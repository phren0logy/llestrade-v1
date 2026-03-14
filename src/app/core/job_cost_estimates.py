"""Local cost forecasting helpers for bulk analysis and report jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.bulk_analysis_runner import (
    BulkAnalysisDocument,
    PromptBundle,
    combine_chunk_summaries,
    generate_chunks,
    load_prompts,
    render_system_prompt,
    render_user_prompt,
    should_chunk,
)
from src.app.core.bulk_prompt_context import build_bulk_placeholders
from src.app.core.bulk_recovery import BulkRecoveryStore
from src.app.core.llm_catalog import calculate_usage_cost
from src.app.core.llm_operation_settings import LLMOperationSettings, normalize_context_window_override
from src.app.core.placeholders.system import SourceFileContext
from src.app.core.project_manager import ProjectMetadata
from src.app.core.prompt_placeholders import format_prompt
from src.app.core.refinement_prompt import read_generation_prompt, read_refinement_prompt
from src.app.core.report_prompt_context import (
    build_report_base_placeholders,
    build_report_generation_placeholders,
    build_report_refinement_placeholders,
)
from src.app.core.report_template_sections import load_template_sections
from src.common.llm.request_budget import (
    compute_preflight_input_budget,
    compute_request_input_budget,
    estimate_request_input_tokens,
    evaluate_request_budget,
)
from src.common.llm.tokens import TokenCounter
from src.app.workers.bulk_analysis_worker import (
    _DEFAULT_MAX_OUTPUT_TOKENS as _BULK_MAX_TOKENS,
    _MIN_CHUNK_TOKEN_TARGET,
    BulkAnalysisWorker,
    ProviderConfig as BulkProviderConfig,
)
from src.app.workers.bulk_reduce_worker import (
    _MIN_INPUT_TOKEN_BUDGET,
    BulkReduceWorker,
    ProviderConfig as ReduceProviderConfig,
)
from src.app.workers.report_common import ReportWorkerBase, _MIN_REPORT_INPUT_BUDGET
from src.app.workers.llm_backend import PydanticAIDirectBackend


@dataclass(frozen=True, slots=True)
class CostForecast:
    available: bool
    best_estimate: float | None
    ceiling: float | None
    spent_actual: float | None = None
    remaining_best_estimate: float | None = None
    remaining_ceiling: float | None = None
    projected_total_best_estimate: float | None = None
    projected_total_ceiling: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "available": self.available,
            "best_estimate": self.best_estimate,
            "ceiling": self.ceiling,
            "spent_actual": self.spent_actual,
            "remaining_best_estimate": self.remaining_best_estimate,
            "remaining_ceiling": self.remaining_ceiling,
            "projected_total_best_estimate": self.projected_total_best_estimate,
            "projected_total_ceiling": self.projected_total_ceiling,
            "reason": self.reason,
        }


def format_currency(amount: float | None) -> str:
    if amount is None:
        return "Unavailable"
    if amount < 1:
        return f"${amount:.2f}"
    return f"${amount:,.2f}"


def format_forecast_inline(forecast: CostForecast) -> str:
    if not forecast.available:
        return "Est. cost unavailable"
    return (
        f"Best {format_currency(forecast.best_estimate)}"
        f" | Ceiling {format_currency(forecast.ceiling)}"
    )


def format_forecast_confirmation(forecast: CostForecast) -> str:
    if not forecast.available:
        return "Estimated cost unavailable for this run."
    if forecast.spent_actual is None:
        return (
            f"Best estimate: {format_currency(forecast.best_estimate)}\n"
            f"Ceiling: {format_currency(forecast.ceiling)}"
        )
    return (
        f"Spent actual: {format_currency(forecast.spent_actual)}\n"
        f"Remaining best estimate: {format_currency(forecast.remaining_best_estimate)}\n"
        f"Remaining ceiling: {format_currency(forecast.remaining_ceiling)}\n"
        f"Projected total best estimate: {format_currency(forecast.projected_total_best_estimate)}\n"
        f"Projected total ceiling: {format_currency(forecast.projected_total_ceiling)}"
    )


def _estimate_output_tokens(input_tokens: int, *, kind: str, max_tokens: int, mode: str) -> int:
    if mode == "ceiling":
        return max_tokens
    if kind in {"bulk_chunk", "reduce_chunk"}:
        return min(max_tokens, max(512, round(input_tokens * 0.10)))
    if kind in {"bulk_combine", "reduce_combine"}:
        return min(max_tokens, max(1024, round(input_tokens * 0.18)))
    if kind == "report_section":
        return min(max_tokens, max(1024, round(input_tokens * 0.25)))
    if kind == "report_refinement":
        return min(max_tokens, max(2048, round(input_tokens * 0.35)))
    return min(max_tokens, max(512, round(input_tokens * 0.10)))


def _request_cost(provider_id: str, model_id: str | None, *, input_tokens: int, output_tokens: int) -> float | None:
    value = calculate_usage_cost(
        provider_id=provider_id,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return float(value) if value is not None else None


def _sum_costs(costs: Iterable[float | None]) -> float | None:
    total = 0.0
    seen = False
    for value in costs:
        if value is None:
            return None
        total += float(value)
        seen = True
    return total if seen else 0.0


def _token_estimate(text: str, *, provider_id: str, model_id: str | None) -> int:
    return estimate_request_input_tokens(
        system_prompt="",
        user_prompt=text,
        provider_id=provider_id,
        model_id=model_id,
    )


def _combine_prompt_tokens(
    *,
    document_name: str,
    metadata: ProjectMetadata | None,
    placeholder_values: Mapping[str, str],
    summary_tokens: Sequence[int],
    system_prompt: str,
    provider_id: str,
    model_id: str | None,
) -> int:
    fake_summaries = [("summary " * max(token_count, 1)).strip() for token_count in summary_tokens]
    prompt, _ = combine_chunk_summaries(
        fake_summaries,
        document_name=document_name,
        metadata=metadata,
        placeholder_values=placeholder_values,
    )
    return estimate_request_input_tokens(
        system_prompt=system_prompt,
        user_prompt=prompt,
        provider_id=provider_id,
        model_id=model_id,
    )


def _plan_hierarchical_combine_cost(
    *,
    document_name: str,
    metadata: ProjectMetadata | None,
    placeholder_values: Mapping[str, str],
    system_prompt: str,
    provider_id: str,
    model_id: str | None,
    summary_tokens: Sequence[int],
    input_budget: int,
    max_tokens: int,
    mode: str,
    completed_keys: set[str] | None = None,
    key_prefix: str = "",
) -> float | None:
    current = list(summary_tokens)
    costs: list[float | None] = []
    completed_keys = completed_keys or set()
    level = 0
    while len(current) > 1:
        level += 1
        next_level: list[int] = []
        batch: list[int] = []
        batch_index = 1
        for summary in current:
            trial = batch + [summary]
            prompt_tokens = _combine_prompt_tokens(
                document_name=document_name,
                metadata=metadata,
                placeholder_values=placeholder_values,
                summary_tokens=trial,
                system_prompt=system_prompt,
                provider_id=provider_id,
                model_id=model_id,
            )
            if batch and prompt_tokens > input_budget:
                batch_key = f"{key_prefix}{level}:{batch_index}"
                if batch_key not in completed_keys:
                    batch_prompt_tokens = _combine_prompt_tokens(
                        document_name=document_name,
                        metadata=metadata,
                        placeholder_values=placeholder_values,
                        summary_tokens=batch,
                        system_prompt=system_prompt,
                        provider_id=provider_id,
                        model_id=model_id,
                    )
                    output_tokens = _estimate_output_tokens(
                        batch_prompt_tokens,
                        kind="bulk_combine",
                        max_tokens=max_tokens,
                        mode=mode,
                    )
                    costs.append(
                        _request_cost(
                            provider_id,
                            model_id,
                            input_tokens=batch_prompt_tokens,
                            output_tokens=output_tokens,
                        )
                    )
                next_level.append(
                    _estimate_output_tokens(
                        _combine_prompt_tokens(
                            document_name=document_name,
                            metadata=metadata,
                            placeholder_values=placeholder_values,
                            summary_tokens=batch,
                            system_prompt=system_prompt,
                            provider_id=provider_id,
                            model_id=model_id,
                        ),
                        kind="bulk_combine",
                        max_tokens=max_tokens,
                        mode=mode,
                    )
                )
                batch = [summary]
                batch_index += 1
            else:
                batch = trial
        if batch:
            batch_key = f"{key_prefix}{level}:{batch_index}"
            if batch_key not in completed_keys:
                batch_prompt_tokens = _combine_prompt_tokens(
                    document_name=document_name,
                    metadata=metadata,
                    placeholder_values=placeholder_values,
                    summary_tokens=batch,
                    system_prompt=system_prompt,
                    provider_id=provider_id,
                    model_id=model_id,
                )
                output_tokens = _estimate_output_tokens(
                    batch_prompt_tokens,
                    kind="bulk_combine",
                    max_tokens=max_tokens,
                    mode=mode,
                )
                costs.append(
                    _request_cost(
                        provider_id,
                        model_id,
                        input_tokens=batch_prompt_tokens,
                        output_tokens=output_tokens,
                    )
                )
            next_level.append(
                _estimate_output_tokens(
                    _combine_prompt_tokens(
                        document_name=document_name,
                        metadata=metadata,
                        placeholder_values=placeholder_values,
                        summary_tokens=batch,
                        system_prompt=system_prompt,
                        provider_id=provider_id,
                        model_id=model_id,
                    ),
                    kind="bulk_combine",
                    max_tokens=max_tokens,
                    mode=mode,
                )
            )
        current = next_level
    return _sum_costs(costs)


def estimate_bulk_map_cost(
    *,
    project_dir: Path,
    group: BulkAnalysisGroup,
    files: Sequence[str],
    metadata: ProjectMetadata | None,
    placeholder_values: Mapping[str, str],
    project_name: str,
    force_rerun: bool,
) -> CostForecast:
    backend = PydanticAIDirectBackend()
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=files,
        metadata=metadata,
        force_rerun=force_rerun,
        placeholder_values=placeholder_values,
        project_name=project_name,
        llm_backend=backend,
    )
    provider_config = worker._resolve_provider()
    bundle = load_prompts(project_dir, group, metadata)
    global_placeholders = worker._build_placeholder_map()
    system_prompt = render_system_prompt(bundle, metadata, placeholder_values=global_placeholders)
    documents = [
        BulkAnalysisDocument(
            source_path=project_dir / "converted_documents" / relative,
            relative_path=relative,
            output_path=project_dir / "bulk_analysis" / group.folder_name / Path(relative).with_suffix("").with_name(Path(relative).stem + "_analysis.md"),
        )
        for relative in files
    ]
    recovery = BulkRecoveryStore(project_dir / "bulk_analysis" / (getattr(group, "slug", None) or group.folder_name))
    recovery_manifest = recovery.load_map_manifest()
    recovery_docs = dict(recovery_manifest.get("documents") or {})

    best_costs: list[float | None] = []
    ceiling_costs: list[float | None] = []
    spent_actual = 0.0 if not force_rerun else None
    if spent_actual is not None:
        spent_actual = float(recovery_manifest.get("actual_cost", 0.0) or 0.0)

    for document in documents:
        body, _, source_context = worker._load_document(document)
        doc_placeholders = worker._build_document_placeholders(global_placeholders, source_context)
        override_window = normalize_context_window_override(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            context_window=getattr(group, "model_context_window", None),
        )
        if isinstance(override_window, int) and override_window > 0:
            raw_context_window = int(override_window)
            token_count = _token_estimate(body, provider_id=provider_config.provider_id, model_id=provider_config.model)
            default_chunk_tokens = max(int(raw_context_window * 0.5), _MIN_CHUNK_TOKEN_TARGET)
            needs_chunking = token_count > default_chunk_tokens
        else:
            needs_chunking, token_count, default_chunk_tokens = should_chunk(
                body,
                provider_config.provider_id,
                provider_config.model,
            )
            raw_context_window = TokenCounter.get_model_context_window(
                provider_config.model or provider_config.provider_id,
                ratio=1.0,
                provider_id=provider_config.provider_id,
            )
        _, input_budget = compute_request_input_budget(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            raw_context_window=raw_context_window,
            max_output_tokens=_BULK_MAX_TOKENS,
            minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
        )
        input_budget = compute_preflight_input_budget(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            runtime_input_budget=input_budget,
            minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
        ) or _MIN_CHUNK_TOKEN_TARGET
        full_prompt = render_user_prompt(
            bundle,
            metadata,
            document.relative_path,
            body,
            placeholder_values=doc_placeholders,
        )
        full_request = evaluate_request_budget(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            system_prompt=system_prompt,
            user_prompt=full_prompt,
            raw_context_window=raw_context_window,
            max_output_tokens=_BULK_MAX_TOKENS,
            minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
        )
        full_prompt_tokens = full_request.input_tokens
        if full_prompt_tokens > input_budget:
            needs_chunking = True

        doc_recovery = dict(recovery_docs.get(document.relative_path) or {})
        completed_chunks = {
            key for key, item in dict(doc_recovery.get("chunks") or {}).items()
            if isinstance(item, dict) and item.get("status") == "complete"
        }
        completed_batches = {
            key for key, item in dict(doc_recovery.get("batches") or {}).items()
            if isinstance(item, dict) and item.get("status") == "complete"
        }

        if not needs_chunking:
            if not force_rerun and doc_recovery.get("status") == "complete":
                continue
            best_costs.append(
                _request_cost(
                    provider_config.provider_id,
                    provider_config.model,
                    input_tokens=full_prompt_tokens,
                    output_tokens=_estimate_output_tokens(
                        full_prompt_tokens,
                        kind="bulk_chunk",
                        max_tokens=_BULK_MAX_TOKENS,
                        mode="best",
                    ),
                )
            )
            ceiling_costs.append(
                _request_cost(
                    provider_config.provider_id,
                    provider_config.model,
                    input_tokens=full_prompt_tokens,
                    output_tokens=_estimate_output_tokens(
                        full_prompt_tokens,
                        kind="bulk_chunk",
                        max_tokens=_BULK_MAX_TOKENS,
                        mode="ceiling",
                    ),
                )
            )
            continue

        chunks = worker._generate_fitting_chunks(
            provider=object(),
            provider_config=provider_config,
            bundle=bundle,
            system_prompt=system_prompt,
            document=document,
            body=body,
            placeholder_values=doc_placeholders,
            input_budget=input_budget,
            initial_chunk_tokens=worker._chunk_target_from_budget(
                input_budget=input_budget,
                default_chunk_tokens=default_chunk_tokens,
            ),
        )
        best_summary_tokens: list[int] = []
        ceiling_summary_tokens: list[int] = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = render_user_prompt(
                bundle,
                metadata,
                document.relative_path,
                chunk,
                chunk_index=idx,
                chunk_total=len(chunks),
                placeholder_values=doc_placeholders,
            )
            input_tokens = evaluate_request_budget(
                provider_id=provider_config.provider_id,
                model_id=provider_config.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                raw_context_window=raw_context_window,
                max_output_tokens=_BULK_MAX_TOKENS,
                minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
            )
            input_tokens = input_tokens.input_tokens
            best_out = _estimate_output_tokens(input_tokens, kind="bulk_chunk", max_tokens=_BULK_MAX_TOKENS, mode="best")
            ceiling_out = _estimate_output_tokens(input_tokens, kind="bulk_chunk", max_tokens=_BULK_MAX_TOKENS, mode="ceiling")
            best_summary_tokens.append(best_out)
            ceiling_summary_tokens.append(ceiling_out)
            if force_rerun or str(idx) not in completed_chunks:
                best_costs.append(_request_cost(provider_config.provider_id, provider_config.model, input_tokens=input_tokens, output_tokens=best_out))
                ceiling_costs.append(_request_cost(provider_config.provider_id, provider_config.model, input_tokens=input_tokens, output_tokens=ceiling_out))

        best_costs.append(
            _plan_hierarchical_combine_cost(
                document_name=document.relative_path,
                metadata=metadata,
                placeholder_values=doc_placeholders,
                system_prompt=system_prompt,
                provider_id=provider_config.provider_id,
                model_id=provider_config.model,
                summary_tokens=best_summary_tokens,
                input_budget=input_budget,
                max_tokens=_BULK_MAX_TOKENS,
                mode="best",
                completed_keys=set() if force_rerun else completed_batches,
            )
        )
        ceiling_costs.append(
            _plan_hierarchical_combine_cost(
                document_name=document.relative_path,
                metadata=metadata,
                placeholder_values=doc_placeholders,
                system_prompt=system_prompt,
                provider_id=provider_config.provider_id,
                model_id=provider_config.model,
                summary_tokens=ceiling_summary_tokens,
                input_budget=input_budget,
                max_tokens=_BULK_MAX_TOKENS,
                mode="ceiling",
                completed_keys=set() if force_rerun else completed_batches,
            )
        )

    best = _sum_costs(best_costs)
    ceiling = _sum_costs(ceiling_costs)
    if best is None or ceiling is None:
        return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Pricing unavailable")
    if spent_actual is None:
        return CostForecast(available=True, best_estimate=best, ceiling=ceiling)
    return CostForecast(
        available=True,
        best_estimate=best,
        ceiling=ceiling,
        spent_actual=spent_actual,
        remaining_best_estimate=best,
        remaining_ceiling=ceiling,
        projected_total_best_estimate=spent_actual + best,
        projected_total_ceiling=spent_actual + ceiling,
    )


def estimate_bulk_reduce_cost(
    *,
    project_dir: Path,
    group: BulkAnalysisGroup,
    metadata: ProjectMetadata | None,
    placeholder_values: Mapping[str, str],
    project_name: str,
    force_rerun: bool,
) -> CostForecast:
    backend = PydanticAIDirectBackend()
    worker = BulkReduceWorker(
        project_dir=project_dir,
        group=group,
        metadata=metadata,
        force_rerun=force_rerun,
        placeholder_values=placeholder_values,
        project_name=project_name,
        llm_backend=backend,
    )
    provider_cfg = worker._resolve_provider()
    inputs = worker._resolve_inputs()
    if not inputs:
        return CostForecast(available=True, best_estimate=0.0, ceiling=0.0)
    bundle = load_prompts(project_dir, group, metadata)
    contexts = {}
    for _, path, _ in inputs:
        for ctx in worker._extract_source_contexts(path):
            contexts[ctx.relative_path] = ctx
    placeholders = worker._build_placeholder_map(reduce_sources=list(contexts.values()))
    system_prompt = render_system_prompt(bundle, metadata, placeholder_values=placeholders)
    content = worker._assemble_combined_content(inputs)
    raw_context_window = normalize_context_window_override(
        provider_id=provider_cfg.provider_id,
        model_id=provider_cfg.model,
        context_window=getattr(group, "model_context_window", None),
    )
    _, input_budget = compute_request_input_budget(
        provider_id=provider_cfg.provider_id,
        model_id=provider_cfg.model,
        explicit_context_window=raw_context_window if isinstance(raw_context_window, int) and raw_context_window > 0 else None,
        max_output_tokens=32_000,
        minimum_budget=_MIN_INPUT_TOKEN_BUDGET,
    )
    input_budget = compute_preflight_input_budget(
        provider_id=provider_cfg.provider_id,
        model_id=provider_cfg.model,
        runtime_input_budget=input_budget,
        minimum_budget=_MIN_INPUT_TOKEN_BUDGET,
    ) or _MIN_INPUT_TOKEN_BUDGET
    needs_chunking, token_count, max_tokens = should_chunk(content, provider_cfg.provider_id, provider_cfg.model)
    recovery = BulkRecoveryStore(project_dir / "bulk_analysis" / (getattr(group, "slug", None) or group.folder_name))
    manifest = recovery.load_reduce_manifest()
    completed_chunks = {
        key for key, item in dict((manifest.get("chunks") or {}).get("items") or {}).items()
        if isinstance(item, dict) and item.get("status") == "complete"
    }
    completed_batches = {
        key for key, item in dict(manifest.get("batches") or {}).items()
        if isinstance(item, dict) and item.get("status") == "complete"
    }
    spent_actual = None if force_rerun else float(manifest.get("actual_cost", 0.0) or 0.0)

    if not needs_chunking:
        prompt = render_user_prompt(bundle, metadata, group.name, content, placeholder_values=placeholders)
        input_tokens = evaluate_request_budget(
            provider_id=provider_cfg.provider_id,
            model_id=provider_cfg.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=32_000,
            explicit_context_window=raw_context_window if isinstance(raw_context_window, int) and raw_context_window > 0 else None,
            minimum_budget=_MIN_INPUT_TOKEN_BUDGET,
        ).input_tokens
        best = _request_cost(
            provider_cfg.provider_id, provider_cfg.model,
            input_tokens=input_tokens,
            output_tokens=_estimate_output_tokens(input_tokens, kind="reduce_combine", max_tokens=32_000, mode="best"),
        )
        ceiling = _request_cost(
            provider_cfg.provider_id, provider_cfg.model,
            input_tokens=input_tokens,
            output_tokens=_estimate_output_tokens(input_tokens, kind="reduce_combine", max_tokens=32_000, mode="ceiling"),
        )
        if best is None or ceiling is None:
            return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Pricing unavailable")
        if spent_actual is None:
            return CostForecast(available=True, best_estimate=best, ceiling=ceiling)
        return CostForecast(
            available=True,
            best_estimate=best,
            ceiling=ceiling,
            spent_actual=spent_actual,
            remaining_best_estimate=best,
            remaining_ceiling=ceiling,
            projected_total_best_estimate=spent_actual + best,
            projected_total_ceiling=spent_actual + ceiling,
        )

    chunks = generate_chunks(content, max_tokens)
    best_costs: list[float | None] = []
    ceiling_costs: list[float | None] = []
    best_summary_tokens: list[int] = []
    ceiling_summary_tokens: list[int] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = render_user_prompt(
            bundle,
            metadata,
            group.name,
            chunk,
            chunk_index=idx,
            chunk_total=len(chunks),
            placeholder_values=placeholders,
        )
        input_tokens = evaluate_request_budget(
            provider_id=provider_cfg.provider_id,
            model_id=provider_cfg.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=32_000,
            explicit_context_window=raw_context_window if isinstance(raw_context_window, int) and raw_context_window > 0 else None,
            minimum_budget=_MIN_INPUT_TOKEN_BUDGET,
        ).input_tokens
        best_out = _estimate_output_tokens(input_tokens, kind="reduce_chunk", max_tokens=32_000, mode="best")
        ceiling_out = _estimate_output_tokens(input_tokens, kind="reduce_chunk", max_tokens=32_000, mode="ceiling")
        best_summary_tokens.append(best_out)
        ceiling_summary_tokens.append(ceiling_out)
        if force_rerun or str(idx) not in completed_chunks:
            best_costs.append(_request_cost(provider_cfg.provider_id, provider_cfg.model, input_tokens=input_tokens, output_tokens=best_out))
            ceiling_costs.append(_request_cost(provider_cfg.provider_id, provider_cfg.model, input_tokens=input_tokens, output_tokens=ceiling_out))
    best_costs.append(
        _plan_hierarchical_combine_cost(
            document_name=group.name,
            metadata=metadata,
            placeholder_values=placeholders,
            system_prompt=system_prompt,
            provider_id=provider_cfg.provider_id,
            model_id=provider_cfg.model,
            summary_tokens=best_summary_tokens,
            input_budget=input_budget,
            max_tokens=32_000,
            mode="best",
            completed_keys=set() if force_rerun else completed_batches,
        )
    )
    ceiling_costs.append(
        _plan_hierarchical_combine_cost(
            document_name=group.name,
            metadata=metadata,
            placeholder_values=placeholders,
            system_prompt=system_prompt,
            provider_id=provider_cfg.provider_id,
            model_id=provider_cfg.model,
            summary_tokens=ceiling_summary_tokens,
            input_budget=input_budget,
            max_tokens=32_000,
            mode="ceiling",
            completed_keys=set() if force_rerun else completed_batches,
        )
    )
    best = _sum_costs(best_costs)
    ceiling = _sum_costs(ceiling_costs)
    if best is None or ceiling is None:
        return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Pricing unavailable")
    if spent_actual is None:
        return CostForecast(available=True, best_estimate=best, ceiling=ceiling)
    return CostForecast(
        available=True,
        best_estimate=best,
        ceiling=ceiling,
        spent_actual=spent_actual,
        remaining_best_estimate=best,
        remaining_ceiling=ceiling,
        projected_total_best_estimate=spent_actual + best,
        projected_total_ceiling=spent_actual + ceiling,
    )


class _EstimateReportWorker(ReportWorkerBase):
    def _run(self) -> None:  # pragma: no cover - never executed
        raise NotImplementedError


def estimate_report_draft_cost(
    *,
    project_dir: Path,
    inputs: Sequence[tuple[str, str]],
    llm_settings: LLMOperationSettings,
    template_path: Path,
    transcript_path: Path | None,
    generation_user_prompt_path: Path,
    generation_system_prompt_path: Path,
    metadata: ProjectMetadata,
    placeholder_values: Mapping[str, str],
    project_name: str,
    max_report_tokens: int = 60_000,
) -> CostForecast:
    worker = _EstimateReportWorker(
        worker_name="report-estimate",
        project_dir=project_dir,
        inputs=inputs,
        provider_id=llm_settings.provider_id,
        model=llm_settings.model_id,
        custom_model=None,
        context_window=llm_settings.context_window,
        use_reasoning=llm_settings.use_reasoning,
        reasoning=llm_settings.reasoning,
        metadata=metadata,
        placeholder_values=placeholder_values,
        project_name=project_name,
        max_report_tokens=max_report_tokens,
        llm_backend=PydanticAIDirectBackend(),
    )
    placeholder_map = worker._placeholder_map()
    combined_content, _ = worker._combine_inputs()
    transcript_text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path and transcript_path.exists() else ""
    system_template = generation_system_prompt_path.read_text(encoding="utf-8").strip()
    system_prompt = format_prompt(system_template, placeholder_map)
    user_template = read_generation_prompt(generation_user_prompt_path)
    sections = load_template_sections(template_path)
    best_costs: list[float | None] = []
    ceiling_costs: list[float | None] = []
    for section in sections:
        user_placeholders = build_report_generation_placeholders(
            base_placeholders=placeholder_map,
            template_section=section.body.strip(),
            section_title=section.title or "",
            transcript=transcript_text,
            additional_documents=combined_content.strip(),
        )
        prompt = format_prompt(user_template, user_placeholders)
        input_tokens = evaluate_request_budget(
            provider_id=llm_settings.provider_id,
            model_id=llm_settings.model_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=max_report_tokens,
            explicit_context_window=llm_settings.context_window,
            minimum_budget=_MIN_REPORT_INPUT_BUDGET,
        ).input_tokens
        best_out = _estimate_output_tokens(input_tokens, kind="report_section", max_tokens=max_report_tokens, mode="best")
        ceiling_out = _estimate_output_tokens(input_tokens, kind="report_section", max_tokens=max_report_tokens, mode="ceiling")
        best_costs.append(_request_cost(llm_settings.provider_id, llm_settings.model_id, input_tokens=input_tokens, output_tokens=best_out))
        ceiling_costs.append(_request_cost(llm_settings.provider_id, llm_settings.model_id, input_tokens=input_tokens, output_tokens=ceiling_out))
    best = _sum_costs(best_costs)
    ceiling = _sum_costs(ceiling_costs)
    if best is None or ceiling is None:
        return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Pricing unavailable")
    return CostForecast(available=True, best_estimate=best, ceiling=ceiling)


def estimate_report_refinement_cost(
    *,
    project_dir: Path,
    inputs: Sequence[tuple[str, str]],
    llm_settings: LLMOperationSettings,
    draft_path: Path,
    template_path: Path | None,
    transcript_path: Path | None,
    refinement_user_prompt_path: Path,
    refinement_system_prompt_path: Path,
    metadata: ProjectMetadata,
    placeholder_values: Mapping[str, str],
    project_name: str,
    max_report_tokens: int = 60_000,
) -> CostForecast:
    worker = _EstimateReportWorker(
        worker_name="report-estimate",
        project_dir=project_dir,
        inputs=inputs,
        provider_id=llm_settings.provider_id,
        model=llm_settings.model_id,
        custom_model=None,
        context_window=llm_settings.context_window,
        use_reasoning=llm_settings.use_reasoning,
        reasoning=llm_settings.reasoning,
        metadata=metadata,
        placeholder_values=placeholder_values,
        project_name=project_name,
        max_report_tokens=max_report_tokens,
        llm_backend=PydanticAIDirectBackend(),
    )
    placeholder_map = worker._placeholder_map()
    draft_text = draft_path.read_text(encoding="utf-8").strip()
    template_text = template_path.read_text(encoding="utf-8").strip() if template_path and template_path.exists() else ""
    transcript_text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path and transcript_path.exists() else ""
    user_template = read_refinement_prompt(refinement_user_prompt_path)
    system_template = refinement_system_prompt_path.read_text(encoding="utf-8").strip()
    system_prompt = format_prompt(system_template, placeholder_map)
    user_placeholders = build_report_refinement_placeholders(
        base_placeholders=placeholder_map,
        draft_report=draft_text,
        template=template_text,
        transcript=transcript_text,
    )
    prompt = format_prompt(user_template, user_placeholders)
    input_tokens = evaluate_request_budget(
        provider_id=llm_settings.provider_id,
        model_id=llm_settings.model_id,
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_output_tokens=max_report_tokens,
        explicit_context_window=llm_settings.context_window,
        minimum_budget=_MIN_REPORT_INPUT_BUDGET,
    ).input_tokens
    best = _request_cost(
        llm_settings.provider_id,
        llm_settings.model_id,
        input_tokens=input_tokens,
        output_tokens=_estimate_output_tokens(input_tokens, kind="report_refinement", max_tokens=max_report_tokens, mode="best"),
    )
    ceiling = _request_cost(
        llm_settings.provider_id,
        llm_settings.model_id,
        input_tokens=input_tokens,
        output_tokens=_estimate_output_tokens(input_tokens, kind="report_refinement", max_tokens=max_report_tokens, mode="ceiling"),
    )
    if best is None or ceiling is None:
        return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Pricing unavailable")
    return CostForecast(available=True, best_estimate=best, ceiling=ceiling)


__all__ = [
    "CostForecast",
    "estimate_bulk_map_cost",
    "estimate_bulk_reduce_cost",
    "estimate_report_draft_cost",
    "estimate_report_refinement_cost",
    "format_currency",
    "format_forecast_confirmation",
    "format_forecast_inline",
]
