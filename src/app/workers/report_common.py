"""Shared helpers for report generation and refinement workers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.app.core.citations import CitationRecordStats, CitationStore, parse_citation_tokens
from src.app.core.llm_catalog import calculate_usage_cost
from src.app.core.llm_operation_settings import LLMReasoningSettings
from src.app.core.report_prompt_context import build_report_base_placeholders
from src.app.core.project_manager import ProjectMetadata
from src.app.core.report_inputs import category_display_name
from src.common.llm.budgets import compute_input_token_budget
from src.common.llm.request_budget import (
    compute_request_input_budget,
    evaluate_request_budget,
)
from src.common.llm.tokens import TokenCounter
from src.common.markdown import PromptReference, SourceReference, compute_file_checksum

from .base import DashboardWorker
from .llm_backend import (
    PydanticAIDirectBackend,
    LLMExecutionBackend,
    LLMInvocationRequest,
    LLMProviderRequest,
    backend_transport_name,
)
from .progress import WorkerProgressDetail
_CITATION_ID_RE = re.compile(r"^ev_[a-z0-9]{8,64}$")
_MIN_REPORT_INPUT_BUDGET = 4_000


class ReportWorkerBase(DashboardWorker):
    """Base class with shared plumbing for draft and refinement workers."""

    def __init__(
        self,
        *,
        worker_name: str,
        project_dir: Path,
        inputs: Sequence[tuple[str, str]],
        provider_id: str,
        model: str,
        custom_model: Optional[str],
        context_window: Optional[int],
        use_reasoning: bool,
        reasoning: Mapping[str, Any] | LLMReasoningSettings | None,
        metadata: ProjectMetadata,
        placeholder_values: Mapping[str, str] | None,
        project_name: str,
        max_report_tokens: int,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        super().__init__(worker_name=worker_name)
        self._project_dir = project_dir
        self._inputs = list(inputs)
        self._provider_id = provider_id
        self._llm_backend: LLMExecutionBackend = llm_backend or PydanticAIDirectBackend()
        self._model = self._llm_backend.normalize_model(provider_id, model)
        raw_custom = custom_model.strip() if custom_model else None
        self._custom_model = self._llm_backend.normalize_model(provider_id, raw_custom)
        self._context_window = context_window
        self._use_reasoning = use_reasoning
        self._reasoning = LLMReasoningSettings.from_value(reasoning, legacy_use_reasoning=use_reasoning)
        self._metadata = metadata
        self._max_report_tokens = max_report_tokens
        self._base_placeholders = dict(placeholder_values or {})
        effective_name = project_name or metadata.case_name or project_dir.name
        self._project_name = effective_name
        self._run_timestamp = datetime.now(timezone.utc)
        try:
            self._citation_store: CitationStore | None = CitationStore(project_dir)
        except Exception:
            self._citation_store = None
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _emit_progress_detail(
        self,
        *,
        run_kind: str,
        phase: str,
        label: str,
        percent: int | None = None,
        detail: str | None = None,
        section_index: int | None = None,
        section_total: int | None = None,
        section_title: str | None = None,
    ) -> None:
        signal = getattr(self, "progress_detail", None)
        if signal is None:
            return
        signal.emit(
            WorkerProgressDetail(
                run_kind=run_kind,
                phase=phase,
                label=label,
                percent=percent,
                section_index=section_index,
                section_total=section_total,
                section_title=section_title,
                detail=detail,
            )
        )

    # ------------------------------------------------------------------
    # Placeholder helpers
    # ------------------------------------------------------------------
    def _placeholder_map(self) -> Dict[str, str]:
        return build_report_base_placeholders(
            base_placeholders=self._base_placeholders,
            project_name=self._project_name,
            project_dir=self._project_dir,
            timestamp=self._run_timestamp,
        )

    # ------------------------------------------------------------------
    # Input aggregation
    # ------------------------------------------------------------------
    def _combine_inputs(self) -> tuple[str, List[dict]]:
        lines: List[str] = []
        metadata: List[dict] = []
        for category, relative in self._inputs:
            absolute = (self._project_dir / relative).resolve()
            if not absolute.exists():
                raise FileNotFoundError(f"Selected input missing: {relative}")
            if absolute.suffix.lower() not in {".md", ".txt"}:
                raise RuntimeError(f"Unsupported input type: {relative}")
            content = absolute.read_text(encoding="utf-8")
            section_header = self._render_section_header(category, relative)
            lines.append(f"<!--- report-input: {category} | {relative} --->")
            lines.append(section_header)
            lines.append(content.strip())
            lines.append("")
            token_info = TokenCounter.count(
                text=content,
                provider=self._provider_id,
                model=self._custom_model or self._model,
            )
            token_count = (
                int(token_info.get("token_count"))
                if token_info.get("success") and token_info.get("token_count") is not None
                else max(len(content) // 4, 1)
            )
            metadata.append(
                {
                    "category": category,
                    "relative_path": relative,
                    "absolute_path": str(absolute),
                    "token_count": token_count,
                }
            )
        if not lines:
            return "", metadata
        combined = "\n".join(lines).strip() + "\n"
        return combined, metadata

    def _render_section_header(self, category: str, relative: str) -> str:
        title = category_display_name(category)
        return f"# {title}: {relative}\n"

    def _input_sources(self, items: Sequence[dict]) -> List[SourceReference]:
        sources: List[SourceReference] = []
        for item in items:
            absolute_raw = item.get("absolute_path")
            if not absolute_raw:
                continue
            abs_path = Path(absolute_raw).expanduser()
            relative = item.get("relative_path") or self._relative_to_project(abs_path)
            category = item.get("category") or "input"
            sources.append(
                SourceReference(
                    path=abs_path,
                    relative=relative,
                    kind=(abs_path.suffix.lstrip(".") or "file"),
                    role=category,
                    checksum=compute_file_checksum(abs_path),
                )
            )
        return sources

    def _optional_source(self, path: Optional[Path], *, role: str) -> Optional[SourceReference]:
        if path is None:
            return None
        return self._file_source(path, role=role)

    def _file_source(
        self,
        path: Optional[Path],
        *,
        role: str,
        checksum: Optional[str] = None,
    ) -> Optional[SourceReference]:
        if path is None:
            return None
        return SourceReference(
            path=path,
            relative=self._relative_to_project(path),
            kind=(path.suffix.lstrip(".") or "file"),
            role=role,
            checksum=checksum or compute_file_checksum(path),
        )

    def _prompt_reference(self, path: Path, *, role: str) -> PromptReference:
        return PromptReference(path=path, role=role)

    def _relative_to_project(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._project_dir.resolve()).as_posix()
        except Exception:
            return path.name

    def _input_token_limit(self, *, max_output_tokens: int) -> int | None:
        _, input_budget = compute_request_input_budget(
            provider_id=self._provider_id,
            model_id=self._custom_model or self._model,
            max_output_tokens=max_output_tokens,
            explicit_context_window=self._context_window,
            minimum_budget=_MIN_REPORT_INPUT_BUDGET,
        )
        return input_budget

    def _evaluate_request_budget(
        self,
        provider: object,
        *,
        prompt: str,
        system_prompt: str,
        max_output_tokens: int,
        temperature: float,
    ):
        return evaluate_request_budget(
            provider_id=self._provider_id,
            model_id=self._custom_model or self._model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=max_output_tokens,
            explicit_context_window=self._context_window,
            minimum_budget=_MIN_REPORT_INPUT_BUDGET,
            exact_token_counter=lambda: self._llm_backend.count_input_tokens(
                provider,
                LLMInvocationRequest(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self._custom_model or self._model,
                    model_settings=self._llm_backend.build_model_settings(
                        self._provider_id,
                        self._custom_model or self._model,
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                        use_reasoning=self._use_reasoning,
                        reasoning=self._reasoning,
                    ),
                ),
            ),
        )

    def _record_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        self._usage_totals["input_tokens"] += input_tokens
        self._usage_totals["output_tokens"] += output_tokens
        self._usage_totals["total_tokens"] += input_tokens + output_tokens

    def _usage_summary(self) -> dict[str, int]:
        return dict(self._usage_totals)

    def _total_cost(self) -> float | None:
        amount = calculate_usage_cost(
            provider_id=self._provider_id,
            model_id=self._custom_model or self._model,
            input_tokens=self._usage_totals["input_tokens"],
            output_tokens=self._usage_totals["output_tokens"],
        )
        if amount is None:
            return None
        return float(amount)

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------
    def _append_citation_ledger(self, prompt: str, ledger: str) -> str:
        if not ledger.strip():
            return prompt
        return f"{prompt.rstrip()}\n\n{ledger.rstrip()}\n"

    def _build_report_evidence_ledger(self, inputs_metadata: Sequence[dict]) -> str:
        sections: list[str] = []

        converted_relatives: list[str] = []
        existing_ids: list[str] = []
        seen_ids: set[str] = set()

        for item in inputs_metadata:
            relative = str(item.get("relative_path") or "")
            absolute_raw = item.get("absolute_path")
            if relative.startswith("converted_documents/"):
                converted_relatives.append(relative[len("converted_documents/"):])

            if not absolute_raw:
                continue
            try:
                text = Path(str(absolute_raw)).read_text(encoding="utf-8")
            except Exception:
                continue
            for token in parse_citation_tokens(text):
                if token.ev_id in seen_ids or not _CITATION_ID_RE.match(token.ev_id):
                    continue
                seen_ids.add(token.ev_id)
                existing_ids.append(token.ev_id)
                if len(existing_ids) >= 220:
                    break
            if len(existing_ids) >= 220:
                break

        if self._citation_store is not None and converted_relatives:
            converted_relatives = list(dict.fromkeys(converted_relatives))
            try:
                ledger = self._citation_store.build_evidence_ledger_for_documents(
                    relative_paths=converted_relatives,
                    max_per_document=25,
                    max_total=220,
                )
            except Exception:
                ledger = ""
            if ledger.strip():
                sections.append(ledger.strip())

        if existing_ids:
            lines = [
                "## Existing Citation IDs",
                "Reuse these IDs when the claim is supported by already-cited evidence.",
                "",
            ]
            lines.extend(f"- {ev_id}" for ev_id in existing_ids[:220])
            sections.append("\n".join(lines).strip())

        if not sections:
            return ""
        return "\n\n".join(sections).strip() + "\n"

    def _record_output_citations(
        self,
        *,
        output_path: Path,
        output_text: str,
        generator: str,
    ) -> CitationRecordStats | None:
        if self._citation_store is None:
            return None
        try:
            stats = self._citation_store.record_output_citations(
                output_path=output_path,
                output_text=output_text,
                generator=generator,
                prompt_hash=None,
            )
        except Exception:
            self.logger.debug("Failed to record report citations for %s", output_path, exc_info=True)
            return None

        if stats.total > 0 and (stats.warning > 0 or stats.invalid > 0):
            self.logger.warning(
                "%s citations for %s: valid=%s warning=%s invalid=%s",
                self.job_tag,
                output_path.name,
                stats.valid,
                stats.warning,
                stats.invalid,
            )
        return stats

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------
    def _create_provider(self):
        return self._llm_backend.create_provider(
            LLMProviderRequest(
                provider_id=self._provider_id,
                model=self._custom_model or self._model,
            )
        )

    def _llm_execution_summary(self) -> str:
        model_name = self._custom_model or self._model or "<default>"
        return (
            f"Using {backend_transport_name(self._llm_backend)} backend: "
            f"{self._provider_id}/{model_name}"
        )

__all__ = ["ReportWorkerBase"]
