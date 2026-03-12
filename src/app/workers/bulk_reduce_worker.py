"""Worker that builds a combined document and runs a single prompt."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import frontmatter
from PySide6.QtCore import Signal

from src.app.core.azure_artifacts import is_azure_raw_artifact
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.llm_catalog import calculate_usage_cost
from src.app.core.llm_operation_settings import normalize_context_window_override
from src.app.core.bulk_recovery import BulkRecoveryStore
from src.app.core.bulk_analysis_runner import (
    BulkAnalysisCancelled,
    PromptBundle,
    combine_chunk_summaries,
    combine_chunk_summaries_hierarchical,
    generate_chunks,
    load_prompts,
    render_system_prompt,
    render_user_prompt,
    should_chunk,
)
from src.app.core.citations import CitationRecordStats, CitationStore
from src.app.core.bulk_paths import (
    iter_map_outputs,
    iter_map_outputs_under,
    normalize_map_relative,
    resolve_map_output_path,
)
from src.app.core.bulk_prompt_context import build_bulk_placeholders
from src.app.core.placeholders.system import SourceFileContext
from src.app.core.project_manager import ProjectMetadata
from src.common.llm.budgets import compute_input_token_budget
from src.common.llm.tokens import TokenCounter
from src.config.observability import trace_operation
from src.common.markdown import (
    PromptReference,
    SourceReference,
    apply_frontmatter,
    build_document_metadata,
    compute_file_checksum,
    infer_project_path,
)

from .base import DashboardWorker
from .checkpoint_manager import CheckpointManager, _sha256
from .llm_backend import (
    LLMExecutionBackend,
    LLMInvocationRequest,
    LLMProviderRequest,
    PydanticAIDirectBackend,
    backend_route_name,
    backend_transport_name,
)
from .stage_contracts import BulkReduceStageInput, stage_trace_attributes

LOGGER = logging.getLogger(__name__)


_MANIFEST_VERSION = 2
_PAGE_MARKER_RE = re.compile(r"<!---\\s*.+?#page=(\\d+)\\s*--->")
_MIN_INPUT_TOKEN_BUDGET = 4_000

_DYNAMIC_REDUCE_KEYS: frozenset[str] = frozenset(
    {
        "document_content",
        "chunk_index",
        "chunk_total",
    }
)


def _manifest_path(project_dir: Path, group: BulkAnalysisGroup) -> Path:
    slug = getattr(group, "slug", None) or group.folder_name
    return project_dir / "bulk_analysis" / slug / "reduce" / "manifest.json"


def _default_manifest() -> Dict[str, object]:
    return {
        "version": _MANIFEST_VERSION,
        "signature": None,
        "chunks": {"count": 0, "done": [], "checksums": {}},
        "batches": {},
        "finalized": False,
    }


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return _default_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_manifest()
    if not isinstance(data, dict):
        return _default_manifest()
    chunks = data.get("chunks") if isinstance(data.get("chunks"), dict) else {"count": 0, "done": [], "checksums": {}}
    batches = data.get("batches") if isinstance(data.get("batches"), dict) else {}
    return {
        "version": data.get("version", _MANIFEST_VERSION),
        "signature": data.get("signature"),
        "placeholders": data.get("placeholders"),
        "chunks": chunks,
        "batches": batches,
        "finalized": bool(data.get("finalized", False)),
    }


def _save_manifest(path: Path, manifest: Dict[str, object]) -> None:
    payload = {
        "version": _MANIFEST_VERSION,
        "signature": manifest.get("signature"),
        "placeholders": manifest.get("placeholders"),
        "chunks": manifest.get("chunks", {"count": 0, "done": [], "checksums": {}}),
        "batches": manifest.get("batches", {}),
        "finalized": bool(manifest.get("finalized", False)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stable_placeholders(placeholders: Mapping[str, str]) -> Dict[str, str]:
    """Filter out volatile placeholder values that should not affect rerun signatures."""

    def _keep(key: str) -> bool:
        if key == "timestamp":
            return False
        if key.startswith("source_pdf_") and key != "source_pdf_filename":
            return False
        if key.startswith("reduce_source_") and key not in {"reduce_source_count"}:
            return False
        return True

    return {k: v for k, v in placeholders.items() if _keep(k)}


def _compute_prompt_hash(
    bundle: PromptBundle,
    provider_cfg: ProviderConfig,
    group: BulkAnalysisGroup,
    metadata: Optional[ProjectMetadata],
    placeholder_values: Mapping[str, str] | None = None,
) -> str:
    metadata_summary: Dict[str, str] = {}
    if metadata:
        metadata_summary = {
            "case_name": metadata.case_name,
            "subject_name": metadata.subject_name,
            "date_of_birth": metadata.date_of_birth,
            "case_description": metadata.case_description,
        }
    payload = {
        "system_template": bundle.system_template,
        "user_template": bundle.user_template,
        "provider_id": provider_cfg.provider_id,
        "model": provider_cfg.model,
        "temperature": provider_cfg.temperature,
        "group_operation": group.operation,
        "use_reasoning": group.use_reasoning,
        "model_context_window": group.model_context_window,
        "system_prompt_path": group.system_prompt_path,
        "user_prompt_path": group.user_prompt_path,
        "metadata": metadata_summary,
        "placeholder_requirements": group.placeholder_requirements,
    }
    if placeholder_values:
        payload["placeholders"] = {k: placeholder_values.get(k, "") for k in sorted(placeholder_values)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _inputs_signature(inputs: Sequence[tuple[str, Path, str]]) -> list[dict[str, object]]:
    signature: list[dict[str, object]] = []
    for kind, path, rel in inputs:
        try:
            checksum = compute_file_checksum(path)
        except Exception:
            checksum = ""
        entry: dict[str, object] = {"kind": kind, "path": rel, "checksum": checksum}
        signature.append(entry)
    signature.sort(key=lambda item: (item["kind"], item["path"]))
    return signature


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    model: Optional[str]
    temperature: float


class BulkReduceWorker(DashboardWorker):
    """Combine selected inputs and run a single LLM prompt to produce one output."""

    progress = Signal(int, int, str)  # completed, total, status text
    file_failed = Signal(str, str)  # path, error
    finished = Signal(int, int)  # successes, failures
    log_message = Signal(str)
    cost_calculated = Signal(float, str, str)

    def __init__(
        self,
        *,
        project_dir: Path,
        group: BulkAnalysisGroup,
        metadata: Optional[ProjectMetadata],
        force_rerun: bool = False,
        placeholder_values: Mapping[str, str] | None = None,
        project_name: str = "",
        estimate_summary: Mapping[str, object] | None = None,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        super().__init__(worker_name="bulk_reduce")
        self._project_dir = project_dir
        self._group = group
        self._metadata = metadata
        self._force_rerun = force_rerun
        self._base_placeholders = dict(placeholder_values or {})
        self._project_name = project_name
        self._estimate_summary = dict(estimate_summary or {})
        self._run_timestamp = datetime.now(timezone.utc)
        self._llm_backend: LLMExecutionBackend = llm_backend or PydanticAIDirectBackend()
        try:
            self._citation_store: CitationStore | None = CitationStore(project_dir)
        except Exception:
            self._citation_store = None
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    # ------------------------------------------------------------------
    # QRunnable API
    # ------------------------------------------------------------------
    def _build_placeholder_map(
        self,
        *,
        source: Optional[SourceFileContext] = None,
        reduce_sources: Optional[Sequence[SourceFileContext]] = None,
    ) -> Dict[str, str]:
        return build_bulk_placeholders(
            base_placeholders=self._base_placeholders,
            project_name=self._project_name,
            timestamp=self._run_timestamp,
            source=source,
            reduce_sources=reduce_sources,
        )

    def _enforce_placeholder_requirements(
        self,
        placeholders: Mapping[str, str],
        *,
        context: str,
        dynamic_keys: Iterable[str],
    ) -> None:
        requirements = getattr(self._group, "placeholder_requirements", None) or {}
        if not requirements:
            return

        missing_required: list[str] = []
        missing_optional: list[str] = []
        dynamic = set(dynamic_keys)

        for key, required in requirements.items():
            if key in dynamic:
                continue
            value = (placeholders.get(key) or "").strip()
            if value:
                continue
            if required:
                missing_required.append(key)
            else:
                missing_optional.append(key)

        if missing_optional:
            self.log_message.emit(
                f"{context}: optional placeholders without values: "
                + ", ".join(f"{{{name}}}" for name in sorted(missing_optional))
            )

        if missing_required:
            formatted = ", ".join(f"{{{name}}}" for name in sorted(missing_required))
            raise RuntimeError(f"{context}: required placeholders missing values: {formatted}")

    def _serialise_placeholders(self, placeholders: Mapping[str, str]) -> Dict[str, str]:
        return {key: placeholders.get(key, "") for key in sorted(placeholders)}

    def _resolve_source_context(
        self,
        *,
        relative_hint: Optional[str],
        path_hint: Optional[str],
        fallback_relative: str,
    ) -> SourceFileContext:
        rel_raw = (relative_hint or "").strip()
        path_raw = (path_hint or "").strip()

        absolute: Path
        if path_raw:
            candidate = Path(path_raw).expanduser()
            if not candidate.is_absolute():
                candidate = (self._project_dir / candidate).resolve()
            absolute = candidate
        else:
            absolute = (self._project_dir / fallback_relative).resolve()

        if not rel_raw:
            try:
                rel_raw = absolute.relative_to(self._project_dir).as_posix()
            except Exception:
                rel_raw = absolute.name

        return SourceFileContext(absolute_path=absolute, relative_path=rel_raw)

    def _extract_source_contexts(self, path: Path) -> List[SourceFileContext]:
        try:
            raw = path.read_text(encoding="utf-8")
            post = frontmatter.loads(raw)
            metadata = dict(post.metadata or {})
        except Exception:
            metadata = {}

        contexts: List[SourceFileContext] = []
        try:
            fallback_relative = path.relative_to(self._project_dir).as_posix()
        except Exception:
            fallback_relative = path.name
        sources = metadata.get("sources")
        if not sources and isinstance(metadata.get("metadata"), dict):
            sources = metadata["metadata"].get("sources")
        if isinstance(sources, list):
            for entry in sources:
                if not isinstance(entry, dict):
                    continue
                context = self._resolve_source_context(
                    relative_hint=entry.get("relative") if isinstance(entry.get("relative"), str) else None,
                    path_hint=entry.get("path") if isinstance(entry.get("path"), str) else None,
                    fallback_relative=fallback_relative,
                )
                contexts.append(context)
        if not contexts:
            rel_path = fallback_relative
            contexts.append(SourceFileContext(absolute_path=path.resolve(), relative_path=rel_path))
        unique: Dict[str, SourceFileContext] = {}
        for ctx in contexts:
            unique[ctx.relative_path] = ctx
        return list(unique.values())

    def _run(self) -> None:  # pragma: no cover - executed in worker thread
        try:
            provider_cfg = self._resolve_provider()
            self.log_message.emit(
                f"Using {backend_transport_name(self._llm_backend)} backend: "
                f"{provider_cfg.provider_id}/{provider_cfg.model or '<default>'}"
            )
            bundle = load_prompts(self._project_dir, self._group, self._metadata)

            inputs = self._resolve_inputs()
            total = len(inputs)
            if total == 0:
                self.log_message.emit("No inputs selected for combined operation.")
                self.finished.emit(0, 0)
                return

            aggregate_contexts_map: Dict[str, SourceFileContext] = {}
            for _, path, _ in inputs:
                for ctx in self._extract_source_contexts(path):
                    aggregate_contexts_map[ctx.relative_path] = ctx
            aggregate_contexts = list(aggregate_contexts_map.values())

            placeholders_global = self._build_placeholder_map(reduce_sources=aggregate_contexts)

            self._enforce_placeholder_requirements(
                placeholders_global,
                context=f"combined analysis '{self._group.name}'",
                dynamic_keys=_DYNAMIC_REDUCE_KEYS,
            )

            system_prompt = render_system_prompt(
                bundle,
                self._metadata,
                placeholder_values=placeholders_global,
            )
            prompt_hash = _compute_prompt_hash(
                bundle,
                provider_cfg,
                self._group,
                self._metadata,
                placeholder_values=self._base_placeholders,
            )

            provider = self._create_provider(provider_cfg)
            if provider is None:
                raise RuntimeError("Reduce provider failed to initialise")

            signature_inputs = _inputs_signature(inputs)
            state_manifest_path = _manifest_path(self._project_dir, self._group)
            previous = _load_manifest(state_manifest_path)
            signature_placeholders = _stable_placeholders(self._serialise_placeholders(placeholders_global))
            signature = {
                "prompt_hash": prompt_hash,
                "inputs": signature_inputs,
                "placeholders": signature_placeholders,
            }

            slug = getattr(self._group, "slug", None) or self._group.folder_name
            checkpoint_mgr = CheckpointManager(
                self._project_dir / "bulk_analysis" / slug / "reduce" / "checkpoints"
            )
            recovery_store = BulkRecoveryStore(self._project_dir / "bulk_analysis" / slug)
            recovery_store.import_legacy_reduce(
                checkpoint_root=checkpoint_mgr.base_dir,
                legacy_manifest=previous,
            )
            recovery_manifest = recovery_store.load_reduce_manifest()

            previous_sig = previous.get("signature") or {}
            def _paths(sig: dict[str, object]) -> list[tuple[str, str]]:
                inputs_list = sig.get("inputs") or []
                return [(item.get("kind"), item.get("path")) for item in inputs_list if isinstance(item, dict)]

            same_inputs = (
                previous.get("version") == _MANIFEST_VERSION
                and previous_sig.get("prompt_hash") == prompt_hash
                and _paths(previous_sig) == _paths({"inputs": signature_inputs})
            )

            # Reset checkpoints if version mismatch or inputs/prompt changed
            if previous.get("version") != _MANIFEST_VERSION or not same_inputs:
                checkpoint_mgr.clear_reduce()
                previous = _default_manifest()
            if self._force_rerun or recovery_manifest.get("signature") != signature:
                recovery_store.clear_reduce()
                recovery_manifest = recovery_store.load_reduce_manifest()

            was_finalized = bool(previous.get("finalized"))
            recovery_complete = bool(recovery_manifest.get("finalized")) and str(recovery_manifest.get("status") or "") == "complete"
            current_manifest = previous
            current_manifest["signature"] = signature
            current_manifest.setdefault("chunks", {"count": 0, "done": [], "checksums": {}})
            current_manifest.setdefault("batches", {})
            current_manifest["finalized"] = False
            manifest = current_manifest
            recovery_manifest["signature"] = signature

            if not self._force_rerun and same_inputs and was_finalized and recovery_complete:
                self.log_message.emit("Combined inputs unchanged; skipping run.")
                self.finished.emit(0, 0)
                return
            recovery_manifest["finalized"] = False
            recovery_manifest["status"] = "running"

            self.log_message.emit(
                f"Starting combined bulk analysis for '{self._group.name}' ({total} input file(s))."
            )

            if total == 1:
                status_message = "Reading 1 input file…"
            else:
                status_message = f"Reading {total} input files…"
            self.progress.emit(0, 1, status_message)
            combined_content = self._assemble_combined_content(inputs)
            evidence_ledger = self._build_reduce_evidence_ledger(inputs)

            if self.is_cancelled():
                raise BulkAnalysisCancelled

            override_window = getattr(self._group, "model_context_window", None)
            if isinstance(override_window, int) and override_window > 0:
                from src.common.llm.tokens import TokenCounter

                token_info = TokenCounter.count(
                    text=combined_content,
                    provider=provider_cfg.provider_id,
                    model=provider_cfg.model or "",
                )
                token_count = token_info.get("token_count") if token_info.get("success") else len(combined_content) // 4
                max_tokens = max(int(override_window * 0.5), 4000)
                needs_chunking = token_count > max_tokens
            else:
                needs_chunking, token_count, max_tokens = should_chunk(
                    combined_content, provider_cfg.provider_id, provider_cfg.model
                )
            self.log_message.emit(
                f"Combined content tokens={token_count}, chunking={'yes' if needs_chunking else 'no'}"
            )

            run_details: Dict[str, object] = {
                "token_count": token_count,
                "max_tokens": max_tokens,
                "chunking": bool(needs_chunking),
            }
            input_budget = self._max_input_budget(provider_cfg, max_output_tokens=32_000)
            run_details["input_budget"] = input_budget

            if not needs_chunking:
                prompt = render_user_prompt(
                    bundle,
                    self._metadata,
                    self._group.name,
                    combined_content,
                    placeholder_values=placeholders_global,
                )
                prompt = self._append_citation_ledger(prompt, evidence_ledger)
                result = self._invoke_provider(
                    provider,
                    provider_cfg,
                    prompt,
                    system_prompt,
                    input_budget=input_budget,
                )
                run_details["chunk_count"] = 1
                chunk_state = current_manifest.get("chunks", {"count": 0, "done": [], "checksums": {}})
                chunk_state["count"] = 1
                chunk_state["done"] = [1]
                current_manifest["chunks"] = chunk_state
                recovery_manifest["chunks"] = {
                    "count": 1,
                    "items": {},
                }
            else:
                chunks = generate_chunks(combined_content, max_tokens)
                if not chunks:
                    prompt = render_user_prompt(
                        bundle,
                        self._metadata,
                        self._group.name,
                        combined_content,
                        placeholder_values=placeholders_global,
                    )
                    prompt = self._append_citation_ledger(prompt, evidence_ledger)
                    result = self._invoke_provider(
                        provider,
                        provider_cfg,
                        prompt,
                        system_prompt,
                        input_budget=input_budget,
                        on_response=lambda response: self._record_recovery_actuals(
                            recovery_store,
                            recovery_manifest,
                            response,
                            provider_cfg=provider_cfg,
                            stage="reduce_direct",
                        ),
                    )
                    run_details["chunk_count"] = 1
                    run_details["chunking"] = False
                else:
                    chunk_summaries = []
                    total_chunks = len(chunks)
                    run_details["chunk_count"] = total_chunks
                    chunk_state = current_manifest.get("chunks", {"count": 0, "done": [], "checksums": {}})
                    chunk_state["count"] = total_chunks
                    done_set = set(chunk_state.get("done") or [])
                    recovery_chunk_state = recovery_manifest.setdefault("chunks", {"count": 0, "items": {}})
                    recovery_chunk_state["count"] = total_chunks
                    recovery_items = dict(recovery_chunk_state.get("items") or {})

                    for idx, chunk in enumerate(chunks, start=1):
                        if self.is_cancelled():
                            raise BulkAnalysisCancelled

                        chunk_checksum = _sha256(chunk)
                        chunk_key = str(idx)
                        cached = recovery_store.load_payload(recovery_store.reduce_chunk_path(idx))
                        cached_content = None
                        if cached:
                            valid, reason = recovery_store.validate_payload(
                                payload=cached,
                                expected_input_checksum=chunk_checksum,
                            )
                            if valid and recovery_items.get(chunk_key, {}).get("status") == "complete":
                                cached_content = str(cached.get("content") or "")
                            else:
                                recovery_store.quarantine_reduce_payload(
                                    kind="chunk",
                                    identifier=chunk_key,
                                    payload=cached,
                                    reason=reason or "invalid cached chunk payload",
                                )
                                recovery_items[chunk_key] = {
                                    **dict(recovery_items.get(chunk_key) or {}),
                                    "status": "corrupt",
                                    "input_checksum": chunk_checksum,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                    "quarantine_reason": reason or "invalid cached chunk payload",
                                }

                        if cached_content:
                            summary = cached_content
                            self.log_message.emit(f"Reuse chunk {idx}/{total_chunks} from checkpoint")
                        else:
                            prompt = render_user_prompt(
                                bundle,
                                self._metadata,
                                self._group.name,
                                chunk,
                                chunk_index=idx,
                                chunk_total=total_chunks,
                                placeholder_values=placeholders_global,
                            )
                            chunk_ledger = self._build_reduce_chunk_ledger(
                                chunk=chunk,
                                base_ledger=evidence_ledger,
                            )
                            prompt = self._append_citation_ledger(prompt, chunk_ledger)
                            summary = self._invoke_provider(
                                provider,
                                provider_cfg,
                                prompt,
                                system_prompt,
                                input_budget=input_budget,
                                on_response=lambda response, chunk_index=chunk_key: self._record_recovery_actuals(
                                    recovery_store,
                                    recovery_manifest,
                                    response,
                                    provider_cfg=provider_cfg,
                                    stage="reduce_chunk",
                                    item=recovery_items.setdefault(chunk_index, {}),
                                ),
                            )
                            payload = recovery_store.save_payload(
                                recovery_store.reduce_chunk_path(idx),
                                content=summary,
                                input_checksum=chunk_checksum,
                            )
                            recovery_items[chunk_key] = {
                                **dict(recovery_items.get(chunk_key) or {}),
                                "status": "complete",
                                "input_checksum": payload.get("input_checksum"),
                                "content_checksum": payload.get("content_checksum"),
                                "updated_at": payload.get("updated_at"),
                            }

                        chunk_state.setdefault("checksums", {})[str(idx)] = chunk_checksum
                        done_set.add(idx)
                        chunk_state["done"] = sorted(done_set)
                        current_manifest["chunks"] = chunk_state
                        recovery_chunk_state["items"] = recovery_items
                        recovery_manifest["chunks"] = recovery_chunk_state
                        try:
                            _save_manifest(state_manifest_path, current_manifest)
                        except Exception:
                            self.logger.debug("Failed to persist chunk manifest update", exc_info=True)
                        recovery_store.save_reduce_manifest(recovery_manifest)

                        chunk_summaries.append(summary)

                    # Use hierarchical reduction to combine chunk summaries
                    # This handles large documents that would exceed token limits
                    def invoke_combine(prompt: str) -> str:
                        """Wrapper for provider invocation during hierarchical reduction."""
                        return self._invoke_provider(
                            provider,
                            provider_cfg,
                            prompt,
                            system_prompt,
                            input_budget=input_budget,
                        )

                    def load_batch(level: int, batch_index: int, checksum: str) -> Optional[str]:
                        batch_key = f"{level}:{batch_index}"
                        cached = recovery_store.load_payload(recovery_store.reduce_batch_path(level, batch_index))
                        if cached:
                            valid, reason = recovery_store.validate_payload(payload=cached, expected_input_checksum=checksum)
                            if valid and recovery_manifest.get("batches", {}).get(batch_key, {}).get("status") == "complete":
                                return str(cached.get("content") or "")
                            recovery_store.quarantine_reduce_payload(
                                kind="batch",
                                identifier=batch_key.replace(":", "_"),
                                payload=cached,
                                reason=reason or "invalid cached batch payload",
                            )
                            batches = dict(recovery_manifest.get("batches") or {})
                            batches[batch_key] = {
                                **dict(batches.get(batch_key) or {}),
                                "status": "corrupt",
                                "input_checksum": checksum,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                                "quarantine_reason": reason or "invalid cached batch payload",
                            }
                            recovery_manifest["batches"] = batches
                            recovery_store.save_reduce_manifest(recovery_manifest)
                        return None

                    def save_batch(level: int, batch_index: int, checksum: str, content: str) -> None:
                        payload = recovery_store.save_payload(
                            recovery_store.reduce_batch_path(level, batch_index),
                            content=content,
                            input_checksum=checksum,
                        )
                        batches = current_manifest.setdefault("batches", {})
                        batches[f"{level}:{batch_index}"] = checksum
                        current_manifest["batches"] = batches
                        recovery_batches = dict(recovery_manifest.get("batches") or {})
                        recovery_batches[f"{level}:{batch_index}"] = {
                            "status": "complete",
                            "input_checksum": payload.get("input_checksum"),
                            "content_checksum": payload.get("content_checksum"),
                            "updated_at": payload.get("updated_at"),
                        }
                        recovery_manifest["batches"] = recovery_batches
                        _save_manifest(state_manifest_path, current_manifest)
                        recovery_store.save_reduce_manifest(recovery_manifest)

                    result = combine_chunk_summaries_hierarchical(
                        chunk_summaries,
                        document_name=self._group.name,
                        metadata=self._metadata,
                        placeholder_values=placeholders_global,
                        provider_id=provider_cfg.provider_id,
                        model=provider_cfg.model,
                        invoke_fn=invoke_combine,
                        is_cancelled_fn=self.is_cancelled,
                        load_batch_fn=load_batch,
                        save_batch_fn=save_batch,
                    )

            # Persist
            output_path, run_manifest_path = self._output_paths()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            written_at = datetime.now(timezone.utc)
            metadata = self._build_reduce_metadata(
                output_path=output_path,
                inputs=inputs,
                provider_cfg=provider_cfg,
                prompt_hash=prompt_hash,
                run_details=run_details,
                placeholders=placeholders_global,
                created_at=written_at,
            )
            updated = apply_frontmatter(result, metadata, merge_existing=True)
            output_path.write_text(updated, encoding="utf-8")
            citation_stats = self._record_output_citations(
                output_path=output_path,
                output_text=result,
                prompt_hash=prompt_hash,
            )
            run_manifest = self._build_run_manifest(
                inputs,
                provider_cfg,
                placeholders_global,
                citation_stats=citation_stats,
            )
            run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
            current_manifest["finalized"] = True
            current_manifest["ran_at"] = written_at.isoformat()
            current_manifest["group_id"] = self._group.group_id
            current_manifest["group_slug"] = getattr(self._group, "slug", None) or self._group.folder_name
            _save_manifest(state_manifest_path, current_manifest)
            recovery_manifest["finalized"] = True
            recovery_manifest["ran_at"] = written_at.isoformat()
            recovery_manifest["status"] = "complete"
            recovery_store.save_reduce_manifest(recovery_manifest)

            self.progress.emit(1, 1, "Completed")
            total_cost = self._total_cost(provider_id=provider_cfg.provider_id, model_id=provider_cfg.model)
            if total_cost is not None:
                self.cost_calculated.emit(total_cost, provider_cfg.provider_id, "bulk_combined")
            self.finished.emit(1, 0)

        except BulkAnalysisCancelled:
            if "recovery_store" in locals() and "recovery_manifest" in locals():
                recovery_manifest["status"] = "cancelled"
                recovery_store.save_reduce_manifest(recovery_manifest)
            self.log_message.emit("Combined operation cancelled.")
            self.finished.emit(0, 0)
        except Exception as exc:  # pragma: no cover - defensive
            if "recovery_store" in locals() and "recovery_manifest" in locals():
                recovery_manifest["status"] = "failed"
                recovery_store.save_reduce_manifest(recovery_manifest)
            self.logger.exception("BulkReduceWorker crashed: %s", exc)
            self.log_message.emit(f"Combined operation error: {exc}")
            self.finished.emit(0, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_reduce_metadata(
        self,
        *,
        output_path: Path,
        inputs: Sequence[tuple[str, Path, str]],
        provider_cfg: ProviderConfig,
        prompt_hash: str,
        run_details: Dict[str, object],
        placeholders: Mapping[str, str],
        created_at: datetime,
    ) -> Dict[str, object]:
        project_path = infer_project_path(output_path)
        sources = self._source_references(inputs)
        prompts = self._prompt_references()
        extra: Dict[str, object] = {
            "group_id": self._group.group_id,
            "group_name": self._group.name,
            "group_operation": self._group.operation,
            "prompt_hash": prompt_hash,
            "provider_id": provider_cfg.provider_id,
            "model": provider_cfg.model,
            "input_count": len(inputs),
        }
        if self._estimate_summary:
            extra["cost_estimate"] = dict(self._estimate_summary)
        extra.update(run_details)
        extra["placeholders"] = self._serialise_placeholders(placeholders)
        return build_document_metadata(
            project_path=project_path,
            generator="bulk_reduce_worker",
            created_at=created_at,
            sources=sources,
            prompts=prompts,
            extra=extra,
        )

    def _source_references(self, inputs: Sequence[tuple[str, Path, str]]) -> Sequence[SourceReference]:
        refs: List[SourceReference] = []
        for kind, path, rel in inputs:
            refs.append(
                SourceReference(
                    path=path,
                    relative=rel,
                    kind=(path.suffix.lstrip(".") or "file"),
                    role=kind,
                    checksum=compute_file_checksum(path),
                )
            )
        return refs

    def _prompt_references(self) -> Sequence[PromptReference]:
        references: List[PromptReference] = []
        system_path = (self._group.system_prompt_path or "").strip()
        if system_path:
            references.append(
                PromptReference(
                    path=self._resolve_prompt_path(system_path),
                    role="system",
                )
            )
        else:
            references.append(PromptReference(identifier="document_analysis_system_prompt", role="system"))

        user_path = (self._group.user_prompt_path or "").strip()
        if user_path:
            references.append(
                PromptReference(
                    path=self._resolve_prompt_path(user_path),
                    role="user",
                )
            )
        else:
            references.append(PromptReference(identifier="document_bulk_analysis_prompt", role="user"))

        return [ref for ref in references if ref.to_dict()]

    def _resolve_prompt_path(self, prompt_path: str) -> Path:
        candidate = Path(prompt_path)
        if candidate.is_absolute():
            return candidate
        return (self._project_dir / candidate).resolve()

    def _resolve_inputs(self) -> list[tuple[str, Path, str]]:
        """Return list of (kind, abs_path, rel_key) where kind in {'converted','map'}.

        rel_key is used in the manifest:
          - converted:  "converted/<relative>"
          - map:        "map/<slug>/<relative>"
        """
        items: list[tuple[str, Path, str]] = []

        # Converted documents
        conv_root = self._project_dir / "converted_documents"
        for rel in (self._group.combine_converted_files or []):
            rel = rel.strip("/")
            if not rel:
                continue
            candidate = conv_root / rel
            if is_azure_raw_artifact(candidate):
                continue
            items.append(("converted", candidate, f"converted/{rel}"))
        for rel_dir in (self._group.combine_converted_directories or []):
            rel_dir = rel_dir.strip("/")
            base = conv_root / rel_dir
            if base.exists():
                for f in base.rglob("*.md"):
                    if is_azure_raw_artifact(f):
                        continue
                    items.append(("converted", f, f"converted/{f.relative_to(conv_root).as_posix()}"))

        # Map outputs under bulk_analysis
        for slug in (self._group.combine_map_groups or []):
            slug = slug.strip()
            if not slug:
                continue
            for path, rel in iter_map_outputs(self._project_dir, slug):
                items.append(("map", path, f"map/{slug}/{rel}"))

        for rel_dir in (self._group.combine_map_directories or []):
            rel_dir = rel_dir.strip("/")
            if not rel_dir:
                continue
            parts = rel_dir.split("/", 1)
            if len(parts) != 2:
                continue
            slug, remainder = parts
            slug = slug.strip()
            if not slug:
                continue
            normalized = normalize_map_relative(remainder)
            for path, rel in iter_map_outputs_under(self._project_dir, slug, normalized):
                items.append(("map", path, f"map/{slug}/{rel}"))

        for rel in (self._group.combine_map_files or []):
            rel = rel.strip("/")
            if not rel:
                continue
            parts = rel.split("/", 1)
            if len(parts) != 2:
                continue
            slug, remainder = parts
            slug = slug.strip()
            if not slug:
                continue
            normalized = normalize_map_relative(remainder)
            if not normalized:
                continue
            path = resolve_map_output_path(self._project_dir, slug, normalized)
            items.append(("map", path, f"map/{slug}/{normalized}"))

        # De-duplicate exact same files (by abs path) but keep list order stable
        seen: set[Path] = set()
        deduped: list[tuple[str, Path, str]] = []
        for kind, path, key in items:
            if path in seen:
                continue
            seen.add(path)
            deduped.append((kind, path, key))

        # Apply ordering
        order = (self._group.combine_order or "path").lower()
        if order == "mtime":
            deduped.sort(key=lambda it: it[1].stat().st_mtime if it[1].exists() else 0)
        else:
            deduped.sort(key=lambda it: it[1].as_posix())

        return deduped

    def _assemble_combined_content(self, inputs: Sequence[tuple[str, Path, str]]) -> str:
        parts: list[str] = []
        for _, abs_path, rel_key in inputs:
            if self.is_cancelled():
                raise BulkAnalysisCancelled
            try:
                text = abs_path.read_text(encoding="utf-8")
            except Exception as exc:
                self.file_failed.emit(rel_key, str(exc))
                text = ""
            # HTML comments to avoid altering markdown structure
            parts.append(f"<!--- section-begin: {rel_key} --->\n")
            parts.append(text.rstrip() + "\n")
            parts.append("<!--- section-end --->\n\n")
        return "".join(parts).rstrip() + "\n"

    def _append_citation_ledger(self, prompt: str, ledger: str) -> str:
        if not ledger.strip():
            return prompt
        return f"{prompt.rstrip()}\n\n{ledger.rstrip()}\n"

    def _build_reduce_evidence_ledger(self, inputs: Sequence[tuple[str, Path, str]]) -> str:
        if self._citation_store is None:
            return ""
        converted_relatives: list[str] = []
        converted_prefix = "converted/"
        for kind, _, rel_key in inputs:
            if kind != "converted":
                continue
            if rel_key.startswith(converted_prefix):
                converted_relatives.append(rel_key[len(converted_prefix):])
        converted_relatives = list(dict.fromkeys(converted_relatives))
        if not converted_relatives:
            return ""
        try:
            return self._citation_store.build_evidence_ledger_for_documents(
                relative_paths=converted_relatives,
                max_per_document=30,
                max_total=220,
            )
        except Exception:
            self.logger.debug("Failed to build reduce citation ledger", exc_info=True)
            return ""

    def _build_reduce_chunk_ledger(self, *, chunk: str, base_ledger: str) -> str:
        if not base_ledger.strip():
            return ""
        pages = {int(match.group(1)) for match in _PAGE_MARKER_RE.finditer(chunk)}
        if not pages:
            return base_ledger

        filtered_lines: list[str] = []
        for line in base_ledger.splitlines():
            if "|p" not in line:
                filtered_lines.append(line)
                continue
            page_match = re.search(r"\|p(\d+)\]", line)
            if page_match and int(page_match.group(1)) in pages:
                filtered_lines.append(line)
        if len(filtered_lines) < 4:
            return base_ledger
        return "\n".join(filtered_lines).strip() + "\n"

    def _record_output_citations(
        self,
        *,
        output_path: Path,
        output_text: str,
        prompt_hash: str,
    ) -> CitationRecordStats | None:
        if self._citation_store is None:
            return None
        try:
            stats = self._citation_store.record_output_citations(
                output_path=output_path,
                output_text=output_text,
                generator="bulk_reduce_worker",
                prompt_hash=prompt_hash,
            )
        except Exception:
            self.logger.debug("Failed to record reduce citations", exc_info=True)
            return None

        if stats.total > 0 and (stats.warning > 0 or stats.invalid > 0):
            self.log_message.emit(
                f"Citations {output_path.name}: valid={stats.valid}, "
                f"warning={stats.warning}, invalid={stats.invalid}"
            )
        return stats

    def _resolve_provider(self) -> ProviderConfig:
        provider_id = str(self._group.provider_id or "").strip()
        if not provider_id:
            raise RuntimeError(
                "Bulk analysis group has no saved provider selection. Edit the group and choose a provider."
            )
        model = self._llm_backend.normalize_model(provider_id, self._group.model or None)
        temperature = 0.1
        return ProviderConfig(provider_id=provider_id, model=model, temperature=temperature)

    def _create_provider(self, config: ProviderConfig) -> object:
        return self._llm_backend.create_provider(
            LLMProviderRequest(
                provider_id=config.provider_id,
                model=config.model,
            )
        )

    def _invoke_provider(
        self,
        provider: object,
        provider_cfg: ProviderConfig,
        prompt: str,
        system_prompt: str,
        *,
        max_tokens: int = 32_000,
        input_budget: Optional[int] = None,
        on_response=None,
    ) -> str:
        if self._cancel_event.is_set():
            raise BulkAnalysisCancelled
        stage_input = BulkReduceStageInput(
            group_id=self._group.group_id,
            group_name=self._group.name,
            group_slug=getattr(self._group, "slug", None) or self._group.folder_name,
            transport=backend_transport_name(self._llm_backend),
            provider_id=provider_cfg.provider_id,
            model=provider_cfg.model,
            reasoning=bool(getattr(self._group, "use_reasoning", False)),
            gateway_route=backend_route_name(self._llm_backend),
            max_tokens=max_tokens,
            temperature=provider_cfg.temperature,
        )
        trace_attributes = stage_trace_attributes(stage_input)
        with trace_operation("bulk_reduce.invoke_llm", trace_attributes):
            response = self._llm_backend.invoke_response(
                provider,
                LLMInvocationRequest(
                    prompt=prompt,
                    model=provider_cfg.model,
                    system_prompt=system_prompt,
                    model_settings=self._llm_backend.build_model_settings(
                        provider_cfg.provider_id,
                        provider_cfg.model,
                        temperature=provider_cfg.temperature,
                        max_tokens=max_tokens,
                        use_reasoning=getattr(self._group, "use_reasoning", False),
                    ),
                    input_tokens_limit=input_budget,
                ),
            )
        if on_response is not None:
            on_response(response)
        self._record_response_usage(response)
        content = str(response.text or "").strip()
        if not content:
            raise RuntimeError("LLM returned empty response")
        return content

    def _max_input_budget(self, provider_cfg: ProviderConfig, *, max_output_tokens: int) -> int | None:
        raw_context_window = normalize_context_window_override(
            provider_id=provider_cfg.provider_id,
            model_id=provider_cfg.model,
            context_window=getattr(self._group, "model_context_window", None),
        )
        if not isinstance(raw_context_window, int) or raw_context_window <= 0:
            model_key = provider_cfg.model or provider_cfg.provider_id
            raw_context_window = TokenCounter.get_model_context_window(
                model_key,
                ratio=1.0,
                provider_id=provider_cfg.provider_id,
            )

        return compute_input_token_budget(
            raw_context_window=raw_context_window,
            max_output_tokens=max_output_tokens,
            minimum_budget=_MIN_INPUT_TOKEN_BUDGET,
        )

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M")

    def _record_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage_totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        self._usage_totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)

    def _response_usage_summary(
        self,
        response: object,
        *,
        provider_cfg: ProviderConfig,
    ) -> dict[str, int | float]:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        cost = calculate_usage_cost(
            provider_id=provider_cfg.provider_id,
            model_id=provider_cfg.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": float(cost) if cost is not None else 0.0,
        }

    def _record_recovery_actuals(
        self,
        recovery_store: BulkRecoveryStore,
        recovery_manifest: dict[str, object],
        response: object,
        *,
        provider_cfg: ProviderConfig,
        stage: str,
        item: dict[str, object] | None = None,
    ) -> None:
        usage = self._response_usage_summary(response, provider_cfg=provider_cfg)
        recovery_store.add_actuals(
            recovery_manifest,
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            cost=float(usage["cost"]),
        )
        if item is not None:
            item["actual_usage"] = {
                "input_tokens": int(usage["input_tokens"]),
                "output_tokens": int(usage["output_tokens"]),
            }
            item["actual_cost"] = float(usage["cost"])
            item["last_stage"] = stage

    def _total_cost(self, *, provider_id: str | None, model_id: str | None) -> float | None:
        if not provider_id:
            return None
        amount = calculate_usage_cost(
            provider_id=provider_id,
            model_id=model_id,
            input_tokens=self._usage_totals["input_tokens"],
            output_tokens=self._usage_totals["output_tokens"],
        )
        if amount is None:
            return None
        return float(amount)

    def _output_paths(self) -> tuple[Path, Path]:
        slug = getattr(self._group, "slug", None) or self._group.folder_name
        out_dir = self._project_dir / "bulk_analysis" / slug / "reduce"
        name = (self._group.combine_output_template or "combined_{timestamp}.md").replace(
            "{timestamp}", self._timestamp()
        )
        if not name.endswith(".md"):
            name = name + ".md"
        out_md = out_dir / name
        out_manifest = out_md.with_suffix(".manifest.json")
        return out_md, out_manifest

    def _build_run_manifest(
        self,
        inputs: Sequence[tuple[str, Path, str]],
        provider_cfg: ProviderConfig,
        placeholders: Mapping[str, str],
        *,
        citation_stats: CitationRecordStats | None = None,
    ) -> dict:
        manifest_inputs = []
        for kind, path, rel in inputs:
            try:
                stat_result = path.stat()
                mtime = float(stat_result.st_mtime)
            except OSError:
                mtime = 0.0
            manifest_inputs.append({"kind": kind, "path": rel, "mtime": round(mtime, 6)})

        manifest = {
            "version": 1,
            "group_id": self._group.group_id,
            "group_slug": getattr(self._group, "slug", None) or self._group.folder_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": manifest_inputs,
            "provider": provider_cfg.provider_id,
            "model": provider_cfg.model,
            "temperature": provider_cfg.temperature,
            "system_prompt_path": self._group.system_prompt_path,
            "user_prompt_path": self._group.user_prompt_path,
            "placeholders": self._serialise_placeholders(placeholders),
        }
        if self._estimate_summary:
            manifest["cost_estimate"] = dict(self._estimate_summary)
        if citation_stats is not None:
            manifest["citations"] = {
                "total": citation_stats.total,
                "valid": citation_stats.valid,
                "warning": citation_stats.warning,
                "invalid": citation_stats.invalid,
            }
        return manifest
