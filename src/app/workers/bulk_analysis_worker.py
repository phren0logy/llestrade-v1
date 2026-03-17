"""Worker for executing bulk-analysis runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import frontmatter
from PySide6.QtCore import Signal

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.llm_catalog import calculate_usage_cost
from src.app.core.llm_operation_settings import normalize_context_window_override
from src.app.core.bulk_recovery import BulkRecoveryStore, build_bulk_prompt_state, bulk_prompt_recovery_signature
from src.app.core.bulk_analysis_runner import (
    BulkAnalysisCancelled,
    BulkAnalysisDocument,
    PromptBundle,
    combine_chunk_summaries_hierarchical,
    generate_chunks,
    load_prompts,
    prepare_documents,
    render_system_prompt,
    render_user_prompt,
    should_chunk,
)
from src.app.core.citations import PAGE_MARKER_RE, CitationLedgerEntry, CitationRecordStats, CitationStore
from src.app.core.bulk_prompt_context import build_bulk_placeholders
from src.app.core.placeholders.system import SourceFileContext
from src.app.core.prompt_assembly import append_generated_prompt_section
from src.app.core.project_manager import ProjectMetadata
from src.common.llm.budgets import compute_input_token_budget
from src.common.llm.request_budget import (
    compute_preflight_input_budget,
    count_request_input_tokens,
    estimate_text_input_tokens,
    evaluate_request_budget,
    resolve_request_raw_context_window,
)
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
    PydanticAIGatewayBackend,
    backend_route_name,
    backend_transport_name,
    extract_http_status_error_details,
)
from .progress import WorkerProgressDetail
from .stage_contracts import BulkMapStageInput, stage_trace_attributes


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    model: Optional[str]


@dataclass
class _ProviderPromptLimitError(RuntimeError):
    configured_limit: int | None
    actual_tokens: int | None
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class _InvocationResult:
    response: object
    content: str
    usage: dict[str, int | float]


@dataclass(frozen=True)
class _ChunkTaskSpec:
    chunk_index: int
    chunk_total: int
    chunk_checksum: str
    prompt: str
    system_prompt: str
    context_label: str
    trace_attributes: dict[str, object]


@dataclass(frozen=True)
class _ChunkTaskResult:
    chunk_index: int
    chunk_checksum: str
    summary: str
    usage: dict[str, int | float]


@dataclass
class _GatewayRateLimitError(RuntimeError):
    status_code: int
    retry_after_seconds: float | None
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class _GatewayServerError(RuntimeError):
    status_code: int
    retry_after_seconds: float | None
    message: str

    def __str__(self) -> str:
        return self.message




_MANIFEST_VERSION = 2
_MTIME_TOLERANCE = 1e-6
_DEFAULT_MAX_OUTPUT_TOKENS = 32_000
_GATEWAY_MAP_MAX_OUTPUT_TOKENS = 12_000
_MIN_CHUNK_TOKEN_TARGET = 4_000
_MAX_PROVIDER_LIMIT_RETRIES = 5
_MAX_GATEWAY_RATE_LIMIT_RETRIES = 3
_MAX_GATEWAY_SERVER_RETRIES = 3
_DEFAULT_GATEWAY_RATE_LIMIT_RETRY_DELAY_SECONDS = 15.0
_DEFAULT_CHUNK_FANOUT_CONCURRENCY = 4
_DEFAULT_CHUNK_TARGET_RATIO = 0.50
_DEFAULT_REQUEST_OVERHEAD_SAFETY_MARGIN = 1_024
# Keep extra headroom for gateway Anthropic bulk runs to reduce rate-limit pressure.
_GATEWAY_ANTHROPIC_REQUEST_BUDGET_RATIO = 0.40
_GATEWAY_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504, 524})
_CONFIGURED_LIMIT_RE = re.compile(
    r"configured limit of (?P<limit>\d+) tokens.*?resulted in (?P<actual>\d+) tokens",
    re.IGNORECASE | re.DOTALL,
)
_MAXIMUM_LIMIT_RE = re.compile(
    r"(?P<actual>\d+) tokens?\s*>\s*(?P<limit>\d+)\s*maximum",
    re.IGNORECASE,
)
_INPUT_LIMIT_RE = re.compile(
    r"Exceeded the input_tokens_limit of (?P<limit>\d+) \(input_tokens=(?P<actual>\d+)\)",
    re.IGNORECASE,
)
_LOCAL_PROMPT_BUDGET_RE = re.compile(
    r"Prompt exceeds model input budget for .*?: (?P<actual>\d+) tokens > (?P<limit>\d+) budget",
    re.IGNORECASE,
)

_DYNAMIC_GLOBAL_KEYS: frozenset[str] = frozenset(
    {
        "document_name",
        "document_content",
        "source_pdf_filename",
        "source_pdf_relative_path",
        "source_pdf_absolute_path",
        "source_pdf_absolute_url",
        "reduce_source_list",
        "reduce_source_table",
        "reduce_source_count",
        "chunk_index",
        "chunk_total",
    }
)

_DYNAMIC_DOCUMENT_KEYS: frozenset[str] = frozenset(
    {
        "document_name",
        "document_content",
        "chunk_index",
        "chunk_total",
        "reduce_source_list",
        "reduce_source_table",
        "reduce_source_count",
    }
)


def _manifest_path(project_dir: Path, group: BulkAnalysisGroup) -> Path:
    return project_dir / "bulk_analysis" / group.folder_name / "manifest.json"


def _default_manifest() -> Dict[str, object]:
    return {"version": _MANIFEST_VERSION, "signature": None, "prompt_state": None, "documents": {}}


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return _default_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_manifest()

    if not isinstance(data, dict):
        return _default_manifest()

    documents = data.get("documents", {})
    if not isinstance(documents, dict):
        documents = {}

    return {
        "version": data.get("version", _MANIFEST_VERSION),
        "signature": data.get("signature"),
        "prompt_state": data.get("prompt_state"),
        "documents": documents,
    }


def _save_manifest(path: Path, manifest: Dict[str, object]) -> None:
    payload = {
        "version": _MANIFEST_VERSION,
        "signature": manifest.get("signature"),
        "prompt_state": manifest.get("prompt_state"),
        "documents": manifest.get("documents", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stable_placeholders(placeholders: Mapping[str, str]) -> Dict[str, str]:
    """Filter out volatile placeholder values that should not affect rerun signatures."""
    return {k: v for k, v in placeholders.items() if k != "timestamp"}


def _compute_prompt_hash(
    bundle: PromptBundle,
    provider_config: ProviderConfig,
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
        "provider_id": provider_config.provider_id,
        "model": provider_config.model,
        "group_operation": group.operation,
        "use_reasoning": group.use_reasoning,
        "model_context_window": group.model_context_window,
        "metadata": metadata_summary,
        "placeholder_requirements": group.placeholder_requirements,
    }
    if placeholder_values:
        payload["placeholders"] = {k: placeholder_values.get(k, "") for k in sorted(placeholder_values)}

    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _should_process_document(
    entry: Optional[Dict[str, object]],
    source_mtime: float,
    prompt_hash: str,
    output_exists: bool,
) -> bool:
    if not output_exists:
        return True
    if entry is None:
        return True

    stored_mtime = entry.get("source_mtime")
    stored_hash = entry.get("prompt_hash")

    if stored_mtime is None or stored_hash is None:
        return True

    try:
        if abs(float(stored_mtime) - float(source_mtime)) > _MTIME_TOLERANCE:
            return True
    except (TypeError, ValueError):
        return True

    return stored_hash != prompt_hash


def _map_recovery_needs_work(entry: Mapping[str, object] | None) -> bool:
    if not isinstance(entry, Mapping) or not entry:
        return False
    if str(entry.get("status") or "") != "complete":
        return True
    for item in dict(entry.get("chunks") or {}).values():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status") or "") != "complete":
            return True
    for item in dict(entry.get("batches") or {}).values():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status") or "") != "complete":
            return True
    return False

class BulkAnalysisWorker(DashboardWorker):
    """Run bulk analysis summaries on the thread pool."""

    progress = Signal(int, int, str)  # completed, total, relative path
    progress_detail = Signal(object)
    file_failed = Signal(str, str)  # relative path, error message
    finished = Signal(int, int)  # successes, failures
    log_message = Signal(str)
    cost_calculated = Signal(float, str, str)

    def __init__(
        self,
        *,
        project_dir: Path,
        group: BulkAnalysisGroup,
        files: Sequence[str],
        metadata: Optional[ProjectMetadata],
        default_provider: Tuple[str, Optional[str]] = ("anthropic", None),
        force_rerun: bool = False,
        preserve_recovery_on_signature_mismatch: bool = False,
        placeholder_values: Mapping[str, str] | None = None,
        project_name: str = "",
        estimate_summary: Mapping[str, object] | None = None,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        super().__init__(worker_name="bulk_analysis")

        self._project_dir = project_dir
        self._group = group
        self._files = list(files)
        self._metadata = metadata
        self._default_provider = default_provider
        self._force_rerun = force_rerun
        self._preserve_recovery_on_signature_mismatch = preserve_recovery_on_signature_mismatch
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
        self._prompt_token_cache: dict[str, tuple[int, str]] = {}
        self._provider_input_budget_overrides: dict[tuple[str, str], int] = {}
        self._current_document_index = 0
        self._total_documents = 0
        self._current_document_path: str | None = None

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

    def _emit_progress_detail(
        self,
        *,
        phase: str,
        label: str,
        document_path: str | None = None,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        detail: str | None = None,
        chunks_completed: int | None = None,
        chunks_in_flight: int | None = None,
    ) -> None:
        percent = self._bulk_progress_percent(
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            phase=phase,
            chunks_completed=chunks_completed,
        )
        self.progress_detail.emit(
            WorkerProgressDetail(
                run_kind="bulk_map",
                phase=phase,
                label=label,
                percent=percent,
                completed=max(self._current_document_index - 1, 0),
                total=self._total_documents or None,
                document_path=document_path or self._current_document_path,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                detail=detail,
                chunks_completed=chunks_completed,
                chunks_in_flight=chunks_in_flight,
            )
        )

    def _bulk_progress_percent(
        self,
        *,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        phase: str | None = None,
        chunks_completed: int | None = None,
    ) -> int:
        if self._total_documents <= 0:
            return 0
        doc_position = max(self._current_document_index - 1, 0)
        fraction = float(doc_position)
        if chunk_total:
            if phase == "combining":
                fraction += 0.98
            elif chunks_completed is not None:
                fraction += min(max(chunks_completed, 0), chunk_total) / chunk_total
            elif phase == "chunk_complete":
                fraction += min(max(chunk_index or 0, 0), chunk_total) / chunk_total
            else:
                fraction += max((chunk_index or 1) - 1, 0) / chunk_total
        elif phase == "document_complete":
            fraction += 1.0
        elif phase in {"document_started", "processing_document"}:
            fraction += 0.02
        return max(0, min(int((fraction / self._total_documents) * 100), 100))

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

    def _extract_source_context(self, metadata: Dict[str, object], document: BulkAnalysisDocument) -> SourceFileContext:
        sources = metadata.get("sources")
        if not sources and isinstance(metadata.get("metadata"), dict):
            sources = metadata["metadata"].get("sources")
        if isinstance(sources, list) and sources:
            entry = sources[0] or {}
            if isinstance(entry, dict):
                return self._resolve_source_context(
                    relative_hint=entry.get("relative"),
                    path_hint=entry.get("path"),
                    fallback_relative=document.relative_path,
                )

        rel_path = document.relative_path
        absolute = (self._project_dir / rel_path).resolve()
        return SourceFileContext(absolute_path=absolute, relative_path=rel_path)

    def _load_document(self, document: BulkAnalysisDocument) -> tuple[str, Dict[str, object], SourceFileContext]:
        raw = document.source_path.read_text(encoding="utf-8")
        try:
            post = frontmatter.loads(raw)
            body = post.content or ""
            metadata = dict(post.metadata or {})
        except Exception:
            body = raw
            metadata = {}
        source_context = self._extract_source_context(metadata, document)
        return body, metadata, source_context

    def _run(self) -> None:  # pragma: no cover - executed in worker thread
        provider: Optional[object] = None
        successes = 0
        failures = 0
        skipped = 0
        manifest: Optional[Dict[str, object]] = None
        manifest_path: Optional[Path] = None
        recovery_manifest: Optional[Dict[str, object]] = None
        recovery_store: Optional[BulkRecoveryStore] = None

        try:
            documents = prepare_documents(self._project_dir, self._group, self._files)
            total = len(documents)
            self._total_documents = total
            if total == 0:
                self.log_message.emit("No documents resolved for bulk analysis run.")
                self.logger.info("%s no documents to process", self.job_tag)
                self.finished.emit(0, 0)
                return

            self.logger.info("%s starting bulk analysis (docs=%s)", self.job_tag, total)
            provider_config = self._resolve_provider()
            self.log_message.emit(
                f"Using {backend_transport_name(self._llm_backend)} backend: "
                f"{provider_config.provider_id}/{provider_config.model or '<default>'}"
            )
            bundle = load_prompts(self._project_dir, self._group, self._metadata)
            global_placeholders = self._build_placeholder_map()

            self._enforce_placeholder_requirements(
                global_placeholders,
                context="bulk analysis",
                dynamic_keys=_DYNAMIC_GLOBAL_KEYS,
            )

            system_prompt = render_system_prompt(
                bundle,
                self._metadata,
                placeholder_values=global_placeholders,
            )
            provider = self._create_provider(provider_config)
            if provider is None:
                raise RuntimeError("Bulk analysis provider failed to initialise")

            prompt_hash = _compute_prompt_hash(
                bundle,
                provider_config,
                self._group,
                self._metadata,
                placeholder_values=self._base_placeholders,
            )
            prompt_state = build_bulk_prompt_state(
                self._project_dir,
                self._group,
                system_template=bundle.system_template,
                user_template=bundle.user_template,
            )
            slug = getattr(self._group, "slug", None) or self._group.folder_name
            legacy_checkpoint_mgr = CheckpointManager(
                self._project_dir / "bulk_analysis" / slug / "map" / "checkpoints"
            )
            recovery_store = BulkRecoveryStore(self._project_dir / "bulk_analysis" / slug)
            signature = {
                "prompt_recovery_hash": bulk_prompt_recovery_signature(
                    prompt_state,
                    provider_id=provider_config.provider_id,
                    model=provider_config.model,
                    operation=self._group.operation,
                    use_reasoning=self._group.use_reasoning,
                    model_context_window=self._group.model_context_window,
                    placeholder_requirements=self._group.placeholder_requirements,
                    metadata=self._metadata,
                    placeholder_values=self._base_placeholders,
                ),
                "placeholders": _stable_placeholders(self._serialise_placeholders(global_placeholders)),
            }
            manifest_path = _manifest_path(self._project_dir, self._group)
            manifest = _load_manifest(manifest_path)
            recovery_store.import_legacy_map(
                checkpoint_root=legacy_checkpoint_mgr.base_dir,
                legacy_manifest=manifest,
            )
            recovery_manifest = recovery_store.load_map_manifest()
            if manifest.get("version") != _MANIFEST_VERSION or (
                manifest.get("signature") != signature and not self._preserve_recovery_on_signature_mismatch
            ):
                manifest = _default_manifest()
            if self._force_rerun or (
                recovery_manifest.get("signature") != signature
                and not self._preserve_recovery_on_signature_mismatch
            ):
                recovery_store.clear_map()
                recovery_manifest = recovery_store.load_map_manifest()
            manifest["signature"] = signature
            manifest["prompt_state"] = prompt_state
            recovery_manifest["signature"] = signature
            recovery_manifest["prompt_state"] = prompt_state
            recovery_manifest["status"] = "running"
            entries = manifest.setdefault("documents", {})  # type: ignore[arg-type]
            recovery_manifest.setdefault("documents", {})  # type: ignore[arg-type]

            for index, document in enumerate(documents, start=1):
                if self.is_cancelled():
                    raise BulkAnalysisCancelled
                self._current_document_index = index
                self._current_document_path = document.relative_path

                try:
                    source_mtime = document.source_path.stat().st_mtime
                except FileNotFoundError:
                    source_mtime = 0.0

                entry = entries.get(document.relative_path)
                recovery_entry = (
                    dict((recovery_manifest or {}).get("documents", {}).get(document.relative_path) or {})
                    if recovery_manifest is not None
                    else {}
                )
                output_exists = document.output_path.exists()
                recovery_needs_work = _map_recovery_needs_work(recovery_entry)
                prompt_hash_for_skip = prompt_hash
                if (
                    self._preserve_recovery_on_signature_mismatch
                    and not recovery_needs_work
                    and isinstance(entry, dict)
                    and entry.get("prompt_hash")
                ):
                    prompt_hash_for_skip = str(entry.get("prompt_hash"))
                if (
                    not self._force_rerun
                    and not recovery_needs_work
                    and not _should_process_document(entry, source_mtime, prompt_hash_for_skip, output_exists)
                ):
                    skipped += 1
                    self.log_message.emit(f"Skipping {document.relative_path} (unchanged)")
                    if isinstance(entry, dict):
                        entry["ran_at"] = datetime.now(timezone.utc).isoformat()
                    progress_count = successes + failures + skipped
                    self.logger.debug(
                        "%s progress %s/%s %s",
                        self.job_tag,
                        progress_count,
                        total,
                        document.relative_path,
                    )
                    self.progress.emit(progress_count, total, document.relative_path)
                    continue

                try:
                    self._emit_progress_detail(
                        phase="document_started",
                        label=f"Processing document {index}/{total}",
                        document_path=document.relative_path,
                    )
                    summary, run_details, doc_placeholders = self._process_document(
                        provider,
                        provider_config,
                        bundle,
                        system_prompt,
                        document,
                        global_placeholders,
                        manifest,
                        prompt_hash,
                        manifest_path,
                        recovery_store,
                        recovery_manifest,
                    )
                except BulkAnalysisCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - propagate via signal
                    failures += 1
                    self.logger.exception("%s failed %s", self.job_tag, document.source_path)
                    self.file_failed.emit(document.relative_path, str(exc))
                else:
                    try:
                        document.output_path.parent.mkdir(parents=True, exist_ok=True)
                        written_at = datetime.now(timezone.utc)
                        metadata = self._build_summary_metadata(
                            document,
                            provider_config,
                            prompt_hash,
                            run_details,
                            created_at=written_at,
                        )
                        updated = apply_frontmatter(summary, metadata, merge_existing=True)
                        document.output_path.write_text(updated, encoding="utf-8")
                    except Exception as exc:  # noqa: BLE001 - propagate via signal
                        failures += 1
                        self.logger.exception("%s write failed %s", self.job_tag, document.output_path)
                        self.file_failed.emit(document.relative_path, str(exc))
                    else:
                        citation_stats = self._record_output_citations(
                            output_path=document.output_path,
                            output_text=summary,
                            prompt_hash=prompt_hash,
                            label_mapping=run_details.get("citation_label_mapping"),
                        )
                        successes += 1
                        ran_timestamp = written_at.isoformat()
                        entry_payload: Dict[str, object] = {
                            "source_mtime": round(source_mtime, 6),
                            "prompt_hash": prompt_hash,
                            "ran_at": ran_timestamp,
                            "placeholders": self._serialise_placeholders(doc_placeholders),
                        }
                        if citation_stats is not None:
                            entry_payload["citations"] = {
                                "total": citation_stats.total,
                                "valid": citation_stats.valid,
                                "warning": citation_stats.warning,
                                "invalid": citation_stats.invalid,
                            }
                        entries[document.relative_path] = entry_payload

                        progress_count = successes + failures + skipped
                        self.logger.debug(
                            "%s progress %s/%s %s",
                            self.job_tag,
                            progress_count,
                            total,
                            document.relative_path,
                        )
                        self._emit_progress_detail(
                            phase="document_complete",
                            label=f"Completed document {index}/{total}",
                            document_path=document.relative_path,
                            detail=f"Saved {document.output_path.name}",
                        )
                        self.progress.emit(progress_count, total, document.relative_path)

        except BulkAnalysisCancelled:
            if recovery_manifest is not None:
                recovery_manifest["status"] = "cancelled"
            self.log_message.emit("Bulk analysis run cancelled.")
            self.logger.info("%s cancelled", self.job_tag)
        except Exception as exc:  # pragma: no cover - defensive logging
            if recovery_manifest is not None:
                recovery_manifest["status"] = "failed"
            self.logger.exception("%s worker crashed: %s", self.job_tag, exc)
            self.log_message.emit(f"Bulk analysis worker encountered an error: {exc}")
            failures = max(failures, 1)
        finally:
            if provider and hasattr(provider, "deleteLater"):
                provider.deleteLater()
            if manifest is not None and manifest_path is not None:
                try:
                    _save_manifest(manifest_path, manifest)
                except Exception:
                    self.logger.debug("%s failed to save bulk analysis manifest", self.job_tag, exc_info=True)
            if recovery_store is not None and recovery_manifest is not None:
                try:
                    recovery_store.save_map_manifest(recovery_manifest)
                except Exception:
                    self.logger.debug("%s failed to save bulk analysis recovery manifest", self.job_tag, exc_info=True)
            if skipped:
                self.log_message.emit(f"Skipped {skipped} document(s) (no changes detected)")
            if recovery_manifest is not None and recovery_manifest.get("status") == "running":
                recovery_manifest["status"] = "complete" if failures == 0 and not self.is_cancelled() else "failed"
            total_cost = self._total_cost(
                provider_id=provider_config.provider_id if "provider_config" in locals() else None,
                model_id=provider_config.model if "provider_config" in locals() else None,
            )
            if total_cost is not None:
                self.cost_calculated.emit(total_cost, provider_config.provider_id, "bulk_map")
            self.logger.info("%s finished: successes=%s failures=%s skipped=%s", self.job_tag, successes, failures, skipped)
            self.finished.emit(successes, failures)
    def cancel(self) -> None:
        super().cancel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_document(
        self,
        provider: object,
        provider_config: ProviderConfig,
        bundle: PromptBundle,
        system_prompt: str,
        document: BulkAnalysisDocument,
        global_placeholders: Dict[str, str],
        manifest: Dict[str, object],
        prompt_hash: str,
        manifest_path: Path,
        recovery_store: BulkRecoveryStore | None = None,
        recovery_manifest: Dict[str, object] | None = None,
        checkpoint_mgr: CheckpointManager | None = None,
    ) -> tuple[str, Dict[str, object], Dict[str, str]]:
        _ = checkpoint_mgr
        if recovery_store is None:
            recovery_store = BulkRecoveryStore(self._project_dir / "bulk_analysis" / self._group.folder_name)
        if recovery_manifest is None:
            recovery_manifest = recovery_store.load_map_manifest()
        if self.is_cancelled():
            raise BulkAnalysisCancelled

        body, metadata, source_context = self._load_document(document)
        doc_placeholders = self._build_document_placeholders(global_placeholders, source_context)

        self._enforce_placeholder_requirements(
            doc_placeholders,
            context=f"bulk analysis document '{document.relative_path}'",
            dynamic_keys=_DYNAMIC_DOCUMENT_KEYS,
        )

        raw_context_window = self._resolve_raw_context_window(provider_config)
        map_max_tokens = self._map_max_output_tokens(provider_config)
        needs_chunking, token_count, default_chunk_tokens = should_chunk(
            body,
            provider_config.provider_id,
            provider_config.model,
            raw_context_window=raw_context_window,
        )

        runtime_input_budget = self._effective_input_budget(
            provider_config,
            self._max_input_budget(
                raw_context_window=raw_context_window,
                max_output_tokens=map_max_tokens,
            ),
        )
        input_budget = self._operational_request_budget(provider_config, runtime_input_budget)
        preflight_input_budget = compute_preflight_input_budget(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            runtime_input_budget=input_budget,
            minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
        )
        full_prompt = render_user_prompt(
            bundle,
            self._metadata,
            document.relative_path,
            body,
            placeholder_values=doc_placeholders,
        )
        citation_entries = self._build_document_citation_entries(
            relative_path=document.relative_path,
            content=body,
        )
        full_citation_appendix = self._render_citation_appendix(entries=citation_entries)
        citation_label_mapping = {entry.citation_label: entry.ev_id for entry in citation_entries}
        effective_system_prompt = self._append_citation_appendix(system_prompt, full_citation_appendix)
        full_prompt_tokens = self._count_prompt_tokens(
            provider,
            provider_config,
            effective_system_prompt,
            full_prompt,
            max_tokens=map_max_tokens,
            allow_backend_preflight=False,
        )
        if preflight_input_budget is not None and full_prompt_tokens > preflight_input_budget:
            needs_chunking = True

        run_details: Dict[str, object] = {
            "token_count": token_count,
            "full_prompt_tokens": full_prompt_tokens,
            "max_tokens": default_chunk_tokens,
            "runtime_input_budget_tokens": runtime_input_budget,
            "input_budget_tokens": input_budget,
            "preflight_input_budget_tokens": preflight_input_budget,
            "chunking": bool(needs_chunking),
        }

        self.log_message.emit(
            f"Processing {document.relative_path} ({token_count} tokens, "
            f"prompt={full_prompt_tokens}, chunking={'yes' if needs_chunking else 'no'})"
        )
        self.logger.debug(
            "%s processing %s tokens=%s prompt_tokens=%s chunking=%s",
            self.job_tag,
            document.relative_path,
            token_count,
            full_prompt_tokens,
            'yes' if needs_chunking else 'no',
        )
        prompt_overhead_budget = self._chunk_prompt_overhead_budget(
            provider=provider,
            provider_config=provider_config,
            bundle=bundle,
            system_prompt=effective_system_prompt,
            document=document,
            placeholder_values=doc_placeholders,
            max_tokens=map_max_tokens,
        )
        fallback_chunk_target_tokens: int | None = None

        if not needs_chunking:
            run_details["chunk_count"] = 1
            try:
                self._emit_progress_detail(
                    phase="processing_document",
                    label=f"Running document analysis for {document.relative_path}",
                    document_path=document.relative_path,
                    detail="Single request",
                )
                result = self._invoke_provider(
                    provider,
                    provider_config,
                    full_prompt,
                    effective_system_prompt,
                    max_tokens=map_max_tokens,
                    input_budget=input_budget,
                    preflight_input_budget=preflight_input_budget,
                    context_label=f"document '{document.relative_path}'",
                )
                run_details["citation_label_mapping"] = dict(citation_label_mapping)
                return result, run_details, doc_placeholders
            except _ProviderPromptLimitError as exc:
                learned_limit = self._register_provider_input_budget(provider_config, exc.configured_limit)
                if learned_limit is None:
                    raise
                needs_chunking = True
                run_details["chunking"] = True
                previous_runtime_input_budget = runtime_input_budget
                runtime_input_budget = learned_limit
                input_budget = self._operational_request_budget(provider_config, runtime_input_budget)
                run_details["runtime_input_budget_tokens"] = runtime_input_budget
                run_details["input_budget_tokens"] = input_budget
                preflight_input_budget = compute_preflight_input_budget(
                    provider_id=provider_config.provider_id,
                    model_id=provider_config.model,
                    runtime_input_budget=input_budget,
                    minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
                )
                run_details["preflight_input_budget_tokens"] = preflight_input_budget
                self.log_message.emit(
                    f"Provider reduced effective prompt budget for {document.relative_path}: "
                    f"{previous_runtime_input_budget} -> {learned_limit} tokens"
                )
            except (_GatewayRateLimitError, _GatewayServerError) as exc:
                needs_chunking = True
                run_details["chunking"] = True
                fallback_chunk_target_tokens = self._gateway_fallback_chunk_target(
                    token_count=token_count,
                    request_budget=preflight_input_budget or input_budget,
                    prompt_overhead_budget=prompt_overhead_budget,
                )
                if isinstance(exc, _GatewayRateLimitError):
                    delay_seconds = self._gateway_rate_limit_retry_delay(exc, 1)
                    self.log_message.emit(
                        f"Gateway rate limited {document.relative_path}; "
                        f"retrying in {delay_seconds:.1f}s with chunked processing"
                    )
                    time.sleep(delay_seconds)
                else:
                    self.log_message.emit(
                        f"Gateway server error {exc.status_code} for {document.relative_path}; "
                        f"retrying with chunked processing"
                    )

        chunk_fit_count_mode = self._chunk_fit_count_mode(provider_config)
        chunk_target_tokens = self._initial_chunk_content_budget(
            request_budget=preflight_input_budget or input_budget,
            prompt_overhead_budget=prompt_overhead_budget,
            default_chunk_tokens=default_chunk_tokens,
        )
        if fallback_chunk_target_tokens is not None:
            chunk_target_tokens = fallback_chunk_target_tokens
        if not self._uses_operational_gateway_budget(provider_config):
            chunk_target_tokens = self._chunk_target_from_budget(
                input_budget=preflight_input_budget or input_budget,
                default_chunk_tokens=chunk_target_tokens,
            )
        run_details["request_overhead_tokens"] = prompt_overhead_budget
        run_details["chunk_fit_count_mode"] = chunk_fit_count_mode
        provider_limit_retries = 0
        gateway_rate_limit_retries = 0
        gateway_server_retries = 0
        chunk_fanout_override: int | None = None
        while True:
            runtime_input_budget = self._effective_input_budget(provider_config, runtime_input_budget)
            input_budget = self._operational_request_budget(provider_config, runtime_input_budget)
            run_details["runtime_input_budget_tokens"] = runtime_input_budget
            run_details["input_budget_tokens"] = input_budget
            preflight_input_budget = compute_preflight_input_budget(
                provider_id=provider_config.provider_id,
                model_id=provider_config.model,
                runtime_input_budget=input_budget,
                minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
            )
            run_details["preflight_input_budget_tokens"] = preflight_input_budget
            adjusted_chunk_target = self._initial_chunk_content_budget(
                request_budget=preflight_input_budget or input_budget,
                prompt_overhead_budget=prompt_overhead_budget,
                default_chunk_tokens=chunk_target_tokens,
            )
            if not self._uses_operational_gateway_budget(provider_config):
                adjusted_chunk_target = self._chunk_target_from_budget(
                    input_budget=preflight_input_budget or input_budget,
                    default_chunk_tokens=adjusted_chunk_target,
                )
            if (
                preflight_input_budget is not None
                and input_budget is not None
                and preflight_input_budget != input_budget
            ):
                self.log_message.emit(
                    f"Chunking {document.relative_path} with runtime budget {runtime_input_budget} tokens, "
                    f"request budget {input_budget} tokens, preflight budget {preflight_input_budget} tokens, "
                    f"overhead {prompt_overhead_budget} tokens, and target chunk size "
                    f"{adjusted_chunk_target} tokens ({chunk_fit_count_mode} chunk fit)"
                )
            else:
                self.log_message.emit(
                    f"Chunking {document.relative_path} with request budget {input_budget} tokens, "
                    f"overhead {prompt_overhead_budget} tokens, and target chunk size "
                    f"{adjusted_chunk_target} tokens ({chunk_fit_count_mode} chunk fit)"
                )
            chunks = self._generate_fitting_chunks(
                provider=provider,
                provider_config=provider_config,
                bundle=bundle,
                system_prompt=effective_system_prompt,
                document=document,
                body=body,
                placeholder_values=doc_placeholders,
                input_budget=preflight_input_budget or input_budget,
                initial_chunk_tokens=adjusted_chunk_target,
                max_tokens=map_max_tokens,
                citation_entries=citation_entries,
            )
            if not chunks:
                run_details["chunk_count"] = 1
                run_details["chunking"] = False
                self._emit_progress_detail(
                    phase="processing_document",
                    label=f"Running document analysis for {document.relative_path}",
                    document_path=document.relative_path,
                    detail="Single request",
                )
                result = self._invoke_provider(
                    provider,
                    provider_config,
                    full_prompt,
                    effective_system_prompt,
                    max_tokens=map_max_tokens,
                    input_budget=input_budget,
                    preflight_input_budget=preflight_input_budget,
                    context_label=f"document '{document.relative_path}'",
                )
                run_details["citation_label_mapping"] = dict(citation_label_mapping)
                return result, run_details, doc_placeholders
            try:
                self._emit_progress_detail(
                    phase="chunking",
                    label=f"Split {document.relative_path} into {len(chunks)} chunks",
                    document_path=document.relative_path,
                    chunk_total=len(chunks),
                )
                return self._process_chunked_document(
                    provider=provider,
                    provider_config=provider_config,
                    bundle=bundle,
                    system_prompt=effective_system_prompt,
                    document=document,
                    doc_placeholders=doc_placeholders,
                    manifest=manifest,
                    prompt_hash=prompt_hash,
                    manifest_path=manifest_path,
                    recovery_store=recovery_store,
                    recovery_manifest=recovery_manifest,
                    run_details=run_details,
                    body=body,
                    chunks=chunks,
                    raw_context_window=raw_context_window,
                    input_budget=input_budget,
                    preflight_input_budget=preflight_input_budget or input_budget,
                    map_max_tokens=map_max_tokens,
                    chunk_fanout_override=chunk_fanout_override,
                    citation_entries=citation_entries,
                    citation_label_mapping=citation_label_mapping,
                )
            except _ProviderPromptLimitError as exc:
                provider_limit_retries += 1
                learned_limit = self._register_provider_input_budget(provider_config, exc.configured_limit)
                if provider_limit_retries >= _MAX_PROVIDER_LIMIT_RETRIES:
                    raise RuntimeError(
                        f"Provider repeatedly rejected prompt size for {document.relative_path}: {exc}"
                    ) from exc
                next_target = self._reduced_chunk_target_from_limit_error(
                    adjusted_chunk_target,
                    limit_error=exc,
                    input_budget=input_budget,
                )
                if next_target >= adjusted_chunk_target and learned_limit is None:
                    raise RuntimeError(str(exc)) from exc
                self.log_message.emit(
                    f"Prompt budget reduced for {document.relative_path}: "
                    f"limit={exc.configured_limit or input_budget} actual={exc.actual_tokens or 'unknown'}; "
                    f"reducing chunk target {adjusted_chunk_target} -> {next_target}"
                )
                documents = manifest.setdefault("documents", {})  # type: ignore[assignment]
                if isinstance(documents, dict):
                    documents.pop(document.relative_path, None)
                    _save_manifest(manifest_path, manifest)
                recovery_documents = recovery_manifest.setdefault("documents", {})  # type: ignore[assignment]
                if isinstance(recovery_documents, dict):
                    recovery_documents.pop(document.relative_path, None)
                recovery_store.clear_map_document(document.relative_path)
                recovery_store.save_map_manifest(recovery_manifest)
                chunk_target_tokens = next_target
            except _GatewayRateLimitError as exc:
                gateway_rate_limit_retries += 1
                if gateway_rate_limit_retries >= _MAX_GATEWAY_RATE_LIMIT_RETRIES:
                    raise RuntimeError(
                        f"Gateway repeatedly rate limited {document.relative_path}: {exc}"
                    ) from exc
                chunk_fanout_override = 1
                delay_seconds = self._gateway_rate_limit_retry_delay(exc, gateway_rate_limit_retries)
                self.log_message.emit(
                    f"Gateway rate limited {document.relative_path}; "
                    f"retrying in {delay_seconds:.1f}s with serial chunk processing"
                )
                time.sleep(delay_seconds)
            except _GatewayServerError as exc:
                gateway_server_retries += 1
                if gateway_server_retries >= _MAX_GATEWAY_SERVER_RETRIES:
                    raise RuntimeError(
                        f"Gateway repeatedly failed {document.relative_path} with {exc.status_code}: {exc}"
                    ) from exc
                chunk_fanout_override = 1
                next_target = max(int(adjusted_chunk_target * 0.75), _MIN_CHUNK_TOKEN_TARGET)
                self.log_message.emit(
                    f"Gateway server error {exc.status_code} for {document.relative_path}; "
                    f"retrying in {self._gateway_rate_limit_retry_delay(exc, gateway_server_retries):.1f}s "
                    f"with smaller chunks {adjusted_chunk_target} -> {next_target}"
                )
                documents = manifest.setdefault("documents", {})  # type: ignore[assignment]
                if isinstance(documents, dict):
                    documents.pop(document.relative_path, None)
                    _save_manifest(manifest_path, manifest)
                recovery_documents = recovery_manifest.setdefault("documents", {})  # type: ignore[assignment]
                if isinstance(recovery_documents, dict):
                    recovery_documents.pop(document.relative_path, None)
                recovery_store.clear_map_document(document.relative_path)
                recovery_store.save_map_manifest(recovery_manifest)
                chunk_target_tokens = next_target
                time.sleep(self._gateway_rate_limit_retry_delay(exc, gateway_server_retries))

    def _process_chunked_document(
        self,
        *,
        provider: object,
        provider_config: ProviderConfig,
        bundle: PromptBundle,
        system_prompt: str,
        document: BulkAnalysisDocument,
        doc_placeholders: Dict[str, str],
        manifest: Dict[str, object],
        prompt_hash: str,
        manifest_path: Path,
        recovery_store: BulkRecoveryStore,
        recovery_manifest: Dict[str, object],
        run_details: Dict[str, object],
        body: str,
        chunks: List[str],
        raw_context_window: int,
        input_budget: int,
        preflight_input_budget: int,
        map_max_tokens: int,
        chunk_fanout_override: int | None = None,
        citation_entries: Sequence[CitationLedgerEntry] = (),
        citation_label_mapping: Mapping[str, str] | None = None,
    ) -> tuple[str, Dict[str, object], Dict[str, str]]:
        documents = manifest.setdefault("documents", {})  # type: ignore[assignment]
        entry: Dict[str, object] = dict(documents.get(document.relative_path, {}) or {})
        recovery_documents = recovery_manifest.setdefault("documents", {})  # type: ignore[assignment]
        recovery_entry: Dict[str, object] = dict(recovery_documents.get(document.relative_path, {}) or {})
        source_checksum = _sha256(body)
        total_chunks = len(chunks)
        needs_reset = (
            entry.get("source_checksum") != source_checksum
            or entry.get("prompt_hash") != prompt_hash
            or entry.get("chunk_count") != total_chunks
        )
        if needs_reset:
            entry = {"chunks_done": [], "checksums": {}}
            recovery_store.clear_map_document(document.relative_path)
            recovery_entry = {"chunks": {}, "batches": {}, "status": "incomplete"}

        entry["source_checksum"] = source_checksum
        entry["prompt_hash"] = prompt_hash
        entry["chunk_count"] = total_chunks
        entry.setdefault("chunks_done", [])
        entry.setdefault("checksums", {})
        documents[document.relative_path] = entry
        recovery_entry["source_checksum"] = source_checksum
        recovery_entry["prompt_hash"] = prompt_hash
        recovery_entry["chunk_count"] = total_chunks
        recovery_entry.setdefault("chunks", {})
        recovery_entry.setdefault("batches", {})
        recovery_entry.setdefault("status", "incomplete")
        recovery_documents[document.relative_path] = recovery_entry

        chunk_summaries: List[str | None] = [None] * total_chunks
        run_details["chunk_count"] = total_chunks
        done_set = set(entry.get("chunks_done") or [])
        checksums: Dict[str, str] = dict(entry.get("checksums") or {})
        recovery_chunks: Dict[str, Dict[str, object]] = dict(recovery_entry.get("chunks") or {})
        pending_specs: Deque[_ChunkTaskSpec] = deque()
        completed_chunks = 0

        for idx, chunk in enumerate(chunks, start=1):
            if self.is_cancelled():
                raise BulkAnalysisCancelled

            chunk_checksum = _sha256(chunk)
            chunk_key = str(idx)
            cached = recovery_store.load_payload(recovery_store.map_chunk_path(document.relative_path, idx))
            cached_content = None
            if cached:
                valid, reason = recovery_store.validate_payload(
                    payload=cached,
                    expected_input_checksum=chunk_checksum,
                )
                if valid and recovery_chunks.get(chunk_key, {}).get("status") == "complete":
                    cached_content = str(cached.get("content") or "")
                else:
                    recovery_store.quarantine_map_payload(
                        document_rel=document.relative_path,
                        kind="chunk",
                        identifier=chunk_key,
                        payload=cached,
                        reason=reason or "invalid cached chunk payload",
                    )
                    recovery_chunks[chunk_key] = {
                        **recovery_chunks.get(chunk_key, {}),
                        "status": "corrupt",
                        "input_checksum": chunk_checksum,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "quarantine_reason": reason or "invalid cached chunk payload",
                    }

            if cached_content:
                with trace_operation(
                    "bulk_analysis.chunk",
                    self._chunk_trace_attributes(
                        provider_config=provider_config,
                        document_path=document.relative_path,
                        chunk_index=idx,
                        chunk_total=total_chunks,
                    ),
                ):
                    self._emit_progress_detail(
                        phase="chunk_complete",
                        label=f"Reused chunk {idx}/{total_chunks}",
                        document_path=document.relative_path,
                        chunk_index=idx,
                        chunk_total=total_chunks,
                        detail="Recovered from checkpoint",
                        chunks_completed=completed_chunks + 1,
                        chunks_in_flight=0,
                    )
                    summary = cached_content
                    self.log_message.emit(
                        f"Reusing chunk {idx}/{total_chunks} for {document.relative_path} from checkpoint"
                    )
                self._persist_chunk_completion(
                    document=document,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    recovery_store=recovery_store,
                    recovery_manifest=recovery_manifest,
                    recovery_documents=recovery_documents,
                    recovery_entry=recovery_entry,
                    recovery_chunks=recovery_chunks,
                    entry=entry,
                    documents=documents,
                    done_set=done_set,
                    checksums=checksums,
                    chunk_index=idx,
                    chunk_checksum=chunk_checksum,
                    summary=summary,
                    usage_summary=None,
                )
                chunk_summaries[idx - 1] = summary
                completed_chunks += 1
            else:
                pending_specs.append(
                    self._build_chunk_task_spec(
                        bundle=bundle,
                        document=document,
                        doc_placeholders=doc_placeholders,
                        chunk=chunk,
                        chunk_index=idx,
                        chunk_total=total_chunks,
                        chunk_checksum=chunk_checksum,
                        provider_config=provider_config,
                        system_prompt=system_prompt,
                        citation_entries=citation_entries,
                    )
                )

        chunk_failures: list[Exception] = []
        prompt_limit_error: _ProviderPromptLimitError | None = None
        gateway_rate_limit_error: _GatewayRateLimitError | None = None
        gateway_server_error: _GatewayServerError | None = None
        fanout_limit = (
            min(
                self._effective_chunk_fanout_max_concurrency(
                    provider_config,
                    override=chunk_fanout_override,
                ),
                len(pending_specs),
            )
            if pending_specs
            else 0
        )

        if fanout_limit > 0:
            in_flight: dict[Future[_ChunkTaskResult], _ChunkTaskSpec] = {}
            stop_submitting = False
            with ThreadPoolExecutor(max_workers=fanout_limit, thread_name_prefix="bulk-map-chunk") as executor:
                self._submit_chunk_tasks(
                    executor=executor,
                    max_workers=fanout_limit,
                    in_flight=in_flight,
                    pending_specs=pending_specs,
                    provider_config=provider_config,
                    system_prompt=system_prompt,
                    map_max_tokens=map_max_tokens,
                    input_budget=input_budget,
                    preflight_input_budget=preflight_input_budget,
                    document=document,
                    chunks_completed=completed_chunks,
                )
                while in_flight:
                    done_futures, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        _spec = in_flight.pop(future)
                        try:
                            chunk_result = future.result()
                        except _ProviderPromptLimitError as exc:
                            if prompt_limit_error is None:
                                prompt_limit_error = exc
                                stop_submitting = True
                                pending_specs.clear()
                                self._cancel_pending_chunk_futures(in_flight)
                        except _GatewayRateLimitError as exc:
                            if gateway_rate_limit_error is None:
                                gateway_rate_limit_error = exc
                                stop_submitting = True
                                pending_specs.clear()
                                self._cancel_pending_chunk_futures(in_flight)
                        except _GatewayServerError as exc:
                            if gateway_server_error is None:
                                gateway_server_error = exc
                                stop_submitting = True
                                pending_specs.clear()
                                self._cancel_pending_chunk_futures(in_flight)
                        except BulkAnalysisCancelled:
                            stop_submitting = True
                            pending_specs.clear()
                            self._cancel_pending_chunk_futures(in_flight)
                            if not any(isinstance(error, BulkAnalysisCancelled) for error in chunk_failures):
                                chunk_failures.append(BulkAnalysisCancelled())
                        except Exception as exc:  # noqa: BLE001 - settle in-flight work before failing
                            stop_submitting = True
                            pending_specs.clear()
                            self._cancel_pending_chunk_futures(in_flight)
                            chunk_failures.append(exc)
                        else:
                            if prompt_limit_error is None:
                                self._persist_chunk_completion(
                                    document=document,
                                    manifest=manifest,
                                    manifest_path=manifest_path,
                                    recovery_store=recovery_store,
                                    recovery_manifest=recovery_manifest,
                                    recovery_documents=recovery_documents,
                                    recovery_entry=recovery_entry,
                                    recovery_chunks=recovery_chunks,
                                    entry=entry,
                                    documents=documents,
                                    done_set=done_set,
                                    checksums=checksums,
                                    chunk_index=chunk_result.chunk_index,
                                    chunk_checksum=chunk_result.chunk_checksum,
                                    summary=chunk_result.summary,
                                    usage_summary=chunk_result.usage,
                                )
                                chunk_summaries[chunk_result.chunk_index - 1] = chunk_result.summary
                                completed_chunks += 1
                                self._emit_progress_detail(
                                    phase="chunk_complete",
                                    label=f"Completed {completed_chunks}/{total_chunks} chunks",
                                    document_path=document.relative_path,
                                    chunk_index=chunk_result.chunk_index,
                                    chunk_total=total_chunks,
                                    chunks_completed=completed_chunks,
                                    chunks_in_flight=len(in_flight),
                                    detail=self._chunk_progress_detail_text(
                                        chunks_completed=completed_chunks,
                                        chunk_total=total_chunks,
                                        chunks_in_flight=len(in_flight),
                                    ),
                                )

                    if self.is_cancelled():
                        stop_submitting = True
                        pending_specs.clear()
                        self._cancel_pending_chunk_futures(in_flight)
                        if not any(isinstance(error, BulkAnalysisCancelled) for error in chunk_failures):
                            chunk_failures.append(BulkAnalysisCancelled())

                    if not stop_submitting:
                        self._submit_chunk_tasks(
                            executor=executor,
                            max_workers=fanout_limit,
                            in_flight=in_flight,
                            pending_specs=pending_specs,
                            provider_config=provider_config,
                            system_prompt=system_prompt,
                            map_max_tokens=map_max_tokens,
                            input_budget=input_budget,
                            preflight_input_budget=preflight_input_budget,
                            document=document,
                            chunks_completed=completed_chunks,
                        )

        if prompt_limit_error is not None:
            raise prompt_limit_error
        if gateway_rate_limit_error is not None:
            raise gateway_rate_limit_error
        if gateway_server_error is not None:
            raise gateway_server_error
        if any(isinstance(error, BulkAnalysisCancelled) for error in chunk_failures):
            raise BulkAnalysisCancelled
        if chunk_failures:
            raise chunk_failures[0]

        ordered_chunk_summaries = [summary for summary in chunk_summaries if summary is not None]
        if len(ordered_chunk_summaries) != total_chunks:
            raise RuntimeError(
                f"Incomplete chunk results for {document.relative_path}: "
                f"{len(ordered_chunk_summaries)}/{total_chunks} chunks available"
            )

        def invoke_combine(prompt: str) -> str:
            self._emit_progress_detail(
                phase="combining",
                label=f"Combining {total_chunks} chunk summaries",
                document_path=document.relative_path,
                chunk_index=total_chunks,
                chunk_total=total_chunks,
                chunks_completed=total_chunks,
                chunks_in_flight=0,
            )
            return self._invoke_provider(
                provider,
                provider_config,
                prompt,
                system_prompt,
                input_budget=input_budget,
                preflight_input_budget=preflight_input_budget,
                context_label=f"combine summary for '{document.relative_path}'",
                on_response=lambda response: self._record_recovery_actuals(
                    recovery_store,
                    recovery_manifest,
                    response,
                    provider_config=provider_config,
                    stage="map_batch",
                ),
            )

        def count_combine_prompt_tokens(prompt: str) -> int:
            return self._count_prompt_tokens(
                provider,
                provider_config,
                system_prompt,
                prompt,
                max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
                allow_backend_preflight=not self._defer_exact_backend_preflight(provider_config),
            )

        def load_batch(level: int, batch_index: int, checksum: str) -> Optional[str]:
            batch_key = f"{level}:{batch_index}"
            payload = recovery_store.load_payload(recovery_store.map_batch_path(document.relative_path, level, batch_index))
            if not payload:
                return None
            valid, reason = recovery_store.validate_payload(payload=payload, expected_input_checksum=checksum)
            if valid and recovery_entry.get("batches", {}).get(batch_key, {}).get("status") == "complete":
                return str(payload.get("content") or "")
            recovery_store.quarantine_map_payload(
                document_rel=document.relative_path,
                kind="batch",
                identifier=batch_key.replace(":", "_"),
                payload=payload,
                reason=reason or "invalid cached batch payload",
            )
            batches = dict(recovery_entry.get("batches") or {})
            batches[batch_key] = {
                **dict(batches.get(batch_key) or {}),
                "status": "corrupt",
                "input_checksum": checksum,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "quarantine_reason": reason or "invalid cached batch payload",
            }
            recovery_entry["batches"] = batches
            recovery_documents[document.relative_path] = recovery_entry
            recovery_store.save_map_manifest(recovery_manifest)
            return None

        def save_batch(level: int, batch_index: int, checksum: str, content: str) -> None:
            batch_key = f"{level}:{batch_index}"
            payload = recovery_store.save_payload(
                recovery_store.map_batch_path(document.relative_path, level, batch_index),
                content=content,
                input_checksum=checksum,
            )
            batches = dict(recovery_entry.get("batches") or {})
            batches[batch_key] = {
                "status": "complete",
                "input_checksum": payload.get("input_checksum"),
                "content_checksum": payload.get("content_checksum"),
                "updated_at": payload.get("updated_at"),
            }
            recovery_entry["batches"] = batches
            recovery_documents[document.relative_path] = recovery_entry
            recovery_store.save_map_manifest(recovery_manifest)

        result = combine_chunk_summaries_hierarchical(
            ordered_chunk_summaries,
            document_name=document.relative_path,
            metadata=self._metadata,
            placeholder_values=doc_placeholders,
            provider_id=provider_config.provider_id,
            model=provider_config.model,
            raw_context_window=raw_context_window,
            invoke_fn=invoke_combine,
            is_cancelled_fn=self.is_cancelled,
            load_batch_fn=load_batch,
            save_batch_fn=save_batch,
            input_budget=preflight_input_budget,
            runtime_input_budget=input_budget,
            count_prompt_tokens_fn=count_combine_prompt_tokens,
        )
        entry["status"] = "complete"
        entry["ran_at"] = datetime.now(timezone.utc).isoformat()
        documents[document.relative_path] = entry
        recovery_entry["status"] = "complete"
        recovery_entry["ran_at"] = entry["ran_at"]
        recovery_documents[document.relative_path] = recovery_entry
        _save_manifest(manifest_path, manifest)
        recovery_store.save_map_manifest(recovery_manifest)
        run_details["citation_label_mapping"] = dict(citation_label_mapping or {})
        return result, run_details, doc_placeholders

    def _chunk_fanout_max_concurrency(self) -> int:
        raw = os.getenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "").strip()
        if not raw:
            return _DEFAULT_CHUNK_FANOUT_CONCURRENCY
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_CHUNK_FANOUT_CONCURRENCY
        return value if value > 0 else _DEFAULT_CHUNK_FANOUT_CONCURRENCY

    def _effective_chunk_fanout_max_concurrency(
        self,
        provider_config: ProviderConfig,
        *,
        override: int | None = None,
    ) -> int:
        configured = self._chunk_fanout_max_concurrency()
        explicit_override = os.getenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "").strip()
        if (
            not explicit_override
            and backend_transport_name(self._llm_backend) == "gateway"
            and provider_config.provider_id == "anthropic"
        ):
            configured = 1
        if isinstance(override, int) and override > 0:
            configured = min(configured, override)
        return max(configured, 1)

    @staticmethod
    def _gateway_rate_limit_retry_delay(
        exc: _GatewayRateLimitError | _GatewayServerError,
        attempt: int,
    ) -> float:
        retry_after = exc.retry_after_seconds
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return float(retry_after)
        return min(_DEFAULT_GATEWAY_RATE_LIMIT_RETRY_DELAY_SECONDS * max(attempt, 1), 60.0)

    def _chunk_trace_attributes(
        self,
        *,
        provider_config: ProviderConfig,
        document_path: str,
        chunk_index: int,
        chunk_total: int,
    ) -> dict[str, object]:
        return {
            "llestrade.transport": backend_transport_name(self._llm_backend),
            "llestrade.provider_id": provider_config.provider_id,
            "llestrade.model": provider_config.model,
            "llestrade.worker": "bulk_analysis",
            "llestrade.stage": "bulk_map_chunk",
            "llestrade.group_id": self._group.group_id,
            "llestrade.group_name": self._group.name,
            "llestrade.group_slug": getattr(self._group, "slug", None) or self._group.folder_name,
            "llestrade.document_path": document_path,
            "llestrade.chunk_index": chunk_index,
            "llestrade.chunk_total": chunk_total,
        }

    def _build_chunk_task_spec(
        self,
        *,
        bundle: PromptBundle,
        document: BulkAnalysisDocument,
        doc_placeholders: Mapping[str, str],
        chunk: str,
        chunk_index: int,
        chunk_total: int,
        chunk_checksum: str,
        provider_config: ProviderConfig,
        system_prompt: str,
        citation_entries: Sequence[CitationLedgerEntry],
    ) -> _ChunkTaskSpec:
        chunk_prompt = render_user_prompt(
            bundle,
            self._metadata,
            document.relative_path,
            chunk,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            placeholder_values=doc_placeholders,
        )
        chunk_entries = self._filter_citation_entries_for_content(
            entries=citation_entries,
            content=chunk,
        )
        chunk_system_prompt = self._append_citation_appendix(
            system_prompt,
            self._render_citation_appendix(entries=chunk_entries),
        )
        return _ChunkTaskSpec(
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            chunk_checksum=chunk_checksum,
            prompt=chunk_prompt,
            system_prompt=chunk_system_prompt,
            context_label=f"chunk {chunk_index}/{chunk_total} for '{document.relative_path}'",
            trace_attributes=self._chunk_trace_attributes(
                provider_config=provider_config,
                document_path=document.relative_path,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
            ),
        )

    def _submit_chunk_tasks(
        self,
        *,
        executor: ThreadPoolExecutor,
        max_workers: int,
        in_flight: dict[Future[_ChunkTaskResult], _ChunkTaskSpec],
        pending_specs: Deque[_ChunkTaskSpec],
        provider_config: ProviderConfig,
        system_prompt: str,
        map_max_tokens: int,
        input_budget: int,
        preflight_input_budget: int,
        document: BulkAnalysisDocument,
        chunks_completed: int,
    ) -> None:
        while pending_specs and len(in_flight) < max_workers and not self.is_cancelled():
            spec = pending_specs.popleft()
            future = executor.submit(
                self._execute_chunk_task,
                provider_config=provider_config,
                spec=spec,
                map_max_tokens=map_max_tokens,
                input_budget=input_budget,
                preflight_input_budget=preflight_input_budget,
            )
            in_flight[future] = spec
            self._emit_progress_detail(
                phase="chunk_started",
                label="Processing chunks",
                document_path=document.relative_path,
                chunk_index=spec.chunk_index,
                chunk_total=spec.chunk_total,
                chunks_completed=chunks_completed,
                chunks_in_flight=len(in_flight),
                detail=self._chunk_progress_detail_text(
                    chunks_completed=chunks_completed,
                    chunk_total=spec.chunk_total,
                    chunks_in_flight=len(in_flight),
                ),
            )

    def _execute_chunk_task(
        self,
        *,
        provider_config: ProviderConfig,
        spec: _ChunkTaskSpec,
        map_max_tokens: int,
        input_budget: int,
        preflight_input_budget: int,
    ) -> _ChunkTaskResult:
        provider = self._create_provider(provider_config)
        try:
            with trace_operation("bulk_analysis.chunk", spec.trace_attributes):
                invocation = self._invoke_provider_result(
                    provider,
                    provider_config,
                    spec.prompt,
                    spec.system_prompt,
                    max_tokens=map_max_tokens,
                    input_budget=input_budget,
                    preflight_input_budget=preflight_input_budget,
                    context_label=spec.context_label,
                )
            return _ChunkTaskResult(
                chunk_index=spec.chunk_index,
                chunk_checksum=spec.chunk_checksum,
                summary=invocation.content,
                usage=invocation.usage,
            )
        finally:
            if provider is not None and hasattr(provider, "deleteLater"):
                provider.deleteLater()

    @staticmethod
    def _cancel_pending_chunk_futures(in_flight: Mapping[Future[Any], _ChunkTaskSpec]) -> None:
        for future in in_flight:
            future.cancel()

    @staticmethod
    def _chunk_progress_detail_text(
        *,
        chunks_completed: int,
        chunk_total: int,
        chunks_in_flight: int,
    ) -> str:
        detail = f"Completed {chunks_completed}/{chunk_total} chunks"
        if chunks_in_flight > 0:
            detail += f", {chunks_in_flight} in flight"
        return detail

    def _persist_chunk_completion(
        self,
        *,
        document: BulkAnalysisDocument,
        manifest: Dict[str, object],
        manifest_path: Path,
        recovery_store: BulkRecoveryStore,
        recovery_manifest: Dict[str, object],
        recovery_documents: Dict[str, object],
        recovery_entry: Dict[str, object],
        recovery_chunks: Dict[str, Dict[str, object]],
        entry: Dict[str, object],
        documents: Dict[str, object],
        done_set: set[int],
        checksums: Dict[str, str],
        chunk_index: int,
        chunk_checksum: str,
        summary: str,
        usage_summary: Mapping[str, int | float] | None,
    ) -> None:
        chunk_key = str(chunk_index)
        if usage_summary is not None:
            self._record_usage_summary(usage_summary)
        payload = recovery_store.save_payload(
            recovery_store.map_chunk_path(document.relative_path, chunk_index),
            content=summary,
            input_checksum=chunk_checksum,
        )
        recovery_chunks[chunk_key] = {
            **recovery_chunks.get(chunk_key, {}),
            "status": "complete",
            "input_checksum": payload.get("input_checksum"),
            "content_checksum": payload.get("content_checksum"),
            "updated_at": payload.get("updated_at"),
        }
        if usage_summary is not None:
            self._record_recovery_usage_summary(
                recovery_store,
                recovery_manifest,
                usage_summary,
                stage="map_chunk",
                item=recovery_chunks[chunk_key],
            )
        checksums[chunk_key] = chunk_checksum
        done_set.add(chunk_index)
        entry["checksums"] = checksums
        entry["chunks_done"] = sorted(done_set)
        documents[document.relative_path] = entry
        recovery_entry["chunks"] = recovery_chunks
        recovery_documents[document.relative_path] = recovery_entry
        try:
            _save_manifest(manifest_path, manifest)
        except Exception:
            self.logger.debug("Failed to persist map chunk manifest update", exc_info=True)
        recovery_store.save_map_manifest(recovery_manifest)

    def _max_input_budget(self, *, raw_context_window: int, max_output_tokens: int) -> int:
        budget = compute_input_token_budget(
            raw_context_window=raw_context_window,
            max_output_tokens=max_output_tokens,
            minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
        )
        return budget or _MIN_CHUNK_TOKEN_TARGET

    def _resolve_raw_context_window(self, provider_config: ProviderConfig) -> int:
        explicit_context_window = normalize_context_window_override(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            context_window=getattr(self._group, "model_context_window", None),
            transport=backend_transport_name(self._llm_backend),
        )
        raw_context_window = resolve_request_raw_context_window(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            explicit_context_window=explicit_context_window,
            transport=backend_transport_name(self._llm_backend),
        )
        if isinstance(raw_context_window, int) and raw_context_window > 0:
            return raw_context_window
        raise RuntimeError(
            f"Unknown context window for provider={provider_config.provider_id} "
            f"model={provider_config.model or ''}. Choose a preset model with metadata or enter a context window."
        )

    @staticmethod
    def _chunk_target_from_budget(*, input_budget: int, default_chunk_tokens: int) -> int:
        return max(
            min(default_chunk_tokens, int(input_budget * _DEFAULT_CHUNK_TARGET_RATIO)),
            _MIN_CHUNK_TOKEN_TARGET,
        )

    def _operational_request_budget(self, provider_config: ProviderConfig, runtime_input_budget: int) -> int:
        budget = runtime_input_budget
        if self._uses_operational_gateway_budget(provider_config):
            budget = int(runtime_input_budget * _GATEWAY_ANTHROPIC_REQUEST_BUDGET_RATIO)
        return max(min(runtime_input_budget, budget), _MIN_CHUNK_TOKEN_TARGET)

    def _uses_operational_gateway_budget(self, provider_config: ProviderConfig) -> bool:
        return (
            backend_transport_name(self._llm_backend) == "gateway"
            and provider_config.provider_id in {"anthropic", "anthropic_bedrock"}
        )

    @staticmethod
    def _initial_chunk_content_budget(
        *,
        request_budget: int,
        prompt_overhead_budget: int,
        default_chunk_tokens: int,
    ) -> int:
        available = max(request_budget - prompt_overhead_budget, _MIN_CHUNK_TOKEN_TARGET)
        return max(min(default_chunk_tokens, available), _MIN_CHUNK_TOKEN_TARGET)

    def _chunk_chars_per_token(self, provider_config: ProviderConfig) -> int:
        if provider_config.provider_id in {"anthropic", "anthropic_bedrock"}:
            return 3
        return 4

    def _chunk_prompt_overhead_budget(
        self,
        *,
        provider: object,
        provider_config: ProviderConfig,
        bundle: PromptBundle,
        system_prompt: str,
        document: BulkAnalysisDocument,
        placeholder_values: Mapping[str, str],
        max_tokens: int,
    ) -> int:
        overhead_prompt = render_user_prompt(
            bundle,
            self._metadata,
            document.relative_path,
            "",
            chunk_index=1,
            chunk_total=1,
            placeholder_values=placeholder_values,
        )
        overhead_tokens = self._count_prompt_tokens(
            provider,
            provider_config,
            system_prompt,
            overhead_prompt,
            max_tokens=max_tokens,
            allow_backend_preflight=False,
        )
        return overhead_tokens + _DEFAULT_REQUEST_OVERHEAD_SAFETY_MARGIN

    def _map_max_output_tokens(self, provider_config: ProviderConfig) -> int:
        if self._uses_operational_gateway_budget(provider_config):
            return _GATEWAY_MAP_MAX_OUTPUT_TOKENS
        return _DEFAULT_MAX_OUTPUT_TOKENS

    @staticmethod
    def _gateway_fallback_chunk_target(
        *,
        token_count: int,
        request_budget: int,
        prompt_overhead_budget: int,
    ) -> int:
        available = max(request_budget - prompt_overhead_budget, _MIN_CHUNK_TOKEN_TARGET)
        reduced = max(token_count // 2, _MIN_CHUNK_TOKEN_TARGET)
        return max(min(available, reduced), _MIN_CHUNK_TOKEN_TARGET)

    def _chunk_fit_count_mode(self, provider_config: ProviderConfig) -> str:
        capabilities = self._llm_backend.capabilities(
            provider_config.provider_id,
            provider_config.model,
        )
        if isinstance(self._llm_backend, PydanticAIGatewayBackend):
            return "estimate"
        if capabilities.supports_pre_request_token_count:
            return "exact"
        return "estimate"

    def _effective_input_budget(self, provider_config: ProviderConfig, input_budget: int) -> int:
        override = self._provider_input_budget_overrides.get(
            (provider_config.provider_id, provider_config.model or "")
        )
        if override is None:
            return input_budget
        return max(min(input_budget, override), _MIN_CHUNK_TOKEN_TARGET)

    def _register_provider_input_budget(
        self,
        provider_config: ProviderConfig,
        configured_limit: int | None,
    ) -> int | None:
        if configured_limit is None or configured_limit <= 0:
            return None
        key = (provider_config.provider_id, provider_config.model or "")
        current = self._provider_input_budget_overrides.get(key)
        next_limit = configured_limit if current is None else min(current, configured_limit)
        self._provider_input_budget_overrides[key] = max(next_limit, _MIN_CHUNK_TOKEN_TARGET)
        return self._provider_input_budget_overrides[key]

    @staticmethod
    def _parse_provider_prompt_limit_error(exc: Exception) -> _ProviderPromptLimitError | None:
        body = getattr(exc, "body", None)
        message_parts = [str(exc)]
        if isinstance(body, str):
            message_parts.append(body)
        elif isinstance(body, Mapping):
            try:
                message_parts.append(json.dumps(body, sort_keys=True))
            except Exception:
                message_parts.append(str(body))

        haystack = "\n".join(part for part in message_parts if part)
        if not haystack:
            return None

        match = _CONFIGURED_LIMIT_RE.search(haystack)
        if match:
            return _ProviderPromptLimitError(
                configured_limit=int(match.group("limit")),
                actual_tokens=int(match.group("actual")),
                message=haystack,
            )

        match = _MAXIMUM_LIMIT_RE.search(haystack)
        if match:
            return _ProviderPromptLimitError(
                configured_limit=int(match.group("limit")),
                actual_tokens=int(match.group("actual")),
                message=haystack,
            )

        match = _INPUT_LIMIT_RE.search(haystack)
        if match:
            return _ProviderPromptLimitError(
                configured_limit=int(match.group("limit")),
                actual_tokens=int(match.group("actual")),
                message=haystack,
            )

        match = _LOCAL_PROMPT_BUDGET_RE.search(haystack)
        if match:
            return _ProviderPromptLimitError(
                configured_limit=int(match.group("limit")),
                actual_tokens=int(match.group("actual")),
                message=haystack,
            )

        return None

    @staticmethod
    def _reduced_chunk_target_from_limit_error(
        current_target: int,
        *,
        limit_error: _ProviderPromptLimitError,
        input_budget: int,
    ) -> int:
        if limit_error.actual_tokens and limit_error.configured_limit and limit_error.actual_tokens > 0:
            scale = (limit_error.configured_limit / limit_error.actual_tokens) * 0.9
            if scale < 1.0:
                reduced = int(current_target * scale)
                return max(min(reduced, current_target - 1), _MIN_CHUNK_TOKEN_TARGET)

        fallback_limit = limit_error.configured_limit or input_budget
        reduced = int(min(current_target, fallback_limit) * _DEFAULT_CHUNK_TARGET_RATIO)
        return max(min(reduced, current_target - 1), _MIN_CHUNK_TOKEN_TARGET)

    def _count_prompt_tokens_with_mode(
        self,
        provider: object,
        provider_config: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        allow_backend_preflight: bool = True,
    ) -> tuple[int, str]:
        combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
        if not combined_prompt:
            return 0, "estimate"

        cache_key = self._prompt_token_cache_key(
            provider_id=provider_config.provider_id,
            model=provider_config.model,
            combined_prompt=combined_prompt,
            max_tokens=max_tokens,
            use_reasoning=bool(getattr(self._group, "use_reasoning", False)),
            mode="backend" if allow_backend_preflight else "estimate",
        )
        cached = self._prompt_token_cache.get(cache_key)
        if cached is not None:
            return cached

        def _exact_token_counter() -> int | None:
            try:
                return self._llm_backend.count_input_tokens(
                    provider,
                    LLMInvocationRequest(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=provider_config.model,
                        model_settings=self._llm_backend.build_model_settings(
                            provider_config.provider_id,
                            provider_config.model,
                            temperature=0.1,
                            max_tokens=max_tokens,
                            use_reasoning=bool(getattr(self._group, "use_reasoning", False)),
                            reasoning=getattr(self._group, "reasoning", {}),
                        ),
                    )
                )
            except Exception:
                self.logger.debug("Backend token preflight failed; falling back to local estimate", exc_info=True)
                return None

        result, mode = count_request_input_tokens(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
            exact_token_counter=_exact_token_counter if allow_backend_preflight else None,
        )
        cached_value = (result, mode)
        self._prompt_token_cache[cache_key] = cached_value
        return cached_value

    def _count_prompt_tokens(
        self,
        provider: object,
        provider_config: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        allow_backend_preflight: bool = True,
    ) -> int:
        result, _mode = self._count_prompt_tokens_with_mode(
            provider,
            provider_config,
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            allow_backend_preflight=allow_backend_preflight,
        )
        return result

    def _estimate_prompt_tokens(self, *, text: str, provider: str, model: str) -> int:
        return estimate_text_input_tokens(
            text=text,
            provider_id=provider,
            model_id=model,
        )

    def _prompt_token_cache_key(
        self,
        *,
        provider_id: str,
        model: str | None,
        combined_prompt: str,
        max_tokens: int,
        use_reasoning: bool,
        mode: str,
    ) -> str:
        payload = {
            "provider_id": provider_id,
            "model": model or "",
            "prompt_sha": _sha256(combined_prompt),
            "max_tokens": max_tokens,
            "use_reasoning": use_reasoning,
            "mode": mode,
        }
        return _sha256(json.dumps(payload, sort_keys=True))

    def _defer_exact_backend_preflight(self, provider_config: ProviderConfig) -> bool:
        if not isinstance(self._llm_backend, (PydanticAIDirectBackend, PydanticAIGatewayBackend)):
            return False
        return self._llm_backend.capabilities(
            provider_config.provider_id,
            provider_config.model,
        ).supports_pre_request_token_count

    def _generate_fitting_chunks(
        self,
        *,
        provider: object,
        provider_config: ProviderConfig,
        bundle: PromptBundle,
        system_prompt: str,
        document: BulkAnalysisDocument,
        body: str,
        placeholder_values: Mapping[str, str],
        input_budget: int,
        initial_chunk_tokens: int,
        max_tokens: int,
        citation_entries: Sequence[CitationLedgerEntry] = (),
    ) -> List[str]:
        chunk_target = max(initial_chunk_tokens, _MIN_CHUNK_TOKEN_TARGET)
        allow_backend_preflight = self._chunk_fit_count_mode(provider_config) == "exact"
        chars_per_token = self._chunk_chars_per_token(provider_config)
        while True:
            chunks = generate_chunks(body, chunk_target, chars_per_token=chars_per_token)
            if not chunks:
                return []

            total_chunks = len(chunks)
            oversized: List[tuple[int, int]] = []
            for idx, chunk in enumerate(chunks, start=1):
                prompt = render_user_prompt(
                    bundle,
                    self._metadata,
                    document.relative_path,
                    chunk,
                    chunk_index=idx,
                    chunk_total=total_chunks,
                    placeholder_values=placeholder_values,
                )
                chunk_entries = self._filter_citation_entries_for_content(
                    entries=citation_entries,
                    content=chunk,
                )
                chunk_system_prompt = self._append_citation_appendix(
                    system_prompt,
                    self._render_citation_appendix(entries=chunk_entries),
                )
                prompt_tokens, count_mode = self._count_prompt_tokens_with_mode(
                    provider,
                    provider_config,
                    chunk_system_prompt,
                    prompt,
                    max_tokens=max_tokens,
                    allow_backend_preflight=allow_backend_preflight,
                )
                if prompt_tokens > input_budget:
                    oversized.append((idx, prompt_tokens))
                    if count_mode == "exact":
                        self.logger.debug(
                            "%s exact chunk-fit count exceeded budget for %s chunk=%s/%s tokens=%s budget=%s",
                            self.job_tag,
                            document.relative_path,
                            idx,
                            total_chunks,
                            prompt_tokens,
                            input_budget,
                        )
                    else:
                        self.logger.debug(
                            "%s estimated chunk request for %s chunk=%s/%s tokens=%s budget=%s",
                            self.job_tag,
                            document.relative_path,
                            idx,
                            total_chunks,
                            prompt_tokens,
                            input_budget,
                        )

            if not oversized:
                return chunks

            if chunk_target <= _MIN_CHUNK_TOKEN_TARGET:
                worst_chunk, worst_tokens = max(oversized, key=lambda item: item[1])
                raise RuntimeError(
                    f"Prompt exceeds model input budget for {document.relative_path} "
                    f"(chunk {worst_chunk}: {worst_tokens} tokens > {input_budget} budget). "
                    "Try a model with a larger context window or simplify the prompt template."
                )

            previous_target = chunk_target
            chunk_target = max(int(chunk_target * _DEFAULT_CHUNK_TARGET_RATIO), _MIN_CHUNK_TOKEN_TARGET)
            self.log_message.emit(
                f"Reducing chunk target for {document.relative_path}: {previous_target} -> {chunk_target} tokens"
            )

    def _invoke_provider(
        self,
        provider: object,
        provider_config: ProviderConfig,
        prompt: str,
        system_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        input_budget: Optional[int] = None,
        preflight_input_budget: Optional[int] = None,
        context_label: str = "bulk analysis request",
        on_response=None,
    ) -> str:
        result = self._invoke_provider_result(
            provider,
            provider_config,
            prompt,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            input_budget=input_budget,
            preflight_input_budget=preflight_input_budget,
            context_label=context_label,
        )
        if on_response is not None:
            on_response(result.response)
        self._record_usage_summary(result.usage)
        return result.content

    def _invoke_provider_result(
        self,
        provider: object,
        provider_config: ProviderConfig,
        prompt: str,
        system_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        input_budget: Optional[int] = None,
        preflight_input_budget: Optional[int] = None,
        context_label: str = "bulk analysis request",
    ) -> _InvocationResult:
        if self._cancel_event.is_set():
            raise BulkAnalysisCancelled

        if input_budget is not None:
            defer_exact_backend_preflight = self._defer_exact_backend_preflight(provider_config)

            def _exact_token_counter() -> int | None:
                return self._llm_backend.count_input_tokens(
                    provider,
                    LLMInvocationRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=provider_config.model,
                        model_settings=self._llm_backend.build_model_settings(
                            provider_config.provider_id,
                            provider_config.model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            use_reasoning=getattr(self._group, "use_reasoning", False),
                            reasoning=getattr(self._group, "reasoning", {}),
                        ),
                    ),
                )

            evaluation = evaluate_request_budget(
                provider_id=provider_config.provider_id,
                model_id=provider_config.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_output_tokens=max_tokens,
                minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
                runtime_input_budget_limit=input_budget,
                transport=backend_transport_name(self._llm_backend),
                exact_token_counter=_exact_token_counter if not defer_exact_backend_preflight else None,
            )
            if not evaluation.fits:
                comparison_budget = (
                    evaluation.preflight_input_budget
                    if evaluation.preflight_input_budget is not None
                    else preflight_input_budget if preflight_input_budget is not None else input_budget
                )
                raise _ProviderPromptLimitError(
                    configured_limit=int(comparison_budget) if comparison_budget is not None else input_budget,
                    actual_tokens=evaluation.input_tokens,
                    message=(
                        f"Prompt exceeds model input budget for {context_label}: "
                        f"{evaluation.input_tokens} tokens > {comparison_budget} budget"
                    ),
                )

        stage_input = BulkMapStageInput(
            group_id=self._group.group_id,
            group_name=self._group.name,
            group_slug=getattr(self._group, "slug", None) or self._group.folder_name,
            transport=backend_transport_name(self._llm_backend),
            provider_id=provider_config.provider_id,
            model=provider_config.model,
            reasoning=bool(getattr(self._group, "use_reasoning", False)),
            gateway_route=backend_route_name(self._llm_backend),
            context_label=context_label,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        trace_attributes = stage_trace_attributes(stage_input)
        with trace_operation("bulk_analysis.invoke_llm", trace_attributes):
            try:
                response = self._llm_backend.invoke_response(
                    provider,
                    LLMInvocationRequest(
                        prompt=prompt,
                        model=provider_config.model,
                        system_prompt=system_prompt,
                        model_settings=self._llm_backend.build_model_settings(
                            provider_config.provider_id,
                            provider_config.model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            use_reasoning=getattr(self._group, "use_reasoning", False),
                            reasoning=getattr(self._group, "reasoning", {}),
                        ),
                        input_tokens_limit=input_budget,
                    ),
                )
            except Exception as exc:
                parsed_limit_error = self._parse_provider_prompt_limit_error(exc)
                if parsed_limit_error is not None:
                    raise parsed_limit_error from exc
                gateway_error = extract_http_status_error_details(exc)
                if (
                    backend_transport_name(self._llm_backend) == "gateway"
                    and gateway_error is not None
                    and gateway_error.status_code in _GATEWAY_RETRYABLE_STATUS_CODES
                ):
                    if gateway_error.status_code == 429:
                        raise _GatewayRateLimitError(
                            status_code=gateway_error.status_code,
                            retry_after_seconds=gateway_error.retry_after_seconds,
                            message=f"Gateway rate limit for {context_label}: {gateway_error.message}",
                        ) from exc
                    raise _GatewayServerError(
                        status_code=gateway_error.status_code,
                        retry_after_seconds=gateway_error.retry_after_seconds,
                        message=f"Gateway server error for {context_label}: {gateway_error.message}",
                    ) from exc
                raise
        content = str(response.text or "").strip()
        if not content:
            raise RuntimeError("LLM returned empty response")
        return _InvocationResult(
            response=response,
            content=content,
            usage=self._response_usage_summary(response, provider_config=provider_config),
        )

    def _record_usage_summary(self, usage_summary: Mapping[str, int | float]) -> None:
        self._usage_totals["input_tokens"] += int(usage_summary.get("input_tokens", 0) or 0)
        self._usage_totals["output_tokens"] += int(usage_summary.get("output_tokens", 0) or 0)

    def _response_usage_summary(
        self,
        response: object,
        *,
        provider_config: ProviderConfig,
    ) -> dict[str, int | float]:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        cost = calculate_usage_cost(
            provider_id=provider_config.provider_id,
            model_id=provider_config.model,
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
        provider_config: ProviderConfig,
        stage: str,
        item: dict[str, object] | None = None,
    ) -> None:
        usage = self._response_usage_summary(response, provider_config=provider_config)
        self._record_recovery_usage_summary(
            recovery_store,
            recovery_manifest,
            usage,
            stage=stage,
            item=item,
        )

    def _record_recovery_usage_summary(
        self,
        recovery_store: BulkRecoveryStore,
        recovery_manifest: dict[str, object],
        usage: Mapping[str, int | float],
        *,
        stage: str,
        item: dict[str, object] | None = None,
    ) -> None:
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

    def _resolve_provider(self) -> ProviderConfig:
        provider_id = str(self._group.provider_id or "").strip()
        if not provider_id:
            raise RuntimeError(
                "Bulk analysis group has no saved provider selection. Edit the group and choose a provider."
            )
        model = self._llm_backend.normalize_model(
            provider_id,
            self._group.model or self._default_provider[1],
        )
        return ProviderConfig(provider_id=provider_id, model=model)

    def _create_provider(
        self,
        config: ProviderConfig,
    ) -> object:
        return self._llm_backend.create_provider(
            LLMProviderRequest(
                provider_id=config.provider_id,
                model=config.model,
            )
        )

    def _build_summary_metadata(
        self,
        document: BulkAnalysisDocument,
        provider_config: ProviderConfig,
        prompt_hash: str,
        run_details: Dict[str, object],
        *,
        created_at: datetime,
    ) -> Dict[str, object]:
        project_path = infer_project_path(document.output_path)
        sources = [
            SourceReference(
                path=document.source_path,
                relative=document.relative_path,
                kind=(document.source_path.suffix.lstrip(".") or "file"),
                role="converted-document",
                checksum=compute_file_checksum(document.source_path),
            )
        ]
        prompts = self._prompt_references()
        extra: Dict[str, object] = {
            "group_id": self._group.group_id,
            "group_name": self._group.name,
            "group_operation": self._group.operation,
            "prompt_hash": prompt_hash,
            "provider_id": provider_config.provider_id,
            "model": provider_config.model,
        }
        if self._estimate_summary:
            extra["cost_estimate"] = dict(self._estimate_summary)
        extra.update(run_details)
        return build_document_metadata(
            project_path=project_path,
            generator="bulk_analysis_worker",
            created_at=created_at,
            sources=sources,
            prompts=prompts,
            extra=extra,
        )

    def _build_document_placeholders(
        self,
        base: Mapping[str, str],
        source_context: SourceFileContext | None,
    ) -> Dict[str, str]:
        doc_placeholders = dict(base)
        doc_placeholders.update(
            self._build_placeholder_map(
                source=source_context,
            )
        )
        return doc_placeholders

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

    def _append_citation_appendix(self, system_prompt: str, appendix: str) -> str:
        return append_generated_prompt_section(system_prompt, appendix)

    def _build_document_evidence_ledger(self, *, relative_path: str, content: str) -> str:
        entries = self._build_document_citation_entries(relative_path=relative_path, content=content)
        return self._render_citation_appendix(entries=entries)

    def _build_document_citation_entries(
        self,
        *,
        relative_path: str,
        content: str,
    ) -> list[CitationLedgerEntry]:
        if self._citation_store is None:
            return []
        pages = self._extract_page_numbers(content)
        if not pages:
            return []
        try:
            return self._citation_store.list_local_citation_entries(
                relative_path=relative_path,
                page_numbers=pages,
                max_entries=120,
            )
        except Exception:
            self.logger.debug(
                "%s failed to build citation entries for %s",
                self.job_tag,
                relative_path,
                exc_info=True,
            )
            return []

    def _render_citation_appendix(
        self,
        *,
        entries: Sequence[CitationLedgerEntry],
    ) -> str:
        if self._citation_store is None:
            return ""
        try:
            return self._citation_store.render_local_citation_appendix(entries)
        except Exception:
            self.logger.debug("%s failed to render citation appendix", self.job_tag, exc_info=True)
            return ""

    def _filter_citation_entries_for_content(
        self,
        *,
        entries: Sequence[CitationLedgerEntry],
        content: str,
    ) -> list[CitationLedgerEntry]:
        pages = set(self._extract_page_numbers(content))
        if not pages:
            return list(entries)
        return [entry for entry in entries if entry.page_number in pages]

    def _extract_page_numbers(self, content: str) -> list[int]:
        pages = [int(match.group(1)) for match in PAGE_MARKER_RE.finditer(content)]
        if not pages:
            return []
        ordered = list(dict.fromkeys(pages))
        return ordered[:40]

    def _record_output_citations(
        self,
        *,
        output_path: Path,
        output_text: str,
        prompt_hash: str,
        label_mapping: Mapping[str, str] | None = None,
    ) -> CitationRecordStats | None:
        if self._citation_store is None:
            return None
        try:
            stats = self._citation_store.record_output_citations(
                output_path=output_path,
                output_text=output_text,
                generator="bulk_analysis_worker",
                prompt_hash=prompt_hash,
                label_mapping=label_mapping,
            )
        except Exception:
            self.logger.debug(
                "%s failed to record citations for %s",
                self.job_tag,
                output_path,
                exc_info=True,
            )
            return None

        if stats.total > 0 and (stats.warning > 0 or stats.invalid > 0):
            self.log_message.emit(
                f"Citations {output_path.name}: valid={stats.valid}, "
                f"warning={stats.warning}, invalid={stats.invalid}"
            )
        return stats

    def _prompt_references(self) -> List[PromptReference]:
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
