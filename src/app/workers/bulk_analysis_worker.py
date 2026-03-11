"""Worker for executing bulk-analysis runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import frontmatter
from PySide6.QtCore import Signal

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
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
from src.app.core.citations import CitationRecordStats, CitationStore
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
    LegacyProviderBackend,
)
from .stage_contracts import BulkMapStageInput, stage_trace_attributes


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    model: Optional[str]




_MANIFEST_VERSION = 2
_MTIME_TOLERANCE = 1e-6
_DEFAULT_MAX_OUTPUT_TOKENS = 32_000
_MIN_CHUNK_TOKEN_TARGET = 4_000
_PAGE_MARKER_RE = re.compile(r"<!---\\s*.+?#page=(\\d+)\\s*--->")

_DYNAMIC_GLOBAL_KEYS: frozenset[str] = frozenset(
    {
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
    return {"version": _MANIFEST_VERSION, "signature": None, "documents": {}}


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
        "documents": documents,
    }


def _save_manifest(path: Path, manifest: Dict[str, object]) -> None:
    payload = {
        "version": _MANIFEST_VERSION,
        "signature": manifest.get("signature"),
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

class BulkAnalysisWorker(DashboardWorker):
    """Run bulk analysis summaries on the thread pool."""

    progress = Signal(int, int, str)  # completed, total, relative path
    file_failed = Signal(str, str)  # relative path, error message
    finished = Signal(int, int)  # successes, failures
    log_message = Signal(str)

    def __init__(
        self,
        *,
        project_dir: Path,
        group: BulkAnalysisGroup,
        files: Sequence[str],
        metadata: Optional[ProjectMetadata],
        default_provider: Tuple[str, Optional[str]] = ("anthropic", None),
        force_rerun: bool = False,
        placeholder_values: Mapping[str, str] | None = None,
        project_name: str = "",
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        super().__init__(worker_name="bulk_analysis")

        self._project_dir = project_dir
        self._group = group
        self._files = list(files)
        self._metadata = metadata
        self._default_provider = default_provider
        self._force_rerun = force_rerun
        self._base_placeholders = dict(placeholder_values or {})
        self._project_name = project_name
        self._run_timestamp = datetime.now(timezone.utc)
        self._llm_backend: LLMExecutionBackend = llm_backend or LegacyProviderBackend()
        try:
            self._citation_store: CitationStore | None = CitationStore(project_dir)
        except Exception:
            self._citation_store = None

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

        try:
            documents = prepare_documents(self._project_dir, self._group, self._files)
            total = len(documents)
            if total == 0:
                self.log_message.emit("No documents resolved for bulk analysis run.")
                self.logger.info("%s no documents to process", self.job_tag)
                self.finished.emit(0, 0)
                return

            self.logger.info("%s starting bulk analysis (docs=%s)", self.job_tag, total)
            provider_config = self._resolve_provider()
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
            slug = getattr(self._group, "slug", None) or self._group.folder_name
            checkpoint_mgr = CheckpointManager(
                self._project_dir / "bulk_analysis" / slug / "map" / "checkpoints"
            )
            signature = {
                "prompt_hash": prompt_hash,
                "placeholders": _stable_placeholders(self._serialise_placeholders(global_placeholders)),
            }
            manifest_path = _manifest_path(self._project_dir, self._group)
            manifest = _load_manifest(manifest_path)
            if manifest.get("version") != _MANIFEST_VERSION or manifest.get("signature") != signature:
                checkpoint_mgr.clear_all()
                manifest = _default_manifest()
            manifest["signature"] = signature
            entries = manifest.setdefault("documents", {})  # type: ignore[arg-type]

            for index, document in enumerate(documents, start=1):
                if self.is_cancelled():
                    raise BulkAnalysisCancelled

                try:
                    source_mtime = document.source_path.stat().st_mtime
                except FileNotFoundError:
                    source_mtime = 0.0

                entry = entries.get(document.relative_path)
                output_exists = document.output_path.exists()
                if not self._force_rerun and not _should_process_document(entry, source_mtime, prompt_hash, output_exists):
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
                    summary, run_details, doc_placeholders = self._process_document(
                        provider,
                        provider_config,
                        bundle,
                        system_prompt,
                        document,
                        global_placeholders,
                        checkpoint_mgr,
                        manifest,
                        prompt_hash,
                        manifest_path,
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
                self.progress.emit(progress_count, total, document.relative_path)

        except BulkAnalysisCancelled:
            self.log_message.emit("Bulk analysis run cancelled.")
            self.logger.info("%s cancelled", self.job_tag)
        except Exception as exc:  # pragma: no cover - defensive logging
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
            if skipped:
                self.log_message.emit(f"Skipped {skipped} document(s) (no changes detected)")
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
        checkpoint_mgr: CheckpointManager,
        manifest: Dict[str, object],
        prompt_hash: str,
        manifest_path: Path,
    ) -> tuple[str, Dict[str, object], Dict[str, str]]:
        if self.is_cancelled():
            raise BulkAnalysisCancelled

        body, metadata, source_context = self._load_document(document)
        doc_placeholders = self._build_document_placeholders(global_placeholders, source_context)

        self._enforce_placeholder_requirements(
            doc_placeholders,
            context=f"bulk analysis document '{document.relative_path}'",
            dynamic_keys=_DYNAMIC_DOCUMENT_KEYS,
        )

        override_window = getattr(self._group, "model_context_window", None)
        if isinstance(override_window, int) and override_window > 0:
            raw_context_window = int(override_window)
            body_token_info = TokenCounter.count(
                text=body,
                provider=provider_config.provider_id,
                model=provider_config.model or "",
            )
            token_count = int(body_token_info.get("token_count") or 0) if body_token_info.get("success") else max(len(body) // 3, 1)
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
            )

        input_budget = self._max_input_budget(
            raw_context_window=raw_context_window,
            max_output_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
        )
        full_prompt = render_user_prompt(
            bundle,
            self._metadata,
            document.relative_path,
            body,
            placeholder_values=doc_placeholders,
        )
        full_ledger = self._build_document_evidence_ledger(
            relative_path=document.relative_path,
            content=body,
        )
        full_prompt = self._append_citation_ledger(full_prompt, full_ledger)
        full_prompt_tokens = self._count_prompt_tokens(
            provider,
            provider_config,
            system_prompt,
            full_prompt,
        )
        if full_prompt_tokens > input_budget:
            needs_chunking = True

        run_details: Dict[str, object] = {
            "token_count": token_count,
            "full_prompt_tokens": full_prompt_tokens,
            "max_tokens": default_chunk_tokens,
            "input_budget_tokens": input_budget,
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

        if not needs_chunking:
            run_details["chunk_count"] = 1
            result = self._invoke_provider(
                provider,
                provider_config,
                full_prompt,
                system_prompt,
                input_budget=input_budget,
                context_label=f"document '{document.relative_path}'",
            )
            return result, run_details, doc_placeholders

        chunk_target_tokens = max(
            min(default_chunk_tokens, int(input_budget * 0.75)),
            _MIN_CHUNK_TOKEN_TARGET,
        )
        chunks = self._generate_fitting_chunks(
            provider=provider,
            provider_config=provider_config,
            bundle=bundle,
            system_prompt=system_prompt,
            document=document,
            body=body,
            placeholder_values=doc_placeholders,
            input_budget=input_budget,
            initial_chunk_tokens=chunk_target_tokens,
        )
        if not chunks:
            run_details["chunk_count"] = 1
            run_details["chunking"] = False
            result = self._invoke_provider(
                provider,
                provider_config,
                full_prompt,
                system_prompt,
                input_budget=input_budget,
                context_label=f"document '{document.relative_path}'",
            )
            return result, run_details, doc_placeholders

        documents = manifest.setdefault("documents", {})  # type: ignore[assignment]
        entry: Dict[str, object] = dict(documents.get(document.relative_path, {}) or {})
        source_checksum = _sha256(body)
        total_chunks = len(chunks)
        needs_reset = (
            entry.get("source_checksum") != source_checksum
            or entry.get("prompt_hash") != prompt_hash
            or entry.get("chunk_count") != total_chunks
        )
        if needs_reset:
            checkpoint_mgr.clear_map_document(document.relative_path)
            entry = {"chunks_done": [], "checksums": {}}

        entry["source_checksum"] = source_checksum
        entry["prompt_hash"] = prompt_hash
        entry["chunk_count"] = total_chunks
        entry.setdefault("chunks_done", [])
        entry.setdefault("checksums", {})
        documents[document.relative_path] = entry

        chunk_summaries: List[str] = []
        run_details["chunk_count"] = total_chunks
        done_set = set(entry.get("chunks_done") or [])
        checksums: Dict[str, str] = dict(entry.get("checksums") or {})

        for idx, chunk in enumerate(chunks, start=1):
            if self.is_cancelled():
                raise BulkAnalysisCancelled

            chunk_checksum = _sha256(chunk)
            cached = checkpoint_mgr.load_map_chunk(document.relative_path, idx)
            cached_content = None
            if (
                cached
                and cached.get("input_checksum") == chunk_checksum
                and cached.get("content")
                and cached.get("content_checksum") == _sha256(cached.get("content"))
                and checksums.get(str(idx)) == chunk_checksum
            ):
                cached_content = cached.get("content")

            if cached_content:
                summary = cached_content
                self.log_message.emit(
                    f"Reusing chunk {idx}/{total_chunks} for {document.relative_path} from checkpoint"
                )
            else:
                chunk_prompt = render_user_prompt(
                    bundle,
                    self._metadata,
                    document.relative_path,
                    chunk,
                    chunk_index=idx,
                    chunk_total=total_chunks,
                    placeholder_values=doc_placeholders,
                )
                chunk_ledger = self._build_document_evidence_ledger(
                    relative_path=document.relative_path,
                    content=chunk,
                )
                chunk_prompt = self._append_citation_ledger(chunk_prompt, chunk_ledger)
                summary = self._invoke_provider(
                    provider,
                    provider_config,
                    chunk_prompt,
                    system_prompt,
                    input_budget=input_budget,
                    context_label=f"chunk {idx}/{total_chunks} for '{document.relative_path}'",
                )
                checkpoint_mgr.save_map_chunk(document.relative_path, idx, summary, chunk_checksum)

            checksums[str(idx)] = chunk_checksum
            done_set.add(idx)
            entry["checksums"] = checksums
            entry["chunks_done"] = sorted(done_set)
            documents[document.relative_path] = entry
            try:
                _save_manifest(manifest_path, manifest)
            except Exception:
                self.logger.debug("Failed to persist map chunk manifest update", exc_info=True)

            chunk_summaries.append(summary)

        def invoke_combine(prompt: str) -> str:
            return self._invoke_provider(
                provider,
                provider_config,
                prompt,
                system_prompt,
                input_budget=input_budget,
                context_label=f"combine summary for '{document.relative_path}'",
            )

        result = combine_chunk_summaries_hierarchical(
            chunk_summaries,
            document_name=document.relative_path,
            metadata=self._metadata,
            placeholder_values=doc_placeholders,
            provider_id=provider_config.provider_id,
            model=provider_config.model,
            invoke_fn=invoke_combine,
            is_cancelled_fn=self.is_cancelled,
        )
        entry["status"] = "complete"
        entry["ran_at"] = datetime.now(timezone.utc).isoformat()
        documents[document.relative_path] = entry
        try:
            _save_manifest(manifest_path, manifest)
        finally:
            checkpoint_mgr.clear_map_document(document.relative_path)
        return result, run_details, doc_placeholders

    def _max_input_budget(self, *, raw_context_window: int, max_output_tokens: int) -> int:
        budget = compute_input_token_budget(
            raw_context_window=raw_context_window,
            max_output_tokens=max_output_tokens,
            minimum_budget=_MIN_CHUNK_TOKEN_TARGET,
        )
        return budget or _MIN_CHUNK_TOKEN_TARGET

    def _count_prompt_tokens(
        self,
        provider: object,
        provider_config: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> int:
        combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
        if not combined_prompt:
            return 0

        try:
            counted = self._llm_backend.count_input_tokens(
                provider,
                LLMInvocationRequest(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=provider_config.model,
                    model_settings={
                        "temperature": 0.1,
                        "max_tokens": max_tokens,
                    },
                ),
            )
            if counted is not None and counted >= 0:
                return int(counted)
        except Exception:
            self.logger.debug("Backend token preflight failed; falling back to local estimate", exc_info=True)

        token_info = TokenCounter.count(
            text=combined_prompt,
            provider=provider_config.provider_id,
            model=provider_config.model or "",
        )
        if token_info.get("success"):
            counted = int(token_info.get("token_count") or 0)
            if provider_config.provider_id in {"anthropic", "anthropic_bedrock"}:
                return max(counted, max(len(combined_prompt) // 3, 1))
            if counted > 0:
                return counted
        return max(len(combined_prompt) // 3, 1)

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
    ) -> List[str]:
        chunk_target = max(initial_chunk_tokens, _MIN_CHUNK_TOKEN_TARGET)
        while True:
            chunks = generate_chunks(body, chunk_target)
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
                chunk_ledger = self._build_document_evidence_ledger(
                    relative_path=document.relative_path,
                    content=chunk,
                )
                prompt = self._append_citation_ledger(prompt, chunk_ledger)
                prompt_tokens = self._count_prompt_tokens(
                    provider,
                    provider_config,
                    system_prompt,
                    prompt,
                    max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
                )
                if prompt_tokens > input_budget:
                    oversized.append((idx, prompt_tokens))

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
            chunk_target = max(int(chunk_target * 0.75), _MIN_CHUNK_TOKEN_TARGET)
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
        context_label: str = "bulk analysis request",
    ) -> str:
        if self._cancel_event.is_set():
            raise BulkAnalysisCancelled

        if input_budget is not None:
            prompt_tokens = self._count_prompt_tokens(
                provider,
                provider_config,
                system_prompt,
                prompt,
                max_tokens=max_tokens,
            )
            if prompt_tokens > input_budget:
                raise RuntimeError(
                    f"Prompt exceeds model input budget for {context_label}: "
                    f"{prompt_tokens} tokens > {input_budget} budget"
                )

        stage_input = BulkMapStageInput(
            group_id=self._group.group_id,
            group_name=self._group.name,
            group_slug=getattr(self._group, "slug", None) or self._group.folder_name,
            provider_id=provider_config.provider_id,
            model=provider_config.model,
            context_label=context_label,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        trace_attributes = stage_trace_attributes(stage_input)
        with trace_operation("bulk_analysis.invoke_llm", trace_attributes):
            response = self._llm_backend.invoke(
                provider,
                LLMInvocationRequest(
                    prompt=prompt,
                    model=provider_config.model,
                    system_prompt=system_prompt,
                    model_settings={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    input_tokens_limit=input_budget,
                ),
            )
        if not response.success:
            raise RuntimeError(response.error or "Unknown LLM error")
        content = response.content.strip()
        if not content:
            raise RuntimeError("LLM returned empty response")
        return content

    def _resolve_provider(self) -> ProviderConfig:
        provider_id = self._group.provider_id or self._default_provider[0] or "anthropic"
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

    def _append_citation_ledger(self, prompt: str, ledger: str) -> str:
        if not ledger.strip():
            return prompt
        return f"{prompt.rstrip()}\\n\\n{ledger.rstrip()}\\n"

    def _build_document_evidence_ledger(self, *, relative_path: str, content: str) -> str:
        if self._citation_store is None:
            return ""
        pages = self._extract_page_numbers(content)
        try:
            return self._citation_store.build_evidence_ledger(
                relative_path=relative_path,
                page_numbers=pages,
                max_entries=120,
            )
        except Exception:
            self.logger.debug(
                "%s failed to build evidence ledger for %s",
                self.job_tag,
                relative_path,
                exc_info=True,
            )
            return ""

    def _extract_page_numbers(self, content: str) -> list[int]:
        pages = [int(match.group(1)) for match in _PAGE_MARKER_RE.finditer(content)]
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
    ) -> CitationRecordStats | None:
        if self._citation_store is None:
            return None
        try:
            stats = self._citation_store.record_output_citations(
                output_path=output_path,
                output_text=output_text,
                generator="bulk_analysis_worker",
                prompt_hash=prompt_hash,
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
