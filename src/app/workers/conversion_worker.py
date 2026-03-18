"""Background worker for local Docling MLX document conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PySide6.QtCore import Signal

from src.app.core.citations import CitationStore
from src.app.core.conversion_helpers import ConversionHelper, registry
from src.app.core.docling_local import (
    DEFAULT_VLM_PRESET,
    DoclingLocalError,
    assert_local_docling_runtime,
    convert_pdf_to_doctags,
)
from src.common.markdown import compute_file_checksum

from src.app.core.conversion_manager import ConversionJob
from .base import DashboardWorker


class FatalConversionError(RuntimeError):
    """Raised when a batch should stop immediately due to shared configuration failure."""


class ConversionWorker(DashboardWorker):
    """Run conversion jobs against the experimental local Docling backend."""

    progress = Signal(int, int, str)
    file_failed = Signal(str, str)
    fatal_error = Signal(str)
    finished = Signal(int, int)

    def __init__(
        self,
        jobs: Iterable[ConversionJob],
        *,
        helper: str = "docling",
        options: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__(worker_name="conversion")
        self._jobs: List[ConversionJob] = list(jobs)
        self._helper_id = helper or "docling"
        self._options = dict(options or {})
        self._helper_cache: Optional[ConversionHelper] = None

    def _run(self) -> None:  # pragma: no cover - executed in worker thread
        total = len(self._jobs)
        self.logger.info("%s starting conversion (jobs=%s)", self.job_tag, total)
        successes = 0
        failures = 0
        for job in self._jobs:
            if self.is_cancelled():
                self.logger.info("%s cancelled after %s/%s jobs", self.job_tag, successes + failures, total)
                break
            try:
                self._execute(job)
            except FatalConversionError as exc:
                failures += 1
                self.logger.error("%s fatal conversion error for %s: %s", self.job_tag, job.source_path, exc)
                self.fatal_error.emit(str(exc))
                break
            except Exception as exc:  # noqa: BLE001
                failures += 1
                self.logger.exception("%s failed %s", self.job_tag, job.source_path)
                self.file_failed.emit(str(job.source_path), str(exc))
            else:
                successes += 1
            finally:
                self.progress.emit(successes + failures, total, job.display_name)
        self.logger.info("%s finished: successes=%s failures=%s", self.job_tag, successes, failures)
        self.finished.emit(successes, failures)

    def _execute(self, job: ConversionJob) -> None:
        if job.conversion_type != "pdf":
            raise RuntimeError("This experimental branch currently supports PDF conversion only.")
        self._convert_with_docling(job)

    def _convert_with_docling(self, job: ConversionJob) -> None:
        job.destination_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_mode, vlm_preset, _standard_profile = self._docling_options()
        if pipeline_mode == "standard_only":
            raise FatalConversionError(
                "The local experimental branch does not support the standard Docling pipeline yet. "
                "Use the MLX VLM path for now."
            )
        try:
            assert_local_docling_runtime()
        except DoclingLocalError as exc:
            raise FatalConversionError(str(exc)) from exc

        try:
            result = convert_pdf_to_doctags(
                source_path=job.source_path,
                vlm_preset=vlm_preset,
            )
        except DoclingLocalError as exc:
            raise RuntimeError(str(exc)) from exc

        doctags_text = result.doctags_content
        job.destination_path.write_text(doctags_text, encoding="utf-8")
        self._index_citations_for_output(
            job,
            job.destination_path,
            doctags_text=doctags_text,
            pages_detected=result.page_count,
            pages_pdf=self._pdf_page_count(job.source_path),
            pipeline_mode=pipeline_mode,
            vlm_preset=vlm_preset,
            standard_profile=None,
        )

    def _docling_options(self) -> tuple[str, str, str | None]:
        pipeline_mode = str(self._options.get("pipeline_mode") or "vlm_primary").strip() or "vlm_primary"
        vlm_preset = str(self._options.get("vlm_preset") or DEFAULT_VLM_PRESET).strip() or DEFAULT_VLM_PRESET
        standard_profile_raw = str(self._options.get("standard_profile") or "").strip()
        standard_profile = standard_profile_raw or None
        return pipeline_mode, vlm_preset, standard_profile

    def _helper(self) -> ConversionHelper:
        if self._helper_cache is not None:
            return self._helper_cache
        helper = registry().get(self._helper_id)
        if helper is None:
            helper = registry().default_helper()
        self._helper_cache = helper
        return helper

    def _project_context(self, job: ConversionJob) -> tuple[Path | None, str]:
        dest = job.destination_path.resolve()
        parts = dest.parts
        try:
            idx = parts.index("converted_documents")
        except ValueError:
            return None, job.source_path.name

        project_dir = Path(*parts[:idx])
        project_resolved = project_dir.resolve()
        source_resolved = job.source_path.resolve()
        try:
            rel = source_resolved.relative_to(project_resolved).as_posix()
        except Exception:
            import os as _os

            rel = Path(_os.path.relpath(source_resolved, project_resolved)).as_posix()
        return project_dir, rel

    def _index_citations_for_output(
        self,
        job: ConversionJob,
        final_path: Path,
        *,
        doctags_text: str,
        pages_detected: int | None,
        pages_pdf: int | None,
        pipeline_mode: str,
        vlm_preset: str,
        standard_profile: str | None,
    ) -> None:
        project_dir, source_relative = self._project_context(job)
        if project_dir is None:
            return

        converted_root = (project_dir / "converted_documents").resolve()
        try:
            relative_path = final_path.resolve().relative_to(converted_root).as_posix()
        except Exception:
            return

        try:
            store = CitationStore(project_dir)
            stats = store.index_doctags_document(
                relative_path=relative_path,
                doctags_text=doctags_text,
                source_checksum=compute_file_checksum(job.source_path),
                pages_pdf=pages_pdf,
                pages_detected=pages_detected,
                source_relative_path=source_relative or job.relative_path,
                source_absolute_path=job.source_path.resolve().as_posix(),
                pipeline_mode=pipeline_mode,
                vlm_preset=vlm_preset,
                standard_profile=standard_profile,
            )
            self.logger.info(
                "%s indexed citation evidence for %s (segments=%s, geometry=%s)",
                self.job_tag,
                relative_path,
                stats.segments_indexed,
                stats.geometry_spans_indexed,
            )
        except Exception as exc:
            self.logger.warning(
                "%s failed to index citation evidence for %s: %s",
                self.job_tag,
                relative_path,
                exc,
            )

    def _pdf_page_count(self, source_path: Path) -> int | None:
        try:
            import fitz  # PyMuPDF

            return len(fitz.open(source_path))
        except Exception:
            return None


__all__ = ["ConversionWorker", "FatalConversionError"]
