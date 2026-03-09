"""Services for orchestrating report draft and refinement workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from shiboken6 import isValid

from src.app.core.project_manager import ProjectMetadata
from src.app.workers.llm_backend import LLMExecutionBackend
from src.app.workers import WorkerCoordinator
from src.app.workers.report_worker import DraftReportWorker, ReportRefinementWorker


@dataclass(slots=True)
class ReportDraftJobConfig:
    """Configuration payload required to launch a draft worker."""

    project_dir: Path
    inputs: Sequence[tuple[str, str]]
    provider_id: str
    model: str
    custom_model: Optional[str]
    context_window: Optional[int]
    template_path: Path
    transcript_path: Optional[Path]
    generation_user_prompt_path: Path
    generation_system_prompt_path: Path
    metadata: ProjectMetadata
    max_report_tokens: int = 60_000
    placeholder_values: Mapping[str, str] | None = None
    project_name: str = ""


@dataclass(slots=True)
class ReportRefinementJobConfig:
    """Configuration payload required to launch a refinement worker."""

    project_dir: Path
    draft_path: Path
    inputs: Sequence[tuple[str, str]]
    provider_id: str
    model: str
    custom_model: Optional[str]
    context_window: Optional[int]
    template_path: Optional[Path]
    transcript_path: Optional[Path]
    refinement_user_prompt_path: Path
    refinement_system_prompt_path: Path
    metadata: ProjectMetadata
    max_report_tokens: int = 60_000
    placeholder_values: Mapping[str, str] | None = None
    project_name: str = ""


class ReportsService:
    """Create and manage report draft and refinement workers for the UI."""

    _DRAFT_KEY = "report:draft"
    _REFINE_KEY = "report:refine"

    def __init__(
        self,
        workers: WorkerCoordinator,
        *,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        self._workers = workers
        self._llm_backend = llm_backend

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return any(
            self._workers.get(key) is not None
            for key in (self._DRAFT_KEY, self._REFINE_KEY)
        )

    # ------------------------------------------------------------------
    # Draft orchestration
    # ------------------------------------------------------------------
    def run_draft(
        self,
        config: ReportDraftJobConfig,
        *,
        on_started: Callable[[], None],
        on_progress: Callable[[int, str], None],
        on_log: Callable[[str], None],
        on_finished: Callable[[dict], None],
        on_failed: Callable[[str], None],
    ) -> bool:
        if self.is_running():
            return False

        worker = DraftReportWorker(
            project_dir=config.project_dir,
            inputs=list(config.inputs),
            provider_id=config.provider_id,
            model=config.model,
            custom_model=config.custom_model,
            context_window=config.context_window,
            template_path=config.template_path,
            transcript_path=config.transcript_path,
            generation_user_prompt_path=config.generation_user_prompt_path,
            generation_system_prompt_path=config.generation_system_prompt_path,
            metadata=config.metadata,
            max_report_tokens=config.max_report_tokens,
            placeholder_values=config.placeholder_values,
            project_name=config.project_name,
            llm_backend=self._llm_backend,
        )

        return self._start_worker(
            key=self._DRAFT_KEY,
            worker=worker,
            on_started=on_started,
            on_progress=on_progress,
            on_log=on_log,
            on_finished=on_finished,
            on_failed=on_failed,
        )

    # ------------------------------------------------------------------
    # Refinement orchestration
    # ------------------------------------------------------------------
    def run_refinement(
        self,
        config: ReportRefinementJobConfig,
        *,
        on_started: Callable[[], None],
        on_progress: Callable[[int, str], None],
        on_log: Callable[[str], None],
        on_finished: Callable[[dict], None],
        on_failed: Callable[[str], None],
    ) -> bool:
        if self.is_running():
            return False

        worker = ReportRefinementWorker(
            project_dir=config.project_dir,
            draft_path=config.draft_path,
            inputs=list(config.inputs),
            provider_id=config.provider_id,
            model=config.model,
            custom_model=config.custom_model,
            context_window=config.context_window,
            template_path=config.template_path,
            transcript_path=config.transcript_path,
            refinement_user_prompt_path=config.refinement_user_prompt_path,
            refinement_system_prompt_path=config.refinement_system_prompt_path,
            metadata=config.metadata,
            max_report_tokens=config.max_report_tokens,
            placeholder_values=config.placeholder_values,
            project_name=config.project_name,
            llm_backend=self._llm_backend,
        )

        return self._start_worker(
            key=self._REFINE_KEY,
            worker=worker,
            on_started=on_started,
            on_progress=on_progress,
            on_log=on_log,
            on_finished=on_finished,
            on_failed=on_failed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _start_worker(
        self,
        *,
        key: str,
        worker,
        on_started: Callable[[], None],
        on_progress: Callable[[int, str], None],
        on_log: Callable[[str], None],
        on_finished: Callable[[dict], None],
        on_failed: Callable[[str], None],
    ) -> bool:
        worker.progress.connect(on_progress)
        worker.log_message.connect(on_log)
        worker.finished.connect(lambda result, w=worker, k=key: self._handle_finished(k, w, result, on_finished))
        worker.failed.connect(lambda message, w=worker, k=key: self._handle_failed(k, w, message, on_failed))

        on_started()
        self._workers.start(key, worker)
        return True

    def _handle_finished(
        self,
        key: str,
        worker,
        result: dict,
        callback: Callable[[dict], None],
    ) -> None:
        stored = self._workers.pop(key)
        if worker and isValid(worker):
            worker.deleteLater()
        if stored and stored is not worker and isValid(stored):
            stored.deleteLater()
        callback(result)

    def _handle_failed(
        self,
        key: str,
        worker,
        message: str,
        callback: Callable[[str], None],
    ) -> None:
        stored = self._workers.pop(key)
        if worker and isValid(worker):
            worker.deleteLater()
        if stored and stored is not worker and isValid(stored):
            stored.deleteLater()
        callback(message)


__all__ = [
    "ReportDraftJobConfig",
    "ReportRefinementJobConfig",
    "ReportsService",
]
