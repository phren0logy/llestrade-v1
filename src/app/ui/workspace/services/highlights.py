"""Services consolidating highlight runner orchestration."""

from __future__ import annotations

from typing import Callable, Optional

from shiboken6 import isValid

from src.app.core.highlight_manager import HighlightJob
from src.app.core.project_manager import ProjectManager
from src.app.workers import WorkerCoordinator
from src.app.workers.highlight_worker import HighlightExtractionSummary, HighlightWorker


class HighlightsService:
    """Prepare and execute highlight extraction batches."""

    _WORKER_KEY = "highlights:run"

    def __init__(self, workers: WorkerCoordinator) -> None:
        self._workers = workers

    def is_running(self) -> bool:
        return self._workers.get(self._WORKER_KEY) is not None

    def cancel(self) -> bool:
        return self._workers.cancel(self._WORKER_KEY)

    def run(
        self,
        *,
        project_manager: ProjectManager,
        on_started: Callable[[int], None],
        on_progress: Callable[[int, int, str], None],
        on_failed: Callable[[str, str], None],
        on_finished: Callable[[HighlightExtractionSummary | None, int, int], None],
    ) -> bool:
        jobs = self._build_jobs(project_manager)
        if not jobs:
            return False

        worker = HighlightWorker(jobs)

        worker.progress.connect(on_progress)
        worker.file_failed.connect(on_failed)
        worker.finished.connect(
            lambda success, failed, w=worker: self._handle_finished(w, success, failed, on_finished)
        )

        on_started(len(jobs))
        self._workers.start(self._WORKER_KEY, worker)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_jobs(self, project_manager: ProjectManager) -> list[HighlightJob]:
        jobs = project_manager.build_highlight_jobs()
        return list(jobs or [])

    def _handle_finished(
        self,
        worker: HighlightWorker,
        successes: int,
        failures: int,
        callback: Callable[[Optional[HighlightExtractionSummary], int, int], None],
    ) -> None:
        stored = self._workers.pop(self._WORKER_KEY)
        if worker and isValid(worker):
            worker.deleteLater()
        if stored and stored is not worker and isValid(stored):
            stored.deleteLater()

        callback(worker.summary, successes, failures)


__all__ = ["HighlightsService"]
