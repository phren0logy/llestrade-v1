"""Service helpers for bulk analysis worker orchestration."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

from shiboken6 import isValid

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.project_manager import ProjectMetadata
from src.app.workers import WorkerCoordinator
from src.app.workers import BulkAnalysisWorker, BulkReduceWorker
from src.app.workers.llm_backend import LLMExecutionBackend


class BulkAnalysisService:
    """Wrap worker lifecycle management for bulk analysis operations."""

    _MAP_KEY_PREFIX = "bulk:"
    _COMBINED_KEY_PREFIX = "combine:"

    def __init__(
        self,
        workers: WorkerCoordinator,
        *,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        self._workers = workers
        self._llm_backend = llm_backend

    # ------------------------------------------------------------------
    # Map (per-document) runs
    # ------------------------------------------------------------------
    def run_map(
        self,
        *,
        project_dir,
        group: BulkAnalysisGroup,
        files: Sequence[str],
        metadata: Optional[ProjectMetadata],
        default_provider: tuple[str, str | None],
        force_rerun: bool,
        placeholder_values: Mapping[str, str],
        project_name: str,
        on_progress: Callable[[str, int, int, str], None],
        on_failed: Callable[[str, str, str], None],
        on_log: Callable[[str, str], None],
        on_finished: Callable[[str, int, int], None],
    ) -> bool:
        key = self._map_key(group.group_id)
        if self._workers.get(key):
            return False

        worker = BulkAnalysisWorker(
            project_dir=project_dir,
            group=group,
            files=list(files),
            metadata=metadata,
            default_provider=default_provider,
            force_rerun=force_rerun,
            placeholder_values=placeholder_values,
            project_name=project_name,
            llm_backend=self._llm_backend,
        )

        gid = group.group_id
        worker.progress.connect(lambda done, total, rel, g=gid: on_progress(g, done, total, rel))
        worker.file_failed.connect(lambda rel, err, g=gid: on_failed(g, rel, err))
        worker.log_message.connect(lambda message, g=gid: on_log(g, message))
        worker.finished.connect(
            lambda success, failed, w=worker, g=gid: self._handle_finished(
                key, w, success, failed, lambda s=success, f=failed, group_id=g: on_finished(group_id, s, f)
            )
        )

        self._workers.start(key, worker)
        return True

    # ------------------------------------------------------------------
    # Combined runs
    # ------------------------------------------------------------------
    def run_combined(
        self,
        *,
        project_dir,
        group: BulkAnalysisGroup,
        metadata: Optional[ProjectMetadata],
        force_rerun: bool,
        placeholder_values: Mapping[str, str],
        project_name: str,
        on_progress: Callable[[str, int, int, str], None],
        on_failed: Callable[[str, str, str], None],
        on_log: Callable[[str, str], None],
        on_finished: Callable[[str, int, int], None],
    ) -> bool:
        key = self._combined_key(group.group_id)
        if self._workers.get(key):
            return False

        worker = BulkReduceWorker(
            project_dir=project_dir,
            group=group,
            metadata=metadata,
            force_rerun=force_rerun,
            placeholder_values=placeholder_values,
            project_name=project_name,
            llm_backend=self._llm_backend,
        )

        gid = group.group_id
        worker.progress.connect(lambda done, total, msg, g=gid: on_progress(g, done, total, msg))
        worker.file_failed.connect(lambda rel, err, g=gid: on_failed(g, rel, err))
        worker.log_message.connect(lambda message, g=gid: on_log(g, message))
        worker.finished.connect(
            lambda success, failed, w=worker, g=gid: self._handle_finished(
                key, w, success, failed, lambda s=success, f=failed, group_id=g: on_finished(group_id, s, f)
            )
        )

        self._workers.start(key, worker)
        return True

    # ------------------------------------------------------------------
    # Cancellation helpers
    # ------------------------------------------------------------------
    def cancel(self, group_id: str) -> bool:
        for key in (self._map_key(group_id), self._combined_key(group_id)):
            worker = self._workers.get(key)
            if worker:
                worker.cancel()
                return True
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _handle_finished(
        self,
        key: str,
        worker,
        successes: int,
        failures: int,
        callback: Callable[[], None],
    ) -> None:
        stored = self._workers.pop(key)
        if worker and isValid(worker):
            worker.deleteLater()
        if stored and stored is not worker and isValid(stored):
            stored.deleteLater()
        callback()

    def _map_key(self, group_id: str) -> str:
        return f"{self._MAP_KEY_PREFIX}{group_id}"

    def _combined_key(self, group_id: str) -> str:
        return f"{self._COMBINED_KEY_PREFIX}{group_id}"


__all__ = ["BulkAnalysisService"]
