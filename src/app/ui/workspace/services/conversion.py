"""Service helpers for dashboard document conversion orchestration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Mapping, Sequence
from uuid import uuid4

from shiboken6 import isValid

from src.app.core.conversion_manager import ConversionJob
from src.app.workers import ConversionWorker, WorkerCoordinator


@dataclass
class _ConversionBatchState:
    batch_id: str
    helper: str
    options: Dict[str, object]
    total: int
    pending: Deque[ConversionJob]
    in_flight: Dict[str, tuple[ConversionWorker, ConversionJob]] = field(default_factory=dict)
    next_worker_index: int = 0
    processed: int = 0
    successes: int = 0
    failures: int = 0
    fatal_error: str | None = None
    starting: bool = False
    needs_pump: bool = False


class ConversionService:
    """Run conversion batches with bounded parallelism."""

    _KEY_PREFIX = "conversion"

    def __init__(self, workers: WorkerCoordinator, *, max_workers: int = 3) -> None:
        self._workers = workers
        self._max_workers = max(1, int(max_workers))
        self._batch: _ConversionBatchState | None = None

    def is_running(self) -> bool:
        return self._batch is not None

    def run(
        self,
        *,
        jobs: Sequence[ConversionJob],
        helper: str,
        options: Mapping[str, object] | None,
        on_progress: Callable[[int, int, str], None],
        on_failed: Callable[[str, str], None],
        on_fatal: Callable[[str], None],
        on_finished: Callable[[int, int], None],
    ) -> bool:
        if self._batch is not None:
            return False

        queued = deque(jobs)
        if not queued:
            return False

        state = _ConversionBatchState(
            batch_id=uuid4().hex[:8],
            helper=helper or "azure_di",
            options=dict(options or {}),
            total=len(queued),
            pending=queued,
        )
        self._batch = state
        self._start_more(
            state,
            on_progress=on_progress,
            on_failed=on_failed,
            on_fatal=on_fatal,
            on_finished=on_finished,
        )
        self._maybe_finalize(state, on_finished)
        return True

    def cancel(self) -> bool:
        state = self._batch
        if state is None:
            return False
        state.pending.clear()
        self._workers.cancel_many(state.in_flight.keys())
        return True

    def _start_more(
        self,
        state: _ConversionBatchState,
        *,
        on_progress: Callable[[int, int, str], None],
        on_failed: Callable[[str, str], None],
        on_fatal: Callable[[str], None],
        on_finished: Callable[[int, int], None],
    ) -> None:
        if self._batch is not state:
            return
        if state.starting:
            state.needs_pump = True
            return

        state.starting = True
        try:
            while self._batch is state:
                state.needs_pump = False
                while (
                    self._batch is state
                    and state.fatal_error is None
                    and state.pending
                    and len(state.in_flight) < self._max_workers
                ):
                    job = state.pending.popleft()
                    worker_key = self._worker_key(state.batch_id, state.next_worker_index)
                    state.next_worker_index += 1
                    worker = ConversionWorker([job], helper=state.helper, options=state.options)
                    state.in_flight[worker_key] = (worker, job)

                    worker.file_failed.connect(on_failed)
                    worker.fatal_error.connect(
                        lambda error, key=worker_key: self._handle_fatal(
                            state,
                            key,
                            error,
                            on_fatal=on_fatal,
                        )
                    )
                    worker.finished.connect(
                        lambda successes, failures, key=worker_key, w=worker, j=job: self._handle_finished(
                            state,
                            key,
                            w,
                            j,
                            successes,
                            failures,
                            on_progress=on_progress,
                            on_failed=on_failed,
                            on_fatal=on_fatal,
                            on_finished=on_finished,
                        )
                    )
                    self._workers.start(worker_key, worker)

                if not state.needs_pump:
                    break
        finally:
            state.starting = False

    def _handle_fatal(
        self,
        state: _ConversionBatchState,
        worker_key: str,
        error: str,
        *,
        on_fatal: Callable[[str], None],
    ) -> None:
        if self._batch is not state:
            return
        if worker_key not in state.in_flight:
            return
        if state.fatal_error is not None:
            return

        state.fatal_error = error
        state.pending.clear()
        cancel_keys = [key for key in state.in_flight.keys() if key != worker_key]
        self._workers.cancel_many(cancel_keys)
        on_fatal(error)

    def _handle_finished(
        self,
        state: _ConversionBatchState,
        worker_key: str,
        worker: ConversionWorker,
        job: ConversionJob,
        successes: int,
        failures: int,
        *,
        on_progress: Callable[[int, int, str], None],
        on_failed: Callable[[str, str], None],
        on_fatal: Callable[[str], None],
        on_finished: Callable[[int, int], None],
    ) -> None:
        stored = self._workers.pop(worker_key)
        _, tracked_job = state.in_flight.pop(worker_key, (worker, job))
        self._delete_worker(worker)
        if stored is not worker:
            self._delete_worker(stored)

        completed = successes + failures
        state.processed += completed
        state.successes += successes
        state.failures += failures
        if completed:
            on_progress(state.processed, state.total, tracked_job.display_name)

        if self._batch is state and state.fatal_error is None:
            self._start_more(
                state,
                on_progress=on_progress,
                on_failed=on_failed,
                on_fatal=on_fatal,
                on_finished=on_finished,
            )
        self._maybe_finalize(state, on_finished)

    def _maybe_finalize(
        self,
        state: _ConversionBatchState,
        on_finished: Callable[[int, int], None],
    ) -> None:
        if self._batch is not state:
            return
        if state.pending or state.in_flight:
            return
        self._batch = None
        on_finished(state.successes, state.failures)

    def _delete_worker(self, worker: ConversionWorker | None) -> None:
        if worker and isValid(worker):
            worker.deleteLater()

    def _worker_key(self, batch_id: str, index: int) -> str:
        return f"{self._KEY_PREFIX}:{batch_id}:{index}"


__all__ = ["ConversionService"]
