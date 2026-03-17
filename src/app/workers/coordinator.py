"""Simple coordinator for dashboard workers."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

from PySide6.QtCore import QThreadPool

from .base import DashboardWorker


class WorkerCoordinator:
    """Manage dashboard workers on the shared thread pool.

    Tracks workers by identifier so callers can cancel or remove them without
    storing additional book-keeping structures. Identifiers are opaque strings
    chosen by the caller (e.g., "conversion:run" or "bulk:group-id").
    """

    def __init__(self, pool: Optional[QThreadPool] = None) -> None:
        from .pool import get_worker_pool

        self._pool: QThreadPool = pool or get_worker_pool()
        self._workers: Dict[str, DashboardWorker] = {}
        self._logger = logging.getLogger(__name__ + ".coordinator")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, key: str, worker: DashboardWorker) -> None:
        """Start `worker` on the thread pool and register it under `key`."""
        self._workers[key] = worker
        try:
            self._logger.info("%s enqueued key=%s", getattr(worker, "job_tag", "[unknown]"), key)
        except Exception:
            pass
        self._pool.start(worker)

    def register(self, key: str, worker: DashboardWorker) -> None:
        """Store `worker` under `key` without starting it."""
        self._workers[key] = worker
        try:
            self._logger.debug("%s registered key=%s", getattr(worker, "job_tag", "[unknown]"), key)
        except Exception:
            pass

    def get(self, key: str) -> Optional[DashboardWorker]:
        return self._workers.get(key)

    def pop(self, key: str) -> Optional[DashboardWorker]:
        return self._workers.pop(key, None)

    def cancel(self, key: str) -> bool:
        worker = self._workers.get(key)
        if not worker:
            return False
        worker.cancel()
        try:
            self._logger.info("%s cancel requested key=%s", getattr(worker, "job_tag", "[unknown]"), key)
        except Exception:
            pass
        return True

    def cancel_many(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.cancel(key)

    def cancel_all(self) -> list[str]:
        keys = list(self._workers.keys())
        self.cancel_many(keys)
        return keys

    def active_keys(self) -> list[str]:
        return list(self._workers.keys())

    def wait_for_done(self, timeout_ms: int) -> bool:
        timeout = max(-1, int(timeout_ms))
        try:
            return bool(self._pool.waitForDone(timeout))
        except TypeError:  # pragma: no cover - older Qt variants
            self._pool.waitForDone()
            return True

    def clear(self) -> None:
        # Cancel and safely delete all tracked workers
        try:
            from shiboken6 import isValid  # type: ignore
        except Exception:  # pragma: no cover - fallback if not available
            def isValid(obj):
                return True

        for worker in list(self._workers.values()):
            try:
                worker.cancel()
            except Exception:
                pass
            try:
                if isValid(worker):
                    try:
                        self._logger.debug("%s cleared", getattr(worker, "job_tag", "[unknown]"))
                    except Exception:
                        pass
                    worker.deleteLater()
            except Exception:
                pass
        self._workers.clear()


__all__ = ["WorkerCoordinator"]
