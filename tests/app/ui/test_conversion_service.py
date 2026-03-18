from __future__ import annotations

from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtWidgets import QApplication

from src.app.core.conversion_manager import ConversionJob
from src.app.ui.workspace.services.conversion import ConversionService
from src.app.workers import WorkerCoordinator

_ = PySide6


class _CaptureThreadPool:
    def __init__(self) -> None:
        self.started: list[QRunnable] = []

    def start(self, worker: QRunnable) -> None:
        self.started.append(worker)


class _StubConversionWorker(QObject, QRunnable):
    progress = Signal(int, int, str)
    file_failed = Signal(str, str)
    fatal_error = Signal(str)
    finished = Signal(int, int)

    def __init__(self, jobs, *, helper: str = "docling", options=None) -> None:  # noqa: ANN001
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.setAutoDelete(False)
        self.job = list(jobs)[0]
        self.helper = helper
        self.options = dict(options or {})
        self.cancel_called = False

    def run(self) -> None:  # pragma: no cover - workers are manually driven in tests
        return

    def cancel(self) -> None:
        self.cancel_called = True


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _jobs(tmp_path: Path, count: int) -> list[ConversionJob]:
    jobs: list[ConversionJob] = []
    for idx in range(count):
        jobs.append(
            ConversionJob(
                source_path=tmp_path / f"doc-{idx}.pdf",
                relative_path=f"folder/doc-{idx}.pdf",
                destination_path=tmp_path / "converted" / f"doc-{idx}.pdf.doctags.txt",
                conversion_type="pdf",
            )
        )
    return jobs


def test_conversion_service_limits_parallel_workers_and_replenishes_queue(
    qt_app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    pool = _CaptureThreadPool()
    coordinator = WorkerCoordinator(pool)
    service = ConversionService(coordinator, max_workers=3)
    jobs = _jobs(tmp_path, 5)

    progress: list[tuple[int, int, str]] = []
    finished: list[tuple[int, int]] = []

    monkeypatch.setattr(
        "src.app.ui.workspace.services.conversion.ConversionWorker",
        _StubConversionWorker,
    )

    started = service.run(
        jobs=jobs,
        helper="docling",
        options={},
        on_progress=lambda processed, total, relative: progress.append((processed, total, relative)),
        on_failed=lambda *_args: None,
        on_fatal=lambda _error: None,
        on_finished=lambda success, failure: finished.append((success, failure)),
    )

    assert started is True
    assert len(pool.started) == 3

    first, second, third = pool.started[:3]
    assert isinstance(first, _StubConversionWorker)
    assert isinstance(second, _StubConversionWorker)
    assert isinstance(third, _StubConversionWorker)

    first.finished.emit(1, 0)
    assert progress == [(1, 5, "folder/doc-0.pdf")]
    assert len(pool.started) == 4

    second.finished.emit(1, 0)
    assert progress[-1] == (2, 5, "folder/doc-1.pdf")
    assert len(pool.started) == 5

    third.finished.emit(1, 0)
    pool.started[3].finished.emit(1, 0)
    pool.started[4].finished.emit(1, 0)

    assert progress[-1] == (5, 5, "folder/doc-4.pdf")
    assert finished == [(5, 0)]
    assert service.is_running() is False


def test_conversion_service_stops_launching_new_jobs_after_fatal_error(
    qt_app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    pool = _CaptureThreadPool()
    coordinator = WorkerCoordinator(pool)
    service = ConversionService(coordinator, max_workers=3)
    jobs = _jobs(tmp_path, 4)

    fatals: list[str] = []
    finished: list[tuple[int, int]] = []

    monkeypatch.setattr(
        "src.app.ui.workspace.services.conversion.ConversionWorker",
        _StubConversionWorker,
    )

    service.run(
        jobs=jobs,
        helper="docling",
        options={},
        on_progress=lambda *_args: None,
        on_failed=lambda *_args: None,
        on_fatal=fatals.append,
        on_finished=lambda success, failure: finished.append((success, failure)),
    )

    assert len(pool.started) == 3
    first, second, third = pool.started[:3]
    assert isinstance(first, _StubConversionWorker)
    assert isinstance(second, _StubConversionWorker)
    assert isinstance(third, _StubConversionWorker)

    first.fatal_error.emit("Docling runtime unavailable")
    assert second.cancel_called is True
    assert third.cancel_called is True

    first.finished.emit(0, 1)
    second.finished.emit(1, 0)
    third.finished.emit(1, 0)

    assert fatals == ["Docling runtime unavailable"]
    assert len(pool.started) == 3
    assert finished == [(2, 1)]
    assert service.is_running() is False
