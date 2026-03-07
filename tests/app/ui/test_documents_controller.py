"""Focused tests for DocumentsController run orchestration paths."""

from __future__ import annotations

from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

_ = PySide6

from src.app.core.conversion_manager import ConversionJob
from src.app.core.conversion_manager import ConversionPlan
from src.app.core.conversion_manager import DuplicateSource
from src.app.ui.workspace.controllers.documents import DocumentsController
from src.app.ui.workspace.documents_tab import DocumentsTab


class _WorkspaceStub(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._conversion_running = False
        self._inflight_sources: set[Path] = set()


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_controller() -> tuple[_WorkspaceStub, DocumentsController, list[list[ConversionJob]]]:
    workspace = _WorkspaceStub()
    tab = DocumentsTab(parent=workspace)
    captured_runs: list[list[ConversionJob]] = []

    def _run_conversion(jobs: list[ConversionJob]) -> None:
        captured_runs.append(jobs)

    controller = DocumentsController(workspace, tab, _run_conversion)
    controller._project_manager = object()  # type: ignore[assignment]
    return workspace, controller, captured_runs


def test_trigger_conversion_runs_after_confirmation(monkeypatch: pytest.MonkeyPatch, qt_app: QApplication, tmp_path: Path) -> None:
    assert qt_app is not None
    workspace, controller, captured_runs = _build_controller()

    job = ConversionJob(
        source_path=tmp_path / "source.pdf",
        relative_path="source.pdf",
        destination_path=tmp_path / "converted" / "source.md",
        conversion_type="pdf",
    )

    monkeypatch.setattr(controller, "collect_conversion_jobs", lambda _inflight: ([job], []))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    controller.trigger_conversion(auto_run=False)

    assert len(captured_runs) == 1
    assert captured_runs[0] == [job]

    workspace.deleteLater()


def test_trigger_conversion_auto_run_skips_confirmation(monkeypatch: pytest.MonkeyPatch, qt_app: QApplication, tmp_path: Path) -> None:
    assert qt_app is not None
    workspace, controller, captured_runs = _build_controller()

    job = ConversionJob(
        source_path=tmp_path / "doc.txt",
        relative_path="doc.txt",
        destination_path=tmp_path / "converted" / "doc.md",
        conversion_type="text",
    )

    monkeypatch.setattr(controller, "collect_conversion_jobs", lambda _inflight: ([job], []))

    def _unexpected_question(*_args, **_kwargs):
        raise AssertionError("question dialog should not be shown for auto runs")

    monkeypatch.setattr(QMessageBox, "question", _unexpected_question)

    controller.trigger_conversion(auto_run=True)

    assert len(captured_runs) == 1
    assert captured_runs[0] == [job]

    workspace.deleteLater()


def test_trigger_conversion_shows_info_when_no_jobs(monkeypatch: pytest.MonkeyPatch, qt_app: QApplication) -> None:
    assert qt_app is not None
    workspace, controller, captured_runs = _build_controller()

    notices: list[str] = []
    monkeypatch.setattr(controller, "collect_conversion_jobs", lambda _inflight: ([], []))
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, message: notices.append(message),
    )

    controller.trigger_conversion(auto_run=False)

    assert captured_runs == []
    assert notices == ["No new files detected."]

    workspace.deleteLater()


def test_collect_conversion_jobs_filters_inflight_and_preserves_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    workspace, controller, _captured_runs = _build_controller()

    first = ConversionJob(
        source_path=tmp_path / "a.pdf",
        relative_path="a.pdf",
        destination_path=tmp_path / "converted" / "a.md",
        conversion_type="pdf",
    )
    second = ConversionJob(
        source_path=tmp_path / "b.pdf",
        relative_path="b.pdf",
        destination_path=tmp_path / "converted" / "b.md",
        conversion_type="pdf",
    )
    duplicate = DuplicateSource(
        digest="abc123",
        primary_relative="a.pdf",
        duplicate_relative="b.pdf",
    )

    monkeypatch.setattr(
        "src.app.ui.workspace.controllers.documents.build_conversion_jobs",
        lambda _pm: ConversionPlan(jobs=(first, second), duplicates=(duplicate,)),
    )

    jobs, duplicates = controller.collect_conversion_jobs({second.source_path})

    assert jobs == [first]
    assert list(duplicates) == [duplicate]

    workspace.deleteLater()
