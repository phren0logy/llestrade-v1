"""Focused tests for ReportsController placeholder-validation flow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

_ = PySide6

import src.app.ui.workspace.controllers.reports as reports_module
from src.app.ui.workspace.controllers.reports import ReportsController
from src.app.ui.workspace.reports_tab import ReportsTab


class _ServiceStub:
    def is_running(self) -> bool:
        return False


class _ManagerStub:
    def __init__(self, project_dir: Path, values: dict[str, str]) -> None:
        self.project_dir = project_dir
        self._values = values

    def placeholder_mapping(self) -> dict[str, str]:
        return dict(self._values)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_controller(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ReportsController:
    workspace = QWidget()
    tab = ReportsTab(parent=workspace)
    return ReportsController(workspace, tab, service=_ServiceStub())


def test_validate_placeholders_can_block_run_on_missing_required(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    controller._tab.generation_user_prompt_edit.setText("Prompt needs {client_name}")

    monkeypatch.setattr(controller, "_read_prompt_file", lambda path: path)
    monkeypatch.setattr(
        reports_module,
        "get_prompt_spec",
        lambda _key: SimpleNamespace(required=("client_name",), optional=()),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    allowed = controller._validate_placeholders_before_run(
        include_generation=True,
        include_refinement=False,
    )

    assert allowed is False


def test_validate_placeholders_can_continue_on_user_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    controller._tab.generation_user_prompt_edit.setText("Prompt needs {client_name}")

    monkeypatch.setattr(controller, "_read_prompt_file", lambda path: path)
    monkeypatch.setattr(
        reports_module,
        "get_prompt_spec",
        lambda _key: SimpleNamespace(required=("client_name",), optional=()),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    allowed = controller._validate_placeholders_before_run(
        include_generation=True,
        include_refinement=False,
    )

    assert allowed is True
