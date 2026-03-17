"""Focused tests for main-window orchestration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

_ = PySide6

import src.app.main_window as main_window_module


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_window(monkeypatch: pytest.MonkeyPatch) -> main_window_module.SimplifiedMainWindow:
    monkeypatch.setattr(main_window_module.QTimer, "singleShot", staticmethod(lambda *_a, **_k: None))
    return main_window_module.SimplifiedMainWindow()


def test_project_relative_path_handles_internal_and_external(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    window = _build_window(monkeypatch)

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    inside = project_dir / "sources" / "doc.pdf"
    inside.parent.mkdir(parents=True)
    inside.write_text("x", encoding="utf-8")

    outside = tmp_path / "external" / "doc.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_text("x", encoding="utf-8")

    inside_rel = window._project_relative_path(project_dir, inside)
    outside_rel = window._project_relative_path(project_dir, outside)

    assert inside_rel == "sources/doc.pdf"
    assert outside_rel == Path(os.path.relpath(outside.resolve(), project_dir.resolve())).as_posix()

    window.deleteLater()


def test_update_window_title_uses_case_name(monkeypatch: pytest.MonkeyPatch, qt_app: QApplication) -> None:
    assert qt_app is not None
    window = _build_window(monkeypatch)

    window._update_window_title(None)
    assert window.windowTitle() == "Llestrade"

    manager_stub = SimpleNamespace(metadata=SimpleNamespace(case_name="Case A"))
    window._update_window_title(manager_stub)
    assert window.windowTitle() == "Llestrade — Case A"

    window.deleteLater()


def test_activate_workspace_does_not_reapply_project_to_created_workspace(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    window = _build_window(monkeypatch)

    class _WorkspaceStub:
        def __init__(self) -> None:
            self.set_project_calls = 0

        def set_project(self, _manager) -> None:
            self.set_project_calls += 1

    workspace = _WorkspaceStub()
    manager_stub = SimpleNamespace(metadata=SimpleNamespace(case_name="Case A"))

    monkeypatch.setattr(window, "_teardown_workspace", lambda close_project=True: None)
    monkeypatch.setattr(window, "_display_workspace", lambda _workspace: None)
    monkeypatch.setattr(window, "_update_window_title", lambda _manager: None)
    monkeypatch.setattr(window.workspace_controller, "create_workspace", lambda _manager: workspace)

    window._activate_workspace(manager_stub)

    assert workspace.set_project_calls == 0

    window.deleteLater()


def test_close_event_aborts_when_workspace_close_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    window = _build_window(monkeypatch)

    monkeypatch.setattr(window, "_shutdown_workspace_for_exit", lambda: False)

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is False

    window.deleteLater()


def test_close_event_tears_down_workspace_before_stopping_catalog_refresh(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    window = _build_window(monkeypatch)
    called = {"shutdown": 0, "refresh": 0}

    monkeypatch.setattr(
        window,
        "_shutdown_workspace_for_exit",
        lambda: called.__setitem__("shutdown", called["shutdown"] + 1) or True,
    )
    monkeypatch.setattr(
        main_window_module,
        "stop_background_catalog_refresh",
        lambda: called.__setitem__("refresh", called["refresh"] + 1),
    )

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is True
    assert called == {"shutdown": 1, "refresh": 1}

    window.deleteLater()
