"""Focused tests for main-window orchestration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip("PySide6")
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
