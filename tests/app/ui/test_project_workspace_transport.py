from __future__ import annotations

from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_ = PySide6

from src.app.core.feature_flags import FeatureFlags
from src.app.core.secure_settings import SecureSettings
from src.app.ui.stages import project_workspace as project_workspace_module
from src.app.ui.stages.project_workspace import ProjectWorkspace


class _DirectBackend:
    pass


class _GatewayBackend:
    pass


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_project_workspace_defaults_to_direct_transport(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(project_workspace_module, "PydanticAIDirectBackend", lambda: _DirectBackend())
    monkeypatch.setattr(project_workspace_module, "PydanticAIGatewayBackend", lambda: _GatewayBackend())
    monkeypatch.setattr(
        project_workspace_module,
        "backend_transport_name",
        lambda backend: "gateway" if isinstance(backend, _GatewayBackend) else "direct",
    )

    workspace = ProjectWorkspace(feature_flags=FeatureFlags(pydantic_ai_gateway_enabled=True))
    try:
        assert workspace._llm_transport == "direct"
        assert isinstance(workspace._bulk_service._llm_backend, _DirectBackend)
    finally:
        workspace.deleteLater()


def test_project_workspace_uses_gateway_transport_when_selected_and_available(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    settings = SecureSettings()
    settings.set("llm_transport_mode", "gateway")

    monkeypatch.setattr(project_workspace_module, "PydanticAIDirectBackend", lambda: _DirectBackend())
    monkeypatch.setattr(project_workspace_module, "PydanticAIGatewayBackend", lambda: _GatewayBackend())
    monkeypatch.setattr(
        project_workspace_module,
        "backend_transport_name",
        lambda backend: "gateway" if isinstance(backend, _GatewayBackend) else "direct",
    )

    workspace = ProjectWorkspace(feature_flags=FeatureFlags(pydantic_ai_gateway_enabled=True))
    try:
        assert workspace._llm_transport == "gateway"
        assert isinstance(workspace._bulk_service._llm_backend, _GatewayBackend)
    finally:
        workspace.deleteLater()


def test_project_workspace_falls_back_to_direct_when_gateway_disabled(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    settings = SecureSettings()
    settings.set("llm_transport_mode", "gateway")

    monkeypatch.setattr(project_workspace_module, "PydanticAIDirectBackend", lambda: _DirectBackend())
    monkeypatch.setattr(project_workspace_module, "PydanticAIGatewayBackend", lambda: _GatewayBackend())
    monkeypatch.setattr(
        project_workspace_module,
        "backend_transport_name",
        lambda backend: "gateway" if isinstance(backend, _GatewayBackend) else "direct",
    )

    workspace = ProjectWorkspace(feature_flags=FeatureFlags(pydantic_ai_gateway_enabled=False))
    try:
        assert workspace._llm_transport == "direct"
        assert isinstance(workspace._bulk_service._llm_backend, _DirectBackend)
    finally:
        workspace.deleteLater()
