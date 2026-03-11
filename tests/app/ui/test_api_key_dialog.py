from __future__ import annotations

from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

_ = PySide6

from src.app.core.secure_settings import SecureSettings
from src.app.ui.widgets.api_key_dialog import APIKeyDialog


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_api_key_dialog_loads_and_saves_gateway_settings(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)

    settings = SecureSettings()
    settings.set(
        "pydantic_ai_gateway_settings",
        {
            "base_url": "https://gateway.example.com",
            "route": "llestrade",
        },
    )
    settings.set_api_key("pydantic_ai_gateway", "gateway-key-1")

    dialog = APIKeyDialog(settings)
    try:
        assert dialog.gateway_base_url.text() == "https://gateway.example.com"
        assert dialog.gateway_route.text() == "llestrade"
        assert dialog.gateway_api_key.property("has_saved_key") is True
        assert dialog.gateway_api_key.text() == "*" * 20

        dialog.gateway_base_url.setText("https://gateway.internal.example.com")
        dialog.gateway_route.setText("bulk")
        dialog.gateway_api_key.setText("gateway-key-2")
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    reloaded = SecureSettings()
    assert reloaded.get("pydantic_ai_gateway_settings") == {
        "base_url": "https://gateway.internal.example.com",
        "route": "bulk",
    }
    assert reloaded.get_api_key("pydantic_ai_gateway") == "gateway-key-2"


def test_api_key_dialog_loads_and_saves_phoenix_content_policy(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)

    settings = SecureSettings()
    settings.set(
        "phoenix_settings",
        {
            "enabled": True,
            "target": "local_phoenix",
            "port": 6006,
            "project": "forensic-report-drafter",
            "export_fixtures": False,
            "content_policy": "redacted",
            "include_binary_content": True,
        },
    )

    dialog = APIKeyDialog(settings)
    try:
        assert dialog.phoenix_enabled.isChecked() is True
        assert dialog.phoenix_content_policy.currentData() == "redacted"
        assert dialog.phoenix_include_binary_content.isChecked() is True

        policy_index = dialog.phoenix_content_policy.findData("unredacted")
        dialog.phoenix_content_policy.setCurrentIndex(policy_index)
        dialog.phoenix_include_binary_content.setChecked(False)
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    reloaded = SecureSettings()
    assert reloaded.get("phoenix_settings") == {
        "enabled": True,
        "target": "local_phoenix",
        "port": 6006,
        "project": "forensic-report-drafter",
        "export_fixtures": False,
        "content_policy": "unredacted",
        "include_binary_content": False,
    }


def test_api_key_dialog_excludes_bedrock_controls(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))

    dialog = APIKeyDialog(SecureSettings())
    try:
        assert not hasattr(dialog, "bedrock_model_combo")
        assert not hasattr(dialog, "bedrock_profile")
        assert "aws_bedrock_profile" not in dialog.config_fields
        assert "aws_bedrock_region" not in dialog.config_fields
    finally:
        dialog.deleteLater()
