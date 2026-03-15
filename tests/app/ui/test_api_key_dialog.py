from __future__ import annotations

from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

_ = PySide6

import src.app.ui.widgets.api_key_dialog as api_key_dialog_module
from src.app.core.secure_settings import SecureKeyStorageError, SecureSettings
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


def test_api_key_dialog_show_reveals_saved_gateway_key_without_overwriting(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)

    settings = SecureSettings()
    settings.set_api_key("pydantic_ai_gateway", "gateway-key-1")

    dialog = APIKeyDialog(settings)
    try:
        show_button = next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "Show" and button.parent() is dialog.gateway_api_key.parent()
        )
        assert dialog.gateway_api_key.text() == "*" * 20

        show_button.click()
        assert dialog.gateway_api_key.text() == "gateway-key-1"
        assert show_button.text() == "Hide"

        show_button.click()
        assert dialog.gateway_api_key.text() == "*" * 20
        assert show_button.text() == "Show"

        dialog.save_keys()
    finally:
        dialog.deleteLater()

    reloaded = SecureSettings()
    assert reloaded.get_api_key("pydantic_ai_gateway") == "gateway-key-1"


def test_api_key_dialog_show_reveals_saved_provider_keys_for_all_secret_fields(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)

    settings = SecureSettings()
    settings.set_api_key("anthropic", "anthropic-key-1")
    settings.set_api_key("azure_openai", "azure-openai-key-1")
    settings.set_api_key("azure_di", "azure-di-key-1")

    dialog = APIKeyDialog(settings)
    try:
        assert dialog.api_fields["anthropic"].text() == "*" * 20
        assert dialog.api_fields["azure_openai"].text() == "*" * 20
        assert dialog.api_fields["azure_di"].text() == "*" * 20

        for provider, expected in (
            ("anthropic", "anthropic-key-1"),
            ("azure_openai", "azure-openai-key-1"),
            ("azure_di", "azure-di-key-1"),
        ):
            field = dialog.api_fields[provider]
            show_button = next(
                button
                for button in field.parent().findChildren(QPushButton)
                if button.text() in {"Show", "Hide"}
            )
            show_button.click()
            assert field.text() == expected
            show_button.click()
            assert field.text() == "*" * 20
    finally:
        dialog.deleteLater()


def test_api_key_dialog_reports_mismatched_saved_key(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    settings = SecureSettings()

    def _fake_set_api_key(provider: str, api_key: str) -> bool:
        settings._api_key_cache[provider] = "different-key"
        return True

    monkeypatch.setattr(settings, "set_api_key", _fake_set_api_key)
    monkeypatch.setattr(settings, "get_api_key", lambda provider: "different-key")

    dialog = APIKeyDialog(settings)
    try:
        dialog.gateway_api_key.setText("gateway-key-expected")
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    assert warnings
    assert "does not match the value entered" in warnings[0]


def test_api_key_dialog_reports_secure_storage_failure(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    settings = SecureSettings()

    def _fail_set_api_key(provider: str, api_key: str) -> bool:
        _ = provider, api_key
        raise SecureKeyStorageError("keychain write failed")

    monkeypatch.setattr(settings, "set_api_key", _fail_set_api_key)

    dialog = APIKeyDialog(settings)
    try:
        dialog.gateway_api_key.setText("gateway-key-expected")
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    assert warnings
    assert "Error saving pydantic_ai_gateway: keychain write failed" in warnings[0]


def test_api_key_dialog_resets_gateway_probe_cache_and_warns_on_invalid_gateway_key(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    settings = SecureSettings()
    settings.set_api_key("pydantic_ai_gateway", "gateway-key-1")

    reset_calls = {"count": 0}
    catalog_reset_calls = {"count": 0}
    gateway_refresh_calls = {"count": 0}

    def _fake_reset() -> None:
        reset_calls["count"] += 1

    def _fake_catalog_reset() -> None:
        catalog_reset_calls["count"] += 1

    def _fake_gateway_refresh(*, force: bool = False) -> None:
        assert force is True
        gateway_refresh_calls["count"] += 1

    monkeypatch.setattr(api_key_dialog_module, "reset_gateway_access_check_cache", _fake_reset)
    monkeypatch.setattr(api_key_dialog_module, "reset_provider_catalog_cache", _fake_catalog_reset)
    monkeypatch.setattr(api_key_dialog_module, "refresh_gateway_provider_catalog", _fake_gateway_refresh)
    monkeypatch.setattr(
        api_key_dialog_module,
        "default_provider_catalog_for_transport",
        lambda **_kwargs: (type("Provider", (), {"provider_id": "anthropic", "models": (type("Model", (), {"model_id": "claude-sonnet-4-5"})(),)})(),),
    )

    class _Result:
        ok = False
        kind = "auth_invalid"
        status_code = 401
        message = "Unauthorized - Key not found"
        route = "bulk"
        provider_id = "anthropic"
        model = "claude-sonnet-4-5"

    class _Backend:
        def __init__(self, *, api_key=None, base_url=None, route=None) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.route = route

        def verify_gateway_access(self, provider_id: str, model: str | None, *, timeout_seconds: float = 5.0, force: bool = False):  # noqa: ANN001
            assert self.api_key == "gateway-key-2"
            assert self.base_url == "https://gateway.example.com"
            assert self.route == "bulk"
            assert provider_id == "anthropic"
            assert model == "claude-sonnet-4-5"
            assert timeout_seconds == 5.0
            assert force is True
            return _Result()

    monkeypatch.setattr(
        api_key_dialog_module,
        "PydanticAIGatewayBackend",
        _Backend,
    )

    dialog = APIKeyDialog(settings)
    try:
        dialog.gateway_base_url.setText("https://gateway.example.com")
        dialog.gateway_route.setText("bulk")
        dialog.gateway_api_key.setText("gateway-key-2")
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    assert reset_calls["count"] == 1
    assert catalog_reset_calls["count"] == 1
    assert gateway_refresh_calls["count"] == 1
    assert warnings
    assert "rejected by the gateway" in warnings[0]


def test_api_key_dialog_loads_and_saves_observability_settings(
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
        "observability_settings",
        {
            "enabled": True,
            "target": "phoenix_local",
            "project": "forensic-report-drafter",
            "content_policy": "redacted",
            "include_binary_content": True,
            "phoenix_port": 6006,
            "otlp_endpoint": None,
            "otlp_headers": {},
        },
    )

    dialog = APIKeyDialog(settings)
    try:
        assert dialog.observability_enabled.isChecked() is True
        assert dialog.observability_target.currentData() == "phoenix_local"
        assert dialog.observability_content_policy.currentData() == "redacted"
        assert dialog.observability_include_binary_content.isChecked() is True

        policy_index = dialog.observability_content_policy.findData("unredacted")
        dialog.observability_content_policy.setCurrentIndex(policy_index)
        dialog.observability_include_binary_content.setChecked(False)
        otlp_index = dialog.observability_target.findData("otlp_http")
        dialog.observability_target.setCurrentIndex(otlp_index)
        dialog.observability_otlp_endpoint.setText("https://otel.example.com/v1/traces")
        dialog.observability_otlp_headers.setPlainText('{"Authorization": "Bearer token"}')
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    reloaded = SecureSettings()
    assert reloaded.get("observability_settings") == {
        "enabled": True,
        "target": "otlp_http",
        "project": "forensic-report-drafter",
        "content_policy": "unredacted",
        "include_binary_content": False,
        "phoenix_port": None,
        "otlp_endpoint": "https://otel.example.com/v1/traces",
        "otlp_headers": {"Authorization": "Bearer token"},
    }


def test_api_key_dialog_rejects_invalid_otlp_headers(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    settings = SecureSettings()

    dialog = APIKeyDialog(settings)
    try:
        otlp_index = dialog.observability_target.findData("otlp_http")
        dialog.observability_target.setCurrentIndex(otlp_index)
        dialog.observability_enabled.setChecked(True)
        dialog.observability_otlp_endpoint.setText("https://otel.example.com/v1/traces")
        dialog.observability_otlp_headers.setPlainText('["not", "a", "dict"]')
        dialog.save_keys()
    finally:
        dialog.deleteLater()

    assert warnings
    assert "Invalid OTLP headers JSON" in warnings[0]


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
