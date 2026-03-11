from __future__ import annotations

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_ = PySide6

from src.app.core.llm_operation_settings import LLMOperationSettings
from src.app.ui.widgets import LLMSettingsPanel


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_panel_provider_switch_updates_model_options(qt_app: QApplication) -> None:
    assert qt_app is not None
    panel = LLMSettingsPanel()
    try:
        provider_index = panel.provider_combo.findData("openai")
        assert provider_index != -1
        panel.provider_combo.setCurrentIndex(provider_index)

        model_ids = [panel.model_combo.itemData(index) for index in range(panel.model_combo.count())]
        assert "gpt-4.1" in model_ids
    finally:
        panel.deleteLater()


def test_panel_returns_provider_bound_custom_model_settings(qt_app: QApplication) -> None:
    assert qt_app is not None
    panel = LLMSettingsPanel()
    try:
        panel.set_settings(
            LLMOperationSettings(
                provider_id="gemini",
                model_id="gemini-2.5-flash",
                context_window=1_000_000,
                use_reasoning=True,
            )
        )

        settings, error = panel.current_settings()

        assert error is None
        assert settings is not None
        assert settings.provider_id == "gemini"
        assert settings.model_id == "gemini-2.5-flash"
        assert settings.context_window == 1_000_000
        assert settings.use_reasoning is True
    finally:
        panel.deleteLater()


def test_panel_advanced_reasoning_starts_collapsed(qt_app: QApplication) -> None:
    assert qt_app is not None
    panel = LLMSettingsPanel()
    try:
        assert panel.advanced_toggle.isChecked() is False
        assert panel.advanced_frame.isVisible() is False
    finally:
        panel.deleteLater()
