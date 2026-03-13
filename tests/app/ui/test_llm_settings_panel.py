from __future__ import annotations

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_ = PySide6

from src.app.core import llm_catalog
from src.app.core.llm_operation_settings import LLMOperationSettings
from src.app.ui.widgets import LLMSettingsPanel
from src.app.ui.widgets import llm_settings_panel as llm_settings_panel_module


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def stub_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_reasoning = llm_catalog.LLMReasoningCapabilities(
        supports_reasoning_controls=True,
        can_disable_reasoning=True,
        controls=("toggle", "budget"),
        budget_step=1024,
        default_state="off",
    )
    openai_reasoning = llm_catalog.LLMReasoningCapabilities(
        supports_reasoning_controls=True,
        can_disable_reasoning=True,
        controls=("toggle", "effort"),
        allowed_efforts=("low", "medium", "high"),
        default_state="off",
    )
    gemini_reasoning = llm_catalog.LLMReasoningCapabilities(
        supports_reasoning_controls=True,
        can_disable_reasoning=True,
        controls=("toggle", "budget"),
        budget_step=1024,
        default_state="off",
    )
    providers = (
        llm_catalog.LLMProviderOption(
            provider_id="anthropic",
            label="Anthropic Claude",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="claude-sonnet-4-5",
                    label="Claude Sonnet 4.5",
                    context_window=200_000,
                    input_price_label="$3/1M",
                    output_price_label="$15/1M",
                    reasoning_capabilities=anthropic_reasoning,
                ),
            ),
        ),
        llm_catalog.LLMProviderOption(
            provider_id="openai",
            label="OpenAI",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="gpt-4.1",
                    label="GPT-4.1",
                    context_window=1_047_576,
                    input_price_label="$2/1M",
                    output_price_label="$8/1M",
                    reasoning_capabilities=openai_reasoning,
                ),
            ),
        ),
        llm_catalog.LLMProviderOption(
            provider_id="gemini",
            label="Google Gemini",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="gemini-2.5-pro",
                    label="Gemini 2.5 Pro",
                    context_window=1_048_576,
                    input_price_label="$1.25/1M",
                    output_price_label="$10/1M",
                    reasoning_capabilities=gemini_reasoning,
                ),
                llm_catalog.LLMModelOption(
                    model_id="gemini-2.5-flash",
                    label="Gemini 2.5 Flash",
                    context_window=1_048_576,
                    input_price_label="$0.3/1M",
                    output_price_label="$2.5/1M",
                    reasoning_capabilities=gemini_reasoning,
                ),
            ),
        ),
    )
    model_lookup = {
        (provider.provider_id, model.model_id): model
        for provider in providers
        for model in provider.models
    }

    monkeypatch.setattr(
        llm_settings_panel_module,
        "default_provider_catalog_for_transport",
        lambda include_azure=False, transport="direct": providers,
    )
    monkeypatch.setattr(
        llm_settings_panel_module,
        "resolve_catalog_model",
        lambda provider_id, model_id, transport="direct": model_lookup.get((provider_id, str(model_id or "").strip())),
    )


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


def test_panel_exposes_provider_native_reasoning_controls(qt_app: QApplication) -> None:
    assert qt_app is not None
    panel = LLMSettingsPanel()
    try:
        provider_index = panel.provider_combo.findData("openai")
        panel.provider_combo.setCurrentIndex(provider_index)

        state_options = [panel.reasoning_state_combo.itemText(i) for i in range(panel.reasoning_state_combo.count())]
        effort_options = [panel.reasoning_effort_combo.itemData(i) for i in range(panel.reasoning_effort_combo.count())]

        assert state_options == ["Off", "On"]
        assert effort_options == ["low", "medium", "high"]
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


def test_panel_shows_catalog_details_for_selected_model(qt_app: QApplication) -> None:
    assert qt_app is not None
    panel = LLMSettingsPanel()
    try:
        provider_index = panel.provider_combo.findData("openai")
        panel.provider_combo.setCurrentIndex(provider_index)

        model_index = panel.model_combo.findData("gpt-4.1")
        panel.model_combo.setCurrentIndex(model_index)

        details = panel.model_details_label.text()
        assert "Context:" in details
        assert "Input:" in details
        assert "Output:" in details
    finally:
        panel.deleteLater()
