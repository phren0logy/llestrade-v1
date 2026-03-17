from src.app.core import llm_catalog
from src.app.core.llm_operation_settings import (
    LLMReasoningSettings,
    default_provider_catalog,
    infer_provider_id_from_model,
    settings_from_report_preferences,
)


def test_settings_from_report_preferences_normalizes_legacy_custom_provider() -> None:
    settings = settings_from_report_preferences(
        provider_id="custom",
        model="",
        custom_model="gpt-4.1",
        context_window=100_000,
        use_reasoning=True,
    )

    assert settings.provider_id == "openai"
    assert settings.model_id == "gpt-4.1"
    assert settings.context_window is None
    assert settings.use_reasoning is True


def test_settings_from_report_preferences_preserves_rich_reasoning_settings() -> None:
    settings = settings_from_report_preferences(
        provider_id="openai",
        model="gpt-5.1",
        custom_model="",
        context_window=None,
        reasoning={
            "state": "on",
            "effort": "high",
        },
    )

    assert settings.reasoning == LLMReasoningSettings(state="on", effort="high")
    assert settings.use_reasoning is True


def test_settings_from_report_preferences_preserves_context_for_unknown_custom_model() -> None:
    settings = settings_from_report_preferences(
        provider_id="custom",
        model="",
        custom_model="gpt-custom-preview",
        context_window=100_000,
        use_reasoning=False,
    )

    assert settings.provider_id == "openai"
    assert settings.model_id == "gpt-custom-preview"
    assert settings.context_window == 100_000


def test_infer_provider_id_from_model_supports_current_selector_families() -> None:
    assert infer_provider_id_from_model("claude-sonnet-4-5") == "anthropic"
    assert infer_provider_id_from_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0") == "anthropic_bedrock"
    assert infer_provider_id_from_model("gemini-2.5-pro") == "gemini"
    assert infer_provider_id_from_model("gpt-4.1") == "openai"


def test_default_provider_catalog_exposes_openai_and_gemini_but_not_azure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_catalog,
        "_gemini_runtime_models",
        lambda: (
            llm_catalog.LLMModelOption(
                model_id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                context_window=1_048_576,
            ),
        ),
    )
    provider_ids = {option.provider_id for option in default_provider_catalog()}

    assert "anthropic" in provider_ids
    assert "openai" in provider_ids
    assert "gemini" in provider_ids
    assert "azure_openai" not in provider_ids
