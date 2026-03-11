"""Shared LLM selection settings and provider catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Sequence

from src.app.core.secure_settings import SecureSettings
from src.common.llm.bedrock_catalog import DEFAULT_BEDROCK_MODELS, list_bedrock_models

SUPPORTED_SELECTOR_PROVIDER_IDS: tuple[str, ...] = (
    "anthropic",
    "anthropic_bedrock",
    "openai",
    "gemini",
)


@dataclass(frozen=True, slots=True)
class LLMModelOption:
    """Single preset model entry for a provider."""

    model_id: str
    label: str


@dataclass(frozen=True, slots=True)
class LLMProviderOption:
    """Selector metadata for a supported LLM provider."""

    provider_id: str
    label: str
    models: tuple[LLMModelOption, ...]


@dataclass(frozen=True, slots=True)
class LLMOperationSettings:
    """Portable LLM selection shared across UI and workers."""

    provider_id: str
    model_id: str
    context_window: Optional[int] = None
    use_reasoning: bool = False

    @property
    def custom_model_id(self) -> str | None:
        return self.model_id if self.context_window is not None else None


def infer_provider_id_from_model(model_id: str | None) -> str | None:
    """Infer a provider from a legacy custom-model string when possible."""
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return None
    if normalized.startswith("anthropic."):
        return "anthropic_bedrock"
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("gpt-") or normalized.startswith("o1") or normalized.startswith("o3"):
        return "openai"
    if normalized.startswith("gemini"):
        return "gemini"
    return None


def settings_from_report_preferences(
    *,
    provider_id: str | None,
    model: str | None,
    custom_model: str | None,
    context_window: int | None,
    use_reasoning: bool = False,
) -> LLMOperationSettings:
    """Normalize persisted report prefs into shared operation settings."""
    provider = str(provider_id or "").strip()
    stored_model = str(model or "").strip()
    custom = str(custom_model or "").strip()

    if provider == "custom":
        inferred = infer_provider_id_from_model(custom or stored_model)
        provider = inferred or "anthropic"

    if not provider:
        provider = "anthropic"

    model_id = custom or stored_model or default_model_for_provider(provider)
    return LLMOperationSettings(
        provider_id=provider,
        model_id=model_id,
        context_window=context_window if custom else None,
        use_reasoning=bool(use_reasoning),
    )


def default_provider_catalog(
    *,
    include_azure: bool = False,
) -> tuple[LLMProviderOption, ...]:
    """Return the shared provider/model selector catalog for the UI."""
    providers: list[LLMProviderOption] = [
        LLMProviderOption(
            provider_id="anthropic",
            label="Anthropic Claude",
            models=(
                LLMModelOption("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
                LLMModelOption("claude-opus-4-6", "Claude Opus 4.6"),
            ),
        ),
        LLMProviderOption(
            provider_id="anthropic_bedrock",
            label="AWS Bedrock (Claude)",
            models=_bedrock_model_options(),
        ),
        LLMProviderOption(
            provider_id="openai",
            label="OpenAI",
            models=(
                LLMModelOption(
                    default_model_for_provider("openai"),
                    default_model_for_provider("openai"),
                ),
            ),
        ),
        LLMProviderOption(
            provider_id="gemini",
            label="Google Gemini",
            models=(
                LLMModelOption(
                    default_model_for_provider("gemini"),
                    default_model_for_provider("gemini"),
                ),
            ),
        ),
    ]
    if include_azure:
        providers.append(
            LLMProviderOption(
                provider_id="azure_openai",
                label="Azure OpenAI",
                models=(
                    LLMModelOption(
                        default_model_for_provider("azure_openai"),
                        default_model_for_provider("azure_openai"),
                    ),
                ),
            )
        )
    return tuple(provider for provider in providers if provider.models)


def provider_option_map(
    options: Sequence[LLMProviderOption],
) -> dict[str, LLMProviderOption]:
    return {option.provider_id: option for option in options}


def _bedrock_model_options() -> tuple[LLMModelOption, ...]:
    try:
        settings = SecureSettings()
        bedrock_settings = settings.get("aws_bedrock_settings", {}) or {}
        bedrock_models = list_bedrock_models(
            region=bedrock_settings.get("region"),
            profile=bedrock_settings.get("profile"),
        )
    except Exception:
        bedrock_models = list(DEFAULT_BEDROCK_MODELS)

    return tuple(
        LLMModelOption(model.model_id, model.name or model.model_id)
        for model in bedrock_models
    )


def default_model_for_provider(provider_id: str) -> str:
    defaults = {
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "anthropic_bedrock": os.getenv("BEDROCK_ANTHROPIC_MODEL", "anthropic.claude-sonnet-4-5-v1"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4.1"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        "azure_openai": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
    }
    return defaults.get(provider_id, "")


__all__ = [
    "LLMModelOption",
    "LLMOperationSettings",
    "LLMProviderOption",
    "SUPPORTED_SELECTOR_PROVIDER_IDS",
    "default_provider_catalog",
    "infer_provider_id_from_model",
    "provider_option_map",
    "settings_from_report_preferences",
]
