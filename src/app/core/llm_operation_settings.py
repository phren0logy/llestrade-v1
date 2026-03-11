"""Shared LLM selection settings and provider catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.app.core.llm_catalog import (
    LLMModelOption,
    LLMProviderOption,
    SUPPORTED_SELECTOR_PROVIDER_IDS,
    default_model_for_provider,
    default_provider_catalog,
    provider_option_map,
)


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

    model_id = custom or stored_model or default_model_for_provider(provider) or ""
    return LLMOperationSettings(
        provider_id=provider,
        model_id=model_id,
        context_window=context_window if custom else None,
        use_reasoning=bool(use_reasoning),
    )


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
