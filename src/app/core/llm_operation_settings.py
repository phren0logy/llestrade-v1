"""Shared LLM selection settings and provider catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.app.core.llm_catalog import (
    CatalogTransport,
    LLMModelOption,
    LLMProviderOption,
    SUPPORTED_SELECTOR_PROVIDER_IDS,
    default_model_for_provider,
    default_provider_catalog,
    default_provider_catalog_for_transport,
    provider_option_map,
    resolve_catalog_model,
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


def normalize_context_window_override(
    *,
    provider_id: str | None,
    model_id: str | None,
    context_window: int | None,
    transport: CatalogTransport = "direct",
) -> int | None:
    """Keep explicit context overrides only for unknown custom models."""

    if not isinstance(context_window, int) or context_window <= 0:
        return None

    provider = str(provider_id or "").strip()
    model = str(model_id or "").strip()
    resolved = resolve_catalog_model(provider, model, transport=transport) if provider and model else None
    if resolved is not None and resolved.context_window is not None:
        return None

    return int(context_window)


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
    transport: CatalogTransport = "direct",
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

    model_id = custom or stored_model or default_model_for_provider(provider, transport=transport) or ""
    return LLMOperationSettings(
        provider_id=provider,
        model_id=model_id,
        context_window=normalize_context_window_override(
            provider_id=provider,
            model_id=model_id,
            context_window=context_window if custom else None,
            transport=transport,
        ),
        use_reasoning=bool(use_reasoning),
    )


__all__ = [
    "LLMModelOption",
    "LLMOperationSettings",
    "LLMProviderOption",
    "SUPPORTED_SELECTOR_PROVIDER_IDS",
    "CatalogTransport",
    "default_provider_catalog",
    "default_provider_catalog_for_transport",
    "infer_provider_id_from_model",
    "normalize_context_window_override",
    "provider_option_map",
    "settings_from_report_preferences",
]
