"""Catalog and pricing helpers backed by ``genai-prices``."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import logging
import os
from threading import Lock
from typing import Optional, Sequence

from genai_prices import UpdatePrices
from genai_prices.data_snapshot import DataSnapshot, get_snapshot
from pydantic_ai.usage import RequestUsage

logger = logging.getLogger(__name__)

SUPPORTED_SELECTOR_PROVIDER_IDS: tuple[str, ...] = (
    "anthropic",
    "openai",
    "gemini",
)

_CATALOG_PROVIDER_IDS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google",
    "azure_openai": "azure",
}

_PROVIDER_LABELS: dict[str, str] = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "azure_openai": "Azure OpenAI",
}

_DEFAULT_MODEL_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
    "azure_openai": "AZURE_OPENAI_DEPLOYMENT_NAME",
}

_MODEL_FAMILY_MATCHERS: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude",),
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "gemini": ("gemini",),
    "azure_openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
}

_catalog_updater: UpdatePrices | None = None
_catalog_updater_lock = Lock()


@dataclass(frozen=True, slots=True)
class LLMModelOption:
    """Single preset model entry for a provider."""

    model_id: str
    label: str
    context_window: int | None = None
    input_price_label: str | None = None
    output_price_label: str | None = None


@dataclass(frozen=True, slots=True)
class LLMProviderOption:
    """Selector metadata for a supported LLM provider."""

    provider_id: str
    label: str
    models: tuple[LLMModelOption, ...]


def start_background_catalog_refresh(*, wait: bool | float = False) -> None:
    """Start a session-scoped background updater for ``genai-prices``."""

    global _catalog_updater

    with _catalog_updater_lock:
        if _catalog_updater is not None:
            return

        updater = UpdatePrices()
        try:
            updater.start(wait=wait)
        except Exception:
            logger.warning("Unable to start genai-prices background refresh", exc_info=True)
            try:
                updater.stop()
            except Exception:
                logger.debug("Failed stopping genai-prices updater after startup failure", exc_info=True)
            return

        _catalog_updater = updater


def stop_background_catalog_refresh() -> None:
    """Stop the session-scoped updater if it is running."""

    global _catalog_updater

    with _catalog_updater_lock:
        updater = _catalog_updater
        _catalog_updater = None

    if updater is None:
        return

    try:
        updater.stop()
    except Exception:
        logger.warning("Unable to stop genai-prices background refresh cleanly", exc_info=True)


def provider_option_map(
    options: Sequence[LLMProviderOption],
) -> dict[str, LLMProviderOption]:
    return {option.provider_id: option for option in options}


def default_provider_catalog(*, include_azure: bool = False) -> tuple[LLMProviderOption, ...]:
    """Return selector-ready provider/model metadata."""

    provider_ids = list(SUPPORTED_SELECTOR_PROVIDER_IDS)
    if include_azure:
        provider_ids.append("azure_openai")

    options: list[LLMProviderOption] = []
    for provider_id in provider_ids:
        models = tuple(_iter_selector_models(provider_id))
        if not models:
            continue
        options.append(
            LLMProviderOption(
                provider_id=provider_id,
                label=_PROVIDER_LABELS.get(provider_id, provider_id),
                models=models,
            )
        )
    return tuple(options)


def default_model_for_provider(provider_id: str) -> str | None:
    """Return the current default preset model for ``provider_id``."""

    env_var = _DEFAULT_MODEL_ENV_VARS.get(provider_id)
    env_model = os.getenv(env_var, "").strip() if env_var else ""
    if env_model:
        matched = resolve_catalog_model(provider_id, env_model)
        if matched and matched.context_window:
            return matched.model_id

    provider = _find_provider(provider_id)
    if provider is None:
        return None
    if provider.models:
        return provider.models[0].model_id
    return None


def resolve_catalog_model(provider_id: str, model_id: str | None) -> LLMModelOption | None:
    """Resolve ``model_id`` to catalog metadata when possible."""

    normalized = str(model_id or "").strip()
    if not normalized:
        return None

    snapshot = _snapshot()
    upstream_provider_id = _CATALOG_PROVIDER_IDS.get(provider_id)
    if not upstream_provider_id:
        return None

    try:
        _provider, model = snapshot.find_provider_model(
            normalized,
            None,
            upstream_provider_id,
            None,
        )
    except LookupError:
        return None

    return LLMModelOption(
        model_id=model.id,
        label=model.name or model.id,
        context_window=model.context_window if _valid_context_window(model.context_window) else None,
        input_price_label=_price_label(getattr(model.prices, "input_mtok", None)),
        output_price_label=_price_label(getattr(model.prices, "output_mtok", None)),
    )


def resolve_model_context_window(provider_id: str, model_id: str | None) -> int | None:
    """Return the raw context window for ``model_id`` when the catalog knows it."""

    model = resolve_catalog_model(provider_id, model_id)
    if model is None:
        return None
    return model.context_window


def calculate_usage_cost(
    *,
    provider_id: str,
    model_id: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Decimal | None:
    """Calculate total cost for a completed request when pricing is known."""

    upstream_provider_id = _CATALOG_PROVIDER_IDS.get(provider_id)
    normalized_model = str(model_id or "").strip()
    if not upstream_provider_id or not normalized_model:
        return None

    usage = RequestUsage(
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
    )
    try:
        calculation = _snapshot().calc(
            usage,
            normalized_model,
            provider_id=upstream_provider_id,
            provider_api_url=None,
            genai_request_timestamp=None,
        )
    except LookupError:
        return None
    except Exception:
        logger.debug(
            "Unable to calculate model cost for provider=%s model=%s",
            provider_id,
            normalized_model,
            exc_info=True,
        )
        return None

    return calculation.total_price


def _snapshot() -> DataSnapshot:
    return get_snapshot()


def _find_provider(provider_id: str) -> LLMProviderOption | None:
    for provider in default_provider_catalog(include_azure=True):
        if provider.provider_id == provider_id:
            return provider
    return None


def _iter_selector_models(provider_id: str) -> Sequence[LLMModelOption]:
    snapshot = _snapshot()
    upstream_provider_id = _CATALOG_PROVIDER_IDS.get(provider_id)
    if not upstream_provider_id:
        return ()

    provider = next((item for item in snapshot.providers if item.id == upstream_provider_id), None)
    if provider is None:
        return ()

    models: list[LLMModelOption] = []
    for model in provider.models:
        if getattr(model, "deprecated", False):
            continue
        if not _is_supported_model_family(provider_id, model.id):
            continue
        if not _valid_context_window(model.context_window):
            continue
        models.append(
            LLMModelOption(
                model_id=model.id,
                label=model.name or model.id,
                context_window=int(model.context_window),
                input_price_label=_price_label(getattr(model.prices, "input_mtok", None)),
                output_price_label=_price_label(getattr(model.prices, "output_mtok", None)),
            )
        )
    return models


def _is_supported_model_family(provider_id: str, model_id: str) -> bool:
    prefixes = _MODEL_FAMILY_MATCHERS.get(provider_id, ())
    normalized = (model_id or "").lower()
    if not prefixes:
        return True
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _valid_context_window(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _price_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return _format_decimal(value)

    base = getattr(value, "base", None)
    tiers = getattr(value, "tiers", None)
    if isinstance(base, Decimal):
        if isinstance(tiers, list) and tiers:
            last_tier = tiers[-1]
            last_price = getattr(last_tier, "price", None)
            if isinstance(last_price, Decimal):
                return f"{_format_decimal(base)} -> {_format_decimal(last_price)}"
        return _format_decimal(base)

    return None


def _format_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return f"${normalized}/1M"


__all__ = [
    "LLMModelOption",
    "LLMProviderOption",
    "SUPPORTED_SELECTOR_PROVIDER_IDS",
    "calculate_usage_cost",
    "default_model_for_provider",
    "default_provider_catalog",
    "provider_option_map",
    "resolve_catalog_model",
    "resolve_model_context_window",
    "start_background_catalog_refresh",
    "stop_background_catalog_refresh",
]
