"""Catalog and pricing helpers using provider-native data first."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
import json
import logging
import os
import re
import time
from threading import Lock
from typing import Any, Literal, Optional, Sequence

from genai_prices import UpdatePrices
from genai_prices.data_snapshot import DataSnapshot, get_snapshot
import httpx
from pydantic_ai.usage import RequestUsage

from src.config.paths import app_config_dir

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
    "anthropic_bedrock": ("anthropic.claude", "claude"),
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "gemini": ("gemini",),
    "azure_openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
}

# Anthropic's published 1M context window requires an explicit beta header.
# The current worker/gateway request path does not send that header, so runtime
# budgeting must cap Anthropic models to the standard 200k request window.
_RUNTIME_CONTEXT_WINDOW_CAPS: dict[str, int] = {
    "anthropic": 200_000,
}

_catalog_updater: UpdatePrices | None = None
_catalog_updater_lock = Lock()
_provider_selector_models: dict[tuple[str, str, str], tuple["LLMModelOption", ...]] = {}
_provider_selector_models_lock = Lock()
_gateway_provider_catalogs: dict[str, tuple["_GatewayCatalogProvider", ...]] = {}
_gateway_provider_catalogs_lock = Lock()
_PROVIDER_CACHE_TTL_SECONDS = 6 * 60 * 60
_PROVIDER_CACHE_MAX_STALE_SECONDS = 7 * 24 * 60 * 60
_BEDROCK_ANTHROPIC_ALIASES: dict[str, str] = {
    "anthropic.claude-sonnet-4-5-v1": "claude-sonnet-4-5",
    "anthropic.claude-sonnet-4-5-v1:0": "claude-sonnet-4-5",
    "anthropic.claude-opus-4-6-v1": "claude-opus-4-6",
    "anthropic.claude-opus-4-1-20250805-v1:0": "claude-opus-4-1-20250805",
}

CatalogTransport = Literal["direct", "gateway"]


@dataclass(frozen=True, slots=True)
class _DiscoveredModel:
    model_id: str
    label: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    lifecycle_status: str = "stable"


@dataclass(frozen=True, slots=True)
class LLMReasoningCapabilities:
    """Normalized reasoning capabilities for a model."""

    supports_reasoning_controls: bool = False
    can_disable_reasoning: bool = False
    controls: tuple[str, ...] = ()
    allowed_efforts: tuple[str, ...] = ()
    allowed_levels: tuple[str, ...] = ()
    budget_min: int | None = None
    budget_max: int | None = None
    budget_step: int | None = None
    default_state: str = "off"
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class LLMModelProvenance:
    availability: str = "unknown"
    context: str | None = None
    pricing: str | None = None
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class LLMModelOption:
    """Single preset model entry for a provider."""

    model_id: str
    label: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_price_label: str | None = None
    output_price_label: str | None = None
    lifecycle_status: str = "stable"
    reasoning_capabilities: LLMReasoningCapabilities = LLMReasoningCapabilities()
    provenance: LLMModelProvenance = LLMModelProvenance()


@dataclass(frozen=True, slots=True)
class LLMProviderOption:
    """Selector metadata for a supported LLM provider."""

    provider_id: str
    label: str
    models: tuple[LLMModelOption, ...]


@dataclass(frozen=True, slots=True)
class _GatewayCatalogProvider:
    provider_id: str
    label: str
    route: str | None
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
    
    return default_provider_catalog_for_transport(include_azure=include_azure, transport="direct")


def default_provider_catalog_for_transport(
    *,
    include_azure: bool = False,
    transport: CatalogTransport = "direct",
) -> tuple[LLMProviderOption, ...]:
    """Return selector-ready provider/model metadata for a transport."""

    if transport == "gateway":
        providers = [
            LLMProviderOption(
                provider_id=provider.provider_id,
                label=provider.label,
                models=provider.models,
            )
            for provider in _gateway_provider_catalog()
            if include_azure or provider.provider_id != "azure_openai"
        ]
        return tuple(providers)

    provider_ids = list(SUPPORTED_SELECTOR_PROVIDER_IDS)
    if include_azure:
        provider_ids.append("azure_openai")

    options: list[LLMProviderOption] = []
    for provider_id in provider_ids:
        models = tuple(_iter_selector_models(provider_id, transport=transport))
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


def default_model_for_provider(
    provider_id: str,
    *,
    transport: CatalogTransport = "direct",
) -> str | None:
    """Return the current default preset model for ``provider_id``."""

    env_var = _DEFAULT_MODEL_ENV_VARS.get(provider_id)
    env_model = os.getenv(env_var, "").strip() if env_var else ""
    if env_model:
        matched = resolve_catalog_model(provider_id, env_model, transport=transport)
        if matched and matched.context_window:
            return matched.model_id

    provider = _find_provider(provider_id, transport=transport)
    if provider is None:
        return None
    for model in provider.models:
        if model.context_window:
            return model.model_id
    if provider.models:
        return provider.models[0].model_id
    return None


def runtime_default_model_for_provider(
    provider_id: str,
    *,
    transport: CatalogTransport = "direct",
) -> str | None:
    """Return the best current runtime model, even if selector metadata is incomplete."""

    env_var = _DEFAULT_MODEL_ENV_VARS.get(provider_id)
    env_model = os.getenv(env_var, "").strip() if env_var else ""
    if env_model:
        return env_model

    provider = _find_provider(provider_id, transport=transport)
    if provider is not None:
        for model in provider.models:
            if model.context_window:
                return model.model_id
        if provider.models:
            return provider.models[0].model_id

    if transport == "gateway":
        return None

    snapshot = _snapshot()
    upstream_provider_id = _CATALOG_PROVIDER_IDS.get(provider_id)
    if not upstream_provider_id:
        return None

    provider = next((item for item in snapshot.providers if item.id == upstream_provider_id), None)
    if provider is None:
        return None

    candidates = [
        model
        for model in provider.models
        if not getattr(model, "deprecated", False)
        and _is_supported_model_family(provider_id, model.id)
    ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda model: _model_sort_key(
            provider_id,
            model_id=model.id,
            label=model.name or model.id,
            context_window=model.context_window if _valid_context_window(model.context_window) else None,
        ),
        reverse=True,
    )
    return candidates[0].id


def resolve_catalog_model(
    provider_id: str,
    model_id: str | None,
    *,
    transport: CatalogTransport = "direct",
) -> LLMModelOption | None:
    """Resolve ``model_id`` to catalog metadata when possible."""

    normalized = str(model_id or "").strip()
    if not normalized:
        return None

    for model in _iter_selector_models(provider_id, transport=transport):
        if model.model_id == normalized:
            return model

    if transport == "gateway":
        return None

    return _resolve_snapshot_catalog_model(provider_id, normalized)


def resolve_model_context_window(
    provider_id: str,
    model_id: str | None,
    *,
    transport: CatalogTransport = "direct",
) -> int | None:
    """Return the raw context window for ``model_id`` when the catalog knows it."""

    model = resolve_catalog_model(provider_id, model_id, transport=transport)
    if model is None:
        return None
    return model.context_window


def _runtime_context_window(
    provider_id: str,
    model_id: str,
    context_window: int | None,
) -> int | None:
    if not _valid_context_window(context_window):
        return None
    cap = _RUNTIME_CONTEXT_WINDOW_CAPS.get(provider_id)
    if cap is None:
        return int(context_window)
    return min(int(context_window), cap)


def _fingerprint(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return "default"
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _cache_scope(provider_id: str, *, transport: CatalogTransport) -> str:
    if transport == "gateway":
        gateway_settings = _read_gateway_settings()
        base_url = str(gateway_settings.get("base_url", "") or "").strip()
        route = str(gateway_settings.get("route", "") or "").strip()
        api_key = _resolve_api_key("pydantic_ai_gateway", env_vars=("PYDANTIC_AI_GATEWAY_API_KEY",))
        return _fingerprint(f"{provider_id}|{base_url}|{route}|{api_key}")

    if provider_id == "gemini":
        key = _resolve_gemini_api_key()
    elif provider_id == "azure_openai":
        azure = _read_azure_settings()
        key = f"{azure.get('endpoint','')}|{azure.get('deployment_name','')}|{_resolve_api_key('azure_openai', env_vars=('AZURE_OPENAI_API_KEY',))}"
    else:
        env_var_map = {
            "openai": ("OPENAI_API_KEY",),
            "anthropic": ("ANTHROPIC_API_KEY",),
            "anthropic_bedrock": (),
        }
        key = _resolve_api_key(provider_id, env_vars=env_var_map.get(provider_id, ()))
    return _fingerprint(f"{provider_id}|{key}")


def _provider_cache_path(provider_id: str, *, transport: CatalogTransport) -> Any:
    cache_dir = app_config_dir() / "model_catalog_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scope = _cache_scope(provider_id, transport=transport)
    return cache_dir / f"{transport}_{provider_id}_{scope}.json"


def _gateway_catalog_scope() -> str:
    gateway_settings = _read_gateway_settings()
    base_url = str(gateway_settings.get("base_url", "") or "").strip()
    route = str(gateway_settings.get("route", "") or "").strip()
    api_key = _resolve_api_key("pydantic_ai_gateway", env_vars=("PYDANTIC_AI_GATEWAY_API_KEY",))
    return _fingerprint(f"{base_url}|{route}|{api_key}")


def _gateway_catalog_cache_path() -> Any:
    cache_dir = app_config_dir() / "model_catalog_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gateway_catalog_{_gateway_catalog_scope()}.json"


def _load_cached_discovered_models(
    provider_id: str,
    *,
    transport: CatalogTransport,
) -> tuple[_DiscoveredModel, ...]:
    cache_path = _provider_cache_path(provider_id, transport=transport)
    if not cache_path.exists():
        return ()
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Unable to read cached model metadata for %s/%s", transport, provider_id, exc_info=True)
        return ()

    cached_at = float(payload.get("cached_at") or 0)
    age_seconds = max(time.time() - cached_at, 0.0)
    if age_seconds > _PROVIDER_CACHE_MAX_STALE_SECONDS:
        return ()

    items = payload.get("models")
    if not isinstance(items, list):
        return ()

    models: list[_DiscoveredModel] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", "") or "").strip()
        if not model_id:
            continue
        context_window = item.get("context_window")
        max_output_tokens = item.get("max_output_tokens")
        models.append(
            _DiscoveredModel(
                model_id=model_id,
                label=str(item.get("label", "") or model_id).strip() or model_id,
                context_window=int(context_window) if _valid_context_window(context_window) else None,
                max_output_tokens=int(max_output_tokens) if _valid_context_window(max_output_tokens) else None,
                lifecycle_status=str(item.get("lifecycle_status", "stable") or "stable"),
            )
        )
    return tuple(models)


def _write_cached_discovered_models(
    provider_id: str,
    *,
    transport: CatalogTransport,
    discovered: Sequence[_DiscoveredModel],
) -> None:
    if not discovered:
        return
    payload = {
        "cached_at": time.time(),
        "models": [asdict(model) for model in discovered],
    }
    try:
        _provider_cache_path(provider_id, transport=transport).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Unable to write cached model metadata for %s/%s", transport, provider_id, exc_info=True)


def _load_cached_gateway_provider_catalog() -> tuple[_GatewayCatalogProvider, ...]:
    cache_path = _gateway_catalog_cache_path()
    if not cache_path.exists():
        return ()
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Unable to read cached gateway provider catalog", exc_info=True)
        return ()

    cached_at = float(payload.get("cached_at") or 0)
    age_seconds = max(time.time() - cached_at, 0.0)
    if age_seconds > _PROVIDER_CACHE_MAX_STALE_SECONDS:
        return ()

    providers_payload = payload.get("providers")
    if not isinstance(providers_payload, list):
        return ()
    providers: list[_GatewayCatalogProvider] = []
    for item in providers_payload:
        if not isinstance(item, dict):
            continue
        provider = _gateway_provider_option_from_payload(item)
        if provider is None:
            continue
        providers.append(
            _GatewayCatalogProvider(
                provider_id=provider.provider_id,
                label=provider.label,
                route=str(item.get("route") or "").strip() or None,
                models=provider.models,
            )
        )
    return tuple(providers)


def _write_cached_gateway_provider_catalog(
    providers: Sequence[_GatewayCatalogProvider],
) -> None:
    if not providers:
        return

    payload = {
        "cached_at": time.time(),
        "providers": [
            {
                "provider_id": provider.provider_id,
                "label": provider.label,
                "route": provider.route,
                "models": [asdict(model) for model in provider.models],
            }
            for provider in providers
        ],
    }
    try:
        _gateway_catalog_cache_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Unable to write cached gateway provider catalog", exc_info=True)


def _read_settings_payload() -> dict[str, Any]:
    settings_path = app_config_dir() / "app_settings.json"
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        logger.debug("Unable to read app settings payload", exc_info=True)
        return {}


def _read_gateway_settings() -> dict[str, Any]:
    payload = _read_settings_payload()
    gateway_settings = payload.get("pydantic_ai_gateway_settings")
    return gateway_settings if isinstance(gateway_settings, dict) else {}


def _read_azure_settings() -> dict[str, Any]:
    payload = _read_settings_payload()
    azure_settings = payload.get("azure_openai_settings")
    return azure_settings if isinstance(azure_settings, dict) else {}


def _read_bedrock_settings() -> dict[str, Any]:
    payload = _read_settings_payload()
    bedrock_settings = payload.get("aws_bedrock_settings")
    return bedrock_settings if isinstance(bedrock_settings, dict) else {}


def _reasoning_capabilities_for_model(
    provider_id: str,
    model_id: str,
) -> LLMReasoningCapabilities:
    normalized = model_id.lower()
    if provider_id in {"openai", "azure_openai"}:
        try:
            from pydantic_ai.profiles.openai import openai_model_profile

            profile = openai_model_profile(model_id)
            if not profile.openai_supports_reasoning:
                return LLMReasoningCapabilities()
            return LLMReasoningCapabilities(
                supports_reasoning_controls=True,
                can_disable_reasoning=bool(profile.openai_supports_reasoning_effort_none),
                controls=("toggle", "effort"),
                allowed_efforts=("low", "medium", "high"),
                default_state="on",
                notes=(
                    "This model always reasons; Off is unavailable."
                    if not profile.openai_supports_reasoning_effort_none
                    else None
                ),
            )
        except Exception:
            logger.debug("Unable to derive OpenAI reasoning capabilities for %s", model_id, exc_info=True)
            return LLMReasoningCapabilities()

    if provider_id in {"anthropic", "anthropic_bedrock"}:
        return LLMReasoningCapabilities(
            supports_reasoning_controls=True,
            can_disable_reasoning=True,
            controls=("toggle", "budget"),
            budget_step=1024,
            default_state="off",
        )

    if provider_id == "gemini":
        if normalized.startswith("gemini-3"):
            return LLMReasoningCapabilities(
                supports_reasoning_controls=True,
                can_disable_reasoning=False,
                controls=("level",),
                allowed_levels=("MINIMAL", "LOW", "MEDIUM", "HIGH"),
                default_state="on",
                notes="Gemini 3 model families use thinking levels and may not fully disable thinking.",
            )
        return LLMReasoningCapabilities(
            supports_reasoning_controls=True,
            can_disable_reasoning=True,
            controls=("toggle", "budget"),
            default_state="off",
        )

    return LLMReasoningCapabilities()


def resolve_reasoning_capabilities(
    provider_id: str,
    model_id: str | None,
    *,
    transport: CatalogTransport = "direct",
) -> LLMReasoningCapabilities:
    normalized = str(model_id or "").strip()
    if not normalized:
        normalized = default_model_for_provider(provider_id, transport=transport) or ""
    if not normalized:
        return LLMReasoningCapabilities()

    model = resolve_catalog_model(provider_id, normalized, transport=transport)
    if model is not None and model.reasoning_capabilities.supports_reasoning_controls:
        return model.reasoning_capabilities
    return _reasoning_capabilities_for_model(provider_id, normalized)


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


def _find_provider(
    provider_id: str,
    *,
    transport: CatalogTransport = "direct",
) -> LLMProviderOption | None:
    for provider in default_provider_catalog_for_transport(include_azure=True, transport=transport):
        if provider.provider_id == provider_id:
            return provider
    return None


def _iter_selector_models(
    provider_id: str,
    *,
    transport: CatalogTransport = "direct",
) -> Sequence[LLMModelOption]:
    scope = _cache_scope(provider_id, transport=transport)
    cache_key = (transport, provider_id, scope)
    with _provider_selector_models_lock:
        cached = _provider_selector_models.get(cache_key)
    if cached is not None:
        return cached

    if transport == "gateway":
        provider = next(
            (item for item in _gateway_provider_catalog() if item.provider_id == provider_id),
            None,
        )
        models = provider.models if provider is not None else ()
    elif provider_id == "gemini":
        models = _gemini_runtime_models()
    elif provider_id == "openai":
        models = _openai_runtime_models(transport=transport)
    elif provider_id == "anthropic":
        models = _anthropic_runtime_models(transport=transport)
    elif provider_id == "azure_openai":
        models = _azure_runtime_models()
    elif provider_id == "anthropic_bedrock":
        models = _bedrock_runtime_models()
    else:
        models = tuple(_iter_snapshot_selector_models(provider_id))

    with _provider_selector_models_lock:
        _provider_selector_models[cache_key] = tuple(models)
        return _provider_selector_models[cache_key]


def reset_provider_catalog_cache() -> None:
    """Clear the in-memory provider model cache."""

    with _provider_selector_models_lock:
        _provider_selector_models.clear()
    with _gateway_provider_catalogs_lock:
        _gateway_provider_catalogs.clear()


def reset_gemini_model_cache() -> None:
    """Backward-compatible alias for clearing provider model caches."""

    reset_provider_catalog_cache()


def _gateway_provider_catalog() -> tuple[_GatewayCatalogProvider, ...]:
    scope = _gateway_catalog_scope()
    with _gateway_provider_catalogs_lock:
        cached = _gateway_provider_catalogs.get(scope)
    if cached is not None:
        return cached

    discovered = _discover_gateway_provider_catalog_live()
    if discovered:
        _write_cached_gateway_provider_catalog(discovered)
        providers = discovered
    else:
        providers = _load_cached_gateway_provider_catalog()

    with _gateway_provider_catalogs_lock:
        _gateway_provider_catalogs[scope] = tuple(providers)

    return tuple(providers)


def _discover_gateway_provider_catalog_live() -> tuple[_GatewayCatalogProvider, ...]:
    gateway_settings = _read_gateway_settings()
    base_url = str(gateway_settings.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return ()
    api_key = _resolve_api_key("pydantic_ai_gateway", env_vars=("PYDANTIC_AI_GATEWAY_API_KEY",))
    if not api_key:
        return ()

    route = str(gateway_settings.get("route") or "").strip()
    params: dict[str, str] = {}
    if route:
        params["route"] = route

    try:
        response = httpx.get(
            f"{base_url}/metadata/catalog",
            params=params or None,
            headers={"Authorization": api_key},
            timeout=15.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.debug("Unable to discover gateway provider catalog", exc_info=True)
        return ()

    providers_payload = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers_payload, list):
        return ()

    providers: list[_GatewayCatalogProvider] = []
    for item in providers_payload:
        if not isinstance(item, dict):
            continue
        provider = _gateway_provider_option_from_payload(item)
        if provider is None:
            continue
        providers.append(
            _GatewayCatalogProvider(
                provider_id=provider.provider_id,
                label=provider.label,
                route=str(item.get("route") or "").strip() or None,
                models=provider.models,
            )
        )
    return tuple(providers)


def _gateway_provider_option_from_payload(payload: dict[str, Any]) -> LLMProviderOption | None:
    provider_id = _normalize_gateway_provider_id(
        payload.get("provider_id"),
        payload.get("upstream_provider_id"),
    )
    if not provider_id:
        return None

    models_payload = payload.get("models")
    if not isinstance(models_payload, list):
        return None

    models = [
        model
        for item in models_payload
        if isinstance(item, dict)
        for model in (_gateway_model_option_from_payload(provider_id, item),)
        if model is not None
    ]
    models.sort(
        key=lambda model: _model_sort_key(
            provider_id,
            model_id=model.model_id,
            label=model.label,
            context_window=model.context_window,
        ),
        reverse=True,
    )
    return LLMProviderOption(
        provider_id=provider_id,
        label=str(payload.get("label") or "").strip() or _PROVIDER_LABELS.get(provider_id, provider_id),
        models=tuple(models),
    )


def _normalize_gateway_provider_id(provider_id: Any, upstream_provider_id: Any) -> str:
    normalized = str(provider_id or "").strip()
    upstream = str(upstream_provider_id or "").strip()
    if normalized in {"gemini", "google-vertex"} or upstream == "google-vertex":
        return "gemini"
    if normalized in {"anthropic_bedrock", "bedrock"} or upstream == "bedrock":
        return "anthropic_bedrock"
    return normalized


def _gateway_model_option_from_payload(provider_id: str, payload: dict[str, Any]) -> LLMModelOption | None:
    model_id = str(payload.get("model_id", "") or "").strip()
    if not model_id:
        return None

    reasoning = _merge_gateway_reasoning_capabilities(
        _reasoning_capabilities_for_model(provider_id, model_id),
        payload.get("reasoning_capabilities"),
    )
    sources = payload.get("sources")
    sources_map = sources if isinstance(sources, dict) else {}

    return LLMModelOption(
        model_id=model_id,
        label=str(payload.get("display_name", "") or model_id).strip() or model_id,
        context_window=_runtime_context_window(provider_id, model_id, _coerce_positive_int(payload.get("context_window"))),
        max_output_tokens=_coerce_positive_int(payload.get("max_output_tokens")),
        input_price_label=_price_label(payload.get("pricing_input_per_million")),
        output_price_label=_price_label(payload.get("pricing_output_per_million")),
        lifecycle_status=str(payload.get("lifecycle_status", "stable") or "stable"),
        reasoning_capabilities=reasoning,
        provenance=LLMModelProvenance(
            availability=str(sources_map.get("availability") or "unknown"),
            context=str(sources_map.get("context_window") or "").strip() or None,
            pricing=str(sources_map.get("pricing") or "").strip() or None,
            reasoning=str(sources_map.get("reasoning_capabilities") or "").strip() or None,
        ),
    )


def _merge_gateway_reasoning_capabilities(
    fallback: LLMReasoningCapabilities,
    payload: Any,
) -> LLMReasoningCapabilities:
    if not isinstance(payload, dict):
        return fallback

    supports = payload.get("supports_reasoning_controls")
    can_disable = payload.get("can_disable_reasoning")
    controls = payload.get("controls")
    return LLMReasoningCapabilities(
        supports_reasoning_controls=bool(supports) if isinstance(supports, bool) else fallback.supports_reasoning_controls,
        can_disable_reasoning=bool(can_disable) if isinstance(can_disable, bool) else fallback.can_disable_reasoning,
        controls=tuple(
            str(control).strip()
            for control in (controls or ())
            if str(control).strip()
        )
        or fallback.controls,
        allowed_efforts=fallback.allowed_efforts,
        allowed_levels=fallback.allowed_levels,
        budget_min=fallback.budget_min,
        budget_max=fallback.budget_max,
        budget_step=fallback.budget_step,
        default_state=fallback.default_state,
        notes=str(payload.get("notes") or "").strip() or fallback.notes,
    )


def _iter_snapshot_selector_models(
    provider_id: str,
    *,
    exclude_preview: bool = False,
) -> tuple[LLMModelOption, ...]:
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
        if provider_id == "gemini" and _is_excluded_gemini_model(model.id):
            continue
        if exclude_preview and _is_preview_model_id(model.id):
            continue
        models.append(
            LLMModelOption(
                model_id=model.id,
                label=model.name or model.id,
                context_window=_runtime_context_window(
                    provider_id,
                    model.id,
                    int(model.context_window) if _valid_context_window(model.context_window) else None,
                ),
                max_output_tokens=None,
                input_price_label=_price_label(getattr(model.prices, "input_mtok", None)),
                output_price_label=_price_label(getattr(model.prices, "output_mtok", None)),
                lifecycle_status="preview" if _is_preview_model_id(model.id) else "stable",
                reasoning_capabilities=_reasoning_capabilities_for_model(provider_id, model.id),
                provenance=LLMModelProvenance(
                    availability="genai-prices",
                    context="genai-prices" if _valid_context_window(model.context_window) else None,
                    pricing="genai-prices",
                    reasoning="provider-rules",
                ),
            )
        )
    models.sort(
        key=lambda model: _model_sort_key(
            provider_id,
            model_id=model.model_id,
            label=model.label,
            context_window=model.context_window,
        ),
        reverse=True,
    )
    return tuple(models)


def _build_discovered_model_options(
    provider_id: str,
    discovered: Sequence[_DiscoveredModel],
) -> tuple[LLMModelOption, ...]:
    models: list[LLMModelOption] = []
    for item in discovered:
        catalog_model = _resolve_snapshot_catalog_model(provider_id, item.model_id)
        context_window = item.context_window
        if context_window is None and catalog_model is not None:
            context_window = catalog_model.context_window
        models.append(
            LLMModelOption(
                model_id=item.model_id,
                label=item.label,
                context_window=_runtime_context_window(provider_id, item.model_id, context_window),
                max_output_tokens=item.max_output_tokens,
                input_price_label=catalog_model.input_price_label if catalog_model else None,
                output_price_label=catalog_model.output_price_label if catalog_model else None,
                lifecycle_status=item.lifecycle_status,
                reasoning_capabilities=_reasoning_capabilities_for_model(provider_id, item.model_id),
                provenance=LLMModelProvenance(
                    availability="provider-native",
                    context="provider-native" if item.context_window is not None else (
                        catalog_model.provenance.context if catalog_model else None
                    ),
                    pricing=catalog_model.provenance.pricing if catalog_model else None,
                    reasoning="provider-rules",
                ),
            )
        )
    models.sort(
        key=lambda model: _model_sort_key(
            provider_id,
            model_id=model.model_id,
            label=model.label,
            context_window=model.context_window,
        ),
        reverse=True,
    )
    return tuple(models)


def _openai_runtime_models(*, transport: CatalogTransport) -> tuple[LLMModelOption, ...]:
    discovered = _discover_models_with_cache("openai", transport=transport, fetcher=_discover_openai_models_live)
    if discovered:
        return _build_discovered_model_options("openai", discovered)
    return _iter_snapshot_selector_models("openai")


def _anthropic_runtime_models(*, transport: CatalogTransport) -> tuple[LLMModelOption, ...]:
    discovered = _discover_models_with_cache("anthropic", transport=transport, fetcher=_discover_anthropic_models_live)
    if discovered:
        return _build_discovered_model_options("anthropic", discovered)
    return _iter_snapshot_selector_models("anthropic")


def _azure_runtime_models() -> tuple[LLMModelOption, ...]:
    azure_settings = _read_azure_settings()
    deployment = str(
        azure_settings.get("deployment")
        or azure_settings.get("deployment_name")
        or ""
    ).strip()
    if deployment:
        return _build_discovered_model_options(
            "azure_openai",
            (
                _DiscoveredModel(
                    model_id=deployment,
                    label=deployment,
                    lifecycle_status="stable",
                ),
            ),
        )
    return _iter_snapshot_selector_models("azure_openai")


def _bedrock_runtime_models() -> tuple[LLMModelOption, ...]:
    discovered = _discover_models_with_cache(
        "anthropic_bedrock",
        transport="direct",
        fetcher=_discover_bedrock_models_live,
    )
    if discovered:
        return _build_discovered_model_options("anthropic_bedrock", discovered)
    return ()


def _gateway_runtime_models(provider_id: str) -> tuple[LLMModelOption, ...]:
    discovered = _discover_models_with_cache(
        provider_id,
        transport="gateway",
        fetcher=lambda: _discover_gateway_models(provider_id),
    )
    if discovered:
        return _build_discovered_model_options(provider_id, discovered)
    if provider_id == "gemini":
        return _iter_snapshot_selector_models("gemini", exclude_preview=True)
    return _iter_snapshot_selector_models(provider_id)


def _gemini_runtime_models() -> tuple[LLMModelOption, ...]:
    discovered = _discover_models_with_cache("gemini", transport="direct", fetcher=_discover_gemini_models_live)
    if discovered:
        return _build_gemini_model_options(discovered)
    return _iter_snapshot_selector_models("gemini")


def _build_gemini_model_options(
    discovered: Sequence[_DiscoveredModel],
) -> tuple[LLMModelOption, ...]:
    models: list[LLMModelOption] = []
    for discovered_model in discovered:
        catalog_model = _resolve_snapshot_catalog_model("gemini", discovered_model.model_id)
        models.append(
            LLMModelOption(
                model_id=discovered_model.model_id,
                label=discovered_model.label,
                context_window=discovered_model.context_window,
                max_output_tokens=discovered_model.max_output_tokens,
                input_price_label=catalog_model.input_price_label if catalog_model else None,
                output_price_label=catalog_model.output_price_label if catalog_model else None,
                lifecycle_status=discovered_model.lifecycle_status,
                reasoning_capabilities=_reasoning_capabilities_for_model("gemini", discovered_model.model_id),
                provenance=LLMModelProvenance(
                    availability="provider-native",
                    context="provider-native",
                    pricing=catalog_model.provenance.pricing if catalog_model else None,
                    reasoning="provider-rules",
                ),
            )
        )
    models.sort(
        key=lambda model: _model_sort_key(
            "gemini",
            model_id=model.model_id,
            label=model.label,
            context_window=model.context_window,
        ),
        reverse=True,
    )
    return tuple(models)


def _discover_models_with_cache(
    provider_id: str,
    *,
    transport: CatalogTransport,
    fetcher: Any,
) -> tuple[_DiscoveredModel, ...]:
    try:
        discovered = tuple(fetcher() or ())
    except Exception:
        logger.debug("Unable to discover models for %s/%s", transport, provider_id, exc_info=True)
        discovered = ()

    if discovered:
        _write_cached_discovered_models(provider_id, transport=transport, discovered=discovered)
        return discovered

    return _load_cached_discovered_models(provider_id, transport=transport)


def _discover_gemini_models_live() -> tuple[_DiscoveredModel, ...]:
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return ()

    try:
        from google import genai
    except Exception:
        logger.debug("google-genai is unavailable; skipping Gemini model discovery", exc_info=True)
        return ()

    try:
        client = genai.Client(api_key=api_key)
        models: list[_DiscoveredModel] = []
        for model in client.models.list(config={"page_size": 100}):
            discovered = _parse_gemini_discovered_model(model)
            if discovered is not None:
                models.append(discovered)
        if not models:
            return ()
        models.sort(
            key=lambda model: _model_sort_key(
                "gemini",
                model_id=model.model_id,
                label=model.label,
                context_window=model.context_window,
            ),
            reverse=True,
        )
        return tuple(models)
    except Exception:
        logger.debug("Unable to discover Gemini models live", exc_info=True)
        return ()


def _discover_openai_models_live() -> tuple[_DiscoveredModel, ...]:
    api_key = _resolve_api_key("openai", env_vars=("OPENAI_API_KEY",))
    if not api_key:
        return ()

    try:
        from openai import OpenAI
    except Exception:
        logger.debug("openai package is unavailable; skipping OpenAI model discovery", exc_info=True)
        return ()

    try:
        client = OpenAI(api_key=api_key)
        models: list[_DiscoveredModel] = []
        for model in client.models.list():
            model_id = str(getattr(model, "id", "") or "").strip()
            if not model_id or not _is_supported_model_family("openai", model_id):
                continue
            models.append(_DiscoveredModel(model_id=model_id, label=model_id))
        return tuple(models)
    except Exception:
        logger.debug("Unable to discover OpenAI models live", exc_info=True)
        return ()


def _discover_anthropic_models_live() -> tuple[_DiscoveredModel, ...]:
    api_key = _resolve_api_key("anthropic", env_vars=("ANTHROPIC_API_KEY",))
    if not api_key:
        return ()

    try:
        import anthropic
    except Exception:
        logger.debug("anthropic package is unavailable; skipping Anthropic model discovery", exc_info=True)
        return ()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        models: list[_DiscoveredModel] = []
        for model in client.models.list():
            model_id = str(getattr(model, "id", "") or "").strip()
            if not model_id or not _is_supported_model_family("anthropic", model_id):
                continue
            label = str(getattr(model, "display_name", "") or model_id).strip() or model_id
            models.append(_DiscoveredModel(model_id=model_id, label=label))
        return tuple(models)
    except Exception:
        logger.debug("Unable to discover Anthropic models live", exc_info=True)
        return ()


def _discover_bedrock_models_live() -> tuple[_DiscoveredModel, ...]:
    try:
        import boto3
    except Exception:
        logger.debug("boto3 is unavailable; skipping Bedrock model discovery", exc_info=True)
        return ()

    settings = _read_bedrock_settings()
    profile = str(settings.get("profile") or "").strip() or None
    region = (
        str(settings.get("region") or "").strip()
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or None
    )
    try:
        session = boto3.session.Session(profile_name=profile, region_name=region)
        client = session.client("bedrock", region_name=region)
        response = client.list_foundation_models(byProvider="Anthropic")
    except Exception:
        logger.debug("Unable to discover Bedrock Anthropic models live", exc_info=True)
        return ()

    models: list[_DiscoveredModel] = []
    for item in response.get("modelSummaries", []) or ():
        raw_model_id = str(item.get("modelId") or "").strip()
        if not raw_model_id:
            continue
        model_id = _BEDROCK_ANTHROPIC_ALIASES.get(raw_model_id, raw_model_id)
        if not _is_supported_model_family("anthropic_bedrock", model_id):
            continue
        label = str(item.get("modelName") or model_id).strip() or model_id
        models.append(
            _DiscoveredModel(
                model_id=model_id,
                label=label,
                lifecycle_status="preview" if _is_preview_model_id(model_id) else "stable",
            )
        )
    return tuple(models)


def _parse_gemini_discovered_model(model: object) -> _DiscoveredModel | None:
    model_name = str(getattr(model, "name", "") or "")
    if not model_name.startswith("models/gemini"):
        return None
    if not _supports_gemini_generate_content(model):
        return None

    normalized_model_id = model_name.removeprefix("models/").strip()
    if not normalized_model_id or _is_excluded_gemini_model(normalized_model_id):
        return None

    context_window = getattr(model, "input_token_limit", None)
    if not _valid_context_window(context_window):
        return None

    label = str(getattr(model, "display_name", "") or normalized_model_id).strip() or normalized_model_id
    output_token_limit = getattr(model, "output_token_limit", None)
    return _DiscoveredModel(
        model_id=normalized_model_id,
        label=label,
        context_window=int(context_window),
        max_output_tokens=int(output_token_limit) if _valid_context_window(output_token_limit) else None,
        lifecycle_status="preview" if _is_preview_model_id(normalized_model_id) else "stable",
    )


def _supports_gemini_generate_content(model: object) -> bool:
    actions = getattr(model, "supported_actions", None) or ()
    normalized_actions = []
    for action in actions:
        raw = str(getattr(action, "name", action) or "").strip().lower()
        normalized_actions.append(raw.replace("_", "").replace("-", ""))
    return any("generatecontent" in action for action in normalized_actions)


def _is_excluded_gemini_model(model_id: str) -> bool:
    normalized = model_id.lower()
    excluded_tokens = (
        "latest",
        "image",
        "vision",
        "audio",
        "native-audio",
        "tts",
        "speech",
        "robotics",
        "computer-use",
        "embedding",
        "aqa",
        "learnlm",
        "live",
    )
    return any(token in normalized for token in excluded_tokens)


def _resolve_api_key(provider_id: str, *, env_vars: Sequence[str]) -> str | None:
    for env_var in env_vars:
        env_key = os.getenv(env_var)
        if env_key and env_key.strip():
            return env_key.strip()

    try:
        import keyring
        from src.app.core.secure_settings import keyring_service_name

        provider_aliases = (provider_id,)
        if provider_id == "gemini":
            provider_aliases = ("gemini", "google")
        elif provider_id == "azure_openai":
            provider_aliases = ("azure_openai", "azure")
        elif provider_id == "pydantic_ai_gateway":
            provider_aliases = ("pydantic_ai_gateway",)
        service_name = keyring_service_name()
        for alias in provider_aliases:
            key = keyring.get_password(service_name, f"api_key_{alias}")
            if key:
                return key
    except Exception:
        logger.debug("Unable to load %s API key from keyring", provider_id, exc_info=True)

    settings_path = app_config_dir() / "app_settings.json"
    if not settings_path.exists():
        return None

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Unable to read %s API key from settings file", provider_id, exc_info=True)
        return None

    api_keys = payload.get("api_keys")
    if not isinstance(api_keys, dict):
        return None

    provider_aliases = (provider_id,)
    if provider_id == "gemini":
        provider_aliases = ("gemini", "google")
    elif provider_id == "azure_openai":
        provider_aliases = ("azure_openai", "azure")
    elif provider_id == "pydantic_ai_gateway":
        provider_aliases = ("pydantic_ai_gateway",)
    for alias in provider_aliases:
        key = api_keys.get(alias)
        if isinstance(key, str) and key.strip():
            return key.strip()
    return None


def _resolve_gemini_api_key() -> str | None:
    return _resolve_api_key("gemini", env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"))


def _discover_gateway_models(provider_id: str) -> tuple[_DiscoveredModel, ...]:
    gateway_settings = _read_gateway_settings()
    base_url = str(gateway_settings.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return ()
    api_key = _resolve_api_key("pydantic_ai_gateway", env_vars=("PYDANTIC_AI_GATEWAY_API_KEY",))
    if not api_key:
        return ()

    route = str(gateway_settings.get("route") or "").strip()
    params = {"provider": provider_id}
    if route:
        params["route"] = route
    try:
        response = httpx.get(
            f"{base_url}/metadata/models",
            params=params,
            headers={"Authorization": api_key},
            timeout=15.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.debug("Unable to discover gateway models for %s", provider_id, exc_info=True)
        return ()

    if isinstance(payload, list):
        models_payload = payload
    elif isinstance(payload, dict):
        models_payload = payload.get("models")
    else:
        models_payload = None
    if not isinstance(models_payload, list):
        return ()
    models: list[_DiscoveredModel] = []
    for item in models_payload:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", "") or "").strip()
        if not model_id:
            continue
        context_window = item.get("context_window")
        max_output_tokens = item.get("max_output_tokens")
        models.append(
            _DiscoveredModel(
                model_id=model_id,
                label=str(item.get("display_name", "") or model_id).strip() or model_id,
                context_window=int(context_window) if _valid_context_window(context_window) else None,
                max_output_tokens=int(max_output_tokens) if _valid_context_window(max_output_tokens) else None,
                lifecycle_status=str(item.get("lifecycle_status", "stable") or "stable"),
            )
        )
    return tuple(models)


def _resolve_snapshot_catalog_model(provider_id: str, model_id: str) -> LLMModelOption | None:
    snapshot = _snapshot()
    upstream_provider_id = _CATALOG_PROVIDER_IDS.get(provider_id)
    if not upstream_provider_id:
        return None

    try:
        _provider, model = snapshot.find_provider_model(
            model_id,
            None,
            upstream_provider_id,
            None,
        )
    except LookupError:
        return None

    return LLMModelOption(
        model_id=model.id,
        label=model.name or model.id,
        context_window=_runtime_context_window(
            provider_id,
            model.id,
            model.context_window if _valid_context_window(model.context_window) else None,
        ),
        max_output_tokens=None,
        input_price_label=_price_label(getattr(model.prices, "input_mtok", None)),
        output_price_label=_price_label(getattr(model.prices, "output_mtok", None)),
        lifecycle_status="preview" if _is_preview_model_id(model.id) else "stable",
        reasoning_capabilities=_reasoning_capabilities_for_model(provider_id, model.id),
        provenance=LLMModelProvenance(
            availability="genai-prices",
            context="genai-prices" if _valid_context_window(model.context_window) else None,
            pricing="genai-prices",
            reasoning="provider-rules",
        ),
    )


def _is_supported_model_family(provider_id: str, model_id: str) -> bool:
    prefixes = _MODEL_FAMILY_MATCHERS.get(provider_id, ())
    normalized = (model_id or "").lower()
    if not prefixes:
        return True
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _valid_context_window(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _is_preview_model_id(model_id: str) -> bool:
    normalized = model_id.lower()
    return any(token in normalized for token in ("preview", "beta", "experimental", "-exp"))


def _model_sort_key(
    provider_id: str,
    *,
    model_id: str,
    label: str,
    context_window: int | None,
) -> tuple[int, int, int, tuple[int, ...], str]:
    normalized = f"{model_id} {label}".lower()
    preview_penalty = -20 if _is_preview_model_id(normalized) else 0
    modality_penalty = -30 if any(token in normalized for token in ("vision", "image", "audio")) else 0
    context_bonus = 5 if _valid_context_window(context_window) else 0
    return (
        _provider_family_rank(provider_id, normalized),
        preview_penalty,
        modality_penalty,
        context_bonus,
        _version_tuple(model_id),
        model_id,
    )


def _provider_family_rank(provider_id: str, normalized: str) -> int:
    if provider_id == "anthropic":
        if "sonnet" in normalized:
            return 300
        if "haiku" in normalized:
            return 200
        if "opus" in normalized:
            return 100
        return 0
    if provider_id == "gemini":
        version_rank = 0
        if "gemini-3" in normalized:
            version_rank = 500
        elif "gemini-2.5" in normalized:
            version_rank = 450
        elif "gemini-2.0" in normalized:
            version_rank = 400
        elif "1.5" in normalized:
            version_rank = 300
        elif "gemini-pro" in normalized:
            version_rank = 200

        family_rank = 0
        if " pro" in normalized or normalized.startswith("gemini-pro") or "-pro" in normalized:
            family_rank = 30
        elif "flash" in normalized:
            family_rank = 20
        elif "lite" in normalized:
            family_rank = 10
        return version_rank + family_rank
    if provider_id in {"openai", "azure_openai"}:
        if "gpt-5" in normalized:
            return 400
        if "gpt-4.1" in normalized:
            return 350
        if "gpt-4o" in normalized:
            return 300
        if normalized.startswith("o4"):
            return 260
        if normalized.startswith("o3"):
            return 240
        if normalized.startswith("o1"):
            return 220
        if "gpt-4" in normalized:
            return 200
        if "gpt-3.5" in normalized:
            return 100
        return 0
    return 0


def _version_tuple(model_id: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", model_id)]
    return tuple(parts) if parts else (0,)


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _price_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return _format_decimal(value)
    if isinstance(value, (int, float)):
        return _format_decimal(Decimal(str(value)))
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return _format_decimal(Decimal(normalized))
        except Exception:
            return None

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
    "CatalogTransport",
    "LLMModelOption",
    "LLMProviderOption",
    "SUPPORTED_SELECTOR_PROVIDER_IDS",
    "calculate_usage_cost",
    "default_model_for_provider",
    "default_provider_catalog",
    "default_provider_catalog_for_transport",
    "provider_option_map",
    "reset_gemini_model_cache",
    "reset_provider_catalog_cache",
    "resolve_catalog_model",
    "resolve_model_context_window",
    "resolve_reasoning_capabilities",
    "LLMReasoningCapabilities",
    "LLMModelProvenance",
    "runtime_default_model_for_provider",
    "start_background_catalog_refresh",
    "stop_background_catalog_refresh",
]
