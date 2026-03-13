"""Catalog and pricing helpers backed by ``genai-prices``."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import logging
import os
import re
from threading import Lock
from typing import Optional, Sequence

from genai_prices import UpdatePrices
from genai_prices.data_snapshot import DataSnapshot, get_snapshot
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
_gemini_selector_models: tuple["LLMModelOption", ...] | None = None
_gemini_selector_models_lock = Lock()


@dataclass(frozen=True, slots=True)
class _GeminiDiscoveredModel:
    model_id: str
    label: str
    context_window: int


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


def runtime_default_model_for_provider(provider_id: str) -> str | None:
    """Return the best current runtime model, even if selector metadata is incomplete."""

    env_var = _DEFAULT_MODEL_ENV_VARS.get(provider_id)
    env_model = os.getenv(env_var, "").strip() if env_var else ""
    if env_model:
        return env_model

    if provider_id == "gemini":
        models = _gemini_runtime_models()
        if models:
            return models[0].model_id

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


def resolve_catalog_model(provider_id: str, model_id: str | None) -> LLMModelOption | None:
    """Resolve ``model_id`` to catalog metadata when possible."""

    normalized = str(model_id or "").strip()
    if not normalized:
        return None

    if provider_id == "gemini":
        for model in _gemini_runtime_models():
            if model.model_id == normalized:
                return model

    return _resolve_snapshot_catalog_model(provider_id, normalized)


def resolve_model_context_window(provider_id: str, model_id: str | None) -> int | None:
    """Return the raw context window for ``model_id`` when the catalog knows it."""

    model = resolve_catalog_model(provider_id, model_id)
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
    if provider_id == "gemini":
        return _gemini_runtime_models()

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
                context_window=_runtime_context_window(
                    provider_id,
                    model.id,
                    int(model.context_window),
                ),
                input_price_label=_price_label(getattr(model.prices, "input_mtok", None)),
                output_price_label=_price_label(getattr(model.prices, "output_mtok", None)),
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
    return models


def reset_gemini_model_cache() -> None:
    """Clear the in-memory Gemini model cache."""

    global _gemini_selector_models
    with _gemini_selector_models_lock:
        _gemini_selector_models = None


def _gemini_runtime_models() -> tuple[LLMModelOption, ...]:
    global _gemini_selector_models

    with _gemini_selector_models_lock:
        if _gemini_selector_models is not None:
            return _gemini_selector_models

        discovered = _discover_gemini_models_live()
        if discovered:
            models = _build_gemini_model_options(discovered)
            _write_gemini_models_cache(discovered)
            _gemini_selector_models = models
            return models

        cached = _load_gemini_models_from_cache()
        models = _build_gemini_model_options(cached)
        _gemini_selector_models = models
        return models


def _build_gemini_model_options(
    discovered: Sequence[_GeminiDiscoveredModel],
) -> tuple[LLMModelOption, ...]:
    models: list[LLMModelOption] = []
    for discovered_model in discovered:
        catalog_model = _resolve_snapshot_catalog_model("gemini", discovered_model.model_id)
        models.append(
            LLMModelOption(
                model_id=discovered_model.model_id,
                label=discovered_model.label,
                context_window=discovered_model.context_window,
                input_price_label=catalog_model.input_price_label if catalog_model else None,
                output_price_label=catalog_model.output_price_label if catalog_model else None,
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


def _discover_gemini_models_live() -> tuple[_GeminiDiscoveredModel, ...]:
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
        models: list[_GeminiDiscoveredModel] = []
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


def _parse_gemini_discovered_model(model: object) -> _GeminiDiscoveredModel | None:
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
    return _GeminiDiscoveredModel(
        model_id=normalized_model_id,
        label=label,
        context_window=int(context_window),
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


def _resolve_gemini_api_key() -> str | None:
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    try:
        import keyring

        for provider_id in ("gemini", "google"):
            key = keyring.get_password("Llestrade", f"api_key_{provider_id}")
            if key:
                return key
    except Exception:
        logger.debug("Unable to load Gemini API key from keyring", exc_info=True)

    settings_path = app_config_dir() / "app_settings.json"
    if not settings_path.exists():
        return None

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Unable to read Gemini API key from settings file", exc_info=True)
        return None

    api_keys = payload.get("api_keys")
    if not isinstance(api_keys, dict):
        return None

    for provider_id in ("gemini", "google"):
        key = api_keys.get(provider_id)
        if isinstance(key, str) and key.strip():
            return key.strip()
    return None


def _gemini_cache_path():
    return app_config_dir() / "gemini_models_cache.json"


def _load_gemini_models_from_cache() -> tuple[_GeminiDiscoveredModel, ...]:
    cache_path = _gemini_cache_path()
    if not cache_path.exists():
        return ()
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Unable to read Gemini model cache", exc_info=True)
        return ()

    items = payload.get("models")
    if not isinstance(items, list):
        return ()

    models: list[_GeminiDiscoveredModel] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", "")).strip()
        label = str(item.get("label", "")).strip() or model_id
        context_window = item.get("context_window")
        if not model_id or not _valid_context_window(context_window):
            continue
        if _is_excluded_gemini_model(model_id):
            continue
        models.append(
            _GeminiDiscoveredModel(
                model_id=model_id,
                label=label,
                context_window=int(context_window),
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


def _write_gemini_models_cache(discovered: Sequence[_GeminiDiscoveredModel]) -> None:
    if not discovered:
        return

    payload = {
        "models": [
            {
                "model_id": model.model_id,
                "label": model.label,
                "context_window": model.context_window,
            }
            for model in discovered
        ],
    }
    try:
        _gemini_cache_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Unable to write Gemini model cache", exc_info=True)


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
        input_price_label=_price_label(getattr(model.prices, "input_mtok", None)),
        output_price_label=_price_label(getattr(model.prices, "output_mtok", None)),
    )


def _is_supported_model_family(provider_id: str, model_id: str) -> bool:
    prefixes = _MODEL_FAMILY_MATCHERS.get(provider_id, ())
    normalized = (model_id or "").lower()
    if not prefixes:
        return True
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _valid_context_window(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _model_sort_key(
    provider_id: str,
    *,
    model_id: str,
    label: str,
    context_window: int | None,
) -> tuple[int, int, int, tuple[int, ...], str]:
    normalized = f"{model_id} {label}".lower()
    preview_penalty = -20 if any(token in normalized for token in ("preview", "beta", "experimental")) else 0
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
        if " pro" in normalized or normalized.startswith("gemini-pro") or "-pro" in normalized:
            return 300
        if "flash" in normalized:
            return 200
        if "lite" in normalized:
            return 100
        return 0
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
    "reset_gemini_model_cache",
    "resolve_catalog_model",
    "resolve_model_context_window",
    "runtime_default_model_for_provider",
    "start_background_catalog_refresh",
    "stop_background_catalog_refresh",
]
