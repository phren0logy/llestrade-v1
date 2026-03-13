"""LLM execution backend contracts used by worker pipelines."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Protocol

import httpx
from pydantic_ai.settings import ModelSettings

from src.app.core.llm_catalog import default_model_for_provider, resolve_reasoning_capabilities
from src.app.core.llm_operation_settings import LLMReasoningSettings

logger = logging.getLogger(__name__)

_MODEL_REQUEST_RETRY_ATTEMPTS = 3
_MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS = 0.5
_MODEL_REQUEST_RETRY_MAX_DELAY_SECONDS = 2.0

_BEDROCK_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4-5": "anthropic.claude-sonnet-4-5-v1",
    "claude-sonnet-4-5-20250929": "anthropic.claude-sonnet-4-5-v1",
    "claude-opus-4-6": "anthropic.claude-opus-4-6-v1",
    "claude-opus-4-1-20250805": "anthropic.claude-opus-4-1-20250805-v1:0",
}

_DIRECT_PROVIDER_NAMES: Mapping[str, str] = {
    "anthropic": "anthropic",
    "anthropic_bedrock": "bedrock",
    "azure_openai": "azure",
    "gemini": "google-gla",
    "openai": "openai",
}

_GATEWAY_UPSTREAM_PROVIDERS: Mapping[str, str] = {
    "anthropic": "anthropic",
    "anthropic_bedrock": "bedrock",
    "gemini": "gemini",
    "openai": "openai",
    "azure_openai": "openai",
}

ReasoningMode = Literal["none", "anthropic", "google", "openai"]
GatewayAccessCheckKind = Literal[
    "ok",
    "missing_config",
    "auth_invalid",
    "auth_forbidden",
    "route_missing",
    "timeout",
    "unreachable",
    "server_error",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class LLMInvocationRequest:
    """Typed request contract for a single LLM invocation."""

    prompt: str
    system_prompt: Optional[str]
    model: Optional[str]
    model_settings: ModelSettings = field(default_factory=dict)
    input_tokens_limit: Optional[int] = None


@dataclass(frozen=True, slots=True)
class LLMProviderCapabilities:
    """Capability metadata for a provider/model pair as exposed by the backend."""

    provider_id: str
    model: str | None
    reasoning_mode: ReasoningMode
    supports_pre_request_token_count: bool

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_mode != "none"


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    """Lightweight provider metadata used when no native client is required."""

    provider_name: str
    default_model: str
    transport: str = "direct"


@dataclass(frozen=True, slots=True)
class DirectProviderMetadata:
    """Pydantic AI-backed provider metadata for non-gateway direct requests."""

    app_provider_id: str
    provider_name: str
    default_model: str
    provider: Any
    transport: str = "direct"


@dataclass(frozen=True, slots=True)
class LLMProviderRequest:
    """Typed contract for initializing a provider or provider metadata."""

    provider_id: str
    model: Optional[str]


@dataclass(frozen=True, slots=True)
class GatewayAccessCheck:
    """Classified result for a live gateway probe."""

    ok: bool
    kind: GatewayAccessCheckKind
    status_code: int | None
    message: str
    base_url: str | None
    route: str | None
    provider_id: str
    model: str | None


_GATEWAY_ACCESS_CACHE_TTL_SECONDS = 300.0
_gateway_access_check_cache: dict[str, tuple[float, GatewayAccessCheck]] = {}


def reset_gateway_access_check_cache() -> None:
    """Clear cached gateway probe results."""

    _gateway_access_check_cache.clear()


def _gateway_access_cache_key(
    *,
    api_key: str | None,
    base_url: str | None,
    route: str | None,
    provider_id: str,
    model: str | None,
) -> str:
    payload = "\0".join(
        (
            api_key or "",
            base_url or "",
            route or "",
            provider_id,
            model or "",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _gateway_access_ok(
    *,
    provider_id: str,
    model: str | None,
    base_url: str | None,
    route: str | None,
    message: str = "Gateway verification succeeded.",
) -> GatewayAccessCheck:
    return GatewayAccessCheck(
        ok=True,
        kind="ok",
        status_code=200,
        message=message,
        base_url=base_url,
        route=route,
        provider_id=provider_id,
        model=model,
    )


def backend_transport_name(backend: Any) -> str:
    if isinstance(backend, PydanticAIGatewayBackend):
        return "gateway"
    return "direct"


def backend_route_name(backend: Any) -> str | None:
    if isinstance(backend, PydanticAIGatewayBackend):
        return backend.current_route()
    return None

def normalize_model_name(provider_id: str, model: Optional[str]) -> str | None:
    """Normalize explicit model selections for a provider."""
    if model is None:
        return None

    normalized = str(model).strip()
    if not normalized:
        return None

    if provider_id == "anthropic_bedrock":
        return _BEDROCK_MODEL_ALIASES.get(normalized, normalized)

    return normalized


def resolve_model_name(
    provider_id: str,
    model: Optional[str],
    *,
    transport: str = "direct",
) -> str | None:
    """Normalize a model name and fall back to the provider default when needed."""
    normalized = normalize_model_name(provider_id, model)
    if normalized:
        return normalized

    fallback = default_model_for_provider(provider_id, transport=transport)
    return normalize_model_name(provider_id, fallback)


def provider_capabilities(provider_id: str, model: Optional[str]) -> LLMProviderCapabilities:
    """Return centralized capability metadata for a provider/model pair."""

    resolved_model = normalize_model_name(provider_id, model)
    reasoning_mode: ReasoningMode = "none"
    supports_pre_request_token_count = False

    if provider_id in {"anthropic", "anthropic_bedrock"}:
        reasoning_mode = "anthropic"
        supports_pre_request_token_count = True
    elif provider_id == "gemini":
        reasoning_mode = "google"
        supports_pre_request_token_count = True
    elif provider_id in {"openai", "azure_openai"}:
        reasoning_mode = "openai"

    return LLMProviderCapabilities(
        provider_id=provider_id,
        model=resolved_model,
        reasoning_mode=reasoning_mode,
        supports_pre_request_token_count=supports_pre_request_token_count,
    )


def _build_messages(request: LLMInvocationRequest) -> list[Any]:
    from pydantic_ai.messages import ModelRequest

    return [
        ModelRequest.user_text_prompt(
            request.prompt,
            instructions=request.system_prompt or None,
        )
    ]


def _build_model_settings(request: LLMInvocationRequest) -> ModelSettings:
    return dict(request.model_settings)


def _reasoning_budget_tokens(max_tokens: int) -> int:
    return min(max(max_tokens // 8, 1_024), 8_192)


def _openai_supports_sampling(*, model: str | None, use_reasoning: bool) -> bool:
    if use_reasoning:
        return False
    if not model:
        return True

    try:
        from pydantic_ai.profiles.openai import openai_model_profile
    except Exception:
        logger.debug("Unable to load OpenAI model profile; keeping sampling settings", exc_info=True)
        return True

    profile = openai_model_profile(model)
    if not profile.openai_supports_reasoning:
        return True
    return bool(profile.openai_supports_reasoning_effort_none)


def _normalize_reasoning_settings(
    *,
    reasoning: LLMReasoningSettings | Mapping[str, Any] | None,
    use_reasoning: bool,
) -> LLMReasoningSettings:
    return LLMReasoningSettings.from_value(reasoning, legacy_use_reasoning=use_reasoning)


def build_model_settings(
    provider_id: str,
    model: Optional[str],
    *,
    temperature: float,
    max_tokens: int,
    use_reasoning: bool = False,
    reasoning: LLMReasoningSettings | Mapping[str, Any] | None = None,
    base_settings: ModelSettings | None = None,
) -> ModelSettings:
    """Build provider-shaped model settings for a single request."""

    normalized_reasoning = _normalize_reasoning_settings(reasoning=reasoning, use_reasoning=use_reasoning)
    model_reasoning_capabilities = resolve_reasoning_capabilities(provider_id, model)
    settings: ModelSettings = dict(base_settings or {})
    if provider_id not in {"openai", "azure_openai"} or _openai_supports_sampling(
        model=model,
        use_reasoning=normalized_reasoning.enabled,
    ):
        settings["temperature"] = temperature
    settings["max_tokens"] = max_tokens

    capabilities = provider_capabilities(provider_id, model)
    if not capabilities.supports_reasoning or not model_reasoning_capabilities.supports_reasoning_controls:
        if normalized_reasoning.enabled:
            raise RuntimeError(
                f"Provider '{provider_id}' does not support reasoning mode in the current LLM backend."
            )
        return settings

    if provider_id in {"openai", "azure_openai"}:
        if normalized_reasoning.state == "off" and model_reasoning_capabilities.can_disable_reasoning:
            settings["openai_reasoning_effort"] = "none"
            return settings
        if normalized_reasoning.state in {"on", "default"} or not model_reasoning_capabilities.can_disable_reasoning:
            effort = normalized_reasoning.effort or "medium"
            if model_reasoning_capabilities.allowed_efforts and effort not in model_reasoning_capabilities.allowed_efforts:
                effort = model_reasoning_capabilities.allowed_efforts[0]
            settings["openai_reasoning_effort"] = effort
            settings["openai_reasoning_summary"] = "detailed"
        return settings

    if not normalized_reasoning.enabled:
        return settings

    if provider_id == "anthropic":
        settings["anthropic_thinking"] = {
            "type": "enabled",
            "budget_tokens": normalized_reasoning.budget_tokens or _reasoning_budget_tokens(max_tokens),
        }
        settings["anthropic_effort"] = "medium"
    elif provider_id == "anthropic_bedrock":
        settings["bedrock_additional_model_requests_fields"] = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": normalized_reasoning.budget_tokens or _reasoning_budget_tokens(max_tokens),
            }
        }
    elif provider_id == "gemini":
        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if model_reasoning_capabilities.allowed_levels:
            level = normalized_reasoning.level or "MEDIUM"
            if level not in model_reasoning_capabilities.allowed_levels:
                level = model_reasoning_capabilities.allowed_levels[0]
            thinking_config["thinking_level"] = level.upper()
        elif normalized_reasoning.budget_tokens:
            thinking_config["thinking_budget"] = normalized_reasoning.budget_tokens
        settings["gemini_thinking_config"] = thinking_config

    return settings


def supported_direct_provider_ids() -> tuple[str, ...]:
    return tuple(sorted(_DIRECT_PROVIDER_NAMES))


def supported_gateway_provider_ids() -> tuple[str, ...]:
    return tuple(sorted(_GATEWAY_UPSTREAM_PROVIDERS))


def _pydantic_ai_instrumentation() -> Any | None:
    try:
        from src.config.observability import get_model_instrumentation_settings

        return get_model_instrumentation_settings()
    except Exception:
        logger.debug("Unable to load Pydantic AI instrumentation settings", exc_info=True)
        return None


def _count_tokens_with_model(
    *,
    model: Any,
    request: LLMInvocationRequest,
    provider_id: str,
    model_name: str,
    error_prefix: str,
) -> int | None:
    if not request.prompt and not request.system_prompt:
        return 0

    try:
        from pydantic_ai.models import ModelRequestParameters

        usage = asyncio.run(
            model.count_tokens(
                _build_messages(request),
                _build_model_settings(request),
                ModelRequestParameters(),
            )
        )
        counted = int(getattr(usage, "input_tokens", 0) or 0)
        return counted if counted >= 0 else None
    except NotImplementedError:
        return None
    except Exception:
        logger.debug(
            "%s for provider=%s model=%s",
            error_prefix,
            provider_id,
            model_name,
            exc_info=True,
        )
        return None


def _require_response_text(
    *,
    response: Any,
    provider_name: str,
    model_name: str,
) -> Any:
    content = str(response.text or "").strip()
    if not content:
        raise RuntimeError(
            f"LLM returned empty response for provider={provider_name} model={model_name}"
        )
    return response


def _usage_limits_for_request(*, provider_id: str, request: LLMInvocationRequest) -> Any | None:
    if request.input_tokens_limit is None or request.input_tokens_limit <= 0:
        return None

    from pydantic_ai.usage import UsageLimits

    return UsageLimits(
        input_tokens_limit=request.input_tokens_limit,
        count_tokens_before_request=provider_capabilities(provider_id, request.model).supports_pre_request_token_count,
    )


def _check_before_request(
    *,
    usage_limits: Any,
    count_input_tokens_fn: Any,
) -> None:
    from pydantic_ai.usage import RunUsage

    usage = RunUsage(requests=0)
    if usage_limits.count_tokens_before_request:
        counted = count_input_tokens_fn()
        if counted is not None and counted >= 0:
            usage.input_tokens = int(counted)

    usage_limits.check_before_request(usage)


def _check_after_response(*, response: Any, usage_limits: Any) -> None:
    from pydantic_ai.usage import RunUsage

    usage = RunUsage(requests=1)
    usage.incr(response.usage)
    usage_limits.check_tokens(usage)


def _is_retryable_model_request_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError)):
        return True
    return False


def _retry_delay_seconds(attempt_number: int) -> float:
    delay = _MODEL_REQUEST_RETRY_BASE_DELAY_SECONDS * (2 ** max(attempt_number - 1, 0))
    return min(delay, _MODEL_REQUEST_RETRY_MAX_DELAY_SECONDS)


def _invoke_model_response_sync(
    *,
    model: Any,
    request: LLMInvocationRequest,
    provider_id: str,
    model_name: str | None,
    count_input_tokens_fn: Any,
    enable_pre_request_counting: bool = True,
) -> Any:
    from pydantic_ai.direct import model_request_sync

    usage_limits = _usage_limits_for_request(provider_id=provider_id, request=request)
    if usage_limits is not None and enable_pre_request_counting:
        _check_before_request(
            usage_limits=usage_limits,
            count_input_tokens_fn=count_input_tokens_fn,
        )

    messages = _build_messages(request)
    model_settings = _build_model_settings(request)
    instrumentation = _pydantic_ai_instrumentation()

    response = None
    for attempt in range(1, _MODEL_REQUEST_RETRY_ATTEMPTS + 1):
        try:
            response = model_request_sync(
                model,
                messages,
                model_settings=model_settings,
                instrument=instrumentation,
            )
            break
        except Exception as exc:
            if not _is_retryable_model_request_error(exc) or attempt >= _MODEL_REQUEST_RETRY_ATTEMPTS:
                raise
            delay_seconds = _retry_delay_seconds(attempt)
            logger.warning(
                "Transient LLM transport failure for provider=%s model=%s attempt=%s/%s; retrying in %.2fs",
                provider_id,
                model_name or request.model or "",
                attempt,
                _MODEL_REQUEST_RETRY_ATTEMPTS,
                delay_seconds,
                exc_info=True,
            )
            time.sleep(delay_seconds)

    if response is None:
        raise RuntimeError("LLM request completed without a response")
    if usage_limits is not None:
        _check_after_response(response=response, usage_limits=usage_limits)
    return response


class LLMExecutionBackend(Protocol):
    """Contract for pluggable LLM execution backends."""

    def create_provider(self, request: LLMProviderRequest) -> object:
        """Return a provider handle suitable for `invoke`."""

    def normalize_model(self, provider_id: str, model: Optional[str]) -> str | None:
        """Normalize an explicit model selection without applying defaults."""

    def resolve_model(self, provider_id: str, model: Optional[str]) -> str | None:
        """Resolve the runtime model name, applying backend defaults when needed."""

    def capabilities(self, provider_id: str, model: Optional[str]) -> LLMProviderCapabilities:
        """Describe backend capabilities for the given provider/model."""

    def build_model_settings(
        self,
        provider_id: str,
        model: Optional[str],
        *,
        temperature: float,
        max_tokens: int,
        use_reasoning: bool = False,
        reasoning: LLMReasoningSettings | Mapping[str, Any] | None = None,
        base_settings: ModelSettings | None = None,
    ) -> ModelSettings:
        """Build provider-shaped model settings for a request."""

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        """Return input token count when the backend can measure it."""

    def invoke_response(self, provider: Any, request: LLMInvocationRequest) -> Any:
        """Execute `request` and return the raw Pydantic AI `ModelResponse`."""


class PydanticAIDirectBackend:
    """Default backend that uses direct Pydantic AI providers for worker requests."""

    _PYDANTIC_PROVIDER_NAMES: Mapping[str, str] = _DIRECT_PROVIDER_NAMES

    def normalize_model(self, provider_id: str, model: Optional[str]) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: Optional[str]) -> str | None:
        return resolve_model_name(provider_id, model, transport="direct")

    def capabilities(self, provider_id: str, model: Optional[str]) -> LLMProviderCapabilities:
        return provider_capabilities(provider_id, model)

    def build_model_settings(
        self,
        provider_id: str,
        model: Optional[str],
        *,
        temperature: float,
        max_tokens: int,
        use_reasoning: bool = False,
        reasoning: LLMReasoningSettings | Mapping[str, Any] | None = None,
        base_settings: ModelSettings | None = None,
    ) -> ModelSettings:
        return build_model_settings(
            provider_id,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_reasoning=use_reasoning,
            reasoning=reasoning,
            base_settings=base_settings,
        )

    def create_provider(self, request: LLMProviderRequest) -> object:
        from src.app.core.secure_settings import SecureSettings

        settings = SecureSettings()
        direct_provider = self._create_pydantic_provider_metadata(settings=settings, request=request)
        if direct_provider is not None:
            return direct_provider

        supported = ", ".join(sorted(self._PYDANTIC_PROVIDER_NAMES))
        raise RuntimeError(
            f"Provider '{request.provider_id}' is not supported by the worker LLM backend. "
            f"Supported providers: {supported}."
        )

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        if not isinstance(provider, DirectProviderMetadata):
            return None
        return self._count_input_tokens_direct(provider, request)

    def invoke_response(self, provider: Any, request: LLMInvocationRequest) -> Any:
        if not isinstance(provider, DirectProviderMetadata):
            raise RuntimeError("Direct provider backend requires DirectProviderMetadata")
        return self._invoke_direct_response(provider, request)

    def _create_pydantic_provider_metadata(
        self,
        *,
        settings: Any,
        request: LLMProviderRequest,
    ) -> DirectProviderMetadata | None:
        provider_name = self._PYDANTIC_PROVIDER_NAMES.get(request.provider_id)
        if provider_name is None:
            return None

        provider = self._build_pydantic_provider(
            settings=settings,
            app_provider_id=request.provider_id,
            provider_name=provider_name,
        )
        return DirectProviderMetadata(
            app_provider_id=request.provider_id,
            provider_name=provider_name,
            default_model=self.resolve_model(request.provider_id, request.model) or "",
            provider=provider,
        )

    def _build_pydantic_provider(
        self,
        *,
        settings: Any,
        app_provider_id: str,
        provider_name: str,
    ) -> Any:
        api_key = str(settings.get_api_key(app_provider_id) or "").strip()
        if provider_name == "anthropic":
            from pydantic_ai.providers.anthropic import AnthropicProvider

            if not api_key:
                raise RuntimeError("No Anthropic API key configured for direct provider requests.")
            return AnthropicProvider(api_key=api_key)
        if provider_name == "google-gla":
            from pydantic_ai.providers.google import GoogleProvider

            if not api_key:
                raise RuntimeError("No Gemini API key configured for direct provider requests.")
            return GoogleProvider(api_key=api_key)
        if provider_name == "openai":
            from pydantic_ai.providers.openai import OpenAIProvider

            if not api_key:
                raise RuntimeError("No OpenAI API key configured for direct provider requests.")
            return OpenAIProvider(api_key=api_key)
        if provider_name == "azure":
            from pydantic_ai.providers.azure import AzureProvider

            azure_settings = settings.get("azure_openai_settings", {}) or {}
            if not api_key:
                raise RuntimeError("No Azure OpenAI API key configured for direct provider requests.")
            if not str(azure_settings.get("endpoint") or "").strip():
                raise RuntimeError("No Azure OpenAI endpoint configured for direct provider requests.")
            return AzureProvider(
                api_key=api_key,
                azure_endpoint=azure_settings.get("endpoint"),
                api_version=azure_settings.get("api_version"),
            )
        if provider_name == "bedrock":
            from pydantic_ai.providers.bedrock import BedrockProvider

            bedrock_settings = settings.get("aws_bedrock_settings", {}) or {}
            return BedrockProvider(
                region_name=bedrock_settings.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
                profile_name=bedrock_settings.get("profile"),
            )
        raise RuntimeError(f"Unsupported Pydantic AI provider '{provider_name}'")

    def _count_input_tokens_direct(
        self,
        provider: DirectProviderMetadata,
        request: LLMInvocationRequest,
    ) -> int | None:
        model_name = self.resolve_model(provider.app_provider_id, request.model or provider.default_model)
        if not model_name:
            return None
        return _count_tokens_with_model(
            model=self._build_direct_model(provider=provider, model_name=model_name),
            request=request,
            provider_id=provider.app_provider_id,
            model_name=model_name,
            error_prefix="Direct provider token preflight failed",
        )

    def _invoke_direct_response(
        self,
        provider: DirectProviderMetadata,
        request: LLMInvocationRequest,
    ) -> Any:
        model_name = self.resolve_model(provider.app_provider_id, request.model or provider.default_model)
        if not model_name:
            raise RuntimeError("No model configured for direct provider backend")
        response = _invoke_model_response_sync(
            model=self._build_direct_model(provider=provider, model_name=model_name),
            request=request,
            provider_id=provider.app_provider_id,
            model_name=model_name,
            count_input_tokens_fn=lambda: self._count_input_tokens_direct(provider, request),
        )
        return _require_response_text(
            response=response,
            provider_name=provider.app_provider_id,
            model_name=model_name,
        )
    @staticmethod
    def _build_direct_model(*, provider: DirectProviderMetadata, model_name: str) -> Any:
        from pydantic_ai.models import infer_model

        return infer_model(
            f"{provider.provider_name}:{model_name}",
            provider_factory=lambda _provider_name: provider.provider,
        )


class PydanticAIGatewayBackend:
    """LLM backend using Pydantic AI Gateway model providers.

    This backend is intentionally scoped for worker-style single-shot calls.
    Agent/tool orchestration remains in worker code.
    """

    _RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
    _GATEWAY_RETRY_ATTEMPTS = 3
    _GATEWAY_RETRY_MAX_WAIT_SECONDS = 30.0
    _DEFAULT_GATEWAY_MAX_CONCURRENCY = 4
    _VERIFY_PROMPT = "Return OK."
    _VERIFY_SYSTEM_PROMPT = "Reply with the single word OK."

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        route: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self._explicit_api_key = api_key
        self._explicit_base_url = base_url
        self._explicit_route = route
        self._max_concurrency = self._resolve_max_concurrency(max_concurrency)
        self._http_client: Any | None = None
        self._concurrency_limiter: Any | None = None

    @staticmethod
    def _load_base_url_from_settings() -> str | None:
        try:
            from src.app.core.secure_settings import SecureSettings

            settings = SecureSettings()
            gateway_settings = settings.get("pydantic_ai_gateway_settings", {}) or {}
            base_url = str(gateway_settings.get("base_url") or "").strip()
            return base_url or None
        except Exception:
            return None

    @staticmethod
    def _load_api_key_from_settings() -> str | None:
        try:
            from src.app.core.secure_settings import SecureSettings

            settings = SecureSettings()
            api_key = settings.get_api_key("pydantic_ai_gateway")
            value = str(api_key or "").strip()
            return value or None
        except Exception:
            return None

    @staticmethod
    def _load_route_from_settings() -> str | None:
        try:
            from src.app.core.secure_settings import SecureSettings

            settings = SecureSettings()
            gateway_settings = settings.get("pydantic_ai_gateway_settings", {}) or {}
            route = str(gateway_settings.get("route") or "").strip()
            return route or None
        except Exception:
            return None

    def normalize_model(self, provider_id: str, model: Optional[str]) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: Optional[str]) -> str | None:
        return resolve_model_name(provider_id, model, transport="gateway")

    def capabilities(self, provider_id: str, model: Optional[str]) -> LLMProviderCapabilities:
        return provider_capabilities(provider_id, model)

    def build_model_settings(
        self,
        provider_id: str,
        model: Optional[str],
        *,
        temperature: float,
        max_tokens: int,
        use_reasoning: bool = False,
        reasoning: LLMReasoningSettings | Mapping[str, Any] | None = None,
        base_settings: ModelSettings | None = None,
    ) -> ModelSettings:
        return build_model_settings(
            provider_id,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_reasoning=use_reasoning,
            reasoning=reasoning,
            base_settings=base_settings,
        )

    def create_provider(self, request: LLMProviderRequest) -> object:
        if request.provider_id not in _GATEWAY_UPSTREAM_PROVIDERS:
            supported = ", ".join(supported_gateway_provider_ids())
            raise RuntimeError(
                f"Provider '{request.provider_id}' is not supported by the Pydantic AI Gateway backend. "
                f"Supported providers: {supported}."
            )
        return ProviderMetadata(
            provider_name=request.provider_id,
            default_model=self.resolve_model(request.provider_id, request.model) or "",
            transport="gateway",
        )

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        provider_id = self._resolve_provider_id(provider)
        if not provider_id:
            return None

        model_name = self.resolve_model(
            provider_id,
            request.model or self._resolve_default_model(provider),
        )
        if not model_name:
            return None
        return _count_tokens_with_model(
            model=self._build_model(provider_id=provider_id, model_name=model_name),
            request=request,
            provider_id=provider_id,
            model_name=model_name,
            error_prefix="Gateway token preflight failed",
        )

    def verify_gateway_access(
        self,
        provider_id: str,
        model: str | None,
        *,
        timeout_seconds: float = 5.0,
        force: bool = False,
    ) -> GatewayAccessCheck:
        resolved_model = self.resolve_model(provider_id, model)
        base_url = self._current_base_url()
        route = self._current_route()
        api_key = self._current_api_key()

        if provider_id not in _GATEWAY_UPSTREAM_PROVIDERS:
            return GatewayAccessCheck(
                ok=False,
                kind="missing_config",
                status_code=None,
                message=f"Provider '{provider_id}' is not supported by the Pydantic AI Gateway backend.",
                base_url=base_url,
                route=route,
                provider_id=provider_id,
                model=resolved_model or model,
            )
        if not api_key:
            return GatewayAccessCheck(
                ok=False,
                kind="missing_config",
                status_code=None,
                message="No Pydantic AI Gateway app key is configured.",
                base_url=base_url,
                route=route,
                provider_id=provider_id,
                model=resolved_model or model,
            )
        if not resolved_model:
            return GatewayAccessCheck(
                ok=False,
                kind="missing_config",
                status_code=None,
                message=f"No model is configured for gateway provider '{provider_id}'.",
                base_url=base_url,
                route=route,
                provider_id=provider_id,
                model=model,
            )

        cache_key = _gateway_access_cache_key(
            api_key=api_key,
            base_url=base_url,
            route=route,
            provider_id=provider_id,
            model=resolved_model,
        )
        now = time.time()
        if not force:
            cached_entry = _gateway_access_check_cache.get(cache_key)
            if cached_entry and cached_entry[0] > now:
                return cached_entry[1]
            if cached_entry:
                _gateway_access_check_cache.pop(cache_key, None)

        result = self._probe_gateway_access(
            provider_id=provider_id,
            model_name=resolved_model,
            api_key=api_key,
            base_url=base_url,
            route=route,
            timeout_seconds=timeout_seconds,
        )
        if result.ok or result.kind in {"auth_invalid", "auth_forbidden"}:
            _gateway_access_check_cache[cache_key] = (
                now + _GATEWAY_ACCESS_CACHE_TTL_SECONDS,
                result,
            )
        return result

    def invoke_response(self, provider: Any, request: LLMInvocationRequest) -> Any:
        provider_id = self._resolve_provider_id(provider)
        if not provider_id:
            raise RuntimeError("Unable to resolve provider ID for Gateway backend")

        model_name = self.resolve_model(
            provider_id,
            request.model or self._resolve_default_model(provider),
        )
        if not model_name:
            raise RuntimeError("No model configured for Gateway backend")

        response = _invoke_model_response_sync(
            model=self._build_model(provider_id=provider_id, model_name=model_name),
            request=request,
            provider_id=provider_id,
            model_name=model_name,
            count_input_tokens_fn=lambda: self.count_input_tokens(provider, request),
            # Gateway sync requests intentionally skip exact token preflight because the
            # cached async gateway client may otherwise be reused across event loops.
            enable_pre_request_counting=False,
        )
        return _require_response_text(
            response=response,
            provider_name=f"gateway/{provider_id}",
            model_name=model_name,
        )

    def _build_model(self, *, provider_id: str, model_name: str, http_client: Any | None = None) -> Any:
        from pydantic_ai.models import infer_model
        from pydantic_ai.models.concurrency import limit_model_concurrency

        model = infer_model(
            self._gateway_model_id(provider_id=provider_id, model_name=model_name),
            provider_factory=lambda provider_name: self._gateway_provider_factory(
                provider_name,
                http_client=http_client,
            ),
        )
        limiter = self._gateway_concurrency_limiter()
        if limiter is None:
            return model
        return limit_model_concurrency(model, limiter=limiter)

    @staticmethod
    def _resolve_provider_id(provider: Any) -> str | None:
        value = getattr(provider, "provider_name", None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if not value:
            return None
        return str(value)

    @staticmethod
    def _resolve_default_model(provider: Any) -> str | None:
        value = getattr(provider, "default_model", None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if not value:
            return None
        return str(value)

    def _gateway_provider_factory(self, provider_name: str, *, http_client: Any | None = None) -> Any:
        from pydantic_ai.providers.gateway import gateway_provider

        upstream_provider = provider_name.removeprefix("gateway/")
        return gateway_provider(
            upstream_provider,
            api_key=self._current_api_key(),
            base_url=self._current_base_url(),
            route=self._current_route(),
            http_client=http_client or self._gateway_http_client(),
        )

    def _current_api_key(self) -> str | None:
        return (
            self._explicit_api_key
            or os.getenv("PYDANTIC_AI_GATEWAY_API_KEY")
            or os.getenv("PAIG_API_KEY")
            or self._load_api_key_from_settings()
        )

    def _current_base_url(self) -> str | None:
        return (
            self._explicit_base_url
            or os.getenv("PYDANTIC_AI_GATEWAY_BASE_URL")
            or os.getenv("PAIG_BASE_URL")
            or self._load_base_url_from_settings()
        )

    def _current_route(self) -> str | None:
        return (
            self._explicit_route
            or os.getenv("PYDANTIC_AI_GATEWAY_ROUTE")
            or self._load_route_from_settings()
        )

    def current_route(self) -> str | None:
        return self._current_route()

    def _gateway_http_client(self) -> Any | None:
        if self._http_client is not None:
            return self._http_client

        self._http_client = self._build_gateway_http_client()
        return self._http_client

    def _build_gateway_http_client(self, *, timeout_seconds: float | None = None) -> Any | None:
        try:
            from httpx import AsyncClient, HTTPStatusError, TimeoutException, TransportError
            from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after
            from tenacity import retry_if_exception_type, stop_after_attempt
        except ImportError:
            logger.debug("Gateway retry transport unavailable; using default Pydantic AI HTTP client", exc_info=True)
            return None

        retry_transport = AsyncTenacityTransport(
            {
                "retry": retry_if_exception_type((HTTPStatusError, TimeoutException, TransportError)),
                "wait": wait_retry_after(max_wait=self._GATEWAY_RETRY_MAX_WAIT_SECONDS),
                "stop": stop_after_attempt(self._GATEWAY_RETRY_ATTEMPTS),
                "reraise": True,
            },
            validate_response=self._raise_for_retryable_gateway_response,
        )
        client_kwargs: dict[str, Any] = {"transport": retry_transport}
        if timeout_seconds and timeout_seconds > 0:
            client_kwargs["timeout"] = timeout_seconds
        return AsyncClient(**client_kwargs)

    @staticmethod
    def _resolve_max_concurrency(value: int | None) -> int | None:
        if value is not None:
            return value if value > 0 else None

        raw = os.getenv("PYDANTIC_AI_GATEWAY_MAX_CONCURRENCY")
        if raw is None:
            return PydanticAIGatewayBackend._DEFAULT_GATEWAY_MAX_CONCURRENCY

        try:
            parsed = int(raw)
        except ValueError:
            logger.warning("Invalid PYDANTIC_AI_GATEWAY_MAX_CONCURRENCY=%r; using default", raw)
            return PydanticAIGatewayBackend._DEFAULT_GATEWAY_MAX_CONCURRENCY

        return parsed if parsed > 0 else None

    def _gateway_concurrency_limiter(self) -> Any | None:
        if self._max_concurrency is None:
            return None
        if self._concurrency_limiter is not None:
            return self._concurrency_limiter

        from pydantic_ai.concurrency import ConcurrencyLimiter

        self._concurrency_limiter = ConcurrencyLimiter(
            self._max_concurrency,
            name="pydantic-ai-gateway",
        )
        return self._concurrency_limiter

    @classmethod
    def _raise_for_retryable_gateway_response(cls, response: Any) -> None:
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code in cls._RETRYABLE_STATUS_CODES:
            response.raise_for_status()

    @classmethod
    def _gateway_model_id(cls, *, provider_id: str, model_name: str) -> str:
        upstream_provider = cls._gateway_upstream_provider(provider_id)
        if not upstream_provider:
            raise RuntimeError(f"Provider '{provider_id}' is not supported by Pydantic AI Gateway backend")
        return f"gateway/{upstream_provider}:{model_name}"

    @staticmethod
    def _gateway_upstream_provider(provider_id: str) -> str | None:
        return _GATEWAY_UPSTREAM_PROVIDERS.get(provider_id)

    def _probe_gateway_access(
        self,
        *,
        provider_id: str,
        model_name: str,
        api_key: str,
        base_url: str | None,
        route: str | None,
        timeout_seconds: float,
    ) -> GatewayAccessCheck:
        verify_client = self._build_gateway_http_client(timeout_seconds=timeout_seconds)
        request = LLMInvocationRequest(
            prompt=self._VERIFY_PROMPT,
            system_prompt=self._VERIFY_SYSTEM_PROMPT,
            model=model_name,
            model_settings=self.build_model_settings(
                provider_id,
                model_name,
                temperature=0.0,
                max_tokens=8,
                use_reasoning=False,
            ),
        )
        try:
            _invoke_model_response_sync(
                model=self._build_model(provider_id=provider_id, model_name=model_name, http_client=verify_client),
                request=request,
                provider_id=provider_id,
                model_name=model_name,
                count_input_tokens_fn=lambda: None,
                enable_pre_request_counting=False,
            )
            return _gateway_access_ok(
                provider_id=provider_id,
                model=model_name,
                base_url=base_url,
                route=route,
            )
        except Exception as exc:
            return self._classify_gateway_probe_error(
                exc,
                provider_id=provider_id,
                model_name=model_name,
                base_url=base_url,
                route=route,
            )
        finally:
            if verify_client is not None:
                try:
                    asyncio.run(verify_client.aclose())
                except Exception:
                    logger.debug("Unable to close temporary gateway probe HTTP client", exc_info=True)

    @classmethod
    def _classify_gateway_probe_error(
        cls,
        exc: Exception,
        *,
        provider_id: str,
        model_name: str,
        base_url: str | None,
        route: str | None,
    ) -> GatewayAccessCheck:
        from pydantic_ai.exceptions import ModelHTTPError

        if isinstance(exc, ModelHTTPError):
            status_code = int(getattr(exc, "status_code", 0) or 0) or None
            message = cls._gateway_probe_error_message(exc)
            kind: GatewayAccessCheckKind = "unknown"
            lower_message = message.lower()
            if status_code == 401:
                kind = "auth_invalid"
            elif status_code == 403:
                kind = "auth_forbidden"
            elif status_code == 404 or "route not found" in lower_message or "no providers available for route" in lower_message:
                kind = "route_missing"
            elif status_code is not None and status_code >= 500:
                kind = "server_error"
            return GatewayAccessCheck(
                ok=False,
                kind=kind,
                status_code=status_code,
                message=message,
                base_url=base_url,
                route=route,
                provider_id=provider_id,
                model=model_name,
            )
        if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
            return GatewayAccessCheck(
                ok=False,
                kind="timeout",
                status_code=None,
                message=str(exc) or "Gateway request timed out.",
                base_url=base_url,
                route=route,
                provider_id=provider_id,
                model=model_name,
            )
        if isinstance(exc, httpx.TransportError):
            return GatewayAccessCheck(
                ok=False,
                kind="unreachable",
                status_code=None,
                message=str(exc) or "Gateway request could not reach the server.",
                base_url=base_url,
                route=route,
                provider_id=provider_id,
                model=model_name,
            )
        return GatewayAccessCheck(
            ok=False,
            kind="unknown",
            status_code=getattr(exc, "status_code", None),
            message=str(exc) or exc.__class__.__name__,
            base_url=base_url,
            route=route,
            provider_id=provider_id,
            model=model_name,
        )

    @staticmethod
    def _gateway_probe_error_message(exc: Exception) -> str:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            message = str(body.get("message") or body.get("error") or "").strip()
            status = str(body.get("status") or "").strip()
            if message and status and status.lower() not in message.lower():
                return f"{message} ({status})"
            if message:
                return message
            if status:
                return status
        if body is not None:
            text = str(body).strip()
            if text:
                return text
        return str(exc) or exc.__class__.__name__

__all__ = [
    "GatewayAccessCheck",
    "ProviderMetadata",
    "LLMExecutionBackend",
    "LLMProviderRequest",
    "LLMInvocationRequest",
    "LLMProviderCapabilities",
    "PydanticAIDirectBackend",
    "PydanticAIGatewayBackend",
    "default_model_for_provider",
    "build_model_settings",
    "backend_transport_name",
    "backend_route_name",
    "normalize_model_name",
    "provider_capabilities",
    "resolve_model_name",
    "reset_gateway_access_check_cache",
    "supported_direct_provider_ids",
    "supported_gateway_provider_ids",
]
