"""LLM execution backend contracts used by worker pipelines."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Protocol

import httpx
from pydantic_ai.settings import ModelSettings

from src.app.core.llm_catalog import default_model_for_provider

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


def build_model_settings(
    provider_id: str,
    model: Optional[str],
    *,
    temperature: float,
    max_tokens: int,
    use_reasoning: bool = False,
    base_settings: ModelSettings | None = None,
) -> ModelSettings:
    """Build provider-shaped model settings for a single request."""

    settings: ModelSettings = dict(base_settings or {})
    settings["temperature"] = temperature
    settings["max_tokens"] = max_tokens

    if not use_reasoning:
        return settings

    capabilities = provider_capabilities(provider_id, model)
    if not capabilities.supports_reasoning:
        raise RuntimeError(
            f"Provider '{provider_id}' does not support reasoning mode in the current LLM backend."
        )

    if provider_id == "anthropic":
        settings["anthropic_thinking"] = {
            "type": "enabled",
            "budget_tokens": _reasoning_budget_tokens(max_tokens),
        }
        settings["anthropic_effort"] = "medium"
    elif provider_id == "anthropic_bedrock":
        settings["bedrock_additional_model_requests_fields"] = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": _reasoning_budget_tokens(max_tokens),
            }
        }
    elif provider_id == "gemini":
        settings["gemini_thinking_config"] = {
            "include_thoughts": True,
        }
    elif provider_id in {"openai", "azure_openai"}:
        settings["openai_reasoning_effort"] = "medium"
        settings["openai_reasoning_summary"] = "detailed"

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
        base_settings: ModelSettings | None = None,
    ) -> ModelSettings:
        return build_model_settings(
            provider_id,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_reasoning=use_reasoning,
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
        base_settings: ModelSettings | None = None,
    ) -> ModelSettings:
        return build_model_settings(
            provider_id,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_reasoning=use_reasoning,
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

    def _build_model(self, *, provider_id: str, model_name: str) -> Any:
        from pydantic_ai.models import infer_model
        from pydantic_ai.models.concurrency import limit_model_concurrency

        model = infer_model(
            self._gateway_model_id(provider_id=provider_id, model_name=model_name),
            provider_factory=self._gateway_provider_factory,
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

    def _gateway_provider_factory(self, provider_name: str) -> Any:
        from pydantic_ai.providers.gateway import gateway_provider

        upstream_provider = provider_name.removeprefix("gateway/")
        return gateway_provider(
            upstream_provider,
            api_key=self._current_api_key(),
            base_url=self._current_base_url(),
            route=self._current_route(),
            http_client=self._gateway_http_client(),
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

    def _build_gateway_http_client(self) -> Any | None:
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
        return AsyncClient(transport=retry_transport)

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

__all__ = [
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
    "supported_direct_provider_ids",
    "supported_gateway_provider_ids",
]
