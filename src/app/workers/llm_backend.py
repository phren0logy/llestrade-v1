"""LLM execution backend contracts used by worker pipelines."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMInvocationRequest:
    """Typed request contract for a single LLM invocation."""

    prompt: str
    system_prompt: Optional[str]
    model: Optional[str]
    temperature: float
    max_tokens: int
    input_tokens_limit: Optional[int] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMInvocationResult:
    """Typed response contract returned by execution backends."""

    success: bool
    content: str
    error: Optional[str]
    usage: Dict[str, Any]
    provider: Optional[str]
    model: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_provider_response(cls, payload: Mapping[str, Any]) -> "LLMInvocationResult":
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return cls(
            success=bool(payload.get("success")),
            content=str(payload.get("content") or ""),
            error=(str(payload.get("error")) if payload.get("error") else None),
            usage=dict(usage),
            provider=(str(payload.get("provider")) if payload.get("provider") else None),
            model=(str(payload.get("model")) if payload.get("model") else None),
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    """Lightweight provider metadata used when no native client is required."""

    provider_name: str
    default_model: str


@dataclass(frozen=True, slots=True)
class LLMProviderRequest:
    """Typed contract for initializing a provider or provider metadata."""

    provider_id: str
    model: Optional[str]
    system_prompt: Optional[str]


def default_model_for_provider(provider_id: str) -> str:
    """Return a stable fallback model when native providers are not initialized."""
    defaults = {
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "anthropic_bedrock": os.getenv("BEDROCK_ANTHROPIC_MODEL", "anthropic.claude-sonnet-4-5-v1"),
        "azure_openai": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4.1"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
    }
    return defaults.get(provider_id, "")


class LLMExecutionBackend(Protocol):
    """Contract for pluggable LLM execution backends."""

    def requires_native_provider(self) -> bool:
        """Whether worker code must initialize a native provider client."""

    def create_provider(self, request: LLMProviderRequest) -> object:
        """Return a provider handle suitable for `invoke`."""

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        """Return input token count when the backend can measure it."""

    def invoke(self, provider: Any, request: LLMInvocationRequest) -> LLMInvocationResult:
        """Execute `request` against the given provider."""


class LegacyProviderBackend:
    """Default backend that forwards directly to existing provider.generate calls."""

    def requires_native_provider(self) -> bool:
        return True

    def create_provider(self, request: LLMProviderRequest) -> object:
        from src.app.core.secure_settings import SecureSettings
        from src.common.llm.factory import create_provider

        settings = SecureSettings()
        provider_id = request.provider_id
        api_key = settings.get_api_key(provider_id)
        kwargs: Dict[str, Any] = {
            "provider": provider_id,
            "default_system_prompt": request.system_prompt,
            "api_key": api_key,
        }
        if provider_id == "azure_openai":
            azure_settings = settings.get("azure_openai_settings", {}) or {}
            kwargs["azure_endpoint"] = azure_settings.get("endpoint")
            kwargs["api_version"] = azure_settings.get("api_version")
        elif provider_id == "anthropic_bedrock":
            bedrock_settings = settings.get("aws_bedrock_settings", {}) or {}
            kwargs["aws_region"] = bedrock_settings.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            kwargs["aws_profile"] = bedrock_settings.get("profile")

        provider = create_provider(**kwargs)
        if provider is None or not getattr(provider, "initialized", False):
            raise RuntimeError(
                f"Unable to initialise provider '{provider_id}'. Check API keys and model configuration in Settings."
            )
        return provider

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        count_tokens = getattr(provider, "count_tokens", None)
        if not callable(count_tokens):
            return None

        combined_prompt = self._combine_prompt(request)
        if not combined_prompt:
            return 0

        try:
            result = count_tokens(text=combined_prompt)
        except Exception:
            logger.debug("Native provider token preflight failed", exc_info=True)
            return None

        return self._extract_token_count(result)

    def invoke(self, provider: Any, request: LLMInvocationRequest) -> LLMInvocationResult:
        kwargs: Dict[str, Any] = {
            "prompt": request.prompt,
            "model": request.model,
            "system_prompt": request.system_prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        kwargs.update(dict(request.extra))
        response = provider.generate(**kwargs)
        if not isinstance(response, dict):
            response = {"success": False, "error": "Provider returned non-dict response"}
        return LLMInvocationResult.from_provider_response(response)

    @staticmethod
    def _combine_prompt(request: LLMInvocationRequest) -> str:
        return f"{(request.system_prompt or '').strip()}\n\n{request.prompt.strip()}".strip()

    @staticmethod
    def _extract_token_count(payload: Any) -> int | None:
        if not isinstance(payload, Mapping):
            return None
        if not payload.get("success"):
            return None
        token_count = payload.get("token_count")
        if token_count is None:
            return None
        counted = int(token_count or 0)
        return counted if counted >= 0 else None


class PydanticAIGatewayBackend:
    """LLM backend using Pydantic AI Gateway model providers.

    This backend is intentionally scoped for worker-style single-shot calls.
    Agent/tool orchestration remains in worker code.
    """

    _FORWARDED_SETTINGS: tuple[str, ...] = (
        "top_p",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
    )
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
        fallback_backend: LLMExecutionBackend | None = None,
        fallback_on_error: bool = False,
    ) -> None:
        self._api_key = (
            api_key
            or os.getenv("PYDANTIC_AI_GATEWAY_API_KEY")
            or os.getenv("PAIG_API_KEY")
            or self._load_api_key_from_settings()
        )
        self._base_url = (
            base_url
            or os.getenv("PYDANTIC_AI_GATEWAY_BASE_URL")
            or os.getenv("PAIG_BASE_URL")
            or self._load_base_url_from_settings()
        )
        self._route = route or os.getenv("PYDANTIC_AI_GATEWAY_ROUTE")
        self._max_concurrency = self._resolve_max_concurrency(max_concurrency)
        self._fallback_backend = fallback_backend
        self._fallback_on_error = fallback_on_error
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

    def requires_native_provider(self) -> bool:
        return False

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(
            provider_name=request.provider_id,
            default_model=request.model or default_model_for_provider(request.provider_id),
        )

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        provider_id = self._resolve_provider_id(provider)
        if not provider_id:
            return None

        model_name = request.model or self._resolve_default_model(provider)
        if not model_name:
            return None

        if not request.prompt and not request.system_prompt:
            return 0

        try:
            model = self._build_model(provider_id=provider_id, model_name=model_name)
            from pydantic_ai.models import ModelRequestParameters

            usage = asyncio.run(
                model.count_tokens(
                    self._build_messages(request),
                    self._build_model_settings(request),
                    ModelRequestParameters(),
                )
            )
            counted = int(getattr(usage, "input_tokens", 0) or 0)
            return counted if counted >= 0 else None
        except NotImplementedError:
            return None
        except Exception:
            logger.debug(
                "Gateway token preflight failed for provider=%s model=%s",
                provider_id,
                model_name,
                exc_info=True,
            )
            return None

    def invoke(self, provider: Any, request: LLMInvocationRequest) -> LLMInvocationResult:
        provider_id = self._resolve_provider_id(provider)
        if not provider_id:
            return self._fallback_or_error(
                provider=provider,
                request=request,
                reason="Unable to resolve provider ID for Gateway backend",
            )

        model_name = request.model or self._resolve_default_model(provider)
        if not model_name:
            return self._fallback_or_error(
                provider=provider,
                request=request,
                reason="No model configured for Gateway backend",
            )

        try:
            model = self._build_model(provider_id=provider_id, model_name=model_name)
            from pydantic_ai.direct import model_request_sync

            usage_limits = self._usage_limits_for_request(provider_id=provider_id, request=request)
            if usage_limits is not None:
                self._check_before_request(
                    provider=provider,
                    provider_id=provider_id,
                    request=request,
                    usage_limits=usage_limits,
                )

            response = model_request_sync(
                model,
                self._build_messages(request),
                model_settings=self._build_model_settings(request),
            )
            if usage_limits is not None:
                self._check_after_response(response=response, usage_limits=usage_limits)

            content = str(response.text or "").strip()
            if not content:
                return LLMInvocationResult(
                    success=False,
                    content="",
                    error="LLM returned empty response",
                    usage={},
                    provider=f"gateway/{provider_id}",
                    model=model_name,
                    raw={},
                )

            usage_obj = response.usage
            usage = {
                "requests": int(getattr(usage_obj, "requests", 0) or 0),
                "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0)
                + int(getattr(usage_obj, "output_tokens", 0) or 0),
            }
            details = getattr(usage_obj, "details", None)
            if isinstance(details, dict) and details:
                usage["details"] = dict(details)

            raw = self._response_metadata(response)

            return LLMInvocationResult(
                success=True,
                content=content,
                error=None,
                usage=usage,
                provider=f"gateway/{provider_id}",
                model=str(getattr(response, "model_name", None) or model_name),
                raw=raw,
            )
        except Exception as exc:
            if "not supported by Pydantic AI Gateway backend" in str(exc):
                return self._fallback_or_error(
                    provider=provider,
                    request=request,
                    reason=f"Gateway invocation failed: {exc}",
                )
            return self._fallback_or_error(
                provider=provider,
                request=request,
                reason=f"Gateway invocation failed: {exc}",
                fallback_on_error=True,
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

    def _fallback_or_error(
        self,
        *,
        provider: Any,
        request: LLMInvocationRequest,
        reason: str,
        fallback_on_error: bool = False,
    ) -> LLMInvocationResult:
        if (
            self._fallback_backend
            and (not fallback_on_error or self._fallback_on_error)
            and self._can_use_fallback_backend(provider)
        ):
            return self._fallback_backend.invoke(provider, request)
        return LLMInvocationResult(
            success=False,
            content="",
            error=reason,
            usage={},
            provider=None,
            model=request.model,
            raw={},
        )

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

    def _can_use_fallback_backend(self, provider: Any) -> bool:
        if not self._fallback_backend:
            return False
        if not isinstance(self._fallback_backend, LegacyProviderBackend):
            return True
        generate = getattr(provider, "generate", None)
        return callable(generate)

    def _gateway_provider_factory(self, provider_name: str) -> Any:
        from pydantic_ai.providers.gateway import gateway_provider

        upstream_provider = provider_name.removeprefix("gateway/")
        return gateway_provider(
            upstream_provider,
            api_key=self._api_key,
            base_url=self._base_url,
            route=self._route,
            http_client=self._gateway_http_client(),
        )

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
        mapping = {
            "anthropic": "anthropic",
            "anthropic_bedrock": "bedrock",
            "gemini": "gemini",
            "openai": "openai",
            "azure_openai": "openai",
        }
        return mapping.get(provider_id)

    @staticmethod
    def _build_messages(request: LLMInvocationRequest) -> list[Any]:
        from pydantic_ai.messages import ModelRequest

        return [
            ModelRequest.user_text_prompt(
                request.prompt,
                instructions=request.system_prompt or None,
            )
        ]

    def _build_model_settings(self, request: LLMInvocationRequest) -> Dict[str, Any]:
        model_settings: Dict[str, Any] = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        for key in self._FORWARDED_SETTINGS:
            if key in request.extra:
                model_settings[key] = request.extra[key]
        return model_settings

    @staticmethod
    def _response_metadata(response: Any) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}
        provider_response_id = getattr(response, "provider_response_id", None)
        finish_reason = getattr(response, "finish_reason", None)
        provider_name = getattr(response, "provider_name", None)
        provider_url = getattr(response, "provider_url", None)
        if provider_response_id:
            raw["provider_response_id"] = str(provider_response_id)
        if finish_reason:
            raw["finish_reason"] = str(finish_reason)
        if provider_name:
            raw["provider_name"] = str(provider_name)
        if provider_url:
            raw["provider_url"] = str(provider_url)
        provider_details = getattr(response, "provider_details", None)
        if isinstance(provider_details, dict) and provider_details:
            raw["provider_details"] = dict(provider_details)
        return raw

    @staticmethod
    def _supports_pre_request_count(provider_id: str) -> bool:
        return provider_id in {"anthropic", "anthropic_bedrock", "gemini"}

    def _usage_limits_for_request(self, *, provider_id: str, request: LLMInvocationRequest) -> Any | None:
        if request.input_tokens_limit is None or request.input_tokens_limit <= 0:
            return None

        from pydantic_ai.usage import UsageLimits

        return UsageLimits(
            input_tokens_limit=request.input_tokens_limit,
            count_tokens_before_request=self._supports_pre_request_count(provider_id),
        )

    def _check_before_request(
        self,
        *,
        provider: Any,
        provider_id: str,
        request: LLMInvocationRequest,
        usage_limits: Any,
    ) -> None:
        from pydantic_ai.usage import RunUsage

        usage = RunUsage(requests=0)
        if usage_limits.count_tokens_before_request:
            counted = self.count_input_tokens(provider, request)
            if counted is not None and counted >= 0:
                usage.input_tokens = int(counted)

        usage_limits.check_before_request(usage)

    @staticmethod
    def _check_after_response(*, response: Any, usage_limits: Any) -> None:
        from pydantic_ai.usage import RunUsage

        usage = RunUsage(requests=1)
        usage.incr(response.usage)
        usage_limits.check_tokens(usage)


__all__ = [
    "ProviderMetadata",
    "LLMExecutionBackend",
    "LLMProviderRequest",
    "LLMInvocationRequest",
    "LLMInvocationResult",
    "LegacyProviderBackend",
    "PydanticAIGatewayBackend",
    "default_model_for_provider",
]
