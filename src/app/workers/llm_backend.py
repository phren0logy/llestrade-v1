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

    def count_input_tokens(self, provider: Any, request: LLMInvocationRequest) -> int | None:
        """Return input token count when the backend can measure it."""

    def invoke(self, provider: Any, request: LLMInvocationRequest) -> LLMInvocationResult:
        """Execute `request` against the given provider."""


class LegacyProviderBackend:
    """Default backend that forwards directly to existing provider.generate calls."""

    def requires_native_provider(self) -> bool:
        return True

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

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        route: str | None = None,
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
        self._fallback_backend = fallback_backend
        self._fallback_on_error = fallback_on_error

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
            from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
            from pydantic_ai.models import ModelRequestParameters

            parts: list[Any] = []
            if request.system_prompt:
                parts.append(SystemPromptPart(content=request.system_prompt))
            if request.prompt:
                parts.append(UserPromptPart(content=request.prompt))

            model_settings: Dict[str, Any] = {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            for key in self._FORWARDED_SETTINGS:
                if key in request.extra:
                    model_settings[key] = request.extra[key]

            usage = asyncio.run(
                model.count_tokens(
                    [ModelRequest(parts=parts)],
                    model_settings,
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
            from pydantic_ai import Agent

            model_settings: Dict[str, Any] = {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            for key in self._FORWARDED_SETTINGS:
                if key in request.extra:
                    model_settings[key] = request.extra[key]

            result = Agent(
                model=model,
                system_prompt=request.system_prompt or "",
                retries=1,
            ).run_sync(
                request.prompt,
                model_settings=model_settings,
            )

            content = str(result.output or "").strip()
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

            usage_obj = result.usage()
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

            run_id = getattr(result, "run_id", None)
            raw: Dict[str, Any] = {}
            if run_id:
                raw["run_id"] = str(run_id)

            return LLMInvocationResult(
                success=True,
                content=content,
                error=None,
                usage=usage,
                provider=f"gateway/{provider_id}",
                model=model_name,
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
        from pydantic_ai.providers.gateway import gateway_provider

        if provider_id == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel

            gateway = gateway_provider(
                "anthropic", api_key=self._api_key, base_url=self._base_url, route=self._route
            )
            return AnthropicModel(model_name, provider=gateway)

        if provider_id == "anthropic_bedrock":
            from pydantic_ai.models.bedrock import BedrockConverseModel

            gateway = gateway_provider(
                "bedrock", api_key=self._api_key, base_url=self._base_url, route=self._route
            )
            return BedrockConverseModel(model_name, provider=gateway)

        if provider_id == "gemini":
            from pydantic_ai.models.google import GoogleModel

            gateway = gateway_provider(
                "google-vertex", api_key=self._api_key, base_url=self._base_url, route=self._route
            )
            return GoogleModel(model_name, provider=gateway)

        if provider_id in {"openai", "azure_openai"}:
            from pydantic_ai.models.openai import OpenAIChatModel

            gateway = gateway_provider(
                "openai", api_key=self._api_key, base_url=self._base_url, route=self._route
            )
            return OpenAIChatModel(model_name, provider=gateway)

        raise RuntimeError(f"Provider '{provider_id}' is not supported by Pydantic AI Gateway backend")

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


__all__ = [
    "ProviderMetadata",
    "LLMExecutionBackend",
    "LLMInvocationRequest",
    "LLMInvocationResult",
    "LegacyProviderBackend",
    "PydanticAIGatewayBackend",
    "default_model_for_provider",
]
