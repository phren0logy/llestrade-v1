"""Unit tests for worker LLM backend contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict

import httpx
import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from src.app.workers.llm_backend import (
    DirectProviderMetadata,
    GatewayAccessCheck,
    LLMInvocationRequest,
    LLMProviderCapabilities,
    LLMProviderRequest,
    PydanticAIDirectBackend,
    PydanticAIGatewayBackend,
    ProviderMetadata,
    build_model_settings,
    extract_http_status_error_details,
    normalize_model_name,
    provider_capabilities,
    reset_gateway_access_check_cache,
    resolve_model_name,
    supported_direct_provider_ids,
    supported_gateway_provider_ids,
)


class _ProviderMeta:
    def __init__(self, name: str, default_model: str = "default-model") -> None:
        self.provider_name = name
        self.default_model = default_model


@pytest.fixture(autouse=True)
def gateway_test_event_loop() -> Any:
    reset_gateway_access_check_cache()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()
        reset_gateway_access_check_cache()


def test_direct_provider_backend_requires_direct_provider_metadata() -> None:
    backend = PydanticAIDirectBackend()

    with pytest.raises(RuntimeError, match="Direct provider backend requires DirectProviderMetadata"):
        backend.invoke_response(
            object(),
            LLMInvocationRequest(
                prompt="summarize",
                system_prompt="system",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 2048},
            ),
        )


def test_direct_provider_backend_count_input_tokens_requires_direct_provider_metadata() -> None:
    backend = PydanticAIDirectBackend()

    result = backend.count_input_tokens(
        object(),
        LLMInvocationRequest(
            prompt="summarize",
            system_prompt=None,
            model=None,
            model_settings={},
        ),
    )

    assert result is None


def test_normalize_model_name_translates_bedrock_aliases() -> None:
    assert normalize_model_name("anthropic_bedrock", "claude-sonnet-4-5") == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert normalize_model_name("anthropic", "claude-sonnet-4-5") == "claude-sonnet-4-5"


def test_resolve_model_name_uses_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = {
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-4.1",
        "gemini": "gemini-2.5-pro",
        "azure_openai": None,
        "anthropic_bedrock": None,
    }
    monkeypatch.setattr(
        "src.app.workers.llm_backend.default_model_for_provider",
        lambda provider_id, transport="direct": defaults.get(provider_id),
    )
    assert str(resolve_model_name("anthropic", None)).startswith("claude")
    assert str(resolve_model_name("openai", None)).startswith(("gpt-", "o"))
    assert str(resolve_model_name("gemini", None)).startswith("gemini")
    assert resolve_model_name("azure_openai", None) is None
    assert resolve_model_name("anthropic_bedrock", None) is None


def test_provider_capabilities_centralize_reasoning_and_preflight_support() -> None:
    assert provider_capabilities("anthropic", "claude-sonnet-4-5") == LLMProviderCapabilities(
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        reasoning_mode="anthropic",
        supports_pre_request_token_count=True,
    )
    assert provider_capabilities("gemini", "gemini-2.5-pro") == LLMProviderCapabilities(
        provider_id="gemini",
        model="gemini-2.5-pro",
        reasoning_mode="google",
        supports_pre_request_token_count=True,
    )
    assert provider_capabilities("openai", "gpt-4.1") == LLMProviderCapabilities(
        provider_id="openai",
        model="gpt-4.1",
        reasoning_mode="openai",
        supports_pre_request_token_count=False,
    )


def test_build_model_settings_adds_provider_specific_reasoning_controls() -> None:
    anthropic_settings = build_model_settings(
        "anthropic",
        "claude-sonnet-4-5",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=True,
    )
    assert anthropic_settings["anthropic_thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert anthropic_settings["temperature"] == 1.0

    bedrock_settings = build_model_settings(
        "anthropic_bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=True,
    )
    assert bedrock_settings["bedrock_additional_model_requests_fields"] == {
        "thinking": {"type": "enabled", "budget_tokens": 4000}
    }
    assert bedrock_settings["temperature"] == 1.0
    gemini_settings = build_model_settings(
        "gemini",
        "gemini-2.5-pro",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=True,
    )
    assert gemini_settings["gemini_thinking_config"] == {"include_thoughts": True}
    assert gemini_settings["temperature"] == 0.2
    openai_settings = build_model_settings(
        "openai",
        "gpt-5.1",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=True,
    )
    assert openai_settings["openai_reasoning_effort"] == "medium"
    assert openai_settings["openai_reasoning_summary"] == "detailed"
    assert "temperature" not in openai_settings


def test_build_model_settings_uses_default_temperature_for_gemini_3_reasoning() -> None:
    settings = build_model_settings(
        "gemini",
        "gemini-3-flash",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=True,
    )

    assert settings["gemini_thinking_config"] == {
        "include_thoughts": True,
        "thinking_level": "MEDIUM",
    }
    assert "temperature" not in settings


def test_build_model_settings_keeps_sampling_for_openai_models_when_supported() -> None:
    settings = build_model_settings(
        "openai",
        "gpt-5.1",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=False,
    )
    assert settings["temperature"] == 0.2


def test_build_model_settings_drops_sampling_for_always_reasoning_openai_models() -> None:
    settings = build_model_settings(
        "openai",
        "gpt-5-mini",
        temperature=0.2,
        max_tokens=32_000,
        use_reasoning=False,
    )
    assert "temperature" not in settings


def test_build_model_settings_rejects_unsupported_reasoning_provider() -> None:
    with pytest.raises(RuntimeError, match="does not support reasoning mode"):
        build_model_settings(
            "unsupported_provider",
            "custom-model",
            temperature=0.2,
            max_tokens=1024,
            use_reasoning=True,
        )


def test_direct_provider_backend_create_provider_loads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _StubSettings:
        def get_api_key(self, provider_id: str) -> str | None:
            assert provider_id == "azure_openai"
            return "azure-key"

        def get(self, key: str, default: object = None) -> object:
            if key == "azure_openai_settings":
                return {"endpoint": "https://azure.example.com", "api_version": "2025-01-01-preview"}
            return default

    class _FakeAzureProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _StubSettings)
    monkeypatch.setattr("pydantic_ai.providers.azure.AzureProvider", _FakeAzureProvider)

    backend = PydanticAIDirectBackend()
    provider = backend.create_provider(
        LLMProviderRequest(
            provider_id="azure_openai",
            model="gpt-4.1",
        )
    )

    assert isinstance(provider, DirectProviderMetadata)
    assert provider.app_provider_id == "azure_openai"
    assert provider.provider_name == "azure"
    assert provider.default_model == "gpt-4.1"
    assert captured == {
        "api_key": "azure-key",
        "azure_endpoint": "https://azure.example.com",
        "api_version": "2025-01-01-preview",
    }


def test_direct_provider_backend_create_provider_rejects_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubSettings:
        def get_api_key(self, provider_id: str) -> str | None:  # noqa: ARG002
            return None

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _StubSettings)

    backend = PydanticAIDirectBackend()

    with pytest.raises(RuntimeError, match="not supported by the worker LLM backend"):
        backend.create_provider(
            LLMProviderRequest(
                provider_id="unsupported_provider",
                model="custom-model",
            )
        )


def test_direct_provider_backend_requires_api_key_for_direct_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubSettings:
        def get_api_key(self, provider_id: str) -> str | None:
            assert provider_id == "anthropic"
            return None

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _StubSettings)

    backend = PydanticAIDirectBackend()

    with pytest.raises(RuntimeError, match="No Anthropic API key configured"):
        backend.create_provider(
            LLMProviderRequest(
                provider_id="anthropic",
                model="claude-sonnet-4-5",
            )
        )


def test_direct_provider_backend_direct_provider_uses_pydantic_ai_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        captured["call"] = {
            "model": model,
            "messages": messages,
            "model_settings": dict(model_settings),
            "model_request_parameters": model_request_parameters,
            "instrument": instrument,
        }
        return ModelResponse(
            parts=[TextPart("Direct response")],
            usage=RequestUsage(input_tokens=25, output_tokens=10),
            model_name="claude-sonnet-4-5",
            provider_name="anthropic",
            provider_url="https://api.anthropic.com",
            provider_response_id="resp-direct",
            finish_reason="stop",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)
    monkeypatch.setattr("src.app.workers.llm_backend._pydantic_ai_instrumentation", lambda: "instrumented")

    backend = PydanticAIDirectBackend()
    monkeypatch.setattr(
        backend,
        "_build_direct_model",
        lambda **kwargs: {"provider": kwargs["provider"].provider_name, "model_name": kwargs["model_name"]},
        raising=True,
    )

    provider = DirectProviderMetadata(
        app_provider_id="anthropic",
        provider_name="anthropic",
        default_model="claude-sonnet-4-5",
        provider=object(),
    )

    response = backend.invoke_response(
        provider,
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System prompt",
            model="claude-sonnet-4-5",
            model_settings={"temperature": 0.2, "max_tokens": 512},
        ),
    )

    assert isinstance(response, ModelResponse)
    assert response.text == "Direct response"
    assert response.provider_name == "anthropic"
    assert response.model_name == "claude-sonnet-4-5"
    assert response.usage.input_tokens == 25
    assert response.usage.output_tokens == 10
    assert captured["call"]["messages"][0].instructions == "System prompt"
    assert captured["call"]["messages"][0].parts[0].content == "Summarize this"
    assert captured["call"]["model_settings"] == {"temperature": 0.2, "max_tokens": 512}
    assert captured["call"]["instrument"] == "instrumented"


def test_direct_provider_backend_invoke_response_returns_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        return ModelResponse(
            parts=[TextPart("Direct response")],
            usage=RequestUsage(input_tokens=25, output_tokens=10),
            model_name="claude-sonnet-4-5",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIDirectBackend()
    monkeypatch.setattr(
        backend,
        "_build_direct_model",
        lambda **kwargs: {"provider": kwargs["provider"].provider_name, "model_name": kwargs["model_name"]},
        raising=True,
    )
    provider = DirectProviderMetadata(
        app_provider_id="anthropic",
        provider_name="anthropic",
        default_model="claude-sonnet-4-5",
        provider=object(),
    )

    response = backend.invoke_response(
        provider,
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System prompt",
            model="claude-sonnet-4-5",
            model_settings={"temperature": 0.2, "max_tokens": 512},
        ),
    )

    assert isinstance(response, ModelResponse)
    assert response.text == "Direct response"


def test_direct_provider_backend_retries_transient_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _flaky_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.RemoteProtocolError("incomplete chunked read")
        return ModelResponse(
            parts=[TextPart("Recovered response")],
            usage=RequestUsage(input_tokens=25, output_tokens=10),
            model_name="claude-sonnet-4-5",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _flaky_model_request_sync)
    monkeypatch.setattr("src.app.workers.llm_backend.time.sleep", lambda _seconds: None)

    backend = PydanticAIDirectBackend()
    monkeypatch.setattr(
        backend,
        "_build_direct_model",
        lambda **kwargs: {"provider": kwargs["provider"].provider_name, "model_name": kwargs["model_name"]},
        raising=True,
    )
    provider = DirectProviderMetadata(
        app_provider_id="anthropic",
        provider_name="anthropic",
        default_model="claude-sonnet-4-5",
        provider=object(),
    )

    response = backend.invoke_response(
        provider,
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System prompt",
            model="claude-sonnet-4-5",
            model_settings={"temperature": 0.2, "max_tokens": 512},
        ),
    )

    assert response.text == "Recovered response"
    assert calls["count"] == 2


def test_direct_provider_backend_does_not_retry_non_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _failing_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        calls["count"] += 1
        raise RuntimeError("semantic failure")

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _failing_model_request_sync)
    monkeypatch.setattr("src.app.workers.llm_backend.time.sleep", lambda _seconds: None)

    backend = PydanticAIDirectBackend()
    monkeypatch.setattr(
        backend,
        "_build_direct_model",
        lambda **kwargs: {"provider": kwargs["provider"].provider_name, "model_name": kwargs["model_name"]},
        raising=True,
    )
    provider = DirectProviderMetadata(
        app_provider_id="anthropic",
        provider_name="anthropic",
        default_model="claude-sonnet-4-5",
        provider=object(),
    )

    with pytest.raises(RuntimeError, match="semantic failure"):
        backend.invoke_response(
            provider,
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System prompt",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 512},
            ),
        )

    assert calls["count"] == 1


def test_direct_provider_backend_enforces_input_limit_before_request_when_counting_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_model_request_sync(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise AssertionError("model_request_sync should not be called when preflight limit is exceeded")

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fail_model_request_sync)

    backend = PydanticAIDirectBackend()
    provider = DirectProviderMetadata(
        app_provider_id="anthropic",
        provider_name="anthropic",
        default_model="claude-sonnet-4-5",
        provider=object(),
    )
    monkeypatch.setattr(backend, "_build_direct_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(backend, "_count_input_tokens_direct", lambda provider, request: 500, raising=True)

    with pytest.raises(Exception, match="input_tokens_limit"):
        backend.invoke_response(
            provider,
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 512},
                input_tokens_limit=400,
            ),
        )


def test_direct_provider_backend_enforces_input_limit_after_response_when_precount_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        return ModelResponse(
            parts=[TextPart("Direct response")],
            usage=RequestUsage(input_tokens=600, output_tokens=45),
            model_name="gpt-4.1",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIDirectBackend()
    provider = DirectProviderMetadata(
        app_provider_id="openai",
        provider_name="openai",
        default_model="gpt-4.1",
        provider=object(),
    )
    monkeypatch.setattr(backend, "_build_direct_model", lambda **_kwargs: object(), raising=True)

    with pytest.raises(Exception, match="Exceeded the input_tokens_limit"):
        backend.invoke_response(
            provider,
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System",
                model="gpt-4.1",
                model_settings={"temperature": 0.2, "max_tokens": 512},
                input_tokens_limit=400,
            ),
        )


def test_gateway_backend_create_provider_returns_metadata() -> None:
    backend = PydanticAIGatewayBackend(api_key="gateway-key")

    provider = backend.create_provider(
        LLMProviderRequest(
            provider_id="anthropic",
            model="claude-sonnet-4-5",
        )
    )

    assert provider == ProviderMetadata(
        provider_name="anthropic",
        default_model="claude-sonnet-4-5",
        transport="gateway",
    )


def test_gateway_backend_create_provider_rejects_unsupported_provider() -> None:
    backend = PydanticAIGatewayBackend(api_key="gateway-key")

    with pytest.raises(RuntimeError, match="not supported by the Pydantic AI Gateway backend"):
        backend.create_provider(
            LLMProviderRequest(
                provider_id="unsupported_provider",
                model="custom-model",
            )
        )


def test_gateway_backend_invoke_response_returns_model_response_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        captured["call"] = {
            "model": model,
            "messages": messages,
            "model_settings": dict(model_settings),
            "model_request_parameters": model_request_parameters,
            "instrument": instrument,
        }
        return ModelResponse(
            parts=[TextPart("Gateway response")],
            usage=RequestUsage(input_tokens=120, output_tokens=45, details={"cached": 0}),
            model_name="claude-sonnet-4-5",
            provider_name="gateway/anthropic",
            provider_url="https://gateway.example.com/anthropic",
            provider_response_id="resp-123",
            finish_reason="stop",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)
    monkeypatch.setattr("src.app.workers.llm_backend._pydantic_ai_instrumentation", lambda: "instrumented")
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")
    monkeypatch.setattr(
        backend,
        "_build_model",
        lambda **kwargs: {"provider_id": kwargs["provider_id"], "model_name": kwargs["model_name"]},
        raising=True,
    )

    response = backend.invoke_response(
        _ProviderMeta("anthropic"),
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System prompt",
            model="claude-sonnet-4-5",
            model_settings={
                "temperature": 0.2,
                "max_tokens": 4096,
                "reasoning_effort": "medium",
            },
        ),
    )

    assert isinstance(response, ModelResponse)
    assert response.text == "Gateway response"
    assert response.provider_name == "gateway/anthropic"
    assert response.model_name == "claude-sonnet-4-5"
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 45
    assert response.provider_response_id == "resp-123"
    assert response.finish_reason == "stop"
    assert response.provider_url == "https://gateway.example.com/anthropic"
    assert captured["call"]["messages"][0].instructions == "System prompt"
    assert captured["call"]["messages"][0].parts[0].content == "Summarize this"
    assert captured["call"]["model_settings"]["temperature"] == 0.2
    assert captured["call"]["model_settings"]["max_tokens"] == 4096
    assert captured["call"]["model_settings"]["reasoning_effort"] == "medium"
    assert captured["call"]["instrument"] == "instrumented"


def test_gateway_backend_rejects_unsupported_provider() -> None:
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")

    with pytest.raises(RuntimeError, match="not supported by Pydantic AI Gateway backend"):
        backend.invoke_response(
            ProviderMetadata(provider_name="unsupported_provider", default_model="legacy-model"),
            LLMInvocationRequest(
                prompt="test",
                system_prompt="sys",
                model=None,
                model_settings={"temperature": 0.1, "max_tokens": 128},
            ),
        )


def test_supported_provider_lists_are_explicit() -> None:
    assert supported_direct_provider_ids() == (
        "anthropic",
        "anthropic_bedrock",
        "azure_openai",
        "gemini",
        "openai",
    )
    assert supported_gateway_provider_ids() == (
        "anthropic",
        "anthropic_bedrock",
        "azure_openai",
        "gemini",
        "openai",
    )


def test_direct_backend_capabilities_delegate_to_shared_capability_map() -> None:
    backend = PydanticAIDirectBackend()

    assert backend.capabilities("anthropic_bedrock", "claude-sonnet-4-5").reasoning_mode == "anthropic"
    assert backend.capabilities("anthropic_bedrock", "claude-sonnet-4-5").supports_pre_request_token_count is True
    assert backend.build_model_settings(
        "gemini",
        "gemini-2.5-pro",
        temperature=0.2,
        max_tokens=1024,
        use_reasoning=True,
    )["gemini_thinking_config"] == {"include_thoughts": True}


def test_gateway_backend_capabilities_delegate_to_shared_capability_map() -> None:
    backend = PydanticAIGatewayBackend(api_key="gateway-key")

    assert backend.capabilities("openai", "gpt-5.1").reasoning_mode == "openai"
    assert backend.capabilities("openai", "gpt-5.1").supports_pre_request_token_count is False
    settings = backend.build_model_settings(
        "openai",
        "gpt-5.1",
        temperature=0.2,
        max_tokens=1024,
        use_reasoning=True,
    )
    assert settings["openai_reasoning_effort"] == "medium"


def test_gateway_backend_rejects_unsupported_provider_metadata() -> None:
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")

    with pytest.raises(RuntimeError, match="not supported by Pydantic AI Gateway backend"):
        backend.invoke_response(
            ProviderMetadata(provider_name="unsupported_provider", default_model="custom-model"),
            LLMInvocationRequest(
                prompt="test",
                system_prompt="sys",
                model=None,
                model_settings={"temperature": 0.1, "max_tokens": 128},
            ),
        )


def test_gateway_backend_reports_configuration_error_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("PAIG_API_KEY", raising=False)
    backend = PydanticAIGatewayBackend(api_key=None, base_url=None)

    with pytest.raises(Exception):
        backend.invoke_response(
            _ProviderMeta("anthropic", default_model="claude-sonnet-4-5"),
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 512},
            ),
        )


def test_gateway_backend_count_input_tokens_uses_model_token_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeModel:
        async def count_tokens(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
            captured["messages"] = messages
            captured["model_settings"] = model_settings
            captured["model_request_parameters"] = model_request_parameters
            return SimpleNamespace(input_tokens=321)

    backend = PydanticAIGatewayBackend(api_key="pylf_test_key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: _FakeModel(), raising=True)

    token_count = backend.count_input_tokens(
        _ProviderMeta("anthropic", default_model="claude-sonnet-4-5"),
        LLMInvocationRequest(
            prompt="User prompt",
            system_prompt="System prompt",
            model="claude-sonnet-4-5",
            model_settings={
                "temperature": 0.2,
                "max_tokens": 2048,
                "reasoning_effort": "medium",
            },
        ),
    )

    assert token_count == 321
    assert len(captured["messages"]) == 1
    assert captured["messages"][0].instructions == "System prompt"
    parts = captured["messages"][0].parts
    assert [type(part).__name__ for part in parts] == ["UserPromptPart"]
    assert parts[0].content == "User prompt"
    assert captured["model_settings"]["temperature"] == 0.2
    assert captured["model_settings"]["max_tokens"] == 2048
    assert captured["model_settings"]["reasoning_effort"] == "medium"


def test_gateway_backend_count_input_tokens_returns_none_when_model_does_not_support_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeModel:
        async def count_tokens(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:  # noqa: ARG002
            raise NotImplementedError("Token counting ahead of the request is not supported")

    backend = PydanticAIGatewayBackend(api_key="pylf_test_key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: _FakeModel(), raising=True)

    token_count = backend.count_input_tokens(
        _ProviderMeta("openai", default_model="gpt-4.1"),
        LLMInvocationRequest(
            prompt="User prompt",
            system_prompt="System prompt",
            model="gpt-4.1",
            model_settings={"temperature": 0.2, "max_tokens": 2048},
        ),
    )

    assert token_count is None


def test_gateway_backend_loads_base_url_from_secure_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("PAIG_BASE_URL", raising=False)

    class _FakeSecureSettings:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "pydantic_ai_gateway_settings":
                return {"base_url": "https://gateway.example.com"}
            return default

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _FakeSecureSettings)

    backend = PydanticAIGatewayBackend(api_key="pylf_test_key", base_url=None)

    assert backend._current_base_url() == "https://gateway.example.com"


def test_gateway_backend_loads_route_from_secure_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_ROUTE", raising=False)

    class _FakeSecureSettings:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "pydantic_ai_gateway_settings":
                return {"route": "llestrade"}
            return default

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _FakeSecureSettings)

    backend = PydanticAIGatewayBackend(api_key="pylf_test_key", base_url="https://gateway.example.com")

    assert backend._current_route() == "llestrade"


def test_gateway_backend_loads_api_key_from_secure_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("PAIG_API_KEY", raising=False)

    class _FakeSecureSettings:
        def get_api_key(self, provider: str) -> Any:
            if provider == "pydantic_ai_gateway":
                return "gateway-key-from-settings"
            return None

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _FakeSecureSettings)

    backend = PydanticAIGatewayBackend(api_key=None, base_url="https://gateway.example.com")

    assert backend._current_api_key() == "gateway-key-from-settings"


def test_gateway_backend_verify_gateway_access_classifies_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise ModelHTTPError(
            401,
            "gemini-2.5-flash-lite",
            {"message": "Unauthorized - Key not found", "status": "Unauthorized"},
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(backend, "_build_gateway_http_client", lambda **_kwargs: None, raising=True)

    result = backend.verify_gateway_access("gemini", "gemini-2.5-flash-lite", force=True)

    assert result.ok is False
    assert result.kind == "auth_invalid"
    assert result.status_code == 401
    assert result.message == "Unauthorized - Key not found"


def test_gateway_backend_verify_gateway_access_classifies_forbidden_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise ModelHTTPError(
            403,
            "claude-sonnet-4-5",
            {"message": "Unauthorized - Key disabled", "status": "Forbidden"},
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(backend, "_build_gateway_http_client", lambda **_kwargs: None, raising=True)

    result = backend.verify_gateway_access("anthropic", "claude-sonnet-4-5", force=True)

    assert result.ok is False
    assert result.kind == "auth_forbidden"
    assert result.status_code == 403


def test_gateway_backend_verify_gateway_access_classifies_route_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise ModelHTTPError(
            404,
            "claude-sonnet-4-5",
            {"message": "Route not found: bulk", "status": "Not Found"},
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIGatewayBackend(
        api_key="gateway-key",
        base_url="https://gateway.example.com",
        route="bulk",
    )
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(backend, "_build_gateway_http_client", lambda **_kwargs: None, raising=True)

    result = backend.verify_gateway_access("anthropic", "claude-sonnet-4-5", force=True)

    assert result.ok is False
    assert result.kind == "route_missing"
    assert result.status_code == 404
    assert result.route == "bulk"


def test_gateway_backend_verify_gateway_access_classifies_hidden_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://gateway.example.com/anthropic/v1/messages?beta=true")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "7"},
        json={"message": "Provider capacity reached for anthropic. Retry soon."},
    )
    status_error = httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)

    def _fake_model_request_sync(*_args, **_kwargs):  # noqa: ANN002, ANN003
        error = ModelAPIError("claude-sonnet-4-6", "Connection error.")
        error.__cause__ = status_error
        raise error

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(backend, "_build_gateway_http_client", lambda **_kwargs: None, raising=True)

    result = backend.verify_gateway_access("anthropic", "claude-sonnet-4-6", force=True)

    assert result.ok is False
    assert result.kind == "rate_limited"
    assert result.status_code == 429
    assert result.retry_after_seconds == 7.0
    assert "Provider capacity reached for anthropic" in result.message


def test_gateway_backend_verify_gateway_access_uses_env_over_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeSecureSettings:
        def get_api_key(self, provider: str) -> str | None:
            assert provider == "pydantic_ai_gateway"
            return "settings-key"

        def get(self, key: str, default: object = None) -> object:
            if key == "pydantic_ai_gateway_settings":
                return {"base_url": "https://settings.example.com", "route": "settings-route"}
            return default

    def _fake_probe(*, provider_id: str, model_name: str, api_key: str, base_url: str | None, route: str | None, timeout_seconds: float):  # noqa: ANN001
        captured.update(
            provider_id=provider_id,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            route=route,
            timeout_seconds=timeout_seconds,
        )
        return GatewayAccessCheck(
            ok=True,
            kind="ok",
            status_code=200,
            message="ok",
            base_url=base_url,
            route=route,
            provider_id=provider_id,
            model=model_name,
        )

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _FakeSecureSettings)
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "env-key")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_ROUTE", "env-route")

    backend = PydanticAIGatewayBackend(api_key=None, base_url=None, route=None)
    monkeypatch.setattr(backend, "_probe_gateway_access", _fake_probe, raising=True)

    result = backend.verify_gateway_access("anthropic", "claude-sonnet-4-5", force=True)

    assert result.ok is True
    assert captured == {
        "provider_id": "anthropic",
        "model_name": "claude-sonnet-4-5",
        "api_key": "env-key",
        "base_url": "https://env.example.com",
        "route": "env-route",
        "timeout_seconds": 5.0,
    }


def test_gateway_backend_verify_gateway_access_caches_successes_and_auth_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _fake_probe(**kwargs):  # noqa: ANN003
        calls["count"] += 1
        return GatewayAccessCheck(
            ok=False,
            kind="auth_invalid",
            status_code=401,
            message="Unauthorized - Key not found",
            base_url=kwargs["base_url"],
            route=kwargs["route"],
            provider_id=kwargs["provider_id"],
            model=kwargs["model_name"],
        )

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_probe_gateway_access", _fake_probe, raising=True)

    first = backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")
    second = backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")

    assert first.kind == "auth_invalid"
    assert second.kind == "auth_invalid"
    assert calls["count"] == 1


def test_gateway_backend_verify_gateway_access_cache_reset_forces_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _fake_probe(**kwargs):  # noqa: ANN003
        calls["count"] += 1
        return GatewayAccessCheck(
            ok=True,
            kind="ok",
            status_code=200,
            message="ok",
            base_url=kwargs["base_url"],
            route=kwargs["route"],
            provider_id=kwargs["provider_id"],
            model=kwargs["model_name"],
        )

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_probe_gateway_access", _fake_probe, raising=True)

    backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")
    reset_gateway_access_check_cache()
    backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")

    assert calls["count"] == 2


def test_gateway_backend_verify_gateway_access_does_not_cache_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _fake_probe(**kwargs):  # noqa: ANN003
        calls["count"] += 1
        return GatewayAccessCheck(
            ok=False,
            kind="unreachable",
            status_code=None,
            message="connect failed",
            base_url=kwargs["base_url"],
            route=kwargs["route"],
            provider_id=kwargs["provider_id"],
            model=kwargs["model_name"],
        )

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_probe_gateway_access", _fake_probe, raising=True)

    backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")
    backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")

    assert calls["count"] == 2


def test_gateway_backend_verify_gateway_access_does_not_cache_rate_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _fake_probe(**kwargs):  # noqa: ANN003
        calls["count"] += 1
        return GatewayAccessCheck(
            ok=False,
            kind="rate_limited",
            status_code=429,
            message="Provider capacity reached for anthropic. Retry soon.",
            base_url=kwargs["base_url"],
            route=kwargs["route"],
            provider_id=kwargs["provider_id"],
            model=kwargs["model_name"],
            retry_after_seconds=5.0,
        )

    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_probe_gateway_access", _fake_probe, raising=True)

    backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")
    backend.verify_gateway_access("anthropic", "claude-sonnet-4-5")

    assert calls["count"] == 2


def test_gateway_backend_skips_pre_request_token_counting_in_sync_invoke_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        return ModelResponse(
            parts=[TextPart("Gateway response")],
            usage=RequestUsage(input_tokens=350, output_tokens=45),
            model_name="claude-sonnet-4-5",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIGatewayBackend(api_key="pylf_test_key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(
        backend,
        "count_input_tokens",
        lambda provider, request: (_ for _ in ()).throw(AssertionError("gateway preflight should be skipped")),
        raising=True,
    )

    response = backend.invoke_response(
        _ProviderMeta("anthropic", default_model="claude-sonnet-4-5"),
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System",
            model="claude-sonnet-4-5",
            model_settings={"temperature": 0.2, "max_tokens": 512},
            input_tokens_limit=400,
        ),
    )

    assert response.text == "Gateway response"


def test_gateway_backend_enforces_input_limit_after_response_when_sync_gateway_precount_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        return ModelResponse(
            parts=[TextPart("Gateway response")],
            usage=RequestUsage(input_tokens=600, output_tokens=45),
            model_name="gpt-4.1",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)

    backend = PydanticAIGatewayBackend(api_key="pylf_test_key", base_url="https://gateway.example.com")
    monkeypatch.setattr(backend, "_build_model", lambda **_kwargs: object(), raising=True)
    monkeypatch.setattr(
        backend,
        "count_input_tokens",
        lambda provider, request: (_ for _ in ()).throw(AssertionError("gateway preflight should be skipped")),
        raising=True,
    )

    with pytest.raises(Exception, match="Exceeded the input_tokens_limit"):
        backend.invoke_response(
            _ProviderMeta("anthropic", default_model="claude-sonnet-4-5"),
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 512},
                input_tokens_limit=400,
            ),
        )


def test_gateway_backend_reports_configuration_error_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("PAIG_API_KEY", raising=False)
    backend = PydanticAIGatewayBackend(api_key=None, base_url=None)

    with pytest.raises(Exception):
        backend.invoke_response(
            ProviderMetadata(provider_name="anthropic", default_model="claude-sonnet-4-5"),
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 512},
            ),
        )


def test_gateway_backend_reports_empty_output_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        return ModelResponse(
            parts=[TextPart("   ")],
            usage=RequestUsage(input_tokens=20, output_tokens=0),
            model_name="claude-sonnet-4-5",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _fake_model_request_sync)
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")
    monkeypatch.setattr(
        backend,
        "_build_model",
        lambda **kwargs: {"provider_id": kwargs["provider_id"], "model_name": kwargs["model_name"]},
        raising=True,
    )

    with pytest.raises(RuntimeError, match="LLM returned empty response"):
        backend.invoke_response(
            _ProviderMeta("anthropic"),
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System prompt",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 4096},
            ),
        )


def test_gateway_backend_retries_transient_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _flaky_model_request_sync(
        model: Any,
        messages: Any,
        *,
        model_settings: Dict[str, Any],
        model_request_parameters: Any = None,
        instrument: Any = None,
    ) -> ModelResponse:
        _ = model, messages, model_settings, model_request_parameters, instrument
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.RemoteProtocolError("incomplete chunked read")
        return ModelResponse(
            parts=[TextPart("Gateway recovered")],
            usage=RequestUsage(input_tokens=20, output_tokens=10),
            model_name="claude-sonnet-4-6",
        )

    monkeypatch.setattr("pydantic_ai.direct.model_request_sync", _flaky_model_request_sync)
    monkeypatch.setattr("src.app.workers.llm_backend.time.sleep", lambda _seconds: None)
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")
    monkeypatch.setattr(
        backend,
        "_build_model",
        lambda **kwargs: {"provider_id": kwargs["provider_id"], "model_name": kwargs["model_name"]},
        raising=True,
    )

    response = backend.invoke_response(
        _ProviderMeta("anthropic"),
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System prompt",
            model="claude-sonnet-4-6",
            model_settings={"temperature": 0.2, "max_tokens": 4096},
        ),
    )

    assert response.text == "Gateway recovered"
    assert calls["count"] == 2


def test_gateway_backend_build_model_uses_canonical_gateway_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_infer_model(model: str, provider_factory: Any) -> Any:
        captured["model"] = model
        captured["provider"] = provider_factory("gateway/gemini")
        return {"model": model}

    def _fake_limit_model_concurrency(model: Any, limiter: Any) -> Any:
        captured["limited_model"] = model
        captured["limiter"] = limiter
        return {"limited": model}

    def _fake_gateway_provider(
        upstream_provider: str,
        /,
        *,
        route: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> dict[str, Any]:
        return {
            "upstream_provider": upstream_provider,
            "route": route,
            "api_key": api_key,
            "base_url": base_url,
            "http_client": http_client,
        }

    monkeypatch.setattr("pydantic_ai.models.infer_model", _fake_infer_model)
    monkeypatch.setattr(
        "pydantic_ai.models.concurrency.limit_model_concurrency",
        _fake_limit_model_concurrency,
    )
    monkeypatch.setattr("pydantic_ai.providers.gateway.gateway_provider", _fake_gateway_provider)

    backend = PydanticAIGatewayBackend(
        api_key="gateway-key",
        base_url="https://gateway.example.com",
        route="llestrade",
        max_concurrency=2,
    )
    model = backend._build_model(provider_id="gemini", model_name="gemini-2.5-pro")

    assert model == {"limited": {"model": "gateway/gemini:gemini-2.5-pro"}}
    assert captured["model"] == "gateway/gemini:gemini-2.5-pro"
    assert captured["limited_model"] == {"model": "gateway/gemini:gemini-2.5-pro"}
    assert captured["limiter"].max_running == 2
    assert captured["provider"] == {
        "upstream_provider": "gemini",
        "route": "llestrade",
        "api_key": "gateway-key",
        "base_url": "https://gateway.example.com",
        "http_client": backend._gateway_http_client(),
    }
    assert captured["provider"]["http_client"] is not None


def test_gateway_backend_uses_latest_settings_values_when_not_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class _StubSettings:
        api_key = "gateway-key-1"
        base_url = "https://gateway-1.example.com"
        route = "route-1"

        def get_api_key(self, provider_id: str) -> str | None:
            assert provider_id == "pydantic_ai_gateway"
            return self.api_key

        def get(self, key: str, default: object = None) -> object:
            if key == "pydantic_ai_gateway_settings":
                return {
                    "base_url": self.base_url,
                    "route": self.route,
                }
            return default

    def _fake_gateway_provider(
        upstream_provider: str,
        /,
        *,
        route: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> dict[str, Any]:
        value = {
            "upstream_provider": upstream_provider,
            "route": route,
            "api_key": api_key,
            "base_url": base_url,
            "http_client": http_client,
        }
        captured.append(value)
        return value

    monkeypatch.setattr("src.app.core.secure_settings.SecureSettings", _StubSettings)
    monkeypatch.setattr("pydantic_ai.providers.gateway.gateway_provider", _fake_gateway_provider)

    backend = PydanticAIGatewayBackend(api_key=None, base_url=None, route=None)

    first = backend._gateway_provider_factory("gateway/anthropic")
    _StubSettings.api_key = "gateway-key-2"
    _StubSettings.base_url = "https://gateway-2.example.com"
    _StubSettings.route = "route-2"
    second = backend._gateway_provider_factory("gateway/anthropic")

    assert first["api_key"] == "gateway-key-1"
    assert first["base_url"] == "https://gateway-1.example.com"
    assert first["route"] == "route-1"
    assert second["api_key"] == "gateway-key-2"
    assert second["base_url"] == "https://gateway-2.example.com"
    assert second["route"] == "route-2"
    assert len(captured) == 2


@pytest.mark.parametrize("status_code", [429, 524])
def test_gateway_backend_only_marks_transient_gateway_statuses_for_retry(status_code: int) -> None:
    retryable = httpx.Response(status_code, request=httpx.Request("POST", "https://gateway.example.com"))
    with pytest.raises(httpx.HTTPStatusError):
        PydanticAIGatewayBackend._raise_for_retryable_gateway_response(retryable)

    non_retryable = httpx.Response(400, request=httpx.Request("POST", "https://gateway.example.com"))
    PydanticAIGatewayBackend._raise_for_retryable_gateway_response(non_retryable)


def test_extract_http_status_error_details_from_httpx_status_error() -> None:
    request = httpx.Request("POST", "https://gateway.example.com/anthropic/v1/messages?beta=true")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "12"},
        json={"error": {"message": "Too Many Requests"}},
    )
    error = httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)

    details = extract_http_status_error_details(error)

    assert details is not None
    assert details.status_code == 429
    assert details.retry_after_seconds == 12.0
    assert "Too Many Requests" in details.message


def test_extract_http_status_error_details_walks_exception_chain() -> None:
    request = httpx.Request("POST", "https://gateway.example.com/anthropic/v1/messages?beta=true")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "3"},
        json={"error": {"message": "Rate limited upstream"}},
    )
    status_error = httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)
    error = ModelAPIError("claude-sonnet-4-6", "Connection error.")
    error.__cause__ = status_error

    details = extract_http_status_error_details(error)

    assert details is not None
    assert details.status_code == 429
    assert details.retry_after_seconds == 3.0
    assert "Rate limited upstream" in details.message


def test_gateway_backend_can_disable_concurrency_limit() -> None:
    backend = PydanticAIGatewayBackend(api_key="gateway-key", max_concurrency=0)
    assert backend._gateway_concurrency_limiter() is None


def test_gateway_backend_returns_error_when_provider_metadata_is_missing() -> None:
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")

    with pytest.raises(RuntimeError, match="Unable to resolve provider ID for Gateway backend"):
        backend.invoke_response(
            object(),
            LLMInvocationRequest(
                prompt="Summarize this",
                system_prompt="System prompt",
                model="claude-sonnet-4-5",
                model_settings={"temperature": 0.2, "max_tokens": 4096},
            ),
        )
