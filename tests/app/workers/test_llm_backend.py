"""Unit tests for worker LLM backend contracts."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.app.workers.llm_backend import (
    LLMInvocationRequest,
    LLMInvocationResult,
    LegacyProviderBackend,
    PydanticAIGatewayBackend,
)


class _StubProvider:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[Dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self.response


class _ProviderMeta:
    def __init__(self, name: str, default_model: str = "default-model") -> None:
        self.provider_name = name
        self.default_model = default_model


def test_legacy_provider_backend_invokes_provider_and_normalizes_result() -> None:
    provider = _StubProvider(
        {
            "success": True,
            "content": "hello",
            "usage": {"output_tokens": 42},
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        }
    )
    backend = LegacyProviderBackend()

    result = backend.invoke(
        provider,
        LLMInvocationRequest(
            prompt="summarize",
            system_prompt="system",
            model="claude-sonnet-4-5",
            temperature=0.2,
            max_tokens=2048,
            extra={"reasoning_effort": "medium"},
        ),
    )

    assert provider.calls == [
        {
            "prompt": "summarize",
            "system_prompt": "system",
            "model": "claude-sonnet-4-5",
            "temperature": 0.2,
            "max_tokens": 2048,
            "reasoning_effort": "medium",
        }
    ]
    assert result.success is True
    assert result.content == "hello"
    assert result.error is None
    assert result.usage == {"output_tokens": 42}
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-4-5"


def test_legacy_provider_backend_handles_non_dict_provider_response() -> None:
    provider = _StubProvider("unexpected")
    backend = LegacyProviderBackend()

    result = backend.invoke(
        provider,
        LLMInvocationRequest(
            prompt="summarize",
            system_prompt=None,
            model=None,
            temperature=0.1,
            max_tokens=256,
        ),
    )

    assert result.success is False
    assert result.content == ""
    assert result.error == "Provider returned non-dict response"


def test_invocation_result_normalizes_non_dict_usage() -> None:
    result = LLMInvocationResult.from_provider_response(
        {
            "success": True,
            "content": "ok",
            "usage": "invalid",
            "error": None,
        }
    )

    assert result.success is True
    assert result.content == "ok"
    assert result.usage == {}


def test_gateway_backend_success_normalizes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Usage:
        requests = 1
        input_tokens = 120
        output_tokens = 45
        details = {"cached": 0}

    class _RunResult:
        output = "Gateway response"
        run_id = "run-123"

        @staticmethod
        def usage() -> _Usage:
            return _Usage()

    captured: dict[str, Any] = {}

    class _FakeAgent:
        def __init__(self, *, model: Any, system_prompt: str, retries: int) -> None:
            captured["init"] = {
                "model": model,
                "system_prompt": system_prompt,
                "retries": retries,
            }

        def run_sync(self, prompt: str, *, model_settings: Dict[str, Any]) -> _RunResult:
            captured["run"] = {
                "prompt": prompt,
                "model_settings": dict(model_settings),
            }
            return _RunResult()

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai, "Agent", _FakeAgent, raising=True)
    backend = PydanticAIGatewayBackend(api_key="pylf_test_key")
    monkeypatch.setattr(
        backend,
        "_build_model",
        lambda **kwargs: {"provider_id": kwargs["provider_id"], "model_name": kwargs["model_name"]},
        raising=True,
    )

    result = backend.invoke(
        _ProviderMeta("anthropic"),
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System prompt",
            model="claude-sonnet-4-5",
            temperature=0.2,
            max_tokens=4096,
            extra={"reasoning_effort": "medium"},
        ),
    )

    assert result.success is True
    assert result.content == "Gateway response"
    assert result.error is None
    assert result.provider == "gateway/anthropic"
    assert result.model == "claude-sonnet-4-5"
    assert result.usage["input_tokens"] == 120
    assert result.usage["output_tokens"] == 45
    assert result.usage["total_tokens"] == 165
    assert result.raw == {"run_id": "run-123"}
    assert captured["init"]["system_prompt"] == "System prompt"
    assert captured["run"]["prompt"] == "Summarize this"
    assert captured["run"]["model_settings"]["temperature"] == 0.2
    assert captured["run"]["model_settings"]["max_tokens"] == 4096
    assert captured["run"]["model_settings"]["reasoning_effort"] == "medium"


def test_gateway_backend_uses_fallback_for_unsupported_provider() -> None:
    provider = _StubProvider(
        {
            "success": True,
            "content": "fallback content",
            "usage": {"output_tokens": 7},
            "provider": "legacy",
            "model": "legacy-model",
        }
    )
    provider.provider_name = "unsupported_provider"
    provider.default_model = "legacy-model"

    backend = PydanticAIGatewayBackend(
        api_key="pylf_test_key",
        fallback_backend=LegacyProviderBackend(),
    )

    result = backend.invoke(
        provider,
        LLMInvocationRequest(
            prompt="test",
            system_prompt="sys",
            model=None,
            temperature=0.1,
            max_tokens=128,
        ),
    )

    assert result.success is True
    assert result.content == "fallback content"
    assert result.provider == "legacy"


def test_gateway_backend_reports_configuration_error_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("PAIG_API_KEY", raising=False)
    backend = PydanticAIGatewayBackend(api_key=None, base_url=None)

    result = backend.invoke(
        _ProviderMeta("anthropic", default_model="claude-sonnet-4-5"),
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System",
            model="claude-sonnet-4-5",
            temperature=0.2,
            max_tokens=512,
        ),
    )

    assert result.success is False
    assert result.error is not None
    assert "Gateway invocation failed" in result.error


def test_gateway_backend_can_fallback_on_error_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("PAIG_API_KEY", raising=False)

    provider = _StubProvider(
        {
            "success": True,
            "content": "legacy fallback",
            "usage": {"output_tokens": 9},
            "provider": "legacy",
            "model": "legacy-model",
        }
    )
    provider.provider_name = "anthropic"
    provider.default_model = "claude-sonnet-4-5"

    backend = PydanticAIGatewayBackend(
        api_key=None,
        base_url=None,
        fallback_backend=LegacyProviderBackend(),
        fallback_on_error=True,
    )

    result = backend.invoke(
        provider,
        LLMInvocationRequest(
            prompt="Summarize this",
            system_prompt="System",
            model="claude-sonnet-4-5",
            temperature=0.2,
            max_tokens=512,
        ),
    )

    assert result.success is True
    assert result.content == "legacy fallback"
    assert result.provider == "legacy"
