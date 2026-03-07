from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.common.llm.providers.anthropic import AnthropicProvider


class _FakeMessagesAPI:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=123)


class _FakeAnthropicClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.messages = _FakeMessagesAPI(response=response, error=error)


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, response=None, error: Exception | None = None):
    class FakeAuthenticationError(Exception):
        pass

    class FakeRateLimitError(Exception):
        pass

    fake_module = SimpleNamespace(
        Anthropic=lambda **kwargs: _FakeAnthropicClient(response=response, error=error),
        AuthenticationError=FakeAuthenticationError,
        RateLimitError=FakeRateLimitError,
        BadRequestError=Exception,
        NotFoundError=Exception,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return fake_module


def test_generate_success_extracts_content(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello world")],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    _install_fake_anthropic(monkeypatch, response=response)

    provider = AnthropicProvider(api_key="test-key")
    result = provider.generate("hi")

    assert provider.initialized is True
    assert result["success"] is True
    assert result["content"] == "hello world"
    assert result["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


def test_generate_maps_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAuthenticationError(Exception):
        pass

    class FakeRateLimitError(Exception):
        pass

    fake_module = SimpleNamespace(
        Anthropic=lambda **kwargs: _FakeAnthropicClient(error=FakeRateLimitError("slow down")),
        AuthenticationError=FakeAuthenticationError,
        RateLimitError=FakeRateLimitError,
        BadRequestError=Exception,
        NotFoundError=Exception,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    provider = AnthropicProvider(api_key="test-key")
    result = provider.generate("hi")

    assert result["success"] is False
    assert "Rate limit exceeded" in result["error"]


def test_count_tokens_uses_provider_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch, response=SimpleNamespace(content=[], usage=None))

    provider = AnthropicProvider(api_key="test-key")
    result = provider.count_tokens(text="hello")

    assert result["success"] is True
    assert result["token_count"] == 123
