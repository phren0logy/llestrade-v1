from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.llm.providers.azure_openai import AzureOpenAIProvider


class _FakeCompletionsAPI:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class _FakeAzureClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.chat = SimpleNamespace(completions=_FakeCompletionsAPI(response=response, error=error))


@pytest.fixture
def _azure_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-01-01-preview")


def test_generate_success_parses_text_and_usage(monkeypatch: pytest.MonkeyPatch, _azure_env) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=[{"type": "text", "text": "hello"}]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr(
        "src.common.llm.providers.azure_openai.openai.AzureOpenAI",
        lambda **kwargs: _FakeAzureClient(response=completion),
    )

    provider = AzureOpenAIProvider(default_system_prompt="system")

    result = provider.generate("hi")

    assert provider.initialized is True
    assert result["success"] is True
    assert result["content"] == "hello"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_generate_maps_not_found_errors(monkeypatch: pytest.MonkeyPatch, _azure_env) -> None:
    class FakeNotFoundError(Exception):
        pass

    monkeypatch.setattr("src.common.llm.providers.azure_openai.openai.NotFoundError", FakeNotFoundError)
    monkeypatch.setattr(
        "src.common.llm.providers.azure_openai.openai.AzureOpenAI",
        lambda **kwargs: _FakeAzureClient(error=FakeNotFoundError("missing deployment")),
    )

    provider = AzureOpenAIProvider()
    result = provider.generate("hi", model="my-deployment")

    assert result["success"] is False
    assert "Deployment not found" in result["error"]


def test_count_tokens_requires_input(_azure_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.common.llm.providers.azure_openai.openai.AzureOpenAI",
        lambda **kwargs: _FakeAzureClient(response=SimpleNamespace(choices=[], usage=None)),
    )

    provider = AzureOpenAIProvider()

    result = provider.count_tokens()

    assert result["success"] is False
    assert "Either messages or text" in result["error"]
