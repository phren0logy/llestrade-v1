from __future__ import annotations

from src.config.observability import PhoenixObservability


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_record_result_attributes_with_dict_usage() -> None:
    obs = PhoenixObservability()
    span = _FakeSpan()

    obs._record_result_attributes(
        span,
        {
            "content": "response text",
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        },
    )

    assert span.attributes["llm.output_messages"] == "response text"
    assert span.attributes["llm.token_count.prompt"] == 11
    assert span.attributes["llm.token_count.completion"] == 7
    assert span.attributes["llm.token_count.total"] == 18


def test_detect_provider_from_module_name() -> None:
    obs = PhoenixObservability()

    assert obs._detect_provider("src.common.llm.providers.anthropic") == "anthropic"
    assert obs._detect_provider("src.common.llm.providers.azure_openai") == "openai"
    assert obs._detect_provider("src.common.llm.providers.gemini") == "google"
    assert obs._detect_provider("src.something.else") == "unknown"


def test_pydantic_ai_instrumentation_redacts_content_when_enabled(monkeypatch) -> None:
    obs = PhoenixObservability()
    obs.enabled = True
    obs._content_policy = "redacted"

    class _FakeInstrumentationSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "pydantic_ai.models.instrumented.InstrumentationSettings",
        _FakeInstrumentationSettings,
    )

    instrumentation = obs.pydantic_ai_instrumentation()

    assert instrumentation is not None
    assert instrumentation.kwargs["include_content"] is False
    assert instrumentation.kwargs["include_binary_content"] is False
    assert obs.pydantic_ai_instrumentation() is instrumentation


def test_pydantic_ai_instrumentation_defaults_to_unredacted_for_local_phoenix(monkeypatch) -> None:
    obs = PhoenixObservability()
    obs.enabled = True
    obs._content_policy = "unredacted"
    obs._include_binary_content = False

    class _FakeInstrumentationSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "pydantic_ai.models.instrumented.InstrumentationSettings",
        _FakeInstrumentationSettings,
    )

    instrumentation = obs.pydantic_ai_instrumentation()

    assert instrumentation is not None
    assert instrumentation.kwargs["include_content"] is True
    assert instrumentation.kwargs["include_binary_content"] is False


def test_content_policy_defaults_are_target_aware() -> None:
    assert PhoenixObservability._resolve_content_policy({}, target="local_phoenix") == "unredacted"
    assert PhoenixObservability._resolve_content_policy({}, target="remote_otel") == "redacted"
