from __future__ import annotations

from src.config.observability import ObservabilityRuntime, ObservabilitySettings


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_record_result_attributes_with_dict_usage() -> None:
    runtime = ObservabilityRuntime()
    span = _FakeSpan()

    runtime._record_result_attributes(
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
    runtime = ObservabilityRuntime()

    assert runtime._detect_provider("src.common.llm.providers.anthropic") == "anthropic"
    assert runtime._detect_provider("src.common.llm.providers.azure_openai") == "openai"
    assert runtime._detect_provider("src.common.llm.providers.gemini") == "google"
    assert runtime._detect_provider("src.something.else") == "unknown"


def test_get_model_instrumentation_settings_redacts_content_when_enabled(monkeypatch) -> None:
    runtime = ObservabilityRuntime()
    runtime.enabled = True
    runtime.settings = ObservabilitySettings(
        enabled=True,
        target="otlp_http",
        content_policy="redacted",
        include_binary_content=False,
    )
    runtime.tracer_provider = object()

    class _FakeInstrumentationSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "pydantic_ai.models.instrumented.InstrumentationSettings",
        _FakeInstrumentationSettings,
    )

    instrumentation = runtime.get_model_instrumentation_settings()

    assert instrumentation is not None
    assert instrumentation.kwargs["tracer_provider"] is runtime.tracer_provider
    assert instrumentation.kwargs["include_content"] is False
    assert instrumentation.kwargs["include_binary_content"] is False
    assert runtime.get_model_instrumentation_settings() is instrumentation


def test_get_model_instrumentation_settings_defaults_to_unredacted_for_local_phoenix(monkeypatch) -> None:
    runtime = ObservabilityRuntime()
    runtime.enabled = True
    runtime.settings = ObservabilitySettings(
        enabled=True,
        target="phoenix_local",
        content_policy="unredacted",
        include_binary_content=False,
    )
    runtime.tracer_provider = object()

    class _FakeInstrumentationSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "pydantic_ai.models.instrumented.InstrumentationSettings",
        _FakeInstrumentationSettings,
    )

    instrumentation = runtime.get_model_instrumentation_settings()

    assert instrumentation is not None
    assert instrumentation.kwargs["include_content"] is True
    assert instrumentation.kwargs["include_binary_content"] is False


def test_content_policy_defaults_are_target_aware() -> None:
    assert ObservabilitySettings._resolve_content_policy(None, target="phoenix_local") == "unredacted"
    assert ObservabilitySettings._resolve_content_policy(None, target="otlp_http") == "redacted"


def test_initialize_uses_otlp_http_target(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeExporter:
        def __init__(self, **kwargs):
            captured["exporter_kwargs"] = kwargs

    monkeypatch.setattr("src.config.observability.OTLPSpanExporter", _FakeExporter)

    runtime = ObservabilityRuntime()
    provider = runtime.initialize(
        {
            "enabled": True,
            "target": "otlp_http",
            "project": "llestrade-tests",
            "content_policy": "redacted",
            "otlp_endpoint": "https://otel.example.com/v1/traces",
            "otlp_headers": {"Authorization": "Bearer token"},
        }
    )

    assert provider is not None
    assert captured["exporter_kwargs"] == {
        "endpoint": "https://otel.example.com/v1/traces",
        "headers": {"Authorization": "Bearer token"},
    }
    runtime.shutdown()


def test_initialize_uses_local_phoenix_target(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeExporter:
        def __init__(self, **kwargs):
            captured["exporter_kwargs"] = kwargs

    monkeypatch.setattr("src.config.observability.OTLPSpanExporter", _FakeExporter)
    monkeypatch.setattr("src.config.observability.PHOENIX_AVAILABLE", True)
    monkeypatch.setattr("src.config.observability.px", type("_Phoenix", (), {"launch_app": staticmethod(lambda: captured.setdefault("launched", True))}))
    monkeypatch.setattr(ObservabilityRuntime, "_is_port_open", staticmethod(lambda port: False))

    runtime = ObservabilityRuntime()
    provider = runtime.initialize(
        {
            "enabled": True,
            "target": "phoenix_local",
            "project": "llestrade-tests",
            "phoenix_port": 7777,
        }
    )

    assert provider is not None
    assert captured["launched"] is True
    assert captured["exporter_kwargs"] == {
        "endpoint": "http://127.0.0.1:7777/v1/traces",
        "headers": {},
    }
    runtime.shutdown()
