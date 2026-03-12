"""Target-neutral OpenTelemetry runtime helpers."""

from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

try:
    import phoenix as px

    PHOENIX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    PHOENIX_AVAILABLE = False
    px = None


ObservabilityTarget = str
ContentPolicy = str


@dataclass(frozen=True, slots=True)
class ObservabilitySettings:
    enabled: bool = False
    target: ObservabilityTarget = "phoenix_local"
    project: str = "forensic-report-drafter"
    content_policy: ContentPolicy = "unredacted"
    include_binary_content: bool = False
    phoenix_port: int | None = 6006
    otlp_endpoint: str | None = None
    otlp_headers: dict[str, str] | None = None

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any] | None = None) -> "ObservabilitySettings":
        raw = dict(settings or {})
        target = cls._resolve_target(raw.get("target"))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            target=target,
            project=str(raw.get("project") or "forensic-report-drafter").strip() or "forensic-report-drafter",
            content_policy=cls._resolve_content_policy(raw.get("content_policy"), target=target),
            include_binary_content=bool(raw.get("include_binary_content", False)),
            phoenix_port=cls._resolve_port(raw.get("phoenix_port")),
            otlp_endpoint=cls._resolve_endpoint(raw.get("otlp_endpoint")),
            otlp_headers=cls._resolve_headers(raw.get("otlp_headers")),
        )

    @staticmethod
    def _resolve_target(value: Any) -> ObservabilityTarget:
        target = str(value or "phoenix_local").strip().lower()
        if target not in {"phoenix_local", "otlp_http"}:
            return "phoenix_local"
        return target

    @staticmethod
    def _resolve_content_policy(value: Any, *, target: ObservabilityTarget) -> ContentPolicy:
        policy = str(value or "").strip().lower()
        if policy in {"redacted", "unredacted"}:
            return policy
        return "unredacted" if target == "phoenix_local" else "redacted"

    @staticmethod
    def _resolve_port(value: Any) -> int | None:
        if value in {None, ""}:
            return 6006
        try:
            return int(value)
        except (TypeError, ValueError):
            return 6006

    @staticmethod
    def _resolve_endpoint(value: Any) -> str | None:
        endpoint = str(value or "").strip()
        return endpoint or None

    @staticmethod
    def _resolve_headers(value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            return {}
        headers: dict[str, str] = {}
        for key, item in value.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            headers[key_text] = str(item)
        return headers


class ObservabilityRuntime:
    """Configures target-neutral OTEL tracing and model instrumentation."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.enabled = False
        self.settings = ObservabilitySettings()
        self.tracer_provider: TracerProvider | None = None
        self.tracer: Any = None
        self._model_instrumentation: Any = None

    def initialize(self, settings: Mapping[str, Any] | None = None) -> TracerProvider | None:
        self.shutdown()
        self.settings = ObservabilitySettings.from_mapping(settings)
        self.enabled = self.settings.enabled
        if not self.enabled:
            self.logger.info("Observability disabled")
            return None

        try:
            exporter = self._create_exporter(self.settings)
            resource = Resource.create(
                {
                    "service.name": "llestrade",
                    "llestrade.project": self.settings.project,
                    "llestrade.observability.target": self.settings.target,
                }
            )
            self.tracer_provider = TracerProvider(resource=resource)
            self.tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            self.tracer = self.tracer_provider.get_tracer("llestrade.observability")
            self._model_instrumentation = None
            self.logger.info(
                "Observability initialized target=%s project=%s",
                self.settings.target,
                self.settings.project,
            )
            return self.tracer_provider
        except Exception as exc:
            self.logger.error("Failed to initialize observability: %s", exc)
            self.shutdown()
            return None

    def _create_exporter(self, settings: ObservabilitySettings) -> OTLPSpanExporter:
        if settings.target == "phoenix_local":
            endpoint = self._ensure_local_phoenix(settings)
            return OTLPSpanExporter(endpoint=endpoint, headers={})
        if settings.target == "otlp_http":
            if not settings.otlp_endpoint:
                raise RuntimeError("OTLP endpoint is required when target is 'otlp_http'")
            return OTLPSpanExporter(
                endpoint=settings.otlp_endpoint,
                headers=dict(settings.otlp_headers or {}),
            )
        raise RuntimeError(f"Unsupported observability target: {settings.target}")

    def _ensure_local_phoenix(self, settings: ObservabilitySettings) -> str:
        if not PHOENIX_AVAILABLE:
            raise RuntimeError("Phoenix is not installed; install arize-phoenix to use the local Phoenix target")
        port = settings.phoenix_port or 6006
        if not self._is_port_open(port):
            os.environ["PHOENIX_PORT"] = str(port)
            os.environ["PHOENIX_HOST"] = "127.0.0.1"
            px.launch_app()
        return f"http://127.0.0.1:{port}/v1/traces"

    @staticmethod
    def _is_port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @contextmanager
    def trace_operation(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self.enabled or self.tracer is None:
            yield None
            return

        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, "" if value is None else str(value))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def trace_llm_call(self, model_name: Optional[str] = None):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                runtime_model = kwargs.get("model") or model_name or "unknown"
                attributes = {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": runtime_model,
                    "llm.provider": self._detect_provider(func.__module__),
                }
                if "temperature" in kwargs:
                    attributes["llm.temperature"] = kwargs["temperature"]

                with self.trace_operation(f"llm.{func.__name__}", attributes) as span:
                    result = func(*args, **kwargs)
                    if span is not None:
                        self._record_result_attributes(span, result)
                    return result

            return wrapper

        return decorator

    def get_model_instrumentation_settings(self) -> Any | None:
        if not self.enabled or self.tracer_provider is None:
            return None
        if self._model_instrumentation is not None:
            return self._model_instrumentation

        try:
            from pydantic_ai.models.instrumented import InstrumentationSettings

            self._model_instrumentation = InstrumentationSettings(
                tracer_provider=self.tracer_provider,
                include_content=self.settings.content_policy == "unredacted",
                include_binary_content=self.settings.include_binary_content,
            )
        except Exception:
            self.logger.debug("Failed to initialize model instrumentation settings", exc_info=True)
            return None

        return self._model_instrumentation

    def shutdown(self) -> None:
        if self.tracer_provider is not None:
            try:
                self.tracer_provider.shutdown()
            except Exception:  # pragma: no cover - defensive
                self.logger.debug("Failed to shutdown tracer provider", exc_info=True)
        self.tracer_provider = None
        self.tracer = None
        self._model_instrumentation = None
        self.enabled = False
        self.settings = ObservabilitySettings()

    def _record_result_attributes(self, span: Any, result: Any) -> None:
        if isinstance(result, dict):
            content = result.get("content")
            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            if isinstance(content, str):
                span.set_attribute("llm.output_messages", content[:1000])
            if usage:
                span.set_attribute("llm.token_count.prompt", int(usage.get("input_tokens", 0) or 0))
                span.set_attribute("llm.token_count.completion", int(usage.get("output_tokens", 0) or 0))
                span.set_attribute("llm.token_count.total", int(usage.get("total_tokens", 0) or 0))
            return

        if hasattr(result, "content"):
            span.set_attribute("llm.output_messages", str(getattr(result, "content"))[:1000])

    @staticmethod
    def _detect_provider(module_name: str) -> str:
        module = module_name.lower()
        if "anthropic" in module:
            return "anthropic"
        if "openai" in module:
            return "openai"
        if "gemini" in module or "google" in module:
            return "google"
        return "unknown"


observability = ObservabilityRuntime()


def initialize_observability(settings: Mapping[str, Any] | None = None):
    return observability.initialize(settings)


def trace_llm_call(model_name: Optional[str] = None):
    return observability.trace_llm_call(model_name)


@contextmanager
def trace_operation(name: str, attributes: Optional[Dict[str, Any]] = None):
    with observability.trace_operation(name, attributes) as span:
        yield span


def get_model_instrumentation_settings() -> Any | None:
    return observability.get_model_instrumentation_settings()
