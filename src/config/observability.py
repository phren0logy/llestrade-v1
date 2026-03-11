"""Phoenix observability integration for local tracing."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import phoenix as px
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from phoenix.otel import register

    PHOENIX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    PHOENIX_AVAILABLE = False
    px = None
    trace = None
    register = None
    Status = None
    StatusCode = None


class PhoenixObservability:
    """Manages Arize Phoenix startup and tracing decorators."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.enabled = False
        self.client: Any = None
        self.tracer: Any = None
        self._pydantic_ai_instrumentation: Any = None
        self.project_name = "forensic-report-drafter"
        self.export_fixtures = False
        self.fixtures_dir = Path("var/test_output/fixtures")

    def initialize(self, settings: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if not PHOENIX_AVAILABLE:
            self.logger.warning("Phoenix not available. Install with: uv add --dev arize-phoenix")
            return None

        phoenix_settings = (settings or {}).get("phoenix_settings", {})
        self.enabled = bool(phoenix_settings.get("enabled", os.getenv("PHOENIX_ENABLED", "false").lower() == "true"))
        if not self.enabled:
            self.logger.info("Phoenix observability disabled")
            return None

        port = int(phoenix_settings.get("port", os.getenv("PHOENIX_PORT", "6006")))
        self.project_name = str(phoenix_settings.get("project", os.getenv("PHOENIX_PROJECT", "forensic-report-drafter")))
        self.export_fixtures = bool(
            phoenix_settings.get(
                "export_fixtures",
                os.getenv("PHOENIX_EXPORT_FIXTURES", "false").lower() == "true",
            )
        )

        try:
            if not self._is_port_open(port):
                os.environ["PHOENIX_PORT"] = str(port)
                os.environ["PHOENIX_HOST"] = "127.0.0.1"
                px.launch_app()

            register(project_name=self.project_name, endpoint=f"http://127.0.0.1:{port}/v1/traces")
            self.tracer = trace.get_tracer(__name__)
            self.client = px.Client()

            if self.export_fixtures:
                self.fixtures_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Phoenix initialized at http://127.0.0.1:%s", port)
            return self.client

        except Exception as exc:
            self.logger.error("Failed to initialize Phoenix: %s", exc)
            self.enabled = False
            self.client = None
            self.tracer = None
            self._pydantic_ai_instrumentation = None
            return None

    def _is_port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @contextmanager
    def trace_operation(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self.enabled or not self.tracer:
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
                    if self.export_fixtures:
                        self._save_fixture(runtime_model, kwargs, result)
                    return result

            return wrapper

        return decorator

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

    def _detect_provider(self, module_name: str) -> str:
        module = module_name.lower()
        if "anthropic" in module:
            return "anthropic"
        if "openai" in module:
            return "openai"
        if "gemini" in module or "google" in module:
            return "google"
        return "unknown"

    def _save_fixture(self, model: str, inputs: Dict[str, Any], output: Any) -> None:
        if not self.export_fixtures:
            return

        try:
            prompt = str(inputs.get("prompt", ""))
            prompt_hash = abs(hash(prompt[:500])) % (10**8)
            filename = f"{model}_{prompt_hash}_{int(time.time())}.json"
            payload = {
                "model": model,
                "input": {
                    "prompt": prompt[:1000],
                    "temperature": inputs.get("temperature"),
                    "max_tokens": inputs.get("max_tokens"),
                },
                "output": output if isinstance(output, dict) else str(output),
            }
            with (self.fixtures_dir / filename).open("w", encoding="utf-8") as fixture:
                json.dump(payload, fixture, indent=2)
        except Exception as exc:  # pragma: no cover - optional diagnostics path
            self.logger.warning("Failed to save fixture: %s", exc)

    def get_traces(self) -> Optional[list[Any]]:
        if not self.client:
            return None
        self.logger.warning(
            "Phoenix trace retrieval is not implemented; get_traces() currently returns an empty list."
        )
        return []

    def shutdown(self) -> None:
        self.client = None
        self.tracer = None
        self._pydantic_ai_instrumentation = None

    def pydantic_ai_instrumentation(self) -> Any | None:
        """Return redacted Pydantic AI instrumentation settings when Phoenix is enabled."""
        if not self.enabled or not PHOENIX_AVAILABLE:
            return None
        if self._pydantic_ai_instrumentation is not None:
            return self._pydantic_ai_instrumentation

        try:
            from pydantic_ai.models.instrumented import InstrumentationSettings

            self._pydantic_ai_instrumentation = InstrumentationSettings(
                include_content=False,
                include_binary_content=False,
            )
        except Exception:
            self.logger.debug("Failed to initialize Pydantic AI instrumentation", exc_info=True)
            return None

        return self._pydantic_ai_instrumentation


phoenix = PhoenixObservability()


def setup_observability(settings: Optional[Dict[str, Any]] = None):
    return phoenix.initialize(settings)


def trace_llm_call(model_name: Optional[str] = None):
    return phoenix.trace_llm_call(model_name)


@contextmanager
def trace_operation(name: str, attributes: Optional[Dict[str, Any]] = None):
    with phoenix.trace_operation(name, attributes) as span:
        yield span


def get_pydantic_ai_instrumentation() -> Any | None:
    return phoenix.pydantic_ai_instrumentation()
