"""Azure OpenAI provider implementation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import openai
from PySide6.QtCore import QObject

from src.config.observability import trace_llm_call

from ..base import BaseLLMProvider
from ..tokens import TokenCounter

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    """Provider for Azure OpenAI API."""

    def __init__(
        self,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        debug: bool = False,
        parent: Optional[QObject] = None,
    ):
        super().__init__(timeout, max_retries, default_system_prompt, debug, parent)

        self.client: Optional[openai.AzureOpenAI] = None
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version

        self._init_client(api_key, azure_endpoint, api_version)
        self._instrument_if_enabled()

    def _instrument_if_enabled(self) -> None:
        if os.getenv("PHOENIX_ENABLED", "false").lower() != "true":
            return

        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
            logger.info("OpenAI instrumentation enabled")
        except Exception as exc:  # pragma: no cover - optional instrumentation
            logger.warning("Could not enable OpenAI instrumentation: %s", exc)

    def _init_client(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        resolved_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        resolved_api_version = api_version or os.getenv("OPENAI_API_VERSION")

        if not resolved_api_key or resolved_api_key == "your_azure_openai_api_key":
            self.emit_error("Azure OpenAI API key not configured")
            return
        if not resolved_endpoint or resolved_endpoint == "your_azure_openai_endpoint":
            self.emit_error("Azure OpenAI endpoint not configured")
            return
        if not resolved_api_version or resolved_api_version == "your_api_version":
            self.emit_error("Azure OpenAI API version not configured")
            return

        self.api_key = resolved_api_key
        self.azure_endpoint = resolved_endpoint
        self.api_version = resolved_api_version

        try:
            self.client = openai.AzureOpenAI(
                api_key=resolved_api_key,
                azure_endpoint=resolved_endpoint,
                api_version=resolved_api_version,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self.set_initialized(True)
        except Exception as exc:
            logger.error("Failed to initialize Azure OpenAI client: %s", exc)
            self.emit_error(f"Failed to initialize Azure OpenAI: {exc}")
            self.client = None
            self.set_initialized(False)

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    @property
    def default_model(self) -> str:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

    def _retryable_error_types(self) -> tuple[type[BaseException], ...]:
        candidates = [
            "APITimeoutError",
            "APIConnectionError",
            "RateLimitError",
            "InternalServerError",
        ]
        retryable: list[type[BaseException]] = []
        for name in candidates:
            error_type = getattr(openai, name, None)
            if isinstance(error_type, type):
                retryable.append(error_type)
        return tuple(retryable)

    def _execute_with_retries(self, request_fn):
        retryable = self._retryable_error_types()
        attempt = 0
        delay = 1.0

        while True:
            try:
                return request_fn()
            except Exception as exc:  # pragma: no cover - retry behavior validated by unit tests
                can_retry = bool(retryable) and isinstance(exc, retryable)
                if not can_retry or attempt >= self.max_retries:
                    raise

                attempt += 1
                logger.warning(
                    "Retrying Azure OpenAI request after %s (%s/%s)",
                    type(exc).__name__,
                    attempt,
                    self.max_retries,
                )
                self.emit_progress(min(90, 10 + (attempt * 20)), f"Retrying request ({attempt}/{self.max_retries})")
                time.sleep(delay)
                delay *= 2

    def _extract_response_text(self, completion: Any) -> str:
        if not getattr(completion, "choices", None):
            return ""

        message = completion.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text" and hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)

        return str(content)

    def _extract_usage(self, completion: Any, messages: List[Dict[str, Any]], content: str) -> Dict[str, int]:
        usage = getattr(completion, "usage", None)
        if usage is not None:
            return {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }

        input_tokens = self.count_tokens(messages=messages).get("token_count", 0)
        output_tokens = self.count_tokens(text=content).get("token_count", 0)
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens) + int(output_tokens),
        }

    @trace_llm_call()
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Azure OpenAI client not initialized", "provider": self.provider_name}

        selected_model = model or self.default_model
        active_system_prompt = system_prompt or self.default_system_prompt

        messages: List[Dict[str, str]] = []
        if active_system_prompt:
            messages.append({"role": "system", "content": active_system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.debug:
            logger.debug(
                "Azure OpenAI request model=%s endpoint=%s api_version=%s",
                selected_model,
                self.azure_endpoint,
                self.api_version,
            )

        self.emit_progress(10, "Sending request to Azure OpenAI...")

        try:
            started = time.time()
            completion = self._execute_with_retries(
                lambda: self.client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
            )
            elapsed = time.time() - started
            if self.debug:
                logger.debug("Azure OpenAI response received in %.2fs", elapsed)

            content = self._extract_response_text(completion)
            usage = self._extract_usage(completion, messages, content)

            self.emit_progress(100, "Response received")
            result = {
                "success": True,
                "content": content,
                "usage": usage,
                "provider": self.provider_name,
                "model": selected_model,
            }
            self.emit_response(result)
            return result

        except Exception as exc:
            auth_type = getattr(openai, "AuthenticationError", None)
            not_found_type = getattr(openai, "NotFoundError", None)
            rate_limit_type = getattr(openai, "RateLimitError", None)
            connection_type = getattr(openai, "APIConnectionError", None)

            if isinstance(auth_type, type) and isinstance(exc, auth_type):
                message = f"Authentication error: {exc}"
            elif isinstance(not_found_type, type) and isinstance(exc, not_found_type):
                message = f"Deployment not found for model '{selected_model}': {exc}"
            elif isinstance(rate_limit_type, type) and isinstance(exc, rate_limit_type):
                message = f"Rate limit exceeded: {exc}"
            elif isinstance(connection_type, type) and isinstance(exc, connection_type):
                message = f"Connection error: {exc}"
            else:
                message = f"API error: {exc}"

            logger.error("Azure OpenAI request failed: %s", message)
            self.emit_error(message)
            return {"success": False, "error": message, "provider": self.provider_name}

    def count_tokens(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if text is None and messages is None:
            return {"success": False, "error": "Either messages or text must be provided"}

        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self.default_model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")

            if text is not None:
                return {"success": True, "token_count": len(encoding.encode(text))}

            token_count = 0
            assert messages is not None
            for message in messages:
                token_count += 3
                for key, value in message.items():
                    if isinstance(value, str):
                        token_count += len(encoding.encode(value))
                    elif key == "content" and isinstance(value, list):
                        for part in value:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                token_count += len(encoding.encode(part["text"]))
                    if key == "name":
                        token_count += 1

            token_count += 3
            return {"success": True, "token_count": token_count}

        except ImportError:
            logger.warning("tiktoken is not installed, using TokenCounter fallback")
            return TokenCounter.count(text=text, messages=messages, provider=self.provider_name)
        except Exception as exc:
            logger.error("Token counting failed: %s", exc)
            return TokenCounter.count(text=text, messages=messages, provider=self.provider_name)

    def generate_with_pdf(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "PDF processing is not supported by Azure OpenAI",
            "provider": self.provider_name,
        }

    def generate_with_thinking(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Extended thinking is not supported by Azure OpenAI",
            "provider": self.provider_name,
        }
