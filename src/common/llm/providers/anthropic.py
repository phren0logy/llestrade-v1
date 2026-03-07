"""Anthropic Claude provider implementation."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject

from src.config.observability import trace_llm_call

from ..base import BaseLLMProvider
from ..tokens import TokenCounter

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude API."""

    def __init__(
        self,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        debug: bool = False,
        parent: Optional[QObject] = None,
    ):
        super().__init__(timeout, max_retries, default_system_prompt, debug, parent)

        self.client: Any = None
        self._anthropic_module: Any = None
        self._init_client(api_key)
        self._instrument_if_enabled()

    def _instrument_if_enabled(self) -> None:
        if os.getenv("PHOENIX_ENABLED", "false").lower() != "true":
            return

        try:
            from openinference.instrumentation.anthropic import AnthropicInstrumentor

            AnthropicInstrumentor().instrument()
            logger.info("Anthropic instrumentation enabled")
        except Exception as exc:  # pragma: no cover - optional instrumentation
            logger.warning("Could not enable Anthropic instrumentation: %s", exc)

    def _init_client(self, api_key: Optional[str] = None) -> None:
        try:
            import anthropic

            resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not resolved_api_key or resolved_api_key == "your_api_key_here":
                self.emit_error("ANTHROPIC_API_KEY not configured")
                return

            self._anthropic_module = anthropic
            self.client = anthropic.Anthropic(
                api_key=resolved_api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self.set_initialized(True)

        except ImportError:
            logger.error("anthropic package not installed")
            self.emit_error("Anthropic package not installed")
        except Exception as exc:
            logger.error("Failed to initialize Anthropic client: %s", exc)
            self.emit_error(f"Failed to initialize Anthropic: {exc}")
            self.client = None
            self.set_initialized(False)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    def _extract_text_from_blocks(self, blocks: Any) -> str:
        if not blocks:
            return ""

        parts: list[str] = []
        for block in blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text" and hasattr(block, "text"):
                parts.append(str(block.text))
                continue
            if block_type == "output_text" and hasattr(block, "text"):
                parts.append(str(block.text))
                continue
            if isinstance(block, dict):
                if block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
                    parts.append(block["text"])

        return "".join(parts)

    def _extract_thinking_text(self, blocks: Any) -> str:
        if not blocks:
            return ""

        parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "thinking" and hasattr(block, "thinking"):
                parts.append(str(block.thinking))
            elif isinstance(block, dict) and block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                parts.append(block["thinking"])
        return "".join(parts)

    def _extract_usage(self, message: Any) -> Dict[str, int]:
        usage = getattr(message, "usage", None)
        if usage is None:
            return {}

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _build_error_message(self, exc: Exception) -> str:
        anthropic = self._anthropic_module
        message = str(exc)

        auth_type = getattr(anthropic, "AuthenticationError", None)
        rate_limit_type = getattr(anthropic, "RateLimitError", None)
        bad_request_type = getattr(anthropic, "BadRequestError", None)
        not_found_type = getattr(anthropic, "NotFoundError", None)

        if isinstance(auth_type, type) and isinstance(exc, auth_type):
            return f"Authentication error: {message}"
        if isinstance(rate_limit_type, type) and isinstance(exc, rate_limit_type):
            return f"Rate limit exceeded: {message}"
        if isinstance(bad_request_type, type) and isinstance(exc, bad_request_type):
            return f"Request error: {message}"
        if isinstance(not_found_type, type) and isinstance(exc, not_found_type):
            return f"Model not found: {message}"
        return f"API error: {message}"

    def _compose_message_kwargs(
        self,
        *,
        prompt_content: Any,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        thinking_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt_content}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if thinking_budget is not None:
            kwargs["temperature"] = 1.0
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        return kwargs

    @trace_llm_call()
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Anthropic client not initialized", "provider": self.provider_name}

        selected_model = model or self.default_model
        active_system_prompt = system_prompt or self.default_system_prompt

        try:
            self.emit_progress(10, "Sending request to Anthropic...")

            message = self.client.messages.create(
                **self._compose_message_kwargs(
                    prompt_content=prompt,
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=active_system_prompt,
                )
            )

            content = self._extract_text_from_blocks(getattr(message, "content", None))
            usage = self._extract_usage(message)

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
            error_message = self._build_error_message(exc)
            logger.error("Anthropic request failed: %s", error_message)
            self.emit_error(error_message)
            return {"success": False, "error": error_message, "provider": self.provider_name}

    def count_tokens(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Anthropic client not initialized"}

        if messages is None and text is None:
            return {"success": False, "error": "Either messages or text must be provided"}

        if messages is None and text is not None:
            messages = [{"role": "user", "content": text}]

        try:
            response = self.client.messages.count_tokens(model=self.default_model, messages=messages)
            count = int(getattr(response, "input_tokens", 0) or 0)
            return {"success": True, "token_count": count}
        except Exception as exc:
            logger.debug("Anthropic count_tokens failed, using fallback: %s", exc)
            return TokenCounter.count(text=text, messages=messages, provider=self.provider_name)

    def generate_with_pdf(
        self,
        prompt: str,
        pdf_file_path: str,
        model: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Anthropic client not initialized", "provider": self.provider_name}

        if not os.path.exists(pdf_file_path):
            return {"success": False, "error": f"PDF file not found: {pdf_file_path}", "provider": self.provider_name}

        selected_model = model or self.default_model
        active_system_prompt = system_prompt or self.default_system_prompt

        try:
            with open(pdf_file_path, "rb") as f:
                pdf_data = f.read()

            payload = [
                {"type": "text", "text": prompt},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.b64encode(pdf_data).decode("utf-8"),
                    },
                },
            ]

            self.emit_progress(10, "Sending PDF to Anthropic...")
            message = self.client.messages.create(
                **self._compose_message_kwargs(
                    prompt_content=payload,
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=active_system_prompt,
                )
            )

            content = self._extract_text_from_blocks(getattr(message, "content", None))
            usage = self._extract_usage(message)

            self.emit_progress(100, "PDF processed")
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
            error_message = self._build_error_message(exc)
            logger.error("Anthropic PDF request failed: %s", error_message)
            self.emit_error(error_message)
            return {"success": False, "error": error_message, "provider": self.provider_name}

    def generate_with_thinking(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 1.0,
        system_prompt: Optional[str] = None,
        thinking_budget: int = 16000,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Anthropic client not initialized", "provider": self.provider_name}

        selected_model = model or self.default_model
        active_system_prompt = system_prompt or self.default_system_prompt
        budget = min(thinking_budget, max(1, max_tokens - 1000))

        try:
            self.emit_progress(10, "Processing with extended thinking...")
            message = self.client.messages.create(
                **self._compose_message_kwargs(
                    prompt_content=prompt,
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=active_system_prompt,
                    thinking_budget=budget,
                )
            )

            blocks = getattr(message, "content", None)
            content = self._extract_text_from_blocks(blocks)
            thinking = self._extract_thinking_text(blocks)
            usage = self._extract_usage(message)

            self.emit_progress(100, "Thinking complete")
            result = {
                "success": True,
                "content": content,
                "thinking": thinking,
                "usage": usage,
                "provider": self.provider_name,
                "model": selected_model,
            }
            self.emit_response(result)
            return result

        except Exception as exc:
            error_message = self._build_error_message(exc)
            logger.error("Anthropic thinking request failed: %s", error_message)
            self.emit_error(error_message)
            return {"success": False, "error": error_message, "provider": self.provider_name}

    def generate_with_pdf_and_thinking(
        self,
        prompt: str,
        pdf_file_path: str,
        model: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 1.0,
        system_prompt: Optional[str] = None,
        thinking_budget_tokens: int = 16000,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Anthropic client not initialized", "provider": self.provider_name}

        if not os.path.exists(pdf_file_path):
            return {"success": False, "error": f"PDF file not found: {pdf_file_path}", "provider": self.provider_name}

        selected_model = model or self.default_model
        active_system_prompt = system_prompt or self.default_system_prompt
        budget = min(thinking_budget_tokens, max(1, max_tokens - 1000))

        try:
            with open(pdf_file_path, "rb") as f:
                pdf_data = f.read()

            payload = [
                {"type": "text", "text": prompt},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.b64encode(pdf_data).decode("utf-8"),
                    },
                },
            ]

            self.emit_progress(10, "Processing PDF with extended thinking...")
            message = self.client.messages.create(
                **self._compose_message_kwargs(
                    prompt_content=payload,
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=active_system_prompt,
                    thinking_budget=budget,
                )
            )

            blocks = getattr(message, "content", None)
            content = self._extract_text_from_blocks(blocks)
            thinking = self._extract_thinking_text(blocks)
            usage = self._extract_usage(message)

            self.emit_progress(100, "PDF processing with thinking complete")
            result = {
                "success": True,
                "content": content,
                "thinking": thinking,
                "usage": usage,
                "provider": self.provider_name,
                "model": selected_model,
            }
            self.emit_response(result)
            return result

        except Exception as exc:
            error_message = self._build_error_message(exc)
            logger.error("Anthropic PDF thinking request failed: %s", error_message)
            self.emit_error(error_message)
            return {"success": False, "error": error_message, "provider": self.provider_name}
