"""Google Gemini provider implementation using google-genai SDK."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject

from ..base import BaseLLMProvider
from ..tokens import TokenCounter

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini API."""

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
        self._genai_module: Any = None
        self._default_model_name: str = os.getenv("GEMINI_MODEL", "").strip()
        self._init_client(api_key)

    def _resolve_api_key(self, explicit_key: Optional[str]) -> Optional[str]:
        if explicit_key:
            return explicit_key

        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if env_key:
            return env_key

        try:
            from src.app.core.secure_settings import SecureSettings

            settings = SecureSettings()
            return settings.get_api_key("gemini") or settings.get_api_key("google")
        except Exception as exc:  # pragma: no cover - defensive keychain fallback
            logger.debug("Unable to load Gemini key from SecureSettings: %s", exc)
            return None

    def _init_client(self, api_key: Optional[str]) -> None:
        try:
            from google import genai

            resolved_key = self._resolve_api_key(api_key)
            if not resolved_key:
                self.emit_error("Gemini API key not configured")
                return

            self._genai_module = genai
            self.client = genai.Client(api_key=resolved_key)
            self.set_initialized(True)
        except ImportError:
            self.emit_error("google-genai package not installed")
        except Exception as exc:
            logger.error("Failed to initialize Gemini client: %s", exc)
            self.emit_error(f"Failed to initialize Gemini: {exc}")
            self.client = None
            self.set_initialized(False)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        if self._default_model_name:
            return self._default_model_name

        from src.app.core.llm_catalog import runtime_default_model_for_provider

        self._default_model_name = runtime_default_model_for_provider("gemini") or ""
        return self._default_model_name

    def _build_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        active_system = system_prompt or self.default_system_prompt
        if not active_system:
            return prompt
        return f"System instructions:\n{active_system}\n\nUser request:\n{prompt}"

    def _usage_from_response(self, response: Any) -> Dict[str, int]:
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return {}

        input_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage_metadata, "total_token_count", 0) or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _generate_content(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> Any:
        assert self.client is not None
        return self.client.models.generate_content(
            model=model,
            contents=self._build_prompt(prompt, system_prompt),
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            },
        )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 200000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Gemini client not initialized", "provider": self.provider_name}

        selected_model = model or self.default_model

        try:
            self.emit_progress(10, "Sending request to Gemini...")
            response = self._generate_content(
                prompt=prompt,
                model=selected_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

            content = str(getattr(response, "text", "") or "")
            if not content:
                return {
                    "success": False,
                    "error": "Gemini response did not contain text output",
                    "provider": self.provider_name,
                }

            usage = self._usage_from_response(response)
            if not usage:
                usage = {
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(content) // 4,
                    "total_tokens": (len(prompt) + len(content)) // 4,
                }

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
            message = f"Gemini API error: {exc}"
            logger.error(message)
            self.emit_error(message)
            return {"success": False, "error": message, "provider": self.provider_name}

    def count_tokens(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Gemini client not initialized"}

        if text is None and messages is None:
            return {"success": False, "error": "Either messages or text must be provided"}

        if text is None and messages is not None:
            text_parts: list[str] = []
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            text_parts.append(item["text"])
            text = "\n".join(text_parts)

        try:
            assert text is not None
            response = self.client.models.count_tokens(
                model=self.default_model,
                contents=text,
            )
            token_count = int(getattr(response, "total_tokens", 0) or 0)
            if token_count > 0:
                return {"success": True, "token_count": token_count, "estimated": False}
        except Exception as exc:
            logger.debug("Gemini count_tokens failed, using fallback: %s", exc)

        return TokenCounter.count(text=text, messages=messages, provider=self.provider_name)

    def generate_with_thinking(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 200000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.initialized or not self.client:
            return {"success": False, "error": "Gemini client not initialized", "provider": self.provider_name}

        selected_model = model or self.default_model
        reasoning_prompt = (
            "Solve this request with clear, step-by-step reasoning.\\n"
            'Format your response with sections named "## Thinking" and "## Answer".\\n\\n'
            f"{prompt}"
        )

        try:
            self.emit_progress(10, "Processing with structured reasoning...")
            response = self._generate_content(
                prompt=reasoning_prompt,
                model=selected_model,
                max_tokens=max_tokens,
                temperature=max(0.2, min(0.5, temperature)),
                system_prompt=system_prompt,
            )
            full_text = str(getattr(response, "text", "") or "")
            if not full_text:
                return {
                    "success": False,
                    "error": "Gemini response did not contain text output",
                    "provider": self.provider_name,
                }

            thinking = full_text
            answer = full_text
            if "## Thinking" in full_text and "## Answer" in full_text:
                _, after_thinking = full_text.split("## Thinking", 1)
                thinking_section, answer_section = after_thinking.split("## Answer", 1)
                thinking = thinking_section.strip()
                answer = answer_section.strip()

            usage = self._usage_from_response(response)
            if not usage:
                usage = {
                    "input_tokens": len(reasoning_prompt) // 4,
                    "output_tokens": len(full_text) // 4,
                    "total_tokens": (len(reasoning_prompt) + len(full_text)) // 4,
                }

            self.emit_progress(100, "Reasoning complete")
            result = {
                "success": True,
                "content": answer,
                "thinking": thinking,
                "usage": usage,
                "provider": self.provider_name,
                "model": selected_model,
            }
            self.emit_response(result)
            return result
        except Exception as exc:
            message = f"Reasoning error: {exc}"
            logger.error(message)
            self.emit_error(message)
            return {"success": False, "error": message, "provider": self.provider_name}
