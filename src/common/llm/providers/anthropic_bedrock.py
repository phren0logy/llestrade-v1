"""
Anthropic Claude provider implementation via AWS Bedrock.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject

from ..base import BaseLLMProvider
from ..bedrock_catalog import BedrockModel, DEFAULT_BEDROCK_MODELS, list_bedrock_models
from ..tokens import TokenCounter

logger = logging.getLogger(__name__)
_botocore_credentials_logger = logging.getLogger("botocore.credentials")

_ATTEMPT_LOCK = Lock()
_MAX_INIT_ATTEMPTS = 3
_INIT_ATTEMPTS = 0


@contextmanager
def _suppress_botocore_credentials_info():
    previous_level = _botocore_credentials_logger.level
    _botocore_credentials_logger.setLevel(max(previous_level, logging.WARNING))
    try:
        yield
    finally:
        _botocore_credentials_logger.setLevel(previous_level)


def _should_attempt_initialization() -> bool:
    with _ATTEMPT_LOCK:
        return _INIT_ATTEMPTS < _MAX_INIT_ATTEMPTS


def _record_attempt(success: bool, message: Optional[str] = None) -> int:
    """
    Track credential check attempts and handle logging.

    Returns the number of failed attempts recorded (0 on success).
    """
    global _INIT_ATTEMPTS
    with _ATTEMPT_LOCK:
        if success:
            _INIT_ATTEMPTS = 0
            return 0

        _INIT_ATTEMPTS += 1
        attempt = _INIT_ATTEMPTS

    if message:
        if attempt < _MAX_INIT_ATTEMPTS:
            logger.warning(message)
        elif attempt == _MAX_INIT_ATTEMPTS:
            logger.warning(f"{message} Automatic retries will stop until restart or manual reset.")
        else:
            logger.debug(message)

    return attempt


class AnthropicBedrockProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models running on AWS Bedrock."""

    def __init__(
        self,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_system_prompt: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_profile: Optional[str] = None,
        debug: bool = False,
        parent: Optional[QObject] = None,
    ):
        """
        Initialize the Anthropic Bedrock provider.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Number of automatic retries for transient failures.
            default_system_prompt: Default system prompt.
            aws_region: Optional explicit AWS region.
            aws_profile: Optional AWS profile name.
            debug: Enable verbose logging.
            parent: Qt parent object.
        """
        super().__init__(timeout, max_retries, default_system_prompt, debug, parent)

        self.client = None
        self._aws_region = aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self._aws_profile = aws_profile
        self._available_models: List[BedrockModel] = list(DEFAULT_BEDROCK_MODELS)

        preferred_model_id = os.getenv("AWS_BEDROCK_DEFAULT_MODEL") or ""
        self._default_model_id = self._select_default_model(preferred_model_id)

        if not _should_attempt_initialization():
            logger.debug(
                "Skipping Anthropic Bedrock initialization after %d failed attempts in this session",
                _MAX_INIT_ATTEMPTS,
            )
            return

        self._load_model_catalog()
        self._default_model_id = self._select_default_model(preferred_model_id)

        self._init_client()

    def _select_default_model(self, preferred_model_id: str) -> str:
        """Choose the default model, preferring the configured identifier."""
        if preferred_model_id:
            for model in self._available_models:
                if model.model_id == preferred_model_id:
                    return model.model_id
        if self._available_models:
            return self._available_models[0].model_id
        return preferred_model_id or DEFAULT_BEDROCK_MODELS[0].model_id

    def _load_model_catalog(self):
        """Load the list of available Anthropic models from Bedrock."""
        try:
            self._available_models = list_bedrock_models(
                region=self._aws_region,
                profile=self._aws_profile,
            )
            if self.debug:
                logger.debug(
                    "Discovered %d Bedrock Claude models", len(self._available_models)
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Unable to discover Bedrock models: %s", exc)
            self._available_models = list(DEFAULT_BEDROCK_MODELS)

    def _init_client(self):
        """Initialise the Anthropic Bedrock client."""
        try:
            import anthropic  # type: ignore

            client_kwargs: Dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }

            session = None
            if self._aws_profile:
                try:
                    import boto3  # type: ignore

                    with _suppress_botocore_credentials_info():
                        session = boto3.Session(
                            profile_name=self._aws_profile,
                            region_name=self._aws_region,
                        )
                    client_kwargs["session"] = session
                except Exception as exc:
                    logger.error("Failed to create AWS session for profile '%s': %s", self._aws_profile, exc)
                    message = (
                        f"Unable to create AWS session for profile '{self._aws_profile}'. "
                        "Ensure it exists by running 'aws configure sso' or 'aws configure'."
                    )
                    _record_attempt(False, message)
                    self.emit_error(message)
                    return

            if self._aws_region:
                client_kwargs["region_name"] = self._aws_region

            self.client = anthropic.AnthropicBedrock(**client_kwargs)
            self._test_connection()
        except ImportError:
            logger.error(
                "anthropic package not installed - please install with: uv add 'anthropic[bedrock]'"
            )
            self.emit_error("Anthropic package with Bedrock extra not installed")
        except Exception as exc:
            logger.error("Error initialising Anthropic Bedrock client: %s", exc)
            self.emit_error(f"Failed to initialise Anthropic Bedrock client: {exc}")
            self.client = None

    def _test_connection(self):
        """Verify that Bedrock credentials are available."""
        if not self.client:
            return

        success = False
        try:
            # Bedrock may not yet support count_tokens; handle gracefully.
            test_messages = [{"role": "user", "content": "Hello Claude"}]
            try:
                self.client.messages.count_tokens(
                    model=self.default_model,
                    messages=test_messages,
                )
                success = True
            except Exception as exc:
                logger.debug("Bedrock token counting unavailable: %s", exc)
                if hasattr(self.client, "models"):
                    try:
                        self.client.models.list()
                        success = True
                    except Exception as inner_exc:
                        logger.debug("Bedrock model listing failed: %s", inner_exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.error("Anthropic Bedrock connection test failed: %s", exc)
            success = False

        if success:
            logger.info("Anthropic Bedrock client initialised successfully")
            _record_attempt(True)
            self.set_initialized(True)
        else:
            message = "Could not verify AWS Bedrock credentials. Ensure `aws configure` has been run."
            attempts = _record_attempt(False, message)
            if attempts <= _MAX_INIT_ATTEMPTS:
                self.emit_error("Anthropic Bedrock not configured")
            self.client = None
            self.set_initialized(False)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic_bedrock"

    @property
    def default_model(self) -> str:
        """Return the default Bedrock model ID."""
        return self._default_model_id

    @property
    def available_models(self) -> List[BedrockModel]:
        """Expose the cached list of available models."""
        return list(self._available_models)

    @classmethod
    def reset_backoff(cls):
        """Allow callers to retry AWS credential checks after failures."""
        global _INIT_ATTEMPTS
        with _ATTEMPT_LOCK:
            _INIT_ATTEMPTS = 0
        logger.debug("AWS Bedrock credential check counter reset")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response using Anthropic Claude on Bedrock."""
        if not self.initialized or not self.client:
            return {"success": False, "error": "Anthropic Bedrock client not initialised"}

        try:
            selected_model = model or self.default_model
            effective_system_prompt = system_prompt or self.default_system_prompt

            options = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if effective_system_prompt:
                options["system"] = effective_system_prompt

            if self.debug:
                logger.debug("Bedrock Request - Model: %s", selected_model)
                logger.debug("Bedrock Request - Prompt length: %d", len(prompt))
                logger.debug("Bedrock Request - Temperature: %s", temperature)

            self.emit_progress(10, "Sending request to AWS Bedrock…")

            start_time = time.time()
            message = self.client.messages.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )
            elapsed_time = time.time() - start_time

            content = ""
            if hasattr(message, "content") and message.content:
                try:
                    content = "".join(part.text for part in message.content if hasattr(part, "text"))
                except Exception:
                    content = message.content[0].text if hasattr(message.content[0], "text") else str(message.content)

            if self.debug:
                logger.debug("Bedrock Response received in %.2f seconds", elapsed_time)

            usage = {}
            if hasattr(message, "usage") and message.usage:
                usage = {
                    "input_tokens": getattr(message.usage, "input_tokens", None),
                    "output_tokens": getattr(message.usage, "output_tokens", None),
                }

            response = {
                "success": True,
                "content": content,
                "model": selected_model,
                "usage": usage,
                "elapsed": elapsed_time,
            }
            self.emit_response(response)
            self.emit_progress(100, "AWS Bedrock response received")
            return response

        except Exception as exc:
            logger.error("Anthropic Bedrock generate call failed: %s", exc)
            error_message = (
                "Anthropic Bedrock request failed. Verify AWS credentials with 'aws sts get-caller-identity'. "
                f"Details: {exc}"
            )
            self.emit_error(error_message)
            return {"success": False, "error": error_message}

    def count_tokens(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Estimate token counts for Bedrock models."""
        return TokenCounter.count(
            text=text,
            messages=messages,
            provider=self.provider_name,
            model=self.default_model,
        )


__all__ = ["AnthropicBedrockProvider"]
