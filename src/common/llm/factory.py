"""
Factory for creating LLM providers.
"""

import logging
from typing import Dict, List, Optional, Any

from PySide6.QtCore import QObject

from .base import BaseLLMProvider
from .providers import (
    AnthropicProvider,
    AnthropicBedrockProvider,
    GeminiProvider,
    AzureOpenAIProvider,
)

logger = logging.getLogger(__name__)


def create_provider(
    provider: str = "auto",
    timeout: float = 600.0,
    max_retries: int = 2,
    default_system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
    debug: bool = False,
    parent: Optional[QObject] = None
) -> Optional[BaseLLMProvider]:
    """
    Create an LLM provider instance.
    
    Args:
        provider: Provider name ("anthropic", "gemini", "azure_openai", or "auto")
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        default_system_prompt: Default system prompt
        api_key: Optional API key (provider-specific)
        azure_endpoint: Azure OpenAI specific endpoint
        api_version: Azure OpenAI specific API version
        debug: Debug mode flag
        parent: Parent QObject
        
    Returns:
        Initialized provider instance or None if initialization fails
    """
    # Auto-detect provider based on available API keys
    if provider == "auto":
        logger.info("Auto-detecting LLM provider...")
        
        # Try Anthropic first
        logger.info("Trying to initialize Anthropic provider...")
        anthropic = AnthropicProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            api_key=api_key,
            debug=debug,
            parent=parent
        )
        if anthropic.initialized:
            logger.info("Successfully auto-selected Anthropic provider")
            return anthropic

        # Try Gemini
        logger.info("Anthropic not available, trying Gemini provider...")
        gemini = GeminiProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            api_key=api_key,
            debug=debug,
            parent=parent
        )
        if gemini.initialized:
            logger.info("Successfully auto-selected Gemini provider")
            return gemini
        
        # Try Azure OpenAI
        logger.info("Gemini not available, trying Azure OpenAI provider...")
        azure = AzureOpenAIProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            debug=debug,
            parent=parent
        )
        if azure.initialized:
            logger.info("Successfully auto-selected Azure OpenAI provider")
            return azure
        
        logger.error("Auto-detection failed - no providers available")
        return None
    
    # Create specific provider
    if provider == "anthropic":
        logger.info("Creating Anthropic provider...")
        return AnthropicProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            api_key=api_key,
            debug=debug,
            parent=parent
        )

    if provider == "anthropic_bedrock":
        logger.info("Creating Anthropic Bedrock provider...")
        return AnthropicBedrockProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            aws_region=aws_region,
            aws_profile=aws_profile,
            debug=debug,
            parent=parent,
        )
    
    elif provider == "gemini":
        logger.info("Creating Gemini provider...")
        return GeminiProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            api_key=api_key,
            debug=debug,
            parent=parent
        )
    
    elif provider == "azure_openai":
        logger.info("Creating Azure OpenAI provider...")
        return AzureOpenAIProvider(
            timeout=timeout,
            max_retries=max_retries,
            default_system_prompt=default_system_prompt,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            debug=debug,
            parent=parent
        )
    
    else:
        logger.error(f"Unknown provider: {provider}")
        raise ValueError(f"Unknown provider: {provider}. Supported: anthropic, gemini, azure_openai, auto")


def get_available_providers() -> List[Dict[str, Any]]:
    """
    Get list of available providers and their status.
    
    Returns:
        List of dicts with provider info including availability
    """
    providers = []

    # Check Anthropic
    anthropic = AnthropicProvider()
    providers.append({
        "id": "anthropic",
        "name": "Anthropic Claude",
        "available": anthropic.initialized,
        "default_model": anthropic.default_model,
        "supports_pdf": True,
        "supports_thinking": True,
    })

    # Check Gemini
    gemini = GeminiProvider()
    providers.append({
        "id": "gemini",
        "name": "Google Gemini",
        "available": gemini.initialized,
        "default_model": gemini.default_model,
        "supports_pdf": False,
        "supports_thinking": True,  # Via structured prompting
    })
    
    # Check Azure OpenAI
    azure = AzureOpenAIProvider()
    providers.append({
        "id": "azure_openai",
        "name": "Azure OpenAI",
        "available": azure.initialized,
        "default_model": azure.default_model,
        "supports_pdf": False,
        "supports_thinking": False,
    })
    
    return providers
