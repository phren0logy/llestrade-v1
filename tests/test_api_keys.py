#!/usr/bin/env python
"""
Test script for API key detection and client initialization
"""

import logging
import os
import sys
from pathlib import Path

import pytest

# Ensure we can import from the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.llm import create_provider
from src.common.llm.providers import GeminiProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

pytestmark = [pytest.mark.live_provider, pytest.mark.integration]

def test_api_keys():
    """Test API key detection for both providers."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if gemini_key:
        masked_key = f"{gemini_key[:4]}...{gemini_key[-4:]}" if len(gemini_key) > 8 else "[too short]"
        logging.info(f"Found Gemini API key: {masked_key}")
    else:
        logging.error("No Gemini API key found")
        
    if anthropic_key:
        masked_key = f"{anthropic_key[:4]}...{anthropic_key[-4:]}" if len(anthropic_key) > 8 else "[too short]"
        logging.info(f"Found Anthropic API key: {masked_key}")
    else:
        logging.error("No Anthropic API key found")
    
    assert gemini_key is not None and anthropic_key is not None, "Both Gemini and Anthropic API keys must be available"

def test_direct_gemini_provider():
    """Test direct creation of Gemini provider."""
    logging.info("Creating Gemini provider directly...")
    
    try:
        # Directly create Gemini provider
        provider = create_provider("gemini")
        
        assert isinstance(provider, GeminiProvider), f"Expected GeminiProvider, got {provider.__class__.__name__}"
        logging.info("Created GeminiProvider instance")
        
        assert provider.initialized, "Gemini provider should be initialized"
        logging.info("Gemini provider initialized successfully")
        model_name = provider.default_model
        assert model_name, "Gemini provider should expose a default model"
        
        # Test simple request
        response = provider.generate(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant.",
            model=model_name,
            temperature=0.1
        )
        
        assert response["success"], f"Gemini response failed: {response.get('error', 'Unknown error')}"
        logging.info(f"Gemini response: {response['content'][:50]}...")
        
    except Exception as e:
        logging.error(f"Error creating Gemini provider: {str(e)}")
        raise

def main():
    """Run all tests."""
    logging.info("Testing API key availability...")
    test_api_keys()
    
    logging.info("Testing direct Gemini provider creation...")
    test_direct_gemini_provider()
    
    logging.info("🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
