#!/usr/bin/env python
"""
Test both Gemini and Anthropic clients working in the same script
"""

import logging
import os
import sys

import pytest

from src.common.llm import create_provider
from src.common.llm.providers.anthropic import AnthropicProvider
from src.common.llm.providers.gemini import GeminiProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

pytestmark = [pytest.mark.live_provider, pytest.mark.integration]

def test_both_clients():
    """Test initializing and using both Gemini and Anthropic clients."""
    
    # Initialize both clients directly
    logging.info("Initializing Anthropic client...")
    anthropic_client = create_provider(provider="anthropic")
    
    logging.info("Initializing Gemini client...")
    gemini_client = create_provider(provider="gemini")
    
    # Test if both clients initialized (using hasattr to avoid isinstance issues with mocks)
    anthropic_working = hasattr(anthropic_client, 'initialized') and anthropic_client.initialized
    gemini_working = hasattr(gemini_client, 'initialized') and gemini_client.initialized
    
    logging.info(f"Anthropic client initialized: {anthropic_working}")
    logging.info(f"Gemini client initialized: {gemini_working}")
    
    # Test request with both if possible
    success = True
    
    # Test Anthropic if available
    if anthropic_working:
        try:
            logging.info("Testing Anthropic response...")
            anthropic_model = anthropic_client.default_model
            assert anthropic_model, "Anthropic client should expose a default model"
            response = anthropic_client.generate(
                prompt="What is the capital of France?",
                model=anthropic_model,
                temperature=0.1
            )
            
            if response["success"]:
                logging.info(f"Anthropic response: {response['content'][:50]}...")
            else:
                logging.error(f"Anthropic response failed: {response.get('error', 'Unknown error')}")
                success = False
        except Exception as e:
            logging.error(f"Error generating Anthropic response: {str(e)}")
            success = False
    
    # Test Gemini if available  
    if gemini_working:
        try:
            logging.info("Testing Gemini response...")
            gemini_model = gemini_client.default_model
            assert gemini_model, "Gemini client should expose a default model"
            response = gemini_client.generate(
                prompt="What is the capital of France?",
                model=gemini_model,
                temperature=0.1
            )
            
            if response["success"]:
                logging.info(f"Gemini response: {response['content'][:50]}...")
            else:
                logging.error(f"Gemini response failed: {response.get('error', 'Unknown error')}")
                success = False
        except Exception as e:
            logging.error(f"Error generating Gemini response: {str(e)}")
            success = False
    
    assert success and (anthropic_working or gemini_working), "At least one client should work and all tests should pass"

def main():
    """Run all tests."""
    test_both_clients()
    
    logging.info("🎉 Tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
