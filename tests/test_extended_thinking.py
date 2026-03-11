#!/usr/bin/env python
"""
Test script for Gemini extended thinking capabilities
"""

import logging
import sys
from pathlib import Path

import pytest

# Ensure we can import from the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.llm import create_provider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

pytestmark = [pytest.mark.live_provider, pytest.mark.integration]

def test_gemini_thinking():
    """Test that the Gemini client can use extended thinking."""
    logging.info("Testing Gemini extended thinking...")
    
    # Create a Gemini provider
    provider = create_provider("gemini")
    
    assert provider.initialized, "Failed to initialize Gemini provider"
    logging.info("✅ Gemini provider initialized successfully")
    model_name = provider.default_model
    assert model_name, "Gemini provider should expose a default model"
    
    # Test extended thinking
    # Complex prompt that should trigger multi-step thinking
    complex_prompt = """
    Solve this step by step:
    
    If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?
    Then, if another train travels the same distance but takes 3 hours, what is its speed?
    Finally, if both trains start at the same time from stations that are 300 miles apart
    and travel toward each other, how long will it take for them to meet?
    """
    
    response = provider.generate_with_thinking(
        prompt=complex_prompt,
        model=model_name,
        temperature=0.7,
        thinking_budget=2000,
    )
    
    assert response["success"], f"Gemini extended thinking failed: {response.get('error', 'Unknown error')}"
    logging.info(f"✅ Gemini extended thinking response received")
    logging.info(f"Content: {response['content'][:200]}...")
    if "thinking" in response and response["thinking"]:
        logging.info(f"Thinking: {response['thinking'][:200]}...")
    else:
        logging.info("No thinking content returned")

def main():
    """Run tests and exit with appropriate status code."""
    test_gemini_thinking()
    logging.info("🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
