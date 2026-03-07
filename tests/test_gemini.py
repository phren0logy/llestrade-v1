"""
Test script for Gemini API integration.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Ensure we can import from the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import our LLM utils
from src.common.llm import create_provider
from src.common.llm.providers.gemini import GeminiProvider

# Test direct Google Generative AI import
try:
    import google.generativeai as genai
    logging.info("Successfully imported google.generativeai package directly")
except ImportError as e:
    logging.error(f"Error importing google.generativeai: {str(e)}")

pytestmark = [pytest.mark.live_provider, pytest.mark.integration]


def test_gemini_client():
    """Test the Gemini client initialization and basic response generation."""
    logging.info("Testing Gemini client initialization...")
    
    # Create a Gemini client directly
    gemini_client = create_provider(provider="gemini")
    
    assert gemini_client.initialized, "Failed to initialize Gemini client"
    logging.info("✅ Gemini client initialized successfully")
    
    # Test a simple response
    logging.info("Testing Gemini response generation...")
    response = gemini_client.generate(
        prompt="What is the capital of France?",
        model="gemini-2.5-pro-preview-05-06",
        temperature=0.1,
    )
    
    assert response["success"], f"Gemini response failed: {response.get('error', 'Unknown error')}"
    logging.info(f"✅ Gemini response received: {response['content'][:100]}...")

def test_auto_client_fallback():
    """Test the auto client selection with fallback to Gemini."""
    logging.info("Testing auto client selection capabilities...")
    
    # Get the original API key value
    original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # First test with Anthropic key present
    try:
        if original_anthropic_key:
            logging.info("Testing auto client with Anthropic API key available...")
            auto_client = create_provider(provider="auto")
            
            # Check if Anthropic was selected (using hasattr to avoid isinstance issues)
            if hasattr(auto_client, 'initialized') and auto_client.initialized:
                logging.info("✅ Auto client correctly selected a working client (likely Anthropic)")
                return  # Early return if a working client is available
    finally:
        pass
    
    # Now test fallback to Gemini by removing Anthropic key
    try:
        if original_anthropic_key:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            logging.info("Testing fallback to Gemini (removed Anthropic key)...")
        
        auto_client = create_provider(provider="auto")
        
        assert hasattr(auto_client, 'initialized') and auto_client.initialized, "Auto client selection should have initialized a working client"
        logging.info("✅ Auto client selection correctly fell back to a working client")
        
        # Test a response
        response = auto_client.generate(
            prompt="Tell me a short joke.",
            model="gemini-2.5-pro-preview-05-06",
            temperature=0.7,
        )
        
        assert response["success"], f"Auto client response failed: {response.get('error', 'Unknown error')}"
        logging.info(f"✅ Auto client response: {response['content'][:100]}...")
        
    finally:
        # Restore original API key
        if original_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = original_anthropic_key

def main():
    """Run all tests."""
    logging.info("=== GEMINI API INTEGRATION TESTS ===")
    
    # Test direct Gemini client
    test_gemini_client()
    
    # Test auto client fallback
    test_auto_client_fallback()
    
    logging.info("🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
