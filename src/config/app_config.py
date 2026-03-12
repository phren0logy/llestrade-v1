import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.app.core.llm_catalog import default_model_for_provider, runtime_default_model_for_provider
# Load environment variables
load_dotenv()

# Direct imports from new llm package
from src.common.llm.factory import create_provider
from src.common.llm.base import BaseLLMProvider

SETTINGS_FILE = Path("var/app_settings.json")


def _default_model(provider_id: str) -> str | None:
    """Resolve a runtime default model without hardcoding repo-owned ids."""
    if provider_id == "azure_openai":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if provider_id == "anthropic_bedrock":
        return os.getenv("AWS_BEDROCK_DEFAULT_MODEL")
    return runtime_default_model_for_provider(provider_id) or default_model_for_provider(provider_id)


def _default_settings() -> dict[str, Any]:
    return {
        "selected_llm_provider_id": "anthropic",
        "llm_provider_configs": {
            "anthropic": {
                "enabled": True,
                "label": "Anthropic Claude",
                "default_model": _default_model("anthropic"),
            },
            "gemini": {
                "enabled": True,
                "label": "Google Gemini",
                "default_model": _default_model("gemini"),
            },
            "azure_openai": {
                "enabled": True,
                "label": "Azure OpenAI",
                "default_deployment_name": _default_model("azure_openai"),
                "azure_endpoint": None,
                "api_version": None,
            },
        },
        "general_settings": {
            "debug_mode": False,
            "default_system_prompt": "You are a helpful AI assistant.",
        },
    }

def get_available_providers_and_models() -> list[dict[str, str]]:
    """
    Loads app settings and returns a list of available (enabled) LLM providers
    and their default models.

    Returns:
        A list of dictionaries, where each dictionary contains:
        'id': provider_id (e.g., "anthropic")
        'label': provider_label (e.g., "Anthropic Claude")
        'model': default_model_name or deployment_id
        'display_name': A user-friendly string like "Anthropic Claude (claude-sonnet-4-5)"
    """
    settings = load_app_settings()
    providers_and_models = []
    provider_configs = settings.get("llm_provider_configs", {})

    for provider_id, config in provider_configs.items():
        if config.get("enabled", False):
            label = config.get("label", provider_id)
            model = None
            if provider_id == "azure_openai":
                model = config.get("default_deployment_name")
            else:
                model = config.get("default_model") or _default_model(provider_id)

            if model: # Only add if a model/deployment is specified
                display_name = f"{label} ({model})"
                providers_and_models.append({
                    "id": provider_id,
                    "label": label,
                    "model": model,
                    "display_name": display_name
                })
            else:
                logging.warning(f"Provider '{label}' (ID: {provider_id}) is enabled but has no default model/deployment configured. It will not be available for selection.")
    return providers_and_models

def load_app_settings() -> dict:
    """Loads application settings from SETTINGS_FILE, creates it with defaults if not found."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_FILE.exists():
        logging.info(f"'{SETTINGS_FILE}' not found. Creating with default settings.")
        defaults = _default_settings()
        save_app_settings(defaults)
        return defaults
    try:
        with SETTINGS_FILE.open('r', encoding='utf-8') as f:
            settings = json.load(f)
            
        # Override Azure deployment name from environment if available
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if azure_deployment and "llm_provider_configs" in settings and "azure_openai" in settings["llm_provider_configs"]:
            settings["llm_provider_configs"]["azure_openai"]["default_deployment_name"] = azure_deployment
            logging.debug(f"Using Azure OpenAI deployment name from environment: {azure_deployment}")
            
        return settings
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error loading '{SETTINGS_FILE}': {e}. Returning default settings.")
        return _default_settings()

def save_app_settings(settings: dict):
    """Saves the provided settings dictionary to SETTINGS_FILE."""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SETTINGS_FILE.open('w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        logging.info(f"Settings saved to '{SETTINGS_FILE}'.")
    except IOError as e:
        logging.error(f"Error saving settings to '{SETTINGS_FILE}': {e}")

def get_configured_llm_provider(
    provider_id_override: Optional[str] = None,
    model_override: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Attempts to initialize an LLM provider using the new direct API.
    Uses the selected provider from app_settings.json by default.
    Can be overridden with specific provider_id and model.

    Args:
        provider_id_override: If provided, use this provider_id instead of the one in settings.
        model_override: If provided, use this model/deployment name.

    Returns:
        A dictionary with the provider instance and its configuration if successful,
        None otherwise. Errors are logged.
        
    Note: This function replaces get_configured_llm_client() and returns
    a provider instead of a client wrapper.
    """
    settings = load_app_settings()

    selected_provider_id = provider_id_override if provider_id_override else settings.get("selected_llm_provider_id")

    if not selected_provider_id:
        logging.error(
            f"No LLM provider selected (either via override or in '{SETTINGS_FILE}'). "
            "Please configure one via the UI or ensure override is passed."
        )
        return None

    provider_configs = settings.get("llm_provider_configs", {})
    specific_config = provider_configs.get(selected_provider_id)

    if not specific_config:
        logging.error(
            f"Configuration for selected provider '{selected_provider_id}' not found "
            f"in '{SETTINGS_FILE}'."
        )
        return None

    if not specific_config.get("enabled", False):
        logging.error(
            f"Selected LLM provider '{selected_provider_id}' "
            f"(Label: '{specific_config.get('label', 'N/A')}') is disabled in settings."
        )
        return None

    general_cfg = settings.get("general_settings", {})
    provider_label = specific_config.get("label", selected_provider_id)

    factory_args = {
        "provider": selected_provider_id,
        "api_key": None,  # Providers will pick up from environment variables
        "default_system_prompt": general_cfg.get("default_system_prompt"),
        "debug": general_cfg.get("debug_mode", False),
        # Consider adding timeout, max_retries to general_settings or provider_configs if needed
    }

    effective_model_name = None
    if selected_provider_id == "azure_openai":
        # Try to get Azure settings from SecureSettings if available
        try:
            from src.app.core import SecureSettings
            secure_settings = SecureSettings()
            azure_settings = secure_settings.get("azure_openai_settings", {})
            
            # Use SecureSettings values if available, otherwise fall back to config
            factory_args["azure_endpoint"] = azure_settings.get("endpoint") or specific_config.get("azure_endpoint")
            factory_args["api_version"] = azure_settings.get("api_version") or specific_config.get("api_version")
            
            # For deployment name, check SecureSettings first
            effective_model_name = model_override or azure_settings.get("deployment") or specific_config.get("default_deployment_name")
        except Exception as e:
            logging.debug(f"Could not load SecureSettings for Azure OpenAI: {e}")
            # Fall back to original behavior
            factory_args["azure_endpoint"] = specific_config.get("azure_endpoint")
            factory_args["api_version"] = specific_config.get("api_version")
            effective_model_name = (
                model_override
                or specific_config.get("default_deployment_name")
                or _default_model("azure_openai")
            )
        
        if not effective_model_name:
            logging.error(
                f"Azure OpenAI is selected, but its 'deployment_name' is not configured "
                f"(either via override, SecureSettings, or in '{SETTINGS_FILE}' for provider '{provider_label}'). This is required."
            )
            return None
    elif selected_provider_id == "anthropic_bedrock":
        try:
            from src.app.core import SecureSettings

            secure_settings = SecureSettings()
            bedrock_settings = secure_settings.get("aws_bedrock_settings", {}) or {}
        except Exception as e:  # pragma: no cover - defensive fallback
            logging.debug(f"Could not load SecureSettings for AWS Bedrock: {e}")
            bedrock_settings = {}

        factory_args["aws_region"] = bedrock_settings.get("region")
        factory_args["aws_profile"] = bedrock_settings.get("profile")
        effective_model_name = (
            model_override
            or bedrock_settings.get("preferred_model")
            or specific_config.get("default_model")
            or _default_model("anthropic_bedrock")
        )
    else:  # For Anthropic, Gemini, etc.
        effective_model_name = (
            model_override
            or specific_config.get("default_model")
            or _default_model(selected_provider_id)
        )
        if not effective_model_name:
            logging.error(
                f"Provider '{provider_label}' is selected, but its 'default_model' "
                f"is not configured (either via override or in '{SETTINGS_FILE}')."
            )
            return None

    logging.info(f"Attempting to initialize LLM provider: {provider_label} "
                 f"(ID: {selected_provider_id}) with model/deployment: '{effective_model_name}'")

    provider: Optional[BaseLLMProvider] = create_provider(**factory_args)

    if provider and provider.initialized:
        logging.info(f"Successfully initialized LLM provider: {provider_label}")
        return {
            "provider": provider,  # Changed from "client" to "provider"
            "provider_id": selected_provider_id,
            "provider_label": provider_label,
            "effective_model_name": effective_model_name,
        }
    else:
        logging.error(
            f"Failed to initialize the selected LLM provider: {provider_label} (ID: {selected_provider_id}). "
            "Please check its API key in environment variables, any specific configurations "
            f"(like Azure deployment name: '{effective_model_name if selected_provider_id == 'azure_openai' else 'N/A'}') "
            "in settings, and network connectivity. The provider's own logs may have more details."
        )
        return None

# Backward compatibility alias
get_configured_llm_client = get_configured_llm_provider

if __name__ == '__main__':
    # Basic example of how to use these functions
    # Setup basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    active_provider_info = get_configured_llm_provider()
    
    document_id_example = "DOC-EXAMPLE-001"

    if active_provider_info:
        provider_instance = active_provider_info["provider"]
        model_name_to_use = active_provider_info["effective_model_name"]
        active_provider_label = active_provider_info["provider_label"]

        print(f"Successfully obtained LLM provider: {active_provider_label} "
              f"for document '{document_id_example}' using model/deployment: '{model_name_to_use}'")
        
        # Example: Using the provider with new API
        # test_prompt = "What is the capital of France?"
        # response = provider_instance.generate(prompt=test_prompt, model=model_name_to_use)
        
        # if response["success"]:
        #     print(f"Response from {active_provider_label}: {response['content']}")
        # else:
        #     print(f"API call failed for {active_provider_label}. Error: {response['error']}")
    else:
        print(f"Failed to obtain an active LLM provider for document '{document_id_example}'. "
              "Check logs for details. Document processing cannot proceed and will need to be re-run.")
