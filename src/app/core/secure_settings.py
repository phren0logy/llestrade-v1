"""
Secure settings management for the new UI.
Handles API keys and sensitive configuration using OS keychain.
"""

import json
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import Dict, Optional, Any
from threading import Lock

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.warning("keyring module not available - API keys will be stored in plain text")

from PySide6.QtCore import QObject, Signal, QSettings
from src.config.paths import app_config_dir

# Shared cache so multiple SecureSettings instances reuse a single keychain lookup.
_GLOBAL_API_KEY_CACHE: Dict[str, Optional[str]] = {}
_CACHE_LOCK = Lock()
_CACHE_MISS = object()
_LEGACY_SETTINGS_ENV_VAR = "FRD_SETTINGS_DIR"


class SecureSettings(QObject):
    """Manages application settings with OS keychain integration for sensitive data."""
    
    settings_changed = Signal()
    api_key_changed = Signal(str)  # provider name
    
    SERVICE_NAME = "Llestrade"
    SETTINGS_FILE = "app_settings.json"
    
    def __init__(self, settings_dir: Optional[Path] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Determine settings directory
        env_override = os.getenv("LLESTRADE_SETTINGS_DIR") or os.getenv(_LEGACY_SETTINGS_ENV_VAR)
        if settings_dir:
            self.settings_dir = settings_dir
        elif env_override:
            self.settings_dir = Path(env_override).expanduser()
        else:
            self.settings_dir = app_config_dir()
        
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.settings_dir / self.SETTINGS_FILE
        
        # Cache for API keys to avoid repeated keychain access (shared across instances)
        self._api_key_cache = _GLOBAL_API_KEY_CACHE
        
        # Load regular settings
        self._settings = self._load_settings()
        
        # Qt settings for UI preferences
        self.qt_settings = QSettings(
            os.getenv("LLESTRADE_QSETTINGS_ORG", "Llestrade"),
            os.getenv("LLESTRADE_QSETTINGS_APP", "Settings"),
        )
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data.setdefault("feature_flags", {})
                    return data
            except Exception as e:
                self.logger.error(f"Error loading settings: {e}")
        return self._get_default_settings()
    
    def _save_settings(self):
        """Save settings to JSON file."""
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(self._settings, f, indent=2)
            self.settings_changed.emit()
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings."""
        return {
            "version": "1.0",
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4-5-20250929",
            "pydantic_ai_gateway_settings": {
                "base_url": None,
                "route": None,
            },
            "phoenix_settings": {
                "enabled": False,
                "target": "local_phoenix",
                "port": 6006,
                "project": "forensic-report-drafter",
                "export_fixtures": False,
                "content_policy": "unredacted",
                "include_binary_content": False,
            },
            "aws_bedrock_settings": {
                "profile": None,
                "region": None,
                "preferred_model": None,
            },
            "output_directory": str(Path.home() / "Documents" / "ForensicReports"),
            "auto_save_interval": 60,  # seconds
            "theme": "light",
            "font_size": 12,
            "window_geometry": None,
            "recent_projects": [],
            "max_recent_projects": 10,
            "show_welcome_screen": True,
            "check_for_updates": True,
            "telemetry_enabled": False,
            "debug_mode": False,
            "feature_flags": {},
        }
    
    # API Key Management (Secure)
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Store API key securely in OS keychain."""
        if not api_key:
            return self.remove_api_key(provider)
            
        if KEYRING_AVAILABLE:
            try:
                keyring.set_password(
                    self.SERVICE_NAME,
                    f"api_key_{provider}",
                    api_key
                )
                with _CACHE_LOCK:
                    self._api_key_cache[provider] = api_key
                self.api_key_changed.emit(provider)
                self.logger.info(f"API key for {provider} stored securely")
                return True
            except Exception as e:
                self.logger.error(f"Failed to store API key securely: {e}")
                return False
        else:
            # Fallback to settings file (less secure)
            self.logger.warning(f"Storing API key for {provider} in settings file (not secure)")
            if "api_keys" not in self._settings:
                self._settings["api_keys"] = {}
            self._settings["api_keys"][provider] = api_key
            self._save_settings()
            with _CACHE_LOCK:
                self._api_key_cache[provider] = api_key
            self.api_key_changed.emit(provider)
            return True
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve API key from OS keychain."""
        with _CACHE_LOCK:
            cached = self._api_key_cache.get(provider, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                return cached

            # Cache miss: hit the keychain (or fallback) while holding the lock
            key: Optional[str] = None
            if KEYRING_AVAILABLE:
                try:
                    key = keyring.get_password(self.SERVICE_NAME, f"api_key_{provider}")
                    if key:
                        self._api_key_cache[provider] = key
                        return key
                except Exception as e:
                    self.logger.error(f"Failed to retrieve API key: {e}")

            # Fallback to settings file
            api_keys = self._settings.get("api_keys", {})
            key = api_keys.get(provider)
            self._api_key_cache[provider] = key
            return key
    
    def remove_api_key(self, provider: str) -> bool:
        """Remove API key from OS keychain."""
        success = False
        
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password(
                    self.SERVICE_NAME,
                    f"api_key_{provider}"
                )
                success = True
            except Exception:
                pass  # Key might not exist
        
        # Also remove from settings file
        if "api_keys" in self._settings and provider in self._settings["api_keys"]:
            del self._settings["api_keys"][provider]
            self._save_settings()
            success = True
        
        # Clear cache
        with _CACHE_LOCK:
            self._api_key_cache.pop(provider, None)
        
        if success:
            self.api_key_changed.emit(provider)
            
        return success
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key exists for provider."""
        return self.get_api_key(provider) is not None
    
    # Regular Settings
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a setting value."""
        self._settings[key] = value
        self._save_settings()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value (alias for get)."""
        return self.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set a setting value (alias for set)."""
        self.set(key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings (excluding sensitive data)."""
        settings_copy = self._settings.copy()
        # Remove sensitive data
        settings_copy.pop("api_keys", None)
        return settings_copy
    
    # Recent Projects
    def add_recent_project(self, project_path: str, project_info: dict = None):
        """Add a project to recent projects list with metadata."""
        recent = self._settings.get("recent_projects", [])
        
        # Remove if already exists (check by path)
        recent = [p for p in recent if (isinstance(p, dict) and p.get('path') != project_path) or (isinstance(p, str) and p != project_path)]
        
        # Create project entry
        if project_info:
            entry = project_info
            entry['path'] = project_path
        else:
            entry = {
                'path': project_path,
                'name': Path(project_path).stem,
                'last_modified': datetime.now().isoformat()
            }
        
        # Add to front
        recent.insert(0, entry)
        
        # Limit size
        max_recent = self._settings.get("max_recent_projects", 10)
        recent = recent[:max_recent]
        
        self._settings["recent_projects"] = recent
        self._save_settings()
    
    def get_recent_projects(self) -> list:
        """Get list of recent projects with backward compatibility."""
        recent = self._settings.get("recent_projects", [])
        
        # Convert old string format to new dict format
        converted = []
        changed = False
        for item in recent:
            if isinstance(item, str):
                # Old format - convert to dict
                entry = {
                    'path': item,
                    'name': Path(item).stem,
                    'last_modified': ''
                }
                changed = True
            else:
                entry = item

            project_path_raw = str(entry.get("path", "")).strip()
            if not project_path_raw:
                changed = True
                continue

            project_path = Path(project_path_raw).expanduser()
            if not project_path.exists():
                changed = True
                continue

            converted.append(entry)

        if changed and converted != recent:
            self._settings["recent_projects"] = converted
            self._save_settings()

        return converted
    
    def remove_recent_project(self, project_path: str):
        """Remove a project from recent projects list."""
        recent = self._settings.get("recent_projects", [])
        
        # Remove by path (handle both old string and new dict format)
        recent = [p for p in recent if (isinstance(p, dict) and p.get('path') != project_path) or (isinstance(p, str) and p != project_path)]
        
        self._settings["recent_projects"] = recent
        self._save_settings()
    
    def clear_recent_projects(self):
        """Clear recent projects list."""
        self._settings["recent_projects"] = []
        self._save_settings()
    
    # Window State
    def save_window_geometry(self, geometry: bytes):
        """Save window geometry."""
        self.qt_settings.setValue("window/geometry", geometry)
    
    def get_window_geometry(self) -> Optional[bytes]:
        """Get saved window geometry."""
        return self.qt_settings.value("window/geometry")
    
    def save_window_state(self, state: bytes):
        """Save window state (for restoring docks, toolbars, etc)."""
        self.qt_settings.setValue("window/state", state)
    
    def get_window_state(self) -> Optional[bytes]:
        """Get saved window state."""
        return self.qt_settings.value("window/state")
    
    # Utility Methods
    def reset_to_defaults(self):
        """Reset all settings to defaults (except API keys)."""
        # Preserve API keys
        api_keys = self._settings.get("api_keys", {})
        
        # Reset to defaults
        self._settings = self._get_default_settings()
        
        # Restore API keys
        if api_keys:
            self._settings["api_keys"] = api_keys
        
        self._save_settings()
    
    def export_settings(self, path: Path):
        """Export settings to file (excluding sensitive data)."""
        export_data = self.get_all()
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_settings(self, path: Path):
        """Import settings from file."""
        try:
            with open(path, 'r') as f:
                imported = json.load(f)
            
            # Don't import sensitive data
            imported.pop("api_keys", None)
            
            # Merge with current settings
            self._settings.update(imported)
            self._save_settings()
            
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            raise
