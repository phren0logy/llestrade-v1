"""
Secure settings management for the new UI.
Handles API keys and sensitive configuration using OS keychain.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.warning("keyring module not available - secure keychain storage is unavailable")

from PySide6.QtCore import QObject, Signal, QSettings
from src.app.core.llm_catalog import default_model_for_provider, suspend_secure_settings_lookup
from src.config.paths import app_config_dir

# Shared cache so multiple SecureSettings instances reuse a single keychain lookup.
_GLOBAL_API_KEY_CACHE: Dict[tuple[str, str], Optional[str]] = {}
_CACHE_LOCK = Lock()
_CACHE_MISS = object()
_LEGACY_SETTINGS_ENV_VAR = "FRD_SETTINGS_DIR"
_KEYRING_SERVICE_ENV_VAR = "LLESTRADE_KEYRING_SERVICE_NAME"
_QSETTINGS_PATH_ENV_VAR = "LLESTRADE_QSETTINGS_PATH"
_KEYCHAIN_ACCOUNT = "app_secrets"
_LEGACY_KEYCHAIN_SERVICE_NAME = "Llestrade"
_PROVIDER_CANONICAL_NAMES = {
    "azure": "azure_openai",
    "google": "gemini",
}
_LEGACY_KEYCHAIN_PROVIDERS: Tuple[str, ...] = (
    "anthropic",
    "azure",
    "azure_di",
    "azure_openai",
    "gemini",
    "google",
    "pydantic_ai_gateway",
)


class SecureKeyStorageError(RuntimeError):
    """Raised when secure keychain storage is unavailable or inconsistent."""


def keyring_service_name(default: str = "LLestrade") -> str:
    override = os.getenv(_KEYRING_SERVICE_ENV_VAR)
    if override and override.strip():
        return override.strip()
    return default


def reset_api_key_cache() -> None:
    """Clear the shared SecureSettings API key cache."""
    with _CACHE_LOCK:
        _GLOBAL_API_KEY_CACHE.clear()


def _canonical_provider_name(provider: str) -> str:
    normalized = str(provider or "").strip()
    return _PROVIDER_CANONICAL_NAMES.get(normalized, normalized)


def _provider_aliases(provider: str) -> Tuple[str, ...]:
    canonical = _canonical_provider_name(provider)
    aliases = [canonical]
    for alias, resolved in _PROVIDER_CANONICAL_NAMES.items():
        if resolved == canonical:
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


class SecureSettings(QObject):
    """Manages application settings with OS keychain integration for sensitive data."""
    
    settings_changed = Signal()
    api_key_changed = Signal(str)  # provider name
    
    SERVICE_NAME = "LLestrade"
    SETTINGS_FILE = "app_settings.json"
    
    def __init__(self, settings_dir: Optional[Path] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.service_name = keyring_service_name(self.SERVICE_NAME)
        
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
        qsettings_path = os.getenv(_QSETTINGS_PATH_ENV_VAR)
        if qsettings_path and qsettings_path.strip():
            qt_settings_path = Path(qsettings_path).expanduser()
            qt_settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.qt_settings = QSettings(str(qt_settings_path), QSettings.IniFormat)
        else:
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
        with suspend_secure_settings_lookup():
            default_model = default_model_for_provider("anthropic") or ""
        return {
            "version": "1.0",
            "llm_provider": "anthropic",
            "llm_model": default_model,
            "pydantic_ai_gateway_settings": {
                "base_url": None,
                "route": None,
            },
            "observability_settings": {
                "enabled": False,
                "target": "phoenix_local",
                "project": "forensic-report-drafter",
                "content_policy": "unredacted",
                "include_binary_content": False,
                "phoenix_port": 6006,
                "otlp_endpoint": None,
                "otlp_headers": {},
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
    def _cache_key(self, provider: str) -> tuple[str, str]:
        return self.service_name, _canonical_provider_name(provider)

    def _legacy_settings_bundle(self) -> Dict[str, str]:
        api_keys = self._settings.get("api_keys", {})
        if not isinstance(api_keys, dict):
            return {}
        bundle: Dict[str, str] = {}
        for provider, value in api_keys.items():
            if isinstance(provider, str) and isinstance(value, str) and value.strip():
                bundle.setdefault(_canonical_provider_name(provider), value.strip())
        return bundle

    def _migration_service_names(self) -> tuple[str, ...]:
        if self.service_name == self.SERVICE_NAME:
            return (self.service_name, _LEGACY_KEYCHAIN_SERVICE_NAME)
        if self.service_name == _LEGACY_KEYCHAIN_SERVICE_NAME:
            return (self.service_name, self.SERVICE_NAME)
        return (self.service_name,)

    def _read_keyring_bundle(self, service_name: str | None = None) -> Dict[str, str]:
        if not KEYRING_AVAILABLE:
            return {}
        try:
            raw = keyring.get_password(service_name or self.service_name, _KEYCHAIN_ACCOUNT)
        except Exception as exc:  # pragma: no cover - backend-specific
            raise SecureKeyStorageError(f"Failed to read keychain bundle: {exc}") from exc
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise SecureKeyStorageError("Keychain bundle is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise SecureKeyStorageError("Keychain bundle must be a JSON object")

        bundle: Dict[str, str] = {}
        for provider, value in payload.items():
            if isinstance(provider, str) and isinstance(value, str) and value.strip():
                bundle[_canonical_provider_name(provider)] = value.strip()
        return bundle

    def _write_keyring_bundle(self, bundle: Dict[str, str]) -> Dict[str, str]:
        if not KEYRING_AVAILABLE:
            raise SecureKeyStorageError("System keychain support is unavailable")
        sanitized = {
            _canonical_provider_name(provider): value.strip()
            for provider, value in bundle.items()
            if isinstance(provider, str) and isinstance(value, str) and value.strip()
        }
        try:
            if sanitized:
                keyring.set_password(
                    self.service_name,
                    _KEYCHAIN_ACCOUNT,
                    json.dumps(sanitized, sort_keys=True),
                )
            else:
                keyring.delete_password(self.service_name, _KEYCHAIN_ACCOUNT)
        except Exception as exc:  # pragma: no cover - backend-specific
            raise SecureKeyStorageError(f"Failed to write keychain bundle: {exc}") from exc

        stored = self._read_keyring_bundle(self.service_name)
        if stored != sanitized:
            raise SecureKeyStorageError("Keychain bundle verification failed after save")
        return stored

    def _collect_legacy_keyring_entries(self) -> tuple[Dict[str, str], list[tuple[str, str]]]:
        collected: Dict[str, str] = {}
        stale_entries: list[tuple[str, str]] = []
        if not KEYRING_AVAILABLE:
            return collected, stale_entries

        for service_name in self._migration_service_names():
            if service_name != self.service_name:
                try:
                    legacy_bundle = self._read_keyring_bundle(service_name)
                except SecureKeyStorageError:
                    legacy_bundle = {}
                if legacy_bundle:
                    for provider, value in legacy_bundle.items():
                        collected.setdefault(provider, value)
                    stale_entries.append((service_name, _KEYCHAIN_ACCOUNT))

            for provider in _LEGACY_KEYCHAIN_PROVIDERS:
                account = f"api_key_{provider}"
                try:
                    value = keyring.get_password(service_name, account)
                except Exception:  # pragma: no cover - backend-specific
                    continue
                if not value or not value.strip():
                    continue
                collected.setdefault(_canonical_provider_name(provider), value.strip())
                stale_entries.append((service_name, account))
        return collected, stale_entries

    def _delete_keyring_entry(self, service_name: str, account: str) -> None:
        if not KEYRING_AVAILABLE:
            return
        try:
            keyring.delete_password(service_name, account)
        except Exception:
            pass

    def _remove_legacy_settings_key(self, provider: str) -> bool:
        api_keys = self._settings.get("api_keys")
        if not isinstance(api_keys, dict):
            return False
        changed = False
        for alias in _provider_aliases(provider):
            if alias in api_keys:
                del api_keys[alias]
                changed = True
        if changed:
            if api_keys:
                self._settings["api_keys"] = api_keys
            else:
                self._settings.pop("api_keys", None)
            self._save_settings()
        return changed

    def _load_keyring_bundle(self) -> Dict[str, str]:
        if not KEYRING_AVAILABLE:
            return {}
        bundle = self._read_keyring_bundle(self.service_name)
        legacy_bundle, stale_entries = self._collect_legacy_keyring_entries()

        changed = False
        for provider, value in legacy_bundle.items():
            if provider not in bundle:
                bundle[provider] = value
                changed = True

        if changed:
            bundle = self._write_keyring_bundle(bundle)

        if stale_entries and bundle:
            for service_name, account in stale_entries:
                self._delete_keyring_entry(service_name, account)

        return bundle

    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Store API key securely in OS keychain."""
        if not api_key:
            return self.remove_api_key(provider)
        canonical_provider = _canonical_provider_name(provider)

        bundle = self._load_keyring_bundle()
        bundle[canonical_provider] = api_key
        stored_bundle = self._write_keyring_bundle(bundle)
        stored_value = stored_bundle.get(canonical_provider)
        if stored_value != api_key:
            raise SecureKeyStorageError(
                f"Keychain verification mismatch while saving {canonical_provider}"
            )

        self._remove_legacy_settings_key(canonical_provider)
        with _CACHE_LOCK:
            self._api_key_cache[self._cache_key(canonical_provider)] = stored_value
        self.api_key_changed.emit(canonical_provider)
        self.logger.info("API key for %s stored securely in keychain bundle", canonical_provider)
        return True
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve API key from OS keychain."""
        canonical_provider = _canonical_provider_name(provider)
        cache_key = self._cache_key(canonical_provider)
        with _CACHE_LOCK:
            cached = self._api_key_cache.get(cache_key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                return cached

            key: Optional[str] = None
            if KEYRING_AVAILABLE:
                try:
                    bundle = self._load_keyring_bundle()
                    key = bundle.get(canonical_provider)
                    if key:
                        self._api_key_cache[cache_key] = key
                        return key
                except Exception as exc:
                    self.logger.error("Failed to retrieve API key from keychain bundle: %s", exc)

            # Backward-compatible fallback for legacy plaintext settings files.
            key = self._legacy_settings_bundle().get(canonical_provider)
            self._api_key_cache[cache_key] = key
            return key
    
    def remove_api_key(self, provider: str) -> bool:
        """Remove API key from OS keychain."""
        canonical_provider = _canonical_provider_name(provider)
        success = False

        if KEYRING_AVAILABLE:
            bundle = self._load_keyring_bundle()
            if canonical_provider in bundle:
                del bundle[canonical_provider]
                self._write_keyring_bundle(bundle)
                success = True
            for service_name in self._migration_service_names():
                for alias in _provider_aliases(canonical_provider):
                    account = f"api_key_{alias}"
                    try:
                        if keyring.get_password(service_name, account):
                            success = True
                    except Exception:
                        pass
                    self._delete_keyring_entry(service_name, account)

        if self._remove_legacy_settings_key(canonical_provider):
            success = True

        # Clear cache
        with _CACHE_LOCK:
            self._api_key_cache.pop(self._cache_key(canonical_provider), None)

        if success:
            self.api_key_changed.emit(canonical_provider)

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
