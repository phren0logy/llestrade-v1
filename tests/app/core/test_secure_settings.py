from __future__ import annotations

import json
from pathlib import Path

import keyring
import pytest

from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.core.llm_catalog import default_model_for_provider
from src.app.core.secure_settings import (
    SecureKeyStorageError,
    SecureSettings,
    keyring_service_name,
)
from src.config.paths import app_user_root


def test_app_user_root_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-user-root"
    monkeypatch.setenv("LLESTRADE_USER_ROOT", str(override))

    resolved = app_user_root()

    assert resolved == override
    assert resolved.exists()


def test_secure_settings_supports_legacy_settings_env(monkeypatch, tmp_path: Path) -> None:
    settings_dir = tmp_path / "legacy-settings"
    monkeypatch.delenv("LLESTRADE_SETTINGS_DIR", raising=False)
    monkeypatch.setenv("FRD_SETTINGS_DIR", str(settings_dir))

    settings = SecureSettings()

    assert settings.settings_dir == settings_dir
    assert settings.settings_path == settings_dir / SecureSettings.SETTINGS_FILE


def test_secure_settings_supports_keyring_service_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLESTRADE_KEYRING_SERVICE_NAME", "LlestradeTests-Keyring")

    settings = SecureSettings(settings_dir=tmp_path / "settings")

    assert keyring_service_name() == "LlestradeTests-Keyring"
    assert settings.service_name == "LlestradeTests-Keyring"


def test_get_recent_projects_prunes_missing_entries(tmp_path: Path) -> None:
    settings_dir = tmp_path / "settings"
    settings = SecureSettings(settings_dir=settings_dir)

    existing_project = tmp_path / "project-a" / "project.frpd"
    existing_project.parent.mkdir(parents=True)
    existing_project.write_text("{}", encoding="utf-8")

    missing_project = tmp_path / "project-b" / "project.frpd"

    settings.set(
        "recent_projects",
        [
            {"path": str(existing_project), "name": "Project A", "last_modified": ""},
            {"path": str(missing_project), "name": "Project B", "last_modified": ""},
        ],
    )

    recent = settings.get_recent_projects()

    assert recent == [{"path": str(existing_project), "name": "Project A", "last_modified": ""}]
    assert settings.get("recent_projects") == recent


def test_secure_settings_defaults_use_catalog_backed_llm_model(tmp_path: Path) -> None:
    settings = SecureSettings(settings_dir=tmp_path / "settings")

    assert settings.get("llm_provider") == "anthropic"
    assert settings.get("llm_model") == (default_model_for_provider("anthropic") or "")


def test_secure_settings_uses_test_keyring_backend() -> None:
    backend = keyring.get_keyring()

    assert "macOS" not in type(backend).__module__


def test_set_api_key_persists_single_bundle_and_not_settings_file(tmp_path: Path) -> None:
    settings = SecureSettings(settings_dir=tmp_path / "settings")

    settings.set_api_key("anthropic", "anthropic-key-1")
    settings.set_api_key("google", "gemini-key-1")

    payload = keyring.get_password(settings.service_name, "app_secrets")
    assert payload is not None
    bundle = json.loads(payload)
    assert bundle == {
        "anthropic": "anthropic-key-1",
        "gemini": "gemini-key-1",
    }

    if settings.settings_path.exists():
        stored_settings = json.loads(settings.settings_path.read_text(encoding="utf-8"))
        assert "api_keys" not in stored_settings


def test_secure_settings_migrates_legacy_provider_entries_and_cleans_them_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLESTRADE_KEYRING_SERVICE_NAME", "LLestrade")
    keyring.set_password("Llestrade", "api_key_anthropic", "legacy-anthropic")
    keyring.set_password("Llestrade", "api_key_google", "legacy-gemini")

    settings = SecureSettings(settings_dir=tmp_path / "settings")

    assert settings.get_api_key("anthropic") == "legacy-anthropic"
    assert settings.get_api_key("gemini") == "legacy-gemini"

    migrated = keyring.get_password("LLestrade", "app_secrets")
    assert migrated is not None
    assert json.loads(migrated) == {
        "anthropic": "legacy-anthropic",
        "gemini": "legacy-gemini",
    }
    assert keyring.get_password("Llestrade", "api_key_anthropic") is None
    assert keyring.get_password("Llestrade", "api_key_google") is None


def test_set_api_key_fails_closed_without_plaintext_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = SecureSettings(settings_dir=tmp_path / "settings")

    def _fail_set_password(*_args, **_kwargs) -> None:
        raise RuntimeError("keychain unavailable")

    monkeypatch.setattr(keyring, "set_password", _fail_set_password)

    with pytest.raises(SecureKeyStorageError, match="Failed to write keychain bundle"):
        settings.set_api_key("anthropic", "anthropic-key-1")

    if settings.settings_path.exists():
        stored_settings = json.loads(settings.settings_path.read_text(encoding="utf-8"))
        assert "api_keys" not in stored_settings


def test_project_manager_recent_projects_stay_in_test_settings(tmp_path: Path) -> None:
    manager = ProjectManager()
    project_path = manager.create_project(tmp_path, ProjectMetadata(case_name="Recent Isolation"))

    settings = SecureSettings()
    recent = settings.get_recent_projects()

    assert recent
    assert recent[0]["path"] == str(project_path)
    assert Path(settings.settings_path).resolve().is_relative_to((tmp_path / "settings").resolve())
