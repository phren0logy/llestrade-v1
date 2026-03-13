from __future__ import annotations

from pathlib import Path

from src.app.core.llm_catalog import default_model_for_provider
from src.app.core.secure_settings import SecureSettings, keyring_service_name
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
