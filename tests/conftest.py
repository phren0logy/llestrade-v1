"""Shared pytest configuration for dashboard tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _live_provider_enabled() -> bool:
    return _env_flag("RUN_LIVE_PROVIDER_TESTS", default=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply marker taxonomy and gate live-provider tests by default."""

    run_live = _live_provider_enabled()
    skip_live = pytest.mark.skip(
        reason="live provider tests are disabled (set RUN_LIVE_PROVIDER_TESTS=1 to enable)"
    )

    for item in items:
        path = Path(str(item.fspath))
        path_text = path.as_posix()
        filename = path.name

        if "/tests/unit/" in path_text or "/tests/common/" in path_text:
            item.add_marker(pytest.mark.unit)
        if "/tests/app/core/" in path_text:
            item.add_marker(pytest.mark.core)
        if "/tests/app/workers/" in path_text:
            item.add_marker(pytest.mark.worker)
        if "/tests/app/ui/" in path_text or filename == "test_qt.py":
            item.add_marker(pytest.mark.ui)

        live_provider_files = {
            "test_api_keys.py",
            "test_both_clients.py",
            "test_gemini.py",
            "test_extended_thinking.py",
        }
        if filename in live_provider_files:
            item.add_marker(pytest.mark.live_provider)
            item.add_marker(pytest.mark.integration)

        if filename == "test_large_document_processing.py":
            item.add_marker(pytest.mark.integration)

        if (
            filename.startswith("test_")
            and "app/" not in path_text
            and "/tests/unit/" not in path_text
            and "/tests/common/" not in path_text
        ):
            item.add_marker(pytest.mark.integration)

        if "live_provider" in item.keywords and not run_live:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _isolate_settings_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep tests from writing into the user's real settings/profile directories."""

    user_root = tmp_path / "user_root"
    user_root.mkdir()
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    monkeypatch.setenv("LLESTRADE_USER_ROOT", str(user_root))
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(settings_dir))
    monkeypatch.setenv("LLESTRADE_QSETTINGS_ORG", "LlestradeTests")
    monkeypatch.setenv("LLESTRADE_QSETTINGS_APP", f"Settings-{tmp_path.name}")
    yield


@pytest.fixture(scope="session", autouse=True)
def _env_keys_from_keychain():
    """Populate API key environment variables from keychain/.env if missing.

    This is only enabled for live-provider runs (RUN_LIVE_PROVIDER_TESTS=1).
    """
    if not _live_provider_enabled():
        yield
        return

    # First try to load values from a local .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # If env vars are still missing, try OS keychain via SecureSettings
    try:
        from src.app.core.secure_settings import SecureSettings

        settings = SecureSettings()

        gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not gemini:
            gemini = settings.get_api_key("gemini") or settings.get_api_key("google")
        if gemini and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = gemini

        anthropic = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic:
            anthropic = settings.get_api_key("anthropic")
        if anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = anthropic
    except Exception:
        # Keychain may not be available in some CI/sandboxed environments
        pass
    yield
