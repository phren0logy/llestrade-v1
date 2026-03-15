"""Shared pytest configuration for dashboard tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

try:
    import keyring
    from keyring.backend import KeyringBackend
except Exception:  # pragma: no cover - test environment bootstrap
    keyring = None
    KeyringBackend = object  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SESSION_TEST_ROOT = Path(tempfile.mkdtemp(prefix="llestrade-pytest-"))


class _MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def clear(self) -> None:
        self._store.clear()

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


_TEST_KEYRING = _MemoryKeyring() if keyring is not None else None


def _session_env_paths() -> tuple[Path, Path]:
    user_root = _SESSION_TEST_ROOT / "user_root"
    settings_dir = _SESSION_TEST_ROOT / "settings"
    user_root.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)
    return user_root, settings_dir


def _install_test_environment() -> None:
    user_root, settings_dir = _session_env_paths()
    os.environ["LLESTRADE_USER_ROOT"] = str(user_root)
    os.environ["LLESTRADE_SETTINGS_DIR"] = str(settings_dir)
    os.environ["LLESTRADE_KEYRING_SERVICE_NAME"] = "LLestradeTests"
    os.environ["LLESTRADE_QSETTINGS_ORG"] = "LLestradeTests"
    os.environ["LLESTRADE_QSETTINGS_APP"] = "Settings-session"
    os.environ["LLESTRADE_QSETTINGS_PATH"] = str(settings_dir / "qt_settings.ini")


_install_test_environment()


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _live_provider_enabled() -> bool:
    return _env_flag("RUN_LIVE_PROVIDER_TESTS", default=False)


def pytest_configure(config: pytest.Config) -> None:
    _ = config
    _install_test_environment()
    if keyring is not None and _TEST_KEYRING is not None:
        keyring.set_keyring(_TEST_KEYRING)


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
    from src.app.core.secure_settings import reset_api_key_cache

    if _TEST_KEYRING is not None:
        _TEST_KEYRING.clear()
    reset_api_key_cache()

    user_root = tmp_path / "user_root"
    user_root.mkdir()
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    monkeypatch.setenv("LLESTRADE_USER_ROOT", str(user_root))
    monkeypatch.setenv("LLESTRADE_SETTINGS_DIR", str(settings_dir))
    monkeypatch.setenv("LLESTRADE_KEYRING_SERVICE_NAME", f"LLestradeTests-{tmp_path.name}")
    monkeypatch.setenv("LLESTRADE_QSETTINGS_ORG", "LLestradeTests")
    monkeypatch.setenv("LLESTRADE_QSETTINGS_APP", f"Settings-{tmp_path.name}")
    monkeypatch.setenv("LLESTRADE_QSETTINGS_PATH", str(settings_dir / "qt_settings.ini"))
    yield
    if _TEST_KEYRING is not None:
        _TEST_KEYRING.clear()
    reset_api_key_cache()


@pytest.fixture(autouse=True)
def _stub_gateway_preflight_for_ui_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Prevent GUI tests from reaching the live gateway preflight path."""

    if "ui" not in request.keywords:
        yield
        return

    from src.app.workers.llm_backend import GatewayAccessCheck, PydanticAIGatewayBackend

    def _ok_gateway_check(self, provider_id: str, model: str | None, *, timeout_seconds: float = 5.0, force: bool = False):  # noqa: ANN001
        _ = self, timeout_seconds, force
        return GatewayAccessCheck(
            ok=True,
            kind="ok",
            status_code=200,
            message="stubbed for UI tests",
            base_url="https://gateway.test",
            route="llm",
            provider_id=provider_id,
            model=model,
        )

    monkeypatch.setattr(PydanticAIGatewayBackend, "verify_gateway_access", _ok_gateway_check)
    yield


@pytest.fixture(scope="session", autouse=True)
def _env_keys_from_dotenv():
    """Populate live-provider env vars from dotenv files when explicitly enabled.

    This is only enabled for live-provider runs (RUN_LIVE_PROVIDER_TESTS=1).
    """
    if not _live_provider_enabled():
        yield
        return

    # First try to load values from local dotenv files if present.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env.live", override=False)
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass
    yield
