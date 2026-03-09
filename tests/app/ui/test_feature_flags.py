"""Tests for dashboard feature flags."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.app.core.feature_flags import FeatureFlags


class _StubSettings:
    def __init__(self, payload: Dict[str, Any] | None = None) -> None:
        self._payload = payload or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._payload.get(key, default)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in FeatureFlags.ENV_MAPPING.values():
        monkeypatch.delenv(key, raising=False)


def test_defaults_without_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

    flags = FeatureFlags.from_settings(None)

    assert flags.dashboard_workspace_enabled is True
    assert flags.bulk_analysis_groups_enabled is True
    assert flags.auto_run_conversion_on_create is True
    assert flags.pydantic_ai_gateway_enabled is True


def test_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = _StubSettings(
        {
            "feature_flags": {
                "bulk_analysis_groups_enabled": True,
                "auto_run_conversion_on_create": False,
                "pydantic_ai_gateway_enabled": False,
            }
        }
    )

    flags = FeatureFlags.from_settings(settings)

    assert flags.bulk_analysis_groups_enabled is True
    assert flags.auto_run_conversion_on_create is False
    assert flags.pydantic_ai_gateway_enabled is False


def test_environment_has_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FRD_ENABLE_BULK_ANALYSIS_GROUPS", "yes")
    monkeypatch.setenv("FRD_AUTO_RUN_CONVERSION", "0")
    monkeypatch.setenv("FRD_ENABLE_PYDANTIC_AI_GATEWAY", "false")

    settings = _StubSettings(
        {
            "feature_flags": {
                "bulk_analysis_groups_enabled": False,
                "auto_run_conversion_on_create": True,
                "pydantic_ai_gateway_enabled": True,
            }
        }
    )

    flags = FeatureFlags.from_settings(settings)

    assert flags.bulk_analysis_groups_enabled is True
    assert flags.auto_run_conversion_on_create is False
    assert flags.pydantic_ai_gateway_enabled is False
