from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from src.app.core import llm_catalog


@dataclass
class _FakePrices:
    input_mtok: Decimal | None = None
    output_mtok: Decimal | None = None


@dataclass
class _FakeModel:
    id: str
    name: str
    context_window: int | None
    prices: _FakePrices = field(default_factory=_FakePrices)
    deprecated: bool = False


@dataclass
class _FakeProvider:
    id: str
    models: list[_FakeModel]


@dataclass
class _FakeSnapshot:
    providers: list[_FakeProvider]

    def find_provider_model(
        self,
        model_id: str,
        _provider_api_url: object,
        provider_id: str,
        _genai_request_timestamp: object,
    ) -> tuple[_FakeProvider, _FakeModel]:
        for provider in self.providers:
            if provider.id != provider_id:
                continue
            for model in provider.models:
                if model.id == model_id:
                    return provider, model
        raise LookupError(model_id)


@pytest.fixture(autouse=True)
def reset_gemini_cache() -> None:
    llm_catalog.reset_gemini_model_cache()
    yield
    llm_catalog.reset_gemini_model_cache()


def test_runtime_default_model_for_gemini_prefers_stable_pro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_catalog,
        "_discover_gemini_models_live",
        lambda: (
            llm_catalog._GeminiDiscoveredModel(
                model_id="gemini-2.5-flash",
                label="Gemini 2.5 Flash",
                context_window=1_048_576,
            ),
            llm_catalog._GeminiDiscoveredModel(
                model_id="gemini-2.5-pro-preview-03-25",
                label="Gemini 2.5 Pro Preview",
                context_window=1_048_576,
            ),
            llm_catalog._GeminiDiscoveredModel(
                model_id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                context_window=1_048_576,
            ),
        ),
    )
    monkeypatch.setattr(llm_catalog, "_write_gemini_models_cache", lambda _models: None)

    assert llm_catalog.runtime_default_model_for_provider("gemini") == "gemini-2.5-pro"


def test_resolve_catalog_model_for_gemini_uses_live_context_and_catalog_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _FakeSnapshot(
        providers=[
            _FakeProvider(
                id="google",
                models=[
                    _FakeModel(
                        id="gemini-2.5-pro",
                        name="Gemini 2.5 Pro",
                        context_window=999_999,
                        prices=_FakePrices(
                            input_mtok=Decimal("1.25"),
                            output_mtok=Decimal("10"),
                        ),
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(llm_catalog, "_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        llm_catalog,
        "_discover_gemini_models_live",
        lambda: (
            llm_catalog._GeminiDiscoveredModel(
                model_id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                context_window=2_097_152,
            ),
        ),
    )
    monkeypatch.setattr(llm_catalog, "_write_gemini_models_cache", lambda _models: None)

    resolved = llm_catalog.resolve_catalog_model("gemini", "gemini-2.5-pro")

    assert resolved is not None
    assert resolved.context_window == 2_097_152
    assert resolved.input_price_label == "$1.25/1M"
    assert resolved.output_price_label == "$10/1M"


def test_runtime_default_model_for_gemini_uses_cache_when_live_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "config"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "gemini_models_cache.json"
    cache_path.write_text(
        """
        {
          "models": [
            {
              "context_window": 1048576,
              "label": "Gemini 2.5 Pro",
              "model_id": "gemini-2.5-pro"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(llm_catalog, "app_config_dir", lambda: cache_dir)
    monkeypatch.setattr(llm_catalog, "_discover_gemini_models_live", lambda: ())

    assert llm_catalog.runtime_default_model_for_provider("gemini") == "gemini-2.5-pro"


def test_parse_gemini_discovered_model_allows_preview_but_filters_non_text_variants() -> None:
    preview = type(
        "Model",
        (),
        {
            "name": "models/gemini-2.5-pro-preview-03-25",
            "display_name": "Gemini 2.5 Pro Preview",
            "input_token_limit": 1_048_576,
            "supported_actions": ["generateContent"],
        },
    )()
    image = type(
        "Model",
        (),
        {
            "name": "models/gemini-2.0-flash-preview-image-generation",
            "display_name": "Gemini 2.0 Flash Image Generation",
            "input_token_limit": 32_768,
            "supported_actions": ["generateContent"],
        },
    )()
    stable = type(
        "Model",
        (),
        {
            "name": "models/gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "input_token_limit": 1_048_576,
            "supported_actions": ["generateContent"],
        },
    )()

    parsed_preview = llm_catalog._parse_gemini_discovered_model(preview)
    assert parsed_preview is not None
    assert parsed_preview.model_id == "gemini-2.5-pro-preview-03-25"
    assert parsed_preview.context_window == 1_048_576
    assert llm_catalog._parse_gemini_discovered_model(image) is None

    parsed = llm_catalog._parse_gemini_discovered_model(stable)
    assert parsed is not None
    assert parsed.model_id == "gemini-2.5-pro"
    assert parsed.context_window == 1_048_576


def test_resolve_catalog_model_caps_anthropic_runtime_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _FakeSnapshot(
        providers=[
            _FakeProvider(
                id="anthropic",
                models=[
                    _FakeModel(
                        id="claude-sonnet-4-5",
                        name="Claude Sonnet 4.5",
                        context_window=1_000_000,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(llm_catalog, "_snapshot", lambda: snapshot)

    resolved = llm_catalog.resolve_catalog_model("anthropic", "claude-sonnet-4-5")

    assert resolved is not None
    assert resolved.context_window == 200_000
