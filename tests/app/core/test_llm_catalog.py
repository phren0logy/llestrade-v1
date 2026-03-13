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
            llm_catalog._DiscoveredModel(
                model_id="gemini-2.5-flash",
                label="Gemini 2.5 Flash",
                context_window=1_048_576,
            ),
            llm_catalog._DiscoveredModel(
                model_id="gemini-2.5-pro-preview-03-25",
                label="Gemini 2.5 Pro Preview",
                context_window=1_048_576,
            ),
            llm_catalog._DiscoveredModel(
                model_id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                context_window=1_048_576,
            ),
        ),
    )
    monkeypatch.setattr(llm_catalog, "_write_cached_discovered_models", lambda *args, **kwargs: None)

    assert llm_catalog.runtime_default_model_for_provider("gemini") == "gemini-2.5-pro"


def test_runtime_default_model_for_gateway_gemini_ignores_direct_preview_discovery(
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
                        context_window=1_048_576,
                    ),
                    _FakeModel(
                        id="gemini-3-flash-preview",
                        name="Gemini 3 Flash Preview",
                        context_window=1_048_576,
                    ),
                ],
            )
        ]
    )
    monkeypatch.setattr(llm_catalog, "_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        llm_catalog,
        "_discover_gemini_models_live",
        lambda: (
            llm_catalog._DiscoveredModel(
                model_id="gemini-3-flash-preview",
                label="Gemini 3 Flash Preview",
                context_window=1_048_576,
            ),
        ),
    )

    assert llm_catalog.runtime_default_model_for_provider("gemini", transport="gateway") == "gemini-2.5-pro"


def test_gateway_gemini_catalog_uses_snapshot_and_excludes_image_variants(
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
                        context_window=None,
                    ),
                    _FakeModel(
                        id="gemini-2.5-flash-image",
                        name="Gemini 2.5 Flash Image",
                        context_window=1_000_000,
                    ),
                ],
            )
        ]
    )
    monkeypatch.setattr(llm_catalog, "_snapshot", lambda: snapshot)

    catalog = llm_catalog.default_provider_catalog_for_transport(transport="gateway")
    gemini = next(provider for provider in catalog if provider.provider_id == "gemini")
    model_ids = {model.model_id for model in gemini.models}

    assert "gemini-2.5-pro" in model_ids
    assert "gemini-2.5-flash-image" not in model_ids


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
            llm_catalog._DiscoveredModel(
                model_id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                context_window=2_097_152,
            ),
        ),
    )
    monkeypatch.setattr(llm_catalog, "_write_cached_discovered_models", lambda *args, **kwargs: None)

    resolved = llm_catalog.resolve_catalog_model("gemini", "gemini-2.5-pro")

    assert resolved is not None
    assert resolved.context_window == 2_097_152
    assert resolved.input_price_label == "$1.25/1M"
    assert resolved.output_price_label == "$10/1M"


def test_default_provider_catalog_for_transport_uses_live_openai_discovery_with_snapshot_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _FakeSnapshot(
        providers=[
            _FakeProvider(
                id="openai",
                models=[
                    _FakeModel(
                        id="gpt-5.4",
                        name="GPT-5.4",
                        context_window=400_000,
                        prices=_FakePrices(input_mtok=Decimal("2.5"), output_mtok=Decimal("12.5")),
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(llm_catalog, "_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        llm_catalog,
        "_discover_openai_models_live",
        lambda: (
            llm_catalog._DiscoveredModel(
                model_id="gpt-5.4",
                label="gpt-5.4",
                context_window=None,
            ),
        ),
    )

    catalog = llm_catalog.default_provider_catalog_for_transport(transport="direct")
    openai = next(provider for provider in catalog if provider.provider_id == "openai")
    resolved = next(model for model in openai.models if model.model_id == "gpt-5.4")

    assert resolved.context_window == 400_000
    assert resolved.input_price_label == "$2.5/1M"
    assert resolved.output_price_label == "$12.5/1M"


def test_runtime_default_model_for_gemini_uses_cache_when_live_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(llm_catalog, "_discover_gemini_models_live", lambda: ())
    monkeypatch.setattr(
        llm_catalog,
        "_load_cached_discovered_models",
        lambda provider_id, transport="direct": (
            llm_catalog._DiscoveredModel(
                model_id="gemini-2.5-pro",
                label="Gemini 2.5 Pro",
                context_window=1_048_576,
            ),
        )
        if provider_id == "gemini" and transport == "direct"
        else (),
    )

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
