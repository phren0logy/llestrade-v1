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
    llm_catalog.reset_provider_catalog_cache()
    yield
    llm_catalog.reset_provider_catalog_cache()


def _cached_gateway_catalog(
    *providers: llm_catalog._GatewayCatalogProvider,
    cached_at: float,
) -> llm_catalog._CachedGatewayCatalog:
    return llm_catalog._CachedGatewayCatalog(
        providers=tuple(providers),
        cached_at=cached_at,
    )


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


def test_runtime_default_model_for_gateway_uses_gateway_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_catalog,
        "_discover_gateway_provider_catalog_live",
        lambda: (
            llm_catalog._GatewayCatalogProvider(
                provider_id="gemini",
                label="Google Gemini",
                route="gemini",
                models=(
                    llm_catalog.LLMModelOption(
                        model_id="gemini-2.5-pro",
                        label="Gemini 2.5 Pro",
                        context_window=1_048_576,
                    ),
                ),
            ),
        ),
    )

    assert llm_catalog.runtime_default_model_for_provider("gemini", transport="gateway") == "gemini-2.5-pro"


def test_runtime_default_model_for_gateway_returns_none_without_gateway_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_catalog, "_discover_gateway_provider_catalog_live", lambda: ())
    monkeypatch.setattr(llm_catalog, "_load_cached_gateway_provider_catalog", lambda max_age_seconds=None: None)
    monkeypatch.setattr(
        llm_catalog,
        "_snapshot",
        lambda: _FakeSnapshot(
            providers=[
                _FakeProvider(
                    id="google",
                    models=[
                        _FakeModel(
                            id="gemini-2.5-pro",
                            name="Gemini 2.5 Pro",
                            context_window=1_048_576,
                        )
                    ],
                )
            ]
        ),
    )

    assert llm_catalog.runtime_default_model_for_provider("gemini", transport="gateway") is None


def test_refresh_gateway_provider_catalog_uses_fresh_disk_cache_without_live_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_700_000_000.0
    disk_catalog = _cached_gateway_catalog(
        llm_catalog._GatewayCatalogProvider(
            provider_id="gemini",
            label="Google Gemini",
            route="bulk",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="gemini-2.5-pro",
                    label="Gemini 2.5 Pro",
                    context_window=1_048_576,
                ),
            ),
        ),
        cached_at=now - 3600,
    )
    live_calls = {"count": 0}

    monkeypatch.setattr(llm_catalog, "_gateway_catalog_scope", lambda: "gateway-scope")
    monkeypatch.setattr(llm_catalog.time, "time", lambda: now)
    monkeypatch.setattr(
        llm_catalog,
        "_load_cached_gateway_provider_catalog",
        lambda max_age_seconds=None: disk_catalog
        if max_age_seconds is None or (now - disk_catalog.cached_at) <= max_age_seconds
        else None,
    )
    monkeypatch.setattr(
        llm_catalog,
        "_discover_gateway_provider_catalog_live",
        lambda: live_calls.__setitem__("count", live_calls["count"] + 1) or (),
    )

    llm_catalog.refresh_gateway_provider_catalog()

    catalog = llm_catalog.default_provider_catalog_for_transport(transport="gateway")

    assert live_calls["count"] == 0
    assert [provider.provider_id for provider in catalog] == ["gemini"]
    assert catalog[0].models[0].model_id == "gemini-2.5-pro"


def test_refresh_gateway_provider_catalog_fetches_live_when_cache_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_700_000_000.0
    stale_catalog = _cached_gateway_catalog(
        llm_catalog._GatewayCatalogProvider(
            provider_id="gemini",
            label="Google Gemini",
            route="bulk",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="gemini-2.5-flash",
                    label="Gemini 2.5 Flash",
                    context_window=1_048_576,
                ),
            ),
        ),
        cached_at=now - (llm_catalog._GATEWAY_CATALOG_FRESHNESS_SECONDS + 10),
    )
    live_catalog = (
        llm_catalog._GatewayCatalogProvider(
            provider_id="gemini",
            label="Google Gemini",
            route="bulk",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="gemini-2.5-pro",
                    label="Gemini 2.5 Pro",
                    context_window=1_048_576,
                ),
            ),
        ),
    )
    live_calls = {"count": 0}

    monkeypatch.setattr(llm_catalog, "_gateway_catalog_scope", lambda: "gateway-scope")
    monkeypatch.setattr(llm_catalog.time, "time", lambda: now)
    monkeypatch.setattr(
        llm_catalog,
        "_load_cached_gateway_provider_catalog",
        lambda max_age_seconds=None: stale_catalog
        if max_age_seconds is None
        else None,
    )
    monkeypatch.setattr(llm_catalog, "_write_cached_gateway_provider_catalog", lambda *args, **kwargs: None)

    def _discover_live():
        live_calls["count"] += 1
        return live_catalog

    monkeypatch.setattr(llm_catalog, "_discover_gateway_provider_catalog_live", _discover_live)

    llm_catalog.refresh_gateway_provider_catalog()

    resolved = llm_catalog.runtime_default_model_for_provider("gemini", transport="gateway")

    assert live_calls["count"] == 1
    assert resolved == "gemini-2.5-pro"


def test_refresh_gateway_provider_catalog_falls_back_to_stale_cache_when_live_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_700_000_000.0
    stale_catalog = _cached_gateway_catalog(
        llm_catalog._GatewayCatalogProvider(
            provider_id="anthropic",
            label="Anthropic Claude",
            route="bulk",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="claude-sonnet-4-5",
                    label="Claude Sonnet 4.5",
                    context_window=200_000,
                ),
            ),
        ),
        cached_at=now - (llm_catalog._GATEWAY_CATALOG_FRESHNESS_SECONDS + 10),
    )

    monkeypatch.setattr(llm_catalog, "_gateway_catalog_scope", lambda: "gateway-scope")
    monkeypatch.setattr(llm_catalog.time, "time", lambda: now)
    monkeypatch.setattr(
        llm_catalog,
        "_load_cached_gateway_provider_catalog",
        lambda max_age_seconds=None: stale_catalog
        if max_age_seconds is None
        else None,
    )
    monkeypatch.setattr(llm_catalog, "_discover_gateway_provider_catalog_live", lambda: ())

    llm_catalog.refresh_gateway_provider_catalog()

    catalog = llm_catalog.default_provider_catalog_for_transport(transport="gateway")

    assert [provider.provider_id for provider in catalog] == ["anthropic"]
    assert catalog[0].models[0].model_id == "claude-sonnet-4-5"


def test_gateway_provider_catalog_uses_gateway_payload_without_snapshot_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_catalog,
        "_discover_gateway_provider_catalog_live",
        lambda: (
            llm_catalog._GatewayCatalogProvider(
                provider_id="gemini",
                label="Google Gemini",
                route="gemini",
                models=(
                    llm_catalog.LLMModelOption(
                        model_id="gemini-3-flash",
                        label="Gemini 3 Flash",
                        context_window=2_000_000,
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        llm_catalog,
        "_snapshot",
        lambda: _FakeSnapshot(
            providers=[
                _FakeProvider(
                    id="google",
                    models=[
                        _FakeModel(
                            id="gemini-2.5-flash-image",
                            name="Gemini 2.5 Flash Image",
                            context_window=1_000_000,
                        )
                    ],
                )
            ]
        ),
    )

    catalog = llm_catalog.default_provider_catalog_for_transport(transport="gateway")
    gemini = next(provider for provider in catalog if provider.provider_id == "gemini")

    assert [model.model_id for model in gemini.models] == ["gemini-3-flash"]


def test_gateway_provider_payload_normalizes_google_vertex_to_gemini() -> None:
    provider = llm_catalog._gateway_provider_option_from_payload(
        {
            "provider_id": "google-vertex",
            "upstream_provider_id": "google-vertex",
            "label": "Google Gemini",
            "models": [
                {
                    "model_id": "gemini-2.5-pro",
                    "display_name": "Gemini 2.5 Pro",
                    "context_window": 1_048_576,
                }
            ],
        }
    )

    assert provider is not None
    assert provider.provider_id == "gemini"
    assert [model.model_id for model in provider.models] == ["gemini-2.5-pro"]


def test_gateway_provider_payload_normalizes_bedrock_to_anthropic_bedrock() -> None:
    provider = llm_catalog._gateway_provider_option_from_payload(
        {
            "provider_id": "bedrock",
            "upstream_provider_id": "bedrock",
            "label": "AWS Bedrock (Claude)",
            "models": [
                {
                    "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "display_name": "Claude Sonnet 4.5",
                    "context_window": 1_000_000,
                }
            ],
        }
    )

    assert provider is not None
    assert provider.provider_id == "anthropic_bedrock"
    assert provider.label == "AWS Bedrock (Claude)"
    assert [model.model_id for model in provider.models] == ["us.anthropic.claude-sonnet-4-5-20250929-v1:0"]


def test_gateway_provider_payload_preserves_missing_context_until_gateway_provides_it() -> None:
    provider = llm_catalog._gateway_provider_option_from_payload(
        {
            "provider_id": "anthropic",
            "upstream_provider_id": "anthropic",
            "label": "Anthropic Claude",
            "models": [
                {
                    "model_id": "claude-opus-4-6",
                }
            ],
        }
    )

    assert provider is not None
    assert provider.provider_id == "anthropic"
    assert [model.model_id for model in provider.models] == ["claude-opus-4-6"]
    model = provider.models[0]
    assert model.label == "claude-opus-4-6"
    assert model.context_window is None
    assert model.input_price_label is None
    assert model.output_price_label is None
    assert model.provenance.context is None
    assert model.provenance.pricing is None


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


def test_resolve_model_context_window_uses_temporary_openai_fallback_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_catalog, "_iter_selector_models", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(llm_catalog, "_resolve_snapshot_catalog_model", lambda *_args, **_kwargs: None)

    assert llm_catalog.resolve_model_context_window("openai", "o3") == 200_000
    assert llm_catalog.resolve_model_context_window("openai", "o4-mini") == 200_000
    assert llm_catalog.resolve_model_context_window("azure_openai", "o1") == 128_000
    assert llm_catalog.resolve_model_context_window("anthropic", "unknown-model") is None


def test_gateway_model_option_from_payload_uses_gateway_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_catalog,
        "_reasoning_capabilities_for_model",
        lambda provider_id, model_id: llm_catalog.LLMReasoningCapabilities(
            supports_reasoning_controls=False,
            can_disable_reasoning=False,
            controls=(),
            allowed_efforts=("low", "medium", "high"),
            default_state="off",
        ),
    )

    resolved = llm_catalog._gateway_model_option_from_payload(
        "openai",
        {
            "model_id": "gpt-5.4",
            "display_name": "GPT-5.4",
            "context_window": "400000",
            "max_output_tokens": 16000,
            "pricing_input_per_million": "2.5",
            "pricing_output_per_million": 12.5,
            "lifecycle_status": "stable",
            "reasoning_capabilities": {
                "supports_reasoning_controls": True,
                "can_disable_reasoning": True,
                "controls": ["toggle", "effort"],
                "notes": "Gateway supplied reasoning metadata.",
            },
            "sources": {
                "availability": "gateway",
                "context_window": "gateway",
                "pricing": "gateway",
                "reasoning_capabilities": "gateway",
            },
        },
    )

    assert resolved is not None
    assert resolved.context_window == 400_000
    assert resolved.max_output_tokens == 16_000
    assert resolved.input_price_label == "$2.5/1M"
    assert resolved.output_price_label == "$12.5/1M"
    assert resolved.reasoning_capabilities.supports_reasoning_controls is True
    assert resolved.reasoning_capabilities.can_disable_reasoning is True
    assert resolved.reasoning_capabilities.controls == ("toggle", "effort")
    assert resolved.reasoning_capabilities.allowed_efforts == ("low", "medium", "high")
    assert resolved.provenance.pricing == "gateway"


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


def test_resolve_catalog_model_uses_anthropic_metadata_context_window(
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
    assert resolved.context_window == 1_000_000


def test_resolve_catalog_model_normalizes_bedrock_inference_profile_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _FakeSnapshot(
        providers=[
            _FakeProvider(
                id="anthropic",
                models=[
                    _FakeModel(
                        id="claude-sonnet-4-6",
                        name="Claude Sonnet 4.6",
                        context_window=1_000_000,
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(llm_catalog, "_snapshot", lambda: snapshot)

    resolved = llm_catalog.resolve_catalog_model("anthropic_bedrock", "us.anthropic.claude-sonnet-4-6")

    assert resolved is not None
    assert resolved.model_id == "claude-sonnet-4-6"
    assert resolved.context_window == 1_000_000
