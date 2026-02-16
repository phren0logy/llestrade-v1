import sys
import types

from src.common.llm import bedrock_catalog as catalog
from src.common.llm.bedrock_catalog import DEFAULT_BEDROCK_MODELS
from src.common.llm.providers import anthropic_bedrock


def test_list_bedrock_models_falls_back_to_defaults(monkeypatch):
    """If AWS discovery fails, the helper should return the bundled defaults."""

    def _fail_session(profile):
        raise RuntimeError("no aws session")

    monkeypatch.setattr(catalog, "_create_session", _fail_session)
    monkeypatch.setattr(catalog, "_cached_models", [], raising=False)
    monkeypatch.setattr(catalog, "_cache_key", (None, None), raising=False)
    monkeypatch.setattr(catalog, "_cache_expiry", 0.0, raising=False)

    models = catalog.list_bedrock_models(region=None, profile=None, force_refresh=True)

    assert [model.model_id for model in models] == [
        model.model_id for model in DEFAULT_BEDROCK_MODELS
    ]


def test_anthropic_bedrock_provider_initialises_with_stubbed_client(monkeypatch):
    """The provider should initialise and generate responses with a stubbed client."""

    fake_models = [
        catalog.BedrockModel(
            model_id="anthropic.fake-sonnet",
            name="Claude Fake Sonnet",
            provider_name="Anthropic",
            region="us-west-2",
        )
    ]

    monkeypatch.setattr(
        anthropic_bedrock, "list_bedrock_models", lambda region=None, profile=None: fake_models
    )

    class _FakeUsage:
        input_tokens = 10
        output_tokens = 20

    class _FakeResponse:
        def __init__(self):
            self.content = [types.SimpleNamespace(text="stub output")]
            self.usage = _FakeUsage()

    class _FakeMessages:
        def count_tokens(self, model, messages):
            return {"tokens": 42}

        def create(self, *args, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        def __init__(self, **kwargs):
            self.messages = _FakeMessages()
            self.models = types.SimpleNamespace(list=lambda: None)

    fake_anthropic = types.SimpleNamespace(AnthropicBedrock=lambda **kwargs: _FakeClient())
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    provider = anthropic_bedrock.AnthropicBedrockProvider(debug=True)

    assert provider.initialized
    assert provider.default_model == fake_models[0].model_id
    assert provider.available_models[0].model_id == fake_models[0].model_id

    result = provider.generate(prompt="Hello", model=None)
    assert result["success"]
    assert "stub output" in result["content"]
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 20


def test_list_bedrock_models_filters_out_pre_41_models(monkeypatch):
    """Discovery should include only Claude 4.1+ models."""

    class _FakeBedrockClient:
        def __init__(self):
            self.meta = types.SimpleNamespace(region_name="us-east-1")

        def list_foundation_models(self, **_kwargs):
            return {
                "modelSummaries": [
                    {
                        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "modelName": "Claude 3.5 Sonnet",
                        "providerName": "Anthropic",
                    },
                    {
                        "modelId": "anthropic.claude-opus-4-1-20250805-v1:0",
                        "modelName": "Claude Opus 4.1",
                        "providerName": "Anthropic",
                    },
                    {
                        "modelId": "anthropic.claude-opus-4-6-v1",
                        "modelName": "Claude Opus 4.6",
                        "providerName": "Anthropic",
                    },
                ]
            }

    class _FakeSession:
        def client(self, *_args, **_kwargs):
            return _FakeBedrockClient()

    monkeypatch.setattr(catalog, "_create_session", lambda profile: _FakeSession())
    monkeypatch.setattr(catalog, "_cached_models", [], raising=False)
    monkeypatch.setattr(catalog, "_cache_key", (None, None), raising=False)
    monkeypatch.setattr(catalog, "_cache_expiry", 0.0, raising=False)

    models = catalog.list_bedrock_models(region=None, profile=None, force_refresh=True)
    model_ids = [model.model_id for model in models]

    assert "anthropic.claude-3-5-sonnet-20240620-v1:0" not in model_ids
    assert "anthropic.claude-opus-4-1-20250805-v1:0" in model_ids
    assert "anthropic.claude-opus-4-6-v1" in model_ids
