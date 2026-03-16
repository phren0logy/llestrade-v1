from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Sequence

import httpx
import pytest
from pydantic_ai.exceptions import ModelAPIError, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from src.app.workers.checkpoint_manager import CheckpointManager

from src.app.core.bulk_analysis_runner import PromptBundle
from src.app.core.placeholders.system import SourceFileContext
from src.app.core.project_manager import ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.workers import bulk_analysis_worker as worker_module
from src.app.workers.bulk_analysis_worker import (
    BulkAnalysisWorker,
    ProviderConfig,
    _ProviderPromptLimitError,
    _compute_prompt_hash,
    _load_manifest,
    _manifest_path,
    _save_manifest,
    _should_process_document,
)
from src.app.workers.llm_backend import (
    LLMExecutionBackend,
    LLMInvocationRequest,
    LLMProviderRequest,
    PydanticAIGatewayBackend,
    ProviderMetadata,
    build_model_settings,
    normalize_model_name,
    provider_capabilities,
    resolve_model_name,
)


def _model_response(
    content: str,
    *,
    model_name: str = "claude",
    output_tokens: int = 1,
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content)],
        usage=RequestUsage(input_tokens=1, output_tokens=output_tokens),
        model_name=model_name,
    )


class _NoNativeBackend(LLMExecutionBackend):
    def normalize_model(self, provider_id: str, model: str | None) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: str | None) -> str | None:
        return resolve_model_name(provider_id, model)

    def capabilities(self, provider_id: str, model: str | None):
        return provider_capabilities(provider_id, model)

    def build_model_settings(self, provider_id: str, model: str | None, **kwargs):
        return build_model_settings(provider_id, model, **kwargs)

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(provider_name=request.provider_id, default_model=request.model or "default-model")

    def invoke_response(self, provider, request: LLMInvocationRequest) -> ModelResponse:  # noqa: ANN001
        _ = provider
        return _model_response("summary", model_name=request.model or "claude", output_tokens=1)


class _ResultBackend(LLMExecutionBackend):
    def __init__(self, result: ModelResponse | Exception) -> None:
        self._result = result

    def normalize_model(self, provider_id: str, model: str | None) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: str | None) -> str | None:
        return resolve_model_name(provider_id, model)

    def capabilities(self, provider_id: str, model: str | None):
        return provider_capabilities(provider_id, model)

    def build_model_settings(self, provider_id: str, model: str | None, **kwargs):
        return build_model_settings(provider_id, model, **kwargs)

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(provider_name=request.provider_id, default_model=request.model or "default-model")

    def invoke_response(self, provider, request: LLMInvocationRequest) -> ModelResponse:  # noqa: ANN001
        _ = provider, request
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _CountingBackend(LLMExecutionBackend):
    def __init__(self, *, token_count: int) -> None:
        self.token_count = token_count
        self.invoked = False
        self.requests: list[LLMInvocationRequest] = []

    def normalize_model(self, provider_id: str, model: str | None) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: str | None) -> str | None:
        return resolve_model_name(provider_id, model)

    def capabilities(self, provider_id: str, model: str | None):
        return provider_capabilities(provider_id, model)

    def build_model_settings(self, provider_id: str, model: str | None, **kwargs):
        return build_model_settings(provider_id, model, **kwargs)

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(provider_name=request.provider_id, default_model=request.model or "default-model")

    def count_input_tokens(self, provider, request: LLMInvocationRequest) -> int | None:  # noqa: ANN001
        _ = provider, request
        return self.token_count

    def invoke_response(self, provider, request: LLMInvocationRequest) -> ModelResponse:  # noqa: ANN001
        _ = provider, request
        self.invoked = True
        self.requests.append(request)
        return _model_response("summary", model_name="claude-sonnet-4-5", output_tokens=1)


class _FlakyUsageLimitBackend(_NoNativeBackend):
    def __init__(self) -> None:
        self.calls = 0

    def invoke_response(self, provider, request: LLMInvocationRequest) -> ModelResponse:  # noqa: ANN001
        _ = provider, request
        self.calls += 1
        if self.calls == 1:
            raise UsageLimitExceeded(
                "Exceeded the input_tokens_limit of 638220 (input_tokens=693218)"
            )
        return _model_response("summary", model_name=request.model or "claude", output_tokens=1)


class _FlakyGatewayRateLimitBackend(PydanticAIGatewayBackend):
    def __init__(self) -> None:
        super().__init__(api_key="gateway-key", base_url="https://gateway.example.com")
        self.calls = 0

    def invoke_response(self, provider, request: LLMInvocationRequest) -> ModelResponse:  # noqa: ANN001
        _ = provider
        self.calls += 1
        if self.calls == 2:
            http_request = httpx.Request("POST", "https://gateway.example.com/anthropic/v1/messages?beta=true")
            http_response = httpx.Response(
                429,
                request=http_request,
                headers={"retry-after": "0"},
                json={"error": {"message": "Too Many Requests"}},
            )
            raise ModelAPIError(request.model or "claude-sonnet-4-6", "Connection error.") from httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=http_request,
                response=http_response,
            )
        return _model_response("summary", model_name=request.model or "claude-sonnet-4-6", output_tokens=1)


class _FlakyGatewayServerBackend(PydanticAIGatewayBackend):
    def __init__(self) -> None:
        super().__init__(api_key="gateway-key", base_url="https://gateway.example.com")
        self.calls = 0

    def invoke_response(self, provider, request: LLMInvocationRequest) -> ModelResponse:  # noqa: ANN001
        _ = provider
        self.calls += 1
        if self.calls == 1:
            http_request = httpx.Request("POST", "https://gateway.example.com/anthropic/v1/messages?beta=true")
            http_response = httpx.Response(
                524,
                request=http_request,
                headers={"retry-after": "0"},
                text="A timeout occurred",
            )
            raise ModelAPIError(request.model or "claude-sonnet-4-6", "Connection error.") from httpx.HTTPStatusError(
                "524 A timeout occurred",
                request=http_request,
                response=http_response,
            )
        return _model_response("summary", model_name=request.model or "claude-sonnet-4-6", output_tokens=1)


def _capture_traces(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object] | None]]:
    recorded: list[tuple[str, dict[str, object] | None]] = []

    @contextmanager
    def _fake_trace_operation(name: str, attributes=None):  # noqa: ANN001
        recorded.append((name, dict(attributes) if attributes is not None else None))
        yield None

    monkeypatch.setattr(worker_module, "trace_operation", _fake_trace_operation)
    return recorded


def test_should_process_document_handles_skips(tmp_path: Path) -> None:
    entry = {
        "source_mtime": 123.456001,
        "prompt_hash": "abc",
    }

    assert not _should_process_document(entry, 123.456, "abc", output_exists=True)
    assert _should_process_document(entry, 123.456, "def", output_exists=True)
    assert _should_process_document(entry, 999.0, "abc", output_exists=True)
    assert _should_process_document(entry, 123.456, "abc", output_exists=False)
    assert _should_process_document(None, 123.456, "abc", output_exists=True)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    path = _manifest_path(tmp_path, group)

    manifest = {
        "version": 2,
        "signature": None,
        "documents": {
            "doc.md": {
                "source_mtime": 1.23,
                "prompt_hash": "deadbeef",
                "ran_at": "2025-01-01T00:00:00+00:00",
            }
        },
    }
    _save_manifest(path, manifest)
    loaded = _load_manifest(path)
    assert loaded == manifest


def test_compute_prompt_hash_changes_on_prompt_and_settings() -> None:
    group = BulkAnalysisGroup.create("Group")
    metadata = ProjectMetadata(case_name="Case A", subject_name="Subject", case_description="Desc")
    bundle = PromptBundle(system_template="System", user_template="User {document_content}")
    config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5")

    first = _compute_prompt_hash(bundle, config, group, metadata)

    group.use_reasoning = True
    second = _compute_prompt_hash(bundle, config, group, metadata)
    assert first != second

    group.use_reasoning = False
    config_alt = ProviderConfig(provider_id="anthropic", model="claude-opus-4-6")
    third = _compute_prompt_hash(bundle, config_alt, group, metadata)
    assert first != third

    metadata.case_name = "Case B"
    fourth = _compute_prompt_hash(bundle, config_alt, group, metadata)
    assert third != fourth


def test_bulk_map_create_provider_skips_native_bootstrap_for_no_native_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )

    def _fail_create_provider(**_kwargs):  # noqa: ANN001
        raise AssertionError("create_provider should not be called for no-native backend")

    monkeypatch.setattr(worker_module, "create_provider", _fail_create_provider, raising=False)
    provider = worker._create_provider(
        ProviderConfig(provider_id="anthropic", model=None),
    )

    assert provider.provider_name == "anthropic"
    assert provider.default_model


def test_bulk_map_resolve_provider_normalizes_bedrock_model(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "anthropic_bedrock"
    group.model = "claude-sonnet-4-5"
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )

    config = worker._resolve_provider()

    assert config.provider_id == "anthropic_bedrock"
    assert config.model == "anthropic.claude-sonnet-4-5-v1"


def test_bulk_map_resolve_provider_requires_saved_provider_selection(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )

    with pytest.raises(RuntimeError, match="no saved provider selection"):
        worker._resolve_provider()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            RuntimeError("Gateway timeout"),
            "Gateway timeout",
        ),
        (
            _model_response("   ", model_name="claude-sonnet-4-5", output_tokens=1),
            "LLM returned empty response",
        ),
    ],
)
def test_bulk_map_invoke_provider_raises_for_failed_or_empty_backend_result(
    tmp_path: Path,
    result: ModelResponse | Exception,
    message: str,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_ResultBackend(result),
    )

    with pytest.raises(RuntimeError, match=message):
        worker._invoke_provider(
            provider=object(),
            provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
            prompt="Prompt",
            system_prompt="System",
        )


def test_bulk_map_invoke_provider_raises_cancelled_when_worker_cancelled(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )
    worker.cancel()

    with pytest.raises(worker_module.BulkAnalysisCancelled):
        worker._invoke_provider(
            provider=object(),
            provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
            prompt="Prompt",
            system_prompt="System",
        )


def test_bulk_map_trace_attributes_match_between_legacy_and_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.slug = "group-slug"
    config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5")

    legacy_worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_ResultBackend(
            _model_response("summary", model_name="claude-sonnet-4-5", output_tokens=1)
        ),
    )
    legacy_traces = _capture_traces(monkeypatch)
    legacy_result = legacy_worker._invoke_provider(
        provider=object(),
        provider_config=config,
        prompt="Prompt",
        system_prompt="System",
        context_label="document 'doc.md'",
    )

    gateway_worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_ResultBackend(
            _model_response("summary", model_name="claude-sonnet-4-5", output_tokens=1)
        ),
    )
    gateway_traces = _capture_traces(monkeypatch)
    gateway_result = gateway_worker._invoke_provider(
        provider=object(),
        provider_config=config,
        prompt="Prompt",
        system_prompt="System",
        context_label="document 'doc.md'",
    )

    assert legacy_result == "summary"
    assert gateway_result == "summary"
    assert legacy_traces == gateway_traces == [
        (
            "bulk_analysis.invoke_llm",
            {
                "llestrade.transport": "direct",
                "llestrade.provider_id": "anthropic",
                "llestrade.model": "claude-sonnet-4-5",
                "llestrade.reasoning": False,
                "llestrade.max_tokens": 32000,
                "llestrade.temperature": 0.1,
                "llestrade.worker": "bulk_analysis",
                "llestrade.stage": "bulk_map",
                "llestrade.group_id": group.group_id,
                "llestrade.group_name": "Group",
                "llestrade.group_slug": "group-slug",
                "llestrade.context_label": "document 'doc.md'",
            },
        )
    ]


def test_bulk_worker_uses_backend_token_count_for_gateway_preflight(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    backend = _CountingBackend(token_count=500)
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )

    with pytest.raises(_ProviderPromptLimitError, match="500 tokens > 400 budget"):
        worker._invoke_provider(
            provider=worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5")),
            provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
            prompt="Prompt",
            system_prompt="System",
            input_budget=400,
        )

    assert backend.invoked is False


def test_bulk_worker_chunk_fit_uses_local_estimates_for_real_gateway_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com"),
    )

    monkeypatch.setattr(worker._llm_backend, "count_input_tokens", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("backend count should not be used during chunk fit")))

    document = worker_module.BulkAnalysisDocument(
        source_path=tmp_path / "doc.md",
        relative_path="doc.md",
        output_path=tmp_path / "out.md",
    )
    bundle = worker_module.PromptBundle(
        system_template="System",
        user_template="Analyze {document_content}",
    )

    chunks = worker._generate_fitting_chunks(
        provider=object(),
        provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6"),
        bundle=bundle,
        system_prompt="System",
        document=document,
        body="Short body text",
        placeholder_values={},
        input_budget=10_000,
        initial_chunk_tokens=4_000,
        max_tokens=worker._map_max_output_tokens(
            ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6")
        ),
    )

    assert chunks


def test_bulk_worker_real_gateway_backend_skips_duplicate_exact_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    backend = PydanticAIGatewayBackend(api_key="gateway-key", base_url="https://gateway.example.com")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )

    monkeypatch.setattr(
        backend,
        "count_input_tokens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("duplicate exact preflight should be deferred to backend invoke")),
    )
    monkeypatch.setattr(
        backend,
        "invoke_response",
        lambda *_args, **_kwargs: _model_response("summary", model_name="claude-sonnet-4-6", output_tokens=1),
    )

    result = worker._invoke_provider(
        provider=object(),
        provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6"),
        prompt="Prompt",
        system_prompt="System",
        input_budget=10_000,
    )

    assert result == "summary"


def test_bulk_worker_parses_provider_prompt_limit_error(tmp_path: Path) -> None:
    class _PromptLimitBackend(_ResultBackend):
        def __init__(self) -> None:
            error = RuntimeError("context_length_exceeded")
            error.body = {
                "error": {
                    "message": (
                        "Input tokens exceed the configured limit of 272000 tokens. "
                        "Your messages resulted in 380475 tokens. Please reduce the length "
                        "of the messages."
                    ),
                    "code": "context_length_exceeded",
                }
            }
            super().__init__(error)

    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_PromptLimitBackend(),
    )

    with pytest.raises(_ProviderPromptLimitError) as exc_info:
        worker._invoke_provider(
            provider=object(),
            provider_config=ProviderConfig(provider_id="openai", model="gpt-5-mini"),
            prompt="Prompt",
            system_prompt="System",
            input_budget=300_000,
        )

    assert exc_info.value.configured_limit == 272_000
    assert exc_info.value.actual_tokens == 380_475


def test_bulk_worker_parses_usage_limit_exceeded_error(tmp_path: Path) -> None:
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_ResultBackend(
            UsageLimitExceeded("Exceeded the input_tokens_limit of 638220 (input_tokens=693218)")
        ),
    )

    with pytest.raises(_ProviderPromptLimitError) as exc_info:
        worker._invoke_provider(
            provider=object(),
            provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
            prompt="Prompt",
            system_prompt="System",
            input_budget=638_220,
        )

    assert exc_info.value.configured_limit == 638_220
    assert exc_info.value.actual_tokens == 693_218


def test_bulk_worker_retries_chunked_document_after_provider_limit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    source_path = project_dir / "converted_documents" / "doc.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("body", encoding="utf-8")

    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="converted_documents/doc.md",
        output_path=project_dir / "bulk_analysis" / "group" / "doc.md",
    )
    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "openai"
    group.model = "gpt-5-mini"
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )
    bundle = PromptBundle(system_template="System", user_template="Analyze {document_content}")
    provider = worker._create_provider(ProviderConfig(provider_id="openai", model="gpt-5-mini"))
    source_context = SourceFileContext(
        absolute_path=source_path.resolve(),
        relative_path=document.relative_path,
    )
    manifest_path = _manifest_path(project_dir, group)
    recovery_store = worker_module.BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    recovery_manifest = recovery_store.load_map_manifest()

    monkeypatch.setattr(worker, "_load_document", lambda _document: ("body", {}, source_context))
    monkeypatch.setattr(worker_module, "should_chunk", lambda *_args, **_kwargs: (True, 3_704_406, 200_000))
    monkeypatch.setattr(worker, "_count_prompt_tokens", lambda *_args, **_kwargs: 1_000)

    chunk_targets: list[int] = []

    def _fake_generate(*, initial_chunk_tokens, **_kwargs):
        chunk_targets.append(initial_chunk_tokens)
        if initial_chunk_tokens > 60_000:
            return ["chunk-one"]
        return ["chunk-one", "chunk-two"]

    monkeypatch.setattr(worker, "_generate_fitting_chunks", _fake_generate)

    invoke_calls: list[str] = []

    def _fake_execute_chunk_task(self, *, spec, **_kwargs):  # noqa: ANN001
        invoke_calls.append(spec.context_label)
        if spec.context_label.startswith("chunk 1/1"):
            raise _ProviderPromptLimitError(
                configured_limit=272_000,
                actual_tokens=380_475,
                message="Input tokens exceed the configured limit of 272000 tokens.",
            )
        return worker_module._ChunkTaskResult(
            chunk_index=spec.chunk_index,
            chunk_checksum=spec.chunk_checksum,
            summary="summary",
            usage={"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        )

    monkeypatch.setattr(BulkAnalysisWorker, "_execute_chunk_task", _fake_execute_chunk_task)
    monkeypatch.setattr(
        worker_module,
        "combine_chunk_summaries_hierarchical",
        lambda summaries, **_kwargs: "combined:" + "|".join(summaries),
    )

    result, run_details, _placeholders = worker._process_document(
        provider=provider,
        provider_config=ProviderConfig(provider_id="openai", model="gpt-5-mini"),
        bundle=bundle,
        system_prompt="System",
        document=document,
        global_placeholders={},
        manifest={"version": 2, "signature": None, "documents": {}},
        prompt_hash="prompt-hash",
        manifest_path=manifest_path,
        recovery_store=recovery_store,
        recovery_manifest=recovery_manifest,
    )

    assert result == "combined:summary|summary"
    assert run_details["chunk_count"] == 2
    assert chunk_targets[0] == 96_888
    assert chunk_targets[0] > chunk_targets[1]
    assert invoke_calls[0] == "chunk 1/1 for 'converted_documents/doc.md'"


def test_bulk_worker_retries_chunked_document_after_usage_limit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    source_path = project_dir / "converted_documents" / "doc.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("body", encoding="utf-8")

    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="converted_documents/doc.md",
        output_path=project_dir / "bulk_analysis" / "group" / "doc.md",
    )
    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "anthropic"
    group.model = "claude-sonnet-4-5"
    group.model_context_window = 1_000_000
    backend = _FlakyUsageLimitBackend()
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )
    bundle = PromptBundle(system_template="System", user_template="Analyze {document_content}")
    provider = worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"))
    source_context = SourceFileContext(
        absolute_path=source_path.resolve(),
        relative_path=document.relative_path,
    )
    manifest_path = _manifest_path(project_dir, group)
    recovery_store = worker_module.BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    recovery_manifest = recovery_store.load_map_manifest()

    monkeypatch.setattr(worker, "_load_document", lambda _document: ("body", {}, source_context))
    monkeypatch.setattr(worker_module, "should_chunk", lambda *_args, **_kwargs: (True, 4_939_208, 200_000))
    monkeypatch.setattr(worker, "_count_prompt_tokens", lambda *_args, **_kwargs: 1_000)

    chunk_targets: list[int] = []

    def _fake_generate(*, initial_chunk_tokens, **_kwargs):
        chunk_targets.append(initial_chunk_tokens)
        if initial_chunk_tokens > 180_000:
            return ["chunk-one"]
        return ["chunk-one", "chunk-two"]

    monkeypatch.setattr(worker, "_generate_fitting_chunks", _fake_generate)
    monkeypatch.setattr(
        worker_module,
        "combine_chunk_summaries_hierarchical",
        lambda summaries, **_kwargs: "combined:" + "|".join(summaries),
    )

    result, run_details, _placeholders = worker._process_document(
        provider=provider,
        provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
        bundle=bundle,
        system_prompt="System",
        document=document,
        global_placeholders={},
        manifest={"version": 2, "signature": None, "documents": {}},
        prompt_hash="prompt-hash",
        manifest_path=manifest_path,
        recovery_store=recovery_store,
        recovery_manifest=recovery_manifest,
    )

    assert result == "combined:summary|summary"
    assert run_details["chunk_count"] == 2
    assert chunk_targets[0] == 200_000
    assert chunk_targets[0] > chunk_targets[1]


def test_bulk_worker_chunk_fit_uses_exact_count_for_direct_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _CountingBackend(token_count=12_000)
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )
    provider = worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"))
    document = worker_module.BulkAnalysisDocument(
        source_path=tmp_path / "doc.md",
        relative_path="doc.md",
        output_path=tmp_path / "out.md",
    )
    bundle = worker_module.PromptBundle(
        system_template="System",
        user_template="Analyze {document_content}",
    )
    count_calls = {"value": 0}

    def _count_input_tokens(provider, request):  # noqa: ANN001
        _ = provider, request
        count_calls["value"] += 1
        return 12_000

    monkeypatch.setattr(backend, "count_input_tokens", _count_input_tokens)

    with pytest.raises(RuntimeError, match="Prompt exceeds model input budget"):
        worker._generate_fitting_chunks(
            provider=provider,
            provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
            bundle=bundle,
            system_prompt="System",
            document=document,
            body="Short body text",
            placeholder_values={},
            input_budget=10_000,
            initial_chunk_tokens=4_000,
            max_tokens=worker._map_max_output_tokens(
                ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5")
            ),
        )

    assert count_calls["value"] > 0


def test_bulk_worker_applies_reasoning_settings_to_llm_request(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.use_reasoning = True
    backend = _CountingBackend(token_count=10)
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )

    result = worker._invoke_provider(
        provider=worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5")),
        provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
        prompt="Prompt",
        system_prompt="System",
    )

    assert result == "summary"
    assert backend.invoked is True
    assert backend.requests[0].model_settings["anthropic_thinking"] == {
        "type": "enabled",
        "budget_tokens": 4000,
    }


def test_bulk_worker_force_rerun_reprocesses(tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = qtbot
    project_dir = tmp_path
    converted = project_dir / "converted_documents" / "folder"
    converted.mkdir(parents=True, exist_ok=True)
    converted_doc = converted / "doc.md"
    converted_doc.write_text("content", encoding="utf-8")

    group = BulkAnalysisGroup.create("Group")
    metadata = ProjectMetadata(case_name="Case")
    call_count = {"value": 0}

    def fake_prepare(project_dir: Path, group: BulkAnalysisGroup, selected: Sequence[str]):
        from src.app.core.bulk_analysis_runner import BulkAnalysisDocument

        source_path = project_dir / "converted_documents" / "folder" / "doc.md"
        output_path = project_dir / "bulk_analysis" / group.folder_name / "folder" / "doc_analysis.md"
        return [BulkAnalysisDocument(source_path, "folder/doc.md", output_path)]

    monkeypatch.setattr(worker_module, "prepare_documents", fake_prepare)
    monkeypatch.setattr(
        worker_module,
        "load_prompts",
        lambda *_args, **_kwargs: PromptBundle("System", "User {document_content}"),
    )
    monkeypatch.setattr(
        BulkAnalysisWorker,
        "_resolve_provider",
        lambda self: ProviderConfig("anthropic", "model"),
    )
    monkeypatch.setattr(
        BulkAnalysisWorker,
        "_create_provider",
        lambda self, *_: object(),
    )

    def fake_process(self, *_args, **_kwargs):
        call_count["value"] += 1
        placeholders = {"document_name": "folder/doc.md"}
        return "summary", {"chunk_count": 1, "chunking": False, "token_count": 10, "max_tokens": 4000}, placeholders

    monkeypatch.setattr(BulkAnalysisWorker, "_process_document", fake_process)

    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=["folder/doc.md"],
        metadata=metadata,
        force_rerun=False,
        placeholder_values={},
        project_name="Case",
    )
    worker._run()
    assert call_count["value"] == 1

    worker_skip = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=["folder/doc.md"],
        metadata=metadata,
        force_rerun=False,
        placeholder_values={},
        project_name="Case",
    )
    worker_skip._run()
    assert call_count["value"] == 1, "expected skip when inputs unchanged"

    worker_force = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=["folder/doc.md"],
        metadata=metadata,
        force_rerun=True,
        placeholder_values={},
        project_name="Case",
    )
    worker_force._run()
    assert call_count["value"] == 2, "force re-run should process again"


def test_bulk_worker_applies_placeholder_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    pdf_path = project_dir / "sources" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_text("pdf bytes", encoding="utf-8")

    import frontmatter

    source_path = converted / "doc.md"
    post = frontmatter.Post("Body text", metadata={"sources": [{"path": str(pdf_path)}]})
    source_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    output_path = project_dir / "bulk_analysis" / "group" / "doc.md"
    metadata = ProjectMetadata(case_name="Case Name")
    group = BulkAnalysisGroup.create("Group", files=["doc.md"])

    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=["doc.md"],
        metadata=metadata,
        force_rerun=True,
        placeholder_values={"client_name": "ACME"},
        project_name="Project XYZ",
    )

    bundle = worker_module.PromptBundle(
        system_template="System for {project_name}",
        user_template="Summary of {source_pdf_filename} for {client_name}",
    )

    provider_config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5")
    captured: dict[str, list[str]] = {"system": [], "user": []}

    def fake_invoke(self, provider, config, prompt, system_prompt, **_kwargs):  # noqa: ANN001
        captured["system"].append(system_prompt)
        captured["user"].append(prompt)
        return "summary"

    monkeypatch.setattr(BulkAnalysisWorker, "_invoke_provider", fake_invoke)
    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="doc.md",
        output_path=output_path,
    )
    global_placeholders = worker._build_placeholder_map()
    system_prompt = worker_module.render_system_prompt(
        bundle, metadata, placeholder_values=global_placeholders
    )
    checkpoint_mgr = CheckpointManager(project_dir / "bulk_analysis" / "group" / "map" / "checkpoints")
    manifest: dict[str, object] = {"version": 2, "signature": {"prompt_hash": "hash", "placeholders": {}}, "documents": {}}
    prompt_hash = "hash"
    manifest_path = worker_module._manifest_path(project_dir, group)

    worker._process_document(
        provider=object(),
        provider_config=provider_config,
        bundle=bundle,
        system_prompt=system_prompt,
        document=document,
        global_placeholders=global_placeholders,
        checkpoint_mgr=checkpoint_mgr,
        manifest=manifest,
        prompt_hash=prompt_hash,
        manifest_path=manifest_path,
    )

    assert captured["system"][0] == "System for Project XYZ"
    assert "doc.pdf" in captured["user"][0]
    assert "ACME" in captured["user"][0]


def test_bulk_worker_placeholder_requirements_skip_dynamic_document_name(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.placeholder_requirements = {
        "document_name": False,
        "subject_name": False,
    }
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )
    logs: list[str] = []
    worker.log_message.connect(logs.append)

    worker._enforce_placeholder_requirements(
        {"subject_name": ""},
        context="bulk analysis document 'doc.md'",
        dynamic_keys=worker_module._DYNAMIC_DOCUMENT_KEYS,
    )

    assert logs
    assert "{subject_name}" in logs[0]
    assert "{document_name}" not in logs[0]


def test_bulk_worker_global_placeholder_requirements_skip_dynamic_document_name(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.placeholder_requirements = {
        "document_name": False,
        "subject_name": False,
    }
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )
    logs: list[str] = []
    worker.log_message.connect(logs.append)

    worker._enforce_placeholder_requirements(
        {"subject_name": ""},
        context="bulk analysis",
        dynamic_keys=worker_module._DYNAMIC_GLOBAL_KEYS,
    )

    assert logs
    assert "{subject_name}" in logs[0]
    assert "{document_name}" not in logs[0]


def test_invoke_provider_rejects_over_budget_prompt(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=True,
        llm_backend=_CountingBackend(token_count=250_000),
    )

    config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5-20250929")

    with pytest.raises(_ProviderPromptLimitError, match="Prompt exceeds model input budget"):
        worker._invoke_provider(
            object(),
            config,
            "user prompt",
            "system prompt",
            input_budget=10_000,
            context_label="document 'doc.md'",
        )


def test_invoke_provider_uses_runtime_budget_without_double_applying_openai_preflight(
    tmp_path: Path,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    backend = _CountingBackend(token_count=189_199)
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=True,
        llm_backend=backend,
    )

    config = ProviderConfig(provider_id="openai", model="gpt-5-mini")

    result = worker._invoke_provider(
        object(),
        config,
        "user prompt",
        "system prompt",
        input_budget=242_220,
        preflight_input_budget=193_776,
        context_label="combine summary for 'doc.md'",
    )

    assert result == "summary"
    assert backend.invoked is True
    assert len(backend.requests) == 1
    assert backend.requests[0].input_tokens_limit == 242_220

def test_process_document_forces_chunking_when_full_prompt_exceeds_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    source_path = converted / "doc.md"
    source_path.write_text("Body text", encoding="utf-8")

    output_path = project_dir / "bulk_analysis" / "group" / "doc_analysis.md"
    metadata = ProjectMetadata(case_name="Case Name")
    group = BulkAnalysisGroup.create("Group", files=["doc.md"])
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=["doc.md"],
        metadata=metadata,
        force_rerun=True,
        placeholder_values={},
        project_name="Project XYZ",
        llm_backend=_ResultBackend(
            _model_response("summary", model_name="claude-sonnet-4-5-20250929", output_tokens=1)
        ),
    )

    bundle = worker_module.PromptBundle(
        system_template="System for {project_name}",
        user_template="Analyze {document_content}",
    )
    provider_config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5-20250929")

    monkeypatch.setattr(worker_module, "should_chunk", lambda *_args, **_kwargs: (False, 100, 130_000))
    monkeypatch.setattr(worker, "_max_input_budget", lambda **_kwargs: 50_000)
    monkeypatch.setattr(worker_module, "generate_chunks", lambda *_args, **_kwargs: ["chunk-1", "chunk-2"])

    def fake_count(
        _provider,
        _config,
        _system,
        prompt,
        *,
        max_tokens=32_000,
        allow_backend_preflight=True,
    ):  # noqa: ANN001
        assert max_tokens > 0
        _ = allow_backend_preflight
        if "chunk 1 of 2" in prompt:
            return 8_000
        if "chunk 2 of 2" in prompt:
            return 9_000
        if "Create a unified bulk analysis" in prompt:
            return 10_000
        return 80_000  # full prompt triggers forced chunking

    monkeypatch.setattr(worker, "_count_prompt_tokens", fake_count)

    checkpoint_mgr = CheckpointManager(project_dir / "bulk_analysis" / "group" / "map" / "checkpoints")
    manifest: dict[str, object] = {"version": 2, "signature": {"prompt_hash": "hash", "placeholders": {}}, "documents": {}}
    prompt_hash = "hash"
    manifest_path = worker_module._manifest_path(project_dir, group)
    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="doc.md",
        output_path=output_path,
    )
    global_placeholders = worker._build_placeholder_map()
    system_prompt = worker_module.render_system_prompt(
        bundle,
        metadata,
        placeholder_values=global_placeholders,
    )

    summary, run_details, _ = worker._process_document(
        provider=object(),
        provider_config=provider_config,
        bundle=bundle,
        system_prompt=system_prompt,
        document=document,
        global_placeholders=global_placeholders,
        checkpoint_mgr=checkpoint_mgr,
        manifest=manifest,
        prompt_hash=prompt_hash,
        manifest_path=manifest_path,
    )

    assert summary == "summary"
    assert run_details["chunking"] is True
    assert run_details["chunk_count"] == 2
    assert run_details["full_prompt_tokens"] == 80_000


def test_bulk_worker_surfaces_gateway_spend_limit_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    source_path = converted / "doc.md"
    source_path.write_text("Body text", encoding="utf-8")

    group = BulkAnalysisGroup.create("Group", files=["doc.md"])
    metadata = ProjectMetadata(case_name="Case")

    monkeypatch.setattr(
        worker_module,
        "load_prompts",
        lambda *_args, **_kwargs: PromptBundle("System", "Analyze {document_content}"),
    )
    monkeypatch.setattr(
        BulkAnalysisWorker,
        "_resolve_provider",
        lambda self: ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
    )
    monkeypatch.setattr(
        BulkAnalysisWorker,
        "_create_provider",
        lambda self, *_: object(),
    )

    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=["doc.md"],
        metadata=metadata,
        force_rerun=True,
        llm_backend=_ResultBackend(
            RuntimeError("Gateway spend limit exceeded")
        ),
    )

    failures: list[tuple[str, str]] = []
    finished: list[tuple[int, int]] = []
    worker.file_failed.connect(lambda path, error: failures.append((path, error)))
    worker.finished.connect(lambda successes, failure_count: finished.append((successes, failure_count)))

    worker._run()

    assert failures == [("doc.md", "Gateway spend limit exceeded")]
    assert finished == [(0, 1)]


def test_bulk_worker_chunk_fanout_uses_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )

    assert worker._chunk_fanout_max_concurrency() == 4

    monkeypatch.setenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "8")
    assert worker._chunk_fanout_max_concurrency() == 8

    monkeypatch.setenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "0")
    assert worker._chunk_fanout_max_concurrency() == 4

    monkeypatch.setenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "invalid")
    assert worker._chunk_fanout_max_concurrency() == 4


def test_bulk_worker_effective_gateway_chunk_fanout_defaults_to_serial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", raising=False)
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_FlakyGatewayRateLimitBackend(),
    )

    assert worker._effective_chunk_fanout_max_concurrency(
        ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6")
    ) == 1


def test_bulk_worker_effective_gateway_chunk_fanout_respects_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "8")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_FlakyGatewayRateLimitBackend(),
    )

    assert worker._effective_chunk_fanout_max_concurrency(
        ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6")
    ) == 8


def test_bulk_worker_map_max_output_tokens_reduce_gateway_anthropic_requests(tmp_path: Path) -> None:
    gateway_worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_FlakyGatewayRateLimitBackend(),
    )
    direct_worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )

    assert gateway_worker._map_max_output_tokens(
        ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6")
    ) == 12_000
    assert direct_worker._map_max_output_tokens(
        ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6")
    ) == 32_000


def test_bulk_worker_gateway_anthropic_operational_budget_uses_metadata_budget_ratio(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.model_context_window = 1_000_000
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_FlakyGatewayRateLimitBackend(),
    )

    runtime_budget = worker._max_input_budget(
        raw_context_window=1_000_000,
        max_output_tokens=worker._map_max_output_tokens(
            ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6")
        ),
    )
    operational_budget = worker._operational_request_budget(
        ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-6"),
        runtime_budget,
    )

    assert runtime_budget == 651_420
    assert operational_budget == 260_568


def test_bulk_worker_chunked_document_reassembles_parallel_results_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    source_path = project_dir / "converted_documents" / "doc.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("body", encoding="utf-8")

    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "openai"
    group.model = "gpt-5-mini"
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )
    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="converted_documents/doc.md",
        output_path=project_dir / "bulk_analysis" / "group" / "doc.md",
    )
    manifest = {"version": 2, "signature": None, "documents": {}}
    manifest_path = worker_module._manifest_path(project_dir, group)
    recovery_store = worker_module.BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    recovery_manifest = recovery_store.load_map_manifest()
    completion_order: list[int] = []

    monkeypatch.setenv("LLESTRADE_BULK_MAP_CHUNK_MAX_CONCURRENCY", "3")
    monkeypatch.setattr(BulkAnalysisWorker, "_create_provider", lambda self, *_args, **_kwargs: object())

    def _fake_execute_chunk_task(self, *, spec, **_kwargs):  # noqa: ANN001
        time.sleep({1: 0.05, 2: 0.01, 3: 0.03}[spec.chunk_index])
        completion_order.append(spec.chunk_index)
        return worker_module._ChunkTaskResult(
            chunk_index=spec.chunk_index,
            chunk_checksum=spec.chunk_checksum,
            summary=f"summary-{spec.chunk_index}",
            usage={"input_tokens": spec.chunk_index, "output_tokens": 1, "cost": 0.1 * spec.chunk_index},
        )

    monkeypatch.setattr(BulkAnalysisWorker, "_execute_chunk_task", _fake_execute_chunk_task)
    monkeypatch.setattr(
        worker_module,
        "combine_chunk_summaries_hierarchical",
        lambda summaries, **_kwargs: "combined:" + "|".join(summaries),
    )

    result, run_details, _ = worker._process_chunked_document(
        provider=object(),
        provider_config=ProviderConfig(provider_id="openai", model="gpt-5-mini"),
        bundle=PromptBundle(system_template="System", user_template="Analyze {document_content}"),
        system_prompt="System",
        document=document,
        doc_placeholders={},
        manifest=manifest,
        prompt_hash="prompt-hash",
        manifest_path=manifest_path,
        recovery_store=recovery_store,
        recovery_manifest=recovery_manifest,
        run_details={},
        body="body",
        chunks=["chunk-1", "chunk-2", "chunk-3"],
        raw_context_window=400_000,
        input_budget=100_000,
        preflight_input_budget=80_000,
        map_max_tokens=32_000,
    )

    assert completion_order != [1, 2, 3]
    assert sorted(completion_order) == [1, 2, 3]
    assert result == "combined:summary-1|summary-2|summary-3"
    assert run_details["chunk_count"] == 3
    assert manifest["documents"]["converted_documents/doc.md"]["chunks_done"] == [1, 2, 3]
    assert recovery_manifest["actual_usage"] == {"input_tokens": 6, "output_tokens": 3}
    assert recovery_manifest["actual_cost"] == pytest.approx(0.6)


def test_bulk_worker_passes_runtime_budget_and_count_callback_to_hierarchical_combine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    source_path = project_dir / "converted_documents" / "doc.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("body", encoding="utf-8")

    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "openai"
    group.model = "gpt-5-mini"
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_CountingBackend(token_count=4321),
    )
    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="converted_documents/doc.md",
        output_path=project_dir / "bulk_analysis" / "group" / "doc.md",
    )
    manifest = {"version": 2, "signature": None, "documents": {}}
    manifest_path = worker_module._manifest_path(project_dir, group)
    recovery_store = worker_module.BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    recovery_manifest = recovery_store.load_map_manifest()

    monkeypatch.setattr(BulkAnalysisWorker, "_create_provider", lambda self, *_args, **_kwargs: object())
    monkeypatch.setattr(
        BulkAnalysisWorker,
        "_execute_chunk_task",
        lambda self, *, spec, **_kwargs: worker_module._ChunkTaskResult(
            chunk_index=spec.chunk_index,
            chunk_checksum=spec.chunk_checksum,
            summary=f"summary-{spec.chunk_index}",
            usage={"input_tokens": 1, "output_tokens": 1, "cost": 0.1},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_hierarchical(summaries, **kwargs):  # noqa: ANN001
        captured["summaries"] = list(summaries)
        captured["input_budget"] = kwargs["input_budget"]
        captured["runtime_input_budget"] = kwargs["runtime_input_budget"]
        captured["raw_context_window"] = kwargs["raw_context_window"]
        captured["counted"] = kwargs["count_prompt_tokens_fn"]("combine prompt")
        return "combined"

    monkeypatch.setattr(worker_module, "combine_chunk_summaries_hierarchical", _fake_hierarchical)

    result, run_details, _ = worker._process_chunked_document(
        provider=object(),
        provider_config=ProviderConfig(provider_id="openai", model="gpt-5-mini"),
        bundle=PromptBundle(system_template="System", user_template="Analyze {document_content}"),
        system_prompt="System",
        document=document,
        doc_placeholders={},
        manifest=manifest,
        prompt_hash="prompt-hash",
        manifest_path=manifest_path,
        recovery_store=recovery_store,
        recovery_manifest=recovery_manifest,
        run_details={},
        body="body",
        chunks=["chunk-1", "chunk-2"],
        raw_context_window=400_000,
        input_budget=242_220,
        preflight_input_budget=193_776,
        map_max_tokens=32_000,
    )

    assert result == "combined"
    assert run_details["chunk_count"] == 2
    assert captured["summaries"] == ["summary-1", "summary-2"]
    assert captured["input_budget"] == 193_776
    assert captured["runtime_input_budget"] == 242_220
    assert captured["raw_context_window"] == 400_000
    assert captured["counted"] == 4321


def test_bulk_worker_retries_gateway_rate_limit_without_rechunking_and_reuses_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    source_path = project_dir / "converted_documents" / "doc.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("body", encoding="utf-8")

    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="converted_documents/doc.md",
        output_path=project_dir / "bulk_analysis" / "group" / "doc.md",
    )
    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "anthropic"
    group.model = "claude-sonnet-4-5"
    group.model_context_window = 1_000_000
    backend = _FlakyGatewayRateLimitBackend()
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )
    bundle = PromptBundle(system_template="System", user_template="Analyze {document_content}")
    provider = worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"))
    source_context = SourceFileContext(
        absolute_path=source_path.resolve(),
        relative_path=document.relative_path,
    )
    manifest_path = _manifest_path(project_dir, group)
    recovery_store = worker_module.BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    recovery_manifest = recovery_store.load_map_manifest()

    monkeypatch.setattr(worker, "_load_document", lambda _document: ("body", {}, source_context))
    monkeypatch.setattr(worker_module, "should_chunk", lambda *_args, **_kwargs: (True, 4_939_208, 200_000))
    monkeypatch.setattr(worker, "_count_prompt_tokens", lambda *_args, **_kwargs: 1_000)
    monkeypatch.setattr(worker, "_gateway_rate_limit_retry_delay", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        worker_module,
        "combine_chunk_summaries_hierarchical",
        lambda summaries, **_kwargs: "combined:" + "|".join(summaries),
    )

    chunk_targets: list[int] = []

    def _fake_generate(*, initial_chunk_tokens, **_kwargs):
        chunk_targets.append(initial_chunk_tokens)
        return ["chunk-one", "chunk-two"]

    monkeypatch.setattr(worker, "_generate_fitting_chunks", _fake_generate)

    result, run_details, _placeholders = worker._process_document(
        provider=provider,
        provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
        bundle=bundle,
        system_prompt="System",
        document=document,
        global_placeholders={},
        manifest={"version": 2, "signature": None, "documents": {}},
        prompt_hash="prompt-hash",
        manifest_path=manifest_path,
        recovery_store=recovery_store,
        recovery_manifest=recovery_manifest,
    )

    assert result == "combined:summary|summary"
    assert run_details["chunk_count"] == 2
    assert chunk_targets == [200_000, 200_000]
    assert backend.calls == 3


def test_bulk_worker_retries_gateway_server_error_with_smaller_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    source_path = project_dir / "converted_documents" / "doc.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("body", encoding="utf-8")

    document = worker_module.BulkAnalysisDocument(
        source_path=source_path,
        relative_path="converted_documents/doc.md",
        output_path=project_dir / "bulk_analysis" / "group" / "doc.md",
    )
    group = BulkAnalysisGroup.create("Group")
    group.provider_id = "anthropic"
    group.model = "claude-sonnet-4-5"
    group.model_context_window = 1_000_000
    backend = _FlakyGatewayServerBackend()
    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )
    bundle = PromptBundle(system_template="System", user_template="Analyze {document_content}")
    provider = worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"))
    source_context = SourceFileContext(
        absolute_path=source_path.resolve(),
        relative_path=document.relative_path,
    )
    manifest_path = _manifest_path(project_dir, group)
    recovery_store = worker_module.BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    recovery_manifest = recovery_store.load_map_manifest()

    monkeypatch.setattr(worker, "_load_document", lambda _document: ("body", {}, source_context))
    monkeypatch.setattr(worker_module, "should_chunk", lambda *_args, **_kwargs: (True, 4_939_208, 200_000))
    monkeypatch.setattr(worker, "_count_prompt_tokens", lambda *_args, **_kwargs: 1_000)
    monkeypatch.setattr(worker, "_gateway_rate_limit_retry_delay", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        worker_module,
        "combine_chunk_summaries_hierarchical",
        lambda summaries, **_kwargs: "combined:" + "|".join(summaries),
    )

    chunk_targets: list[int] = []

    def _fake_generate(*, initial_chunk_tokens, **_kwargs):
        chunk_targets.append(initial_chunk_tokens)
        if initial_chunk_tokens > 150_000:
            return ["chunk-one"]
        return ["chunk-one", "chunk-two"]

    monkeypatch.setattr(worker, "_generate_fitting_chunks", _fake_generate)

    result, run_details, _placeholders = worker._process_document(
        provider=provider,
        provider_config=ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5"),
        bundle=bundle,
        system_prompt="System",
        document=document,
        global_placeholders={},
        manifest={"version": 2, "signature": None, "documents": {}},
        prompt_hash="prompt-hash",
        manifest_path=manifest_path,
        recovery_store=recovery_store,
        recovery_manifest=recovery_manifest,
    )

    assert result == "combined:summary|summary"
    assert run_details["chunk_count"] == 2
    assert chunk_targets == [200_000, 150_000]
    assert backend.calls == 3


def test_bulk_worker_extract_page_numbers_uses_canonical_marker_regex(tmp_path: Path) -> None:
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )

    content = "\n".join(
        [
            "<!--- AC DH 000001.pdf#page=1 --->",
            "First page",
            "<!--- AC DH 000001.pdf#page=1 --->",
            "<!--- AC DH 000001.pdf#page=2 --->",
        ]
    )

    assert worker._extract_page_numbers(content) == [1, 2]


def test_bulk_worker_skips_ledger_when_chunk_has_no_page_markers(tmp_path: Path) -> None:
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=BulkAnalysisGroup.create("Group"),
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )

    class _FailingCitationStore:
        def build_evidence_ledger(self, **_kwargs):  # noqa: ANN003
            raise AssertionError("ledger should not be built without page markers")

    worker._citation_store = _FailingCitationStore()  # type: ignore[assignment]

    assert worker._build_document_evidence_ledger(relative_path="doc.md", content="no markers here") == ""
