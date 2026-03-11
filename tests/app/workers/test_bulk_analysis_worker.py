from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import pytest

from src.app.workers.checkpoint_manager import CheckpointManager

from src.app.core.bulk_analysis_runner import PromptBundle
from src.app.core.project_manager import ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.workers import bulk_analysis_worker as worker_module
from src.app.workers.bulk_analysis_worker import (
    BulkAnalysisWorker,
    ProviderConfig,
    _compute_prompt_hash,
    _load_manifest,
    _manifest_path,
    _save_manifest,
    _should_process_document,
)
from src.app.workers.llm_backend import (
    LLMExecutionBackend,
    LLMInvocationRequest,
    LLMInvocationResult,
    LLMProviderRequest,
    ProviderMetadata,
    normalize_model_name,
    resolve_model_name,
)


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, *, prompt: str, model=None, system_prompt=None, temperature=0.1, max_tokens=32000):  # noqa: ANN001
        self.calls.append(prompt)
        return {"success": True, "content": "summary"}

    def count_tokens(self, text=None, messages=None):  # noqa: ANN001
        content = text or ""
        return {"success": True, "token_count": max(len(content) // 4, 1)}


class _NoNativeBackend(LLMExecutionBackend):
    def requires_native_provider(self) -> bool:
        return False

    def normalize_model(self, provider_id: str, model: str | None) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: str | None) -> str | None:
        return resolve_model_name(provider_id, model)

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(provider_name=request.provider_id, default_model=request.model or "default-model")

    def invoke(self, provider, request: LLMInvocationRequest) -> LLMInvocationResult:  # noqa: ANN001
        return LLMInvocationResult(
            success=True,
            content="summary",
            error=None,
            usage={"output_tokens": 1},
            provider="gateway/anthropic",
            model=request.model,
            raw={},
        )


class _ResultBackend(LLMExecutionBackend):
    def __init__(self, result: LLMInvocationResult) -> None:
        self._result = result

    def requires_native_provider(self) -> bool:
        return False

    def normalize_model(self, provider_id: str, model: str | None) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: str | None) -> str | None:
        return resolve_model_name(provider_id, model)

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(provider_name=request.provider_id, default_model=request.model or "default-model")

    def invoke(self, provider, request: LLMInvocationRequest) -> LLMInvocationResult:  # noqa: ANN001
        _ = provider, request
        return self._result


class _CountingBackend(LLMExecutionBackend):
    def __init__(self, *, token_count: int) -> None:
        self.token_count = token_count
        self.invoked = False

    def requires_native_provider(self) -> bool:
        return False

    def normalize_model(self, provider_id: str, model: str | None) -> str | None:
        return normalize_model_name(provider_id, model)

    def resolve_model(self, provider_id: str, model: str | None) -> str | None:
        return resolve_model_name(provider_id, model)

    def create_provider(self, request: LLMProviderRequest) -> object:
        return ProviderMetadata(provider_name=request.provider_id, default_model=request.model or "default-model")

    def count_input_tokens(self, provider, request: LLMInvocationRequest) -> int | None:  # noqa: ANN001
        _ = provider, request
        return self.token_count

    def invoke(self, provider, request: LLMInvocationRequest) -> LLMInvocationResult:  # noqa: ANN001
        _ = provider, request
        self.invoked = True
        return LLMInvocationResult(
            success=True,
            content="summary",
            error=None,
            usage={},
            provider="gateway/anthropic",
            model="claude",
            raw={},
        )


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
    config = ProviderConfig(provider_id="anthropic", model="claude")

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
        "system prompt",
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


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            LLMInvocationResult(
                success=False,
                content="",
                error="Gateway timeout",
                usage={},
                provider="gateway/anthropic",
                model="claude",
                raw={},
            ),
            "Gateway timeout",
        ),
        (
            LLMInvocationResult(
                success=True,
                content="   ",
                error=None,
                usage={},
                provider="gateway/anthropic",
                model="claude",
                raw={},
            ),
            "LLM returned empty response",
        ),
    ],
)
def test_bulk_map_invoke_provider_raises_for_failed_or_empty_backend_result(
    tmp_path: Path,
    result: LLMInvocationResult,
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
            provider_config=ProviderConfig(provider_id="anthropic", model="claude"),
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
            provider_config=ProviderConfig(provider_id="anthropic", model="claude"),
            prompt="Prompt",
            system_prompt="System",
        )


def test_bulk_map_trace_attributes_match_between_legacy_and_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.slug = "group-slug"
    config = ProviderConfig(provider_id="anthropic", model="claude")

    legacy_worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )
    legacy_traces = _capture_traces(monkeypatch)
    legacy_result = legacy_worker._invoke_provider(
        provider=_FakeProvider(),
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
            LLMInvocationResult(
                success=True,
                content="summary",
                error=None,
                usage={"output_tokens": 1},
                provider="gateway/anthropic",
                model="claude",
                raw={},
            )
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
                "llestrade.provider_id": "anthropic",
                "llestrade.model": "claude",
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

    with pytest.raises(RuntimeError, match="500 tokens > 400 budget"):
        worker._invoke_provider(
            provider=worker._create_provider(ProviderConfig(provider_id="anthropic", model="claude"), "System"),
            provider_config=ProviderConfig(provider_id="anthropic", model="claude"),
            prompt="Prompt",
            system_prompt="System",
            input_budget=400,
        )

    assert backend.invoked is False


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

    provider_config = ProviderConfig(provider_id="anthropic", model="model")
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


def test_invoke_provider_rejects_over_budget_prompt(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=True,
    )

    provider = _FakeProvider()
    config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5-20250929")

    def fake_count(*_args, **_kwargs):  # noqa: ANN001
        return 250_000

    worker._count_prompt_tokens = fake_count  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Prompt exceeds model input budget"):
        worker._invoke_provider(
            provider,  # type: ignore[arg-type]
            config,
            "user prompt",
            "system prompt",
            input_budget=10_000,
            context_label="document 'doc.md'",
        )


def test_invoke_provider_compat_supports_legacy_override_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkAnalysisWorker(
        project_dir=tmp_path,
        group=group,
        files=[],
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=True,
    )
    config = ProviderConfig(provider_id="anthropic", model="claude")

    def legacy_invoke(self, provider, provider_config, prompt, system_prompt):  # noqa: ANN001
        return "summary"

    monkeypatch.setattr(BulkAnalysisWorker, "_invoke_provider", legacy_invoke)
    caplog.clear()
    caplog.set_level("WARNING")

    result = worker._invoke_provider_compat(
        object(),  # type: ignore[arg-type]
        config,
        "prompt",
        "system",
        input_budget=5_000,
        context_label="legacy-override",
    )

    assert result == "summary"
    assert any("dropped unsupported kwargs" in rec.message for rec in caplog.records)


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
    )

    bundle = worker_module.PromptBundle(
        system_template="System for {project_name}",
        user_template="Analyze {document_content}",
    )
    provider_config = ProviderConfig(provider_id="anthropic", model="claude-sonnet-4-5-20250929")
    provider = _FakeProvider()

    monkeypatch.setattr(worker_module, "should_chunk", lambda *_args, **_kwargs: (False, 100, 130_000))
    monkeypatch.setattr(worker, "_max_input_budget", lambda **_kwargs: 50_000)
    monkeypatch.setattr(worker_module, "generate_chunks", lambda *_args, **_kwargs: ["chunk-1", "chunk-2"])

    def fake_count(_provider, _config, _system, prompt):  # noqa: ANN001
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
        provider=provider,  # type: ignore[arg-type]
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
        lambda self: ProviderConfig(provider_id="anthropic", model="claude"),
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
            LLMInvocationResult(
                success=False,
                content="",
                error="Gateway spend limit exceeded",
                usage={},
                provider="gateway/anthropic",
                model="claude",
                raw={},
            )
        ),
    )

    failures: list[tuple[str, str]] = []
    finished: list[tuple[int, int]] = []
    worker.file_failed.connect(lambda path, error: failures.append((path, error)))
    worker.finished.connect(lambda successes, failure_count: finished.append((successes, failure_count)))

    worker._run()

    assert failures == [("doc.md", "Gateway spend limit exceeded")]
    assert finished == [(0, 1)]
