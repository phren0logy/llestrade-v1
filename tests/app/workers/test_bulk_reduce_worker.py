from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from src.app.core.project_manager import ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.workers import bulk_reduce_worker as reduce_module
from src.app.workers.llm_backend import LLMExecutionBackend, LLMInvocationRequest, LLMInvocationResult
from src.app.workers.bulk_reduce_worker import BulkReduceWorker, ProviderConfig


class _NoNativeBackend(LLMExecutionBackend):
    def requires_native_provider(self) -> bool:
        return False

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

    def invoke(self, provider, request: LLMInvocationRequest) -> LLMInvocationResult:  # noqa: ANN001
        _ = provider, request
        return self._result


class _CapturingBackend(LLMExecutionBackend):
    def __init__(self, result: LLMInvocationResult) -> None:
        self._result = result
        self.requests: list[LLMInvocationRequest] = []

    def requires_native_provider(self) -> bool:
        return False

    def invoke(self, provider, request: LLMInvocationRequest) -> LLMInvocationResult:  # noqa: ANN001
        _ = provider
        self.requests.append(request)
        return self._result


def _capture_traces(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object] | None]]:
    recorded: list[tuple[str, dict[str, object] | None]] = []

    @contextmanager
    def _fake_trace_operation(name: str, attributes=None):  # noqa: ANN001
        recorded.append((name, dict(attributes) if attributes is not None else None))
        yield None

    monkeypatch.setattr(reduce_module, "trace_operation", _fake_trace_operation)
    return recorded


def test_bulk_reduce_worker_force_rerun(tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = qtbot
    project_dir = tmp_path
    converted = project_dir / "converted_documents" / "folder"
    converted.mkdir(parents=True, exist_ok=True)
    converted_doc = converted / "doc.md"
    converted_doc.write_text("content", encoding="utf-8")

    group = BulkAnalysisGroup.create("Group")
    group.combine_converted_files = ["folder/doc.md"]
    metadata = ProjectMetadata(case_name="Case")

    call_count = {"value": 0}

    monkeypatch.setattr(
        reduce_module,
        "load_prompts",
        lambda *_args, **_kwargs: reduce_module.PromptBundle("System", "User {document_content}"),
    )
    monkeypatch.setattr(
        BulkReduceWorker,
        "_resolve_provider",
        lambda self: ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
    )
    monkeypatch.setattr(
        BulkReduceWorker,
        "_create_provider",
        lambda self, *_: object(),
    )
    monkeypatch.setattr(
        reduce_module,
        "should_chunk",
        lambda *_args, **_kwargs: (False, 1000, 2000),
    )

    def fake_invoke(self, *args, **kwargs):  # noqa: ANN001
        call_count["value"] += 1
        return "summary"

    monkeypatch.setattr(BulkReduceWorker, "_invoke_provider", fake_invoke)

    worker = BulkReduceWorker(
        project_dir=project_dir,
        group=group,
        metadata=metadata,
        force_rerun=False,
        placeholder_values={},
        project_name="Case",
    )
    worker._run()
    assert call_count["value"] == 1

    skip_worker = BulkReduceWorker(
        project_dir=project_dir,
        group=group,
        metadata=metadata,
        force_rerun=False,
        placeholder_values={},
        project_name="Case",
    )
    skip_worker._run()
    assert call_count["value"] == 1

    force_worker = BulkReduceWorker(
        project_dir=project_dir,
        group=group,
        metadata=metadata,
        force_rerun=True,
        placeholder_values={},
        project_name="Case",
    )
    force_worker._run()
    assert call_count["value"] == 2


def test_bulk_reduce_worker_applies_placeholder_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path
    converted_dir = project_dir / "converted_documents"
    converted_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = project_dir / "sources" / "doc space.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_text("pdf", encoding="utf-8")

    import frontmatter

    doc_path = converted_dir / "doc.md"
    post = frontmatter.Post("Content", metadata={"sources": [{"path": str(pdf_path)}]})
    doc_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    group = BulkAnalysisGroup.create("Group")
    group.combine_converted_files = ["doc.md"]

    metadata = ProjectMetadata(case_name="Case")

    monkeypatch.setattr(
        reduce_module,
        "load_prompts",
        lambda *_args, **_kwargs: reduce_module.PromptBundle(
            "System for {project_name}",
            "Combined {reduce_source_list} {client_name} {source_pdf_absolute_url}",
        ),
    )
    monkeypatch.setattr(
        BulkReduceWorker,
        "_resolve_provider",
        lambda self: ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
    )
    monkeypatch.setattr(
        BulkReduceWorker,
        "_create_provider",
        lambda self, *_: object(),
    )
    monkeypatch.setattr(
        reduce_module,
        "should_chunk",
        lambda *_args, **_kwargs: (False, 100, 2000),
    )

    captured: dict[str, list[str]] = {"system": [], "user": []}

    def fake_invoke(self, provider, cfg, prompt, system_prompt, **kwargs):  # noqa: ANN001
        _ = kwargs
        captured["system"].append(system_prompt)
        captured["user"].append(prompt)
        return "summary"

    monkeypatch.setattr(BulkReduceWorker, "_invoke_provider", fake_invoke)

    worker = BulkReduceWorker(
        project_dir=project_dir,
        group=group,
        metadata=metadata,
        force_rerun=True,
        placeholder_values={"client_name": "ACME"},
        project_name="Case Project",
    )
    worker._run()

    assert captured["system"], "expected system prompt captured"
    assert captured["user"], "expected user prompt captured"
    assert captured["system"][0] == "System for Case Project"
    user_prompt = captured["user"][0]
    assert "doc space.pdf" in user_prompt
    assert "ACME" in user_prompt
    from urllib.parse import quote

    expected_url = quote(pdf_path.resolve().as_posix(), safe="/:")
    assert expected_url in user_prompt


def test_bulk_reduce_create_provider_skips_native_bootstrap_for_no_native_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkReduceWorker(
        project_dir=tmp_path,
        group=group,
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_NoNativeBackend(),
    )

    def _fail_create_provider(**_kwargs):  # noqa: ANN001
        raise AssertionError("create_provider should not be called for no-native backend")

    monkeypatch.setattr(reduce_module, "create_provider", _fail_create_provider, raising=False)
    provider = worker._create_provider(
        ProviderConfig(provider_id="anthropic", model=None, temperature=0.1),
        "system prompt",
    )

    assert provider.provider_name == "anthropic"
    assert provider.default_model


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            LLMInvocationResult(
                success=False,
                content="",
                error="Gateway provider rejected request",
                usage={},
                provider="gateway/anthropic",
                model="claude",
                raw={},
            ),
            "Gateway provider rejected request",
        ),
        (
            LLMInvocationResult(
                success=True,
                content=" ",
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
def test_bulk_reduce_invoke_provider_raises_for_failed_or_empty_backend_result(
    tmp_path: Path,
    result: LLMInvocationResult,
    message: str,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkReduceWorker(
        project_dir=tmp_path,
        group=group,
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=_ResultBackend(result),
    )

    with pytest.raises(RuntimeError, match=message):
        worker._invoke_provider(
            provider=object(),
            provider_cfg=ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
            prompt="Prompt",
            system_prompt="System",
        )


def test_bulk_reduce_invoke_provider_raises_cancelled_when_worker_cancelled(tmp_path: Path) -> None:
    group = BulkAnalysisGroup.create("Group")
    worker = BulkReduceWorker(
        project_dir=tmp_path,
        group=group,
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )
    worker.cancel()

    with pytest.raises(reduce_module.BulkAnalysisCancelled):
        worker._invoke_provider(
            provider=object(),
            provider_cfg=ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
            prompt="Prompt",
            system_prompt="System",
        )


def test_bulk_reduce_trace_attributes_match_between_legacy_and_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.slug = "group-slug"
    config = ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1)

    class _LegacyProvider:
        def generate(self, **kwargs):  # noqa: ANN003, ANN001
            _ = kwargs
            return {"success": True, "content": "summary"}

    legacy_worker = BulkReduceWorker(
        project_dir=tmp_path,
        group=group,
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
    )
    legacy_traces = _capture_traces(monkeypatch)
    legacy_result = legacy_worker._invoke_provider(
        provider=_LegacyProvider(),
        provider_cfg=config,
        prompt="Prompt",
        system_prompt="System",
    )

    gateway_worker = BulkReduceWorker(
        project_dir=tmp_path,
        group=group,
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
        provider_cfg=config,
        prompt="Prompt",
        system_prompt="System",
    )

    assert legacy_result == "summary"
    assert gateway_result == "summary"
    assert legacy_traces == gateway_traces == [
        (
            "bulk_reduce.invoke_llm",
            {
                "llestrade.provider_id": "anthropic",
                "llestrade.model": "claude",
                "llestrade.max_tokens": 32000,
                "llestrade.temperature": 0.1,
                "llestrade.worker": "bulk_reduce",
                "llestrade.stage": "bulk_reduce",
                "llestrade.group_id": group.group_id,
                "llestrade.group_name": "Group",
                "llestrade.group_slug": "group-slug",
            },
        )
    ]


def test_bulk_reduce_passes_computed_input_budget_to_backend(
    tmp_path: Path,
) -> None:
    group = BulkAnalysisGroup.create("Group")
    group.model_context_window = 100_000
    backend = _CapturingBackend(
        LLMInvocationResult(
            success=True,
            content="summary",
            error=None,
            usage={"output_tokens": 1},
            provider="gateway/anthropic",
            model="claude",
            raw={},
        )
    )
    worker = BulkReduceWorker(
        project_dir=tmp_path,
        group=group,
        metadata=ProjectMetadata(case_name="Case"),
        force_rerun=False,
        llm_backend=backend,
    )

    result = worker._invoke_provider(
        provider=object(),
        provider_cfg=ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
        prompt="Prompt",
        system_prompt="System",
        input_budget=worker._max_input_budget(
            ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
            max_output_tokens=32_000,
        ),
    )

    assert result == "summary"
    assert len(backend.requests) == 1
    assert backend.requests[0].input_tokens_limit == 67_000


def test_bulk_reduce_worker_surfaces_gateway_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    converted_doc = converted / "doc.md"
    converted_doc.write_text("content", encoding="utf-8")

    group = BulkAnalysisGroup.create("Group")
    group.combine_converted_files = ["doc.md"]
    metadata = ProjectMetadata(case_name="Case")

    monkeypatch.setattr(
        reduce_module,
        "load_prompts",
        lambda *_args, **_kwargs: reduce_module.PromptBundle("System", "User {document_content}"),
    )
    monkeypatch.setattr(
        BulkReduceWorker,
        "_resolve_provider",
        lambda self: ProviderConfig(provider_id="anthropic", model="claude", temperature=0.1),
    )
    monkeypatch.setattr(
        BulkReduceWorker,
        "_create_provider",
        lambda self, *_: object(),
    )
    monkeypatch.setattr(
        reduce_module,
        "should_chunk",
        lambda *_args, **_kwargs: (False, 100, 2000),
    )

    worker = BulkReduceWorker(
        project_dir=project_dir,
        group=group,
        metadata=metadata,
        force_rerun=True,
        llm_backend=_ResultBackend(
            LLMInvocationResult(
                success=False,
                content="",
                error="Gateway timeout",
                usage={},
                provider="gateway/anthropic",
                model="claude",
                raw={},
            )
        ),
    )

    finished: list[tuple[int, int]] = []
    logs: list[str] = []
    worker.finished.connect(lambda successes, failures: finished.append((successes, failures)))
    worker.log_message.connect(logs.append)

    worker._run()

    assert finished == [(0, 1)]
    assert any("Gateway timeout" in message for message in logs)
