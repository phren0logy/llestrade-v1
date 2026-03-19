from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Sequence

import frontmatter
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart
from pydantic_ai.usage import RequestUsage

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_ = PySide6

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.converted_documents import converted_artifact_relative
from src.app.core.project_manager import ProjectMetadata
from src.app.core.report_inputs import REPORT_CATEGORY_CONVERTED
from src.app.workers import bulk_analysis_worker, bulk_reduce_worker, report_common, report_worker
from src.app.workers.bulk_analysis_worker import BulkAnalysisWorker
from src.app.workers.bulk_reduce_worker import BulkReduceWorker
from src.app.workers.llm_backend import (
    LLMExecutionBackend,
    LLMInvocationRequest,
    LLMProviderRequest,
    ProviderMetadata,
    build_model_settings,
    normalize_model_name,
    provider_capabilities,
    resolve_model_name,
)
from src.app.workers.report_worker import DraftReportWorker, ReportRefinementWorker

_FIXED_NOW = datetime(2026, 3, 9, 12, 34, 56, tzinfo=timezone.utc)
_FIXED_EPOCH = _FIXED_NOW.timestamp()


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return _FIXED_NOW.astimezone().replace(tzinfo=None)
        return _FIXED_NOW.astimezone(tz)


def _model_response(
    content: str,
    *,
    model_name: str = "claude-sonnet-4-5",
    output_tokens: int = 101,
    reasoning: str | None = None,
) -> ModelResponse:
    parts: list[object] = []
    if reasoning:
        parts.append(ThinkingPart(reasoning))
    parts.append(TextPart(content))
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(input_tokens=1, output_tokens=output_tokens),
        model_name=model_name,
    )


class _SequenceBackend(LLMExecutionBackend):
    def __init__(self, payloads: Sequence[ModelResponse | Exception]) -> None:
        self._payloads = list(payloads)

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
        if not self._payloads:
            raise AssertionError("Sequence backend called more times than expected")
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _freeze_worker_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_common, "datetime", _FrozenDateTime, raising=True)
    monkeypatch.setattr(report_worker, "datetime", _FrozenDateTime, raising=True)
    monkeypatch.setattr(bulk_analysis_worker, "datetime", _FrozenDateTime, raising=True)
    monkeypatch.setattr(bulk_reduce_worker, "datetime", _FrozenDateTime, raising=True)


def _set_fixed_mtime(path: Path) -> None:
    os.utime(path, (_FIXED_EPOCH, _FIXED_EPOCH))


def _normalize_value(value: Any, project_dir: Path) -> Any:
    root_text = project_dir.resolve().as_posix()
    relative_root_text = project_dir.as_posix()
    if isinstance(value, dict):
        return {key: _normalize_value(item, project_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item, project_dir) for item in value]
    if isinstance(value, str):
        normalized = value.replace(root_text, "<PROJECT_ROOT>")
        return normalized.replace(relative_root_text, "<PROJECT_ROOT>")
    return value


def _read_json_snapshot(path: Path, project_dir: Path) -> Any:
    return _normalize_value(json.loads(path.read_text(encoding="utf-8")), project_dir)


def _read_markdown_snapshot(path: Path, project_dir: Path) -> dict[str, Any]:
    document = frontmatter.loads(path.read_text(encoding="utf-8"))
    metadata = _normalize_value(dict(document.metadata), project_dir)
    sources = metadata.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            relative = str(source.get("relative") or "")
            if relative.startswith("reports/") and "checksum" in source:
                source["checksum"] = "<GENERATED_CHECKSUM>"
    return {
        "metadata": metadata,
        "content": document.content,
    }


def _prepare_report_project(project_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    converted_dir = project_dir / "converted_documents"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted_doc = converted_dir / "doc.md"
    converted_doc.write_text("# Heading\nBody", encoding="utf-8")

    template_dir = project_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / "report-template.md"
    template_path.write_text(
        dedent(
            """\
            # Section One

            Details for section one.

            # Section Two

            Details for section two.
            """
        ),
        encoding="utf-8",
    )

    prompt_dir = project_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    generation_user_prompt_path = prompt_dir / "generation_user_prompt.md"
    generation_user_prompt_path.write_text(
        (
            "## Prompt\n\n"
            "Write the report section described in {template_section}.\n\n"
            "Transcript (if any):\n{transcript}\n\n"
            "Other sources:\n{additional_documents}\n"
        ),
        encoding="utf-8",
    )
    refinement_user_prompt_path = prompt_dir / "refinement_user_prompt.md"
    refinement_user_prompt_path.write_text(
        (
            "## Instructions\n\nPolish the draft into final prose.\n\n"
            "## Prompt\n\n"
            "<draft>{draft_report}</draft>\n\n"
            "<template>{template}</template>\n\n"
            "<transcript>{transcript}</transcript>\n"
        ),
        encoding="utf-8",
    )

    system_prompt_dir = project_dir / "system_prompts"
    system_prompt_dir.mkdir(parents=True, exist_ok=True)
    generation_system_prompt_path = system_prompt_dir / "generation.md"
    generation_system_prompt_path.write_text("You are drafting for {client_name}.", encoding="utf-8")
    refinement_system_prompt_path = system_prompt_dir / "refinement.md"
    refinement_system_prompt_path.write_text("You are refining for {client_name}.", encoding="utf-8")

    for path in (
        converted_doc,
        template_path,
        generation_user_prompt_path,
        refinement_user_prompt_path,
        generation_system_prompt_path,
        refinement_system_prompt_path,
    ):
        _set_fixed_mtime(path)

    return (
        template_path,
        generation_user_prompt_path,
        refinement_user_prompt_path,
        generation_system_prompt_path,
        refinement_system_prompt_path,
    )


def _prepare_bulk_project(project_dir: Path) -> tuple[BulkAnalysisGroup, BulkAnalysisGroup]:
    converted_dir = project_dir / "converted_documents"
    converted_dir.mkdir(parents=True, exist_ok=True)
    relative_path = converted_artifact_relative("doc.pdf")
    converted_doc = converted_dir / relative_path
    converted_doc.parent.mkdir(parents=True, exist_ok=True)
    converted_doc.write_text("<loc_0><loc_0><loc_200><loc_40>Body text for parity testing.\n", encoding="utf-8")

    prompt_dir = project_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = prompt_dir / "bulk_system.md"
    system_prompt.write_text("System for {project_name}", encoding="utf-8")
    user_prompt = prompt_dir / "bulk_user.md"
    user_prompt.write_text("Analyze {document_name}\n\n{document_content}", encoding="utf-8")
    reduce_user_prompt = prompt_dir / "reduce_user.md"
    reduce_user_prompt.write_text("Combine {reduce_source_count} source(s)\n\n{document_content}", encoding="utf-8")

    for path in (converted_doc, system_prompt, user_prompt, reduce_user_prompt):
        _set_fixed_mtime(path)

    map_group = BulkAnalysisGroup.create(
        "Parity Map",
        files=[relative_path],
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        system_prompt_path="prompts/bulk_system.md",
        user_prompt_path="prompts/bulk_user.md",
    )
    map_group.slug = "parity-map"
    map_group.group_id = "group-map"

    reduce_group = BulkAnalysisGroup.create(
        "Parity Reduce",
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        system_prompt_path="prompts/bulk_system.md",
        user_prompt_path="prompts/reduce_user.md",
    )
    reduce_group.slug = "parity-reduce"
    reduce_group.group_id = "group-reduce"
    reduce_group.operation = "combined"
    reduce_group.combine_converted_files = [relative_path]
    reduce_group.combine_output_template = "combined_output.md"

    return map_group, reduce_group


def _gateway_result(
    content: str,
    *,
    model: str = "claude-sonnet-4-5",
    output_tokens: int = 101,
    reasoning: str | None = None,
) -> ModelResponse:
    return _model_response(
        content,
        model_name=model,
        output_tokens=output_tokens,
        reasoning=reasoning,
    )


def _run_report_pipeline(
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy: bool,
) -> dict[str, Any]:
    (
        template_path,
        generation_user_prompt_path,
        refinement_user_prompt_path,
        generation_system_prompt_path,
        refinement_system_prompt_path,
    ) = _prepare_report_project(project_dir)
    metadata = ProjectMetadata(case_name="Case")
    placeholder_values = {"client_name": "ACME Inc"}

    if legacy:
        draft_backend = _SequenceBackend(
            [
                _model_response("Section output 1", model_name="claude-sonnet-4-5", output_tokens=101),
                _model_response("Section output 2", model_name="claude-sonnet-4-5", output_tokens=102),
            ]
        )
    else:
        draft_backend = _SequenceBackend(
            [
                _gateway_result("Section output 1"),
                _gateway_result("Section output 2", output_tokens=102),
            ]
        )

    draft_worker = DraftReportWorker(
        project_dir=project_dir,
        inputs=[(REPORT_CATEGORY_CONVERTED, "converted_documents/doc.md")],
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        custom_model=None,
        context_window=None,
        template_path=template_path,
        transcript_path=None,
        generation_user_prompt_path=generation_user_prompt_path,
        generation_system_prompt_path=generation_system_prompt_path,
        metadata=metadata,
        placeholder_values=placeholder_values,
        project_name="Case",
        llm_backend=draft_backend,
    )

    draft_failures: list[str] = []
    draft_results: list[dict[str, Any]] = []
    draft_worker.failed.connect(draft_failures.append)
    draft_worker.finished.connect(lambda payload: draft_results.append(payload))
    draft_worker.run()
    assert not draft_failures
    assert draft_results

    draft_result = draft_results[0]
    draft_path = Path(draft_result["draft_path"])

    if legacy:
        refine_backend = _SequenceBackend(
            [
                _model_response(
                    "Refined content",
                    model_name="claude-sonnet-4-5",
                    output_tokens=111,
                    reasoning="Reasoning trace",
                )
            ]
        )
    else:
        refine_backend = _SequenceBackend(
            [
                _gateway_result(
                    "Refined content",
                    output_tokens=111,
                    reasoning="Reasoning trace",
                )
            ]
        )

    refine_worker = ReportRefinementWorker(
        project_dir=project_dir,
        draft_path=draft_path,
        inputs=[(REPORT_CATEGORY_CONVERTED, "converted_documents/doc.md")],
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        custom_model=None,
        context_window=None,
        template_path=template_path,
        transcript_path=None,
        refinement_user_prompt_path=refinement_user_prompt_path,
        refinement_system_prompt_path=refinement_system_prompt_path,
        metadata=metadata,
        placeholder_values=placeholder_values,
        project_name="Case",
        llm_backend=refine_backend,
    )

    refine_failures: list[str] = []
    refine_results: list[dict[str, Any]] = []
    refine_worker.failed.connect(refine_failures.append)
    refine_worker.finished.connect(lambda payload: refine_results.append(payload))
    refine_worker.run()
    assert not refine_failures
    assert refine_results

    refine_result = refine_results[0]
    return {
        "draft_result": _normalize_value(draft_result, project_dir),
        "draft_markdown": _read_markdown_snapshot(Path(draft_result["draft_path"]), project_dir),
        "draft_manifest": _read_json_snapshot(Path(draft_result["manifest_path"]), project_dir),
        "draft_inputs": _read_markdown_snapshot(Path(draft_result["inputs_path"]), project_dir),
        "refine_result": _normalize_value(refine_result, project_dir),
        "refined_markdown": _read_markdown_snapshot(Path(refine_result["refined_path"]), project_dir),
        "refine_manifest": _read_json_snapshot(Path(refine_result["manifest_path"]), project_dir),
        "refine_inputs": _read_markdown_snapshot(Path(refine_result["inputs_path"]), project_dir),
        "reasoning_markdown": _read_markdown_snapshot(Path(refine_result["reasoning_path"]), project_dir),
    }


def _run_bulk_map(
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy: bool,
) -> dict[str, Any]:
    map_group, _ = _prepare_bulk_project(project_dir)
    metadata = ProjectMetadata(case_name="Case")
    relative_path = converted_artifact_relative("doc.pdf")

    if legacy:
        llm_backend = _SequenceBackend(
            [
                _model_response("summary", model_name="claude-sonnet-4-5", output_tokens=7)
            ]
        )
    else:
        llm_backend = _SequenceBackend([_gateway_result("summary", output_tokens=7)])

    worker = BulkAnalysisWorker(
        project_dir=project_dir,
        group=map_group,
        files=[relative_path],
        metadata=metadata,
        force_rerun=True,
        placeholder_values={},
        project_name="Case",
        llm_backend=llm_backend,
    )
    worker._run()

    output_path = project_dir / "bulk_analysis" / map_group.folder_name / "doc.pdf_analysis.md"
    manifest_path = project_dir / "bulk_analysis" / map_group.folder_name / "manifest.json"
    return {
        "output": _read_markdown_snapshot(output_path, project_dir),
        "manifest": _read_json_snapshot(manifest_path, project_dir),
    }


def _run_bulk_reduce(
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy: bool,
) -> dict[str, Any]:
    _, reduce_group = _prepare_bulk_project(project_dir)
    metadata = ProjectMetadata(case_name="Case")

    if legacy:
        llm_backend = _SequenceBackend(
            [
                _model_response("summary", model_name="claude-sonnet-4-5", output_tokens=7)
            ]
        )
    else:
        llm_backend = _SequenceBackend([_gateway_result("summary", output_tokens=7)])

    worker = BulkReduceWorker(
        project_dir=project_dir,
        group=reduce_group,
        metadata=metadata,
        force_rerun=True,
        placeholder_values={},
        project_name="Case",
        llm_backend=llm_backend,
    )
    worker._run()

    output_path = project_dir / "bulk_analysis" / reduce_group.folder_name / "reduce" / "combined_output.md"
    state_manifest_path = project_dir / "bulk_analysis" / reduce_group.folder_name / "reduce" / "manifest.json"
    run_manifest_path = output_path.with_suffix(".manifest.json")
    return {
        "output": _read_markdown_snapshot(output_path, project_dir),
        "state_manifest": _read_json_snapshot(state_manifest_path, project_dir),
        "run_manifest": _read_json_snapshot(run_manifest_path, project_dir),
    }


def test_report_pipeline_gateway_matches_legacy_artifacts(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    _freeze_worker_time(monkeypatch)

    legacy_snapshot = _run_report_pipeline(tmp_path / "legacy", monkeypatch, legacy=True)
    gateway_snapshot = _run_report_pipeline(tmp_path / "gateway", monkeypatch, legacy=False)

    assert gateway_snapshot == legacy_snapshot


def test_bulk_map_gateway_matches_legacy_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_worker_time(monkeypatch)

    legacy_snapshot = _run_bulk_map(tmp_path / "legacy", monkeypatch, legacy=True)
    gateway_snapshot = _run_bulk_map(tmp_path / "gateway", monkeypatch, legacy=False)

    assert gateway_snapshot == legacy_snapshot


def test_bulk_reduce_gateway_matches_legacy_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_worker_time(monkeypatch)

    legacy_snapshot = _run_bulk_reduce(tmp_path / "legacy", monkeypatch, legacy=True)
    gateway_snapshot = _run_bulk_reduce(tmp_path / "gateway", monkeypatch, legacy=False)

    assert gateway_snapshot == legacy_snapshot
