from __future__ import annotations

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

    def fake_invoke(self, provider, config, prompt, system_prompt):  # noqa: ANN001
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
