from __future__ import annotations

from pathlib import Path

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.bulk_recovery import BulkRecoveryStore
from src.app.core.job_cost_estimates import (
    estimate_bulk_map_cost,
    estimate_report_draft_cost,
)
from src.app.core.llm_operation_settings import LLMOperationSettings
from src.app.core.project_manager import ProjectMetadata


def test_estimate_report_draft_cost_returns_best_and_ceiling(tmp_path: Path) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    report_input = converted / "input.md"
    report_input.write_text("# Input\nBody", encoding="utf-8")

    template = project_dir / "template.md"
    template.write_text("# Section A\nBody\n\n# Section B\nBody", encoding="utf-8")
    generation_user = project_dir / "generation_user.md"
    generation_user.write_text("Section {section_title}\n\n{template_section}\n\n{additional_documents}", encoding="utf-8")
    generation_system = project_dir / "generation_system.md"
    generation_system.write_text("System for {project_name}", encoding="utf-8")

    forecast = estimate_report_draft_cost(
        project_dir=project_dir,
        inputs=[("converted", "converted_documents/input.md")],
        llm_settings=LLMOperationSettings(
            provider_id="anthropic",
            model_id="claude-sonnet-4-5",
            context_window=None,
            use_reasoning=False,
        ),
        template_path=template,
        transcript_path=None,
        generation_user_prompt_path=generation_user,
        generation_system_prompt_path=generation_system,
        metadata=ProjectMetadata(case_name="Case"),
        placeholder_values={},
        project_name="Case Project",
    )

    assert forecast.available is True
    assert forecast.best_estimate is not None
    assert forecast.ceiling is not None
    assert forecast.best_estimate <= forecast.ceiling


def test_estimate_bulk_map_cost_includes_spent_actual_for_resume(tmp_path: Path) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    source = converted / "doc.md"
    source.write_text("# Heading\nBody text", encoding="utf-8")

    group = BulkAnalysisGroup.create(
        "Group",
        files=["doc.md"],
        provider_id="anthropic",
        model="claude-sonnet-4-5",
    )
    store = BulkRecoveryStore(project_dir / "bulk_analysis" / group.folder_name)
    manifest = store.load_map_manifest()
    manifest["actual_cost"] = 1.25
    manifest["documents"] = {
        "doc.md": {
            "status": "incomplete",
            "chunks": {},
            "batches": {},
        }
    }
    store.save_map_manifest(manifest)

    forecast = estimate_bulk_map_cost(
        project_dir=project_dir,
        group=group,
        files=["doc.md"],
        metadata=ProjectMetadata(case_name="Case"),
        placeholder_values={},
        project_name="Case Project",
        force_rerun=False,
    )

    assert forecast.available is True
    assert forecast.spent_actual == 1.25
    assert forecast.remaining_best_estimate is not None
    assert forecast.remaining_ceiling is not None
    assert forecast.projected_total_best_estimate == forecast.spent_actual + forecast.remaining_best_estimate


def test_estimate_bulk_map_cost_supports_bedrock_inference_profile_ids(tmp_path: Path) -> None:
    project_dir = tmp_path
    converted = project_dir / "converted_documents"
    converted.mkdir(parents=True, exist_ok=True)
    source = converted / "doc.md"
    source.write_text("# Heading\nBody text", encoding="utf-8")

    group = BulkAnalysisGroup.create(
        "Group",
        files=["doc.md"],
        provider_id="anthropic_bedrock",
        model="us.anthropic.claude-sonnet-4-6",
    )

    forecast = estimate_bulk_map_cost(
        project_dir=project_dir,
        group=group,
        files=["doc.md"],
        metadata=ProjectMetadata(case_name="Case"),
        placeholder_values={},
        project_name="Case Project",
        force_rerun=False,
    )

    assert forecast.available is True
    assert forecast.best_estimate is not None
    assert forecast.ceiling is not None
