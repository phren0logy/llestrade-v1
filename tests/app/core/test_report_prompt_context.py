from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.app.core.report_prompt_context import (
    build_report_base_placeholders,
    build_report_generation_placeholders,
    build_report_refinement_placeholders,
)
from src.app.core.project_manager import ProjectMetadata


def test_build_report_base_placeholders_includes_system_values(tmp_path: Path) -> None:
    base = {"client": "ACME"}
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

    placeholders = build_report_base_placeholders(
        base_placeholders=base,
        metadata=None,
        project_name="Sample Case",
        project_dir=tmp_path,
        timestamp=timestamp,
    )

    assert placeholders["client"] == "ACME"
    assert placeholders["project_name"] == "Sample Case"
    assert placeholders["timestamp"] == timestamp.isoformat()


def test_build_report_base_placeholders_includes_metadata_defaults(tmp_path: Path) -> None:
    placeholders = build_report_base_placeholders(
        base_placeholders={},
        metadata=ProjectMetadata(
            case_name="Sample Case",
            subject_name="Jane Roe",
            date_of_birth="1970-01-02",
            case_description="Referral context",
        ),
        project_name="Sample Case",
        project_dir=tmp_path,
    )

    assert placeholders["case_name"] == "Sample Case"
    assert placeholders["subject_name"] == "Jane Roe"
    assert placeholders["subject_dob"] == "1970-01-02"
    assert placeholders["case_info"] == "Referral context"


def test_build_report_generation_placeholders_adds_dynamic_values() -> None:
    base = {"existing": "value"}

    placeholders = build_report_generation_placeholders(
        base_placeholders=base,
        template_section="Section text",
        section_title="History",
        transcript="Transcript body",
        additional_documents="Combined docs",
    )

    assert placeholders["template_section"] == "Section text"
    assert placeholders["section_title"] == "History"
    assert placeholders["transcript"] == "Transcript body"
    assert placeholders["additional_documents"] == "Combined docs"
    assert placeholders["document_content"] == "Combined docs"


def test_build_report_refinement_placeholders_adds_dynamic_values() -> None:
    base = {"existing": "value"}

    placeholders = build_report_refinement_placeholders(
        base_placeholders=base,
        draft_report="Draft content",
        template="Template content",
        transcript="Transcript content",
    )

    assert placeholders["draft_report"] == "Draft content"
    assert placeholders["template"] == "Template content"
    assert placeholders["transcript"] == "Transcript content"
