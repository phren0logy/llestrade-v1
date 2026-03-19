"""Shared placeholder helpers for report generation and refinement workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional

from src.app.core.project_manager import ProjectMetadata
from src.app.core.placeholders.system import system_placeholder_map


def build_report_base_placeholders(
    *,
    base_placeholders: Mapping[str, str],
    metadata: ProjectMetadata | None,
    project_name: Optional[str],
    project_dir: Optional[Path],
    timestamp: Optional[datetime] = None,
) -> Dict[str, str]:
    """Return the base placeholder map used for report prompts."""

    effective_name = (project_name or "").strip()
    if not effective_name and project_dir:
        try:
            effective_name = project_dir.resolve().name
        except Exception:
            effective_name = project_dir.name

    ts = timestamp or datetime.now(timezone.utc)

    values: Dict[str, str] = dict(base_placeholders)
    if metadata is not None:
        values.setdefault("case_name", metadata.case_name or "")
        values.setdefault("subject_name", metadata.subject_name or metadata.case_name or "")
        values.setdefault("subject_dob", metadata.date_of_birth or "")
        values.setdefault("case_info", metadata.case_description or "")
    values.update(
        system_placeholder_map(
            project_name=effective_name,
            timestamp=ts,
        )
    )
    return values


def build_report_generation_placeholders(
    *,
    base_placeholders: Mapping[str, str],
    template_section: str,
    section_title: str,
    transcript: str,
    additional_documents: str,
) -> Dict[str, str]:
    """Return placeholders for report draft generation prompts."""

    values = dict(base_placeholders)
    values.update(
        {
            "template_section": template_section or "",
            "section_title": section_title or "",
            "transcript": transcript or "",
            "additional_documents": additional_documents or "",
            "document_content": additional_documents or "",
        }
    )
    return values


def build_report_refinement_placeholders(
    *,
    base_placeholders: Mapping[str, str],
    draft_report: str,
    template: str,
    transcript: str,
) -> Dict[str, str]:
    """Return placeholders for report refinement prompts."""

    values = dict(base_placeholders)
    values.update(
        {
            "draft_report": draft_report or "",
            "template": template or "",
            "transcript": transcript or "",
        }
    )
    return values


__all__ = [
    "build_report_base_placeholders",
    "build_report_generation_placeholders",
    "build_report_refinement_placeholders",
]
