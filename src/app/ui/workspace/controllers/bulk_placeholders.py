"""Placeholder-analysis helpers for bulk-analysis groups."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.bulk_analysis_runner import load_prompts
from src.app.core.placeholders.analyzer import PlaceholderAnalysis, analyse_prompts
from src.app.core.prompt_placeholders import get_prompt_spec

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.app.core.project_manager import ProjectManager


def analyse_group_placeholders(
    project_manager: "ProjectManager" | None,
    group: BulkAnalysisGroup,
    *,
    prompt_loader: Callable = load_prompts,
    prompt_spec_getter: Callable[[str], object | None] = get_prompt_spec,
) -> tuple[PlaceholderAnalysis | None, set[str], set[str]]:
    manager = project_manager
    if not manager or not manager.project_dir:
        return None, set(), set()
    try:
        bundle = prompt_loader(Path(manager.project_dir), group, manager.metadata)
    except Exception:
        return None, set(), set()

    values = manager.placeholder_mapping()

    metadata = getattr(manager, "metadata", None)
    if metadata:
        values.setdefault("subject_name", metadata.subject_name or metadata.case_name or "")
        values.setdefault("subject_dob", metadata.date_of_birth or "")
        values.setdefault("case_info", metadata.case_description or "")
        values.setdefault("case_name", metadata.case_name or "")

    required: set[str] = set()
    optional: set[str] = set()
    if group.placeholder_requirements:
        for key, is_required in group.placeholder_requirements.items():
            if is_required:
                required.add(key)
            else:
                optional.add(key)
    else:
        user_spec = prompt_spec_getter("document_bulk_analysis_prompt")
        if user_spec:
            required.update(user_spec.required)
            optional.update(user_spec.optional)
        system_spec = prompt_spec_getter("document_analysis_system_prompt")
        if system_spec:
            required.update(system_spec.required)
            optional.update(system_spec.optional)

    if group.operation == "per_document":
        required.add("document_content")
        values.setdefault("document_content", "<document>")
        if group.files:
            values.setdefault("document_name", group.files[0])
        else:
            values.setdefault("document_name", "<document>")
    else:
        optional.update(
            {
                "reduce_source_list",
                "reduce_source_table",
                "reduce_source_count",
            }
        )

    dynamic_keys = {
        "document_content",
        "source_pdf_filename",
        "source_pdf_relative_path",
        "source_pdf_absolute_path",
        "source_pdf_absolute_url",
        "reduce_source_list",
        "reduce_source_table",
        "reduce_source_count",
    }

    analysis = analyse_prompts(
        bundle.system_template,
        bundle.user_template,
        available_values=values,
        required_keys=required,
        optional_keys=optional,
    )

    missing_required = set(analysis.missing_required) - dynamic_keys
    missing_optional = set(analysis.missing_optional) - dynamic_keys
    return analysis, missing_required, missing_optional
