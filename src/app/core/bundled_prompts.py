"""Canonical bundled prompt definitions and legacy aliases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BundledPrompt:
    key: str
    filename: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    description: str | None = None
    legacy_keys: tuple[str, ...] = ()
    legacy_filenames: tuple[str, ...] = ()


def _sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(values)))


PROMPTS: tuple[BundledPrompt, ...] = (
    BundledPrompt(
        key="bulk_system",
        filename="bulk_system.md",
        optional=_sorted(("subject_name", "subject_dob", "case_info")),
        description="System prompt for bulk analysis runs.",
        legacy_keys=("document_analysis_system_prompt",),
        legacy_filenames=("document_analysis_system_prompt.md",),
    ),
    BundledPrompt(
        key="bulk_per_document",
        filename="bulk_per_document.md",
        required=("document_content",),
        optional=_sorted(("subject_name", "subject_dob", "case_info", "document_name", "chunk_index", "chunk_total")),
        description="User prompt for per-document bulk analysis.",
        legacy_keys=("document_bulk_analysis_prompt",),
        legacy_filenames=("document_bulk_analysis_prompt.md", "per_document_anaysis.md"),
    ),
    BundledPrompt(
        key="bulk_combined",
        filename="bulk_combined.md",
        required=("document_content",),
        optional=_sorted(("subject_name", "subject_dob", "case_info")),
        description="User prompt for combined bulk analysis.",
        legacy_keys=("integrated_analysis_prompt",),
        legacy_filenames=("integrated_analysis_prompt.md", "comprehensive_analysis_instructions.md"),
    ),
    BundledPrompt(
        key="report_draft_system",
        filename="report_draft_system.md",
        description="System prompt for report section generation.",
        legacy_keys=("report_generation_system_prompt",),
        legacy_filenames=("report_generation_system_prompt.md", "default_generation_system.md"),
    ),
    BundledPrompt(
        key="report_draft_user",
        filename="report_draft_user.md",
        required=("template_section", "transcript", "additional_documents"),
        optional=_sorted(("section_title", "document_content")),
        description="User prompt for report section generation.",
        legacy_keys=("report_generation_user_prompt", "report_generation_instructions", "section_processing"),
        legacy_filenames=(
            "report_generation_user_prompt.md",
            "report_generation_instructions.md",
            "section_processing.md",
            "default_generation_user.md",
        ),
    ),
    BundledPrompt(
        key="report_refine_system",
        filename="report_refine_system.md",
        description="System prompt for report refinement.",
        legacy_keys=("report_refinement_system_prompt",),
        legacy_filenames=("report_refinement_system_prompt.md", "default_refinement_system.md"),
    ),
    BundledPrompt(
        key="report_refine_user",
        filename="report_refine_user.md",
        required=("draft_report", "template"),
        optional=_sorted(("transcript",)),
        description="User prompt for refining the generated report draft.",
        legacy_keys=("refinement_prompt",),
        legacy_filenames=("refinement_prompt.md", "default_refinement_user.md"),
    ),
    BundledPrompt(
        key="default_system",
        filename="default_system.md",
        description="Default global system prompt.",
        legacy_keys=("system_prompt",),
        legacy_filenames=("system_prompt.md",),
    ),
)


PROMPTS_BY_KEY = {prompt.key: prompt for prompt in PROMPTS}
PROMPTS_BY_FILENAME = {prompt.filename: prompt for prompt in PROMPTS}
PROMPTS_BY_ALIAS: dict[str, BundledPrompt] = {}
for prompt in PROMPTS:
    PROMPTS_BY_ALIAS[prompt.key] = prompt
    PROMPTS_BY_ALIAS[prompt.filename] = prompt
    for alias in prompt.legacy_keys:
        PROMPTS_BY_ALIAS[alias] = prompt
    for alias in prompt.legacy_filenames:
        PROMPTS_BY_ALIAS[alias] = prompt


def canonical_prompt_key(value: str) -> str:
    prompt = PROMPTS_BY_ALIAS.get(str(value or "").strip())
    if prompt is None:
        return str(value or "").strip()
    return prompt.key


def canonical_prompt_filename(value: str) -> str:
    prompt = PROMPTS_BY_ALIAS.get(str(value or "").strip())
    if prompt is None:
        return str(value or "").strip()
    return prompt.filename


def canonical_prompt_reference(path_str: str) -> str:
    """Return a normalized prompt path/reference for known bundled prompts."""

    raw = str(path_str or "").strip()
    if not raw:
        return ""

    candidate = Path(raw)
    filename = canonical_prompt_filename(candidate.name)
    if filename == candidate.name:
        return raw

    if candidate.is_absolute():
        return str(candidate.with_name(filename))

    if len(candidate.parts) >= 2 and candidate.parts[0] in {"prompts", "reports"}:
        return f"prompts/{filename}"

    if len(candidate.parts) == 1:
        return filename

    return str(candidate.with_name(filename))


def prompt_definition(value: str) -> BundledPrompt | None:
    return PROMPTS_BY_ALIAS.get(str(value or "").strip())


__all__ = [
    "BundledPrompt",
    "PROMPTS",
    "PROMPTS_BY_ALIAS",
    "PROMPTS_BY_FILENAME",
    "PROMPTS_BY_KEY",
    "canonical_prompt_filename",
    "canonical_prompt_key",
    "canonical_prompt_reference",
    "prompt_definition",
]
