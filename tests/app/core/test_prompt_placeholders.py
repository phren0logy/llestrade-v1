from __future__ import annotations

import pytest

from src.app.core.prompt_placeholders import (
    MissingPlaceholdersError,
    ensure_required_placeholders,
    format_prompt,
    placeholder_summary,
)


def test_ensure_required_placeholders_detects_missing() -> None:
    with pytest.raises(MissingPlaceholdersError) as excinfo:
        ensure_required_placeholders("report_draft_user", "Prompt without placeholders")

    error = excinfo.value
    assert error.prompt_key == "report_draft_user"
    assert error.missing == ("template_section", "transcript", "additional_documents")
    assert "{additional_documents}" in str(error)


def test_format_prompt_preserves_unknown_placeholders() -> None:
    template = "Section: {template_section}\nUnknown: {custom}"
    context = {"template_section": "History", "transcript": "ignored"}
    result = format_prompt(template, context)
    assert "History" in result
    assert "{custom}" in result


def test_placeholder_summary_includes_required_and_optional() -> None:
    summary = placeholder_summary("bulk_per_document")
    assert "{document_content}" in summary
    # Optional placeholders should appear as well.
    assert "{document_name}" in summary
