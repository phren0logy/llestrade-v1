from __future__ import annotations

from src.app.core.bundled_prompts import canonical_prompt_key, canonical_prompt_reference
from src.config.prompt_store import get_repo_prompts_dir


def test_canonical_prompt_key_maps_legacy_identifiers() -> None:
    assert canonical_prompt_key("document_analysis_system_prompt") == "bulk_system"
    assert canonical_prompt_key("document_bulk_analysis_prompt") == "bulk_per_document"
    assert canonical_prompt_key("integrated_analysis_prompt") == "bulk_combined"
    assert canonical_prompt_key("report_generation_user_prompt") == "report_draft_user"
    assert canonical_prompt_key("report_refinement_system_prompt") == "report_refine_system"
    assert canonical_prompt_key("system_prompt") == "default_system"


def test_canonical_prompt_reference_maps_legacy_paths() -> None:
    assert canonical_prompt_reference("document_analysis_system_prompt.md") == "bulk_system.md"
    assert canonical_prompt_reference("prompts/document_bulk_analysis_prompt.md") == "prompts/bulk_per_document.md"
    assert canonical_prompt_reference("prompts/comprehensive_analysis_instructions.md") == "prompts/bulk_combined.md"
    assert canonical_prompt_reference("prompts/default_generation_user.md") == "prompts/report_draft_user.md"


def test_bundled_prompt_resources_use_current_citation_language() -> None:
    prompt_dir = get_repo_prompts_dir()
    prompt_files = sorted(prompt_dir.glob("*.md"))
    assert prompt_files

    for prompt_file in prompt_files:
        content = prompt_file.read_text(encoding="utf-8")
        assert "[CIT:ev_" not in content, prompt_file.name
        assert "Citation Evidence Ledger" not in content, prompt_file.name
        assert "change the extension from .md to .pdf" not in content, prompt_file.name

