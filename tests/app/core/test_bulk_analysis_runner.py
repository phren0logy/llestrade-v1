"""Tests for bulk analysis prompt handling."""

from __future__ import annotations

import pytest

from src.app.core import bulk_analysis_runner as runner
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.prompt_placeholders import MissingPlaceholdersError


class _StubPromptManager:
    def get_template(self, name: str) -> str:
        if name == "document_analysis_system_prompt":
            return "System"
        if name == "document_bulk_analysis_prompt":
            return "Summary without placeholder"
        raise KeyError(name)


def test_load_prompts_requires_document_placeholder(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    group = BulkAnalysisGroup.create("Group")

    monkeypatch.setattr(runner, "PromptManager", lambda: _StubPromptManager())

    with pytest.raises(MissingPlaceholdersError) as excinfo:
        runner.load_prompts(tmp_path, group, metadata=None)

    assert "{document_content}" in str(excinfo.value)


def test_render_prompts_apply_placeholder_values() -> None:
    bundle = runner.PromptBundle(
        system_template="Welcome {project_name} for {client}",
        user_template="{document_content} -- {client}",
    )
    values = {"client": "ACME Corp", "project_name": "Case-42"}
    system = runner.render_system_prompt(bundle, metadata=None, placeholder_values=values)
    user = runner.render_user_prompt(
        bundle,
        metadata=None,
        document_name="doc.md",
        document_content="Body",
        placeholder_values=values,
    )
    assert system == "Welcome Case-42 for ACME Corp"
    assert user.startswith("Body -- ACME Corp")


def test_combine_chunk_summaries_uses_placeholder_values() -> None:
    summaries = ["Summary A", "Summary B"]
    prompt, context = runner.combine_chunk_summaries(
        summaries,
        document_name="Doc1",
        metadata=None,
        placeholder_values={"client": "ACME"},
    )
    assert "ACME" in prompt
    assert context["client"] == "ACME"


def test_combine_chunk_summaries_hierarchical_uses_single_pass_for_small_docs() -> None:
    """Test that small documents use the fast single-pass path."""
    summaries = ["Summary A", "Summary B", "Summary C"]
    invoke_calls = []

    def mock_invoke(prompt: str) -> str:
        invoke_calls.append(prompt)
        return "Combined result"

    result = runner.combine_chunk_summaries_hierarchical(
        summaries,
        document_name="SmallDoc",
        metadata=None,
        placeholder_values=None,
        provider_id="anthropic",
        model="claude-sonnet-4-5-20250929",
        invoke_fn=mock_invoke,
        is_cancelled_fn=None,
    )

    # Should use single-pass (only 1 invoke call)
    assert len(invoke_calls) == 1
    assert result == "Combined result"


def test_combine_chunk_summaries_hierarchical_batches_large_docs() -> None:
    """Test that large documents trigger hierarchical batching."""
    # Create summaries that will exceed token limit when combined
    # Each summary is ~10000 characters (~2500 tokens)
    # Claude's safe window ~130K tokens, and we use ~95% of that for combining (~123K)
    # Need >123K tokens to trigger hierarchical, so use 55 summaries ≈ 137K tokens
    large_summary = "x" * 10000
    summaries = [large_summary] * 55  # 55 summaries ≈ 137,500 tokens

    invoke_calls = []

    def mock_invoke(prompt: str) -> str:
        invoke_calls.append(prompt)
        # Return a smaller summary for next level
        return "Batch summary " + str(len(invoke_calls))

    result = runner.combine_chunk_summaries_hierarchical(
        summaries,
        document_name="LargeDoc",
        metadata=None,
        placeholder_values=None,
        provider_id="anthropic",
        model="claude-sonnet-4-5-20250929",
        invoke_fn=mock_invoke,
        is_cancelled_fn=None,
    )

    # Should use hierarchical reduction (multiple invoke calls)
    assert len(invoke_calls) > 1, f"Expected hierarchical batching for large document, but got {len(invoke_calls)} calls"
    # Final result should be from the last batch
    assert "Batch summary" in result


def test_combine_chunk_summaries_hierarchical_respects_cancellation() -> None:
    """Test that cancellation is respected during hierarchical reduction."""
    # Create large enough summaries to trigger hierarchical batching
    large_summary = "x" * 10000
    summaries = [large_summary] * 60
    cancel_after_first = True

    def is_cancelled():
        return cancel_after_first

    call_count = [0]

    def mock_invoke(prompt: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            # After first invoke, set cancellation flag
            nonlocal cancel_after_first
            cancel_after_first = True
            return "First batch result"
        return "Should not reach here"

    # Should raise BulkAnalysisCancelled when cancellation is detected
    with pytest.raises(runner.BulkAnalysisCancelled):
        runner.combine_chunk_summaries_hierarchical(
            summaries,
            document_name="Doc",
            metadata=None,
            placeholder_values=None,
            provider_id="anthropic",
            model="claude-sonnet-4-5-20250929",
            invoke_fn=mock_invoke,
            is_cancelled_fn=is_cancelled,
        )


def test_combine_chunk_summaries_hierarchical_wraps_errors_with_context() -> None:
    """Test that errors during hierarchical reduction include helpful context."""
    # Create large summaries to trigger hierarchical batching
    large_summary = "x" * 10000
    summaries = [large_summary] * 60

    def mock_invoke(prompt: str) -> str:
        raise RuntimeError("Provider error")

    # Should wrap the error with context about which batch failed
    with pytest.raises(Exception) as excinfo:
        runner.combine_chunk_summaries_hierarchical(
            summaries,
            document_name="Doc",
            metadata=None,
            placeholder_values=None,
            provider_id="anthropic",
            model="claude-sonnet-4-5-20250929",
            invoke_fn=mock_invoke,
            is_cancelled_fn=None,
        )

    # Error message should include context about hierarchical reduction
    error_msg = str(excinfo.value)
    # During hierarchical batching, errors should be wrapped with context
    assert "Hierarchical reduction failed" in error_msg or ("level" in error_msg.lower() and "batch" in error_msg.lower())


def test_generate_chunks_enforces_hard_cap_for_large_sections() -> None:
    max_tokens = 100
    large_section = "A" * 5000
    text = f"# Header\n\n{large_section}"

    chunks = runner.generate_chunks(text, max_tokens=max_tokens)

    assert chunks
    max_chars = max_tokens * 4
    assert all(len(chunk) <= max_chars for chunk in chunks)
