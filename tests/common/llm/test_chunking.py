"""Unit tests for markdown-based chunking strategies."""

from __future__ import annotations

from src.common.llm.chunking import ChunkingStrategy


def _long_section(title: str, repeat: int) -> str:
    body = ("Paragraph text with details.\n" * repeat).strip()
    return f"## {title}\n\n{body}\n"


def test_markdown_headers_split_on_section_boundaries() -> None:
    """Chunks should respect markdown headers when enforcing token budgets."""
    text = "# Document\n\n" + _long_section("Section 1", 40) + "\n" + _long_section("Section 2", 40)
    # max_tokens -> ~400 characters, so each section becomes its own chunk.
    chunks = ChunkingStrategy.markdown_headers(text=text, max_tokens=100, overlap=0)

    assert len(chunks) >= 2
    assert "## Section 1" in chunks[0]
    assert any("## Section 2" in chunk for chunk in chunks[1:])


def test_markdown_headers_apply_character_overlap() -> None:
    """Verifies overlap characters show up at the start of the following chunk."""
    text = "# Doc\n\n" + _long_section("Section A", 30) + "\n" + _long_section("Section B", 30)
    chunks = ChunkingStrategy.markdown_headers(text=text, max_tokens=120, overlap=200)

    assert len(chunks) >= 2
    shared_tail = chunks[0][-200:].strip()
    assert shared_tail
    assert shared_tail in chunks[1]


def test_markdown_headers_handles_documents_without_headers() -> None:
    """Plain text without headers should still return a single chunk with original content."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = ChunkingStrategy.markdown_headers(text=text, max_tokens=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].startswith("Paragraph one.")
    assert "Paragraph three." in chunks[0]
