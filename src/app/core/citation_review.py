"""Helpers for loading review-ready citation payloads for bulk outputs."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import frontmatter

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.bulk_paths import iter_map_outputs
from src.app.core.citations import CitationStore

_LOCAL_CITATION_TOKEN_RE = re.compile(r"\[(C\d{1,5})\]")


@dataclass(frozen=True)
class PdfTarget:
    citation_label: str
    source_path: str | None
    page_number: int | None
    boxes: tuple[dict[str, float], ...]
    snippet: str
    status: str
    reason: str


@dataclass(frozen=True)
class ReviewedCitation:
    citation_label: str
    token: str
    ev_id: str | None
    status: str
    reason: str
    confidence: float
    document_relative_path: str | None
    source_relative_path: str | None
    source_absolute_path: str | None
    page_number: int | None
    snippet: str
    boxes: tuple[dict[str, float], ...]
    mention_count: int

    def to_pdf_target(self) -> PdfTarget:
        return PdfTarget(
            citation_label=self.citation_label,
            source_path=self.source_absolute_path,
            page_number=self.page_number,
            boxes=self.boxes,
            snippet=self.snippet,
            status=self.status,
            reason=self.reason,
        )


@dataclass(frozen=True)
class ReviewedOutput:
    output_path: Path
    relative_key: str
    text: str
    rendered_html: str
    citations: tuple[ReviewedCitation, ...]


class CitationReviewService:
    """Load review data for bulk map outputs."""

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = Path(project_dir).resolve()
        self._store = CitationStore(self._project_dir)

    def list_map_outputs(self, group: BulkAnalysisGroup) -> list[tuple[Path, str]]:
        outputs = list(iter_map_outputs(self._project_dir, group.folder_name))
        outputs.sort(key=lambda item: item[0].stat().st_mtime if item[0].exists() else 0, reverse=True)
        return outputs

    def load_reviewed_output(self, output_path: Path, *, relative_key: str | None = None) -> ReviewedOutput:
        post = frontmatter.load(output_path)
        text = (post.content or "").strip()
        mentions = self._store.list_output_citation_mentions(output_path)
        grouped: dict[str, list] = {}
        for mention in mentions:
            label = mention.citation_label or mention.token.strip("[]")
            grouped.setdefault(label, []).append(mention)

        citations: list[ReviewedCitation] = []
        for label, items in grouped.items():
            primary = items[0]
            bundle = self._store.get_evidence_bundle(primary.ev_id) if primary.ev_id else None
            boxes = tuple(
                box
                for box in (
                    (item.get("normalized_bbox") if isinstance(item, dict) else None)
                    for item in (bundle.geometry if bundle else [])
                )
                if isinstance(box, dict)
            )
            source_absolute_path = primary.source_absolute_path or (bundle.source_absolute_path if bundle else None)
            source_relative_path = primary.source_relative_path or (bundle.source_relative_path if bundle else None)
            if not source_absolute_path and source_relative_path:
                candidate = (self._project_dir / source_relative_path).resolve()
                source_absolute_path = candidate.as_posix()

            citations.append(
                ReviewedCitation(
                    citation_label=label,
                    token=primary.token,
                    ev_id=primary.ev_id,
                    status=primary.status,
                    reason=primary.reason,
                    confidence=primary.confidence,
                    document_relative_path=primary.document_relative_path,
                    source_relative_path=source_relative_path,
                    source_absolute_path=source_absolute_path,
                    page_number=primary.page_number or (bundle.page_number if bundle else None),
                    snippet=(bundle.text if bundle else ""),
                    boxes=boxes,
                    mention_count=len(items),
                )
            )

        citations.sort(key=lambda item: (_citation_sort_key(item.citation_label), item.page_number or 0))
        return ReviewedOutput(
            output_path=output_path,
            relative_key=relative_key or output_path.name,
            text=text,
            rendered_html=_render_output_html(text),
            citations=tuple(citations),
        )


def _citation_sort_key(label: str) -> int:
    match = re.match(r"^C(\d+)$", label)
    return int(match.group(1)) if match else 999999


def _render_output_html(text: str) -> str:
    escaped = html.escape(text)
    rendered = _LOCAL_CITATION_TOKEN_RE.sub(r'<a href="citation:\1">[\1]</a>', escaped)
    return f"<pre style='white-space: pre-wrap; font-family: monospace;'>{rendered}</pre>"


__all__ = [
    "CitationReviewService",
    "PdfTarget",
    "ReviewedCitation",
    "ReviewedOutput",
]
