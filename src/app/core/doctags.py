"""Helpers for working with Docling DocTags artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Mapping, Sequence


DOCTAGS_SUFFIX = ".doctags.txt"
DOCTAGS_GRID_SIZE = 500.0

_PAGE_BREAK_RE = re.compile(r"<page_break\s*/?>", re.IGNORECASE)
_LOC_TOKEN_RE = re.compile(r"<loc_(\d+)>", re.IGNORECASE)
_TAG_TOKEN_RE = re.compile(r"<[^>]+>|[^<]+")
_PAGE_INDEX_RE = re.compile(r"page[_:-]?(\d+)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocTagsBlock:
    page_number: int
    ordinal: int
    text: str
    normalized_text: str
    bbox: dict[str, float] | None
    normalized_bbox: dict[str, float] | None
    page_width: float
    page_height: float
    unit: str


@dataclass(frozen=True)
class DocTagsSpan:
    page_number: int
    ordinal: int
    text: str
    normalized_text: str
    grid_bbox: dict[str, float] | None
    normalized_bbox: dict[str, float] | None


def is_doctags_artifact(path: Path | str) -> bool:
    text = path.as_posix() if isinstance(path, Path) else str(path)
    return text.lower().endswith(DOCTAGS_SUFFIX)


def converted_output_path(relative: str, *, doctags: bool = True) -> Path:
    base = Path(relative)
    if doctags:
        return Path(f"{base.as_posix()}{DOCTAGS_SUFFIX}")
    return base.with_suffix(".md")


def is_supported_converted_input(path: Path | str) -> bool:
    text = path.as_posix() if isinstance(path, Path) else str(path)
    lowered = text.lower()
    return lowered.endswith(DOCTAGS_SUFFIX) or lowered.endswith(".md") or lowered.endswith(".txt")


def prompt_text_from_converted(raw: str, *, path: Path | None = None) -> str:
    if path is not None and is_doctags_artifact(path):
        return doctags_to_text(raw)
    if raw.startswith("---\n"):
        parts = raw.split("\n---\n", 1)
        if len(parts) == 2:
            return parts[1]
    return raw


def doctags_to_text(raw: str) -> str:
    parts: list[str] = []
    for token in _TAG_TOKEN_RE.findall(raw):
        if not token:
            continue
        if token.startswith("<"):
            if _PAGE_BREAK_RE.fullmatch(token):
                parts.append("\n\n")
            elif _LOC_TOKEN_RE.fullmatch(token):
                continue
            else:
                parts.append("\n")
            continue
        parts.append(token)
    text = "".join(parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def render_doctags_text(raw: str) -> str:
    """Return prompt/report-friendly text for a DocTags artifact."""

    return doctags_to_text(raw)


def render_doctags_pages(raw: str) -> list[str]:
    """Return one rendered text block per DocTags page."""

    pages: list[list[str]] = [[]]
    for token in _TAG_TOKEN_RE.findall(raw):
        if not token:
            continue
        if token.startswith("<"):
            if _PAGE_BREAK_RE.fullmatch(token):
                pages.append([])
            elif _LOC_TOKEN_RE.fullmatch(token):
                continue
            else:
                pages[-1].append("\n")
            continue
        pages[-1].append(token)
    rendered: list[str] = []
    for page in pages:
        text = "".join(page)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if text:
            rendered.append(text)
    return rendered


def page_numbers_from_converted(raw: str, *, path: Path | None = None) -> list[int]:
    if path is None or not is_doctags_artifact(path):
        from src.app.core.citations import PAGE_MARKER_RE

        pages = [int(match.group(1)) for match in PAGE_MARKER_RE.finditer(raw)]
        return list(dict.fromkeys(pages))

    pages: list[int] = [1]
    for tag in re.findall(r"<[^>]+>", raw):
        if _PAGE_BREAK_RE.fullmatch(tag):
            pages.append(pages[-1] + 1)
            continue
        match = _PAGE_INDEX_RE.search(tag)
        if match:
            page = int(match.group(1))
            if page not in pages:
                pages.append(page)
    return sorted(set(pages))


def parse_doctags_blocks(raw: str) -> list[DocTagsBlock]:
    page_number = 1
    ordinal = 0
    pending_locs: list[int] = []
    buffer: list[str] = []
    blocks: list[DocTagsBlock] = []

    def flush() -> None:
        nonlocal ordinal
        text = _normalize_plain_text("".join(buffer))
        if not text:
            buffer.clear()
            pending_locs.clear()
            return
        ordinal += 1
        bbox = _bbox_from_locs(pending_locs)
        blocks.append(
            DocTagsBlock(
                page_number=page_number,
                ordinal=ordinal,
                text=text,
                normalized_text=_normalize_text(text),
                bbox=bbox,
                normalized_bbox=_normalize_doctags_bbox(bbox),
                page_width=DOCTAGS_GRID_SIZE,
                page_height=DOCTAGS_GRID_SIZE,
                unit="grid",
            )
        )
        buffer.clear()
        pending_locs.clear()

    for token in _TAG_TOKEN_RE.findall(raw):
        if not token:
            continue
        if token.startswith("<"):
            if _PAGE_BREAK_RE.fullmatch(token):
                flush()
                page_number += 1
                ordinal = 0
                continue

            loc_match = _LOC_TOKEN_RE.fullmatch(token)
            if loc_match:
                pending_locs.append(int(loc_match.group(1)))
                continue

            page_match = _PAGE_INDEX_RE.search(token)
            if page_match:
                flush()
                page_number = max(int(page_match.group(1)), 1)
                ordinal = 0
                continue

            if buffer:
                flush()
            continue

        if token.strip():
            buffer.append(token)

    flush()
    return blocks


def build_geometry_spans_from_doctags(raw: str) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for block in parse_doctags_blocks(raw):
        spans.append(
            {
                "page_number": block.page_number,
                "text": block.text,
                "normalized_text": block.normalized_text,
                "polygon_json": None,
                "bbox_json": json.dumps(block.bbox) if block.bbox is not None else None,
                "normalized_bbox_json": json.dumps(block.normalized_bbox) if block.normalized_bbox is not None else None,
                "page_width": block.page_width,
                "page_height": block.page_height,
                "unit": block.unit,
                "source_path": f"doctags.pages[{block.page_number - 1}].blocks[{block.ordinal - 1}]",
                "chunk_index": block.ordinal - 1,
            }
        )
    return spans


def extract_doctags_spans(raw: str) -> list[DocTagsSpan]:
    spans: list[DocTagsSpan] = []
    for block in parse_doctags_blocks(raw):
        spans.append(
            DocTagsSpan(
                page_number=block.page_number,
                ordinal=block.ordinal,
                text=block.text,
                normalized_text=block.normalized_text,
                grid_bbox=block.bbox,
                normalized_bbox=block.normalized_bbox,
            )
        )
    return spans


def _normalize_plain_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def _bbox_from_locs(values: Sequence[int]) -> dict[str, float] | None:
    if len(values) < 4:
        return None
    xs = [float(v) for idx, v in enumerate(values[:4]) if idx % 2 == 0]
    ys = [float(v) for idx, v in enumerate(values[:4]) if idx % 2 == 1]
    if len(xs) != 2 or len(ys) != 2:
        return None
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def _normalize_doctags_bbox(bbox: Mapping[str, float] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    return {
        "x_min": max(0.0, min(float(bbox["x_min"]) / DOCTAGS_GRID_SIZE, 1.0)),
        "y_min": max(0.0, min(float(bbox["y_min"]) / DOCTAGS_GRID_SIZE, 1.0)),
        "x_max": max(0.0, min(float(bbox["x_max"]) / DOCTAGS_GRID_SIZE, 1.0)),
        "y_max": max(0.0, min(float(bbox["y_max"]) / DOCTAGS_GRID_SIZE, 1.0)),
    }


__all__ = [
    "DOCTAGS_SUFFIX",
    "DocTagsBlock",
    "DocTagsSpan",
    "build_geometry_spans_from_doctags",
    "converted_output_path",
    "doctags_to_text",
    "extract_doctags_spans",
    "is_doctags_artifact",
    "is_supported_converted_input",
    "page_numbers_from_converted",
    "parse_doctags_blocks",
    "prompt_text_from_converted",
    "render_doctags_pages",
    "render_doctags_text",
]
