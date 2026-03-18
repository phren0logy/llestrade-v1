"""Helpers for DocTags-based converted-document storage and loading."""

from __future__ import annotations

from pathlib import Path

from .doctags import render_doctags_text


DOCTAGS_ARTIFACT_SUFFIX = ".doctags.txt"


def converted_artifact_relative(source_relative: str) -> str:
    """Return the converted artifact path for a source-relative input."""

    normalized = str(source_relative or "").strip().lstrip("/")
    if not normalized:
        return DOCTAGS_ARTIFACT_SUFFIX.lstrip(".")
    return f"{normalized}{DOCTAGS_ARTIFACT_SUFFIX}"


def source_relative_from_artifact(relative_path: str) -> str | None:
    """Return the original source-relative path for a converted artifact."""

    normalized = str(relative_path or "").strip().lstrip("/")
    if not normalized.endswith(DOCTAGS_ARTIFACT_SUFFIX):
        return None
    source_relative = normalized[: -len(DOCTAGS_ARTIFACT_SUFFIX)]
    return source_relative or None


def is_doctags_artifact(path: Path | str) -> bool:
    """Return whether the path points to a DocTags converted artifact."""

    text = path.as_posix() if isinstance(path, Path) else str(path)
    return text.lower().endswith(DOCTAGS_ARTIFACT_SUFFIX)


def load_converted_document_text(path: Path) -> str:
    """Load a converted artifact and render it for prompt/report use."""

    raw = path.read_text(encoding="utf-8")
    if is_doctags_artifact(path):
        return render_doctags_text(raw)
    return raw


def display_name_for_converted(relative_path: str) -> str:
    """Return a stable user-facing name for a converted artifact."""

    source_relative = source_relative_from_artifact(relative_path)
    if source_relative:
        return source_relative
    return relative_path


def map_output_stem(relative_path: str) -> str:
    """Return the per-document bulk output stem for a converted artifact."""

    source_relative = source_relative_from_artifact(relative_path) or relative_path
    return Path(source_relative).with_suffix("").name


__all__ = [
    "DOCTAGS_ARTIFACT_SUFFIX",
    "converted_artifact_relative",
    "display_name_for_converted",
    "is_doctags_artifact",
    "load_converted_document_text",
    "map_output_stem",
    "source_relative_from_artifact",
]
