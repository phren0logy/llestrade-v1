"""Utilities for building highlight extraction jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

from .converted_documents import converted_artifact_relative
from .project_manager import ProjectManager


@dataclass(frozen=True)
class HighlightJob:
    """Specification for extracting highlights from a document."""

    source_pdf: Path
    pdf_relative: str
    converted_relative: str
    converted_document: Path
    highlight_relative: str
    highlight_output: Path


def build_highlight_jobs(project_manager: ProjectManager) -> List[HighlightJob]:
    """Return highlight jobs for PDFs with existing converted DocTags outputs."""

    project_dir = project_manager.project_dir
    if not project_dir:
        return []

    state = project_manager.source_state
    if not state.root:
        return []

    root_path = _resolve_root(project_dir, state.root)
    if root_path is None or not root_path.exists():
        return []

    converted_root = project_dir / "converted_documents"
    if not converted_root.exists():
        return []

    highlights_root = project_dir / "highlights"
    documents_root = highlights_root / "documents"

    scan_roots: Set[Path] = set()
    if state.include_root_files:
        scan_roots.add(root_path)

    for folder in state.selected_folders or []:
        candidate = root_path / folder
        if candidate.is_dir():
            scan_roots.add(candidate)

    jobs: List[HighlightJob] = []
    seen_relatives: Set[str] = set()

    for base in sorted(scan_roots):
        for pdf_path in base.rglob("*.pdf"):
            relative = pdf_path.relative_to(root_path).as_posix()
            if relative in seen_relatives:
                continue
            seen_relatives.add(relative)

            converted_relative = converted_artifact_relative(relative)
            converted_document = converted_root / converted_relative
            if not converted_document.exists():
                continue

            highlight_relative = Path(relative).with_name(Path(relative).name + ".highlights.md").as_posix()
            highlight_output = documents_root / highlight_relative

            jobs.append(
                HighlightJob(
                    source_pdf=pdf_path,
                    pdf_relative=relative,
                    converted_relative=converted_relative,
                    converted_document=converted_document,
                    highlight_relative=highlight_relative,
                    highlight_output=highlight_output,
                )
            )

    return jobs


def _resolve_root(project_dir: Path, root_spec: str) -> Optional[Path]:
    path = Path(root_spec)
    if not path.is_absolute():
        path = (project_dir / root_spec).resolve()
    return path


__all__ = ["HighlightJob", "build_highlight_jobs"]
