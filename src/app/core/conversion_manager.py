"""Build conversion jobs for the dashboard pipeline."""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import frontmatter

from .converted_documents import converted_artifact_relative
from .citations import CitationStore
from .project_manager import ProjectManager

LOGGER = logging.getLogger(__name__)

SUPPORTED_PDF_EXTENSIONS = {".pdf"}


@dataclass(frozen=True)
class ConversionJob:
    source_path: Path
    relative_path: str
    destination_path: Path
    conversion_type: str  # copy|text|docx|pdf

    @property
    def display_name(self) -> str:
        return self.relative_path or self.source_path.name


@dataclass(frozen=True)
class DuplicateSource:
    """Record that two source files share identical content."""

    digest: str
    primary_relative: str
    duplicate_relative: str


@dataclass(frozen=True)
class ConversionPlan:
    """Conversion work along with any duplicate notices discovered."""

    jobs: Sequence[ConversionJob]
    duplicates: Sequence[DuplicateSource]

    @classmethod
    def empty(cls) -> "ConversionPlan":
        return cls(jobs=(), duplicates=())


def build_conversion_jobs(project_manager: ProjectManager) -> ConversionPlan:
    """Return jobs required to bring selected folders into converted_documents."""
    project_dir = project_manager.project_dir
    if not project_dir:
        return ConversionPlan.empty()

    state = project_manager.source_state
    if not state.root:
        return ConversionPlan.empty()

    root_path = _resolve_root(project_dir, state.root)
    if not root_path or not root_path.exists():
        LOGGER.warning("Source root %s is not accessible", state.root)
        return ConversionPlan.empty()

    selected = state.selected_folders or []
    if not selected:
        return ConversionPlan.empty()

    jobs: List[ConversionJob] = []
    duplicates: List[DuplicateSource] = []
    seen_sources: set[Path] = set()
    seen_hashes: dict[str, str] = {}
    for folder in selected:
        folder_path = root_path / folder
        if not folder_path.exists() or not folder_path.is_dir():
            LOGGER.debug("Selected folder %s missing under %s", folder, root_path)
            continue
        for source_file in _iter_files(folder_path):
            if source_file in seen_sources:
                continue
            seen_sources.add(source_file)
            relative = source_file.relative_to(root_path).as_posix()
            conversion_type = _classify_conversion(source_file)
            if conversion_type is None:
                continue
            digest = _hash_source(source_file)
            if digest:
                primary = seen_hashes.get(digest)
                if primary:
                    duplicates.append(
                        DuplicateSource(
                            digest=digest,
                            primary_relative=primary,
                            duplicate_relative=relative,
                        )
                    )
                    continue
                seen_hashes[digest] = relative
            destination = _destination_for(project_dir, relative, conversion_type)
            needs_conversion = _needs_conversion(source_file, destination)
            if not needs_conversion:
                continue
            if digest and _has_matching_checksum(destination, digest):
                continue
            jobs.append(
                ConversionJob(
                    source_path=source_file,
                    relative_path=relative,
                    destination_path=destination,
                    conversion_type=conversion_type,
                )
            )
    return ConversionPlan(jobs=tuple(jobs), duplicates=tuple(duplicates))


def _resolve_root(project_dir: Path, root_spec: str) -> Path | None:
    path = Path(root_spec)
    if not path.is_absolute():
        path = (project_dir / root_spec).resolve()
    return path


def _iter_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def _classify_conversion(source_file: Path) -> str | None:
    suffix = source_file.suffix.lower()
    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    return None


def _destination_for(project_dir: Path, relative: str, conversion_type: str) -> Path:
    converted_root = project_dir / "converted_documents"
    converted_root.mkdir(parents=True, exist_ok=True)
    destination = converted_root / converted_artifact_relative(relative)
    return destination


def _needs_conversion(source: Path, destination: Path) -> bool:
    if not destination.exists():
        return True
    try:
        return source.stat().st_mtime > destination.stat().st_mtime
    except OSError:
        return True


def _hash_source(path: Path) -> str | None:
    """Compute a SHA256 digest for the provided file, returning None on failure."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        LOGGER.warning("Unable to hash %s; skipping duplicate detection", path)
        return None


def _has_matching_checksum(destination: Path, expected_digest: str) -> bool:
    if not destination.exists():
        return False
    resolved = destination.resolve()
    project_dir = None
    parts = resolved.parts
    try:
        idx = parts.index("converted_documents")
    except ValueError:
        idx = -1
    if idx > 0:
        project_dir = Path(*parts[:idx])
    if project_dir is not None:
        try:
            converted_root = (project_dir / "converted_documents").resolve()
            relative = resolved.relative_to(converted_root).as_posix()
            metadata = CitationStore(project_dir).get_document_metadata(relative)
            if metadata and metadata.get("source_checksum") == expected_digest:
                return True
        except Exception:
            pass
    try:
        document = frontmatter.load(destination)
    except Exception:
        return False
    metadata = document.metadata if isinstance(document.metadata, dict) else None
    if not metadata:
        return False
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        return False
    for entry in sources:
        if isinstance(entry, dict) and entry.get("checksum") == expected_digest:
            return True
    return False


def copy_existing_markdown(source: Path, destination: Path) -> None:
    raise RuntimeError("This experimental branch does not support markdown copy-through conversion.")
