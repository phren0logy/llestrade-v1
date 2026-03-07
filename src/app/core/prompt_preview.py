"""Utilities for generating bulk-analysis prompt previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import frontmatter

from src.app.core.azure_artifacts import is_azure_raw_artifact
from src.app.core.bulk_analysis_runner import load_prompts
from src.app.core.bulk_analysis_runner import _metadata_context  # type: ignore[attr-defined]
from src.app.core.project_manager import ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.prompt_placeholders import get_prompt_spec, format_prompt
from src.app.core.placeholders.system import SourceFileContext
from src.app.core.bulk_paths import (
    iter_map_outputs,
    iter_map_outputs_under,
    normalize_map_relative,
    resolve_map_output_path,
)
from src.app.core.bulk_prompt_context import build_bulk_placeholders


class PromptPreviewError(RuntimeError):
    """Raised when a prompt preview cannot be generated."""


@dataclass(slots=True)
class PromptPreview:
    system_template: str
    user_template: str
    system_rendered: str
    user_rendered: str
    values: dict[str, str]
    required: set[str]
    optional: set[str]


def generate_prompt_preview(
    project_dir: Path,
    group: BulkAnalysisGroup,
    *,
    metadata: Optional[ProjectMetadata] = None,
    max_content_lines: int = 10,
    placeholder_values: Optional[Mapping[str, str]] = None,
) -> PromptPreview:
    """Return a preview of the system/user prompts for the supplied group."""

    project_dir = Path(project_dir)
    if not project_dir.exists():
        raise PromptPreviewError("Project directory is not available.")

    bundle = load_prompts(project_dir, group, metadata)
    metadata_context = _metadata_context(metadata)

    operation = getattr(group, "operation", "per_document") or "per_document"
    source_context: SourceFileContext | None = None
    reduce_contexts: list[SourceFileContext] = []

    if operation == "combined":
        combined_inputs = _resolve_combined_inputs(project_dir, group, limit=5)
        preview_path = combined_inputs[0] if combined_inputs else None
        reduce_contexts = _build_reduce_contexts(project_dir, combined_inputs)
        source_context = None
        document_name = group.name or "Combined"
    else:
        preview_path, document_name = _resolve_first_per_document_input(project_dir, group)

    if preview_path is None:
        raise PromptPreviewError("No converted files are available for this group.")

    try:
        body, doc_metadata = _read_document(preview_path)
    except FileNotFoundError as exc:  # pragma: no cover - filesystem race
        raise PromptPreviewError(f"Preview source missing: {preview_path}") from exc

    truncated_content = _truncate_markdown(body, max_content_lines)

    if operation != "combined":
        source_context = _extract_primary_source(project_dir, preview_path, doc_metadata, document_name)

    base_placeholder_values: dict[str, str] = dict(placeholder_values or {})

    effective_project_name = (
        (placeholder_values or {}).get("project_name")
        or (metadata.case_name if metadata and metadata.case_name else None)
        or (group.name if getattr(group, "name", None) else None)
        or project_dir.name
    )

    if operation == "combined":
        # Rebuild reduce contexts if we haven't already (combined path might be None earlier)
        if "reduce_contexts" not in locals():
            reduce_contexts = _build_reduce_contexts(project_dir, [])
        source_context = None  # Combined runs don't set per-document source placeholders

    system_placeholders = build_bulk_placeholders(
        base_placeholders=base_placeholder_values,
        project_name=effective_project_name,
        reduce_sources=reduce_contexts,
    )
    system_context = dict(metadata_context)
    system_context.update(system_placeholders)
    system_rendered = format_prompt(bundle.system_template, system_context)

    document_placeholders = build_bulk_placeholders(
        base_placeholders=base_placeholder_values,
        project_name=effective_project_name,
        source=source_context,
        reduce_sources=reduce_contexts,
    )
    user_context = dict(metadata_context)
    user_context.update(
        {
            "document_name": document_name,
            "document_content": truncated_content,
        }
    )
    user_context.update(document_placeholders)
    user_rendered = format_prompt(bundle.user_template, user_context)

    required: set[str] = set()
    optional: set[str] = set()
    system_spec = get_prompt_spec("document_analysis_system_prompt")
    if system_spec:
        required.update(system_spec.required)
        optional.update(system_spec.optional)
    user_spec = get_prompt_spec("document_bulk_analysis_prompt")
    if user_spec:
        required.update(user_spec.required)
        optional.update(user_spec.optional)

    values = dict(system_context)
    values.update(user_context)

    return PromptPreview(
        system_template=bundle.system_template,
        user_template=bundle.user_template,
        system_rendered=system_rendered,
        user_rendered=user_rendered,
        values=values,
        required=required,
        optional=optional,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_first_per_document_input(
    project_dir: Path,
    group: BulkAnalysisGroup,
) -> tuple[Optional[Path], str]:
    converted_root = project_dir / "converted_documents"
    if not converted_root.exists():
        return None, group.files[0] if group.files else group.name

    file_map = {}
    for path in converted_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            if is_azure_raw_artifact(path):
                continue
            relative = path.relative_to(converted_root).as_posix().strip("/")
            file_map[relative] = path

    if not file_map:
        return None, group.files[0] if group.files else group.name

    ordered: list[str] = []
    for entry in group.files or []:
        candidate = entry.strip("/")
        if candidate in file_map:
            ordered.append(candidate)

    for directory in group.directories or []:
        directory = directory.strip("/")
        if not directory:
            continue
        for relative in sorted(file_map):
            if relative == directory or relative.startswith(directory + "/"):
                ordered.append(relative)

    if not ordered:
        ordered = sorted(file_map)

    if not ordered:
        return None, group.files[0] if group.files else group.name

    first_relative = ordered[0]
    return file_map[first_relative], first_relative


def _truncate_markdown(content: str, max_lines: int) -> str:
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    truncated = lines[:max_lines]
    truncated.append("…")
    return "\n".join(truncated)


def _resolve_combined_inputs(project_dir: Path, group: BulkAnalysisGroup, *, limit: Optional[int] = None) -> list[Path]:
    conv_root = project_dir / "converted_documents"
    items: list[Path] = []

    def _add(path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        if is_azure_raw_artifact(path):
            return
        if path in items:
            return
        items.append(path)

    # Explicit converted files
    for rel in group.combine_converted_files or []:
        rel = rel.strip("/")
        if not rel:
            continue
        _add(conv_root / rel)
        if limit and len(items) >= limit:
            return _apply_order(items, group)

    # Converted directories
    for rel_dir in group.combine_converted_directories or []:
        rel_dir = rel_dir.strip("/")
        if not rel_dir:
            continue
        base = conv_root / rel_dir
        if not base.exists():
            continue
        for candidate in sorted(p for p in base.rglob("*.md") if p.is_file()):
            if is_azure_raw_artifact(candidate):
                continue
            _add(candidate)
            if limit and len(items) >= limit:
                return _apply_order(items, group)

    # Entire map groups
    for slug in group.combine_map_groups or []:
        slug = slug.strip()
        if not slug:
            continue
        for path, _ in iter_map_outputs(project_dir, slug):
            _add(path)
            if limit and len(items) >= limit:
                return _apply_order(items, group)

    # Map directories
    for rel_dir in group.combine_map_directories or []:
        rel_dir = rel_dir.strip("/")
        if not rel_dir:
            continue
        parts = rel_dir.split("/", 1)
        if len(parts) != 2:
            continue
        slug, remainder = parts
        slug = slug.strip()
        if not slug:
            continue
        normalized = normalize_map_relative(remainder)
        for path, _ in iter_map_outputs_under(project_dir, slug, normalized):
            _add(path)
            if limit and len(items) >= limit:
                return _apply_order(items, group)

    # Explicit map files
    for rel in group.combine_map_files or []:
        rel = rel.strip("/")
        if not rel:
            continue
        parts = rel.split("/", 1)
        if len(parts) != 2:
            continue
        slug, remainder = parts
        slug = slug.strip()
        if not slug:
            continue
        normalized = normalize_map_relative(remainder)
        if not normalized:
            continue
        _add(resolve_map_output_path(project_dir, slug, normalized))
        if limit and len(items) >= limit:
            return _apply_order(items, group)

    # Fallback to first per-document converted file
    path, _ = _resolve_first_per_document_input(project_dir, group)
    if path:
        _add(path)

    return _apply_order(items, group)


def _apply_order(paths: Sequence[Path], group: BulkAnalysisGroup) -> list[Path]:
    order = (group.combine_order or "path").lower()
    if order == "mtime":
        return sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return sorted(paths, key=lambda p: p.as_posix())


def _read_document(path: Path) -> tuple[str, dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    try:
        post = frontmatter.loads(raw)
    except Exception:
        return raw, {}
    content = post.content or ""
    metadata = dict(post.metadata or {})
    return content, metadata


def _extract_primary_source(
    project_dir: Path,
    path: Path,
    metadata: Mapping[str, object],
    fallback_relative: str,
) -> SourceFileContext | None:
    contexts = _extract_source_contexts(project_dir, path, metadata, fallback_relative)
    return contexts[0] if contexts else None


def _build_reduce_contexts(project_dir: Path, paths: Iterable[Path]) -> list[SourceFileContext]:
    contexts: list[SourceFileContext] = []
    seen: set[str] = set()
    for path in paths:
        fallback_rel = _project_relative(project_dir, path)
        _, metadata = _read_document(path)
        for ctx in _extract_source_contexts(project_dir, path, metadata, fallback_rel):
            if ctx.relative_path in seen:
                continue
            seen.add(ctx.relative_path)
            contexts.append(ctx)
    return contexts


def _extract_source_contexts(
    project_dir: Path,
    path: Path,
    metadata: Mapping[str, object],
    fallback_relative: str,
) -> list[SourceFileContext]:
    sources = metadata.get("sources")
    if not sources and isinstance(metadata.get("metadata"), dict):
        nested = metadata["metadata"]
        if isinstance(nested, Mapping):
            sources = nested.get("sources")

    contexts: list[SourceFileContext] = []
    if isinstance(sources, Iterable):
        for entry in sources:
            if not isinstance(entry, Mapping):
                continue
            context = _resolve_source_context(project_dir, entry, fallback_relative)
            if context:
                contexts.append(context)

    if contexts:
        return contexts
    absolute = path.resolve()
    relative = fallback_relative or absolute.name
    return [SourceFileContext(absolute_path=absolute, relative_path=relative)]


def _resolve_source_context(
    project_dir: Path,
    entry: Mapping[str, object],
    fallback_relative: str,
) -> SourceFileContext | None:
    rel_raw = str(entry.get("relative", "") or "").strip()
    path_raw = str(entry.get("path", "") or "").strip()

    absolute: Path
    if path_raw:
        candidate = Path(path_raw).expanduser()
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        absolute = candidate
    else:
        absolute = (project_dir / fallback_relative).resolve()

    if not rel_raw:
        try:
            rel_raw = absolute.relative_to(project_dir).as_posix()
        except Exception:
            rel_raw = absolute.name

    return SourceFileContext(absolute_path=absolute, relative_path=rel_raw)


def _project_relative(project_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir).as_posix()
    except Exception:
        return path.name


__all__ = [
    "PromptPreview",
    "PromptPreviewError",
    "generate_prompt_preview",
]
