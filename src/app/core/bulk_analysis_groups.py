"""Bulk analysis group persistence helpers."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.app.core.llm_operation_settings import LLMReasoningSettings, normalize_context_window_override

LOGGER = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"
BULK_ANALYSIS_FOLDER = "bulk_analysis"
BULK_ANALYSIS_GROUP_VERSION = "2"


class InvalidBulkAnalysisGroupFormat(Exception):
    """Raised when a bulk analysis group's config format is invalid or unsupported."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    slug = slug.strip("-")
    return slug or "group"


def _ensure_unique_slug(base_dir: Path, candidate: str, existing_slug: Optional[str] = None) -> str:
    if existing_slug:
        return existing_slug
    slug = candidate
    index = 2
    while (base_dir / slug).exists():
        slug = f"{candidate}-{index}"
        index += 1
    return slug


def _groups_root(project_dir: Path) -> Path:
    root = project_dir / BULK_ANALYSIS_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class BulkAnalysisGroup:
    group_id: str
    name: str
    description: str = ""
    files: List[str] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    prompt_template: str = ""
    provider_id: str = ""
    model: str = ""
    # Optional explicit context window (tokens) for custom models
    model_context_window: Optional[int] = None
    system_prompt_path: str = ""
    user_prompt_path: str = ""
    placeholder_requirements: Dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    slug: Optional[str] = None
    version: str = BULK_ANALYSIS_GROUP_VERSION

    # Operation type and combined-operation fields
    # operation: "per_document" applies prompts to each doc individually (map)
    # operation: "combined" merges selected inputs then applies a single prompt once (reduce)
    operation: str = "per_document"

    # Combined operation inputs (project-relative paths)
    combine_converted_files: List[str] = field(default_factory=list)
    combine_converted_directories: List[str] = field(default_factory=list)
    combine_map_groups: List[str] = field(default_factory=list)  # slugs
    combine_map_files: List[str] = field(default_factory=list)   # group_slug/rel/path.md
    combine_map_directories: List[str] = field(default_factory=list)

    # Combined options
    combine_order: str = "path"  # or "mtime"
    combine_output_template: str = "combined_{timestamp}.md"
    use_reasoning: bool = False
    reasoning: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        description: str = "",
        files: Optional[Iterable[str]] = None,
        directories: Optional[Iterable[str]] = None,
        prompt_template: str = "",
        provider_id: str = "",
        model: str = "",
        system_prompt_path: str = "",
        user_prompt_path: str = "",
    ) -> "BulkAnalysisGroup":
        return cls(
            group_id=str(uuid.uuid4()),
            name=name,
            description=description,
            files=list(files or []),
            directories=list(directories or []),
            prompt_template=prompt_template,
            provider_id=provider_id,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
        )

    def touch(self) -> None:
        self.updated_at = _utcnow()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.group_id,
            "name": self.name,
            "description": self.description,
            "files": self.files,
            "directories": self.directories,
            "prompt_template": self.prompt_template,
            "provider_id": self.provider_id,
            "model": self.model,
            "model_context_window": self.model_context_window,
            "system_prompt_path": self.system_prompt_path,
            "user_prompt_path": self.user_prompt_path,
            "placeholder_requirements": self.placeholder_requirements,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "slug": self.slug,
            "version": self.version,
            "operation": self.operation,
            "combine_converted_files": self.combine_converted_files,
            "combine_converted_directories": self.combine_converted_directories,
            "combine_map_groups": self.combine_map_groups,
            "combine_map_files": self.combine_map_files,
            "combine_map_directories": self.combine_map_directories,
            "combine_order": self.combine_order,
            "combine_output_template": self.combine_output_template,
            "use_reasoning": self.use_reasoning,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BulkAnalysisGroup":
        version = str(payload.get("version", BULK_ANALYSIS_GROUP_VERSION))
        if version != BULK_ANALYSIS_GROUP_VERSION:
            raise InvalidBulkAnalysisGroupFormat(
                f"Unsupported bulk analysis group version {version} (expected {BULK_ANALYSIS_GROUP_VERSION})"
            )

        return cls(
            group_id=str(payload.get("id")),
            name=str(payload.get("name", "Untitled Group")),
            description=str(payload.get("description", "")),
            files=list(payload.get("files", [])),
            directories=list(payload.get("directories", [])),
            prompt_template=str(payload.get("prompt_template", "")),
            provider_id=str(payload.get("provider_id", "")),
            model=str(payload.get("model", "")),
            model_context_window=normalize_context_window_override(
                provider_id=str(payload.get("provider_id", "")),
                model_id=str(payload.get("model", "")),
                context_window=(
                    int(payload.get("model_context_window"))
                    if str(payload.get("model_context_window", "")).strip().isdigit()
                    else None
                ),
            ),
            system_prompt_path=str(payload.get("system_prompt_path", "")),
            user_prompt_path=str(payload.get("user_prompt_path", "")),
            placeholder_requirements=dict(payload.get("placeholder_requirements", {})),
            created_at=_parse_datetime(payload.get("created_at")),
            updated_at=_parse_datetime(payload.get("updated_at")),
            slug=payload.get("slug"),
            version=version,
            operation=str(payload.get("operation", "per_document")),
            combine_converted_files=list(payload.get("combine_converted_files", [])),
            combine_converted_directories=list(payload.get("combine_converted_directories", [])),
            combine_map_groups=list(payload.get("combine_map_groups", [])),
            combine_map_files=list(payload.get("combine_map_files", [])),
            combine_map_directories=list(payload.get("combine_map_directories", [])),
            combine_order=str(payload.get("combine_order", "path")),
            combine_output_template=str(payload.get("combine_output_template", "combined_{timestamp}.md")),
            use_reasoning=bool(payload.get("use_reasoning", False)),
            reasoning=LLMReasoningSettings.from_value(
                payload.get("reasoning"),
                legacy_use_reasoning=bool(payload.get("use_reasoning", False)),
            ).to_dict(),
        )

    # Convenience properties
    @property
    def folder_name(self) -> str:
        return self.slug or _slugify(self.name)


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return _utcnow()


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------

def load_bulk_analysis_groups(project_dir: Path) -> List[BulkAnalysisGroup]:
    root = _groups_root(project_dir)
    groups: List[BulkAnalysisGroup] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        config_path = child / CONFIG_FILENAME
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text())
            group = BulkAnalysisGroup.from_dict(data)
            if not group.slug:
                group.slug = child.name
            groups.append(group)
        except InvalidBulkAnalysisGroupFormat as exc:
            LOGGER.warning(
                "Invalid or unsupported bulk analysis group format at %s: %s",
                config_path,
                exc,
            )
            # Skip invalid groups; UI can provide messaging elsewhere if needed.
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load bulk analysis group from %s: %s", config_path, exc)
    return groups


def save_bulk_analysis_group(project_dir: Path, group: BulkAnalysisGroup) -> BulkAnalysisGroup:
    root = _groups_root(project_dir)
    slug = _ensure_unique_slug(root, _slugify(group.name), group.slug)
    group.slug = slug
    group.touch()

    group_dir = root / slug
    group_dir.mkdir(parents=True, exist_ok=True)

    config_path = group_dir / CONFIG_FILENAME
    config_path.write_text(json.dumps(group.to_dict(), indent=2))
    return group


def delete_bulk_analysis_group(project_dir: Path, group: BulkAnalysisGroup) -> None:
    if not group.slug:
        return
    group_dir = _groups_root(project_dir) / group.slug
    if group_dir.exists():
        shutil.rmtree(group_dir, ignore_errors=True)


__all__ = [
    "BulkAnalysisGroup",
    "load_bulk_analysis_groups",
    "save_bulk_analysis_group",
    "delete_bulk_analysis_group",
    "InvalidBulkAnalysisGroupFormat",
]
