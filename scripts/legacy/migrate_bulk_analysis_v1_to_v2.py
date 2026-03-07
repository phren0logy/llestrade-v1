#!/usr/bin/env python3
"""
One-off migration: upgrade a project's bulk-analysis configuration to the new format (v2).

What it does:
- Ensures a `bulk_analysis/` folder exists (renames legacy `summaries/` if present).
- Normalizes legacy layouts:
  - If per-document outputs live directly under `bulk_analysis/`, creates a default group and moves files under it.
  - Ensures per-document outputs are placed under `bulk_analysis/<group>/outputs/` (moves files as needed).
- Upgrades or creates `bulk_analysis/<group>/config.json` to version 2 with sensible defaults.

Notes:
- This script is idempotent and supports a dry-run mode.
- Backups of original config files are written to `backups/<timestamp>-migration/` under the project.

Usage:
  uv run scripts/migrate_bulk_analysis_v1_to_v2.py --project \
      /Users/andy/Library/CloudStorage/OneDrive-TagInc/Active/Jones-v-WA --yes

"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------------
# Helpers
# -------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_project_dir(path: Path) -> Path:
    if path.is_file() and path.name.endswith(".frpd"):
        return path.parent
    candidate = path / "project.frpd"
    if candidate.exists():
        return path
    return path


def _title_from_slug(slug: str) -> str:
    name = slug.replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in name.split()) or "Group"


def _list_group_dirs(bulk_root: Path) -> List[Path]:
    if not bulk_root.exists():
        return []
    return sorted([p for p in bulk_root.iterdir() if p.is_dir()])


def _ensure_outputs_dir(group_dir: Path, *, dry_run: bool) -> Path:
    outputs = group_dir / "outputs"
    if outputs.exists():
        return outputs
    if not dry_run:
        outputs.mkdir(parents=True, exist_ok=True)
    return outputs


def _is_combined_artifact(path: Path) -> bool:
    # Combined outputs (if any) belong under reduce/; avoid moving these
    name = path.name
    return name.startswith("combined_") and name.endswith(".md")


def _iter_legacy_outputs(group_dir: Path) -> List[Path]:
    """Return per-document output files that live directly under the group dir (legacy)."""
    results: List[Path] = []
    for path in group_dir.rglob("*.md"):
        if not path.is_file():
            continue
        # Skip known non-output files
        if path.name == "config.json":
            continue
        if path.suffix != ".md":
            continue
        # Skip new-structure locations
        if (group_dir / "outputs") in path.parents:
            continue
        if (group_dir / "reduce") in path.parents:
            continue
        # Skip combined artifacts (defensive)
        if _is_combined_artifact(path):
            continue
        results.append(path)
    return results


def _normalize_relative_for_selection(group_dir: Path, md_path: Path) -> Optional[str]:
    """Map a per-document output md file to its converted_documents-relative input path.

    Rules:
    - Strip `outputs/` prefix if present.
    - Replace trailing `_analysis.md` with `.md`.
    - Return a project-relative path under converted_documents.
    """
    try:
        rel = md_path.relative_to(group_dir).as_posix()
    except ValueError:
        return None
    # Strip outputs/ prefix
    if rel.startswith("outputs/"):
        rel = rel[len("outputs/") :]
    # Normalize _analysis suffix
    if rel.endswith(".md") and rel[:-3].endswith("_analysis"):
        rel = rel[:-12] + ".md"
    return rel


def _backup_config(project_dir: Path, group_slug: str, config_path: Path, *, dry_run: bool) -> Optional[Path]:
    if not config_path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = project_dir / "backups" / f"{ts}-migration"
    if not dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / f"{group_slug}_config.v1.backup.json"
    if not dry_run:
        shutil.copy2(config_path, backup_path)
    return backup_path


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _infer_provider(info: Dict[str, object]) -> Tuple[str, str]:
    provider = (
        str(info.get("provider_id"))
        or str(info.get("provider"))
        or str(info.get("llm_provider"))
        or "anthropic"
    )
    model = (
        str(info.get("model"))
        or str(info.get("llm_model"))
        or ""
    )
    return provider, model


def _infer_prompts(info: Dict[str, object]) -> Tuple[str, str]:
    system_path = (
        str(info.get("system_prompt_path"))
        or str(info.get("system_prompt"))
        or "resources/prompts/document_analysis_system_prompt.md"
    )
    user_path = (
        str(info.get("user_prompt_path"))
        or str(info.get("user_prompt"))
        or "resources/prompts/document_bulk_analysis_prompt.md"
    )
    return system_path, user_path


@dataclass
class MigrationAction:
    description: str
    src: Optional[Path] = None
    dst: Optional[Path] = None


def migrate_project(project_path: Path, *, dry_run: bool = False) -> List[MigrationAction]:
    actions: List[MigrationAction] = []

    project_dir = _derive_project_dir(project_path)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    # 1) Ensure bulk_analysis/ exists (rename legacy summaries/ if needed)
    bulk_root = project_dir / "bulk_analysis"
    summaries_root = project_dir / "summaries"
    if not bulk_root.exists() and summaries_root.exists():
        actions.append(MigrationAction("Rename summaries/ to bulk_analysis/", summaries_root, bulk_root))
        if not dry_run:
            summaries_root.rename(bulk_root)

    bulk_root.mkdir(parents=True, exist_ok=True) if not dry_run else None

    # Detect whether any immediate children look like v2 groups (contain config.json)
    child_dirs = _list_group_dirs(bulk_root)
    v2_groups = [d for d in child_dirs if (d / "config.json").exists()]

    # 2) Legacy single-group layout: no config.json anywhere under bulk_analysis
    if not v2_groups:
        # Create a default group and move all per-document outputs under bulk_root into it
        default_slug = "legacy"
        default_dir = bulk_root / default_slug
        actions.append(MigrationAction("Create default legacy group", None, default_dir))
        if not dry_run:
            default_dir.mkdir(parents=True, exist_ok=True)

        # Collect all candidate md files recursively under bulk_root (excluding combined and hidden/system files)
        for path in bulk_root.rglob("*.md"):
            if not path.is_file():
                continue
            # Skip any file that would already be under the default group's outputs
            if default_dir in path.parents:
                continue
            # Skip combined artifacts and any stray config-like files
            if _is_combined_artifact(path):
                continue
            if path.name == "config.json":
                continue
            # Compute relative path from bulk_root so we preserve structure under outputs/
            rel = path.relative_to(bulk_root)
            outputs = _ensure_outputs_dir(default_dir, dry_run=dry_run)
            target = outputs / rel
            actions.append(MigrationAction("Move legacy per-document output into default group/outputs/", path, target))
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))

        # Write default group config
        config_path = default_dir / "config.json"
        # Build selection set from the files we just moved
        selection: List[str] = []
        for md in (default_dir / "outputs").rglob("*.md") if (default_dir / "outputs").exists() else []:
            norm = _normalize_relative_for_selection(default_dir, md)
            if norm:
                selection.append(norm)
        selection = sorted({s for s in selection})

        info: Dict[str, object] = {}
        name = _title_from_slug(default_slug)
        provider, model = _infer_provider(info)
        system_prompt, user_prompt = _infer_prompts(info)
        v2_payload: Dict[str, object] = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": "Migrated from legacy layout",
            "files": selection,
            "directories": [],
            "prompt_template": "",
            "provider_id": provider,
            "model": model,
            "system_prompt_path": system_prompt,
            "user_prompt_path": user_prompt,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "slug": default_slug,
            "version": "2",
            "operation": "per_document",
            "combine_converted_files": [],
            "combine_converted_directories": [],
            "combine_map_groups": [],
            "combine_map_files": [],
            "combine_map_directories": [],
            "combine_order": "path",
            "combine_output_template": "combined_{timestamp}.md",
            "use_reasoning": False,
        }
        actions.append(MigrationAction("Write v2 config.json", None, config_path))
        if not dry_run:
            _write_json(config_path, v2_payload)
        return actions

    # 3) v2-like: process each child directory that has a config.json (treat others as non-groups)
    for group_dir in v2_groups:
        slug = group_dir.name
        config_path = group_dir / "config.json"
        info = _read_json(config_path) if config_path.exists() else {}
        version = str(info.get("version", "")) if info else ""

        # 3a) Move legacy top-level md outputs under outputs/
        legacy_outputs = _iter_legacy_outputs(group_dir)
        if legacy_outputs:
            outputs_dir = _ensure_outputs_dir(group_dir, dry_run=dry_run)
            for md in legacy_outputs:
                rel = md.relative_to(group_dir)
                target = outputs_dir / rel
                actions.append(MigrationAction("Move per-document output under outputs/", md, target))
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(md), str(target))

        # Re-scan selection set from current files (prefer outputs/)
        selection: List[str] = []
        outputs_dir = group_dir / "outputs"
        candidates: List[Path] = []
        if outputs_dir.exists():
            candidates.extend([p for p in outputs_dir.rglob("*.md") if p.is_file()])
        else:
            candidates.extend(_iter_legacy_outputs(group_dir))
        for md in candidates:
            norm = _normalize_relative_for_selection(group_dir, md)
            if norm:
                selection.append(norm)
        selection = sorted({s for s in selection})

        # 3b) Backup existing config if migrating
        backup_path = None
        if version != "2":
            backup_path = _backup_config(project_dir, slug, config_path, dry_run=dry_run)
            if backup_path:
                actions.append(MigrationAction("Backup old config", config_path, backup_path))

        # 3c) Build v2 config payload
        # Try to keep name/description if present
        name = str(info.get("name", "")) or _title_from_slug(slug)
        description = str(info.get("description", "")) if info else ""
        provider, model = _infer_provider(info)
        system_prompt, user_prompt = _infer_prompts(info)
        created_at = (
            str(info.get("created_at"))
            if info.get("created_at")
            else (_now_iso())
        )
        updated_at = _now_iso()
        # Prefer existing selection fields; else fall back to inferred selection from outputs
        files = list(info.get("files", [])) if isinstance(info.get("files"), list) else selection
        directories = list(info.get("directories", [])) if isinstance(info.get("directories"), list) else []
        group_id = str(info.get("id") or info.get("group_id") or uuid.uuid4())

        v2_payload: Dict[str, object] = {
            "id": group_id,
            "name": name,
            "description": description,
            "files": files,
            "directories": directories,
            "prompt_template": str(info.get("prompt_template", "")),
            "provider_id": provider,
            "model": model,
            "system_prompt_path": system_prompt,
            "user_prompt_path": user_prompt,
            "created_at": created_at,
            "updated_at": updated_at,
            "slug": slug,
            "version": "2",
            # Operation additions (new in v2)
            "operation": str(info.get("operation", "per_document")) or "per_document",
            "combine_converted_files": list(info.get("combine_converted_files", [])) if isinstance(info.get("combine_converted_files"), list) else [],
            "combine_converted_directories": list(info.get("combine_converted_directories", [])) if isinstance(info.get("combine_converted_directories"), list) else [],
            "combine_map_groups": list(info.get("combine_map_groups", [])) if isinstance(info.get("combine_map_groups"), list) else [],
            "combine_map_files": list(info.get("combine_map_files", [])) if isinstance(info.get("combine_map_files"), list) else [],
            "combine_map_directories": list(info.get("combine_map_directories", [])) if isinstance(info.get("combine_map_directories"), list) else [],
            "combine_order": str(info.get("combine_order", "path")) or "path",
            "combine_output_template": str(info.get("combine_output_template", "combined_{timestamp}.md")) or "combined_{timestamp}.md",
            "use_reasoning": bool(info.get("use_reasoning", False)),
        }

        # 3d) Write config.json (always ensure v2 fields are present)
        actions.append(MigrationAction("Write v2 config.json", None, config_path))
        if not dry_run:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(config_path, v2_payload)

    return actions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate bulk-analysis config to v2 (one-off script)")
    parser.add_argument("--project", required=True, help="Path to project directory or project.frpd file")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without making changes")
    parser.add_argument("--yes", action="store_true", help="Run without interactive confirmation")
    args = parser.parse_args(argv)

    project = Path(args.project).expanduser()
    dry_run = bool(args.dry_run)

    try:
        actions = migrate_project(project, dry_run=dry_run)
        print(f"Planned actions: {len(actions)}")
        for act in actions:
            src = f" {act.src}" if act.src else ""
            dst = f" -> {act.dst}" if act.dst else ""
            print(f"- {act.description}:{src}{dst}")
        if dry_run:
            print("\nDry-run complete. Re-run without --dry-run (and with --yes) to apply.")
            return 0

        if not args.yes:
            reply = input("\nApply these changes? [y/N]: ").strip().lower()
            if reply not in {"y", "yes"}:
                print("Aborted.")
                return 1

        # Apply for real
        # We already performed changes in migrate_project when dry_run=False
        print("\nMigration complete.")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
