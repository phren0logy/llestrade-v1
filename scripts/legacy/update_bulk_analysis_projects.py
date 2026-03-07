#!/usr/bin/env python3
"""
One-off utility to upgrade legacy project assets from the old "summary" naming.

For each supplied project directory the script:
  * Updates bulk-analysis group config files so any reference to
    document_summary_prompt*.md now points at document_bulk_analysis_prompt*.md.
  * Renames prompt files named document_summary_prompt.md within the project tree.

Usage:
    uv run scripts/update_bulk_analysis_projects.py /path/to/project [more...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple


OLD_PROMPT_STEM = "document_summary_prompt"
NEW_PROMPT_STEM = "document_bulk_analysis_prompt"


def _replace_prompt_strings(payload: Any) -> Tuple[Any, bool]:
    """Return a copy of ``payload`` with legacy prompt identifiers swapped."""
    changed = False

    if isinstance(payload, dict):
        updated = {}
        for key, value in payload.items():
            new_value, child_changed = _replace_prompt_strings(value)
            if child_changed:
                changed = True
            updated[key] = new_value
        return updated, changed

    if isinstance(payload, list):
        updated_list = []
        for item in payload:
            new_item, child_changed = _replace_prompt_strings(item)
            if child_changed:
                changed = True
            updated_list.append(new_item)
        return updated_list, changed

    if isinstance(payload, str):
        new_value = payload.replace(OLD_PROMPT_STEM, NEW_PROMPT_STEM)
        if new_value != payload:
            changed = True
        return new_value, changed

    return payload, changed


def _derive_project_dir(path: Path) -> Path:
    """Resolve the project directory from a supplied path."""
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix == ".frpd":
        return path.parent
    return path


def _update_config_file(path: Path) -> bool:
    """Rewrite ``config.json`` if it contains legacy prompt references."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Skipping unreadable config {path}: {exc}", file=sys.stderr)
        return False

    updated, changed = _replace_prompt_strings(data)
    if not changed:
        return False

    path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"✅  Updated prompt references in {path}")
    return True


def _rename_prompt_file(path: Path) -> bool:
    target = path.with_name(path.name.replace(OLD_PROMPT_STEM, NEW_PROMPT_STEM))
    if target.exists():
        if target.resolve() == path.resolve():
            return False
        # Already renamed elsewhere; remove legacy file for safety.
        path.unlink()
        print(f"🗑️  Removed duplicate legacy prompt file {path}")
        return True

    path.rename(target)
    print(f"🔁 Renamed {path} -> {target}")
    return True


def process_project(project_path: Path) -> None:
    project_dir = _derive_project_dir(project_path)
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_path}", file=sys.stderr)
        return

    print(f"\n=== Processing {project_dir} ===")

    config_paths = sorted(project_dir.glob("bulk_analysis/**/config.json"))
    if not config_paths:
        print("ℹ️  No bulk_analysis config files found.")
    config_changes = sum(_update_config_file(path) for path in config_paths)

    prompt_paths = sorted(project_dir.rglob(f"{OLD_PROMPT_STEM}.md"))
    prompt_changes = sum(_rename_prompt_file(path) for path in prompt_paths)

    if config_changes == 0 and prompt_changes == 0:
        print("✅  No updates required.")
    else:
        print(f"Done – configs updated: {config_changes}, prompt files renamed: {prompt_changes}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upgrade legacy bulk analysis prompt references.")
    parser.add_argument(
        "projects",
        nargs="+",
        help="Project directories or project.frpd files to update.",
    )
    args = parser.parse_args(argv)

    for entry in args.projects:
        process_project(Path(entry))

    return 0


if __name__ == "__main__":
    sys.exit(main())
