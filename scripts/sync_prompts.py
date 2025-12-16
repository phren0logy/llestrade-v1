#!/usr/bin/env python3
"""
Sync bundled prompt templates into the user prompt store.

This copies prompts from src/app/resources/prompts into
~/Documents/llestrade/prompts/bundled (or platform equivalent) without
touching custom prompts.

Usage:
  uv run scripts/sync_prompts.py            # Copy new files, skip changed
  uv run scripts/sync_prompts.py --force    # Overwrite changed bundled files
  
  # Alternatively (module mode):
  uv run -m scripts.sync_prompts            # Preferred when using Python module paths
  uv run -m scripts.sync_prompts --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path when invoked as a script
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.prompt_store import (
    get_bundled_dir,
    get_repo_prompts_dir,
    sync_bundled_prompts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync bundled prompts into user store")
    parser.add_argument("--force", action="store_true", help="Overwrite changed bundled files")
    args = parser.parse_args()

    repo_dir = get_repo_prompts_dir()
    bundled_dir = get_bundled_dir()
    print(f"Repo prompts:    {repo_dir}")
    print(f"User (bundled):  {bundled_dir}")

    result = sync_bundled_prompts(force=args.force)
    print("\nSync results:")
    for key in ("copied", "updated", "skipped", "same", "deleted"):
        items = result.get(key, [])
        print(f"- {key}: {len(items)}")
        for name in items:
            print(f"   • {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
