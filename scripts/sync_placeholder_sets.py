#!/usr/bin/env python3
"""
Sync bundled placeholder sets into the user placeholder store.

Copies markdown files from src/app/resources/placeholder_sets into
~/Documents/llestrade/placeholder_sets/bundled (or platform equivalent)
without touching custom placeholder sets.

Usage:
  uv run scripts/sync_placeholder_sets.py         # Copy new files, skip changed
  uv run scripts/sync_placeholder_sets.py --force # Overwrite changed bundled files

  # Alternatively (module mode):
  uv run -m scripts.sync_placeholder_sets            # Preferred when using Python module paths
  uv run -m scripts.sync_placeholder_sets --force
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

from src.config.placeholder_store import (
    get_placeholder_bundled_dir,
    get_repo_placeholder_dir,
    sync_bundled_placeholder_sets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync bundled placeholder sets into user store")
    parser.add_argument("--force", action="store_true", help="Overwrite changed bundled files")
    args = parser.parse_args()

    repo_dir = get_repo_placeholder_dir()
    bundled_dir = get_placeholder_bundled_dir()
    print(f"Repo placeholder sets: {repo_dir}")
    print(f"User (bundled):       {bundled_dir}")

    result = sync_bundled_placeholder_sets(force=args.force)
    print("\nSync results:")
    for key in ("copied", "updated", "skipped", "same", "deleted"):
        items = result.get(key, [])
        print(f"- {key}: {len(items)}")
        for name in items:
            print(f"   • {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

