#!/usr/bin/env python3
"""Compatibility wrapper for scripts/legacy/migrate_llestrade_from_legacy.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "legacy" / "migrate_llestrade_from_legacy.py"
    runpy.run_path(str(target), run_name="__main__")
