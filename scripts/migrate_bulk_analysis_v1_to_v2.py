#!/usr/bin/env python3
"""Compatibility wrapper for scripts/legacy/migrate_bulk_analysis_v1_to_v2.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "legacy" / "migrate_bulk_analysis_v1_to_v2.py"
    runpy.run_path(str(target), run_name="__main__")
