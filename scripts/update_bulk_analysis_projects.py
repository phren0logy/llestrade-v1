#!/usr/bin/env python3
"""Compatibility wrapper for scripts/legacy/update_bulk_analysis_projects.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "legacy" / "update_bulk_analysis_projects.py"
    runpy.run_path(str(target), run_name="__main__")
