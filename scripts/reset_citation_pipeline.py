"""Reset citation-dependent derived artifacts for one project.

This script makes the citation pipeline reset explicit after citation-format or
storage changes. It removes only derived artifacts so the project can be
reconverted and rerun from source documents.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DERIVED_RELATIVE_PATHS: tuple[str, ...] = (
    ".llestrade/citations.db",
    "converted_documents",
    "highlights",
    "bulk_analysis",
    "reports",
)


def collect_reset_paths(project_dir: Path) -> list[Path]:
    """Return existing derived paths that should be removed for a clean rebuild."""

    root = project_dir.expanduser().resolve()
    paths: list[Path] = []
    for relative in DERIVED_RELATIVE_PATHS:
        candidate = root / relative
        if candidate.exists():
            paths.append(candidate)
    return paths


def reset_citation_pipeline(project_dir: Path, *, dry_run: bool = False) -> list[Path]:
    """Remove citation-dependent derived artifacts for ``project_dir``."""

    paths = collect_reset_paths(project_dir)
    if dry_run:
        return paths

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove derived citation-dependent project artifacts so the project "
            "can be reconverted and rerun from source documents."
        )
    )
    parser.add_argument("project_dir", type=Path, help="Project directory to reset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List paths that would be removed without deleting anything",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    project_dir = args.project_dir.expanduser().resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise SystemExit(f"Project directory not found: {project_dir}")

    paths = reset_citation_pipeline(project_dir, dry_run=args.dry_run)
    if not paths:
        print(f"No derived citation artifacts found under {project_dir}")
        return 0

    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {len(paths)} path(s):")
    for path in paths:
        print(f"- {path}")

    if args.dry_run:
        print("\nNext steps after a real reset:")
    else:
        print("\nNext steps:")
    print("1. Reconvert documents from the Documents tab.")
    print("2. Regenerate highlights if needed.")
    print("3. Rerun bulk analysis groups.")
    print("4. Regenerate report drafts/refinements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
