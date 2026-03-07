"""Export project citation data from SQLite to JSON for debugging."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path.as_posix())
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str) -> list[dict]:
    return [dict(row) for row in conn.execute(sql).fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export citations.db to JSON")
    parser.add_argument("project_dir", type=Path, help="Project directory containing .llestrade/citations.db")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON path (default: <project_dir>/citations_export.json)",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.expanduser().resolve()
    db_path = project_dir / ".llestrade" / "citations.db"
    if not db_path.exists():
        raise SystemExit(f"Citation database not found: {db_path}")

    output_path = args.output.expanduser().resolve() if args.output else project_dir / "citations_export.json"

    with _connect(db_path) as conn:
        payload = {
            "db_path": db_path.as_posix(),
            "documents": _rows(conn, "SELECT * FROM documents ORDER BY relative_path"),
            "segments": _rows(conn, "SELECT * FROM segments ORDER BY document_id, page_number, ordinal"),
            "geometry_spans": _rows(conn, "SELECT * FROM geometry_spans ORDER BY document_id, page_number, id"),
            "outputs": _rows(conn, "SELECT * FROM outputs ORDER BY created_at"),
            "output_citations": _rows(conn, "SELECT * FROM output_citations ORDER BY output_id, id"),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported citation data to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
