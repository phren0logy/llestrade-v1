"""Citation storage, indexing, and verification helpers.

This module provides a per-project SQLite store used to:
- index converted markdown segments into deterministic evidence IDs,
- ingest Azure DI geometry spans from raw JSON sidecars,
- parse/verify inline citation tokens in generated outputs,
- retrieve evidence bundles for future agentic verification and UI deep-linking.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

PAGE_MARKER_RE = re.compile(r"<!---\s*.+?#page=(\d+)\s*--->")
CITATION_TOKEN_RE = re.compile(r"\[CIT:(ev_[a-z0-9]{8,64})\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CitationToken:
    token: str
    ev_id: str
    start_offset: int
    end_offset: int
    line: int
    column: int


@dataclass(frozen=True)
class CitationVerification:
    token: CitationToken
    status: str
    reason: str
    confidence: float
    document_relative_path: str | None
    page_number: int | None


@dataclass(frozen=True)
class EvidenceBundle:
    ev_id: str
    document_relative_path: str
    page_number: int
    text: str
    neighbors: Sequence[Mapping[str, object]]
    geometry: Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class IndexStats:
    relative_path: str
    segments_indexed: int
    geometry_spans_indexed: int


@dataclass(frozen=True)
class CitationRecordStats:
    output_path: str
    total: int
    valid: int
    warning: int
    invalid: int


def parse_citation_tokens(text: str) -> list[CitationToken]:
    """Return parsed citation tokens from ``text`` with source positions."""

    tokens: list[CitationToken] = []
    for match in CITATION_TOKEN_RE.finditer(text):
        start = int(match.start())
        line, column = _line_and_column(text, start)
        tokens.append(
            CitationToken(
                token=match.group(0),
                ev_id=match.group(1),
                start_offset=start,
                end_offset=int(match.end()),
                line=line,
                column=column,
            )
        )
    return tokens


class CitationStore:
    """SQLite-backed citation store for one project."""

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = Path(project_dir).resolve()
        self._db_path = self._project_dir / ".llestrade" / "citations.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def index_converted_document(
        self,
        *,
        relative_path: str,
        markdown_text: str,
        source_checksum: str | None,
        azure_raw_json_path: Path | None,
        pages_pdf: int | None,
        pages_detected: int | None,
    ) -> IndexStats:
        """Index converted markdown segments and Azure geometry spans."""

        indexed_at = _utcnow_iso()
        segments = _segment_markdown(markdown_text, source_checksum or "")
        geometry_spans = self._extract_geometry_spans(azure_raw_json_path)

        with self._connection() as conn:
            conn.execute("BEGIN")
            document_id = self._upsert_document(
                conn,
                relative_path=relative_path,
                source_checksum=source_checksum,
                azure_raw_json_path=azure_raw_json_path,
                pages_pdf=pages_pdf,
                pages_detected=pages_detected,
                indexed_at=indexed_at,
            )
            conn.execute("DELETE FROM segments WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM geometry_spans WHERE document_id = ?", (document_id,))

            if segments:
                conn.executemany(
                    (
                        "INSERT INTO segments ("
                        "ev_id, document_id, page_number, ordinal, text, normalized_text, start_offset, end_offset"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        (
                            item["ev_id"],
                            document_id,
                            int(item["page_number"]),
                            int(item["ordinal"]),
                            item["text"],
                            item["normalized_text"],
                            item.get("start_offset"),
                            item.get("end_offset"),
                        )
                        for item in segments
                    ],
                )

            if geometry_spans:
                conn.executemany(
                    (
                        "INSERT INTO geometry_spans ("
                        "document_id, page_number, text, normalized_text, polygon_json, bbox_json, unit, source_path, chunk_index"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        (
                            document_id,
                            int(item["page_number"]),
                            item.get("text", ""),
                            item.get("normalized_text", ""),
                            item.get("polygon_json"),
                            item.get("bbox_json"),
                            item.get("unit"),
                            item.get("source_path"),
                            item.get("chunk_index"),
                        )
                        for item in geometry_spans
                    ],
                )

            conn.commit()

        return IndexStats(
            relative_path=relative_path,
            segments_indexed=len(segments),
            geometry_spans_indexed=len(geometry_spans),
        )

    def build_evidence_ledger(
        self,
        *,
        relative_path: str,
        page_numbers: Sequence[int] | None = None,
        max_entries: int = 120,
        max_excerpt_chars: int = 180,
    ) -> str:
        """Return a compact ledger for prompt-time citation guidance."""

        page_numbers = list(page_numbers or [])

        with self._connection() as conn:
            doc_row = conn.execute(
                "SELECT id FROM documents WHERE relative_path = ?",
                (relative_path,),
            ).fetchone()
            if doc_row is None:
                return ""

            document_id = int(doc_row[0])
            query = (
                "SELECT ev_id, page_number, text FROM segments "
                "WHERE document_id = ? "
            )
            params: list[object] = [document_id]
            if page_numbers:
                placeholders = ",".join("?" for _ in page_numbers)
                query += f"AND page_number IN ({placeholders}) "
                params.extend(page_numbers)
            query += "ORDER BY page_number ASC, ordinal ASC LIMIT ?"
            params.append(int(max_entries))

            rows = conn.execute(query, tuple(params)).fetchall()

        if not rows:
            return ""

        lines: list[str] = []
        lines.append("## Citation Evidence Ledger")
        lines.append("Use only IDs from this ledger for citation markers: [CIT:ev_<id>].")
        lines.append("If evidence is missing, explicitly state that no supporting evidence was found.")
        lines.append("")

        for ev_id, page, text in rows:
            excerpt = _squash_whitespace(str(text))
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[: max_excerpt_chars - 1].rstrip() + "…"
            lines.append(f"- [{ev_id}|p{page}] {excerpt}")

        return "\n".join(lines).strip() + "\n"

    def build_evidence_ledger_for_documents(
        self,
        *,
        relative_paths: Sequence[str],
        max_per_document: int = 40,
        max_total: int = 200,
        max_excerpt_chars: int = 180,
    ) -> str:
        """Return ledger entries across multiple documents."""

        entries: list[tuple[str, str, int, str]] = []
        with self._connection() as conn:
            for rel in relative_paths:
                doc_row = conn.execute(
                    "SELECT id FROM documents WHERE relative_path = ?",
                    (rel,),
                ).fetchone()
                if doc_row is None:
                    continue
                document_id = int(doc_row[0])
                rows = conn.execute(
                    (
                        "SELECT ev_id, page_number, text FROM segments "
                        "WHERE document_id = ? ORDER BY page_number ASC, ordinal ASC LIMIT ?"
                    ),
                    (document_id, int(max_per_document)),
                ).fetchall()
                entries.extend((rel, str(ev_id), int(page), str(text)) for ev_id, page, text in rows)

        if not entries:
            return ""

        entries = entries[:max_total]
        lines: list[str] = []
        lines.append("## Citation Evidence Ledger")
        lines.append("Use only IDs from this ledger for citation markers: [CIT:ev_<id>].")
        lines.append("If evidence is missing, explicitly state that no supporting evidence was found.")
        lines.append("")

        for rel, ev_id, page, text in entries:
            excerpt = _squash_whitespace(text)
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[: max_excerpt_chars - 1].rstrip() + "…"
            lines.append(f"- [{ev_id}|{rel}|p{page}] {excerpt}")

        return "\n".join(lines).strip() + "\n"

    def list_evidence_ids_for_documents(
        self,
        *,
        relative_paths: Sequence[str],
        limit: int = 400,
    ) -> list[str]:
        """Return evidence IDs across multiple documents."""

        out: list[str] = []
        with self._connection() as conn:
            for rel in relative_paths:
                doc_row = conn.execute(
                    "SELECT id FROM documents WHERE relative_path = ?",
                    (rel,),
                ).fetchone()
                if doc_row is None:
                    continue
                rows = conn.execute(
                    "SELECT ev_id FROM segments WHERE document_id = ? ORDER BY page_number, ordinal LIMIT ?",
                    (int(doc_row[0]), int(limit)),
                ).fetchall()
                out.extend(str(row[0]) for row in rows)
                if len(out) >= limit:
                    break
        return out[:limit]

    def verify_citations(self, text: str) -> list[CitationVerification]:
        """Parse and verify citation markers in generated output text."""

        tokens = parse_citation_tokens(text)
        if not tokens:
            return []

        ev_ids = sorted({token.ev_id for token in tokens})
        segment_lookup: dict[str, sqlite3.Row] = {}

        with self._connection() as conn:
            placeholders = ",".join("?" for _ in ev_ids)
            rows = conn.execute(
                (
                    "SELECT s.ev_id, s.page_number, s.normalized_text, d.relative_path, d.id AS document_id "
                    "FROM segments s JOIN documents d ON d.id = s.document_id "
                    f"WHERE s.ev_id IN ({placeholders})"
                ),
                tuple(ev_ids),
            ).fetchall()
            for row in rows:
                segment_lookup[str(row["ev_id"])] = row

            geometry_count: dict[tuple[int, int], int] = {}
            if rows:
                pairs = {(int(row["document_id"]), int(row["page_number"])) for row in rows}
                for document_id, page_number in pairs:
                    count_row = conn.execute(
                        "SELECT COUNT(*) AS count FROM geometry_spans WHERE document_id = ? AND page_number = ?",
                        (document_id, page_number),
                    ).fetchone()
                    geometry_count[(document_id, page_number)] = int(count_row["count"] if count_row else 0)

        verifications: list[CitationVerification] = []
        for token in tokens:
            segment = segment_lookup.get(token.ev_id)
            if segment is None:
                verifications.append(
                    CitationVerification(
                        token=token,
                        status="invalid",
                        reason="unknown evidence id",
                        confidence=0.0,
                        document_relative_path=None,
                        page_number=None,
                    )
                )
                continue

            context = text[max(0, token.start_offset - 240): min(len(text), token.end_offset + 240)]
            context = context.replace(token.token, "")
            overlap = _token_overlap_ratio(_normalize_text(context), str(segment["normalized_text"]))
            doc_id = int(segment["document_id"])
            page_number = int(segment["page_number"])
            geometry_hits = geometry_count.get((doc_id, page_number), 0)

            if overlap < 0.06:
                status = "warning"
                reason = "low textual overlap with cited evidence"
            else:
                status = "valid"
                reason = "verified"

            if geometry_hits <= 0:
                if status == "valid":
                    status = "warning"
                reason = "verified without geometry mapping"

            verifications.append(
                CitationVerification(
                    token=token,
                    status=status,
                    reason=reason,
                    confidence=round(overlap, 4),
                    document_relative_path=str(segment["relative_path"]),
                    page_number=page_number,
                )
            )

        return verifications

    def record_output_citations(
        self,
        *,
        output_path: Path,
        output_text: str,
        generator: str,
        prompt_hash: str | None,
        created_at: datetime | None = None,
    ) -> CitationRecordStats:
        """Persist citation verification results for an output artifact."""

        timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        output_path_resolved = output_path.resolve().as_posix()
        checksum = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        verifications = self.verify_citations(output_text)

        with self._connection() as conn:
            conn.execute("BEGIN")
            output_row = conn.execute(
                "SELECT id FROM outputs WHERE output_path = ?",
                (output_path_resolved,),
            ).fetchone()
            if output_row is None:
                cur = conn.execute(
                    (
                        "INSERT INTO outputs (output_path, checksum, generator, prompt_hash, created_at) "
                        "VALUES (?, ?, ?, ?, ?)"
                    ),
                    (output_path_resolved, checksum, generator, prompt_hash, timestamp),
                )
                output_id = int(cur.lastrowid)
            else:
                output_id = int(output_row[0])
                conn.execute(
                    "UPDATE outputs SET checksum = ?, generator = ?, prompt_hash = ?, created_at = ? WHERE id = ?",
                    (checksum, generator, prompt_hash, timestamp, output_id),
                )
                conn.execute("DELETE FROM output_citations WHERE output_id = ?", (output_id,))

            if verifications:
                conn.executemany(
                    (
                        "INSERT INTO output_citations ("
                        "output_id, token, ev_id, status, reason, confidence, start_offset, line, column, "
                        "document_relative_path, page_number, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        (
                            output_id,
                            item.token.token,
                            item.token.ev_id,
                            item.status,
                            item.reason,
                            float(item.confidence),
                            int(item.token.start_offset),
                            int(item.token.line),
                            int(item.token.column),
                            item.document_relative_path,
                            item.page_number,
                            timestamp,
                        )
                        for item in verifications
                    ],
                )

            conn.commit()

        valid = sum(1 for item in verifications if item.status == "valid")
        warning = sum(1 for item in verifications if item.status == "warning")
        invalid = sum(1 for item in verifications if item.status == "invalid")

        return CitationRecordStats(
            output_path=output_path_resolved,
            total=len(verifications),
            valid=valid,
            warning=warning,
            invalid=invalid,
        )

    def get_evidence_bundle(self, ev_id: str, *, window: int = 2) -> EvidenceBundle | None:
        """Return cited segment, surrounding segments, and geometry candidates."""

        with self._connection() as conn:
            row = conn.execute(
                (
                    "SELECT s.ev_id, s.document_id, s.page_number, s.ordinal, s.text, d.relative_path "
                    "FROM segments s JOIN documents d ON d.id = s.document_id "
                    "WHERE s.ev_id = ?"
                ),
                (ev_id,),
            ).fetchone()
            if row is None:
                return None

            document_id = int(row["document_id"])
            page = int(row["page_number"])
            ordinal = int(row["ordinal"])
            target_norm = _normalize_text(str(row["text"]))

            neighbors_rows = conn.execute(
                (
                    "SELECT ev_id, page_number, ordinal, text FROM segments "
                    "WHERE document_id = ? AND page_number BETWEEN ? AND ? "
                    "ORDER BY page_number ASC, ordinal ASC"
                ),
                (document_id, max(page - 1, 1), page + 1),
            ).fetchall()

            ordered = [
                {
                    "ev_id": str(item["ev_id"]),
                    "page_number": int(item["page_number"]),
                    "ordinal": int(item["ordinal"]),
                    "text": str(item["text"]),
                }
                for item in neighbors_rows
            ]
            idx = next(
                (i for i, item in enumerate(ordered) if item["ev_id"] == ev_id),
                0,
            )
            low = max(0, idx - max(window, 0))
            high = min(len(ordered), idx + max(window, 0) + 1)
            neighbors = ordered[low:high]

            geometry_rows = conn.execute(
                (
                    "SELECT text, polygon_json, bbox_json, unit, source_path, chunk_index, normalized_text "
                    "FROM geometry_spans WHERE document_id = ? AND page_number = ?"
                ),
                (document_id, page),
            ).fetchall()

        scored_geometry: list[tuple[float, dict[str, object]]] = []
        for geo in geometry_rows:
            geo_norm = _normalize_text(str(geo["normalized_text"] or ""))
            score = _token_overlap_ratio(target_norm, geo_norm)
            scored_geometry.append(
                (
                    score,
                    {
                        "text": str(geo["text"] or ""),
                        "polygon": _json_or_none(geo["polygon_json"]),
                        "bbox": _json_or_none(geo["bbox_json"]),
                        "unit": str(geo["unit"] or ""),
                        "source_path": str(geo["source_path"] or ""),
                        "chunk_index": geo["chunk_index"],
                        "score": round(score, 4),
                    },
                )
            )

        scored_geometry.sort(key=lambda item: item[0], reverse=True)
        geometry = [item[1] for item in scored_geometry[:10]]

        return EvidenceBundle(
            ev_id=str(row["ev_id"]),
            document_relative_path=str(row["relative_path"]),
            page_number=page,
            text=str(row["text"]),
            neighbors=neighbors,
            geometry=geometry,
        )

    def _extract_geometry_spans(self, azure_raw_json_path: Path | None) -> list[dict[str, object]]:
        if azure_raw_json_path is None or not azure_raw_json_path.exists():
            return []

        try:
            payload = json.loads(azure_raw_json_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        spans: list[dict[str, object]] = []
        seen: set[tuple[int, str, str]] = set()

        for chunk_index, source_path, result in _iter_analyze_results(payload):
            pages = result.get("pages") if isinstance(result, dict) else None
            if not isinstance(pages, list):
                continue
            for pidx, page in enumerate(pages):
                if not isinstance(page, dict):
                    continue
                page_number = _to_int(page.get("page_number")) or _to_int(page.get("pageNumber")) or (pidx + 1)
                unit = str(page.get("unit") or "")
                lines = page.get("lines")
                if isinstance(lines, list) and lines:
                    for lidx, line in enumerate(lines):
                        if not isinstance(line, dict):
                            continue
                        text = str(line.get("content") or "").strip()
                        if not text:
                            continue
                        polygon = line.get("polygon")
                        bbox = _polygon_to_bbox(polygon)
                        norm = _normalize_text(text)
                        key = (int(page_number), norm, json.dumps(bbox, sort_keys=True) if bbox else "")
                        if key in seen:
                            continue
                        seen.add(key)
                        spans.append(
                            {
                                "page_number": int(page_number),
                                "text": text,
                                "normalized_text": norm,
                                "polygon_json": json.dumps(polygon) if polygon is not None else None,
                                "bbox_json": json.dumps(bbox) if bbox is not None else None,
                                "unit": unit,
                                "source_path": f"{source_path}.pages[{pidx}].lines[{lidx}]",
                                "chunk_index": chunk_index,
                            }
                        )
                    continue

                paragraphs = page.get("paragraphs")
                if not isinstance(paragraphs, list):
                    continue
                for ridx, para in enumerate(paragraphs):
                    if not isinstance(para, dict):
                        continue
                    text = str(para.get("content") or "").strip()
                    if not text:
                        continue
                    polygon = para.get("polygon")
                    bbox = _polygon_to_bbox(polygon)
                    norm = _normalize_text(text)
                    key = (int(page_number), norm, json.dumps(bbox, sort_keys=True) if bbox else "")
                    if key in seen:
                        continue
                    seen.add(key)
                    spans.append(
                        {
                            "page_number": int(page_number),
                            "text": text,
                            "normalized_text": norm,
                            "polygon_json": json.dumps(polygon) if polygon is not None else None,
                            "bbox_json": json.dumps(bbox) if bbox is not None else None,
                            "unit": unit,
                            "source_path": f"{source_path}.pages[{pidx}].paragraphs[{ridx}]",
                            "chunk_index": chunk_index,
                        }
                    )

        return spans

    def _upsert_document(
        self,
        conn: sqlite3.Connection,
        *,
        relative_path: str,
        source_checksum: str | None,
        azure_raw_json_path: Path | None,
        pages_pdf: int | None,
        pages_detected: int | None,
        indexed_at: str,
    ) -> int:
        azure_rel = ""
        if azure_raw_json_path:
            try:
                azure_rel = azure_raw_json_path.resolve().relative_to(self._project_dir).as_posix()
            except Exception:
                azure_rel = azure_raw_json_path.resolve().as_posix()

        row = conn.execute(
            "SELECT id FROM documents WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()
        if row is None:
            cur = conn.execute(
                (
                    "INSERT INTO documents ("
                    "relative_path, source_checksum, azure_raw_json_path, pages_pdf, pages_detected, indexed_at"
                    ") VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (relative_path, source_checksum, azure_rel, pages_pdf, pages_detected, indexed_at),
            )
            return int(cur.lastrowid)

        document_id = int(row[0])
        conn.execute(
            (
                "UPDATE documents SET source_checksum = ?, azure_raw_json_path = ?, pages_pdf = ?, "
                "pages_detected = ?, indexed_at = ? WHERE id = ?"
            ),
            (source_checksum, azure_rel, pages_pdf, pages_detected, indexed_at, document_id),
        )
        return document_id

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version >= _SCHEMA_VERSION:
                return

            if version < 1:
                self._apply_schema_v1(conn)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.commit()

    def _apply_schema_v1(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relative_path TEXT NOT NULL UNIQUE,
                source_checksum TEXT,
                azure_raw_json_path TEXT,
                pages_pdf INTEGER,
                pages_detected INTEGER,
                indexed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS segments (
                ev_id TEXT PRIMARY KEY,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                start_offset INTEGER,
                end_offset INTEGER,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, page_number, ordinal)
            );

            CREATE INDEX IF NOT EXISTS idx_segments_doc_page
            ON segments(document_id, page_number, ordinal);

            CREATE TABLE IF NOT EXISTS geometry_spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT,
                normalized_text TEXT,
                polygon_json TEXT,
                bbox_json TEXT,
                unit TEXT,
                source_path TEXT,
                chunk_index INTEGER,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_geometry_doc_page
            ON geometry_spans(document_id, page_number);

            CREATE TABLE IF NOT EXISTS outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_path TEXT NOT NULL UNIQUE,
                checksum TEXT,
                generator TEXT,
                prompt_hash TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS output_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                ev_id TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                confidence REAL,
                start_offset INTEGER,
                line INTEGER,
                column INTEGER,
                document_relative_path TEXT,
                page_number INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(output_id) REFERENCES outputs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_output_citations_output
            ON output_citations(output_id);

            CREATE INDEX IF NOT EXISTS idx_output_citations_ev
            ON output_citations(ev_id);
            """
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path.as_posix(), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn


def _segment_markdown(markdown_text: str, source_checksum: str) -> list[dict[str, object]]:
    lines = markdown_text.splitlines(keepends=True)
    char_pos = 0
    page_number = 1
    page_ordinals: dict[int, int] = {}

    segments: list[dict[str, object]] = []
    buffer_lines: list[str] = []
    buffer_start: int | None = None

    def flush_buffer(end_pos: int) -> None:
        nonlocal buffer_lines, buffer_start, segments, page_number
        if not buffer_lines:
            return
        block = "".join(buffer_lines).strip()
        if not block:
            buffer_lines = []
            buffer_start = None
            return

        pieces = _split_long_segment(block)
        local_start = buffer_start if buffer_start is not None else end_pos

        for piece in pieces:
            page_ordinals[page_number] = page_ordinals.get(page_number, 0) + 1
            ordinal = page_ordinals[page_number]
            normalized = _normalize_text(piece)
            ev_id = _ev_id(source_checksum, page_number, ordinal, normalized)
            end_offset = local_start + len(piece)
            segments.append(
                {
                    "ev_id": ev_id,
                    "page_number": page_number,
                    "ordinal": ordinal,
                    "text": piece,
                    "normalized_text": normalized,
                    "start_offset": local_start,
                    "end_offset": end_offset,
                }
            )
            local_start = end_offset + 1

        buffer_lines = []
        buffer_start = None

    for line in lines:
        marker = PAGE_MARKER_RE.search(line)
        if marker:
            flush_buffer(char_pos)
            page_number = _to_int(marker.group(1)) or page_number
            char_pos += len(line)
            continue

        if not line.strip():
            flush_buffer(char_pos)
            char_pos += len(line)
            continue

        if buffer_start is None:
            buffer_start = char_pos
        buffer_lines.append(line)
        char_pos += len(line)

    flush_buffer(char_pos)
    return segments


def _split_long_segment(text: str, *, max_chars: int = 1200, target_chars: int = 700) -> list[str]:
    cleaned = _squash_whitespace(text)
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    if not sentences:
        return [cleaned[i:i + max_chars].strip() for i in range(0, len(cleaned), max_chars)]

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= target_chars:
            current = candidate
            continue
        pieces.append(current)
        current = sentence

    if current:
        pieces.append(current)

    bounded: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            bounded.append(piece)
            continue
        bounded.extend(piece[i:i + max_chars].strip() for i in range(0, len(piece), max_chars))

    return [item for item in bounded if item]


def _iter_analyze_results(payload: object) -> Iterator[tuple[int | None, str, dict]]:
    if not isinstance(payload, dict):
        return

    mode = str(payload.get("mode") or "").lower()
    if mode == "chunked" and isinstance(payload.get("chunks"), list):
        for idx, chunk in enumerate(payload.get("chunks") or []):
            if not isinstance(chunk, dict):
                continue
            result = chunk.get("analyze_result")
            if isinstance(result, dict):
                yield idx, f"chunks[{idx}].analyze_result", result
        return

    yield None, "analyze_result", payload


def _token_overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aset = set(a.split())
    bset = set(b.split())
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(len(aset), 1)


def _line_and_column(text: str, offset: int) -> tuple[int, int]:
    prefix = text[:offset]
    line = prefix.count("\n") + 1
    last_newline = prefix.rfind("\n")
    column = offset + 1 if last_newline < 0 else offset - last_newline
    return line, column


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _squash_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _ev_id(source_checksum: str, page_number: int, ordinal: int, normalized_text: str) -> str:
    payload = f"{source_checksum}|{page_number}|{ordinal}|{normalized_text}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"ev_{digest}"


def _polygon_to_bbox(polygon: object) -> dict[str, float] | None:
    if not isinstance(polygon, list):
        return None

    numbers: list[float] = []
    for value in polygon:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            return None

    if len(numbers) < 4:
        return None

    xs = numbers[0::2]
    ys = numbers[1::2]
    if not xs or not ys:
        return None

    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def _to_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _json_or_none(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return value


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CitationRecordStats",
    "CitationStore",
    "CitationToken",
    "CitationVerification",
    "EvidenceBundle",
    "IndexStats",
    "parse_citation_tokens",
]
