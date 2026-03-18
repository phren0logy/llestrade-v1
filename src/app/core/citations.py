"""Citation storage, indexing, and verification helpers.

This module provides a per-project SQLite store used to:
- index converted markdown segments into deterministic evidence IDs,
- ingest Azure DI geometry spans from raw JSON sidecars,
- verify local inline citation labels in generated outputs,
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

from src.app.core.doctags import build_geometry_spans_from_doctags, parse_doctags_blocks

PAGE_MARKER_RE = re.compile(r"<!---\s*.+?#page=(\d+)\s*--->")
LOCAL_CITATION_TOKEN_RE = re.compile(r"\[(C\d{1,5})\]")
INLINE_CITATION_TOKEN_RE = re.compile(r"\[CIT:(ev_[a-z0-9]{8,64})\]|\[(C\d{1,5})\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class EvidenceBundle:
    ev_id: str
    document_relative_path: str
    source_relative_path: str | None
    source_absolute_path: str | None
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


@dataclass(frozen=True)
class CitationLedgerEntry:
    citation_label: str
    ev_id: str
    document_relative_path: str
    page_number: int
    text: str


@dataclass(frozen=True)
class OutputCitationMention:
    token: str
    citation_label: str | None
    ev_id: str | None
    status: str
    reason: str
    confidence: float
    start_offset: int
    line: int
    column: int
    document_relative_path: str | None
    source_relative_path: str | None
    source_absolute_path: str | None
    page_number: int | None

def strip_citation_tokens(text: str) -> str:
    """Remove inline citation markers while preserving surrounding prose."""

    stripped = INLINE_CITATION_TOKEN_RE.sub("", text)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    return stripped


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

    def get_document_metadata(self, relative_path: str) -> dict[str, object] | None:
        with self._connection() as conn:
            row = conn.execute(
                (
                    "SELECT relative_path, source_checksum, source_relative_path, source_absolute_path, pages_pdf, "
                    "pages_detected, content_format, pipeline_mode, vlm_preset, standard_profile "
                    "FROM documents WHERE relative_path = ?"
                ),
                (relative_path,),
            ).fetchone()
        if row is None:
            return None
        return {
            "relative_path": str(row["relative_path"]),
            "source_checksum": str(row["source_checksum"] or "") or None,
            "source_relative_path": str(row["source_relative_path"] or "") or None,
            "source_absolute_path": str(row["source_absolute_path"] or "") or None,
            "pages_pdf": _to_int(row["pages_pdf"]),
            "pages_detected": _to_int(row["pages_detected"]),
            "content_format": str(row["content_format"] or "") or None,
            "pipeline_mode": str(row["pipeline_mode"] or "") or None,
            "vlm_preset": str(row["vlm_preset"] or "") or None,
            "standard_profile": str(row["standard_profile"] or "") or None,
        }

    def index_converted_document(
        self,
        *,
        relative_path: str,
        markdown_text: str,
        source_checksum: str | None,
        azure_raw_json_path: Path | None,
        pages_pdf: int | None,
        pages_detected: int | None,
        source_relative_path: str | None = None,
        source_absolute_path: str | None = None,
        content_format: str = "markdown",
        pipeline_mode: str | None = None,
        vlm_preset: str | None = None,
        standard_profile: str | None = None,
    ) -> IndexStats:
        """Index converted markdown segments and Azure geometry spans."""

        indexed_at = _utcnow_iso()
        if content_format == "doctags":
            segments = _segment_doctags(markdown_text, source_checksum or "")
            geometry_spans = build_geometry_spans_from_doctags(markdown_text)
        else:
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
                source_relative_path=source_relative_path,
                source_absolute_path=source_absolute_path,
                indexed_at=indexed_at,
                content_format=content_format,
                pipeline_mode=pipeline_mode,
                vlm_preset=vlm_preset,
                standard_profile=standard_profile,
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
                        "document_id, page_number, text, normalized_text, polygon_json, bbox_json, "
                        "normalized_bbox_json, page_width, page_height, unit, source_path, chunk_index"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        (
                            document_id,
                            int(item["page_number"]),
                            item.get("text", ""),
                            item.get("normalized_text", ""),
                            item.get("polygon_json"),
                            item.get("bbox_json"),
                            item.get("normalized_bbox_json"),
                            item.get("page_width"),
                            item.get("page_height"),
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

    def index_doctags_document(
        self,
        *,
        relative_path: str,
        doctags_text: str,
        source_checksum: str | None,
        pages_pdf: int | None,
        pages_detected: int | None,
        source_relative_path: str | None = None,
        source_absolute_path: str | None = None,
        pipeline_mode: str | None = "vlm_primary",
        vlm_preset: str | None = "granite_docling",
        standard_profile: str | None = None,
    ) -> IndexStats:
        return self.index_converted_document(
            relative_path=relative_path,
            markdown_text=doctags_text,
            source_checksum=source_checksum,
            azure_raw_json_path=None,
            pages_pdf=pages_pdf,
            pages_detected=pages_detected,
            source_relative_path=source_relative_path,
            source_absolute_path=source_absolute_path,
            content_format="doctags",
            pipeline_mode=pipeline_mode,
            vlm_preset=vlm_preset,
            standard_profile=standard_profile,
        )

    def list_local_citation_entries(
        self,
        *,
        relative_path: str,
        page_numbers: Sequence[int] | None = None,
        max_entries: int = 120,
    ) -> list[CitationLedgerEntry]:
        """Return deterministic local citation labels for one converted document."""

        page_numbers = list(page_numbers or [])
        with self._connection() as conn:
            doc_row = conn.execute(
                "SELECT id FROM documents WHERE relative_path = ?",
                (relative_path,),
            ).fetchone()
            if doc_row is None:
                return []

            query = (
                "SELECT ev_id, page_number, text FROM segments "
                "WHERE document_id = ? "
            )
            params: list[object] = [int(doc_row[0])]
            if page_numbers:
                placeholders = ",".join("?" for _ in page_numbers)
                query += f"AND page_number IN ({placeholders}) "
                params.extend(int(page) for page in page_numbers)
            query += "ORDER BY page_number ASC, ordinal ASC LIMIT ?"
            params.append(int(max_entries))
            rows = conn.execute(query, tuple(params)).fetchall()

        entries: list[CitationLedgerEntry] = []
        for index, row in enumerate(rows, start=1):
            entries.append(
                CitationLedgerEntry(
                    citation_label=f"C{index}",
                    ev_id=str(row["ev_id"]),
                    document_relative_path=relative_path,
                    page_number=int(row["page_number"]),
                    text=str(row["text"]),
                )
            )
        return entries

    def list_local_citation_entries_for_documents(
        self,
        *,
        relative_paths: Sequence[str],
        max_per_document: int = 40,
        max_total: int = 220,
    ) -> list[CitationLedgerEntry]:
        """Return deterministic local labels for segments across multiple documents."""

        raw_entries: list[tuple[str, int, int, str, str]] = []
        with self._connection() as conn:
            for rel in relative_paths:
                doc_row = conn.execute(
                    "SELECT id FROM documents WHERE relative_path = ?",
                    (rel,),
                ).fetchone()
                if doc_row is None:
                    continue
                rows = conn.execute(
                    (
                        "SELECT ev_id, page_number, ordinal, text FROM segments "
                        "WHERE document_id = ? ORDER BY page_number ASC, ordinal ASC LIMIT ?"
                    ),
                    (int(doc_row[0]), int(max_per_document)),
                ).fetchall()
                raw_entries.extend(
                    (rel, int(row["page_number"]), int(row["ordinal"]), str(row["ev_id"]), str(row["text"]))
                    for row in rows
                )

        raw_entries.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        out: list[CitationLedgerEntry] = []
        for index, (rel, page_number, _ordinal, ev_id, text) in enumerate(raw_entries[:max_total], start=1):
            out.append(
                CitationLedgerEntry(
                    citation_label=f"C{index}",
                    ev_id=ev_id,
                    document_relative_path=rel,
                    page_number=page_number,
                    text=text,
                )
            )
        return out

    def list_local_citation_entries_for_evidence_ids(
        self,
        *,
        ev_ids: Sequence[str],
        max_total: int = 220,
    ) -> list[CitationLedgerEntry]:
        """Return deterministic local labels for an explicit evidence-id set."""

        ids = sorted({str(ev_id) for ev_id in ev_ids if ev_id})
        if not ids:
            return []

        with self._connection() as conn:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                (
                    "SELECT s.ev_id, s.page_number, s.ordinal, s.text, d.relative_path "
                    "FROM segments s JOIN documents d ON d.id = s.document_id "
                    f"WHERE s.ev_id IN ({placeholders}) "
                    "ORDER BY d.relative_path ASC, s.page_number ASC, s.ordinal ASC"
                ),
                tuple(ids),
            ).fetchall()

        out: list[CitationLedgerEntry] = []
        for index, row in enumerate(rows[:max_total], start=1):
            out.append(
                CitationLedgerEntry(
                    citation_label=f"C{index}",
                    ev_id=str(row["ev_id"]),
                    document_relative_path=str(row["relative_path"]),
                    page_number=int(row["page_number"]),
                    text=str(row["text"]),
                )
            )
        return out

    def build_local_citation_appendix(
        self,
        *,
        relative_path: str,
        page_numbers: Sequence[int] | None = None,
        max_entries: int = 120,
        max_excerpt_chars: int = 180,
    ) -> tuple[str, dict[str, str]]:
        """Return a bulk-map citation appendix and label-to-evidence mapping."""

        entries = self.list_local_citation_entries(
            relative_path=relative_path,
            page_numbers=page_numbers,
            max_entries=max_entries,
        )
        appendix = self.render_local_citation_appendix(entries, max_excerpt_chars=max_excerpt_chars)
        mapping = {entry.citation_label: entry.ev_id for entry in entries}
        return appendix, mapping

    def build_local_citation_appendix_for_documents(
        self,
        *,
        relative_paths: Sequence[str],
        max_per_document: int = 40,
        max_total: int = 220,
        max_excerpt_chars: int = 180,
    ) -> tuple[str, dict[str, str]]:
        entries = self.list_local_citation_entries_for_documents(
            relative_paths=relative_paths,
            max_per_document=max_per_document,
            max_total=max_total,
        )
        appendix = self.render_local_citation_appendix(entries, max_excerpt_chars=max_excerpt_chars)
        mapping = {entry.citation_label: entry.ev_id for entry in entries}
        return appendix, mapping

    def build_local_citation_appendix_for_evidence_ids(
        self,
        *,
        ev_ids: Sequence[str],
        max_total: int = 220,
        max_excerpt_chars: int = 180,
    ) -> tuple[str, dict[str, str]]:
        entries = self.list_local_citation_entries_for_evidence_ids(
            ev_ids=ev_ids,
            max_total=max_total,
        )
        appendix = self.render_local_citation_appendix(entries, max_excerpt_chars=max_excerpt_chars)
        mapping = {entry.citation_label: entry.ev_id for entry in entries}
        return appendix, mapping

    def render_local_citation_appendix(
        self,
        entries: Sequence[CitationLedgerEntry],
        *,
        max_excerpt_chars: int = 180,
    ) -> str:
        """Render a previously assigned local citation set without renumbering it."""

        if not entries:
            return ""

        lines: list[str] = []
        lines.append("Use only citation labels from this appendix for factual claims.")
        lines.append("Exact inline citation format: [C1], [C2], [C3], etc.")
        lines.append("If supporting evidence is missing, explicitly say the evidence is unavailable.")
        lines.append("")
        lines.append("Available citations:")
        for entry in entries:
            excerpt = _squash_whitespace(entry.text)
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[: max_excerpt_chars - 1].rstrip() + "…"
            lines.append(
                f"- [{entry.citation_label}] {entry.document_relative_path} p. {entry.page_number}: {excerpt}"
            )
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

    def verify_local_citations(
        self,
        text: str,
        *,
        label_mapping: Mapping[str, str],
    ) -> list[dict[str, object]]:
        """Parse and verify local output-scoped citation labels like ``[C1]``."""

        matches = list(LOCAL_CITATION_TOKEN_RE.finditer(text))
        if not matches:
            return []

        ev_ids = [label_mapping[label] for label in {match.group(1) for match in matches} if label in label_mapping]
        segment_lookup, geometry_count = self._segment_lookup_for_ids(ev_ids)

        results: list[dict[str, object]] = []
        for match in matches:
            label = match.group(1)
            start = int(match.start())
            line, column = _line_and_column(text, start)
            ev_id = label_mapping.get(label)
            segment = segment_lookup.get(ev_id) if ev_id else None
            if segment is None:
                results.append(
                    {
                        "token": match.group(0),
                        "citation_label": label,
                        "ev_id": ev_id,
                        "status": "invalid",
                        "reason": "unknown citation label",
                        "confidence": 0.0,
                        "start_offset": start,
                        "line": line,
                        "column": column,
                        "document_relative_path": None,
                        "source_relative_path": None,
                        "source_absolute_path": None,
                        "page_number": None,
                    }
                )
                continue

            context = text[max(0, start - 240): min(len(text), int(match.end()) + 240)]
            context = context.replace(match.group(0), "")
            overlap = _token_overlap_ratio(_normalize_text(context), str(segment["normalized_text"]))
            doc_id = int(segment["document_id"])
            page_number = int(segment["page_number"])
            geometry_hits = geometry_count.get((doc_id, page_number), 0)
            status = "valid"
            reason = "verified"
            if overlap < 0.06:
                status = "warning"
                reason = "low textual overlap with cited evidence"
            if geometry_hits <= 0:
                if status == "valid":
                    status = "warning"
                reason = "verified without geometry mapping"

            results.append(
                {
                    "token": match.group(0),
                    "citation_label": label,
                    "ev_id": ev_id,
                    "status": status,
                    "reason": reason,
                    "confidence": round(overlap, 4),
                    "start_offset": start,
                    "line": line,
                    "column": column,
                    "document_relative_path": str(segment["relative_path"]),
                    "source_relative_path": str(segment["source_relative_path"] or "") or None,
                    "source_absolute_path": str(segment["source_absolute_path"] or "") or None,
                    "page_number": page_number,
                }
            )
        return results

    def record_output_citations(
        self,
        *,
        output_path: Path,
        output_text: str,
        generator: str,
        prompt_hash: str | None,
        label_mapping: Mapping[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> CitationRecordStats:
        """Persist citation verification results for an output artifact."""

        timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        output_path_resolved = output_path.resolve().as_posix()
        checksum = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        local_verifications = self.verify_local_citations(output_text, label_mapping=label_mapping or {})
        total_verifications = len(local_verifications)

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

            if local_verifications:
                conn.executemany(
                    (
                        "INSERT INTO output_citations ("
                        "output_id, token, citation_label, ev_id, status, reason, confidence, start_offset, line, column, "
                        "document_relative_path, source_relative_path, source_absolute_path, page_number, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        (
                            output_id,
                            str(item["token"]),
                            item["citation_label"],
                            item["ev_id"] or "",
                            item["status"],
                            item["reason"],
                            float(item["confidence"]),
                            int(item["start_offset"]),
                            int(item["line"]),
                            int(item["column"]),
                            item["document_relative_path"],
                            item["source_relative_path"],
                            item["source_absolute_path"],
                            item["page_number"],
                            timestamp,
                        )
                        for item in local_verifications
                    ],
                )

            conn.commit()

        valid = sum(1 for item in local_verifications if item["status"] == "valid")
        warning = sum(1 for item in local_verifications if item["status"] == "warning")
        invalid = sum(1 for item in local_verifications if item["status"] == "invalid")

        return CitationRecordStats(
            output_path=output_path_resolved,
            total=total_verifications,
            valid=valid,
            warning=warning,
            invalid=invalid,
        )

    def list_output_citation_mentions(self, output_path: Path) -> list[OutputCitationMention]:
        """Return stored citation mentions for one output artifact."""

        output_path_resolved = output_path.resolve().as_posix()
        with self._connection() as conn:
            rows = conn.execute(
                (
                    "SELECT token, citation_label, ev_id, status, reason, confidence, start_offset, line, column, "
                    "document_relative_path, source_relative_path, source_absolute_path, page_number "
                    "FROM output_citations oc "
                    "JOIN outputs o ON o.id = oc.output_id "
                    "WHERE o.output_path = ? "
                    "ORDER BY oc.start_offset ASC, oc.id ASC"
                ),
                (output_path_resolved,),
            ).fetchall()

        mentions: list[OutputCitationMention] = []
        for row in rows:
            mentions.append(
                OutputCitationMention(
                    token=str(row["token"]),
                    citation_label=str(row["citation_label"] or "") or None,
                    ev_id=str(row["ev_id"] or "") or None,
                    status=str(row["status"]),
                    reason=str(row["reason"] or ""),
                    confidence=float(row["confidence"] or 0.0),
                    start_offset=int(row["start_offset"] or 0),
                    line=int(row["line"] or 0),
                    column=int(row["column"] or 0),
                    document_relative_path=str(row["document_relative_path"] or "") or None,
                    source_relative_path=str(row["source_relative_path"] or "") or None,
                    source_absolute_path=str(row["source_absolute_path"] or "") or None,
                    page_number=_to_int(row["page_number"]),
                )
            )
        return mentions

    def _segment_lookup_for_ids(
        self,
        ev_ids: Iterable[str],
    ) -> tuple[dict[str, sqlite3.Row], dict[tuple[int, int], int]]:
        ids = sorted({str(ev_id) for ev_id in ev_ids if ev_id})
        if not ids:
            return {}, {}

        segment_lookup: dict[str, sqlite3.Row] = {}
        geometry_count: dict[tuple[int, int], int] = {}
        with self._connection() as conn:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                (
                    "SELECT s.ev_id, s.page_number, s.normalized_text, d.relative_path, d.id AS document_id, "
                    "d.source_relative_path, d.source_absolute_path "
                    "FROM segments s JOIN documents d ON d.id = s.document_id "
                    f"WHERE s.ev_id IN ({placeholders})"
                ),
                tuple(ids),
            ).fetchall()
            for row in rows:
                segment_lookup[str(row["ev_id"])] = row

            for document_id, page_number in {(int(row["document_id"]), int(row["page_number"])) for row in rows}:
                count_row = conn.execute(
                    "SELECT COUNT(*) AS count FROM geometry_spans WHERE document_id = ? AND page_number = ?",
                    (document_id, page_number),
                ).fetchone()
                geometry_count[(document_id, page_number)] = int(count_row["count"] if count_row else 0)
        return segment_lookup, geometry_count

    def get_evidence_bundle(self, ev_id: str, *, window: int = 2) -> EvidenceBundle | None:
        """Return cited segment, surrounding segments, and geometry candidates."""

        with self._connection() as conn:
            row = conn.execute(
                (
                    "SELECT s.ev_id, s.document_id, s.page_number, s.ordinal, s.text, d.relative_path, "
                    "d.source_relative_path, d.source_absolute_path "
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
                    "SELECT text, polygon_json, bbox_json, normalized_bbox_json, page_width, page_height, "
                    "unit, source_path, chunk_index, normalized_text "
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
                        "normalized_bbox": _json_or_none(geo["normalized_bbox_json"]),
                        "page_width": geo["page_width"],
                        "page_height": geo["page_height"],
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
            source_relative_path=str(row["source_relative_path"] or "") or None,
            source_absolute_path=str(row["source_absolute_path"] or "") or None,
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
                page_width = _to_float(page.get("width"))
                page_height = _to_float(page.get("height"))
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
                        normalized_bbox = _normalize_bbox(bbox, page_width=page_width, page_height=page_height)
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
                                "normalized_bbox_json": json.dumps(normalized_bbox) if normalized_bbox is not None else None,
                                "page_width": page_width,
                                "page_height": page_height,
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
                    normalized_bbox = _normalize_bbox(bbox, page_width=page_width, page_height=page_height)
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
                            "normalized_bbox_json": json.dumps(normalized_bbox) if normalized_bbox is not None else None,
                            "page_width": page_width,
                            "page_height": page_height,
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
        source_relative_path: str | None,
        source_absolute_path: str | None,
        indexed_at: str,
        content_format: str | None,
        pipeline_mode: str | None,
        vlm_preset: str | None,
        standard_profile: str | None,
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
                    "relative_path, source_checksum, azure_raw_json_path, pages_pdf, pages_detected, "
                    "source_relative_path, source_absolute_path, content_format, pipeline_mode, "
                    "vlm_preset, standard_profile, indexed_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    relative_path,
                    source_checksum,
                    azure_rel,
                    pages_pdf,
                    pages_detected,
                    source_relative_path,
                    source_absolute_path,
                    content_format,
                    pipeline_mode,
                    vlm_preset,
                    standard_profile,
                    indexed_at,
                ),
            )
            return int(cur.lastrowid)

        document_id = int(row[0])
        conn.execute(
            (
                "UPDATE documents SET source_checksum = ?, azure_raw_json_path = ?, pages_pdf = ?, "
                "pages_detected = ?, source_relative_path = ?, source_absolute_path = ?, content_format = ?, "
                "pipeline_mode = ?, vlm_preset = ?, standard_profile = ?, indexed_at = ? WHERE id = ?"
            ),
            (
                source_checksum,
                azure_rel,
                pages_pdf,
                pages_detected,
                source_relative_path,
                source_absolute_path,
                content_format,
                pipeline_mode,
                vlm_preset,
                standard_profile,
                indexed_at,
                document_id,
            ),
        )
        return document_id

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version >= _SCHEMA_VERSION:
                return

            if version < 1:
                self._apply_schema_v1(conn)
                version = 1

            if version < 2:
                self._apply_schema_v2(conn)
                version = 2

            if version < 3:
                self._apply_schema_v3(conn)
                version = 3

            conn.execute(f"PRAGMA user_version = {version}")
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
                source_relative_path TEXT,
                source_absolute_path TEXT,
                content_format TEXT,
                pipeline_mode TEXT,
                vlm_preset TEXT,
                standard_profile TEXT,
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
                normalized_bbox_json TEXT,
                page_width REAL,
                page_height REAL,
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
                citation_label TEXT,
                ev_id TEXT,
                status TEXT NOT NULL,
                reason TEXT,
                confidence REAL,
                start_offset INTEGER,
                line INTEGER,
                column INTEGER,
                document_relative_path TEXT,
                source_relative_path TEXT,
                source_absolute_path TEXT,
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

    def _apply_schema_v2(self, conn: sqlite3.Connection) -> None:
        statements = [
            "ALTER TABLE documents ADD COLUMN source_relative_path TEXT",
            "ALTER TABLE documents ADD COLUMN source_absolute_path TEXT",
            "ALTER TABLE geometry_spans ADD COLUMN normalized_bbox_json TEXT",
            "ALTER TABLE geometry_spans ADD COLUMN page_width REAL",
            "ALTER TABLE geometry_spans ADD COLUMN page_height REAL",
            "ALTER TABLE output_citations ADD COLUMN citation_label TEXT",
            "ALTER TABLE output_citations ADD COLUMN source_relative_path TEXT",
            "ALTER TABLE output_citations ADD COLUMN source_absolute_path TEXT",
        ]
        for statement in statements:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        conn.execute("CREATE INDEX IF NOT EXISTS idx_output_citations_label ON output_citations(citation_label)")

    def _apply_schema_v3(self, conn: sqlite3.Connection) -> None:
        statements = [
            "ALTER TABLE documents ADD COLUMN content_format TEXT",
            "ALTER TABLE documents ADD COLUMN pipeline_mode TEXT",
            "ALTER TABLE documents ADD COLUMN vlm_preset TEXT",
            "ALTER TABLE documents ADD COLUMN standard_profile TEXT",
        ]
        for statement in statements:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

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


def _segment_doctags(doctags_text: str, source_checksum: str) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for block in parse_doctags_blocks(doctags_text):
        pieces = _split_long_segment(block.text)
        for offset, piece in enumerate(pieces):
            ordinal = (block.ordinal - 1) * 100 + offset + 1
            normalized = _normalize_text(piece)
            ev_id = _ev_id(source_checksum, block.page_number, ordinal, normalized)
            segments.append(
                {
                    "ev_id": ev_id,
                    "page_number": block.page_number,
                    "ordinal": ordinal,
                    "text": piece,
                    "normalized_text": normalized,
                    "start_offset": None,
                    "end_offset": None,
                }
            )
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


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_bbox(
    bbox: Mapping[str, object] | None,
    *,
    page_width: float | None,
    page_height: float | None,
) -> dict[str, float] | None:
    if not bbox or not page_width or not page_height or page_width <= 0 or page_height <= 0:
        return None
    x_min = _to_float(bbox.get("x_min"))
    x_max = _to_float(bbox.get("x_max"))
    y_min = _to_float(bbox.get("y_min"))
    y_max = _to_float(bbox.get("y_max"))
    if None in {x_min, x_max, y_min, y_max}:
        return None
    return {
        "x_min": max(0.0, min(float(x_min) / page_width, 1.0)),
        "y_min": max(0.0, min(float(y_min) / page_height, 1.0)),
        "x_max": max(0.0, min(float(x_max) / page_width, 1.0)),
        "y_max": max(0.0, min(float(y_max) / page_height, 1.0)),
    }


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
    "CitationLedgerEntry",
    "CitationRecordStats",
    "CitationStore",
    "EvidenceBundle",
    "IndexStats",
    "OutputCitationMention",
    "strip_citation_tokens",
]
