"""Shared progress payloads for long-running worker operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkerProgressDetail:
    """Structured progress snapshot for GUI updates and trace mirroring."""

    run_kind: str
    phase: str
    label: str
    percent: int | None = None
    completed: int | None = None
    total: int | None = None
    document_path: str | None = None
    chunk_index: int | None = None
    chunk_total: int | None = None
    section_index: int | None = None
    section_total: int | None = None
    section_title: str | None = None
    detail: str | None = None
    chunks_completed: int | None = None
    chunks_in_flight: int | None = None


__all__ = ["WorkerProgressDetail"]
