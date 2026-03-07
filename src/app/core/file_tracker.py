"""File tracking utilities for dashboard workflows."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from src.app.core.azure_artifacts import is_azure_raw_artifact
from src.app.core.bulk_paths import (
    iter_map_outputs,
    iter_map_outputs_under,
    normalize_map_relative,
    resolve_map_output_path,
)

if TYPE_CHECKING:
    from .bulk_analysis_groups import BulkAnalysisGroup

LOGGER = logging.getLogger(__name__)

TRACKER_FILENAME = "file_tracker.json"
TRACKER_VERSION = "1"


@dataclass
class FileTrackerSnapshot:
    """Summary of document state within a project."""

    timestamp: datetime
    counts: Dict[str, int] = field(default_factory=dict)
    files: Dict[str, List[str]] = field(default_factory=dict)
    missing: Dict[str, List[str]] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    version: str = TRACKER_VERSION

    @property
    def imported_count(self) -> int:
        return self.counts.get("imported", 0)

    @property
    def bulk_analysis_count(self) -> int:
        return self.counts.get("bulk_analysis", 0)

    @property
    def summaries_count(self) -> int:
        """Compatibility alias for legacy callers."""
        return self.bulk_analysis_count

    @property
    def highlights_count(self) -> int:
        return self.counts.get("highlights", 0)

    def to_dashboard_metrics(self) -> "DashboardMetrics":
        """Translate the snapshot into lightweight dashboard metrics."""
        return DashboardMetrics.from_snapshot(self)

    def to_json(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "counts": self.counts,
            "files": self.files,
            "missing": self.missing,
            "notes": self.notes,
            "version": self.version,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "FileTrackerSnapshot":
        timestamp = payload.get("timestamp")
        counts = dict(payload.get("counts", {}))
        files = {k: list(v) for k, v in dict(payload.get("files", {})).items()}
        missing = {k: list(v) for k, v in dict(payload.get("missing", {})).items()}

        if "bulk_analysis" not in counts and "summaries" in counts:
            counts["bulk_analysis"] = counts.pop("summaries")
        if "bulk_analysis" not in files and "summaries" in files:
            files["bulk_analysis"] = files.pop("summaries")
        if "bulk_analysis_missing" not in missing and "summaries_missing" in missing:
            missing["bulk_analysis_missing"] = missing.pop("summaries_missing")

        return cls(
            timestamp=(
                datetime.fromisoformat(timestamp)
                if isinstance(timestamp, str)
                else datetime.now(timezone.utc)
            ),
            counts=counts,
            files=files,
            missing=missing,
            notes=dict(payload.get("notes", {})),
            version=str(payload.get("version", TRACKER_VERSION)),
        )


@dataclass(frozen=True)
class DashboardMetrics:
    """Aggregated counts surfaced to the dashboard and welcome views."""

    last_scan: Optional[datetime]
    imported_total: int = 0
    highlights_total: int = 0
    bulk_analysis_total: int = 0
    pending_highlights: int = 0
    pending_bulk_analysis: int = 0
    notes: Dict[str, str] = field(default_factory=dict)
    snapshot_version: str = TRACKER_VERSION

    @classmethod
    def empty(cls) -> "DashboardMetrics":
        return cls(
            last_scan=None,
            imported_total=0,
            highlights_total=0,
            bulk_analysis_total=0,
            pending_highlights=0,
            pending_bulk_analysis=0,
            notes={},
            snapshot_version=TRACKER_VERSION,
        )

    @classmethod
    def from_snapshot(cls, snapshot: FileTrackerSnapshot) -> "DashboardMetrics":
        return cls(
            last_scan=snapshot.timestamp,
            imported_total=snapshot.imported_count,
            highlights_total=snapshot.highlights_count,
            bulk_analysis_total=snapshot.bulk_analysis_count,
            pending_highlights=len(snapshot.missing.get("highlights_missing", [])),
            pending_bulk_analysis=len(snapshot.missing.get("bulk_analysis_missing", [])),
            notes=dict(snapshot.notes),
            snapshot_version=snapshot.version,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "imported_total": self.imported_total,
            "highlights_total": self.highlights_total,
            "bulk_analysis_total": self.bulk_analysis_total,
            "pending_highlights": self.pending_highlights,
            "pending_bulk_analysis": self.pending_bulk_analysis,
            "notes": dict(self.notes),
            "snapshot_version": self.snapshot_version,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object] | None) -> "DashboardMetrics":
        if not payload:
            return cls.empty()
        last_scan_value = payload.get("last_scan") if isinstance(payload, dict) else None
        last_scan: Optional[datetime]
        if isinstance(last_scan_value, str):
            try:
                last_scan = datetime.fromisoformat(last_scan_value)
            except ValueError:
                last_scan = None
        else:
            last_scan = None
        bulk_total = payload.get("bulk_analysis_total") if isinstance(payload, dict) else None
        if bulk_total is None and isinstance(payload, dict):
            bulk_total = payload.get("summaries_total")

        pending_bulk = payload.get("pending_bulk_analysis") if isinstance(payload, dict) else None
        if pending_bulk is None and isinstance(payload, dict):
            pending_bulk = payload.get("pending_summaries")

        highlights_total = payload.get("highlights_total") if isinstance(payload, dict) else None
        pending_highlights = payload.get("pending_highlights") if isinstance(payload, dict) else None

        return cls(
            last_scan=last_scan,
            imported_total=int(payload.get("imported_total", 0)),
            highlights_total=int(highlights_total or 0),
            bulk_analysis_total=int(bulk_total or 0),
            pending_highlights=int(pending_highlights or 0),
            pending_bulk_analysis=int(pending_bulk or 0),
            notes=dict(payload.get("notes", {})),
            snapshot_version=str(payload.get("snapshot_version", TRACKER_VERSION)),
        )

class FileTracker:
    """Track files within a project directory.

    The tracker inspects the canonical subdirectories:
    - converted_documents/
    - bulk_analysis/

    Each `scan()` collects counts and missing counterparts, then persists
    the snapshot to `file_tracker.json` under the project root.
    """

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.snapshot: Optional[FileTrackerSnapshot] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> Optional[FileTrackerSnapshot]:
        """Load the previously stored snapshot, if present."""
        tracker_path = self._tracker_file()
        if not tracker_path.exists():
            LOGGER.debug("No file tracker cache at %s", tracker_path)
            return None
        try:
            payload = json.loads(tracker_path.read_text())
            self.snapshot = FileTrackerSnapshot.from_json(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load file tracker snapshot: %s", exc)
            self.snapshot = None
        return self.snapshot

    def scan(self) -> FileTrackerSnapshot:
        """Walk the project directories and generate a fresh snapshot."""
        imported = self._gather_files("converted_documents")
        converted_root = self.project_path / "converted_documents"
        bulk_analysis = self._filter_bulk_analysis_files(
            self._gather_files("bulk_analysis")
        )
        highlights_files = self._gather_files("highlights")
        highlights_normalized = self._normalize_highlight_files(highlights_files)

        normalized_bulk_for_docs = self._normalize_bulk_outputs(bulk_analysis, imported)

        # Determine which converted files originated from PDFs (eligible for highlights)
        imported_pdf: set[str] = set()
        for rel in imported:
            try:
                if _converted_is_pdf(converted_root / rel):
                    imported_pdf.add(rel)
            except Exception:
                # Defensive: treat unreadable entries as non-PDF
                continue

        counts = {
            "imported": len(imported),
            "imported_pdf": len(imported_pdf),
            "bulk_analysis": len(normalized_bulk_for_docs),
            "highlights": len(highlights_normalized),
        }

        files = {
            "imported": sorted(imported),
            "imported_pdf": sorted(imported_pdf),
            "bulk_analysis": sorted(bulk_analysis),
            "highlights": sorted(highlights_files),
        }

        missing = {
            "bulk_analysis_missing": sorted(imported - normalized_bulk_for_docs),
            # Only PDFs are eligible for highlights; compute missing accordingly
            "highlights_missing": sorted(imported_pdf - highlights_normalized),
        }

        snapshot = FileTrackerSnapshot(
            timestamp=datetime.now(timezone.utc),
            counts=counts,
            files=files,
            missing=missing,
        )
        self.snapshot = snapshot
        self._write_snapshot(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tracker_file(self) -> Path:
        return self.project_path / TRACKER_FILENAME

    def _gather_files(self, folder_name: str) -> set[str]:
        folder = self.project_path / folder_name
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            return set()

        collected: set[str] = set()
        for path in folder.rglob("*"):
            if path.is_file():
                if folder_name == "converted_documents" and is_azure_raw_artifact(path):
                    continue
                collected.add(path.relative_to(folder).as_posix())
        return collected

    def _filter_bulk_analysis_files(self, files: set[str]) -> set[str]:
        """Return bulk-analysis output paths that should be counted."""

        if not files:
            return files

        filtered: set[str] = set()
        for path in files:
            parts = path.split("/")
            if not parts:
                continue
            leaf = parts[-1]
            if leaf == ".DS_Store":
                continue
            if leaf == "config.json":
                continue
            filtered.add(path)
        return filtered

    def _normalize_highlight_files(self, files: set[str]) -> set[str]:
        if not files:
            return set()

        normalized: set[str] = set()
        for path in files:
            normalized_entry = _normalize_highlight_entry(path)
            if normalized_entry:
                normalized.add(normalized_entry)
        return normalized

    def _normalize_bulk_outputs(self, files: set[str], converted: set[str]) -> set[str]:
        """Map bulk-analysis artefacts back to converted document paths."""

        if not files:
            return set()

        normalized: set[str] = set()
        for path in files:
            # Direct matches (legacy layout) pass straight through.
            if path in converted:
                normalized.add(path)
                continue
            parsed = _parse_bulk_output_entry(path)
            if not parsed:
                continue
            slug, resolved = parsed

            candidates = [resolved]
            if slug and not resolved.startswith(slug + "/"):
                candidates.append(f"{slug}/{resolved}")

            for candidate in candidates:
                if candidate in converted:
                    normalized.add(candidate)
                    break
        return normalized

    def _write_snapshot(self, snapshot: FileTrackerSnapshot) -> None:
        tracker_path = self._tracker_file()
        try:
            tracker_path.write_text(json.dumps(snapshot.to_json(), indent=2))
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to persist file tracker snapshot: %s", exc)


@dataclass(frozen=True)
class WorkspaceGroupMetrics:
    """Coverage details for a single bulk-analysis group.

    `converted_files` contains the converted-document relative paths that the group may
    operate on. Counts are derived from these files so callers do not need to re-run
    filtering logic in the UI layer.
    """

    group_id: str
    name: str
    slug: str
    converted_files: tuple[str, ...]
    converted_count: int
    bulk_analysis_total: int
    pending_bulk_analysis: int
    pending_files: tuple[str, ...]
    # Operation type and combined-operation metrics
    operation: str = "per_document"
    combined_input_count: int = 0
    combined_latest_path: str | None = None
    combined_latest_at: datetime | None = None
    combined_is_stale: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "slug": self.slug,
            "converted_files": list(self.converted_files),
            "converted_count": self.converted_count,
            "bulk_analysis_total": self.bulk_analysis_total,
            "pending_bulk_analysis": self.pending_bulk_analysis,
            "pending_files": list(self.pending_files),
            "operation": self.operation,
            "combined_input_count": self.combined_input_count,
            "combined_latest_path": self.combined_latest_path,
            "combined_latest_at": self.combined_latest_at.isoformat() if self.combined_latest_at else None,
            "combined_is_stale": self.combined_is_stale,
        }


@dataclass(frozen=True)
class WorkspaceMetrics:
    """Aggregated dashboard + group metrics for workspace consumption."""

    dashboard: "DashboardMetrics"
    highlights_missing: tuple[str, ...]
    bulk_missing: tuple[str, ...]
    groups: Dict[str, WorkspaceGroupMetrics]

    def to_dict(self) -> Dict[str, object]:
        return {
            "dashboard": self.dashboard.to_dict(),
            "highlights_missing": list(self.highlights_missing),
            "bulk_missing": list(self.bulk_missing),
            "groups": {group_id: metrics.to_dict() for group_id, metrics in self.groups.items()},
        }


def build_workspace_metrics(
    *,
    snapshot: "FileTrackerSnapshot | None",
    dashboard: "DashboardMetrics",
    bulk_analysis_groups: Sequence["BulkAnalysisGroup"],
    project_dir: Path | None = None,
) -> WorkspaceMetrics:
    """Translate raw tracker data into workspace-friendly metrics."""

    if snapshot is None:
        highlights_missing: tuple[str, ...] = tuple()
        bulk_missing: tuple[str, ...] = tuple()
        groups: Dict[str, WorkspaceGroupMetrics] = {}
        return WorkspaceMetrics(
            dashboard=dashboard,
            highlights_missing=highlights_missing,
            bulk_missing=bulk_missing,
            groups=groups,
        )

    converted_files = set(snapshot.files.get("imported", []))
    converted_pdf_files = set(snapshot.files.get("imported_pdf", [])) or converted_files
    bulk_files = set(snapshot.files.get("bulk_analysis", []))
    highlight_files = set(snapshot.files.get("highlights", []))

    highlights_normalized = {
        normalized
        for entry in highlight_files
        if (normalized := _normalize_highlight_entry(entry)) is not None
    }

    normalized_bulk_files: set[str] = set()
    bulk_outputs_by_group: Dict[str, set[str]] = defaultdict(set)
    for relative_path in bulk_files:
        parsed = _parse_bulk_output_entry(relative_path)
        if not parsed:
            continue
        slug, normalized = parsed
        normalized_bulk_files.add(normalized)
        bulk_outputs_by_group[slug].add(normalized)

    # Only PDFs are eligible for highlights; prefer imported_pdf if available
    highlights_missing = tuple(sorted(converted_pdf_files - highlights_normalized))
    bulk_missing = tuple(sorted(converted_files - normalized_bulk_files))

    group_metrics: Dict[str, WorkspaceGroupMetrics] = {}
    for group in bulk_analysis_groups:
        slug = getattr(group, "slug", None) or group.folder_name
        converted_subset = _resolve_group_converted_paths(group, converted_files)
        group_outputs = bulk_outputs_by_group.get(slug, set())
        bulk_subset = {path for path in converted_subset if path in group_outputs}

        pending_bulk = len(converted_subset) - len(bulk_subset)
        pending_files = tuple(sorted(converted_subset - bulk_subset))

        # Defaults for combined fields
        op_type = getattr(group, "operation", "per_document") or "per_document"
        combined_input_count = 0
        combined_latest_path: str | None = None
        combined_latest_at: datetime | None = None
        combined_is_stale = False

        # If the group represents a combined operation, compute inputs and status.
        if op_type == "combined" and project_dir is not None:
            combined_input_count, combined_latest_path, combined_latest_at, combined_is_stale = (
                _compute_combined_status(project_dir, group)
            )

        metrics = WorkspaceGroupMetrics(
            group_id=group.group_id,
            name=group.name,
            slug=slug,
            converted_files=tuple(sorted(converted_subset)),
            converted_count=len(converted_subset),
            bulk_analysis_total=len(bulk_subset),
            pending_bulk_analysis=max(pending_bulk, 0),
            pending_files=pending_files,
            operation=op_type,
            combined_input_count=combined_input_count,
            combined_latest_path=combined_latest_path,
            combined_latest_at=combined_latest_at,
            combined_is_stale=combined_is_stale,
        )
        group_metrics[group.group_id] = metrics

    return WorkspaceMetrics(
        dashboard=dashboard,
        highlights_missing=highlights_missing,
        bulk_missing=bulk_missing,
        groups=group_metrics,
    )


def _iter_project_files(root: Path, rel_dirs: Sequence[str]) -> set[str]:
    selected: set[str] = set()
    normalized_rel = {d.strip("/") for d in rel_dirs}
    for rel in list(normalized_rel):
        base = (root / rel).resolve()
        if not base.exists():
            continue
        if base.is_file():
            if is_azure_raw_artifact(base):
                continue
            selected.add(rel)
            continue
        for path in base.rglob("*.md"):
            if is_azure_raw_artifact(path):
                continue
            try:
                selected.add(path.relative_to(root).as_posix())
            except ValueError:
                # Path not under root; skip
                continue
    return selected


def _compute_combined_status(project_dir: Path, group: "BulkAnalysisGroup") -> tuple[int, str | None, datetime | None, bool]:
    # Build selection from converted_documents
    conv_root = project_dir / "converted_documents"
    converted_selected: set[str] = set()
    for rel in group.combine_converted_files or []:
        rel = rel.strip("/")
        if rel:
            converted_selected.add(rel)
    converted_selected |= _iter_project_files(conv_root, group.combine_converted_directories or [])

    # Build selection from per-document outputs under bulk_analysis
    ba_root = project_dir / "bulk_analysis"
    map_selected: set[str] = set()
    map_paths: Dict[str, Path] = {}

    for slug in group.combine_map_groups or []:
        slug = slug.strip()
        if not slug:
            continue
        for path, rel in iter_map_outputs(project_dir, slug):
            key = f"{slug}/{rel}"
            map_selected.add(key)
            map_paths.setdefault(key, path)

    for rel in group.combine_map_directories or []:
        rel = rel.strip("/")
        if not rel:
            continue
        parts = rel.split("/", 1)
        if len(parts) != 2:
            continue
        slug, remainder = parts
        slug = slug.strip()
        if not slug:
            continue
        normalized = normalize_map_relative(remainder)
        for path, rel_path in iter_map_outputs_under(project_dir, slug, normalized):
            key = f"{slug}/{rel_path}"
            map_selected.add(key)
            map_paths.setdefault(key, path)

    for rel in group.combine_map_files or []:
        rel = rel.strip("/")
        if not rel:
            continue
        parts = rel.split("/", 1)
        if len(parts) != 2:
            continue
        slug, remainder = parts
        slug = slug.strip()
        if not slug:
            continue
        normalized = normalize_map_relative(remainder)
        if not normalized:
            continue
        key = f"{slug}/{normalized}"
        map_selected.add(key)
        map_paths.setdefault(key, resolve_map_output_path(project_dir, slug, normalized))

    # Latest combined artifact under reduce/
    reduce_dir = ba_root / (getattr(group, "slug", None) or group.folder_name) / "reduce"
    latest_path: Path | None = None
    latest_mtime: float | None = None
    if reduce_dir.exists():
        for f in reduce_dir.glob("combined_*.md"):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = f

    latest_ts: datetime | None = None
    latest_rel: str | None = None
    if latest_path is not None:
        try:
            latest_ts = datetime.fromtimestamp(int(latest_mtime or 0))
            latest_rel = latest_path.relative_to(project_dir).as_posix()
        except Exception:
            latest_ts = None
            latest_rel = None

    # Staleness: if no artifact and there are inputs → stale
    inputs_count = len(converted_selected) + len(map_selected)
    if inputs_count == 0:
        return 0, latest_rel, latest_ts, False
    if latest_path is None:
        return inputs_count, latest_rel, latest_ts, True

    # Compare mtimes with manifest if present; else fallback to simple mtime comparison
    manifest = latest_path.with_suffix(".manifest.json")
    recorded: dict[str, float] = {}
    high_precision: set[str] = set()
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
            for entry in payload.get("inputs", []):
                path = entry.get("path")
                mtime_ns = entry.get("mtime_ns")
                mtime = entry.get("mtime")
                if isinstance(path, str):
                    if isinstance(mtime_ns, (int, float)):
                        recorded[path] = float(mtime_ns) / 1_000_000_000
                        high_precision.add(path)
                    elif isinstance(mtime, (int, float)):
                        recorded[path] = float(mtime)
        except Exception:
            recorded = {}
            high_precision = set()

    def _current_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    stale = False
    # Converted inputs: key namespace "converted/" for manifest paths
    for rel in converted_selected:
        key = f"converted/{rel}"
        current = _current_mtime(conv_root / rel)
        recorded_m = recorded.get(key)
        if recorded_m is None or recorded_m <= 0:
            stale = True
            break
        tolerance = 1e-6 if key in high_precision else 0.5
        if current - recorded_m > tolerance:
            stale = True
            break

    if not stale:
        for rel in map_selected:
            # rel is "slug/relative.md"
            key = f"map/{rel}"
            parts = rel.split("/", 1)
            slug = parts[0]
            remainder = parts[1] if len(parts) == 2 else ""
            normalized = normalize_map_relative(remainder)
            path = map_paths.get(rel)
            if path is None:
                path = resolve_map_output_path(project_dir, slug, normalized)
                map_paths[rel] = path
            current = _current_mtime(path)
            recorded_m = recorded.get(key)
            if recorded_m is None or recorded_m <= 0:
                stale = True
                break
            tolerance = 1e-6 if key in high_precision else 0.5
            if current - recorded_m > tolerance:
                stale = True
                break

    return inputs_count, latest_rel, latest_ts, stale


def _resolve_group_converted_paths(
    group: "BulkAnalysisGroup",
    converted_paths: set[str],
) -> set[str]:
    """Return the converted-document paths covered by the supplied group."""

    if not converted_paths:
        return set()

    selected: set[str] = set()
    # Only consider markdown or text files for per-document operations
    normalised_converted = {
        path.strip("/")
        for path in converted_paths
        if path.lower().endswith(".md") or path.lower().endswith(".txt")
    }

    for path in group.files:
        candidate = path.strip("/")
        if candidate and candidate in normalised_converted:
            selected.add(candidate)

    for directory in group.directories:
        normalised = directory.strip("/")
        if not normalised:
            selected.update(normalised_converted)
            continue
        prefix = normalised + "/"
        for path in normalised_converted:
            if path == normalised or path.startswith(prefix):
                selected.add(path)

    return selected


def _normalize_highlight_entry(relative_path: str) -> str | None:
    suffix = ".highlights.md"
    if not relative_path.endswith(suffix):
        return None
    if relative_path.startswith("colors/"):
        return None
    normalized = relative_path
    if normalized.startswith("documents/"):
        normalized = normalized[len("documents/") :]
    if not normalized:
        return None
    base = normalized[: -len(suffix)] + ".md"
    return base


def _converted_is_pdf(path: Path) -> bool:
    """Return True if the converted markdown at `path` originated from a PDF.

    Heuristic: inspect the YAML front-matter inserted by the converter and
    look for a line `source_format: pdf`. Reads only the first ~200 lines.
    """
    try:
        # Read a small portion of the file; YAML header is at the top
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline()
            if not first.startswith("---"):
                return False
            for _ in range(0, 200):
                line = fh.readline()
                if not line:
                    break
                if line.strip() == "---":
                    break
                # Cheap check without YAML parser
                if line.strip().lower().startswith("source_format:"):
                    return line.strip().lower().endswith("pdf")
    except Exception:
        return False
    return False


def _parse_bulk_output_entry(relative_path: str) -> tuple[str, str] | None:
    """Return (group_slug, normalized_path) for a bulk output entry."""

    if not relative_path:
        return None
    parts = relative_path.split("/", 1)
    if len(parts) != 2:
        return None
    slug, remainder = parts
    if not slug or not remainder:
        return None
    if remainder.startswith("outputs/"):
        remainder = remainder[len("outputs/") :]
    if not remainder:
        return None

    normalized = remainder
    if remainder.endswith(".md") and remainder[:-3].endswith("_analysis"):
        normalized = remainder[:-12] + ".md"

    return slug, normalized


__all__ = [
    "FileTracker",
    "FileTrackerSnapshot",
    "DashboardMetrics",
    "WorkspaceMetrics",
    "WorkspaceGroupMetrics",
    "build_workspace_metrics",
]
