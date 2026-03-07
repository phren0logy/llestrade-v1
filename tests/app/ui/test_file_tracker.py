import json
from pathlib import Path
import pytest

from src.app.core.file_tracker import FileTracker


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    for folder in ("converted_documents", "bulk_analysis"):
        (tmp_path / folder).mkdir()
    (tmp_path / "highlights" / "documents").mkdir(parents=True)
    return tmp_path


def write_file(path: Path, name: str, content: str = "sample") -> None:
    full_path = path / name
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)


def load_tracker_snapshot(root: Path) -> dict:
    payload = json.loads((root / "file_tracker.json").read_text())
    payload["counts"] = {k: int(v) for k, v in payload.get("counts", {}).items()}
    return payload


def test_scan_empty_project_creates_tracker(project_root: Path):
    tracker = FileTracker(project_root)
    snapshot = tracker.scan()

    assert snapshot.imported_count == 0
    assert snapshot.bulk_analysis_count == 0
    assert snapshot.highlights_count == 0
    assert snapshot.missing["bulk_analysis_missing"] == []
    assert snapshot.missing["highlights_missing"] == []

    stored = load_tracker_snapshot(project_root)
    # Core counters must be present and zero; imported_pdf may also be present for PDF-only metrics
    assert stored["counts"].get("imported") == 0
    assert stored["counts"].get("bulk_analysis") == 0
    assert stored["counts"].get("highlights") == 0
    if "imported_pdf" in stored["counts"]:
        assert stored["counts"]["imported_pdf"] == 0
    # Files lists are empty for a fresh project; imported_pdf may also be present
    assert stored["files"].get("imported", []) == []
    assert stored["files"].get("bulk_analysis", []) == []
    assert stored["files"].get("highlights", []) == []
    if "imported_pdf" in stored["files"]:
        assert stored["files"]["imported_pdf"] == []


def test_scan_detects_missing_bulk_outputs(project_root: Path):
    # Mark converted doc as originating from PDF so highlights are expected
    write_file(
        project_root / "converted_documents",
        "case/doc1.md",
        "---\nsource_format: pdf\n---\ncontent",
    )

    tracker = FileTracker(project_root)
    snapshot = tracker.scan()

    assert snapshot.imported_count == 1
    assert snapshot.bulk_analysis_count == 0
    assert snapshot.missing["bulk_analysis_missing"] == ["case/doc1.md"]
    assert snapshot.missing["highlights_missing"] == ["case/doc1.md"]


def test_scan_updates_when_new_files_added(project_root: Path):
    tracker = FileTracker(project_root)
    tracker.scan()

    write_file(
        project_root / "converted_documents",
        "doc1.md",
        "---\nsource_format: pdf\n---\ncontent",
    )
    snapshot = tracker.scan()
    assert snapshot.imported_count == 1
    assert tracker.snapshot is snapshot
    assert snapshot.missing["highlights_missing"] == ["doc1.md"]

    write_file(project_root / "bulk_analysis", "doc1.md")
    snapshot = tracker.scan()
    assert snapshot.bulk_analysis_count == 1
    assert snapshot.files["bulk_analysis"] == ["doc1.md"]
    assert snapshot.missing["bulk_analysis_missing"] == []

    write_file(project_root / "highlights" / "documents", "doc1.highlights.md")
    snapshot = tracker.scan()
    assert snapshot.highlights_count == 1
    assert snapshot.missing["highlights_missing"] == []


def test_scan_normalises_summary_bulk_outputs(project_root: Path) -> None:
    write_file(
        project_root / "converted_documents",
        "case/doc1.md",
        "---\nsource_format: pdf\n---\ncontent",
    )
    write_file(
        project_root / "bulk_analysis",
        "summary/case/doc1_analysis.md",
    )

    tracker = FileTracker(project_root)
    snapshot = tracker.scan()

    assert snapshot.bulk_analysis_count == 1
    assert snapshot.missing["bulk_analysis_missing"] == []
    assert snapshot.counts["bulk_analysis"] == 1


def test_load_returns_none_when_no_snapshot(project_root: Path):
    tracker = FileTracker(project_root)
    assert tracker.load() is None


def test_load_reads_previous_snapshot(project_root: Path):
    tracker = FileTracker(project_root)
    tracker.scan()

    loaded = FileTracker(project_root).load()
    assert loaded is not None
    assert loaded.counts == tracker.snapshot.counts


def test_scan_ignores_bulk_analysis_noise_files(project_root: Path) -> None:
    write_file(project_root / "converted_documents", "case/doc1.md")
    write_file(project_root / "bulk_analysis", "case/doc1_analysis.md")
    write_file(project_root / "bulk_analysis", ".DS_Store", "")
    write_file(project_root / "bulk_analysis", "case/.DS_Store", "")
    write_file(project_root / "bulk_analysis", "case/config.json", "{}")

    tracker = FileTracker(project_root)
    snapshot = tracker.scan()

    assert snapshot.bulk_analysis_count == 1
    assert snapshot.files["bulk_analysis"] == ["case/doc1_analysis.md"]


def test_scan_ignores_azure_raw_sidecars(project_root: Path) -> None:
    write_file(project_root / "converted_documents", "case/doc1.md")
    write_file(project_root / "converted_documents", "case/doc1.azure.raw.md")
    write_file(project_root / "converted_documents", "case/doc1.azure.raw.json", "{}")

    tracker = FileTracker(project_root)
    snapshot = tracker.scan()

    assert snapshot.imported_count == 1
    assert snapshot.files["imported"] == ["case/doc1.md"]
