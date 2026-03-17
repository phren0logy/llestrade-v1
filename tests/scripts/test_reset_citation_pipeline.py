from __future__ import annotations

from pathlib import Path

from scripts.reset_citation_pipeline import collect_reset_paths, reset_citation_pipeline


def test_collect_reset_paths_returns_existing_derived_artifacts(tmp_path: Path) -> None:
    (tmp_path / ".llestrade").mkdir()
    (tmp_path / ".llestrade" / "citations.db").write_text("db", encoding="utf-8")
    (tmp_path / "converted_documents").mkdir()
    (tmp_path / "bulk_analysis").mkdir()

    paths = collect_reset_paths(tmp_path)

    assert paths == [
        tmp_path / ".llestrade" / "citations.db",
        tmp_path / "converted_documents",
        tmp_path / "bulk_analysis",
    ]


def test_reset_citation_pipeline_dry_run_preserves_files(tmp_path: Path) -> None:
    (tmp_path / ".llestrade").mkdir()
    (tmp_path / ".llestrade" / "citations.db").write_text("db", encoding="utf-8")
    (tmp_path / "reports").mkdir()

    removed = reset_citation_pipeline(tmp_path, dry_run=True)

    assert removed == [
        tmp_path / ".llestrade" / "citations.db",
        tmp_path / "reports",
    ]
    assert (tmp_path / ".llestrade" / "citations.db").exists()
    assert (tmp_path / "reports").exists()


def test_reset_citation_pipeline_removes_derived_artifacts(tmp_path: Path) -> None:
    (tmp_path / ".llestrade").mkdir()
    (tmp_path / ".llestrade" / "citations.db").write_text("db", encoding="utf-8")
    (tmp_path / "converted_documents").mkdir()
    (tmp_path / "highlights").mkdir()
    (tmp_path / "bulk_analysis").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "templates").mkdir()

    removed = reset_citation_pipeline(tmp_path)

    assert removed == [
        tmp_path / ".llestrade" / "citations.db",
        tmp_path / "converted_documents",
        tmp_path / "highlights",
        tmp_path / "bulk_analysis",
        tmp_path / "reports",
    ]
    assert not (tmp_path / ".llestrade" / "citations.db").exists()
    assert not (tmp_path / "converted_documents").exists()
    assert not (tmp_path / "highlights").exists()
    assert not (tmp_path / "bulk_analysis").exists()
    assert not (tmp_path / "reports").exists()
    assert (tmp_path / "templates").exists()
