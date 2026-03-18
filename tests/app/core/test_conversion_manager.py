from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from src.app.core.citations import CitationStore
from src.app.core.converted_documents import converted_artifact_relative
from src.app.core.conversion_manager import build_conversion_jobs


def _manager(project_dir: Path, *, root: str = "source", selected: list[str] | None = None):
    return SimpleNamespace(
        project_dir=project_dir,
        source_state=SimpleNamespace(root=root, selected_folders=list(selected or [])),
    )


def _write_source(project_dir: Path, relative: str, content: bytes) -> Path:
    path = project_dir / "source" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_converted_with_checksum(project_dir: Path, relative: str, checksum: str) -> Path:
    destination = project_dir / "converted_documents" / converted_artifact_relative(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "<page_header><loc_0><loc_0><loc_50><loc_20>converted</page_header>\n",
        encoding="utf-8",
    )
    CitationStore(project_dir).index_doctags_document(
        relative_path=destination.relative_to(project_dir / "converted_documents").as_posix(),
        doctags_text=destination.read_text(encoding="utf-8"),
        source_checksum=checksum,
        pages_pdf=1,
        pages_detected=1,
    )
    return destination


def test_build_conversion_jobs_returns_empty_when_source_root_missing(tmp_path: Path) -> None:
    manager = _manager(tmp_path, root="does-not-exist", selected=["docs"])
    plan = build_conversion_jobs(manager)
    assert plan.jobs == ()
    assert plan.duplicates == ()


def test_build_conversion_jobs_skips_when_checksum_matches_even_if_mtime_newer(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "docs/report.pdf", b"pdf-contents")
    checksum = hashlib.sha256(b"pdf-contents").hexdigest()
    destination = _write_converted_with_checksum(tmp_path, "docs/report.pdf", checksum)

    os.utime(destination, (1_000, 1_000))
    os.utime(source, (2_000, 2_000))

    manager = _manager(tmp_path, selected=["docs"])
    plan = build_conversion_jobs(manager)

    assert plan.jobs == ()
    assert plan.duplicates == ()


def test_build_conversion_jobs_adds_job_when_checksum_mismatch(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "docs/report.pdf", b"pdf-contents")
    destination = _write_converted_with_checksum(tmp_path, "docs/report.pdf", "different-checksum")

    os.utime(destination, (1_000, 1_000))
    os.utime(source, (2_000, 2_000))

    manager = _manager(tmp_path, selected=["docs"])
    plan = build_conversion_jobs(manager)

    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.relative_path == "docs/report.pdf"
    assert job.conversion_type == "pdf"
    assert job.destination_path == destination


def test_build_conversion_jobs_skips_when_destination_is_newer(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "docs/notes.pdf", b"pdf")
    destination = tmp_path / "converted_documents" / converted_artifact_relative("docs/notes.pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("converted", encoding="utf-8")

    os.utime(source, (1_000, 1_000))
    os.utime(destination, (2_000, 2_000))

    manager = _manager(tmp_path, selected=["docs"])
    plan = build_conversion_jobs(manager)

    assert plan.jobs == ()


def test_build_conversion_jobs_reports_duplicates_and_converts_only_one(tmp_path: Path) -> None:
    _write_source(tmp_path, "folder-a/doc1.pdf", b"same-content")
    _write_source(tmp_path, "folder-b/doc2.pdf", b"same-content")

    manager = _manager(tmp_path, selected=["folder-a", "folder-b"])
    plan = build_conversion_jobs(manager)

    assert len(plan.jobs) == 1
    assert len(plan.duplicates) == 1

    converted_relatives = {"folder-a/doc1.pdf", "folder-b/doc2.pdf"}
    assert plan.jobs[0].relative_path in converted_relatives

    duplicate = plan.duplicates[0]
    assert {duplicate.primary_relative, duplicate.duplicate_relative} == converted_relatives
    assert duplicate.primary_relative != duplicate.duplicate_relative


def test_build_conversion_jobs_ignores_hidden_and_unsupported_files(tmp_path: Path) -> None:
    _write_source(tmp_path, ".hidden.pdf", b"hidden")
    _write_source(tmp_path, "docs/data.csv", b"ignored")
    _write_source(tmp_path, "docs/note.txt", b"copy-me")
    _write_source(tmp_path, "docs/note.pdf", b"pdf")

    manager = _manager(tmp_path, selected=["", "docs"])
    plan = build_conversion_jobs(manager)

    assert len(plan.jobs) == 1
    assert plan.jobs[0].relative_path == "docs/note.pdf"
    assert plan.jobs[0].conversion_type == "pdf"
