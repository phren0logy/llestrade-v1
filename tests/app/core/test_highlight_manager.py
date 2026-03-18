from __future__ import annotations

from pathlib import Path

import fitz

from src.app.core.converted_documents import converted_artifact_relative
from src.app.core.highlight_manager import build_highlight_jobs
from src.app.core.project_manager import ProjectManager, ProjectMetadata, SourceTreeState
from src.app.workers.highlight_worker import HighlightWorker


def _create_pdf(path: Path, text: str, *, highlight: bool) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=12)
    if highlight:
        rects = page.search_for(text)
        for rect in rects:
            annot = page.add_highlight_annot(rect)
            annot.update()
    doc.save(path)
    doc.close()


def _setup_manager(tmp_path: Path) -> tuple[ProjectManager, Path, Path]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "converted_documents").mkdir()

    sources_root = tmp_path / "sources"
    (sources_root / "folder").mkdir(parents=True)

    manager = ProjectManager()
    manager.project_dir = project_root
    manager.project_path = project_root / "project.frpd"
    manager.metric_path = None
    manager.source_state = SourceTreeState(
        root=str(sources_root),
        selected_folders=["folder"],
        include_root_files=False,
    )
    manager.metadata = ProjectMetadata(case_name="Highlight Demo")

    return manager, project_root, sources_root


def test_build_highlight_jobs(tmp_path: Path) -> None:
    manager, project_root, sources_root = _setup_manager(tmp_path)

    pdf_path = sources_root / "folder" / "doc.pdf"
    _create_pdf(pdf_path, "Important", highlight=True)
    converted_path = project_root / "converted_documents" / converted_artifact_relative("folder/doc.pdf")
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    converted_path.write_text("content", encoding="utf-8")

    jobs = build_highlight_jobs(manager)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.source_pdf == pdf_path
    assert job.pdf_relative == "folder/doc.pdf"
    assert job.converted_relative == "folder/doc.pdf.doctags.txt"
    assert job.highlight_output == project_root / "highlights" / "documents" / "folder/doc.pdf.highlights.md"


def test_highlight_worker_creates_output(tmp_path: Path) -> None:
    manager, project_root, sources_root = _setup_manager(tmp_path)

    pdf_path = sources_root / "folder" / "doc.pdf"
    _create_pdf(pdf_path, "Annotated", highlight=True)
    converted_path = project_root / "converted_documents" / converted_artifact_relative("folder/doc.pdf")
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    converted_path.write_text("content", encoding="utf-8")

    jobs = build_highlight_jobs(manager)
    assert jobs
    job = jobs[0]

    worker = HighlightWorker([job])
    collection = worker._process_job(job)

    assert job.highlight_output.exists()
    content = job.highlight_output.read_text(encoding="utf-8")
    assert "Annotated" in content
    assert "folder/doc.pdf" in content
    assert collection is not None
    assert not collection.is_empty()


def test_highlight_worker_creates_placeholder_when_no_highlights(tmp_path: Path) -> None:
    manager, project_root, sources_root = _setup_manager(tmp_path)

    pdf_path = sources_root / "folder" / "doc.pdf"
    _create_pdf(pdf_path, "Plain", highlight=False)
    converted_path = project_root / "converted_documents" / converted_artifact_relative("folder/doc.pdf")
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    converted_path.write_text("content", encoding="utf-8")

    jobs = build_highlight_jobs(manager)
    assert jobs
    job = jobs[0]

    worker = HighlightWorker([job])
    collection = worker._process_job(job)

    assert job.highlight_output.exists()
    content = job.highlight_output.read_text(encoding="utf-8")
    assert "No highlights found" in content
    assert collection.is_empty()


def test_highlight_worker_writes_color_aggregates(tmp_path: Path) -> None:
    manager, project_root, sources_root = _setup_manager(tmp_path)

    pdf_path = sources_root / "folder" / "doc.pdf"
    _create_pdf(pdf_path, "Color text", highlight=True)
    converted_path = project_root / "converted_documents" / converted_artifact_relative("folder/doc.pdf")
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    converted_path.write_text("content", encoding="utf-8")

    jobs = build_highlight_jobs(manager)
    assert jobs

    worker = HighlightWorker(jobs)
    worker._run()

    colors_dir = project_root / "highlights" / "colors"
    files = list(colors_dir.glob("*.md"))
    assert files, "Expected aggregated color files to be created"
    content = files[0].read_text(encoding="utf-8")
    assert "Page" in content
    assert "folder/doc.pdf" in content
    assert worker.summary is not None
    assert worker.summary.color_files_written == len(files)


def test_highlight_worker_migrates_legacy_colors_directory(tmp_path: Path) -> None:
    manager, project_root, sources_root = _setup_manager(tmp_path)

    pdf_path = sources_root / "folder" / "doc.pdf"
    _create_pdf(pdf_path, "Legacy Colors", highlight=True)
    converted_path = project_root / "converted_documents" / converted_artifact_relative("folder/doc.pdf")
    converted_path.parent.mkdir(parents=True, exist_ok=True)
    converted_path.write_text("content", encoding="utf-8")

    legacy_colors = project_root / "highlights" / "documents" / "colors"
    legacy_colors.mkdir(parents=True, exist_ok=True)
    (legacy_colors / "old-color.md").write_text("stale", encoding="utf-8")

    jobs = build_highlight_jobs(manager)
    assert jobs

    worker = HighlightWorker(jobs)
    worker._run()

    assert not legacy_colors.exists()
    colors_dir = project_root / "highlights" / "colors"
    assert colors_dir.exists()
