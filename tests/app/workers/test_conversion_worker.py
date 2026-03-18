from __future__ import annotations

from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from src.app.core.citations import CitationStore
from src.app.core.conversion_manager import ConversionJob
from src.app.core.converted_documents import converted_artifact_relative
from src.app.core.docling_local import DoclingConversionResult, DoclingLocalError
from src.app.workers.conversion_worker import ConversionWorker, FatalConversionError

_ = PySide6


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_convert_pdf_with_local_docling_writes_doctags_and_indexes(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert qt_app is not None

    project_dir = tmp_path / "project"
    source = project_dir / "case" / "sample.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.7")

    destination = project_dir / "converted_documents" / converted_artifact_relative("case/sample.pdf")
    job = ConversionJob(
        source_path=source,
        relative_path="case/sample.pdf",
        destination_path=destination,
        conversion_type="pdf",
    )

    monkeypatch.setattr("src.app.workers.conversion_worker.assert_local_docling_runtime", lambda: None)
    monkeypatch.setattr(
        "src.app.workers.conversion_worker.convert_pdf_to_doctags",
        lambda **_kwargs: DoclingConversionResult(
            doctags_content="<page_header><loc_0><loc_0><loc_200><loc_40>Patient reported insomnia.</page_header>",
            text_content="Patient reported insomnia.",
            json_content=None,
            filename="sample.pdf",
            status="success",
            processing_time=0.25,
            page_count=1,
        ),
    )
    monkeypatch.setattr("src.app.workers.conversion_worker.ConversionWorker._pdf_page_count", lambda _self, _path: 1)

    worker = ConversionWorker([job], helper="docling")
    worker._convert_with_docling(job)

    assert destination.exists()
    store = CitationStore(project_dir)
    metadata = store.get_document_metadata("case/sample.pdf.doctags.txt")
    assert metadata is not None
    assert metadata["content_format"] == "doctags"
    assert metadata["pipeline_mode"] == "vlm_primary"
    assert metadata["vlm_preset"] == "granite_docling"
    ids = store.list_evidence_ids_for_documents(relative_paths=["case/sample.pdf.doctags.txt"])
    assert ids


def test_convert_with_local_docling_surfaces_runtime_guard_as_fatal(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert qt_app is not None

    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF")
    job = ConversionJob(
        source_path=source,
        relative_path="sample.pdf",
        destination_path=tmp_path / "converted_documents" / "sample.pdf.doctags.txt",
        conversion_type="pdf",
    )

    monkeypatch.setattr(
        "src.app.workers.conversion_worker.assert_local_docling_runtime",
        lambda: (_ for _ in ()).throw(DoclingLocalError("Local Docling MLX dependencies are not installed")),
    )

    worker = ConversionWorker([job], helper="docling")
    with pytest.raises(FatalConversionError, match="Local Docling MLX dependencies are not installed"):
        worker._convert_with_docling(job)


def test_conversion_worker_emits_fatal_error_once_and_stops_batch(tmp_path: Path) -> None:
    first = ConversionJob(
        source_path=tmp_path / "first.pdf",
        relative_path="first.pdf",
        destination_path=tmp_path / "converted" / "first.pdf.doctags.txt",
        conversion_type="pdf",
    )
    second = ConversionJob(
        source_path=tmp_path / "second.pdf",
        relative_path="second.pdf",
        destination_path=tmp_path / "converted" / "second.pdf.doctags.txt",
        conversion_type="pdf",
    )

    worker = ConversionWorker([first, second], helper="docling")
    fatal_messages: list[str] = []
    file_failures: list[tuple[str, str]] = []
    finished: list[tuple[int, int]] = []
    seen_jobs: list[str] = []

    def _execute(job: ConversionJob) -> None:
        seen_jobs.append(job.relative_path)
        raise FatalConversionError("Local Docling runtime is unavailable")

    worker.fatal_error.connect(fatal_messages.append)
    worker.file_failed.connect(lambda path, error: file_failures.append((path, error)))
    worker.finished.connect(lambda successes, failures: finished.append((successes, failures)))
    worker._execute = _execute  # type: ignore[method-assign]

    worker._run()

    assert seen_jobs == ["first.pdf"]
    assert fatal_messages == ["Local Docling runtime is unavailable"]
    assert file_failures == []
    assert finished == [(0, 1)]
