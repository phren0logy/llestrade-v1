from __future__ import annotations

import hashlib
from pathlib import Path

import frontmatter
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from src.app.core.citations import CitationStore
from src.app.core.conversion_manager import ConversionJob
from src.app.workers.conversion_worker import ConversionWorker, FatalConversionError

_ = PySide6


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _write_converted_frontmatter(destination: Path, checksum: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"---\nsources:\n  - checksum: {checksum}\n---\nconverted\n",
        encoding="utf-8",
    )


def test_can_reuse_azure_raw_requires_valid_sidecars_and_checksum(qt_app: QApplication, tmp_path: Path) -> None:
    assert qt_app is not None

    source_pdf = tmp_path / "sample.pdf"
    source_pdf.write_bytes(b"pdf")
    checksum = hashlib.sha256(b"pdf").hexdigest()

    destination = tmp_path / "converted" / "sample.md"
    raw_markdown = tmp_path / "converted" / "sample.azure.raw.md"
    raw_json = tmp_path / "converted" / "sample.azure.raw.json"
    _write_converted_frontmatter(destination, checksum)
    raw_markdown.write_text("cached markdown", encoding="utf-8")
    raw_json.write_text('{"ok": true}', encoding="utf-8")

    worker = ConversionWorker([], helper="azure_di")
    assert worker._can_reuse_azure_raw(
        final_path=destination,
        raw_markdown_path=raw_markdown,
        raw_json_path=raw_json,
        source_checksum=checksum,
    )

    raw_json.write_text("{invalid-json", encoding="utf-8")
    assert not worker._can_reuse_azure_raw(
        final_path=destination,
        raw_markdown_path=raw_markdown,
        raw_json_path=raw_json,
        source_checksum=checksum,
    )

    _write_converted_frontmatter(destination, "different")
    raw_json.write_text('{"ok": true}', encoding="utf-8")
    assert not worker._can_reuse_azure_raw(
        final_path=destination,
        raw_markdown_path=raw_markdown,
        raw_json_path=raw_json,
        source_checksum=checksum,
    )


def test_convert_pdf_with_azure_reprocesses_when_raw_json_invalid(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert qt_app is not None

    job = ConversionJob(
        source_path=tmp_path / "sample.pdf",
        relative_path="sample.pdf",
        destination_path=tmp_path / "converted" / "sample.md",
        conversion_type="pdf",
    )
    job.source_path.parent.mkdir(parents=True, exist_ok=True)
    job.source_path.write_bytes(b"pdf")

    checksum = hashlib.sha256(b"pdf").hexdigest()
    _write_converted_frontmatter(job.destination_path, checksum)
    raw_markdown = job.destination_path.parent / "sample.azure.raw.md"
    raw_json = job.destination_path.parent / "sample.azure.raw.json"
    raw_markdown.write_text("cached raw\n<!-- PageBreak -->\npage2", encoding="utf-8")
    raw_json.write_text("{invalid-json", encoding="utf-8")

    class StubSettings:
        def get(self, key, default=None):
            if key == "azure_di_settings":
                return {"endpoint": "https://example"}
            return default

        def get_api_key(self, provider):
            return "secret" if provider == "azure_di" else None

    called = {"value": 0}

    def fake_process(_self, source_path, output_dir, json_dir, endpoint, key):  # noqa: ANN001
        called["value"] += 1
        produced_markdown = Path(output_dir) / "sample.md"
        produced_json = Path(output_dir) / "sample.json"
        produced_markdown.write_text("fresh markdown\n<!-- PageBreak -->\npage2", encoding="utf-8")
        produced_json.write_text('{"fresh": true}', encoding="utf-8")
        return str(produced_json), str(produced_markdown)

    monkeypatch.setattr("src.app.workers.conversion_worker.SecureSettings", StubSettings)
    monkeypatch.setattr("src.app.workers.conversion_worker.ConversionWorker._process_with_azure", fake_process)

    worker = ConversionWorker([job], helper="azure_di")
    worker._convert_pdf_with_azure(job)

    assert called["value"] == 1
    post = frontmatter.load(job.destination_path)
    assert post.metadata["azure_raw_cached"] is False
    assert raw_json.exists()
    assert "fresh" in raw_json.read_text(encoding="utf-8")


def test_index_citations_for_output_creates_project_db(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None

    project_dir = tmp_path / "project"
    source_pdf = project_dir / "source.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"pdf")

    destination = project_dir / "converted_documents" / "case" / "source.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "---\n"
        "sources:\n"
        "  - checksum: abc123\n"
        "pages_detected: 1\n"
        "pages_pdf: 1\n"
        "---\n"
        "<!--- case/source.pdf#page=1 --->\n"
        "Patient reported insomnia and anxiety.\n",
        encoding="utf-8",
    )

    job = ConversionJob(
        source_path=source_pdf,
        relative_path="case/source.pdf",
        destination_path=destination,
        conversion_type="pdf",
    )

    worker = ConversionWorker([], helper="azure_di")
    worker._index_citations_for_output(job, destination)

    store = CitationStore(project_dir)
    ids = store.list_evidence_ids_for_documents(relative_paths=["case/source.md"])
    assert ids


def test_azure_credentials_normalize_endpoint_and_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Settings:
        def get(self, key, default=None):
            if key == "azure_di_settings":
                return {"endpoint": "https://example.cognitiveservices.azure.com/"}
            return default

        def get_api_key(self, provider):
            return "secret" if provider == "azure_di" else None

    monkeypatch.setattr("src.app.workers.conversion_worker.SecureSettings", _Settings)

    worker = ConversionWorker([], helper="azure_di")
    endpoint, key = worker._azure_credentials()

    assert endpoint == "https://example.cognitiveservices.azure.com"
    assert key == "secret"

    class _InvalidSettings(_Settings):
        def get(self, key, default=None):
            if key == "azure_di_settings":
                return {"endpoint": "http://example"}
            return default

    monkeypatch.setattr("src.app.workers.conversion_worker.SecureSettings", _InvalidSettings)

    with pytest.raises(FatalConversionError, match="endpoint is invalid"):
        worker._azure_credentials()


def test_conversion_worker_emits_fatal_error_once_and_stops_batch(
    tmp_path: Path,
) -> None:
    first = ConversionJob(
        source_path=tmp_path / "first.pdf",
        relative_path="first.pdf",
        destination_path=tmp_path / "converted" / "first.md",
        conversion_type="pdf",
    )
    second = ConversionJob(
        source_path=tmp_path / "second.pdf",
        relative_path="second.pdf",
        destination_path=tmp_path / "converted" / "second.md",
        conversion_type="pdf",
    )

    worker = ConversionWorker([first, second], helper="azure_di")
    fatal_messages: list[str] = []
    file_failures: list[tuple[str, str]] = []
    finished: list[tuple[int, int]] = []
    seen_jobs: list[str] = []

    def _execute(job: ConversionJob) -> None:
        seen_jobs.append(job.relative_path)
        raise FatalConversionError("Azure credentials rejected")

    worker.fatal_error.connect(fatal_messages.append)
    worker.file_failed.connect(lambda path, error: file_failures.append((path, error)))
    worker.finished.connect(lambda successes, failures: finished.append((successes, failures)))
    worker._execute = _execute  # type: ignore[method-assign]

    worker._run()

    assert seen_jobs == ["first.pdf"]
    assert fatal_messages == ["Azure credentials rejected"]
    assert file_failures == []
    assert finished == [(0, 1)]
