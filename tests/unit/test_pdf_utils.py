from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import pdf_utils


class _FakeAnalyzeResult:
    def __init__(self, *, content: str, kind: str) -> None:
        self.content = content
        self._kind = kind

    def as_dict(self) -> dict[str, str]:
        return {"kind": self._kind}


class _FakePoller:
    def __init__(self, result: _FakeAnalyzeResult) -> None:
        self._result = result

    def result(self, timeout: int = 0) -> _FakeAnalyzeResult:  # noqa: ARG002
        return self._result


class _FakeDocumentIntelligenceClient:
    def __init__(self, *, endpoint: str, credential: object) -> None:  # noqa: ARG002
        self.calls: list[str] = []

    def begin_analyze_document(
        self,
        model_id: str,  # noqa: ARG002
        stream,
        output_content_format=None,
        pages: str | None = None,  # noqa: ARG002
    ) -> _FakePoller:
        # Regression guard: this raises ValueError if a closed stream is passed.
        pos = stream.tell()
        _ = stream.read(1)
        stream.seek(pos)

        if output_content_format == pdf_utils.DocumentContentFormat.MARKDOWN:
            self.calls.append("markdown")
            return _FakePoller(
                _FakeAnalyzeResult(content="Markdown output", kind="markdown")
            )

        self.calls.append("json")
        return _FakePoller(_FakeAnalyzeResult(content="", kind="json"))


def test_process_pdf_with_azure_keeps_stream_open_for_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, _FakeDocumentIntelligenceClient] = {}

    def _build_client(*, endpoint: str, credential: object) -> _FakeDocumentIntelligenceClient:
        client = _FakeDocumentIntelligenceClient(endpoint=endpoint, credential=credential)
        captured["client"] = client
        return client

    monkeypatch.setattr(pdf_utils, "DocumentIntelligenceClient", _build_client)
    monkeypatch.setattr(pdf_utils, "get_pdf_page_count", lambda _path: 1)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    json_path, markdown_path = pdf_utils.process_pdf_with_azure(
        str(pdf_path),
        str(output_dir),
        str(output_dir),
        str(output_dir),
        "https://example.cognitiveservices.azure.com/",
        "secret",
    )

    assert json_path is not None
    assert Path(markdown_path).exists()
    assert Path(json_path).exists()
    assert captured["client"].calls == ["markdown", "json"]
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["kind"] == "json"


def test_process_pdf_with_azure_does_not_retry_auth_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    class _FailingClient:
        def __init__(self, *, endpoint: str, credential: object) -> None:  # noqa: ARG002
            pass

        def begin_analyze_document(self, *args, **kwargs):  # noqa: ANN002, ANN003
            calls["count"] += 1
            raise pdf_utils.ClientAuthenticationError(
                message="bad credentials",
                response=SimpleNamespace(status_code=401, reason="Unauthorized"),
            )

    monkeypatch.setattr(pdf_utils, "DocumentIntelligenceClient", _FailingClient)
    monkeypatch.setattr(pdf_utils, "get_pdf_page_count", lambda _path: 1)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(pdf_utils.AzureDocumentIntelligenceAuthError, match="authentication failed"):
        pdf_utils.process_pdf_with_azure(
            str(pdf_path),
            str(output_dir),
            str(output_dir),
            str(output_dir),
            "https://example.cognitiveservices.azure.com/",
            "secret",
        )

    assert calls["count"] == 1


def test_process_pdf_with_azure_rejects_invalid_endpoint(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(pdf_utils.AzureDocumentIntelligenceConfigurationError, match="Invalid Azure Document Intelligence endpoint"):
        pdf_utils.process_pdf_with_azure(
            str(pdf_path),
            str(output_dir),
            str(output_dir),
            str(output_dir),
            "example.cognitiveservices.azure.com/",
            "secret",
        )


def test_process_pdf_with_azure_does_not_emit_f0_warnings_for_s0_docs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(pdf_utils, "DocumentIntelligenceClient", _FakeDocumentIntelligenceClient)
    monkeypatch.setattr(pdf_utils, "get_pdf_page_count", lambda _path: 140)
    monkeypatch.setattr(pdf_utils.os.path, "getsize", lambda _path: int(8.2 * 1024 * 1024))

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    pdf_utils.process_pdf_with_azure(
        str(pdf_path),
        str(output_dir),
        str(output_dir),
        str(output_dir),
        "https://example.cognitiveservices.azure.com/",
        "secret",
    )

    output = capsys.readouterr().out
    assert "F0 tier" not in output
    assert "first 2 pages" not in output


def test_process_pdf_with_azure_chunks_only_above_2000_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        pdf_utils,
        "DocumentIntelligenceClient",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(pdf_utils, "get_pdf_page_count", lambda _path: 2001)

    def _fake_chunked(**kwargs):  # noqa: ANN003
        assert kwargs["max_pages"] == 2000
        return "chunked markdown", [{"pages": "1-2000"}]

    monkeypatch.setattr(pdf_utils, "_azure_markdown_chunked", _fake_chunked)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    json_path, markdown_path = pdf_utils.process_pdf_with_azure(
        str(pdf_path),
        str(output_dir),
        str(output_dir),
        str(output_dir),
        "https://example.cognitiveservices.azure.com/",
        "secret",
    )

    assert json_path is not None
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["chunk_size"] == 2000
    assert payload["page_count"] == 2001
    assert "chunked markdown" in Path(markdown_path).read_text(encoding="utf-8")
