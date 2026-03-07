from __future__ import annotations

import json
from pathlib import Path

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

