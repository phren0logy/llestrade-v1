from __future__ import annotations

import json
from pathlib import Path

from src.app.core.citations import CitationStore, parse_citation_tokens


def test_parse_citation_tokens_reports_positions() -> None:
    text = "Line one [CIT:ev_1234abcd5678ef00]\nNext [CIT:ev_abcdef0123456789]"
    tokens = parse_citation_tokens(text)

    assert [token.ev_id for token in tokens] == [
        "ev_1234abcd5678ef00",
        "ev_abcdef0123456789",
    ]
    assert tokens[0].line == 1
    assert tokens[1].line == 2
    assert tokens[0].column > 1


def test_citation_store_indexes_and_verifies_with_geometry(tmp_path: Path) -> None:
    project_dir = tmp_path
    store = CitationStore(project_dir)

    raw_json = project_dir / "converted_documents" / "case" / "doc.azure.raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 1,
                        "unit": "pixel",
                        "lines": [
                            {
                                "content": "Patient reported insomnia and anxiety.",
                                "polygon": [0, 0, 100, 0, 100, 10, 0, 10],
                            }
                        ],
                    },
                    {
                        "page_number": 2,
                        "unit": "pixel",
                        "lines": [
                            {
                                "content": "Second page medication list.",
                                "polygon": [0, 20, 100, 20, 100, 30, 0, 30],
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    markdown = (
        "<!--- case/doc.pdf#page=1 --->\n"
        "Patient reported insomnia and anxiety.\n\n"
        "Additional details are documented.\n\n"
        "<!--- case/doc.pdf#page=2 --->\n"
        "Second page medication list.\n"
    )

    stats = store.index_converted_document(
        relative_path="case/doc.md",
        markdown_text=markdown,
        source_checksum="checksum-1",
        azure_raw_json_path=raw_json,
        pages_pdf=2,
        pages_detected=2,
    )
    assert stats.segments_indexed >= 2
    assert stats.geometry_spans_indexed >= 2

    ids = store.list_evidence_ids_for_documents(relative_paths=["case/doc.md"])
    assert ids
    ev_id = ids[0]

    verified = store.verify_citations(
        f"Patient reported insomnia and anxiety [CIT:{ev_id}]"
    )
    assert len(verified) == 1
    assert verified[0].status == "valid"

    unknown = store.verify_citations("Unsupported claim [CIT:ev_deadbeefdeadbeef]")
    assert len(unknown) == 1
    assert unknown[0].status == "invalid"

    output_path = project_dir / "bulk_analysis" / "group" / "outputs" / "doc_analysis.md"
    record = store.record_output_citations(
        output_path=output_path,
        output_text=f"Patient reported insomnia and anxiety [CIT:{ev_id}]",
        generator="bulk_analysis_worker",
        prompt_hash="hash-1",
    )
    assert record.total == 1
    assert record.valid == 1

    bundle = store.get_evidence_bundle(ev_id)
    assert bundle is not None
    assert bundle.document_relative_path == "case/doc.md"
    assert bundle.geometry


def test_citation_store_ingests_chunked_azure_json(tmp_path: Path) -> None:
    project_dir = tmp_path
    store = CitationStore(project_dir)

    raw_json = project_dir / "converted_documents" / "case" / "chunked.azure.raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.write_text(
        json.dumps(
            {
                "mode": "chunked",
                "chunks": [
                    {
                        "range": {"start": 1, "end": 2},
                        "analyze_result": {
                            "pages": [
                                {
                                    "page_number": 1,
                                    "unit": "pixel",
                                    "lines": [
                                        {
                                            "content": "Chunked page one evidence.",
                                            "polygon": [0, 0, 20, 0, 20, 10, 0, 10],
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    markdown = (
        "<!--- case/chunked.pdf#page=1 --->\n"
        "Chunked page one evidence.\n"
    )

    stats = store.index_converted_document(
        relative_path="case/chunked.md",
        markdown_text=markdown,
        source_checksum="checksum-2",
        azure_raw_json_path=raw_json,
        pages_pdf=1,
        pages_detected=1,
    )

    assert stats.segments_indexed >= 1
    assert stats.geometry_spans_indexed >= 1
