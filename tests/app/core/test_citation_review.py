from __future__ import annotations

import json
from pathlib import Path

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.citation_review import CitationReviewService
from src.app.core.citations import CitationStore


def test_citation_review_service_loads_reviewed_output(tmp_path: Path) -> None:
    project_dir = tmp_path
    source_pdf = project_dir / "case" / "doc.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4")

    raw_json = project_dir / "converted_documents" / "case" / "doc.azure.raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 1,
                        "width": 100,
                        "height": 200,
                        "unit": "pixel",
                        "lines": [
                            {
                                "content": "Patient reported insomnia and anxiety.",
                                "polygon": [10, 20, 60, 20, 60, 40, 10, 40],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    store = CitationStore(project_dir)
    store.index_converted_document(
        relative_path="case/doc.md",
        markdown_text="<!--- case/doc.pdf#page=1 --->\nPatient reported insomnia and anxiety.\n",
        source_checksum="review-checksum",
        azure_raw_json_path=raw_json,
        pages_pdf=1,
        pages_detected=1,
        source_relative_path="case/doc.pdf",
        source_absolute_path=source_pdf.resolve().as_posix(),
    )
    _, mapping = store.build_local_citation_appendix(relative_path="case/doc.md", page_numbers=[1])

    output_path = project_dir / "bulk_analysis" / "demo" / "case" / "doc_analysis.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("Claim text [C1]\n", encoding="utf-8")
    store.record_output_citations(
        output_path=output_path,
        output_text="Claim text [C1]\n",
        generator="bulk_analysis_worker",
        prompt_hash="review-hash",
        label_mapping=mapping,
    )

    service = CitationReviewService(project_dir)
    group = BulkAnalysisGroup.create("Demo")
    outputs = service.list_map_outputs(group)
    assert outputs

    reviewed = service.load_reviewed_output(output_path, relative_key="case/doc_analysis.md")
    assert reviewed.citations[0].citation_label == "C1"
    assert reviewed.citations[0].source_absolute_path == source_pdf.resolve().as_posix()
    assert reviewed.citations[0].boxes[0]["x_min"] == 0.1
