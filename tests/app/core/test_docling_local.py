from __future__ import annotations

from types import SimpleNamespace

from docling_core.types.doc.base import BoundingBox, CoordOrigin

from src.app.core.docling_local import _normalize_document_bboxes


def test_normalize_document_bboxes_repairs_inverted_top_left_bbox() -> None:
    bbox = BoundingBox(l=10.0, t=634.032, r=20.0, b=582.624, coord_origin=CoordOrigin.TOPLEFT)
    doc = SimpleNamespace(
        texts=[SimpleNamespace(prov=[SimpleNamespace(bbox=bbox)])],
        pages={},
    )

    repaired = _normalize_document_bboxes(doc)

    assert repaired == 1
    fixed = doc.texts[0].prov[0].bbox
    assert fixed.as_tuple() == (10.0, 582.624, 20.0, 634.032)
    assert fixed.coord_origin == CoordOrigin.TOPLEFT


def test_normalize_document_bboxes_repairs_inverted_bottom_left_bbox() -> None:
    bbox = BoundingBox(l=10.0, t=582.624, r=20.0, b=634.032, coord_origin=CoordOrigin.BOTTOMLEFT)
    doc = SimpleNamespace(
        texts=[SimpleNamespace(prov=[SimpleNamespace(bbox=bbox)])],
        pages={},
    )

    repaired = _normalize_document_bboxes(doc)

    assert repaired == 1
    fixed = doc.texts[0].prov[0].bbox
    assert fixed.as_tuple() == (10.0, 582.624, 20.0, 634.032)
    assert fixed.coord_origin == CoordOrigin.BOTTOMLEFT

