"""Local Docling MLX runtime helpers for experimental PDF conversion."""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.app.core.doctags import render_doctags_pages, render_doctags_text

DEFAULT_VLM_PRESET = "granite_docling"


class DoclingLocalError(RuntimeError):
    """Raised when the local Docling runtime is unavailable or conversion fails."""


@dataclass(frozen=True)
class DoclingConversionResult:
    doctags_content: str
    text_content: str
    json_content: Mapping[str, Any] | None
    filename: str
    status: str
    processing_time: float | None
    page_count: int | None = None


def assert_local_docling_runtime() -> None:
    """Raise a user-facing error when the local MLX runtime is not usable."""

    if sys.platform != "darwin":
        raise DoclingLocalError(
            "This experimental branch only supports local Docling conversion on macOS."
        )
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        raise DoclingLocalError(
            "This experimental branch requires Apple Silicon for the local MLX runtime."
        )

    missing: list[str] = []
    for module_name in ("docling", "mlx"):
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise DoclingLocalError(
            f"Local Docling MLX dependencies are not installed: {joined}. "
            "Install the Docling VLM and MLX packages for this branch."
        )


def convert_pdf_to_doctags(
    *,
    source_path: Path,
    vlm_preset: str = DEFAULT_VLM_PRESET,
) -> DoclingConversionResult:
    """Convert one PDF to DocTags using the local Docling MLX runtime."""

    if source_path.suffix.lower() != ".pdf":
        raise DoclingLocalError(
            "This experimental branch currently supports PDF conversion only."
        )

    assert_local_docling_runtime()

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
        from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except Exception as exc:  # pragma: no cover - exercised via runtime guard tests
        raise DoclingLocalError(
            "Unable to import Docling MLX conversion modules. "
            "Install the Docling VLM runtime for this branch."
        ) from exc

    preset = (vlm_preset or DEFAULT_VLM_PRESET).strip() or DEFAULT_VLM_PRESET
    started = time.perf_counter()
    try:
        vlm_options = VlmConvertOptions.from_preset(
            preset,
            engine_options=MlxVlmEngineOptions(),
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
                ),
            }
        )
        conversion = converter.convert(source=str(source_path))
        document = conversion.document
        doctags_content = document.export_to_doctags()
    except Exception as exc:
        raise DoclingLocalError(f"Local Docling conversion failed for {source_path.name}: {exc}") from exc

    doctags_content = doctags_content.strip()
    if not doctags_content:
        raise DoclingLocalError(
            f"Local Docling conversion returned no DocTags content for {source_path.name}."
        )

    page_count = _doc_page_count(document)
    if not page_count:
        page_count = len(render_doctags_pages(doctags_content)) or None

    return DoclingConversionResult(
        doctags_content=doctags_content,
        text_content=render_doctags_text(doctags_content),
        json_content=None,
        filename=source_path.name,
        status="success",
        processing_time=round(time.perf_counter() - started, 4),
        page_count=page_count,
    )


def _doc_page_count(document: object) -> int | None:
    pages = getattr(document, "pages", None)
    if pages is None:
        return None
    try:
        return len(pages)
    except Exception:
        return None


__all__ = [
    "DEFAULT_VLM_PRESET",
    "DoclingConversionResult",
    "DoclingLocalError",
    "assert_local_docling_runtime",
    "convert_pdf_to_doctags",
]
