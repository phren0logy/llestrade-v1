"""File/path/input helpers for ReportsController."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

from PySide6.QtWidgets import QMessageBox

from src.app.core.azure_artifacts import is_azure_raw_artifact
from src.app.core.bulk_paths import iter_map_outputs
from src.app.core.citations import strip_citation_tokens
from src.app.core.report_inputs import (
    REPORT_CATEGORY_BULK_COMBINED,
    REPORT_CATEGORY_BULK_MAP,
    REPORT_CATEGORY_CONVERTED,
    REPORT_CATEGORY_HIGHLIGHT_COLOR,
    REPORT_CATEGORY_HIGHLIGHT_DOCUMENT,
    ReportInputDescriptor,
)
from src.app.core.report_template_sections import load_template_sections
from src.config.prompt_store import get_bundled_dir, get_custom_dir, get_repo_prompts_dir


def optional_path(value: str) -> Optional[Path]:
    value = value.strip()
    return Path(value).expanduser() if value else None


def validate_required_path(value: str, title: str, message: str, workspace) -> Optional[Path]:
    value = value.strip()
    if not value:
        QMessageBox.warning(workspace, title, message)
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        QMessageBox.warning(workspace, title, f"The selected file does not exist:\n{path}")
        return None
    return path


def validate_prompt_path(
    value: str,
    title: str,
    message: str,
    *,
    validator,
    reader,
    workspace,
) -> Optional[Path]:
    path = validate_required_path(value, title, message, workspace)
    if path is None:
        return None
    try:
        content = reader(path)
        validator(content)
    except ValueError as exc:
        QMessageBox.warning(workspace, title, str(exc))
        return None
    except Exception:
        QMessageBox.warning(workspace, title, "Unable to read the selected prompt.")
        return None
    return path


def read_prompt_file(path_str: str, project_dir: Optional[Path]) -> str:
    path_str = (path_str or "").strip()
    if not path_str:
        return ""
    if not project_dir:
        return ""
    path = Path(normalize_prompt_path(path_str)).expanduser()
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                project_dir / path,
                get_custom_dir() / path,
                get_bundled_dir() / path,
                get_repo_prompts_dir() / path,
            ]
        )
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""


def normalize_prompt_path(path_str: str) -> str:
    path_str = (path_str or "").strip()
    if not path_str:
        return ""

    path = Path(path_str).expanduser()
    if not path.is_absolute():
        return path_str

    try:
        relative = path.relative_to(get_repo_prompts_dir())
    except Exception:
        return path_str

    for root in (get_custom_dir(), get_bundled_dir()):
        candidate = root / relative
        if candidate.exists():
            return str(candidate)
    return path_str


def safe_read_text(path: Optional[Path]) -> str:
    if not path:
        return ""
    try:
        return path.expanduser().read_text(encoding="utf-8")
    except Exception:
        return ""


def preview_template_section(template_path: Optional[Path]) -> tuple[str, str]:
    if not template_path:
        return "", ""
    try:
        sections = load_template_sections(template_path)
    except Exception:
        return "", ""
    if not sections:
        return "", ""
    first = sections[0]
    return first.body.strip(), first.title or ""


def preview_additional_documents(
    project_dir: Optional[Path],
    descriptors: Sequence[ReportInputDescriptor],
) -> str:
    if not project_dir:
        return ""
    lines: list[str] = []
    for descriptor in descriptors:
        candidate = (project_dir / descriptor.relative_path).resolve()
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            content = strip_citation_tokens(candidate.read_text(encoding="utf-8")).strip()
        except Exception:
            continue
        header = f"# {descriptor.label} ({descriptor.category})"
        lines.extend(["<!-- preview: report-input -->", header, content, ""])
    return "\n".join(lines).strip()


def resolve_selected_inputs(selected_inputs: set[str]) -> List[tuple[str, str]]:
    selected_pairs: List[tuple[str, str]] = []
    for key in sorted(selected_inputs):
        if ":" not in key:
            continue
        category, relative = key.split(":", 1)
        selected_pairs.append((category, relative))
    return selected_pairs


def collect_report_inputs(project_dir: Optional[Path]) -> List[ReportInputDescriptor]:
    if not project_dir:
        return []
    descriptors: List[ReportInputDescriptor] = []

    def add_descriptor(category: str, absolute: Path, label: str) -> None:
        descriptors.append(
            ReportInputDescriptor(
                category=category,
                relative_path=absolute.relative_to(project_dir).as_posix(),
                label=label,
            )
        )

    converted_root = project_dir / "converted_documents"
    if converted_root.exists():
        for path in sorted(converted_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                if is_azure_raw_artifact(path):
                    continue
                add_descriptor(
                    REPORT_CATEGORY_CONVERTED,
                    path,
                    path.relative_to(converted_root).as_posix(),
                )

    bulk_root = project_dir / "bulk_analysis"
    if bulk_root.exists():
        for slug_dir in sorted(bulk_root.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            for path, rel in sorted(iter_map_outputs(project_dir, slug), key=lambda item: item[1]):
                add_descriptor(
                    REPORT_CATEGORY_BULK_MAP,
                    path,
                    f"{slug}/{rel}",
                )
            reduce_dir = slug_dir / "reduce"
            if reduce_dir.exists():
                for path in sorted(reduce_dir.rglob("*.md")):
                    add_descriptor(
                        REPORT_CATEGORY_BULK_COMBINED,
                        path,
                        f"{slug_dir.name}/reduce/{path.relative_to(reduce_dir).as_posix()}",
                    )

    highlight_docs = project_dir / "highlights" / "documents"
    if highlight_docs.exists():
        for path in sorted(highlight_docs.rglob("*.md")):
            add_descriptor(
                REPORT_CATEGORY_HIGHLIGHT_DOCUMENT,
                path,
                path.relative_to(highlight_docs).as_posix(),
            )

    highlight_colors = project_dir / "highlights" / "colors"
    if highlight_colors.exists():
        for path in sorted(highlight_colors.glob("*.md")):
            add_descriptor(
                REPORT_CATEGORY_HIGHLIGHT_COLOR,
                path,
                path.relative_to(highlight_colors).as_posix(),
            )

    return descriptors


def default_prompt_path(filename: str) -> str:
    for candidate in (
        get_bundled_dir() / "reports" / filename,
        get_repo_prompts_dir() / "reports" / filename,
    ):
        if candidate.exists():
            return str(candidate)
    return ""


def safe_initial_path(provider: Callable[[], Path], fallback: Path) -> Path:
    try:
        return Path(provider())
    except Exception:
        return fallback
