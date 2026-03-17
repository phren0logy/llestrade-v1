"""View helpers for bulk-analysis controller rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Sequence, Set

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidgetItem,
    QWidget,
)

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.file_tracker import WorkspaceGroupMetrics
from src.app.core.placeholders.analyzer import PlaceholderAnalysis
from src.app.workers.progress import WorkerProgressDetail


def build_status_text(
    *,
    group: BulkAnalysisGroup,
    metrics: WorkspaceGroupMetrics | None,
    running_groups: Set[str],
    cancelling_groups: Set[str],
    progress_map: Dict[str, tuple[int, int]],
    progress_details: Dict[str, WorkerProgressDetail],
) -> str:
    gid = group.group_id
    op_type = getattr(metrics, "operation", "per_document") if metrics else group.operation or "per_document"

    if gid in cancelling_groups:
        return "Cancelling…"
    if gid in running_groups:
        completed, total = progress_map.get(gid, (0, 0))
        detail = progress_details.get(gid)
        if detail and detail.chunk_total and detail.chunks_completed is not None:
            chunk_text = f"{detail.chunks_completed}/{detail.chunk_total} chunks"
            if detail.chunks_in_flight:
                chunk_text += f", {detail.chunks_in_flight} in flight"
            if total:
                return f"Running ({completed}/{total}, {chunk_text})"
            return f"Running ({chunk_text})"
        if detail and detail.chunk_total:
            chunk_text = f"chunk {detail.chunk_index or 0}/{detail.chunk_total}"
            if total:
                return f"Running ({completed}/{total}, {chunk_text})"
            return f"Running ({chunk_text})"
        if detail and detail.phase == "combining":
            if total:
                return f"Running ({completed}/{total}, combining)"
            return "Running (combining)"
        if total:
            return f"Running ({completed}/{total})"
        return "Running…"

    if op_type == "combined":
        input_count = getattr(metrics, "combined_input_count", 0) if metrics else 0
        if input_count == 0:
            return "No inputs"
        if metrics and metrics.reduce_corrupt_chunks:
            return f"Corrupt reduce ({metrics.reduce_corrupt_chunks})"
        if metrics and metrics.reduce_resumable_chunks:
            return f"Resumable reduce ({metrics.reduce_resumable_chunks})"
        if metrics and getattr(metrics, "combined_is_stale", False):
            return "Stale"
        return "Ready"

    converted_count = metrics.converted_count if metrics else 0
    if not converted_count:
        return "No converted files"
    if metrics and metrics.map_corrupt_chunks:
        return f"Corrupt chunks ({metrics.map_corrupt_chunks})"
    if metrics and metrics.map_resumable_chunks:
        return f"Resumable ({metrics.map_resumable_chunks} chunks)"
    if metrics and metrics.pending_bulk_analysis:
        return f"Pending bulk ({metrics.pending_bulk_analysis})"
    return "Ready"


def build_placeholder_item(
    analysis: PlaceholderAnalysis | None,
    missing_required: set[str],
    missing_optional: set[str],
) -> QTableWidgetItem:
    item = QTableWidgetItem("—")
    item.setTextAlignment(Qt.AlignCenter)
    if not analysis:
        return item

    tooltip_lines: List[str] = []
    if analysis.used:
        tooltip_lines.append("Used placeholders: " + ", ".join(sorted(analysis.used)))
    unused = analysis.unused - analysis.used
    if unused:
        tooltip_lines.append("Unused placeholders: " + ", ".join(sorted(unused)))

    if missing_required:
        item.setText(f"Missing required ({len(missing_required)})")
        item.setForeground(QColor(178, 34, 34))
        tooltip_lines.append(
            "Missing required: " + ", ".join(sorted(f"{{{key}}}" for key in missing_required))
        )
    elif missing_optional:
        item.setText(f"Missing optional ({len(missing_optional)})")
        item.setForeground(QColor(184, 134, 11))
        tooltip_lines.append(
            "Missing optional: " + ", ".join(sorted(f"{{{key}}}" for key in missing_optional))
        )
    else:
        item.setText("OK")
        item.setForeground(QColor(27, 94, 32))

    if tooltip_lines:
        item.setToolTip("\n".join(tooltip_lines))
    return item


def build_action_widget(
    *,
    group: BulkAnalysisGroup,
    metrics: WorkspaceGroupMetrics | None,
    is_running: bool,
    is_cancelling: bool,
    on_run_map: Callable[[BulkAnalysisGroup, bool], None],
    on_run_combined: Callable[[BulkAnalysisGroup, bool], None],
    on_cancel: Callable[[BulkAnalysisGroup], None],
    on_recover: Callable[[BulkAnalysisGroup], None],
    on_edit: Callable[[BulkAnalysisGroup], None],
    on_open_group_folder: Callable[[BulkAnalysisGroup], None],
    on_review_outputs: Callable[[BulkAnalysisGroup], None],
    on_preview_prompt: Callable[[BulkAnalysisGroup], None],
    on_open_latest_combined: Callable[[BulkAnalysisGroup], None],
    on_delete: Callable[[BulkAnalysisGroup], None],
) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    op_type = getattr(metrics, "operation", "per_document") if metrics else group.operation or "per_document"
    if op_type == "combined":
        input_count = getattr(metrics, "combined_input_count", 0) if metrics else 0

        run_combined = QPushButton("Run Combined")
        run_combined.setEnabled(input_count > 0 and not is_running)
        run_combined.clicked.connect(lambda _, g=group: on_run_combined(g, False))
        layout.addWidget(run_combined)

        run_combined_all = QPushButton("Run Combined All")
        run_combined_all.setEnabled(input_count > 0 and not is_running)
        run_combined_all.clicked.connect(lambda _, g=group: on_run_combined(g, True))
        layout.addWidget(run_combined_all)
    else:
        pending_count = metrics.pending_bulk_analysis if metrics else None
        converted_count = metrics.converted_count if metrics else 0

        run_pending = QPushButton("Run Pending")
        run_pending.setEnabled((pending_count or 0) > 0 and not is_running)
        run_pending.clicked.connect(lambda _, g=group: on_run_map(g, False))
        layout.addWidget(run_pending)

        run_all = QPushButton("Run All")
        run_all.setEnabled(converted_count > 0 and not is_running)
        run_all.clicked.connect(lambda _, g=group: on_run_map(g, True))
        layout.addWidget(run_all)

    cancel_button = QPushButton("Cancel")
    cancel_button.setEnabled(is_running)
    cancel_button.clicked.connect(lambda _, g=group: on_cancel(g))
    layout.addWidget(cancel_button)

    recover_button = QPushButton("Recover…")
    recover_button.setEnabled(not is_running and not is_cancelling)
    recover_button.clicked.connect(lambda _, g=group: on_recover(g))
    layout.addWidget(recover_button)

    edit_button = QPushButton("Edit…")
    edit_button.setEnabled(not is_running and not is_cancelling)
    edit_button.clicked.connect(lambda _, g=group: on_edit(g))
    layout.addWidget(edit_button)

    open_button = QPushButton("Open Folder")
    open_button.clicked.connect(lambda _, g=group: on_open_group_folder(g))
    layout.addWidget(open_button)

    if op_type != "combined":
        review_button = QPushButton("Review Outputs…")
        review_button.setEnabled(not is_running and not is_cancelling)
        review_button.clicked.connect(lambda _, g=group: on_review_outputs(g))
        layout.addWidget(review_button)

    prompt_button = QPushButton("Preview Prompt")
    prompt_button.clicked.connect(lambda _, g=group: on_preview_prompt(g))
    layout.addWidget(prompt_button)

    combined_button = QPushButton("Open Combined Output")
    combined_button.clicked.connect(lambda _, g=group: on_open_latest_combined(g))
    layout.addWidget(combined_button)

    delete_button = QPushButton("Delete")
    delete_button.clicked.connect(lambda _, g=group: on_delete(g))
    layout.addWidget(delete_button)

    layout.addStretch()
    return widget


def append_log_message(log_text, message: str, *, timestamp: bool = True) -> None:
    formatted = message
    if timestamp:
        formatted = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    cursor = log_text.textCursor()
    cursor.movePosition(QTextCursor.End)
    cursor.insertText(formatted + "\n")
    log_text.setTextCursor(cursor)
