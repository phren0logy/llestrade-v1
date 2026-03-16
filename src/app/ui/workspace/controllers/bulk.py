"""Business-logic controller for the bulk analysis tab."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set, TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
    QTableWidgetItem,
    QTreeWidgetItem,
    QWidget,
)

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.bulk_analysis_runner import load_prompts
from src.app.core.file_tracker import WorkspaceGroupMetrics, WorkspaceMetrics
from src.app.core.bulk_recovery import BulkRecoveryStore
from src.app.core.job_cost_estimates import (
    CostForecast,
    estimate_bulk_map_cost,
    estimate_bulk_reduce_cost,
    format_forecast_confirmation,
    format_forecast_inline,
)
from src.app.core.prompt_placeholders import get_prompt_spec
from src.app.ui.workspace.bulk_tab import BulkAnalysisTab
from src.app.ui.workspace.services import BulkAnalysisService
from src.app.ui.dialogs.bulk_recovery_dialog import BulkRecoveryDialog, RecoveryAction
from src.app.core.placeholders.analyzer import PlaceholderAnalysis
from src.app.workers.progress import WorkerProgressDetail
from .bulk_placeholders import analyse_group_placeholders
from .bulk_view import (
    append_log_message as append_log_to_widget,
    build_action_widget,
    build_placeholder_item,
    build_status_text,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.app.core.project_manager import ProjectManager

LOGGER = logging.getLogger(__name__)


class BulkAnalysisController:
    """Render and co-ordinate bulk analysis group state."""

    def __init__(
        self,
        tab: BulkAnalysisTab,
        *,
        workspace: QWidget,
        service: BulkAnalysisService,
        on_create_group: Callable[[], None],
        on_refresh_requested: Callable[[], None],
        on_refresh_groups: Callable[[], None],
        on_refresh_metrics: Callable[[], None],
        on_edit_group: Callable[[BulkAnalysisGroup], None],
        on_open_group_folder: Callable[[BulkAnalysisGroup], None],
        on_show_prompt_preview: Callable[[BulkAnalysisGroup], None],
        on_open_latest_combined: Callable[[BulkAnalysisGroup], None],
        on_delete_group: Callable[[BulkAnalysisGroup], None],
    ) -> None:
        self._tab = tab
        self._workspace = workspace
        self._service = service

        self._project_manager: Optional["ProjectManager"] = None
        self._latest_metrics: WorkspaceMetrics | None = None
        self._feature_enabled = True

        self._on_create_group = on_create_group
        self._on_refresh_requested = on_refresh_requested
        self._on_refresh_groups = on_refresh_groups
        self._on_refresh_metrics = on_refresh_metrics
        self._on_edit_group = on_edit_group
        self._on_open_group_folder = on_open_group_folder
        self._on_show_prompt_preview = on_show_prompt_preview
        self._on_open_latest_combined = on_open_latest_combined
        self._on_delete_group = on_delete_group

        self._info_message: str = "No bulk analysis groups yet."
        self._running_groups: Set[str] = set()
        self._cancelling_groups: Set[str] = set()
        self._progress_map: Dict[str, tuple[int, int]] = {}
        self._progress_details: Dict[str, WorkerProgressDetail] = {}
        self._active_progress_group_id: str | None = None
        self._failures: Dict[str, List[str]] = {}
        self._estimate_cache: Dict[str, CostForecast] = {}

        self._tab.create_button.clicked.connect(self._on_create_group)
        self._tab.refresh_button.clicked.connect(self._on_refresh_requested)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def set_feature_enabled(self, enabled: bool) -> None:
        self._feature_enabled = enabled
        self._tab.setEnabled(enabled)

    def set_project(self, project_manager: Optional["ProjectManager"]) -> None:
        self._project_manager = project_manager
        self._running_groups.clear()
        self._cancelling_groups.clear()
        self._progress_map.clear()
        self._progress_details.clear()
        self._active_progress_group_id = None
        self._failures.clear()
        self._estimate_cache.clear()
        self._latest_metrics = None
        self._tab.table.setRowCount(0)
        self._tab.empty_label.show()
        self._tab.log_text.clear()
        self._info_message = "No bulk analysis groups yet."
        self._tab.info_label.setText(self._info_message)
        self._tab.group_tree.clear()
        self._reset_active_progress_widget()

    @property
    def tab(self) -> BulkAnalysisTab:
        """Expose the underlying tab widget for testing or orchestration."""
        return self._tab

    @property
    def running_groups(self) -> Set[str]:
        return set(self._running_groups)

    def is_running(self, group_id: str) -> bool:
        return group_id in self._running_groups

    def progress_for(self, group_id: str) -> Optional[tuple[int, int]]:
        return self._progress_map.get(group_id)

    def is_cancelling(self, group_id: str) -> bool:
        return group_id in self._cancelling_groups

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def refresh(
        self,
        *,
        groups: Sequence[BulkAnalysisGroup],
        workspace_metrics: WorkspaceMetrics | None,
    ) -> None:
        if not self._feature_enabled:
            return

        self._latest_metrics = workspace_metrics

        if not groups:
            self._tab.table.setRowCount(0)
            self._tab.empty_label.show()
        else:
            self._tab.empty_label.hide()
            self._tab.table.setRowCount(len(groups))

        total_docs = 0
        group_metrics: Dict[str, WorkspaceGroupMetrics] = {}
        if workspace_metrics:
            total_docs = workspace_metrics.dashboard.imported_total
            group_metrics = workspace_metrics.groups

        known_ids = {group.group_id for group in groups}
        self._prune_stale_states(known_ids)

        for row, group in enumerate(groups):
            metrics = group_metrics.get(group.group_id)
            self._populate_row(
                row=row,
                group=group,
                total_docs=total_docs,
                metrics=metrics,
            )

        if not groups:
            self._info_message = "No bulk analysis groups yet."
            self._tab.info_label.setText(self._info_message)
        elif self._running_groups or self._cancelling_groups:
            self._tab.info_label.setText(self._info_message)
        else:
            self._info_message = f"{len(groups)} bulk analysis group(s)"
            self._tab.info_label.setText(self._info_message)

        self._refresh_group_tree(groups)

    def _populate_row(
        self,
        *,
        row: int,
        group: BulkAnalysisGroup,
        total_docs: int,
        metrics: WorkspaceGroupMetrics | None,
    ) -> None:
        table = self._tab.table
        description = group.description or ""

        name_item = QTableWidgetItem(group.name)
        name_item.setData(Qt.UserRole, group.group_id)
        table.setItem(row, 0, name_item)

        op_type = getattr(metrics, "operation", "per_document") if metrics else group.operation or "per_document"
        converted_count = metrics.converted_count if metrics else 0
        if op_type == "combined":
            input_count = getattr(metrics, "combined_input_count", 0) if metrics else 0
            last_run_count = getattr(metrics, "combined_last_run_input_count", None) if metrics else None
            coverage_text = f"Combined — Inputs: {input_count}"
            if isinstance(last_run_count, int):
                coverage_text += f" (last run: {last_run_count})"
        else:
            coverage_text = f"{converted_count} of {total_docs}" if total_docs else str(converted_count)
        coverage_item = QTableWidgetItem(coverage_text)
        coverage_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 1, coverage_item)

        updated_text = group.updated_at.strftime("%Y-%m-%d %H:%M")
        updated_item = QTableWidgetItem(updated_text)
        updated_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 2, updated_item)

        status_item = QTableWidgetItem(self._status_text(group, metrics))
        status_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 3, status_item)

        estimate_item = QTableWidgetItem(self._estimate_text(group))
        estimate_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 4, estimate_item)

        analysis, missing_required, missing_optional = self._analyse_placeholders(group)
        placeholder_item = build_placeholder_item(
            analysis,
            missing_required,
            missing_optional,
        )
        table.setItem(row, 5, placeholder_item)

        action_widget = self._build_action_widget(group, metrics)
        table.setCellWidget(row, 6, action_widget)

        tooltip_lines: List[str] = []
        if description:
            tooltip_lines.append(description)
        if group.directories:
            tooltip_lines.append("Directories: " + ", ".join(group.directories))
        extra_files = sorted(set(group.files))
        if extra_files:
            tooltip_lines.append("Files: " + ", ".join(extra_files))
        if metrics and metrics.converted_files:
            tooltip_lines.append(
                "Converted files (" + str(metrics.converted_count) + "): " + ", ".join(metrics.converted_files)
            )
        if op_type == "combined" and metrics:
            if metrics.combined_latest_path:
                tooltip_lines.append("Latest combined: " + metrics.combined_latest_path)
            if metrics.combined_last_run_input_count is not None:
                tooltip_lines.append("Last run input count: " + str(metrics.combined_last_run_input_count))
        if tooltip_lines:
            name_item.setToolTip("\n".join(tooltip_lines))

    def _status_text(
        self,
        group: BulkAnalysisGroup,
        metrics: WorkspaceGroupMetrics | None,
    ) -> str:
        return build_status_text(
            group=group,
            metrics=metrics,
            running_groups=self._running_groups,
            cancelling_groups=self._cancelling_groups,
            progress_map=self._progress_map,
            progress_details=self._progress_details,
        )

    def _build_action_widget(
        self,
        group: BulkAnalysisGroup,
        metrics: WorkspaceGroupMetrics | None,
    ) -> QWidget:
        return build_action_widget(
            group=group,
            metrics=metrics,
            is_running=group.group_id in self._running_groups,
            is_cancelling=group.group_id in self._cancelling_groups,
            on_run_map=lambda g, force: self.start_map_run(g, force),
            on_run_combined=lambda g, force: self.start_combined_run(g, force),
            on_cancel=self.cancel_run,
            on_recover=self.open_recovery_dialog,
            on_edit=self._on_edit_group,
            on_open_group_folder=self._on_open_group_folder,
            on_preview_prompt=self._on_show_prompt_preview,
            on_open_latest_combined=self._on_open_latest_combined,
            on_delete=self._on_delete_group,
        )

    def _refresh_group_tree(self, groups: Sequence[BulkAnalysisGroup]) -> None:
        tree = self._tab.group_tree
        tree.clear()
        for group in groups:
            parent = QTreeWidgetItem([group.name])
            for directory in sorted(group.directories):
                parent.addChild(QTreeWidgetItem(["Directory", directory]))
            for file_path in sorted(set(group.files)):
                parent.addChild(QTreeWidgetItem(["File", file_path]))
            tree.addTopLevelItem(parent)
        tree.expandAll()

    def _analyse_placeholders(self, group: BulkAnalysisGroup) -> tuple[PlaceholderAnalysis | None, set[str], set[str]]:
        return analyse_group_placeholders(
            self._project_manager,
            group,
            prompt_loader=load_prompts,
            prompt_spec_getter=get_prompt_spec,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def auto_run_pending_groups(self, groups: Sequence[BulkAnalysisGroup]) -> int:
        """Start pending per-document runs without modal prompts.

        Returns the number of groups successfully started.
        """
        started = 0
        for group in groups:
            if (group.operation or "per_document") != "per_document":
                continue
            if group.group_id in self._running_groups or group.group_id in self._cancelling_groups:
                continue
            if self.start_map_run(group, force_rerun=False, interactive=False):
                started += 1
        return started

    def start_map_run(
        self,
        group: BulkAnalysisGroup,
        force_rerun: bool,
        *,
        interactive: bool = True,
        selected_files: Sequence[str] | None = None,
    ) -> bool:
        if not self._feature_enabled:
            return False
        manager = self._project_manager
        if not manager or not manager.project_dir:
            if interactive:
                QMessageBox.warning(self._workspace, "Bulk Analysis", "The project directory is not available.")
            return False

        gid = group.group_id
        if gid in self._running_groups:
            if interactive:
                QMessageBox.information(
                    self._workspace,
                    "Already Running",
                    f"Bulk analysis for '{group.name}' is already in progress.",
                )
            return False

        metrics = self._resolve_group_metrics(gid)
        if not metrics:
            if interactive:
                QMessageBox.warning(
                    self._workspace,
                    "Bulk Analysis",
                    "Project metrics are unavailable. Re-scan sources before running bulk analysis.",
                )
            else:
                self._handle_log(gid, f"Skipping '{group.name}': workspace metrics unavailable.")
            self._on_refresh_groups()
            return False

        if selected_files is not None:
            files = list(selected_files)
            if not files:
                return False
        elif force_rerun:
            files = list(metrics.converted_files)
            if not files:
                if interactive:
                    QMessageBox.warning(
                        self._workspace,
                        "No Converted Documents",
                        "This group does not have any converted documents yet. Run conversion first.",
                    )
                return False
        else:
            pending = list(metrics.pending_files)
            if pending:
                files = pending
            else:
                if interactive:
                    QMessageBox.information(
                        self._workspace,
                        "Up to Date",
                        "All documents already have bulk analysis results. Use 'Run All' to re-process everything.",
                    )
                return False

        analysis, missing_required, missing_optional = self._analyse_placeholders(group)
        if analysis:
            if missing_required or missing_optional:
                if not interactive:
                    missing = sorted(missing_required | missing_optional)
                    missing_text = ", ".join(f"{{{name}}}" for name in missing)
                    self._handle_log(
                        gid,
                        f"Skipping '{group.name}': unresolved placeholders ({missing_text}).",
                    )
                    return False
                messages: list[str] = []
                if missing_required:
                    messages.append(
                        "Required placeholders without values:\n  - "
                        + "\n  - ".join(sorted(f"{{{name}}}" for name in missing_required))
                    )
                if missing_optional:
                    messages.append(
                        "Optional placeholders without values:\n  - "
                        + "\n  - ".join(sorted(f"{{{name}}}" for name in missing_optional))
                    )
                messages.append("Continue with the run?")
                reply = QMessageBox.question(
                    self._workspace,
                    "Placeholder Values Missing",
                    "\n\n".join(messages),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return False

        forecast = self._forecast_map_run(group, files=files, force_rerun=force_rerun)
        if interactive and not self._confirm_run_with_forecast(
            title="Bulk Analysis Estimate",
            group_name=group.name,
            mode_label="all documents" if force_rerun else "pending documents",
            forecast=forecast,
        ):
            return False

        provider_default = (
            (manager.settings or {}).get("llm_provider", ""),
            (manager.settings or {}).get("llm_model", ""),
        )
        if not self._verify_gateway_before_run(
            provider_id=str(group.provider_id or "").strip(),
            model=group.model or provider_default[1],
            title="Bulk Analysis",
            interactive=interactive,
        ):
            return False

        self._running_groups.add(gid)
        self._progress_map[gid] = (0, len(files))
        self._failures[gid] = []
        self._cancelling_groups.discard(gid)

        started = self._service.run_map(
            project_dir=manager.project_dir,
            group=group,
            files=files,
            metadata=manager.metadata,
            default_provider=provider_default,
            force_rerun=force_rerun,
            placeholder_values=manager.project_placeholder_values(),
            project_name=manager.project_name,
            estimate_summary=forecast.to_dict() if forecast.available else None,
            on_progress=self._handle_progress,
            on_progress_detail=self._handle_progress_detail,
            on_failed=self._handle_failed,
            on_log=self._handle_log,
            on_finished=lambda group_id, successes, failures: self._handle_finished(
                group_id,
                successes,
                failures,
                operation="map",
            ),
            on_cost=self._on_run_cost,
        )
        if not started:
            self._running_groups.discard(gid)
            self._progress_map.pop(gid, None)
            self._failures.pop(gid, None)
            if interactive:
                QMessageBox.information(
                    self._workspace,
                    "Already Running",
                    f"Bulk analysis for '{group.name}' is already in progress.",
                )
            return False

        mode_label = "all documents" if force_rerun else "pending documents"
        self._handle_log(gid, f"Starting bulk analysis for '{group.name}' ({len(files)} {mode_label}).")
        self._on_refresh_groups()
        return True

    def start_combined_run(self, group: BulkAnalysisGroup, force_rerun: bool) -> None:
        if not self._feature_enabled:
            return
        manager = self._project_manager
        if not manager or not manager.project_dir:
            QMessageBox.warning(self._workspace, "Bulk Analysis", "The project directory is not available.")
            return

        gid = group.group_id
        if gid in self._running_groups:
            QMessageBox.information(
                self._workspace,
                "Already Running",
                f"Combined operation for '{group.name}' is already in progress.",
            )
            return

        metrics = self._resolve_group_metrics(gid)
        if not metrics:
            QMessageBox.warning(
                self._workspace,
                "Bulk Analysis",
                "Project metrics are unavailable. Refresh groups before running combined analysis.",
            )
            self._on_refresh_groups()
            return

        input_count = getattr(metrics, "combined_input_count", 0)
        if input_count <= 0:
            QMessageBox.warning(
                self._workspace,
                "No Inputs",
                "This combined group does not currently resolve any inputs.",
            )
            return

        if not force_rerun and getattr(metrics, "combined_is_stale", False):
            last_run_count = getattr(metrics, "combined_last_run_input_count", None)
            summary = "Inputs have changed since the latest combined output."
            if isinstance(last_run_count, int):
                summary += (
                    f"\n\nCurrent resolved inputs: {input_count}\n"
                    f"Last run input count: {last_run_count}"
                )
            reply = QMessageBox.question(
                self._workspace,
                "Stale Combined Inputs",
                summary + "\n\nContinue with the run?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if force_rerun:
            confirm = QMessageBox.question(
                self._workspace,
                "Force Combined Re-run",
                (
                    "This will recompute the combined analysis and overwrite the latest output.\n\n"
                    "Do you want to proceed?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return

        analysis, missing_required, missing_optional = self._analyse_placeholders(group)
        if analysis:
            if missing_required or missing_optional:
                messages: list[str] = []
                if missing_required:
                    messages.append(
                        "Required placeholders without values:\n  - "
                        + "\n  - ".join(sorted(f"{{{name}}}" for name in missing_required))
                    )
                if missing_optional:
                    messages.append(
                        "Optional placeholders without values:\n  - "
                        + "\n  - ".join(sorted(f"{{{name}}}" for name in missing_optional))
                    )
                messages.append("Continue with the run?")
                reply = QMessageBox.question(
                    self._workspace,
                    "Placeholder Values Missing",
                    "\n\n".join(messages),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        forecast = self._forecast_combined_run(group, force_rerun=force_rerun)
        if not self._confirm_run_with_forecast(
            title="Combined Analysis Estimate",
            group_name=group.name,
            mode_label="force rerun" if force_rerun else "standard run",
            forecast=forecast,
        ):
            return
        if not self._verify_gateway_before_run(
            provider_id=str(group.provider_id or "").strip(),
            model=group.model or None,
            title="Bulk Analysis",
        ):
            return

        self._running_groups.add(gid)
        self._progress_map[gid] = (0, 1)
        self._failures[gid] = []
        self._cancelling_groups.discard(gid)

        started = self._service.run_combined(
            project_dir=manager.project_dir,
            group=group,
            metadata=manager.metadata,
            force_rerun=force_rerun,
            placeholder_values=manager.project_placeholder_values(),
            project_name=manager.project_name,
            estimate_summary=forecast.to_dict() if forecast.available else None,
            on_progress=self._handle_progress,
            on_progress_detail=self._handle_progress_detail,
            on_failed=self._handle_failed,
            on_log=self._handle_log,
            on_finished=lambda group_id, successes, failures: self._handle_finished(
                group_id,
                successes,
                failures,
                operation="combined",
            ),
            on_cost=self._on_run_cost,
        )
        if not started:
            self._running_groups.discard(gid)
            self._progress_map.pop(gid, None)
            self._failures.pop(gid, None)
            QMessageBox.information(
                self._workspace,
                "Already Running",
                f"Combined operation for '{group.name}' is already in progress.",
            )
            return

        mode_label = "force" if force_rerun else "standard"
        self._handle_log(gid, f"Starting combined operation for '{group.name}' ({mode_label}).")
        self._on_refresh_groups()

    def cancel_run(self, group: BulkAnalysisGroup) -> None:
        if not self._feature_enabled:
            return
        gid = group.group_id
        if self._service.cancel(gid):
            self._cancelling_groups.add(gid)
            self.set_info_message("Cancelling bulk analysis…")
        else:
            self._progress_map.pop(gid, None)
            self._failures.pop(gid, None)
            self._cancelling_groups.discard(gid)
            self.set_info_message("Bulk analysis cancelled.")
        self._on_refresh_groups()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
    def _handle_progress(self, group_id: str, completed: int, total: int, relative_path: str) -> None:
        if group_id in self._cancelling_groups:
            return
        self._progress_map[group_id] = (completed, total)
        self._refresh_groups_safely(relative_path)

    def _handle_progress_detail(self, group_id: str, detail: object) -> None:
        if group_id in self._cancelling_groups or not isinstance(detail, WorkerProgressDetail):
            return
        self._progress_details[group_id] = detail
        self._active_progress_group_id = group_id
        self._update_active_progress_widget(group_id, detail)
        self._on_refresh_groups()

    def _handle_failed(self, group_id: str, relative_path: str, error: str) -> None:
        LOGGER.error("Bulk analysis failed for %s: %s", relative_path, error)
        self._failures.setdefault(group_id, []).append(f"{relative_path}: {error}")

    def _handle_log(self, group_id: str, message: str) -> None:
        LOGGER.info("[BulkAnalysis][%s] %s", group_id, message)
        append_log_to_widget(self._tab.log_text, message, timestamp=True)
        self.set_info_message(message)

    def _handle_finished(self, group_id: str, successes: int, failures: int, *, operation: str) -> None:
        self._running_groups.discard(group_id)
        self._progress_map.pop(group_id, None)
        self._progress_details.pop(group_id, None)
        errors = self._failures.pop(group_id, [])

        was_cancelled = group_id in self._cancelling_groups
        if was_cancelled:
            self._cancelling_groups.discard(group_id)
            completion_message = "Bulk analysis cancelled."
        elif failures:
            completion_message = f"Bulk analysis completed with {failures} error(s)."
        else:
            completion_message = "Bulk analysis completed."

        self._handle_log(group_id, completion_message)

        if errors and not was_cancelled:
            QMessageBox.warning(
                self._workspace,
                "Bulk Analysis Issues",
                "Some documents failed during bulk analysis:\n" + "\n".join(errors),
            )

        if self._on_refresh_metrics:
            self._on_refresh_metrics()
        self._refresh_active_progress_group()
        self._on_refresh_groups()

    def _on_run_cost(self, amount: float, provider: str, stage: str) -> None:
        manager = self._project_manager
        if not manager:
            return
        manager.add_cost(amount, provider, stage)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_info_message(self, message: str) -> None:
        self._info_message = message
        self._tab.info_label.setText(message)

    def append_log_message(self, message: str) -> None:
        append_log_to_widget(self._tab.log_text, message, timestamp=False)

    def _refresh_groups_safely(self, relative_path: str) -> None:
        self._tab.info_label.setText(f"Processing… {relative_path}")
        self._on_refresh_groups()

    def _update_active_progress_widget(self, group_id: str, detail: WorkerProgressDetail) -> None:
        group_name = group_id
        manager = self._project_manager
        if manager:
            for group in manager.list_bulk_analysis_groups():
                if group.group_id == group_id:
                    group_name = group.name
                    break
        self._tab.active_progress_group_label.setText(group_name)
        self._tab.active_progress_bar.setValue(detail.percent or 0)
        self._tab.active_progress_status_label.setText(detail.label)
        self._tab.active_progress_detail_label.setText(self._format_progress_detail(detail))
        self._tab.active_progress_widget.show()

    def _format_progress_detail(self, detail: WorkerProgressDetail) -> str:
        parts: list[str] = []
        if detail.document_path:
            parts.append(detail.document_path)
        if detail.chunk_total and detail.chunks_completed is not None:
            chunk_text = f"{detail.chunks_completed}/{detail.chunk_total} chunks"
            if detail.chunks_in_flight:
                chunk_text += f", {detail.chunks_in_flight} in flight"
            parts.append(chunk_text)
        elif detail.chunk_total:
            parts.append(f"Chunk {detail.chunk_index or 0}/{detail.chunk_total}")
        if detail.section_total:
            title = detail.section_title or "Untitled"
            parts.append(f"Section {detail.section_index or 0}/{detail.section_total}: {title}")
        if detail.detail:
            parts.append(detail.detail)
        return " | ".join(parts)

    def _reset_active_progress_widget(self) -> None:
        self._tab.active_progress_group_label.clear()
        self._tab.active_progress_bar.setValue(0)
        self._tab.active_progress_status_label.clear()
        self._tab.active_progress_detail_label.clear()
        self._tab.active_progress_widget.hide()

    def _refresh_active_progress_group(self) -> None:
        active = self._active_progress_group_id
        if active and active in self._running_groups and active in self._progress_details:
            self._update_active_progress_widget(active, self._progress_details[active])
            return
        if self._progress_details:
            next_group_id = next(iter(self._progress_details))
            self._active_progress_group_id = next_group_id
            self._update_active_progress_widget(next_group_id, self._progress_details[next_group_id])
            return
        self._active_progress_group_id = None
        self._reset_active_progress_widget()

    def _resolve_group_metrics(self, group_id: str) -> Optional[WorkspaceGroupMetrics]:
        if self._latest_metrics and group_id in self._latest_metrics.groups:
            return self._latest_metrics.groups[group_id]

        manager = self._project_manager
        if not manager:
            return None
        try:
            metrics = manager.get_workspace_metrics()
        except Exception:
            LOGGER.exception("Failed to refresh workspace metrics for bulk analysis")
            return None
        self._latest_metrics = metrics
        return metrics.groups.get(group_id)

    def _prune_stale_states(self, valid_ids: Set[str]) -> None:
        for state_map in (self._progress_map, self._progress_details, self._failures):
            for gid in list(state_map.keys()):
                if gid not in valid_ids:
                    state_map.pop(gid, None)
        self._running_groups.intersection_update(valid_ids)
        self._cancelling_groups.intersection_update(valid_ids)
        self._estimate_cache = {
            gid: forecast
            for gid, forecast in self._estimate_cache.items()
            if gid in valid_ids
        }
        if self._active_progress_group_id not in self._running_groups:
            self._refresh_active_progress_group()

    def _estimate_text(self, group: BulkAnalysisGroup) -> str:
        forecast = self._estimate_cache.get(group.group_id)
        if forecast is None:
            return "—"
        return format_forecast_inline(forecast)

    def _forecast_map_run(
        self,
        group: BulkAnalysisGroup,
        *,
        files: Sequence[str],
        force_rerun: bool,
    ) -> CostForecast:
        manager = self._project_manager
        if not manager or not manager.project_dir:
            return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Project unavailable")
        forecast = estimate_bulk_map_cost(
            project_dir=manager.project_dir,
            group=group,
            files=files,
            metadata=manager.metadata,
            placeholder_values=manager.project_placeholder_values(),
            project_name=manager.project_name,
            force_rerun=force_rerun,
        )
        self._estimate_cache[group.group_id] = forecast
        return forecast

    def _forecast_combined_run(
        self,
        group: BulkAnalysisGroup,
        *,
        force_rerun: bool,
    ) -> CostForecast:
        manager = self._project_manager
        if not manager or not manager.project_dir:
            return CostForecast(available=False, best_estimate=None, ceiling=None, reason="Project unavailable")
        forecast = estimate_bulk_reduce_cost(
            project_dir=manager.project_dir,
            group=group,
            metadata=manager.metadata,
            placeholder_values=manager.project_placeholder_values(),
            project_name=manager.project_name,
            force_rerun=force_rerun,
        )
        self._estimate_cache[group.group_id] = forecast
        return forecast

    def _confirm_run_with_forecast(
        self,
        *,
        title: str,
        group_name: str,
        mode_label: str,
        forecast: CostForecast,
    ) -> bool:
        message = (
            f"Start '{group_name}' for {mode_label}?\n\n"
            f"{format_forecast_confirmation(forecast)}"
        )
        reply = QMessageBox.question(
            self._workspace,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def _verify_gateway_before_run(
        self,
        *,
        provider_id: str,
        model: str | None,
        title: str,
        interactive: bool = True,
    ) -> bool:
        if not provider_id:
            return True
        result = self._service.verify_gateway_access(provider_id=provider_id, model=model)
        if result.ok:
            return True
        if result.kind == "rate_limited":
            if not interactive:
                retry_after = (
                    f" Retry-After: {result.retry_after_seconds:.1f}s."
                    if result.retry_after_seconds is not None
                    else ""
                )
                target = f"{result.provider_id}/{result.model or '<default>'}"
                route = f" via route '{result.route}'" if result.route else ""
                self._handle_log(
                    "bulk",
                    f"Skipping run for {target}{route}: gateway probe returned HTTP 429.{retry_after}",
                )
                return False
            reply = QMessageBox.question(
                self._workspace,
                title,
                self._gateway_failure_message(result),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return reply == QMessageBox.Yes
        QMessageBox.warning(
            self._workspace,
            title,
            self._gateway_failure_message(result),
        )
        return False

    @staticmethod
    def _gateway_failure_message(result) -> str:  # noqa: ANN001
        model_label = result.model or "<default>"
        target = f"{result.provider_id}/{model_label}"
        route = f"\nRoute: {result.route}" if result.route else ""
        status = f"\nGateway response: HTTP {result.status_code}" if result.status_code else ""
        retry_after = (
            f"\nRetry-After: {result.retry_after_seconds:.1f}s"
            if result.retry_after_seconds is not None
            else ""
        )
        if result.kind in {"auth_invalid", "auth_forbidden"}:
            return (
                "The saved Pydantic AI Gateway app key was rejected by the gateway.\n\n"
                f"Requested selection: {target}{route}{status}\n"
                f"Gateway message: {result.message}\n\n"
                "Update Settings > API Keys > Pydantic AI Gateway App Key, then try again."
            )
        if result.kind == "route_missing":
            return (
                "The configured gateway route/provider mapping is not available for this run.\n\n"
                f"Requested selection: {target}{route}{status}\n"
                f"Gateway message: {result.message}\n\n"
                "Check the selected provider/model and the optional gateway route in Settings."
            )
        if result.kind == "missing_config":
            return (
                "Gateway mode is enabled, but the gateway configuration is incomplete.\n\n"
                f"Requested selection: {target}\n"
                f"Gateway message: {result.message}\n\n"
                "Set the Pydantic AI Gateway App Key in Settings > API Keys before starting the run."
            )
        if result.kind == "rate_limited":
            return (
                "The gateway is temporarily rate limited for this provider, but you can still continue.\n\n"
                f"Requested selection: {target}{route}{status}{retry_after}\n"
                f"Gateway message: {result.message}\n\n"
                "Choose Continue to start the run anyway and let runtime retries/backoff handle the request, "
                "or Cancel to wait and try again later."
            )
        if result.kind in {"timeout", "unreachable", "server_error"}:
            return (
                "The gateway is currently unavailable, so the run was not started.\n\n"
                f"Requested selection: {target}{route}{status}\n"
                f"Gateway message: {result.message}"
            )
        return (
            "The gateway access check failed, so the run was not started.\n\n"
            f"Requested selection: {target}{route}{status}\n"
            f"Gateway message: {result.message}"
        )

    def open_recovery_dialog(self, group: BulkAnalysisGroup) -> None:
        manager = self._project_manager
        if not manager or not manager.project_dir:
            return
        store = BulkRecoveryStore(manager.project_dir / "bulk_analysis" / (getattr(group, "slug", None) or group.folder_name))
        dialog = BulkRecoveryDialog(
            group=group,
            store=store,
            parent=self._workspace,
        )
        result = dialog.exec()
        action = dialog.selected_action
        self._on_refresh_metrics()
        self._on_refresh_groups()
        if result != QDialog.Accepted or action is None:
            return
        if action == RecoveryAction.RESUME:
            if (group.operation or "per_document") == "combined":
                self.start_combined_run(group, False)
            else:
                documents = dialog.selected_documents()
                self.start_map_run(group, False, selected_files=documents or None)
            return
        if action == RecoveryAction.RERUN_SELECTED:
            if (group.operation or "per_document") == "combined":
                self.start_combined_run(group, False)
            else:
                documents = dialog.selected_documents()
                self.start_map_run(group, False, selected_files=documents or None)


__all__ = ["BulkAnalysisController"]
