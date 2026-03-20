"""Dashboard workspace for the new UI."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Sequence, Set

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QColor, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QDialog,
)
from PySide6.QtWidgets import QHeaderView

from src.app.core.conversion_manager import ConversionJob
from src.app.core.feature_flags import FeatureFlags
from src.app.core.file_tracker import WorkspaceMetrics
from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.core.secure_settings import SecureSettings
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.ui.dialogs.project_settings_dialog import ProjectSettingsDialog
from src.app.ui.dialogs.bulk_analysis_group_dialog import BulkAnalysisGroupDialog
from src.app.ui.dialogs.prompt_preview_dialog import PromptPreviewDialog
from src.app.ui.workspace import BulkAnalysisTab, DocumentsTab, HighlightsTab, ReportsTab
from src.app.ui.workspace.controllers import (
    BulkAnalysisController,
    DocumentsController,
    HighlightsController,
    ReportsController,
)
from src.app.ui.workspace.qt_flags import (
    ITEM_IS_ENABLED,
    ITEM_IS_TRISTATE,
    ITEM_IS_USER_CHECKABLE,
)
from src.app.ui.workspace.services import (
    BulkAnalysisService,
    ConversionService,
    HighlightsService,
    ReportsService,
)
from src.app.workers import WorkerCoordinator, get_worker_pool
from src.app.workers.highlight_worker import HighlightExtractionSummary
from src.app.workers.llm_backend import PydanticAIDirectBackend, PydanticAIGatewayBackend
from src.app.workers.llm_backend import backend_transport_name
from src.app.core.prompt_preview import generate_prompt_preview, PromptPreviewError
from src.app.ui.widgets import BannerAction, SmartBanner

LOGGER = logging.getLogger(__name__)


class ProjectWorkspace(QWidget):
    """Dashboard workspace showing documents and bulk analysis groups."""

    home_requested = Signal()

    def __init__(
        self,
        project_manager: Optional[ProjectManager] = None,
        parent: Optional[QWidget] = None,
        *,
        feature_flags: Optional[FeatureFlags] = None,
    ) -> None:
        super().__init__(parent)
        self._feature_flags = feature_flags or FeatureFlags()
        self._project_manager: Optional[ProjectManager] = None
        self._project_path_label = QLabel()
        self._metadata_label: QLabel | None = None
        self._edit_metadata_button: QPushButton | None = None
        self._home_button: QPushButton | None = None
        self._workspace_metrics: WorkspaceMetrics | None = None
        self._thread_pool = get_worker_pool()
        self._workers = WorkerCoordinator(self._thread_pool)
        self._inflight_sources: set[Path] = set()
        self._conversion_running = False
        self._conversion_total = 0
        self._conversion_fatal_error: str | None = None
        self._bulk_analysis_tab: BulkAnalysisTab | None = None
        self._bulk_controller: BulkAnalysisController | None = None
        self._bulk_banner: SmartBanner | None = None
        self._highlights_banner: SmartBanner | None = None
        self._highlights_tab: HighlightsTab | None = None
        self._highlights_controller: HighlightsController | None = None
        self._reports_tab: ReportsTab | None = None
        self._reports_controller: ReportsController | None = None

        self._documents_controller: DocumentsController | None = None
        self._conversion_service = ConversionService(self._workers, max_workers=1)
        self._highlight_service = HighlightsService(self._workers)
        llm_backend = self._build_llm_backend()
        self._llm_transport = backend_transport_name(llm_backend)
        self._bulk_service = BulkAnalysisService(self._workers, llm_backend=llm_backend)
        self._reports_service = ReportsService(self._workers, llm_backend=llm_backend)
        self._build_ui()
        if project_manager:
            self.set_project(project_manager)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)

        self._home_button = QPushButton("home")
        self._home_button.setCursor(Qt.PointingHandCursor)
        self._home_button.setFlat(True)
        self._home_button.clicked.connect(self.home_requested.emit)
        top_bar.addWidget(self._home_button)

        self._project_path_label.setStyleSheet("font-weight: 600;")
        top_bar.addWidget(self._project_path_label)
        top_bar.addStretch()

        layout.addLayout(top_bar)

        self._tabs = QTabWidget()
        self._documents_tab = self._build_documents_tab()
        self._tabs.addTab(self._documents_tab, "Documents")
        self._highlights_tab = self._build_highlights_tab()
        self._tabs.addTab(self._highlights_tab, "Highlights")
        if self._feature_flags.bulk_analysis_groups_enabled:
            self._bulk_analysis_tab = self._build_bulk_analysis_tab()
            self._tabs.addTab(self._bulk_analysis_tab, "Bulk Analysis")
        self._reports_tab = self._build_reports_tab()
        self._tabs.addTab(self._reports_tab, "Reports")
        metadata_row = QHBoxLayout()
        metadata_row.setContentsMargins(0, 0, 0, 0)
        self._metadata_label = QLabel("Subject: — | DOB: —")
        self._metadata_label.setStyleSheet("color: #555;")
        metadata_row.addWidget(self._metadata_label)
        metadata_row.addStretch()
        self._edit_metadata_button = QPushButton("Edit Project Info…")
        self._edit_metadata_button.setEnabled(False)
        self._edit_metadata_button.clicked.connect(self._edit_project_metadata)
        metadata_row.addWidget(self._edit_metadata_button)
        layout.addLayout(metadata_row)
        layout.addWidget(self._tabs)

    def _build_documents_tab(self) -> QWidget:
        tab = DocumentsTab(parent=self)
        self._documents_controller = DocumentsController(self, tab, self._start_conversion)
        self._source_root_label = tab.source_root_label
        self._counts_label = tab.counts_label
        self._last_scan_label = tab.last_scan_label
        self._rescan_button = tab.rescan_button
        self._source_tree = tab.source_tree
        self._root_warning_label = tab.root_warning_label
        self._highlights_banner = tab.highlights_banner
        self._bulk_banner = tab.bulk_banner

        self._rescan_button.clicked.connect(
            lambda: self._trigger_conversion(auto_run=False, show_no_new_notice=True)
        )
        self._source_tree.itemChanged.connect(self._on_source_item_changed)
        return tab

    def _build_highlights_tab(self) -> HighlightsTab:
        tab = HighlightsTab(parent=self)
        self._highlights_controller = HighlightsController(
            self,
            tab,
            on_extract_requested=self._handle_highlight_extract_requested,
            counts_label=self._counts_label,
        )
        return tab

    def _build_reports_tab(self) -> ReportsTab:
        tab = ReportsTab(parent=self, llm_transport=self._llm_transport)
        self._reports_controller = ReportsController(self, tab, service=self._reports_service)
        return tab

    def _build_bulk_analysis_tab(self) -> QWidget:
        tab = BulkAnalysisTab(parent=self)
        self._bulk_analysis_tab = tab
        self._bulk_controller = BulkAnalysisController(
            tab,
            workspace=self,
            service=self._bulk_service,
            on_create_group=self._show_create_group_dialog,
            on_refresh_requested=self.refresh,
            on_refresh_groups=self._refresh_bulk_analysis_groups,
            on_refresh_metrics=self._refresh_file_tracker,
            on_edit_group=self._show_edit_group_dialog,
            on_open_group_folder=self._open_group_folder,
            on_show_prompt_preview=self._show_group_prompt_preview,
            on_open_latest_combined=self._open_latest_combined,
            on_delete_group=self._confirm_delete_group,
        )
        return tab

    def set_project(self, project_manager: ProjectManager) -> None:
        """Attach the workspace to a project manager."""
        had_prior_scan = bool(project_manager.source_state.last_scan)
        self._project_manager = project_manager
        self._documents_controller.set_project(project_manager)
        if self._bulk_controller:
            self._bulk_controller.set_project(project_manager)
        if self._highlights_controller:
            self._highlights_controller.set_project(project_manager)
            self._highlights_controller.set_conversion_running(self._conversion_running)
        if self._reports_controller:
            self._reports_controller.set_project(project_manager)
        self._workers.clear()
        self._workspace_metrics = None
        project_dir = project_manager.project_dir
        project_path = Path(project_dir).resolve() if project_dir else None
        self._project_path_label.setText(
            f"Active project: {project_path}" if project_path else "Active project: (unsaved)"
        )
        if self._edit_metadata_button:
            self._edit_metadata_button.setEnabled(True)
        self._update_metadata_label()
        self.refresh()
        self._maybe_focus_bulk_tab_on_open()
        if self._project_manager and had_prior_scan:
            QTimer.singleShot(
                0,
                lambda: self._trigger_conversion(auto_run=False, show_no_new_notice=False),
            )

    def project_manager(self) -> Optional[ProjectManager]:
        return self._project_manager

    @property
    def bulk_controller(self) -> BulkAnalysisController | None:
        return self._bulk_controller

    @property
    def bulk_tab(self) -> BulkAnalysisTab | None:
        return self._bulk_analysis_tab

    @property
    def reports_controller(self) -> ReportsController | None:
        return self._reports_controller

    def _build_llm_backend(self):
        transport_mode = str(SecureSettings().get("llm_transport_mode", "direct") or "direct").strip().lower()
        if transport_mode == "gateway" and self._feature_flags.pydantic_ai_gateway_enabled:
            return PydanticAIGatewayBackend()
        return PydanticAIDirectBackend()

    def active_job_labels(self) -> list[str]:
        labels: list[str] = []
        if self._conversion_service.is_running():
            labels.append("document conversion")
        if self._highlights_controller and self._highlights_controller.is_running():
            labels.append("highlight extraction")
        if self._bulk_controller and self._bulk_controller.running_groups:
            count = len(self._bulk_controller.running_groups)
            labels.append(f"bulk analysis ({count} run{'s' if count != 1 else ''})")
        if self._reports_controller and self._reports_controller.is_running():
            labels.append("report generation")
        return labels

    def has_active_jobs(self) -> bool:
        return bool(self.active_job_labels())

    def cancel_active_jobs(self) -> None:
        self._conversion_service.cancel()
        self._highlight_service.cancel()
        if self._bulk_controller:
            self._bulk_controller.cancel_all_runs()
        if self._reports_controller:
            self._reports_controller.cancel()

    def wait_for_jobs_done(self, timeout_ms: int) -> bool:
        return self._workers.wait_for_done(timeout_ms)

    def shutdown(self) -> None:
        """Cancel background work before disposing of the workspace."""

        self.cancel_active_jobs()
        self._workers.clear()
        self._documents_controller.shutdown()
        if self._bulk_controller:
            self._bulk_controller.set_project(None)
        if self._highlights_controller:
            self._highlights_controller.set_project(None)
        if self._reports_controller:
            self._reports_controller.shutdown()

    def refresh(self) -> None:
        if self._documents_controller:
            metrics = self._documents_controller.refresh()
            if metrics is not None:
                self._workspace_metrics = metrics
        else:
            self._populate_source_tree()
            self._update_source_root_label()
            self._refresh_file_tracker()
        self._refresh_reports_view()
        self._refresh_highlights_view()
        if self._feature_flags.bulk_analysis_groups_enabled:
            self._refresh_bulk_analysis_groups()
        self._update_metadata_label()

    def _update_metadata_label(self) -> None:
        if not self._metadata_label:
            return
        metadata = self._project_manager.metadata if self._project_manager else None
        if not metadata:
            self._metadata_label.setText("Subject: — | DOB: —")
            return

        subject = metadata.subject_name.strip() if metadata.subject_name else "—"
        dob = metadata.date_of_birth.strip() if metadata.date_of_birth else "—"
        parts = [f"Subject: {subject}", f"DOB: {dob}"]
        if metadata.case_description:
            first_line = metadata.case_description.strip().splitlines()[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "…"
            parts.append(f"Case info: {first_line}")
        self._metadata_label.setText(" | ".join(parts))

    def _edit_project_metadata(self) -> None:
        if not self._project_manager:
            return

        dialog = ProjectSettingsDialog(self._project_manager, self)
        if dialog.exec() == QDialog.Accepted:
            self._update_metadata_label()

    def begin_initial_conversion(self) -> None:
        """Trigger an initial scan/conversion after project creation."""
        if self._feature_flags.auto_run_conversion_on_create:
            self._trigger_conversion(auto_run=True, show_no_new_notice=False)

    def _refresh_file_tracker(self) -> None:
        if self._documents_controller:
            metrics = self._documents_controller.refresh_file_tracker()
            if metrics is not None:
                self._workspace_metrics = metrics
            return
        if not self._project_manager:
            return
        try:
            self._workspace_metrics = self._project_manager.get_workspace_metrics(refresh=True)
        except Exception:
            self._counts_label.setText("Scan failed")
            if self._highlights_banner:
                self._highlights_banner.reset()
            if self._bulk_banner:
                self._bulk_banner.reset()
            return

        metrics = self._workspace_metrics.dashboard

        converted_total = metrics.imported_total
        if converted_total:
            pdf_total = metrics.highlights_total + metrics.pending_highlights
            highlight_text = f"Highlights: {metrics.highlights_total} of {pdf_total}"
            if metrics.pending_highlights:
                highlight_text += f" (pending {metrics.pending_highlights})"
            bulk_text = f"Bulk analysis: {metrics.bulk_analysis_total} of {converted_total}"
            if metrics.pending_bulk_analysis:
                bulk_text += f" (pending {metrics.pending_bulk_analysis})"
            counts_text = (
                f"Converted: {converted_total} | "
                f"{highlight_text} | "
                f"{bulk_text}"
            )
        else:
            counts_text = "Converted: 0 | Highlights: 0 | Bulk analysis: 0"
        self._counts_label.setText(counts_text)

        bulk_missing = list(self._workspace_metrics.bulk_missing)
        highlights_missing = list(self._workspace_metrics.highlights_missing)

        if self._highlights_banner:
            if highlights_missing:
                total = metrics.pending_highlights or len(highlights_missing)
                plural = "s" if total != 1 else ""
                self._highlights_banner.set_role("warning")
                self._highlights_banner.set_message(
                    f"Highlights pending for {total} PDF{plural}.",
                    "Review the queue to see which documents still need highlights.",
                )
                self._highlights_banner.set_actions(
                    [
                        BannerAction(
                            label="Review list",
                            callback=self.show_pending_highlights,
                            is_default=True,
                        )
                    ]
                )
                self._highlights_banner.show()
            else:
                self._highlights_banner.reset()

        if self._bulk_banner:
            if bulk_missing:
                total = metrics.pending_bulk_analysis or len(bulk_missing)
                plural = "s" if total != 1 else ""
                self._bulk_banner.set_role("warning")
                self._bulk_banner.set_message(
                    f"Bulk analysis pending for {total} document{plural}.",
                    "Open the Bulk Analysis tab to create or refresh group runs.",
                )
                self._bulk_banner.set_actions(
                    [
                        BannerAction(
                            label="Open Bulk Analysis",
                            callback=self.show_bulk_analysis_tab,
                            is_default=True,
                        )
                    ]
                )
                self._bulk_banner.show()
            else:
                self._bulk_banner.reset()

        self._update_last_scan_label()

    def _refresh_highlights_view(self) -> None:
        if not self._highlights_controller:
            return
        highlight_state = self._project_manager.highlight_state if self._project_manager else None
        self._highlights_controller.set_conversion_running(self._conversion_running)
        self._highlights_controller.refresh(
            metrics=self._workspace_metrics,
            highlight_state=highlight_state,
        )

    def show_pending_highlights(self) -> None:
        """Switch to the highlights tab and focus the pending list."""
        if not self._highlights_tab:
            return
        index = self._tabs.indexOf(self._highlights_tab)
        if index != -1:
            self._tabs.setCurrentIndex(index)
        if self._highlights_controller:
            self._highlights_controller.show_pending_list()

    def show_bulk_analysis_tab(self) -> None:
        """Switch to the bulk analysis tab if available."""
        if not self._bulk_analysis_tab:
            return
        index = self._tabs.indexOf(self._bulk_analysis_tab)
        if index != -1:
            self._tabs.setCurrentIndex(index)

    def _refresh_reports_view(self) -> None:
        if self._reports_controller:
            self._reports_controller.refresh()

    def _handle_highlight_extract_requested(self) -> None:
        if not self._project_manager or not self._highlights_controller:
            return
        if self._highlight_service.is_running() or self._highlights_controller.is_running():
            QMessageBox.information(
                self,
                "Highlights",
                "Highlight extraction is already in progress.",
            )
            return

        started = self._highlight_service.run(
            project_manager=self._project_manager,
            on_started=self._highlights_controller.begin_extraction,
            on_progress=self._highlights_controller.update_progress,
            on_failed=self._highlights_controller.record_failure,
            on_finished=self._on_highlight_run_finished,
        )
        if not started:
            QMessageBox.information(
                self,
                "Highlights",
                "No converted PDFs are ready for highlight extraction.",
            )

    def _on_highlight_run_finished(
        self,
        summary: HighlightExtractionSummary | None,
        successes: int,
        failures: int,
    ) -> None:
        if self._highlights_controller:
            self._highlights_controller.finish(summary=summary, failures=failures)

        if summary and self._project_manager:
            self._project_manager.record_highlight_run(
                generated_at=summary.generated_at,
                documents_processed=summary.documents_processed,
                documents_with_highlights=summary.documents_with_highlights,
                total_highlights=summary.total_highlights,
                color_files_written=summary.color_files_written,
            )

        self._refresh_file_tracker()
        self._refresh_highlights_view()

    # ------------------------------------------------------------------
    # Source tree helpers
    # ------------------------------------------------------------------
    def _trigger_conversion(
        self,
        auto_run: bool,
        *,
        show_no_new_notice: bool = True,
    ) -> None:
        if self._documents_controller:
            self._documents_controller.trigger_conversion(
                auto_run,
                show_no_new_notice=show_no_new_notice,
            )

    def _handle_rescan_clicked(self) -> None:
        self._trigger_conversion(auto_run=False, show_no_new_notice=True)

    def _maybe_focus_bulk_tab_on_open(self) -> None:
        if not self._bulk_analysis_tab or not self._workspace_metrics:
            return
        if self._workspace_metrics.dashboard.imported_total <= 0:
            return
        index = self._tabs.indexOf(self._bulk_analysis_tab)
        if index != -1:
            self._tabs.setCurrentIndex(index)

    def _select_source_root(self) -> None:
        if not self._project_manager or not self._project_manager.project_dir:
            QMessageBox.warning(self, "Project Required", "Create or open a project first.")
            return
        start_dir = str(self._resolve_source_root() or self._project_manager.project_dir)
        chosen = QFileDialog.getExistingDirectory(self, "Select Source Folder", start_dir)
        if not chosen:
            return
        chosen_path = Path(chosen)
        if self._documents_controller:
            relative = self._documents_controller.to_project_relative(chosen_path)
        else:
            relative = self._to_project_relative(chosen_path)
        self._project_manager.update_source_state(
            root=relative,
            selected_folders=[],
            warnings=[],
        )
        if self._documents_controller:
            self._documents_controller.populate_source_tree()
            self._documents_controller.update_source_root_label()
            self._documents_controller.update_last_scan_label()
            self._documents_controller.set_root_warning([])
        else:
            self._populate_source_tree()
            self._update_source_root_label()
            self._update_last_scan_label()
            self._set_root_warning([])


    def _update_last_scan_label(self) -> None:
        if self._documents_controller:
            self._documents_controller.update_last_scan_label()
            return
        if not self._project_manager:
            self._last_scan_label.setText("")
            return
        metrics = None
        if self._workspace_metrics:
            metrics = self._workspace_metrics.dashboard
        elif self._project_manager.dashboard_metrics:
            metrics = self._project_manager.dashboard_metrics
        last_scan = metrics.last_scan if metrics else None
        if not last_scan and self._project_manager.source_state.last_scan:
            try:
                last_scan = datetime.fromisoformat(self._project_manager.source_state.last_scan)
            except ValueError:
                last_scan = self._project_manager.source_state.last_scan

        if not last_scan:
            self._last_scan_label.setText("Last scan: never")
            return
        if isinstance(last_scan, datetime):
            display = last_scan.strftime("Last scan: %Y-%m-%d %H:%M")
        else:
            display = f"Last scan: {last_scan}"
        self._last_scan_label.setText(display)

    def _populate_source_tree(self) -> None:
        if self._documents_controller:
            self._documents_controller.populate_source_tree()

    def _populate_directory_contents(self, *args, **kwargs) -> None:
        if self._documents_controller:
            self._documents_controller.populate_directory_contents(*args, **kwargs)

    def _iter_directories(self, root_path: Path) -> List[str]:
        if self._documents_controller:
            return self._documents_controller.iter_directories(root_path)
        return []

    def _normalise_relative_path(self, path: str) -> str:
        if self._documents_controller:
            return self._documents_controller.normalise_relative_path(path)
        return Path(path.strip('/')).as_posix() if path else ''

    def _apply_directory_flags(self, item: QTreeWidgetItem) -> None:
        if self._documents_controller:
            self._documents_controller.apply_directory_flags(item)

    def _should_skip_source_entry(self, entry: Path) -> bool:
        if self._documents_controller:
            return self._documents_controller.should_skip_source_entry(entry)
        return False

    def _is_path_tracked(self, relative: str, tracked: Set[str]) -> bool:
        if self._documents_controller:
            return self._documents_controller.is_path_tracked(relative, tracked)
        return False

    def _compute_new_directories(self, actual: Set[str], selected: Set[str], acknowledged: Set[str]) -> List[str]:
        if self._documents_controller:
            return self._documents_controller.compute_new_directories(actual, selected, acknowledged)
        return []

    def _mark_directory_as_new(self, relative: str) -> None:
        if self._documents_controller:
            self._documents_controller.mark_directory_as_new(relative)

    def _clear_new_directory_marker(self, relative: str) -> None:
        if self._documents_controller:
            self._documents_controller.clear_new_directory_marker(relative)

    def _acknowledge_directories(self, directories: Sequence[str]) -> None:
        if self._documents_controller:
            self._documents_controller.acknowledge_directories(directories)

    def _prompt_for_new_directories(self, new_dirs: Sequence[str]) -> None:
        if self._documents_controller:
            self._documents_controller.prompt_for_new_directories(new_dirs)

    def _expand_to_item(self, item: QTreeWidgetItem) -> None:
        if self._documents_controller:
            self._documents_controller.expand_to_item(item)

    def _handle_missing_directories(self, missing_dirs: Sequence[str]) -> None:
        if self._documents_controller:
            self._documents_controller.handle_missing_directories(missing_dirs)

    def _on_source_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._documents_controller:
            self._documents_controller.handle_source_item_changed(item, column)

    def _cascade_source_check_state(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        if self._documents_controller:
            self._documents_controller.cascade_source_check_state(item, state)

    def _update_parent_source_state(self, item: Optional[QTreeWidgetItem]) -> None:
        if self._documents_controller:
            self._documents_controller.update_parent_source_state(item)

    def _update_selected_folders_from_tree(self) -> None:
        if self._documents_controller:
            self._documents_controller.update_selected_folders_from_tree()

    def _collect_selected_directories(self) -> List[str]:
        if self._documents_controller:
            return self._documents_controller.collect_selected_directories()
        return []

    def _collect_selected_directories_from_item(self, item: QTreeWidgetItem, results: Set[str]) -> None:
        if self._documents_controller:
            self._documents_controller.collect_selected_directories_from_item(item, results)

    def _set_root_warning(self, warnings: List[str]) -> None:
        if self._documents_controller:
            self._documents_controller.set_root_warning(warnings)
        elif self._root_warning_label:
            if warnings:
                self._root_warning_label.setText("\n".join(warnings))
                self._root_warning_label.show()
            else:
                self._root_warning_label.clear()
                self._root_warning_label.hide()

    def _compute_root_warnings(self, root_path: Path) -> List[str]:
        if self._documents_controller:
            return self._documents_controller.compute_root_warnings(root_path)
        return []

    def _update_source_root_label(self) -> None:
        if self._documents_controller:
            self._documents_controller.update_source_root_label()
            return
        if not self._project_manager:
            self._source_root_label.setText("Source root: not set")
            return
        root_path = self._resolve_source_root()
        if not root_path or not root_path.exists():
            self._source_root_label.setText("Source root: not set")
        else:
            self._source_root_label.setText(f"Source root: {root_path}")

    def _prompt_reselect_source_root(self) -> None:
        if self._documents_controller:
            self._documents_controller.prompt_reselect_source_root()

    def _schedule_file_tracker_refresh(self) -> None:
        if self._documents_controller:
            self._documents_controller.schedule_file_tracker_refresh()

    def _run_scheduled_file_tracker_refresh(self) -> None:
        if self._documents_controller:
            self._documents_controller.run_scheduled_file_tracker_refresh()

    def _find_child(self, parent: QTreeWidgetItem, name: str) -> Optional[QTreeWidgetItem]:
        for index in range(parent.childCount()):
            child = parent.child(index)
            if child.text(0) == name:
                return child
        return None

    def _resolve_source_root(self) -> Optional[Path]:
        if not self._project_manager or not self._project_manager.project_dir:
            return None
        root_spec = (self._project_manager.source_state.root or "").strip()
        if not root_spec:
            return None
        root_path = Path(root_spec)
        if not root_path.is_absolute():
            root_path = (self._project_manager.project_dir / root_path).resolve()
        return root_path

    def _to_project_relative(self, path: Path) -> str:
        if not self._project_manager or not self._project_manager.project_dir:
            return path.as_posix()
        project_dir = Path(self._project_manager.project_dir).resolve()
        try:
            rel = path.resolve().relative_to(project_dir)
            return rel.as_posix()
        except ValueError:
            rel_str = os.path.relpath(path.resolve(), project_dir)
            return Path(rel_str).as_posix()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _start_conversion(self, jobs: List[ConversionJob]) -> None:
        if not self._project_manager or not jobs:
            return

        self._conversion_running = True
        self._conversion_total = len(jobs)
        self._conversion_errors: List[str] = []
        self._conversion_fatal_error: str | None = None
        self._rescan_button.setEnabled(False)
        self._counts_label.setText(f"Converting documents (0/{self._conversion_total})…")
        if self._highlights_controller:
            self._highlights_controller.set_conversion_running(True)

        helper = self._project_manager.conversion_settings.helper
        options = dict(self._project_manager.conversion_settings.options or {})
        started = self._conversion_service.run(
            jobs=jobs,
            helper=helper,
            options=options,
            on_progress=self._on_conversion_progress,
            on_failed=self._on_conversion_failed,
            on_fatal=self._on_conversion_fatal,
            on_finished=lambda success, failed, js=jobs: self._on_conversion_finished(
                None, js, success, failed
            ),
        )
        if not started:
            self._conversion_running = False
            self._rescan_button.setEnabled(True)
            if self._highlights_controller:
                self._highlights_controller.set_conversion_running(False)
            return

        self._inflight_sources.update(job.source_path for job in jobs)

    def _on_conversion_progress(self, processed: int, total: int, relative_path: str) -> None:
        self._counts_label.setText(f"Converting documents ({processed}/{total})… {relative_path}")

    def _on_conversion_failed(self, source_path: str, error: str) -> None:
        message = f"{Path(source_path).name}: {error}"
        self._conversion_errors.append(message)

    def _on_conversion_fatal(self, error: str) -> None:
        self._conversion_fatal_error = error
        self._counts_label.setText("Stopping conversion due to Azure configuration error…")

    def _on_conversion_finished(
        self,
        worker,
        jobs: Sequence[ConversionJob],
        successes: int,
        failures: int,
    ) -> None:
        for job in jobs:
            self._inflight_sources.discard(job.source_path)

        timestamp = datetime.now(timezone.utc).isoformat()
        warnings = self._documents_controller.current_warnings if self._documents_controller else []
        self._project_manager.update_source_state(last_scan=timestamp, warnings=warnings)
        self._update_last_scan_label()

        self._conversion_running = False
        self._rescan_button.setEnabled(True)
        if self._highlights_controller:
            self._highlights_controller.set_conversion_running(False)
        self._refresh_file_tracker()
        self._refresh_highlights_view()
        if self._feature_flags.bulk_analysis_groups_enabled:
            self._refresh_bulk_analysis_groups()
            self._auto_run_pending_bulk_groups()
        if self._conversion_fatal_error:
            QMessageBox.critical(
                self,
                "Conversion Stopped",
                self._conversion_fatal_error,
            )
        elif failures:
            error_text = "\n".join(self._conversion_errors) or "Unknown errors"
            QMessageBox.warning(
                self,
                "Conversion Issues",
                "Some documents failed to convert:\n\n" + error_text,
            )

    def _auto_run_pending_bulk_groups(self) -> None:
        if not self._bulk_controller or not self._project_manager:
            return
        try:
            groups = self._project_manager.list_bulk_analysis_groups()
        except Exception:
            LOGGER.exception("Failed to list bulk analysis groups for auto-run")
            return
        started = self._bulk_controller.auto_run_pending_groups(groups)
        if started:
            LOGGER.info("Auto-started pending bulk runs for %s group(s)", started)

    # ------------------------------------------------------------------
    # Highlight extraction helpers
    # ------------------------------------------------------------------
    def _refresh_bulk_analysis_groups(self) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        if not self._bulk_controller:
            return

        groups: Sequence[BulkAnalysisGroup] = []
        if self._project_manager:
            try:
                groups = self._project_manager.list_bulk_analysis_groups()
            except Exception:
                groups = []

        self._bulk_controller.refresh(
            groups=groups,
            workspace_metrics=self._workspace_metrics,
        )

    def _show_edit_group_dialog(self, group: BulkAnalysisGroup) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        manager = self._project_manager
        if not manager or not manager.project_dir:
            return
        existing_names = [
            candidate.name
            for candidate in manager.list_bulk_analysis_groups()
            if candidate.group_id != group.group_id
        ]
        dialog = BulkAnalysisGroupDialog(
            manager.project_dir,
            self,
            metadata=manager.metadata,
            placeholder_values=manager.placeholder_mapping(),
            existing_group=group,
            existing_names=existing_names,
            llm_transport=self._llm_transport,
        )
        if dialog.exec() == QDialog.Accepted:
            try:
                updated_group = dialog.build_group()
                manager.save_bulk_analysis_group(updated_group)
            except Exception as exc:
                QMessageBox.critical(self, "Update Bulk Analysis Group Failed", str(exc))
            else:
                self.refresh()

    def _show_create_group_dialog(self) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        if not self._project_manager or not self._project_manager.project_dir:
            return
        existing_names = [group.name for group in self._project_manager.list_bulk_analysis_groups()]
        dialog = BulkAnalysisGroupDialog(
            self._project_manager.project_dir,
            self,
            metadata=self._project_manager.metadata,
            placeholder_values=self._project_manager.placeholder_mapping(),
            existing_names=existing_names,
            llm_transport=self._llm_transport,
        )
        if dialog.exec() == QDialog.Accepted:
            group = dialog.build_group()
            try:
                self._project_manager.save_bulk_analysis_group(group)
            except Exception as exc:
                QMessageBox.critical(self, "Create Bulk Analysis Group Failed", str(exc))
            else:
                self.refresh()

    def _confirm_delete_group(self, group: BulkAnalysisGroup) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        if not self._project_manager:
            return
        message = (
            f"Delete bulk analysis group '{group.name}'?\n\n"
            "All generated bulk analysis outputs stored in this group will be deleted."
        )
        reply = QMessageBox.question(
            self,
            "Delete Bulk Analysis Group",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            if not self._project_manager.delete_bulk_analysis_group(group.group_id):
                QMessageBox.warning(self, "Delete Failed", "Could not delete the bulk analysis group.")
            else:
                if self._bulk_controller:
                    self._bulk_controller.cancel_run(group)
                self.refresh()

    def _open_group_folder(self, group: BulkAnalysisGroup) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        if not self._project_manager or not self._project_manager.project_dir:
            return
        folder = self._project_manager.project_dir / "bulk_analysis" / group.folder_name
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _show_group_prompt_preview(self, group: BulkAnalysisGroup) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        if not self._project_manager or not self._project_manager.project_dir:
            QMessageBox.warning(self, "Prompt Preview", "Open a project to preview prompts.")
            return

        project_dir = Path(self._project_manager.project_dir)
        metadata = self._project_manager.metadata

        try:
            preview = generate_prompt_preview(
                project_dir,
                group,
                metadata=metadata,
                placeholder_values=self._project_manager.placeholder_mapping(),
            )
        except PromptPreviewError as exc:
            QMessageBox.warning(self, "Prompt Preview", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Prompt preview failed: %s", exc)
            QMessageBox.warning(self, "Prompt Preview", "Failed to generate prompt preview.")
            return

        dialog = PromptPreviewDialog(self)
        dialog.set_preview(preview)
        dialog.exec()

    def _open_latest_combined(self, group: BulkAnalysisGroup) -> None:
        if not self._feature_flags.bulk_analysis_groups_enabled:
            return
        if not self._project_manager or not self._project_manager.project_dir:
            return
        folder = self._project_manager.project_dir / "bulk_analysis" / group.folder_name / "reduce"
        if not folder.exists():
            QMessageBox.information(self, "No Outputs", "No combined outputs found for this operation.")
            return
        latest = None
        latest_m = None
        for f in folder.glob("combined_*.md"):
            try:
                m = f.stat().st_mtime
            except OSError:
                continue
            if latest is None or m > (latest_m or 0):
                latest = f
                latest_m = m
        if latest is None:
            QMessageBox.information(self, "No Outputs", "No combined outputs found for this operation.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(latest)))

    # ------------------------------------------------------------------
    # Worker coordination helpers
    # ------------------------------------------------------------------
    def _bulk_key(self, group_id: str) -> str:
        return f"bulk:{group_id}"
