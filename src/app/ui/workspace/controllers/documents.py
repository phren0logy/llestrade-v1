"""Business-logic controller for the workspace Documents tab."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem

from src.app.core.conversion_manager import (
    ConversionJob,
    DuplicateSource,
    build_conversion_jobs,
)
from src.app.core.file_tracker import WorkspaceMetrics
from src.app.core.project_manager import ProjectManager
from src.app.ui.workspace.documents_tab import DocumentsTab
from .documents_tree_state import (
    apply_directory_flags as apply_directory_flags_item,
    cascade_source_check_state as cascade_tree_check_state,
    collect_selected_directories_from_item as collect_checked_directories_from_item,
    is_path_tracked as is_tree_path_tracked,
    normalise_relative_path as normalise_tree_relative_path,
    should_skip_source_entry as should_skip_tree_entry,
    update_parent_source_state as update_parent_tree_state,
)
from .documents_view import (
    set_root_warning as render_root_warning,
    update_bulk_banner as render_bulk_banner,
    update_highlights_banner as render_highlights_banner,
    update_last_scan_label as render_last_scan_label,
    update_source_root_label as render_source_root_label,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.app.ui.stages.project_workspace import ProjectWorkspace


class DocumentsController:
    """Coordinate state and interactions for the documents tab."""

    def __init__(
        self,
        workspace: "ProjectWorkspace",
        tab: DocumentsTab,
        run_conversion: Callable[[List[ConversionJob]], None],
    ) -> None:
        self._workspace = workspace
        self._tab = tab
        self._run_conversion = run_conversion
        self._project_manager: Optional[ProjectManager] = None
        self._source_tree_nodes: Dict[str, QTreeWidgetItem] = {}
        self._block_source_tree_signal = False
        self._new_directory_alerts: Set[str] = set()
        self._new_dir_prompt_active = False
        self._pending_file_tracker_refresh = False
        self._current_warnings: List[str] = []
        self._missing_root_prompted = False
        self._workspace_metrics: WorkspaceMetrics | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def set_project(self, project_manager: Optional[ProjectManager]) -> None:
        self._project_manager = project_manager
        self._workspace_metrics = None
        self._pending_file_tracker_refresh = False
        self._source_tree_nodes.clear()
        self._block_source_tree_signal = False
        self._new_directory_alerts.clear()
        self._new_dir_prompt_active = False
        self._current_warnings = []
        self._missing_root_prompted = False
        self._tab.highlights_banner.reset()
        self._tab.bulk_banner.reset()

        tree = self._tab.source_tree
        tree.clear()
        tree.setDisabled(project_manager is None)

        if project_manager is None:
            self._tab.source_root_label.setText("Source root: not set")
            self._tab.counts_label.setText("Converted: 0 | Highlights: 0 | Bulk analysis: 0")
            self._tab.last_scan_label.setText("")
        self.set_root_warning([])

    def shutdown(self) -> None:
        self._source_tree_nodes.clear()
        self._new_directory_alerts.clear()
        self._pending_file_tracker_refresh = False
        self._workspace_metrics = None

    # ------------------------------------------------------------------
    # Public API used by ProjectWorkspace
    # ------------------------------------------------------------------
    def refresh(self) -> WorkspaceMetrics | None:
        self.populate_source_tree()
        self.update_source_root_label()
        metrics = self.refresh_file_tracker()
        return metrics

    def trigger_conversion(
        self,
        auto_run: bool,
        *,
        show_no_new_notice: bool = True,
    ) -> bool:
        project_manager = self._project_manager
        if not project_manager:
            return False
        workspace = self._workspace
        if workspace._conversion_running:
            QMessageBox.information(
                workspace,
                "Conversion Running",
                "Document conversion is already in progress.",
            )
            return False

        jobs, duplicates = self.collect_conversion_jobs(workspace._inflight_sources)
        if duplicates:
            self.show_duplicate_notice(duplicates)
        if not jobs:
            if show_no_new_notice and not auto_run and not duplicates:
                QMessageBox.information(workspace, "Conversion", "No new files detected.")
            return False

        if not auto_run:
            reply = QMessageBox.question(
                workspace,
                "Convert Documents",
                f"Convert {len(jobs)} new PDF document(s) to DocTags now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return False

        self._run_conversion(jobs)
        return True

    def collect_conversion_jobs(
        self,
        inflight_sources: Set[Path],
    ) -> tuple[list[ConversionJob], Sequence[DuplicateSource]]:
        if not self._project_manager:
            return [], []
        plan = build_conversion_jobs(self._project_manager)
        jobs = [job for job in plan.jobs if job.source_path not in inflight_sources]
        return jobs, plan.duplicates

    def show_duplicate_notice(self, duplicates: Sequence[DuplicateSource]) -> None:
        if not duplicates:
            return
        workspace = self._workspace
        preview_limit = 10
        listed = list(duplicates[:preview_limit])
        lines = [
            f"- {duplicate.duplicate_relative} matches {duplicate.primary_relative}"
            for duplicate in listed
        ]
        remaining = len(duplicates) - len(listed)
        if remaining > 0:
            lines.append(f"…and {remaining} more duplicate files.")

        message = (
            "Duplicate files detected. Matching documents will be skipped to avoid converting"
            " the same content twice.\n\n"
            + "\n".join(lines)
        )
        QMessageBox.warning(workspace, "Duplicate Files Skipped", message)

    def refresh_file_tracker(self) -> WorkspaceMetrics | None:
        project_manager = self._project_manager
        counts_label = self._tab.counts_label
        if not project_manager:
            counts_label.setText("Converted: 0 | Highlights: 0 | Bulk analysis: 0")
            self._tab.highlights_banner.reset()
            self._tab.bulk_banner.reset()
            self._workspace_metrics = None
            return None

        try:
            self._workspace_metrics = project_manager.get_workspace_metrics(refresh=True)
        except Exception:
            counts_label.setText("Scan failed")
            self._tab.highlights_banner.reset()
            self._tab.bulk_banner.reset()
            return None

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
            counts_label.setText(
                f"Converted: {converted_total} | {highlight_text} | {bulk_text}"
            )
        else:
            counts_label.setText("Converted: 0 | Highlights: 0 | Bulk analysis: 0")

        highlights_missing = list(self._workspace_metrics.highlights_missing)
        bulk_missing = list(self._workspace_metrics.bulk_missing)

        self._update_highlights_banner(highlights_missing, metrics.pending_highlights)
        self._update_bulk_banner(bulk_missing, metrics.pending_bulk_analysis)

        self.update_last_scan_label()
        return self._workspace_metrics

    def update_source_root_label(self) -> None:
        root_path = self.resolve_source_root()
        render_source_root_label(self._tab.source_root_label, root_path)

    def update_last_scan_label(self) -> None:
        label = self._tab.last_scan_label
        project_manager = self._project_manager
        metrics = None
        if self._workspace_metrics:
            metrics = self._workspace_metrics.dashboard
        elif project_manager and project_manager.dashboard_metrics:
            metrics = project_manager.dashboard_metrics

        last_scan = metrics.last_scan if metrics else None
        if not last_scan and project_manager and project_manager.source_state.last_scan:
            try:
                last_scan = datetime.fromisoformat(project_manager.source_state.last_scan)
            except ValueError:
                last_scan = project_manager.source_state.last_scan

        render_last_scan_label(label, last_scan)

    def populate_source_tree(self) -> None:
        tree = self._tab.source_tree
        tree.clear()
        self._source_tree_nodes.clear()
        project_manager = self._project_manager

        if not project_manager:
            tree.setDisabled(True)
            self.set_root_warning([])
            return

        root_path = self.resolve_source_root()
        if not root_path or not root_path.exists():
            tree.setDisabled(True)
            warning = [
                "Source folder missing. Update the project location to resume scanning."
                if project_manager.source_state.root
                else "Select a source folder to begin tracking documents."
            ]
            self.set_root_warning(warning)
            project_manager.update_source_state(warnings=warning)
            if not self._missing_root_prompted:
                self._missing_root_prompted = True
                self.prompt_reselect_source_root()
            return

        tree.setDisabled(False)
        self._missing_root_prompted = False

        selected_set = {
            self.normalise_relative_path(path)
            for path in (project_manager.source_state.selected_folders or [])
        }
        acknowledged_set = {
            self.normalise_relative_path(path)
            for path in (project_manager.source_state.acknowledged_folders or [])
        }

        self._current_warnings = self.compute_root_warnings(root_path)
        self.set_root_warning(self._current_warnings)
        project_manager.update_source_state(warnings=self._current_warnings)

        root_label = root_path.name or root_path.as_posix()
        root_item = QTreeWidgetItem([root_label])
        root_item.setData(0, Qt.UserRole, ("root", ""))
        self.apply_directory_flags(root_item)
        root_item.setCheckState(0, Qt.Checked if "" in selected_set else Qt.Unchecked)
        self._source_tree_nodes[""] = root_item
        tree.addTopLevelItem(root_item)

        processed_dirs: Set[Path] = set()
        self._populate_directory_contents(
            parent_item=root_item,
            directory=root_path,
            processed=processed_dirs,
            root_path=root_path,
            selected=selected_set,
        )
        tree.expandItem(root_item)

        all_directories = {self.normalise_relative_path(entry) for entry in self.iter_directories(root_path)}
        known_set = {
            self.normalise_relative_path(entry)
            for entry in (project_manager.source_state.known_folders or [])
        }
        if not acknowledged_set and all_directories:
            acknowledged_set = set(all_directories)
            initial_ack = sorted(acknowledged_set)
            if initial_ack != (project_manager.source_state.acknowledged_folders or []):
                project_manager.update_source_state(acknowledged_folders=initial_ack)

        new_directories = sorted(path for path in all_directories if path and path not in known_set)
        for relative in new_directories:
            self.mark_directory_as_new(relative)
        if new_directories:
            self.prompt_for_new_directories(new_directories)

        removed_directories = {path for path in known_set if path and path not in all_directories}
        selected_missing = {path for path in selected_set if path and path not in all_directories}
        missing_directories = sorted(removed_directories | selected_missing)
        if missing_directories:
            self.handle_missing_directories(missing_directories)

        known_snapshot = sorted(all_directories)
        if known_snapshot != (project_manager.source_state.known_folders or []):
            project_manager.update_source_state(known_folders=known_snapshot)

    def handle_source_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._block_source_tree_signal:
            return

        data = item.data(0, Qt.UserRole)
        if not data:
            return
        node_type, relative = data
        if node_type not in {"dir", "root"}:
            return

        state = item.checkState(0)
        self._block_source_tree_signal = True
        try:
            if state in (Qt.Checked, Qt.Unchecked):
                self.cascade_source_check_state(item, state)
                self.update_parent_source_state(item.parent())
        finally:
            self._block_source_tree_signal = False

        self.update_selected_folders_from_tree()
        if node_type == "dir" and relative:
            self.acknowledge_directories([relative])

    def schedule_file_tracker_refresh(self) -> None:
        if self._pending_file_tracker_refresh:
            return
        self._pending_file_tracker_refresh = True
        QTimer.singleShot(200, self.run_scheduled_file_tracker_refresh)

    def run_scheduled_file_tracker_refresh(self) -> None:
        self._pending_file_tracker_refresh = False
        metrics = self.refresh_file_tracker()
        if metrics is not None:
            self._workspace._workspace_metrics = metrics  # keep stage state in sync

    def resolve_source_root(self) -> Optional[Path]:
        project_manager = self._project_manager
        if not project_manager or not project_manager.project_dir:
            return None
        root_spec = (project_manager.source_state.root or "").strip()
        if not root_spec:
            return None
        root_path = Path(root_spec)
        if not root_path.is_absolute():
            root_path = (project_manager.project_dir / root_path).resolve()
        return root_path

    def to_project_relative(self, path: Path) -> str:
        project_manager = self._project_manager
        if not project_manager or not project_manager.project_dir:
            return path.as_posix()
        project_dir = Path(project_manager.project_dir).resolve()
        try:
            rel = path.resolve().relative_to(project_dir)
            return rel.as_posix()
        except ValueError:
            rel_str = os.path.relpath(path.resolve(), project_dir)
            return Path(rel_str).as_posix()

    @property
    def current_warnings(self) -> List[str]:
        return list(self._current_warnings)

    @property
    def workspace_metrics(self) -> WorkspaceMetrics | None:
        return self._workspace_metrics

    # ------------------------------------------------------------------
    # Banner helpers
    # ------------------------------------------------------------------
    def _update_highlights_banner(self, missing: Sequence[str], pending_count: int) -> None:
        render_highlights_banner(
            self._tab.highlights_banner,
            missing,
            pending_count,
            self._workspace.show_pending_highlights,
        )

    def _update_bulk_banner(self, missing: Sequence[str], pending_count: int) -> None:
        render_bulk_banner(
            self._tab.bulk_banner,
            missing,
            pending_count,
            self._workspace.show_bulk_analysis_tab,
        )

    # ------------------------------------------------------------------
    # Tree helpers (adapted from original ProjectWorkspace implementation)
    # ------------------------------------------------------------------
    def populate_directory_contents(self, *args, **kwargs) -> None:
        self._populate_directory_contents(*args, **kwargs)

    def _populate_directory_contents(
        self,
        parent_item: QTreeWidgetItem,
        directory: Path,
        processed: Set[Path],
        root_path: Path,
        selected: Set[str],
    ) -> None:
        directory = directory.resolve()
        if directory in processed:
            return
        processed.add(directory)

        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda entry: (not entry.is_dir(), entry.name.lower()),
            )
        except Exception:
            return

        for entry in entries:
            if self.should_skip_source_entry(entry):
                continue
            try:
                relative = entry.relative_to(root_path).as_posix()
            except ValueError:
                continue

            if entry.is_dir():
                node = self._source_tree_nodes.get(relative)
                if node is None:
                    node = QTreeWidgetItem([entry.name])
                    node.setData(0, Qt.UserRole, ("dir", relative))
                    self.apply_directory_flags(node)
                    node.setCheckState(
                        0,
                        Qt.Checked if self.is_path_tracked(relative, selected) else Qt.Unchecked,
                    )
                    self._source_tree_nodes[relative] = node
                    parent_item.addChild(node)
                else:
                    self.apply_directory_flags(node)
                self._populate_directory_contents(node, entry, processed, root_path, selected)
            else:
                file_item = QTreeWidgetItem([entry.name])
                file_item.setData(0, Qt.UserRole, ("file", relative))
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsUserCheckable)
                parent_item.addChild(file_item)

    def iter_directories(self, root_path: Path) -> List[str]:
        results: List[str] = []
        for path in root_path.rglob("*"):
            if path.is_dir():
                try:
                    rel = path.relative_to(root_path).as_posix()
                except ValueError:
                    continue
                results.append(rel)
        return results

    def normalise_relative_path(self, path: str) -> str:
        return normalise_tree_relative_path(path)

    def apply_directory_flags(self, item: QTreeWidgetItem) -> None:
        apply_directory_flags_item(item)

    def should_skip_source_entry(self, entry: Path) -> bool:
        return should_skip_tree_entry(entry)

    def is_path_tracked(self, relative: str, tracked: Set[str]) -> bool:
        return is_tree_path_tracked(relative, tracked)

    def compute_new_directories(
        self,
        actual: Set[str],
        selected: Set[str],
        acknowledged: Set[str],
    ) -> List[str]:
        new_entries: List[str] = []
        for entry in actual:
            normalized = self.normalise_relative_path(entry)
            if not normalized:
                continue
            if self.is_path_tracked(normalized, selected):
                continue
            if self.is_path_tracked(normalized, acknowledged):
                continue
            new_entries.append(normalized)
        return sorted(new_entries)

    def mark_directory_as_new(self, relative: str) -> None:
        normalized = self.normalise_relative_path(relative)
        if not normalized:
            return
        item = self._source_tree_nodes.get(normalized)
        if not item:
            return
        item.setBackground(0, QBrush(QColor("#fff3bf")))
        self._new_directory_alerts.add(normalized)
        self.expand_to_item(item)

    def clear_new_directory_marker(self, relative: str) -> None:
        normalized = self.normalise_relative_path(relative)
        if not normalized:
            return
        item = self._source_tree_nodes.get(normalized)
        if item:
            item.setBackground(0, QBrush())
        self._new_directory_alerts.discard(normalized)

    def acknowledge_directories(self, directories: Sequence[str]) -> None:
        project_manager = self._project_manager
        if not project_manager or not directories:
            return
        ack_set = {
            self.normalise_relative_path(entry)
            for entry in (project_manager.source_state.acknowledged_folders or [])
        }
        updated = False
        for entry in directories:
            normalized = self.normalise_relative_path(entry)
            if not normalized:
                continue
            if normalized not in ack_set:
                ack_set.add(normalized)
                updated = True
            self.clear_new_directory_marker(normalized)
        if updated:
            project_manager.update_source_state(acknowledged_folders=sorted(ack_set))

    def prompt_for_new_directories(self, new_dirs: Sequence[str]) -> None:
        if not new_dirs or self._new_dir_prompt_active:
            return
        self._new_dir_prompt_active = True
        try:
            preview = "\n".join(f"- {entry}" for entry in new_dirs[:10])
            remaining = len(new_dirs) - min(len(new_dirs), 10)
            message = "New folders were detected under the source root.\n\n" + preview
            if remaining > 0:
                message += f"\n… and {remaining} more."
            box = QMessageBox(self._workspace)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("New Folders Detected")
            box.setText(message)
            review_button = box.addButton("Review Folders", QMessageBox.AcceptRole)
            box.addButton("Ignore for Now", QMessageBox.RejectRole)
            box.setDefaultButton(review_button)
            box.exec()
            if box.clickedButton() == review_button:
                tabs = getattr(self._workspace, "_tabs", None)
                documents_tab = getattr(self._workspace, "_documents_tab", None)
                if tabs is not None and documents_tab is not None:
                    tabs.setCurrentWidget(documents_tab)
                for entry in new_dirs:
                    self.highlight_directory_item(entry)
            else:
                self.acknowledge_directories(new_dirs)
        finally:
            self._new_dir_prompt_active = False

    def expand_to_item(self, item: QTreeWidgetItem) -> None:
        tree = self._tab.source_tree
        current = item.parent()
        while current:
            tree.expandItem(current)
            current = current.parent()

    def highlight_directory_item(self, relative: str) -> None:
        tree = self._tab.source_tree
        normalized = self.normalise_relative_path(relative)
        if not normalized:
            return
        item = self._source_tree_nodes.get(normalized)
        if not item:
            return
        tree.setCurrentItem(item)
        self.expand_to_item(item)
        tree.scrollToItem(item)

    def handle_missing_directories(self, missing: Sequence[str]) -> None:
        project_manager = self._project_manager
        if not project_manager or not missing:
            return
        converted_root = (project_manager.project_dir / "converted_documents").resolve()
        highlights_root = (project_manager.project_dir / "highlights").resolve()
        colors_root = (project_manager.project_dir / "highlights" / "colors").resolve()

        selected = set(project_manager.source_state.selected_folders or [])
        acknowledged = set(project_manager.source_state.acknowledged_folders or [])
        selected.difference_update(missing)
        acknowledged.difference_update(missing)
        project_manager.update_source_state(
            selected_folders=sorted(selected),
            acknowledged_folders=sorted(acknowledged),
        )

        for entry in missing:
            self.clear_new_directory_marker(entry)

        self.cleanup_removed_directories(converted_root, highlights_root, colors_root, missing)

    def cleanup_removed_directories(
        self,
        converted_root: Path,
        highlights_root: Path,
        colors_root: Path,
        missing: Sequence[str],
    ) -> None:
        for entry in missing:
            relative = Path(entry)
            target = (converted_root / relative).resolve()
            highlight_target = (highlights_root / relative).resolve()
            color_target = (colors_root / relative).resolve()
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if highlight_target.exists():
                shutil.rmtree(highlight_target, ignore_errors=True)
            if color_target.exists():
                shutil.rmtree(color_target, ignore_errors=True)

        bulk_root = (converted_root.parent / "bulk_analysis").resolve()
        if not bulk_root.exists():
            return
        for slug_dir in bulk_root.iterdir():
            reduce_dir = slug_dir / "reduce"
            map_dir = slug_dir / "map"
            for candidate in (reduce_dir, map_dir):
                if candidate.exists() and not any(candidate.iterdir()):
                    shutil.rmtree(candidate, ignore_errors=True)

    def cascade_source_check_state(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        cascade_tree_check_state(item, state)

    def update_parent_source_state(self, item: Optional[QTreeWidgetItem]) -> None:
        update_parent_tree_state(item)

    def update_selected_folders_from_tree(self) -> None:
        project_manager = self._project_manager
        if not project_manager:
            return
        selected = self.collect_selected_directories()
        ack_set = {
            self.normalise_relative_path(entry)
            for entry in (project_manager.source_state.acknowledged_folders or [])
        }
        ack_set.update(selected)
        project_manager.update_source_state(
            selected_folders=selected,
            acknowledged_folders=sorted(ack_set),
        )
        self.schedule_file_tracker_refresh()
        feature_flags = getattr(self._workspace, "_feature_flags", None)
        if feature_flags and feature_flags.bulk_analysis_groups_enabled:
            self._workspace._refresh_bulk_analysis_groups()

    def collect_selected_directories(self) -> List[str]:
        tree = self._tab.source_tree
        results: Set[str] = set()
        root = tree.invisibleRootItem()
        for index in range(root.childCount()):
            self.collect_selected_directories_from_item(root.child(index), results)
        return sorted(results)

    def collect_selected_directories_from_item(self, item: QTreeWidgetItem, results: Set[str]) -> None:
        collect_checked_directories_from_item(item, results, self.normalise_relative_path)

    # ------------------------------------------------------------------
    # Warning helpers
    # ------------------------------------------------------------------
    def set_root_warning(self, warnings: List[str]) -> None:
        render_root_warning(self._tab.root_warning_label, warnings)

    def compute_root_warnings(self, root_path: Path) -> List[str]:
        warnings: List[str] = []
        has_root_files = any(child.is_file() for child in root_path.iterdir())
        if has_root_files:
            warnings.append(
                "Files in the source root will be skipped. Move them into subfolders so they can be processed."
            )
        project_manager = self._project_manager
        selected = project_manager.source_state.selected_folders if project_manager else []
        if not selected:
            warnings.append("Select at least one folder to include in scanning.")
        return warnings

    def prompt_reselect_source_root(self) -> None:
        project_manager = self._project_manager
        if not project_manager:
            return
        root_spec = project_manager.source_state.root
        if not root_spec:
            return
        response = QMessageBox.question(
            self._workspace,
            "Source Folder Missing",
            "The previously selected source folder cannot be found.\n"
            "Do you want to locate it now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if response == QMessageBox.Yes:
            self._missing_root_prompted = False
            self._workspace._select_source_root()
