"""Welcome stage for the dashboard UI."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt, Signal, QSize, QUrl
from PySide6.QtGui import QFont, QIcon, QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.app.core import ProjectManager, SecureSettings
from src.app.core.file_tracker import DashboardMetrics, FileTracker

LOGGER = logging.getLogger(__name__)


class WelcomeStage(QWidget):
    """Landing view that surfaces recent projects and quick actions."""

    new_project_requested = Signal()
    project_opened = Signal(Path)
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = SecureSettings()
        self._build_ui()
        self._update_api_status()

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        layout.addWidget(self._build_header())

        content_row = QHBoxLayout()
        content_row.setSpacing(30)
        self._recent_projects_group = QGroupBox("Recent Projects")
        self._recent_projects_layout = QVBoxLayout(self._recent_projects_group)
        self._populate_recent_projects()
        content_row.addWidget(self._recent_projects_group, 2)
        content_row.addLayout(self._build_side_panel(), 1)
        layout.addLayout(content_row)
        layout.addStretch()

    def _build_header(self) -> QWidget:
        header = QWidget()
        h_layout = QVBoxLayout(header)
        title = QLabel("Welcome to Llestrade")
        font = QFont(title.font())
        font.setPointSize(24)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Launch or resume a case to begin drafting your report")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 14px;")

        h_layout.addWidget(title)
        h_layout.addWidget(subtitle)
        return header

    def _populate_recent_projects(self) -> None:
        while self._recent_projects_layout.count():
            item = self._recent_projects_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        recent_projects = self.settings.get_recent_projects()
        if not recent_projects:
            empty = QLabel("No recent projects")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: #999; padding: 40px;")
            self._recent_projects_layout.addWidget(empty)
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(12)

        for index, project_info in enumerate(recent_projects[:6]):
            row, col = divmod(index, 2)
            grid.addWidget(self._build_project_card(project_info), row, col)

        scroll.setWidget(container)
        self._recent_projects_layout.addWidget(scroll)

    def _build_project_card(self, project_info: Dict) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Box)
        card.setStyleSheet(
            "QFrame { border: 1px solid #ddd; border-radius: 8px; background: white; padding: 16px; }"
            "QFrame:hover { border-color: #2196f3; background: #f5f5f5; }"
        )
        card.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        name = QLabel(project_info.get("name", "Untitled Project"))
        font = QFont(name.font())
        font.setBold(True)
        font.setPointSize(12)
        name.setFont(font)
        layout.addWidget(name)

        metadata = project_info.get("metadata", {})
        case_label = QLabel(f"Case: {metadata.get('case_name', 'N/A')}")
        case_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(case_label)

        subject_name = metadata.get("subject_name")
        if subject_name:
            subject_label = QLabel(f"Subject: {subject_name}")
            subject_label.setStyleSheet("color: #666; font-size: 11px;")
            layout.addWidget(subject_label)

        dob = metadata.get("date_of_birth")
        if dob:
            dob_label = QLabel(f"DOB: {dob}")
            dob_label.setStyleSheet("color: #666; font-size: 11px;")
            layout.addWidget(dob_label)

        last_modified = project_info.get("last_modified")
        if last_modified:
            try:
                timestamp = datetime.fromisoformat(last_modified)
                formatted = timestamp.strftime("%b %d, %Y at %I:%M %p")
                modified_label = QLabel(f"Modified: {formatted}")
                modified_label.setStyleSheet("color: #999; font-size: 10px;")
                layout.addWidget(modified_label)
            except ValueError:
                LOGGER.debug("Failed to parse last_modified timestamp: %s", last_modified)

        project_path = Path(project_info.get("path", "")).expanduser()

        stats_label = QLabel(self._project_stats_text(project_path))
        stats_label.setWordWrap(True)
        stats_label.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(stats_label)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        open_project_btn = QPushButton("Open")
        open_project_btn.setMinimumHeight(28)
        open_project_btn.clicked.connect(lambda _, path=project_path: self._open_project_path(path))
        button_row.addWidget(open_project_btn)

        open_folder_btn = QPushButton("Open Folder")
        open_folder_btn.setMinimumHeight(28)
        open_folder_btn.clicked.connect(lambda _, path=project_path: self._open_project_folder(path))
        button_row.addWidget(open_folder_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setMinimumHeight(28)
        delete_btn.setStyleSheet("QPushButton { color: #d32f2f; }")
        delete_btn.clicked.connect(lambda _, path=project_path: self._confirm_delete_project(path))
        button_row.addWidget(delete_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        if not project_path.exists():
            card.setStyleSheet(
                "QFrame { border: 1px solid #e57373; border-radius: 8px; background: #fff6f7; padding: 16px; }"
            )
            stats_label.setText("Project file missing. Use Delete to remove from list.")
            open_project_btn.setEnabled(False)
            open_folder_btn.setEnabled(False)
        else:
            def _open(_event) -> None:
                self.project_opened.emit(project_path)

            card.mousePressEvent = _open  # type: ignore[assignment]
        return card

    def _build_side_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.addWidget(self._build_quick_actions())
        layout.addWidget(self._build_api_status())
        layout.addWidget(self._build_quick_start())
        layout.addStretch()
        return layout

    def _build_quick_actions(self) -> QWidget:
        group = QGroupBox("Quick Actions")
        layout = QVBoxLayout(group)

        new_btn = QPushButton("📄 New Project")
        new_btn.setMinimumHeight(48)
        new_btn.setStyleSheet(
            "QPushButton { background-color: #2196f3; color: white; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        new_btn.clicked.connect(self.new_project_requested.emit)
        layout.addWidget(new_btn)

        open_btn = QPushButton("📂 Open Project")
        open_btn.setMinimumHeight(44)
        open_btn.setStyleSheet(
            "QPushButton { border: 2px solid #2196f3; color: #2196f3; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e3f2fd; }"
        )
        open_btn.clicked.connect(self._open_project_dialog)
        layout.addWidget(open_btn)

        return group

    # ------------------------------------------------------------------
    # Project helpers
    # ------------------------------------------------------------------
    def _project_stats_text(self, project_path: Path) -> str:
        metrics_text = self._metrics_text(project_path)
        if metrics_text:
            return metrics_text

        if not project_path.exists():
            return "Project file not found."

        fallback_metrics = self._tracker_metrics(project_path.parent)
        if fallback_metrics.imported_total:
            return self._format_metrics(fallback_metrics)
        return "No converted documents yet."

    def _metrics_text(self, project_path: Path) -> str:
        metrics = ProjectManager.read_dashboard_metrics_from_disk(project_path)

        if (
            metrics.last_scan is None
            and metrics.imported_total == 0
            and metrics.bulk_analysis_total == 0
            and metrics.pending_bulk_analysis == 0
        ):
            return ""
        return self._format_metrics(metrics)

    def _format_metrics(self, metrics: DashboardMetrics) -> str:
        converted = metrics.imported_total
        if not converted:
            summary = "Converted: 0 | Highlights: 0 | Bulk analysis: 0"
        else:
            pdf_total = metrics.highlights_total + metrics.pending_highlights
            highlight_text = f"Highlights: {metrics.highlights_total} of {pdf_total}"
            if metrics.pending_highlights:
                highlight_text += f" (pending {metrics.pending_highlights})"
            bulk_text = f"Bulk analysis: {metrics.bulk_analysis_total} of {converted}"
            if metrics.pending_bulk_analysis:
                bulk_text += f" (pending {metrics.pending_bulk_analysis})"
            summary = f"Converted: {converted} | {highlight_text} | {bulk_text}"

        last_scan = metrics.last_scan
        if isinstance(last_scan, datetime):
            try:
                formatted = last_scan.astimezone().strftime("%Y-%m-%d %H:%M")
            except ValueError:
                formatted = last_scan.strftime("%Y-%m-%d %H:%M")
            summary += f" - Last scan {formatted}"
        return summary

    def _tracker_metrics(self, project_dir: Path) -> DashboardMetrics:
        tracker = FileTracker(project_dir)
        snapshot = tracker.load()
        if snapshot is None:
            try:
                snapshot = tracker.scan()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.debug("Failed to scan tracker metrics for %s: %s", project_dir, exc)
                return DashboardMetrics.empty()
        try:
            return snapshot.to_dashboard_metrics()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.debug("Failed to translate tracker snapshot for %s: %s", project_dir, exc)
            return DashboardMetrics.empty()

    def _open_project_path(self, project_path: Path) -> None:
        if project_path.exists():
            self.project_opened.emit(project_path)
            self._populate_recent_projects()
        else:
            QMessageBox.warning(
                self,
                "Project Missing",
                f"The project path could not be found:\n{project_path}",
            )
            self.settings.remove_recent_project(str(project_path))
            self._populate_recent_projects()

    def _open_project_folder(self, project_path: Path) -> None:
        folder = project_path.parent if project_path.suffix == ProjectManager.PROJECT_EXTENSION else project_path
        if not folder.exists():
            QMessageBox.warning(
                self,
                "Folder Missing",
                f"The project folder could not be found:\n{folder}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _confirm_delete_project(self, project_path: Path) -> None:
        if project_path.suffix != ProjectManager.PROJECT_EXTENSION:
            candidate = project_path / ProjectManager.PROJECT_FILENAME
            if candidate.exists():
                project_path = candidate
        if not project_path.exists():
            self.settings.remove_recent_project(str(project_path))
            self._populate_recent_projects()
            return
        project_dir = project_path.parent
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Delete project '{project_dir.name}' and all generated files?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(project_dir, ignore_errors=False)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Delete Failed",
                f"Failed to delete the project folder:\n{exc}",
            )
            return
        self.settings.remove_recent_project(str(project_path))
        self._populate_recent_projects()
        QMessageBox.information(self, "Project Deleted", "The project was deleted successfully.")

    def _build_api_status(self) -> QWidget:
        self._api_group = QGroupBox("Settings")
        self._api_layout = QVBoxLayout(self._api_group)
        configure_btn = QPushButton("Open Settings")
        configure_btn.clicked.connect(self.settings_requested.emit)
        self._api_layout.addWidget(configure_btn)
        return self._api_group

    def _build_quick_start(self) -> QWidget:
        group = QGroupBox("Quick Start Guide")
        layout = QVBoxLayout(group)
        for step in (
            "1. Configure settings and API keys",
            "2. Create or open a project",
            "3. Import documents into the workspace",
            "4. Process and analyze content",
            "5. Generate and refine your report",
        ):
            label = QLabel(step)
            label.setWordWrap(True)
            label.setStyleSheet("padding: 2px 0;")
            layout.addWidget(label)
        return group

    # ------------------------------------------------------------------
    # Actions & helpers
    # ------------------------------------------------------------------
    def _open_project_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(Path.home()),
            "Forensic Report Project (*.frpd)",
        )
        if file_path:
            self.project_opened.emit(Path(file_path))
            self._populate_recent_projects()

    def showEvent(self, event: QShowEvent) -> None:  # pragma: no cover - Qt lifecycle
        super().showEvent(event)
        self._populate_recent_projects()
        self._update_api_status()

    def _update_api_status(self) -> None:
        while self._api_layout.count() > 1:
            item = self._api_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        providers = [
            ("anthropic", "Anthropic (Claude)", "🤖"),
            ("anthropic_bedrock", "AWS Bedrock (Claude)", "🛡️"),
            ("gemini", "Google Gemini", "✨"),
            ("azure_openai", "Azure OpenAI", "☁️"),
        ]
        for provider_id, label, icon in providers:
            if provider_id == "anthropic_bedrock":
                available = self._is_bedrock_available()
            else:
                available = self.settings.has_api_key(provider_id)
            row = QLabel(f"{icon} {label} — {'Configured' if available else 'Missing'}")
            color = "#4caf50" if available else "#f44336"
            row.setStyleSheet(f"color: {color};")
            self._api_layout.insertWidget(self._api_layout.count() - 1, row)

    def refresh_api_status(self) -> None:
        """Refresh provider status labels without relying on periodic polling."""
        self._update_api_status()

    def get_recent_projects(self) -> List[Dict]:  # pragma: no cover - compatibility shim
        return self.settings.get_recent_projects()

    def cleanup(self) -> None:
        pass

    def _is_bedrock_available(self) -> bool:
        try:
            bedrock_settings = self.settings.get("aws_bedrock_settings", {}) or {}
        except Exception:
            bedrock_settings = {}

        profile = str(bedrock_settings.get("profile") or "").strip()
        if profile:
            return True

        # Consider standard AWS credential hints as "configured" for welcome-stage status.
        env_markers = (
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        )
        if any(os.getenv(name) for name in env_markers):
            return True

        aws_dir = Path.home() / ".aws"
        return (aws_dir / "credentials").exists() or (aws_dir / "config").exists()


__all__ = ["WelcomeStage"]
