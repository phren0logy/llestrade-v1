"""Presentation widget for the bulk analysis tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTextEdit,
    QHeaderView,
)


class BulkAnalysisTab(QWidget):
    """Encapsulate the bulk-analysis dashboard UI widgets."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.create_button = QPushButton("New Bulk Analysis…")
        self.refresh_button = QPushButton("Refresh")

        self.info_label = QLabel("No bulk analysis groups yet.")
        self.active_progress_group_label = QLabel("")
        self.active_progress_group_label.setStyleSheet("font-weight: 600;")
        self.active_progress_bar = QProgressBar()
        self.active_progress_bar.setRange(0, 100)
        self.active_progress_bar.setValue(0)
        self.active_progress_status_label = QLabel("")
        self.active_progress_detail_label = QLabel("")
        self.active_progress_detail_label.setStyleSheet("color: #666;")
        self.active_progress_widget = QWidget()
        self.active_progress_widget.hide()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Group", "Coverage", "Updated", "Status", "Est. Cost", "Placeholders", "Actions"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.empty_label = QLabel("No bulk analysis groups created yet.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #666; padding: 20px;")
        self.empty_label.hide()

        self.group_tree = QTreeWidget()
        self.group_tree.setHeaderHidden(True)
        self.group_tree.setUniformRowHeights(True)
        self.group_tree.setSelectionMode(QAbstractItemView.NoSelection)
        self.group_tree.setFocusPolicy(Qt.NoFocus)

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(QLabel("Manage bulk analysis groups to organise processed documents."))
        header_layout.addStretch()
        header_layout.addWidget(self.create_button)
        header_layout.addWidget(self.refresh_button)
        layout.addLayout(header_layout)

        layout.addWidget(self.info_label)
        progress_layout = QVBoxLayout(self.active_progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(4)
        progress_layout.addWidget(self.active_progress_group_label)
        progress_layout.addWidget(self.active_progress_bar)
        progress_layout.addWidget(self.active_progress_status_label)
        progress_layout.addWidget(self.active_progress_detail_label)
        layout.addWidget(self.active_progress_widget)
        layout.addWidget(self.log_text)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        table_container = QVBoxLayout()
        table_container.setSpacing(6)
        table_container.addWidget(self.table)
        table_container.addWidget(self.empty_label)
        content_layout.addLayout(table_container, 2)

        tree_container = QVBoxLayout()
        tree_container.setSpacing(4)
        tree_label = QLabel("Folder coverage (read-only)")
        tree_label.setStyleSheet("color: #666; font-size: 11px;")
        tree_container.addWidget(tree_label)
        tree_container.addWidget(self.group_tree)
        content_layout.addLayout(tree_container, 1)

        layout.addLayout(content_layout)
