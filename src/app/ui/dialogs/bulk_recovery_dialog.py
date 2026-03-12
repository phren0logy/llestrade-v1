"""Dialog for inspecting and repairing bulk recovery state."""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.bulk_recovery import BulkRecoveryStore


class RecoveryAction(str, Enum):
    RESUME = "resume"
    RERUN_SELECTED = "rerun_selected"


class BulkRecoveryDialog(QDialog):
    """Inspect bulk recovery state and mark units for regeneration."""

    def __init__(
        self,
        *,
        group: BulkAnalysisGroup,
        store: BulkRecoveryStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Recover {group.name}")
        self.resize(760, 520)
        self._group = group
        self._store = store
        self.selected_action: RecoveryAction | None = None

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("color: #555;")

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Unit", "Status", "Updated", "Notes"])
        self._tree.setSelectionMode(QTreeWidget.ExtendedSelection)

        self._resume_button = QPushButton("Resume")
        self._rerun_button = QPushButton("Rerun Selected Chunks")
        self._compromise_button = QPushButton("Mark Compromised")
        self._reset_document_button = QPushButton("Reset Document")
        self._reset_stage_button = QPushButton("Reset Stage")
        self._close_button = QPushButton("Close")

        self._build_ui()
        self._connect()
        self._reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._summary_label)
        layout.addWidget(self._tree)

        buttons = QHBoxLayout()
        buttons.addWidget(self._resume_button)
        buttons.addWidget(self._rerun_button)
        buttons.addWidget(self._compromise_button)
        buttons.addWidget(self._reset_document_button)
        buttons.addWidget(self._reset_stage_button)
        buttons.addStretch()
        buttons.addWidget(self._close_button)
        layout.addLayout(buttons)

    def _connect(self) -> None:
        self._resume_button.clicked.connect(self._resume)
        self._rerun_button.clicked.connect(self._rerun_selected)
        self._compromise_button.clicked.connect(self._mark_compromised)
        self._reset_document_button.clicked.connect(self._reset_document)
        self._reset_stage_button.clicked.connect(self._reset_stage)
        self._close_button.clicked.connect(self.reject)

    def _reload(self) -> None:
        self._tree.clear()
        if (self._group.operation or "per_document") == "combined":
            self._load_reduce()
        else:
            self._load_map()

    def _load_map(self) -> None:
        manifest = self._store.load_map_manifest()
        documents = dict(manifest.get("documents") or {})
        corrupt = 0
        resumable = 0
        for document_rel, entry in sorted(documents.items()):
            if not isinstance(entry, dict):
                continue
            parent = QTreeWidgetItem([document_rel, str(entry.get("status") or "incomplete"), str(entry.get("ran_at") or ""), ""])
            parent.setData(0, Qt.UserRole, ("document", document_rel))
            chunks = dict(entry.get("chunks") or {})
            batches = dict(entry.get("batches") or {})
            for chunk_key, payload in sorted(chunks.items(), key=lambda item: int(item[0])):
                status = str((payload or {}).get("status") or "incomplete")
                note = str((payload or {}).get("quarantine_reason") or "")
                item = QTreeWidgetItem([f"Chunk {chunk_key}", status, str((payload or {}).get("updated_at") or ""), note])
                item.setData(0, Qt.UserRole, ("chunk", document_rel, int(chunk_key)))
                parent.addChild(item)
                if status != "complete":
                    resumable += 1
                if status in {"corrupt", "compromised"}:
                    corrupt += 1
            for batch_key, payload in sorted(batches.items()):
                status = str((payload or {}).get("status") or "incomplete")
                note = str((payload or {}).get("quarantine_reason") or "")
                item = QTreeWidgetItem([f"Batch {batch_key}", status, str((payload or {}).get("updated_at") or ""), note])
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                parent.addChild(item)
            self._tree.addTopLevelItem(parent)
        self._tree.expandAll()
        self._summary_label.setText(
            f"Latest map recovery state. Resumable chunks: {resumable}. Corrupt/compromised chunks: {corrupt}."
        )

    def _load_reduce(self) -> None:
        manifest = self._store.load_reduce_manifest()
        chunks = dict((manifest.get("chunks") or {}).get("items") or {})
        root = QTreeWidgetItem(["Combined stage", str(manifest.get("status") or "idle"), str(manifest.get("ran_at") or ""), ""])
        for chunk_key, payload in sorted(chunks.items(), key=lambda item: int(item[0])):
            status = str((payload or {}).get("status") or "incomplete")
            note = str((payload or {}).get("quarantine_reason") or "")
            item = QTreeWidgetItem([f"Chunk {chunk_key}", status, str((payload or {}).get("updated_at") or ""), note])
            item.setData(0, Qt.UserRole, ("reduce_chunk", int(chunk_key)))
            root.addChild(item)
        for batch_key, payload in sorted(dict(manifest.get("batches") or {}).items()):
            status = str((payload or {}).get("status") or "incomplete")
            note = str((payload or {}).get("quarantine_reason") or "")
            item = QTreeWidgetItem([f"Batch {batch_key}", status, str((payload or {}).get("updated_at") or ""), note])
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            root.addChild(item)
        self._tree.addTopLevelItem(root)
        self._tree.expandAll()
        self._summary_label.setText(
            f"Latest combined recovery state. Finalized: {'yes' if manifest.get('finalized') else 'no'}."
        )

    def _selected_payloads(self) -> list[tuple]:
        payloads: list[tuple] = []
        for item in self._tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data:
                payloads.append(tuple(data))
        return payloads

    def selected_documents(self) -> list[str]:
        documents: set[str] = set()
        for payload in self._selected_payloads():
            if payload[0] == "document":
                documents.add(str(payload[1]))
            elif payload[0] == "chunk":
                documents.add(str(payload[1]))
        return sorted(documents)

    def _resume(self) -> None:
        self.selected_action = RecoveryAction.RESUME
        self.accept()

    def _rerun_selected(self) -> None:
        payloads = self._selected_payloads()
        if not payloads:
            QMessageBox.information(self, "Recover", "Select one or more chunks to rerun.")
            return
        changed = False
        for payload in payloads:
            if payload[0] == "chunk":
                self._store.mark_map_chunk_compromised(
                    document_rel=str(payload[1]),
                    index=int(payload[2]),
                    reason="manually selected for rerun",
                )
                changed = True
            elif payload[0] == "reduce_chunk":
                self._store.mark_reduce_chunk_compromised(
                    index=int(payload[1]),
                    reason="manually selected for rerun",
                )
                changed = True
        if not changed:
            QMessageBox.information(self, "Recover", "The current selection does not include rerunnable chunks.")
            return
        self.selected_action = RecoveryAction.RERUN_SELECTED
        self.accept()

    def _mark_compromised(self) -> None:
        payloads = self._selected_payloads()
        changed = False
        for payload in payloads:
            if payload[0] == "chunk":
                self._store.mark_map_chunk_compromised(
                    document_rel=str(payload[1]),
                    index=int(payload[2]),
                    reason="manually marked compromised",
                )
                changed = True
            elif payload[0] == "reduce_chunk":
                self._store.mark_reduce_chunk_compromised(
                    index=int(payload[1]),
                    reason="manually marked compromised",
                )
                changed = True
        if changed:
            self._reload()

    def _reset_document(self) -> None:
        payloads = self._selected_payloads()
        documents = {str(payload[1]) for payload in payloads if payload and payload[0] in {"document", "chunk"}}
        if not documents:
            QMessageBox.information(self, "Recover", "Select a document or one of its chunks.")
            return
        for document_rel in documents:
            self._store.reset_map_document(document_rel)
        self._reload()

    def _reset_stage(self) -> None:
        if (self._group.operation or "per_document") == "combined":
            self._store.reset_reduce()
        else:
            self._store.clear_map()
            self._store.save_map_manifest(self._store.load_map_manifest())
        self._reload()


__all__ = ["BulkRecoveryDialog", "RecoveryAction"]
