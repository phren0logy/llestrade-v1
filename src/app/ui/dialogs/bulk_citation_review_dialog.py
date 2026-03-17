"""Dialog for reviewing bulk map outputs with clickable citation navigation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.citation_review import CitationReviewService, ReviewedCitation, ReviewedOutput
from src.app.ui.widgets.pdf_citation_viewer import PdfCitationViewer


class BulkCitationReviewDialog(QDialog):
    """Review bulk map outputs and jump to supporting PDF regions."""

    def __init__(
        self,
        *,
        project_dir: Path,
        group: BulkAnalysisGroup,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Citation Review: {group.name}")
        self.resize(1280, 860)

        self._group = group
        self._service = CitationReviewService(project_dir)
        self._current_output: ReviewedOutput | None = None
        self._citations_by_label: dict[str, ReviewedCitation] = {}

        self._output_combo = QComboBox()
        self._output_combo.currentIndexChanged.connect(self._load_selected_output)

        self._output_text = QTextBrowser()
        self._output_text.setOpenLinks(False)
        self._output_text.anchorClicked.connect(self._handle_anchor_clicked)

        self._citation_list = QListWidget()
        self._citation_list.currentItemChanged.connect(self._handle_list_changed)

        self._active_label = QLabel("No citation selected.")
        self._active_label.setWordWrap(True)
        self._viewer = PdfCitationViewer()

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Output"))
        left_layout.addWidget(self._output_combo)
        left_layout.addWidget(QLabel("Generated Output"))
        left_layout.addWidget(self._output_text, stretch=2)
        left_layout.addWidget(QLabel("Citations"))
        left_layout.addWidget(self._citation_list, stretch=1)
        left_layout.addWidget(self._active_label)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        outputs = self._service.list_map_outputs(group)
        if not outputs:
            QMessageBox.information(self, "Citation Review", "No map outputs are available for this group yet.")
            self._output_combo.setEnabled(False)
            self._output_text.setPlainText("No outputs available.")
            return

        for path, rel in outputs:
            self._output_combo.addItem(rel, (path.as_posix(), rel))
        self._load_selected_output()

    def _load_selected_output(self) -> None:
        data = self._output_combo.currentData()
        if not data:
            return
        output_path_str, relative_key = data
        output_path = Path(output_path_str)
        self._current_output = self._service.load_reviewed_output(output_path, relative_key=relative_key)
        self._citations_by_label = {
            citation.citation_label: citation
            for citation in self._current_output.citations
        }
        self._output_text.setHtml(self._current_output.rendered_html)
        self._citation_list.clear()
        for citation in self._current_output.citations:
            page_text = f"p. {citation.page_number}" if citation.page_number else "page unavailable"
            source_text = citation.source_relative_path or citation.document_relative_path or "unknown source"
            item = QListWidgetItem(
                f"{citation.citation_label} • {source_text} • {page_text} • {citation.status}"
            )
            item.setData(0x0100, citation.citation_label)
            self._citation_list.addItem(item)

        if self._current_output.citations:
            self._citation_list.setCurrentRow(0)
        else:
            self._active_label.setText("No stored citations found for this output.")
            self._viewer.clear_view("No citations available.")

    def _handle_anchor_clicked(self, url: QUrl) -> None:
        label = url.toString().removeprefix("citation:")
        self._set_active_citation(label)

    def _handle_list_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        label = current.data(0x0100)
        if isinstance(label, str):
            self._set_active_citation(label)

    def _set_active_citation(self, label: str) -> None:
        citation = self._citations_by_label.get(label)
        if citation is None:
            return
        if self._citation_list.currentItem() is None or self._citation_list.currentItem().data(0x0100) != label:
            for idx in range(self._citation_list.count()):
                item = self._citation_list.item(idx)
                if item.data(0x0100) == label:
                    self._citation_list.setCurrentRow(idx)
                    break

        source_text = citation.source_relative_path or citation.document_relative_path or "unknown source"
        page_text = f"page {citation.page_number}" if citation.page_number else "page unavailable"
        self._active_label.setText(
            f"{citation.citation_label} • {source_text} • {page_text} • {citation.status}: {citation.reason}"
        )
        self._viewer.set_target(citation.to_pdf_target())


__all__ = ["BulkCitationReviewDialog"]
