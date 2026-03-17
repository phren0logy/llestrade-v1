"""Simple single-page PyMuPDF viewer for citation review."""

from __future__ import annotations

from pathlib import Path

import fitz
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.app.core.citation_review import PdfTarget


class PdfCitationViewer(QWidget):
    """Render one PDF page at a time with normalized bbox overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._doc: fitz.Document | None = None
        self._source_path: str | None = None
        self._page_number = 1
        self._boxes: tuple[dict[str, float], ...] = ()
        self._zoom = 1.25
        self._fit_width = True

        self._prev_button = QPushButton("Prev")
        self._next_button = QPushButton("Next")
        self._zoom_out_button = QPushButton("-")
        self._zoom_in_button = QPushButton("+")
        self._fit_button = QPushButton("Fit Width")
        self._page_label = QLabel("No PDF selected")
        self._snippet_label = QLabel("")
        self._snippet_label.setWordWrap(True)
        self._snippet_label.setStyleSheet("color: #666;")

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self._image_label.setBackgroundRole(self.backgroundRole())
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._image_label)

        controls = QHBoxLayout()
        controls.addWidget(self._prev_button)
        controls.addWidget(self._next_button)
        controls.addWidget(self._zoom_out_button)
        controls.addWidget(self._zoom_in_button)
        controls.addWidget(self._fit_button)
        controls.addStretch(1)
        controls.addWidget(self._page_label)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._scroll, stretch=1)
        layout.addWidget(self._snippet_label)

        self._prev_button.clicked.connect(self._go_prev)
        self._next_button.clicked.connect(self._go_next)
        self._zoom_out_button.clicked.connect(lambda: self._adjust_zoom(0.85))
        self._zoom_in_button.clicked.connect(lambda: self._adjust_zoom(1.15))
        self._fit_button.clicked.connect(self._set_fit_width)
        self._update_buttons()

    def set_target(self, target: PdfTarget) -> None:
        if not target.source_path:
            self.clear_view("Source PDF unavailable.")
            self._snippet_label.setText(target.snippet or "")
            return

        source_path = Path(target.source_path).expanduser().resolve().as_posix()
        if source_path != self._source_path:
            self._close_document()
            self._doc = fitz.open(source_path)
            self._source_path = source_path

        self._page_number = max(int(target.page_number or 1), 1)
        self._boxes = tuple(target.boxes or ())
        self._snippet_label.setText(target.snippet or "")
        self._render_page()

    def clear_view(self, message: str = "No PDF selected.") -> None:
        self._image_label.clear()
        self._page_label.setText(message)
        self._boxes = ()
        self._update_buttons()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._fit_width and self._doc is not None:
            self._render_page()

    def _set_fit_width(self) -> None:
        self._fit_width = True
        self._render_page()

    def _adjust_zoom(self, factor: float) -> None:
        self._fit_width = False
        self._zoom = max(0.25, min(self._zoom * factor, 6.0))
        self._render_page()

    def _go_prev(self) -> None:
        if self._doc is None:
            return
        if self._page_number > 1:
            self._page_number -= 1
            self._render_page()

    def _go_next(self) -> None:
        if self._doc is None:
            return
        if self._page_number < len(self._doc):
            self._page_number += 1
            self._render_page()

    def _render_page(self) -> None:
        if self._doc is None:
            self.clear_view()
            return

        page_index = max(min(self._page_number - 1, len(self._doc) - 1), 0)
        page = self._doc.load_page(page_index)
        if self._fit_width:
            viewport_width = max(self._scroll.viewport().width() - 24, 100)
            page_width = max(page.rect.width, 1.0)
            self._zoom = max(min(viewport_width / page_width, 4.0), 0.25)

        pix = page.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom), alpha=False)
        fmt = QImage.Format_RGB888
        image = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
        rendered = QPixmap.fromImage(image)

        if self._boxes:
            painter = QPainter(rendered)
            pen = QPen(QColor(220, 58, 37))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(QColor(220, 58, 37, 48))
            width = rendered.width()
            height = rendered.height()
            for box in self._boxes:
                try:
                    x = float(box.get("x_min", 0.0)) * width
                    y = float(box.get("y_min", 0.0)) * height
                    w = max(float(box.get("x_max", 0.0)) * width - x, 2.0)
                    h = max(float(box.get("y_max", 0.0)) * height - y, 2.0)
                except (TypeError, ValueError):
                    continue
                painter.drawRect(int(x), int(y), int(w), int(h))
            painter.end()

        self._image_label.setPixmap(rendered)
        self._page_number = page_index + 1
        self._page_label.setText(f"{Path(self._source_path or '').name} • Page {self._page_number} of {len(self._doc)}")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_doc = self._doc is not None
        self._prev_button.setEnabled(has_doc and self._page_number > 1)
        self._next_button.setEnabled(has_doc and self._doc is not None and self._page_number < len(self._doc))
        self._zoom_in_button.setEnabled(has_doc)
        self._zoom_out_button.setEnabled(has_doc)
        self._fit_button.setEnabled(has_doc)

    def _close_document(self) -> None:
        if self._doc is not None:
            self._doc.close()
        self._doc = None

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._close_document()
        super().closeEvent(event)


__all__ = ["PdfCitationViewer"]
