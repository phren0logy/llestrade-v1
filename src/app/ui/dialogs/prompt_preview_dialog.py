"""Modal dialog that displays rendered system/user prompts side by side."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.app.core.placeholders.analyzer import (
    PlaceholderAnalysis,
    analyse_prompts,
    build_preview_styles,
    highlight_placeholders_raw,
    render_preview_html,
)
from src.app.core.prompt_assembly import GENERATED_CITATION_APPENDIX_TITLE
from src.app.core.prompt_preview import PromptPreview

class PromptPreviewDialog(QDialog):
    """Simple dialog that renders the system and user prompts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prompt Preview")
        self.resize(980, 600)

        self._preview = None
        self._analysis: PlaceholderAnalysis | None = None
        self._raw_system_html = ""
        self._raw_user_html = ""
        self._preview_system_html = ""
        self._preview_user_html = ""

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(12)
        toggle_row.addWidget(QLabel("View:"))
        self._raw_button = QRadioButton("Raw (placeholders highlighted)")
        self._preview_button = QRadioButton("Preview (with substitutions)")
        self._raw_button.setChecked(True)
        toggle_group = QButtonGroup(self)
        toggle_group.addButton(self._raw_button)
        toggle_group.addButton(self._preview_button)
        self._raw_button.toggled.connect(self._update_views)
        toggle_row.addWidget(self._raw_button)
        toggle_row.addWidget(self._preview_button)
        toggle_row.addStretch(1)
        main_layout.addLayout(toggle_row)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        self._system_edit = self._create_editor("System Prompt")
        self._user_edit = self._create_editor("User Prompt")

        content_row.addWidget(self._system_edit, stretch=1)
        content_row.addWidget(self._user_edit, stretch=1)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(6)

        self._usage_label = QLabel("Placeholder Usage")
        self._usage_label.setStyleSheet("font-weight: 600;")
        sidebar.addWidget(self._usage_label)

        self._used_list = QListWidget()
        self._used_list.setMinimumWidth(220)
        sidebar.addWidget(QLabel("Used in prompts"))
        sidebar.addWidget(self._used_list, stretch=1)

        self._unused_list = QListWidget()
        sidebar.addWidget(QLabel("Available but unused"))
        sidebar.addWidget(self._unused_list, stretch=1)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #b23; font-weight: 600;")
        self._warning_label.hide()
        sidebar.addWidget(self._warning_label)
        sidebar.addStretch(1)

        content_row.addLayout(sidebar)
        main_layout.addLayout(content_row, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        main_layout.addWidget(buttons)

    def _create_editor(self, placeholder: str) -> QTextEdit:
        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setAcceptRichText(True)
        editor.setPlaceholderText(placeholder)
        editor.setLineWrapMode(QTextEdit.WidgetWidth)
        return editor

    def set_preview(self, preview: PromptPreview, *, required: Iterable[str] | None = None, optional: Iterable[str] | None = None) -> None:
        """Populate the dialog with prompt templates and placeholder metadata."""

        required_set = set(required or preview.required)
        optional_set = set(optional or preview.optional)

        analysis = analyse_prompts(
            preview.system_template,
            preview.user_template,
            available_values=preview.values,
            required_keys=required_set,
            optional_keys=optional_set,
        )

        css = build_preview_styles()

        def _html_or_placeholder(template: str, raw_renderer, preview_renderer) -> tuple[str, str]:
            if template.strip():
                raw_html = css + "<pre>" + raw_renderer(template) + "</pre>"
                preview_html = css + "<pre>" + preview_renderer(template) + "</pre>"
            else:
                blank = css + "<pre><em>No preview available</em></pre>"
                raw_html = preview_html = blank
            return raw_html, preview_html

        self._raw_system_html, _ = _html_or_placeholder(
            preview.system_template,
            lambda tpl: highlight_placeholders_raw(tpl, values=preview.values, required=required_set),
            lambda tpl: render_preview_html(tpl, values=preview.values, required=required_set),
        )
        self._raw_user_html, _ = _html_or_placeholder(
            preview.user_template,
            lambda tpl: highlight_placeholders_raw(tpl, values=preview.values, required=required_set),
            lambda tpl: render_preview_html(tpl, values=preview.values, required=required_set),
        )

        if preview.system_appendix.strip():
            appendix_html = (
                f"\n\n<hr><div style='font-weight:600; margin: 0.6rem 0 0.4rem 0;'>"
                f"{GENERATED_CITATION_APPENDIX_TITLE}</div><pre>{preview.system_appendix}</pre>"
            )
            self._raw_system_html += appendix_html
        if preview.user_appendix.strip():
            appendix_html = (
                f"\n\n<hr><div style='font-weight:600; margin: 0.6rem 0 0.4rem 0;'>"
                "Generated User Appendix</div><pre>"
                f"{preview.user_appendix}</pre>"
            )
            self._raw_user_html += appendix_html

        self._preview_system_html = css + "<pre>" + render_preview_html(
            preview.system_rendered,
            values=preview.values,
            required=required_set,
        ) + "</pre>"
        self._preview_user_html = css + "<pre>" + render_preview_html(
            preview.user_rendered,
            values=preview.values,
            required=required_set,
        ) + "</pre>"

        self._preview = preview
        self._analysis = analysis
        self._populate_usage_lists()
        self._update_views()

    def _populate_usage_lists(self) -> None:
        self._used_list.clear()
        self._unused_list.clear()
        if not self._analysis:
            return

        used_lines: list[str] = []
        missing_required = sorted(self._analysis.missing_required)
        missing_optional = sorted(self._analysis.missing_optional)

        for usage in self._analysis.usages:
            label = usage.name
            flags: list[str] = []
            if usage.required:
                flags.append("required")
            if usage.name in self._analysis.used:
                flags.append("used")
            if not usage.has_value:
                flags.append("missing")
            suffix = "" if not flags else f" ({', '.join(flags)})"
            item_text = f"{label}{suffix}"
            if usage.name in self._analysis.used:
                item = QListWidgetItem(item_text)
                if not usage.has_value:
                    item.setForeground(Qt.red)
                elif usage.required:
                    item.setForeground(Qt.darkGreen)
                self._used_list.addItem(item)
            else:
                item = QListWidgetItem(item_text)
                item.setForeground(Qt.darkGray)
                self._unused_list.addItem(item)

        warnings: list[str] = []
        if missing_required:
            warnings.append(
                "Missing required placeholders: " + ", ".join(f"{{{name}}}" for name in missing_required)
            )
        if missing_optional:
            warnings.append(
                "Missing optional placeholders: " + ", ".join(f"{{{name}}}" for name in missing_optional)
            )

        if warnings:
            self._warning_label.setText("\n".join(warnings))
            self._warning_label.show()
        else:
            self._warning_label.hide()

    def _update_views(self) -> None:
        if self._raw_button.isChecked():
            self._system_edit.setHtml(self._raw_system_html)
            self._user_edit.setHtml(self._raw_user_html)
        else:
            self._system_edit.setHtml(self._preview_system_html)
            self._user_edit.setHtml(self._preview_user_html)
        self._system_edit.moveCursor(QTextCursor.Start)
        self._user_edit.moveCursor(QTextCursor.Start)


__all__ = ["PromptPreviewDialog"]
