"""Presentation widget for the reports tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.widgets import LLMSettingsPanel


class CollapsibleGroupBox(QWidget):
    """Simple collapsible container with a caption row and framed body."""

    def __init__(self, title: str, *, parent: QWidget | None = None, collapsed: bool = False) -> None:
        super().__init__(parent)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: 600; }")

        self._content = QFrame(self)
        self._content.setFrameShape(QFrame.StyledPanel)
        self._content.setFrameShadow(QFrame.Plain)
        self._content.setVisible(not collapsed)

        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 12, 12)
        self._content_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

        self._toggle.toggled.connect(self._on_toggled)

    def content_layout(self) -> QVBoxLayout:
        """Expose the layout hosting the section contents."""

        return self._content_layout

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)


class ReportsTab(QWidget):
    """Encapsulate the UI elements used by the reports workflow."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.inputs_tree = QTreeWidget()
        self.inputs_tree.setColumnCount(2)
        self.inputs_tree.setHeaderLabels(["Input", "Project Path"])
        self.inputs_tree.setUniformRowHeights(True)
        self.inputs_tree.setSelectionMode(QTreeWidget.NoSelection)

        self.llm_settings_panel = LLMSettingsPanel(parent=self)
        self.provider_combo = self.llm_settings_panel.provider_combo
        self.model_combo = self.llm_settings_panel.model_combo
        self.custom_model_label = self.llm_settings_panel.custom_model_label
        self.custom_model_edit = self.llm_settings_panel.custom_model_edit
        self.custom_context_label = self.llm_settings_panel.custom_context_label
        self.custom_context_spin = self.llm_settings_panel.custom_context_spin
        self.reasoning_checkbox = self.llm_settings_panel.reasoning_checkbox
        self.advanced_reasoning_toggle = self.llm_settings_panel.advanced_toggle

        self.template_edit = QLineEdit()
        self.template_browse_button = QPushButton("Browse…")

        self.transcript_edit = QLineEdit()
        self.transcript_browse_button = QPushButton("Browse…")

        self.generation_user_prompt_edit = QLineEdit()
        self.generation_user_prompt_browse = QPushButton("Browse…")
        self.generation_user_prompt_preview = QPushButton("Preview generation prompt")

        self.generation_system_prompt_edit = QLineEdit()
        self.generation_system_prompt_browse = QPushButton("Browse…")
        self.refinement_user_prompt_edit = QLineEdit()
        self.refinement_user_prompt_browse = QPushButton("Browse…")
        self.refinement_user_prompt_preview = QPushButton("Preview refinement prompt")

        self.refinement_system_prompt_edit = QLineEdit()
        self.refinement_system_prompt_browse = QPushButton("Browse…")

        self.refine_draft_edit = QLineEdit()
        self.refine_draft_browse_button = QPushButton("Browse…")

        self.generate_draft_button = QPushButton("Generate Draft")
        self.run_refinement_button = QPushButton("Run Refinement")
        self.open_reports_button = QPushButton("Open Reports Folder")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(140)

        self.history_list = QTreeWidget()
        self.history_list.setColumnCount(3)
        self.history_list.setHeaderLabels(["Generated", "Model", "Outputs"])

        self.open_draft_button = QPushButton("Open Draft")
        self.open_refined_button = QPushButton("Open Refined")
        self.open_reasoning_button = QPushButton("Open Reasoning")
        self.open_manifest_button = QPushButton("Open Manifest")
        self.open_inputs_button = QPushButton("Open Inputs")

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        inputs_group = QGroupBox("Select Inputs")
        inputs_layout = QVBoxLayout(inputs_group)
        inputs_layout.addWidget(self.inputs_tree)
        top_layout.addWidget(inputs_group, 1)

        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(6)

        config_layout.addWidget(self.llm_settings_panel)

        template_label = QLabel("Template (required):")
        config_layout.addWidget(template_label)
        template_row = QHBoxLayout()
        template_row.setContentsMargins(0, 0, 0, 0)
        template_row.addWidget(self.template_edit)
        template_row.addWidget(self.template_browse_button)
        config_layout.addLayout(template_row)

        transcript_label = QLabel("Transcript (optional):")
        config_layout.addWidget(transcript_label)
        transcript_row = QHBoxLayout()
        transcript_row.setContentsMargins(0, 0, 0, 0)
        transcript_row.addWidget(self.transcript_edit)
        transcript_row.addWidget(self.transcript_browse_button)
        config_layout.addLayout(transcript_row)

        generation_group = CollapsibleGroupBox("Generate Draft", parent=self)
        generation_layout = generation_group.content_layout()

        generation_system_label = QLabel("Generation system prompt:")
        generation_layout.addWidget(generation_system_label)
        generation_system_row = QHBoxLayout()
        generation_system_row.setContentsMargins(0, 0, 0, 0)
        generation_system_row.addWidget(self.generation_system_prompt_edit)
        generation_system_row.addWidget(self.generation_system_prompt_browse)
        generation_layout.addLayout(generation_system_row)

        generation_user_label = QLabel("Generation user prompt:")
        generation_layout.addWidget(generation_user_label)
        generation_user_row = QHBoxLayout()
        generation_user_row.setContentsMargins(0, 0, 0, 0)
        generation_user_row.addWidget(self.generation_user_prompt_edit)
        generation_user_row.addWidget(self.generation_user_prompt_browse)
        generation_layout.addLayout(generation_user_row)
        generation_layout.addWidget(self.generation_user_prompt_preview)

        generation_button_row = QHBoxLayout()
        generation_button_row.setContentsMargins(0, 0, 0, 0)
        generation_button_row.addWidget(self.generate_draft_button)
        generation_button_row.addStretch()
        generation_layout.addLayout(generation_button_row)
        self.generate_estimate_label = QLabel("Est. cost unavailable")
        self.generate_estimate_label.setStyleSheet("color: #666;")
        generation_layout.addWidget(self.generate_estimate_label)

        config_layout.addWidget(generation_group)

        refinement_group = CollapsibleGroupBox("Refine Draft", parent=self)
        refinement_layout = refinement_group.content_layout()

        refinement_system_label = QLabel("Refinement system prompt:")
        refinement_layout.addWidget(refinement_system_label)
        refinement_system_row = QHBoxLayout()
        refinement_system_row.setContentsMargins(0, 0, 0, 0)
        refinement_system_row.addWidget(self.refinement_system_prompt_edit)
        refinement_system_row.addWidget(self.refinement_system_prompt_browse)
        refinement_layout.addLayout(refinement_system_row)

        refinement_user_label = QLabel("Refinement user prompt:")
        refinement_layout.addWidget(refinement_user_label)
        refinement_user_row = QHBoxLayout()
        refinement_user_row.setContentsMargins(0, 0, 0, 0)
        refinement_user_row.addWidget(self.refinement_user_prompt_edit)
        refinement_user_row.addWidget(self.refinement_user_prompt_browse)
        refinement_layout.addLayout(refinement_user_row)
        refinement_layout.addWidget(self.refinement_user_prompt_preview)

        refinement_draft_label = QLabel("Existing draft for refinement:")
        refinement_layout.addWidget(refinement_draft_label)
        refinement_draft_row = QHBoxLayout()
        refinement_draft_row.setContentsMargins(0, 0, 0, 0)
        refinement_draft_row.addWidget(self.refine_draft_edit)
        refinement_draft_row.addWidget(self.refine_draft_browse_button)
        refinement_layout.addLayout(refinement_draft_row)

        refinement_button_row = QHBoxLayout()
        refinement_button_row.setContentsMargins(0, 0, 0, 0)
        refinement_button_row.addWidget(self.run_refinement_button)
        refinement_button_row.addStretch()
        refinement_layout.addLayout(refinement_button_row)
        self.refinement_estimate_label = QLabel("Est. cost unavailable")
        self.refinement_estimate_label.setStyleSheet("color: #666;")
        refinement_layout.addWidget(self.refinement_estimate_label)

        config_layout.addWidget(refinement_group)

        top_layout.addWidget(config_group, 1)
        layout.addLayout(top_layout)

        open_row = QHBoxLayout()
        open_row.setContentsMargins(0, 0, 0, 0)
        open_row.addWidget(self.open_reports_button)
        open_row.addStretch()
        layout.addLayout(open_row)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text)

        history_group = QGroupBox("Recent Reports")
        history_layout = QVBoxLayout(history_group)
        history_layout.addWidget(self.history_list)

        history_buttons = QHBoxLayout()
        history_buttons.addWidget(self.open_draft_button)
        history_buttons.addWidget(self.open_refined_button)
        history_buttons.addWidget(self.open_reasoning_button)
        history_buttons.addWidget(self.open_manifest_button)
        history_buttons.addWidget(self.open_inputs_button)
        history_buttons.addStretch()
        history_layout.addLayout(history_buttons)

        layout.addWidget(history_group)


__all__ = ["ReportsTab"]
