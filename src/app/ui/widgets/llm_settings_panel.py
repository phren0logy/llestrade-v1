"""Shared provider/model/reasoning selector for LLM-backed workflows."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.core.llm_operation_settings import (
    LLMOperationSettings,
    default_provider_catalog,
    provider_option_map,
)
from src.app.core.llm_catalog import resolve_catalog_model

_CUSTOM_MODEL_SENTINEL = "__custom_model__"


class LLMSettingsPanel(QWidget):
    """Reusable selector for shared LLM workflow settings."""

    settings_changed = Signal()

    def __init__(self, *, parent: QWidget | None = None, include_azure: bool = False) -> None:
        super().__init__(parent)
        self._provider_options = default_provider_catalog(include_azure=include_azure)
        self._provider_map = provider_option_map(self._provider_options)
        self._invalid_selection_message: str | None = None
        self._build_ui()
        self._populate_provider_options()
        self._refresh_model_options()
        self._update_custom_state()
        self._update_model_details()

    def current_settings(self) -> tuple[LLMOperationSettings | None, str | None]:
        if self._invalid_selection_message:
            return None, self._invalid_selection_message

        provider_id = self.provider_combo.currentData()
        if not provider_id:
            return None, "Select a provider before continuing."

        model_data = self.model_combo.currentData()
        if model_data == _CUSTOM_MODEL_SENTINEL:
            custom_model = self.custom_model_edit.text().strip()
            if not custom_model:
                return None, "Enter a custom model id before continuing."
            resolved = resolve_catalog_model(str(provider_id), custom_model)
            context_window = int(self.custom_context_spin.value())
            if context_window <= 0:
                if resolved and resolved.context_window:
                    context_window = int(resolved.context_window)
                else:
                    return None, "Enter a context window before continuing."
            return (
                LLMOperationSettings(
                    provider_id=str(provider_id),
                    model_id=custom_model,
                    context_window=context_window,
                    use_reasoning=self.reasoning_checkbox.isChecked(),
                ),
                None,
            )

        model_id = str(model_data or "").strip()
        if not model_id:
            return None, "Select a model before continuing."

        return (
            LLMOperationSettings(
                provider_id=str(provider_id),
                model_id=model_id,
                context_window=None,
                use_reasoning=self.reasoning_checkbox.isChecked(),
            ),
            None,
        )

    def set_settings(self, settings: LLMOperationSettings) -> None:
        provider_index = self.provider_combo.findData(settings.provider_id)
        self.provider_combo.blockSignals(True)
        self.provider_combo.setCurrentIndex(provider_index)
        self.provider_combo.blockSignals(False)

        self._refresh_model_options()

        self._invalid_selection_message = None
        model_index = self.model_combo.findData(settings.model_id)
        custom_mode = settings.context_window is not None or model_index == -1
        if provider_index == -1:
            self.provider_combo.setCurrentIndex(-1)
            custom_mode = True
            self._invalid_selection_message = (
                "The saved provider is no longer supported in the shared selector. "
                "Choose a new provider before continuing."
            )
        if custom_mode:
            custom_index = self.model_combo.findData(_CUSTOM_MODEL_SENTINEL)
            if custom_index != -1:
                self.model_combo.setCurrentIndex(custom_index)
            self.custom_model_edit.setText(settings.model_id)
            self.custom_context_spin.setValue(int(settings.context_window or 0))
            if model_index == -1 and settings.context_window is None and provider_index != -1:
                self._invalid_selection_message = (
                    "The saved model is no longer available. Choose a new preset or enter "
                    "a custom context window before continuing."
                )
        elif model_index != -1:
            self.model_combo.setCurrentIndex(model_index)
            self.custom_model_edit.clear()
            self.custom_context_spin.setValue(0)

        self.reasoning_checkbox.setChecked(bool(settings.use_reasoning))
        self._update_custom_state()
        self._update_model_details()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        provider_row = QHBoxLayout()
        provider_row.setContentsMargins(0, 0, 0, 0)
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.setEditable(False)
        provider_row.addWidget(self.provider_combo)
        layout.addLayout(provider_row)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(0, 0, 0, 0)
        self.custom_model_label = QLabel("Custom model id:")
        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("Enter a provider-specific model id")
        custom_row.addWidget(self.custom_model_label)
        custom_row.addWidget(self.custom_model_edit)
        layout.addLayout(custom_row)

        context_row = QHBoxLayout()
        context_row.setContentsMargins(0, 0, 0, 0)
        self.custom_context_label = QLabel("Context window:")
        self.custom_context_spin = QSpinBox()
        self.custom_context_spin.setRange(0, 4_000_000)
        self.custom_context_spin.setSingleStep(1_000)
        self.custom_context_spin.setSpecialValueText("Use catalog value")
        self.custom_context_spin.setValue(0)
        context_row.addWidget(self.custom_context_label)
        context_row.addWidget(self.custom_context_spin)
        context_row.addWidget(QLabel("tokens"))
        context_row.addStretch(1)
        layout.addLayout(context_row)

        self.model_details_label = QLabel("")
        self.model_details_label.setWordWrap(True)
        self.model_details_label.setStyleSheet("color: #666;")
        layout.addWidget(self.model_details_label)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #8a4b00;")
        self.validation_label.setVisible(False)
        layout.addWidget(self.validation_label)

        self.reasoning_checkbox = QCheckBox("Reasoning")
        layout.addWidget(self.reasoning_checkbox)

        self.advanced_toggle = QToolButton(self)
        self.advanced_toggle.setText("Advanced reasoning settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        layout.addWidget(self.advanced_toggle)

        self.advanced_frame = QFrame(self)
        self.advanced_frame.setFrameShape(QFrame.StyledPanel)
        self.advanced_frame.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(12, 8, 12, 12)
        advanced_layout.setSpacing(4)
        note = QLabel(
            "Reasoning uses safe provider defaults when enabled. "
            "Provider-specific advanced controls are not exposed yet."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        advanced_layout.addWidget(note)
        layout.addWidget(self.advanced_frame)

        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.custom_model_edit.textChanged.connect(self._on_custom_model_changed)
        self.custom_context_spin.valueChanged.connect(self._on_custom_context_changed)
        self.reasoning_checkbox.toggled.connect(self.settings_changed)
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)

    def _populate_provider_options(self) -> None:
        self.provider_combo.clear()
        for option in self._provider_options:
            self.provider_combo.addItem(option.label, option.provider_id)

    def _refresh_model_options(self) -> None:
        provider_id = str(self.provider_combo.currentData() or "").strip()
        option = self._provider_map.get(provider_id)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if option is not None:
            for model in option.models:
                self.model_combo.addItem(model.label, model.model_id)
        self.model_combo.addItem("Custom model…", _CUSTOM_MODEL_SENTINEL)
        if self.model_combo.count() > 0 and self.model_combo.currentIndex() < 0:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

    def _on_provider_changed(self) -> None:
        self._invalid_selection_message = None
        self._refresh_model_options()
        self._update_custom_state()
        self._update_model_details()
        self.settings_changed.emit()

    def _on_model_changed(self) -> None:
        self._invalid_selection_message = None
        self._update_custom_state()
        self._update_model_details()
        self.settings_changed.emit()

    def _on_custom_model_changed(self) -> None:
        self._invalid_selection_message = None
        self._update_model_details()
        self.settings_changed.emit()

    def _on_custom_context_changed(self) -> None:
        self._invalid_selection_message = None
        self._update_model_details()
        self.settings_changed.emit()

    def _update_custom_state(self) -> None:
        is_custom = self.model_combo.currentData() == _CUSTOM_MODEL_SENTINEL
        for widget in (
            self.custom_model_label,
            self.custom_model_edit,
            self.custom_context_label,
            self.custom_context_spin,
        ):
            widget.setVisible(is_custom)
        self.validation_label.setVisible(bool(self._invalid_selection_message))
        self.validation_label.setText(self._invalid_selection_message or "")

    def _on_advanced_toggled(self, checked: bool) -> None:
        self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.advanced_frame.setVisible(checked)

    def _update_model_details(self) -> None:
        provider_id = str(self.provider_combo.currentData() or "").strip()
        model_data = self.model_combo.currentData()
        custom_mode = model_data == _CUSTOM_MODEL_SENTINEL
        model_id = self.custom_model_edit.text().strip() if custom_mode else str(model_data or "").strip()

        if not provider_id or not model_id:
            self.model_details_label.clear()
            return

        option = resolve_catalog_model(provider_id, model_id)
        if option is None:
            self.model_details_label.setText("Catalog details unavailable for this model.")
            return

        details: list[str] = []
        if option.context_window:
            details.append(f"Context: {option.context_window:,} tokens")
        if option.input_price_label:
            details.append(f"Input: {option.input_price_label}")
        if option.output_price_label:
            details.append(f"Output: {option.output_price_label}")
        if custom_mode and self.custom_context_spin.value() <= 0 and option.context_window is None:
            self.model_details_label.setText("Enter a context window to use this custom model.")
            return
        self.model_details_label.setText(" | ".join(details) if details else "Catalog details unavailable for this model.")


__all__ = ["LLMSettingsPanel"]
