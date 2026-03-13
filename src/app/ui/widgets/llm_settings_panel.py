"""Shared provider/model/reasoning selector for LLM-backed workflows."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.core.llm_catalog import (
    LLMReasoningCapabilities,
    reset_gemini_model_cache,
    resolve_catalog_model,
)
from src.app.core.llm_operation_settings import (
    CatalogTransport,
    LLMOperationSettings,
    LLMReasoningSettings,
    default_provider_catalog_for_transport,
    provider_option_map,
)

_CUSTOM_MODEL_SENTINEL = "__custom_model__"


class LLMSettingsPanel(QWidget):
    """Reusable selector for shared LLM workflow settings."""

    settings_changed = Signal()

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        include_azure: bool = False,
        transport: CatalogTransport = "direct",
    ) -> None:
        super().__init__(parent)
        self._transport = transport
        self._include_azure = include_azure
        self._provider_options = ()
        self._provider_map = {}
        self._invalid_selection_message: str | None = None
        self._build_ui()
        self._reload_catalog()
        self._refresh_model_options()
        self._update_custom_state()
        self._update_reasoning_controls()
        self._update_model_details()

    @property
    def transport(self) -> CatalogTransport:
        return self._transport

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
            resolved = resolve_catalog_model(str(provider_id), custom_model, transport=self._transport)
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
                    reasoning=self._current_reasoning_settings(),
                ),
                None,
            )

        model_id = str(model_data or "").strip()
        if not model_id:
            return None, "Select a model before continuing."

        resolved = resolve_catalog_model(str(provider_id), model_id, transport=self._transport)
        if resolved is not None and resolved.context_window is None:
            context_window = int(self.custom_context_spin.value())
            if context_window <= 0:
                return None, "Enter a context window before continuing."
        else:
            context_window = None

        return (
            LLMOperationSettings(
                provider_id=str(provider_id),
                model_id=model_id,
                context_window=context_window,
                reasoning=self._current_reasoning_settings(),
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
        resolved = resolve_catalog_model(settings.provider_id, settings.model_id, transport=self._transport)
        needs_context_override = resolved is not None and resolved.context_window is None and settings.context_window is not None
        custom_mode = (settings.context_window is not None and not needs_context_override) or model_index == -1
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
            self.custom_context_spin.setValue(int(settings.context_window or 0))

        self._update_custom_state()
        self._update_reasoning_controls()
        self._set_reasoning_settings(settings.reasoning)
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
        self.refresh_models_button = QPushButton("Refresh")
        self.refresh_models_button.setAutoDefault(False)
        model_row.addWidget(self.refresh_models_button)
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

        reasoning_row = QHBoxLayout()
        reasoning_row.setContentsMargins(0, 0, 0, 0)
        reasoning_row.addWidget(QLabel("Reasoning:"))
        self.reasoning_state_combo = QComboBox()
        reasoning_row.addWidget(self.reasoning_state_combo)
        layout.addLayout(reasoning_row)

        effort_row = QHBoxLayout()
        effort_row.setContentsMargins(0, 0, 0, 0)
        self.reasoning_effort_label = QLabel("Effort:")
        self.reasoning_effort_combo = QComboBox()
        effort_row.addWidget(self.reasoning_effort_label)
        effort_row.addWidget(self.reasoning_effort_combo)
        layout.addLayout(effort_row)

        level_row = QHBoxLayout()
        level_row.setContentsMargins(0, 0, 0, 0)
        self.reasoning_level_label = QLabel("Level:")
        self.reasoning_level_combo = QComboBox()
        level_row.addWidget(self.reasoning_level_label)
        level_row.addWidget(self.reasoning_level_combo)
        layout.addLayout(level_row)

        budget_row = QHBoxLayout()
        budget_row.setContentsMargins(0, 0, 0, 0)
        self.reasoning_budget_label = QLabel("Budget:")
        self.reasoning_budget_spin = QSpinBox()
        self.reasoning_budget_spin.setRange(0, 256_000)
        self.reasoning_budget_spin.setSingleStep(1_024)
        self.reasoning_budget_spin.setSpecialValueText("Provider default")
        budget_row.addWidget(self.reasoning_budget_label)
        budget_row.addWidget(self.reasoning_budget_spin)
        budget_row.addWidget(QLabel("tokens"))
        budget_row.addStretch(1)
        layout.addLayout(budget_row)

        self.reasoning_note_label = QLabel("")
        self.reasoning_note_label.setWordWrap(True)
        self.reasoning_note_label.setStyleSheet("color: #666;")
        layout.addWidget(self.reasoning_note_label)

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
            "Reasoning controls come from the selected provider/model when available. "
            "Unavailable controls are hidden automatically."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        advanced_layout.addWidget(note)
        layout.addWidget(self.advanced_frame)

        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.custom_model_edit.textChanged.connect(self._on_custom_model_changed)
        self.custom_context_spin.valueChanged.connect(self._on_custom_context_changed)
        self.refresh_models_button.clicked.connect(self._refresh_models_clicked)
        self.reasoning_state_combo.currentIndexChanged.connect(self._on_reasoning_changed)
        self.reasoning_effort_combo.currentIndexChanged.connect(self._on_reasoning_changed)
        self.reasoning_level_combo.currentIndexChanged.connect(self._on_reasoning_changed)
        self.reasoning_budget_spin.valueChanged.connect(self._on_reasoning_changed)
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)

    def _reload_catalog(self) -> None:
        self._provider_options = default_provider_catalog_for_transport(
            include_azure=self._include_azure,
            transport=self._transport,
        )
        self._provider_map = provider_option_map(self._provider_options)
        self._populate_provider_options()

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
        self._update_reasoning_controls()
        self._update_model_details()
        self.settings_changed.emit()

    def _on_model_changed(self) -> None:
        self._invalid_selection_message = None
        self._update_custom_state()
        self._update_reasoning_controls()
        self._update_model_details()
        self.settings_changed.emit()

    def _on_custom_model_changed(self) -> None:
        self._invalid_selection_message = None
        self._update_reasoning_controls()
        self._update_model_details()
        self.settings_changed.emit()

    def _on_custom_context_changed(self) -> None:
        self._invalid_selection_message = None
        self._update_model_details()
        self.settings_changed.emit()

    def _update_custom_state(self) -> None:
        is_custom = self.model_combo.currentData() == _CUSTOM_MODEL_SENTINEL
        provider_id = str(self.provider_combo.currentData() or "").strip()
        model_id = self.custom_model_edit.text().strip() if is_custom else str(self.model_combo.currentData() or "").strip()
        resolved = resolve_catalog_model(provider_id, model_id, transport=self._transport) if provider_id and model_id else None
        needs_context = bool(is_custom or (resolved is not None and resolved.context_window is None))

        self.custom_model_label.setVisible(is_custom)
        self.custom_model_edit.setVisible(is_custom)
        self.custom_context_label.setVisible(needs_context)
        self.custom_context_spin.setVisible(needs_context)
        self.validation_label.setVisible(bool(self._invalid_selection_message))
        self.validation_label.setText(self._invalid_selection_message or "")

    def _refresh_models_clicked(self) -> None:
        current_provider = self.provider_combo.currentData()
        current_model = self.model_combo.currentData()
        reset_gemini_model_cache()
        self._reload_catalog()
        provider_index = self.provider_combo.findData(current_provider)
        if provider_index != -1:
            self.provider_combo.setCurrentIndex(provider_index)
        self._refresh_model_options()
        model_index = self.model_combo.findData(current_model)
        if model_index != -1:
            self.model_combo.setCurrentIndex(model_index)
        self._update_reasoning_controls()
        self._update_model_details()
        self.settings_changed.emit()

    def _on_reasoning_changed(self) -> None:
        self._update_reasoning_controls()
        self.settings_changed.emit()

    def _on_advanced_toggled(self, checked: bool) -> None:
        self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.advanced_frame.setVisible(checked)

    def _current_reasoning_capabilities(self) -> LLMReasoningCapabilities:
        provider_id = str(self.provider_combo.currentData() or "").strip()
        model_data = self.model_combo.currentData()
        custom_mode = model_data == _CUSTOM_MODEL_SENTINEL
        model_id = self.custom_model_edit.text().strip() if custom_mode else str(model_data or "").strip()
        option = resolve_catalog_model(provider_id, model_id, transport=self._transport) if provider_id and model_id else None
        if option is None:
            return LLMReasoningCapabilities()
        return option.reasoning_capabilities

    def _current_reasoning_settings(self) -> LLMReasoningSettings:
        capabilities = self._current_reasoning_capabilities()
        enabled = capabilities.supports_reasoning_controls and (
            capabilities.can_disable_reasoning is False
            or str(self.reasoning_state_combo.currentData() or "off") == "on"
        )
        effort = str(self.reasoning_effort_combo.currentData() or "").strip() or None
        level = str(self.reasoning_level_combo.currentData() or "").strip() or None
        budget_tokens = int(self.reasoning_budget_spin.value()) or None
        return LLMReasoningSettings(
            state="on" if enabled else "off",
            effort=effort,
            level=level,
            budget_tokens=budget_tokens,
        )

    def _set_reasoning_settings(self, settings: LLMReasoningSettings) -> None:
        self._update_reasoning_controls()
        desired_state = settings.state or ("on" if settings.enabled else "off")
        state_index = self.reasoning_state_combo.findData(desired_state)
        if state_index != -1:
            self.reasoning_state_combo.blockSignals(True)
            self.reasoning_state_combo.setCurrentIndex(state_index)
            self.reasoning_state_combo.blockSignals(False)
        effort_index = self.reasoning_effort_combo.findData(settings.effort)
        if effort_index != -1:
            self.reasoning_effort_combo.blockSignals(True)
            self.reasoning_effort_combo.setCurrentIndex(effort_index)
            self.reasoning_effort_combo.blockSignals(False)
        level_index = self.reasoning_level_combo.findData(settings.level)
        if level_index != -1:
            self.reasoning_level_combo.blockSignals(True)
            self.reasoning_level_combo.setCurrentIndex(level_index)
            self.reasoning_level_combo.blockSignals(False)
        self.reasoning_budget_spin.blockSignals(True)
        self.reasoning_budget_spin.setValue(int(settings.budget_tokens or 0))
        self.reasoning_budget_spin.blockSignals(False)

    def _update_reasoning_controls(self) -> None:
        capabilities = self._current_reasoning_capabilities()

        self.reasoning_state_combo.blockSignals(True)
        self.reasoning_state_combo.clear()
        if not capabilities.supports_reasoning_controls:
            self.reasoning_state_combo.addItem("Not supported", "off")
            self.reasoning_state_combo.setEnabled(False)
        else:
            self.reasoning_state_combo.addItem("Off", "off")
            self.reasoning_state_combo.addItem("On", "on")
            self.reasoning_state_combo.setEnabled(capabilities.can_disable_reasoning)
            if not capabilities.can_disable_reasoning:
                index = self.reasoning_state_combo.findData("on")
                self.reasoning_state_combo.setCurrentIndex(index)
        self.reasoning_state_combo.blockSignals(False)

        enabled = bool(capabilities.supports_reasoning_controls) and (
            not capabilities.can_disable_reasoning
            or str(self.reasoning_state_combo.currentData() or "off") == "on"
        )

        self.reasoning_effort_combo.blockSignals(True)
        self.reasoning_effort_combo.clear()
        for effort in capabilities.allowed_efforts:
            self.reasoning_effort_combo.addItem(effort.title(), effort)
        if self.reasoning_effort_combo.count() > 0 and self.reasoning_effort_combo.currentIndex() < 0:
            self.reasoning_effort_combo.setCurrentIndex(0)
        self.reasoning_effort_combo.blockSignals(False)
        show_effort = enabled and bool(capabilities.allowed_efforts)
        self.reasoning_effort_label.setVisible(show_effort)
        self.reasoning_effort_combo.setVisible(show_effort)

        self.reasoning_level_combo.blockSignals(True)
        self.reasoning_level_combo.clear()
        for level in capabilities.allowed_levels:
            self.reasoning_level_combo.addItem(level.title(), level)
        if self.reasoning_level_combo.count() > 0 and self.reasoning_level_combo.currentIndex() < 0:
            self.reasoning_level_combo.setCurrentIndex(0)
        self.reasoning_level_combo.blockSignals(False)
        show_level = enabled and bool(capabilities.allowed_levels)
        self.reasoning_level_label.setVisible(show_level)
        self.reasoning_level_combo.setVisible(show_level)

        if capabilities.budget_min is not None or capabilities.budget_max is not None:
            self.reasoning_budget_spin.setRange(
                int(capabilities.budget_min or 0),
                int(capabilities.budget_max or 256_000),
            )
            self.reasoning_budget_spin.setSingleStep(int(capabilities.budget_step or 1_024))
        show_budget = enabled and "budget" in capabilities.controls
        self.reasoning_budget_label.setVisible(show_budget)
        self.reasoning_budget_spin.setVisible(show_budget)

        note = capabilities.notes or ""
        if not capabilities.supports_reasoning_controls:
            note = "This model does not expose provider-native reasoning controls."
        self.reasoning_note_label.setVisible(bool(note))
        self.reasoning_note_label.setText(note)

    def _update_model_details(self) -> None:
        provider_id = str(self.provider_combo.currentData() or "").strip()
        model_data = self.model_combo.currentData()
        custom_mode = model_data == _CUSTOM_MODEL_SENTINEL
        model_id = self.custom_model_edit.text().strip() if custom_mode else str(model_data or "").strip()

        if not provider_id or not model_id:
            self.model_details_label.clear()
            return

        option = resolve_catalog_model(provider_id, model_id, transport=self._transport)
        if option is None:
            self.model_details_label.setText("Catalog details unavailable for this model.")
            return

        details: list[str] = []
        if option.lifecycle_status != "stable":
            details.append(f"Status: {option.lifecycle_status.title()}")
        if option.context_window:
            details.append(f"Context: {option.context_window:,} tokens")
        else:
            details.append("Context: enter manually")
        if option.max_output_tokens:
            details.append(f"Max output: {option.max_output_tokens:,}")
        if option.input_price_label:
            details.append(f"Input: {option.input_price_label}")
        if option.output_price_label:
            details.append(f"Output: {option.output_price_label}")
        if option.provenance.availability:
            details.append(f"Source: {option.provenance.availability}")
        if self.custom_context_spin.isVisible() and self.custom_context_spin.value() <= 0 and option.context_window is None:
            self.model_details_label.setText("Enter a context window to use this model.")
            return
        self.model_details_label.setText(" | ".join(details) if details else "Catalog details unavailable for this model.")


__all__ = ["LLMSettingsPanel"]
