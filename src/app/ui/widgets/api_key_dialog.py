"""
API key configuration dialog for the new UI.
"""

import logging
import json
from typing import Any, Dict
from urllib.parse import urlparse

from src.app.core.feature_flags import FeatureFlags
from src.app.core.llm_catalog import (
    refresh_gateway_provider_catalog,
    reset_provider_catalog_cache,
)
from src.app.core.llm_operation_settings import default_provider_catalog_for_transport
from src.app.workers.llm_backend import (
    PydanticAIGatewayBackend,
    reset_gateway_access_check_cache,
)

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QGroupBox,
    QDialogButtonBox, QMessageBox, QScrollArea,
    QWidget, QTabWidget, QCheckBox, QComboBox, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl


class APIKeyDialog(QDialog):
    """Dialog for configuring API keys and service endpoints."""

    _OBSERVABILITY_CONTENT_POLICIES = (
        ("Unredacted", "unredacted"),
        ("Redacted", "redacted"),
    )
    _OBSERVABILITY_TARGETS = (
        ("Local Phoenix", "phoenix_local"),
        ("OTLP Endpoint", "otlp_http"),
    )
    _MASKED_SECRET = "*" * 20
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._feature_flags = FeatureFlags.from_settings(settings)
        self._gateway_transport_available = self._feature_flags.pydantic_ai_gateway_enabled
        self._transport_restart_notice = (
            "LLM transport changes apply the next time you reopen the project or restart the app."
        )
        
        self.setWindowTitle("Configure API Keys & Services")
        self.setModal(True)
        self.resize(600, 700)
        
        self.api_fields: Dict[str, QLineEdit] = {}
        self.config_fields: Dict[str, QLineEdit] = {}
        
        self.setup_ui()
        self.load_keys()
    
    def setup_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Description
        desc = QLabel(
            "Configure your API keys and service endpoints. "
            "Keys are stored securely in your system's keychain."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # LLM Providers tab
        llm_tab = self._create_llm_tab()
        tabs.addTab(llm_tab, "LLM Providers")
        
        # Azure Services tab
        azure_tab = self._create_azure_tab()
        tabs.addTab(azure_tab, "Azure Services")
        
        # Observability tab
        observability_tab = self._create_observability_tab()
        tabs.addTab(observability_tab, "Observability")
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.save_keys)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _create_llm_tab(self) -> QWidget:
        """Create the LLM providers tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scroll area for providers
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # LLM Providers
        providers = [
            ("Anthropic (Claude)", "anthropic", "https://console.anthropic.com/"),
            ("Google Gemini", "gemini", "https://makersuite.google.com/app/apikey"),
        ]

        for display_name, key, url in providers:
            group = QGroupBox(display_name)
            group_layout = QVBoxLayout(group)
            
            # API key input
            key_layout = QHBoxLayout()

            field, show_btn = self._create_secret_field(
                key,
                f"Enter {display_name} API key...",
            )
            key_layout.addWidget(field)
            key_layout.addWidget(show_btn)
            
            group_layout.addLayout(key_layout)
            
            # Help text with URL
            help_text = QLabel(f'<a href="{url}">Get API key →</a>')
            help_text.setOpenExternalLinks(True)
            help_text.setStyleSheet("color: #1976d2;")
            group_layout.addWidget(help_text)

            scroll_layout.addWidget(group)

        gateway_group = QGroupBox("Pydantic AI Gateway App Key")
        gateway_layout = QFormLayout(gateway_group)

        self.transport_mode_combo = QComboBox()
        self.transport_mode_combo.addItem("Direct provider requests", "direct")
        gateway_index = self.transport_mode_combo.count()
        self.transport_mode_combo.addItem("Pydantic AI Gateway (PAIG)", "gateway")
        if not self._gateway_transport_available:
            item = self.transport_mode_combo.model().item(gateway_index)
            if item is not None:
                item.setEnabled(False)
        gateway_layout.addRow("LLM Transport:", self.transport_mode_combo)

        transport_help = QLabel(
            "Direct is the recommended default for large bulk runs. "
            "PAIG remains available as an explicit opt-in."
        )
        transport_help.setWordWrap(True)
        gateway_layout.addRow("", transport_help)

        self.transport_mode_note = QLabel(self._transport_restart_notice)
        self.transport_mode_note.setWordWrap(True)
        self.transport_mode_note.setStyleSheet("color: #666;")
        gateway_layout.addRow("", self.transport_mode_note)

        self.gateway_unavailable_note = QLabel(
            "Pydantic AI Gateway is currently unavailable in this environment, so direct transport will be used."
        )
        self.gateway_unavailable_note.setWordWrap(True)
        self.gateway_unavailable_note.setStyleSheet("color: #8a6d3b;")
        self.gateway_unavailable_note.setVisible(not self._gateway_transport_available)
        gateway_layout.addRow("", self.gateway_unavailable_note)

        gateway_info = QLabel(
            "This app key is the credential used for provider/model selections whenever gateway mode is enabled. "
            "Set a custom base URL only for a self-hosted gateway deployment."
        )
        gateway_info.setWordWrap(True)
        gateway_layout.addRow(gateway_info)

        gateway_key_row = QHBoxLayout()
        self.gateway_api_key, gateway_show_btn = self._create_secret_field(
            "pydantic_ai_gateway",
            "Enter Pydantic AI Gateway App Key...",
        )
        gateway_key_row.addWidget(self.gateway_api_key)
        gateway_key_row.addWidget(gateway_show_btn)
        gateway_layout.addRow("App Key:", gateway_key_row)

        self.gateway_base_url = QLineEdit()
        self.gateway_base_url.setPlaceholderText("https://gateway.example.com")
        gateway_layout.addRow("Base URL:", self.gateway_base_url)
        self.config_fields["gateway_base_url"] = self.gateway_base_url

        self.gateway_route = QLineEdit()
        self.gateway_route.setPlaceholderText("Optional routing group or route override")
        gateway_layout.addRow("Route:", self.gateway_route)
        self.config_fields["gateway_route"] = self.gateway_route

        gateway_help = QLabel(
            "Use the custom domain from your self-hosted gateway deployment, for example "
            "<code>https://gateway.example.com</code>. Set a route only if your self-hosted "
            "gateway uses named routing groups or a non-default route."
        )
        gateway_help.setWordWrap(True)
        gateway_help.setTextFormat(Qt.RichText)
        gateway_layout.addRow("", gateway_help)

        scroll_layout.addWidget(gateway_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        return widget
    
    def _create_azure_tab(self) -> QWidget:
        """Create the Azure services tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Azure OpenAI Settings
        openai_group = QGroupBox("Azure OpenAI Configuration")
        openai_layout = QFormLayout(openai_group)

        ak_row = QHBoxLayout()
        self.azure_openai_key, ak_show_btn = self._create_secret_field(
            "azure_openai",
            "Enter Azure OpenAI API key…",
        )
        ak_row.addWidget(self.azure_openai_key)
        ak_row.addWidget(ak_show_btn)
        openai_layout.addRow("API Key:", ak_row)

        self.azure_endpoint = QLineEdit()
        self.azure_endpoint.setPlaceholderText("https://your-resource.openai.azure.com/")
        openai_layout.addRow("Endpoint:", self.azure_endpoint)
        self.config_fields["azure_endpoint"] = self.azure_endpoint

        self.azure_deployment = QLineEdit()
        self.azure_deployment.setPlaceholderText("e.g., gpt-4.1")
        openai_layout.addRow("Deployment Name:", self.azure_deployment)
        self.config_fields["azure_deployment"] = self.azure_deployment

        self.azure_api_version = QLineEdit()
        self.azure_api_version.setPlaceholderText("e.g., 2025-01-01-preview")
        openai_layout.addRow("API Version:", self.azure_api_version)
        self.config_fields["azure_api_version"] = self.azure_api_version

        help_text_openai = QLabel('<a href="https://portal.azure.com/">Configure in Azure Portal →</a>')
        help_text_openai.setOpenExternalLinks(True)
        help_text_openai.setStyleSheet("color: #1976d2;")
        openai_layout.addRow("", help_text_openai)

        layout.addWidget(openai_group)

        layout.addStretch()
        return widget

    def _create_secret_field(self, provider: str, placeholder: str) -> tuple[QLineEdit, QPushButton]:
        field = QLineEdit()
        field.setEchoMode(QLineEdit.Password)
        field.setPlaceholderText(placeholder)
        field.setProperty("has_saved_key", False)
        field.setProperty("saved_key_value", "")
        field.setProperty("key_dirty", False)
        field.textEdited.connect(lambda _text, f=field: self._mark_secret_field_dirty(f))
        self.api_fields[provider] = field

        show_btn = QPushButton("Show")
        show_btn.setCheckable(True)
        show_btn.setMaximumWidth(60)
        show_btn.toggled.connect(
            lambda checked, f=field, b=show_btn: self._toggle_secret_visibility(f, b, checked)
        )
        return field, show_btn

    def _mark_secret_field_dirty(self, field: QLineEdit) -> None:
        field.setProperty("key_dirty", True)

    def _set_secret_field_value(self, field: QLineEdit, value: str) -> None:
        field.blockSignals(True)
        field.setText(value)
        field.blockSignals(False)

    def _toggle_secret_visibility(self, field: QLineEdit, button: QPushButton, checked: bool) -> None:
        saved_value = str(field.property("saved_key_value") or "")
        current_text = field.text()

        if checked:
            if (
                field.property("has_saved_key")
                and not field.property("key_dirty")
                and current_text == self._MASKED_SECRET
            ):
                self._set_secret_field_value(field, saved_value)
            field.setEchoMode(QLineEdit.Normal)
            button.setText("Hide")
            return

        if (
            field.property("has_saved_key")
            and not field.property("key_dirty")
            and current_text == saved_value
        ):
            self._set_secret_field_value(field, self._MASKED_SECRET)
        field.setEchoMode(QLineEdit.Password)
        button.setText("Show")
    
    def _open_phoenix_ui(self):
        """Open Phoenix UI in browser."""
        port = self.observability_phoenix_port.text() or "6006"
        QDesktopServices.openUrl(QUrl(f"http://localhost:{port}"))

    def _sync_observability_target_fields(self) -> None:
        target = self.observability_target.currentData() or "phoenix_local"
        is_phoenix = target == "phoenix_local"
        self.observability_phoenix_port_row.setVisible(is_phoenix)
        self.observability_open_phoenix_button.setVisible(is_phoenix)
        self.observability_otlp_endpoint_row.setVisible(not is_phoenix)
        self.observability_otlp_headers_row.setVisible(not is_phoenix)

    def _create_observability_tab(self) -> QWidget:
        """Create the generic observability tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel(
            "Configure OTEL-based tracing for model calls and worker spans. "
            "Use Local Phoenix for private local tracing, or point the app at a generic OTLP HTTP endpoint."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        observability_group = QGroupBox("Observability Configuration")
        observability_layout = QVBoxLayout(observability_group)

        self.observability_enabled = QCheckBox("Enable Observability")
        self.observability_enabled.setToolTip(
            "When enabled, worker spans and instrumented model calls are exported via OpenTelemetry."
        )
        observability_layout.addWidget(self.observability_enabled)

        form = QFormLayout()
        self.observability_target = QComboBox()
        for label, value in self._OBSERVABILITY_TARGETS:
            self.observability_target.addItem(label, value)
        form.addRow("Target:", self.observability_target)

        self.observability_project = QLineEdit()
        self.observability_project.setText("forensic-report-drafter")
        self.observability_project.setPlaceholderText("Project name for organizing traces")
        form.addRow("Project Name:", self.observability_project)

        self.observability_content_policy = QComboBox()
        for label, value in self._OBSERVABILITY_CONTENT_POLICIES:
            self.observability_content_policy.addItem(label, value)
        self.observability_content_policy.setToolTip(
            "Choose whether model instrumentation includes full prompt/response text."
        )
        form.addRow("Content Policy:", self.observability_content_policy)

        self.observability_include_binary_content = QCheckBox("Include binary/multimodal content")
        self.observability_include_binary_content.setToolTip(
            "Include binary or multimodal payload bodies in model instrumentation."
        )
        observability_layout.addLayout(form)
        observability_layout.addWidget(self.observability_include_binary_content)

        self.observability_policy_note = QLabel(
            "Use unredacted traces only for trusted local environments. Remote OTLP endpoints should normally use redacted content."
        )
        self.observability_policy_note.setWordWrap(True)
        observability_layout.addWidget(self.observability_policy_note)

        target_specific_form = QFormLayout()
        self.observability_phoenix_port = QLineEdit()
        self.observability_phoenix_port.setText("6006")
        self.observability_phoenix_port.setPlaceholderText("6006")
        self.observability_phoenix_port_row = QWidget()
        phoenix_port_row_layout = QHBoxLayout(self.observability_phoenix_port_row)
        phoenix_port_row_layout.setContentsMargins(0, 0, 0, 0)
        phoenix_port_row_layout.addWidget(self.observability_phoenix_port)
        target_specific_form.addRow("Local Phoenix Port:", self.observability_phoenix_port_row)

        self.observability_otlp_endpoint = QLineEdit()
        self.observability_otlp_endpoint.setPlaceholderText("https://otel.example.com/v1/traces")
        self.observability_otlp_endpoint_row = QWidget()
        otlp_endpoint_row_layout = QHBoxLayout(self.observability_otlp_endpoint_row)
        otlp_endpoint_row_layout.setContentsMargins(0, 0, 0, 0)
        otlp_endpoint_row_layout.addWidget(self.observability_otlp_endpoint)
        target_specific_form.addRow("OTLP Endpoint:", self.observability_otlp_endpoint_row)

        self.observability_otlp_headers = QTextEdit()
        self.observability_otlp_headers.setPlaceholderText('{\n  "Authorization": "Bearer ..."\n}')
        self.observability_otlp_headers.setFixedHeight(100)
        self.observability_otlp_headers_row = QWidget()
        otlp_headers_row_layout = QHBoxLayout(self.observability_otlp_headers_row)
        otlp_headers_row_layout.setContentsMargins(0, 0, 0, 0)
        otlp_headers_row_layout.addWidget(self.observability_otlp_headers)
        target_specific_form.addRow("OTLP Headers:", self.observability_otlp_headers_row)
        observability_layout.addLayout(target_specific_form)

        help_text = QLabel('<a href="https://opentelemetry.io/docs/">Learn more about OpenTelemetry →</a>')
        help_text.setOpenExternalLinks(True)
        help_text.setStyleSheet("color: #1976d2;")
        observability_layout.addWidget(help_text)

        self.observability_open_phoenix_button = QPushButton("Open Phoenix UI")
        self.observability_open_phoenix_button.clicked.connect(self._open_phoenix_ui)
        observability_layout.addWidget(self.observability_open_phoenix_button)

        self.observability_target.currentIndexChanged.connect(self._sync_observability_target_fields)

        self.config_fields["observability_enabled"] = self.observability_enabled
        self.config_fields["observability_target"] = self.observability_target
        self.config_fields["observability_project"] = self.observability_project
        self.config_fields["observability_content_policy"] = self.observability_content_policy
        self.config_fields["observability_include_binary_content"] = self.observability_include_binary_content
        self.config_fields["observability_phoenix_port"] = self.observability_phoenix_port
        self.config_fields["observability_otlp_endpoint"] = self.observability_otlp_endpoint
        self.config_fields["observability_otlp_headers"] = self.observability_otlp_headers

        layout.addWidget(observability_group)
        layout.addStretch()
        self._sync_observability_target_fields()
        return widget
    
    def load_keys(self):
        """Load existing API keys and settings."""
        # Load API keys (masked)
        for provider, field in self.api_fields.items():
            key = str(self.settings.get_api_key(provider) or "")
            if key:
                self._set_secret_field_value(field, self._MASKED_SECRET)
                field.setProperty("has_saved_key", True)
                field.setProperty("saved_key_value", key)
                field.setProperty("key_dirty", False)
            else:
                self._set_secret_field_value(field, "")
                field.setProperty("has_saved_key", False)
                field.setProperty("saved_key_value", "")
                field.setProperty("key_dirty", False)
        
        # Load Azure OpenAI settings
        azure_settings = self.settings.get("azure_openai_settings", {})
        if "endpoint" in azure_settings:
            self.azure_endpoint.setText(azure_settings["endpoint"])
        if "deployment" in azure_settings:
            self.azure_deployment.setText(azure_settings["deployment"])
        if "api_version" in azure_settings:
            self.azure_api_version.setText(azure_settings["api_version"])
        
        gateway_settings = self.settings.get("pydantic_ai_gateway_settings", {}) or {}
        base_url = str(gateway_settings.get("base_url") or "").strip()
        if base_url:
            self.gateway_base_url.setText(base_url)
        route = str(gateway_settings.get("route") or "").strip()
        if route:
            self.gateway_route.setText(route)

        transport_mode = str(self.settings.get("llm_transport_mode", "direct") or "direct").strip().lower()
        if transport_mode == "gateway" and not self._gateway_transport_available:
            transport_mode = "direct"
        transport_index = self.transport_mode_combo.findData(transport_mode)
        self.transport_mode_combo.setCurrentIndex(transport_index if transport_index >= 0 else 0)

        observability_settings = self.settings.get("observability_settings", {}) or {}
        self.observability_enabled.setChecked(bool(observability_settings.get("enabled", False)))
        target = str(observability_settings.get("target") or "phoenix_local").strip().lower()
        target_index = self.observability_target.findData(target)
        self.observability_target.setCurrentIndex(target_index if target_index >= 0 else 0)
        if "project" in observability_settings:
            self.observability_project.setText(str(observability_settings["project"]))
        content_policy = str(observability_settings.get("content_policy") or "unredacted").strip().lower()
        policy_index = self.observability_content_policy.findData(content_policy)
        self.observability_content_policy.setCurrentIndex(policy_index if policy_index >= 0 else 0)
        self.observability_include_binary_content.setChecked(
            bool(observability_settings.get("include_binary_content", False))
        )
        if "phoenix_port" in observability_settings and observability_settings["phoenix_port"] is not None:
            self.observability_phoenix_port.setText(str(observability_settings["phoenix_port"]))
        if "otlp_endpoint" in observability_settings and observability_settings["otlp_endpoint"]:
            self.observability_otlp_endpoint.setText(str(observability_settings["otlp_endpoint"]))
        headers = observability_settings.get("otlp_headers") or {}
        self.observability_otlp_headers.setPlainText(
            json.dumps(headers, indent=2, sort_keys=True) if headers else "{}"
        )
        self._sync_observability_target_fields()
    
    def save_keys(self):
        """Save API keys and settings to secure storage."""
        saved_count = 0
        errors = []
        warnings = []
        notices = []
        previous_gateway_settings = self.settings.get("pydantic_ai_gateway_settings", {}) or {}
        previous_gateway_key = str(self.settings.get_api_key("pydantic_ai_gateway") or "").strip()
        previous_transport_mode = str(self.settings.get("llm_transport_mode", "direct") or "direct").strip().lower()
        requested_transport_mode = str(self.transport_mode_combo.currentData() or "direct").strip().lower()
        transport_mode = (
            requested_transport_mode
            if requested_transport_mode == "gateway" and self._gateway_transport_available
            else "direct"
        )
        self.settings.set("llm_transport_mode", transport_mode)
        transport_changed = transport_mode != previous_transport_mode
        if requested_transport_mode == "gateway" and not self._gateway_transport_available:
            notices.append(
                "PAIG transport is currently unavailable in this environment, so the app will continue using direct transport."
            )
        if transport_changed:
            notices.append(self._transport_restart_notice)
        
        # Save API keys
        for provider, field in self.api_fields.items():
            text = field.text().strip()
            saved_value = str(field.property("saved_key_value") or "")
            has_saved_key = bool(field.property("has_saved_key"))
            key_dirty = bool(field.property("key_dirty"))
            
            if has_saved_key and not key_dirty and text in {self._MASKED_SECRET, saved_value}:
                continue
            
            # Save or remove key
            if text and not text.startswith("*"):
                try:
                    if self.settings.set_api_key(provider, text):
                        stored_value = str(self.settings.get_api_key(provider) or "")
                        if stored_value != text:
                            errors.append(f"Saved {provider} key does not match the value entered")
                        else:
                            saved_count += 1
                    else:
                        errors.append(f"Failed to save {provider} key")
                except Exception as e:
                    errors.append(f"Error saving {provider}: {str(e)}")
            elif not text and has_saved_key:
                # Remove key if field was cleared
                self.settings.remove_api_key(provider)
        
        # Save Azure OpenAI settings
        azure_endpoint = self.azure_endpoint.text().strip()
        azure_deployment = self.azure_deployment.text().strip()
        azure_api_version = self.azure_api_version.text().strip()
        
        if any([azure_endpoint, azure_deployment, azure_api_version]):
            azure_settings = {}
            if azure_endpoint:
                azure_settings["endpoint"] = azure_endpoint
            if azure_deployment:
                azure_settings["deployment"] = azure_deployment
            if azure_api_version:
                azure_settings["api_version"] = azure_api_version
            self.settings.set("azure_openai_settings", azure_settings)
        
        gateway_base_url = self.gateway_base_url.text().strip()
        gateway_route = self.gateway_route.text().strip()
        self.settings.set(
            "pydantic_ai_gateway_settings",
            {
                "base_url": gateway_base_url or None,
                "route": gateway_route or None,
            },
        )
        gateway_effective_key = self._effective_secret_field_value(self.gateway_api_key)
        gateway_changed = (
            gateway_effective_key != previous_gateway_key
            or (str(previous_gateway_settings.get("base_url") or "").strip() or None) != (gateway_base_url or None)
            or (str(previous_gateway_settings.get("route") or "").strip() or None) != (gateway_route or None)
        )

        target = self.observability_target.currentData() or "phoenix_local"
        observability_headers: dict[str, str] = {}
        headers_text = self.observability_otlp_headers.toPlainText().strip() or "{}"
        try:
            parsed_headers: Any = json.loads(headers_text)
            if not isinstance(parsed_headers, dict):
                raise ValueError("OTLP headers must be a JSON object")
            observability_headers = {str(key): str(value) for key, value in parsed_headers.items()}
        except Exception as exc:
            errors.append(f"Invalid OTLP headers JSON: {exc}")

        phoenix_port: int | None = None
        if target == "phoenix_local":
            try:
                phoenix_port = int(self.observability_phoenix_port.text().strip() or "6006")
            except ValueError:
                errors.append("Local Phoenix port must be a valid integer")
        otlp_endpoint = self.observability_otlp_endpoint.text().strip() or None
        if target == "otlp_http" and not otlp_endpoint:
            errors.append("OTLP endpoint is required when the target is OTLP Endpoint")

        # Show result
        if errors:
            QMessageBox.warning(
                self,
                "Save Errors",
                "Some keys could not be saved:\n" + "\n".join(errors)
            )
        else:
            observability_settings = {
                "enabled": self.observability_enabled.isChecked(),
                "target": target,
                "project": self.observability_project.text().strip() or "forensic-report-drafter",
                "content_policy": self.observability_content_policy.currentData() or "unredacted",
                "include_binary_content": self.observability_include_binary_content.isChecked(),
                "phoenix_port": phoenix_port,
                "otlp_endpoint": otlp_endpoint,
                "otlp_headers": observability_headers,
            }
            self.settings.set("observability_settings", observability_settings)
            if gateway_changed:
                reset_gateway_access_check_cache()
                reset_provider_catalog_cache()
                refresh_gateway_provider_catalog(force=True)
                if transport_mode == "gateway":
                    gateway_warning = self._validate_saved_gateway_settings(
                        api_key=gateway_effective_key or None,
                        base_url=gateway_base_url or None,
                        route=gateway_route or None,
                    )
                    if gateway_warning:
                        warnings.append(gateway_warning)
            if warnings:
                QMessageBox.warning(
                    self,
                    "Gateway Validation Warning",
                    "\n\n".join(warnings + notices),
                )
            elif saved_count > 0 or notices:
                message_parts = notices[:]
                if saved_count > 0:
                    message_parts.insert(0, f"Successfully saved {saved_count} API key(s) and settings.")
                elif not message_parts:
                    message_parts.append("Settings saved.")
                QMessageBox.information(
                    self,
                    "Settings Saved",
                    "\n\n".join(message_parts).strip()
                )
            self.accept()

    @staticmethod
    def _effective_secret_field_value(field: QLineEdit) -> str:
        text = field.text().strip()
        if bool(field.property("has_saved_key")) and not bool(field.property("key_dirty")):
            if text == APIKeyDialog._MASKED_SECRET:
                return str(field.property("saved_key_value") or "").strip()
        if text == APIKeyDialog._MASKED_SECRET:
            return str(field.property("saved_key_value") or "").strip()
        return text

    @staticmethod
    def _normalize_https_endpoint(value: str, *, label: str) -> tuple[str, str | None]:
        normalized = str(value or "").strip().rstrip("/")
        if not normalized:
            return "", None
        parsed = urlparse(normalized)
        if parsed.scheme != "https" or not parsed.netloc:
            return "", f"{label} endpoint must be a complete https:// URL"
        return normalized, None

    def _validate_saved_gateway_settings(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        route: str | None,
    ) -> str | None:
        provider_catalog = default_provider_catalog_for_transport(include_azure=True, transport="gateway")
        probe_provider = next((provider for provider in provider_catalog if provider.models), None)
        if probe_provider is None:
            return None
        probe_model = probe_provider.models[0].model_id if probe_provider.models else None
        backend = PydanticAIGatewayBackend(
            api_key=api_key,
            base_url=base_url,
            route=route,
        )
        result = backend.verify_gateway_access(
            probe_provider.provider_id,
            probe_model,
            timeout_seconds=5.0,
            force=True,
        )
        if result.ok:
            return None
        route_text = f"\nRoute: {result.route}" if result.route else ""
        status_text = f"\nGateway response: HTTP {result.status_code}" if result.status_code else ""
        if result.kind in {"auth_invalid", "auth_forbidden"}:
            return (
                "The saved Pydantic AI Gateway app key was rejected by the gateway.\n\n"
                f"Validation probe: {result.provider_id}/{result.model or '<default>'}{route_text}{status_text}\n"
                f"Gateway message: {result.message}\n\n"
                "Bulk and report runs will be blocked until this key is corrected."
            )
        if result.kind == "route_missing":
            return (
                "The saved gateway route/provider configuration could not be validated with the default probe.\n\n"
                f"Validation probe: {result.provider_id}/{result.model or '<default>'}{route_text}{status_text}\n"
                f"Gateway message: {result.message}\n\n"
                "Runs will validate the selected provider/model again before they start."
            )
        if result.kind == "missing_config":
            return (
                "Gateway mode is enabled, but the saved gateway configuration is incomplete.\n\n"
                f"Gateway message: {result.message}"
            )
        return (
            "The saved gateway settings could not be validated.\n\n"
            f"Validation probe: {result.provider_id}/{result.model or '<default>'}{route_text}{status_text}\n"
            f"Gateway message: {result.message}\n\n"
            "Runs will validate the selected provider/model again before they start."
        )
    
    @staticmethod
    def configure_api_keys(settings, parent=None):
        """Static method to show the dialog."""
        dialog = APIKeyDialog(settings, parent)
        return dialog.exec() == QDialog.Accepted
