"""
API key configuration dialog for the new UI.
"""

import logging
from typing import Dict

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QGroupBox,
    QDialogButtonBox, QMessageBox, QScrollArea,
    QWidget, QTabWidget, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl


class APIKeyDialog(QDialog):
    """Dialog for configuring API keys and service endpoints."""

    _PHOENIX_CONTENT_POLICIES = (
        ("Unredacted (Local Phoenix)", "unredacted"),
        ("Redacted", "redacted"),
    )
    _MASKED_SECRET = "*" * 20
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
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
            "Keys are stored securely in your system's keychain when available."
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
        
        # Phoenix Observability tab
        phoenix_tab = self._create_phoenix_tab()
        tabs.addTab(phoenix_tab, "Phoenix (Observability)")
        
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

        gateway_info = QLabel(
            "Optional custom base URL for a self-hosted gateway. Leave blank to use the managed gateway "
            "or the environment variable override."
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
        
        # Azure Document Intelligence Settings
        di_group = QGroupBox("Azure Document Intelligence")
        di_layout = QVBoxLayout(di_group)
        
        # API key input
        key_layout = QHBoxLayout()
        
        self.azure_di_key, show_btn = self._create_secret_field(
            "azure_di",
            "Enter Azure Document Intelligence API key...",
        )
        key_layout.addWidget(self.azure_di_key)
        key_layout.addWidget(show_btn)
        
        di_layout.addLayout(key_layout)
        
        # Endpoint
        endpoint_layout = QFormLayout()
        self.azure_di_endpoint = QLineEdit()
        self.azure_di_endpoint.setPlaceholderText("https://your-resource.cognitiveservices.azure.com/")
        endpoint_layout.addRow("Endpoint:", self.azure_di_endpoint)
        self.config_fields["azure_di_endpoint"] = self.azure_di_endpoint
        
        di_layout.addLayout(endpoint_layout)
        
        # Help text
        help_text = QLabel('<a href="https://portal.azure.com/">Configure in Azure Portal →</a>')
        help_text.setOpenExternalLinks(True)
        help_text.setStyleSheet("color: #1976d2;")
        di_layout.addWidget(help_text)
        
        layout.addWidget(di_group)
        
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
        port = self.phoenix_port.text() or "6006"
        QDesktopServices.openUrl(QUrl(f"http://localhost:{port}"))
    
    def _create_phoenix_tab(self) -> QWidget:
        """Create the Phoenix observability tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Phoenix info
        info = QLabel(
            "Arize Phoenix provides local LLM observability and debugging. "
            "Enable Phoenix to trace LLM calls, capture costs, and debug issues. "
            "Phoenix runs locally on your machine for complete data privacy."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Phoenix Settings
        phoenix_group = QGroupBox("Phoenix Configuration")
        phoenix_layout = QVBoxLayout(phoenix_group)
        
        # Enable/Disable Phoenix
        self.phoenix_enabled = QCheckBox("Enable Phoenix Observability")
        self.phoenix_enabled.setToolTip(
            "When enabled, Phoenix will trace all LLM calls locally"
        )
        phoenix_layout.addWidget(self.phoenix_enabled)
        
        # Phoenix Port
        port_layout = QFormLayout()
        self.phoenix_port = QLineEdit()
        self.phoenix_port.setText("6006")  # Default port
        self.phoenix_port.setPlaceholderText("Default: 6006")
        port_layout.addRow("Local Port:", self.phoenix_port)
        phoenix_layout.addLayout(port_layout)
        
        # Project Name
        project_layout = QFormLayout()
        self.phoenix_project = QLineEdit()
        self.phoenix_project.setText("forensic-report-drafter")
        self.phoenix_project.setPlaceholderText("Project name for organizing traces")
        project_layout.addRow("Project Name:", self.phoenix_project)
        phoenix_layout.addLayout(project_layout)
        
        # Export Fixtures Option
        self.phoenix_export_fixtures = QCheckBox("Export traces as test fixtures")
        self.phoenix_export_fixtures.setToolTip(
            "Automatically save LLM responses as test fixtures for mocking"
        )
        phoenix_layout.addWidget(self.phoenix_export_fixtures)

        policy_layout = QFormLayout()
        self.phoenix_content_policy = QComboBox()
        for label, value in self._PHOENIX_CONTENT_POLICIES:
            self.phoenix_content_policy.addItem(label, value)
        self.phoenix_content_policy.setToolTip(
            "Choose whether model instrumentation includes full prompt/response text."
        )
        policy_layout.addRow("Content Policy:", self.phoenix_content_policy)
        phoenix_layout.addLayout(policy_layout)

        self.phoenix_include_binary_content = QCheckBox("Include binary/multimodal content")
        self.phoenix_include_binary_content.setToolTip(
            "Include binary or multimodal payload bodies in model instrumentation."
        )
        phoenix_layout.addWidget(self.phoenix_include_binary_content)

        self.phoenix_policy_note = QLabel(
            "Unredacted traces are appropriate for trusted local Phoenix deployments. "
            "Use redacted traces if you later forward telemetry to a remote OTEL backend."
        )
        self.phoenix_policy_note.setWordWrap(True)
        phoenix_layout.addWidget(self.phoenix_policy_note)

        # Store config fields
        self.config_fields["phoenix_enabled"] = self.phoenix_enabled
        self.config_fields["phoenix_port"] = self.phoenix_port
        self.config_fields["phoenix_project"] = self.phoenix_project
        self.config_fields["phoenix_export_fixtures"] = self.phoenix_export_fixtures
        self.config_fields["phoenix_content_policy"] = self.phoenix_content_policy
        
        # Help text
        help_text = QLabel('<a href="https://phoenix.arize.com/">Learn more about Phoenix →</a>')
        help_text.setOpenExternalLinks(True)
        help_text.setStyleSheet("color: #1976d2;")
        phoenix_layout.addWidget(help_text)
        
        # View Phoenix UI Button
        view_button = QPushButton("Open Phoenix UI")
        view_button.clicked.connect(self._open_phoenix_ui)
        phoenix_layout.addWidget(view_button)
        
        layout.addWidget(phoenix_group)
        
        layout.addStretch()
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
        
        # Load Azure DI settings
        azure_di_settings = self.settings.get("azure_di_settings", {})
        if "endpoint" in azure_di_settings:
            self.azure_di_endpoint.setText(azure_di_settings["endpoint"])

        gateway_settings = self.settings.get("pydantic_ai_gateway_settings", {}) or {}
        base_url = str(gateway_settings.get("base_url") or "").strip()
        if base_url:
            self.gateway_base_url.setText(base_url)
        route = str(gateway_settings.get("route") or "").strip()
        if route:
            self.gateway_route.setText(route)

        # Load Phoenix settings
        phoenix_settings = self.settings.get("phoenix_settings", {})
        self.phoenix_enabled.setChecked(phoenix_settings.get("enabled", False))
        if "port" in phoenix_settings:
            self.phoenix_port.setText(str(phoenix_settings["port"]))
        if "project" in phoenix_settings:
            self.phoenix_project.setText(phoenix_settings["project"])
        self.phoenix_export_fixtures.setChecked(
            phoenix_settings.get("export_fixtures", False)
        )
        content_policy = str(phoenix_settings.get("content_policy") or "unredacted").strip().lower()
        policy_index = self.phoenix_content_policy.findData(content_policy)
        self.phoenix_content_policy.setCurrentIndex(policy_index if policy_index >= 0 else 0)
        self.phoenix_include_binary_content.setChecked(
            bool(phoenix_settings.get("include_binary_content", False))
        )
    
    def save_keys(self):
        """Save API keys and settings to secure storage."""
        saved_count = 0
        errors = []
        
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
        
        # Save Azure DI settings
        azure_di_endpoint = self.azure_di_endpoint.text().strip()
        if azure_di_endpoint:
            azure_di_settings = {"endpoint": azure_di_endpoint}
            self.settings.set("azure_di_settings", azure_di_settings)

        gateway_base_url = self.gateway_base_url.text().strip()
        gateway_route = self.gateway_route.text().strip()
        self.settings.set(
            "pydantic_ai_gateway_settings",
            {
                "base_url": gateway_base_url or None,
                "route": gateway_route or None,
            },
        )

        # Save Phoenix settings
        phoenix_settings = {
            "enabled": self.phoenix_enabled.isChecked(),
            "target": "local_phoenix",
            "port": int(self.phoenix_port.text() or 6006),
            "project": self.phoenix_project.text().strip() or "forensic-report-drafter",
            "export_fixtures": self.phoenix_export_fixtures.isChecked(),
            "content_policy": self.phoenix_content_policy.currentData() or "unredacted",
            "include_binary_content": self.phoenix_include_binary_content.isChecked(),
        }
        self.settings.set("phoenix_settings", phoenix_settings)
        
        # Show result
        if errors:
            QMessageBox.warning(
                self,
                "Save Errors",
                "Some keys could not be saved:\n" + "\n".join(errors)
            )
        else:
            if saved_count > 0:
                QMessageBox.information(
                    self,
                    "Settings Saved",
                    f"Successfully saved {saved_count} API key(s) and settings."
                )
            self.accept()
    
    @staticmethod
    def configure_api_keys(settings, parent=None):
        """Static method to show the dialog."""
        dialog = APIKeyDialog(settings, parent)
        return dialog.exec() == QDialog.Accepted
