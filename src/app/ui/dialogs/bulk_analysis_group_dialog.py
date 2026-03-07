"""Dialog for creating bulk analysis groups."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Optional, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.config.prompt_store import get_bundled_dir, get_custom_dir
from src.config.paths import app_resource_root
from src.app.core.azure_artifacts import is_azure_raw_artifact
from src.app.core.bulk_paths import iter_map_outputs
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.project_manager import ProjectMetadata
from src.app.core.placeholders.analyzer import find_placeholders
from src.app.core.prompt_placeholders import get_prompt_spec
from src.app.core.prompt_preview import generate_prompt_preview, PromptPreviewError
from src.app.core.prompt_placeholders import placeholder_summary
from .prompt_preview_dialog import PromptPreviewDialog
from src.app.core.secure_settings import SecureSettings
from src.common.llm.bedrock_catalog import DEFAULT_BEDROCK_MODELS, list_bedrock_models

DEFAULT_SYSTEM_PROMPT = "prompts/document_analysis_system_prompt.md"
DEFAULT_USER_PROMPT = "prompts/document_bulk_analysis_prompt.md"


class BulkAnalysisGroupDialog(QDialog):
    """Collect information needed to create a bulk analysis group."""

    def __init__(
        self,
        project_dir: Path,
        parent: Optional[QWidget] = None,
        *,
        metadata: Optional[ProjectMetadata] = None,
        placeholder_values: Optional[Mapping[str, str]] = None,
        existing_group: Optional[BulkAnalysisGroup] = None,
    ) -> None:
        super().__init__(parent)
        self._project_dir = project_dir
        self._metadata = metadata
        self._placeholder_values = dict(placeholder_values or {})
        self._placeholder_requirements: Dict[str, bool] = {}
        self._group: Optional[BulkAnalysisGroup] = None
        self._existing_group = existing_group
        self._tree_nodes: Dict[Tuple[str, bool], QTreeWidgetItem] = {}
        self._map_tree_nodes: Dict[Tuple[str, str], QTreeWidgetItem] = {}
        self.setWindowTitle("Edit Bulk Analysis" if existing_group else "New Bulk Analysis")
        self.setModal(True)
        self._build_ui()
        self._populate_file_tree()
        self._populate_map_outputs_tree()
        self._apply_initial_group_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_group(self) -> BulkAnalysisGroup:
        if not self._group:
            raise RuntimeError("Dialog was not accepted; no bulk analysis group available")
        return self._group

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Clinical Records")
        form.addRow("Name", self.name_edit)

        self.description_edit = QPlainTextEdit()
        self.description_edit.setPlaceholderText("Optional description")
        self.description_edit.setFixedHeight(60)
        form.addRow("Description", self.description_edit)

        # Operation selector
        self.operation_combo = QComboBox()
        self.operation_combo.addItem("Per-document", "per_document")
        self.operation_combo.addItem("Combined", "combined")
        self.operation_combo.currentIndexChanged.connect(self._on_operation_changed)
        form.addRow("Operation", self.operation_combo)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.itemChanged.connect(self._on_tree_item_changed)
        self._block_tree_signal = False
        form.addRow("Documents", self.file_tree)

        # Map outputs tree group (visible for Combined)
        self.map_tree_group = QGroupBox("Per-document Outputs (optional)")
        map_layout = QVBoxLayout(self.map_tree_group)
        self.map_tree = QTreeWidget()
        self.map_tree.setHeaderHidden(True)
        self.map_tree.setUniformRowHeights(True)
        self.map_tree.itemChanged.connect(self._on_map_tree_item_changed)
        map_layout.addWidget(self.map_tree)
        form.addRow(self.map_tree_group)

        self.manual_files_label = QLabel("Extra Files")
        self.manual_files_edit = QPlainTextEdit()
        self.manual_files_edit.setPlaceholderText("Additional files (one per line, optional)")
        self.manual_files_edit.setMinimumHeight(60)
        form.addRow(self.manual_files_label, self.manual_files_edit)

        self.system_prompt_edit = QLineEdit()
        self.system_prompt_edit.setToolTip(
            placeholder_summary("document_analysis_system_prompt")
        )
        self.system_prompt_button = QPushButton("Browse…")
        self.system_prompt_button.clicked.connect(lambda: self._choose_prompt_file(self.system_prompt_edit))
        form.addRow("System Prompt", self._wrap_with_button(self.system_prompt_edit, self.system_prompt_button))
        self._initialise_prompt_path(
            self.system_prompt_edit,
            "document_analysis_system_prompt.md",
            DEFAULT_SYSTEM_PROMPT,
        )
        self.system_prompt_edit.editingFinished.connect(self._refresh_placeholder_requirements)

        self.user_prompt_edit = QLineEdit()
        self.user_prompt_edit.setToolTip(
            placeholder_summary("document_bulk_analysis_prompt")
        )
        self.user_prompt_button = QPushButton("Browse…")
        self.user_prompt_button.clicked.connect(lambda: self._choose_prompt_file(self.user_prompt_edit))
        form.addRow("User Prompt", self._wrap_with_button(self.user_prompt_edit, self.user_prompt_button))
        self._initialise_prompt_path(
            self.user_prompt_edit,
            "document_bulk_analysis_prompt.md",
            DEFAULT_USER_PROMPT,
        )
        self.user_prompt_edit.editingFinished.connect(self._refresh_placeholder_requirements)

        self._placeholder_table = QTableWidget(0, 2)
        self._placeholder_table.setHorizontalHeaderLabels(["Placeholder", "Required"])
        self._placeholder_table.verticalHeader().setVisible(False)
        header = self._placeholder_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._placeholder_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._placeholder_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._placeholder_table.setFocusPolicy(Qt.NoFocus)
        form.addRow("Placeholder Requirements", self._placeholder_table)

        self.preview_prompt_button = QPushButton("Preview Prompt")
        self.preview_prompt_button.clicked.connect(self._preview_prompt)
        form.addRow("", self.preview_prompt_button)

        self.model_combo = self._build_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model", self.model_combo)

        self.custom_model_label = QLabel("Custom model")
        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("e.g., claude-sonnet-4-5-20250929")
        form.addRow(self.custom_model_label, self.custom_model_edit)

        self.custom_context_label = QLabel("Custom context window (tokens)")
        self.custom_context_spin = QSpinBox()
        self.custom_context_spin.setRange(10000, 4000000)
        self.custom_context_spin.setSingleStep(1000)
        self.custom_context_spin.setValue(200000)
        form.addRow(self.custom_context_label, self.custom_context_spin)

        # Combined options
        self.order_combo = QComboBox()
        self.order_combo.addItem("By path", "path")
        self.order_combo.addItem("By modified time", "mtime")
        form.addRow("Combined Order", self.order_combo)

        self.output_template_edit = QLineEdit()
        self.output_template_edit.setPlaceholderText("combined_{timestamp}.md")
        self.output_template_edit.setText("combined_{timestamp}.md")
        form.addRow("Output Template", self.output_template_edit)

        self.reasoning_checkbox = QCheckBox("Use reasoning (thinking models)")
        form.addRow("Reasoning", self.reasoning_checkbox)

        layout.addLayout(form)
        self._refresh_placeholder_requirements()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initial visibility
        self._on_operation_changed()
        self._on_model_changed()

    def _is_allowed_file(self, path: Path) -> bool:
        """Return True if the file should be selectable (only .md or .txt)."""
        try:
            return path.suffix.lower() in {".md", ".txt"} and not is_azure_raw_artifact(path)
        except Exception:
            return False

    def _wrap_with_button(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        widget = QWidget()
        h_layout = QHBoxLayout(widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(line_edit)
        h_layout.addWidget(button)
        return widget

    def _build_model_combo(self):
        combo = QComboBox()
        combo.setEditable(False)
        # Limit to Anthropic with two supported models and a Custom option
        combo.addItem("Custom…", ("custom", ""))
        combo.addItem(
            "Anthropic Claude (claude-sonnet-4-5-20250929)",
            ("anthropic", "claude-sonnet-4-5-20250929"),
        )
        combo.addItem(
            "Anthropic Claude (claude-opus-4-6)",
            ("anthropic", "claude-opus-4-6"),
        )

        # Append AWS Bedrock Claude models discovered via AWS CLI credentials
        bedrock_models = []
        try:
            settings = SecureSettings()
            bedrock_settings = settings.get("aws_bedrock_settings", {}) or {}
            bedrock_models = list_bedrock_models(
                region=bedrock_settings.get("region"),
                profile=bedrock_settings.get("profile"),
            )
        except Exception:
            bedrock_models = list(DEFAULT_BEDROCK_MODELS)

        for model in bedrock_models:
            label = f"AWS Bedrock Claude ({model.name})"
            combo.addItem(label, ("anthropic_bedrock", model.model_id))

        return combo

    def _on_model_changed(self) -> None:
        data = self.model_combo.currentData()
        is_custom = bool(data) and data[0] == "custom"
        self.custom_model_label.setVisible(is_custom)
        self.custom_model_edit.setVisible(is_custom)
        self.custom_context_label.setVisible(is_custom)
        self.custom_context_spin.setVisible(is_custom)

    def _choose_prompt_file(self, line_edit: QLineEdit) -> None:
        # Default browse folder to user prompt store (custom), with safe fallbacks
        try:
            initial_dir = get_custom_dir()
        except Exception:
            initial_dir = self._project_dir if self._project_dir else Path.cwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prompt File",
            str(initial_dir),
            "Prompt Files (*.txt *.md *.prompt);;All Files (*)",
        )
        if file_path:
            line_edit.setText(self._normalise_path(Path(file_path)))
            self._refresh_placeholder_requirements()

    def _initialise_prompt_path(self, line_edit: QLineEdit, filename: str, fallback: str) -> None:
        path: Optional[Path] = None
        try:
            candidate = get_bundled_dir() / filename
            if candidate.exists():
                path = candidate
        except Exception:
            path = None

        if path is None:
            resource_root = app_resource_root()
            resource_fallback = resource_root / fallback
            if resource_fallback.exists():
                path = resource_fallback
            else:
                path = resource_fallback

        line_edit.setText(self._normalise_path(path))

    def _refresh_placeholder_requirements(self) -> None:
        system_text = self._read_prompt_template(self.system_prompt_edit.text().strip())
        user_text = self._read_prompt_template(self.user_prompt_edit.text().strip())
        placeholders = sorted(find_placeholders(system_text) | find_placeholders(user_text))

        required_defaults: set[str] = set()
        optional_defaults: set[str] = set()

        system_spec = get_prompt_spec("document_analysis_system_prompt")
        if system_spec:
            required_defaults.update(system_spec.required)
            optional_defaults.update(system_spec.optional)

        user_spec = get_prompt_spec("document_bulk_analysis_prompt")
        if user_spec:
            required_defaults.update(user_spec.required)
            optional_defaults.update(user_spec.optional)

        if self.operation_combo.currentData() == "per_document":
            required_defaults.add("document_content")

        # Preserve existing choices where possible
        previous = dict(self._placeholder_requirements)
        self._placeholder_requirements = {}

        self._placeholder_table.blockSignals(True)
        self._placeholder_table.setRowCount(len(placeholders))
        for row, name in enumerate(placeholders):
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemIsEnabled)
            self._placeholder_table.setItem(row, 0, item)
            checkbox = QCheckBox()
            checkbox.setTristate(False)
            checkbox.setChecked(previous.get(name, name in required_defaults))
            checkbox.stateChanged.connect(lambda state, key=name: self._on_placeholder_requirement_changed(key, state == Qt.Checked))
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(checkbox)
            self._placeholder_table.setCellWidget(row, 1, container)
            self._placeholder_requirements[name] = checkbox.isChecked()
        self._placeholder_table.blockSignals(False)

    def _on_placeholder_requirement_changed(self, key: str, required: bool) -> None:
        self._placeholder_requirements[key] = required

    def _read_prompt_template(self, path_str: str) -> str:
        path_str = (path_str or "").strip()
        if not path_str:
            return ""
        path = Path(path_str).expanduser()
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if self._project_dir:
                candidates.append((self._project_dir / path).resolve())
            candidates.append((get_custom_dir() / path).resolve())
            candidates.append((get_bundled_dir() / path).resolve())
            resource_root = app_resource_root()
            candidates.append((resource_root / path).resolve())
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate.read_text(encoding="utf-8")
            except Exception:
                continue
        return ""

    def _on_operation_changed(self) -> None:
        combined = self.operation_combo.currentData() == "combined"
        self.map_tree_group.setVisible(combined)
        # Combined options are still useful to adjust ahead of time
        self.order_combo.setEnabled(combined)
        self.output_template_edit.setEnabled(combined)
        self.reasoning_checkbox.setEnabled(True)
        # Show Extra Files only for Combined
        self.manual_files_label.setVisible(combined)
        self.manual_files_edit.setVisible(combined)
        self._refresh_placeholder_requirements()

    def _on_map_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        # Mirror tri-state behavior used in the converted-docs tree
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        node_type, _ = data
        state = item.checkState(0)
        # Propagate to children when a directory/group toggles
        if node_type in ("map-dir", "map-group") and state in (Qt.Checked, Qt.Unchecked):
            for index in range(item.childCount()):
                child = item.child(index)
                child.setCheckState(0, state)
        # Update ancestors' partial/checked state
        self._sync_parent_state(item.parent())
        return

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------
    def _handle_accept(self) -> None:
        group = self._build_group_instance()
        if group is None:
            return
        self._group = group
        self.accept()

    def _build_group_instance(self) -> Optional[BulkAnalysisGroup]:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please provide a name for the bulk analysis group.")
            return None

        tree_files, directories = self._collect_selection()
        manual_files_all = [
            self._normalise_text(line.strip())
            for line in self.manual_files_edit.toPlainText().splitlines()
            if line.strip()
        ]
        files_set = set()
        for path in tree_files:
            if not path:
                continue
            if Path(path).suffix.lower() in {".md", ".txt"}:
                files_set.add(self._normalise_text(path))
        for path in manual_files_all:
            if not path:
                continue
            if Path(path).suffix.lower() in {".md", ".txt"}:
                files_set.add(self._normalise_text(path))
        files = sorted(files_set)

        directories = sorted({self._normalise_directory(path) for path in directories if path})

        combo_data = self.model_combo.currentData()
        provider_id, model = combo_data if combo_data else ("", "")
        is_custom_selection = combo_data and combo_data[0] == "custom"
        if is_custom_selection:
            provider_id = "anthropic"
            model = self.custom_model_edit.text().strip()
            if not model:
                QMessageBox.warning(self, "Missing Model", "Please enter a model id for the custom option.")
                return None
            custom_window = int(self.custom_context_spin.value())
        else:
            custom_window = None

        description = self.description_edit.toPlainText().strip()
        system_prompt = self._normalise_text(self.system_prompt_edit.text().strip())
        user_prompt = self._normalise_text(self.user_prompt_edit.text().strip())
        placeholder_settings = dict(self._placeholder_requirements)

        op = self.operation_combo.currentData() or "per_document"
        combine_order = self.order_combo.currentData() or "path"
        combine_template = self.output_template_edit.text().strip() or "combined_{timestamp}.md"

        map_groups: list[str] = []
        map_dirs: list[str] = []
        map_files: list[str] = []
        if op == "combined":
            root = self.map_tree.invisibleRootItem()
            nodes = [root]
            while nodes:
                node = nodes.pop()
                for i in range(node.childCount()):
                    child = node.child(i)
                    nodes.append(child)
                    data = child.data(0, Qt.UserRole)
                    if not data:
                        continue
                    kind, value = data
                    if child.checkState(0) != Qt.Checked:
                        continue
                    if kind == "map-group":
                        map_groups.append(str(value))
                    elif kind == "map-dir":
                        map_dirs.append(str(value))
                    elif kind == "map-file":
                        map_files.append(str(value))

        group_kwargs = {
            "name": name,
            "description": description,
            "provider_id": provider_id,
            "model": model,
            "system_prompt_path": system_prompt,
            "user_prompt_path": user_prompt,
            "model_context_window": custom_window,
            "placeholder_requirements": placeholder_settings,
            "use_reasoning": self.reasoning_checkbox.isChecked(),
        }

        if op == "combined":
            group_kwargs.update(
                {
                    "operation": "combined",
                    "files": [],
                    "directories": [],
                    "combine_converted_files": files,
                    "combine_converted_directories": directories,
                    "combine_map_files": sorted(set(map_files)),
                    "combine_map_groups": sorted(set(map_groups)),
                    "combine_map_directories": sorted(set(map_dirs)),
                    "combine_order": combine_order,
                    "combine_output_template": combine_template,
                }
            )
        else:
            group_kwargs.update(
                {
                    "operation": "per_document",
                    "files": files,
                    "directories": directories,
                    "combine_converted_files": [],
                    "combine_converted_directories": [],
                    "combine_map_files": [],
                    "combine_map_groups": [],
                    "combine_map_directories": [],
                }
            )

        existing = self._existing_group
        if existing:
            return replace(existing, **group_kwargs)

        base_group = BulkAnalysisGroup.create(
            name=name,
            description=description,
            files=group_kwargs.get("files", []),
            directories=group_kwargs.get("directories", []),
            provider_id=provider_id,
            model=model,
            system_prompt_path=system_prompt,
            user_prompt_path=user_prompt,
        )
        for key, value in group_kwargs.items():
            setattr(base_group, key, value)
        return base_group

    def _preview_prompt(self) -> None:
        group = self._build_group_instance()
        if group is None:
            return
        try:
            preview = generate_prompt_preview(
                self._project_dir,
                group,
                metadata=self._metadata,
                placeholder_values=self._placeholder_values,
            )
        except PromptPreviewError as exc:
            QMessageBox.warning(self, "Prompt Preview", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Prompt Preview", "Failed to generate prompt preview.")
            return

        dialog = PromptPreviewDialog(self)
        required_keys = {key for key, is_required in self._placeholder_requirements.items() if is_required}
        dialog.set_preview(preview, required=required_keys)
        dialog.exec()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _populate_file_tree(self) -> None:
        self.file_tree.clear()
        if not self._project_dir:
            notice = QTreeWidgetItem(["No project directory available."])
            notice.setFlags(Qt.NoItemFlags)
            self.file_tree.addTopLevelItem(notice)
            return

        converted_root = self._project_dir / "converted_documents"
        if not converted_root.exists():
            notice = QTreeWidgetItem(["No converted documents found. Run conversion first."])
            notice.setFlags(Qt.NoItemFlags)
            self.file_tree.addTopLevelItem(notice)
            return

        converted_files = []
        for path in converted_root.rglob("*"):
            if not path.is_file():
                continue
            # Hide Azure DI artefacts from the picker to avoid confusing users.
            if any(
                part in {".azure-di", ".azure_di"} or part.startswith(".azure-di") or part.startswith(".azure_di")
                for part in path.parts
            ):
                continue
            # Only allow markdown or text files
            if not self._is_allowed_file(path):
                continue
            converted_files.append(path.relative_to(converted_root).as_posix())

        converted_files.sort()

        if not converted_files:
            notice = QTreeWidgetItem(["Converted folder is empty. Run conversion first."])
            notice.setFlags(Qt.NoItemFlags)
            self.file_tree.addTopLevelItem(notice)
            return

        self._tree_nodes = {}
        self._block_tree_signal = True
        for path in converted_files:
            self._add_path_to_tree(path)
        self.file_tree.expandAll()
        self._block_tree_signal = False

    def _populate_map_outputs_tree(self) -> None:
        self.map_tree.clear()
        self._map_tree_nodes = {}
        if not self._project_dir:
            info = QTreeWidgetItem(["No project directory available."])
            info.setFlags(Qt.NoItemFlags)
            self.map_tree.addTopLevelItem(info)
            return
        ba_root = self._project_dir / "bulk_analysis"
        if not ba_root.exists():
            info = QTreeWidgetItem(["No per-document outputs found."])
            info.setFlags(Qt.NoItemFlags)
            self.map_tree.addTopLevelItem(info)
            return

        added = False
        for slug_dir in sorted(ba_root.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            outputs = list(iter_map_outputs(self._project_dir, slug))
            if not outputs:
                continue

            added = True
            group_item = QTreeWidgetItem([slug])
            group_item.setData(0, Qt.UserRole, ("map-group", slug))
            flags = group_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            group_item.setFlags(flags)
            group_item.setCheckState(0, Qt.Unchecked)
            self.map_tree.addTopLevelItem(group_item)
            self._map_tree_nodes[("map-group", slug)] = group_item

            tree_nodes: dict[tuple[str, bool], QTreeWidgetItem] = {}
            for _, rel in sorted(outputs, key=lambda item: item[1]):
                self._add_map_path_to_tree(group_item, slug, rel, tree_nodes)

        if not added:
            info = QTreeWidgetItem(["No per-document outputs found."])
            info.setFlags(Qt.NoItemFlags)
            self.map_tree.addTopLevelItem(info)

    def _apply_initial_group_state(self) -> None:
        group = self._existing_group
        if not group:
            return

        self.name_edit.setText(group.name)
        self.description_edit.setPlainText(group.description or "")
        self.system_prompt_edit.setText(self._normalise_text(group.system_prompt_path))
        self.user_prompt_edit.setText(self._normalise_text(group.user_prompt_path))
        self.reasoning_checkbox.setChecked(group.use_reasoning)
        if group.combine_output_template:
            self.output_template_edit.setText(group.combine_output_template)
        current_order = group.combine_order or "path"
        index = self.order_combo.findData(current_order)
        if index != -1:
            self.order_combo.setCurrentIndex(index)

        self._placeholder_requirements = dict(group.placeholder_requirements or {})

        # Ensure the operation UI matches the saved group without emitting intermediate signals.
        operation = group.operation or "per_document"
        self.operation_combo.blockSignals(True)
        op_index = self.operation_combo.findData(operation)
        if op_index == -1:
            op_index = 0
        self.operation_combo.setCurrentIndex(op_index)
        self.operation_combo.blockSignals(False)
        self._on_operation_changed()

        self._apply_model_selection(group)

        manual_entries = self._restore_file_tree_selection(group)
        if operation == "combined":
            manual_text = "\n".join(manual_entries)
            self.manual_files_edit.setPlainText(manual_text)
            self._restore_map_tree_selection(group)
        else:
            self.manual_files_edit.clear()

        if group.model_context_window:
            self.custom_context_spin.setValue(int(group.model_context_window))

        self._refresh_placeholder_requirements()

    def _apply_model_selection(self, group: BulkAnalysisGroup) -> None:
        provider = group.provider_id or ""
        model_id = group.model or ""

        target_index = -1
        for idx in range(self.model_combo.count()):
            data = self.model_combo.itemData(idx)
            if data == (provider, model_id):
                target_index = idx
                break

        if provider == "anthropic_bedrock" and target_index == -1 and model_id:
            label = f"AWS Bedrock Claude ({model_id})"
            self.model_combo.addItem(label, (provider, model_id))
            target_index = self.model_combo.count() - 1

        if target_index == -1:
            # Fallback to custom configuration.
            target_index = 0

        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(target_index)
        self.model_combo.blockSignals(False)
        self._on_model_changed()

        current_data = self.model_combo.currentData()
        if current_data and current_data[0] == "custom":
            self.custom_model_edit.setText(model_id)
        else:
            self.custom_model_edit.clear()

    def _restore_file_tree_selection(self, group: BulkAnalysisGroup) -> List[str]:
        manual_entries: List[str] = []
        seen_manual: set[str] = set()
        files = group.combine_converted_files if group.operation == "combined" else group.files
        directories = group.combine_converted_directories if group.operation == "combined" else group.directories

        self._block_tree_signal = True
        for directory in directories:
            item = self._tree_nodes.get((directory, False))
            if item:
                item.setCheckState(0, Qt.Checked)
                self._sync_parent_state(item.parent())
        for file_path in files:
            item = self._tree_nodes.get((file_path, True))
            if item:
                item.setCheckState(0, Qt.Checked)
                self._sync_parent_state(item.parent())
            else:
                if file_path not in seen_manual:
                    manual_entries.append(file_path)
                    seen_manual.add(file_path)
        self._block_tree_signal = False
        return manual_entries

    def _restore_map_tree_selection(self, group: BulkAnalysisGroup) -> None:
        self.map_tree.blockSignals(True)

        for slug in group.combine_map_groups:
            item = self._map_tree_nodes.get(("map-group", slug))
            if item:
                self._set_map_item_state(item, Qt.Checked)

        for directory in group.combine_map_directories:
            item = self._map_tree_nodes.get(("map-dir", directory))
            if item:
                self._set_map_item_state(item, Qt.Checked)

        for file_path in group.combine_map_files:
            item = self._map_tree_nodes.get(("map-file", file_path))
            if item:
                item.setCheckState(0, Qt.Checked)
                self._sync_parent_state(item.parent())

        self.map_tree.blockSignals(False)

    def _set_map_item_state(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        item.setCheckState(0, state)
        for index in range(item.childCount()):
            child = item.child(index)
            self._set_map_item_state(child, state)
        self._sync_parent_state(item.parent())

    def _add_path_to_tree(self, relative_path: str) -> None:
        parts = relative_path.split("/")
        current_path = ""
        parent_item = self.file_tree.invisibleRootItem()

        for index, part in enumerate(parts):
            current_path = f"{current_path}/{part}" if current_path else part
            is_file = index == len(parts) - 1
            key = (current_path, is_file)

            existing = self._tree_nodes.get(key)
            if existing:
                parent_item = existing
                continue

            item = QTreeWidgetItem(parent_item, [part])
            item.setData(0, Qt.UserRole, ("file" if is_file else "dir", current_path))
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if not is_file:
                flags |= Qt.ItemFlag.ItemIsAutoTristate
            item.setFlags(flags)
            item.setCheckState(0, Qt.Unchecked)

            if not is_file:
                self._tree_nodes[(current_path, False)] = item
            else:
                self._tree_nodes[(current_path, True)] = item

            parent_item = item

    def _add_map_path_to_tree(self, group_item: QTreeWidgetItem, slug: str, relative_path: str, cache: dict) -> None:
        parts = relative_path.split("/")
        current_path = ""
        parent_item = group_item
        for index, part in enumerate(parts):
            current_path = f"{current_path}/{part}" if current_path else part
            is_file = index == len(parts) - 1
            key = (f"{slug}/{current_path}", is_file)
            existing = cache.get(key)
            if existing:
                parent_item = existing
                continue
            item = QTreeWidgetItem(parent_item, [part])
            if is_file:
                value = f"{slug}/{current_path}"
                item.setData(0, Qt.UserRole, ("map-file", value))
                self._map_tree_nodes[("map-file", value)] = item
            else:
                value = f"{slug}/{current_path}"
                item.setData(0, Qt.UserRole, ("map-dir", value))
                self._map_tree_nodes[("map-dir", value)] = item
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if not is_file:
                flags |= Qt.ItemFlag.ItemIsAutoTristate
            item.setFlags(flags)
            item.setCheckState(0, Qt.Unchecked)
            cache[key] = item
            parent_item = item

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._block_tree_signal:
            return
        node_type, _ = item.data(0, Qt.UserRole) or (None, None)
        state = item.checkState(0)

        self._block_tree_signal = True
        if node_type == "dir" and state in (Qt.Checked, Qt.Unchecked):
            for index in range(item.childCount()):
                child = item.child(index)
                child.setCheckState(0, state)

        self._sync_parent_state(item.parent())
        self._block_tree_signal = False

    def _sync_parent_state(self, parent: Optional[QTreeWidgetItem]) -> None:
        while parent is not None:
            checked = unchecked = 0
            for index in range(parent.childCount()):
                state = parent.child(index).checkState(0)
                if state == Qt.Checked:
                    checked += 1
                elif state == Qt.Unchecked:
                    unchecked += 1
                else:
                    checked += 1
                    unchecked += 1
            if checked and not unchecked:
                parent.setCheckState(0, Qt.Checked)
            elif unchecked and not checked:
                parent.setCheckState(0, Qt.Unchecked)
            else:
                parent.setCheckState(0, Qt.PartiallyChecked)
            parent = parent.parent()

    def _collect_selection(self) -> tuple[List[str], List[str]]:
        files: List[str] = []
        directories: List[str] = []
        root = self.file_tree.invisibleRootItem()
        for index in range(root.childCount()):
            self._collect_from_item(root.child(index), files, directories)
        return files, directories

    def _collect_from_item(self, item: QTreeWidgetItem, files: List[str], directories: List[str]) -> None:
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        node_type, path = data
        state = item.checkState(0)

        if node_type == "dir":
            if state == Qt.Checked:
                directories.append(path)
                return
        elif node_type == "file" and state == Qt.Checked:
            files.append(path)

        for index in range(item.childCount()):
            self._collect_from_item(item.child(index), files, directories)
    def _normalise_text(self, text: str) -> str:
        if not text:
            return ""
        path = Path(text)
        if not path.is_absolute():
            return text
        return self._normalise_path(path)

    def _normalise_path(self, path: Path) -> str:
        if not path:
            return ""
        if self._project_dir:
            try:
                project_dir = self._project_dir.resolve()
                relative = path.resolve().relative_to(project_dir)
                return str(relative)
            except Exception:
                pass
        return str(path)

    def _normalise_directory(self, path: str) -> str:
        normalised = self._normalise_text(path)
        return normalised.strip("/")
