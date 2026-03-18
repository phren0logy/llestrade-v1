"""Tests for new project creation dialog and conversion helper plumbing."""

from __future__ import annotations

from pathlib import Path
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

_ = PySide6

from src.app.core import llm_catalog
from src.app.core.llm_catalog import default_model_for_provider
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.core.converted_documents import converted_artifact_relative
from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.ui.dialogs.new_project_dialog import NewProjectDialog
from src.app.ui.dialogs.project_metadata_dialog import ProjectMetadataDialog
from src.app.ui.dialogs.bulk_analysis_group_dialog import BulkAnalysisGroupDialog
from src.app.ui.widgets import llm_settings_panel as llm_settings_panel_module
from src.app.core.file_tracker import FileTracker
from src.app.ui.stages.project_workspace import ProjectWorkspace


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Ensure a QApplication instance exists for widget-based tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def stub_llm_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_reasoning = llm_catalog.LLMReasoningCapabilities(
        supports_reasoning_controls=True,
        can_disable_reasoning=True,
        controls=("toggle", "budget"),
        budget_step=1024,
        default_state="off",
    )
    providers = (
        llm_catalog.LLMProviderOption(
            provider_id="anthropic",
            label="Anthropic Claude",
            models=(
                llm_catalog.LLMModelOption(
                    model_id="claude-sonnet-4-5",
                    label="Claude Sonnet 4.5",
                    context_window=200_000,
                    input_price_label="$3/1M",
                    output_price_label="$15/1M",
                    reasoning_capabilities=anthropic_reasoning,
                ),
                llm_catalog.LLMModelOption(
                    model_id="claude-opus-4-1",
                    label="Claude Opus 4.1",
                    context_window=200_000,
                    input_price_label="$15/1M",
                    output_price_label="$75/1M",
                    reasoning_capabilities=anthropic_reasoning,
                ),
            ),
        ),
    )
    model_lookup = {
        (provider.provider_id, model.model_id): model
        for provider in providers
        for model in provider.models
    }

    monkeypatch.setattr(
        llm_settings_panel_module,
        "default_provider_catalog_for_transport",
        lambda include_azure=False, transport="direct": providers,
    )
    monkeypatch.setattr(
        llm_settings_panel_module,
        "resolve_catalog_model",
        lambda provider_id, model_id, transport="direct": model_lookup.get((provider_id, str(model_id or "").strip())),
    )


def test_new_project_dialog_collects_helper_and_preview(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    source_root = tmp_path / "source"
    (source_root / "bundle").mkdir(parents=True)

    output_base = tmp_path / "output"
    output_base.mkdir()
    (output_base / "Case-Name").mkdir()

    dialog = NewProjectDialog()
    dialog._project_name_edit.setText("Case Name")
    dialog._subject_name_edit.setText("Jane Doe")
    dialog._dob_edit.setText("1975-08-19")
    dialog._case_info_edit.setPlainText("Referral for competency evaluation")
    dialog._source_root = source_root
    dialog._source_line.setText(source_root.as_posix())
    dialog._populate_tree()

    dialog._output_base = output_base
    dialog._output_line.setText(output_base.as_posix())
    dialog._update_folder_preview()

    assert dialog._helper_combo.count() == 1
    helper_id = dialog._helper_combo.itemData(0)
    assert helper_id == "docling"

    dialog._on_accept()
    config = dialog.result_config()
    assert config is not None
    assert config.subject_name == "Jane Doe"
    assert config.date_of_birth == "1975-08-19"
    assert config.case_description == "Referral for competency evaluation"
    assert config.conversion_helper == "docling"
    assert config.conversion_options == {}
    assert config.output_base == output_base
    assert config.selected_folders == ["bundle"]
    assert "Case-Name-2" in dialog._folder_preview_label.text()

    dialog.deleteLater()

def test_project_manager_update_conversion_helper_replaces_options(qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.conversion_settings.options = {"legacy": True}

    manager.update_conversion_helper("docling")
    assert manager.conversion_settings.helper == "docling"
    assert manager.conversion_settings.options == {}


def test_new_project_uses_catalog_backed_llm_defaults(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()

    manager.create_project(tmp_path, ProjectMetadata(case_name="Default Model Demo"))

    expected_model = default_model_for_provider("anthropic") or ""
    assert manager.settings["llm_provider"] == "anthropic"
    assert manager.settings["llm_model"] == expected_model
    assert manager.report_state.last_provider == "anthropic"
    assert manager.report_state.last_model == expected_model


def test_file_tracker_counts_converted_documents(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    project_root = tmp_path / "projects"
    project_root.mkdir()

    manager = ProjectManager()
    manager.create_project(project_root, ProjectMetadata(case_name="Tracker Demo"))

    converted_folder = manager.project_dir / "converted_documents"
    converted_folder.mkdir(parents=True, exist_ok=True)
    (converted_folder / converted_artifact_relative("example.pdf")).write_text("content")

    tracker = FileTracker(manager.project_dir)
    snapshot = tracker.scan()

    assert snapshot.imported_count == 1


def test_workspace_prompts_for_missing_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(
        projects_root,
        ProjectMetadata(
            case_name="Missing Source",
            subject_name="Alex Parker",
            date_of_birth="1981-02-14",
            case_description="Guardianship evaluation",
        ),
    )

    missing_root = tmp_path / "external_source"
    manager.update_source_state(root=missing_root.as_posix(), selected_folders=[], warnings=[])

    prompt_called = {"count": 0}

    def fake_question(*_args, **_kwargs):
        prompt_called["count"] += 1
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", fake_question)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)

    assert prompt_called["count"] == 1
    assert manager.source_state.warnings
    assert "Source folder" in manager.source_state.warnings[0]
    assert workspace._metadata_label is not None
    metadata_text = workspace._metadata_label.text()
    assert "Alex Parker" in metadata_text
    assert "1981-02-14" in metadata_text
    workspace.deleteLater()


def test_summary_group_dialog_lists_converted_documents(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Converted Demo"))

    converted_root = manager.project_dir / "converted_documents" / "folder"
    converted_root.mkdir(parents=True)
    (converted_root / Path(converted_artifact_relative("folder/doc.pdf")).name).write_text("content")

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        tree = dialog.file_tree
        top_labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert "folder" in top_labels
        folder_item = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0) == "folder")
        child_names = [folder_item.child(i).text(0) for i in range(folder_item.childCount())]
        assert "doc.pdf.doctags.txt" in child_names
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_requires_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Validation Demo"))

    converted_root = manager.project_dir / "converted_documents" / "folder"
    converted_root.mkdir(parents=True)
    (converted_root / Path(converted_artifact_relative("folder/doc.pdf")).name).write_text("content")

    warnings: list[str] = []

    def fake_warning(_parent, _title, message):
        warnings.append(message)
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        dialog.name_edit.setText("Empty Group")
        group = dialog._build_group_instance()
        assert group is None
        assert warnings
        assert "Select at least one converted file or directory" in warnings[-1]
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_marks_legacy_group_without_provider_invalid(
    tmp_path: Path,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Legacy Group"))

    converted_root = manager.project_dir / "converted_documents" / "folder"
    converted_root.mkdir(parents=True)
    (converted_root / Path(converted_artifact_relative("folder/doc.pdf")).name).write_text("content")

    legacy_group = BulkAnalysisGroup.create(
        "Legacy Group",
        files=[converted_artifact_relative("folder/doc.pdf")],
    )
    legacy_group.provider_id = ""
    legacy_group.model = ""

    dialog = BulkAnalysisGroupDialog(manager.project_dir, existing_group=legacy_group)
    try:
        settings, error = dialog.llm_settings_panel.current_settings()
        assert settings is None
        assert error is not None
        assert "saved provider" in error.lower()
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_llm_settings_section_has_room(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Bulk Layout"))

    converted_root = manager.project_dir / "converted_documents" / "folder"
    converted_root.mkdir(parents=True)
    (converted_root / Path(converted_artifact_relative("folder/doc.pdf")).name).write_text("content", encoding="utf-8")

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        dialog.show()
        qt_app.processEvents()

        assert dialog.minimumWidth() == 760
        assert dialog.llm_settings_group.title() == "LLM Settings"
        assert dialog.llm_settings_panel.width() >= 420
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_persists_enabled_reasoning(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Reasoning Restore"))

    converted_root = manager.project_dir / "converted_documents" / "folder"
    converted_root.mkdir(parents=True)
    (converted_root / Path(converted_artifact_relative("folder/doc.pdf")).name).write_text("content", encoding="utf-8")

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        dialog.name_edit.setText("Reasoning Group")
        folder_item = dialog.file_tree.topLevelItem(0)
        assert folder_item is not None
        file_item = folder_item.child(0)
        assert file_item is not None
        file_item.setCheckState(0, Qt.Checked)

        state_index = dialog.reasoning_state_combo.findData("on")
        assert state_index != -1
        dialog.reasoning_state_combo.setCurrentIndex(state_index)
        qt_app.processEvents()

        group = dialog._build_group_instance()

        assert group is not None
        assert group.use_reasoning is True
        assert group.reasoning.get("state") == "on"
    finally:
        dialog.deleteLater()

    reopened = BulkAnalysisGroupDialog(manager.project_dir, existing_group=group)
    try:
        qt_app.processEvents()
        assert reopened.reasoning_state_combo.currentData() == "on"
    finally:
        reopened.deleteLater()


def test_bulk_group_dialog_requires_combined_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Combined Validation"))

    warnings: list[str] = []

    def fake_warning(_parent, _title, message):
        warnings.append(message)
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        dialog.name_edit.setText("Empty Combined")
        index = dialog.operation_combo.findData("combined")
        assert index != -1
        dialog.operation_combo.setCurrentIndex(index)
        group = dialog._build_group_instance()
        assert group is None
        assert warnings
        assert "Select at least one converted or map output input" in warnings[-1]
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_canonicalizes_map_group_selection(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Combined Canonical"))

    map_dir = manager.project_dir / "bulk_analysis" / "per-document-summaries" / "Conner"
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "doc1_analysis.md").write_text("analysis one", encoding="utf-8")
    (map_dir / "doc2_analysis.md").write_text("analysis two", encoding="utf-8")

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        dialog.name_edit.setText("Canonical Combined")
        dialog.model_combo.setCurrentIndex(1)
        index = dialog.operation_combo.findData("combined")
        assert index != -1
        dialog.operation_combo.setCurrentIndex(index)

        map_group_item = dialog.map_tree.topLevelItem(0)
        assert map_group_item is not None
        assert map_group_item.text(0) == "per-document-summaries"
        map_group_item.setCheckState(0, Qt.Checked)

        group = dialog._build_group_instance()
        assert group is not None
        assert group.combine_map_groups == ["per-document-summaries"]
        assert group.combine_map_directories == []
        assert group.combine_map_files == []
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_shows_effective_combined_input_counts(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Combined Counts"))

    converted_dir = manager.project_dir / "converted_documents" / "folder"
    converted_dir.mkdir(parents=True, exist_ok=True)
    (converted_dir / Path(converted_artifact_relative("folder/doc.pdf")).name).write_text("converted", encoding="utf-8")

    map_dir = manager.project_dir / "bulk_analysis" / "per-document-summaries" / "folder"
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "doc_analysis.md").write_text("analysis", encoding="utf-8")

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        index = dialog.operation_combo.findData("combined")
        assert index != -1
        dialog.operation_combo.setCurrentIndex(index)

        folder_item = dialog.file_tree.topLevelItem(0)
        assert folder_item is not None
        converted_item = folder_item.child(0)
        assert converted_item is not None
        converted_item.setCheckState(0, Qt.Checked)

        map_group_item = dialog.map_tree.topLevelItem(0)
        assert map_group_item is not None
        map_group_item.setCheckState(0, Qt.Checked)

        qt_app.processEvents()
        assert dialog.combined_inputs_label.text() == "Effective inputs: 2 (converted 1, map 1)"
    finally:
        dialog.deleteLater()


def test_bulk_group_dialog_rejects_duplicate_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Duplicate Name Validation"))

    warnings: list[str] = []

    def fake_warning(_parent, _title, message):
        warnings.append(message)
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)

    dialog = BulkAnalysisGroupDialog(
        manager.project_dir,
        existing_names=["Clinical Records"],
    )
    try:
        dialog.name_edit.setText("clinical records")
        group = dialog._build_group_instance()
        assert group is None
        assert warnings
        assert "already exists" in warnings[-1]
    finally:
        dialog.deleteLater()


def test_workspace_scan_on_open_checks_for_new_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Scan On Open"))

    source_root = tmp_path / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    manager.update_source_state(
        root=source_root.as_posix(),
        selected_folders=[],
        warnings=[],
        last_scan="2026-03-08T12:00:00",
    )

    calls: list[tuple[bool, bool]] = []

    monkeypatch.setattr(
        "src.app.ui.stages.project_workspace.QTimer.singleShot",
        staticmethod(lambda _delay, fn: fn()),
    )

    workspace = ProjectWorkspace()
    monkeypatch.setattr(
        workspace,
        "_trigger_conversion",
        lambda auto_run, show_no_new_notice=True: calls.append((auto_run, show_no_new_notice)),
    )

    workspace.set_project(manager)

    assert calls == [(False, False)]
    workspace.deleteLater()


def test_workspace_scan_on_open_skips_when_no_last_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qt_app: QApplication,
) -> None:
    assert qt_app is not None
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="No Scan On Open"))

    source_root = tmp_path / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    manager.update_source_state(
        root=source_root.as_posix(),
        selected_folders=[],
        warnings=[],
        last_scan=None,
    )

    calls: list[tuple[bool, bool]] = []

    monkeypatch.setattr(
        "src.app.ui.stages.project_workspace.QTimer.singleShot",
        staticmethod(lambda _delay, fn: fn()),
    )

    workspace = ProjectWorkspace()
    monkeypatch.setattr(
        workspace,
        "_trigger_conversion",
        lambda auto_run, show_no_new_notice=True: calls.append((auto_run, show_no_new_notice)),
    )

    workspace.set_project(manager)

    assert calls == []
    workspace.deleteLater()


def test_project_metadata_dialog_updates_fields(qt_app: QApplication) -> None:
    assert qt_app is not None
    original = ProjectMetadata(
        case_name="Sample Case",
        subject_name="Initial Subject",
        date_of_birth="1990-05-10",
        case_description="Initial description",
    )

    dialog = ProjectMetadataDialog(original)
    dialog._subject_edit.setText("Updated Subject")
    dialog._dob_edit.setText("1991-06-11")
    dialog._case_info_edit.setPlainText("Updated details")

    result = dialog.result_metadata()
    assert result.subject_name == "Updated Subject"
    assert result.date_of_birth == "1991-06-11"
    assert result.case_description == "Updated details"

    dialog.deleteLater()
