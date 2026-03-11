"""Tests for new project creation dialog and conversion helper plumbing."""

from __future__ import annotations

import frontmatter
import hashlib
from pathlib import Path
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

_ = PySide6

from src.app.core.llm_catalog import default_model_for_provider
from src.app.core.conversion_manager import ConversionJob
from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.ui.dialogs.new_project_dialog import NewProjectDialog
from src.app.ui.dialogs.project_metadata_dialog import ProjectMetadataDialog
from src.app.ui.dialogs.bulk_analysis_group_dialog import BulkAnalysisGroupDialog
from src.app.workers.conversion_worker import ConversionWorker
from src.app.core.file_tracker import FileTracker
from src.app.ui.stages.project_workspace import ProjectWorkspace


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Ensure a QApplication instance exists for widget-based tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
    assert helper_id == "azure_di"

    dialog._on_accept()
    config = dialog.result_config()
    assert config is not None
    assert config.subject_name == "Jane Doe"
    assert config.date_of_birth == "1975-08-19"
    assert config.case_description == "Referral for competency evaluation"
    assert config.conversion_helper == "azure_di"
    assert config.conversion_options == {}
    assert config.output_base == output_base
    assert config.selected_folders == ["bundle"]
    assert "Case-Name-2" in dialog._folder_preview_label.text()

    dialog.deleteLater()


def test_conversion_worker_uses_azure_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = ConversionJob(
        source_path=tmp_path / "sample.pdf",
        relative_path="sample.pdf",
        destination_path=tmp_path / "converted" / "sample.md",
        conversion_type="pdf",
    )

    job.source_path.parent.mkdir(parents=True, exist_ok=True)
    job.source_path.write_bytes(b"pdf")

    produced_markdown = job.destination_path.parent / "sample.md"
    produced_json = job.destination_path.parent / "sample.json"
    raw_markdown = job.destination_path.parent / "sample.azure.raw.md"
    raw_json = job.destination_path.parent / "sample.azure.raw.json"

    def fake_process(_self, source_path, output_dir, json_dir, endpoint, key):
        assert json_dir == output_dir
        produced_markdown.parent.mkdir(parents=True, exist_ok=True)
        produced_markdown.write_text("azure output\n<!-- PageBreak -->\nnext page")
        produced_json.write_text('{"kind":"raw"}')
        return str(produced_json), str(produced_markdown)

    class StubSettings:
        def __init__(self) -> None:
            pass

        def get(self, key, default=None):
            if key == "azure_di_settings":
                return {"endpoint": "https://example"}
            return default

        def get_api_key(self, provider):
            if provider == "azure_di":
                return "secret"
            return None

    monkeypatch.setattr("src.app.workers.conversion_worker.SecureSettings", StubSettings)
    monkeypatch.setattr(
        "src.app.workers.conversion_worker.ConversionWorker._process_with_azure",
        fake_process,
    )

    worker = ConversionWorker([job], helper="azure_di")
    worker._convert_pdf_with_azure(job)

    content = job.destination_path.read_text()
    assert "<!--- sample.pdf#page=1 --->" in content
    assert raw_markdown.exists()
    assert raw_json.exists()
    assert not produced_json.exists()

    post = frontmatter.load(job.destination_path)
    assert post.metadata["azure_raw_markdown_path"].endswith("/converted/sample.azure.raw.md")
    assert post.metadata["azure_raw_json_path"].endswith("/converted/sample.azure.raw.json")
    assert post.metadata["azure_raw_cached"] is False


def test_conversion_worker_reuses_cached_raw_azure_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = ConversionJob(
        source_path=tmp_path / "sample.pdf",
        relative_path="sample.pdf",
        destination_path=tmp_path / "converted" / "sample.md",
        conversion_type="pdf",
    )
    job.source_path.parent.mkdir(parents=True, exist_ok=True)
    job.source_path.write_bytes(b"pdf")
    job.destination_path.parent.mkdir(parents=True, exist_ok=True)

    checksum = hashlib.sha256(b"pdf").hexdigest()
    job.destination_path.write_text(f"---\nsources:\n  - checksum: {checksum}\n---\nold", encoding="utf-8")
    raw_markdown = job.destination_path.parent / "sample.azure.raw.md"
    raw_json = job.destination_path.parent / "sample.azure.raw.json"
    raw_markdown.write_text("cached raw\n<!-- PageBreak -->\npage2", encoding="utf-8")
    raw_json.write_text('{"cached": true}', encoding="utf-8")

    class StubSettings:
        def get(self, key, default=None):
            if key == "azure_di_settings":
                return {"endpoint": "https://example"}
            return default

        def get_api_key(self, provider):
            return "secret" if provider == "azure_di" else None

    called = {"value": False}

    def should_not_run(*_args, **_kwargs):
        called["value"] = True
        raise AssertionError("Azure conversion should not run when cache is valid")

    monkeypatch.setattr("src.app.workers.conversion_worker.SecureSettings", StubSettings)
    monkeypatch.setattr(
        "src.app.workers.conversion_worker.ConversionWorker._process_with_azure",
        should_not_run,
    )

    worker = ConversionWorker([job], helper="azure_di")
    worker._convert_pdf_with_azure(job)

    assert called["value"] is False
    post = frontmatter.load(job.destination_path)
    assert post.metadata["azure_raw_cached"] is True


def test_conversion_worker_raises_without_azure_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = ConversionJob(
        source_path=tmp_path / "sample.pdf",
        relative_path="sample.pdf",
        destination_path=tmp_path / "converted" / "sample.md",
        conversion_type="pdf",
    )

    job.source_path.parent.mkdir(parents=True, exist_ok=True)
    job.source_path.write_bytes(b"pdf")

    class EmptySettings:
        def __init__(self) -> None:
            pass

        def get(self, key, default=None):
            return default

        def get_api_key(self, provider):
            return None

    monkeypatch.setattr("src.app.workers.conversion_worker.SecureSettings", EmptySettings)

    worker = ConversionWorker([job], helper="azure_di")
    with pytest.raises(RuntimeError, match="Azure Document Intelligence credentials"):
        worker._convert_pdf_with_azure(job)


def test_project_manager_update_conversion_helper_replaces_options(qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.conversion_settings.options = {"legacy": True}

    manager.update_conversion_helper("azure_di")
    assert manager.conversion_settings.helper == "azure_di"
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
    (converted_folder / "example.md").write_text("content")

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
    (converted_root / "doc.md").write_text("content")

    dialog = BulkAnalysisGroupDialog(manager.project_dir)
    try:
        tree = dialog.file_tree
        top_labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert "folder" in top_labels
        folder_item = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0) == "folder")
        child_names = [folder_item.child(i).text(0) for i in range(folder_item.childCount())]
        assert "doc.md" in child_names
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
    (converted_root / "doc.md").write_text("content")

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
    (converted_dir / "doc.md").write_text("converted", encoding="utf-8")

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
