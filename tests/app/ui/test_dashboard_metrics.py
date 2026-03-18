"""Tests for the dashboard metrics API provided by ProjectManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

_ = PySide6

from src.app.core.converted_documents import converted_artifact_relative
from src.app.core.file_tracker import DashboardMetrics
from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.ui.stages.welcome_stage import WelcomeStage
from src.app.workers import DashboardWorker, get_worker_pool


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_get_dashboard_metrics_scans_when_missing(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Metrics Demo"))

    metrics = manager.get_dashboard_metrics()

    assert metrics.imported_total == 0
    assert metrics.bulk_analysis_total == 0
    assert metrics.pending_bulk_analysis == 0
    assert metrics.last_scan is not None
    assert manager.dashboard_metrics == metrics
    assert manager.source_state.last_scan is not None


def test_dashboard_metrics_refresh_counts_files(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Metrics Files"))

    converted = manager.project_dir / "converted_documents" / "folder"
    converted.mkdir(parents=True, exist_ok=True)
    (converted / Path(converted_artifact_relative("folder/doc1.pdf")).name).write_text("converted")
    (converted / Path(converted_artifact_relative("folder/doc2.pdf")).name).write_text("converted")

    metrics = manager.get_dashboard_metrics(refresh=True)

    assert metrics.imported_total == 2
    assert metrics.bulk_analysis_total == 0
    assert metrics.pending_bulk_analysis == 2
    assert manager.dashboard_metrics == metrics


def test_dashboard_metrics_persist_across_project_reload(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    project_path = manager.create_project(tmp_path, ProjectMetadata(case_name="Persist Metrics"))

    converted = manager.project_dir / "converted_documents"
    converted.mkdir(exist_ok=True)
    (converted / Path(converted_artifact_relative("doc.pdf")).name).write_text("body")

    first_metrics = manager.get_dashboard_metrics(refresh=True)
    manager.save_project()

    reloaded = ProjectManager()
    assert reloaded.load_project(project_path)

    assert reloaded.dashboard_metrics.imported_total == first_metrics.imported_total
    assert reloaded.dashboard_metrics.last_scan is not None

    cached_metrics = reloaded.get_dashboard_metrics()
    assert cached_metrics == reloaded.dashboard_metrics
    assert cached_metrics.imported_total == 1


def test_read_dashboard_metrics_from_disk(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    project_path = manager.create_project(tmp_path, ProjectMetadata(case_name="Disk Metrics"))

    converted = manager.project_dir / "converted_documents"
    converted.mkdir(exist_ok=True)
    (converted / Path(converted_artifact_relative("doc.pdf")).name).write_text("body")

    manager.get_dashboard_metrics(refresh=True)
    manager.save_project()

    metrics = ProjectManager.read_dashboard_metrics_from_disk(project_path)

    assert metrics.imported_total == 1
    assert metrics.last_scan is not None
    assert metrics.pending_bulk_analysis == 1

def test_workspace_metrics_include_group_coverage(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Workspace Coverage"))

    converted_dir = manager.project_dir / "converted_documents" / "folder"
    converted_dir.mkdir(parents=True, exist_ok=True)
    (converted_dir / Path(converted_artifact_relative("folder/doc1.pdf")).name).write_text("converted")
    (converted_dir / Path(converted_artifact_relative("folder/doc2.pdf")).name).write_text("converted")

    group = BulkAnalysisGroup.create(name="Case Files", directories=["folder"])
    manager.save_bulk_analysis_group(group)

    outputs_dir = manager.project_dir / "bulk_analysis" / group.slug / "outputs" / "folder"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "doc1.pdf_analysis.md").write_text("analysis")

    metrics = manager.get_workspace_metrics(refresh=True)

    assert metrics.dashboard.imported_total == 2

    group_metrics = metrics.groups[group.group_id]
    assert group_metrics.converted_count == 2
    assert group_metrics.bulk_analysis_total == 1
    assert group_metrics.pending_bulk_analysis == 1
    assert set(group_metrics.converted_files) == {
        converted_artifact_relative("folder/doc1.pdf"),
        converted_artifact_relative("folder/doc2.pdf"),
    }


def test_workspace_metrics_include_combined_last_run_input_count(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Workspace Combined"))

    map_dir = manager.project_dir / "bulk_analysis" / "per-document-summaries" / "Conner"
    map_dir.mkdir(parents=True, exist_ok=True)
    first_map = map_dir / "doc1_analysis.md"
    second_map = map_dir / "doc2_analysis.md"
    first_map.write_text("analysis one", encoding="utf-8")
    second_map.write_text("analysis two", encoding="utf-8")

    combined = BulkAnalysisGroup.create(name="Comprehensive Summary")
    combined.operation = "combined"
    combined.combine_map_groups = ["per-document-summaries"]
    combined_saved = manager.save_bulk_analysis_group(combined)

    reduce_dir = manager.project_dir / "bulk_analysis" / combined_saved.slug / "reduce"
    reduce_dir.mkdir(parents=True, exist_ok=True)
    combined_md = reduce_dir / "combined_20260307-1010.md"
    combined_md.write_text("combined output", encoding="utf-8")
    combined_manifest = combined_md.with_suffix(".manifest.json")
    combined_manifest.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "kind": "map",
                        "path": "map/per-document-summaries/Conner/doc1_analysis.md",
                        "mtime": first_map.stat().st_mtime,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metrics = manager.get_workspace_metrics(refresh=True)

    group_metrics = metrics.groups[combined_saved.group_id]
    assert group_metrics.operation == "combined"
    assert group_metrics.combined_input_count == 2
    assert group_metrics.combined_last_run_input_count == 1
    assert group_metrics.combined_is_stale is True


def test_overlapping_groups_keep_pending_counts_isolated(tmp_path: Path, qt_app: QApplication) -> None:
    assert qt_app is not None
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Overlap Metrics"))

    converted = manager.project_dir / "converted_documents" / "folder"
    converted.mkdir(parents=True, exist_ok=True)
    (converted / Path(converted_artifact_relative("folder/doc1.pdf")).name).write_text("converted 1", encoding="utf-8")
    (converted / Path(converted_artifact_relative("folder/doc2.pdf")).name).write_text("converted 2", encoding="utf-8")

    group_all = manager.save_bulk_analysis_group(
        BulkAnalysisGroup.create(name="All Docs", directories=["folder"])
    )
    group_single = manager.save_bulk_analysis_group(
        BulkAnalysisGroup.create(name="Single Doc", files=[converted_artifact_relative("folder/doc1.pdf")])
    )

    outputs_dir = manager.project_dir / "bulk_analysis" / group_all.slug / "folder"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "doc1.pdf_analysis.md").write_text("analysis 1", encoding="utf-8")
    (outputs_dir / "doc2.pdf_analysis.md").write_text("analysis 2", encoding="utf-8")

    metrics = manager.get_workspace_metrics(refresh=True)
    all_metrics = metrics.groups[group_all.group_id]
    single_metrics = metrics.groups[group_single.group_id]

    assert all_metrics.pending_bulk_analysis == 0
    assert single_metrics.pending_bulk_analysis == 1
    assert single_metrics.pending_files == (converted_artifact_relative("folder/doc1.pdf"),)


def test_welcome_stage_uses_persisted_metrics(
    tmp_path: Path, qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("FRD_SETTINGS_DIR", str(tmp_path / "settings"))
    manager = ProjectManager()
    project_path = manager.create_project(tmp_path, ProjectMetadata(case_name="Welcome Metrics"))

    converted = manager.project_dir / "converted_documents"
    converted.mkdir(exist_ok=True)
    (converted / Path(converted_artifact_relative("doc.pdf")).name).write_text("body")

    manager.get_dashboard_metrics(refresh=True)
    manager.save_project()

    stage = WelcomeStage()
    try:
        stage.showEvent(QShowEvent())
        stats_text = stage._project_stats_text(manager.project_path)
    finally:
        stage.deleteLater()

    assert "Converted: 1" in stats_text
    assert "Highlights: 0 of 1 (pending 1)" in stats_text
    assert "Bulk analysis: 0 of 1 (pending 1)" in stats_text
    assert "Last scan" in stats_text


def test_welcome_stage_refreshes_on_show_event(
    tmp_path: Path, qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("FRD_SETTINGS_DIR", str(tmp_path / "settings_refresh"))
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Refresh Metrics"))

    converted_dir = manager.project_dir / "converted_documents"
    converted_dir.mkdir(exist_ok=True)
    (converted_dir / Path(converted_artifact_relative("doc.pdf")).name).write_text("body")

    manager.get_dashboard_metrics(refresh=True)
    manager.save_project()

    stage = WelcomeStage()
    try:
        stage.showEvent(QShowEvent())
        qt_app.processEvents()

        stats_label = _find_stats_label(stage)
        assert stats_label is not None
        initial_text = stats_label.text()
        assert "Highlights: 0 of 1 (pending 1)" in initial_text
        assert "Bulk analysis: 0 of 1 (pending 1)" in initial_text

        outputs_dir = manager.project_dir / "bulk_analysis" / "manual" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "doc.pdf_analysis.md").write_text("analysis")

        manager.get_dashboard_metrics(refresh=True)
        manager.save_project()

        stage.showEvent(QShowEvent())
        qt_app.processEvents()

        updated_label = _find_stats_label(stage)
        assert updated_label is not None
        updated_text = updated_label.text()
        assert "Bulk analysis: 1 of 1" in updated_text
    finally:
        stage.deleteLater()


def test_welcome_stage_excludes_bedrock_status(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None
    monkeypatch.setenv("FRD_SETTINGS_DIR", str(tmp_path / "settings_no_bedrock"))

    stage = WelcomeStage()
    try:
        labels = []
        for index in range(stage._api_layout.count()):
            item = stage._api_layout.itemAt(index)
            widget = item.widget()
            if isinstance(widget, QLabel):
                labels.append(widget.text())
        assert not any("Bedrock" in text for text in labels)
    finally:
        stage.deleteLater()


def _find_stats_label(stage: WelcomeStage) -> QLabel | None:
    for index in range(stage._recent_projects_layout.count()):
        item = stage._recent_projects_layout.itemAt(index)
        widget = item.widget()
        if isinstance(widget, QScrollArea):
            container = widget.widget()
            if not container:
                continue
            layout = container.layout()
            if not layout:
                continue
            for row in range(layout.count()):
                card = layout.itemAt(row).widget()
                if card is None:
                    continue
                for label in card.findChildren(QLabel):
                    if "Converted" in label.text():
                        return label
    return None


class _DummyWorker(DashboardWorker):
    def __init__(self) -> None:
        super().__init__(worker_name="dummy")
        self.invoked = False

    def _run(self) -> None:
        self.invoked = True


def test_worker_pool_singleton(qt_app: QApplication) -> None:
    pool_a = get_worker_pool()
    pool_b = get_worker_pool()
    assert pool_a is pool_b
    assert pool_a.maxThreadCount() == 3


def test_dashboard_worker_base_helpers() -> None:
    worker = _DummyWorker()
    assert not worker.is_cancelled()
    worker.run()
    assert worker.invoked
    worker.cancel()
    assert worker.is_cancelled()


def test_dashboard_metrics_from_dict_accepts_legacy_keys() -> None:
    payload = {
        "imported_total": 5,
        "processed_total": 4,
        "summaries_total": 3,
        "pending_processing": 1,
        "pending_summaries": 2,
    }

    metrics = DashboardMetrics.from_dict(payload)

    assert metrics.bulk_analysis_total == 3
    assert metrics.pending_bulk_analysis == 2
