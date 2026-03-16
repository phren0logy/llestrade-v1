"""Integration tests for ProjectWorkspace bulk analysis behaviour."""

from __future__ import annotations

from pathlib import Path
import pytest
from PySide6.QtCore import QCoreApplication, QObject, QRunnable, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton, QMessageBox

from src.app.core.bulk_analysis_runner import PromptBundle
from src.app.core.job_cost_estimates import CostForecast

from src.app.core.file_tracker import FileTracker, WorkspaceGroupMetrics
from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.app.ui.stages import project_workspace
from src.app.ui.stages.project_workspace import ProjectWorkspace
from src.app.ui.workspace.bulk_tab import BulkAnalysisTab
from src.app.workers import bulk_analysis_worker
from src.app.workers.llm_backend import GatewayAccessCheck
from src.app.workers.progress import WorkerProgressDetail


class _ImmediateThreadPool:
    """Thread pool stub that executes QRunnables synchronously."""

    def __init__(self) -> None:
        self.last_worker = None

    def start(self, worker: QRunnable) -> None:  # pragma: no cover - trivial stub
        self.last_worker = worker
        worker.run()


class _CaptureThreadPool:
    """Thread pool stub that captures the worker without executing it."""

    def __init__(self) -> None:
        self.last_worker: QRunnable | None = None

    def start(self, worker: QRunnable) -> None:  # pragma: no cover - trivial stub
        self.last_worker = worker


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _create_project_with_group(tmp_path: Path) -> tuple[ProjectManager, BulkAnalysisGroup]:
    projects_root = tmp_path / "projects"
    projects_root.mkdir()

    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Bulk Analysis Demo"))

    converted_doc = manager.project_dir / "converted_documents" / "folder" / "record.md"
    converted_doc.parent.mkdir(parents=True, exist_ok=True)
    converted_doc.write_text("# Heading\nBody", encoding="utf-8")

    # Ensure the tracker sees our converted document so the workspace resolves it.
    FileTracker(manager.project_dir).scan()

    group = BulkAnalysisGroup.create(
        name="Demo Group",
        files=["folder/record.md"],
        provider_id="anthropic",
        model="claude-sonnet-4-5",
    )
    group.model_context_window = 200_000
    saved = manager.save_bulk_analysis_group(group)
    return manager, saved


def _create_project_with_combined_group(tmp_path: Path) -> tuple[ProjectManager, BulkAnalysisGroup]:
    projects_root = tmp_path / "projects"
    projects_root.mkdir()

    manager = ProjectManager()
    manager.create_project(projects_root, ProjectMetadata(case_name="Combined Demo"))

    converted_doc = manager.project_dir / "converted_documents" / "folder" / "record.md"
    converted_doc.parent.mkdir(parents=True, exist_ok=True)
    converted_doc.write_text("# Heading\nBody", encoding="utf-8")
    FileTracker(manager.project_dir).scan()

    group = BulkAnalysisGroup.create(
        name="Combined Demo",
        provider_id="anthropic",
        model="claude-sonnet-4-5",
    )
    group.operation = "combined"
    group.combine_converted_files = ["folder/record.md"]
    group.model_context_window = 200_000
    saved = manager.save_bulk_analysis_group(group)
    return manager, saved


def _find_button(action_widget, text: str) -> QPushButton:
    for button in action_widget.findChildren(QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"Button with text '{text}' not found")


def test_bulk_analysis_tab_uses_fixed_system_font_for_log(qt_app: QApplication) -> None:
    assert qt_app is not None

    tab = BulkAnalysisTab()
    expected = QFontDatabase.systemFont(QFontDatabase.FixedFont)

    assert tab.log_text.font().family() == expected.family()
    assert tab.log_text.font().pointSize() == 11

    tab.deleteLater()


def test_workspace_run_executes_worker_and_updates_ui(tmp_path: Path, qt_app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    pool = _ImmediateThreadPool()
    monkeypatch.setattr(project_workspace, "get_worker_pool", lambda: pool)

    monkeypatch.setattr(
        bulk_analysis_worker.BulkAnalysisWorker,
        "_create_provider",
        lambda self, config: object(),
    )
    monkeypatch.setattr(
        bulk_analysis_worker.BulkAnalysisWorker,
        "_invoke_provider",
        lambda self, provider, config, prompt, system_prompt, temperature=0.1, max_tokens=32000, **kwargs: "Summary output",
    )
    monkeypatch.setattr(
        bulk_analysis_worker,
        "should_chunk",
        lambda content, provider_id, model_name, **_kwargs: (False, 10, 8192),
    )

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None
    monkeypatch.setattr(controller, "_verify_gateway_before_run", lambda **_kwargs: True)
    table = controller.tab.table
    assert table.rowCount() == 1

    action_widget = table.cellWidget(0, 6)
    run_button = _find_button(action_widget, "Run Pending")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    run_button.click()
    for _ in range(10):
        QCoreApplication.processEvents()
        if not controller.is_running(group.group_id):
            break
    else:
        pytest.fail("Bulk analysis run did not complete")

    expected_output = manager.project_dir / "bulk_analysis" / group.slug / "folder" / "record_analysis.md"
    assert expected_output.exists()
    assert "Summary output" in expected_output.read_text(encoding="utf-8")

    assert controller.progress_for(group.group_id) is None
    assert controller.progress_for(group.group_id) is None
    assert run_button.isEnabled()

    status_item = table.item(0, 3)
    assert status_item is not None
    assert status_item.text() == "Ready"

    workspace.deleteLater()


def test_workspace_defaults_to_bulk_tab_when_converted_docs_exist(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, _group = _create_project_with_group(tmp_path)
    pool = _CaptureThreadPool()
    monkeypatch.setattr(project_workspace, "get_worker_pool", lambda: pool)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    current_text = workspace._tabs.tabText(workspace._tabs.currentIndex())
    assert current_text == "Bulk Analysis"

    workspace.deleteLater()


def test_workspace_conversion_finish_auto_runs_pending_groups(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    pool = _CaptureThreadPool()
    monkeypatch.setattr(project_workspace, "get_worker_pool", lambda: pool)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    captured_group_ids: list[str] = []

    def fake_auto_run(groups):
        captured_group_ids.extend(candidate.group_id for candidate in groups)
        return 1

    assert workspace.bulk_controller is not None
    monkeypatch.setattr(workspace.bulk_controller, "auto_run_pending_groups", fake_auto_run)

    workspace._on_conversion_finished(worker=None, jobs=[], successes=1, failures=0)

    assert captured_group_ids == [group.group_id]

    workspace.deleteLater()


def test_workspace_conversion_finish_shows_single_fatal_error(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, _group = _create_project_with_group(tmp_path)
    pool = _CaptureThreadPool()
    monkeypatch.setattr(project_workspace, "get_worker_pool", lambda: pool)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    criticals: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: criticals.append(message) or QMessageBox.Ok,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    workspace._conversion_errors = ["sample.pdf: generic failure"]
    workspace._conversion_fatal_error = "Azure credentials rejected"
    workspace._on_conversion_finished(worker=None, jobs=[], successes=0, failures=1)

    assert criticals == ["Azure credentials rejected"]
    assert warnings == []

    workspace.deleteLater()


class _StubBulkAnalysisWorker(QObject, QRunnable):
    """Bulk-analysis worker stub that supports cancellation flow."""

    progress = Signal(int, int, str)
    file_failed = Signal(str, str)
    finished = Signal(int, int)
    log_message = Signal(str)

    def __init__(
        self,
        *,
        project_dir: Path,
        group: BulkAnalysisGroup,
        files: list[str],
        metadata: ProjectMetadata | None,
        default_provider: tuple[str, str | None],
        force_rerun: bool = False,
        preserve_recovery_on_signature_mismatch: bool = False,
        placeholder_values: dict[str, str] | None = None,
        project_name: str = "",
        estimate_summary=None,  # noqa: ANN001
        llm_backend=None,  # noqa: ANN001
    ) -> None:
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.setAutoDelete(True)
        self.group = group
        self.cancel_called = False
        self.force_rerun = force_rerun
        self.preserve_recovery_on_signature_mismatch = preserve_recovery_on_signature_mismatch

    def run(self) -> None:  # pragma: no cover - trivial stub
        self.log_message.emit("started")

    def cancel(self) -> None:  # pragma: no cover - trivial stub
        self.cancel_called = True


def test_workspace_cancel_updates_status_and_cleans_state(tmp_path: Path, qt_app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    pool = _CaptureThreadPool()
    monkeypatch.setattr(project_workspace, "get_worker_pool", lambda: pool)
    monkeypatch.setattr("src.app.ui.workspace.services.bulk.BulkAnalysisWorker", _StubBulkAnalysisWorker)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None
    monkeypatch.setattr(controller, "_verify_gateway_before_run", lambda **_kwargs: True)
    table = controller.tab.table
    action_widget = table.cellWidget(0, 6)
    run_button = _find_button(action_widget, "Run Pending")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    run_button.click()
    QCoreApplication.processEvents()

    action_widget = table.cellWidget(0, 6)
    cancel_button = _find_button(action_widget, "Cancel")
    run_button = _find_button(action_widget, "Run Pending")

    assert pool.last_worker is not None
    worker = pool.last_worker
    assert isinstance(worker, _StubBulkAnalysisWorker)

    assert controller.is_running(group.group_id)
    assert controller.progress_for(group.group_id) == (0, 1)

    cancel_button.click()
    for _ in range(10):
        QCoreApplication.processEvents()
        if controller.is_cancelling(group.group_id):
            break
    else:
        pytest.fail("Cancellation did not register")

    assert worker.cancel_called is True
    assert controller.is_cancelling(group.group_id)

    # Simulate the worker completing after cancellation.
    worker.finished.emit(0, 0)
    for _ in range(10):
        QCoreApplication.processEvents()
        if not controller.is_running(group.group_id):
            break
    else:
        pytest.fail("Bulk analysis worker did not finish after cancellation")

    assert not controller.is_running(group.group_id)
    assert not controller.is_cancelling(group.group_id)
    assert controller.progress_for(group.group_id) is None

    refreshed_widget = table.cellWidget(0, 6)
    refreshed_run = _find_button(refreshed_widget, "Run Pending")
    refreshed_cancel = _find_button(refreshed_widget, "Cancel")
    assert refreshed_run.isEnabled()
    assert not refreshed_cancel.isEnabled()

    workspace.deleteLater()


def test_workspace_bulk_progress_detail_updates_active_progress_widget(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    pool = _CaptureThreadPool()
    monkeypatch.setattr(project_workspace, "get_worker_pool", lambda: pool)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None
    controller._running_groups.add(group.group_id)
    controller._progress_map[group.group_id] = (0, 1)

    controller._handle_progress_detail(
        group.group_id,
        WorkerProgressDetail(
            run_kind="bulk_map",
            phase="chunk_started",
            label="Processing chunks",
            percent=12,
            completed=0,
            total=1,
            document_path="folder/record.md",
            chunk_index=3,
            chunk_total=24,
            chunks_completed=3,
            chunks_in_flight=2,
        ),
    )

    assert controller.tab.active_progress_widget.isHidden() is False
    assert controller.tab.active_progress_bar.value() == 12
    detail_text = controller.tab.active_progress_detail_label.text()
    assert "3/24 chunks" in detail_text
    assert "2 in flight" in detail_text

    workspace.deleteLater()


def test_workspace_open_recovery_dialog_uses_qdialog_accept_code(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    class _FakeRecoveryDialog:
        def __init__(self, **_kwargs):  # noqa: ANN003
            self.selected_action = None

        def exec(self) -> int:
            return QDialog.Accepted

        def selected_documents(self) -> list[str]:
            return []

    monkeypatch.setattr("src.app.ui.workspace.controllers.bulk.BulkRecoveryDialog", _FakeRecoveryDialog)

    controller.open_recovery_dialog(group)

    workspace.deleteLater()


def test_placeholder_analysis_includes_metadata_values(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    metadata = manager.metadata
    assert metadata is not None
    metadata.subject_name = "Jane Roe"
    metadata.date_of_birth = "1970-01-01"
    metadata.case_description = "Sample case description."

    bundle = PromptBundle(
        system_template="System uses {subject_name} and {case_info}",
        user_template="User prompt for {subject_name} born {subject_dob}",
    )
    monkeypatch.setattr(
        "src.app.ui.workspace.controllers.bulk.load_prompts",
        lambda *_args, **_kwargs: bundle,
    )

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    controller = workspace.bulk_controller
    assert controller is not None

    analysis, missing_required, missing_optional = controller._analyse_placeholders(group)
    assert analysis is not None
    assert "subject_name" not in missing_optional
    assert "subject_dob" not in missing_optional
    assert "case_info" not in missing_optional
    assert "document_name" not in missing_optional

    workspace.deleteLater()


def test_combined_run_prompts_when_inputs_are_stale(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_combined_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None
    monkeypatch.setattr(controller, "_verify_gateway_before_run", lambda **_kwargs: True)

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=(),
        converted_count=0,
        bulk_analysis_total=0,
        pending_bulk_analysis=0,
        pending_files=(),
        operation="combined",
        combined_input_count=6,
        combined_latest_path="bulk_analysis/combined-demo/reduce/combined_20260307-1010.md",
        combined_latest_at=None,
        combined_is_stale=True,
        combined_last_run_input_count=4,
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))

    prompts: list[tuple[str, str]] = []

    def fake_question(_parent, title, text, *_args, **_kwargs):
        prompts.append((title, text))
        return QMessageBox.No

    run_called = {"value": False}

    def fake_run_combined(**_kwargs):  # noqa: ANN003
        run_called["value"] = True
        return True

    monkeypatch.setattr(QMessageBox, "question", fake_question)
    monkeypatch.setattr(controller._service, "run_combined", fake_run_combined)

    controller.start_combined_run(group, force_rerun=False)

    assert run_called["value"] is False
    assert prompts
    assert prompts[0][0] == "Stale Combined Inputs"
    assert "Current resolved inputs: 6" in prompts[0][1]
    assert "Last run input count: 4" in prompts[0][1]

    workspace.deleteLater()


def test_bulk_map_run_passes_estimate_summary_to_service(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(
            available=True,
            best_estimate=1.0,
            ceiling=2.0,
        ),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    captured: dict[str, object] = {}

    def fake_run_map(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return True

    monkeypatch.setattr(controller._service, "run_map", fake_run_map)

    assert controller.start_map_run(group, force_rerun=False) is True
    assert captured["estimate_summary"] == {
        "available": True,
        "best_estimate": 1.0,
        "ceiling": 2.0,
        "spent_actual": None,
        "remaining_best_estimate": None,
        "remaining_ceiling": None,
        "projected_total_best_estimate": None,
        "projected_total_ceiling": None,
        "reason": None,
    }

    workspace.deleteLater()


def test_bulk_map_run_blocks_when_gateway_key_is_rejected(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(
        controller._service,
        "verify_gateway_access",
        lambda **_kwargs: GatewayAccessCheck(
            ok=False,
            kind="auth_invalid",
            status_code=401,
            message="Unauthorized - Key not found",
            base_url="https://gateway.example.com",
            route="bulk",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
        ),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    run_called = {"value": False}
    monkeypatch.setattr(controller._service, "run_map", lambda **_kwargs: run_called.__setitem__("value", True) or True)

    assert controller.start_map_run(group, force_rerun=False) is False
    assert run_called["value"] is False
    assert warnings
    assert "Pydantic AI Gateway app key was rejected" in warnings[0]

    workspace.deleteLater()


def test_bulk_map_run_can_continue_when_gateway_probe_is_rate_limited(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    replies = iter((QMessageBox.Yes, QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(
        controller._service,
        "verify_gateway_access",
        lambda **_kwargs: GatewayAccessCheck(
            ok=False,
            kind="rate_limited",
            status_code=429,
            message="Provider capacity reached for anthropic. Retry soon.",
            base_url="https://gateway.example.com",
            route="bulk",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
            retry_after_seconds=7.0,
        ),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    run_called = {"value": False}
    monkeypatch.setattr(controller._service, "run_map", lambda **_kwargs: run_called.__setitem__("value", True) or True)

    assert controller.start_map_run(group, force_rerun=False) is True
    assert run_called["value"] is True
    assert warnings == []

    workspace.deleteLater()


def test_bulk_map_run_stays_blocked_when_user_cancels_rate_limited_gateway_probe(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    replies = iter((QMessageBox.Yes, QMessageBox.No))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(
        controller._service,
        "verify_gateway_access",
        lambda **_kwargs: GatewayAccessCheck(
            ok=False,
            kind="rate_limited",
            status_code=429,
            message="Provider capacity reached for anthropic. Retry soon.",
            base_url="https://gateway.example.com",
            route="bulk",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
            retry_after_seconds=7.0,
        ),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    run_called = {"value": False}
    monkeypatch.setattr(controller._service, "run_map", lambda **_kwargs: run_called.__setitem__("value", True) or True)

    assert controller.start_map_run(group, force_rerun=False) is False
    assert run_called["value"] is False
    assert warnings == []

    workspace.deleteLater()


def test_bulk_auto_run_skips_rate_limited_gateway_probe_without_modal(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    monkeypatch.setattr(
        controller._service,
        "verify_gateway_access",
        lambda **_kwargs: GatewayAccessCheck(
            ok=False,
            kind="rate_limited",
            status_code=429,
            message="Provider capacity reached for anthropic. Retry soon.",
            base_url="https://gateway.example.com",
            route="bulk",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
            retry_after_seconds=7.0,
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("question dialog should not open")),
    )

    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(controller, "_handle_log", lambda group_id, message: logs.append((group_id, message)))

    run_called = {"value": False}
    monkeypatch.setattr(controller._service, "run_map", lambda **_kwargs: run_called.__setitem__("value", True) or True)

    assert controller.auto_run_pending_groups([group]) == 0
    assert run_called["value"] is False
    assert logs == [
        (
            "bulk",
            "Skipping run for anthropic/claude-sonnet-4-5 via route 'bulk': gateway probe returned HTTP 429. Retry-After: 7.0s.",
        )
    ]

    workspace.deleteLater()


def test_combined_run_blocks_when_gateway_route_is_missing(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_combined_group(tmp_path)

    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=(),
        converted_count=0,
        bulk_analysis_total=0,
        pending_bulk_analysis=0,
        pending_files=(),
        operation="combined",
        combined_input_count=2,
        combined_latest_path=None,
        combined_latest_at=None,
        combined_is_stale=False,
        combined_last_run_input_count=None,
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_combined_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(
        controller._service,
        "verify_gateway_access",
        lambda **_kwargs: GatewayAccessCheck(
            ok=False,
            kind="route_missing",
            status_code=404,
            message="Route not found: bulk",
            base_url="https://gateway.example.com",
            route="bulk",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
        ),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    run_called = {"value": False}
    monkeypatch.setattr(controller._service, "run_combined", lambda **_kwargs: run_called.__setitem__("value", True) or True)

    controller.start_combined_run(group, force_rerun=False)

    assert run_called["value"] is False
    assert warnings
    assert "route/provider mapping is not available" in warnings[0]

    workspace.deleteLater()


def test_bulk_map_run_prompt_change_resume_keeps_remaining_docs(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    monkeypatch.setattr(controller, "_verify_gateway_before_run", lambda **_kwargs: True)
    monkeypatch.setattr(
        controller,
        "_load_map_recovery_manifest",
        lambda _group: {
            "prompt_state": {
                "system": {"logical_name": "prompt.md", "content_hash": "old", "missing": False},
                "user": {"logical_name": "user.md", "content_hash": "same", "missing": False},
            }
        },
    )
    monkeypatch.setattr(
        "src.app.ui.workspace.controllers.bulk.capture_bulk_prompt_state",
        lambda *_args, **_kwargs: {
            "system": {"logical_name": "prompt.md", "content_hash": "new", "missing": False},
            "user": {"logical_name": "user.md", "content_hash": "same", "missing": False},
        },
    )

    replies = iter((QMessageBox.Yes, QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(replies))

    captured: dict[str, object] = {}
    monkeypatch.setattr(controller._service, "run_map", lambda **kwargs: captured.update(kwargs) or True)

    assert controller.start_map_run(group, force_rerun=False) is True
    assert captured["preserve_recovery_on_signature_mismatch"] is True
    assert captured["force_rerun"] is False

    workspace.deleteLater()


def test_bulk_map_run_missing_prompt_defaults_to_restart_with_replacement(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    replacement_prompt = manager.project_dir / "prompts" / "replacement_system.md"
    replacement_prompt.parent.mkdir(parents=True, exist_ok=True)
    replacement_prompt.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_map_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    monkeypatch.setattr(controller, "_verify_gateway_before_run", lambda **_kwargs: True)
    monkeypatch.setattr(
        controller,
        "_load_map_recovery_manifest",
        lambda _group: {
            "prompt_state": {
                "system": {"logical_name": "missing.md", "content_hash": "old", "missing": False},
            }
        },
    )

    def fake_capture(_project_dir, candidate_group, _metadata):
        if candidate_group.system_prompt_path == str(replacement_prompt):
            return {
                "system": {"logical_name": replacement_prompt.name, "content_hash": "new", "missing": False},
                "user": {"logical_name": "user.md", "content_hash": "same", "missing": False},
            }
        return {
            "system": {"logical_name": "missing.md", "content_hash": "old", "missing": True},
            "user": {"logical_name": "user.md", "content_hash": "same", "missing": False},
        }

    monkeypatch.setattr("src.app.ui.workspace.controllers.bulk.capture_bulk_prompt_state", fake_capture)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(replacement_prompt), "Markdown Files (*.md)"))
    replies = iter((QMessageBox.No, QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(replies))

    captured: dict[str, object] = {}
    monkeypatch.setattr(controller._service, "run_map", lambda **kwargs: captured.update(kwargs) or True)

    assert controller.start_map_run(group, force_rerun=False) is True
    assert captured["force_rerun"] is True
    assert captured["preserve_recovery_on_signature_mismatch"] is False
    assert captured["group"].system_prompt_path == str(replacement_prompt)

    workspace.deleteLater()


def test_bulk_map_auto_run_skips_when_prompt_recovery_needs_review(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_group(tmp_path)
    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=("folder/record.md",),
        converted_count=1,
        bulk_analysis_total=0,
        pending_bulk_analysis=1,
        pending_files=("folder/record.md",),
        operation="per_document",
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(
        controller,
        "_load_map_recovery_manifest",
        lambda _group: {"prompt_state": {"system": {"logical_name": "prompt.md", "content_hash": "old", "missing": False}}},
    )
    monkeypatch.setattr(
        "src.app.ui.workspace.controllers.bulk.capture_bulk_prompt_state",
        lambda *_args, **_kwargs: {"system": {"logical_name": "prompt.md", "content_hash": "new", "missing": False}},
    )

    logs: list[str] = []
    monkeypatch.setattr(controller, "_handle_log", lambda _gid, message: logs.append(message))
    run_called = {"value": False}
    monkeypatch.setattr(controller._service, "run_map", lambda **_kwargs: run_called.__setitem__("value", True) or True)

    assert controller.start_map_run(group, force_rerun=False, interactive=False) is False
    assert run_called["value"] is False
    assert logs
    assert "prompt recovery requires manual review" in logs[0]

    workspace.deleteLater()


def test_combined_run_prompt_change_forces_restart(
    tmp_path: Path,
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qt_app is not None

    manager, group = _create_project_with_combined_group(tmp_path)
    workspace = ProjectWorkspace()
    workspace.set_project(manager)
    QCoreApplication.processEvents()

    controller = workspace.bulk_controller
    assert controller is not None

    metrics = WorkspaceGroupMetrics(
        group_id=group.group_id,
        name=group.name,
        slug=group.slug or group.folder_name,
        converted_files=(),
        converted_count=0,
        bulk_analysis_total=0,
        pending_bulk_analysis=0,
        pending_files=(),
        operation="combined",
        combined_input_count=1,
        combined_latest_path="bulk_analysis/combined-demo/reduce/combined.md",
        combined_latest_at=None,
        combined_is_stale=False,
        combined_last_run_input_count=1,
    )

    monkeypatch.setattr(controller, "_resolve_group_metrics", lambda _group_id: metrics)
    monkeypatch.setattr(controller, "_analyse_placeholders", lambda _group: (None, set(), set()))
    monkeypatch.setattr(
        controller,
        "_forecast_combined_run",
        lambda *_args, **_kwargs: CostForecast(available=True, best_estimate=1.0, ceiling=2.0),
    )
    monkeypatch.setattr(controller, "_verify_gateway_before_run", lambda **_kwargs: True)
    monkeypatch.setattr(
        controller,
        "_load_reduce_recovery_manifest",
        lambda _group: {
            "prompt_state": {
                "system": {"logical_name": "prompt.md", "content_hash": "old", "missing": False},
            }
        },
    )
    monkeypatch.setattr(
        "src.app.ui.workspace.controllers.bulk.capture_bulk_prompt_state",
        lambda *_args, **_kwargs: {
            "system": {"logical_name": "prompt.md", "content_hash": "new", "missing": False},
        },
    )

    replies = iter((QMessageBox.Yes, QMessageBox.Yes, QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(replies))

    captured: dict[str, object] = {}
    monkeypatch.setattr(controller._service, "run_combined", lambda **kwargs: captured.update(kwargs) or True)

    controller.start_combined_run(group, force_rerun=False)

    assert captured["force_rerun"] is True

    workspace.deleteLater()
