"""Focused tests for ReportsController placeholder-validation flow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

_ = PySide6

import src.app.ui.workspace.controllers.reports as reports_module
from src.app.ui.workspace.controllers.reports import ReportsController
from src.app.ui.workspace.reports_tab import ReportsTab
from src.app.workers.llm_backend import GatewayAccessCheck
from src.app.workers.progress import WorkerProgressDetail


class _ServiceStub:
    def __init__(self) -> None:
        self.gateway_result = GatewayAccessCheck(
            ok=True,
            kind="ok",
            status_code=200,
            message="ok",
            base_url=None,
            route=None,
            provider_id="anthropic",
            model="claude-sonnet-4-5",
        )
        self.run_draft_called = False
        self.run_refinement_called = False

    def is_running(self) -> bool:
        return False

    def verify_gateway_access(
        self,
        *,
        provider_id: str,
        model: str | None,
        timeout_seconds: float = 5.0,
        force: bool = False,
    ) -> GatewayAccessCheck:
        _ = provider_id, model, timeout_seconds, force
        return self.gateway_result

    def run_draft(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        self.run_draft_called = True
        return True

    def run_refinement(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        self.run_refinement_called = True
        return True


class _ManagerStub:
    def __init__(self, project_dir: Path, values: dict[str, str]) -> None:
        self.project_dir = project_dir
        self._values = values
        self.metadata = None
        self.project_name = "Case"
        self.saved_preferences: dict[str, object] | None = None

    def placeholder_mapping(self) -> dict[str, str]:
        return dict(self._values)

    def project_placeholder_values(self) -> dict[str, str]:
        return dict(self._values)

    def update_report_preferences(self, **kwargs) -> None:  # noqa: ANN003
        self.saved_preferences = dict(kwargs)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_controller(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ReportsController:
    workspace = QWidget()
    tab = ReportsTab(parent=workspace)
    return ReportsController(workspace, tab, service=_ServiceStub())


def test_validate_placeholders_can_block_run_on_missing_required(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    controller._tab.generation_user_prompt_edit.setText("Prompt needs {client_name}")

    monkeypatch.setattr(controller, "_read_prompt_file", lambda path: path)
    monkeypatch.setattr(
        reports_module,
        "get_prompt_spec",
        lambda _key: SimpleNamespace(required=("client_name",), optional=()),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    allowed = controller._validate_placeholders_before_run(
        include_generation=True,
        include_refinement=False,
    )

    assert allowed is False


def test_validate_placeholders_can_continue_on_user_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    controller._tab.generation_user_prompt_edit.setText("Prompt needs {client_name}")

    monkeypatch.setattr(controller, "_read_prompt_file", lambda path: path)
    monkeypatch.setattr(
        reports_module,
        "get_prompt_spec",
        lambda _key: SimpleNamespace(required=("client_name",), optional=()),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    allowed = controller._validate_placeholders_before_run(
        include_generation=True,
        include_refinement=False,
    )

    assert allowed is True


def test_start_draft_job_blocks_when_gateway_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)
    service = controller._service
    assert isinstance(service, _ServiceStub)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    template_path = tmp_path / "template.md"
    template_path.write_text("template", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    service.gateway_result = GatewayAccessCheck(
        ok=False,
        kind="auth_invalid",
        status_code=401,
        message="Unauthorized - Key not found",
        base_url="https://gateway.example.com",
        route="bulk",
        provider_id="anthropic",
        model="claude-sonnet-4-5",
    )

    monkeypatch.setattr(controller, "_validate_placeholders_before_run", lambda **_kwargs: True)
    monkeypatch.setattr(
        controller,
        "_resolve_llm_settings",
        lambda: SimpleNamespace(provider_id="anthropic", model_id="claude-sonnet-4-5", custom_model_id=None),
    )
    monkeypatch.setattr(controller, "_validate_required_path", lambda *_args, **_kwargs: template_path)
    monkeypatch.setattr(controller, "_validate_prompt_path", lambda *_args, **_kwargs: prompt_path)
    monkeypatch.setattr(controller, "_optional_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller, "_resolve_selected_inputs", lambda: [("converted", "doc.md")])
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    controller._start_draft_job()

    assert service.run_draft_called is False
    assert warnings
    assert "Pydantic AI Gateway app key was rejected" in warnings[0]


def test_start_draft_job_can_continue_when_gateway_probe_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)
    service = controller._service
    assert isinstance(service, _ServiceStub)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    template_path = tmp_path / "template.md"
    template_path.write_text("template", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    service.gateway_result = GatewayAccessCheck(
        ok=False,
        kind="rate_limited",
        status_code=429,
        message="Provider capacity reached for anthropic. Retry soon.",
        base_url="https://gateway.example.com",
        route="bulk",
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        retry_after_seconds=7.0,
    )

    monkeypatch.setattr(controller, "_validate_placeholders_before_run", lambda **_kwargs: True)
    monkeypatch.setattr(
        controller,
        "_resolve_llm_settings",
        lambda: SimpleNamespace(
            provider_id="anthropic",
            model_id="claude-sonnet-4-5",
            custom_model_id=None,
            context_window=200000,
            use_reasoning=False,
            reasoning=SimpleNamespace(to_dict=lambda: {}),
        ),
    )
    monkeypatch.setattr(controller, "_validate_required_path", lambda *_args, **_kwargs: template_path)
    monkeypatch.setattr(controller, "_validate_prompt_path", lambda *_args, **_kwargs: prompt_path)
    monkeypatch.setattr(controller, "_optional_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller, "_resolve_selected_inputs", lambda: [("converted", "doc.md")])
    monkeypatch.setattr(controller, "_confirm_report_forecast", lambda **_kwargs: True)
    replies = iter((QMessageBox.Yes, QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(replies))

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    controller._start_draft_job()

    assert service.run_draft_called is True
    assert warnings == []


def test_start_draft_job_stays_blocked_when_user_cancels_rate_limited_gateway_probe(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)
    service = controller._service
    assert isinstance(service, _ServiceStub)

    controller._project_manager = _ManagerStub(tmp_path, values={})
    template_path = tmp_path / "template.md"
    template_path.write_text("template", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    service.gateway_result = GatewayAccessCheck(
        ok=False,
        kind="rate_limited",
        status_code=429,
        message="Provider capacity reached for anthropic. Retry soon.",
        base_url="https://gateway.example.com",
        route="bulk",
        provider_id="anthropic",
        model="claude-sonnet-4-5",
        retry_after_seconds=7.0,
    )

    monkeypatch.setattr(controller, "_validate_placeholders_before_run", lambda **_kwargs: True)
    monkeypatch.setattr(
        controller,
        "_resolve_llm_settings",
        lambda: SimpleNamespace(provider_id="anthropic", model_id="claude-sonnet-4-5", custom_model_id=None),
    )
    monkeypatch.setattr(controller, "_validate_required_path", lambda *_args, **_kwargs: template_path)
    monkeypatch.setattr(controller, "_validate_prompt_path", lambda *_args, **_kwargs: prompt_path)
    monkeypatch.setattr(controller, "_optional_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller, "_resolve_selected_inputs", lambda: [("converted", "doc.md")])
    monkeypatch.setattr(controller, "_confirm_report_forecast", lambda **_kwargs: True)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message) or QMessageBox.Ok,
    )

    controller._start_draft_job()

    assert service.run_draft_called is False
    assert warnings == []


def test_report_progress_detail_updates_progress_label(
    monkeypatch: pytest.MonkeyPatch,
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    assert qt_app is not None
    controller = _build_controller(monkeypatch, tmp_path)

    controller._on_report_progress_detail(
        WorkerProgressDetail(
            run_kind="report_draft",
            phase="section_started",
            label="Generating section 2 of 4",
            percent=35,
            section_index=2,
            section_total=4,
            section_title="Background",
        )
    )

    assert controller._tab.progress_bar.value() == 35
    assert "Section 2/4: Background" in controller._tab.progress_detail_label.text()
