from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.app.core.project_manager import ProjectManager, ProjectMetadata
from src.app.ui.workspace.controllers.reports_history import persist_report_history


def test_report_history_entry_round_trips_estimate_metadata(tmp_path: Path) -> None:
    manager = ProjectManager()
    manager.create_project(tmp_path, ProjectMetadata(case_name="Report History Demo"))

    draft_path = manager.project_dir / "reports" / "draft.md"
    draft_path.write_text("draft", encoding="utf-8")

    manager.record_report_draft_run(
        timestamp=datetime.now(timezone.utc),
        draft_path=draft_path,
        manifest_path=None,
        inputs_path=None,
        provider="openai",
        model="gpt-5",
        custom_model=None,
        context_window=200000,
        inputs=["converted/a.md"],
        template_path="templates/t.md",
        transcript_path=None,
        generation_user_prompt="prompts/generation.md",
        generation_system_prompt="prompts/system.md",
        draft_tokens=1234,
        estimated_best_cost=1.25,
        estimated_ceiling_cost=2.5,
    )
    manager.save_project()

    reloaded = ProjectManager()
    assert reloaded.load_project(manager.project_path) is True

    entry = reloaded.report_state.history[0]
    assert entry.estimated_best_cost == 1.25
    assert entry.estimated_ceiling_cost == 2.5
    assert entry.draft_tokens == 1234


def test_persist_report_history_forwards_cost_estimate_metadata() -> None:
    captured: dict[str, object] = {}

    class _ManagerStub:
        def record_report_draft_run(self, **kwargs) -> None:  # noqa: ANN003
            captured["draft"] = kwargs

        def record_report_refinement_run(self, **kwargs) -> None:  # noqa: ANN003
            captured["refinement"] = kwargs

    manager = _ManagerStub()

    persist_report_history(
        manager,
        "draft",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "draft_path": "/tmp/draft.md",
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "inputs": [],
            "cost_estimate": {
                "best_estimate": 3.0,
                "ceiling": 6.0,
            },
        },
    )

    draft_kwargs = captured["draft"]
    assert draft_kwargs["estimated_best_cost"] == 3.0
    assert draft_kwargs["estimated_ceiling_cost"] == 6.0
