"""Unit tests for typed stage contracts and trace attributes."""

from __future__ import annotations

from src.app.workers.stage_contracts import (
    BulkMapStageInput,
    BulkReduceStageInput,
    ReportDraftStageInput,
    ReportRefineStageInput,
    stage_trace_attributes,
)


def test_bulk_map_stage_trace_attributes_include_group_context() -> None:
    attrs = stage_trace_attributes(
        BulkMapStageInput(
            group_id="group-1",
            group_name="Clinical",
            group_slug="clinical",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
            context_label="doc-a.md",
            max_tokens=32000,
            temperature=0.1,
        )
    )

    assert attrs["llestrade.worker"] == "bulk_analysis"
    assert attrs["llestrade.stage"] == "bulk_map"
    assert attrs["llestrade.group_id"] == "group-1"
    assert attrs["llestrade.group_name"] == "Clinical"
    assert attrs["llestrade.group_slug"] == "clinical"
    assert attrs["llestrade.context_label"] == "doc-a.md"


def test_bulk_reduce_stage_trace_attributes_include_reduce_stage() -> None:
    attrs = stage_trace_attributes(
        BulkReduceStageInput(
            group_id="group-1",
            group_name="Clinical",
            group_slug="clinical",
            provider_id="anthropic",
            model=None,
            max_tokens=16000,
            temperature=0.2,
        )
    )

    assert attrs["llestrade.worker"] == "bulk_reduce"
    assert attrs["llestrade.stage"] == "bulk_reduce"
    assert attrs["llestrade.model"] is None


def test_report_stages_produce_expected_attributes() -> None:
    draft_attrs = stage_trace_attributes(
        ReportDraftStageInput(
            section_index=2,
            section_total=5,
            section_title="Findings",
            provider_id="anthropic",
            model="claude-sonnet-4-5",
            max_tokens=60000,
            temperature=0.2,
        )
    )
    refine_attrs = stage_trace_attributes(
        ReportRefineStageInput(
            provider_id="anthropic",
            model="claude-sonnet-4-5",
            max_tokens=60000,
            temperature=0.2,
        )
    )

    assert draft_attrs["llestrade.worker"] == "report_draft"
    assert draft_attrs["llestrade.stage"] == "report_draft"
    assert draft_attrs["llestrade.section_index"] == 2
    assert draft_attrs["llestrade.section_total"] == 5
    assert draft_attrs["llestrade.section_title"] == "Findings"

    assert refine_attrs["llestrade.worker"] == "report_refine"
    assert refine_attrs["llestrade.stage"] == "report_refine"
