"""Typed stage contracts and trace-attribute mapping for worker LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True, slots=True)
class BulkMapStageInput:
    group_id: str
    group_name: str
    group_slug: str
    provider_id: str
    model: Optional[str]
    context_label: str
    max_tokens: int
    temperature: float
    transport: str = "direct"
    reasoning: bool = False
    gateway_route: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BulkReduceStageInput:
    group_id: str
    group_name: str
    group_slug: str
    provider_id: str
    model: Optional[str]
    max_tokens: int
    temperature: float
    transport: str = "direct"
    reasoning: bool = False
    gateway_route: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ReportDraftStageInput:
    section_index: int
    section_total: int
    section_title: str
    provider_id: str
    model: str
    max_tokens: int
    temperature: float
    transport: str = "direct"
    reasoning: bool = False
    gateway_route: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ReportRefineStageInput:
    provider_id: str
    model: str
    max_tokens: int
    temperature: float
    transport: str = "direct"
    reasoning: bool = False
    gateway_route: Optional[str] = None


StageInput = Union[
    BulkMapStageInput,
    BulkReduceStageInput,
    ReportDraftStageInput,
    ReportRefineStageInput,
]


def stage_trace_attributes(stage: StageInput) -> Dict[str, object]:
    """Produce stable trace attributes from typed stage inputs."""
    attrs: Dict[str, object] = {
        "llestrade.transport": stage.transport,
        "llestrade.provider_id": stage.provider_id,
        "llestrade.model": stage.model,
        "llestrade.reasoning": stage.reasoning,
        "llestrade.max_tokens": stage.max_tokens,
        "llestrade.temperature": stage.temperature,
    }
    if stage.gateway_route:
        attrs["llestrade.gateway.route"] = stage.gateway_route

    if isinstance(stage, BulkMapStageInput):
        attrs.update(
            {
                "llestrade.worker": "bulk_analysis",
                "llestrade.stage": "bulk_map",
                "llestrade.group_id": stage.group_id,
                "llestrade.group_name": stage.group_name,
                "llestrade.group_slug": stage.group_slug,
                "llestrade.context_label": stage.context_label,
            }
        )
    elif isinstance(stage, BulkReduceStageInput):
        attrs.update(
            {
                "llestrade.worker": "bulk_reduce",
                "llestrade.stage": "bulk_reduce",
                "llestrade.group_id": stage.group_id,
                "llestrade.group_name": stage.group_name,
                "llestrade.group_slug": stage.group_slug,
            }
        )
    elif isinstance(stage, ReportDraftStageInput):
        attrs.update(
            {
                "llestrade.worker": "report_draft",
                "llestrade.stage": "report_draft",
                "llestrade.section_index": stage.section_index,
                "llestrade.section_total": stage.section_total,
                "llestrade.section_title": stage.section_title,
            }
        )
    else:
        attrs.update(
            {
                "llestrade.worker": "report_refine",
                "llestrade.stage": "report_refine",
            }
        )

    return attrs


__all__ = [
    "BulkMapStageInput",
    "BulkReduceStageInput",
    "ReportDraftStageInput",
    "ReportRefineStageInput",
    "stage_trace_attributes",
]
