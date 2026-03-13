"""Report-history helpers shared by ReportsController methods."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


@dataclass(slots=True)
class HistorySelection:
    draft_path: Optional[Path]
    refined_path: Optional[Path]
    reasoning_path: Optional[Path]
    manifest_path: Optional[Path]
    inputs_path: Optional[Path]


def _maybe_path(raw: object) -> Optional[Path]:
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser()
    except Exception:
        return None


def persist_report_history(manager, run_type: str, result: Dict[str, object]) -> None:
    """Persist a report run into project-manager history."""
    timestamp_raw = str(result.get("timestamp"))
    try:
        timestamp = datetime.fromisoformat(timestamp_raw)
    except Exception:
        timestamp = datetime.now(timezone.utc)

    draft_path = _maybe_path(result.get("draft_path"))
    refined_path = _maybe_path(result.get("refined_path"))
    reasoning_path = _maybe_path(result.get("reasoning_path"))
    manifest_path = _maybe_path(result.get("manifest_path"))
    inputs_path = _maybe_path(result.get("inputs_path"))

    provider = str(result.get("provider", "anthropic"))
    model = str(result.get("model", ""))
    custom_model = result.get("custom_model")
    context_window = result.get("context_window")
    use_reasoning = bool(result.get("use_reasoning", False))
    reasoning = result.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    try:
        context_window_int = int(context_window) if context_window is not None else None
    except (ValueError, TypeError):
        context_window_int = None
    inputs = list(result.get("inputs", []))

    template_value = str(result.get("template_path") or "").strip() or None
    transcript_value = str(result.get("transcript_path") or "").strip() or None

    generation_user_prompt = str(result.get("generation_user_prompt") or "").strip() or None
    generation_system_prompt = str(result.get("generation_system_prompt") or "").strip() or None
    refinement_user_prompt = str(result.get("refinement_user_prompt") or "").strip() or None
    refinement_system_prompt = str(result.get("refinement_system_prompt") or "").strip() or None
    estimate_payload = result.get("cost_estimate")
    estimate_payload = estimate_payload if isinstance(estimate_payload, dict) else {}

    if run_type == "refinement" and refined_path is not None:
        manager.record_report_refinement_run(
            timestamp=timestamp,
            draft_path=(draft_path or refined_path),
            refined_path=refined_path,
            reasoning_path=reasoning_path,
            manifest_path=manifest_path,
            inputs_path=inputs_path,
            provider=provider,
            model=model,
            custom_model=str(custom_model) if custom_model else None,
            context_window=context_window_int,
            inputs=inputs,
            use_reasoning=use_reasoning,
            reasoning=reasoning,
            template_path=template_value,
            transcript_path=transcript_value,
            refinement_user_prompt=refinement_user_prompt,
            refinement_system_prompt=refinement_system_prompt,
            refined_tokens=result.get("refinement_tokens"),
            estimated_best_cost=estimate_payload.get("best_estimate"),
            estimated_ceiling_cost=estimate_payload.get("ceiling"),
        )
        return

    if draft_path is None and result.get("draft_path"):
        draft_path = Path(str(result["draft_path"])).expanduser()
    if draft_path is None:
        return
    manager.record_report_draft_run(
        timestamp=timestamp,
        draft_path=draft_path,
        manifest_path=manifest_path,
        inputs_path=inputs_path,
        provider=provider,
        model=model,
        custom_model=str(custom_model) if custom_model else None,
        context_window=context_window_int,
        inputs=inputs,
        use_reasoning=use_reasoning,
        reasoning=reasoning,
        template_path=template_value,
        transcript_path=transcript_value,
        generation_user_prompt=generation_user_prompt,
        generation_system_prompt=generation_system_prompt,
        draft_tokens=result.get("draft_tokens"),
        estimated_best_cost=estimate_payload.get("best_estimate"),
        estimated_ceiling_cost=estimate_payload.get("ceiling"),
    )


def current_history_selection(manager, index_value: object) -> Optional[HistorySelection]:
    """Resolve currently-selected report history record into file paths."""
    if index_value is None:
        return None
    try:
        entry = manager.report_state.history[int(index_value)]
    except (ValueError, TypeError, IndexError):
        return None
    return HistorySelection(
        draft_path=Path(entry.draft_path) if entry.draft_path else None,
        refined_path=Path(entry.refined_path) if entry.refined_path else None,
        reasoning_path=Path(entry.reasoning_path) if entry.reasoning_path else None,
        manifest_path=Path(entry.manifest_path) if entry.manifest_path else None,
        inputs_path=Path(entry.inputs_path) if entry.inputs_path else None,
    )
