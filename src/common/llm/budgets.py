"""Shared token-budget helpers for model invocation."""

from __future__ import annotations

from typing import Optional


DEFAULT_INPUT_BUDGET_RATIO = 0.85
DEFAULT_INPUT_BUDGET_BUFFER = 1_000
DEFAULT_MIN_INPUT_BUDGET = 4_000


def compute_input_token_budget(
    *,
    raw_context_window: Optional[int],
    max_output_tokens: int,
    ratio: float = DEFAULT_INPUT_BUDGET_RATIO,
    buffer_tokens: int = DEFAULT_INPUT_BUDGET_BUFFER,
    minimum_budget: int = DEFAULT_MIN_INPUT_BUDGET,
) -> int | None:
    """Return a conservative input-token budget for a request.

    The budget leaves explicit space for output tokens and an additional buffer,
    while also capping the usable prompt size to a fixed percentage of the raw
    model window. `None` indicates that no meaningful budget could be derived.
    """

    if raw_context_window is None or raw_context_window <= 0:
        return None

    ratio_budget = int(raw_context_window * ratio)
    available_budget = raw_context_window - max_output_tokens - buffer_tokens
    budget = min(ratio_budget, available_budget)
    if budget <= 0:
        return None

    return max(int(budget), minimum_budget)


__all__ = [
    "DEFAULT_INPUT_BUDGET_BUFFER",
    "DEFAULT_INPUT_BUDGET_RATIO",
    "DEFAULT_MIN_INPUT_BUDGET",
    "compute_input_token_budget",
]
