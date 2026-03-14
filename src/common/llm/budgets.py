"""Shared token-budget helpers for model invocation."""

from __future__ import annotations

from typing import Optional


DEFAULT_INPUT_BUDGET_RATIO = 0.66
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

    The budget first reserves explicit space for output tokens and an
    additional buffer, then applies a global safety ratio to the remaining
    prompt space. `None` indicates that no meaningful budget could be derived.
    """

    if raw_context_window is None or raw_context_window <= 0:
        return None

    available_budget = raw_context_window - max_output_tokens - buffer_tokens
    if available_budget <= 0:
        return None

    budget = int(available_budget * ratio)
    if budget <= 0:
        return None

    return max(int(budget), minimum_budget)


__all__ = [
    "DEFAULT_INPUT_BUDGET_BUFFER",
    "DEFAULT_INPUT_BUDGET_RATIO",
    "DEFAULT_MIN_INPUT_BUDGET",
    "compute_input_token_budget",
]
