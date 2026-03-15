"""Shared request token counting and budget evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .budgets import DEFAULT_MIN_INPUT_BUDGET, compute_input_token_budget
from .tokens import TokenCounter

RequestCountMode = Literal["exact", "estimate"]
RequestCatalogTransport = Literal["direct", "gateway"]
DEFAULT_PREFLIGHT_BUDGET_RATIO = 1.0
OPENAI_PREFLIGHT_BUDGET_RATIO = 0.80
_OPENAI_PROVIDER_IDS = {"openai", "azure_openai"}


@dataclass(frozen=True, slots=True)
class RequestBudgetEvaluation:
    raw_context_window: int | None
    runtime_input_budget: int | None
    preflight_input_budget: int | None
    input_tokens: int
    fits: bool
    count_mode: RequestCountMode

    @property
    def input_budget(self) -> int | None:
        """Backward-compatible alias for the preflight planning budget."""
        return self.preflight_input_budget


def preflight_budget_ratio(*, provider_id: str, model_id: str | None = None) -> float:
    _ = model_id
    if provider_id in _OPENAI_PROVIDER_IDS:
        return OPENAI_PREFLIGHT_BUDGET_RATIO
    return DEFAULT_PREFLIGHT_BUDGET_RATIO


def compute_preflight_input_budget(
    *,
    provider_id: str,
    model_id: str | None,
    runtime_input_budget: int | None,
    minimum_budget: int = DEFAULT_MIN_INPUT_BUDGET,
) -> int | None:
    if runtime_input_budget is None or runtime_input_budget <= 0:
        return None

    ratio = preflight_budget_ratio(provider_id=provider_id, model_id=model_id)
    if ratio >= 1.0:
        return int(runtime_input_budget)

    reduced_budget = int(runtime_input_budget * ratio)
    if reduced_budget <= 0:
        return None
    return max(min(reduced_budget, int(runtime_input_budget)), minimum_budget)


def resolve_request_raw_context_window(
    *,
    provider_id: str,
    model_id: str | None,
    explicit_context_window: int | None = None,
    transport: RequestCatalogTransport = "direct",
) -> int | None:
    if isinstance(explicit_context_window, int) and explicit_context_window > 0:
        return int(explicit_context_window)

    try:
        from src.app.core.llm_catalog import resolve_model_context_window

        resolved = resolve_model_context_window(
            provider_id=provider_id,
            model_id=model_id,
            transport=transport,
        )
        return int(resolved) if isinstance(resolved, int) and resolved > 0 else None
    except Exception:
        return None


def compute_request_input_budget(
    *,
    provider_id: str,
    model_id: str | None,
    max_output_tokens: int,
    explicit_context_window: int | None = None,
    raw_context_window: int | None = None,
    minimum_budget: int = DEFAULT_MIN_INPUT_BUDGET,
    runtime_input_budget_limit: int | None = None,
    input_budget_limit: int | None = None,
    transport: RequestCatalogTransport = "direct",
) -> tuple[int | None, int | None]:
    resolved_runtime_limit = (
        int(runtime_input_budget_limit)
        if isinstance(runtime_input_budget_limit, int) and runtime_input_budget_limit > 0
        else int(input_budget_limit)
        if isinstance(input_budget_limit, int) and input_budget_limit > 0
        else None
    )
    resolved_raw_context_window = (
        int(raw_context_window)
        if isinstance(raw_context_window, int) and raw_context_window > 0
        else resolve_request_raw_context_window(
            provider_id=provider_id,
            model_id=model_id,
            explicit_context_window=explicit_context_window,
            transport=transport,
        )
    )
    input_budget = compute_input_token_budget(
        raw_context_window=resolved_raw_context_window,
        max_output_tokens=max_output_tokens,
        minimum_budget=minimum_budget,
    )
    if resolved_runtime_limit is not None:
        input_budget = (
            min(input_budget, resolved_runtime_limit)
            if isinstance(input_budget, int) and input_budget > 0
            else resolved_runtime_limit
        )
    return resolved_raw_context_window, input_budget


def estimate_request_input_tokens(
    *,
    system_prompt: str,
    user_prompt: str,
    provider_id: str,
    model_id: str | None,
) -> int:
    combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
    if not combined_prompt:
        return 0

    token_info = TokenCounter.count(
        text=combined_prompt,
        provider=provider_id,
        model=model_id or "",
    )
    conservative = max(len(combined_prompt) // 3, 1)
    if token_info.get("success"):
        counted = int(token_info.get("token_count") or 0)
        if provider_id in {"anthropic", "anthropic_bedrock"}:
            return max(counted, conservative)
        if counted > 0:
            return counted
    return conservative


def count_request_input_tokens(
    *,
    system_prompt: str,
    user_prompt: str,
    provider_id: str,
    model_id: str | None,
    exact_token_counter: Callable[[], int | None] | None = None,
) -> tuple[int, RequestCountMode]:
    if exact_token_counter is not None:
        try:
            counted = exact_token_counter()
        except Exception:
            counted = None
        if counted is not None and counted >= 0:
            return int(counted), "exact"

    return (
        estimate_request_input_tokens(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id,
            model_id=model_id,
        ),
        "estimate",
    )


def evaluate_request_budget(
    *,
    provider_id: str,
    model_id: str | None,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    explicit_context_window: int | None = None,
    raw_context_window: int | None = None,
    minimum_budget: int = DEFAULT_MIN_INPUT_BUDGET,
    runtime_input_budget_limit: int | None = None,
    input_budget_limit: int | None = None,
    transport: RequestCatalogTransport = "direct",
    exact_token_counter: Callable[[], int | None] | None = None,
) -> RequestBudgetEvaluation:
    resolved_raw_context_window, runtime_input_budget = compute_request_input_budget(
        provider_id=provider_id,
        model_id=model_id,
        max_output_tokens=max_output_tokens,
        explicit_context_window=explicit_context_window,
        raw_context_window=raw_context_window,
        minimum_budget=minimum_budget,
        runtime_input_budget_limit=runtime_input_budget_limit,
        input_budget_limit=input_budget_limit,
        transport=transport,
    )
    preflight_input_budget = compute_preflight_input_budget(
        provider_id=provider_id,
        model_id=model_id,
        runtime_input_budget=runtime_input_budget,
        minimum_budget=minimum_budget,
    )
    input_tokens, count_mode = count_request_input_tokens(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider_id=provider_id,
        model_id=model_id,
        exact_token_counter=exact_token_counter,
    )
    fits = preflight_input_budget is None or input_tokens <= preflight_input_budget
    return RequestBudgetEvaluation(
        raw_context_window=resolved_raw_context_window,
        runtime_input_budget=runtime_input_budget,
        preflight_input_budget=preflight_input_budget,
        input_tokens=input_tokens,
        fits=fits,
        count_mode=count_mode,
    )


__all__ = [
    "RequestBudgetEvaluation",
    "RequestCatalogTransport",
    "RequestCountMode",
    "compute_preflight_input_budget",
    "compute_request_input_budget",
    "count_request_input_tokens",
    "estimate_request_input_tokens",
    "evaluate_request_budget",
    "preflight_budget_ratio",
    "resolve_request_raw_context_window",
]
