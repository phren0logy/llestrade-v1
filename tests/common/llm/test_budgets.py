from __future__ import annotations

from src.app.core import llm_catalog
from src.common.llm.budgets import compute_input_token_budget
from src.common.llm.request_budget import (
    compute_preflight_input_budget,
    compute_request_input_budget,
    count_request_input_tokens,
    evaluate_request_budget,
    resolve_request_raw_context_window,
)


def test_compute_input_token_budget_applies_ratio_after_reserving_output_and_buffer() -> None:
    budget = compute_input_token_budget(
        raw_context_window=400_000,
        max_output_tokens=32_000,
    )

    assert budget == 242_220


def test_compute_input_token_budget_returns_none_when_reserved_space_exhausts_window() -> None:
    budget = compute_input_token_budget(
        raw_context_window=30_000,
        max_output_tokens=32_000,
    )

    assert budget is None


def test_count_request_input_tokens_prefers_exact_counter() -> None:
    input_tokens, count_mode = count_request_input_tokens(
        system_prompt="System",
        user_prompt="Prompt",
        provider_id="openai",
        model_id="gpt-5-mini",
        exact_token_counter=lambda: 1234,
    )

    assert input_tokens == 1234
    assert count_mode == "exact"


def test_compute_request_input_budget_applies_explicit_limit_override() -> None:
    raw_context_window, input_budget = compute_request_input_budget(
        provider_id="openai",
        model_id="gpt-5-mini",
        raw_context_window=400_000,
        max_output_tokens=32_000,
        runtime_input_budget_limit=200_000,
    )

    assert raw_context_window == 400_000
    assert input_budget == 200_000


def test_compute_preflight_input_budget_applies_openai_safety_margin() -> None:
    assert compute_preflight_input_budget(
        provider_id="openai",
        model_id="gpt-5-mini",
        runtime_input_budget=242_220,
    ) == 193_776


def test_evaluate_request_budget_reports_fit_against_shared_budget() -> None:
    evaluation = evaluate_request_budget(
        provider_id="openai",
        model_id="gpt-5-mini",
        system_prompt="System",
        user_prompt="Prompt",
        raw_context_window=100_000,
        max_output_tokens=20_000,
        exact_token_counter=lambda: 52_140,
    )

    assert evaluation.runtime_input_budget == 52_140
    assert evaluation.preflight_input_budget == 41_712
    assert evaluation.input_budget == 41_712
    assert evaluation.input_tokens == 52_140
    assert evaluation.fits is False
    assert evaluation.count_mode == "exact"


def test_resolve_request_raw_context_window_uses_transport_aware_catalog_lookup(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_resolve_model_context_window(provider_id: str, model_id: str | None, *, transport: str = "direct"):
        captured["provider_id"] = provider_id
        captured["model_id"] = model_id or ""
        captured["transport"] = transport
        return 200_000

    monkeypatch.setattr(llm_catalog, "resolve_model_context_window", _fake_resolve_model_context_window)

    resolved = resolve_request_raw_context_window(
        provider_id="openai",
        model_id="o3",
        transport="gateway",
    )

    assert resolved == 200_000
    assert captured == {
        "provider_id": "openai",
        "model_id": "o3",
        "transport": "gateway",
    }
