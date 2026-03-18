"""Tests for conversion helper registry and integration points."""

from __future__ import annotations

from src.app.core.conversion_helpers import registry, find_helper


def test_registry_lists_helpers() -> None:
    helpers = registry().list_helpers()
    helper_ids = {helper.helper_id for helper in helpers}
    assert helper_ids == {"docling"}


def test_find_helper_returns_correct_entry() -> None:
    helper = find_helper("docling")
    assert helper.helper_id == "docling"
    assert "Docling" in helper.name


def test_helper_options_exposed() -> None:
    helper = find_helper("docling")
    assert list(helper.options) == []
