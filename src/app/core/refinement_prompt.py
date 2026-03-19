"""Helpers for validation of report user prompt files."""

from __future__ import annotations

from pathlib import Path

from .prompt_placeholders import ensure_required_placeholders


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_refinement_prompt(path: Path) -> str:
    """Return the raw refinement user prompt content."""

    return _read_prompt(path)


def validate_refinement_prompt(content: str) -> None:
    """Ensure required refinement placeholders exist."""

    ensure_required_placeholders("report_refine_user", content)


def read_generation_prompt(path: Path) -> str:
    """Return the raw generation user prompt content."""

    return _read_prompt(path)


def validate_generation_prompt(content: str) -> None:
    """Ensure required generation placeholders exist."""

    ensure_required_placeholders("report_draft_user", content)


__all__ = [
    "read_refinement_prompt",
    "validate_refinement_prompt",
    "read_generation_prompt",
    "validate_generation_prompt",
]
