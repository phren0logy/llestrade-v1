"""Helpers for assembling effective prompts from editable templates plus generated text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

GENERATED_CITATION_APPENDIX_TITLE = "Generated Citation Appendix"


def append_generated_prompt_section(
    base_text: str,
    generated_text: str,
    *,
    title: str = GENERATED_CITATION_APPENDIX_TITLE,
) -> str:
    """Append a labeled generated section to prompt text."""

    appendix = (generated_text or "").strip()
    base = (base_text or "").rstrip()
    if not appendix:
        return base

    divider = f"--- {title} ---"
    if not base:
        return f"{divider}\n{appendix}\n"
    return f"{base}\n\n{divider}\n{appendix}\n"


@dataclass(frozen=True)
class PromptAssembly:
    """Effective prompt payload plus the generated sections used to build it."""

    system_template: str
    user_template: str
    system_appendix: str = ""
    user_appendix: str = ""
    system_effective: str = ""
    user_effective: str = ""
    values: Mapping[str, str] = field(default_factory=dict)


__all__ = [
    "GENERATED_CITATION_APPENDIX_TITLE",
    "PromptAssembly",
    "append_generated_prompt_section",
]
