"""Helpers for Azure DI raw artifact naming and filtering."""

from __future__ import annotations

from pathlib import Path

AZURE_RAW_MARKDOWN_SUFFIX = ".azure.raw.md"
AZURE_RAW_JSON_SUFFIX = ".azure.raw.json"


def is_azure_raw_artifact(path: Path | str) -> bool:
    """Return True when the path points at an Azure raw sidecar artifact."""
    text = path.as_posix() if isinstance(path, Path) else str(path)
    lowered = text.lower()
    return lowered.endswith(AZURE_RAW_MARKDOWN_SUFFIX) or lowered.endswith(AZURE_RAW_JSON_SUFFIX)

