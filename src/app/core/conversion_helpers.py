"""Registry for document conversion helpers used by the dashboard pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ConversionHelper:
    """Represents a conversion helper implementation."""

    helper_id: str
    name: str
    description: str
    supported_extensions: Iterable[str]
    options: Iterable[Any] = field(default_factory=list)
    executor: Optional[Callable[..., None]] = None  # injected later by workers


class HelperRegistry:
    """In-memory registry of available conversion helpers."""

    def __init__(self) -> None:
        self._helpers: Dict[str, ConversionHelper] = {}
        self._default_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, helper: ConversionHelper) -> None:
        self._helpers[helper.helper_id] = helper
        if self._default_id is None:
            self._default_id = helper.helper_id

    def get(self, helper_id: str) -> Optional[ConversionHelper]:
        return self._helpers.get(helper_id)

    def list_helpers(self) -> List[ConversionHelper]:
        return list(self._helpers.values())

    def default_helper(self) -> ConversionHelper:
        if self._default_id is None:
            raise KeyError("No conversion helpers registered")
        helper = self.get(self._default_id)
        if helper is None:
            raise KeyError("Default conversion helper is not registered")
        return helper


# ----------------------------------------------------------------------
# Default registry instance with built-in helpers
# ----------------------------------------------------------------------

_registry = HelperRegistry()

_registry.register(
    ConversionHelper(
        helper_id="docling",
        name="Local Docling MLX",
        description=(
            "Converts project PDFs through a local Docling MLX runtime and stores"
            " DocTags-only artifacts under converted_documents/."
        ),
        supported_extensions=[".pdf"],
    )
)


def registry() -> HelperRegistry:
    """Return the global helper registry instance."""
    return _registry


def find_helper(helper_id: str) -> ConversionHelper:
    helper = _registry.get(helper_id)
    if helper is None:
        raise KeyError(f"Unknown conversion helper: {helper_id}")
    return helper


__all__ = [
    "ConversionHelper",
    "HelperRegistry",
    "registry",
    "find_helper",
]
