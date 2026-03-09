"""View-rendering helpers for the Documents controller."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from src.app.ui.widgets import BannerAction


def update_source_root_label(label, root_path: Path | None) -> None:
    if not root_path or not root_path.exists():
        label.setText("Source root: not set")
    else:
        label.setText(f"Source root: {root_path}")


def update_last_scan_label(label, last_scan: datetime | str | None) -> None:
    if not last_scan:
        label.setText("Last scan: never")
        return

    if isinstance(last_scan, datetime):
        display = last_scan.strftime("Last scan: %Y-%m-%d %H:%M")
    else:
        display = f"Last scan: {last_scan}"
    label.setText(display)


def set_root_warning(label, warnings: Sequence[str]) -> None:
    if warnings:
        label.setText("\n".join(warnings))
        label.show()
    else:
        label.clear()
        label.hide()


def update_highlights_banner(
    banner,
    missing: Sequence[str],
    pending_count: int,
    on_review: Callable[[], None],
) -> None:
    if not missing:
        banner.reset()
        return

    total = pending_count or len(missing)
    plural = "s" if total != 1 else ""
    banner.set_role("warning")
    banner.set_message(
        f"Highlights pending for {total} PDF{plural}.",
        "Review the queue to see which documents still need highlights.",
    )
    banner.set_actions(
        [
            BannerAction(
                label="Review list",
                callback=on_review,
                is_default=True,
            )
        ]
    )
    banner.show()


def update_bulk_banner(
    banner,
    missing: Sequence[str],
    pending_count: int,
    on_open_bulk: Callable[[], None],
) -> None:
    if not missing:
        banner.reset()
        return

    total = pending_count or len(missing)
    plural = "s" if total != 1 else ""
    banner.set_role("warning")
    banner.set_message(
        f"Bulk analysis pending for {total} document{plural}.",
        "Open the Bulk Analysis tab to create or refresh group runs.",
    )
    banner.set_actions(
        [
            BannerAction(
                label="Open Bulk Analysis",
                callback=on_open_bulk,
                is_default=True,
            )
        ]
    )
    banner.show()
