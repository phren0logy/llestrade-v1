"""Tree-state helpers for the Documents controller."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem

from src.app.ui.workspace.qt_flags import (
    ITEM_IS_ENABLED,
    ITEM_IS_TRISTATE,
    ITEM_IS_USER_CHECKABLE,
)


def normalise_relative_path(path: str) -> str:
    if not path:
        return ""
    return Path(path.strip("/")).as_posix()


def apply_directory_flags(item: QTreeWidgetItem) -> None:
    flags = item.flags()
    flags |= ITEM_IS_ENABLED | ITEM_IS_USER_CHECKABLE
    flags &= ~ITEM_IS_TRISTATE
    item.setFlags(flags)


def should_skip_source_entry(entry: Path) -> bool:
    return any(
        part in {".azure-di", ".azure_di"} or part.startswith(".azure-di") or part.startswith(".azure_di")
        for part in entry.parts
    )


def is_path_tracked(relative: str, tracked: Set[str]) -> bool:
    candidate = normalise_relative_path(relative)
    if not candidate:
        return "" in tracked
    while True:
        if candidate in tracked:
            return True
        if "/" not in candidate:
            candidate = ""
        else:
            candidate = candidate.rsplit("/", 1)[0]
        if candidate == "":
            return "" in tracked


def cascade_source_check_state(item: QTreeWidgetItem, state: Qt.CheckState) -> None:
    for index in range(item.childCount()):
        child = item.child(index)
        if child.flags() & Qt.ItemIsUserCheckable:
            child.setCheckState(0, state)
        cascade_source_check_state(child, state)


def update_parent_source_state(item: Optional[QTreeWidgetItem]) -> None:
    while item is not None and item.flags() & Qt.ItemIsUserCheckable:
        checked = unchecked = 0
        for index in range(item.childCount()):
            child_state = item.child(index).checkState(0)
            if child_state == Qt.Checked:
                checked += 1
            elif child_state == Qt.Unchecked:
                unchecked += 1
            else:
                checked += 1
                unchecked += 1
        if checked and unchecked:
            item.setCheckState(0, Qt.PartiallyChecked)
        elif checked:
            item.setCheckState(0, Qt.Checked)
        else:
            item.setCheckState(0, Qt.Unchecked)
        item = item.parent()


def collect_selected_directories_from_item(
    item: QTreeWidgetItem,
    results: Set[str],
    normalise: Callable[[str], str] = normalise_relative_path,
) -> None:
    data = item.data(0, Qt.UserRole)
    if not data:
        return
    node_type, relative = data
    state = item.checkState(0)
    if node_type == "dir" and state == Qt.Checked:
        results.add(normalise(relative))
        return
    for index in range(item.childCount()):
        collect_selected_directories_from_item(item.child(index), results, normalise)
