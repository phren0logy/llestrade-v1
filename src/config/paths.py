"""
App paths and cross-platform user directories for Llestrade.

Moves user-visible files from the legacy hidden folder (~/.forensic_report_drafter)
to a visible Documents folder: ~/Documents/llestrade (and platform equivalents).
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Optional


APP_FOLDER_NAME = "llestrade"
LEGACY_HIDDEN_ROOT = Path.home() / ".forensic_report_drafter"  # legacy reference (for scripts only)


def _xdg_documents_dir() -> Optional[Path]:
    """Best-effort attempt to read Linux XDG documents directory."""
    try:
        config = Path.home() / ".config" / "user-dirs.dirs"
        if not config.exists():
            return None
        text = config.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("XDG_DOCUMENTS_DIR"):
                parts = line.split("=", 1)
                if len(parts) != 2:
                    continue
                value = parts[1].strip().strip('"')
                # Replace $HOME token
                if value.startswith("$HOME/"):
                    value = str(Path.home() / value.split("/", 1)[1])
                p = Path(value).expanduser()
                return p
    except Exception:
        return None
    return None


def documents_dir() -> Path:
    """Return a user-visible Documents directory across platforms."""
    home = Path.home()
    if sys.platform.startswith("win"):
        candidates = [home / "Documents", home / "My Documents"]
    elif sys.platform == "darwin":
        candidates = [home / "Documents"]
    else:
        xdg = _xdg_documents_dir()
        candidates = [xdg] if xdg else [home / "Documents"]

    for c in candidates:
        if c and c.exists():
            return c
    # Fallback to home if Documents isn't present
    return home


def app_user_root() -> Path:
    env_override = os.getenv("LLESTRADE_USER_ROOT")
    if env_override:
        root = Path(env_override).expanduser()
    else:
        root = documents_dir() / APP_FOLDER_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def app_config_dir() -> Path:
    p = app_user_root() / "config"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_prompts_root() -> Path:
    p = app_user_root() / "prompts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_templates_root() -> Path:
    p = app_user_root() / "templates"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_placeholder_sets_root() -> Path:
    p = app_user_root() / "placeholder_sets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_logs_dir() -> Path:
    p = app_user_root() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_crashes_dir() -> Path:
    p = app_user_root() / "crashes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_base_dir() -> Path:
    """Return the root directory containing application resources.

    When the app is frozen (PyInstaller), sys._MEIPASS points to the unpacked
    bundle. During development we fall back to the repository root.
    """

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[2]


def app_resource_root() -> Path:
    """Return the folder that holds bundled static resources."""

    base = app_base_dir()
    candidates = [
        base / "src" / "app" / "resources",
        base / "app" / "resources",
        base / "resources",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


__all__ = [
    "documents_dir",
    "app_user_root",
    "app_config_dir",
    "app_prompts_root",
    "app_templates_root",
    "app_placeholder_sets_root",
    "app_logs_dir",
    "app_crashes_dir",
    "app_base_dir",
    "app_resource_root",
]
