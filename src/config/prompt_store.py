"""
Prompt storage and synchronization utilities.

Design:
- Keep all prompts in a user-writable config folder outside the app bundle.
- Maintain two categories:
  - bundled/: copies of prompts shipped with the app (managed by the app)
  - custom/: user-authored prompts (never overwritten)
- On install/update, sync bundled prompts from the app resources into bundled/.
  Custom prompts are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .paths import (
    app_prompts_root,
    app_resource_root,
    app_templates_root,
)


def get_prompts_root() -> Path:
    return app_prompts_root()


def get_templates_root() -> Path:
    return app_templates_root()


def get_bundled_dir() -> Path:
    path = get_prompts_root() / "bundled"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_custom_dir() -> Path:
    path = get_prompts_root() / "custom"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_template_bundled_dir() -> Path:
    path = get_templates_root() / "bundled"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_template_custom_dir() -> Path:
    path = get_templates_root() / "custom"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_repo_prompts_dir() -> Path:
    """Return the path to prompts bundled with the application source tree.

    Normal dev layout: src/app/resources/prompts
    Provide a few fallbacks to be resilient to where this file is located.
    """
    resource_root = app_resource_root()
    path = resource_root / "prompts"
    if path.exists():
        return path
    return path


def get_repo_templates_dir() -> Path:
    """Return the path to templates bundled with the application source tree."""

    resource_root = app_resource_root()
    path = resource_root / "templates"
    if path.exists():
        return path
    return path


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_md_files(folder: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if not folder.exists():
        return files
    for p in sorted(folder.glob("*.md")):
        if p.is_file():
            files[p.name] = p
    return files


def _manifest_path(bundled_dir: Path) -> Path:
    return bundled_dir / ".manifest.json"


def _load_manifest(bundled_dir: Path) -> dict:
    path = _manifest_path(bundled_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(bundled_dir: Path, payload: dict) -> None:
    path = _manifest_path(bundled_dir)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest() -> dict:
    return _load_manifest(get_bundled_dir())


def save_manifest(payload: dict) -> None:
    _save_manifest(get_bundled_dir(), payload)


def load_template_manifest() -> dict:
    return _load_manifest(get_template_bundled_dir())


def save_template_manifest(payload: dict) -> None:
    _save_manifest(get_template_bundled_dir(), payload)


def compute_repo_digest(repo_dir: Path) -> dict:
    files = _collect_md_files(repo_dir)
    entries = {name: _hash_file(path) for name, path in files.items()}
    combined = hashlib.sha256()
    for name in sorted(entries):
        combined.update(name.encode("utf-8"))
        combined.update(entries[name].encode("utf-8"))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "digest": combined.hexdigest(),
    }


def _sync_resource(repo_dir: Path, bundled_dir: Path, *, force: bool) -> dict:
    repo_files = _collect_md_files(repo_dir)

    # ensure directories exist even if empty
    bundled_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(bundled_dir)
    repo_digest = compute_repo_digest(repo_dir)

    copied: List[str] = []
    updated: List[str] = []
    skipped: List[str] = []
    same: List[str] = []
    deleted: List[str] = []

    for name, src in repo_files.items():
        dst = bundled_dir / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied.append(name)
            continue
        src_hash = _hash_file(src)
        dst_hash = _hash_file(dst)
        if src_hash == dst_hash:
            same.append(name)
        else:
            if force:
                dst.write_bytes(src.read_bytes())
                updated.append(name)
            else:
                skipped.append(name)

    # Remove stale bundled files that no longer exist in the repo.
    existing_files = _collect_md_files(bundled_dir)
    for name, path in existing_files.items():
        if name not in repo_files:
            try:
                path.unlink()
                deleted.append(name)
            except OSError:
                # Ignore delete failures; leave file intact.
                continue

    _save_manifest(
        bundled_dir,
        {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "repo_digest": repo_digest,
            "copied": copied,
            "updated": updated,
            "skipped": skipped,
            "same": same,
            "deleted": deleted,
            "previous_manifest": manifest,
        },
    )

    return {
        "copied": copied,
        "updated": updated,
        "skipped": skipped,
        "same": same,
        "deleted": deleted,
    }


def sync_bundled_prompts(*, force: bool = False) -> dict:
    """Sync bundled prompts from the app resources into the user store."""

    return _sync_resource(get_repo_prompts_dir(), get_bundled_dir(), force=force)


def sync_bundled_templates(*, force: bool = False) -> dict:
    """Sync bundled templates from the app resources into the user store."""

    return _sync_resource(get_repo_templates_dir(), get_template_bundled_dir(), force=force)


__all__ = [
    "get_prompts_root",
    "get_templates_root",
    "get_bundled_dir",
    "get_custom_dir",
    "get_template_bundled_dir",
    "get_template_custom_dir",
    "get_repo_prompts_dir",
    "get_repo_templates_dir",
    "sync_bundled_prompts",
    "sync_bundled_templates",
    "load_manifest",
    "save_manifest",
    "load_template_manifest",
    "save_template_manifest",
]
