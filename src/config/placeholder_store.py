"""
Placeholder set storage and synchronisation utilities.

This mirrors the prompt/template resource handling:
- Bundled placeholder sets live under the application resources and are synced into a
  user-writable ``placeholder_sets/bundled`` directory.
- Custom placeholder sets authored by users live under ``placeholder_sets/custom``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from .paths import (
    app_placeholder_sets_root,
    app_resource_root,
)

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_placeholder_root() -> Path:
    return app_placeholder_sets_root()


def get_placeholder_bundled_dir() -> Path:
    path = get_placeholder_root() / "bundled"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_placeholder_custom_dir() -> Path:
    path = get_placeholder_root() / "custom"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_repo_placeholder_dir() -> Path:
    resource_root = app_resource_root()
    path = resource_root / "placeholder_sets"
    if path.exists():
        return path
    return path


# ---------------------------------------------------------------------------
# Synchronisation helpers
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collect_md_files(folder: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if not folder.exists():
        return files
    for entry in sorted(folder.glob("*.md")):
        if entry.is_file():
            files[entry.name] = entry
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
        elif force:
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
                # Ignore delete failures; leave the file in place.
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


def sync_bundled_placeholder_sets(*, force: bool = False) -> dict:
    """Synchronise bundled placeholder sets into the user workspace."""

    repo_dir = get_repo_placeholder_dir()
    bundled_dir = get_placeholder_bundled_dir()
    return _sync_resource(repo_dir, bundled_dir, force=force)


__all__ = [
    "get_placeholder_root",
    "get_placeholder_bundled_dir",
    "get_placeholder_custom_dir",
    "get_repo_placeholder_dir",
    "sync_bundled_placeholder_sets",
    "compute_repo_digest",
]
