#!/usr/bin/env python3
"""
One-time migration script: move legacy Forensic Report Drafter data to Llestrade paths
and migrate keyring namespace from "ForensicReportDrafter" to "Llestrade".

Moves (if present):
- ~/.forensic_report_drafter/prompts -> ~/Documents/llestrade/prompts
- ~/.forensic_report_drafter/config  -> ~/Documents/llestrade/config
- ~/.forensic_report_drafter/logs    -> ~/Documents/llestrade/logs
- ~/.forensic_report_drafter/crashes -> ~/Documents/llestrade/crashes

Keyring:
- Copies api keys from service "ForensicReportDrafter" to "Llestrade" for providers:
  anthropic, gemini, azure_openai, azure_di
  (use --remove-legacy-keys to delete legacy entries after copy)

Usage:
  uv run scripts/migrate_llestrade_from_legacy.py --yes
  uv run scripts/migrate_llestrade_from_legacy.py --dry-run
  uv run scripts/migrate_llestrade_from_legacy.py --remove-legacy-keys --yes
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


LEGACY_ROOT = Path.home() / ".forensic_report_drafter"


def _documents_dir() -> Path:
    # Keep simple: prefer ~/Documents if it exists; else fallback to home
    docs = Path.home() / "Documents"
    return docs if docs.exists() else Path.home()


def _llestrade_root() -> Path:
    root = _documents_dir() / "llestrade"
    root.mkdir(parents=True, exist_ok=True)
    return root


def migrate_folders(*, dry_run: bool = False) -> list[str]:
    actions: list[str] = []
    if not LEGACY_ROOT.exists():
        actions.append("Legacy folder not found; nothing to migrate.")
        return actions
    target = _llestrade_root()
    for name in ("prompts", "config", "logs", "crashes"):
        src = LEGACY_ROOT / name
        dst = target / name
        if src.exists() and not dst.exists():
            actions.append(f"Move {src} -> {dst}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        elif src.exists() and dst.exists():
            actions.append(f"Skip {src} (destination exists)")
    return actions


def migrate_keyring(*, dry_run: bool = False, remove_legacy: bool = False) -> list[str]:
    actions: list[str] = []
    try:
        import keyring  # type: ignore
    except Exception:
        return [
            "keyring not available; skipping API key migration",
            "Install 'keyring' if you want to migrate keys via this script.",
        ]

    providers = ["anthropic", "gemini", "azure_openai", "azure_di"]
    for provider in providers:
        legacy_key = keyring.get_password("ForensicReportDrafter", f"api_key_{provider}")
        if not legacy_key:
            actions.append(f"No legacy key for {provider}")
            continue
        existing = keyring.get_password("Llestrade", f"api_key_{provider}")
        if existing:
            actions.append(f"Llestrade key already set for {provider}; skipping copy")
        else:
            actions.append(f"Copy keyring entry {provider}: ForensicReportDrafter -> Llestrade")
            if not dry_run:
                keyring.set_password("Llestrade", f"api_key_{provider}", legacy_key)
        if remove_legacy and not dry_run:
            try:
                keyring.delete_password("ForensicReportDrafter", f"api_key_{provider}")
                actions.append(f"Removed legacy key for {provider}")
            except Exception:
                actions.append(f"Failed to remove legacy key for {provider} (ignored)")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy data to Llestrade paths and keyring")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without making changes")
    parser.add_argument("--remove-legacy-keys", action="store_true", help="Delete old keyring entries after copying")
    parser.add_argument("--yes", action="store_true", help="Run without interactive confirmation")
    args = parser.parse_args()

    print("Legacy root:", LEGACY_ROOT)
    print("Target root:", _llestrade_root())

    folder_actions = migrate_folders(dry_run=args.dry_run)
    keyring_actions = migrate_keyring(dry_run=args.dry_run, remove_legacy=args.remove_legacy_keys)

    print("\nPlanned actions:")
    for a in folder_actions + keyring_actions:
        print("-", a)

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to apply.")
        return 0

    if not args.yes:
        ans = input("\nApply these changes? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return 1

    print("\nMigration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

