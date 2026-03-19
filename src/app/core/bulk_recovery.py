"""Durable recovery helpers for bulk map and combined/reduce workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from src.app.core.bulk_analysis_groups import BulkAnalysisGroup
from src.config.paths import app_base_dir, app_resource_root
from src.config.prompt_store import get_bundled_dir, get_custom_dir

if TYPE_CHECKING:
    from src.app.core.project_manager import ProjectMetadata

RECOVERY_VERSION = 1
DEFAULT_BULK_SYSTEM_PROMPT_IDENTIFIER = "bulk_system"
DEFAULT_BULK_USER_PROMPT_IDENTIFIER = "bulk_per_document"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BulkPromptCompatibility:
    kind: str
    roles: tuple[str, ...]


def _normalize_prompt_path(value: str | None) -> str:
    return str(value or "").strip()


def _logical_prompt_name(configured_path: str, identifier: str) -> str:
    if configured_path:
        return Path(configured_path).name or configured_path
    return identifier


def _explicit_prompt_search_paths(project_dir: Path, prompt_path: str) -> list[Path]:
    candidate = Path(prompt_path).expanduser()
    if candidate.is_absolute():
        return [candidate]

    search_paths: list[Path] = []
    if project_dir:
        search_paths.append((project_dir / candidate).resolve())
    bundle_root = app_base_dir()
    resource_root = app_resource_root()
    app_root = resource_root.parent
    search_paths.extend(
        [
            (bundle_root / candidate).resolve(),
            (app_root / candidate).resolve(),
            (resource_root / candidate).resolve(),
        ]
    )
    try:
        custom_dir = get_custom_dir()
        search_paths.append((custom_dir / candidate).resolve())
        search_paths.append((custom_dir / candidate.name).resolve())
    except Exception:
        pass
    try:
        bundled_dir = get_bundled_dir()
        search_paths.append((bundled_dir / candidate).resolve())
        search_paths.append((bundled_dir / candidate.name).resolve())
    except Exception:
        pass
    return search_paths


def resolve_bulk_prompt_path(project_dir: Path, prompt_path: str | None) -> Path | None:
    normalized = _normalize_prompt_path(prompt_path)
    if not normalized:
        return None

    seen: set[Path] = set()
    for path in _explicit_prompt_search_paths(project_dir, normalized):
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def _default_prompt_path(identifier: str) -> Path | None:
    filename = f"{identifier}.md"
    for root in (get_custom_dir(), get_bundled_dir()):
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def build_bulk_prompt_state(
    project_dir: Path,
    group: BulkAnalysisGroup,
    *,
    system_template: str,
    user_template: str,
) -> dict[str, dict[str, Any]]:
    def _build_role_state(role: str, configured_path: str, identifier: str, content: str) -> dict[str, Any]:
        normalized = _normalize_prompt_path(configured_path)
        resolved = resolve_bulk_prompt_path(project_dir, normalized) if normalized else _default_prompt_path(identifier)
        return {
            "role": role,
            "configured_path": normalized,
            "identifier": None if normalized else identifier,
            "logical_name": _logical_prompt_name(normalized, identifier),
            "resolved_path": str(resolved) if resolved is not None else None,
            "exists": bool(resolved is not None or not normalized),
            "missing": bool(normalized and resolved is None),
            "content_hash": _sha256(content or ""),
        }

    return {
        "system": _build_role_state(
            "system",
            group.system_prompt_path,
            DEFAULT_BULK_SYSTEM_PROMPT_IDENTIFIER,
            system_template,
        ),
        "user": _build_role_state(
            "user",
            group.user_prompt_path,
            DEFAULT_BULK_USER_PROMPT_IDENTIFIER,
            user_template,
        ),
    }


def capture_bulk_prompt_state(
    project_dir: Path,
    group: BulkAnalysisGroup,
    metadata: ProjectMetadata | None,
) -> dict[str, dict[str, Any]]:
    from src.app.core.bulk_analysis_runner import load_prompts

    bundle = load_prompts(project_dir, group, metadata)
    return build_bulk_prompt_state(
        project_dir,
        group,
        system_template=bundle.system_template,
        user_template=bundle.user_template,
    )


def bulk_prompt_recovery_signature(
    prompt_state: Mapping[str, Mapping[str, Any]],
    *,
    provider_id: str,
    model: str | None,
    operation: str,
    use_reasoning: bool,
    model_context_window: int | None,
    placeholder_requirements: Mapping[str, bool] | None,
    metadata: ProjectMetadata | None,
    placeholder_values: Mapping[str, str] | None = None,
) -> str:
    metadata_summary: dict[str, str] = {}
    if metadata is not None:
        metadata_summary = {
            "case_name": metadata.case_name,
            "subject_name": metadata.subject_name,
            "date_of_birth": metadata.date_of_birth,
            "case_description": metadata.case_description,
        }

    payload: dict[str, Any] = {
        "prompt_identity": {
            role: {
                "logical_name": str((state or {}).get("logical_name") or ""),
                "identifier": (state or {}).get("identifier"),
            }
            for role, state in sorted(prompt_state.items())
        },
        "provider_id": provider_id,
        "model": model,
        "group_operation": operation,
        "use_reasoning": use_reasoning,
        "model_context_window": model_context_window,
        "metadata": metadata_summary,
        "placeholder_requirements": dict(placeholder_requirements or {}),
    }
    if placeholder_values:
        payload["placeholders"] = {k: placeholder_values.get(k, "") for k in sorted(placeholder_values)}
    return _sha256(json.dumps(payload, sort_keys=True))


def classify_bulk_prompt_compatibility(
    previous_state: Mapping[str, Mapping[str, Any]] | None,
    current_state: Mapping[str, Mapping[str, Any]],
    *,
    fallback_prompt_hash_mismatch: bool = False,
) -> BulkPromptCompatibility:
    missing_roles: list[str] = []
    replaced_roles: list[str] = []
    changed_roles: list[str] = []

    for role, current in current_state.items():
        if bool((current or {}).get("missing")):
            missing_roles.append(role)
            continue

        previous = dict((previous_state or {}).get(role) or {})
        if not previous:
            continue

        previous_name = str(previous.get("logical_name") or "")
        current_name = str((current or {}).get("logical_name") or "")
        if previous_name and current_name and previous_name != current_name:
            replaced_roles.append(role)
            continue

        previous_hash = str(previous.get("content_hash") or "")
        current_hash = str((current or {}).get("content_hash") or "")
        if previous_hash and current_hash and previous_hash != current_hash:
            changed_roles.append(role)

    if missing_roles:
        return BulkPromptCompatibility(kind="missing", roles=tuple(sorted(missing_roles)))
    if replaced_roles:
        return BulkPromptCompatibility(kind="replaced", roles=tuple(sorted(replaced_roles)))
    if changed_roles:
        return BulkPromptCompatibility(kind="same_identity_changed", roles=tuple(sorted(changed_roles)))
    if fallback_prompt_hash_mismatch:
        return BulkPromptCompatibility(kind="same_identity_changed", roles=("system", "user"))
    return BulkPromptCompatibility(kind="unchanged", roles=())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _safe_rel_path(relative_path: str) -> Path:
    return Path(Path(relative_path).as_posix())


def _default_usage_totals() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0}


def _default_map_manifest() -> dict[str, Any]:
    return {
        "version": RECOVERY_VERSION,
        "signature": None,
        "updated_at": None,
        "actual_usage": _default_usage_totals(),
        "actual_cost": 0.0,
        "documents": {},
    }


def _default_reduce_manifest() -> dict[str, Any]:
    return {
        "version": RECOVERY_VERSION,
        "signature": None,
        "updated_at": None,
        "actual_usage": _default_usage_totals(),
        "actual_cost": 0.0,
        "status": "idle",
        "chunks": {"count": 0, "items": {}},
        "batches": {},
        "finalized": False,
        "ran_at": None,
    }


class BulkRecoveryStore:
    """Persist durable latest-run recovery state under a hidden group folder."""

    def __init__(self, group_root: Path) -> None:
        self._group_root = group_root
        self._root = group_root / ".recovery"

    # ------------------------------------------------------------------
    # Stage roots
    # ------------------------------------------------------------------
    def map_root(self) -> Path:
        return self._root / "map" / "latest"

    def reduce_root(self) -> Path:
        return self._root / "reduce" / "latest"

    def map_manifest_path(self) -> Path:
        return self.map_root() / "manifest.json"

    def reduce_manifest_path(self) -> Path:
        return self.reduce_root() / "manifest.json"

    # ------------------------------------------------------------------
    # Map manifest/payload helpers
    # ------------------------------------------------------------------
    def load_map_manifest(self) -> dict[str, Any]:
        payload = _read_json(self.map_manifest_path())
        if not isinstance(payload, dict) or payload.get("version") != RECOVERY_VERSION:
            return _default_map_manifest()
        payload.setdefault("documents", {})
        payload.setdefault("actual_usage", _default_usage_totals())
        payload.setdefault("actual_cost", 0.0)
        return payload

    def save_map_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        payload["version"] = RECOVERY_VERSION
        payload["updated_at"] = _utcnow()
        _write_json(self.map_manifest_path(), payload)

    def clear_map(self) -> None:
        shutil.rmtree(self.map_root(), ignore_errors=True)

    def clear_map_document(self, document_rel: str) -> None:
        shutil.rmtree(self.map_root() / "documents" / _safe_rel_path(document_rel), ignore_errors=True)

    def reset_map_document(self, document_rel: str) -> None:
        manifest = self.load_map_manifest()
        documents = dict(manifest.get("documents") or {})
        documents.pop(document_rel, None)
        manifest["documents"] = documents
        self.clear_map_document(document_rel)
        self.save_map_manifest(manifest)

    def map_chunk_path(self, document_rel: str, index: int) -> Path:
        return self.map_root() / "documents" / _safe_rel_path(document_rel) / "chunks" / f"chunk_{index}.json"

    def map_batch_path(self, document_rel: str, level: int, batch_index: int) -> Path:
        return (
            self.map_root()
            / "documents"
            / _safe_rel_path(document_rel)
            / "batches"
            / f"level_{level}_batch_{batch_index}.json"
        )

    def quarantine_map_payload(
        self,
        *,
        document_rel: str,
        kind: str,
        identifier: str,
        payload: Mapping[str, Any] | None,
        reason: str,
    ) -> None:
        quarantine_path = (
            self.map_root()
            / "quarantine"
            / _safe_rel_path(document_rel)
            / f"{kind}_{identifier}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        record = {
            "kind": kind,
            "identifier": identifier,
            "reason": reason,
            "quarantined_at": _utcnow(),
            "payload": dict(payload or {}),
        }
        _write_json(quarantine_path, record)

    def mark_map_chunk_compromised(self, *, document_rel: str, index: int, reason: str) -> None:
        manifest = self.load_map_manifest()
        documents = dict(manifest.get("documents") or {})
        entry = dict(documents.get(document_rel) or {})
        chunks = dict(entry.get("chunks") or {})
        chunk_key = str(index)
        payload = self.load_payload(self.map_chunk_path(document_rel, index))
        if payload is not None:
            self.quarantine_map_payload(
                document_rel=document_rel,
                kind="chunk",
                identifier=chunk_key,
                payload=payload,
                reason=reason,
            )
        try:
            self.map_chunk_path(document_rel, index).unlink()
        except FileNotFoundError:
            pass
        chunks[chunk_key] = {
            **dict(chunks.get(chunk_key) or {}),
            "status": "compromised",
            "updated_at": _utcnow(),
            "quarantine_reason": reason,
        }
        entry["chunks"] = chunks
        entry["status"] = "incomplete"
        entry["batches"] = {}
        documents[document_rel] = entry
        manifest["documents"] = documents
        shutil.rmtree(self.map_root() / "documents" / _safe_rel_path(document_rel) / "batches", ignore_errors=True)
        self.save_map_manifest(manifest)

    # ------------------------------------------------------------------
    # Reduce manifest/payload helpers
    # ------------------------------------------------------------------
    def load_reduce_manifest(self) -> dict[str, Any]:
        payload = _read_json(self.reduce_manifest_path())
        if not isinstance(payload, dict) or payload.get("version") != RECOVERY_VERSION:
            return _default_reduce_manifest()
        payload.setdefault("chunks", {"count": 0, "items": {}})
        payload.setdefault("batches", {})
        payload.setdefault("actual_usage", _default_usage_totals())
        payload.setdefault("actual_cost", 0.0)
        return payload

    def save_reduce_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        payload["version"] = RECOVERY_VERSION
        payload["updated_at"] = _utcnow()
        _write_json(self.reduce_manifest_path(), payload)

    def clear_reduce(self) -> None:
        shutil.rmtree(self.reduce_root(), ignore_errors=True)

    def reset_reduce(self) -> None:
        self.clear_reduce()
        self.save_reduce_manifest(_default_reduce_manifest())

    def clear_reduce_batches(self) -> None:
        shutil.rmtree(self.reduce_root() / "batches", ignore_errors=True)

    def reduce_chunk_path(self, index: int) -> Path:
        return self.reduce_root() / "chunks" / f"chunk_{index}.json"

    def reduce_batch_path(self, level: int, batch_index: int) -> Path:
        return self.reduce_root() / "batches" / f"level_{level}_batch_{batch_index}.json"

    def quarantine_reduce_payload(
        self,
        *,
        kind: str,
        identifier: str,
        payload: Mapping[str, Any] | None,
        reason: str,
    ) -> None:
        quarantine_path = (
            self.reduce_root()
            / "quarantine"
            / f"{kind}_{identifier}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        record = {
            "kind": kind,
            "identifier": identifier,
            "reason": reason,
            "quarantined_at": _utcnow(),
            "payload": dict(payload or {}),
        }
        _write_json(quarantine_path, record)

    def mark_reduce_chunk_compromised(self, *, index: int, reason: str) -> None:
        manifest = self.load_reduce_manifest()
        chunks = dict((manifest.get("chunks") or {}).get("items") or {})
        chunk_key = str(index)
        payload = self.load_payload(self.reduce_chunk_path(index))
        if payload is not None:
            self.quarantine_reduce_payload(
                kind="chunk",
                identifier=chunk_key,
                payload=payload,
                reason=reason,
            )
        try:
            self.reduce_chunk_path(index).unlink()
        except FileNotFoundError:
            pass
        chunks[chunk_key] = {
            **dict(chunks.get(chunk_key) or {}),
            "status": "compromised",
            "updated_at": _utcnow(),
            "quarantine_reason": reason,
        }
        manifest["chunks"] = {
            **dict(manifest.get("chunks") or {}),
            "items": chunks,
        }
        manifest["batches"] = {}
        manifest["finalized"] = False
        manifest["status"] = "incomplete"
        self.clear_reduce_batches()
        self.save_reduce_manifest(manifest)

    # ------------------------------------------------------------------
    # Generic payload validation/save helpers
    # ------------------------------------------------------------------
    def load_payload(self, path: Path) -> dict[str, Any] | None:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return None
        return payload

    def save_payload(
        self,
        path: Path,
        *,
        content: str,
        input_checksum: str,
        status: str = "complete",
    ) -> dict[str, Any]:
        payload = {
            "input_checksum": input_checksum,
            "content": content,
            "content_checksum": _sha256(content),
            "status": status,
            "updated_at": _utcnow(),
        }
        _write_json(path, payload)
        return payload

    def validate_payload(
        self,
        *,
        payload: Mapping[str, Any] | None,
        expected_input_checksum: str,
    ) -> tuple[bool, str]:
        if not payload:
            return False, "missing payload"
        content = payload.get("content")
        if not isinstance(content, str) or not content:
            return False, "missing content"
        if payload.get("input_checksum") != expected_input_checksum:
            return False, "input checksum mismatch"
        if payload.get("content_checksum") != _sha256(content):
            return False, "content checksum mismatch"
        return True, ""

    # ------------------------------------------------------------------
    # Aggregate state helpers
    # ------------------------------------------------------------------
    def add_actuals(
        self,
        manifest: dict[str, Any],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float | None = None,
    ) -> None:
        usage = manifest.setdefault("actual_usage", _default_usage_totals())
        usage["input_tokens"] = int(usage.get("input_tokens", 0) or 0) + int(input_tokens or 0)
        usage["output_tokens"] = int(usage.get("output_tokens", 0) or 0) + int(output_tokens or 0)
        manifest["actual_cost"] = float(manifest.get("actual_cost", 0.0) or 0.0) + float(cost or 0.0)

    # ------------------------------------------------------------------
    # Legacy import
    # ------------------------------------------------------------------
    def import_legacy_map(
        self,
        *,
        checkpoint_root: Path,
        legacy_manifest: Mapping[str, Any],
    ) -> None:
        if self.map_manifest_path().exists() or not checkpoint_root.exists():
            return

        manifest = _default_map_manifest()
        manifest["signature"] = legacy_manifest.get("signature")
        documents = manifest.setdefault("documents", {})

        for document_rel, entry in dict(legacy_manifest.get("documents") or {}).items():
            document_entry = {
                "source_checksum": entry.get("source_checksum"),
                "prompt_hash": entry.get("prompt_hash"),
                "chunk_count": int(entry.get("chunk_count", 0) or 0),
                "status": entry.get("status", "incomplete"),
                "chunks": {},
                "batches": {},
                "ran_at": entry.get("ran_at"),
            }
            for idx in entry.get("chunks_done") or []:
                if not isinstance(idx, int):
                    continue
                legacy_path = checkpoint_root / "map" / _safe_rel_path(document_rel) / f"chunk_{idx}.json"
                payload = self.load_payload(legacy_path)
                if payload is None:
                    continue
                _write_json(self.map_chunk_path(document_rel, idx), payload)
                document_entry["chunks"][str(idx)] = {
                    "status": "complete",
                    "input_checksum": payload.get("input_checksum"),
                    "content_checksum": payload.get("content_checksum"),
                    "updated_at": payload.get("updated_at"),
                }
            documents[str(document_rel)] = document_entry

        self.save_map_manifest(manifest)

    def import_legacy_reduce(
        self,
        *,
        checkpoint_root: Path,
        legacy_manifest: Mapping[str, Any],
    ) -> None:
        if self.reduce_manifest_path().exists() or not checkpoint_root.exists():
            return

        manifest = _default_reduce_manifest()
        manifest["signature"] = legacy_manifest.get("signature")
        manifest["finalized"] = bool(legacy_manifest.get("finalized", False))
        manifest["ran_at"] = legacy_manifest.get("ran_at")
        manifest["status"] = "complete" if manifest["finalized"] else "incomplete"

        chunks = manifest.setdefault("chunks", {"count": 0, "items": {}})
        legacy_chunks = dict((legacy_manifest.get("chunks") or {}))
        chunks["count"] = int(legacy_chunks.get("count", 0) or 0)
        for idx in legacy_chunks.get("done") or []:
            if not isinstance(idx, int):
                continue
            legacy_path = checkpoint_root / "reduce" / "chunks" / f"chunk_{idx}.json"
            payload = self.load_payload(legacy_path)
            if payload is None:
                continue
            _write_json(self.reduce_chunk_path(idx), payload)
            chunks.setdefault("items", {})[str(idx)] = {
                "status": "complete",
                "input_checksum": payload.get("input_checksum"),
                "content_checksum": payload.get("content_checksum"),
                "updated_at": payload.get("updated_at"),
            }

        for batch_key, checksum in dict(legacy_manifest.get("batches") or {}).items():
            if not isinstance(batch_key, str) or ":" not in batch_key:
                continue
            level_text, batch_text = batch_key.split(":", 1)
            if not level_text.isdigit() or not batch_text.isdigit():
                continue
            level = int(level_text)
            batch_index = int(batch_text)
            legacy_path = checkpoint_root / "reduce" / "batches" / f"level_{level}_batch_{batch_index}.json"
            payload = self.load_payload(legacy_path)
            if payload is None:
                continue
            _write_json(self.reduce_batch_path(level, batch_index), payload)
            manifest.setdefault("batches", {})[batch_key] = {
                "status": "complete",
                "input_checksum": checksum,
                "content_checksum": payload.get("content_checksum"),
                "updated_at": payload.get("updated_at"),
            }

        self.save_reduce_manifest(manifest)


def recovery_summary(group_root: Path) -> dict[str, Any]:
    """Return a lightweight summary for UI metrics/status."""

    store = BulkRecoveryStore(group_root)
    map_manifest = store.load_map_manifest()
    reduce_manifest = store.load_reduce_manifest()

    resumable_documents = 0
    resumable_chunks = 0
    corrupt_chunks = 0
    for entry in dict(map_manifest.get("documents") or {}).values():
        if not isinstance(entry, dict):
            continue
        chunk_items = dict(entry.get("chunks") or {})
        pending_or_corrupt = 0
        for item in chunk_items.values():
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "")
            if status in {"complete"}:
                continue
            pending_or_corrupt += 1
            resumable_chunks += 1
            if status in {"corrupt", "compromised"}:
                corrupt_chunks += 1
        if pending_or_corrupt > 0:
            resumable_documents += 1

    reduce_chunk_items = dict((reduce_manifest.get("chunks") or {}).get("items") or {})
    reduce_resumable = 0
    reduce_corrupt = 0
    for item in reduce_chunk_items.values():
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "")
        if status != "complete":
            reduce_resumable += 1
            if status in {"corrupt", "compromised"}:
                reduce_corrupt += 1

    return {
        "map_resumable_documents": resumable_documents,
        "map_resumable_chunks": resumable_chunks,
        "map_corrupt_chunks": corrupt_chunks,
        "reduce_resumable_chunks": reduce_resumable,
        "reduce_corrupt_chunks": reduce_corrupt,
        "map_actual_cost": float(map_manifest.get("actual_cost", 0.0) or 0.0),
        "reduce_actual_cost": float(reduce_manifest.get("actual_cost", 0.0) or 0.0),
    }


__all__ = ["BulkRecoveryStore", "RECOVERY_VERSION", "recovery_summary"]
