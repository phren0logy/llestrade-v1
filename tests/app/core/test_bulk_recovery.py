from __future__ import annotations

from pathlib import Path

from src.app.core.bulk_recovery import BulkRecoveryStore


def test_mark_map_chunk_compromised_quarantines_payload_and_clears_batches(tmp_path: Path) -> None:
    group_root = tmp_path / "bulk_analysis" / "group"
    store = BulkRecoveryStore(group_root)
    manifest = store.load_map_manifest()
    manifest["documents"] = {
        "doc.md": {
            "status": "complete",
            "chunks": {
                "1": {"status": "complete"},
            },
            "batches": {
                "1:1": {"status": "complete"},
            },
        }
    }
    store.save_map_manifest(manifest)
    store.save_payload(
        store.map_chunk_path("doc.md", 1),
        content="chunk output",
        input_checksum="abc",
    )
    store.save_payload(
        store.map_batch_path("doc.md", 1, 1),
        content="batch output",
        input_checksum="def",
    )

    store.mark_map_chunk_compromised(document_rel="doc.md", index=1, reason="test compromise")

    updated = store.load_map_manifest()
    entry = updated["documents"]["doc.md"]
    assert entry["status"] == "incomplete"
    assert entry["chunks"]["1"]["status"] == "compromised"
    assert entry["batches"] == {}
    assert not store.map_chunk_path("doc.md", 1).exists()
    assert not store.map_batch_path("doc.md", 1, 1).exists()
    assert any((store.map_root() / "quarantine" / "doc.md").glob("chunk_1_*.json"))


def test_mark_reduce_chunk_compromised_quarantines_payload_and_invalidates_batches(tmp_path: Path) -> None:
    group_root = tmp_path / "bulk_analysis" / "group"
    store = BulkRecoveryStore(group_root)
    manifest = store.load_reduce_manifest()
    manifest["status"] = "complete"
    manifest["finalized"] = True
    manifest["chunks"] = {
        "count": 1,
        "items": {
            "1": {"status": "complete"},
        },
    }
    manifest["batches"] = {"1:1": {"status": "complete"}}
    store.save_reduce_manifest(manifest)
    store.save_payload(store.reduce_chunk_path(1), content="chunk", input_checksum="abc")
    store.save_payload(store.reduce_batch_path(1, 1), content="batch", input_checksum="def")

    store.mark_reduce_chunk_compromised(index=1, reason="test compromise")

    updated = store.load_reduce_manifest()
    assert updated["status"] == "incomplete"
    assert updated["finalized"] is False
    assert updated["chunks"]["items"]["1"]["status"] == "compromised"
    assert updated["batches"] == {}
    assert not store.reduce_chunk_path(1).exists()
    assert not store.reduce_batch_path(1, 1).exists()
