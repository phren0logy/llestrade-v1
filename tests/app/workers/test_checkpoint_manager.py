import json
from pathlib import Path

import pytest

from src.app.core import bulk_analysis_runner as runner
from src.app.workers.checkpoint_manager import CheckpointManager, _sha256


def test_checkpoint_manager_map_roundtrip(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    doc = "doc.md"
    content = "summary content"
    checksum = _sha256("chunk-1")

    mgr.save_map_chunk(doc, 1, content, checksum)
    loaded = mgr.load_map_chunk(doc, 1)

    assert loaded is not None
    assert loaded["input_checksum"] == checksum
    assert loaded["content"] == content


def test_checkpoint_manager_reduce_batch_roundtrip(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    content = "batch result"
    checksum = _sha256("batch-input")

    mgr.save_reduce_batch(1, 2, content, checksum)
    loaded = mgr.load_reduce_batch(1, 2)

    assert loaded is not None
    assert loaded["input_checksum"] == checksum
    assert loaded["content"] == content


def test_hierarchical_combiner_reuses_cached_batches(monkeypatch) -> None:
    # Force hierarchical path by making token counts exceed the threshold.
    monkeypatch.setattr(
        runner.TokenCounter,
        "count",
        staticmethod(
            lambda text, provider, model=None: {
                "success": True,
                "token_count": max(1, len(text) // 4),
            }
        ),
    )
    monkeypatch.setattr(
        runner.TokenCounter,
        "get_model_context_window",
        staticmethod(lambda model, ratio=None: 500),
    )

    summaries = ["x" * 4000, "y" * 4000, "z" * 4000]
    calls = {"invoke": 0}

    def invoke_fn(prompt: str) -> str:
        calls["invoke"] += 1
        return f"combined-{calls['invoke']}"

    cache_hits: list[tuple[int, int]] = []

    def load_batch_fn(level: int, batch_index: int, checksum: str):
        # Only the first batch is cached; others are processed normally.
        if level == 1 and batch_index == 1:
            cache_hits.append((level, batch_index))
            return "cached-batch"
        return None

    saved_batches: list[str] = []

    def save_batch_fn(level: int, batch_index: int, checksum: str, content: str):
        saved_batches.append(f"{level}:{batch_index}:{content}")

    result = runner.combine_chunk_summaries_hierarchical(
        summaries,
        document_name="Doc",
        metadata=None,
        placeholder_values=None,
        provider_id="anthropic",
        model=None,
        invoke_fn=invoke_fn,
        is_cancelled_fn=None,
        load_batch_fn=load_batch_fn,
        save_batch_fn=save_batch_fn,
    )

    # Cached batch reused without invoking provider again.
    assert (1, 1) in cache_hits
    assert all(not entry.startswith("1:1:") for entry in saved_batches)
    assert calls["invoke"] >= 1
    assert result
    assert any(entry.endswith("combined-1") or entry.endswith("combined-2") for entry in saved_batches)
