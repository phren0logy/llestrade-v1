"""Shared helpers for the bulk-analysis worker."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from src.app.core.azure_artifacts import is_azure_raw_artifact
from src.common.llm.chunking import ChunkingStrategy
from src.common.llm.tokens import TokenCounter
from src.config.paths import app_base_dir, app_resource_root
from src.config.prompt_store import get_bundled_dir, get_custom_dir

from .bulk_analysis_groups import BulkAnalysisGroup
from .project_manager import ProjectMetadata
from .prompt_manager import PromptManager
from .prompt_placeholders import ensure_required_placeholders, format_prompt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBundle:
    """Resolved prompts for a bulk analysis run."""

    system_template: str
    user_template: str


@dataclass(frozen=True)
class BulkAnalysisDocument:
    """Represents a single document to analyse."""

    source_path: Path
    relative_path: str
    output_path: Path


class BulkAnalysisCancelled(Exception):
    """Raised when cancellation is requested during processing."""


def prepare_documents(
    project_dir: Path,
    group: BulkAnalysisGroup,
    selected_files: Sequence[str],
) -> List[BulkAnalysisDocument]:
    """Resolve bulk-analysis documents and their output paths."""

    converted_root = project_dir / "converted_documents"
    group_root = project_dir / "bulk_analysis" / group.folder_name
    documents: List[BulkAnalysisDocument] = []

    for relative in selected_files:
        source_path = converted_root / relative
        if not source_path.exists():
            LOGGER.warning("Converted file missing for bulk analysis: %s", source_path)
            continue

        # Only process markdown or text files produced by conversion
        if source_path.suffix.lower() not in {".md", ".txt"}:
            LOGGER.warning("Skipping non-markdown file for bulk analysis: %s", source_path)
            continue
        if is_azure_raw_artifact(source_path):
            LOGGER.debug("Skipping Azure raw sidecar for bulk analysis: %s", source_path)
            continue

        output_relative = Path(relative).with_suffix("")
        output_filename = f"{output_relative.name}_analysis.md"
        output_path = group_root / output_relative.parent / output_filename
        documents.append(
            BulkAnalysisDocument(
                source_path=source_path,
                relative_path=relative,
                output_path=output_path,
            )
        )

    return documents


def load_prompts(
    project_dir: Path,
    group: BulkAnalysisGroup,
    metadata: Optional[ProjectMetadata],
) -> PromptBundle:
    """Return the prompt bundle for the supplied group."""

    prompt_manager = PromptManager()
    system_template = _read_prompt_file(project_dir, group.system_prompt_path)
    if not system_template:
        try:
            system_template = prompt_manager.get_template("document_analysis_system_prompt")
        except KeyError:
            system_template = "You are a forensic assistant."

    user_template = _read_prompt_file(project_dir, group.user_prompt_path)
    if not user_template:
        try:
            user_template = prompt_manager.get_template("document_bulk_analysis_prompt")
        except KeyError:
            user_template = (
                "Summarise the provided document content focusing on key facts, timelines, "
                "and clinical details.\n\n{document_content}"
            )

    ensure_required_placeholders("document_bulk_analysis_prompt", user_template)

    return PromptBundle(system_template=system_template, user_template=user_template)


def render_system_prompt(
    bundle: PromptBundle,
    metadata: Optional[ProjectMetadata],
    *,
    placeholder_values: Mapping[str, str] | None = None,
) -> str:
    """Format the system prompt with available metadata."""

    context = _metadata_context(metadata)
    if placeholder_values:
        context.update({k: v for k, v in placeholder_values.items() if v is not None})
    return format_prompt(bundle.system_template, context)


def render_user_prompt(
    bundle: PromptBundle,
    metadata: Optional[ProjectMetadata],
    document_name: str,
    document_content: str,
    *,
    placeholder_values: Mapping[str, str] | None = None,
    chunk_index: Optional[int] = None,
    chunk_total: Optional[int] = None,
) -> str:
    """Format the user prompt for a document or chunk."""

    context = _metadata_context(metadata)
    context.update(
        {
            "document_name": document_name,
            "document_content": document_content,
        }
    )
    if placeholder_values:
        context.update({k: v for k, v in placeholder_values.items() if v is not None})
    if chunk_index is not None and chunk_total is not None:
        context.update(
            {
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
            }
        )
    prompt = format_prompt(bundle.user_template, context)
    if chunk_index is not None and chunk_total is not None:
        prefix = (
            f"You are analysing chunk {chunk_index} of {chunk_total} from {document_name}.\n\n"
        )
        prompt = prefix + prompt
    return prompt


def should_chunk(
    content: str,
    provider_id: str,
    model_name: Optional[str],
) -> tuple[bool, int, int]:
    """Return whether chunking is required and the relevant token counts."""

    token_info = TokenCounter.count(text=content, provider=provider_id, model=model_name or "")
    conservative_estimate = max(len(content) // 3, 1)
    if token_info.get("success"):
        counted = int(token_info.get("token_count") or 0)
        if provider_id in {"anthropic", "anthropic_bedrock"}:
            tokens = max(counted, conservative_estimate)
        else:
            tokens = max(counted, 1)
    else:
        tokens = conservative_estimate
    context_window = TokenCounter.get_model_context_window(model_name or provider_id)
    max_tokens_per_chunk = max(context_window, 4000)
    return tokens > max_tokens_per_chunk, tokens, max_tokens_per_chunk


def generate_chunks(content: str, max_tokens: int) -> List[str]:
    """Split the document into manageable chunks."""
    chunks = ChunkingStrategy.markdown_headers(
        text=content,
        max_tokens=max_tokens,
        overlap=2000,
    )
    if not chunks:
        return []

    max_chars = max(max_tokens * 4, 1)
    bounded: List[str] = []
    for chunk in chunks:
        chunk_text = chunk.strip()
        if not chunk_text:
            continue
        if len(chunk_text) <= max_chars:
            bounded.append(chunk_text)
            continue

        fallback = ChunkingStrategy.simple_overlap(chunk_text, max_tokens=max_tokens, overlap=200)
        if not fallback:
            fallback = [chunk_text]

        for piece in fallback:
            piece_text = piece.strip()
            if not piece_text:
                continue
            if len(piece_text) <= max_chars:
                bounded.append(piece_text)
                continue
            # Hard cap split to guarantee no chunk can exceed downstream prompt limits.
            for start in range(0, len(piece_text), max_chars):
                segment = piece_text[start:start + max_chars].strip()
                if segment:
                    bounded.append(segment)

    return bounded


def combine_chunk_summaries(
    summaries: Iterable[str],
    *,
    document_name: str,
    metadata: Optional[ProjectMetadata],
    placeholder_values: Mapping[str, str] | None = None,
) -> tuple[str, Dict[str, str]]:
    """Prepare the final prompt for combining chunk summaries."""

    combined = "\n\n---\n\n".join(summary.strip() for summary in summaries if summary.strip())
    context = _metadata_context(metadata)
    context.update(
        {
            "document_name": document_name,
            "chunk_summaries": combined,
        }
    )
    if placeholder_values:
        context.update({k: v for k, v in placeholder_values.items() if v is not None})
    base_template = (
        "Create a unified bulk analysis by combining these partial results of document: {document_name}\n\n"
        "## Partial Results:\n{chunk_summaries}\n\n"
        "Please create a single, coherent deliverable that captures all key information from the document."
    )
    if placeholder_values:
        visible_keys = [key for key, value in placeholder_values.items() if value]
        if visible_keys:
            summary_lines = "\n".join(f"- {key}: {{{key}}}" for key in sorted(set(visible_keys)))
            base_template += (
                "\n\n### Placeholder Context\n"
                "The following project placeholders are provided for additional context:\n"
                f"{summary_lines}"
            )
    prompt = format_prompt(base_template, context)
    return prompt, context

def combine_chunk_summaries_hierarchical(
    summaries: List[str],
    *,
    document_name: str,
    metadata: Optional[ProjectMetadata],
    placeholder_values: Mapping[str, str] | None = None,
    provider_id: str,
    model: Optional[str] = None,
    invoke_fn,
    is_cancelled_fn=None,
    load_batch_fn=None,
    save_batch_fn=None,
) -> str:
    """
    Hierarchically combine summaries using multi-level reduction when needed.

    This function attempts single-pass combination first (for backward compatibility
    with small documents), then falls back to hierarchical batching if the combined
    prompt would exceed the model's context window.

    Args:
        summaries: List of summary strings to combine
        document_name: Name of the document being processed
        metadata: Project metadata for context
        placeholder_values: Additional placeholder values
        provider_id: LLM provider identifier
        model: Model name for token counting
        invoke_fn: Callable that takes a prompt string and returns LLM response
        is_cancelled_fn: Optional callable that returns True if operation should cancel

    Returns:
        Final combined summary string

    Raises:
        BulkAnalysisCancelled: If is_cancelled_fn returns True during processing
    """
    import logging
    logger = logging.getLogger(__name__)

    # Check for cancellation
    if is_cancelled_fn and is_cancelled_fn():
        raise BulkAnalysisCancelled("Operation cancelled before hierarchical reduction")

    # Calculate thresholds from the model's context window with a 65% safety buffer.
    raw_context_window = TokenCounter.get_model_context_window(model or provider_id, ratio=1.0)
    context_window = TokenCounter.get_model_context_window(model or provider_id)
    max_combine_tokens = int(context_window * 0.95)

    logger.info(
        f"Hierarchical reduction starting: {len(summaries)} summaries, "
        f"max_combine_tokens={max_combine_tokens} (~95% of safe window {context_window}, raw {raw_context_window})"
    )

    # Step 1: Try single-pass combination (existing behavior for small documents)
    prompt, context = combine_chunk_summaries(
        summaries,
        document_name=document_name,
        metadata=metadata,
        placeholder_values=placeholder_values,
    )

    # Count tokens in the combined prompt
    prompt_tokens = _count_tokens(prompt, provider_id, model)

    logger.info(f"Single-pass prompt: {prompt_tokens} tokens")

    # If under threshold, use single-pass (fast path)
    if prompt_tokens <= max_combine_tokens:
        logger.info("Single-pass combination fits within limit, invoking provider")
        return invoke_fn(prompt)

    # Step 2: Multi-level hierarchical reduction required
    logger.warning(
        f"Single-pass prompt exceeds limit ({prompt_tokens} > {max_combine_tokens}), "
        f"switching to hierarchical reduction"
    )

    # Process summaries hierarchically
    current_level_summaries = list(summaries)
    level = 0
    target_summary_tokens = max(int(max_combine_tokens * 0.4), 4000)
    condense_template = (
        "You are preparing intermediate notes for {document_name}. "
        "Condense the partial summary below so the output stays well under {token_target} tokens. "
        "Focus on key facts, merge redundant bullets, and keep any page references when possible.\n\n"
        "## Partial Summary\n"
        "{chunk_summaries}\n\n"
        "## Condensed Summary\n"
    )

    while len(current_level_summaries) > 1:
        level += 1
        logger.info(f"Hierarchical level {level}: processing {len(current_level_summaries)} summaries")

        # Check for cancellation
        if is_cancelled_fn and is_cancelled_fn():
            raise BulkAnalysisCancelled(f"Operation cancelled during hierarchical level {level}")

        # First, ensure no single summary exceeds the per-batch budget.
        normalized_level: List[str] = []
        for summary_idx, summary in enumerate(current_level_summaries, start=1):
            summary_tokens = _count_tokens(summary, provider_id, model)
            if summary_tokens <= target_summary_tokens:
                normalized_level.append(summary)
                continue

            logger.info(
                "Level %s, summary %s exceeds target (%s > %s tokens); condensing",
                level,
                summary_idx,
                summary_tokens,
                target_summary_tokens,
            )

            condensed = summary
            attempts = 0
            while attempts < 3:
                attempts += 1
                if is_cancelled_fn and is_cancelled_fn():
                    raise BulkAnalysisCancelled(
                        f"Operation cancelled while condensing summary {summary_idx} in level {level}"
                    )

                condense_context = _metadata_context(metadata)
                condense_context.update(
                    {
                        "document_name": document_name,
                        "chunk_summaries": condensed,
                        "token_target": str(target_summary_tokens),
                    }
                )
                if placeholder_values:
                    condense_context.update({k: v for k, v in placeholder_values.items() if v is not None})
                condense_prompt = format_prompt(condense_template, condense_context)
                condensed = invoke_fn(condense_prompt).strip()
                condensed_tokens = _count_tokens(condensed, provider_id, model)
                logger.info(
                    "Level %s, summary %s condensed attempt %s produced ~%s tokens",
                    level,
                    summary_idx,
                    attempts,
                    condensed_tokens,
                )
                if condensed_tokens <= target_summary_tokens:
                    break

            final_tokens = _count_tokens(condensed, provider_id, model)
            if final_tokens > target_summary_tokens:
                logger.warning(
                    "Level %s, summary %s still above target after condensation (~%s tokens); truncating",
                    level,
                    summary_idx,
                    final_tokens,
                )
                condensed = _truncate_to_tokens(condensed, target_summary_tokens)

            normalized_level.append(condensed)

        current_level_summaries = normalized_level

        # Dynamically determine batch size by testing token counts
        batches = []
        current_batch = []
        current_batch_tokens = 0

        for idx, summary in enumerate(current_level_summaries):
            # Check cancellation periodically
            if is_cancelled_fn and is_cancelled_fn():
                raise BulkAnalysisCancelled(f"Operation cancelled at summary {idx} in level {level}")

            # Count tokens in this summary
            summary_tokens = _count_tokens(summary, provider_id, model)

            # Calculate tokens if we add this summary to current batch
            # Include overhead for separators and prompt template (~500 tokens)
            test_batch = current_batch + [summary]
            test_prompt, _ = combine_chunk_summaries(
                test_batch,
                document_name=document_name,
                metadata=metadata,
                placeholder_values=placeholder_values,
            )
            test_tokens = _count_tokens(test_prompt, provider_id, model)

            # If adding this summary would exceed limit, finalize current batch
            if test_tokens > max_combine_tokens and current_batch:
                batches.append(current_batch)
                logger.info(
                    f"Level {level}, batch {len(batches)}: {len(current_batch)} summaries, "
                    f"~{current_batch_tokens} tokens"
                )
                current_batch = [summary]
                current_batch_tokens = summary_tokens
            else:
                # Add to current batch
                current_batch.append(summary)
                current_batch_tokens = test_tokens

        # Add final batch
        if current_batch:
            batches.append(current_batch)
            logger.info(
                f"Level {level}, batch {len(batches)}: {len(current_batch)} summaries, "
                f"~{current_batch_tokens} tokens"
            )

        logger.info(f"Level {level}: created {len(batches)} batches")

        # Combine each batch
        next_level_summaries = []
        for batch_idx, batch in enumerate(batches):
            # Check cancellation
            if is_cancelled_fn and is_cancelled_fn():
                raise BulkAnalysisCancelled(
                    f"Operation cancelled at batch {batch_idx + 1}/{len(batches)} in level {level}"
                )

            logger.info(
                f"Level {level}, combining batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} summaries)"
            )

            # Determine checksum for this batch to enable checkpoint reuse
            batch_input_checksum = _batch_checksum(batch)
            if load_batch_fn:
                cached = load_batch_fn(level, batch_idx + 1, batch_input_checksum)
                if cached:
                    next_level_summaries.append(cached)
                    logger.info(
                        f"Level {level}, batch {batch_idx + 1} reused cached result (checkpoint)"
                    )
                    continue

            # Create prompt for this batch
            batch_prompt, _ = combine_chunk_summaries(
                batch,
                document_name=document_name,
                metadata=metadata,
                placeholder_values=placeholder_values,
            )

            # Invoke LLM to combine this batch
            try:
                batch_result = invoke_fn(batch_prompt)
                if save_batch_fn:
                    try:
                        save_batch_fn(level, batch_idx + 1, batch_input_checksum, batch_result)
                    except Exception:
                        logger.debug(
                            "Level %s, batch %s failed to persist checkpoint",
                            level,
                            batch_idx + 1,
                            exc_info=True,
                        )
                next_level_summaries.append(batch_result)
                logger.info(f"Level {level}, batch {batch_idx + 1} completed successfully")
            except Exception as e:
                # Wrap error with context
                logger.error(
                    f"Level {level}, batch {batch_idx + 1} failed: {e}",
                    exc_info=True
                )
                raise Exception(
                    f"Hierarchical reduction failed at level {level}, "
                    f"batch {batch_idx + 1}/{len(batches)}: {e}"
                ) from e

        # Move to next level
        current_level_summaries = next_level_summaries
        logger.info(
            f"Level {level} complete: reduced {len(batches)} batches to "
            f"{len(next_level_summaries)} summaries"
        )

    # Final result
    final_summary = current_level_summaries[0]
    logger.info(
        f"Hierarchical reduction complete after {level} levels, "
        f"final summary: {len(final_summary)} characters"
    )

    return final_summary


def _batch_checksum(batch: List[str]) -> str:
    """Return a stable checksum for a batch of summaries."""
    joined = "\n\n---\n\n".join(batch)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _count_tokens(text: str, provider_id: str, model: Optional[str]) -> int:
    """Return token count for text with a len//4 fallback."""
    token_info = TokenCounter.count(text=text, provider=provider_id, model=model or "")
    conservative = max(len(text) // 3, 1)
    if not token_info.get("success"):
        return conservative
    counted = int(token_info.get("token_count") or 0)
    if provider_id in {"anthropic", "anthropic_bedrock"}:
        return max(counted, conservative)
    return max(counted, 1)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to roughly max_tokens using char-based estimate."""
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _read_prompt_file(project_dir: Path, path_str: str | None) -> str:
    if not path_str:
        return ""
    candidate = Path(path_str).expanduser()
    search_paths: List[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        if project_dir:
            search_paths.append((project_dir / candidate).resolve())
        bundle_root = app_base_dir()
        resource_root = app_resource_root()
        app_root = resource_root.parent
        search_paths.append((bundle_root / candidate).resolve())
        # Support paths relative to src/app/… and the resources directory.
        search_paths.append((app_root / candidate).resolve())
        search_paths.append((resource_root / candidate).resolve())

        # Look inside the user prompt store for both matching relative paths and basename fallbacks
        store_paths: List[Path] = []
        try:
            custom_dir = get_custom_dir()
            store_paths.append(custom_dir / candidate)
            store_paths.append(custom_dir / candidate.name)
        except Exception:
            pass
        try:
            bundled_dir = get_bundled_dir()
            store_paths.append(bundled_dir / candidate)
            store_paths.append(bundled_dir / candidate.name)
        except Exception:
            pass
        search_paths.extend(store_paths)

    seen: set[Path] = set()
    for path in search_paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load prompt file %s: %s", path, exc)
            return ""

    LOGGER.warning("Prompt file %s not found in project or application templates", path_str)
    return ""


def _metadata_context(metadata: Optional[ProjectMetadata]) -> Dict[str, str]:
    if not metadata:
        return {
            "subject_name": "",
            "subject_dob": "",
            "case_info": "",
            "case_name": "",
        }
    return {
        "subject_name": metadata.subject_name or metadata.case_name or "",
        "subject_dob": metadata.date_of_birth or "",
        "case_info": metadata.case_description or "",
        "case_name": metadata.case_name or "",
    }


__all__ = [
    "BulkAnalysisCancelled",
    "BulkAnalysisDocument",
    "PromptBundle",
    "combine_chunk_summaries",
    "combine_chunk_summaries_hierarchical",
    "generate_chunks",
    "load_prompts",
    "prepare_documents",
    "render_system_prompt",
    "render_user_prompt",
    "should_chunk",
]
