"""Context control loop — manages τ_soft/τ_hard thresholds and DAG condensation."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from lcm.compaction.escalation import escalated_summarize
from lcm.store.messages import Message, get_unsummarized_messages, total_tokens
from lcm.store.summaries import (
    create_condensed_summary,
    create_leaf_summary,
    get_leaf_summaries,
)

# Default thresholds (in estimated tokens)
TAU_SOFT = 50_000
TAU_HARD = 200_000
BLOCK_SIZE_MIN = 5
BLOCK_SIZE_MAX = 15
CONDENSATION_THRESHOLD = 5  # Condense when 5+ uncondensed leaf summaries exist


@dataclass
class CompactionStats:
    leaf_summaries_created: int = 0
    condensed_summaries_created: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0


async def check_and_compact(
    db: aiosqlite.Connection,
    session_id: str,
    tau_soft: int = TAU_SOFT,
    tau_hard: int = TAU_HARD,
    model: str | None = None,
) -> CompactionStats:
    """Main compaction entry point.

    Checks token count against thresholds and triggers appropriate compaction.
    """
    stats = CompactionStats()
    stats.total_tokens_before = await total_tokens(db, session_id)

    if stats.total_tokens_before >= tau_hard:
        # Blocking: summarize ALL unsummarized messages
        stats = await _compact_all(db, session_id, model=model, stats=stats)
    elif stats.total_tokens_before >= tau_soft:
        # Async: summarize oldest unsummarized messages
        stats = await _compact_oldest(db, session_id, model=model, stats=stats)

    # DAG condensation: condense leaf summaries if enough exist
    stats = await _condense_if_needed(db, session_id, model=model, stats=stats)

    stats.total_tokens_after = await total_tokens(db, session_id)
    return stats


async def _compact_oldest(
    db: aiosqlite.Connection,
    session_id: str,
    model: str | None = None,
    stats: CompactionStats | None = None,
) -> CompactionStats:
    """Summarize the oldest block of unsummarized messages."""
    stats = stats or CompactionStats()
    unsummarized = await get_unsummarized_messages(db, session_id)

    if len(unsummarized) < BLOCK_SIZE_MIN:
        return stats

    # Take one block of messages
    block_size = min(len(unsummarized), BLOCK_SIZE_MAX)
    block = unsummarized[:block_size]
    await _summarize_block(db, session_id, block, model=model)
    stats.leaf_summaries_created += 1

    return stats


async def _compact_all(
    db: aiosqlite.Connection,
    session_id: str,
    model: str | None = None,
    stats: CompactionStats | None = None,
) -> CompactionStats:
    """Summarize ALL unsummarized messages in blocks."""
    stats = stats or CompactionStats()
    unsummarized = await get_unsummarized_messages(db, session_id)

    if not unsummarized:
        return stats

    # Split into blocks
    blocks = _split_into_blocks(unsummarized)
    for block in blocks:
        await _summarize_block(db, session_id, block, model=model)
        stats.leaf_summaries_created += 1

    return stats


def _split_into_blocks(messages: list[Message]) -> list[list[Message]]:
    """Split messages into blocks of BLOCK_SIZE_MIN to BLOCK_SIZE_MAX."""
    blocks: list[list[Message]] = []
    remaining = list(messages)

    while remaining:
        if len(remaining) <= BLOCK_SIZE_MAX:
            blocks.append(remaining)
            break

        # If remaining is just slightly over max, split more evenly
        if len(remaining) <= BLOCK_SIZE_MAX * 2:
            mid = len(remaining) // 2
            blocks.append(remaining[:mid])
            blocks.append(remaining[mid:])
            break

        blocks.append(remaining[:BLOCK_SIZE_MAX])
        remaining = remaining[BLOCK_SIZE_MAX:]

    return blocks


async def _summarize_block(
    db: aiosqlite.Connection,
    session_id: str,
    block: list[Message],
    model: str | None = None,
) -> int:
    """Create a leaf summary for a block of messages."""
    # Build the content to summarize
    content_parts = []
    for msg in block:
        content_parts.append(f"[{msg.role}]: {msg.content}")
    full_content = "\n\n".join(content_parts)

    # Escalated summarization
    result = await escalated_summarize(full_content, target_tokens=500, model=model)

    summary_id = await create_leaf_summary(
        db=db,
        session_id=session_id,
        content=result.content,
        msg_start_id=block[0].id,
        msg_end_id=block[-1].id,
        mode=result.mode,
        token_estimate=result.token_estimate,
    )
    return summary_id


async def _condense_if_needed(
    db: aiosqlite.Connection,
    session_id: str,
    model: str | None = None,
    stats: CompactionStats | None = None,
) -> CompactionStats:
    """Condense uncondensed leaf summaries if there are enough of them."""
    stats = stats or CompactionStats()
    uncondensed = await get_leaf_summaries(db, session_id)

    if len(uncondensed) < CONDENSATION_THRESHOLD:
        return stats

    # Combine all uncondensed leaf summaries into one condensed node
    content_parts = []
    child_ids = []
    for s in uncondensed:
        content_parts.append(s.content)
        child_ids.append(s.id)

    full_content = "\n\n---\n\n".join(content_parts)

    result = await escalated_summarize(full_content, target_tokens=800, model=model)

    await create_condensed_summary(
        db=db,
        session_id=session_id,
        content=result.content,
        child_ids=child_ids,
        mode=result.mode,
        token_estimate=result.token_estimate,
    )
    stats.condensed_summaries_created += 1

    return stats
