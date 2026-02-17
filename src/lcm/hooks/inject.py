"""Summary injection — build context block from top-level DAG nodes after compaction."""

from __future__ import annotations

import aiosqlite

from lcm.store.summaries import get_top_level_summaries


async def build_injection_text(
    db: aiosqlite.Connection,
    session_id: str,
    max_tokens: int = 4000,
) -> str:
    """Build a summary context block for injection after compaction.

    Formats top-level DAG nodes as a concise context block with LCM IDs
    that the assistant can use with lcm_expand to drill into details.
    """
    summaries = await get_top_level_summaries(db, session_id)

    if not summaries:
        return ""

    parts = [
        "# LCM Context Recovery",
        "",
        "The following summaries cover your conversation history. "
        "Use `lcm_expand(summary_id)` to retrieve full messages, "
        "or `lcm_grep(pattern)` to search across all history.",
        "",
    ]

    total_tokens = 0
    for s in summaries:
        if total_tokens + s.token_estimate > max_tokens:
            parts.append(
                f"\n[{len(summaries) - len(parts) + 4} more summaries available — use lcm_status()]"
            )
            break

        msg_range = f"messages {s.msg_start_id}-{s.msg_end_id}" if s.msg_start_id else "no messages"
        parts.append(f"## S{s.id} (L{s.level}, {msg_range})")
        parts.append(s.content)
        parts.append("")
        total_tokens += s.token_estimate

    return "\n".join(parts)
