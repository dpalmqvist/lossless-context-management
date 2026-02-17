"""Status tool: lcm_status — session stats and DAG overview."""

from __future__ import annotations

import aiosqlite

from lcm.store.messages import count_messages, total_tokens
from lcm.store.summaries import (
    Summary,
    count_summaries,
    get_dag_depth,
    get_top_level_summaries,
)


async def lcm_status(db: aiosqlite.Connection, session_id: str) -> dict:
    """Return session stats: message count, summary count, tokens, DAG info."""
    msg_count = await count_messages(db, session_id)
    sum_count = await count_summaries(db, session_id)
    tokens = await total_tokens(db, session_id)
    depth = await get_dag_depth(db, session_id)
    top_level = await get_top_level_summaries(db, session_id)

    return {
        "session_id": session_id,
        "message_count": msg_count,
        "summary_count": sum_count,
        "total_tokens_stored": tokens,
        "dag_depth": depth,
        "top_level_summaries": [
            {
                "id": f"S{s.id}",
                "level": s.level,
                "mode": s.mode,
                "msg_range": f"{s.msg_start_id}-{s.msg_end_id}",
                "tokens": s.token_estimate,
                "preview": s.content[:200],
            }
            for s in top_level
        ],
    }
