"""Memory retrieval tools: lcm_grep, lcm_describe, lcm_expand."""

from __future__ import annotations

import json

import aiosqlite

from lcm.store.files import FileRef, get_file_ref
from lcm.store.messages import (
    Message,
    get_messages_by_range,
    search_messages_fts,
    search_messages_regex,
)
from lcm.store.summaries import (
    Summary,
    get_children,
    get_covering_summary,
    get_summary,
)

PAGE_SIZE = 10


def _format_message(msg: Message) -> dict:
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content[:500] + ("..." if len(msg.content) > 500 else ""),
        "timestamp": msg.timestamp,
        "tokens": msg.token_estimate,
    }


def _format_summary(s: Summary) -> dict:
    return {
        "id": f"S{s.id}",
        "level": s.level,
        "mode": s.mode,
        "content": s.content[:500] + ("..." if len(s.content) > 500 else ""),
        "msg_range": f"{s.msg_start_id}-{s.msg_end_id}",
        "tokens": s.token_estimate,
        "timestamp": s.timestamp,
    }


async def lcm_grep(
    db: aiosqlite.Connection,
    pattern: str,
    session_id: str | None = None,
    summary_id: int | None = None,
    page: int = 1,
    use_regex: bool = False,
) -> dict:
    """Search message history using FTS5 or regex.

    Results are grouped by their covering summary for context.
    Paginated, PAGE_SIZE results per page.
    """
    offset = (page - 1) * PAGE_SIZE

    # If searching within a specific summary's messages
    if summary_id is not None:
        summary = await get_summary(db, summary_id)
        if not summary:
            return {"error": f"Summary S{summary_id} not found"}

        messages = await get_messages_by_range(
            db, summary.msg_start_id, summary.msg_end_id
        )
        # Filter by pattern
        import re

        compiled = re.compile(pattern, re.IGNORECASE) if use_regex else None
        filtered = []
        for msg in messages:
            if use_regex and compiled and compiled.search(msg.content):
                filtered.append(msg)
            elif not use_regex and pattern.lower() in msg.content.lower():
                filtered.append(msg)

        total = len(filtered)
        page_results = filtered[offset : offset + PAGE_SIZE]
    else:
        # Global search
        if use_regex:
            results = await search_messages_regex(
                db, pattern, session_id=session_id, limit=PAGE_SIZE, offset=offset
            )
        else:
            try:
                results = await search_messages_fts(
                    db, pattern, session_id=session_id, limit=PAGE_SIZE, offset=offset
                )
            except Exception:
                # Fall back to regex on FTS5 syntax errors
                results = await search_messages_regex(
                    db, pattern, session_id=session_id, limit=PAGE_SIZE, offset=offset
                )
        page_results = results
        total = len(results)  # Approximate; FTS5 doesn't cheaply give total

    # Group by covering summary
    grouped: dict[str, dict] = {}
    for msg in page_results:
        covering = await get_covering_summary(db, msg.id)
        key = f"S{covering.id}" if covering else "unsummarized"
        if key not in grouped:
            grouped[key] = {
                "summary_id": key,
                "summary_preview": (
                    covering.content[:200] if covering else "No covering summary"
                ),
                "messages": [],
            }
        grouped[key]["messages"].append(_format_message(msg))

    return {
        "pattern": pattern,
        "page": page,
        "page_size": PAGE_SIZE,
        "results": list(grouped.values()),
        "has_more": total >= PAGE_SIZE,
    }


async def lcm_describe(db: aiosqlite.Connection, lcm_id: str) -> dict:
    """Metadata lookup for any LCM ID.

    IDs prefixed with 'S' are summaries, 'F' are files, plain numbers are messages.
    """
    if lcm_id.startswith("S"):
        summary_id = int(lcm_id[1:])
        summary = await get_summary(db, summary_id)
        if not summary:
            return {"error": f"Summary {lcm_id} not found"}

        children = await get_children(db, summary_id)
        return {
            "type": "summary",
            **_format_summary(summary),
            "children": [_format_summary(c) for c in children],
        }

    elif lcm_id.startswith("F"):
        file_id = int(lcm_id[1:])
        file_ref = await get_file_ref(db, file_id)
        if not file_ref:
            return {"error": f"File {lcm_id} not found"}

        return {
            "type": "file",
            "id": f"F{file_ref.id}",
            "path": file_ref.file_path,
            "file_type": file_ref.file_type,
            "size_bytes": file_ref.size_bytes,
            "summary": file_ref.exploration_summary,
            "tokens": file_ref.token_estimate,
        }

    else:
        # Assume message ID
        from lcm.store.messages import get_message

        try:
            msg_id = int(lcm_id)
        except ValueError:
            return {"error": f"Invalid ID format: {lcm_id}"}

        msg = await get_message(db, msg_id)
        if not msg:
            return {"error": f"Message {lcm_id} not found"}

        covering = await get_covering_summary(db, msg.id)
        return {
            "type": "message",
            **_format_message(msg),
            "covering_summary": _format_summary(covering) if covering else None,
            "metadata": msg.metadata,
        }


async def lcm_expand(
    db: aiosqlite.Connection,
    summary_id: int,
    page: int = 1,
) -> dict:
    """Expand a summary to its constituent messages. Paginated."""
    summary = await get_summary(db, summary_id)
    if not summary:
        return {"error": f"Summary S{summary_id} not found"}

    if summary.msg_start_id is None or summary.msg_end_id is None:
        return {
            "summary": _format_summary(summary),
            "messages": [],
            "note": "No message range associated with this summary",
        }

    offset = (page - 1) * PAGE_SIZE
    messages = await get_messages_by_range(db, summary.msg_start_id, summary.msg_end_id)

    total = len(messages)
    page_messages = messages[offset : offset + PAGE_SIZE]

    # Also get child summaries if this is a condensed node
    children = await get_children(db, summary_id)

    return {
        "summary": _format_summary(summary),
        "page": page,
        "page_size": PAGE_SIZE,
        "total_messages": total,
        "messages": [_format_message(m) for m in page_messages],
        "child_summaries": [_format_summary(c) for c in children],
        "has_more": (offset + PAGE_SIZE) < total,
    }
