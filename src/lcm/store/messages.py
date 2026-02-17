"""Message CRUD operations — append-only store with FTS5 search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import aiosqlite


@dataclass
class Message:
    id: int
    session_id: str
    role: str
    content: str
    token_estimate: int
    timestamp: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Message:
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            token_estimate=row["token_estimate"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


async def insert_message(
    db: aiosqlite.Connection,
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> int:
    """Insert a message (append-only). Returns the new message ID."""
    token_est = estimate_tokens(content)
    meta_json = json.dumps(metadata or {})

    cursor = await db.execute(
        """
        INSERT INTO messages (session_id, role, content, token_estimate, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, role, content, token_est, meta_json),
    )
    await db.commit()
    return cursor.lastrowid


async def get_message(db: aiosqlite.Connection, message_id: int) -> Message | None:
    """Get a single message by ID."""
    cursor = await db.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
    row = await cursor.fetchone()
    return Message.from_row(row) if row else None


async def get_messages_by_range(
    db: aiosqlite.Connection,
    start_id: int,
    end_id: int,
) -> list[Message]:
    """Get messages in an ID range (inclusive)."""
    cursor = await db.execute(
        "SELECT * FROM messages WHERE id >= ? AND id <= ? ORDER BY id",
        (start_id, end_id),
    )
    return [Message.from_row(row) for row in await cursor.fetchall()]


async def get_messages_by_session(
    db: aiosqlite.Connection,
    session_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[Message]:
    """Get messages for a session, ordered by ID."""
    cursor = await db.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY id LIMIT ? OFFSET ?",
        (session_id, limit, offset),
    )
    return [Message.from_row(row) for row in await cursor.fetchall()]


async def search_messages_fts(
    db: aiosqlite.Connection,
    query: str,
    session_id: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[Message]:
    """Search messages using FTS5. Query uses FTS5 syntax (AND, OR, NOT, phrases)."""
    if session_id:
        cursor = await db.execute(
            """
            SELECT m.* FROM messages m
            JOIN messages_fts f ON m.id = f.rowid
            WHERE messages_fts MATCH ? AND m.session_id = ?
            ORDER BY rank
            LIMIT ? OFFSET ?
            """,
            (query, session_id, limit, offset),
        )
    else:
        cursor = await db.execute(
            """
            SELECT m.* FROM messages m
            JOIN messages_fts f ON m.id = f.rowid
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ? OFFSET ?
            """,
            (query, limit, offset),
        )
    return [Message.from_row(row) for row in await cursor.fetchall()]


async def search_messages_regex(
    db: aiosqlite.Connection,
    pattern: str,
    session_id: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[Message]:
    """Search messages using Python regex on content."""
    compiled = re.compile(pattern, re.IGNORECASE)

    if session_id:
        cursor = await db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
    else:
        cursor = await db.execute("SELECT * FROM messages ORDER BY id")

    results: list[Message] = []
    skipped = 0
    async for row in cursor:
        if compiled.search(row["content"]):
            if skipped < offset:
                skipped += 1
                continue
            results.append(Message.from_row(row))
            if len(results) >= limit:
                break

    return results


async def count_messages(
    db: aiosqlite.Connection, session_id: str | None = None
) -> int:
    """Count total messages, optionally filtered by session."""
    if session_id:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        )
    else:
        cursor = await db.execute("SELECT COUNT(*) FROM messages")
    row = await cursor.fetchone()
    return row[0]


async def total_tokens(db: aiosqlite.Connection, session_id: str | None = None) -> int:
    """Sum of token estimates for all messages, optionally filtered by session."""
    if session_id:
        cursor = await db.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        )
    else:
        cursor = await db.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages"
        )
    row = await cursor.fetchone()
    return row[0]


async def get_unsummarized_messages(
    db: aiosqlite.Connection, session_id: str
) -> list[Message]:
    """Get messages that don't have a covering summary yet."""
    cursor = await db.execute(
        """
        SELECT m.* FROM messages m
        WHERE m.session_id = ?
        AND NOT EXISTS (
            SELECT 1 FROM summaries s
            WHERE s.session_id = m.session_id
            AND s.msg_start_id <= m.id
            AND s.msg_end_id >= m.id
        )
        ORDER BY m.id
        """,
        (session_id,),
    )
    return [Message.from_row(row) for row in await cursor.fetchall()]
