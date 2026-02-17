"""Summary DAG operations — create, link, traverse summary nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import aiosqlite


@dataclass
class Summary:
    id: int
    session_id: str
    level: int
    content: str
    token_estimate: int
    mode: str
    timestamp: str
    msg_start_id: int | None
    msg_end_id: int | None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Summary:
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            level=row["level"],
            content=row["content"],
            token_estimate=row["token_estimate"],
            mode=row["mode"],
            timestamp=row["timestamp"],
            msg_start_id=row["msg_start_id"],
            msg_end_id=row["msg_end_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


async def create_leaf_summary(
    db: aiosqlite.Connection,
    session_id: str,
    content: str,
    msg_start_id: int,
    msg_end_id: int,
    mode: str = "preserve_details",
    token_estimate: int = 0,
    metadata: dict | None = None,
) -> int:
    """Create a leaf summary node covering a message range. Returns summary ID."""
    if token_estimate == 0:
        token_estimate = max(1, len(content) // 4)

    cursor = await db.execute(
        """
        INSERT INTO summaries (session_id, level, content, token_estimate, mode,
                               msg_start_id, msg_end_id, metadata)
        VALUES (?, 0, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            content,
            token_estimate,
            mode,
            msg_start_id,
            msg_end_id,
            json.dumps(metadata or {}),
        ),
    )
    await db.commit()
    return cursor.lastrowid


async def create_condensed_summary(
    db: aiosqlite.Connection,
    session_id: str,
    content: str,
    child_ids: list[int],
    mode: str = "preserve_details",
    token_estimate: int = 0,
    metadata: dict | None = None,
) -> int:
    """Create a condensed summary that covers multiple child summaries."""
    if token_estimate == 0:
        token_estimate = max(1, len(content) // 4)

    # Find the overall message range from children
    if child_ids:
        placeholders = ",".join("?" * len(child_ids))
        cursor = await db.execute(
            f"""
            SELECT MIN(msg_start_id) as min_start, MAX(msg_end_id) as max_end
            FROM summaries WHERE id IN ({placeholders})
            """,
            child_ids,
        )
        row = await cursor.fetchone()
        msg_start_id = row["min_start"]
        msg_end_id = row["max_end"]
    else:
        msg_start_id = None
        msg_end_id = None

    # Determine level (max child level + 1)
    if child_ids:
        cursor = await db.execute(
            f"SELECT MAX(level) as max_level FROM summaries WHERE id IN ({placeholders})",
            child_ids,
        )
        row = await cursor.fetchone()
        level = (row["max_level"] or 0) + 1
    else:
        level = 1

    cursor = await db.execute(
        """
        INSERT INTO summaries (session_id, level, content, token_estimate, mode,
                               msg_start_id, msg_end_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            level,
            content,
            token_estimate,
            mode,
            msg_start_id,
            msg_end_id,
            json.dumps(metadata or {}),
        ),
    )
    parent_id = cursor.lastrowid

    # Create links
    for child_id in child_ids:
        await db.execute(
            "INSERT INTO summary_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )

    await db.commit()
    return parent_id


async def get_summary(db: aiosqlite.Connection, summary_id: int) -> Summary | None:
    """Get a single summary by ID."""
    cursor = await db.execute("SELECT * FROM summaries WHERE id = ?", (summary_id,))
    row = await cursor.fetchone()
    return Summary.from_row(row) if row else None


async def get_children(db: aiosqlite.Connection, summary_id: int) -> list[Summary]:
    """Get child summaries of a condensed node."""
    cursor = await db.execute(
        """
        SELECT s.* FROM summaries s
        JOIN summary_links sl ON s.id = sl.child_id
        WHERE sl.parent_id = ?
        ORDER BY s.msg_start_id
        """,
        (summary_id,),
    )
    return [Summary.from_row(row) for row in await cursor.fetchall()]


async def get_parents(db: aiosqlite.Connection, summary_id: int) -> list[Summary]:
    """Get parent summaries that contain this node."""
    cursor = await db.execute(
        """
        SELECT s.* FROM summaries s
        JOIN summary_links sl ON s.id = sl.parent_id
        WHERE sl.child_id = ?
        """,
        (summary_id,),
    )
    return [Summary.from_row(row) for row in await cursor.fetchall()]


async def get_covering_summary(
    db: aiosqlite.Connection, message_id: int
) -> Summary | None:
    """Find the highest-level summary covering a given message ID."""
    cursor = await db.execute(
        """
        SELECT s.* FROM summaries s
        WHERE s.msg_start_id <= ? AND s.msg_end_id >= ?
        ORDER BY s.level DESC
        LIMIT 1
        """,
        (message_id, message_id),
    )
    row = await cursor.fetchone()
    return Summary.from_row(row) if row else None


async def get_top_level_summaries(
    db: aiosqlite.Connection, session_id: str
) -> list[Summary]:
    """Get summaries that are not children of any other summary."""
    cursor = await db.execute(
        """
        SELECT s.* FROM summaries s
        WHERE s.session_id = ?
        AND s.id NOT IN (SELECT child_id FROM summary_links)
        ORDER BY s.msg_start_id
        """,
        (session_id,),
    )
    return [Summary.from_row(row) for row in await cursor.fetchall()]


async def get_leaf_summaries(
    db: aiosqlite.Connection, session_id: str
) -> list[Summary]:
    """Get leaf summaries (level 0) that are not yet condensed into a parent."""
    cursor = await db.execute(
        """
        SELECT s.* FROM summaries s
        WHERE s.session_id = ? AND s.level = 0
        AND s.id NOT IN (SELECT child_id FROM summary_links)
        ORDER BY s.msg_start_id
        """,
        (session_id,),
    )
    return [Summary.from_row(row) for row in await cursor.fetchall()]


async def count_summaries(
    db: aiosqlite.Connection, session_id: str | None = None
) -> int:
    """Count summaries, optionally filtered by session."""
    if session_id:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM summaries WHERE session_id = ?", (session_id,)
        )
    else:
        cursor = await db.execute("SELECT COUNT(*) FROM summaries")
    row = await cursor.fetchone()
    return row[0]


async def get_dag_depth(db: aiosqlite.Connection, session_id: str) -> int:
    """Get the maximum depth of the summary DAG for a session."""
    cursor = await db.execute(
        "SELECT COALESCE(MAX(level), 0) FROM summaries WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    return row[0]
