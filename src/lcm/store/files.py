"""Large file reference storage with exploration summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass

import aiosqlite


@dataclass
class FileRef:
    id: int
    session_id: str
    file_path: str
    file_type: str | None
    size_bytes: int | None
    exploration_summary: str | None
    token_estimate: int
    timestamp: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> FileRef:
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            size_bytes=row["size_bytes"],
            exploration_summary=row["exploration_summary"],
            token_estimate=row["token_estimate"],
            timestamp=row["timestamp"],
        )


async def store_file_ref(
    db: aiosqlite.Connection,
    session_id: str,
    file_path: str,
    file_type: str | None = None,
    size_bytes: int | None = None,
    exploration_summary: str | None = None,
) -> int:
    """Store a file reference. Returns the file ID."""
    token_est = max(1, len(exploration_summary) // 4) if exploration_summary else 0

    cursor = await db.execute(
        """
        INSERT INTO large_files (session_id, file_path, file_type, size_bytes,
                                  exploration_summary, token_estimate)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, file_path, file_type, size_bytes, exploration_summary, token_est),
    )
    await db.commit()
    return cursor.lastrowid


async def get_file_ref(db: aiosqlite.Connection, file_id: int) -> FileRef | None:
    """Get a file reference by ID."""
    cursor = await db.execute("SELECT * FROM large_files WHERE id = ?", (file_id,))
    row = await cursor.fetchone()
    return FileRef.from_row(row) if row else None


async def get_files_by_session(
    db: aiosqlite.Connection, session_id: str
) -> list[FileRef]:
    """Get all file references for a session."""
    cursor = await db.execute(
        "SELECT * FROM large_files WHERE session_id = ? ORDER BY id",
        (session_id,),
    )
    return [FileRef.from_row(row) for row in await cursor.fetchall()]


async def search_files_by_path(
    db: aiosqlite.Connection,
    pattern: str,
    session_id: str | None = None,
) -> list[FileRef]:
    """Search file references by path pattern (SQL LIKE)."""
    if session_id:
        cursor = await db.execute(
            "SELECT * FROM large_files WHERE file_path LIKE ? AND session_id = ? ORDER BY id",
            (f"%{pattern}%", session_id),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM large_files WHERE file_path LIKE ? ORDER BY id",
            (f"%{pattern}%",),
        )
    return [FileRef.from_row(row) for row in await cursor.fetchall()]
