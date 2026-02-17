"""SQLite connection manager with FTS5, auto-migration, and WAL mode."""

from __future__ import annotations

import os
from pathlib import Path

import aiosqlite

DEFAULT_DB_DIR = Path.home() / ".lcm"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "lcm.db"

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, list[str]] = {
    1: [
        # Immutable message store
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            token_estimate INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            metadata TEXT DEFAULT '{}'
        )
        """,
        # FTS5 index on message content
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            content='messages',
            content_rowid='id',
            tokenize='porter unicode61'
        )
        """,
        # Triggers to keep FTS5 in sync
        """
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
        END
        """,
        # Summary DAG nodes
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            level INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            token_estimate INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT 'preserve_details',
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            msg_start_id INTEGER,
            msg_end_id INTEGER,
            metadata TEXT DEFAULT '{}'
        )
        """,
        # Links between summary nodes (DAG edges)
        """
        CREATE TABLE IF NOT EXISTS summary_links (
            parent_id INTEGER NOT NULL REFERENCES summaries(id),
            child_id INTEGER NOT NULL REFERENCES summaries(id),
            PRIMARY KEY (parent_id, child_id)
        )
        """,
        # Large file references
        """
        CREATE TABLE IF NOT EXISTS large_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT,
            size_bytes INTEGER,
            exploration_summary TEXT,
            token_estimate INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
        )
        """,
        # Schema version tracking
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )
        """,
        "INSERT OR IGNORE INTO schema_version (version) VALUES (1)",
        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_summaries_msg_range ON summaries(msg_start_id, msg_end_id)",
        "CREATE INDEX IF NOT EXISTS idx_large_files_session ON large_files(session_id)",
    ],
}


async def get_db(db_path: str | Path | None = None) -> aiosqlite.Connection:
    """Open a database connection with WAL mode and run migrations."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row

    # Enable WAL mode and foreign keys
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")

    await _run_migrations(db)
    return db


async def _get_current_version(db: aiosqlite.Connection) -> int:
    """Get the current schema version, or 0 if no schema exists."""
    try:
        cursor = await db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        return row[0] if row and row[0] else 0
    except aiosqlite.OperationalError:
        return 0


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """Run all pending migrations."""
    current = await _get_current_version(db)

    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            for sql in MIGRATIONS[version]:
                await db.execute(sql)
            await db.commit()
