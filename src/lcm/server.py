"""FastMCP server entry point — registers all LCM tools."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import aiosqlite
from mcp.server.fastmcp import FastMCP

from lcm.store.database import get_db

# Global database connection
_db: aiosqlite.Connection | None = None


async def _get_db() -> aiosqlite.Connection:
    global _db
    if _db is None:
        db_path = os.environ.get("LCM_DB_PATH")
        _db = await get_db(db_path)
    return _db


mcp = FastMCP(
    "LCM — Lossless Context Management",
    instructions=(
        "Provides lossless memory retrieval, context preservation, "
        "and parallel data processing for Claude Code."
    ),
)


@mcp.tool()
async def lcm_status(session_id: str = "default") -> str:
    """Get LCM session stats: message count, summary count, tokens stored, DAG depth, and top-level summaries.

    Args:
        session_id: Session identifier (defaults to "default")
    """
    from lcm.tools.status import lcm_status as _status

    db = await _get_db()
    result = await _status(db, session_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def lcm_grep(
    pattern: str,
    session_id: str | None = None,
    summary_id: int | None = None,
    page: int = 1,
    use_regex: bool = False,
) -> str:
    """Search message history using FTS5 full-text search or regex.

    Results are grouped by their covering summary for context. Use summary_id to search within a specific summary's messages.

    Args:
        pattern: Search query (FTS5 syntax or regex pattern)
        session_id: Filter by session (optional)
        summary_id: Search within a specific summary's messages (optional)
        page: Page number for pagination (default: 1, 10 results per page)
        use_regex: Use regex instead of FTS5 (default: False)
    """
    from lcm.tools.memory import lcm_grep as _grep

    db = await _get_db()
    result = await _grep(
        db, pattern, session_id=session_id, summary_id=summary_id,
        page=page, use_regex=use_regex,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def lcm_describe(lcm_id: str) -> str:
    """Look up metadata for any LCM entity by ID.

    ID formats: plain number for messages, "S" prefix for summaries (e.g. "S5"), "F" prefix for files (e.g. "F3").

    Args:
        lcm_id: The LCM ID to look up (e.g. "42", "S5", "F3")
    """
    from lcm.tools.memory import lcm_describe as _describe

    db = await _get_db()
    result = await _describe(db, lcm_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def lcm_expand(summary_id: int, page: int = 1) -> str:
    """Expand a summary to its constituent messages.

    Use this to drill into a summary and see the original messages it covers. Paginated (10 per page).

    Args:
        summary_id: The summary ID (numeric, without the "S" prefix)
        page: Page number for pagination (default: 1)
    """
    from lcm.tools.memory import lcm_expand as _expand

    db = await _get_db()
    result = await _expand(db, summary_id, page=page)
    return json.dumps(result, indent=2)


@mcp.tool()
async def llm_map(
    input_path: str,
    prompt: str,
    output_schema: str = "{}",
    concurrency: int = 16,
) -> str:
    """Process each line of a JSONL file with a stateless LLM call (Haiku).

    Fan out concurrent calls for high throughput. Each item is independently classified/transformed.

    Args:
        input_path: Path to input JSONL file
        prompt: Instructions for processing each item
        output_schema: JSON schema string for output validation (default: "{}" for no validation)
        concurrency: Max concurrent API calls (default: 16)
    """
    from lcm.tools.operators import llm_map as _llm_map

    schema = json.loads(output_schema) if output_schema != "{}" else None
    result = await _llm_map(
        input_path=input_path,
        prompt=prompt,
        output_schema=schema,
        concurrency=concurrency,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def agentic_map(
    input_path: str,
    prompt: str,
    output_schema: str = "{}",
    read_only: bool = True,
    concurrency: int = 4,
) -> str:
    """Process each JSONL item with a multi-turn agent loop (Haiku + tools).

    Each item gets a multi-turn conversation with file read and optional bash access. Lower concurrency than llm_map.

    Args:
        input_path: Path to input JSONL file
        prompt: Instructions for the agent processing each item
        output_schema: JSON schema string for output validation (default: "{}" for no validation)
        read_only: Restrict agent to read-only operations (default: True)
        concurrency: Max concurrent agent loops (default: 4)
    """
    from lcm.tools.operators import agentic_map as _agentic_map

    schema = json.loads(output_schema) if output_schema != "{}" else None
    result = await _agentic_map(
        input_path=input_path,
        prompt=prompt,
        output_schema=schema,
        read_only=read_only,
        concurrency=concurrency,
    )
    return json.dumps(result, indent=2)


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
