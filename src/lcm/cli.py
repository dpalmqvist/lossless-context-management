"""CLI entry points for hook scripts and manual operations."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from lcm.store.database import get_db


def main() -> None:
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        print("Usage: lcm <command> [args]", file=sys.stderr)
        print("Commands: hook, status", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "hook":
        _handle_hook()
    elif command == "status":
        _handle_status()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


def _handle_hook() -> None:
    """Dispatch hook subcommands: capture, inject, init."""
    if len(sys.argv) < 3:
        print("Usage: lcm hook <capture|inject|init>", file=sys.stderr)
        sys.exit(1)

    subcommand = sys.argv[2]
    session_id = os.environ.get("CLAUDE_SESSION_ID", "default")

    if subcommand == "capture":
        asyncio.run(_hook_capture(session_id))
    elif subcommand == "inject":
        asyncio.run(_hook_inject(session_id))
    elif subcommand == "init":
        asyncio.run(_hook_init(session_id))
    else:
        print(f"Unknown hook subcommand: {subcommand}", file=sys.stderr)
        sys.exit(1)


async def _hook_capture(session_id: str) -> None:
    """Capture new messages from Claude Code transcript."""
    from lcm.hooks.capture import capture_new_messages

    db = await get_db()
    try:
        # Read transcript path from environment or auto-detect
        transcript_path = os.environ.get("CLAUDE_TRANSCRIPT_PATH")
        result = await capture_new_messages(db, session_id, transcript_path)
        if result.get("captured", 0) > 0:
            print(json.dumps(result), file=sys.stderr)
    finally:
        await db.close()


async def _hook_inject(session_id: str) -> None:
    """Inject summary context after compaction."""
    from lcm.hooks.inject import build_injection_text

    db = await get_db()
    try:
        text = await build_injection_text(db, session_id)
        if text:
            # Output to stdout — Claude Code reads this as injected context
            print(text)
    finally:
        await db.close()


async def _hook_init(session_id: str) -> None:
    """Initialize LCM for a new session."""
    db = await get_db()
    try:
        # Just ensure the database is initialized
        from lcm.store.messages import count_messages
        count = await count_messages(db, session_id)
        print(
            json.dumps({"session_id": session_id, "existing_messages": count}),
            file=sys.stderr,
        )
    finally:
        await db.close()


def _handle_status() -> None:
    """Print session status."""
    session_id = os.environ.get("CLAUDE_SESSION_ID", "default")
    if len(sys.argv) >= 3:
        session_id = sys.argv[2]

    asyncio.run(_print_status(session_id))


async def _print_status(session_id: str) -> None:
    from lcm.tools.status import lcm_status

    db = await get_db()
    try:
        result = await lcm_status(db, session_id)
        print(json.dumps(result, indent=2))
    finally:
        await db.close()


if __name__ == "__main__":
    main()
