"""Transcript reading + diff — capture new messages from Claude Code."""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiosqlite

from lcm.store.messages import insert_message

# State file tracks last-processed position per session
DEFAULT_STATE_DIR = Path.home() / ".lcm" / "state"


def _state_file(session_id: str, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{session_id}.pos"


def _get_last_position(session_id: str, state_dir: Path = DEFAULT_STATE_DIR) -> int:
    """Get the last-processed line number for a session."""
    sf = _state_file(session_id, state_dir)
    if sf.exists():
        try:
            return int(sf.read_text().strip())
        except ValueError:
            return 0
    return 0


def _set_last_position(session_id: str, position: int, state_dir: Path = DEFAULT_STATE_DIR) -> None:
    """Store the last-processed line number."""
    sf = _state_file(session_id, state_dir)
    sf.write_text(str(position))


def find_transcript_path(session_id: str) -> Path | None:
    """Find the JSONL transcript file for a Claude Code session."""
    # Claude Code stores transcripts in ~/.claude/projects/*/SESSION_ID.jsonl
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None

    for project_dir in claude_dir.iterdir():
        if project_dir.is_dir():
            transcript = project_dir / f"{session_id}.jsonl"
            if transcript.exists():
                return transcript
    return None


async def capture_new_messages(
    db: aiosqlite.Connection,
    session_id: str,
    transcript_path: str | None = None,
    state_dir: Path = DEFAULT_STATE_DIR,
) -> dict:
    """Read new messages from Claude Code transcript and persist them.

    Returns stats about what was captured.
    """
    if transcript_path:
        path = Path(transcript_path)
    else:
        path = find_transcript_path(session_id)

    if not path or not path.exists():
        return {"captured": 0, "error": "Transcript not found"}

    last_pos = _get_last_position(session_id, state_dir)
    new_messages = 0
    final_pos = last_pos

    with open(path) as f:
        for line_num, line in enumerate(f):
            if line_num < last_pos:
                continue

            line = line.strip()
            if not line:
                final_pos = line_num + 1
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                final_pos = line_num + 1
                continue

            # Extract message content based on Claude Code transcript format
            role, content = _extract_message(entry)
            if role and content:
                await insert_message(
                    db=db,
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata={"source": "transcript", "line": line_num},
                )
                new_messages += 1

            final_pos = line_num + 1

    _set_last_position(session_id, final_pos, state_dir)
    return {"captured": new_messages, "last_position": final_pos}


def _extract_message(entry: dict) -> tuple[str | None, str | None]:
    """Extract role and content from a transcript JSONL entry.

    Claude Code transcript entries have varying formats. We handle the common ones.
    """
    # Standard message format
    if "type" in entry and entry["type"] == "message":
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        if isinstance(content, list):
            # Multi-part content (text + tool_use etc.)
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_use":
                        parts.append(
                            f"[Tool: {part.get('name', '?')}({json.dumps(part.get('input', {}))[:200]})]"
                        )
                    elif part.get("type") == "tool_result":
                        result_content = part.get("content", "")
                        if isinstance(result_content, str):
                            parts.append(f"[ToolResult: {result_content[:500]}]")
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        return role, content if content else None

    # User prompt format
    if "type" in entry and entry["type"] == "human":
        return "user", entry.get("message", entry.get("content", ""))

    # Assistant response format
    if "type" in entry and entry["type"] == "assistant":
        return "assistant", entry.get("message", entry.get("content", ""))

    # Tool result format
    if "type" in entry and entry["type"] == "tool_result":
        return "tool", str(entry.get("content", ""))[:1000]

    return None, None
