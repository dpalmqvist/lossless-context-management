"""Tests for hooks: transcript parsing, diff logic, injection formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lcm.hooks.capture import _extract_message, capture_new_messages
from lcm.hooks.inject import build_injection_text
from lcm.store.database import get_db
from lcm.store.messages import count_messages
from lcm.store.summaries import create_leaf_summary


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = await get_db(db_path)
    yield conn
    await conn.close()


# --- Transcript Parsing ---


class TestExtractMessage:
    def test_standard_message(self):
        entry = {"type": "message", "role": "user", "content": "Hello"}
        role, content = _extract_message(entry)
        assert role == "user"
        assert content == "Hello"

    def test_multipart_content(self):
        entry = {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "name": "read", "input": {"path": "/foo"}},
            ],
        }
        role, content = _extract_message(entry)
        assert role == "assistant"
        assert "Let me check" in content
        assert "[Tool: read" in content

    def test_human_format(self):
        entry = {"type": "human", "message": "How do I fix this?"}
        role, content = _extract_message(entry)
        assert role == "user"
        assert content == "How do I fix this?"

    def test_assistant_format(self):
        entry = {"type": "assistant", "message": "Here's the fix"}
        role, content = _extract_message(entry)
        assert role == "assistant"

    def test_tool_result_format(self):
        entry = {"type": "tool_result", "content": "File contents here"}
        role, content = _extract_message(entry)
        assert role == "tool"

    def test_unknown_format(self):
        entry = {"type": "unknown", "data": "something"}
        role, content = _extract_message(entry)
        assert role is None
        assert content is None

    def test_empty_content(self):
        entry = {"type": "message", "role": "user", "content": ""}
        role, content = _extract_message(entry)
        assert role == "user"
        assert content is None  # Empty string → None


# --- Capture Diff Logic ---


class TestCapture:
    async def test_capture_from_transcript(self, db, tmp_path):
        # Create a fake transcript
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"type": "message", "role": "user", "content": "First"}) + "\n"
            + json.dumps({"type": "message", "role": "assistant", "content": "Reply"}) + "\n"
        )

        result = await capture_new_messages(db, "capture_basic", str(transcript))
        assert result["captured"] == 2

        count = await count_messages(db, "capture_basic")
        assert count == 2

    async def test_capture_incremental(self, db, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"type": "message", "role": "user", "content": "First"}) + "\n"
        )

        result1 = await capture_new_messages(db, "capture_incr", str(transcript))
        assert result1["captured"] == 1

        # Append more lines
        with open(transcript, "a") as f:
            f.write(json.dumps({"type": "message", "role": "user", "content": "Second"}) + "\n")

        result2 = await capture_new_messages(db, "capture_incr", str(transcript))
        assert result2["captured"] == 1

        count = await count_messages(db, "capture_incr")
        assert count == 2

    async def test_capture_missing_file(self, db):
        result = await capture_new_messages(db, "capture_missing", "/nonexistent.jsonl")
        assert result["captured"] == 0
        assert "error" in result


# --- Injection Formatting ---


class TestInjection:
    async def test_empty_session(self, db):
        text = await build_injection_text(db, "empty")
        assert text == ""

    async def test_injection_with_summaries(self, db):
        await create_leaf_summary(
            db, "s1", "Set up project structure with FastAPI",
            msg_start_id=1, msg_end_id=10,
        )
        await create_leaf_summary(
            db, "s1", "Implemented authentication module",
            msg_start_id=11, msg_end_id=20,
        )

        text = await build_injection_text(db, "s1")
        assert "LCM Context Recovery" in text
        assert "Set up project" in text
        assert "authentication" in text
        assert "lcm_expand" in text

    async def test_injection_respects_token_limit(self, db):
        # Create a large summary
        await create_leaf_summary(
            db, "s1", "X" * 20000,  # ~5000 tokens
            msg_start_id=1, msg_end_id=10,
        )
        await create_leaf_summary(
            db, "s1", "Y" * 20000,
            msg_start_id=11, msg_end_id=20,
        )

        text = await build_injection_text(db, "s1", max_tokens=2000)
        # Should not include all content
        assert len(text) < 25000
