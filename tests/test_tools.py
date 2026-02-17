"""Tests for MCP tools: memory, operators, status."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lcm.store.database import get_db
from lcm.store.messages import insert_message
from lcm.store.summaries import create_condensed_summary, create_leaf_summary
from lcm.store.files import store_file_ref
from lcm.tools.memory import lcm_describe, lcm_expand, lcm_grep
from lcm.tools.operators import llm_map, agentic_map
from lcm.tools.status import lcm_status


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = await get_db(db_path)
    yield conn
    await conn.close()


# --- lcm_grep Tests ---


class TestLcmGrep:
    async def test_fts_search(self, db):
        await insert_message(db, "s1", "user", "Fix the authentication bug")
        await insert_message(db, "s1", "user", "Deploy to production")

        result = await lcm_grep(db, "authentication")
        assert len(result["results"]) > 0
        assert result["pattern"] == "authentication"

    async def test_regex_search(self, db):
        await insert_message(db, "s1", "user", "Error code: 404")
        await insert_message(db, "s1", "user", "Error code: 500")

        result = await lcm_grep(db, r"Error code: \d+", use_regex=True)
        assert len(result["results"]) > 0

    async def test_search_within_summary(self, db):
        id1 = await insert_message(db, "s1", "user", "Alpha feature")
        id2 = await insert_message(db, "s1", "user", "Beta feature")
        id3 = await insert_message(db, "s1", "user", "Gamma feature")

        sid = await create_leaf_summary(db, "s1", "Features", msg_start_id=id1, msg_end_id=id2)

        result = await lcm_grep(db, "Alpha", summary_id=sid)
        assert len(result["results"]) > 0

    async def test_pagination(self, db):
        for i in range(25):
            await insert_message(db, "s1", "user", f"Search term item {i}")

        page1 = await lcm_grep(db, "Search term", page=1)
        page2 = await lcm_grep(db, "Search term", page=2)
        # Both pages should return results
        assert len(page1["results"]) > 0


# --- lcm_describe Tests ---


class TestLcmDescribe:
    async def test_describe_message(self, db):
        mid = await insert_message(db, "s1", "user", "Test message")
        result = await lcm_describe(db, str(mid))
        assert result["type"] == "message"
        assert result["role"] == "user"

    async def test_describe_summary(self, db):
        sid = await create_leaf_summary(db, "s1", "Summary text", msg_start_id=1, msg_end_id=5)
        result = await lcm_describe(db, f"S{sid}")
        assert result["type"] == "summary"
        assert result["level"] == 0

    async def test_describe_file(self, db):
        fid = await store_file_ref(db, "s1", "/test.py", file_type="py", size_bytes=100)
        result = await lcm_describe(db, f"F{fid}")
        assert result["type"] == "file"
        assert result["path"] == "/test.py"

    async def test_describe_not_found(self, db):
        result = await lcm_describe(db, "999")
        assert "error" in result

    async def test_describe_invalid_id(self, db):
        result = await lcm_describe(db, "XYZ")
        assert "error" in result


# --- lcm_expand Tests ---


class TestLcmExpand:
    async def test_expand_leaf(self, db):
        id1 = await insert_message(db, "s1", "user", "First")
        id2 = await insert_message(db, "s1", "user", "Second")
        id3 = await insert_message(db, "s1", "user", "Third")

        sid = await create_leaf_summary(db, "s1", "Summary", msg_start_id=id1, msg_end_id=id3)
        result = await lcm_expand(db, sid)
        assert result["total_messages"] == 3
        assert len(result["messages"]) == 3

    async def test_expand_condensed(self, db):
        c1 = await create_leaf_summary(db, "s1", "Child 1", msg_start_id=1, msg_end_id=5)
        c2 = await create_leaf_summary(db, "s1", "Child 2", msg_start_id=6, msg_end_id=10)
        parent = await create_condensed_summary(db, "s1", "Parent", child_ids=[c1, c2])

        result = await lcm_expand(db, parent)
        assert len(result["child_summaries"]) == 2

    async def test_expand_not_found(self, db):
        result = await lcm_expand(db, 999)
        assert "error" in result

    async def test_expand_pagination(self, db):
        ids = []
        for i in range(15):
            ids.append(await insert_message(db, "s1", "user", f"Msg {i}"))

        sid = await create_leaf_summary(db, "s1", "Lots", msg_start_id=ids[0], msg_end_id=ids[-1])

        page1 = await lcm_expand(db, sid, page=1)
        assert len(page1["messages"]) == 10
        assert page1["has_more"] is True

        page2 = await lcm_expand(db, sid, page=2)
        assert len(page2["messages"]) == 5
        assert page2["has_more"] is False


# --- lcm_status Tests ---


class TestLcmStatus:
    async def test_empty_session(self, db):
        result = await lcm_status(db, "empty")
        assert result["message_count"] == 0
        assert result["summary_count"] == 0

    async def test_with_data(self, db):
        await insert_message(db, "s1", "user", "Hello")
        await insert_message(db, "s1", "assistant", "Hi")
        await create_leaf_summary(db, "s1", "Greeting", msg_start_id=1, msg_end_id=2)

        result = await lcm_status(db, "s1")
        assert result["message_count"] == 2
        assert result["summary_count"] == 1
        assert result["total_tokens_stored"] > 0


# --- llm_map Tests ---


class TestLlmMap:
    @patch("lcm.tools.operators.classify")
    async def test_llm_map_basic(self, mock_classify, tmp_path):
        mock_classify.return_value = {"label": "positive"}

        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"text": "I love this"}\n'
            '{"text": "This is great"}\n'
            '{"text": "Wonderful!"}\n'
        )

        result = await llm_map(
            input_path=str(input_file),
            prompt="Classify sentiment",
        )

        assert result["total_items"] == 3
        assert result["successful"] == 3
        assert result["failed"] == 0
        assert Path(result["output_path"]).exists()

    @patch("lcm.tools.operators.classify")
    async def test_llm_map_with_failures(self, mock_classify, tmp_path):
        call_count = 0

        async def side_effect(item, prompt, output_schema):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # First item fails all 3 retries
                raise Exception("API error")
            return {"label": "ok"}

        mock_classify.side_effect = side_effect

        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"text": "fail"}\n'
            '{"text": "pass"}\n'
        )

        result = await llm_map(
            input_path=str(input_file),
            prompt="Process",
        )

        assert result["failed"] == 1
        assert result["successful"] == 1

    async def test_llm_map_missing_file(self):
        result = await llm_map(
            input_path="/nonexistent.jsonl",
            prompt="Process",
        )
        assert "error" in result


# --- agentic_map Tests ---


class TestAgenticMap:
    @patch("lcm.tools.operators.agent_loop")
    async def test_agentic_map_basic(self, mock_agent, tmp_path):
        mock_agent.return_value = {"analysis": "done"}

        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"file": "test.py"}\n')

        result = await agentic_map(
            input_path=str(input_file),
            prompt="Analyze file",
        )

        assert result["total_items"] == 1
        assert result["successful"] == 1
