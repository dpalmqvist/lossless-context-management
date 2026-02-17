"""Tests for compaction engine: escalation, thresholds, file ID propagation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from lcm.compaction.engine import (
    _compact_all,
    _compact_oldest,
    _condense_if_needed,
    _split_into_blocks,
    check_and_compact,
    CompactionStats,
)
from lcm.compaction.escalation import EscalationResult, escalated_summarize, _truncate_deterministic
from lcm.store.database import get_db
from lcm.store.messages import insert_message
from lcm.store.summaries import (
    count_summaries,
    get_leaf_summaries,
    get_top_level_summaries,
)


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = await get_db(db_path)
    yield conn
    await conn.close()


# --- Escalation Tests ---


class TestEscalation:
    def test_truncate_deterministic_short(self):
        text = "Short text"
        result = _truncate_deterministic(text, max_tokens=100)
        assert result == text

    def test_truncate_deterministic_long(self):
        text = "A" * 10000
        result = _truncate_deterministic(text, max_tokens=512)
        assert len(result) < len(text)
        assert "[...truncated...]" in result

    @patch("lcm.compaction.escalation.summarize")
    async def test_escalation_level1_success(self, mock_summarize):
        mock_summarize.return_value = "Concise summary"

        result = await escalated_summarize("A" * 5000, target_tokens=500)
        assert result.level == 1
        assert result.mode == "preserve_details"
        assert result.content == "Concise summary"

    @patch("lcm.compaction.escalation.summarize")
    async def test_escalation_passthrough_small(self, mock_summarize):
        result = await escalated_summarize("Short", target_tokens=500)
        assert result.level == 0
        assert result.mode == "passthrough"
        mock_summarize.assert_not_called()

    @patch("lcm.compaction.escalation.summarize")
    async def test_escalation_falls_to_level3(self, mock_summarize):
        # Simulate LLM returning content larger than original (edge case)
        mock_summarize.side_effect = Exception("API error")

        result = await escalated_summarize("B" * 5000, target_tokens=500)
        assert result.level == 3
        assert result.mode == "deterministic_truncate"

    @patch("lcm.compaction.escalation.summarize")
    async def test_escalation_level2_fallback(self, mock_summarize):
        # Level 1 returns something larger, level 2 succeeds
        call_count = 0

        async def side_effect(content, mode, target_tokens, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Level 1: return something larger
                return "X" * len(content)
            else:
                # Level 2: return concise
                return "Bullet summary"

        mock_summarize.side_effect = side_effect

        result = await escalated_summarize("C" * 5000, target_tokens=500)
        assert result.level == 2
        assert result.mode == "bullet_points"


# --- Block Splitting ---


class TestBlockSplitting:
    def test_single_small_block(self):
        from lcm.store.messages import Message

        msgs = [
            Message(id=i, session_id="s1", role="user", content=f"M{i}",
                    token_estimate=10, timestamp="", metadata={})
            for i in range(3)
        ]
        blocks = _split_into_blocks(msgs)
        assert len(blocks) == 1
        assert len(blocks[0]) == 3

    def test_split_large_list(self):
        from lcm.store.messages import Message

        msgs = [
            Message(id=i, session_id="s1", role="user", content=f"M{i}",
                    token_estimate=10, timestamp="", metadata={})
            for i in range(35)
        ]
        blocks = _split_into_blocks(msgs)
        total = sum(len(b) for b in blocks)
        assert total == 35
        for block in blocks:
            assert len(block) <= 15


# --- Engine Integration Tests ---


class TestCompaction:
    @patch("lcm.compaction.engine.escalated_summarize")
    async def test_compact_oldest(self, mock_summarize, db):
        mock_summarize.return_value = EscalationResult(
            content="Summary", level=1, mode="preserve_details", token_estimate=20
        )

        # Insert enough messages
        for i in range(10):
            await insert_message(db, "s1", "user", f"Message {i} " * 50)

        stats = await _compact_oldest(db, "s1")
        assert stats.leaf_summaries_created == 1

    @patch("lcm.compaction.engine.escalated_summarize")
    async def test_compact_all(self, mock_summarize, db):
        mock_summarize.return_value = EscalationResult(
            content="Summary", level=1, mode="preserve_details", token_estimate=20
        )

        for i in range(20):
            await insert_message(db, "s1", "user", f"Message {i} " * 50)

        stats = await _compact_all(db, "s1")
        assert stats.leaf_summaries_created > 0

    @patch("lcm.compaction.engine.escalated_summarize")
    async def test_condense_when_enough_leaves(self, mock_summarize, db):
        mock_summarize.return_value = EscalationResult(
            content="Condensed", level=1, mode="preserve_details", token_estimate=30
        )

        # Create 6 leaf summaries manually
        from lcm.store.summaries import create_leaf_summary

        for i in range(6):
            start = i * 10 + 1
            end = start + 9
            await create_leaf_summary(db, "s1", f"Leaf {i}", msg_start_id=start, msg_end_id=end)

        leaves_before = await get_leaf_summaries(db, "s1")
        assert len(leaves_before) == 6

        stats = await _condense_if_needed(db, "s1")
        assert stats.condensed_summaries_created == 1

        top = await get_top_level_summaries(db, "s1")
        assert len(top) == 1
        assert top[0].level == 1

    @patch("lcm.compaction.engine.escalated_summarize")
    async def test_file_id_propagation(self, mock_summarize, db):
        """Verify condensed summaries inherit the message range from children."""
        mock_summarize.return_value = EscalationResult(
            content="Summary", level=1, mode="preserve_details", token_estimate=20
        )

        from lcm.store.summaries import create_leaf_summary, create_condensed_summary, get_summary

        c1 = await create_leaf_summary(db, "s1", "L1", msg_start_id=1, msg_end_id=50)
        c2 = await create_leaf_summary(db, "s1", "L2", msg_start_id=51, msg_end_id=100)

        parent = await create_condensed_summary(db, "s1", "P", child_ids=[c1, c2])
        p = await get_summary(db, parent)

        assert p.msg_start_id == 1
        assert p.msg_end_id == 100
