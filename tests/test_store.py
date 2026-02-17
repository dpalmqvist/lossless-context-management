"""Tests for the storage layer: messages, summaries, files."""

from __future__ import annotations

import pytest
import aiosqlite

from lcm.store.database import get_db
from lcm.store.messages import (
    count_messages,
    get_message,
    get_messages_by_range,
    get_messages_by_session,
    get_unsummarized_messages,
    insert_message,
    search_messages_fts,
    search_messages_regex,
    total_tokens,
)
from lcm.store.summaries import (
    count_summaries,
    create_condensed_summary,
    create_leaf_summary,
    get_children,
    get_covering_summary,
    get_dag_depth,
    get_leaf_summaries,
    get_parents,
    get_summary,
    get_top_level_summaries,
)
from lcm.store.files import (
    get_file_ref,
    get_files_by_session,
    search_files_by_path,
    store_file_ref,
)


@pytest.fixture
async def db(tmp_path):
    """Create a fresh test database."""
    db_path = tmp_path / "test.db"
    conn = await get_db(db_path)
    yield conn
    await conn.close()


# --- Message Tests ---


class TestMessages:
    async def test_insert_and_get(self, db):
        msg_id = await insert_message(db, "s1", "user", "Hello world")
        msg = await get_message(db, msg_id)
        assert msg is not None
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.session_id == "s1"
        assert msg.token_estimate > 0

    async def test_append_only_ids(self, db):
        id1 = await insert_message(db, "s1", "user", "First")
        id2 = await insert_message(db, "s1", "assistant", "Second")
        assert id2 > id1

    async def test_get_by_range(self, db):
        ids = []
        for i in range(5):
            ids.append(await insert_message(db, "s1", "user", f"Message {i}"))

        msgs = await get_messages_by_range(db, ids[1], ids[3])
        assert len(msgs) == 3
        assert msgs[0].content == "Message 1"
        assert msgs[-1].content == "Message 3"

    async def test_get_by_session(self, db):
        await insert_message(db, "s1", "user", "Session 1")
        await insert_message(db, "s2", "user", "Session 2")
        await insert_message(db, "s1", "user", "Session 1 again")

        msgs = await get_messages_by_session(db, "s1")
        assert len(msgs) == 2

    async def test_fts_search(self, db):
        await insert_message(db, "s1", "user", "Fix the authentication bug in login.py")
        await insert_message(db, "s1", "user", "Add unit tests for the payment module")
        await insert_message(db, "s1", "user", "Deploy to staging environment")

        results = await search_messages_fts(db, "authentication")
        assert len(results) == 1
        assert "authentication" in results[0].content

    async def test_fts_search_with_session_filter(self, db):
        await insert_message(db, "s1", "user", "Python is great")
        await insert_message(db, "s2", "user", "Python is amazing")

        results = await search_messages_fts(db, "Python", session_id="s1")
        assert len(results) == 1

    async def test_regex_search(self, db):
        await insert_message(db, "s1", "user", "Error: NullPointerException at line 42")
        await insert_message(db, "s1", "user", "Fixed the typo in readme")

        results = await search_messages_regex(db, r"Error:.*line \d+")
        assert len(results) == 1
        assert "NullPointerException" in results[0].content

    async def test_count_and_tokens(self, db):
        await insert_message(db, "s1", "user", "Hello")
        await insert_message(db, "s1", "user", "World " * 100)

        assert await count_messages(db, "s1") == 2
        assert await total_tokens(db, "s1") > 0

    async def test_unsummarized_messages(self, db):
        id1 = await insert_message(db, "s1", "user", "First")
        id2 = await insert_message(db, "s1", "user", "Second")
        id3 = await insert_message(db, "s1", "user", "Third")

        unsummarized = await get_unsummarized_messages(db, "s1")
        assert len(unsummarized) == 3

        # Create a summary covering first two
        await create_leaf_summary(db, "s1", "Summary", id1, id2)
        unsummarized = await get_unsummarized_messages(db, "s1")
        assert len(unsummarized) == 1
        assert unsummarized[0].id == id3


# --- Summary Tests ---


class TestSummaries:
    async def test_create_leaf(self, db):
        sid = await create_leaf_summary(
            db, "s1", "This is a summary", msg_start_id=1, msg_end_id=5
        )
        summary = await get_summary(db, sid)
        assert summary is not None
        assert summary.level == 0
        assert summary.content == "This is a summary"
        assert summary.msg_start_id == 1
        assert summary.msg_end_id == 5

    async def test_create_condensed(self, db):
        c1 = await create_leaf_summary(db, "s1", "Part 1", msg_start_id=1, msg_end_id=5)
        c2 = await create_leaf_summary(db, "s1", "Part 2", msg_start_id=6, msg_end_id=10)

        parent = await create_condensed_summary(
            db, "s1", "Combined summary", child_ids=[c1, c2]
        )
        summary = await get_summary(db, parent)
        assert summary.level == 1
        assert summary.msg_start_id == 1
        assert summary.msg_end_id == 10

    async def test_get_children(self, db):
        c1 = await create_leaf_summary(db, "s1", "Child 1", msg_start_id=1, msg_end_id=5)
        c2 = await create_leaf_summary(db, "s1", "Child 2", msg_start_id=6, msg_end_id=10)
        parent = await create_condensed_summary(
            db, "s1", "Parent", child_ids=[c1, c2]
        )

        children = await get_children(db, parent)
        assert len(children) == 2

    async def test_get_parents(self, db):
        c1 = await create_leaf_summary(db, "s1", "Child", msg_start_id=1, msg_end_id=5)
        parent = await create_condensed_summary(
            db, "s1", "Parent", child_ids=[c1]
        )

        parents = await get_parents(db, c1)
        assert len(parents) == 1
        assert parents[0].id == parent

    async def test_covering_summary(self, db):
        id1 = await insert_message(db, "s1", "user", "Msg 1")
        id2 = await insert_message(db, "s1", "user", "Msg 2")
        id3 = await insert_message(db, "s1", "user", "Msg 3")

        await create_leaf_summary(db, "s1", "Covers 1-2", msg_start_id=id1, msg_end_id=id2)

        covering = await get_covering_summary(db, id1)
        assert covering is not None
        assert "Covers 1-2" in covering.content

        covering3 = await get_covering_summary(db, id3)
        assert covering3 is None

    async def test_top_level_summaries(self, db):
        c1 = await create_leaf_summary(db, "s1", "Leaf 1", msg_start_id=1, msg_end_id=5)
        c2 = await create_leaf_summary(db, "s1", "Leaf 2", msg_start_id=6, msg_end_id=10)
        parent = await create_condensed_summary(
            db, "s1", "Condensed", child_ids=[c1, c2]
        )
        c3 = await create_leaf_summary(db, "s1", "Leaf 3", msg_start_id=11, msg_end_id=15)

        top = await get_top_level_summaries(db, "s1")
        # parent and c3 should be top-level (c1, c2 are children)
        assert len(top) == 2
        ids = {s.id for s in top}
        assert parent in ids
        assert c3 in ids

    async def test_leaf_summaries_uncondensed(self, db):
        c1 = await create_leaf_summary(db, "s1", "Leaf 1", msg_start_id=1, msg_end_id=5)
        c2 = await create_leaf_summary(db, "s1", "Leaf 2", msg_start_id=6, msg_end_id=10)
        parent = await create_condensed_summary(
            db, "s1", "Parent", child_ids=[c1]
        )
        # c2 is uncondensed, c1 is in parent
        uncondensed = await get_leaf_summaries(db, "s1")
        assert len(uncondensed) == 1
        assert uncondensed[0].id == c2

    async def test_dag_depth(self, db):
        c1 = await create_leaf_summary(db, "s1", "L0", msg_start_id=1, msg_end_id=5)
        c2 = await create_leaf_summary(db, "s1", "L0", msg_start_id=6, msg_end_id=10)
        p = await create_condensed_summary(db, "s1", "L1", child_ids=[c1, c2])

        assert await get_dag_depth(db, "s1") == 1

    async def test_count(self, db):
        await create_leaf_summary(db, "s1", "A", msg_start_id=1, msg_end_id=5)
        await create_leaf_summary(db, "s1", "B", msg_start_id=6, msg_end_id=10)
        assert await count_summaries(db, "s1") == 2


# --- File Tests ---


class TestFiles:
    async def test_store_and_get(self, db):
        fid = await store_file_ref(
            db, "s1", "/path/to/file.py",
            file_type="py", size_bytes=1234,
            exploration_summary="A Python module",
        )
        ref = await get_file_ref(db, fid)
        assert ref is not None
        assert ref.file_path == "/path/to/file.py"
        assert ref.file_type == "py"
        assert ref.size_bytes == 1234

    async def test_get_by_session(self, db):
        await store_file_ref(db, "s1", "/a.py")
        await store_file_ref(db, "s1", "/b.py")
        await store_file_ref(db, "s2", "/c.py")

        files = await get_files_by_session(db, "s1")
        assert len(files) == 2

    async def test_search_by_path(self, db):
        await store_file_ref(db, "s1", "/src/auth/login.py")
        await store_file_ref(db, "s1", "/src/auth/logout.py")
        await store_file_ref(db, "s1", "/src/main.py")

        results = await search_files_by_path(db, "auth")
        assert len(results) == 2
