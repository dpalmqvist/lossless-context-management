"""End-to-end integration test: full pipeline from message insertion to context injection."""

from __future__ import annotations

import json

import pytest

from lcm.store.database import get_db
from lcm.store.messages import count_messages, insert_message, total_tokens
from lcm.store.summaries import create_leaf_summary
from lcm.tools.memory import lcm_describe, lcm_expand, lcm_grep
from lcm.tools.status import lcm_status
from lcm.hooks.inject import build_injection_text

SESSION = "e2e-test"

CONVERSATION = [
    ("user", "Help me create a FastAPI application with authentication"),
    ("assistant", "I will create a FastAPI app with JWT authentication. Let me start with the project structure."),
    ("user", "Use SQLAlchemy for the database"),
    ("assistant", "Setting up SQLAlchemy with async support. Created models/user.py with User model containing id, email, hashed_password fields."),
    ("user", "Add an endpoint for user registration"),
    ("assistant", "Created POST /api/auth/register endpoint in routes/auth.py. It validates email format, hashes password with bcrypt, and stores in the database."),
    ("user", "Now add login endpoint that returns JWT tokens"),
    ("assistant", "Created POST /api/auth/login endpoint. It verifies credentials, generates access_token (15min) and refresh_token (7d) using python-jose. Tokens include user_id and email claims."),
    ("user", "Add a protected endpoint that requires authentication"),
    ("assistant", "Created GET /api/users/me with Depends(get_current_user). The dependency extracts and validates the JWT from the Authorization header. Returns user profile data."),
    ("user", "Can you add rate limiting to the auth endpoints?"),
    ("assistant", "Added slowapi rate limiting: /register is limited to 5/minute, /login to 10/minute per IP. Configured in middleware.py with Redis backend."),
    ("user", "Write tests for the auth flow"),
    ("assistant", "Created tests/test_auth.py with pytest-asyncio: test_register_success, test_register_duplicate_email, test_login_success, test_login_wrong_password, test_protected_endpoint_with_token, test_protected_endpoint_without_token. All 6 tests pass."),
]


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "e2e.db"
    conn = await get_db(db_path)
    yield conn
    await conn.close()


@pytest.fixture
async def populated_db(db):
    """DB with all conversation messages inserted."""
    for role, content in CONVERSATION:
        await insert_message(db, SESSION, role, content)
    return db


@pytest.fixture
async def summarized_db(populated_db):
    """DB with messages and two leaf summaries."""
    db = populated_db
    await create_leaf_summary(
        db, SESSION,
        "Set up FastAPI project with JWT auth and SQLAlchemy. Created User model with id/email/hashed_password. Added /register and /login endpoints.",
        msg_start_id=1, msg_end_id=8,
    )
    await create_leaf_summary(
        db, SESSION,
        "Added protected GET /users/me endpoint with JWT dependency. Added rate limiting (slowapi+Redis): 5/min register, 10/min login. Wrote 6 auth tests — all pass.",
        msg_start_id=9, msg_end_id=14,
    )
    return db


class TestE2EPipeline:
    async def test_message_insertion(self, populated_db):
        assert await count_messages(populated_db, SESSION) == 14
        assert await total_tokens(populated_db, SESSION) > 0

    async def test_status_before_compaction(self, populated_db):
        status = await lcm_status(populated_db, SESSION)
        assert status["message_count"] == 14
        assert status["summary_count"] == 0
        assert status["dag_depth"] == 0

    async def test_fts_search(self, populated_db):
        result = await lcm_grep(populated_db, "JWT", session_id=SESSION)
        messages = [m for group in result["results"] for m in group["messages"]]
        assert len(messages) >= 2
        assert any("JWT" in m["content"] for m in messages)

    async def test_regex_search(self, populated_db):
        result = await lcm_grep(
            populated_db, r"test_\w+", session_id=SESSION, use_regex=True
        )
        messages = [m for group in result["results"] for m in group["messages"]]
        assert len(messages) >= 1

    async def test_status_after_compaction(self, summarized_db):
        status = await lcm_status(summarized_db, SESSION)
        assert status["message_count"] == 14
        assert status["summary_count"] == 2
        assert len(status["top_level_summaries"]) == 2

    async def test_expand_summary(self, summarized_db):
        result = await lcm_expand(summarized_db, 1)
        assert result["total_messages"] == 8
        assert len(result["messages"]) == 8
        assert result["summary"]["id"] == "S1"

    async def test_describe_summary(self, summarized_db):
        result = await lcm_describe(summarized_db, "S2")
        assert result["type"] == "summary"
        assert result["level"] == 0
        assert "rate limiting" in result["content"]

    async def test_describe_message(self, summarized_db):
        result = await lcm_describe(summarized_db, "1")
        assert result["type"] == "message"
        assert result["role"] == "user"

    async def test_grep_within_summary(self, summarized_db):
        result = await lcm_grep(summarized_db, "rate limiting", summary_id=2)
        messages = [m for group in result["results"] for m in group["messages"]]
        assert len(messages) >= 1
        assert any("rate limiting" in m["content"] for m in messages)

    async def test_injection_text(self, summarized_db):
        text = await build_injection_text(summarized_db, SESSION)
        assert "LCM Context Recovery" in text
        assert "lcm_expand" in text
        assert "S1" in text
        assert "S2" in text
        assert "FastAPI" in text
        assert "rate limiting" in text
