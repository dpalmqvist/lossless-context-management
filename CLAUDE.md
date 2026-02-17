# LCM — Lossless Context Management

## What This Is

MCP server + hooks augmentation layer for Claude Code. Provides lossless memory retrieval, automatic context preservation across compaction events, and parallel data processing operators (llm_map, agentic_map).

## Architecture

- **Python MCP server (stdio)** via FastMCP — `src/lcm/server.py`
- **SQLite + FTS5** immutable message store + hierarchical summary DAG — `src/lcm/store/`
- **Three-level compaction** (preserve_details → bullet_points → deterministic truncation) — `src/lcm/compaction/`
- **Anthropic API (Haiku)** for summarization and operators — `src/lcm/llm/client.py`
- **Hooks** for automatic capture and injection — `src/lcm/hooks/` + `hooks/`

## Project Structure

```
src/lcm/
├── server.py          # FastMCP entry point (6 tools registered)
├── cli.py             # CLI for hook scripts (`lcm hook capture|inject|init`)
├── store/             # SQLite storage layer
│   ├── database.py    # Connection, migrations, FTS5, WAL mode
│   ├── messages.py    # Append-only message CRUD + FTS5/regex search
│   ├── summaries.py   # Summary DAG operations (leaf, condensed, traverse)
│   └── files.py       # Large file references
├── compaction/
│   ├── engine.py      # τ_soft/τ_hard control loop, DAG condensation
│   ├── escalation.py  # Three-level summarization escalation
│   └── file_explorer.py  # Type-aware file analysis
├── tools/
│   ├── memory.py      # lcm_grep, lcm_describe, lcm_expand
│   ├── operators.py   # llm_map, agentic_map
│   └── status.py      # lcm_status
├── hooks/
│   ├── capture.py     # Transcript reading + diff
│   └── inject.py      # Summary injection post-compaction
└── llm/
    └── client.py      # Anthropic API wrapper (summarize, classify, agent_loop)
```

## Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run MCP server directly
uv run python -m lcm.server

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv --directory . run python -m lcm.server

# CLI usage
uv run lcm status [session_id]
uv run lcm hook capture
uv run lcm hook inject
uv run lcm hook init
```

## Testing

- 75 tests (65 unit + 10 E2E), all passing
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- Each test gets a fresh SQLite database via `tmp_path` fixture
- LLM calls are mocked in compaction/operator tests
- No `ANTHROPIC_API_KEY` needed for tests

## Key Conventions

- **Append-only store**: Messages are never modified or deleted
- **Token estimates**: ~4 chars per token (rough heuristic in `messages.py:estimate_tokens`)
- **Summary IDs**: Prefixed with `S` (e.g., `S5`), file IDs with `F` (e.g., `F3`), message IDs are plain integers
- **Pagination**: 10 items per page across all tools
- **Compaction thresholds**: τ_soft = 50K tokens (async), τ_hard = 200K tokens (blocking)
- **DAG condensation**: Triggers when 5+ uncondensed leaf summaries exist

## MCP Tools

| Tool | Purpose |
|------|---------|
| `lcm_status` | Session stats, DAG overview |
| `lcm_grep` | FTS5/regex search across message history |
| `lcm_describe` | Metadata lookup by LCM ID |
| `lcm_expand` | Expand summary to constituent messages |
| `llm_map` | Parallel stateless LLM processing of JSONL |
| `agentic_map` | Parallel multi-turn agent processing of JSONL |

## Environment Variables

- `ANTHROPIC_API_KEY` — Required for summarization and operators (not for tests)
- `LCM_DB_PATH` — Override default database path (`~/.lcm/lcm.db`)
- `CLAUDE_SESSION_ID` — Used by hooks to identify the session
