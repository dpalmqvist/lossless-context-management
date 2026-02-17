# Lossless Context Management for Claude Code

MCP server + hooks augmentation layer providing lossless memory retrieval, automatic context preservation across compaction events, and parallel data processing operators.

Based on the paper [*Lossless Context Management for Long-Horizon Code Agents*](https://arxiv.org/abs/2506.18655) by Ehrlich & Blackman (2025), which describes a deterministic architecture that outperforms Claude Code on long-context benchmarks by maintaining an immutable message store, hierarchical summary DAG, and operator-level recursion tools.

## Features

- **Immutable message store** — SQLite + FTS5 full-text search, append-only
- **Hierarchical summary DAG** — Three-level escalation compaction (preserve_details, bullet_points, deterministic truncation)
- **Automatic context recovery** — Hooks capture messages and inject summaries after compaction
- **MCP tools** — `lcm_status`, `lcm_grep`, `lcm_describe`, `lcm_expand` for memory retrieval
- **Parallel data operators** — `llm_map` and `agentic_map` for processing JSONL files with concurrent Haiku calls

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- `ANTHROPIC_API_KEY` environment variable (for summarization and operators)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dpalmqvist/lossless-context-management.git
cd lossless-context-management
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Register the MCP server with Claude Code

```bash
claude mcp add --transport stdio lcm -- \
  uv --directory /path/to/lossless-context-management run python -m lcm.server
```

Replace `/path/to/lossless-context-management` with the actual path where you cloned the repo.

### 4. Configure hooks

Add the following to your project's `.claude/settings.json` (or merge into an existing one):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/lossless-context-management/hooks/session_start.sh",
            "timeout": 10
          }
        ]
      },
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"type\":\"compact\"}' | /path/to/lossless-context-management/hooks/session_start.sh",
            "timeout": 30
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Edit|Write|Read",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/lossless-context-management/hooks/post_tool_use.sh",
            "timeout": 10,
            "async": true
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/lossless-context-management/hooks/user_prompt_submit.sh",
            "timeout": 10,
            "async": true
          }
        ]
      }
    ]
  }
}
```

Replace `/path/to/lossless-context-management` with the actual path.

### 5. Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

Once installed, the LCM tools are available in Claude Code:

| Tool | Description |
|------|-------------|
| `lcm_status` | Session stats: message count, summary count, tokens, DAG depth |
| `lcm_grep` | Search message history with FTS5 or regex, grouped by covering summary |
| `lcm_describe` | Metadata lookup for any LCM ID (message, summary, or file) |
| `lcm_expand` | Expand a summary to its constituent messages |
| `llm_map` | Process JSONL items in parallel with stateless Haiku calls |
| `agentic_map` | Process JSONL items with multi-turn agent loops |

### CLI

```bash
# Check session status
uv run lcm status [session_id]

# Manual hook operations
uv run lcm hook capture
uv run lcm hook inject
uv run lcm hook init
```

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests (75 tests, ~0.5s)
uv run pytest tests/ -v

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv --directory . run python -m lcm.server
```

## How It Works

1. **Hooks** automatically capture messages from Claude Code's transcript into an append-only SQLite store
2. When token count exceeds **τ_soft** (50K tokens), oldest unsummarized messages are grouped into blocks and summarized into **leaf nodes** in the DAG
3. When 5+ uncondensed leaf nodes accumulate, they are **condensed** into higher-level summary nodes
4. At **τ_hard** (200K tokens), all unsummarized messages are compacted
5. After compaction, the **SessionStart hook** injects a context recovery block with top-level summaries and LCM IDs
6. Claude Code can then use `lcm_expand` and `lcm_grep` to drill into any part of the conversation history

## Reference

> Ehrlich, N., & Blackman, S. (2025). *Lossless Context Management for Long-Horizon Code Agents*. arXiv:2506.18655. https://arxiv.org/abs/2506.18655

## License

MIT
