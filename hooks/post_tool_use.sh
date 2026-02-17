#!/usr/bin/env bash
# Hook: PostToolUse — capture messages after tool execution
# Runs asynchronously (async: true in settings.json)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Read stdin JSON from Claude Code
INPUT=$(cat)

# Extract session_id and transcript_path from stdin JSON
export CLAUDE_SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','default'))" 2>/dev/null || echo "default")
export CLAUDE_TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript_path',''))" 2>/dev/null || echo "")

# Run capture (hook is already async in settings.json)
uv --directory "$PROJECT_DIR" run lcm hook capture
