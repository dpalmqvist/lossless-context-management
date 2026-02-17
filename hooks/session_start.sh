#!/usr/bin/env bash
# Hook: SessionStart — initialize LCM and inject summaries on resume/compact
# Runs on startup, resume, and compact (matcher: "startup|resume|compact")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Read stdin JSON from Claude Code
INPUT=$(cat)

# Extract session_id and transcript_path from stdin JSON
export CLAUDE_SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','default'))" 2>/dev/null || echo "default")
export CLAUDE_TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript_path',''))" 2>/dev/null || echo "")

# Initialize session
uv --directory "$PROJECT_DIR" run lcm hook init 2>/dev/null || true

# Attempt summary injection (no-op if nothing to inject)
uv --directory "$PROJECT_DIR" run lcm hook inject 2>/dev/null || true
