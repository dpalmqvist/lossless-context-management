#!/usr/bin/env bash
# Hook: SessionStart — initialize LCM and inject summaries on resume
# Runs on both startup and resume (matcher: "startup|resume")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Read stdin (Claude Code passes JSON with hook context)
INPUT=$(cat)

export CLAUDE_SESSION_ID="${CLAUDE_SESSION_ID:-default}"

# Initialize session
uv --directory "$PROJECT_DIR" run lcm hook init 2>/dev/null || true

# Attempt summary injection (no-op if nothing to inject)
uv --directory "$PROJECT_DIR" run lcm hook inject 2>/dev/null || true
