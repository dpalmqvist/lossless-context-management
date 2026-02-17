#!/usr/bin/env bash
# Hook: PostToolUse — capture messages after tool execution
# Runs asynchronously to avoid blocking Claude Code

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export CLAUDE_SESSION_ID="${CLAUDE_SESSION_ID:-default}"

# Run capture (hook is already async in settings.json)
uv --directory "$PROJECT_DIR" run lcm hook capture
