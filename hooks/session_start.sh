#!/usr/bin/env bash
# Hook: SessionStart — initialize LCM or inject summaries after compaction
# Reads JSON from stdin with session info

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Read stdin (Claude Code passes JSON with hook context)
INPUT=$(cat)

# Extract session ID if available
export CLAUDE_SESSION_ID="${CLAUDE_SESSION_ID:-default}"

# Check if this is a compaction recovery
IS_COMPACT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Check if this is a post-compaction session start
    print('true' if data.get('type') == 'compact' else 'false')
except:
    print('false')
" 2>/dev/null || echo "false")

if [ "$IS_COMPACT" = "true" ]; then
    # Inject summaries after compaction
    uv --directory "$PROJECT_DIR" run lcm hook inject
else
    # Normal session start — initialize
    uv --directory "$PROJECT_DIR" run lcm hook init 2>/dev/null || true
fi
