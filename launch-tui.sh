#!/usr/bin/env bash
# Launch Memory Vault TUI (Linux / macOS)
# Usage: ./launch-tui.sh [additional args]

set -euo pipefail

cd "$(dirname "$0")"

# Activate venv if it exists (prefer .venv, fallback venv)
if [ -d ".venv/bin" ]; then
    source .venv/bin/activate
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Verify installation
if ! python -c "import memory_vault" 2>/dev/null; then
    echo "❌ memory-vault not installed. Run: pip install -e ."
    exit 1
fi

exec python -m memory_vault browse "$@"
