#!/usr/bin/env bash
# Start the HSI backend (FastAPI). Uses the repo-local .venv so the same script
# works on any host without needing a hardcoded system Python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
HOST="${HSI_BACKEND_HOST:-0.0.0.0}"
PORT="${HSI_BACKEND_PORT:-8000}"

if [ ! -x "$VENV_PY" ]; then
    echo "No venv found. Run: uv venv .venv && uv pip install --python .venv/bin/python -r backend/requirements.txt"
    exit 1
fi

cd "$SCRIPT_DIR/backend"
exec "$VENV_PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
