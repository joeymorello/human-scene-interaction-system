#!/usr/bin/env bash
# Start the HSI backend (FastAPI).
#
# Prefers .venv-ml (Python 3.10 + torch + JOSH deps, set up by
# ml-pipeline/setup_gpu.sh) when present, since that venv can actually run
# the reconstruction pipeline. Falls back to .venv (lightweight backend-only
# deps) for dev work on machines without a GPU or the full ML stack.
#
# Env overrides:
#   HSI_BACKEND_HOST  default 0.0.0.0
#   HSI_BACKEND_PORT  default 8000
#   HSI_VENV          explicit path to a python venv to use (wins over autodetect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HSI_BACKEND_HOST:-0.0.0.0}"
PORT="${HSI_BACKEND_PORT:-8000}"

if [ -n "${HSI_VENV:-}" ]; then
    VENV_PY="$HSI_VENV/bin/python"
elif [ -x "$SCRIPT_DIR/.venv-ml/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv-ml/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
else
    VENV_PY=""
fi

if [ -z "$VENV_PY" ] || [ ! -x "$VENV_PY" ]; then
    echo "No venv found. Either:"
    echo "  - ml-pipeline/setup_gpu.sh         (full ML stack for real pipeline runs)"
    echo "  - uv venv .venv && uv pip install --python .venv/bin/python -r backend/requirements.txt"
    exit 1
fi

echo "Backend using $VENV_PY"

cd "$SCRIPT_DIR/backend"
exec "$VENV_PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
