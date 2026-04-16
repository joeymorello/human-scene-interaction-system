#!/usr/bin/env bash
# Start the HSI frontend (Next.js dev server).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${HSI_FRONTEND_PORT:-3000}"

cd "$SCRIPT_DIR/frontend"
exec npx next dev --port "$PORT" --hostname 0.0.0.0
