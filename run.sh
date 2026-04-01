#!/bin/bash
#
# keepgoing runner — keeps the agent alive across crashes and restarts.
# Usage: ./run.sh [-task "your goal here"]
#
# The model server must be started separately:
#   python scripts/start_server.py                    # With TurboQuant KV cache compression
#   python scripts/start_server.py --no-turboquant    # Standard KV cache
#   mlx_lm.server --model nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx --port 8000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Build once
echo "[$(date)] Building keepgoing..."
go build -o keepgoing .

# Check if model server is running
check_model_server() {
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null || echo "000"
}

wait_for_model_server() {
    echo "[$(date)] Waiting for model server at http://localhost:8000..."
    for i in $(seq 1 30); do
        if [ "$(check_model_server)" = "200" ]; then
            echo "[$(date)] Model server is ready."
            return 0
        fi
        sleep 2
    done
    echo "[$(date)] WARNING: Model server not responding. Agent will retry on each call."
}

# Check server on startup
if [ "$(check_model_server)" != "200" ]; then
    echo "[$(date)] Model server not detected."
    echo "[$(date)] Start it with: python scripts/start_server.py"
    echo "[$(date)]   (or: mlx_lm.server --model nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx --port 8000)"
    wait_for_model_server
fi

# Run agent in a resilient loop
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=10

while true; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Starting agent..."
    echo "=========================================="

    ./keepgoing "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Agent completed successfully."
        break
    fi

    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))

    if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
        echo "[$(date)] Too many consecutive failures ($CONSECUTIVE_FAILURES). Stopping."
        exit 1
    fi

    # Exponential backoff: 5s, 10s, 20s, 40s... capped at 300s
    BACKOFF=$((5 * (2 ** (CONSECUTIVE_FAILURES - 1))))
    if [ $BACKOFF -gt 300 ]; then
        BACKOFF=300
    fi

    echo "[$(date)] Agent exited with code $EXIT_CODE (failure $CONSECUTIVE_FAILURES/$MAX_CONSECUTIVE_FAILURES)."
    echo "[$(date)] Restarting in ${BACKOFF}s..."
    sleep $BACKOFF

    # Reset failure count if model server is healthy
    if [ "$(check_model_server)" = "200" ]; then
        CONSECUTIVE_FAILURES=0
    fi
done
