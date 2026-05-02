#!/usr/bin/env bash
# Full improvement loop: grade → retrain → export → measure → plot
#
# Usage:
#   bash scripts/improve.sh               # 50 images, manual grading of wrong ones
#   bash scripts/improve.sh --n 20        # fewer images
#   bash scripts/improve.sh --grade-all   # manually grade every image
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
HOST="http://127.0.0.1:8000"
PORT=8000
SERVER_PID_FILE="$ROOT/.server.pid"

# ── helpers ──────────────────────────────────────────────────────────────────

header() { echo; echo "  ══════════════════════════════════════════════════"; echo "  $1"; echo "  ══════════════════════════════════════════════════"; echo; }
step()   { echo "  ▸ $*"; }

server_running() {
    curl -sf "$HOST/health" > /dev/null 2>&1
}

start_server() {
    step "Starting inference server on :$PORT ..."
    "$PYTHON" -m uvicorn app.main:app \
        --app-dir "$ROOT/backend" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --log-level warning &
    echo $! > "$SERVER_PID_FILE"
    local attempts=0
    until server_running || [ $attempts -ge 20 ]; do
        sleep 1
        attempts=$((attempts + 1))
    done
    if ! server_running; then
        echo "  ERROR: server failed to start" >&2
        exit 1
    fi
    step "Server ready at $HOST"
}

stop_server() {
    if [ -f "$SERVER_PID_FILE" ]; then
        local pid
        pid=$(cat "$SERVER_PID_FILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$SERVER_PID_FILE"
    fi
    # also catch any uvicorn started externally on this port
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    sleep 1
}

# ── main ─────────────────────────────────────────────────────────────────────

header "STEP 1 — Collect Feedback"

if ! server_running; then
    start_server
    STARTED_SERVER=1
else
    step "Server already running at $HOST"
    STARTED_SERVER=0
fi

"$PYTHON" "$ROOT/scripts/review.py" --host "$HOST" "$@"

header "STEP 2 — Retrain"
step "Training with feedback corrections applied..."
cd "$ROOT"
"$PYTHON" -m model.training.train \
    --config "$ROOT/model/training/aptos_efficientnet_b0.json"

header "STEP 3 — Export"
CHECKPOINT="$ROOT/model/artifacts/checkpoints/aptos_efficientnet_b0_best.pt"
ONNX_OUT="$ROOT/model/artifacts/model.onnx"

if [ ! -f "$CHECKPOINT" ]; then
    echo "  ERROR: checkpoint not found at $CHECKPOINT" >&2
    exit 1
fi

step "Exporting to ONNX..."
"$PYTHON" -m model.export.export_onnx \
    --checkpoint "$CHECKPOINT" \
    --output "$ONNX_OUT"
step "Exported → $ONNX_OUT"

header "STEP 4 — Restart Server"
step "Reloading server with new model..."
stop_server
start_server

header "STEP 5 — Measure Accuracy"
step "Running 50 predictions with updated model..."
"$PYTHON" "$ROOT/scripts/review.py" --host "$HOST" --n 50 --no-grade

header "STEP 6 — Plot"
step "Saving charts..."
"$PYTHON" "$ROOT/scripts/accuracy_plot.py" --save
echo
step "Opening charts..."
open "$ROOT/data/feedback/accuracy_over_time.png" \
     "$ROOT/data/feedback/confusion_matrix.png"   \
     "$ROOT/data/feedback/per_grade_accuracy.png"

echo
echo "  Done. Check the charts to see if accuracy improved."
echo
