#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
DEFAULT_PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "$ROOT_DIR/backend/data" "$ROOT_DIR/model/artifacts" "$ROOT_DIR/data/raw/aptos2019"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python virtualenv is missing or incomplete at $VENV_DIR" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, PIL, numpy" >/dev/null 2>&1; then
  "$PIP_BIN" install -r "$ROOT_DIR/backend/requirements.txt"
fi

if [[ -z "${MEDIMG_ENABLE_MOCK_MODEL:-}" ]]; then
  if [[ -f "${MEDIMG_MODEL_PATH:-$ROOT_DIR/model/artifacts/model.onnx}" ]]; then
    export MEDIMG_ENABLE_MOCK_MODEL=0
  else
    export MEDIMG_ENABLE_MOCK_MODEL=1
  fi
fi

choose_port() {
  local port="$1"
  while lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do
    port=$((port + 1))
  done
  echo "$port"
}

PORT_TO_USE="$(choose_port "$DEFAULT_PORT")"

echo "Starting Medical Imaging AI MVP"
echo "Host: $HOST"
echo "Port: $PORT_TO_USE"
echo "Mock model: $MEDIMG_ENABLE_MOCK_MODEL"
echo "Docs: http://$HOST:$PORT_TO_USE/docs"

exec "$PYTHON_BIN" -m uvicorn app.main:app \
  --app-dir "$ROOT_DIR/backend" \
  --host "$HOST" \
  --port "$PORT_TO_USE" \
  --reload
