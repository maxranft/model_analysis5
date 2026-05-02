#!/usr/bin/env bash
# Roll back model.onnx to any previously exported version.
#
# Usage:
#   bash scripts/rollback.sh          # interactive menu
#   bash scripts/rollback.sh 3        # roll back to version #3 from the list
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_DIR="$ROOT/model/artifacts/versions"
ACTIVE="$ROOT/model/artifacts/model.onnx"

if [[ ! -d "$VERSIONS_DIR" ]] || [[ -z "$(ls "$VERSIONS_DIR"/*.onnx 2>/dev/null)" ]]; then
    echo "  No saved versions found in $VERSIONS_DIR"
    echo "  Versions are created automatically each time you export a model."
    exit 1
fi

echo ""
echo "  Available model versions:"
echo "  ─────────────────────────────────────────────────"

mapfile -t VERSIONS < <(ls -t "$VERSIONS_DIR"/*.onnx)
for i in "${!VERSIONS[@]}"; do
    name="$(basename "${VERSIONS[$i]}")"
    size="$(du -sh "${VERSIONS[$i]}" | cut -f1)"
    marker=""
    if cmp -s "${VERSIONS[$i]}" "$ACTIVE" 2>/dev/null; then
        marker="  ← active"
    fi
    printf "  [%d] %s  (%s)%s\n" "$((i+1))" "$name" "$size" "$marker"
done

echo ""

if [[ -n "${1:-}" ]]; then
    CHOICE="$1"
else
    read -rp "  Enter number to restore (or q to quit): " CHOICE
fi

if [[ "$CHOICE" == "q" ]]; then
    exit 0
fi

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || (( CHOICE < 1 || CHOICE > ${#VERSIONS[@]} )); then
    echo "  Invalid choice." >&2
    exit 1
fi

SELECTED="${VERSIONS[$((CHOICE-1))]}"
echo ""
echo "  Restoring → $(basename "$SELECTED")"
cp "$SELECTED" "$ACTIVE"

echo "  Restarting server..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2
"$ROOT/.venv/bin/python" -m uvicorn app.main:app \
    --app-dir "$ROOT/backend" \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level warning &
sleep 3
echo "  Done. Server running with $(basename "$SELECTED")"
echo ""
