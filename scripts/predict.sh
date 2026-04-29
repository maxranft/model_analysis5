#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-}"
SYMPTOMS="${2:-}"
HOST="${MEDIMG_HOST:-127.0.0.1}"
PORT="${MEDIMG_PORT:-8001}"
BASE="http://$HOST:$PORT"

if [[ -z "$IMAGE" ]]; then
  echo "Usage: scripts/predict.sh <image_path> [symptoms]"
  echo "  image_path   path to a PNG, JPEG, WebP, or TIFF file"
  echo "  symptoms     optional symptom description in quotes"
  exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
  echo "Error: file not found: $IMAGE"
  exit 1
fi

# Check server is up
if ! curl -sf "$BASE/health" >/dev/null 2>&1; then
  echo "Error: server is not reachable at $BASE"
  echo "Run: bash scripts/start_project.sh"
  exit 1
fi

echo ""
echo "Image   : $IMAGE"
[[ -n "$SYMPTOMS" ]] && echo "Symptoms: $SYMPTOMS"
echo ""

ARGS=(-s -X POST "$BASE/triage" -F "image=@$IMAGE")
[[ -n "$SYMPTOMS" ]] && ARGS+=(-F "symptoms=$SYMPTOMS")

RESPONSE=$(curl "${ARGS[@]}")

# Parse with python and print cleanly
python3 - "$RESPONSE" <<'PYEOF'
import sys, json

raw = sys.argv[1]
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("Error: unexpected response from server")
    print(raw)
    sys.exit(1)

if "detail" in data:
    print(f"Error: {data['detail']}")
    sys.exit(1)

label_map = {
    "no_dr":           "No Diabetic Retinopathy",
    "mild_dr":         "Mild DR",
    "moderate_dr":     "Moderate DR",
    "severe_dr":       "Severe DR",
    "proliferative_dr":"Proliferative DR",
}
triage_map = {
    "routine_review": "Routine review",
    "refer_for_review": "Refer for review",
    "retake_image":   "Retake image",
}

triage  = triage_map.get(data["triage_label"], data["triage_label"])
conf    = data["confidence"] * 100
version = data["model_version"]
req_id  = data["request_id"]

print(f"  Recommendation : {triage}")
print(f"  Confidence     : {conf:.1f}%")
print("")
print("  Top findings:")
for i, f in enumerate(data["top_findings"], 1):
    name  = label_map.get(f["label"], f["label"])
    score = f["score"] * 100
    bar   = "█" * int(score / 5)
    print(f"    {i}. {name:<28} {score:5.1f}%  {bar}")
print("")
print(f"  Model   : {version}")
print(f"  Request : {req_id}")
print("")
print(f"  ⚠  {data['disclaimer']}")
print("")
PYEOF
