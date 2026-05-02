#!/usr/bin/env bash
# Download the full APTOS 2019 training image set in one zip.
#
# Requirements:
#   1. Place your Kaggle token at ~/.kaggle/kaggle.json
#      Get it from: kaggle.com → Settings → API → Create New Token
#   2. Accept competition rules at:
#      https://www.kaggle.com/c/aptos2019-blindness-detection
#
# Usage:
#   bash scripts/download_aptos_train_images.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="$ROOT_DIR/data/raw/aptos2019"
META_DIR="$TRAIN_ROOT/meta"
IMG_DIR="$TRAIN_ROOT/train_images"
TMP_DIR="$TRAIN_ROOT/.download_tmp"
KAGGLE="$ROOT_DIR/.venv/bin/kaggle"
COMPETITION="aptos2019-blindness-detection"

mkdir -p "$META_DIR" "$IMG_DIR" "$TMP_DIR"

# ── Check credentials ─────────────────────────────────────────────────────────
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo ""
  echo "  ERROR: Kaggle token not found."
  echo ""
  echo "  Fix:"
  echo "    1. Go to kaggle.com → Settings → API → Create New Token"
  echo "    2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
  echo "    3. chmod 600 ~/.kaggle/kaggle.json"
  echo "    4. Re-run this script"
  echo ""
  exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"

# ── Download metadata CSV if missing ─────────────────────────────────────────
if [[ ! -f "$META_DIR/train.csv" ]]; then
  echo "Downloading train.csv..."
  "$KAGGLE" competitions download \
    -c "$COMPETITION" \
    -f train.csv \
    -p "$TMP_DIR" \
    --quiet
  if [[ -f "$TMP_DIR/train.csv.zip" ]]; then
    unzip -o -q "$TMP_DIR/train.csv.zip" -d "$META_DIR"
  else
    mv "$TMP_DIR/train.csv" "$META_DIR/train.csv"
  fi
  echo "  Saved → $META_DIR/train.csv"
fi

# ── Count already-downloaded images ──────────────────────────────────────────
TOTAL=$(tail -n +2 "$META_DIR/train.csv" | wc -l | tr -d ' ')
EXISTING=$(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "  Training images : $EXISTING / $TOTAL already on disk"
echo ""

if [[ "$EXISTING" -ge "$TOTAL" ]]; then
  echo "  All images already downloaded. Nothing to do."
  exit 0
fi

# ── Download full competition dataset ────────────────────────────────────────
ZIP_PATH="$TMP_DIR/${COMPETITION}.zip"

echo "  Downloading full dataset (~11 GB — this will take a while)..."
echo ""

"$KAGGLE" competitions download \
  -c "$COMPETITION" \
  -p "$TMP_DIR"

# ── Unzip ─────────────────────────────────────────────────────────────────────
echo ""
echo "  Extracting images..."
unzip -o -q "$ZIP_PATH" -d "$TRAIN_ROOT"

# move train_images into expected location if nested
if [[ -d "$TRAIN_ROOT/train_images" ]]; then
  : # already in the right place
fi

# ── Clean up zip ──────────────────────────────────────────────────────────────
rm -f "$ZIP_PATH"
rmdir "$TMP_DIR" 2>/dev/null || true

FINAL=$(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "  Done — $FINAL images in $IMG_DIR"
