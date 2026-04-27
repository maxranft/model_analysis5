#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${1:-$ROOT_DIR/data/raw/aptos2019}"
META_DIR="$TRAIN_ROOT/meta"
IMG_DIR="$TRAIN_ROOT/train_images"
ZIP_DIR="$TRAIN_ROOT/train_image_zips"
TMP_LIST="$TRAIN_ROOT/.train_ids.txt"
JOBS="${APTOS_DOWNLOAD_JOBS:-6}"

mkdir -p "$META_DIR" "$IMG_DIR" "$ZIP_DIR"

if [[ ! -f "$META_DIR/train.csv" ]]; then
  "$ROOT_DIR/scripts/download_aptos_competition.sh" "$TRAIN_ROOT"
fi

if [[ ! -f "$META_DIR/train.csv" ]]; then
  echo "Missing $META_DIR/train.csv"
  exit 1
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "$ROOT_DIR/.kaggle/kaggle.json" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Kaggle credentials not found."
  exit 1
fi

tail -n +2 "$META_DIR/train.csv" | cut -d, -f1 > "$TMP_LIST"

download_one() {
  local id="$1"
  local zip_path="$ZIP_DIR/${id}.png.zip"
  local raw_path="$ZIP_DIR/${id}.png"
  local image_path="$IMG_DIR/${id}.png"

  if [[ -f "$image_path" ]]; then
    return 0
  fi

  "$ROOT_DIR/.venv/bin/kaggle" competitions download \
    -c aptos2019-blindness-detection \
    -f "train_images/${id}.png" \
    -p "$ZIP_DIR" \
    -o \
    -q

  if [[ -f "$zip_path" ]]; then
    unzip -o -q "$zip_path" -d "$IMG_DIR"
  elif [[ -f "$raw_path" ]]; then
    mv -f "$raw_path" "$image_path"
  else
    echo "Missing downloaded artifact for $id" >&2
    return 1
  fi
}

export ROOT_DIR IMG_DIR ZIP_DIR
export -f download_one

xargs -P "$JOBS" -n 1 -I {} bash -lc 'download_one "$@"' _ {} < "$TMP_LIST"

echo "Downloaded training images into $IMG_DIR"
