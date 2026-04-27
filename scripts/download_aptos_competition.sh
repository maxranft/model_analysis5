#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="${KAGGLE_CONFIG_DIR:-$ROOT_DIR/.kaggle}"
TARGET_DIR="${1:-$ROOT_DIR/data/raw/aptos2019}"
ARCHIVE_PATH="$TARGET_DIR/aptos2019-blindness-detection.zip"

mkdir -p "$KAGGLE_DIR" "$TARGET_DIR"

if [[ ! -f "$KAGGLE_DIR/kaggle.json" && ! -f "$HOME/.kaggle/kaggle.json" && -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "Kaggle credentials not found."
  echo "Put kaggle.json in $KAGGLE_DIR or ~/.kaggle/, or export KAGGLE_API_TOKEN."
  exit 1
fi

export KAGGLE_CONFIG_DIR="$KAGGLE_DIR"
"$ROOT_DIR/.venv/bin/kaggle" competitions download \
  -c aptos2019-blindness-detection \
  -p "$TARGET_DIR"

if [[ -f "$ARCHIVE_PATH" ]]; then
  unzip -o "$ARCHIVE_PATH" -d "$TARGET_DIR"
fi

echo "APTOS data downloaded to $TARGET_DIR"
