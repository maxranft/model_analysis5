#!/usr/bin/env python3
"""
Batch predict and review fundus images against ground truth.

Usage:
  python scripts/review.py          # 50 random images
  python scripts/review.py --n 20   # custom count
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT         = Path(__file__).resolve().parents[1]
TRAIN_CSV    = ROOT / "data/raw/aptos2019/meta/train.csv"
IMG_DIR      = ROOT / "data/raw/aptos2019/train_images"
FEEDBACK_CSV = ROOT / "data/feedback/corrections.csv"

GRADE_TO_LABEL = {
    0: "no_dr",
    1: "mild_dr",
    2: "moderate_dr",
    3: "severe_dr",
    4: "proliferative_dr",
}
LABEL_TO_GRADE = {v: k for k, v in GRADE_TO_LABEL.items()}
DISPLAY = {
    "no_dr":            "No DR           ",
    "mild_dr":          "Mild DR         ",
    "moderate_dr":      "Moderate DR     ",
    "severe_dr":        "Severe DR       ",
    "proliferative_dr": "Proliferative DR",
}


def load_ground_truth() -> dict[str, int]:
    with TRAIN_CSV.open() as f:
        return {row["id_code"]: int(row["diagnosis"]) for row in csv.DictReader(f)}


def available_images(gt: dict[str, int]) -> list[tuple[str, Path, int]]:
    return [
        (id_code, IMG_DIR / f"{id_code}.png", grade)
        for id_code, grade in gt.items()
        if (IMG_DIR / f"{id_code}.png").exists()
    ]


def predict(client: httpx.Client, path: Path) -> dict:
    with path.open("rb") as f:
        resp = client.post("/triage", files={"image": (path.name, f, "image/png")}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def print_detail(n: int, total: int, id_code: str, result: dict, actual_grade: int) -> None:
    pred_label = result["top_findings"][0]["label"]
    actual_label = GRADE_TO_LABEL[actual_grade]
    correct = LABEL_TO_GRADE.get(pred_label) == actual_grade
    marker = "✓" if correct else "✗"

    print(f"\n  [{n}/{total}]  {id_code}  {marker}")
    print(f"  {'Predicted':<10}: {DISPLAY.get(pred_label, pred_label)}  "
          f"{result['confidence']*100:.1f}%  {bar(result['confidence'])}")
    print(f"  {'Actual':<10}: {DISPLAY.get(actual_label, actual_label)}")
    print()
    print("  Grades: 0=No DR  1=Mild  2=Moderate  3=Severe  4=Proliferative")
    print("  [Enter] accept ground truth   [0-4] override   [s] skip")
    print("  > ", end="", flush=True)


def ask_correction(actual_grade: int) -> int | None:
    raw = input().strip().lower()
    if raw == "s":
        return None
    if raw in "01234":
        return int(raw)
    return actual_grade


def save_feedback(records: list[dict]) -> None:
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "id_code",
        "predicted_grade", "predicted_label",
        "actual_grade", "actual_label",
        "confirmed_grade", "confirmed_label",
        "confidence", "correct",
    ]
    write_header = not FEEDBACK_CSV.exists()
    with FEEDBACK_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Images to review (default 50)")
    parser.add_argument("--host", default="http://127.0.0.1:8001")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    gt   = load_ground_truth()
    pool = available_images(gt)

    if not pool:
        print("No images found. Run: bash scripts/download_aptos_train_images.sh")
        sys.exit(1)

    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    total  = len(sample)

    print(f"\n  Batch Review — {total} images")
    print(f"  {'─' * 50}")

    with httpx.Client(base_url=args.host) as client:
        try:
            client.get("/health", timeout=5).raise_for_status()
        except Exception:
            print(f"\n  Server not reachable at {args.host}")
            print("  Run: bash scripts/start_project.sh")
            sys.exit(1)

        # ── Step 1: run all predictions ──────────────────────────
        print("\n  Running predictions...\n")
        results = []
        for i, (id_code, path, actual_grade) in enumerate(sample, 1):
            try:
                result      = predict(client, path)
                pred_label  = result["top_findings"][0]["label"]
                pred_grade  = LABEL_TO_GRADE.get(pred_label, -1)
                correct     = pred_grade == actual_grade
                marker      = "✓" if correct else "✗"
                results.append({
                    "id_code":      id_code,
                    "path":         path,
                    "actual_grade": actual_grade,
                    "pred_grade":   pred_grade,
                    "pred_label":   pred_label,
                    "confidence":   result["confidence"],
                    "result":       result,
                    "correct":      correct,
                })
                print(f"  {marker} [{i:>3}/{total}]  {id_code}  "
                      f"{DISPLAY.get(pred_label,'?')}  {result['confidence']*100:.0f}%")
            except Exception as exc:
                print(f"  ! [{i:>3}/{total}]  {id_code}  error: {exc}")

    # ── Step 2: summary ──────────────────────────────────────────
    n_correct   = sum(1 for r in results if r["correct"])
    n_incorrect = len(results) - n_correct
    print(f"\n  {'─' * 50}")
    print(f"  Accuracy: {n_correct}/{total}  ({n_correct/total*100:.1f}%)")

    grade_errors: dict[str, int] = {}
    for r in results:
        if not r["correct"]:
            key = f"{GRADE_TO_LABEL[r['actual_grade']]} → {r['pred_label']}"
            grade_errors[key] = grade_errors.get(key, 0) + 1
    if grade_errors:
        print("\n  Common mistakes:")
        for pattern, count in sorted(grade_errors.items(), key=lambda x: -x[1]):
            print(f"    {pattern}  (×{count})")

    # ── Step 3: interactive review ───────────────────────────────
    incorrect = [r for r in results if not r["correct"]]
    feedback_records: list[dict] = []
    now = datetime.now(UTC).isoformat()

    if not incorrect:
        print("\n  All predictions matched ground truth — nothing to correct.")
    else:
        print(f"\n  {'─' * 50}")
        print(f"  Reviewing {n_incorrect} incorrect prediction(s).")

        for i, r in enumerate(incorrect, 1):
            print_detail(i, n_incorrect, r["id_code"], r["result"], r["actual_grade"])
            confirmed_grade = ask_correction(r["actual_grade"])
            if confirmed_grade is None:
                continue
            feedback_records.append({
                "timestamp":       now,
                "id_code":         r["id_code"],
                "predicted_grade": r["pred_grade"],
                "predicted_label": r["pred_label"],
                "actual_grade":    r["actual_grade"],
                "actual_label":    GRADE_TO_LABEL[r["actual_grade"]],
                "confirmed_grade": confirmed_grade,
                "confirmed_label": GRADE_TO_LABEL[confirmed_grade],
                "confidence":      r["confidence"],
                "correct":         False,
            })

    # Save correct predictions too — useful as confirmed training signal
    for r in results:
        if r["correct"]:
            label = GRADE_TO_LABEL[r["actual_grade"]]
            feedback_records.append({
                "timestamp":       now,
                "id_code":         r["id_code"],
                "predicted_grade": r["pred_grade"],
                "predicted_label": r["pred_label"],
                "actual_grade":    r["actual_grade"],
                "actual_label":    label,
                "confirmed_grade": r["actual_grade"],
                "confirmed_label": label,
                "confidence":      r["confidence"],
                "correct":         True,
            })

    if feedback_records:
        save_feedback(feedback_records)
        rel = FEEDBACK_CSV.relative_to(ROOT)
        print(f"\n  Feedback saved → {rel}  ({len(feedback_records)} records)")

    if n_incorrect:
        print(f"\n  To retrain with corrections:")
        print(f"    python -m model.training.train")
    print()


if __name__ == "__main__":
    main()
