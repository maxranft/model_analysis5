#!/usr/bin/env python3

#!Run with: open data/feedback/accuracy_over_time.png


#!Demo script: python scripts/accuracy_plot.py --demo --save

"""
Plot model accuracy and training history.

Usage:
  python scripts/accuracy_plot.py            # all charts, one point per session
  python scripts/accuracy_plot.py --daily    # group by calendar day
  python scripts/accuracy_plot.py --save     # save PNGs instead of displaying

Charts produced:
  accuracy_over_time.png  — review session accuracy over runs
  confusion_matrix.png    — predicted vs actual DR grade
  per_grade_accuracy.png  — accuracy broken down by grade
  training_history.png    — epoch-level val QWK + best QWK per training run
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
FEEDBACK_CSV   = ROOT / "data/feedback/corrections.csv"
METRICS_JSON   = ROOT / "model/artifacts/checkpoints/metrics.json"
EXPERIMENT_LOG = ROOT / "model/artifacts/experiment_log.csv"
OUTPUT_DIR     = ROOT / "data/feedback"

GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
GRADE_KEYS   = ["no_dr", "mild_dr", "moderate_dr", "severe_dr", "proliferative_dr"]


def load_records() -> list[dict]:
    if not FEEDBACK_CSV.exists():
        print(f"No feedback data found at {FEEDBACK_CSV.relative_to(ROOT)}")
        print("Run  python scripts/review.py  first to generate feedback.")
        sys.exit(1)
    with FEEDBACK_CSV.open() as f:
        return list(csv.DictReader(f))


def group_by_session(records: list[dict]) -> list[tuple[datetime, float, int, float]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r["timestamp"]].append(r)
    points = []
    for ts_str, rows in sorted(buckets.items()):
        ts = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
        correct = sum(1 for r in rows if r["correct"].strip().lower() == "true")
        acc = correct / len(rows) * 100
        avg_conf = sum(float(r["confidence"]) for r in rows) / len(rows) * 100
        points.append((ts, acc, len(rows), avg_conf))
    return points


def group_by_day(records: list[dict]) -> list[tuple[datetime, float, int, float]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        day = datetime.fromisoformat(r["timestamp"]).strftime("%Y-%m-%d")
        buckets[day].append(r)
    points = []
    for day_str, rows in sorted(buckets.items()):
        ts = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        correct = sum(1 for r in rows if r["correct"].strip().lower() == "true")
        acc = correct / len(rows) * 100
        avg_conf = sum(float(r["confidence"]) for r in rows) / len(rows) * 100
        points.append((ts, acc, len(rows), avg_conf))
    return points


def rolling_avg(values: list[float], window: int = 3) -> list[float | None]:
    out: list[float | None] = []
    for i, _ in enumerate(values):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(sum(values[i + 1 - window : i + 1]) / window)
    return out


def plot_accuracy_over_time(
    points: list[tuple[datetime, float, int, float]], mode: str, save: bool
) -> None:
    import matplotlib.pyplot as plt

    accuracies  = [p[1] for p in points]
    counts      = [p[2] for p in points]
    confidences = [p[3] for p in points]
    reps        = list(range(1, len(points) + 1))

    roll   = rolling_avg(accuracies, window=min(3, len(accuracies)))
    roll_x = [reps[i] for i, v in enumerate(roll) if v is not None]
    roll_y = [v for v in roll if v is not None]

    tick_labels = [f"Run {i}" for i in reps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Model Accuracy by Review Run", fontsize=14, fontweight="bold")

    bubble_sizes = [max(c * 4, 30) for c in counts]
    ax1.scatter(reps, accuracies, s=bubble_sizes, color="#4C72B0", alpha=0.8,
                zorder=3, label="Accuracy (bubble ∝ sample size)")
    ax1.plot(reps, accuracies, color="#4C72B0", alpha=0.3, linewidth=1)
    if roll_y:
        ax1.plot(roll_x, roll_y, color="#DD8452", linewidth=2,
                 label=f"{min(3, len(accuracies))}-run rolling avg")
    ax1.axhline(y=sum(accuracies) / len(accuracies), color="gray",
                linestyle="--", linewidth=1, alpha=0.6, label="Overall avg")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    for rep, acc, n in zip(reps, accuracies, counts):
        ax1.annotate(f"{acc:.0f}%\n(n={n})", xy=(rep, acc),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=7, color="#333333")

    ax2.plot(reps, confidences, color="#55A868", linewidth=2, marker="o",
             markersize=5, label="Avg confidence")
    ax2.set_ylabel("Avg Confidence (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xlabel("Review Run")
    ax2.set_xticks(reps)
    ax2.set_xticklabels(tick_labels, fontsize=8)

    plt.tight_layout()
    _finish(fig, "accuracy_over_time.png", save)


def plot_confusion_matrix(records: list[dict], save: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n = 5
    matrix = [[0] * n for _ in range(n)]
    for r in records:
        actual    = int(r["confirmed_grade"])
        predicted = int(r["predicted_grade"])
        if 0 <= actual < n and 0 <= predicted < n:
            matrix[actual][predicted] += 1

    mat = np.array(matrix, dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Confusion Matrix\n(row-normalised — shade = % of actual grade predicted as each class)",
                 fontsize=11, fontweight="bold")

    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of actual grade")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(GRADE_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(GRADE_LABELS, fontsize=9)
    ax.set_xlabel("Predicted Grade", fontsize=10)
    ax.set_ylabel("Actual Grade", fontsize=10)

    for i in range(n):
        for j in range(n):
            count = int(matrix[i][j])
            pct   = norm[i][j] * 100
            color = "white" if norm[i][j] > 0.55 else "#222222"
            ax.text(j, i, f"{pct:.0f}%\n({count})",
                    ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    _finish(fig, "confusion_matrix.png", save)


def plot_per_grade_accuracy(records: list[dict], save: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    grade_correct = defaultdict(int)
    grade_total   = defaultdict(int)
    for r in records:
        g = int(r["actual_grade"])
        grade_total[g]   += 1
        if r["correct"].strip().lower() == "true":
            grade_correct[g] += 1

    grades    = list(range(5))
    accuracies = [
        grade_correct[g] / grade_total[g] * 100 if grade_total[g] else 0
        for g in grades
    ]
    counts = [grade_total[g] for g in grades]

    cmap   = plt.get_cmap("RdYlGn")
    colors = [cmap(a / 100) for a in accuracies]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Accuracy by DR Grade", fontsize=14, fontweight="bold")

    bars = ax.bar(grades, accuracies, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(grades)
    ax.set_xticklabels(
        [f"Grade {g}\n{GRADE_LABELS[g]}" for g in grades], fontsize=9
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(
        y=sum(1 for r in records if r["correct"].strip().lower() == "true") / len(records) * 100,
        color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Overall avg",
    )
    ax.legend(fontsize=8)

    for bar, acc, n in zip(bars, accuracies, counts):
        label = f"{acc:.0f}%" if n > 0 else "no data"
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 2,
                f"{label}\n(n={n})", ha="center", va="bottom", fontsize=8, color="#222222")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Accuracy %")

    plt.tight_layout()
    _finish(fig, "per_grade_accuracy.png", save)


def plot_training_history(save: bool) -> None:
    import json

    import matplotlib.pyplot as plt

    has_metrics = METRICS_JSON.exists()
    has_log     = EXPERIMENT_LOG.exists()

    if not has_metrics and not has_log:
        print("  No training history found — skipping training history chart.")
        print(f"  (looked for {METRICS_JSON.relative_to(ROOT)} and {EXPERIMENT_LOG.relative_to(ROOT)})")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Training History — QWK", fontsize=14, fontweight="bold")
    ax_epoch, ax_runs = axes

    # Top panel: epoch-by-epoch QWK for the most recent training run
    if has_metrics:
        with METRICS_JSON.open() as f:
            summary = json.load(f)
        history = summary.get("history", [])
        if history:
            epochs   = [e["epoch"] for e in history]
            val_qwk  = [float(e["val"]["qwk"]) for e in history]
            train_acc = [float(e["train"]["accuracy"]) for e in history]
            best_qwk  = max(val_qwk)
            best_ep   = epochs[val_qwk.index(best_qwk)]

            ax_epoch.plot(epochs, val_qwk, color="#4C72B0", linewidth=2,
                          marker="o", markersize=5, label="Val QWK")
            ax_epoch.axhline(y=best_qwk, color="#DD8452", linestyle="--",
                             linewidth=1.2, alpha=0.8,
                             label=f"Best QWK {best_qwk:.4f} (epoch {best_ep})")
            ax_epoch.axvline(x=best_ep, color="#DD8452", linestyle=":",
                             linewidth=1, alpha=0.6)
            ax_epoch.fill_between(epochs, val_qwk, alpha=0.08, color="#4C72B0")
            ax_epoch.set_ylabel("Quadratic Weighted Kappa")
            ax_epoch.set_ylim(0, 1.05)
            ax_epoch.set_xlabel("Epoch")
            ax_epoch.legend(fontsize=9, loc="lower right")
            ax_epoch.grid(axis="y", alpha=0.3)
            ax_epoch.set_title("Latest Run — Epoch-level Val QWK", fontsize=11)
        else:
            ax_epoch.text(0.5, 0.5, "No epoch history in metrics.json",
                          ha="center", va="center", transform=ax_epoch.transAxes)
    else:
        ax_epoch.text(0.5, 0.5, "metrics.json not found",
                      ha="center", va="center", transform=ax_epoch.transAxes)

    # Bottom panel: best QWK per training run from experiment log
    if has_log:
        with EXPERIMENT_LOG.open() as f:
            runs = list(csv.DictReader(f))
        if runs:
            run_nums = list(range(1, len(runs) + 1))
            run_qwk  = [float(r["best_qwk"]) for r in runs]
            run_labels = [f"Run {i}" for i in run_nums]
            colors = ["#DD8452" if q == max(run_qwk) else "#4C72B0" for q in run_qwk]
            bars = ax_runs.bar(run_nums, run_qwk, color=colors,
                               edgecolor="white", linewidth=0.8, zorder=3)
            ax_runs.set_xticks(run_nums)
            ax_runs.set_xticklabels(run_labels, fontsize=8)
            ax_runs.set_ylabel("Best QWK")
            ax_runs.set_ylim(0, 1.05)
            ax_runs.set_xlabel("Training Run")
            ax_runs.grid(axis="y", alpha=0.3, zorder=0)
            ax_runs.set_title("Best QWK per Training Run", fontsize=11)
            for bar, q in zip(bars, run_qwk):
                ax_runs.text(bar.get_x() + bar.get_width() / 2, q + 0.01,
                             f"{q:.4f}", ha="center", va="bottom", fontsize=8)
        else:
            ax_runs.text(0.5, 0.5, "No training runs in experiment_log.csv",
                         ha="center", va="center", transform=ax_runs.transAxes)
    else:
        ax_runs.text(0.5, 0.5,
                     "experiment_log.csv not found\n(appears after next retrain)",
                     ha="center", va="center", transform=ax_runs.transAxes,
                     fontsize=10)

    plt.tight_layout()
    _finish(fig, "training_history.png", save)


def plot_demo_training_curve(save: bool) -> None:
    """Demonstration curve: shows what training looks like when starting from random
    baseline (~49% majority-class accuracy) and improving with a pretrained backbone."""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    epochs = np.arange(1, 26)

    # Sigmoid ramp from 49 → ~84, plus small per-epoch noise
    midpoint = 10
    steepness = 0.35
    trend = 49 + 41 / (1 + np.exp(-steepness * (epochs - midpoint)))
    noise = rng.normal(0, 0.8, size=len(epochs))
    accuracy = np.clip(trend + noise, 0, 100)

    best_idx = int(np.argmax(accuracy))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Training Accuracy",
        fontsize=14, fontweight="bold", color="#333333",
    )

    ax.plot(epochs, accuracy, color="#4C72B0", linewidth=2, marker="o",
            markersize=4, label="Val accuracy")
    ax.fill_between(epochs, accuracy, alpha=0.08, color="#4C72B0")
    ax.axhline(y=49, color="#999999", linestyle="--", linewidth=1.2,
               label="Majority-class baseline (49%)")
    ax.axvline(x=epochs[best_idx], color="#DD8452", linestyle=":",
               linewidth=1.2, alpha=0.7)
    ax.scatter([epochs[best_idx]], [accuracy[best_idx]], color="#DD8452",
               s=80, zorder=5, label=f"Peak {accuracy[best_idx]:.1f}% (epoch {epochs[best_idx]})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim(30, 100)
    ax.set_xlim(0.5, len(epochs) + 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    _finish(fig, "demo_training_curve.png", save)


def _finish(fig, filename: str, save: bool) -> None:
    import matplotlib.pyplot as plt

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / filename
        fig.savefig(out, dpi=150)
        print(f"  Saved → {out.relative_to(ROOT)}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", action="store_true",
                        help="Aggregate by calendar day instead of per session")
    parser.add_argument("--save", action="store_true",
                        help="Save PNGs to data/feedback/ instead of displaying")
    parser.add_argument("--demo", action="store_true",
                        help="Show a demonstration training curve starting from random baseline")
    args = parser.parse_args()

    if args.demo:
        plot_demo_training_curve(args.save)
        return

    records = load_records()
    if args.daily:
        points = group_by_day(records)
        mode = "day"
    else:
        points = group_by_session(records)
        mode = "session"

    if not points:
        print("No records to plot.")
        sys.exit(1)

    total   = sum(r["correct"].strip().lower() == "true" for r in records)
    overall = total / len(records) * 100
    print(f"\n  {len(records)} total records  |  {len(points)} {mode}(s)  |  overall accuracy {overall:.1f}%\n")

    plot_accuracy_over_time(points, mode, args.save)
    plot_confusion_matrix(records, args.save)
    plot_per_grade_accuracy(records, args.save)
    plot_training_history(args.save)


if __name__ == "__main__":
    main()
