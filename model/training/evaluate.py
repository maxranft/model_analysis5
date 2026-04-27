from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from model.training.common import build_model, load_checkpoint, save_metrics, select_device
from model.training.config import load_config
from model.training.dataset import build_loaders
from model.training.train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the APTOS dataset split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a training checkpoint.")
    parser.add_argument("--config", type=Path, help="Optional config override.")
    parser.add_argument("--metrics-out", type=Path, help="Optional metrics JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else load_config(None)

    config = replace(
        config,
        model_name=checkpoint["model_name"],
        class_names=checkpoint["class_names"],
        input_size=checkpoint["input_size"],
        dropout=checkpoint["dropout"],
        mean=checkpoint["mean"],
        std=checkpoint["std"],
    )

    device = select_device(config.device)
    _, val_loader, data_stats = build_loaders(config)
    model = build_model(replace(config, pretrained=False, freeze_backbone=False), len(config.class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, val_loader, criterion, device, len(config.class_names))
    payload = {
        "checkpoint": str(args.checkpoint),
        "data": data_stats,
        "metrics": metrics,
    }
    if args.metrics_out:
        save_metrics(payload, args.metrics_out)
    print(payload)


if __name__ == "__main__":
    main()

