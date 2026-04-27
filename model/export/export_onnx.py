from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from model.training.common import build_model, load_checkpoint
from model.training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a training checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the saved training checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = load_config(None)
    config = replace(
        config,
        model_name=checkpoint["model_name"],
        pretrained=False,
        freeze_backbone=False,
        input_size=checkpoint["input_size"],
        dropout=checkpoint["dropout"],
        class_names=checkpoint["class_names"],
    )

    model = build_model(config, len(config.class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, config.input_size, config.input_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
    )
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
