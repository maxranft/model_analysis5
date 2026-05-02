from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from model.training.common import build_model, load_checkpoint
from model.training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a training checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def export_model(checkpoint_path: Path, output_path: Path, opset: int) -> float:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
        dynamo=False,
    )
    return float(checkpoint.get("best_qwk", 0.0))


def save_versioned_copy(output_path: Path, qwk: float) -> Path:
    versions_dir = output_path.parent / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    versioned = versions_dir / f"{ts}_qwk{qwk:.4f}.onnx"
    shutil.copy2(output_path, versioned)
    return versioned


def main() -> None:
    args = parse_args()
    qwk = export_model(args.checkpoint, args.output, args.opset)
    print(f"exported → {args.output}  (QWK {qwk:.4f})")
    versioned = save_versioned_copy(args.output, qwk)
    print(f"versioned → {versioned.relative_to(args.output.parent.parent)}")


if __name__ == "__main__":
    main()
