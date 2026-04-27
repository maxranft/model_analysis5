from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, efficientnet_b0, efficientnet_b1

from model.training.config import TrainingConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: TrainingConfig, num_classes: int) -> torch.nn.Module:
    weights = None
    if config.pretrained:
        if config.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
        elif config.model_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.DEFAULT
        else:
            raise ValueError(f"Unsupported model_name: {config.model_name}")

    try:
        if config.model_name == "efficientnet_b0":
            model = efficientnet_b0(weights=weights)
        elif config.model_name == "efficientnet_b1":
            model = efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {config.model_name}")
    except Exception as exc:
        if not config.pretrained:
            raise exc
        print(f"warning: failed to load pretrained weights ({exc}); falling back to random initialization")
        fallback = replace(config, pretrained=False)
        return build_model(fallback, num_classes)

    classifier_in = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=config.dropout),
        torch.nn.Linear(classifier_in, num_classes),
    )

    if config.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def unfreeze_backbone(model: torch.nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = True


def confusion_matrix(predictions: list[int], targets: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def quadratic_weighted_kappa(predictions: list[int], targets: list[int], num_classes: int) -> float:
    if not predictions:
        return 0.0

    observed = np.zeros((num_classes, num_classes), dtype=np.float64)
    for target, prediction in zip(targets, predictions):
        observed[target, prediction] += 1

    target_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(target_hist, pred_hist) / observed.sum()

    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    denom = max((num_classes - 1) ** 2, 1)
    for row in range(num_classes):
        for col in range(num_classes):
            weights[row, col] = ((row - col) ** 2) / denom

    observed_score = np.sum(weights * observed) / observed.sum()
    expected_score = np.sum(weights * expected) / expected.sum()
    if expected_score == 0:
        return 1.0
    return float(1.0 - (observed_score / expected_score))


def per_class_recall(matrix: list[list[int]]) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for index, row in enumerate(matrix):
        total = sum(row)
        recalls[str(index)] = float(row[index] / total) if total else 0.0
    return recalls


def checkpoint_payload(
    model: torch.nn.Module,
    config: TrainingConfig,
    best_metric: float,
    extra_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_name": config.model_name,
        "class_names": config.class_names,
        "input_size": config.input_size,
        "dropout": config.dropout,
        "mean": config.mean,
        "std": config.std,
        "best_qwk": best_metric,
        "config": config.to_dict(),
        "metrics": extra_metrics,
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
