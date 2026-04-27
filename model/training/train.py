from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch
from tqdm import tqdm

from model.training.common import (
    build_model,
    checkpoint_payload,
    confusion_matrix,
    load_checkpoint,
    per_class_recall,
    quadratic_weighted_kappa,
    save_checkpoint,
    save_metrics,
    select_device,
    set_seed,
    unfreeze_backbone,
)
from model.training.config import TrainingConfig, load_config, save_config_template
from model.training.dataset import build_loaders, class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet on the APTOS blindness dataset.")
    parser.add_argument("--config", type=Path, help="Path to a JSON training config.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--save-config-template", type=Path, help="Write a config template and exit.")
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    matrix = confusion_matrix(all_predictions, all_targets, num_classes)
    qwk = quadratic_weighted_kappa(all_predictions, all_targets, num_classes)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "qwk": qwk,
        "confusion_matrix": matrix,
        "per_class_recall": per_class_recall(matrix),
    }


def build_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
    backbone_trainable: bool,
) -> torch.optim.Optimizer:
    if backbone_trainable:
        parameter_groups = [
            {"params": model.features.parameters(), "lr": config.backbone_learning_rate},
            {"params": model.classifier.parameters(), "lr": config.learning_rate},
        ]
    else:
        parameter_groups = [{"params": model.classifier.parameters(), "lr": config.learning_rate}]
    return torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)


def main() -> None:
    args = parse_args()
    if args.save_config_template:
        save_config_template(args.save_config_template)
        return

    config = load_config(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.learning_rate is not None:
        config = replace(config, learning_rate=args.learning_rate)

    set_seed(config.seed)
    device = select_device(config.device)
    train_loader, val_loader, data_stats = build_loaders(config)
    model = build_model(config, num_classes=len(config.class_names)).to(device)

    criterion_kwargs = {"label_smoothing": config.label_smoothing}
    if not config.use_weighted_sampler:
        criterion_kwargs["weight"] = class_weights(train_loader.dataset.records, len(config.class_names)).to(device)
    criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
    optimizer = build_optimizer(model, config, backbone_trainable=not config.freeze_backbone)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_qwk = float("-inf")
    best_payload: dict[str, object] | None = None
    best_epoch = -1
    epochs_since_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        if config.freeze_backbone and epoch == config.unfreeze_at_epoch:
            unfreeze_backbone(model)
            optimizer = build_optimizer(model, config, backbone_trainable=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, len(config.class_names))
        scheduler.step(float(val_metrics["qwk"]))

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(epoch_metrics)

        if float(val_metrics["qwk"]) > best_qwk:
            best_qwk = float(val_metrics["qwk"])
            best_epoch = epoch
            epochs_since_improvement = 0
            best_payload = checkpoint_payload(model, config, best_qwk, {"epoch": epoch, **val_metrics})
            save_checkpoint(best_payload, config.checkpoint_path)
        else:
            epochs_since_improvement += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={float(val_metrics['loss']):.4f} "
            f"val_acc={float(val_metrics['accuracy']):.4f} "
            f"val_qwk={float(val_metrics['qwk']):.4f}"
        )

        if epoch >= config.min_epochs and epochs_since_improvement >= config.early_stopping_patience:
            break

    elapsed = time.perf_counter() - started
    summary = {
        "config": config.to_dict(),
        "device": str(device),
        "data": data_stats,
        "best_epoch": best_epoch,
        "best_qwk": best_qwk,
        "checkpoint_path": str(config.checkpoint_path),
        "elapsed_seconds": elapsed,
        "history": history,
    }
    save_metrics(summary, config.metrics_path)

    if best_payload is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    checkpoint = load_checkpoint(config.checkpoint_path, map_location="cpu")
    print(f"saved checkpoint: {config.checkpoint_path}")
    print(f"best_qwk: {checkpoint['best_qwk']:.4f}")


if __name__ == "__main__":
    main()
