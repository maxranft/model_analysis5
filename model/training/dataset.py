from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from model.training.config import TrainingConfig


@dataclass(slots=True)
class ImageRecord:
    id_code: str
    label: int
    path: Path


class AptosDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, record.label


def load_records(csv_path: Path, image_dir: Path) -> tuple[list[ImageRecord], int]:
    records: list[ImageRecord] = []
    missing = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = image_dir / f"{row['id_code']}.png"
            if not image_path.exists():
                missing += 1
                continue
            records.append(ImageRecord(id_code=row["id_code"], label=int(row["diagnosis"]), path=image_path))
    if not records:
        raise FileNotFoundError(f"No training images found under {image_dir}")
    return records, missing


def stratified_split(records: list[ImageRecord], val_fraction: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    buckets: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        buckets[record.label].append(record)

    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        val_size = max(1, int(len(bucket) * val_fraction))
        val_records.extend(bucket[:val_size])
        train_records.extend(bucket[val_size:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def build_transforms(config: TrainingConfig) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=config.mean, std=config.std)
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.input_size, config.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def class_weights(records: list[ImageRecord], num_classes: int) -> torch.Tensor:
    counts = Counter(record.label for record in records)
    total = sum(counts.values())
    weights = []
    for label in range(num_classes):
        count = counts.get(label, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader, dict[str, int | dict[int, int]]]:
    records, missing = load_records(config.data_csv, config.image_dir)
    train_records, val_records = stratified_split(records, config.val_fraction, config.seed)
    train_transform, eval_transform = build_transforms(config)

    train_dataset = AptosDataset(train_records, train_transform)
    val_dataset = AptosDataset(val_records, eval_transform)

    sampler = None
    if config.use_weighted_sampler:
        weights = class_weights(train_records, len(config.class_names))
        sample_weights = [float(weights[record.label]) for record in train_records]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    stats = {
        "records_total": len(records),
        "records_missing": missing,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "train_class_counts": dict(Counter(record.label for record in train_records)),
        "val_class_counts": dict(Counter(record.label for record in val_records)),
    }
    return train_loader, val_loader, stats

