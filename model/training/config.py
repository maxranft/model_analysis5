from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class TrainingConfig:
    data_csv: Path = field(default_factory=lambda: repo_root() / "data/raw/aptos2019/meta/train.csv")
    image_dir: Path = field(default_factory=lambda: repo_root() / "data/raw/aptos2019/train_images")
    output_dir: Path = field(default_factory=lambda: repo_root() / "model/artifacts/checkpoints")
    metrics_path: Path = field(default_factory=lambda: repo_root() / "model/artifacts/checkpoints/metrics.json")
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_at_epoch: int = 3
    input_size: int = 224
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    label_smoothing: float = 0.0
    early_stopping_patience: int = 4
    min_epochs: int = 3
    use_weighted_sampler: bool = True
    class_names: list[str] = field(default_factory=lambda: ["0", "1", "2", "3", "4"])
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    checkpoint_name: str = "aptos_efficientnet_b0_best.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("data_csv", "image_dir", "output_dir", "metrics_path"):
            data[key] = str(data[key])
        data["checkpoint_path"] = str(self.checkpoint_path)
        return data


def load_config(config_path: str | Path | None = None) -> TrainingConfig:
    config = TrainingConfig()
    if config_path is None:
        return config

    path = Path(config_path)
    payload = json.loads(path.read_text())
    for key in ("data_csv", "image_dir", "output_dir", "metrics_path"):
        if key in payload:
            payload[key] = Path(payload[key])
    return TrainingConfig(**payload)


def save_config_template(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(TrainingConfig().to_dict(), indent=2) + "\n")

