from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _parse_labels(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("normal", "possible_retinopathy", "poor_quality")
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(labels) or ("normal", "possible_retinopathy", "poor_quality")


@dataclass(slots=True)
class Settings:
    app_name: str = "Medical Imaging AI MVP"
    app_version: str = "0.1.0"
    db_path: Path = field(default_factory=lambda: Path(os.getenv("MEDIMG_DB_PATH", "backend/data/app.db")))
    model_path: Path = field(default_factory=lambda: Path(os.getenv("MEDIMG_MODEL_PATH", "model/artifacts/model.onnx")))
    model_version: str = field(default_factory=lambda: os.getenv("MEDIMG_MODEL_VERSION", "efficientnet_b0_fundus_v1"))
    enable_mock_model: bool = field(default_factory=lambda: _bool_env("MEDIMG_ENABLE_MOCK_MODEL", False))
    image_size: int = field(default_factory=lambda: _int_env("MEDIMG_IMAGE_SIZE", 224))
    max_image_bytes: int = field(default_factory=lambda: _int_env("MEDIMG_MAX_IMAGE_BYTES", 5 * 1024 * 1024))
    labels: tuple[str, ...] = field(default_factory=lambda: _parse_labels(os.getenv("MEDIMG_LABELS")))
    allowed_mime_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp", "image/tiff")
    disclaimer: str = "This result is for triage support only and is not a medical diagnosis."
