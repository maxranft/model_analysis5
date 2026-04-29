from __future__ import annotations

import io
import sqlite3
from collections.abc import Iterator

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.core.config import Settings
from app.db.repository import SQLiteRepository
from app.main import create_app
from app.services.inference import PredictionResult


class FakeModel:
    loaded = True
    mode = "test"

    def __init__(self, model_version: str) -> None:
        self.model_version = model_version

    def predict(self, image_tensor, symptoms: str | None = None) -> PredictionResult:
        del image_tensor, symptoms
        return PredictionResult(
            model_version=self.model_version,
            triage_label="refer_for_review",
            scores={
                "no_dr": 0.03,
                "mild_dr": 0.04,
                "moderate_dr": 0.72,
                "severe_dr": 0.14,
                "proliferative_dr": 0.07,
            },
            top_label="moderate_dr",
            confidence=0.72,
            latency_ms=4,
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        model_path=tmp_path / "missing.onnx",
        enable_mock_model=False,
        max_image_bytes=1024 * 1024,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(
        settings=settings,
        model_service=FakeModel(settings.model_version),
        repository=SQLiteRepository(settings.db_path),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (32, 32), color=(128, 32, 32))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def db_connection(settings: Settings):
    connection = sqlite3.connect(settings.db_path)
    yield connection
    connection.close()
