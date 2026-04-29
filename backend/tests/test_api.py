from __future__ import annotations

from app.core.config import Settings


def test_health_endpoint_reports_model_readiness(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["inference_mode"] == "test"


def test_triage_accepts_valid_image_and_logs_prediction(client, image_bytes, db_connection) -> None:
    response = client.post(
        "/triage",
        files={"image": ("scan.png", image_bytes, "image/png")},
        data={"symptoms": "blurred vision", "channel": "telegram"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["triage_label"] == "refer_for_review"
    assert body["top_findings"][0]["label"] == "moderate_dr"

    request_row = db_connection.execute("SELECT status FROM requests").fetchone()
    prediction_row = db_connection.execute("SELECT top_label FROM predictions").fetchone()
    assert request_row[0] == "completed"
    assert prediction_row[0] == "moderate_dr"


def test_triage_rejects_invalid_file_type(client) -> None:
    response = client.post(
        "/triage",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file type."


def test_triage_rejects_corrupt_image_payload(client) -> None:
    response = client.post(
        "/triage",
        files={"image": ("broken.png", b"not actually a png", "image/png")},
    )

    assert response.status_code == 400
    assert "readable image" in response.json()["detail"]


def test_triage_rejects_oversized_image(settings, image_bytes) -> None:
    from fastapi.testclient import TestClient

    from app.db.repository import SQLiteRepository
    from app.main import create_app
    from tests.conftest import FakeModel

    limited_settings = Settings(
        db_path=settings.db_path.parent / "limited.db",
        model_path=settings.model_path,
        enable_mock_model=False,
        max_image_bytes=64,
    )
    app = create_app(
        settings=limited_settings,
        model_service=FakeModel(limited_settings.model_version),
        repository=SQLiteRepository(limited_settings.db_path),
    )
    with TestClient(app) as client:
        response = client.post(
            "/triage",
            files={"image": ("scan.png", image_bytes, "image/png")},
        )

    assert response.status_code == 413
    assert "maximum allowed size" in response.json()["detail"]
