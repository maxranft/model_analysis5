from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.triage import Finding, HealthResponse, TriageResponse
from app.services.inference import ModelUnavailableError
from app.services.preprocessing import ImageValidationError


router = APIRouter()


@router.get("/")
async def root(request: Request) -> dict:
    """Root endpoint with API information."""
    settings = request.app.state.settings
    model_service = request.app.state.model_service
    
    base_url = str(request.base_url).rstrip("/")
    
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "model_status": "loaded" if model_service.loaded else "unavailable",
        "inference_mode": model_service.mode,
        "endpoints": {
            "health": f"{base_url}/health",
            "triage": f"{base_url}/triage",
            "docs": f"{base_url}/docs",
            "openapi": f"{base_url}/openapi.json",
        },
        "usage": {
            "triage": "POST multipart/form-data with 'image' file and optional 'symptoms' text",
            "health": "GET to check service and model readiness",
        },
    }


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    model_service = request.app.state.model_service
    settings = request.app.state.settings
    return HealthResponse(
        status="ok" if model_service.loaded else "degraded",
        model_loaded=model_service.loaded,
        inference_mode=model_service.mode,
        model_version=settings.model_version,
    )


@router.post("/triage", response_model=TriageResponse)
async def triage(
    request: Request,
    image: UploadFile = File(...),
    symptoms: str | None = Form(default=None),
    channel: str | None = Form(default=None),
) -> TriageResponse:
    settings = request.app.state.settings
    repository = request.app.state.repository
    preprocessor = request.app.state.preprocessor
    model_service = request.app.state.model_service

    request_id = f"req_{uuid4().hex[:12]}"
    repository.create_request(
        request_id=request_id,
        timestamp=_timestamp(),
        channel=channel,
        symptoms=symptoms,
        image_filename=image.filename,
        status="received",
    )

    try:
        if image.content_type not in settings.allowed_mime_types:
            raise HTTPException(status_code=415, detail="Unsupported file type.")

        payload = await image.read()
        tensor = preprocessor.preprocess(payload)
        prediction = model_service.predict(tensor, symptoms=symptoms)

        top_findings = [
            Finding(label=label, score=score)
            for label, score in sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        response = TriageResponse(
            request_id=request_id,
            model_version=prediction.model_version,
            triage_label=prediction.triage_label,
            confidence=prediction.confidence,
            top_findings=top_findings,
            disclaimer=settings.disclaimer,
        )

        repository.store_prediction(
            request_id=request_id,
            model_version=prediction.model_version,
            top_label=prediction.top_label,
            confidence=prediction.confidence,
            triage_label=prediction.triage_label,
            raw_scores=prediction.scores,
            latency_ms=prediction.latency_ms,
        )
        repository.update_request_status(request_id, "completed")
        return response
    except ImageValidationError as exc:
        repository.update_request_status(request_id, "rejected")
        repository.store_error(request_id, "image_validation", str(exc), _timestamp())
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        repository.update_request_status(request_id, "failed")
        repository.store_error(request_id, "model_unavailable", str(exc), _timestamp())
        raise HTTPException(status_code=503, detail="Inference service is unavailable.") from exc
    except HTTPException as exc:
        repository.update_request_status(request_id, "rejected")
        repository.store_error(request_id, "request_validation", exc.detail, _timestamp())
        raise exc
    except Exception as exc:
        repository.update_request_status(request_id, "failed")
        repository.store_error(request_id, "internal_error", str(exc), _timestamp())
        raise HTTPException(status_code=500, detail="Internal server error.") from exc