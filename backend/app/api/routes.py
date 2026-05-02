from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from twilio.request_validator import RequestValidator

from app.schemas.triage import Finding, HealthResponse, TriageResponse
from app.services.inference import ModelUnavailableError
from app.services.preprocessing import ImageValidationError

router = APIRouter()

_GRADE_DISPLAY = {
    "no_dr":             "Grade 0 — No DR",
    "mild_dr":           "Grade 1 — Mild DR",
    "moderate_dr":       "Grade 2 — Moderate DR",
    "severe_dr":         "Grade 3 — Severe DR",
    "proliferative_dr":  "Grade 4 — Proliferative DR",
}

_GRADE_ADVICE = {
    "no_dr":             "No signs of diabetic retinopathy detected.",
    "mild_dr":           "Mild changes detected. Routine follow-up recommended.",
    "moderate_dr":       "Moderate DR detected. Please consult an ophthalmologist.",
    "severe_dr":         "Severe DR detected. Prompt ophthalmology referral advised.",
    "proliferative_dr":  "Proliferative DR detected. Urgent ophthalmology review needed.",
}


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
        prediction = model_service.predict(tensor)

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


def _twiml(message: str) -> PlainTextResponse:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{message}</Message></Response>"
    )
    return PlainTextResponse(content=body, media_type="application/xml")


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request) -> PlainTextResponse:
    settings = request.app.state.settings

    # Validate Twilio signature when credentials are configured
    if settings.twilio_auth_token:
        validator = RequestValidator(settings.twilio_auth_token)
        url = str(request.url)
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(url, dict(form_data), signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature.")
    else:
        form_data = await request.form()

    media_url = form_data.get("MediaUrl0", "")
    if not media_url:
        return _twiml("Please send a retinal fundus image and I will grade it for diabetic retinopathy.")

    preprocessor = request.app.state.preprocessor
    model_service = request.app.state.model_service
    repository   = request.app.state.repository

    # Download the image from Twilio's media server
    auth = (settings.twilio_account_sid, settings.twilio_auth_token) if settings.twilio_account_sid else None
    async with httpx.AsyncClient() as client:
        resp = await client.get(media_url, auth=auth, follow_redirects=True, timeout=15)
    if resp.status_code != 200:
        return _twiml("Could not download your image. Please try again.")

    request_id = f"req_{uuid4().hex[:12]}"
    try:
        tensor = preprocessor.preprocess(resp.content)
        prediction = model_service.predict(tensor)
    except (ImageValidationError, ModelUnavailableError) as exc:
        return _twiml(f"Could not process image: {exc}")
    except Exception:
        return _twiml("An error occurred during analysis. Please try again.")

    repository.create_request(request_id=request_id, timestamp=_timestamp(),
                              channel="whatsapp", symptoms=None,
                              image_filename=str(media_url), status="completed")
    repository.store_prediction(request_id=request_id,
                                model_version=prediction.model_version,
                                top_label=prediction.top_label,
                                confidence=prediction.confidence,
                                triage_label=prediction.triage_label,
                                raw_scores=prediction.scores,
                                latency_ms=prediction.latency_ms)

    grade = _GRADE_DISPLAY.get(prediction.top_label, prediction.top_label)
    conf  = round(prediction.confidence * 100)

    explainer = request.app.state.explainer
    explanation = explainer.explain(grade, conf)

    if explanation:
        msg = f"{grade} ({conf}% confidence)\n\n{explanation}"
    else:
        advice = _GRADE_ADVICE.get(prediction.top_label, "")
        msg = (
            f"{grade} ({conf}% confidence)\n\n"
            f"{advice}\n\n"
            f"⚠️ {settings.disclaimer}"
        )
    return _twiml(msg)