from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class TriageResponse(BaseModel):
    request_id: str
    model_version: str
    triage_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_findings: list[Finding]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    inference_mode: str
    model_version: str

