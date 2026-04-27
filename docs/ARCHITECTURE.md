# Medical Imaging AI MVP Architecture

## Goal

Build a lightweight, secure MVP for medical image triage that accepts an uploaded image plus short clinical context, runs a constrained inference pipeline, and returns a structured triage response through a messaging-based AI agent.

This MVP is for triage support and testing only. It is not a diagnostic device, and the system should explicitly avoid presenting output as a medical diagnosis.

## MVP Scope

Included:

- Single-image inference for a defined imaging modality such as fundus scans
- Structured backend API for image upload and prediction
- ONNX-based model inference behind a FastAPI service
- SQLite logging for requests and model outputs
- Messaging agent integration through OpenClaw
- Focused backend and guardrail tests

Excluded from MVP:

- Multi-image studies
- Human-in-the-loop review UI
- Fine-grained patient record management
- Training pipeline orchestration
- Production-grade PHI workflows, regulatory controls, or EHR integrations

## System Overview

The system is split into three runtime components and one testing layer:

1. Model service responsibilities inside the FastAPI backend:
   image preprocessing, ONNX inference, score normalization, response generation
2. Persistence layer:
   SQLite request and prediction logging
3. Agent interface:
   OpenClaw running in Docker and exposed through messaging channels such as Telegram or WhatsApp
4. Test layer:
   pytest-based validation of API behavior, mocked inference, and agent failure handling

## High-Level Data Flow

1. A clinician or tester sends an image and short symptom description to the messaging bot.
2. OpenClaw validates that the request matches the medical imaging workflow.
3. OpenClaw sends a `POST` request to the FastAPI backend with the image and metadata.
4. FastAPI validates the payload, stores a request log entry, and preprocesses the image with Pillow.
5. The backend converts the processed image into the tensor format expected by the ONNX model.
6. ONNX Runtime executes inference and returns class scores.
7. The backend maps scores into a constrained triage response and stores the result in SQLite.
8. OpenClaw reformats the response into plain language and returns it to the user.

## Architecture Decisions

### 1. Model Layer

**Framework:** PyTorch for training and export

**Model choice:** EfficientNet via transfer learning

Reasoning:

- EfficientNet is small enough for MVP deployment and strong enough for image classification baselines.
- Transfer learning reduces dataset requirements and training time.
- Fine-tuning only the classifier head keeps experimentation fast and compute costs low.

**Serving format:** ONNX

Reasoning:

- Removes training-only runtime dependencies from inference
- Produces a portable model artifact
- Works cleanly with `onnxruntime` in a FastAPI service

**Model contract:**

- Input: single RGB image tensor, normalized, fixed resolution such as `224x224`
- Output: class logits or probabilities for a constrained label set
- Versioning: every exported model should carry a model version string used in logs and API responses

### 2. Backend Layer

**Framework:** FastAPI

Responsibilities:

- Accept multipart image upload plus optional symptom text
- Validate MIME type, size, and image readability
- Preprocess images with Pillow
- Run ONNX inference
- Return structured JSON
- Log request metadata and model outputs

**Image preprocessing with Pillow:**

- Open uploaded file safely
- Convert to RGB
- Resize to model input dimensions
- Normalize using the same statistics used during training
- Reject corrupt or unsupported files

**Database:** SQLite

Reasoning:

- Zero-configuration persistence for MVP
- Adequate for request logging and audit traces in local or low-volume testing
- Easy to migrate later to PostgreSQL if concurrency or operational needs increase

Suggested tables:

- `requests`: request id, timestamp, channel, prompt text, image filename, status
- `predictions`: request id, model version, top label, confidence, raw scores, latency ms
- `errors`: request id, error type, message, timestamp

### 3. Agent Interface Layer

**Runtime:** OpenClaw in Docker

Responsibilities:

- Receive images and user text from supported messaging channels
- Trigger only the medical imaging workflow when the input matches expected intent
- Call the FastAPI backend
- Render backend output into a constrained response template
- Handle backend failures without inventing medical output

**Skill behavior:**

1. Accept one image and optional symptom text.
2. Send them to the backend as a multipart request.
3. Return only backend-derived results.
4. If the backend is unavailable, return a fixed fallback message such as:
   `The diagnostic server is currently offline.`
5. Avoid extrapolating beyond the backend response.

## API Contract

### `POST /triage`

Multipart form-data:

- `image`: required file upload
- `symptoms`: optional short text string
- `channel`: optional string such as `telegram`

Response example:

```json
{
  "request_id": "req_123",
  "model_version": "efficientnet_b0_fundus_v1",
  "triage_label": "refer_for_review",
  "confidence": 0.91,
  "top_findings": [
    {
      "label": "possible_retinopathy",
      "score": 0.91
    }
  ],
  "disclaimer": "This result is for triage support only and is not a medical diagnosis."
}
```

### `GET /health`

Returns service status and model readiness.

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Security and Safety Constraints

MVP security should stay narrow and explicit:

- Run OpenClaw in an isolated Docker container
- Keep the FastAPI backend on a private network interface where possible
- Enforce file type and size validation before image decoding
- Strip executable content risk by decoding and re-encoding only through the image pipeline
- Log request IDs rather than patient-identifying information where possible
- Return constrained triage labels, not free-form diagnoses
- Include a disclaimer in every user-facing response

Important limitation:

- If this system is ever used with protected health information, the current MVP stack is not enough on its own. Storage, transport, retention, access control, vendor posture, and audit requirements must be revisited before real clinical deployment.

## Testing Strategy

### Backend tests with pytest

Required coverage:

- Valid image upload returns `200`
- Invalid file type returns `4xx`
- Corrupt image returns `4xx`
- Oversized payload returns `4xx`
- Health endpoint reports model availability

### Mocked inference

Use a mocked ONNX inference layer during routine tests to keep test execution fast and deterministic.

Validate:

- Response schema
- Error handling
- Logging behavior
- Confidence score propagation

### Agent guardrail tests

Validate that OpenClaw:

- Calls the backend only for supported requests
- Returns backend-derived output only
- Uses the offline fallback when the API is unavailable
- Does not fabricate diagnoses or unsupported recommendations

## Deployment Shape

For the MVP, a simple two-container setup is sufficient:

- `backend`: FastAPI app with ONNX model artifact and SQLite volume
- `openclaw`: agent container configured with the medical imaging skill

Suggested future addition:

- Reverse proxy for TLS termination and request limits if exposed beyond local testing

## Recommended Repository Layout

```text
ml.5/
  backend/
    app/
      api/
      core/
      db/
      services/
      schemas/
    tests/
    requirements.txt
  model/
    export/
    training/
    artifacts/
  agent/
    SKILL.md
    docker/
  docs/
    ARCHITECTURE.md
```

## Build Sequence

1. Export a baseline EfficientNet classifier to ONNX.
2. Implement FastAPI endpoints and preprocessing.
3. Add SQLite logging and health checks.
4. Write backend tests with mocked inference.
5. Implement the OpenClaw skill and container wiring.
6. Validate the end-to-end bot workflow with a test image set.

## Risks and Tradeoffs

- SQLite is acceptable for MVP logging but limited under concurrent write load.
- EfficientNet transfer learning is fast to ship but may underperform on narrow or imbalanced clinical datasets.
- Messaging-based UX is fast to pilot but weaker than a dedicated clinical review interface.
- ONNX reduces serving overhead but adds an export compatibility step that must be validated carefully.

## Success Criteria

The MVP is successful when it can:

- Accept a valid medical image from the agent interface
- Return a structured triage response in under a practical test threshold
- Reject malformed uploads safely
- Log requests and predictions reliably
- Fail closed when the backend is unavailable
