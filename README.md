# Medical Imaging AI MVP

This repository contains a lightweight MVP for medical image triage. The working core is a FastAPI backend that validates uploaded images, preprocesses them for model inference, logs requests to SQLite, and returns a constrained triage response.

## Repository Layout

- `backend/`: FastAPI app, SQLite persistence, tests, Dockerfile
- `agent/`: OpenClaw skill contract for the messaging interface
- `model/`: placeholders for training, export, and model artifacts
- `docs/`: architecture and project documentation

## Quick Start

1. Create a virtual environment and install the backend requirements.
2. Start the API with mock inference enabled for local testing:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
MEDIMG_ENABLE_MOCK_MODEL=1 uvicorn app.main:app --app-dir backend --reload
```

The API will be available at `http://127.0.0.1:8000`.

Single-command startup:

```bash
./scripts/start_project.sh
```

This script creates `.venv` if needed, installs backend dependencies when missing, prepares runtime directories, enables mock inference automatically when no ONNX model is present, and starts the API on the first open port starting at `8000`.

## Download APTOS Training Data

The competition data requires Kaggle authentication.

1. Generate `kaggle.json` from your Kaggle account settings.
2. Place it at `.kaggle/kaggle.json` in this repo or at `~/.kaggle/kaggle.json`.
3. Accept the competition rules on Kaggle.
4. Run:

```bash
./scripts/download_aptos_competition.sh
```

The files will be downloaded into `data/raw/aptos2019/`.

To fetch just the APTOS training labels and training images:

```bash
./scripts/download_aptos_train_images.sh
```

This script uses `data/raw/aptos2019/meta/train.csv` and downloads the corresponding `train_images/*.png` files into `data/raw/aptos2019/train_images/`.

## Train The Model

Install training dependencies:

```bash
./.venv/bin/python -m pip install -r model/requirements-train.txt
```

Run training:

```bash
./.venv/bin/python -m model.training.train --config model/training/aptos_efficientnet_b0.json
```

Export the best checkpoint to ONNX:

```bash
./.venv/bin/python -m model.export.export_onnx \
  --checkpoint model/artifacts/checkpoints/aptos_efficientnet_b0_best.pt \
  --output model/artifacts/model.onnx
```

## Endpoints

- `GET /health`: reports service and model readiness
- `POST /triage`: accepts multipart form-data with `image`, optional `symptoms`, and optional `channel`

## Notes

- Without `MEDIMG_ENABLE_MOCK_MODEL=1`, the backend expects an ONNX model artifact at `model/artifacts/model.onnx` unless you override `MEDIMG_MODEL_PATH`.
- This MVP is for triage support and testing only. It is not a diagnostic system.
