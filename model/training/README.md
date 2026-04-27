# Training Notes

This directory contains the PyTorch training pipeline for the APTOS blindness detection dataset.

## Install

```bash
./.venv/bin/python -m pip install -r model/requirements-train.txt
```

## Train

```bash
./.venv/bin/python -m model.training.train --config model/training/aptos_efficientnet_b0.json
```

This will:

- read labels from `data/raw/aptos2019/meta/train.csv`
- read images from `data/raw/aptos2019/train_images/`
- run a stratified train/validation split
- fine-tune EfficientNet with weighted sampling and validation tracking
- save the best checkpoint to `model/artifacts/checkpoints/`

## Evaluate

```bash
./.venv/bin/python -m model.training.evaluate \
  --checkpoint model/artifacts/checkpoints/aptos_efficientnet_b0_best.pt \
  --metrics-out model/artifacts/checkpoints/eval_metrics.json
```

## Outputs

- checkpoint: `model/artifacts/checkpoints/*.pt`
- training metrics: `model/artifacts/checkpoints/metrics.json`
- exported ONNX: `model/artifacts/model.onnx`
