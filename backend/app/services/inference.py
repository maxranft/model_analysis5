from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when the inference backend is not ready."""


@dataclass(slots=True)
class PredictionResult:
    model_version: str
    triage_label: str
    scores: dict[str, float]
    top_label: str
    confidence: float
    latency_ms: int


class OnnxTriageModel:
    def __init__(
        self,
        model_path: Path,
        labels: tuple[str, ...],
        model_version: str,
    ) -> None:
        self.model_path = model_path
        self.labels = labels
        self.model_version = model_version
        self.session = None
        self.input_name = ""
        self.mode = "unavailable"
        self.loaded = False
        self._try_initialize()

    def _try_initialize(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import onnxruntime as ort
        except ImportError:
            return
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.mode = "onnx"
        self.loaded = True

    def predict(self, image_tensor: np.ndarray) -> PredictionResult:
        if not self.loaded:
            raise ModelUnavailableError("Inference model is not loaded.")

        started = time.perf_counter()
        raw = self.session.run(None, {self.input_name: image_tensor})[0]
        logits = np.asarray(raw)[0]

        scores = self._softmax(logits)
        labels_to_scores = {label: float(score) for label, score in zip(self.labels, scores)}
        top_label = max(labels_to_scores, key=labels_to_scores.get)
        confidence = labels_to_scores[top_label]
        latency_ms = math.ceil((time.perf_counter() - started) * 1000)

        return PredictionResult(
            model_version=self.model_version,
            triage_label=self._triage_label_for(top_label),
            scores=labels_to_scores,
            top_label=top_label,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        stabilized = logits - np.max(logits)
        exp = np.exp(stabilized)
        return exp / np.sum(exp)

    @staticmethod
    def _triage_label_for(top_label: str) -> str:
        if top_label == "no_dr":
            return "routine_review"
        return "refer_for_review"
