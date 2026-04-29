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
        enable_mock_model: bool = False,
    ) -> None:
        self.model_path = model_path
        self.labels = labels
        self.model_version = model_version
        self.enable_mock_model = enable_mock_model
        self.session = None
        self.input_name = ""
        self.mode = "unavailable"
        self.loaded = False
        self._label_index = {label: idx for idx, label in enumerate(self.labels)}
        self._try_initialize()

    def _try_initialize(self) -> None:
        if self.model_path.exists():
            try:
                import onnxruntime as ort
            except ImportError:
                self.session = None
            else:
                self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.mode = "onnx"
                self.loaded = True
                return

        if self.enable_mock_model:
            self.mode = "mock"
            self.loaded = True

    def predict(self, image_tensor: np.ndarray, symptoms: str | None = None) -> PredictionResult:
        if not self.loaded:
            raise ModelUnavailableError("Inference model is not loaded.")

        started = time.perf_counter()
        if self.mode == "onnx":
            raw = self.session.run(None, {self.input_name: image_tensor})[0]
            logits = np.asarray(raw)[0]
        else:
            logits = self._mock_logits(image_tensor, symptoms or "")

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

    def _mock_logits(self, image_tensor: np.ndarray, symptoms: str) -> np.ndarray:
        mean_intensity = float(image_tensor.mean())
        contrast = float(image_tensor.std())
        logits = np.linspace(-0.2, 0.2, num=len(self.labels), dtype=np.float32)

        if "no_dr" in self._label_index:
            logits[self._label_index["no_dr"]] += max(0.0, 0.6 - mean_intensity)
        if "moderate_dr" in self._label_index:
            logits[self._label_index["moderate_dr"]] += mean_intensity + contrast
        if "severe_dr" in self._label_index:
            logits[self._label_index["severe_dr"]] += max(0.0, 0.3 - contrast)

        normalized_symptoms = symptoms.lower()
        if "bleed" in normalized_symptoms and "proliferative_dr" in self._label_index:
            logits[self._label_index["proliferative_dr"]] += 0.5
        if "blur" in normalized_symptoms and "moderate_dr" in self._label_index:
            logits[self._label_index["moderate_dr"]] += 0.3

        return logits

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

