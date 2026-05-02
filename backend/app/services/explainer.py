from __future__ import annotations

import httpx

_PROMPT = """\
You are a compassionate medical assistant helping explain diabetic retinopathy (DR) screening results to a patient.

The AI model analysed a retinal fundus image and found:
- Diagnosis: {grade}
- Confidence: {confidence}%

Write exactly 2-3 sentences in plain English:
1. What this grade means for their vision health.
2. What they should do next.
3. A brief reminder that this is a screening tool, not a medical diagnosis.

Be warm, clear, and concise. No bullet points. No headers.\
"""


class OllamaExplainer:
    def __init__(self, base_url: str, model: str, timeout: float = 20.0) -> None:
        self._url = base_url.rstrip("/") + "/api/generate"
        self._model = model
        self._timeout = timeout

    def available(self) -> bool:
        try:
            r = httpx.get(self._url.replace("/api/generate", "/api/tags"), timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def explain(self, grade_display: str, confidence_pct: int) -> str | None:
        prompt = _PROMPT.format(grade=grade_display, confidence=confidence_pct)
        try:
            r = httpx.post(
                self._url,
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except Exception:
            pass
        return None
