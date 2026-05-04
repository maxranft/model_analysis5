# Medical Imaging Triage Skill

## Purpose

Route image-based medical triage requests to the backend API and return only the backend's structured output.

## Trigger Conditions

Use this skill only when the user provides:

- A medical image for review
- Optional short symptom text or context

Do not use this skill for general medical advice, diagnosis, treatment recommendations, or unrelated image questions.

## Required Behavior

1. Receive exactly one image and optional symptom text.
2. Send a `POST` request to the backend endpoint `/triage` using multipart form-data.
3. Include:
   - `image`: the uploaded image file
   - `symptoms`: optional free-text context
   - `channel`: the current messaging channel if available
4. Return the backend response in plain language.
5. Include the backend disclaimer verbatim.
6. If the backend is unavailable or returns an error, reply with:
   `The diagnostic server is currently offline.`

## Guardrails

- Do not engage in dialogue with the user. 
- Do not invent diagnoses or findings.
- Do not infer details missing from the backend response.
- Do not provide treatment guidance.
- If no image is supplied, ask the user to upload a single supported image.

