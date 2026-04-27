# Agent Integration

This directory contains the messaging-agent side of the MVP. The primary artifact is [SKILL.md](/Users/maxranft/Downloads/ml.5/agent/SKILL.md), which constrains the agent to:

- accept one medical image and optional symptom text
- call the backend `POST /triage` endpoint
- return only backend-derived output
- fail closed when the backend is unavailable

Wire this skill into your OpenClaw runtime and point the agent container at the backend URL exposed by your deployment.

