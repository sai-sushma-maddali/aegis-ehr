"""
AegisEHR API — thin FastAPI layer around the Clinical RAG pipeline.

This file does not implement any clinical logic itself. It:
  1. Builds the patient dropdown from the real patient registry
     (core.clinical_agent.get_patient_registry).
  2. Calls core.clinical_agent.ask_clinical_assistant(question) unchanged.
  3. Parses the raw result into a UI-friendly shape via
     core.response_parser.parse_clinical_result, so the frontend never has
     to deal with the pipeline's internal text formatting.

Run with:
    uvicorn backend.main:app --reload --port 8000
(from the aegis-ehr project root, with the Anaconda `base` environment
active — that's the environment with chromadb / sentence-transformers /
scikit-learn / fastapi already installed.)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core import clinical_agent
from backend.core.response_parser import parse_clinical_result

logger = logging.getLogger("aegisehr")

app = FastAPI(title="AegisEHR API")

# The React dev server (Vite) runs on a different port, so CORS must be
# opened for local development.
app.add_middleware(
    CORSMiddleware,
    # Vite moves to 5174, 5175... when 5173 is busy, so accept any local port.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    # Optional: when omitted, the pipeline resolves the patient from the
    # question text itself and refuses if it cannot find exactly one.
    patient_label: str | None = None  # e.g. "Elizabeth Brown (MRN L0)"
    question: str


@app.get("/api/patients")
def list_patients():
    """
    Real patient registry, built from what's actually indexed in ChromaDB
    (core.clinical_agent.get_patient_registry). No hardcoded patient list.
    """
    try:
        registry = clinical_agent.get_patient_registry()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return [
        {"patient_id": patient_id, "patient_name": patient_name}
        for patient_id, patient_name in sorted(registry.items())
    ]


@app.post("/api/ask")
def ask(request: AskRequest):
    """
    Construct "For <name> (MRN <id>): <question>" and run it through the
    unmodified ask_clinical_assistant() pipeline, then return a parsed,
    UI-ready result.
    """
    question = (request.question or "").strip()
    patient_label = (request.patient_label or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    full_question = f"For {patient_label}: {question}" if patient_label else question

    try:
        raw_result = clinical_agent.ask_clinical_assistant(full_question)
    except RuntimeError as error:
        # Pipeline infrastructure not ready (ChromaDB / Ollama unavailable).
        logger.exception("Clinical assistant pipeline unavailable")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        # Patient resolution failures are expected, user-facing errors,
        # not server errors.
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Clinical assistant pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="The clinical assistant could not process this question.",
        ) from error

    return parse_clinical_result(raw_result)


@app.get("/api/health")
def health():
    return {"status": "ok"}
