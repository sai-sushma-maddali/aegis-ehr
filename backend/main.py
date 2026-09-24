"""
Aegis EHR — FastAPI orchestration gateway.

Deterministic chat sequence
---------------------------
1. Tier 1 — threat_signatures Chroma cache (cosine distance <= 0.18)
2. Tier 2 — Qwen local intent triage (allow<=0.15 / block>=0.85)
3. Tier 3 — PHI scrub + Gemma 4 CloudDefender + hot-patch
4. If PASSED — ClinicalAgent RAG over clinical_records; else bypass
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=os.getenv("AEGIS_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("aegis.gateway")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
TEST_RESULTS_PATH = PROJECT_DIR / "tests" / "test_results.json"
DEFAULT_CHROMA_PATH = BACKEND_DIR / "data" / "chroma_aegis_db"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str | None = None
    message: str | None = None
    patient_id: str | None = None
    skip_clinical: bool = False

    @field_validator("prompt", "message")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def resolved_prompt(self) -> str:
        text = self.prompt or self.message
        if not text:
            raise ValueError("Either 'prompt' or 'message' is required.")
        return text


class ChatResponse(BaseModel):
    request_id: str
    timestamp: str
    status: Literal["PASSED", "BLOCKED"]
    source_layer: str
    latency_ms: float
    confidence_score: float
    attack_detected: bool
    attack_details: dict[str, Any] | None = None
    privacy_audit: dict[str, Any]
    clinical_response: str | None = None
    ui_badge: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    edge_guard_ready: bool
    clinical_agent_ready: bool
    ingestion_ready: bool
    offline_mode: bool
    patients_indexed: int
    threat_signatures: int


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.offline_mode = _env_flag("AEGIS_OFFLINE_MODE", default=True)
        self.edge_guard: Any | None = None
        self.clinical_agent: Any | None = None
        self.ingestion: Any | None = None
        self.edge_guard_ready = False
        self.clinical_agent_ready = False
        self.ingestion_ready = False
        self.boot_errors: list[str] = []
        self.request_count = 0
        self.blocked_count = 0
        self.passed_count = 0
        self.total_latency_ms = 0.0
        self.seed_threat_ids: set[str] = set()


STATE = AppState()


def _build_offline_edge_guard(
    chroma_client: Any | None = None,
    embedding_model: Any | None = None,
) -> Any:
    import chromadb

    from core.cloud_defender import CloudDefender
    from core.edge_guard import ClassifierResult, EdgeGuard

    attack_keywords = (
        "ignore", "disregard", "system prompt", "hidden policy", "override",
        "jailbreak", "dan", "code blue", "dr. vance", "select *", "dump",
        "export", "wildcard", "base64", "end of system", "system note",
        "<!-- override", "unrestricted", "bypass", "api keys", "vector db",
        "all_patients", "developer instructions",
    )

    class KeywordClassifier:
        def __call__(self, prompt: str) -> ClassifierResult:
            text = prompt.lower()
            hits = sum(1 for kw in attack_keywords if kw in text)
            if hits >= 2:
                attack_p = 0.96
            elif hits == 1:
                attack_p = 0.50
            else:
                attack_p = 0.04
            safe_p = 1.0 - attack_p
            return ClassifierResult(
                safe_probability=safe_p,
                attack_probability=attack_p,
                predicted_label=(
                    "PROMPT_INJECTION" if attack_p >= safe_p else "SAFE"
                ),
                latency_ms=1.2,
            )

    def cloud_handler(scrubbed: str) -> dict[str, Any]:
        text = scrubbed.lower()
        is_attack = any(
            kw in text
            for kw in (
                "ignore", "override", "jailbreak", "dan", "dump", "export",
                "system note", "unrestricted", "select *", "developer",
            )
        )
        if is_attack:
            return {
                "is_attack": True,
                "attack_class": "DIRECT_INJECTION",
                "threat_level": "HIGH",
                "extracted_intent": "Cloud forensics confirmed adversarial intent.",
                "hot_patch_signature": scrubbed[:240],
                "confidence_score": 0.93,
                "recommended_action": "BLOCK",
            }
        return {
            "is_attack": False,
            "attack_class": None,
            "threat_level": "LOW",
            "extracted_intent": "Cloud forensics judged prompt benign.",
            "hot_patch_signature": None,
            "confidence_score": 0.88,
            "recommended_action": "ALLOW",
        }

    client = chroma_client or chromadb.PersistentClient(
        path=str(DEFAULT_CHROMA_PATH)
    )
    guard = EdgeGuard(
        chroma_client=client,
        collection_name="threat_signatures",
        cloud_defender=CloudDefender(handler=cloud_handler),
        load_classifier=False,
        embedding_model=embedding_model,  # must match seeded MiniLM dims
        classifier_model=object(),
        classifier_tokenizer=object(),
        allow_threshold=0.15,
        block_threshold=0.85,
        vector_similarity_threshold=0.82,
    )
    guard._classify = KeywordClassifier()  # type: ignore[method-assign]
    guard._classifier_ready = True
    return guard


def _init_stack() -> None:
    from core.clinical_agent import ClinicalAgent
    from core.edge_guard import EdgeGuard
    from core.ingestion import EHRIngestion

    chroma_path = Path(
        os.getenv("AEGIS_CHROMA_PATH", str(DEFAULT_CHROMA_PATH))
    )
    reset = _env_flag("AEGIS_RESET_ON_BOOT", default=False)

    try:
        ingestion = EHRIngestion(chroma_path=chroma_path)
        if STATE.offline_mode:
            clinical = ingestion.ingest_clinical_records(
                reset=reset or True,
                prefer_huggingface=False,
            )
            threats = ingestion.seed_threat_signatures(reset=True)
            bootstrap = {"clinical": clinical, "threats": threats}
        else:
            bootstrap = ingestion.bootstrap(reset=reset)

        STATE.ingestion = ingestion
        STATE.ingestion_ready = True
        STATE.seed_threat_ids = {
            row["signature_id"]
            for row in ingestion.list_threat_signatures()
            if (row.get("source") == "seed")
        }
        logger.info(
            "gateway.ingestion_ready",
            extra={
                "patients": bootstrap["clinical"]["patient_count"],
                "threats": bootstrap["threats"]["signature_count"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        STATE.boot_errors.append(f"Ingestion: {type(exc).__name__}: {exc}")
        logger.exception("gateway.ingestion_boot_failed")

    try:
        embed_model = (
            STATE.ingestion._embedding_model
            if STATE.ingestion is not None
            else None
        )
        if STATE.offline_mode or not Path(
            os.getenv("AEGIS_GUARD_MODEL_PATH", "./qwen_medical_guard_merged")
        ).exists():
            client = (
                STATE.ingestion.client if STATE.ingestion is not None else None
            )
            STATE.edge_guard = _build_offline_edge_guard(
                client,
                embedding_model=embed_model,
            )
            STATE.offline_mode = True
        else:
            STATE.edge_guard = EdgeGuard(
                model_path=os.getenv(
                    "AEGIS_GUARD_MODEL_PATH",
                    "./qwen_medical_guard_merged",
                ),
                chroma_client=(
                    STATE.ingestion.client if STATE.ingestion else None
                ),
                chroma_path=chroma_path,
                collection_name="threat_signatures",
                allow_threshold=0.15,
                block_threshold=0.85,
                embedding_model=embed_model,
            )
        STATE.edge_guard_ready = True
        logger.info(
            "gateway.edge_guard_ready",
            extra={"offline": STATE.offline_mode},
        )
    except Exception as exc:  # noqa: BLE001
        STATE.boot_errors.append(f"EdgeGuard: {type(exc).__name__}: {exc}")
        logger.exception("gateway.edge_guard_boot_failed")
        STATE.edge_guard = _build_offline_edge_guard(
            STATE.ingestion.client if STATE.ingestion else None,
            embedding_model=(
                STATE.ingestion._embedding_model
                if STATE.ingestion is not None
                else None
            ),
        )
        STATE.edge_guard_ready = True
        STATE.offline_mode = True

    if _env_flag("AEGIS_SKIP_CLINICAL", default=False):
        logger.info("gateway.clinical_skipped")
        return

    try:
        # Reuse the same chroma path / clinical_records collection when possible.
        agent = ClinicalAgent(
            chroma_path=chroma_path / "clinical_sidecar"
            if STATE.ingestion is None
            else chroma_path,
            collection_name="clinical_records",
            load_embedding_model=True,
            reset_collection=False,
        )
        # If ingestion already filled clinical_records on shared client,
        # refresh registry; otherwise bootstrap via HF/synthetic path.
        if STATE.ingestion is not None:
            # Point clinical agent at ingestion's clinical collection.
            agent.chroma_client = STATE.ingestion.client
            agent.collection = STATE.ingestion.clinical
            agent.refresh_registry()
        if not agent.patient_registry:
            agent.bootstrap_knowledge_base(reset=False)

        STATE.clinical_agent = agent
        STATE.clinical_agent_ready = bool(agent.patient_registry)
        logger.info(
            "gateway.clinical_ready",
            extra={"patients": len(agent.patient_registry)},
        )
    except Exception as exc:  # noqa: BLE001
        STATE.boot_errors.append(
            f"ClinicalAgent: {type(exc).__name__}: {exc}"
        )
        logger.exception("gateway.clinical_boot_failed")


def _normalize_chat_payload(guard_result: dict[str, Any]) -> dict[str, Any]:
    """Map internal EdgeGuard result onto the universal PASSED|BLOCKED schema."""
    status = str(guard_result.get("status", "BLOCKED")).upper()
    if status == "ESCALATED":
        # Public API has no ESCALATED — fail closed.
        status = "BLOCKED"
        guard_result["attack_detected"] = True
        guard_result["ui_badge"] = {
            "label": "Security Review Required",
            "color": "amber",
        }
        if not guard_result.get("source_layer"):
            guard_result["source_layer"] = "CLOUD_DEFENDER_HOTPATCH"

    clinical = guard_result.get("clinical_response")
    if isinstance(clinical, dict):
        clinical = clinical.get("answer") or clinical.get("message")
    if clinical is not None:
        clinical = str(clinical)

    details = guard_result.get("attack_details")
    if status == "PASSED":
        details = None

    return {
        "request_id": guard_result.get("request_id"),
        "timestamp": guard_result.get("timestamp"),
        "status": status,
        "source_layer": guard_result.get("source_layer"),
        "latency_ms": float(guard_result.get("latency_ms") or 0.0),
        "confidence_score": float(guard_result.get("confidence_score") or 0.0),
        "attack_detected": bool(
            guard_result.get("attack_detected", status == "BLOCKED")
        ),
        "attack_details": details,
        "privacy_audit": guard_result.get("privacy_audit")
        or {
            "phi_redacted": False,
            "cloud_escalated": False,
            "scrubbed_entities": [],
        },
        "clinical_response": clinical if status == "PASSED" else None,
        "ui_badge": guard_result.get("ui_badge")
        or {"label": "Security Review Required", "color": "amber"},
    }


def _run_clinical(prompt: str, patient_id: str | None) -> str | None:
    if STATE.clinical_agent is None or not STATE.clinical_agent_ready:
        return (
            "Security check passed, but the clinical assistant is not ready. "
            "Run ingestion bootstrap or unset AEGIS_SKIP_CLINICAL."
        )
    query = prompt
    if patient_id and patient_id.upper() not in prompt.upper():
        query = f"{prompt}\n\n(MRN {patient_id.upper()})"
    try:
        result = STATE.clinical_agent.ask(query)
        return str(result.get("answer") or "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("gateway.clinical_failed")
        return (
            f"Clinical RAG failed after security PASS ({type(exc).__name__})."
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("gateway.startup")
    _init_stack()
    yield
    logger.info("gateway.shutdown")


app = FastAPI(
    title="Aegis EHR Gateway",
    version="1.0.0",
    description="Edge-first clinical AI security gateway for HP ZGX Nano",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv(
            "AEGIS_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if o.strip()
    ] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    threats = 0
    patients = 0
    if STATE.ingestion is not None:
        try:
            threats = STATE.ingestion.threats.count()
        except Exception:  # noqa: BLE001
            threats = 0
    if STATE.clinical_agent is not None:
        patients = len(STATE.clinical_agent.patient_registry or {})
    elif STATE.ingestion is not None:
        try:
            patients = len(
                {
                    (m or {}).get("patient_id")
                    for m in (
                        STATE.ingestion.clinical.get(include=["metadatas"])
                        .get("metadatas")
                        or []
                    )
                    if (m or {}).get("patient_id")
                }
            )
        except Exception:  # noqa: BLE001
            patients = 0

    return HealthResponse(
        status="ok" if STATE.edge_guard_ready else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        edge_guard_ready=STATE.edge_guard_ready,
        clinical_agent_ready=STATE.clinical_agent_ready,
        ingestion_ready=STATE.ingestion_ready,
        offline_mode=STATE.offline_mode,
        patients_indexed=patients,
        threat_signatures=threats,
    )


@app.get("/api/threats")
async def list_threats() -> dict[str, Any]:
    if STATE.ingestion is None and STATE.edge_guard is None:
        raise HTTPException(status_code=503, detail="Threat store unavailable.")
    if STATE.ingestion is not None:
        rows = STATE.ingestion.list_threat_signatures()
    else:
        data = STATE.edge_guard.threat_collection.get(
            include=["documents", "metadatas"]
        )
        rows = [
            {
                "signature_id": sid,
                "signature": doc,
                "attack_class": (meta or {}).get("attack_class"),
                "threat_level": (meta or {}).get("threat_level"),
                "source": (meta or {}).get("source"),
            }
            for sid, doc, meta in zip(
                data.get("ids") or [],
                data.get("documents") or [],
                data.get("metadatas") or [],
            )
        ]
    return {"count": len(rows), "threats": rows}


@app.post("/api/reset")
async def reset_hot_patches() -> dict[str, Any]:
    """Flush dynamic hot-patches and restore the 8 seeded signatures."""
    if STATE.ingestion is None:
        raise HTTPException(status_code=503, detail="Ingestion not ready.")
    result = STATE.ingestion.reset_hot_patches_to_seed()
    STATE.seed_threat_ids = set(result.get("signature_ids") or [])
    # Keep EdgeGuard collection pointer fresh.
    if STATE.edge_guard is not None:
        STATE.edge_guard.threat_collection = STATE.ingestion.threats
    return {
        "status": "ok",
        "message": "Hot-patches flushed; seed threat signatures restored.",
        "threats": result,
    }


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    harness = None
    if TEST_RESULTS_PATH.exists():
        try:
            import json

            harness = json.loads(TEST_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            harness = None
    return {
        "gateway": {
            "request_count": STATE.request_count,
            "passed_count": STATE.passed_count,
            "blocked_count": STATE.blocked_count,
            "mean_latency_ms": (
                round(STATE.total_latency_ms / STATE.request_count, 3)
                if STATE.request_count
                else 0.0
            ),
            "boot_errors": STATE.boot_errors,
            "offline_mode": STATE.offline_mode,
        },
        "harness": harness,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> dict[str, Any]:
    if not STATE.edge_guard_ready or STATE.edge_guard is None:
        raise HTTPException(status_code=503, detail="EdgeGuard is not ready.")

    try:
        prompt = body.resolved_prompt()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(prompt) > 50_000:
        raise HTTPException(status_code=413, detail="Prompt too large.")

    started = time.perf_counter()
    logger.info("gateway.chat", extra={"prompt_chars": len(prompt)})

    guard_result = STATE.edge_guard.evaluate(prompt)
    status = str(guard_result.get("status", "BLOCKED")).upper()

    if status == "PASSED" and not body.skip_clinical:
        guard_result["clinical_response"] = _run_clinical(
            prompt, body.patient_id
        )
    else:
        guard_result["clinical_response"] = None

    payload = _normalize_chat_payload(guard_result)
    # Prefer wall-clock gateway latency for the public field.
    payload["latency_ms"] = round(
        (time.perf_counter() - started) * 1000.0, 3
    )

    STATE.request_count += 1
    STATE.total_latency_ms += payload["latency_ms"]
    if payload["status"] == "PASSED":
        STATE.passed_count += 1
    else:
        STATE.blocked_count += 1

    return payload


@app.exception_handler(Exception)
async def fail_closed_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("gateway.unhandled", extra={"error": type(exc).__name__})
    return JSONResponse(
        status_code=500,
        content={
            "request_id": "REQ-ERROR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "source_layer": "GATEWAY",
            "latency_ms": 0.0,
            "confidence_score": 1.0,
            "attack_detected": True,
            "attack_details": {
                "attack_class": "UNKNOWN",
                "threat_level": "HIGH",
                "signature_id": "",
                "extracted_intent": "Gateway fail-closed on internal error.",
                "hot_patch_signature": None,
            },
            "privacy_audit": {
                "phi_redacted": False,
                "cloud_escalated": False,
                "scrubbed_entities": [],
            },
            "clinical_response": None,
            "ui_badge": {
                "label": "Security Review Required",
                "color": "amber",
            },
        },
    )


if __name__ == "__main__":
    import uvicorn

    if "AEGIS_OFFLINE_MODE" not in os.environ:
        os.environ["AEGIS_OFFLINE_MODE"] = "1"
    if "AEGIS_SKIP_CLINICAL" not in os.environ:
        # Faster local boot; set to 0 when Mistral/Ollama is available.
        os.environ["AEGIS_SKIP_CLINICAL"] = "1"

    uvicorn.run(
        "main:app",
        host=os.getenv("AEGIS_HOST", "0.0.0.0"),
        port=int(os.getenv("AEGIS_PORT", "8000")),
        reload=_env_flag("AEGIS_RELOAD", default=False),
    )
