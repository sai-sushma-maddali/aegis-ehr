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
import json
import os
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from core.document_extract import compose_prompt_with_document, extract_document_text
from core.zrt_metrics import collect_engine_telemetry
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
ACTIVITY_PATH = BACKEND_DIR / "data" / "prompt_activity.json"
MAX_ACTIVITY = 500
_ACTIVITY_LOCK = threading.Lock()


def _load_activity() -> list[dict[str, Any]]:
    if not ACTIVITY_PATH.exists():
        return []
    try:
        loaded = json.loads(ACTIVITY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)][:MAX_ACTIVITY]


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
        self.activity: list[dict[str, Any]] = _load_activity()


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


def _prepare_weight_cache(guard_model_path: str) -> Any | None:
    """Start or reuse the resident model cache. None means load in-process."""
    if STATE.offline_mode or not _env_flag("AEGIS_WEIGHT_CACHE", default=True):
        return None
    if not Path(guard_model_path).exists():
        return None
    from core.weight_cache import (
        RemoteEmbedder,
        cache_is_alive,
        ensure_weight_cache,
    )

    try:
        ensure_weight_cache(guard_model_path)
    except Exception as exc:  # noqa: BLE001
        if cache_is_alive():
            raise
        logger.exception("gateway.weight_cache_unavailable")
        STATE.boot_errors.append(f"Weight cache: {type(exc).__name__}: {exc}")
        return None
    logger.info("gateway.weight_cache_ready")
    return RemoteEmbedder()


def _init_stack() -> None:
    from core.clinical_agent import ClinicalAgent
    from core.edge_guard import DEFAULT_MERGED_MODEL_PATH, EdgeGuard
    from core.ingestion import EHRIngestion

    chroma_path = Path(
        os.getenv("AEGIS_CHROMA_PATH", str(DEFAULT_CHROMA_PATH))
    )
    reset = _env_flag("AEGIS_RESET_ON_BOOT", default=False)
    guard_model_path = os.getenv(
        "AEGIS_GUARD_MODEL_PATH",
        DEFAULT_MERGED_MODEL_PATH,
    )
    remote_embedder = _prepare_weight_cache(guard_model_path)

    try:
        ingestion = EHRIngestion(
            chroma_path=chroma_path,
            embedding_model=remote_embedder,
        )
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
        embed_model = remote_embedder or (
            STATE.ingestion._embedding_model
            if STATE.ingestion is not None
            else None
        )
        if STATE.offline_mode or not Path(guard_model_path).exists():
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
                model_path=guard_model_path,
                chroma_client=(
                    STATE.ingestion.client if STATE.ingestion else None
                ),
                chroma_path=chroma_path,
                collection_name="threat_signatures",
                allow_threshold=0.15,
                block_threshold=0.85,
                embedding_model=embed_model,
                load_classifier=remote_embedder is None,
            )
            if remote_embedder is not None:
                STATE.edge_guard.attach_remote_classifier()
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
            load_embedding_model=remote_embedder is None,
            embedding_model=remote_embedder,
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


def _compose_clinical_question(question: str, patient_label: str | None) -> str:
    text = question.strip()
    label = (patient_label or "").strip()
    if label and label.lower() not in text.lower():
        return f"{text}\n\nPatient: {label}"
    return text


def _subanswer_status(answer: str, fallback_used: bool) -> str:
    if fallback_used:
        return "NEEDS REVIEW"
    match = re.search(
        r"Status:\s*(SUPPORTED|NOT_FOUND|NOT FOUND|ERROR|PARTIAL)",
        answer or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return "NEEDS REVIEW"
    token = match.group(1).upper().replace(" ", "_")
    return {
        "SUPPORTED": "SUPPORTED",
        "NOT_FOUND": "NOT FOUND",
        "PARTIAL": "PARTIAL",
        "ERROR": "NEEDS REVIEW",
    }.get(token, "NEEDS REVIEW")


def _answer_body(answer: str) -> str:
    text = answer or ""
    match = re.search(
        r"Answer:\s*(.*?)(?:\n\s*Source Chunks:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = match.group(1).strip() if match else text.strip()
    return re.sub(r"\s+", " ", body)


def _overall_status(statuses: list[str]) -> str:
    present = set(statuses)
    if not present or present == {"NOT FOUND"}:
        return "NOT FOUND"
    if present == {"SUPPORTED"}:
        return "SUPPORTED"
    if "SUPPORTED" in present:
        return "PARTIAL"
    return "NEEDS REVIEW"


def _match_score(chunk: dict[str, Any]) -> float | None:
    """Cosine match strength for one retrieved chunk, in [0, 1]."""
    similarity = chunk.get("similarity")
    if similarity is not None:
        try:
            return max(0.0, min(1.0, float(similarity)))
        except (TypeError, ValueError):
            pass
    scores: list[float] = []
    for key in ("dense_similarity", "lexical_score"):
        value = chunk.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            scores.append(number)
    if not scores:
        return None
    return max(0.0, min(1.0, max(scores)))


def _shape_clinical_result(result: dict[str, Any]) -> dict[str, Any]:
    """Map ClinicalAgent.ask() onto the frontend answer card."""
    subanswers: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()

    for item in result.get("subanswers") or []:
        answer = str(item.get("answer") or "")
        status = _subanswer_status(answer, bool(item.get("fallback_used")))
        subanswers.append(
            {
                "question": item.get("question") or "",
                "status": status,
                "answer_text": _answer_body(answer),
            }
        )
        for chunk in item.get("retrieved_chunks") or []:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            metadata = chunk.get("metadata") or {}
            preview = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
            citation: dict[str, Any] = {
                "chunk_id": chunk_id,
                "section": str(metadata.get("section") or "Record"),
                "patient_name": str(metadata.get("patient_name") or ""),
                "preview": preview[:400],
            }
            score = _match_score(chunk)
            if score is not None:
                citation["score"] = round(score, 4)
            evidence.append(citation)

    answer = str(result.get("answer") or "").strip()
    if result.get("mode") == "chat":
        return {
            "patient_id": None,
            "patient_name": None,
            "answer": answer,
            "overall_status": "CHAT",
            "subanswers": [
                {
                    "question": (subanswers[0]["question"] if subanswers else ""),
                    "status": "CHAT",
                    "answer_text": answer,
                }
            ],
            "evidence": [],
            "confidence": None,
            "searched": False,
            "latency_seconds": float(result.get("latency_seconds") or 0.0),
        }
    cited_scores = [
        float(item["score"])
        for item in evidence[:3]
        if item.get("score") is not None
    ]
    confidence = (
        round(sum(cited_scores) / len(cited_scores), 4) if cited_scores else None
    )
    if answer and subanswers:
        subanswers[0]["status"] = "SUPPORTED" if evidence else "NOT FOUND"
        subanswers[0]["answer_text"] = answer
    statuses = [item["status"] for item in subanswers]
    overall = "SUPPORTED" if answer and evidence else _overall_status(statuses)
    if answer and not evidence:
        overall = "NOT FOUND"
    return {
        "patient_id": result.get("patient_id"),
        "patient_name": result.get("patient_name"),
        "answer": answer,
        "overall_status": overall,
        "subanswers": subanswers,
        "evidence": evidence[:3],
        "confidence": confidence,
        "searched": True,
        "latency_seconds": float(result.get("latency_seconds") or 0.0),
    }


def _block_notice(guard_result: dict[str, Any]) -> dict[str, Any]:
    """Short, user-facing explanation for a blocked clinical question."""
    source = str(guard_result.get("source_layer") or "")
    summaries = {
        "EDGE_INTENT_TRIAGE": (
            "Tier 2: On-Device Intent Guard classified this message as a "
            "prompt injection and stopped it before any chart was opened."
        ),
        "EDGE_VECTOR_CACHE": (
            "Tier 1: Fast Vector Threat Cache matched a stored attack "
            "signature and stopped this message before Tier 2 ran."
        ),
        "CLOUD_DEFENDER_HOTPATCH": (
            "Tier 3: Cloud Forensic Defender flagged this message, so no "
            "chart was opened."
        ),
    }
    notice: dict[str, Any] = {
        "kicker": {
            "EDGE_INTENT_TRIAGE": "Tier 2: On-Device Intent Guard (Qwen2.5-7B)",
            "EDGE_VECTOR_CACHE": "Tier 1: Fast Vector Threat Cache (ChromaDB)",
            "CLOUD_DEFENDER_HOTPATCH": "Tier 3: Cloud Forensic Defender (Gemma 4)",
        }.get(source, "Security screening"),
        "title": "Request blocked",
        "summary": summaries.get(
            source,
            "This message did not pass security screening, so no chart was opened.",
        ),
    }
    confidence = guard_result.get("confidence_score")
    if isinstance(confidence, (int, float)):
        notice["confidence"] = round(float(confidence), 4)
    return notice


def _record_gateway_outcome(status: str, latency_ms: float) -> None:
    STATE.request_count += 1
    STATE.total_latency_ms += latency_ms
    if status == "PASSED":
        STATE.passed_count += 1
    else:
        STATE.blocked_count += 1


def _activity_event(
    prompt: str,
    guard_result: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    source = str(guard_result.get("source_layer") or "")
    status = str(guard_result.get("status") or "BLOCKED").upper()
    privacy = guard_result.get("privacy_audit") or {}
    went_to_cloud = (
        bool(privacy.get("cloud_escalated"))
        or source == "CLOUD_DEFENDER_HOTPATCH"
        or status == "ESCALATED"
    )
    if went_to_cloud and status == "BLOCKED":
        outcome = "cloud_block"
        label = "Blocked by Tier 3: Cloud Forensic Defender (Gemma 4)"
        summary = "Tier 3: Cloud Forensic Defender reviewed this prompt and blocked it."
    elif went_to_cloud and status == "PASSED":
        outcome = "cloud_pass"
        label = "Escalated to Tier 3: Cloud Forensic Defender (Gemma 4)"
        summary = "Tier 3: Cloud Forensic Defender reviewed this prompt and allowed the chart search."
    elif went_to_cloud:
        outcome = "cloud_sent"
        label = "Escalated to Tier 3: Cloud Forensic Defender (Gemma 4)"
        summary = "Tier 2 was uncertain, so this prompt was sent to Tier 3: Cloud Forensic Defender."
    elif status == "BLOCKED" and source == "EDGE_VECTOR_CACHE":
        outcome = "vector_block"
        label = "Blocked by Tier 1: Fast Vector Threat Cache (ChromaDB)"
        summary = (
            "Tier 1: Fast Vector Threat Cache matched a stored attack signature "
            "and stopped this prompt before Tier 2 ran."
        )
    elif status == "BLOCKED":
        outcome = "edge_block"
        label = "Blocked by Tier 2: On-Device Intent Guard (Qwen2.5-7B)"
        summary = (
            "Tier 2: On-Device Intent Guard classified this prompt as a prompt "
            "injection and stopped it before any chart was opened."
        )
    else:
        outcome = "edge_pass"
        label = "Cleared by Tier 2: On-Device Intent Guard (Qwen2.5-7B)"
        summary = "Tier 2: On-Device Intent Guard cleared this prompt on-site and the charts were searched."

    confidence = guard_result.get("confidence_score")
    diagnostics = guard_result.get("diagnostics") or {}
    details = guard_result.get("attack_details") or {}
    signature_id = str(details.get("signature_id") or "")
    if "phi_exposed" in privacy:
        phi_exposed: bool | None = bool(privacy.get("phi_exposed"))
        phi_audited = True
    elif not went_to_cloud:
        phi_exposed = False
        phi_audited = True
    else:
        phi_exposed = None
        phi_audited = False
    payload = privacy.get("cloud_payload_bytes")
    if isinstance(payload, (int, float)):
        payload_bytes: int | None = int(payload)
    elif not went_to_cloud:
        payload_bytes = 0
    else:
        payload_bytes = None
    attack_class = details.get("attack_class")
    if not attack_class and status == "PASSED":
        attack_class = "Benign"
    return {
        "id": str(guard_result.get("request_id") or f"REQ-{time.time_ns()}"),
        "timestamp": guard_result.get("timestamp")
        or datetime.now(timezone.utc).isoformat(),
        "prompt": (prompt or "").strip()[:2000],
        "outcome": outcome,
        "outcome_label": label,
        "summary": summary,
        "source_layer": source,
        "status": status,
        "confidence": (
            round(float(confidence), 4)
            if isinstance(confidence, (int, float))
            else None
        ),
        "latency_ms": round(float(latency_ms), 1),
        "guard_latency_ms": _optional_ms(guard_result.get("latency_ms")),
        "vector_latency_ms": _optional_ms(diagnostics.get("vector_cache_latency_ms")),
        "classifier_latency_ms": _optional_ms(
            diagnostics.get("classifier_latency_ms")
        ),
        "cloud_latency_ms": _optional_ms(diagnostics.get("cloud_latency_ms")),
        "cloud_payload_bytes": payload_bytes,
        "phi_exposed": phi_exposed,
        "phi_audited": phi_audited,
        "attack_class": attack_class,
        "signature_id": signature_id,
        "hot_patched": bool(
            went_to_cloud and status == "BLOCKED" and signature_id
        ),
    }


def _record_activity(
    prompt: str,
    guard_result: dict[str, Any],
    latency_ms: float,
) -> None:
    event = _activity_event(prompt, guard_result, latency_ms)
    with _ACTIVITY_LOCK:
        STATE.activity.insert(0, event)
        del STATE.activity[MAX_ACTIVITY:]
        try:
            ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
            ACTIVITY_PATH.write_text(
                json.dumps(STATE.activity, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.exception("gateway.activity_save_failed")


def _activity_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "handled_by_guard": sum(
            1 for event in events if event.get("outcome") == "edge_pass"
        ),
        "blocked_by_model": sum(
            1
            for event in events
            if event.get("status") == "BLOCKED"
            and event.get("source_layer") == "EDGE_INTENT_TRIAGE"
        ),
        "blocked_by_cache": sum(
            1
            for event in events
            if event.get("source_layer") == "EDGE_VECTOR_CACHE"
            and event.get("status") == "BLOCKED"
        ),
        "blocked_by_guard": sum(
            1
            for event in events
            if event.get("outcome") in {"edge_block", "vector_block"}
            or (
                event.get("status") == "BLOCKED"
                and event.get("source_layer")
                in {"EDGE_INTENT_TRIAGE", "EDGE_VECTOR_CACHE"}
            )
        ),
        "sent_to_cloud": sum(
            1
            for event in events
            if str(event.get("outcome") or "").startswith("cloud")
        ),
    }


def _optional_ms(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 3)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _event_number(event: dict[str, Any], key: str) -> float | None:
    return _optional_ms(event.get(key))


def _event_vector_latency(event: dict[str, Any]) -> float | None:
    measured = _event_number(event, "vector_latency_ms")
    if measured is not None:
        return measured
    if event.get("source_layer") == "EDGE_VECTOR_CACHE":
        return _event_number(event, "latency_ms")
    return None


def _event_classifier_latency(event: dict[str, Any]) -> float | None:
    return _event_number(event, "classifier_latency_ms")


def _event_cloud_latency(event: dict[str, Any]) -> float | None:
    return _event_number(event, "cloud_latency_ms")


def _event_phi_exposed(event: dict[str, Any]) -> bool | None:
    if "phi_exposed" in event and event.get("phi_exposed") is not None:
        return bool(event.get("phi_exposed"))
    if not str(event.get("outcome") or "").startswith("cloud"):
        return False
    return None


def _event_payload_bytes(event: dict[str, Any]) -> int | None:
    value = event.get("cloud_payload_bytes")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if not str(event.get("outcome") or "").startswith("cloud"):
        return 0
    return None


def _event_attack_class(event: dict[str, Any]) -> str:
    recorded = event.get("attack_class")
    if isinstance(recorded, str) and recorded.strip():
        return recorded.strip()
    outcome = str(event.get("outcome") or "")
    status = str(event.get("status") or "")
    if status == "PASSED" or outcome.endswith("_pass"):
        return "Benign"
    if event.get("source_layer") == "EDGE_VECTOR_CACHE":
        return "Known signature"
    if outcome == "edge_block":
        return "Prompt injection"
    return "Unclassified"


def _tier_label(source: str) -> str:
    return {
        "EDGE_VECTOR_CACHE": "Tier 1: Fast Vector Threat Cache (ChromaDB)",
        "EDGE_INTENT_TRIAGE": "Tier 2: On-Device Intent Guard (Qwen2.5-7B)",
        "CLOUD_DEFENDER_HOTPATCH": "Tier 3: Cloud Forensic Defender (Gemma 4)",
    }.get(source, source or "Recorded path")


def _observed_rps(events: list[dict[str, Any]]) -> dict[str, Any]:
    stamps: list[datetime] = []
    for event in events:
        raw = event.get("timestamp")
        if not raw:
            continue
        try:
            stamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(stamps) < 2:
        return {"requests_per_second": None, "span_seconds": None, "samples": len(stamps)}
    span = (max(stamps) - min(stamps)).total_seconds()
    if span <= 0:
        return {"requests_per_second": None, "span_seconds": 0.0, "samples": len(stamps)}
    return {
        "requests_per_second": round((len(stamps) - 1) / span, 4),
        "span_seconds": round(span, 1),
        "samples": len(stamps),
    }


def _smi_number(token: str) -> float | None:
    cleaned = token.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _unified_memory() -> dict[str, Any] | None:
    """GB10 reports framebuffer memory as N/A; the pool is system memory."""
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                fields[key] = int(rest.strip().split()[0])
        total_kb = fields["MemTotal"]
        available_kb = fields["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    if total_kb <= 0:
        return None
    used_kb = max(total_kb - available_kb, 0)
    return {
        "used_mb": round(used_kb / 1024, 1),
        "total_mb": round(total_kb / 1024, 1),
        "used_fraction": round(used_kb / total_kb, 4),
        "source": "unified",
    }


_WEIGHT_BYTES: dict[str, int | None] | None = None


def _tree_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += (Path(dirpath) / filename).stat().st_size
            except OSError:
                continue
    return total


def _weight_bytes() -> dict[str, int | None]:
    """Model and index sizes. Walked once; these trees do not change per query."""
    global _WEIGHT_BYTES
    if _WEIGHT_BYTES is not None:
        return _WEIGHT_BYTES
    guard_path = None
    if STATE.edge_guard is not None and getattr(STATE.edge_guard, "model_path", None):
        guard_path = Path(STATE.edge_guard.model_path)
    guard = _tree_bytes(guard_path) if guard_path is not None else None
    mistral = _tree_bytes(
        Path("/opt/hp/zrt/models/hf/mistralai/Mistral-7B-Instruct-v0.3")
    )
    chroma = _tree_bytes(DEFAULT_CHROMA_PATH)
    parts = [value for value in (guard, mistral, chroma) if value is not None]
    _WEIGHT_BYTES = {
        "guard": guard,
        "mistral": mistral,
        "chroma": chroma,
        "total": sum(parts) if parts else None,
    }
    return _WEIGHT_BYTES


def _hardware_profile() -> dict[str, Any]:
    """Live GB10 sensors plus on-disk weight sizes. Rated TOPS is the chip spec."""
    name = None
    temp = None
    util = None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        parts = [part.strip() for part in output.strip().splitlines()[0].split(",")]
        name = parts[0] if parts else None
        temp = _smi_number(parts[1]) if len(parts) > 1 else None
        util = _smi_number(parts[2]) if len(parts) > 2 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        name = None
    disk_total = None
    try:
        usage = os.statvfs("/")
        disk_total = int(usage.f_frsize) * int(usage.f_blocks)
    except OSError:
        disk_total = None
    weights = _weight_bytes()
    return {
        "gpu_name": name or "NVIDIA GB10",
        "package_temp_c": temp,
        "utilization_pct": util,
        "rated_tops_fp4": 1000,
        "disk_total_bytes": disk_total,
        "weights_bytes": weights.get("total"),
        "weights_guard_bytes": weights.get("guard"),
        "weights_mistral_bytes": weights.get("mistral"),
        "weights_chroma_bytes": weights.get("chroma"),
    }


def _gpu_memory() -> dict[str, Any] | None:
    utilization = None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        parts = [part.strip() for part in output.strip().splitlines()[0].split(",")]
        used = _smi_number(parts[0]) if parts else None
        total = _smi_number(parts[1]) if len(parts) > 1 else None
        utilization = _smi_number(parts[2]) if len(parts) > 2 else None
        if used is not None and total is not None and total > 0:
            return {
                "used_mb": round(used, 1),
                "total_mb": round(total, 1),
                "used_fraction": round(used / total, 4),
                "source": "gpu",
                "utilization_pct": utilization,
            }
    except (OSError, subprocess.SubprocessError, IndexError):
        utilization = None
    memory = _unified_memory()
    if memory is None and utilization is None:
        return None
    if memory is None:
        return {"utilization_pct": utilization}
    memory["utilization_pct"] = utilization
    return memory


def _threat_rows() -> list[dict[str, Any]]:
    if STATE.ingestion is not None:
        return STATE.ingestion.list_threat_signatures()
    if STATE.edge_guard is None:
        return []
    data = STATE.edge_guard.threat_collection.get(include=["documents", "metadatas"])
    return [
        {
            "signature_id": sid,
            "attack_class": (meta or {}).get("attack_class"),
            "source": (meta or {}).get("source"),
            "created_at": (meta or {}).get("created_at"),
        }
        for sid, meta in zip(data.get("ids") or [], data.get("metadatas") or [])
    ]


def _attack_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        groups.setdefault(_event_attack_class(event), []).append(event)
    rows: list[dict[str, Any]] = []
    for attack_class, bucket in groups.items():
        sources = [str(event.get("source_layer") or "") for event in bucket]
        source = max(set(sources), key=sources.count) if sources else ""
        statuses = [str(event.get("status") or "RECORDED") for event in bucket]
        status = max(set(statuses), key=statuses.count) if statuses else "RECORDED"
        hot_patched = sum(1 for event in bucket if event.get("hot_patched"))
        if hot_patched and hot_patched == len(bucket) and status == "BLOCKED":
            status = "HOT-PATCHED"
        rows.append(
            {
                "attack_class": attack_class,
                "samples": len(bucket),
                "source_layer": source,
                "tier_label": _tier_label(source),
                "latency_ms": _mean(
                    [
                        float(event.get("guard_latency_ms") or event["latency_ms"])
                        for event in bucket
                        if isinstance(
                            event.get("guard_latency_ms") or event.get("latency_ms"),
                            (int, float),
                        )
                    ]
                ),
                "status": status,
            }
        )
    rows.sort(key=lambda row: (-int(row["samples"]), str(row["attack_class"])))
    return rows


def _build_telemetry(events: list[dict[str, Any]]) -> dict[str, Any]:
    vector_latencies = [
        latency
        for event in events
        if (latency := _event_vector_latency(event)) is not None
    ]
    classifier_latencies = [
        latency
        for event in events
        if (latency := _event_classifier_latency(event)) is not None
    ]
    cloud_latencies = [
        latency
        for event in events
        if (latency := _event_cloud_latency(event)) is not None
    ]
    phi_values = [_event_phi_exposed(event) for event in events]
    phi_known = [value for value in phi_values if value is not None]
    phi_exposed = sum(1 for value in phi_known if value)
    local_payloads = [
        payload
        for event in events
        if not str(event.get("outcome") or "").startswith("cloud")
        and (payload := _event_payload_bytes(event)) is not None
    ]
    cloud_payloads = [
        payload
        for event in events
        if str(event.get("outcome") or "").startswith("cloud")
        and (payload := _event_payload_bytes(event)) is not None
    ]
    local_blocks = sum(
        1
        for event in events
        if event.get("outcome") in {"edge_block", "vector_block"}
        or (
            event.get("status") == "BLOCKED"
            and event.get("source_layer") in {"EDGE_INTENT_TRIAGE", "EDGE_VECTOR_CACHE"}
        )
    )
    cloud_blocks = sum(1 for event in events if event.get("outcome") == "cloud_block")
    attack_blocks = local_blocks + cloud_blocks
    cloud_attacks = [
        event for event in events if event.get("outcome") == "cloud_block"
    ]
    hot_patched = sum(1 for event in cloud_attacks if event.get("hot_patched"))
    threats = _threat_rows()
    hot_ids = {
        str(row.get("signature_id"))
        for row in threats
        if row.get("source") == "hot_patch" and row.get("signature_id")
    }
    vector_replays = sum(
        1
        for event in events
        if event.get("source_layer") == "EDGE_VECTOR_CACHE"
        and str(event.get("signature_id") or "") in hot_ids
    )
    tier_latencies = {
        "vector": _mean(vector_latencies),
        "classifier": _mean(classifier_latencies),
        "cloud": _mean(cloud_latencies),
    }
    known_latencies = [value for value in tier_latencies.values() if value is not None]
    slowest = max(known_latencies) if known_latencies else None
    guard = STATE.edge_guard
    agent = STATE.clinical_agent
    defender = getattr(guard, "cloud_defender", None) if guard is not None else None
    tiers = []
    for key, name, source in (
        ("vector", "Tier 1: Fast Vector Threat Cache (ChromaDB)", "EDGE_VECTOR_CACHE"),
        ("classifier", "Tier 2: On-Device Intent Guard (Qwen2.5-7B)", "EDGE_INTENT_TRIAGE"),
        ("cloud", "Tier 3: Cloud Forensic Defender (Gemma 4)", "CLOUD_DEFENDER_HOTPATCH"),
    ):
        decisions = [
            event for event in events if event.get("source_layer") == source
        ]
        cloud_calls = sum(
            1
            for event in decisions
            if str(event.get("outcome") or "").startswith("cloud")
        )
        latency = tier_latencies[key]
        tiers.append(
            {
                "id": key,
                "name": name,
                "latency_ms": latency,
                "samples": (
                    len(vector_latencies)
                    if key == "vector"
                    else len(classifier_latencies)
                    if key == "classifier"
                    else len(cloud_latencies)
                ),
                "decisions": len(decisions),
                "cloud_calls": cloud_calls,
                "bar_fraction": (
                    round(latency / slowest, 4)
                    if latency is not None and slowest
                    else None
                ),
            }
        )
    return {
        "requests": len(events),
        "models": {
            "guard": (
                Path(guard.model_path).name
                if guard is not None and getattr(guard, "model_path", None)
                else None
            ),
            "clinical": (
                getattr(agent, "ollama_model", None) if agent is not None else None
            ),
            "cloud": getattr(defender, "model", None) if defender is not None else None,
        },
        "kpis": {
            "vector_latency_ms": tier_latencies["vector"],
            "vector_samples": len(vector_latencies),
            "classifier_latency_ms": tier_latencies["classifier"],
            "classifier_samples": len(classifier_latencies),
            "phi_exposure_rate": (
                round(phi_exposed / len(phi_known), 4) if phi_known else None
            ),
            "phi_exposed": phi_exposed if phi_known else None,
            "phi_audited": len(phi_known),
            "self_heal_rate": (
                round(hot_patched / len(cloud_attacks), 4) if cloud_attacks else None
            ),
            "self_heal_patched": hot_patched if cloud_attacks else None,
            "self_heal_cloud_attacks": len(cloud_attacks),
        },
        "tiers": tiers,
        "local_blocks": local_blocks,
        "attack_blocks": attack_blocks,
        "local_absorption_rate": (
            round(local_blocks / attack_blocks, 4) if attack_blocks else None
        ),
        "gpu": _gpu_memory(),
        "throughput": _observed_rps(events),
        "egress": {
            "local_mean_bytes": _mean([float(value) for value in local_payloads]),
            "local_samples": len(local_payloads),
            "cloud_mean_bytes": _mean([float(value) for value in cloud_payloads]),
            "cloud_samples": len(cloud_payloads),
        },
        "signatures": {
            "total": len(threats),
            "seed": sum(1 for row in threats if row.get("source") == "seed"),
            "hot_patch": sum(1 for row in threats if row.get("source") == "hot_patch"),
            "available": STATE.ingestion is not None or STATE.edge_guard is not None,
        },
        "attack_rows": _attack_rows(events),
        "hot_patches": [
            {
                "signature_id": row.get("signature_id"),
                "attack_class": row.get("attack_class"),
                "created_at": row.get("created_at"),
                "later_vector_blocks": sum(
                    1
                    for event in events
                    if event.get("source_layer") == "EDGE_VECTOR_CACHE"
                    and event.get("signature_id") == row.get("signature_id")
                ),
            }
            for row in threats
            if row.get("source") == "hot_patch"
        ],
        "vector_replays": vector_replays,
        "hardware": _hardware_profile(),
        "engine": collect_engine_telemetry(),
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


class ChatTurn(BaseModel):
    role: str = "user"
    text: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    patient_label: str | None = None
    history: list[ChatTurn] = Field(default_factory=list)

    @field_validator("question", "patient_label")
    @classmethod
    def _strip_ask(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


@app.get("/api/patients")
async def list_patients() -> list[dict[str, str]]:
    agent = STATE.clinical_agent
    if agent is None:
        return []
    registry = agent.patient_registry or {}
    return [
        {"patient_id": patient_id, "patient_name": patient_name}
        for patient_id, patient_name in sorted(
            registry.items(),
            key=lambda item: item[1].lower(),
        )
    ]


def _signature_count() -> int | None:
    try:
        return len(_threat_rows())
    except Exception:  # noqa: BLE001
        return None


def _phi_tokens_egressed(guard_result: dict[str, Any], prompt: str) -> int:
    """PHI tokens that left the device. Redacted or local requests stay at 0."""
    from core.telemetry_tracker import estimate_tokens

    privacy = guard_result.get("privacy_audit") or {}
    if not privacy.get("cloud_escalated") or not privacy.get("phi_exposed"):
        return 0
    payload = privacy.get("cloud_payload_bytes")
    if isinstance(payload, (int, float)) and not isinstance(payload, bool) and payload > 0:
        return max(1, int(payload) // 4)
    return estimate_tokens(prompt)


def _record_request_telemetry(
    prompt: str,
    guard_result: dict[str, Any],
    latency_ms: float,
    *,
    output_text: str | None = None,
    generation_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from core.telemetry_tracker import estimate_tokens, get_tracker

    diagnostics = guard_result.get("diagnostics") or {}
    source = str(guard_result.get("source_layer") or "")
    usage = generation_usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int):
        input_tokens = estimate_tokens(prompt)
    if not isinstance(output_tokens, int):
        output_tokens = estimate_tokens(output_text) if output_text else 0
    generation_ms = usage.get("generation_ms")
    vector_ms = diagnostics.get("vector_cache_latency_ms")
    cache_hit = (
        source == "EDGE_VECTOR_CACHE"
        and str(guard_result.get("status") or "").upper() == "BLOCKED"
    )
    return get_tracker().record(
        source_layer=source,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generation_ms=(
            float(generation_ms)
            if isinstance(generation_ms, (int, float)) and not isinstance(generation_ms, bool)
            else None
        ),
        phi_tokens_egressed=_phi_tokens_egressed(guard_result, prompt),
        cache_hit=cache_hit,
        cache_hit_latency_ms=(
            float(vector_ms)
            if isinstance(vector_ms, (int, float)) and not isinstance(vector_ms, bool)
            else None
        ),
        signature_count=_signature_count(),
    )


@app.post("/api/ask")
async def ask_clinical(request: Request) -> dict[str, Any]:
    """Screen a question (and optional uploaded document), then answer."""
    content_type = request.headers.get("content-type", "")
    document_name: str | None = None
    document_text: str | None = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        question = str(form.get("question") or "").strip()
        patient_label_raw = form.get("patient_label")
        patient_label = (
            str(patient_label_raw).strip() or None
            if patient_label_raw is not None
            else None
        )
        history_raw = form.get("history")
        try:
            history_payload = json.loads(str(history_raw or "[]"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid history payload.") from exc
        if not isinstance(history_payload, list):
            raise HTTPException(status_code=422, detail="History must be a list.")
        history_turns = [
            ChatTurn.model_validate(item)
            for item in history_payload
            if isinstance(item, dict)
        ]
        upload = form.get("document")
        if upload is not None and hasattr(upload, "read"):
            data = await upload.read()
            document_name = getattr(upload, "filename", None) or "document"
            try:
                document_text = extract_document_text(document_name, data)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not question and not document_text:
            raise HTTPException(status_code=422, detail="A question is required.")
        question = question or "Please review this uploaded document."
    else:
        try:
            body = AskRequest.model_validate(await request.json())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail="Invalid ask payload.") from exc
        question = body.question or ""
        patient_label = body.patient_label
        history_turns = body.history
        if not question:
            raise HTTPException(status_code=422, detail="A question is required.")

    if not STATE.edge_guard_ready or STATE.edge_guard is None:
        raise HTTPException(status_code=503, detail="EdgeGuard is not ready.")
    if STATE.clinical_agent is None or not STATE.clinical_agent_ready:
        raise HTTPException(
            status_code=503,
            detail="The clinical assistant is not ready.",
        )

    prompt = (
        compose_prompt_with_document(question, document_name or "document", document_text)
        if document_text
        else question
    )
    started = time.perf_counter()
    guard_result = STATE.edge_guard.evaluate(prompt)
    status = str(guard_result.get("status", "BLOCKED")).upper()
    if status != "PASSED":
        latency_ms = (time.perf_counter() - started) * 1000.0
        _record_gateway_outcome("BLOCKED", latency_ms)
        _record_activity(prompt, guard_result, latency_ms)
        badge = (guard_result.get("ui_badge") or {}).get("label")
        notice = _block_notice(guard_result)
        if badge:
            notice["label"] = badge
        notice["telemetry"] = _record_request_telemetry(
            prompt, guard_result, latency_ms
        )
        raise HTTPException(status_code=403, detail=notice)

    history = [
        {"role": turn.role, "text": turn.text}
        for turn in history_turns[-8:]
        if turn.text.strip()
    ]
    try:
        result = STATE.clinical_agent.ask(
            prompt,
            history=history,
            patient_hint=patient_label,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("gateway.clinical_failed")
        raise HTTPException(
            status_code=503,
            detail=f"Clinical RAG failed after security PASS ({type(exc).__name__}).",
        ) from exc

    payload = _shape_clinical_result(result)
    latency_ms = (time.perf_counter() - started) * 1000.0
    _record_gateway_outcome("PASSED", latency_ms)
    _record_activity(prompt, guard_result, latency_ms)
    payload["telemetry"] = _record_request_telemetry(
        prompt,
        guard_result,
        latency_ms,
        output_text=str(payload.get("answer") or ""),
        generation_usage=result.get("generation_usage")
        if isinstance(result, dict)
        else None,
    )
    if document_name:
        payload["document_name"] = document_name
    return payload


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


@app.post("/api/dashboard/clear")
async def clear_dashboard() -> dict[str, Any]:
    """Empty the dashboard log and session totals. Seed signatures stay."""
    from core.telemetry_tracker import get_tracker

    with _ACTIVITY_LOCK:
        STATE.activity.clear()
        try:
            ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
            ACTIVITY_PATH.write_text("[]\n", encoding="utf-8")
        except OSError:
            logger.exception("gateway.activity_clear_failed")

    threats: dict[str, Any] | None = None
    if STATE.ingestion is not None:
        threats = STATE.ingestion.reset_hot_patches_to_seed()
        STATE.seed_threat_ids = set(threats.get("signature_ids") or [])
        if STATE.edge_guard is not None:
            STATE.edge_guard.threat_collection = STATE.ingestion.threats

    get_tracker().clear()
    from core.zrt_metrics import mark_session_baseline

    mark_session_baseline()
    return {
        "status": "ok",
        "events": [],
        "counts": _activity_counts([]),
        "threats": threats,
    }


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


@app.get("/api/activity")
async def list_activity() -> dict[str, Any]:
    events = list(STATE.activity)
    return {"counts": _activity_counts(events), "events": events}


@app.get("/api/telemetry")
async def telemetry() -> dict[str, Any]:
    from core.telemetry_tracker import get_tracker

    payload = _build_telemetry(list(STATE.activity))
    payload["telemetry"] = get_tracker().snapshot(signature_count=_signature_count())
    return payload


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
        "telemetry": collect_engine_telemetry(),
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

    _record_gateway_outcome(payload["status"], payload["latency_ms"])
    _record_activity(prompt, guard_result, payload["latency_ms"])
    payload["telemetry"] = _record_request_telemetry(
        prompt,
        guard_result,
        payload["latency_ms"],
        output_text=str(guard_result.get("clinical_response") or ""),
    )

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
        # Faster local boot; set to 0 when zrt is serving Mistral.
        os.environ["AEGIS_SKIP_CLINICAL"] = "1"

    uvicorn.run(
        "main:app",
        host=os.getenv("AEGIS_HOST", "0.0.0.0"),
        port=int(os.getenv("AEGIS_PORT", "8000")),
        reload=_env_flag("AEGIS_RELOAD", default=False),
    )
