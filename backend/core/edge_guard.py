"""
EdgeGuard — three-tier security gateway for the Aegis clinical assistant.

Tier 1: Local ChromaDB threat signature cache
Tier 2: Fine-tuned Qwen2.5-7B binary prompt-injection classifier
Tier 3: Privacy-preserving cloud escalation (PHI scrubbed only)

The local model is a binary classifier (SAFE / PROMPT_INJECTION).
Attack taxonomy is assigned only by the cloud defender layer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from .cloud_defender import (
    VALID_ATTACK_CLASSES,
    VALID_THREAT_LEVELS,
    CloudDefender,
    CloudDefenderError,
    CloudDefenderUnavailable,
    normalize_cloud_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / evaluation metadata (from fine-tuning results)
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PEFT_ADAPTER_ID = "SushmaMaddali/qwen-medical-prompt-guard-lora"
DEFAULT_MERGED_MODEL_PATH = "./qwen_medical_guard_merged"

LABEL_SAFE = 0
LABEL_PROMPT_INJECTION = 1
ID2LABEL = {0: "SAFE", 1: "PROMPT_INJECTION"}
LABEL2ID = {"SAFE": 0, "PROMPT_INJECTION": 1}

DEFAULT_VECTOR_SIMILARITY_THRESHOLD = 0.82
DEFAULT_ALLOW_THRESHOLD = 0.10
DEFAULT_BLOCK_THRESHOLD = 0.90
DEFAULT_THREAT_COLLECTION = "threat_signatures"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Published evaluation numbers — informational only; not used for gating.
MODEL_EVAL_METRICS = {
    "accuracy": 0.9569,
    "precision": 1.0000,
    "recall": 0.9167,
    "f1": 0.9565,
    "roc_auc": 0.9952,
    "confusion_matrix": [[56, 0], [5, 55]],
    "notes": (
        "Binary classifier missed 5 attacks on the original test set. "
        "Do not treat local triage as perfect. "
        "GPU-only median latency on HP ZGX Nano was ~61–63 ms "
        "(p95 ~63–65 ms)."
    ),
}

SOURCE_VECTOR = "EDGE_VECTOR_CACHE"
SOURCE_TRIAGE = "EDGE_INTENT_TRIAGE"
SOURCE_CLOUD = "CLOUD_DEFENDER_HOTPATCH"

STATUS_PASSED = "PASSED"
STATUS_BLOCKED = "BLOCKED"
STATUS_ESCALATED = "ESCALATED"

UI_BADGES = {
    STATUS_PASSED: {
        "label": "Authorized Query (Pass)",
        "color": "green",
    },
    "BLOCKED_VECTOR": {
        "label": "Blocked at Vector Cache",
        "color": "red",
    },
    "BLOCKED_TRIAGE": {
        "label": "Blocked by Edge Intent Triage",
        "color": "red",
    },
    "BLOCKED_CLOUD": {
        "label": "Blocked by Cloud Defender",
        "color": "red",
    },
    STATUS_ESCALATED: {
        "label": "Escalated to Cloud Defender",
        "color": "purple",
    },
    "ERROR": {
        "label": "Security Review Required",
        "color": "amber",
    },
}


# ---------------------------------------------------------------------------
# PHI redaction
# ---------------------------------------------------------------------------

@dataclass
class RedactionResult:
    """Outcome of PHI scrubbing prior to cloud escalation."""

    scrubbed_text: str
    scrubbed_entities: list[str] = field(default_factory=list)
    phi_redacted: bool = False
    success: bool = True
    error: str | None = None


class PHIRedactor:
    """
    Regex-based PHI / PII scrubber for escalation payloads.

    This is a best-effort local filter, not a certified de-identification
    engine. If redaction fails, EdgeGuard must not escalate to the cloud.
    """

    # Order matters: more specific patterns first.
    PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        (
            "EMAIL",
            re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),
        ),
        (
            "PHONE",
            re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
            ),
        ),
        (
            "SSN",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
        (
            "MRN",
            re.compile(
                r"\b(?:MRN|Medical\s+Record\s+Number)\s*[:#-]?\s*L?\d+\b",
                re.IGNORECASE,
            ),
        ),
        (
            "RECORD_ID",
            re.compile(r"\b(?:MRN|Record\s*ID|Patient\s*ID)\s*[:#]?\s*\w+\b", re.IGNORECASE),
        ),
        (
            "DOB",
            re.compile(
                r"\b(?:DOB|Date\s+of\s+Birth)\s*[:#]?\s*"
                r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
                re.IGNORECASE,
            ),
        ),
        (
            "DATE",
            re.compile(
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
            ),
        ),
        (
            "ADDRESS",
            re.compile(
                r"\b\d{1,6}\s+[A-Za-z0-9.';\-\s]{2,40}\b"
                r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?"
                r"|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "PERSON_NAME",
            re.compile(
                r"\b(?:Patient|Pt\.?|Dr\.?|Doctor|Nurse|Clinician|Physician)"
                r"\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
            ),
        ),
        (
            "PROPER_NAME",
            re.compile(
                r"\b(?:Mr\.|Mrs\.|Ms\.|Miss)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
            ),
        ),
    ]

    def redact(self, text: str) -> RedactionResult:
        if not isinstance(text, str):
            return RedactionResult(
                scrubbed_text="",
                success=False,
                error="Input to redactor must be a string.",
            )

        try:
            scrubbed = text
            entities: list[str] = []

            for entity_type, pattern in self.PATTERNS:
                if pattern.search(scrubbed):
                    scrubbed = pattern.sub(f"[REDACTED_{entity_type}]", scrubbed)
                    if entity_type not in entities:
                        entities.append(entity_type)

            return RedactionResult(
                scrubbed_text=scrubbed,
                scrubbed_entities=entities,
                phi_redacted=bool(entities),
                success=True,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed on any redaction error
            logger.exception("phi_redaction_failed")
            return RedactionResult(
                scrubbed_text="",
                success=False,
                error=f"PHI redaction failed: {exc}",
            )


# ---------------------------------------------------------------------------
# Classifier result
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    safe_probability: float
    attack_probability: float
    predicted_label: str
    latency_ms: float


# ---------------------------------------------------------------------------
# EdgeGuard
# ---------------------------------------------------------------------------

class EdgeGuard:
    """
    Edge security gateway for clinical prompts and RAG context.

    Fail-safe defaults: component failures do not silently allow traffic.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MERGED_MODEL_PATH,
        chroma_client: Any | None = None,
        cloud_defender: Any | None = None,
        vector_similarity_threshold: float = DEFAULT_VECTOR_SIMILARITY_THRESHOLD,
        allow_threshold: float = DEFAULT_ALLOW_THRESHOLD,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        max_length: int = 512,
        device: str | None = None,
        *,
        collection_name: str = DEFAULT_THREAT_COLLECTION,
        chroma_path: str | Path | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_model: Any | None = None,
        base_model_id: str = BASE_MODEL_ID,
        peft_adapter_id: str = PEFT_ADAPTER_ID,
        load_classifier: bool = True,
        enable_cloud_escalation: bool = True,
        fail_closed_on_cloud_error: bool = True,
        fail_closed_on_vector_error: bool = True,
        fail_closed_on_classifier_error: bool = True,
        redactor: PHIRedactor | None = None,
        classifier_model: Any | None = None,
        classifier_tokenizer: Any | None = None,
    ) -> None:
        if not 0.0 <= allow_threshold < block_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= allow_threshold < block_threshold <= 1."
            )
        if not 0.0 < vector_similarity_threshold <= 1.0:
            raise ValueError(
                "vector_similarity_threshold must be in (0, 1]."
            )

        self.model_path = model_path
        self.base_model_id = base_model_id
        self.peft_adapter_id = peft_adapter_id
        self.vector_similarity_threshold = float(vector_similarity_threshold)
        self.allow_threshold = float(allow_threshold)
        self.block_threshold = float(block_threshold)
        self.max_length = int(max_length)
        self.collection_name = collection_name
        self.enable_cloud_escalation = bool(enable_cloud_escalation)
        self.fail_closed_on_cloud_error = bool(fail_closed_on_cloud_error)
        self.fail_closed_on_vector_error = bool(fail_closed_on_vector_error)
        self.fail_closed_on_classifier_error = bool(
            fail_closed_on_classifier_error
        )
        self.redactor = redactor or PHIRedactor()
        self.cloud_defender = cloud_defender or CloudDefender()

        self.device = device or self._resolve_device()
        self._torch = None
        self._model = classifier_model
        self._tokenizer = classifier_tokenizer
        self._classifier_ready = (
            classifier_model is not None and classifier_tokenizer is not None
        )
        self._classifier_load_error: str | None = None

        self._embedding_model_name = embedding_model_name
        self._embedding_model = embedding_model
        self._init_chroma(chroma_client, chroma_path)

        if load_classifier and not self._classifier_ready:
            try:
                self._load_classifier()
            except Exception as exc:  # noqa: BLE001
                self._classifier_load_error = str(exc)
                logger.exception(
                    "edge_guard.classifier_load_failed",
                    extra={"error_type": type(exc).__name__},
                )

        logger.info(
            "edge_guard.initialized",
            extra={
                "device": self.device,
                "classifier_ready": self._classifier_ready,
                "collection": self.collection_name,
                "allow_threshold": self.allow_threshold,
                "block_threshold": self.block_threshold,
                "vector_similarity_threshold": self.vector_similarity_threshold,
            },
        )

    # ----- bootstrap helpers -----

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _init_chroma(
        self,
        chroma_client: Any | None,
        chroma_path: str | Path | None,
    ) -> None:
        if chroma_client is not None:
            self.chroma_client = chroma_client
        else:
            path = Path(
                chroma_path
                if chroma_path is not None
                else Path(__file__).resolve().parents[1]
                / "data"
                / "chroma_threat_db"
            )
            path.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(path))

        existing = {c.name for c in self.chroma_client.list_collections()}
        if self.collection_name in existing:
            self.threat_collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        else:
            self.threat_collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _ensure_embedding_model(self) -> Any:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self._embedding_model_name
            )
        return self._embedding_model

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedding_model = self._ensure_embedding_model()
        vectors = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(v) for v in vectors]

    def _load_classifier(self) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        dtype = (
            torch.bfloat16
            if (
                self.device == "cuda"
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
            else torch.float32
        )

        merged = Path(self.model_path)
        if merged.exists():
            logger.info(
                "edge_guard.loading_merged_classifier",
                extra={"model_path": str(merged)},
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(merged),
                trust_remote_code=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(merged),
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            logger.info(
                "edge_guard.loading_peft_classifier",
                extra={
                    "base_model": self.base_model_id,
                    "adapter": self.peft_adapter_id,
                },
            )
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_id,
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, self.peft_adapter_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        model.to(self.device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._classifier_ready = True
        self._classifier_load_error = None

    # ----- public API -----

    def evaluate(self, prompt: str) -> dict[str, Any]:
        """
        Run the three-tier defense pipeline against one prompt / RAG blob.
        """
        started = time.perf_counter()
        request_id = self._new_request_id()
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            validated = self._validate_prompt(prompt)
        except ValueError as exc:
            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_BLOCKED,
                source_layer=SOURCE_TRIAGE,
                latency_ms=self._elapsed_ms(started),
                confidence_score=1.0,
                attack_detected=True,
                attack_details={
                    "attack_class": "UNKNOWN",
                    "threat_level": "HIGH",
                    "signature_id": "",
                    "extracted_intent": f"Invalid input: {exc}",
                    "hot_patch_signature": None,
                },
                privacy_audit=self._empty_privacy_audit(),
                ui_badge=UI_BADGES["ERROR"],
            )

        # Tier 1 — vector threat cache
        try:
            vector_hit = self._check_vector_cache(validated)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "edge_guard.vector_cache_error",
                extra={"request_id": request_id},
            )
            if self.fail_closed_on_vector_error:
                return self._build_response(
                    request_id=request_id,
                    timestamp=timestamp,
                    status=STATUS_BLOCKED,
                    source_layer=SOURCE_VECTOR,
                    latency_ms=self._elapsed_ms(started),
                    confidence_score=1.0,
                    attack_detected=True,
                    attack_details={
                        "attack_class": "UNKNOWN",
                        "threat_level": "HIGH",
                        "signature_id": "",
                        "extracted_intent": (
                            "Vector threat cache unavailable; "
                            f"fail-closed ({type(exc).__name__})."
                        ),
                        "hot_patch_signature": None,
                    },
                    privacy_audit=self._empty_privacy_audit(),
                    ui_badge=UI_BADGES["ERROR"],
                )
            vector_hit = None

        if vector_hit is not None:
            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_BLOCKED,
                source_layer=SOURCE_VECTOR,
                latency_ms=self._elapsed_ms(started),
                confidence_score=float(vector_hit["similarity"]),
                attack_detected=True,
                attack_details={
                    "attack_class": vector_hit.get(
                        "attack_class", "UNKNOWN"
                    ),
                    "threat_level": vector_hit.get(
                        "threat_level", "HIGH"
                    ),
                    "signature_id": vector_hit.get("signature_id", ""),
                    "extracted_intent": (
                        "Matched known attack signature in local "
                        "vector threat cache."
                    ),
                    "hot_patch_signature": None,
                },
                privacy_audit=self._empty_privacy_audit(),
                ui_badge=UI_BADGES["BLOCKED_VECTOR"],
                extras={
                    "vector_cache_latency_ms": vector_hit["latency_ms"],
                    "vector_similarity": vector_hit["similarity"],
                },
            )

        # Tier 2 — local Qwen binary classifier
        try:
            clf = self._classify(validated)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "edge_guard.classifier_error",
                extra={"request_id": request_id},
            )
            if self.fail_closed_on_classifier_error:
                return self._build_response(
                    request_id=request_id,
                    timestamp=timestamp,
                    status=STATUS_BLOCKED,
                    source_layer=SOURCE_TRIAGE,
                    latency_ms=self._elapsed_ms(started),
                    confidence_score=1.0,
                    attack_detected=True,
                    attack_details={
                        "attack_class": "UNKNOWN",
                        "threat_level": "HIGH",
                        "signature_id": "",
                        "extracted_intent": (
                            "Local classifier unavailable; "
                            f"fail-closed ({type(exc).__name__})."
                        ),
                        "hot_patch_signature": None,
                    },
                    privacy_audit=self._empty_privacy_audit(),
                    ui_badge=UI_BADGES["ERROR"],
                )
            raise

        confidence = max(clf.safe_probability, clf.attack_probability)
        triage_extras = {
            "safe_probability": clf.safe_probability,
            "attack_probability": clf.attack_probability,
            "predicted_label": clf.predicted_label,
            "classifier_latency_ms": clf.latency_ms,
            "model_eval_metrics": MODEL_EVAL_METRICS,
        }

        if clf.attack_probability <= self.allow_threshold:
            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_PASSED,
                source_layer=SOURCE_TRIAGE,
                latency_ms=self._elapsed_ms(started),
                confidence_score=confidence,
                attack_detected=False,
                attack_details=None,
                privacy_audit=self._empty_privacy_audit(),
                ui_badge=UI_BADGES[STATUS_PASSED],
                extras=triage_extras,
            )

        if clf.attack_probability >= self.block_threshold:
            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_BLOCKED,
                source_layer=SOURCE_TRIAGE,
                latency_ms=self._elapsed_ms(started),
                confidence_score=confidence,
                attack_detected=True,
                attack_details={
                    # Binary classifier cannot assign taxonomy.
                    "attack_class": "UNKNOWN",
                    "threat_level": self._threat_from_probability(
                        clf.attack_probability
                    ),
                    "signature_id": "",
                    "extracted_intent": (
                        "Local Qwen classifier flagged PROMPT_INJECTION "
                        f"(p_attack={clf.attack_probability:.4f})."
                    ),
                    "hot_patch_signature": None,
                },
                privacy_audit=self._empty_privacy_audit(),
                ui_badge=UI_BADGES["BLOCKED_TRIAGE"],
                extras=triage_extras,
            )

        # Ambiguous band → Tier 3 (optional)
        if not self.enable_cloud_escalation:
            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_ESCALATED,
                source_layer=SOURCE_TRIAGE,
                latency_ms=self._elapsed_ms(started),
                confidence_score=confidence,
                attack_detected=False,
                attack_details={
                    "attack_class": "UNKNOWN",
                    "threat_level": "MEDIUM",
                    "signature_id": "",
                    "extracted_intent": (
                        "Ambiguous local score; cloud escalation disabled."
                    ),
                    "hot_patch_signature": None,
                },
                privacy_audit=self._empty_privacy_audit(),
                ui_badge=UI_BADGES[STATUS_ESCALATED],
                extras=triage_extras,
            )

        return self._escalate_to_cloud(
            prompt=validated,
            request_id=request_id,
            timestamp=timestamp,
            started=started,
            triage_confidence=confidence,
            triage_extras=triage_extras,
        )

    def hot_patch_vector_db(
        self,
        signature: str,
        attack_class: str,
        threat_level: str,
    ) -> str:
        """
        Persist a validated attack signature into the local threat cache.
        Returns the stored signature ID. Deduplicates near-identical text.
        """
        if not isinstance(signature, str) or not signature.strip():
            raise ValueError("signature must be a non-empty string.")

        attack_class = str(attack_class).upper().strip()
        if attack_class not in VALID_ATTACK_CLASSES:
            raise ValueError(
                f"Invalid attack_class {attack_class!r}. "
                f"Expected one of {sorted(VALID_ATTACK_CLASSES)}."
            )

        threat_level = str(threat_level).upper().strip()
        if threat_level not in VALID_THREAT_LEVELS:
            raise ValueError(
                f"Invalid threat_level {threat_level!r}. "
                f"Expected one of {sorted(VALID_THREAT_LEVELS)}."
            )

        clean_signature = signature.strip()
        signature_id = self._signature_id(clean_signature)

        # Exact-ID dedupe
        existing = self.threat_collection.get(ids=[signature_id])
        if existing and existing.get("ids"):
            logger.info(
                "edge_guard.hot_patch_duplicate",
                extra={"signature_id": signature_id},
            )
            return signature_id

        # Near-duplicate by embedding similarity
        embedding_model = self._ensure_embedding_model()
        embedding = self._embed_texts([clean_signature])

        try:
            if self.threat_collection.count() > 0:
                matches = self.threat_collection.query(
                    query_embeddings=embedding,
                    n_results=1,
                    include=["distances", "metadatas"],
                )
                if matches["ids"] and matches["ids"][0]:
                    distance = float(matches["distances"][0][0])
                    similarity = 1.0 - distance
                    if similarity >= self.vector_similarity_threshold:
                        existing_id = matches["ids"][0][0]
                        logger.info(
                            "edge_guard.hot_patch_near_duplicate",
                            extra={
                                "signature_id": existing_id,
                                "similarity": similarity,
                            },
                        )
                        return existing_id
        except Exception:  # noqa: BLE001
            logger.exception("edge_guard.hot_patch_dedupe_query_failed")

        self.threat_collection.upsert(
            ids=[signature_id],
            documents=[clean_signature],
            embeddings=embedding,
            metadatas=[
                {
                    "attack_class": attack_class,
                    "threat_level": threat_level,
                    "source": "hot_patch",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        logger.info(
            "edge_guard.hot_patch_stored",
            extra={
                "signature_id": signature_id,
                "attack_class": attack_class,
                "threat_level": threat_level,
                # Do not log signature body — may still contain sensitive text.
                "signature_chars": len(clean_signature),
            },
        )
        return signature_id

    # ----- tier implementations -----

    def _check_vector_cache(self, prompt: str) -> dict[str, Any] | None:
        """
        Return match metadata when similarity >= threshold, else None.

        Medical terminology alone must not block: only known attack
        signatures stored in the threat collection can match.
        """
        started = time.perf_counter()
        if self.threat_collection.count() == 0:
            return None

        query_embedding = self._embed_texts([prompt])

        results = self.threat_collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        latency_ms = self._elapsed_ms(started)

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = float(results["distances"][0][0])
        similarity = 1.0 - distance
        if similarity < self.vector_similarity_threshold:
            return None

        metadata = results["metadatas"][0][0] or {}
        return {
            "signature_id": results["ids"][0][0],
            "similarity": similarity,
            "distance": distance,
            "latency_ms": latency_ms,
            "attack_class": metadata.get("attack_class", "UNKNOWN"),
            "threat_level": metadata.get("threat_level", "HIGH"),
        }

    def _classify(self, prompt: str) -> ClassifierResult:
        if not self._classifier_ready or self._model is None:
            detail = self._classifier_load_error or "classifier not loaded"
            raise RuntimeError(f"Qwen classifier unavailable: {detail}")

        import torch

        self._torch = torch
        started = time.perf_counter()

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        safe_probability = float(probabilities[LABEL_SAFE].item())
        attack_probability = float(
            probabilities[LABEL_PROMPT_INJECTION].item()
        )
        predicted_idx = int(torch.argmax(probabilities).item())
        predicted_label = ID2LABEL.get(predicted_idx, "SAFE")
        latency_ms = self._elapsed_ms(started)

        return ClassifierResult(
            safe_probability=safe_probability,
            attack_probability=attack_probability,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
        )

    def _escalate_to_cloud(
        self,
        prompt: str,
        request_id: str,
        timestamp: str,
        started: float,
        triage_confidence: float,
        triage_extras: dict[str, Any],
    ) -> dict[str, Any]:
        redaction = self.redactor.redact(prompt)
        privacy_audit = {
            "phi_redacted": redaction.phi_redacted,
            "cloud_escalated": False,
            "scrubbed_entities": list(redaction.scrubbed_entities),
        }

        if not redaction.success:
            logger.error(
                "edge_guard.redaction_failed_abort_escalation",
                extra={"request_id": request_id},
            )
            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_BLOCKED,
                source_layer=SOURCE_CLOUD,
                latency_ms=self._elapsed_ms(started),
                confidence_score=triage_confidence,
                attack_detected=True,
                attack_details={
                    "attack_class": "UNKNOWN",
                    "threat_level": "HIGH",
                    "signature_id": "",
                    "extracted_intent": (
                        "PHI redaction failed; cloud escalation aborted."
                    ),
                    "hot_patch_signature": None,
                },
                privacy_audit=privacy_audit,
                ui_badge=UI_BADGES["ERROR"],
                extras=triage_extras,
            )

        try:
            raw = self.cloud_defender.analyze(redaction.scrubbed_text)
            cloud = normalize_cloud_result(raw)
        except (CloudDefenderUnavailable, CloudDefenderError, TimeoutError) as exc:
            logger.warning(
                "edge_guard.cloud_unavailable",
                extra={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )
            return self._cloud_failure_response(
                request_id=request_id,
                timestamp=timestamp,
                started=started,
                triage_confidence=triage_confidence,
                privacy_audit=privacy_audit,
                triage_extras=triage_extras,
                reason=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "edge_guard.cloud_unexpected_error",
                extra={"request_id": request_id},
            )
            return self._cloud_failure_response(
                request_id=request_id,
                timestamp=timestamp,
                started=started,
                triage_confidence=triage_confidence,
                privacy_audit=privacy_audit,
                triage_extras=triage_extras,
                reason=f"Unexpected cloud error: {type(exc).__name__}",
            )

        privacy_audit["cloud_escalated"] = True
        signature_id = ""
        hot_patch_signature = cloud.get("hot_patch_signature")

        if cloud["is_attack"]:
            if hot_patch_signature:
                try:
                    signature_id = self.hot_patch_vector_db(
                        signature=hot_patch_signature,
                        attack_class=cloud["attack_class"],
                        threat_level=cloud["threat_level"],
                    )
                except ValueError as exc:
                    logger.warning(
                        "edge_guard.hot_patch_rejected",
                        extra={
                            "request_id": request_id,
                            "reason": str(exc),
                        },
                    )
                    signature_id = ""

            return self._build_response(
                request_id=request_id,
                timestamp=timestamp,
                status=STATUS_BLOCKED,
                source_layer=SOURCE_CLOUD,
                latency_ms=self._elapsed_ms(started),
                confidence_score=float(cloud["confidence_score"]),
                attack_detected=True,
                attack_details={
                    "attack_class": cloud["attack_class"],
                    "threat_level": cloud["threat_level"],
                    "signature_id": signature_id,
                    "extracted_intent": cloud["extracted_intent"],
                    "hot_patch_signature": hot_patch_signature,
                },
                privacy_audit=privacy_audit,
                ui_badge=UI_BADGES["BLOCKED_CLOUD"],
                extras=triage_extras,
            )

        return self._build_response(
            request_id=request_id,
            timestamp=timestamp,
            status=STATUS_PASSED,
            source_layer=SOURCE_CLOUD,
            latency_ms=self._elapsed_ms(started),
            confidence_score=float(cloud["confidence_score"]),
            attack_detected=False,
            attack_details=None,
            privacy_audit=privacy_audit,
            ui_badge=UI_BADGES[STATUS_PASSED],
            extras=triage_extras,
        )

    def _cloud_failure_response(
        self,
        request_id: str,
        timestamp: str,
        started: float,
        triage_confidence: float,
        privacy_audit: dict[str, Any],
        triage_extras: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        if self.fail_closed_on_cloud_error:
            status = STATUS_BLOCKED
            attack_detected = True
            threat_level = "HIGH"
            intent = (
                "Cloud defender unavailable; fail-closed. "
                f"Reason: {reason}"
            )
        else:
            status = STATUS_ESCALATED
            attack_detected = False
            threat_level = "MEDIUM"
            intent = (
                "Cloud defender unavailable; awaiting local review. "
                f"Reason: {reason}"
            )

        return self._build_response(
            request_id=request_id,
            timestamp=timestamp,
            status=status,
            source_layer=SOURCE_CLOUD,
            latency_ms=self._elapsed_ms(started),
            confidence_score=triage_confidence,
            attack_detected=attack_detected,
            attack_details={
                "attack_class": "UNKNOWN",
                "threat_level": threat_level,
                "signature_id": "",
                "extracted_intent": intent,
                "hot_patch_signature": None,
            },
            privacy_audit=privacy_audit,
            ui_badge=UI_BADGES["ERROR"],
            extras=triage_extras,
        )

    # ----- helpers -----

    @staticmethod
    def _validate_prompt(prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string.")
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("prompt must be non-empty.")
        if len(cleaned) > 50_000:
            raise ValueError("prompt exceeds maximum allowed length.")
        return cleaned

    @staticmethod
    def _new_request_id() -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        suffix = uuid.uuid4().hex[:4].upper()
        return f"REQ-{day}-{suffix}"

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)

    @staticmethod
    def _signature_id(signature: str) -> str:
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        return f"SIG-{digest}"

    @staticmethod
    def _threat_from_probability(attack_probability: float) -> str:
        if attack_probability >= 0.98:
            return "CRITICAL"
        if attack_probability >= 0.95:
            return "HIGH"
        if attack_probability >= 0.90:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _empty_privacy_audit() -> dict[str, Any]:
        return {
            "phi_redacted": False,
            "cloud_escalated": False,
            "scrubbed_entities": [],
        }

    @staticmethod
    def _build_response(
        *,
        request_id: str,
        timestamp: str,
        status: str,
        source_layer: str,
        latency_ms: float,
        confidence_score: float,
        attack_detected: bool,
        attack_details: dict[str, Any] | None,
        privacy_audit: dict[str, Any],
        ui_badge: dict[str, str],
        clinical_response: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "request_id": request_id,
            "timestamp": timestamp,
            "status": status,
            "source_layer": source_layer,
            "latency_ms": float(latency_ms),
            "confidence_score": float(confidence_score),
            "attack_detected": bool(attack_detected),
            "attack_details": attack_details,
            "privacy_audit": privacy_audit,
            "clinical_response": clinical_response,
            "ui_badge": ui_badge,
        }
        if extras:
            response["diagnostics"] = extras
        return response


__all__ = [
    "EdgeGuard",
    "PHIRedactor",
    "RedactionResult",
    "ClassifierResult",
    "MODEL_EVAL_METRICS",
]
