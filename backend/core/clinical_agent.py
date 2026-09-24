"""
Clinical RAG Assistant — ported from clinical_RAG.ipynb.

This module is a direct, behavior-preserving port of the notebook pipeline
(hybrid dense + lexical retrieval, question decomposition, evidence-grounded
generation via local Mistral through Ollama, and citation/coverage
validation) into an importable Python module so it can be served over HTTP
instead of run cell-by-cell in Jupyter.

Nothing about the retrieval, prompting, or validation logic has been
changed from the notebook — only the packaging (module-level lazy
initialization instead of notebook globals, and the existing persisted
ChromaDB collection is reused instead of re-downloading/re-indexing the
source patient reports on every run).

Public API used by the FastAPI layer (backend/main.py):
    - get_patient_registry() -> dict[str, str]
    - ask_clinical_assistant(question, evidence_top_k=3) -> dict
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from threading import Lock

import chromadb
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PROJECT PATHS — points at the ChromaDB store already built
# and populated by clinical_RAG.ipynb. This module never
# re-downloads or re-indexes patient reports; it only reads
# what the notebook already indexed.
# ============================================================

PROJECT_DIR = Path("/Users/kanika/Desktop/EDGE Hack")
CHROMA_PATH = PROJECT_DIR / "chroma_ehr_db"
COLLECTION_NAME = "clinical_ehr"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

CHUNK_ID_PATTERN = r"\bL\d+_[a-z0-9_]+_s\d{3}_p\d{3}\b"


# ============================================================
# LAZY PIPELINE STATE
# ============================================================
#
# The embedding model and ChromaDB client are expensive to load,
# so they are created once, on first use, and reused across
# requests instead of on every API call or at import time.

class _PipelineState:
    def __init__(self):
        self._lock = Lock()
        self.collection = None
        self.embedding_model = None
        self.patient_registry: dict[str, str] = {}

    def ensure_ready(self):
        if self.collection is not None and self.embedding_model is not None:
            return

        with self._lock:
            if self.collection is not None and self.embedding_model is not None:
                return

            if not CHROMA_PATH.exists():
                raise RuntimeError(
                    f"ChromaDB store not found at {CHROMA_PATH}. "
                    "Run clinical_RAG.ipynb at least once to build the "
                    "patient index before starting the API."
                )

            chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

            try:
                collection = chroma_client.get_collection(name=COLLECTION_NAME)
            except Exception as error:
                raise RuntimeError(
                    f"ChromaDB collection '{COLLECTION_NAME}' was not found. "
                    "Run clinical_RAG.ipynb to index patient data first."
                ) from error

            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            self.collection = collection
            self.embedding_model = embedding_model
            self.patient_registry = build_patient_registry(collection)


_state = _PipelineState()


def get_patient_registry() -> dict:
    """Return the current patient_id -> patient_name registry."""
    _state.ensure_ready()
    return dict(_state.patient_registry)


# ============================================================
# BUILD PATIENT REGISTRY FROM CHROMADB
# ============================================================

def build_patient_registry(collection):
    """
    Build patient_id -> patient_name from indexed metadata.

    If the same MRN appears with inconsistent names, raise instead of
    silently overwriting a registry entry.
    """
    data = collection.get(include=["metadatas"])

    registry = {}

    for metadata in data["metadatas"]:
        patient_id = metadata.get("patient_id")
        patient_name = metadata.get("patient_name")

        if not patient_id or not patient_name:
            continue

        if patient_id in registry and registry[patient_id] != patient_name:
            raise ValueError(
                "Inconsistent patient registry entry "
                f"for {patient_id}: {registry[patient_id]} vs {patient_name}"
            )

        registry[patient_id] = patient_name

    return registry


# ============================================================
# RESOLVE PATIENT FROM QUESTION — SAFE VERSION
# ============================================================

def resolve_patient_from_question(question, patient_registry):
    """
    Resolve exactly one patient from the clinical question.

    Safety rules:
    - Only treat an L-number as MRN when explicitly preceded by
      "MRN" or "Medical Record Number".
    - Do NOT confuse spinal levels such as L4-L5 with an MRN.
    - If patient name and MRN disagree, raise an error.
    - If multiple patients are mentioned, raise an error.
    """
    question_lower = question.lower()

    mrn_matches = re.findall(
        r"\b(?:MRN|Medical\s+Record\s+Number)"
        r"\s*[:#-]?\s*(L\d+)\b",
        question,
        flags=re.IGNORECASE,
    )

    mrn_ids = {mrn.upper() for mrn in mrn_matches}

    if len(mrn_ids) > 1:
        raise ValueError("Multiple MRNs were found in the question.")

    explicit_mrn = next(iter(mrn_ids)) if mrn_ids else None

    if explicit_mrn is not None and explicit_mrn not in patient_registry:
        raise ValueError(
            f"MRN {explicit_mrn} is not present in the patient registry."
        )

    name_matches = []

    for patient_id, patient_name in patient_registry.items():
        name_pattern = r"(?<!\w)" + re.escape(patient_name.lower()) + r"(?!\w)"

        if re.search(name_pattern, question_lower):
            name_matches.append({"patient_id": patient_id, "patient_name": patient_name})

    if len(name_matches) > 1:
        raise ValueError("Multiple patient names were found in the question.")

    name_patient = name_matches[0] if name_matches else None

    if explicit_mrn and name_patient:
        if explicit_mrn != name_patient["patient_id"]:
            raise ValueError(
                "Patient identity conflict: the question names "
                f"{name_patient['patient_name']} ({name_patient['patient_id']}) "
                f"but specifies MRN {explicit_mrn}."
            )
        return name_patient

    if explicit_mrn:
        return {"patient_id": explicit_mrn, "patient_name": patient_registry[explicit_mrn]}

    if name_patient:
        return name_patient

    raise ValueError("Could not safely identify exactly one patient from the question.")


# ============================================================
# CLEAN QUERY BEFORE RETRIEVAL
# ============================================================

def prepare_retrieval_query(question, patient_name=None, patient_id=None):
    """
    Prepare a clinical question for vector/lexical retrieval by stripping
    explicit patient identifiers (name / MRN) so they don't skew semantic
    similarity, while leaving standalone L-numbers (e.g. spinal levels
    such as L4-L5) untouched.
    """
    cleaned = str(question)

    if patient_name:
        cleaned = re.sub(
            rf"{re.escape(patient_name)}(?:['’]s)?",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    if patient_id:
        cleaned = re.sub(
            rf"\b(?:MRN|Medical\s+Record\s+Number)"
            rf"\s*[:#-]?\s*{re.escape(patient_id)}\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"\(\s*\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\bfor\s*,", ",", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s+\?", "?", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# ============================================================
# LEXICAL SEARCH WITH TF-IDF
# ============================================================

def lexical_search(question, patient_id, collection):
    """Keyword/lexical retrieval over one patient's chunks (complements dense search)."""
    patient_data = collection.get(
        where={"patient_id": patient_id},
        include=["documents", "metadatas"],
    )

    ids = patient_data["ids"]
    documents = patient_data["documents"]
    metadatas = patient_data["metadatas"]

    if not ids:
        return []

    texts = [question] + documents

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    query_vector = tfidf_matrix[0]
    document_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(query_vector, document_vectors)[0]

    results = []
    for index, score in enumerate(similarities):
        results.append({
            "chunk_id": ids[index],
            "text": documents[index],
            "metadata": metadatas[index],
            "lexical_score": float(score),
        })

    results.sort(key=lambda item: item["lexical_score"], reverse=True)
    return results


# ============================================================
# DENSE (SEMANTIC) RETRIEVAL
# ============================================================

def retrieve_chunks(question, patient_id, collection, embedding_model, top_k=5):
    patient_data = collection.get(where={"patient_id": patient_id})
    total_chunks = len(patient_data["ids"])

    if total_chunks == 0:
        return []

    query_embedding = embedding_model.encode(
        question, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        where={"patient_id": patient_id},
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for chunk_id, document, metadata, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "chunk_id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": float(distance),
            "similarity": 1.0 - float(distance),
        })

    return retrieved


# ============================================================
# HYBRID RETRIEVAL USING RECIPROCAL RANK FUSION (RRF)
# ============================================================

def hybrid_retrieve(question, patient_id, collection, embedding_model, top_k=5, rrf_k=60):
    """Combine dense (MiniLM) and lexical (TF-IDF) rankings via Reciprocal Rank Fusion."""
    patient_data = collection.get(where={"patient_id": patient_id})
    total_chunks = len(patient_data["ids"])

    if total_chunks == 0:
        return []

    dense_results = retrieve_chunks(
        question=question,
        patient_id=patient_id,
        collection=collection,
        embedding_model=embedding_model,
        top_k=total_chunks,
    )

    lexical_results = lexical_search(question=question, patient_id=patient_id, collection=collection)

    combined = {}

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        combined[chunk_id] = {
            "chunk_id": chunk_id,
            "text": item["text"],
            "metadata": item["metadata"],
            "dense_rank": rank,
            "dense_similarity": item["similarity"],
            "lexical_rank": None,
            "lexical_score": 0.0,
            "rrf_score": 0.0,
        }
        combined[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, item in enumerate(lexical_results, start=1):
        chunk_id = item["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = {
                "chunk_id": chunk_id,
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_rank": None,
                "dense_similarity": 0.0,
                "lexical_rank": rank,
                "lexical_score": item["lexical_score"],
                "rrf_score": 0.0,
            }
        else:
            combined[chunk_id]["lexical_rank"] = rank
            combined[chunk_id]["lexical_score"] = item["lexical_score"]

        combined[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    ranked_results = sorted(combined.values(), key=lambda item: item["rrf_score"], reverse=True)
    return ranked_results[:top_k]


# ============================================================
# BUILD CLEAN RAG CONTEXT
# ============================================================

def build_rag_context(retrieved_chunks):
    context_blocks = []

    for chunk in retrieved_chunks:
        chunk_id = chunk["chunk_id"]
        section_name = chunk["metadata"]["section"]
        text = chunk["text"].strip()

        section_prefix = f"Section: {section_name}"
        if text.startswith(section_prefix):
            text = text[len(section_prefix):].strip()

        block = f"""
SOURCE CHUNK: {chunk_id}
SECTION: {section_name}

{text}
""".strip()

        context_blocks.append(block)

    return ("\n\n" + "=" * 50 + "\n\n").join(context_blocks)


# ============================================================
# LOCAL MISTRAL GENERATION VIA OLLAMA
# ============================================================

def generate_with_mistral(prompt, model=OLLAMA_MODEL, timeout=120):
    """
    Send a prompt to local Mistral through Ollama.

    Raises RuntimeError on connection/generation failure so an error
    message cannot be mistaken for a clinical answer.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except Exception as error:
        raise RuntimeError(
            "Local Mistral generation failed. "
            f"Make sure Ollama is running and {model!r} is installed."
        ) from error


# ============================================================
# DECOMPOSE MULTI-PART CLINICAL QUESTION
# ============================================================

def decompose_question(question):
    """
    Break a multi-part clinical question into independent factual
    sub-questions. Falls back to the original question if the model's
    JSON response can't be parsed.
    """
    prompt = f"""
Split the clinical question below into the smallest independent
factual questions needed to answer it completely.

Rules:
- Do not answer the questions.
- Do not add information.
- Preserve clinically important wording.
- Return ONLY a JSON array of strings.

CLINICAL QUESTION:
{question}
""".strip()

    response = generate_with_mistral(prompt).strip()

    try:
        parsed = json.loads(response)

        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(item, str) and item.strip() for item in parsed)
        ):
            return [item.strip() for item in parsed][:6]

    except json.JSONDecodeError:
        pass

    return [question]


# ============================================================
# RETRIEVE EVIDENCE FOR ONE SUB-QUESTION
# ============================================================

def retrieve_subquestion_evidence(subquestion, patient_id, patient_name, top_k=3):
    retrieval_query = prepare_retrieval_query(
        subquestion, patient_name=patient_name, patient_id=patient_id
    )

    return hybrid_retrieve(
        question=retrieval_query,
        patient_id=patient_id,
        collection=_state.collection,
        embedding_model=_state.embedding_model,
        top_k=top_k,
    )


# ============================================================
# RESPONSE + CITATION INTEGRITY VALIDATION
# ============================================================

def validate_response_integrity(answer, retrieved_chunks):
    """
    Validate response structure and citation integrity. Does NOT claim the
    clinical answer is factually correct — evidence coverage is checked
    separately by verify_answer_against_evidence().
    """
    allowed_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}
    cited_ids = set(re.findall(CHUNK_ID_PATTERN, answer))
    invalid_ids = cited_ids - allowed_ids

    status_match = re.search(r"(?im)^Status:\s*(SUPPORTED|NOT_FOUND)\s*$", answer)
    status = status_match.group(1).upper() if status_match else None

    has_source_section = "Source Chunks:" in answer

    if status == "SUPPORTED":
        valid = has_source_section and len(cited_ids) > 0 and len(invalid_ids) == 0
    elif status == "NOT_FOUND":
        valid = (
            has_source_section
            and len(invalid_ids) == 0
            and (len(cited_ids) > 0 or len(allowed_ids) == 0)
        )
    else:
        valid = False

    return {
        "valid": valid,
        "status": status,
        "cited_ids": sorted(cited_ids),
        "invalid_ids": sorted(invalid_ids),
        "allowed_ids": sorted(allowed_ids),
    }


# ============================================================
# ANSWER + VERIFY ONE CLINICAL SUB-QUESTION
# ============================================================

def verify_answer_against_evidence(subquestion, answer, retrieved_chunks):
    """Verify the generated answer is supported by the retrieved EHR evidence."""
    rag_context = build_rag_context(retrieved_chunks)

    verification_prompt = f"""
You are verifying a clinical RAG answer against retrieved
synthetic Electronic Health Record (EHR) evidence.

QUESTION:
{subquestion}

ANSWER:
{answer}

RETRIEVED EHR EVIDENCE:
{rag_context}

Evaluate BOTH requirements:

1. Every clinical claim in the answer must be explicitly
   supported by the retrieved evidence.

2. The answer must NOT say that information is missing or
   not specified when the retrieved evidence explicitly
   contains the requested information.

Do not use outside medical knowledge.

Return EXACTLY one line:

PASS

or

FAIL: <short reason>
""".strip()

    verdict = generate_with_mistral(verification_prompt).strip()
    passed = verdict.upper().startswith("PASS")

    return {"passed": passed, "verdict": verdict}


def answer_subquestion(subquestion, retrieved_chunks, max_attempts=2):
    """
    Generate one grounded clinical answer.

    Outcomes: SUPPORTED, NOT_FOUND, or ERROR (validation could not be
    satisfied after the allowed retries).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    rag_context = build_rag_context(retrieved_chunks)
    allowed_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
    allowed_ids_text = "\n".join(allowed_ids) if allowed_ids else "(none)"

    base_prompt = f"""
You are a Clinical Assistant working with synthetic
Electronic Health Record (EHR) data.

Answer ONLY the specific clinical question below using ONLY
the retrieved EHR context.

RULES:

1. Use only facts explicitly stated in the retrieved context.

2. Do not use outside medical knowledge.

3. Do not guess or invent information.

4. Do not infer timing, causality, sequence, or relationships
   unless they are explicitly documented.

5. Answer the specific question completely.

6. Do not say information is missing if it is explicitly
   present in the retrieved context.

7. Cite ONLY chunk IDs listed under "Allowed chunk IDs".

8. If the requested information is explicitly present,
   use:

   Status: SUPPORTED

9. If the requested information is genuinely absent from
   the retrieved evidence, use:

   Status: NOT_FOUND

10. For NOT_FOUND, cite the retrieved chunks that were checked.

QUESTION:

{subquestion}

RETRIEVED EHR CONTEXT:

{rag_context}

RESPONSE FORMAT:

Status: SUPPORTED or NOT_FOUND

Answer:
<concise evidence-grounded answer>

Source Chunks:
<supporting or checked chunk IDs>

Allowed chunk IDs:
{allowed_ids_text}
""".strip()

    prompt = base_prompt
    integrity = None
    coverage = None

    for attempt in range(1, max_attempts + 1):
        answer = generate_with_mistral(prompt)

        integrity = validate_response_integrity(answer, retrieved_chunks)
        coverage = verify_answer_against_evidence(subquestion, answer, retrieved_chunks)

        if integrity["valid"] and coverage["passed"]:
            return {
                "answer": answer,
                "integrity_validation": integrity,
                "coverage_validation": coverage,
                "fallback_used": False,
                "error": None,
            }

        prompt = f"""
{base_prompt}

The previous response failed validation.

Previous response:

{answer}

Citation / structure validation:

{integrity}

Evidence coverage validation:

{coverage["verdict"]}

Generate a corrected response.

IMPORTANT:

- Use ONLY the retrieved EHR evidence.
- Do not invent information.
- Do not claim information is missing if the evidence contains it.
- Cite ONLY allowed chunk IDs.
- Follow the required response format exactly.
""".strip()

    checked_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
    checked_text = ", ".join(checked_ids) if checked_ids else "(none)"

    fallback_answer = (
        "Status: ERROR\n\n"
        "Answer:\n"
        "The Clinical Assistant could not produce a response "
        "that passed grounding validation.\n\n"
        "Source Chunks Checked:\n"
        f"{checked_text}"
    )

    return {
        "answer": fallback_answer,
        "integrity_validation": integrity,
        "coverage_validation": coverage,
        "fallback_used": True,
        "error": "GROUNDING_VALIDATION_FAILED",
    }


# ============================================================
# FINAL STABLE CLINICAL RAG PIPELINE
# ============================================================

def ask_clinical_assistant(question, evidence_top_k=3):
    """
    Final clinical RAG pipeline. Identical behavior to clinical_RAG.ipynb.

    Flow:
    Patient resolution -> question decomposition -> patient-filtered
    hybrid retrieval -> answer each information need -> evidence
    verification.
    """
    _state.ensure_ready()

    start_time = time.perf_counter()

    patient = resolve_patient_from_question(question, _state.patient_registry)
    patient_id = patient["patient_id"]
    patient_name = patient["patient_name"]

    subquestions = decompose_question(question)

    if not subquestions:
        subquestions = [question]

    if len(subquestions) > 6:
        subquestions = [question]

    subanswers = []

    for subquestion in subquestions:
        retrieved_chunks = retrieve_subquestion_evidence(
            subquestion=subquestion,
            patient_id=patient_id,
            patient_name=patient_name,
            top_k=evidence_top_k,
        )

        result = answer_subquestion(subquestion=subquestion, retrieved_chunks=retrieved_chunks)

        subanswers.append({
            "question": subquestion,
            "answer": result["answer"],
            "retrieved_chunks": retrieved_chunks,
            "integrity_validation": result["integrity_validation"],
            "coverage_validation": result["coverage_validation"],
            "fallback_used": result.get("fallback_used", False),
        })

    final_answer = "\n\n".join(item["answer"] for item in subanswers)
    total_latency = time.perf_counter() - start_time

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "original_question": question,
        "subquestions": subquestions,
        "subanswers": subanswers,
        "answer": final_answer,
        "latency_seconds": total_latency,
    }
