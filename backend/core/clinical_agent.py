"""
Clinical RAG Assistant for Aegis EHR.

Knowledge base : serag-ai/Synthetic-EHR-Llama
Embeddings     : sentence-transformers/all-MiniLM-L6-v2
Vector store   : ChromaDB
LLM            : Mistral-7B-Instruct-v0.3 via local Ollama

Chunking strategy
-----------------
1. Group / process one patient report at a time.
2. Within each patient, split by clinical subheadings.
3. Prefix every chunk with patient name and MRN before embedding.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import chromadb
import requests
from huggingface_hub import hf_hub_download, list_repo_files
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

REPO_ID = "serag-ai/Synthetic-EHR-Llama"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RAW_DATA_FOLDER = BACKEND_DIR / "data" / "raw"
CHROMA_PATH = BACKEND_DIR / "data" / "chroma_ehr_db"
COLLECTION_NAME = "clinical_records"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"

# MiniLM max = 256 tokens. Keep clinical body under 180 so there is room for
# Patient ID / Patient Name / Section headers in the embedded text.
MAX_CONTENT_TOKENS = 180
CHUNK_OVERLAP_TOKENS = 20

DEFAULT_SEED_REPORTS = (
    "llama/report_0.txt",   # Elizabeth Brown — L0
    "llama/report_1.txt",   # Ricky Johnson — L1
    "llama/report_10.txt",  # Alice Smythe — L10
)

CHUNK_ID_PATTERN = r"\bL\d+_[a-z0-9_]+_s\d{3}_p\d{3}\b"


# ---------------------------------------------------------------------------
# Patient metadata + section parsing
# ---------------------------------------------------------------------------

def extract_patient_metadata(text: str) -> dict[str, str]:
    """Extract patient name and MRN from one EHR report."""
    name_match = re.search(
        r"•?\s*Name:\s*([^\n\r]+)",
        text,
        re.IGNORECASE,
    )
    mrn_match = re.search(
        r"•?\s*Medical Record Number:\s*(L\d+)",
        text,
        re.IGNORECASE,
    )

    if not name_match:
        raise ValueError("Patient name could not be found.")
    if not mrn_match:
        raise ValueError("Medical Record Number could not be found.")

    return {
        "patient_id": mrn_match.group(1).upper().strip(),
        "patient_name": name_match.group(1).strip(),
    }


def is_section_heading(line: str) -> bool:
    """Return True if a line looks like a clinical section heading."""
    line = line.strip()
    if not line or line.startswith("•"):
        return False
    if len(line) > 100 or not line.endswith(":"):
        return False
    heading_text = line[:-1].strip()
    return bool(re.search(r"[A-Za-z]", heading_text))


def extract_sections(text: str) -> list[dict[str, str]]:
    """
    Split one patient EHR report into dynamically detected clinical sections.
    Empty sections are ignored.
    """
    sections: list[dict[str, str]] = []
    current_section: str | None = None
    current_content: list[str] = []

    for line in text.splitlines():
        clean_line = line.strip()

        if is_section_heading(clean_line):
            if current_section is not None:
                section_text = "\n".join(current_content).strip()
                if section_text:
                    sections.append(
                        {"section": current_section, "text": section_text}
                    )
            current_section = clean_line.rstrip(":").strip()
            current_content = []
        elif clean_line and current_section is not None:
            current_content.append(clean_line)

    if current_section is not None:
        section_text = "\n".join(current_content).strip()
        if section_text:
            sections.append({"section": current_section, "text": section_text})

    return sections


# ---------------------------------------------------------------------------
# Chunking (sentence-preserving, tokenizer used only for counting)
# ---------------------------------------------------------------------------

def clean_for_id(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like units while preserving wording."""
    protected = text
    abbreviations = ["Dr.", "Mr.", "Mrs.", "Ms.", "e.g.", "i.e.", "vs."]
    placeholders: dict[str, str] = {}

    for index, abbreviation in enumerate(abbreviations):
        placeholder = f"<ABBR_{index}>"
        placeholders[placeholder] = abbreviation
        protected = protected.replace(abbreviation, placeholder)

    protected = re.sub(r"\b(\d+)\.", r"\1<LIST_DOT>", protected)
    sentences = re.split(r"(?<=[.!?])\s+", protected)

    restored: list[str] = []
    for sentence in sentences:
        sentence = sentence.replace("<LIST_DOT>", ".")
        for placeholder, abbreviation in placeholders.items():
            sentence = sentence.replace(placeholder, abbreviation)
        sentence = sentence.strip()
        if sentence:
            restored.append(sentence)
    return restored


class ClinicalChunker:
    """Patient-grouped, subheading-based EHR chunker."""

    def __init__(
        self,
        tokenizer: Any,
        max_content_tokens: int = MAX_CONTENT_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_content_tokens = max_content_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

    def split_oversized_sentence(
        self,
        sentence: str,
        max_tokens: int,
    ) -> list[str]:
        words = sentence.split()
        pieces: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate_words = current_words + [word]
            candidate = " ".join(candidate_words)
            if current_words and self.count_tokens(candidate) > max_tokens:
                pieces.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words

        if current_words:
            pieces.append(" ".join(current_words))
        return pieces

    def split_long_section(
        self,
        text: str,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> list[str]:
        max_tokens = max_tokens or self.max_content_tokens
        overlap_tokens = (
            overlap_tokens
            if overlap_tokens is not None
            else self.overlap_tokens
        )

        if self.count_tokens(text) <= max_tokens:
            return [text]

        sentences: list[str] = []
        for sentence in split_into_sentences(text):
            if self.count_tokens(sentence) <= max_tokens:
                sentences.append(sentence)
            else:
                sentences.extend(
                    self.split_oversized_sentence(sentence, max_tokens)
                )

        chunks: list[str] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            candidate = " ".join(current_sentences + [sentence]).strip()
            if (
                current_sentences
                and self.count_tokens(candidate) > max_tokens
            ):
                chunks.append(" ".join(current_sentences).strip())

                overlap_sentences: list[str] = []
                for previous_sentence in reversed(current_sentences):
                    overlap_candidate = " ".join(
                        [previous_sentence] + overlap_sentences
                    ).strip()
                    if self.count_tokens(overlap_candidate) > overlap_tokens:
                        break
                    overlap_sentences.insert(0, previous_sentence)

                current_sentences = overlap_sentences

            current_sentences.append(sentence)

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return chunks

    def create_patient_chunks(
        self,
        patient_text: str,
    ) -> list[dict[str, Any]]:
        """
        Create unique, source-preserving RAG chunks for one patient.

        Each chunk is prefixed with Patient ID + Patient Name + Section so
        embeddings carry identity context while Chroma filters by patient_id.
        """
        metadata = extract_patient_metadata(patient_text)
        patient_id = metadata["patient_id"]
        patient_name = metadata["patient_name"]
        patient_sections = extract_sections(patient_text)

        final_chunks: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        section_occurrences: dict[str, int] = {}

        for section in patient_sections:
            section_name = section["section"]
            section_text = section["text"]
            section_id = clean_for_id(section_name)

            section_occurrences[section_id] = (
                section_occurrences.get(section_id, 0) + 1
            )
            section_instance = section_occurrences[section_id]

            parts = self.split_long_section(section_text)

            for part_number, part_text in enumerate(parts, start=1):
                chunk_id = (
                    f"{patient_id}_"
                    f"{section_id}_"
                    f"s{section_instance:03d}_"
                    f"p{part_number:03d}"
                )
                if chunk_id in seen_chunk_ids:
                    raise ValueError(
                        f"Duplicate chunk ID generated: {chunk_id}"
                    )
                seen_chunk_ids.add(chunk_id)

                # Patient name + ID precede each subheading group.
                embedding_text = (
                    f"Patient ID: {patient_id}\n"
                    f"Patient Name: {patient_name}\n"
                    f"Section: {section_name}\n\n"
                    f"{part_text}"
                )

                final_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "section": section_name,
                        "section_instance": section_instance,
                        "part": part_number,
                        "text": embedding_text,
                    }
                )

        return final_chunks


# ---------------------------------------------------------------------------
# Query prep + hybrid retrieval
# ---------------------------------------------------------------------------

def prepare_retrieval_query(
    question: str,
    patient_name: str | None = None,
    patient_id: str | None = None,
) -> str:
    """
    Strip explicit patient identifiers from the retrieval query.

    Chroma already filters by patient_id; removing names/MRNs lets the
    embedding focus on clinical concepts. Standalone L-numbers are kept
    so spinal levels like L4-L5 are not destroyed.
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


def build_rag_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Convert retrieved EHR chunks into clean LLM context blocks."""
    context_blocks: list[str] = []

    for chunk in retrieved_chunks:
        chunk_id = chunk["chunk_id"]
        section_name = chunk["metadata"]["section"]
        text = chunk["text"].strip()

        # Drop identity/section prefixes already present as structured headers.
        text = re.sub(
            r"^Patient ID:\s*.+\nPatient Name:\s*.+\n",
            "",
            text,
            count=1,
        )
        section_prefix = f"Section: {section_name}"
        if text.startswith(section_prefix):
            text = text[len(section_prefix):].strip()

        block = (
            f"SOURCE CHUNK: {chunk_id}\n"
            f"SECTION: {section_name}\n\n"
            f"{text}"
        )
        context_blocks.append(block)

    separator = (
        "\n\n"
        "=================================================="
        "\n\n"
    )
    return separator.join(context_blocks)


def validate_response_integrity(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate response structure and citation integrity."""
    allowed_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}
    cited_ids = set(re.findall(CHUNK_ID_PATTERN, answer))
    invalid_ids = cited_ids - allowed_ids

    status_match = re.search(
        r"(?im)^Status:\s*(SUPPORTED|NOT_FOUND)\s*$",
        answer,
    )
    status = status_match.group(1).upper() if status_match else None
    has_source_section = "Source Chunks:" in answer

    if status == "SUPPORTED":
        valid = (
            has_source_section
            and len(cited_ids) > 0
            and len(invalid_ids) == 0
        )
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


def resolve_patient_from_question(
    question: str,
    patient_registry: dict[str, str],
) -> dict[str, str]:
    """
    Resolve exactly one patient from the clinical question.

    Only treat an L-number as an MRN when explicitly preceded by
    "MRN" or "Medical Record Number" (avoids L4-L5 spinal false positives).
    """
    question_lower = question.lower()

    mrn_matches = re.findall(
        r"\b(?:MRN|Medical\s+Record\s+Number)\s*[:#-]?\s*(L\d+)\b",
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

    name_matches: list[dict[str, str]] = []
    for patient_id, patient_name in patient_registry.items():
        name_pattern = (
            r"(?<!\w)"
            + re.escape(patient_name.lower())
            + r"(?!\w)"
        )
        if re.search(name_pattern, question_lower):
            name_matches.append(
                {"patient_id": patient_id, "patient_name": patient_name}
            )

    if len(name_matches) > 1:
        raise ValueError(
            "Multiple patient names were found in the question."
        )

    name_patient = name_matches[0] if name_matches else None

    if explicit_mrn and name_patient:
        if explicit_mrn != name_patient["patient_id"]:
            raise ValueError(
                "Patient identity conflict: "
                f"the question names {name_patient['patient_name']} "
                f"({name_patient['patient_id']}) "
                f"but specifies MRN {explicit_mrn}."
            )
        return name_patient

    if explicit_mrn:
        return {
            "patient_id": explicit_mrn,
            "patient_name": patient_registry[explicit_mrn],
        }

    if name_patient:
        return name_patient

    raise ValueError(
        "Could not safely identify exactly one patient from the question."
    )


def build_patient_registry(collection: Any) -> dict[str, str]:
    """Build patient_id -> patient_name from indexed Chroma metadata."""
    data = collection.get(include=["metadatas"])
    registry: dict[str, str] = {}

    for metadata in data["metadatas"] or []:
        patient_id = metadata.get("patient_id")
        patient_name = metadata.get("patient_name")
        if not patient_id or not patient_name:
            continue
        if patient_id in registry and registry[patient_id] != patient_name:
            raise ValueError(
                "Inconsistent patient registry entry "
                f"for {patient_id}: "
                f"{registry[patient_id]} vs {patient_name}"
            )
        registry[patient_id] = patient_name

    return registry


# ---------------------------------------------------------------------------
# Clinical Assistant
# ---------------------------------------------------------------------------

class ClinicalAssistant:
    """
    Edge-local clinical RAG assistant.

    Downloads synthetic EHR reports from Hugging Face, chunks them by
    patient then subheading, indexes into ChromaDB with MiniLM, and
    answers clinical questions with local Mistral-7B-Instruct-v0.3.
    """

    def __init__(
        self,
        chroma_path: str | Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        ollama_base_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL,
        raw_data_folder: str | Path = RAW_DATA_FOLDER,
        reset_collection: bool = False,
        load_embedding_model: bool = True,
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.raw_data_folder = Path(raw_data_folder)
        self.raw_data_folder.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model: SentenceTransformer | None = None
        self.chunker: ClinicalChunker | None = None
        self.patient_registry: dict[str, str] = {}

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path)
        )

        existing = [
            c.name for c in self.chroma_client.list_collections()
        ]
        if reset_collection and collection_name in existing:
            self.chroma_client.delete_collection(name=collection_name)
            existing = [
                c.name for c in self.chroma_client.list_collections()
            ]

        if collection_name in existing:
            self.collection = self.chroma_client.get_collection(
                name=collection_name
            )
        else:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        if load_embedding_model:
            self._load_embedding_model()

        self.refresh_registry()

    # ----- model / store bootstrap -----

    def _load_embedding_model(self) -> None:
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.chunker = ClinicalChunker(tokenizer=self.embedding_model.tokenizer)

    def refresh_registry(self) -> dict[str, str]:
        self.patient_registry = build_patient_registry(self.collection)
        return self.patient_registry

    def check_ollama(self) -> list[str]:
        response = requests.get(
            f"{self.ollama_base_url}/api/tags",
            timeout=5,
        )
        response.raise_for_status()
        models = [
            model["name"]
            for model in response.json().get("models", [])
        ]
        return models

    def generate_with_mistral(
        self,
        prompt: str,
        timeout: int = 120,
    ) -> str:
        """Send a prompt to local Mistral-7B-Instruct-v0.3 through Ollama."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as error:
            raise RuntimeError(
                "Local Mistral generation failed. "
                "Make sure Ollama is running and "
                f"{self.ollama_model!r} is installed "
                f"(e.g. `ollama pull {self.ollama_model}`)."
            ) from error

    # ----- knowledge-base ingestion -----

    def list_repo_files(self) -> list[str]:
        return list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    def download_patient_report(self, report_file: str) -> Path:
        """Download one complete patient report from Hugging Face."""
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=report_file,
            repo_type="dataset",
            local_dir=str(self.raw_data_folder),
        )
        return Path(file_path)

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks were provided.")
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not loaded.")

        ids = [chunk["chunk_id"] for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk IDs detected before indexing."
            )

        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "patient_id": chunk["patient_id"],
                "patient_name": chunk["patient_name"],
                "section": chunk["section"],
                "section_instance": chunk["section_instance"],
                "part": chunk["part"],
            }
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self.refresh_registry()

    def ingest_patient_report(
        self,
        report_file: str | None = None,
        patient_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Download (optional) and index one patient report.

        Provide either ``report_file`` (HF path under the dataset) or
        raw ``patient_text``.
        """
        if self.chunker is None:
            raise RuntimeError("Chunker is not initialized.")

        if patient_text is None:
            if not report_file:
                raise ValueError(
                    "Provide report_file or patient_text."
                )
            path = self.download_patient_report(report_file)
            patient_text = path.read_text(encoding="utf-8")

        chunks = self.chunker.create_patient_chunks(patient_text)
        self.index_chunks(chunks)
        return chunks

    def bootstrap_knowledge_base(
        self,
        report_files: tuple[str, ...] | list[str] = DEFAULT_SEED_REPORTS,
        reset: bool = False,
    ) -> dict[str, Any]:
        """
        Index seed Synthetic-EHR-Llama reports into ChromaDB.

        Default seeds: Elizabeth Brown (L0), Ricky Johnson (L1),
        Alice Smythe (L10).
        """
        if reset:
            existing = [
                c.name for c in self.chroma_client.list_collections()
            ]
            if self.collection_name in existing:
                self.chroma_client.delete_collection(
                    name=self.collection_name
                )
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.patient_registry = {}

        indexed: list[dict[str, str]] = []
        for report_file in report_files:
            chunks = self.ingest_patient_report(report_file=report_file)
            indexed.append(
                {
                    "report_file": report_file,
                    "patient_id": chunks[0]["patient_id"],
                    "patient_name": chunks[0]["patient_name"],
                    "chunk_count": str(len(chunks)),
                }
            )

        return {
            "patients": indexed,
            "total_chunks": self.collection.count(),
            "registry": dict(self.patient_registry),
        }

    # ----- retrieval -----

    def lexical_search(
        self,
        question: str,
        patient_id: str,
    ) -> list[dict[str, Any]]:
        patient_data = self.collection.get(
            where={"patient_id": patient_id},
            include=["documents", "metadatas"],
        )
        ids = patient_data["ids"]
        documents = patient_data["documents"]
        metadatas = patient_data["metadatas"]
        if not ids:
            return []

        texts = [question] + documents
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(
            tfidf_matrix[0],
            tfidf_matrix[1:],
        )[0]

        results = [
            {
                "chunk_id": ids[index],
                "text": documents[index],
                "metadata": metadatas[index],
                "lexical_score": float(score),
            }
            for index, score in enumerate(similarities)
        ]
        results.sort(key=lambda item: item["lexical_score"], reverse=True)
        return results

    def retrieve_chunks(
        self,
        question: str,
        patient_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not loaded.")

        patient_data = self.collection.get(
            where={"patient_id": patient_id}
        )
        total_chunks = len(patient_data["ids"])
        if total_chunks == 0:
            return []

        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            where={"patient_id": patient_id},
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": float(distance),
                    "similarity": 1.0 - float(distance),
                }
            )
        return retrieved

    def hybrid_retrieve(
        self,
        question: str,
        patient_id: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Dense MiniLM + TF-IDF lexical retrieval fused with RRF."""
        patient_data = self.collection.get(
            where={"patient_id": patient_id}
        )
        total_chunks = len(patient_data["ids"])
        if total_chunks == 0:
            return []

        dense_results = self.retrieve_chunks(
            question=question,
            patient_id=patient_id,
            top_k=total_chunks,
        )
        lexical_results = self.lexical_search(
            question=question,
            patient_id=patient_id,
        )

        combined: dict[str, dict[str, Any]] = {}

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
                "rrf_score": 1 / (rrf_k + rank),
            }

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

        ranked = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )
        return ranked[:top_k]

    def retrieve_subquestion_evidence(
        self,
        subquestion: str,
        patient_id: str,
        patient_name: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        retrieval_query = prepare_retrieval_query(
            subquestion,
            patient_name=patient_name,
            patient_id=patient_id,
        )
        return self.hybrid_retrieve(
            question=retrieval_query,
            patient_id=patient_id,
            top_k=top_k,
        )

    # ----- generation + validation -----

    def decompose_question(self, question: str) -> list[str]:
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

        response = self.generate_with_mistral(prompt).strip()
        try:
            parsed = json.loads(response)
            if (
                isinstance(parsed, list)
                and parsed
                and all(
                    isinstance(item, str) and item.strip()
                    for item in parsed
                )
            ):
                return [item.strip() for item in parsed][:6]
        except json.JSONDecodeError:
            pass
        return [question]

    def verify_answer_against_evidence(
        self,
        subquestion: str,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
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

        verdict = self.generate_with_mistral(verification_prompt).strip()
        return {
            "passed": verdict.upper().startswith("PASS"),
            "verdict": verdict,
        }

    def answer_subquestion(
        self,
        subquestion: str,
        retrieved_chunks: list[dict[str, Any]],
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        rag_context = build_rag_context(retrieved_chunks)
        allowed_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        allowed_ids_text = (
            "\n".join(allowed_ids) if allowed_ids else "(none)"
        )

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
        integrity: dict[str, Any] | None = None
        coverage: dict[str, Any] | None = None

        for _ in range(1, max_attempts + 1):
            answer = self.generate_with_mistral(prompt)
            integrity = validate_response_integrity(
                answer, retrieved_chunks
            )
            coverage = self.verify_answer_against_evidence(
                subquestion, answer, retrieved_chunks
            )

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

    # ----- public pipeline -----

    def ask(
        self,
        question: str,
        evidence_top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Full clinical RAG pipeline.

        Patient resolution → question decomposition → patient-filtered
        hybrid retrieval → grounded answer per sub-question → verify.
        """
        start_time = time.perf_counter()

        patient = resolve_patient_from_question(
            question, self.patient_registry
        )
        patient_id = patient["patient_id"]
        patient_name = patient["patient_name"]

        subquestions = self.decompose_question(question)
        if not subquestions:
            subquestions = [question]
        if len(subquestions) > 6:
            subquestions = [question]

        subanswers: list[dict[str, Any]] = []
        for subquestion in subquestions:
            retrieved_chunks = self.retrieve_subquestion_evidence(
                subquestion=subquestion,
                patient_id=patient_id,
                patient_name=patient_name,
                top_k=evidence_top_k,
            )
            result = self.answer_subquestion(
                subquestion=subquestion,
                retrieved_chunks=retrieved_chunks,
            )
            subanswers.append(
                {
                    "question": subquestion,
                    "answer": result["answer"],
                    "retrieved_chunks": retrieved_chunks,
                    "integrity_validation": result["integrity_validation"],
                    "coverage_validation": result["coverage_validation"],
                    "fallback_used": result.get("fallback_used", False),
                }
            )

        final_answer = "\n\n".join(
            item["answer"] for item in subanswers
        )

        return {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "original_question": question,
            "subquestions": subquestions,
            "subanswers": subanswers,
            "answer": final_answer,
            "latency_seconds": time.perf_counter() - start_time,
        }

    # Backwards-compatible alias matching the notebook API.
    def ask_clinical_assistant(
        self,
        question: str,
        evidence_top_k: int = 3,
    ) -> dict[str, Any]:
        return self.ask(question, evidence_top_k=evidence_top_k)


# Spec alias used by the FastAPI gateway / hackathon brief.
ClinicalAgent = ClinicalAssistant


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Aegis Clinical RAG Assistant"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Download seed EHR reports and index into ChromaDB",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the Chroma collection before bootstrap",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Clinical question to ask",
    )
    args = parser.parse_args()

    assistant = ClinicalAssistant(reset_collection=False)

    if args.bootstrap:
        print("Bootstrapping Synthetic-EHR-Llama knowledge base...")
        summary = assistant.bootstrap_knowledge_base(reset=args.reset)
        print(json.dumps(summary, indent=2))

    if args.question:
        if not assistant.patient_registry:
            raise SystemExit(
                "No patients indexed. Run with --bootstrap first."
            )
        result = assistant.ask(args.question)
        print("\n=== Clinical Assistant Response ===\n")
        print(f"Patient: {result['patient_name']} ({result['patient_id']})")
        print(f"Latency: {result['latency_seconds']:.2f}s\n")
        print(result["answer"])


if __name__ == "__main__":
    main()
