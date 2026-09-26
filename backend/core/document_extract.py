"""Extract plain text from uploaded clinical attachments."""

from __future__ import annotations

import io
from pathlib import Path

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_CHARS = 40_000


def extract_document_text(filename: str, data: bytes) -> str:
    """Return UTF-8 text from a .txt, .pdf, or .docx upload."""
    if not data:
        raise ValueError("The uploaded document is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document is too large (2 MB limit).")

    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .txt, .pdf, and .docx documents are supported.")

    if suffix == ".txt":
        text = _decode_text(data)
    elif suffix == ".pdf":
        text = _extract_pdf(data)
    else:
        text = _extract_docx(data)

    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        raise ValueError("No readable text was found in that document.")
    if len(cleaned) > MAX_DOCUMENT_CHARS:
        cleaned = cleaned[:MAX_DOCUMENT_CHARS] + "\n…[truncated]"
    return cleaned


def compose_prompt_with_document(question: str, filename: str, document_text: str) -> str:
    """Join the clinician question with the uploaded file for EdgeGuard + RAG."""
    question = (question or "").strip() or "Please review this uploaded document."
    name = Path(filename or "document").name
    body = (document_text or "").strip()
    return (
        f"{question}\n\n"
        f"--- Uploaded document: {name} ---\n"
        f"{body}\n"
        f"--- End of uploaded document ---"
    )


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ValueError("PDF support is not installed on this server.") from exc
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise ValueError("Word document support is not installed on this server.") from exc
    document = Document(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
