"""
Parsing helpers for the Clinical Assistant UI.

`ask_clinical_assistant()` (in clinical_agent.py) returns a raw pipeline
result: a dict of subquestions/subanswers, each subanswer's `answer` field
being free-text in the form:

    Status: SUPPORTED

    Answer:
    <text>

    Source Chunks:
    <ids>

These helpers turn that raw structure into clean, UI-ready values without
touching or reinterpreting the underlying RAG logic. Every value returned
here is either read directly from the backend result or is an honest
"not available" placeholder — nothing is invented.

Kept intentionally separate from clinical_agent.py: this file only reads
the pipeline's output, it never calls the pipeline or influences generation.
"""

from __future__ import annotations

import re

STATUS_LINE_PATTERN = re.compile(r"(?im)^Status:\s*(SUPPORTED|NOT_FOUND|ERROR)\s*$")


def extract_status(raw_answer: str) -> str | None:
    """Pull the `Status: ...` line out of one subanswer's raw text, if present."""
    match = STATUS_LINE_PATTERN.search(raw_answer or "")
    return match.group(1).upper() if match else None


def extract_answer_text(raw_answer: str) -> str:
    """
    Pull the human-readable text under `Answer:` out of one subanswer's raw
    text, stopping before the `Source Chunks:` section. Falls back to the
    raw text if the expected format isn't found (e.g. the ERROR fallback
    answer built by clinical_agent.answer_subquestion).
    """
    if not raw_answer:
        return ""

    match = re.search(
        r"(?is)Answer:\s*(.*?)(?:\n\s*Source Chunks(?: Checked)?:|\Z)",
        raw_answer,
    )

    if match:
        return match.group(1).strip()

    return raw_answer.strip()


def get_overall_status(result: dict) -> str:
    """
    Roll every subanswer's status up into one badge for the top-level
    "Grounded Clinical Answer" card:

        SUPPORTED     - every subanswer is SUPPORTED
        NOT FOUND     - every subanswer is NOT_FOUND
        PARTIAL       - a mix of SUPPORTED / NOT_FOUND
        NEEDS REVIEW  - any subanswer is ERROR, or status is unreadable
    """
    subanswers = result.get("subanswers") or []

    if not subanswers:
        return "NEEDS REVIEW"

    statuses = [extract_status(item.get("answer", "")) for item in subanswers]

    if any(status is None or status == "ERROR" for status in statuses):
        return "NEEDS REVIEW"

    if all(status == "SUPPORTED" for status in statuses):
        return "SUPPORTED"

    if all(status == "NOT_FOUND" for status in statuses):
        return "NOT FOUND"

    return "PARTIAL"


def collect_evidence(result: dict) -> list[dict]:
    """
    Flatten every retrieved chunk across all subanswers into one
    deduplicated list for the "Retrieved Evidence" table:

        [{chunk_id, section, preview}, ...]

    Order follows first appearance (subquestion order, then retrieval
    rank), which is also each chunk's relevance order.
    """
    seen_ids: set[str] = set()
    evidence: list[dict] = []

    for subanswer in result.get("subanswers") or []:
        for chunk in subanswer.get("retrieved_chunks") or []:
            chunk_id = chunk.get("chunk_id")

            if not chunk_id or chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)

            text = (chunk.get("text") or "").strip()
            section = (chunk.get("metadata") or {}).get("section", "Unknown")

            # The stored chunk text is prefixed with "Section: <name>" —
            # strip it since the section already has its own table column.
            section_prefix = f"Section: {section}"
            if text.startswith(section_prefix):
                text = text[len(section_prefix):].strip()

            preview = text[:160] + ("..." if len(text) > 160 else "")

            evidence.append({
                "chunk_id": chunk_id,
                "section": section,
                "preview": preview,
            })

    return evidence


def _rollup_validation_field(subanswers: list[dict], get_passed) -> str:
    """
    Shared PASS/FAIL/PENDING rollup for one validation dimension across
    subanswers. PENDING means the field wasn't available on a subanswer.
    """
    if not subanswers:
        return "PENDING"

    results = []

    for subanswer in subanswers:
        passed = get_passed(subanswer)
        if passed is None:
            return "PENDING"
        results.append(passed)

    return "PASS" if all(results) else "FAIL"


def get_validation_results(result: dict) -> dict:
    """
    Roll up citation integrity and evidence coverage across all
    subanswers into the two badges shown on the "Validation" card.

    Only reports PASS/FAIL when the backend actually returned a
    validation dict for every subanswer; otherwise PENDING.
    """
    subanswers = result.get("subanswers") or []

    def citation_passed(subanswer):
        integrity = subanswer.get("integrity_validation")
        if not isinstance(integrity, dict) or "valid" not in integrity:
            return None
        return bool(integrity["valid"])

    def coverage_passed(subanswer):
        coverage = subanswer.get("coverage_validation")
        if not isinstance(coverage, dict) or "passed" not in coverage:
            return None
        return bool(coverage["passed"])

    return {
        "citation_integrity": _rollup_validation_field(subanswers, citation_passed),
        "evidence_coverage": _rollup_validation_field(subanswers, coverage_passed),
    }


def get_safety_decision(result: dict) -> dict:
    """
    Edge Guard is not wired into the pipeline yet, so there is no real
    security decision to report. This reads an optional `security` key
    on the result (the integration point for when Edge Guard is
    connected) and otherwise honestly reports NOT_EVALUATED rather than
    fabricating a SAFE verdict.

    Expected future shape once connected:
        {"decision": "SAFE" | "BLOCK_LOCAL" | "ESCALATE", "reason": str}
    """
    security = result.get("security") if isinstance(result, dict) else None

    if isinstance(security, dict) and security.get("decision") in (
        "SAFE", "BLOCK_LOCAL", "ESCALATE",
    ):
        return {
            "decision": security["decision"],
            "reason": security.get("reason"),
        }

    return {
        "decision": "NOT_EVALUATED",
        "reason": "Security decision unavailable until Edge Guard is connected.",
    }


def parse_clinical_result(result: dict) -> dict:
    """
    Build the full, UI-ready view of one ask_clinical_assistant() result.
    This is the single entry point main.py calls after the pipeline runs.
    """
    subanswers_view = []

    for subanswer in result.get("subanswers") or []:
        raw_answer = subanswer.get("answer", "")

        subanswers_view.append({
            "question": subanswer.get("question", ""),
            "status": extract_status(raw_answer) or "ERROR",
            "answer_text": extract_answer_text(raw_answer),
            "source_chunk_ids": sorted({
                chunk.get("chunk_id")
                for chunk in subanswer.get("retrieved_chunks") or []
                if chunk.get("chunk_id")
            }),
            "fallback_used": bool(subanswer.get("fallback_used", False)),
        })

    return {
        "patient_id": result.get("patient_id"),
        "patient_name": result.get("patient_name"),
        "original_question": result.get("original_question"),
        "overall_status": get_overall_status(result),
        "subanswers": subanswers_view,
        "evidence": collect_evidence(result),
        "validation": get_validation_results(result),
        "safety": get_safety_decision(result),
        "latency_seconds": result.get("latency_seconds"),
    }
