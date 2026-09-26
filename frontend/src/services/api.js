// Thin fetch wrapper around the Aegis EHR FastAPI gateway.
// Questions go through /api/ask, which screens them before retrieval.

const API_BASE_URL = "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(message, status, notice = null) {
    super(message);
    this.status = status;
    this.notice = notice;
  }
}

const BLOCK_NOTICES = {
  "Blocked by Edge Intent Triage": {
    kicker: "Tier 2: On-Device Intent Guard (Qwen2.5-7B)",
    title: "Request blocked",
    summary:
      "Tier 2: On-Device Intent Guard classified this message as a prompt injection and stopped it before any chart was opened.",
  },
  "Blocked at Vector Cache": {
    kicker: "Tier 1: Fast Vector Threat Cache (ChromaDB)",
    title: "Request blocked",
    summary:
      "Tier 1: Fast Vector Threat Cache matched a known attack pattern and stopped this message before Tier 2 ran.",
  },
  "Blocked by Cloud Defender": {
    kicker: "Tier 3: Cloud Forensic Defender (Gemma 4)",
    title: "Request blocked",
    summary:
      "Tier 3: Cloud Forensic Defender flagged this message, so no chart was opened.",
  },
};

async function handleResponse(response) {
  if (response.ok) {
    return response.json();
  }

  let detail = `Request failed (${response.status})`;
  let notice = null;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
      notice = BLOCK_NOTICES[detail] || null;
    } else if (body?.detail && typeof body.detail === "object") {
      notice = body.detail;
      detail = notice.title || notice.label || detail;
    }
  } catch {
    // response body wasn't JSON — keep the generic message
  }

  throw new ApiError(detail, response.status, notice);
}

/** Real patient registry from the backend (built from indexed ChromaDB metadata). */
export async function fetchPatients() {
  const response = await fetch(`${API_BASE_URL}/api/patients`);
  return handleResponse(response);
}

/**
 * Ask the clinical assistant a question about one patient.
 * patientLabel is optional, e.g. "Elizabeth Brown (MRN L0)"; without it the
 * backend finds the patient named in the question.
 * document is an optional File (.txt / .pdf / .docx) for poisoned-doc tests.
 */
export async function askClinicalAssistant(
  patientLabel,
  question,
  history = [],
  document = null,
) {
  if (document) {
    const form = new FormData();
    form.append("question", question || "");
    if (patientLabel) form.append("patient_label", patientLabel);
    form.append("history", JSON.stringify(history || []));
    form.append("document", document, document.name);
    const response = await fetch(`${API_BASE_URL}/api/ask`, {
      method: "POST",
      body: form,
    });
    return handleResponse(response);
  }

  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_label: patientLabel || null,
      question,
      history,
    }),
  });
  return handleResponse(response);
}

export { ApiError };

export async function fetchActivity() {
  const response = await fetch(`${API_BASE_URL}/api/activity`);
  return handleResponse(response);
}

export async function fetchTelemetry() {
  const response = await fetch(`${API_BASE_URL}/api/telemetry`);
  return handleResponse(response);
}

export async function clearDashboard() {
  const response = await fetch(`${API_BASE_URL}/api/dashboard/clear`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function fetchThreatSummary() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/threats`);
    if (!response.ok) return null;
    const body = await response.json();
    const count = Number(body?.count);
    return Number.isFinite(count) ? count : null;
  } catch {
    return null;
  }
}

/** True when the backend answers its health check. */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}
