// Thin fetch wrapper around the Aegis EHR FastAPI gateway.
// Questions go through /api/ask, which screens them before retrieval.
//
// Use a relative base by default so Vite's /api proxy works. That way
// SSH port-forward of the UI port alone reaches the backend on the host.
// Ports/hosts come from ../aegis.config.json (see vite.config.js).
// Override with VITE_API_BASE_URL only when the UI and API are on
// different hosts (e.g. production split deploy).

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

/** Clinical ask can include guard + multi-step RAG; fail loudly instead of spinning forever. */
const ASK_TIMEOUT_MS = Number(import.meta.env.VITE_ASK_TIMEOUT_MS || 180_000);

class ApiError extends Error {
  constructor(message, status, notice = null) {
    super(message);
    this.status = status;
    this.notice = notice;
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = ASK_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new ApiError(
        `No response within ${Math.round(timeoutMs / 1000)}s. ` +
          "Check that the FastAPI gateway and clinical LLM (see aegis.config.json) " +
          "are running on the host, and that Vite is proxying /api.",
        408,
      );
    }
    throw new ApiError(
      "Cannot reach the Aegis backend. If you are port-forwarding, forward " +
        "only the Vite UI port (with /api proxy) or set VITE_API_BASE_URL to the " +
        "forwarded API origin.",
      0,
    );
  } finally {
    clearTimeout(timer);
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
  const response = await fetchWithTimeout(`${API_BASE_URL}/api/patients`, {}, 15_000);
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
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/ask`, {
      method: "POST",
      body: form,
    });
    return handleResponse(response);
  }

  const response = await fetchWithTimeout(`${API_BASE_URL}/api/ask`, {
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
  const response = await fetchWithTimeout(`${API_BASE_URL}/api/activity`, {}, 15_000);
  return handleResponse(response);
}

export async function fetchTelemetry() {
  const response = await fetchWithTimeout(`${API_BASE_URL}/api/telemetry`, {}, 15_000);
  return handleResponse(response);
}

export async function clearDashboard() {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/dashboard/clear`,
    { method: "POST" },
    15_000,
  );
  return handleResponse(response);
}

export async function fetchThreatSummary() {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/threats`, {}, 15_000);
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
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/health`, {}, 8_000);
    return response.ok;
  } catch {
    return false;
  }
}
