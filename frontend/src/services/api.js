// Thin fetch wrapper around the AegisEHR FastAPI backend.
// This file owns HTTP concerns only — no parsing of clinical content
// happens here, that's done server-side in backend/core/response_parser.py
// so the UI always receives an already-clean shape.

const API_BASE_URL = "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function handleResponse(response) {
  if (response.ok) {
    return response.json();
  }

  let detail = `Request failed (${response.status})`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // response body wasn't JSON — keep the generic message
  }

  throw new ApiError(detail, response.status);
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
 */
export async function askClinicalAssistant(patientLabel, question) {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_label: patientLabel || null, question }),
  });
  return handleResponse(response);
}

export { ApiError };

/** True when the backend answers its health check. */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}
