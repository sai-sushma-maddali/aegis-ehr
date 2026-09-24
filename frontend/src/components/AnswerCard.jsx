import { useState } from "react";

const OVERALL_LABEL = {
  SUPPORTED: { text: "Supported by record", className: "status-green" },
  PARTIAL: { text: "Partially supported", className: "status-amber" },
  "NOT FOUND": { text: "Not found in record", className: "status-gray" },
  "NEEDS REVIEW": { text: "Needs review", className: "status-red" },
};

// The RAG splits every question into sub-questions and always asks who the
// patient is. Those identity lookups are already shown in the header, so
// they are left out of the summary.
const IDENTITY_QUESTION = /who is the patient|patient'?s? (name|mrn)|what is the .*mrn/i;

function cleanAnswerText(text) {
  return (text || "")
    .replace(/\s*\(Source Chunk:[^)]*\)/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function buildSummaryPoints(subanswers) {
  const clinical = subanswers.filter((sub) => !IDENTITY_QUESTION.test(sub.question));
  const candidates = clinical.length > 0 ? clinical : subanswers;

  const seen = new Set();
  const points = [];
  for (const sub of candidates) {
    if (sub.status !== "SUPPORTED") continue;
    const text = cleanAnswerText(sub.answer_text);
    if (!text || seen.has(text)) continue;
    seen.add(text);
    points.push(text);
  }
  return points;
}

export default function AnswerCard({ result, isRunning }) {
  const [showBreakdown, setShowBreakdown] = useState(false);

  if (isRunning) {
    return (
      <div className="card">
        <div className="loading-block">
          <div className="spinner" />
          <div>Searching the patient's record and preparing a grounded answer...</div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const overall = OVERALL_LABEL[result.overall_status] || OVERALL_LABEL["NOT FOUND"];
  const points = buildSummaryPoints(result.subanswers);
  const latency =
    typeof result.latency_seconds === "number" ? `${result.latency_seconds.toFixed(1)}s` : null;

  return (
    <div className="card">
      <div className="answer-header">
        <div>
          <div className="answer-patient">
            {result.patient_name} <span className="answer-mrn">MRN {result.patient_id}</span>
          </div>
        </div>
        <div className="answer-header-right">
          <span className={`status-badge ${overall.className}`}>{overall.text}</span>
          {latency && <span className="answer-latency">Answered in {latency}</span>}
        </div>
      </div>

      {(result.overall_status === "NEEDS REVIEW" || result.overall_status === "PARTIAL") && (
        <div className="answer-warning">
          {result.overall_status === "NEEDS REVIEW"
            ? "This answer could not be fully verified against the patient's record. Do not rely on it without checking the evidence below."
            : "Only part of this question could be answered from the patient's record."}
        </div>
      )}

      {points.length > 0 ? (
        <ul className="answer-points">
          {points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      ) : (
        <div className="empty-state">No supported answer was found in this patient's record.</div>
      )}

      <button
        type="button"
        className="details-toggle"
        onClick={() => setShowBreakdown((open) => !open)}
      >
        {showBreakdown ? "Hide" : "Show"} how the question was analysed ({result.subanswers.length}{" "}
        steps)
      </button>

      {showBreakdown &&
        result.subanswers.map((sub, index) => (
          <div className="subanswer-block" key={index}>
            <div className="subanswer-question">{sub.question}</div>
            <div className="subanswer-text">
              {cleanAnswerText(sub.answer_text) || "No answer text was returned."}
            </div>
          </div>
        ))}
    </div>
  );
}
