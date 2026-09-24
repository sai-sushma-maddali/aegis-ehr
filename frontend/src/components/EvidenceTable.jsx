import { useState } from "react";

const VISIBLE_BY_DEFAULT = 3;

export default function EvidenceTable({ result, isRunning }) {
  const [showAll, setShowAll] = useState(false);

  if (isRunning || !result) return null;

  const evidence = result.evidence || [];
  const shown = showAll ? evidence : evidence.slice(0, VISIBLE_BY_DEFAULT);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Evidence from the patient's record</div>
          <div className="card-subtitle">The passages the answer is based on</div>
        </div>
        <span className="evidence-count">{evidence.length} passages</span>
      </div>

      {evidence.length === 0 ? (
        <div className="empty-state">No matching evidence was retrieved for this question.</div>
      ) : (
        <div className="evidence-list">
          {shown.map((item) => (
            <div className="evidence-item" key={item.chunk_id} title={item.chunk_id}>
              <span className="section-tag">{item.section}</span>
              <div className="evidence-text">{item.preview}</div>
            </div>
          ))}
        </div>
      )}

      {evidence.length > VISIBLE_BY_DEFAULT && (
        <button
          type="button"
          className="details-toggle"
          style={{ marginTop: 12 }}
          onClick={() => setShowAll((open) => !open)}
        >
          {showAll ? "Show fewer" : `Show all ${evidence.length} passages`}
        </button>
      )}
    </div>
  );
}
