import { useState } from "react";

const SOURCE_LIMIT = 3;

function matchPercent(score) {
  return typeof score === "number" ? `${Math.round(score * 100)}% match` : null;
}

export default function EvidenceTable({ result, isRunning }) {
  const [open, setOpen] = useState(false);

  if (isRunning || !result || result.searched === false) return null;

  const sources = (result.evidence || []).slice(0, SOURCE_LIMIT);
  if (sources.length === 0) return null;

  return (
    <div className="sources">
      <button type="button" className="details-toggle" onClick={() => setOpen((current) => !current)}>
        {open ? "Hide sources" : `Sources (${sources.length})`}
      </button>

      {open && (
        <ol className="source-list">
          {sources.map((item, index) => {
            const match = matchPercent(item.score);
            return (
              <li className="source-item" key={item.chunk_id || index}>
                <div className="source-cite">
                  <span className="source-index">{index + 1}</span>
                  <span className="section-tag">{item.section}</span>
                  {item.patient_name ? <span>{item.patient_name}</span> : null}
                  {match ? <span>{match}</span> : null}
                  <span className="source-id">{item.chunk_id}</span>
                </div>
                <div className="evidence-text">{item.preview}</div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
