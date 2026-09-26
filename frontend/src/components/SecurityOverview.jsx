import { useState } from "react";
import { TIER1, TIER2, TIER3 } from "../tierNames.js";
import PromptPipeline from "./PromptPipeline.jsx";

function outcomeClass(outcome) {
  if (outcome === "edge_pass" || outcome === "cloud_pass") return "status-green";
  if (outcome === "edge_block" || outcome === "vector_block" || outcome === "cloud_block") return "status-red";
  return "status-amber";
}

function presentEvent(event) {
  if (event.source_layer === "EDGE_VECTOR_CACHE" && event.status === "BLOCKED") {
    return {
      outcome: "vector_block",
      label: `Blocked by ${TIER1}`,
      summary:
        "Tier 1: Fast Vector Threat Cache matched a stored attack signature and stopped this prompt before Tier 2 ran.",
    };
  }
  if (event.source_layer === "EDGE_INTENT_TRIAGE" && event.status === "BLOCKED") {
    return {
      outcome: event.outcome || "edge_block",
      label: `Blocked by ${TIER2}`,
      summary:
        "Tier 2: On-Device Intent Guard classified this prompt as a prompt injection and stopped it before any chart was opened.",
    };
  }
  if (event.outcome === "cloud_block") {
    return {
      outcome: "cloud_block",
      label: `Blocked by ${TIER3}`,
      summary: event.summary,
    };
  }
  if (String(event.outcome || "").startsWith("cloud")) {
    return {
      outcome: event.outcome,
      label: `Escalated to ${TIER3}`,
      summary: event.summary,
    };
  }
  if (event.outcome === "edge_pass" || event.status === "PASSED") {
    return {
      outcome: event.outcome || "edge_pass",
      label: `Cleared by ${TIER2}`,
      summary: event.summary,
    };
  }
  return {
    outcome: event.outcome,
    label: event.outcome_label,
    summary: event.summary,
  };
}

function formatHistoryMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  if (value >= 100) return `${Math.round(value).toLocaleString()} ms`;
  return `${value.toFixed(1)} ms`;
}

function cloudPartner(event, events) {
  const signatureId = event?.signature_id;
  if (!signatureId) return null;
  return events.find(
    (other) =>
      other.signature_id === signatureId &&
      other.id !== event.id &&
      (other.outcome === "cloud_block" || other.hot_patched),
  );
}

function proofForEvent(event, events) {
  const signatureId = event?.signature_id;
  if (!signatureId) return null;
  const cacheHit = event.source_layer === "EDGE_VECTOR_CACHE" && event.status === "BLOCKED";
  const cloudBlock = event.outcome === "cloud_block" || (event.hot_patched && String(event.outcome || "").startsWith("cloud"));
  if (cacheHit) {
    const partner = cloudPartner(event, events);
    if (!partner) return null;
    const cloudMs = partner.cloud_latency_ms ?? partner.latency_ms;
    const cacheMs = event.latency_ms;
    const factor =
      typeof cloudMs === "number" && typeof cacheMs === "number" && cacheMs > 0 && cloudMs > cacheMs
        ? cloudMs / cacheMs
        : null;
    const speed = factor == null ? "" : ` • ${factor >= 10 ? Math.round(factor).toLocaleString() : factor.toFixed(1)}x speedup`;
    return {
      badge: `Matched ${signatureId.slice(0, 12)}${speed}`,
      link: "↳ Autonomous hot-patch hit (0ms cloud egress)",
    };
  }
  if (cloudBlock && event.hot_patched) {
    return {
      badge: `Hot-Patch Synthesized: ${signatureId.slice(0, 12)}`,
      link: null,
    };
  }
  return null;
}

function formatWhen(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function PromptDetails({ event }) {
  const view = presentEvent(event);
  const confidence =
    typeof event.confidence === "number" ? `${Math.round(event.confidence * 100)}% confidence` : null;
  const latency = formatHistoryMs(event.cloud_latency_ms ?? event.latency_ms);

  return (
    <div className="activity-details">
      <p className="activity-prompt">{event.prompt}</p>
      <p className="activity-summary">{view.summary}</p>
      <div className="activity-meta">
        {confidence && <span>{confidence}</span>}
        {latency && <span>{latency}</span>}
      </div>
      <PromptPipeline event={event} />
    </div>
  );
}

export default function SecurityOverview({ events = [], unavailable = false }) {
  const [openedId, setOpenedId] = useState(null);

  return (
    <section className="security-overview">
      <div className="dashboard-intro">
        <div className="dashboard-title">Security Overview</div>
        <p>Every prompt is kept here, newest first, with the path it followed.</p>
      </div>

      {unavailable && events.length === 0 ? (
        <div className="empty-state">The activity log is not reachable yet.</div>
      ) : events.length === 0 ? (
        <div className="empty-state">No prompts yet. Ask in Clinical Assistant and the result will show up here.</div>
      ) : (
        <div className="activity-list">
          {events.map((event, index) => {
            const open = index === 0 || openedId === event.id;
            const view = presentEvent(event);
            const proof = proofForEvent(event, events);
            const latency = formatHistoryMs(
              event.source_layer === "CLOUD_DEFENDER_HOTPATCH"
                ? event.cloud_latency_ms ?? event.latency_ms
                : event.latency_ms,
            );
            return (
              <article className="activity-item" key={event.id || index}>
                <button
                  type="button"
                  className="activity-row"
                  onClick={() => setOpenedId(open && index !== 0 ? null : event.id)}
                >
                  <span className={`status-badge ${outcomeClass(view.outcome)}`}>{view.label}</span>
                  {latency && <span className="activity-latency">{latency}</span>}
                  <span className="activity-when">{formatWhen(event.timestamp)}</span>
                  {index !== 0 && <span className="activity-preview">{event.prompt}</span>}
                </button>
                {proof && (
                  <div className="activity-proof">
                    <span className="md-pill md-pill-violet">{proof.badge}</span>
                    {proof.link ? <span className="activity-proof-link">{proof.link}</span> : null}
                  </div>
                )}
                {open && <PromptDetails event={event} />}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
