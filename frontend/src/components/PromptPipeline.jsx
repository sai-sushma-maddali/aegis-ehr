function formatStepMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  if (value >= 100) return `${Math.round(value).toLocaleString()} ms`;
  return `${value.toFixed(1)} ms`;
}

function step(id, title, status, tone, latency) {
  return { id, title, status, tone, latency };
}

export function pipelineNodes(event) {
  if (!event) return [];
  const source = event.source_layer;
  const blocked = event.status === "BLOCKED";
  const nodes = [step("ingest", "Request Ingestion", "Completed", "green", null)];

  const phiExposed = event.phi_exposed === true;
  nodes.push(
    step(
      "phi",
      "PHI Sanitizer Check",
      phiExposed ? "PHI left the device" : event.phi_exposed === false ? "Zero-egress verified" : "Completed",
      phiExposed ? "red" : "green",
      null,
    ),
  );

  const cacheHit = source === "EDGE_VECTOR_CACHE" && blocked;
  const vectorMs = formatStepMs(
    event.vector_latency_ms ?? (cacheHit ? event.latency_ms : null),
  );
  nodes.push(
    step(
      "tier1",
      "Tier 1: Fast Vector Threat Cache",
      cacheHit ? "Intercepted" : "Miss",
      cacheHit ? "red" : "gray",
      vectorMs,
    ),
  );
  if (cacheHit) {
    nodes.push(step("block", "Block & Return", "Completed", "red", null));
    return nodes;
  }

  const classMs = formatStepMs(event.classifier_latency_ms);
  const cloudPath = source === "CLOUD_DEFENDER_HOTPATCH" || String(event.outcome || "").startsWith("cloud");
  if (source === "EDGE_INTENT_TRIAGE" && blocked) {
    nodes.push(step("tier2", "Tier 2: On-Device Intent Guard", "Block", "red", classMs));
    nodes.push(step("block", "Block & Return", "Completed", "red", null));
    return nodes;
  }
  if (cloudPath) {
    const cloudBlocked = event.outcome === "cloud_block" || blocked;
    nodes.push(step("tier2", "Tier 2: On-Device Intent Guard", "Escalate", "amber", classMs));
    nodes.push(
      step(
        "tier3",
        "Tier 3: Cloud Forensic Defender",
        cloudBlocked ? "Block" : "Pass",
        cloudBlocked ? "red" : "green",
        formatStepMs(event.cloud_latency_ms),
      ),
    );
    if (event.hot_patched) {
      nodes.push(step("patch", "Synthesize Hot-Patch", "Completed", "violet", null));
      nodes.push(step("chroma", "ChromaDB Ingestion", "Completed", "violet", null));
    }
    if (cloudBlocked) {
      nodes.push(step("block", "Block & Return", "Completed", "red", null));
    } else {
      nodes.push(step("rag", "Clinical RAG Assistant (Mistral-7B)", "Completed", "sky", null));
    }
    return nodes;
  }

  nodes.push(step("tier2", "Tier 2: On-Device Intent Guard", "Pass", "green", classMs));
  nodes.push(step("rag", "Clinical RAG Assistant (Mistral-7B)", "Completed", "sky", null));
  return nodes;
}

function Arrow() {
  return (
    <svg className="pipe-arrow" viewBox="0 0 28 16" aria-hidden="true">
      <path d="M1 8 H18" />
      <path d="M16 3 L25 8 L16 13" />
    </svg>
  );
}

export default function PromptPipeline({ event }) {
  const nodes = pipelineNodes(event);
  if (nodes.length === 0) return null;

  return (
    <div className="pipe" aria-label="Prompt execution path">
      {nodes.map((node, index) => (
        <div className="pipe-step" key={node.id}>
          {index > 0 ? <Arrow /> : null}
          <div className={`pipe-node pipe-${node.tone}`}>
            <span className={`pipe-ring pipe-ring-${node.tone}`} />
            <span className="pipe-title">{node.title}</span>
            <span className="pipe-status">{node.status}</span>
            {node.latency ? <span className="pipe-latency">{node.latency}</span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}
