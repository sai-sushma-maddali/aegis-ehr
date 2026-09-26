import { useEffect, useState } from "react";
import {
  Cpu,
  FileText,
  RefreshCw,
  Shield,
  Zap,
} from "lucide-react";
import { fetchTelemetry } from "../services/api.js";
import { TIER_BY_SOURCE } from "../tierNames.js";

function StatusPill({ children, className }) {
  return <span className={`md-pill ${className || ""}`}>{children}</span>;
}

function formatMs(value) {
  if (value == null || Number.isNaN(Number(value))) return "No samples";
  return `${Number(value).toFixed(1)} ms`;
}

function formatPct(rate) {
  if (rate == null || Number.isNaN(Number(rate))) return "No samples";
  return `${(Number(rate) * 100).toFixed(1)}%`;
}

function speedup(edgeMs, cloudMs) {
  const edge = Number(edgeMs);
  const cloud = Number(cloudMs);
  if (!Number.isFinite(edge) || !Number.isFinite(cloud) || edge <= 0 || cloud <= edge) {
    return null;
  }
  return {
    factor: cloud / edge,
    dropPct: (1 - edge / cloud) * 100,
  };
}

function formatFactor(factor) {
  if (factor >= 10) return `${Math.round(factor)}x`;
  return `${factor.toFixed(1)}x`;
}

function tierTitle(id) {
  if (id === "vector") return TIER_BY_SOURCE.EDGE_VECTOR_CACHE;
  if (id === "classifier") return TIER_BY_SOURCE.EDGE_INTENT_TRIAGE;
  if (id === "cloud") return TIER_BY_SOURCE.CLOUD_DEFENDER_HOTPATCH;
  return id;
}

function displayAttackClass(row) {
  if (row?.attack_class !== "UNKNOWN") return row?.attack_class || "Unclassified";
  if (row?.source_layer === "EDGE_INTENT_TRIAGE") return "HEURISTIC_OVERRIDE";
  return "ADVERSARIAL_PAYLOAD";
}

function displayTierLabel(row, hotPatches) {
  const base = TIER_BY_SOURCE[row?.source_layer] || row?.tier_label;
  const patched =
    row?.status === "HOT-PATCHED" ||
    hotPatches.some((patch) => patch.attack_class === row?.attack_class);
  if (row?.attack_class === "TOOL_EXFILTRATION" && patched && row?.source_layer === "CLOUD_DEFENDER_HOTPATCH") {
    return `${base} + hot-patch write`;
  }
  return base;
}

function statusClass(status) {
  const label = String(status || "").toUpperCase();
  if (label === "PASSED") return "md-pill-green";
  if (label === "BLOCKED") return "md-pill-red";
  if (label.includes("HOT")) return "md-pill-violet";
  return "md-pill-amber";
}

function formatPlainMs(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  const amount = Number(value);
  if (amount >= 100) return `${Math.round(amount).toLocaleString()} ms`;
  return `${amount.toFixed(1)} ms`;
}

function mean(values) {
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function classEvents(events, attackClass) {
  return (events || []).filter((event) => event.attack_class === attackClass);
}

function adaptationFor(events, attackClass) {
  const bucket = classEvents(events, attackClass);
  const initial = mean(
    bucket
      .filter((event) => event.outcome === "cloud_block")
      .map((event) => event.cloud_latency_ms)
      .filter((value) => typeof value === "number"),
  );
  const replay = mean(
    bucket
      .filter((event) => event.source_layer === "EDGE_VECTOR_CACHE" && event.status === "BLOCKED")
      .map((event) => event.vector_latency_ms ?? event.latency_ms)
      .filter((value) => typeof value === "number"),
  );
  if (initial == null || replay == null || replay <= 0 || initial <= replay) return null;
  return { initial, replay, factor: initial / replay };
}

function signatureBadges(events, attackClass) {
  if (!attackClass || attackClass === "Benign") return [];
  const ids = [
    ...new Set(
      classEvents(events, attackClass)
        .map((event) => event.signature_id)
        .filter(Boolean),
    ),
  ];
  return ids
    .map((id) => ({
      id,
      short: id.slice(0, 12),
      hits: (events || []).filter(
        (event) =>
          event.signature_id === id &&
          event.source_layer === "EDGE_VECTOR_CACHE" &&
          event.status === "BLOCKED",
      ).length,
    }))
    .filter(
      (item) =>
        item.hits > 0 ||
        (events || []).some((event) => event.signature_id === item.id && event.hot_patched),
    );
}

function tierSubtitle(tier, tiers, hotPatchCount) {
  const total = tiers.reduce((sum, item) => sum + (item.decisions || 0), 0);
  const count = tier.decisions || 0;
  if (!total) return "No decisions yet";
  const share = `${count} Decision${count === 1 ? "" : "s"} (${((count / total) * 100).toFixed(1)}% traffic share)`;
  const egress = `${tier.cloud_calls || 0} cloud egress`;
  if (tier.id === "cloud") {
    return hotPatchCount > 0 ? `${share} • ${egress} • Hot-patch synthesized` : `${share} • ${egress}`;
  }
  return `${share} • ${egress} • $0.00000 marginal cost`;
}

export default function MetricsDashboard({
  events = [],
  revision = 0,
  clearing = false,
  clearError = "",
  onClear,
}) {
  const [telemetry, setTelemetry] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const payload = await fetchTelemetry();
        if (!cancelled) {
          setTelemetry(payload);
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err?.message || "Telemetry is unavailable.");
        }
      }
    }

    if (revision > 0) setTelemetry(null);
    load();
    const timer = setInterval(load, 4000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [revision]);

  const kpis = telemetry?.kpis || {};
  const signatures = telemetry?.signatures;
  const engine = telemetry?.engine || null;
  const tiers = telemetry?.tiers || [];
  const attackRows = telemetry?.attack_rows || [];
  const hotPatches = telemetry?.hot_patches || [];

  const cards = [
    {
      id: "tier1",
      label: "Tier 1 Intercept Latency",
      value: formatMs(kpis.vector_latency_ms),
      detail:
        kpis.vector_samples > 0
          ? "Mean vector cache match time (< 0.18 cosine distance)."
          : "No vector-cache timings recorded yet",
      icon: Zap,
      accent: "md-accent-green",
    },
    {
      id: "tier2",
      label: "Tier 2 Local Triage Latency",
      value: formatMs(kpis.classifier_latency_ms),
      detail:
        kpis.classifier_samples > 0
          ? "Single forward pass classification (BF16 on-device)."
          : "No classifier timings recorded yet",
      icon: Cpu,
      accent: "md-accent-sky",
    },
    {
      id: "phi",
      label: "Data Sovereignty (HIPAA Audit)",
      value:
        kpis.phi_exposure_rate == null
          ? "No samples"
          : `${formatPct(kpis.phi_exposure_rate)} PHI Exposure`,
      detail:
        kpis.phi_audited > 0
          ? `${kpis.phi_exposed} / ${kpis.phi_audited} audited queries transmitted raw patient identifiers.`
          : "No audited queries yet",
      icon: Shield,
      accent: "md-accent-green",
    },
    {
      id: "patch",
      label: "Autonomous Hot-Patch Rate",
      value:
        kpis.self_heal_rate == null
          ? "No samples"
          : `${formatPct(kpis.self_heal_rate)} Closed Loop`,
      detail:
        kpis.self_heal_cloud_attacks > 0
          ? `${kpis.self_heal_patched} / ${kpis.self_heal_cloud_attacks} novel zero-days persisted to ChromaDB for sub-10ms replay.`
          : "No cloud blocks recorded yet",
      icon: RefreshCw,
      accent: "md-accent-violet",
    },
  ];

  const absorption =
    telemetry?.local_absorption_rate == null
      ? "No blocked prompts recorded yet."
      : `${formatPct(telemetry.local_absorption_rate)} of blocked prompts were stopped on-site (${telemetry.local_blocks} of ${telemetry.attack_blocks}).`;

  const memoryPct = engine?.memory_percent;
  const vectorTier = tiers.find((tier) => tier.id === "vector");
  const cloudTier = tiers.find((tier) => tier.id === "cloud");
  const tierSpeed = speedup(vectorTier?.latency_ms, cloudTier?.latency_ms);

  return (
    <section className="md-shell">
      <div className="md-header">
        <div>
          <p className="md-kicker">HP ZGX Nano</p>
          <h2 className="md-title">Aegis defense telemetry</h2>
        </div>
        <div className="md-header-actions">
          <button
            type="button"
            className="md-btn md-btn-slate"
            onClick={onClear}
            disabled={clearing || !onClear}
          >
            {clearing ? "Clearing…" : "Clear dashboard"}
          </button>
          {clearError ? <p className="md-hint">{clearError}</p> : null}
        </div>
      </div>

      {error ? <p className="md-hint">{error}</p> : null}

      <div className="md-kpi-grid">
        {cards.map((card) => {
          const Icon = card.icon;
          const phiClear = card.id === "phi" && kpis.phi_exposure_rate === 0;
          const healed = card.id === "patch" && kpis.self_heal_rate === 1;
          return (
            <article key={card.id} className="md-card">
              <div className={`md-icon ${card.accent}`}>
                <Icon size={16} />
              </div>
              <div className={`md-value ${phiClear ? "md-value-emerald" : ""} ${healed ? "md-value-violet" : ""}`}>
                {card.value}
              </div>
              {card.id === "tier1" && tierSpeed ? (
                <span className="md-pill md-pill-green md-impact-pill">
                  ⚡ {formatFactor(tierSpeed.factor)} vs. Cloud
                </span>
              ) : null}
              {card.id === "tier2" && kpis.classifier_samples > 0 ? (
                <span className="md-pill md-pill-sky md-impact-pill">Qwen2.5-7B head</span>
              ) : null}
              <div className="md-label">{card.label}</div>
              <p className="md-detail">{card.detail}</p>
            </article>
          );
        })}
      </div>

      <div className="md-split">
        <article className="md-panel">
          <h3 className="md-panel-title">Multi-tier defense breakdown</h3>
          <div className="md-tier-list">
            {tiers.length === 0 ? (
              <p className="md-hint">No recorded decisions yet.</p>
            ) : (
              tiers.map((tier) => (
                <div key={tier.id} className="md-tier">
                  <div className="md-tier-top">
                    <div className="md-tier-name">{tierTitle(tier.id) || tier.name}</div>
                  </div>
                  <div className="md-tier-meta">
                    <span>{tierSubtitle(tier, tiers, hotPatches.length)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
          <p className="md-callout">{absorption}</p>
        </article>

        <article className="md-panel">
          <h3 className="md-panel-title">HP ZGX Nano telemetry</h3>
          <dl className="md-telemetry">
            <div>
              <div className="md-tele-row">
                <dt>
                  <Cpu size={16} />
                  Unified memory (128 GB LPDDR5x)
                </dt>
                <dd>
                  {engine?.system_memory_used_gb == null
                    ? "Not readable"
                    : `${Number(engine.system_memory_used_gb).toFixed(1)} GB / ${Number(engine.system_memory_total_gb).toFixed(1)} GB`}
                </dd>
              </div>
              <div className="md-bar md-bar-thick">
                <div
                  className="md-bar-fill md-bar-green"
                  style={{ width: `${memoryPct == null ? 0 : Math.min(Number(memoryPct), 100).toFixed(1)}%` }}
                />
              </div>
              <p className="md-hint">
                {memoryPct == null
                  ? "Memory usage was not readable on this machine."
                  : `${Number(memoryPct).toFixed(1)}%. Shared between on-device guard, Mistral-7B, ChromaDB, and OS. No discrete GPU framebuffer.`}
              </p>
            </div>
            <div className="md-tele-row">
              <dt>
                <Zap size={16} />
                Clinical Streaming TTFT (P50)
              </dt>
              <dd>
                {engine?.ttft_p50_ms == null ? "No ZRT samples" : `${Number(engine.ttft_p50_ms).toFixed(1)} ms`}
                <span>
                  {engine?.ttft_p90_ms == null
                    ? "Time-to-first-token for local Mistral-7B assistant"
                    : `Time-to-first-token for local Mistral-7B assistant (P90: ${Number(engine.ttft_p90_ms).toFixed(1)} ms | P99: ${Number(engine.ttft_p99_ms).toFixed(1)} ms)`}
                </span>
              </dd>
            </div>
            <div className="md-tele-row">
              <dt>
                <FileText size={16} />
                Prompt Token Screening
              </dt>
              <dd>
                {engine?.prompt_tokens_total == null
                  ? "No ZRT samples"
                  : `${Number(engine.prompt_tokens_total).toLocaleString()} tokens`}
                <span>
                  {engine?.requests_succeeded == null
                    ? "Screened prompts from the local Mistral session"
                    : `Screened across ${engine.requests_succeeded} completed request${engine.requests_succeeded === 1 ? "" : "s"} (${Number(engine.generated_tokens_total || 0).toLocaleString()} generation tokens produced)`}
                </span>
              </dd>
            </div>
            <div className="md-tele-row">
              <dt>
                <Cpu size={16} />
                Compute Architecture & Load
              </dt>
              <dd>
                {engine?.system_cpu_percent == null && engine?.system_gpu_percent == null
                  ? "Not readable"
                  : `${engine?.system_cpu_percent == null ? "—" : `${Number(engine.system_cpu_percent).toFixed(1)}% CPU`} • ${engine?.system_gpu_percent == null ? "—" : `${Number(engine.system_gpu_percent).toFixed(0)}% GPU`}`}
                <span>
                  NVIDIA GB10 Grace Blackwell (20-core Arm, 1,000 TOPS FP4 rating
                  {engine?.package_temp_c == null ? "" : `, ${Number(engine.package_temp_c).toFixed(0)}°C package`})
                </span>
              </dd>
            </div>
            {engine?.e2e_latency_p50_s == null ? null : (
              <p className="md-hint">
                Clinical Synthesis Duration: {Number(engine.e2e_latency_p50_s).toFixed(2)}s P50 (full medical note generation)
              </p>
            )}
          </dl>
        </article>
      </div>

      <article className="md-panel md-table-panel">
        <h3 className="md-panel-title">Recorded prompt classes</h3>
        <div className="md-table-wrap">
          <table className="md-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Recorded prompts</th>
                <th>Deciding tier</th>
                <th>Adaptation</th>
                <th>Active signature / cache hits</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {attackRows.length === 0 ? (
                <tr>
                  <td colSpan={6}>No prompts recorded yet.</td>
                </tr>
              ) : (
                attackRows.map((row) => {
                  const adapt = events.length ? adaptationFor(events, row.attack_class) : null;
                  const badges = events.length ? signatureBadges(events, row.attack_class) : [];
                  const flow =
                    adapt != null
                      ? "Tier 3 Escalation → Tier 1 Hot-Patch"
                      : displayTierLabel(row, hotPatches);
                  return (
                    <tr key={row.attack_class}>
                      <td className="md-strong">{displayAttackClass(row)}</td>
                      <td>{row.samples}</td>
                      <td>{flow}</td>
                      <td>
                        {adapt ? (
                          <span className="md-adapt">
                            Initial: {formatPlainMs(adapt.initial)} (Cloud) → Replay: {formatPlainMs(adapt.replay)} (Tier 1 Cache)
                            <span className="md-pill md-pill-green">⚡ {Math.round(adapt.factor).toLocaleString()}x speedup</span>
                          </span>
                        ) : events.length ? (
                          "—"
                        ) : (
                          formatMs(row.latency_ms)
                        )}
                      </td>
                      <td>
                        {badges.length === 0 ? (
                          "—"
                        ) : (
                          badges.map((badge) => (
                            <span className="md-sig" key={badge.id}>
                              {badge.short} ({badge.hits} hit{badge.hits === 1 ? "" : "s"})
                            </span>
                          ))
                        )}
                      </td>
                      <td>
                        <StatusPill className={statusClass(row.status)}>
                          {row.status}
                        </StatusPill>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </article>

      <article className="md-panel">
        <div className="md-demo-head">
          <div>
            <h3 className="md-panel-title">Stored hot-patches</h3>
            <p className="md-hint">
              Signatures written after Tier 3: Cloud Forensic Defender blocks a prompt, then matched by Tier 1.
            </p>
          </div>
          <StatusPill className="md-pill-slate">
            {signatures?.available ? `${signatures.hot_patch} hot-patch` : "Unavailable"}
          </StatusPill>
        </div>
        <div className="md-log">
          {hotPatches.length === 0 ? (
            <span>No hot-patch signatures are stored.</span>
          ) : (
            hotPatches.map((patch) => (
              <div key={patch.signature_id}>
                {patch.signature_id} · {displayAttackClass(patch)} ·{" "}
                {patch.later_vector_blocks} later vector-cache hit
                {patch.later_vector_blocks === 1 ? "" : "s"}
                {patch.created_at ? ` · ${patch.created_at}` : ""}
              </div>
            ))
          )}
        </div>
      </article>
    </section>
  );
}
