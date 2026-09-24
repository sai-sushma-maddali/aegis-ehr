// Every metric is "Not connected yet" on purpose: the services that measure
// them (signature DB, Edge Guard, Cloud Defender) do not exist yet, and the
// dashboard must never show invented values.

const BLOCKS = [
  {
    title: "Edge vs Cloud",
    subtitle: "Where each request was handled",
    metrics: [
      { label: "Queries Screened", hint: "Every prompt checked before it reaches the assistant" },
      { label: "Handled Locally", hint: "Resolved on-site, without leaving the hospital network" },
      { label: "Cloud Escalations", hint: "Uncertain prompts sent for expert review" },
      { label: "Escalation Ratio", hint: "Cloud requests as a share of all requests" },
    ],
  },
  {
    title: "Threat Protection",
    subtitle: "How well attacks are being stopped",
    metrics: [
      { label: "Attacks Blocked", hint: "Malicious prompts stopped before reaching patient data" },
      { label: "Attack Catch Rate", hint: "Share of real attacks that were stopped" },
      { label: "False Positive Rate", hint: "Legitimate questions wrongly blocked" },
      { label: "Attack Success Rate", hint: "Attacks that got through" },
    ],
  },
  {
    title: "Privacy",
    subtitle: "What leaves the hospital",
    metrics: [
      { label: "Patient Data Sent to Cloud", hint: "Identifiable patient information leaving the site" },
      { label: "Tokens Sent to Cloud", hint: "Amount of text shared for cloud review" },
    ],
  },
  {
    title: "Speed",
    subtitle: "Time added by security screening",
    metrics: [
      { label: "Average Edge Latency", hint: "Local screening time per prompt" },
      { label: "Average Cloud Latency", hint: "Round trip for escalated prompts" },
    ],
  },
  {
    title: "Self-Healing",
    subtitle: "Learning from new attacks",
    metrics: [
      { label: "Signatures Learned", hint: "New attack patterns stored locally" },
      { label: "Replays Blocked Locally", hint: "Repeat attacks caught without a cloud call" },
      { label: "Self-Healing Rate", hint: "Share of replayed new attacks blocked locally" },
    ],
  },
];

function MetricBlock({ block }) {
  return (
    <section className="card metric-block">
      <div className="card-title">{block.title}</div>
      <div className="card-subtitle">{block.subtitle}</div>

      <div className="metric-row">
        {block.metrics.map((metric) => (
          <div className="metric-item" key={metric.label} title={metric.hint}>
            <div className="metric-label">{metric.label}</div>
            <div className="metric-value">Not connected yet</div>
            <div className="metric-hint">{metric.hint}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function DashboardPage() {
  return (
    <div className="dashboard-column">
      <div className="dashboard-intro">
        <div className="dashboard-title">Security overview</div>
        <p>
          How AegisEHR is protecting the clinical assistant. Figures appear here only once the
          matching security service is running and measuring them.
        </p>
      </div>

      {BLOCKS.map((block) => (
        <MetricBlock block={block} key={block.title} />
      ))}

      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Live Security Activity</div>
            <div className="card-subtitle">Safe, blocked, escalated and learned events</div>
          </div>
        </div>
        <div className="empty-state">
          No activity yet. Events will appear here once security screening is connected.
        </div>
      </div>
    </div>
  );
}
