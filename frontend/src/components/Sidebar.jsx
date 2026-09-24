import {
  ShieldIcon,
  GaugeIcon,
  MessageIcon,
  RadarIcon,
  FileTextIcon,
} from "./icons.jsx";

// Only "Clinical Assistant" is wired to a real backend today. Every other
// item is visually present (per the design brief) but disabled — no fake
// functionality behind them.
const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: GaugeIcon, enabled: true },
  { key: "clinical-assistant", label: "Clinical Assistant", icon: MessageIcon, enabled: true },
  { key: "attack-lab", label: "Attack Lab", icon: RadarIcon, enabled: false },
  { key: "audit", label: "Audit / Evaluation", icon: FileTextIcon, enabled: false },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <ShieldIcon width={18} height={18} />
        </div>
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-title">AegisEHR</span>
          <span className="sidebar-brand-subtitle">Secure Clinical AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ key, label, icon: Icon, enabled }) => (
          <button
            key={key}
            type="button"
            className={`sidebar-nav-item ${activePage === key ? "active" : ""} ${
              enabled ? "" : "disabled"
            }`}
            disabled={!enabled}
            onClick={() => enabled && onNavigate(key)}
          >
            <span className="nav-icon-label">
              <Icon />
              {label}
            </span>
            {!enabled && <span className="nav-soon-badge">Soon</span>}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-footer-event">EDGE Hack 2026</div>
        <div>Team CloudZero</div>
      </div>
    </aside>
  );
}
