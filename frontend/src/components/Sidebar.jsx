import {
  GaugeIcon,
  MessageIcon,
} from "./icons.jsx";

const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: GaugeIcon },
  { key: "clinical-assistant", label: "Clinical Assistant", icon: MessageIcon },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <img
          src="/AegisEHR_shield.png"
          alt=""
          className="sidebar-brand-logo"
        />
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-title">AegisEHR</span>
          <span className="sidebar-brand-subtitle">Secure Clinical AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            type="button"
            className={`sidebar-nav-item ${activePage === key ? "active" : ""}`}
            onClick={() => onNavigate(key)}
          >
            <span className="nav-icon-label">
              <Icon />
              {label}
            </span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div>Team ZeroCloud</div>
      </div>
    </aside>
  );
}
