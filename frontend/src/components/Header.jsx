import { useEffect, useState } from "react";
import { checkBackendHealth } from "../services/api.js";
import { ShieldIcon } from "./icons.jsx";

const HEALTH_CHECK_INTERVAL_MS = 15000;

export default function Header() {
  const [backendUp, setBackendUp] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      const isUp = await checkBackendHealth();
      if (!cancelled) setBackendUp(isUp);
    }

    refresh();
    const timer = setInterval(refresh, HEALTH_CHECK_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const dotClass = backendUp === null ? "dot-unknown" : backendUp ? "dot-up" : "dot-down";
  const statusText =
    backendUp === null ? "Checking connection" : backendUp ? "Connected" : "Backend offline";

  return (
    <header className="top-header">
      <div>
        <div className="top-header-title">AegisEHR</div>
        <div className="top-header-subtitle">
          Secure clinical AI with grounded retrieval and safety controls
        </div>
      </div>

      <div className="top-header-right">
        <span className="header-meta">
          <ShieldIcon width={14} height={14} />
          Synthetic patient data
        </span>
        <span className="header-divider" />
        <span className="header-meta">
          <span className={`status-dot ${dotClass}`} />
          Local model · Mistral · {statusText}
        </span>
      </div>
    </header>
  );
}
