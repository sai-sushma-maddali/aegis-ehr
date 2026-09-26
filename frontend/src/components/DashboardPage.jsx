import { useEffect, useState } from "react";
import { clearDashboard, fetchActivity } from "../services/api.js";
import MetricsDashboard from "./MetricsDashboard.jsx";
import SecurityOverview from "./SecurityOverview.jsx";

const POLL_MS = 4000;

export default function DashboardPage() {
  const [events, setEvents] = useState([]);
  const [unavailable, setUnavailable] = useState(false);
  const [revision, setRevision] = useState(0);
  const [clearing, setClearing] = useState(false);
  const [clearError, setClearError] = useState("");

  async function handleClear() {
    setClearing(true);
    setClearError("");
    try {
      await clearDashboard();
      setEvents([]);
      setRevision((value) => value + 1);
    } catch (err) {
      setClearError(err?.message || "Could not clear the dashboard.");
    } finally {
      setClearing(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      try {
        const payload = await fetchActivity();
        if (cancelled) return;
        setEvents(payload.events || []);
        setUnavailable(false);
      } catch {
        if (!cancelled) setUnavailable(true);
      }
    }

    refresh();
    const timer = setInterval(refresh, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="dashboard-column">
      <MetricsDashboard
        events={events}
        revision={revision}
        clearing={clearing}
        clearError={clearError}
        onClear={handleClear}
      />
      <SecurityOverview key={revision} events={events} unavailable={unavailable} />
    </div>
  );
}
