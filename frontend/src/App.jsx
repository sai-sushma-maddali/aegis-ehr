import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Header from "./components/Header.jsx";
import ClinicalAssistantPage from "./components/ClinicalAssistantPage.jsx";
import DashboardPage from "./components/DashboardPage.jsx";

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");

  return (
    <div className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />

      <div className="main-column">
        <Header />

        <div className="page-content">
          {/* Pages stay mounted and are only hidden, so switching tabs
              never resets a question, answer or loaded patient list. */}
          <div hidden={activePage !== "dashboard"}>
            <DashboardPage />
          </div>
          <div hidden={activePage !== "clinical-assistant"}>
            <ClinicalAssistantPage />
          </div>
        </div>
      </div>
    </div>
  );
}
