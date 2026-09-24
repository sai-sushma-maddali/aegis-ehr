import { useEffect, useRef, useState } from "react";
import { askClinicalAssistant, fetchPatients } from "../services/api.js";
import { AlertTriangleIcon } from "./icons.jsx";
import AnswerCard from "./AnswerCard.jsx";
import EvidenceTable from "./EvidenceTable.jsx";

const NO_PATIENT_MESSAGE =
  "I couldn't tell which patient you mean. Please include the patient's name, for example: “What medications is Elizabeth Brown on?”";

// The backend accepts a question with or without a patient. For follow-ups
// ("and her allergies?") we re-attach the patient from the last answer, but
// only when the user did not name anyone themselves.
function mentionsPatient(question, patients) {
  const text = question.toLowerCase();
  const namesOne = patients.some((p) => text.includes(p.patient_name.toLowerCase()));
  return namesOne || /\bmrn\b|medical record number/i.test(question);
}

function friendlyError(error) {
  if (error.status === 422 && /patient|MRN/i.test(error.message)) return NO_PATIENT_MESSAGE;
  return error.message || "The clinical assistant could not process this question.";
}

export default function ClinicalAssistantPage() {
  const [patients, setPatients] = useState([]);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [activePatient, setActivePatient] = useState(null);

  const bottomRef = useRef(null);

  useEffect(() => {
    fetchPatients()
      .then(setPatients)
      .catch(() => setPatients([]));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isRunning]);

  async function send(text) {
    const question = text.trim();
    if (!question || isRunning) return;

    setMessages((current) => [...current, { role: "user", text: question }]);
    setDraft("");
    setIsRunning(true);

    const needsContext = activePatient && !mentionsPatient(question, patients);
    const patientLabel = needsContext
      ? `${activePatient.name} (MRN ${activePatient.id})`
      : null;

    try {
      const result = await askClinicalAssistant(patientLabel, question);
      setActivePatient({ name: result.patient_name, id: result.patient_id });
      setMessages((current) => [...current, { role: "assistant", result }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "error", text: friendlyError(error) }]);
    } finally {
      setIsRunning(false);
    }
  }

  function handleKeyDown(event) {
    // Enter sends, Shift+Enter adds a new line.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(draft);
    }
  }

  const suggestions = patients.slice(0, 3).map((p) => `What medications is ${p.patient_name} on?`);

  return (
    <div className={`chat-page ${messages.length === 0 ? "is-empty" : ""}`}>
      <div className="chat-scroll">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-title">How can I help?</div>
            <p>
              Ask about a patient's record in plain language. Include the patient's name and I'll
              search only their record.
            </p>
            <div className="chat-suggestions">
              {suggestions.map((suggestion) => (
                <button
                  type="button"
                  key={suggestion}
                  className="chat-suggestion"
                  onClick={() => send(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((message, index) => (
              <ChatMessage message={message} key={index} />
            ))}

            {isRunning && (
              <div className="chat-assistant">
                <div className="card">
                  <div className="loading-block">
                    <div className="spinner" />
                    <div>Searching the patient's record. This can take a minute...</div>
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="chat-input-area">
        {activePatient && (
          <div className="chat-context">
            Talking about <strong>{activePatient.name}</strong> (MRN {activePatient.id})
            <button type="button" className="chat-context-clear" onClick={() => setActivePatient(null)}>
              Clear
            </button>
          </div>
        )}

        <div className="chat-input-box">
          <textarea
            className="chat-input"
            rows={1}
            placeholder="Ask about a patient, e.g. What medications is Elizabeth Brown on?"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            aria-label="Message AegisEHR"
          />
          <button
            type="button"
            className="chat-send"
            disabled={!draft.trim() || isRunning}
            onClick={() => send(draft)}
            aria-label="Send"
          >
            →
          </button>
        </div>

        <div className="chat-footnote">
          Synthetic patient data only. Security screening is not connected yet.
        </div>
      </div>
    </div>
  );
}

function ChatMessage({ message }) {
  if (message.role === "user") {
    return (
      <div className="chat-user">
        <div className="chat-user-bubble">{message.text}</div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="chat-assistant">
        <div className="error-banner">
          <AlertTriangleIcon width={16} height={16} />
          <span>{message.text}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-assistant">
      <AnswerCard result={message.result} isRunning={false} />
      <EvidenceTable result={message.result} isRunning={false} />
    </div>
  );
}
