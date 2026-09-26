import { useEffect, useRef, useState } from "react";
import { askClinicalAssistant } from "../services/api.js";
import { TIER1, TIER2, TIER3 } from "../tierNames.js";
import { AlertTriangleIcon, PaperclipIcon, ShieldIcon } from "./icons.jsx";
import AnswerCard from "./AnswerCard.jsx";

const ACCEPTED_DOCS = ".txt,.pdf,.docx,.doc,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

function blockKicker(kicker) {
  const label = String(kicker || "");
  if (
    label === TIER1 ||
    label === TIER2 ||
    label === TIER3
  ) {
    return label;
  }
  if (
    label === "Edge intent triage" ||
    label === "Aegis Guard" ||
    label.includes("Intent Guard") ||
    label.includes("Edge Intent")
  ) {
    return TIER2;
  }
  if (
    label === "Known attack pattern" ||
    label === "Threat cache" ||
    label.includes("Vector Threat Cache")
  ) {
    return TIER1;
  }
  if (label === "Cloud review" || label.includes("Cloud Forensic")) {
    return TIER3;
  }
  return label || "Security screening";
}

function friendlyError(error) {
  return error.message || "The clinical assistant could not process this question.";
}

function historyFromMessages(messages) {
  return messages.slice(-8).flatMap((message) => {
    if (message.role === "user" && message.text) {
      return [{ role: "user", text: message.text }];
    }
    if (message.role === "assistant" && message.result?.answer) {
      return [{ role: "assistant", text: message.result.answer }];
    }
    return [];
  });
}

export default function ClinicalAssistantPage() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [attachment, setAttachment] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const bottomRef = useRef(null);
  const fileRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isRunning]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [draft]);

  async function send(text) {
    const question = text.trim();
    if ((!question && !attachment) || isRunning) return;

    const history = historyFromMessages(messages);
    const file = attachment;
    const displayText = file
      ? `${question || "Please review this uploaded document."}\n\nAttached: ${file.name}`
      : question;
    setMessages((current) => [...current, { role: "user", text: displayText }]);
    setDraft("");
    setAttachment(null);
    if (fileRef.current) fileRef.current.value = "";
    setIsRunning(true);

    try {
      const result = await askClinicalAssistant(
        null,
        question || "Please review this uploaded document.",
        history,
        file,
      );
      setMessages((current) => [...current, { role: "assistant", result }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "error", text: friendlyError(error), notice: error.notice || null },
      ]);
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

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    if (!file) {
      setAttachment(null);
      return;
    }
    const name = file.name.toLowerCase();
    if (!(/\.(txt|pdf|docx|doc)$/.test(name))) {
      setAttachment(null);
      event.target.value = "";
      setMessages((current) => [
        ...current,
        {
          role: "error",
          text: "Only .txt, .pdf, and .docx documents can be uploaded.",
          notice: null,
        },
      ]);
      return;
    }
    setAttachment(file);
  }

  const suggestions = [
    "What can this clinical assistant help me with?",
    "How does the security screening protect patient data?",
    "Can I attach a note or lab report for review?",
  ];
  const canSend = Boolean(draft.trim() || attachment) && !isRunning;

  return (
    <div className={`chat-page ${messages.length === 0 ? "is-empty" : ""}`}>
      <div className="chat-scroll">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-title">How can I help?</div>
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
                    <div>Checking that message...</div>
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="chat-input-area">
        {attachment ? (
          <div className="chat-attachment">
            <span>{attachment.name}</span>
            <button
              type="button"
              className="chat-context-clear"
              onClick={() => {
                setAttachment(null);
                if (fileRef.current) fileRef.current.value = "";
              }}
            >
              Remove
            </button>
          </div>
        ) : null}
        <div className="chat-input-box">
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED_DOCS}
            className="chat-file-input"
            onChange={handleFileChange}
            aria-label="Attach a document"
          />
          <button
            type="button"
            className="chat-attach"
            disabled={isRunning}
            onClick={() => fileRef.current?.click()}
            aria-label="Attach document"
            title="Attach .txt, .pdf, or .docx"
          >
            <PaperclipIcon width={20} height={20} strokeWidth={2.4} />
          </button>
          <textarea
            ref={inputRef}
            className="chat-input"
            rows={1}
            placeholder="Ask a question, or attach a document to review"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            aria-label="Message AegisEHR"
          />
          <button
            type="button"
            className="chat-send"
            disabled={!canSend}
            onClick={() => send(draft)}
            aria-label="Send"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M5 12h12M13 6l6 6-6 6"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>

        <div className="chat-footnote">
          AegisEHR can make mistakes. Check important information before relying on it.
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
    if (message.notice?.summary) {
      const confidence =
        typeof message.notice.confidence === "number"
          ? Math.round(message.notice.confidence * 100)
          : null;
      return (
        <div className="chat-assistant">
          <div className="block-notice">
            <div className="block-notice-icon">
              <ShieldIcon width={18} height={18} />
            </div>
            <div>
              <div className="block-notice-kicker">
                {blockKicker(message.notice.kicker)}
              </div>
              <div className="block-notice-title">{message.notice.title || "Request blocked"}</div>
              <p className="block-notice-summary">{message.notice.summary}</p>
              {confidence !== null && (
                <div className="block-notice-meta">Detection confidence {confidence}%</div>
              )}
            </div>
          </div>
        </div>
      );
    }

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
    </div>
  );
}
