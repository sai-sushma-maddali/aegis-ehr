# AegisEHR

Edge-first clinical AI with a self-healing security harness for **HP ZGX** (Nano, Fury, or a local GPU/CPU laptop).

AegisEHR screens every clinical question (and optional uploaded document) through a three-tier defense stack before any chart retrieval runs. Benign questions are answered on-device with **Mistral-7B** RAG. Attacks are blocked locally when possible, escalated to cloud forensics when uncertain, and hot-patched back into the local threat cache so the next similar prompt dies in milliseconds.

**Team:** ZeroCloud

---

## Configuration (only zrt model paths)

**`aegis.config.json`** at the repo root is the only file operators edit. It has **three paths** — everything else (ports, LoRA id, thresholds, flags, CORS) is in code (`backend/core/config.py`).

```bash
cp aegis.config.example.json aegis.config.json
# edit the three paths after zrt pull
```

```json
{
  "paths": {
    "qwen_base_model": "/opt/hp/zrt/models/hf/Qwen/Qwen2.5-7B-Instruct/main",
    "mistral_model_dir": "/opt/hp/zrt/models/hf/mistralai/Mistral-7B-Instruct-v0.3",
    "zrt_metrics_socket": "/opt/hp/zrt/run/vllm-hf-mistralai-mistral-7b-instruct-v0-3-main.sock"
  }
}
```

| Field | Required? | Purpose |
|-------|-----------|---------|
| `qwen_base_model` | **Yes** for Tier 2 | Local Qwen2.5-7B-Instruct after `zrt pull` (used with HF LoRA in code) |
| `mistral_model_dir` | Optional | Local Mistral folder (dashboard weight size); clinical chat uses zrt `:8080` |
| `zrt_metrics_socket` | Optional | zrt metrics socket for dashboard TTFT |

Leave a path `""` until that asset exists. Confirm folders with `ls /opt/hp/zrt/models/hf/` and `ls /opt/hp/zrt/run/`.

**In code (do not put in JSON):** LoRA `SushmaMaddali/qwen-medical-prompt-guard-lora`, ports `8000`/`5173`, clinical URL `http://127.0.0.1:8080/v1`, seeds at `backend/data/seed_threats.json`, thresholds, CORS. Advanced overrides: `AEGIS_*` env vars.

**Threat seeds:** `backend/data/seed_threats.json` is committed; Chroma is created on first API boot.

---

## Architecture

```
Clinician UI (React / Vite :5173)
        │  /api  (Vite proxy → FastAPI on same host)
        ▼
FastAPI gateway (backend/main.py)
        │
        ├─ Tier 1  Fast Vector Threat Cache (ChromaDB + MiniLM)
        ├─ Tier 2  On-Device Intent Guard (fine-tuned Qwen2.5-7B)
        ├─ Tier 3  Cloud Forensic Defender (Gemma 4) + hot-patch
        └─ Downstream  Clinical RAG (Mistral via clinical_llm_base_url)
```

---

## Prerequisites

- Python **3.10+** (CUDA PyTorch recommended on ZGX)
- Node.js **18+**
- Optional: HP **zrt** (or any OpenAI-compatible server) for clinical answers
- Optional: `export GEMINI_API_KEY=...` in the API shell for Tier 3 cloud (not loaded from `.env` automatically)

---

## Run on ZGX Fury + open the UI on your laptop

Do these steps **in order**. Everything below runs **on Fury** except the final SSH tunnel and browser on the laptop.

### 0. Clone / sync the repo on Fury

```bash
cd ~/Documents   # or wherever you keep the project
# git clone … / sync aegis-ehr
cd aegis-ehr
```

### 1. Pull models with zrt, then edit `aegis.config.json`

```bash
zrt pull 'hf:Qwen/Qwen2.5-7B-Instruct@main'
# Mistral is pulled when you `zrt serve` it (step 3)
```

Edit **only** these three paths in `aegis.config.json` (copy from `aegis.config.example.json` if needed):

```json
{
  "paths": {
    "qwen_base_model": "/opt/hp/zrt/models/hf/Qwen/Qwen2.5-7B-Instruct/main",
    "mistral_model_dir": "/opt/hp/zrt/models/hf/mistralai/Mistral-7B-Instruct-v0.3",
    "zrt_metrics_socket": "/opt/hp/zrt/run/vllm-hf-mistralai-mistral-7b-instruct-v0-3-main.sock"
  }
}
```

Use the real folders on your Fury box (`ls` to confirm). Tier 2 = that Qwen path + the HF LoRA id baked into the code. First API boot may download the LoRA once (needs network).

### 2. Install dependencies (once)

```bash
# Python (use your Fury conda/venv)
pip install -r backend/requirements.txt

cd frontend && npm install && cd ..
```

### 3. Start the clinical LLM (zrt) on Fury

```bash
zrt serve 'hf:mistralai/Mistral-7B-Instruct-v0.3@main' --gpu-memory-fraction 0.30
```

Confirm (still on Fury):

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Clinical LLM URL defaults to `http://127.0.0.1:8080/v1` in code. Override with `AEGIS_CLINICAL_LLM_BASE_URL` only if zrt uses another port.

### 4. Start the FastAPI gateway on Fury

```bash
cd backend
# optional Tier 3:
# export GEMINI_API_KEY=your_key_here
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Health check on Fury:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expect `edge_guard_ready` / `clinical_agent_ready` true when Qwen path + zrt are healthy. First boot builds Chroma (can take several minutes; may download HF LoRA / EHR once).

### 5. Start the Vite UI on Fury

```bash
cd frontend
npm run dev
```

Vite listens on `0.0.0.0:5173` (code default) and proxies `/api` → FastAPI on `127.0.0.1:8000`.

### 6. On your laptop — SSH port-forward **only the UI**

```bash
ssh -L 5173:127.0.0.1:5173 YOUR_USER@zgx-fury
```

### 7. Open the app on the laptop

Browser: **http://127.0.0.1:5173**

Ask a clinical question. The laptop browser talks only to Vite; Vite on Fury proxies `/api` to FastAPI and the clinical LLM on Fury.

**Do not** open `http://127.0.0.1:8000` from the laptop unless you also forwarded port 8000 — that used to leave chat stuck on “Processing…”.

### If chat stays on “Processing…”

On **Fury**, while the laptop UI is open:

1. `curl -s http://127.0.0.1:8000/api/health`
2. `curl -s http://127.0.0.1:8080/v1/models`
3. Watch FastAPI logs — first Tier 2 / RAG call can take tens of seconds
4. Confirm the SSH tunnel is still up and you are using the Vite URL, not `:8000`

---

## Run entirely on one machine (Nano, Fury, or laptop)

Same order as above, skip the SSH step, and open `http://127.0.0.1:5173` on that machine.

For a quick smoke test without models (keyword guard only, no clinical RAG):

```bash
export AEGIS_OFFLINE_MODE=1 AEGIS_SKIP_CLINICAL=1 AEGIS_WEIGHT_CACHE=0
# then start API + UI only (steps 4–5)
```

---

## Using the product

### Clinical Assistant

- Ask chart questions in natural language
- Attach **`.txt` / `.pdf` / `.docx`** to test poisoned-document attacks
- Blocks show the deciding tier name

Sample fixtures:

- `tests/poisoned_docs/mccorkle_lab_addendum.txt`
- `tests/poisoned_docs/mccartney_night_notes.txt`

### Dashboard

- Tier latencies, PHI exposure, hot-patch rate
- Host telemetry from `/proc`, `nvidia-smi`, and optional zrt metrics socket

---

## Main API routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Readiness |
| POST | `/api/ask` | Screen + answer (JSON or multipart with `document`) |
| GET | `/api/activity` | Prompt history |
| GET | `/api/telemetry` | Dashboard aggregates + live engine snapshot |
| GET | `/api/metrics` | Gateway + engine telemetry |
| POST | `/api/dashboard/clear` | Reset dashboard session state |
| POST | `/api/reset` | Flush hot-patches; restore seed threats |
| GET | `/api/threats` | Threat store listing |
| GET | `/api/patients` | Indexed patient registry |

---

## Tests

```bash
cd /path/to/aegis-ehr
PYTHONPATH=backend python -m unittest \
  tests.test_edge_guard \
  tests.test_telemetry \
  tests.test_zrt_metrics \
  tests.test_document_extract -v
```

Note: `test_chromadb_failure_fail_closed` intentionally mocks Chroma down and fail-closes. A short `edge_guard.vector_cache_error` warning is expected, not a crash.

---

## Repository layout

```
aegis-ehr/
├── aegis.config.json           # ← edit this for your machine
├── aegis.config.example.json
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── core/
│       ├── config.py           # loads aegis.config.json
│       ├── edge_guard.py
│       ├── cloud_defender.py
│       ├── clinical_agent.py
│       ├── ingestion.py
│       ├── weight_cache.py
│       └── zrt_metrics.py
├── frontend/
├── tests/
└── fine tuning/
```

---

## Design principles

- **Config, don’t hardcode** — zrt model paths in `aegis.config.json`; other defaults in code
- **Measure, don’t invent** — dashboard numbers from recorded prompts, Chroma, `/proc`, `nvidia-smi`, optional zrt socket
- **Fail closed** — classifier / Chroma unavailable blocks rather than retrieve
- **PHI discipline** — scrub before cloud
