# AegisEHR

Edge-first clinical AI with a self-healing security harness for **HP ZGX** (Nano, Fury, or a local GPU/CPU laptop).

AegisEHR screens every clinical question (and optional uploaded document) through a three-tier defense stack before any chart retrieval runs. Benign questions are answered on-device with **Mistral-7B** RAG. Attacks are blocked locally when possible, escalated to cloud forensics when uncertain, and hot-patched back into the local threat cache so the next similar prompt dies in milliseconds.

**Team:** ZeroCloud

---

## Configuration (one file for every machine)

All machine-specific paths, ports, and labels live in **`aegis.config.json`** at the repo root (backend + Vite both read it).

```bash
cp aegis.config.example.json aegis.config.json   # if you need a fresh copy
# edit aegis.config.json for your host
```

| Field | Purpose |
|-------|---------|
| `paths.guard_merged_model` | Fine-tuned Qwen guard merge folder (empty → offline keyword guard) |
| `paths.qwen_base_model` | Optional local Qwen base for PEFT merge |
| `paths.mistral_model_dir` | Optional on-disk Mistral weights (dashboard size only) |
| `paths.zrt_metrics_socket` | Optional zrt/vLLM metrics UNIX socket |
| `paths.chroma_dir` | ChromaDB directory (relative to repo root OK) |
| `paths.seed_threats_file` | Tier-1 seed attacks JSON (committed: `backend/data/seed_threats.json`) |
| `services.api_*` / `ui_*` | FastAPI and Vite bind host/port |
| `services.clinical_llm_base_url` | OpenAI-compatible LLM (default zrt on `:8080`) |
| `services.weight_cache_*` | Resident guard embedder cache |
| `flags.offline_mode` | `true` = no cloud Tier 3 |
| `flags.skip_clinical` | `true` = skip Mistral RAG (faster boot) |
| `platform.*` | Dashboard labels (works for Nano / Fury / laptop) |

`AEGIS_*` environment variables still override the JSON when set. Point at another file with `AEGIS_CONFIG=/path/to/file.json`.

### Example: ZGX Fury (or Nano) with local models

```json
{
  "platform": {
    "display_name": "HP ZGX Fury",
    "gpu_fallback_name": "NVIDIA GPU",
    "architecture_blurb": "ZGX Fury on-device GPU",
    "memory_label": "System memory",
    "rated_tops_fp4": null
  },
  "paths": {
    "guard_merged_model": "/home/YOUR_USER/models/qwen_medical_guard_merged",
    "qwen_base_model": "/opt/hp/zrt/models/hf/Qwen/Qwen2.5-7B-Instruct/main",
    "mistral_model_dir": "/opt/hp/zrt/models/hf/mistralai/Mistral-7B-Instruct-v0.3",
    "zrt_metrics_socket": "/opt/hp/zrt/run/vllm-hf-mistralai-mistral-7b-instruct-v0-3-main.sock",
    "chroma_dir": "backend/data/chroma_aegis_db",
    "raw_data_dir": "backend/data/raw",
    "ehr_records_file": "backend/data/ehr_records.txt"
  },
  "flags": {
    "offline_mode": false,
    "skip_clinical": false,
    "weight_cache": true,
    "reset_on_boot": false
  }
}
```

Leave model/zrt paths `""` if that asset is missing — the gateway boots with safe fallbacks.

**Threat seed data:** `backend/data/seed_threats.json` ships with 8 canonical attack signatures. On API boot, ingestion embeds them into Chroma `threat_signatures`. Keep that file in git (Chroma DB folders stay gitignored).

---

## Architecture

```
Clinician UI (React / Vite — port from aegis.config.json)
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
- Optional: merged guard weights + `GEMINI_API_KEY` for full Tier 2 / Tier 3

Secrets stay in a gitignored `.env` at the repo root:

```bash
GEMINI_API_KEY=your_key_here
```

---

## Run on ZGX Fury + open the UI on your laptop

Do these steps **in order**. Everything below runs **on Fury** except the final SSH tunnel and browser on the laptop.

### 0. Clone / sync the repo on Fury

```bash
cd ~/Documents   # or wherever you keep the project
# git clone … / sync aegis-ehr
cd aegis-ehr
```

### 1. Edit `aegis.config.json`

Set `paths.guard_merged_model` (and optional zrt paths) for this machine. Set:

```json
"flags": { "offline_mode": false, "skip_clinical": false, "weight_cache": true }
```

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

If you use a different port, update `services.clinical_llm_base_url` in `aegis.config.json`.

### 4. Start the FastAPI gateway on Fury

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Host/port can also come from `aegis.config.json` (`services.api_host` / `api_port`) if you run `python main.py`.

Health check on Fury:

```bash
curl -s http://127.0.0.1:8000/api/health
```

### 5. Start the Vite UI on Fury

```bash
cd frontend
npm run dev
```

Vite binds using `services.ui_host` / `ui_port` from `aegis.config.json` (default `0.0.0.0:5173`) and proxies `/api` → FastAPI on the same machine.

### 6. On your laptop — SSH port-forward **only the UI**

```bash
ssh -L 5173:127.0.0.1:5173 YOUR_USER@zgx-fury
```

If you changed `ui_port` in config, use that port on both sides of `-L`.

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

For a quick laptop smoke test without models:

```json
"flags": { "offline_mode": true, "skip_clinical": true, "weight_cache": false }
```

Then start API + UI only (steps 4–5).

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
- Labels come from `platform.*` in `aegis.config.json`

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

- **Config, don’t hardcode** — paths/ports/labels in `aegis.config.json`
- **Measure, don’t invent** — dashboard numbers from recorded prompts, Chroma, `/proc`, `nvidia-smi`, optional zrt socket
- **Fail closed** — classifier / Chroma unavailable blocks rather than retrieve
- **PHI discipline** — scrub before cloud
