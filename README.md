# AegisEHR

Edge-first clinical AI with a self-healing security harness for the **HP ZGX Nano**.

AegisEHR screens every clinical question (and optional uploaded document) through a three-tier defense stack before any chart retrieval runs. Benign questions are answered on-device with **Mistral-7B** RAG. Attacks are blocked locally when possible, escalated to cloud forensics when uncertain, and hot-patched back into the local threat cache so the next similar prompt dies in milliseconds.

**Team:** ZeroCloud

---

## Problem & use case

Hospital clinical assistants that sit on top of EHR data are exposed to:

- Direct prompt injection and jailbreaks
- Indirect / poisoned document payloads (lab notes, nursing notes)
- Authority impersonation and tool-exfiltration attempts
- Accidental PHI egress when uncertain prompts are sent to the cloud

AegisEHR keeps patient charts on the ZGX Nano, screens every request before retrieval, redacts PHI before any cloud call, and learns from confirmed attacks so repeats never leave the edge again.

---

## Architecture

```
Clinician UI (React / Vite :5173)
        │
        ▼
FastAPI gateway (backend/main.py :8000)
        │
        ├─ Tier 1  Fast Vector Threat Cache (ChromaDB + MiniLM)
        │           cosine similarity ≥ 0.82 → block
        ├─ Tier 2  On-Device Intent Guard (fine-tuned Qwen2.5-7B head)
        │           attack_p ≤ 0.15 allow · ≥ 0.85 block · else escalate
        ├─ Tier 3  Cloud Forensic Defender (Gemma 4) + hot-patch write
        │           PHI scrub before egress · signature stored in Chroma
        └─ Downstream  Clinical RAG Assistant (Mistral-7B via HP zrt)
                        only if the request PASSED screening
```

### Standardized tier names (UI + API)

| Tier | Name |
|------|------|
| 1 | Fast Vector Threat Cache (ChromaDB) |
| 2 | On-Device Intent Guard (Qwen2.5-7B) |
| 3 | Cloud Forensic Defender (Gemma 4) |
| Downstream | Clinical RAG Assistant (Mistral-7B) |

### Self-healing loop

1. Ambiguous prompt → Tier 3 confirms attack  
2. Scrubbed / reviewed text is written into `threat_signatures`  
3. Replay of the same attack → Tier 1 cache hit in ~10 ms (no cloud)

---

## Repository layout

```
aegis-ehr/
├── backend/
│   ├── main.py                 # FastAPI gateway
│   ├── requirements.txt
│   ├── core/
│   │   ├── edge_guard.py       # Tier 1 + 2 + cloud escalate
│   │   ├── cloud_defender.py   # Gemma 4 forensic review
│   │   ├── clinical_agent.py   # Mistral RAG over clinical_records
│   │   ├── ingestion.py        # Chroma bootstrap + seed threats
│   │   ├── document_extract.py # PDF / DOCX / TXT upload parsing
│   │   ├── weight_cache.py     # Resident guard weights on :8091
│   │   ├── telemetry_tracker.py
│   │   └── zrt_metrics.py      # Live ZRT / GB10 hardware metrics
│   └── data/                   # Chroma + activity (gitignored)
├── frontend/                   # React + Vite clinical UI + dashboard
├── tests/                      # Unit tests + poisoned document samples
└── fine tuning/                # Guard fine-tune artifacts / notes
```

---

## Technical stack

| Layer | Choice |
|-------|--------|
| Device | HP ZGX Nano (NVIDIA GB10 Grace Blackwell) |
| Local LLM runtime | HP **zrt** (vLLM) for Mistral-7B on `127.0.0.1:8080` |
| Guard weights | Resident weight cache on `127.0.0.1:8091` (no second CUDA load) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (`threat_signatures`, `clinical_records`) |
| Cloud forensics | Google GenAI Gemma 4 (`GEMINI_API_KEY`) |
| API | FastAPI + Uvicorn |
| UI | React 18 + Vite |

---

## Prerequisites

- Python **3.10+** with a conda/mamba env that has CUDA-capable PyTorch (this project uses `/home/hp6/miniforge3/envs/zgx`)
- Node.js **18+** (frontend)
- HP **zrt** available on PATH (`zrt serve …`)
- Fine-tuned guard merge folder (default path in code):  
  `/home/hp6/jupyterlab/qwen_medical_guard_merged`
- Optional for Tier 3: `GEMINI_API_KEY` in a gitignored `.env` at the repo root

---

## Environment

Create `/home/hp6/Desktop/edge_hack/aegis-ehr/.env` (never commit):

```bash
GEMINI_API_KEY=your_key_here
```

Useful flags:

| Variable | Meaning |
|----------|---------|
| `AEGIS_OFFLINE_MODE=0` | Allow cloud Tier 3 when needed |
| `AEGIS_SKIP_CLINICAL=0` | Enable Mistral clinical RAG |
| `AEGIS_WEIGHT_CACHE=0` | Disable resident weight cache (not recommended on ZGX) |
| `AEGIS_LOG_LEVEL=INFO` | Gateway log level |

---

## Setup

### 1. Python dependencies

```bash
cd /home/hp6/Desktop/edge_hack/aegis-ehr
/home/hp6/miniforge3/envs/zgx/bin/pip install -r backend/requirements.txt
```

### 2. Frontend dependencies

```bash
cd frontend
npm install
```

### 3. Serve Mistral with zrt (clinical answers)

```bash
zrt serve 'hf:mistralai/Mistral-7B-Instruct-v0.3@main' --gpu-memory-fraction 0.30
```

Confirm OpenAI-compatible API:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

### 4. Start the API

From `backend/` (reuses weight cache on `:8091` when healthy):

```bash
cd /home/hp6/Desktop/edge_hack/aegis-ehr/backend
AEGIS_OFFLINE_MODE=0 AEGIS_SKIP_CLINICAL=0 \
  /home/hp6/miniforge3/envs/zgx/bin/python -m uvicorn main:app \
  --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/api/health
```

### 5. Start the UI

```bash
cd /home/hp6/Desktop/edge_hack/aegis-ehr/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open **http://localhost:5173/** (not `:8000`).

---

## Using the product

### Clinical Assistant

- Ask chart questions in natural language
- Attach **`.txt` / `.pdf` / `.docx`** (paperclip) to test **poisoned document** attacks
- Every request is screened before RAG; blocks show the deciding tier name

Sample poisoned fixtures (from the attacker prompt set):

- `tests/poisoned_docs/mccorkle_lab_addendum.txt`
- `tests/poisoned_docs/mccartney_night_notes.txt`

Example:

1. Attach `mccorkle_lab_addendum.txt`
2. Ask: *Please summarize this uploaded follow-up pathology addendum for Michael Mccorkle (MRN L1004).*
3. Expect a Tier 1 or Tier 2 block rather than a chart dump

### Dashboard

- Live KPIs: Tier 1 / Tier 2 latency, PHI exposure, hot-patch rate
- Multi-tier volume & cost share
- HP ZGX Nano telemetry from live memory / CPU / GPU / ZRT histograms
- Prompt history with Airflow-style execution DAG per request
- **Clear dashboard** resets activity, session savings, and ZRT session counters (seeds kept)

---

## Main API routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Readiness |
| POST | `/api/ask` | Screen + answer (JSON or multipart with `document`) |
| GET | `/api/activity` | Prompt history |
| GET | `/api/telemetry` | Dashboard aggregates + live engine snapshot |
| GET | `/api/metrics` | Gateway + ZRT telemetry object |
| POST | `/api/dashboard/clear` | Reset dashboard session state |
| POST | `/api/reset` | Flush hot-patches; restore seed threat signatures |
| GET | `/api/threats` | Threat store listing |
| GET | `/api/patients` | Indexed patient registry |

---

## Tests

```bash
cd /home/hp6/Desktop/edge_hack/aegis-ehr
/home/hp6/miniforge3/envs/zgx/bin/python -m unittest \
  tests.test_edge_guard \
  tests.test_telemetry \
  tests.test_zrt_metrics \
  tests.test_document_extract -v
```

---

## Design principles

- **Measure, don’t invent** — dashboard numbers come from recorded prompts, Chroma, `/proc`, `nvidia-smi`, and the live ZRT metrics socket
- **No second guard load** — API attaches to the resident weight cache on `:8091`
- **Fail closed** — classifier unavailable or gateway errors block rather than retrieve
- **PHI discipline** — scrub before cloud; residency metrics track whether identifiers left the device

---

## Notes for this ZGX deployment

- Unified memory is reported as system LPDDR (GB10 has no separate GPU framebuffer in `nvidia-smi`)
- Do not call `torch.cuda.mem_get_info()` from the API process while models are loaded
- Stopping uvicorn: kill the **python** child that owns `:8000`, not only a shell wrapper
- Activity log path: `backend/data/prompt_activity.json` (gitignored)
