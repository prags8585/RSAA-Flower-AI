# ReSEA — Residency‑Safe Engineering Answers (starter)

This is a **Simple, end‑to‑end starter** for a decentralized, cross‑region RAG assistant.
It runs **three region services** (EU/US/APAC) that each index **their own** docs,
plus a **coordinator** that fans out queries and merges results with **RRF**.
You can add optional **federated re‑ranking** (Flower) from click feedback.

> Works on one laptop (four processes). No cloud required.

---

## 0) Prereqs

- Python 3.10+ (3.11 recommended)
- `pip` (or `uv`) installed
- (Optional) **Ollama** or **vLLM** if you want LLM summarization; not required for first run.

## 1) Setup

```bash
python -m venv .venv && source ./.venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Start the three region services (in three terminals)

**Terminal 1 (EU on :8001):**
```bash
REGION_NAME=EU python -m uvicorn region.region_service:app --port 8001 --reload
```

**Terminal 2 (US on :8002):**
```bash
REGION_NAME=US python -m uvicorn region.region_service:app --port 8002 --reload
```

**Terminal 3 (APAC on :8003):**
```bash
REGION_NAME=APAC python -m uvicorn region.region_service:app --port 8003 --reload
```

> These services keep their **own** indexes in memory and write feedback to `./feedback/{REGION}_feedback.csv`.

## 3) Seed some sample documents (new terminal)

```bash
python scripts/index_sample_data.py
```

## 4) Start the coordinator (fuses results with RRF)

**Terminal 4 (Coordinator on :8000):**
```bash
python -m uvicorn coordinator.coordinator_service:app --port 8000 --reload
```

## 5) Ask a question

```bash
python scripts/query.py "How do we scale the service for a traffic spike?"
```

You’ll see a **stitched answer** with citations. By default we **don’t** call an LLM to keep the starter light.
If you have **Ollama** running locally, export this before step 5 to enable summarization:

```bash
export OLLAMA_BASE_URL=http://localhost:11434      # Windows: set OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
```

## 6) Click feedback → (optional) federated re‑ranking

- Mark top results as helpful to produce per‑region feedback:
  ```bash
  python scripts/feedback.py "How do we scale the service for a traffic spike?"
  ```

- Train a **tiny reranker** *federatively* (simulated) with **Flower**. This script spins up three virtual clients
  (EU/US/APAC) that each read their local `feedback/*.csv`, train logistic regression, and **Flower FedAvg** averages the weights.

  ```bash
  python fed/train_reranker_flower.py --rounds 3
  ```

  The coordinator will automatically pick up the saved model at `models/reranker.joblib` for re‑ranking on the next query.

## 7) That’s it. What just happened?

- **Data stayed local:** Each region answered from its own index.
- **Coordinator only merged scores:** It used **RRF** to fuse lists and stitch an answer.
- **Learning stayed decentralized:** The reranker was trained **per region** and averaged via **Flower**, not by pooling raw docs.

## 8) Troubleshooting

- If you see “model not found” errors in the region service, you’re fine—the default backend uses **TF‑IDF** (no big downloads).
- If you want semantic embeddings, set `USE_ST_EMB=1` in each region terminal **after** installing `sentence-transformers`:
  ```bash
  pip install sentence-transformers
  export USE_ST_EMB=1
  ```
- On Windows PowerShell, use `$env:REGION_NAME="EU"` etc.

## 9) File map

```
resea-starter/
  README.md
  requirements.txt
  coordinator/
    coordinator_service.py
    rrf.py
    reranker.py
    settings.py
  region/
    region_service.py
  fed/
    train_reranker_flower.py
  scripts/
    index_sample_data.py
    query.py
    feedback.py
  data/{eu,us,apac}/*.md
  models/            # reranker.joblib saved here after Flower training
  feedback/          # per-region click logs
```
