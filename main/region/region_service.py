import os
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from pathlib import Path
import csv
import threading
import re
try:
    import yaml 
except Exception:
    yaml = None

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS: list = []          # each: {"uri": str, "text": str, "meta": dict}
VECS: np.ndarray = None  # TF-IDF matrix (n_docs, dim)
_vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)

def _fit_vectors():
    """(Re)build TF-IDF matrix from DOCS."""
    global VECS
    texts = [d["text"] for d in DOCS] if DOCS else []
    if not texts:
        VECS = np.zeros((0, 1), dtype=np.float32)
        return
    VECS = _vectorizer.fit_transform(texts).astype(np.float32).toarray()

def _embed(text: str) -> np.ndarray:
    """Embed a query using same vectorizer."""
    if VECS is None or VECS.size == 0:
        _fit_vectors()
    return _vectorizer.transform([text]).astype(np.float32).toarray()[0]

def _cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a (dim,) and matrix B (n, dim)."""
    an = np.linalg.norm(a) + 1e-12
    Bn = np.linalg.norm(B, axis=1) + 1e-12
    return (B @ (a / an)) / Bn

FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

SENSITIVE_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED:email]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED:ip]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED:aws_key]"),
    (re.compile(r"\b(?:secret|token|apikey|api_key|password)\s*[:=]\s*[^\s,]+", re.I), "[REDACTED:secret]"),
    (re.compile(r"\b[0-9]{13,19}\b"), "[REDACTED:card]"),  # naïve CC-like
]

def strip_front_matter(text: str):
    m = FM_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}
    return meta, text[m.end():]

def redact_text(text: str) -> str:
    # custom mask markers
    text = re.sub(r"\[MASK](.*?)\[/MASK]", "[REDACTED]", text, flags=re.DOTALL | re.IGNORECASE)
    # automatic patterns
    for rx, repl in SENSITIVE_PATTERNS:
        text = rx.sub(repl, text)
    return text


REGION = os.getenv("REGION_NAME", "EU").upper()
USE_ST_EMB = os.getenv("USE_ST_EMB", "0") == "1"

# ---- Embedding backends ----
class Embedder:
    def fit(self, texts: List[str]): ...
    def encode(self, texts: List[str]) -> np.ndarray: ...

class TFIDFEmbedder(Embedder):
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = None

    def fit(self, texts: List[str]):
        self.matrix = self.vec.fit_transform(texts)
        self.matrix = normalize(self.matrix, norm="l2")

    def encode(self, texts: List[str]) -> np.ndarray:
        m = self.vec.transform(texts)
        m = normalize(m, norm="l2")
        return m

try:
    if USE_ST_EMB:
        from sentence_transformers import SentenceTransformer
        class STEmbedder(Embedder):
            def __init__(self):
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.docs = None  # will keep dense vectors
            def fit(self, texts: List[str]):
                self.docs = self.model.encode(texts, normalize_embeddings=True)
            def encode(self, texts: List[str]) -> np.ndarray:
                return self.model.encode(texts, normalize_embeddings=True)
        BACKEND = "sentence-transformers"
        emb = STEmbedder()
    else:
        BACKEND = "tfidf"
        emb = TFIDFEmbedder()
except Exception:
    BACKEND = "tfidf"
    emb = TFIDFEmbedder()

# ---- App state ----
app = FastAPI(title=f"ReSEA Region: {REGION}", version="0.1.0")
DOCS: List[Dict] = []
MUTEX = threading.Lock()

class IndexItem(BaseModel):
    uri: str
    text: str

class IndexBatch(BaseModel):
    items: List[IndexItem]

def rebuild_index():
    texts = [d["text"] for d in DOCS]
    if not texts:
        return
    emb.fit(texts)

@app.get("/healthz")
def healthz():
    return {"region": REGION, "ok": True, "backend": BACKEND, "docs": len(DOCS)}

# @app.post("/index_batch")
# def index_batch(batch: IndexBatch):
#     with MUTEX:
#         for it in batch.items:
#             DOCS.append({"uri": it.uri, "text": it.text})
#         rebuild_index()
#     return {"region": REGION, "indexed": len(batch.items), "total": len(DOCS), "backend": BACKEND}

@app.post("/index_batch")
def index_batch(payload: Dict[str, Any]):
    global DOCS, VECS  # if your code uses these
    items = payload.get("items", [])
    added = 0

    for it in items:
        uri = it["uri"]
        raw = it["text"]

        meta, body = strip_front_matter(raw)

        # Skip entire doc if requested
        if str(meta.get("mask", "")).lower() in ("true", "1") or \
           str(meta.get("do_not_index", "")).lower() in ("true", "1"):
            continue

        # Redact content (you can make this always-on or gated by meta['redact'])
        if str(meta.get("redact", "true")).lower() in ("true", "1"):
            body = redact_text(body)
        else:
            body = redact_text(body)  # safe default: still redact obvious secrets

        # Store doc (adjust to your structure)
        DOCS.append({"uri": uri, "text": body, "meta": meta})
        added += 1

    _fit_vectors()  # call your existing embedding/index refresh
    return {"ok": True, "region": REGION, "added": added, "total": len(DOCS)}



@app.get("/search")
def search(q: str, k: int = 3):
    global DOCS, VECS

    # Ensure vectors exist
    if VECS is None or VECS.size == 0:
        _fit_vectors()
    if VECS.size == 0 or not DOCS:
        return {"region": REGION, "results": []}

    # 1) Embed query and score docs
    qvec = _embed(q)                    # (dim,)
    sims = _cosine_sim(qvec, VECS)      # (n_docs,)

    # 2) Top-k indices by similarity
    k = max(1, min(int(k), len(DOCS)))
    top_idx = np.argsort(-sims)[:k]

    # 3) Build sanitized results
    results = []
    for rank, i in enumerate(top_idx, start=1):
        doc = DOCS[i]
        txt = redact_text(doc["text"])      # ensure no leak at response time
        snippet = txt[:600]                  # <— exactly here
        results.append({
            "uri": doc["uri"],
            "snippet": snippet,
            "score": float(sims[i]),
            "rank": rank,
            "region": REGION,
        })

    return {"region": REGION, "results": results}


# @app.get("/search")
# def search(q: str, k: int = 3):
#     # ... your existing retrieval to get hits: List[Dict{uri,text,score,rank}]
#     hits = _search_impl(q, k)  # pseudo: your existing code

#     sanitized = []
#     for h in hits:
#         txt = redact_text(h["text"])  # double-check redaction
#         # make a short snippet from sanitized text
#         snippet = txt[:600]
#         sanitized.append({
#             "uri": h["uri"],
#             "snippet": snippet,
#             "score": float(h["score"]),
#             "rank": int(h["rank"]),
#             "region": REGION,
#         })
#     return {"region": REGION, "results": sanitized}


class FeedbackItem(BaseModel):
    query: str
    uri: str
    rank: int
    retrieval_score: float
    label: int  # 1 = helpful, 0 = not helpful

@app.post("/feedback")
def feedback(item: FeedbackItem):
    # Simple 3-feature vector expected by reranker: score, inv_rank, snippet_len
    snip_len = 0
    for d in DOCS:
        if d["uri"] == item.uri:
            snip_len = len(d["text"][:400])
            break
    inv_rank = 1.0 / (item.rank + 1.0)
    row = [item.retrieval_score, inv_rank, snip_len, item.label]
    Path("feedback").mkdir(exist_ok=True)
    with open(f"feedback/{REGION}_feedback.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    return {"ok": True, "region": REGION, "saved_to": f"feedback/{REGION}_feedback.csv"}
