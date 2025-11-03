from typing import List, Dict
from pathlib import Path
import joblib
import numpy as np
from .settings import RERANKER_PATH

def _features(item: Dict) -> np.ndarray:
    # Simple, consistent features: [rrf_fused_score, inv_rank, snippet_len]
    inv_rank = 1.0 / (item.get("rank", 1) + 1.0)
    snip_len = float(len(item.get("snippet", "")))
    return np.array([float(item.get("fused_score", 0.0)), inv_rank, snip_len], dtype=float)

def load_reranker():
    p = Path(RERANKER_PATH)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

def apply_reranker(candidates: List[Dict], clf) -> List[Dict]:
    if clf is None or len(candidates) == 0:
        return candidates
    X = np.vstack([_features(c) for c in candidates])
    try:
        scores = clf.predict_proba(X)[:, 1]
    except Exception:
        # fallback if predict_proba missing
        scores = clf.decision_function(X)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    # Sort by reranker score then by fused score
    candidates.sort(key=lambda x: (x.get("rerank_score", 0.0), x.get("fused_score", 0.0)), reverse=True)
    for i, c in enumerate(candidates, start=1):
        c["rank"] = i
    return candidates
