import os
from typing import List, Dict, Optional
from fastapi import FastAPI, Body
import requests
from .settings import REGIONS, RRF_K
from .rrf import rrf_merge
from .reranker import load_reranker, apply_reranker

app = FastAPI(title="ReSEA Coordinator", version="0.1.0")

def query_region(region: Dict, q: str, k: int = 3) -> List[Dict]:
    try:
        resp = requests.get(f"{region['base_url']}/search", params={"q": q, "k": k}, timeout=10)
        data = resp.json()
        items = data.get("results", [])
        # annotate with region for fusion
        for it in items:
            it["region"] = data.get("region", region["name"])
        return items
    except Exception as e:
        return []

def llm_summarize(question: str, snippets: List[Dict]) -> Optional[str]:
    # base = os.getenv("OLLAMA_BASE_URL")
    # model = os.getenv("OLLAMA_MODEL", "llama3")

    base = "http://localhost:11434"
    model = "llama3"
    if not base:
        # LLM disabled: return stitched bullet points
        bullets = "\n".join([f"- [{it.get('from', [it.get('region','?')])[0]}] {it['snippet'][:220]}..." for it in snippets[:3]])
        return f"""No local LLM configured — returning a stitched answer.\nQuestion: {question}\nKey points:\n{bullets}"""
    try:
        import json
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an SRE assistant. Write concise, actionable steps, and include citations like [EU], [US], [APAC] inline."},
                {"role": "user", "content": f"Question: {question}\nUse these snippets with their regions in brackets:\n" + "\n\n".join([f"[{','.join(it.get('from', [it.get('region','?')]))}] {it['snippet']}" for it in snippets])}
            ],
            "stream": False
        }
        r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=60)
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        # fall back to stitched
        bullets = "\n".join([f"- [{it.get('from', [it.get('region','?')])[0]}] {it['snippet'][:220]}..." for it in snippets[:3]])
        return f"""LLM call failed ({e}).\nQuestion: {question}\nKey points:\n{bullets}"""

@app.post("/ask")
def ask(payload: Dict = Body(...)):
    q = payload.get("q", "")
    k = int(payload.get("k", 3))
    # 1) Fan-out
    region_lists = [query_region(r, q, k=k) for r in REGIONS]
    # 2) Fuse with RRF
    fused = rrf_merge(region_lists, k=RRF_K)
    # 3) Optional reranker (from federated flower training)
    clf = load_reranker()
    reranked = apply_reranker(fused, clf)
    # 4) Compose answer (LLM optional)
    answer = llm_summarize(q, reranked[:k])
    # 5) Return
    return {
        "question": q,
        "answer": answer,
        "candidates": reranked[:k],
        "regions": [r["name"] for r in REGIONS],
        "llm": True,
        "reranker_loaded": bool(clf),
    }

@app.get("/regions")
def regions():
    return {"regions": REGIONS}

@app.get("/healthz")
def healthz():
    return {"ok": True}
