import requests, sys

q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do we scale the service for a traffic spike?"
# Ask
resp = requests.post("http://localhost:8000/ask", json={"q": q, "k": 3})
data = resp.json()
# Mark the top-1 as helpful in each origin region
for cand in data["candidates"]:
    for region in cand.get("from", []):
        # send feedback to that region's service
        base = {"EU":"http://localhost:8001","US":"http://localhost:8002","APAC":"http://localhost:8003"}[region]
        payload = {
            "query": q,
            "uri": cand["uri"],
            "rank": cand["rank"],
            "retrieval_score": float(cand.get("fused_score", 0.0)),
            "label": 1
        }
        r = requests.post(f"{base}/feedback", json=payload)
        print(f"[{region}] feedback saved ->", r.json())
print("Now run:  python fed/train_reranker_flower.py --rounds 3")
