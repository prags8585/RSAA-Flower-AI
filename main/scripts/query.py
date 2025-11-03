import sys, requests, json
q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do we scale the service for a traffic spike?"
r = requests.post("http://localhost:8000/ask", json={"q": q, "k": 3})
data = r.json()
print("# Answer")
print(data["answer"])
print("\n# Top Candidates")
for c in data["candidates"]:
    print(f"- [{','.join(c.get('from', []))}] {c['uri']}  (fused={c.get('fused_score', 0):.3f})")
