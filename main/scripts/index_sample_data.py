import requests, os, glob, pathlib, json

regions = [
    ("EU", "http://localhost:8001"),
    ("US", "http://localhost:8002"),
    ("APAC","http://localhost:8003"),
]

base = pathlib.Path(__file__).resolve().parents[1] / "data"
count = 0
for name, url in regions:
    folder = base / name.lower()
    items = []
    for p in sorted(folder.glob("*.md")):
        items.append({"uri": f"{name}:{p.name}", "text": p.read_text(encoding="utf-8")})
    if not items:
        continue
    r = requests.post(f"{url}/index_batch", json={"items": items})
    r.raise_for_status()
    print(f"[{name}] indexed {len(items)} from {folder}")
    count += len(items)

print(f"Total indexed: {count}")
