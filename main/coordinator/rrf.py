from typing import List, Dict

def rrf_merge(region_results: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion.
    region_results: list over regions, each is list of items with keys: uri, snippet, rank, score, region
    Returns: list of merged items with fused_score.
    """
    fused = {}
    for items in region_results:
        for item in items:
            key = (item["uri"], item["snippet"])
            fused.setdefault(key, {"uri": item["uri"], "snippet": item["snippet"], "from": set(), "min_rank": 10**9, "fused_score": 0.0})
            fused[key]["from"].add(item["region"])
            fused[key]["min_rank"] = min(fused[key]["min_rank"], item["rank"])
            fused[key]["fused_score"] += 1.0 / (k + item["rank"])
    merged = []
    for (_k), v in fused.items():
        v["from"] = sorted(list(v["from"]))
        merged.append(v)
    merged.sort(key=lambda x: (x["fused_score"], -x["min_rank"]), reverse=True)
    # add rank field
    for i, m in enumerate(merged, start=1):
        m["rank"] = i
    return merged
