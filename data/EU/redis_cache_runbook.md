# Redis Cache Runbook (EU)

**Issue:** Cache miss ratio above 80%.

**Steps to Resolve:**
1. Check Redis cluster health with `redis-cli info`.
2. Flush only non-critical keys using `EVICT`.
3. If memory > 80%, scale cache cluster nodes.

**Last updated:** 2023-07-14