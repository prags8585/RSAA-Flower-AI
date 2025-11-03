# MySQL Runbook (APAC)

**Issue:** Slow queries detected on orders table.

**Steps to Resolve:**
1. Run `EXPLAIN` on problematic queries.
2. Add index to `customer_id` column if missing.
3. If replication delay > 10s, resync replica.

**Last updated:** 2023-09-18