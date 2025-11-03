# Kafka Broker Runbook (US)

**Issue:** Broker under-replicated partitions.

**Steps to Resolve:**
1. Run `kafka-topics.sh --describe`.
2. Restart broker if partition count > threshold.
3. Add broker if replication lag > 5000.

**Last updated:** 2023-11-02