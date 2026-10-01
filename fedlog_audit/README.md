# fedlog-audit

**A pre-deployment safety auditor + domain-aware fixes for heterogeneous / evolving
federated log anomaly detection.**

This packages the findings of the paper *"Domain-Aware Federated Log Anomaly Detection
under Heterogeneous and Sequentially Arriving Domains"* into a small, usable tool. The core
auditor is **pure Python (zero dependencies)** and works from **summary statistics only** —
each domain's event-id block and its normal-sequence length range — so a federation
coordinator can audit a *planned* deployment without touching any raw logs. That is the same
privacy posture federated learning itself assumes.

## What problem it solves

Two "reassuring" published properties of federated log anomaly detection break in a realistic
heterogeneous/evolving federation. This tool tells you, *before you train*, which failure
you will hit and prescribes the fix:

| Failure | When it fires | Fix the tool prescribes |
|---|---|---|
| **Range-aggregation collapse** (H1) | a broad-length domain shares a global Length range with a tight one | per-domain (domain-aware) ranges |
| **Unbounded growth** (C2) | set-union detector + domains keep arriving | cap the vocabulary / use a fixed-size model |
| **Catastrophic forgetting** (H3) | deep model + *sequential* arrival + *shared* representation + broad-after-narrow | ≥10% replay buffer (tuning-free) or EWC |
| **Routing** | disjoint vocabularies → tag-free; overlapping → open case | vocab-block routing (no oracle tag) |

Every rule cites the measured result it came from.

## Install

```bash
pip install -e .            # auditor only (no third-party deps)
pip install -e ".[research]"  # + numpy/torch/matplotlib to run the experiments in src/
```

## CLI

```bash
# canonical HDFS+BGL example from the paper (both the H1 and the forgetting cell)
fedlog-audit --example

# audit your own federation
fedlog-audit \
  --domain hdfs:0-33:4-300 \
  --domain bgl:33-427:1-900 \
  --arrival sequential --representation embedding --detector deep --order hdfs,bgl
```
`--domain` is `name:idlo-idhi:lenmin-lenmax`. The CLI exits non-zero on any HIGH/CRITICAL
risk, so it can gate a deployment pipeline.

### CI/CD gate

The auditor is designed to run in continuous integration before a federation change:

```bash
# fails the build if the planned onboarding would catastrophically forget
fedlog-audit --domain hdfs:0-33:4-300 --domain bgl:33-427:1-900 \
  --arrival sequential --representation embedding --detector deep \
  --architecture transformer --order hdfs,bgl --json

# the SAME rollout passes once the fix is declared
fedlog-audit ... --mitigation replay --json
```

- `--architecture lstm|transformer` — an attention model shares feature space even under a
  scalar id, so it is flagged where an LSTM would be safe.
- `--mitigation domain-aware,replay,ewc,bounded-vocab` — declare the fixes already in place;
  the auditor clears the risks they address, so a *fixed* configuration passes the gate.
- `--json` — machine-readable report for logging/gating.

A ready-to-use GitHub Action is in `.github/workflows/fedlog-audit.yml`. The auditor's
predictions are validated against measured outcomes (14/15 configurations;
`results/auditor_validation.csv`).

## Python API

```python
from fedlog_audit import DomainSpec, audit_federation

report = audit_federation(
    [DomainSpec("hdfs", (0, 33), (4, 300)),
     DomainSpec("bgl", (33, 427), (1, 900))],
    arrival="sequential", representation="embedding", detector="deep",
    order=["hdfs", "bgl"])

print(report.render())        # human-readable audit
report.worst_level            # 'CRITICAL'
for r in report.risks:
    print(r.name, r.level, r.recommendation)
```

The fix implementations themselves (domain-aware aggregation, vocab-block router, the
lightweight detectors) are re-exported when the research code is on the path:

```python
from fedlog_audit import DomainAwareLength, LengthDetector, KnownEventsDetector
```

## Tests

```bash
python tests/test_auditor.py     # or: pytest tests/test_auditor.py
```
