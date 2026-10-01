# Domain-Aware Federated Log Anomaly Detection — A Safety Map for Heterogeneous, Evolving Federations

When organizations run **federated** anomaly detection over **different, evolving** log sources
(the real deployment), the detectors can **silently stop working** — no error, no crash — losing up
to **99.97 % of their anomaly coverage**. This project shows exactly when that happens, proves why,
and ships fixes that cost a few bytes, plus a **pre-deployment auditor** that catches the risk before
you train.

> **The headline, in measured numbers:** domain-blind mixing drops HDFS Length recall `0.369 → 0.0001`
> (6,209 of 16,838 anomalies silently missed); onboarding a new domain erases `94.5 %` of a deep
> model's detection. The safeguards cost **48 bytes** (per-domain ranges), a replay buffer **smaller
> than the model it protects**, and a **0.002 ms** check. See `results/impact_asymmetry.png`.

Target venue: IEEE Access. Paper source: [`docs/paper/main.tex`](docs/paper/main.tex).

---

## Why it matters to organizations

Federated log analytics is used precisely where logs **can't** be centralized — banks, healthcare,
telecom/IoT, multi-tenant clouds, critical infrastructure — all of which monitor **many** different
log sources and **add new ones continuously**. Continuous *effective* monitoring is also a mandated
control (GDPR Art. 32, HIPAA, PCI-DSS, NIS2). A monitoring system that silently degrades on the next
onboarding is both a security blind spot and a compliance gap. Full motivation and the measured stakes:
[`docs/IMPACT_AND_MOTIVATION.md`](docs/IMPACT_AND_MOTIVATION.md).

## What we found

- **H1 — range aggregation collapses** on the tighter domain under mixing (HDFS Length `0.538 → 0.000`);
  a **domain-aware** fix restores it (`→ 0.561`), routing with **no oracle tag** (100 % accuracy). We
  **prove** this as a containment condition (Proposition 1).
- **H2 — simultaneous mixing is safe** for deep models; **H3 — sequential onboarding catastrophically
  forgets** the earlier domain (`0.70 → 0.04`) *iff* the representation shares a feature space. A clean
  **2×2 safety map**; the failure is **directional** (broad-after-narrow).
- **Generality:** holds across **3 domains** (HDFS, BGL, OpenStack) and **2 architectures** (LSTM **and**
  a Transformer — which forgets even under the scalar input that leaves the LSTM immune).
- **Fixes, compared:** a small **replay buffer** (`0.665 → 0.004`) and **EWC**, benchmarked against
  **A-GEM** — simple replay is Pareto-competitive. All with near-zero overhead.
- **A validated auditor:** from summary statistics alone it predicts which failures a planned
  federation will hit — correct on **14/15** measured configurations.

## The deployable auditor (`fedlog_audit`)

```bash
pip install -e .                 # auditor core has zero third-party deps
fedlog-audit --example           # the canonical HDFS+BGL+OpenStack audit

# gate a planned rollout in CI (exits non-zero on HIGH/CRITICAL risk):
fedlog-audit --domain hdfs:0-33:4-300 --domain bgl:33-427:1-900 \
  --arrival sequential --representation embedding --detector deep \
  --architecture transformer --order hdfs,bgl --json
# ...passes once the fix is declared:
fedlog-audit ... --mitigation replay
```

It reads only each domain's event-id block and length range — **never raw logs** — preserving the
federated privacy model. Ready-to-use GitHub Action: [`.github/workflows/fedlog-audit.yml`](.github/workflows/fedlog-audit.yml).
Interactive dashboard: see the Artifact link in the project notes.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
./reproduce.sh                   # runs the lightweight experiments + analyses + figures
python tests/test_auditor.py     # 9 auditor unit tests
```
The deep-model runs (DeepLog/Transformer, H2/H3/mechanism/mitigations) are CPU-bound and slower; each
has its own `scripts/*.py`. Every number in the paper traces to a CSV in `results/`.

## Repository map

| Path | What |
|---|---|
| `fedlog_audit/` | the pip-installable safety auditor + CLI (the product) |
| `src/` | merged cross-domain vocabulary, dataloader, detectors, aggregation, metrics |
| `scripts/` | one runnable file per experiment + plotting + impact analysis |
| `results/` | CSVs and figures (every paper number traces here) |
| `docs/paper/main.tex` | the paper (8 figures, 13 tables, a proposition + proof) |
| `docs/IMPACT_AND_MOTIVATION.md` | organizational-importance research notes |
| `docs/RESEARCH_LOG.md`, `docs/TILLNOW.md` | lab notebook + session handoff |

## Honest limitations

Our federated DeepLog operates at F1 ≈ 0.70, not the reference's ~0.93 (a documented scalar-input
capacity ceiling; the forgetting result is a *relative* effect, unaffected). OpenStack ships only 4
labelled (order-based) anomalies, so it is used for the range/routing generality, not as a headline
detection score. Both are discussed openly in the paper's Threats to Validity.
