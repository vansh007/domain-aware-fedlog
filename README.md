# Domain-Aware Federated Log Anomaly Detection under Sequential Domain Arrival

Testing whether the invariance and bounded-growth properties reported for federated
log anomaly detection survive a **heterogeneous, evolving** federation.

**Status:** Research scaffold. Not yet run. See `docs/TILLNOW.md` for progress.

**Target venue:** IEEE Access.

---

## The idea in three lines

1. Prior work evaluates federated log anomaly detection only where every client holds
   the *same* dataset (quantity skew).
2. We put HDFS clients and BGL clients in *one* federation, and make new domains arrive
   *sequentially*.
3. We test whether "lightweight methods are distribution-invariant" and "model size is
   bounded" still hold. We predict they don't — and propose a domain-aware aggregation
   fix.

Full reasoning: `docs/RULES_AND_GOALS.md`. Reference-paper facts:
`docs/REFERENCE_PAPER_NOTES.md`.

---

## Quick start

> This scaffold gives you the structure, the merged-vocabulary dataloader design, the
> metrics, configs, and plotting. The reference implementation's model code is **not**
> bundled (it is GPL-3.0 — clone it yourself). Week 1 wires the two together.

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12.x recommended (matches the reference repo).

### 2. Get the reference implementation (Week 1)

```bash
git clone https://github.com/ait-aecid/comparison-fed-centr-efficient-ad reference_repo
# Read reference_repo/Readme.md. Note: they use Flower + config YAMLs, like us.
```

### 3. Get the data

Download HDFS and BGL from LogHub (https://github.com/logpai/loghub) into `data/`.
See `data/README.md` for the expected layout. Datasets are **git-ignored** — never
commit raw logs.

### 4. Week-1 gate: reproduce the baseline

```bash
python scripts/reproduce_baseline.py --dataset hdfs
python scripts/reproduce_baseline.py --dataset bgl
```

If these do not roughly match the reference paper's numbers (see
`docs/REFERENCE_PAPER_NOTES.md`), **stop and report** before writing any mixed-domain
code. That is the single most important checkpoint in the project.

### 5. Week-2: the experiment that matters

```bash
python scripts/run_experiment.py --config configs/hdfs_bgl_mixed.yaml
```

This is the config that does not exist in the reference repo. It builds one federation
with 3 HDFS clients + 2 BGL clients over a merged vocabulary, and reports per-domain F1.

---

## Repository layout

```
domain-aware-fedlog/
├── CLAUDE.md                  # read first — project memory for Claude Code
├── README.md                  # you are here
├── requirements.txt
├── configs/                   # experiment configs (YAML)
│   ├── hdfs_iid.yaml          # single-domain baseline
│   ├── bgl_iid.yaml           # single-domain baseline
│   ├── hdfs_bgl_mixed.yaml    # ⭐ the mixed federation (H1, H2)
│   └── sequential_arrival.yaml# ⭐ HDFS then BGL (H3)
├── src/
│   ├── dataloader.py          # ⭐ merged-vocabulary loader — the core new work
│   ├── vocabulary.py          # global template dictionary across domains
│   ├── methods/               # anomaly detection methods
│   │   ├── length.py          # range-based (predicted to break under mixing)
│   │   ├── known_events.py    # set-union based
│   │   └── deeplog_stub.py    # thin wrapper over reference DeepLog
│   ├── aggregation.py         # FedAvg + our domain-aware strategy (stub)
│   └── metrics.py             # F1, forgetting, model-size growth
├── scripts/
│   ├── reproduce_baseline.py  # Week-1 gate
│   ├── run_experiment.py      # main entry point
│   ├── gather_results.py      # collate results/*.csv
│   └── plot_results.py        # forgetting curve, size-growth plot
├── experiments/               # one append-only log per run
├── results/                   # CSVs + plots (git-ignored)
└── docs/
    ├── TILLNOW.md             # progress log — update every session
    ├── RULES_AND_GOALS.md     # the "why", scope, non-negotiables
    ├── REFERENCE_PAPER_NOTES.md # distilled facts (avoid re-reading the PDF)
    └── DECISIONS.md           # decision log with rationale
```

⭐ = where the actual contribution lives.

---

## What is real vs stub in this scaffold

| Component | State |
|---|---|
| Project structure, configs, docs | Complete |
| `vocabulary.py` (merged dictionary) | Working reference implementation |
| `dataloader.py` | Working skeleton + the merge logic; parser hook is a TODO |
| `methods/length.py`, `known_events.py` | Working (these are simple by design) |
| `metrics.py` (F1, forgetting, size) | Working |
| `deeplog_stub.py` | Stub — wires to the reference repo's DeepLog |
| `aggregation.py` domain-aware strategy | **Stub — this is your Week-4 research** |
| Any experimental numbers | **None. You produce these by running.** |

---

## License

Your own code: choose a license (MIT is fine for a paper artifact). Note the reference
implementation is **GPL-3.0**; if you import or adapt its code directly, GPL obligations
apply to derived files. Keeping our wrappers separate (calling their scripts as a
subprocess) avoids tangling licenses — see `docs/DECISIONS.md`.
