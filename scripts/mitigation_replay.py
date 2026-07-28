#!/usr/bin/env python
"""
mitigation_replay.py — a replay defense for the (sequential, shared-representation) failure.

H3/mechanism showed that an EMBEDDING-input DeepLog catastrophically forgets HDFS when BGL
arrives sequentially (forgetting ~0.665, HDFS F1 0.70 -> 0.04). This script tests whether a
small experience-replay buffer fixes it: when the new domain (BGL) arrives, we keep a
p-fraction sample of the earlier domain's (HDFS) training windows and train on BGL + buffer
jointly (as an extra federated client), instead of BGL alone.

For each seed we run, from the SAME stage-1 (HDFS-only) model:
  - no-replay continuation on BGL only              (reproduces the failure)
  - replay continuation on BGL + p-fraction HDFS    for p in a small sweep
and we report, per condition: HDFS F1 after (retention), BGL F1 after (new-domain learning),
and HDFS forgetting = F1_before - F1_after. A good defense drives forgetting toward 0 while
keeping BGL F1 high.

All in the embedding condition (the only one that forgets). CPU-first, 1%-data federated
regime, rounds=12 (the H2/H3 plateau). Writes results/mitigation_replay.csv.

Usage:
    python scripts/mitigation_replay.py                       # seeds 0..2, default p-sweep
    python scripts/mitigation_replay.py --seeds 0 1 2 3 4 --fracs 0.05 0.1 0.25 0.5
"""

from __future__ import annotations
import argparse
import csv
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import build_federation
from src.methods.deeplog_fed import FedConfig, train_federated, evaluate

HDFS_CLIENTS = 3
BGL_CLIENTS = 2
DATA_ROOT = "data"
ROUNDS = 12
INPUT_MODE = "embedding"   # the failure condition


def _test_of(fed, domain):
    c = fed.clients_of(domain)[0]
    return c.test_sequences, c.test_labels


def run_seed(seed, fracs):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=ROUNDS, local_epochs=1, seed=seed,
                    input_mode=INPUT_MODE)

    hdfs_clients = [c.normal_train for c in fed.clients_of("hdfs")]
    bgl_clients = [c.normal_train for c in fed.clients_of("bgl")]
    hdfs_pool = [s for client in hdfs_clients for s in client]   # pooled HDFS training seqs

    hdfs_test = _test_of(fed, "hdfs")
    bgl_test = _test_of(fed, "bgl")

    # ---- stage 1: HDFS only ----
    model0 = train_federated(hdfs_clients, cfg)
    f1_before = evaluate(model0, *hdfs_test, cfg).f1
    print(f"[seed {seed}] stage1 HDFS F1_before={f1_before:.4f}  (pool={len(hdfs_pool)} seqs)")

    rows = []

    # ---- no replay (reproduce the failure) ----
    m_nr = train_federated(bgl_clients, cfg, init_model=model0)
    hdfs_nr = evaluate(m_nr, *hdfs_test, cfg).f1
    bgl_nr = evaluate(m_nr, *bgl_test, cfg).f1
    rows.append({"seed": seed, "frac": 0.0, "buffer_seqs": 0,
                 "hdfs_before": round(f1_before, 6), "hdfs_after": round(hdfs_nr, 6),
                 "bgl_after": round(bgl_nr, 6), "forgetting": round(f1_before - hdfs_nr, 6)})
    print(f"[seed {seed}]   no-replay      HDFS_after={hdfs_nr:.4f}  BGL_after={bgl_nr:.4f}  "
          f"forget={f1_before - hdfs_nr:+.4f}")

    # ---- replay: BGL + p-fraction HDFS buffer as an extra client ----
    for p in fracs:
        rng = random.Random(1000 + seed)          # deterministic buffer per (seed)
        k = max(1, int(round(p * len(hdfs_pool))))
        buffer = rng.sample(hdfs_pool, k)
        m_r = train_federated(bgl_clients + [buffer], cfg, init_model=model0)
        hdfs_r = evaluate(m_r, *hdfs_test, cfg).f1
        bgl_r = evaluate(m_r, *bgl_test, cfg).f1
        rows.append({"seed": seed, "frac": p, "buffer_seqs": k,
                     "hdfs_before": round(f1_before, 6), "hdfs_after": round(hdfs_r, 6),
                     "bgl_after": round(bgl_r, 6), "forgetting": round(f1_before - hdfs_r, 6)})
        print(f"[seed {seed}]   replay p={p:<5} HDFS_after={hdfs_r:.4f}  BGL_after={bgl_r:.4f}  "
              f"forget={f1_before - hdfs_r:+.4f}  (buf={k})")
    return rows


def main(seeds, fracs):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(s, fracs))

    os.makedirs("results", exist_ok=True)
    out = "results/mitigation_replay.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "frac", "buffer_seqs", "hdfs_before",
                                          "hdfs_after", "bgl_after", "forgetting"])
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 72)
    print(f"REPLAY MITIGATION SUMMARY (n={len(seeds)} seeds: {seeds}, input={INPUT_MODE})")
    print("=" * 72)
    print(f"{'frac':>6} {'buf':>6}  {'HDFS_after':>12} {'BGL_after':>12} {'forgetting':>12}")
    for p in [0.0] + list(fracs):
        sub = [r for r in all_rows if r["frac"] == p]
        if not sub:
            continue

        def ms(key):
            v = [r[key] for r in sub]
            return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
        hm, hs = ms("hdfs_after"); bm, bs = ms("bgl_after"); fm, fs = ms("forgetting")
        buf = round(statistics.fmean([r["buffer_seqs"] for r in sub]))
        label = "no-replay" if p == 0.0 else f"p={p}"
        print(f"{label:>6} {buf:>6}  {hm:>6.4f}+/-{hs:.4f} {bm:>6.4f}+/-{bs:.4f} "
              f"{fm:>+6.4f}+/-{fs:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5])
    args = ap.parse_args()
    main(args.seeds, args.fracs)
