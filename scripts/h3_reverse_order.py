#!/usr/bin/env python
"""
h3_reverse_order.py — is the sequential-arrival forgetting an artifact of arrival ORDER?

The mechanism experiment showed embedding-DeepLog forgets HDFS when domains arrive HDFS->BGL.
A reviewer will ask: is that specific to HDFS-being-first (e.g., because BGL is "bigger"), or
is it a symmetric property of shared-representation sequential arrival? This script reverses
the order (BGL first, then HDFS) and measures how much the EARLIER domain (now BGL) is
forgotten, for BOTH input encodings. If embedding forgets in this direction too and scalar
does not, the 2x2 map's failing cell is order-robust, not a one-off.

Protocol per seed and per input_mode in {scalar, embedding}:
  stage 1: federated train on BGL clients        -> BGL F1_before
  stage 2: continue federated training on HDFS    -> BGL F1_after (retention), HDFS F1 (learned)
  forgetting = BGL F1_before - BGL F1_after
All at the H2/H3 plateau (rounds=12), CPU-first. Writes results/h3_reverse_order.csv.

Read-only import of the DeepLog harness; modifies no existing script or result.

Usage:
    python scripts/h3_reverse_order.py                # seeds 0..2, both input modes
    python scripts/h3_reverse_order.py --seeds 0 1 2
"""

from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import build_federation
from src.methods.deeplog_fed import FedConfig, train_federated, evaluate

HDFS_CLIENTS = 3
BGL_CLIENTS = 2
DATA_ROOT = "data"
ROUNDS = 12


def _test_of(fed, domain):
    c = fed.clients_of(domain)[0]
    return c.test_sequences, c.test_labels


def run_seed(seed, input_mode):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=ROUNDS, local_epochs=1, seed=seed,
                    input_mode=input_mode)

    hdfs_clients = [c.normal_train for c in fed.clients_of("hdfs")]
    bgl_clients = [c.normal_train for c in fed.clients_of("bgl")]
    bgl_test = _test_of(fed, "bgl")
    hdfs_test = _test_of(fed, "hdfs")

    # ---- reverse order: BGL first, then HDFS ----
    model0 = train_federated(bgl_clients, cfg)                 # stage 1: BGL
    bgl_before = evaluate(model0, *bgl_test, cfg).f1
    model1 = train_federated(hdfs_clients, cfg, init_model=model0)  # stage 2: HDFS
    bgl_after = evaluate(model1, *bgl_test, cfg).f1            # retention of earlier (BGL)
    hdfs_after = evaluate(model1, *hdfs_test, cfg).f1          # newly learned domain
    forget = bgl_before - bgl_after
    print(f"[seed {seed}] {input_mode:9s} BGL_before={bgl_before:.4f}  BGL_after={bgl_after:.4f}"
          f"  HDFS_after={hdfs_after:.4f}  forget={forget:+.4f}")
    return {"seed": seed, "input_mode": input_mode,
            "bgl_before": round(bgl_before, 6), "bgl_after": round(bgl_after, 6),
            "hdfs_after": round(hdfs_after, 6), "forgetting": round(forget, 6)}


def main(seeds):
    rows = []
    for mode in ("scalar", "embedding"):
        for s in seeds:
            rows.append(run_seed(s, mode))

    os.makedirs("results", exist_ok=True)
    out = "results/h3_reverse_order.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "input_mode", "bgl_before", "bgl_after",
                                          "hdfs_after", "forgetting"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 68)
    print(f"H3 REVERSE-ORDER (BGL->HDFS) SUMMARY (n={len(seeds)} seeds: {seeds})")
    print("=" * 68)
    for mode in ("scalar", "embedding"):
        sub = [r for r in rows if r["input_mode"] == mode]

        def ms(key):
            v = [r[key] for r in sub]
            return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
        bb, _ = ms("bgl_before"); ba, bas = ms("bgl_after"); fm, fs = ms("forgetting")
        print(f"{mode:9s}  BGL_before={bb:.4f}  BGL_after={ba:.4f}+/-{bas:.4f}  "
              f"forgetting={fm:+.4f}+/-{fs:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    main(args.seeds)
