#!/usr/bin/env python
"""
transformer_mechanism.py — does the forgetting mechanism generalize beyond DeepLog?

The paper shows a scalar-input LSTM (DeepLog) is safe under sequential arrival while a learned
embedding catastrophically forgets, and argues the deciding factor is the INPUT REPRESENTATION
(disjoint magnitude ranges vs. a shared feature space), not the architecture. If that argument is
right, a completely different modern architecture --- a self-attention Transformer --- should show
the SAME 2x2: simultaneous mixing safe under both inputs; sequential arrival catastrophic under the
embedding (shared) input and safe under the scalar (disjoint) input.

This script runs exactly that test with a tiny Transformer next-event detector (same window, same
top-g rule, same FedAvg loop as DeepLog, via model_type='transformer'), for both input encodings,
over multiple seeds. It reuses the identical harness so the comparison to DeepLog is apples-to-apples.

Per (input_mode, seed):
  stage-1 HDFS model  -> HDFS F1 (also the H2 single-domain HDFS point)
  continue on BGL     -> HDFS F1 after (H3 forgetting), BGL F1 after
  simultaneous mixed  -> HDFS F1, BGL F1 (H2)
  single-domain BGL   -> BGL F1 (H2 single-domain BGL point)
Writes results/transformer_mechanism.csv. CPU-first, rounds=12 (the H2/H3 plateau).

Usage:
    python scripts/transformer_mechanism.py                 # seeds 0..2, both inputs
    python scripts/transformer_mechanism.py --seeds 0 1 2 --rounds 12
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


def _test_of(fed, domain):
    c = fed.clients_of(domain)[0]
    return c.test_sequences, c.test_labels


def run(seed, input_mode, rounds):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=rounds, local_epochs=1, seed=seed,
                    input_mode=input_mode, model_type="transformer")

    hdfs_clients = [c.normal_train for c in fed.clients_of("hdfs")]
    bgl_clients = [c.normal_train for c in fed.clients_of("bgl")]
    hdfs_test, bgl_test = _test_of(fed, "hdfs"), _test_of(fed, "bgl")

    # stage-1 HDFS (serves as H3 stage-1 AND H2 single-domain HDFS)
    m0 = train_federated(hdfs_clients, cfg)
    hdfs_before = evaluate(m0, *hdfs_test, cfg).f1

    # H3 sequential: continue on BGL
    m1 = train_federated(bgl_clients, cfg, init_model=m0)
    hdfs_after = evaluate(m1, *hdfs_test, cfg).f1
    bgl_after_seq = evaluate(m1, *bgl_test, cfg).f1
    forget = hdfs_before - hdfs_after

    # H2 simultaneous: mixed
    m_mixed = train_federated(hdfs_clients + bgl_clients, cfg)
    hdfs_mixed = evaluate(m_mixed, *hdfs_test, cfg).f1
    bgl_mixed = evaluate(m_mixed, *bgl_test, cfg).f1

    # H2 single-domain BGL
    m_bgl = train_federated(bgl_clients, cfg)
    bgl_single = evaluate(m_bgl, *bgl_test, cfg).f1

    print(f"[seed {seed}] {input_mode:9s}  H3 forget={forget:+.4f} (HDFS {hdfs_before:.3f}->{hdfs_after:.3f})"
          f"  |  H2 HDFS drop={hdfs_before - hdfs_mixed:+.4f}  BGL drop={bgl_single - bgl_mixed:+.4f}")
    return {"seed": seed, "input_mode": input_mode,
            "hdfs_before": round(hdfs_before, 6), "hdfs_after_seq": round(hdfs_after, 6),
            "bgl_after_seq": round(bgl_after_seq, 6), "forgetting": round(forget, 6),
            "hdfs_single": round(hdfs_before, 6), "hdfs_mixed": round(hdfs_mixed, 6),
            "bgl_single": round(bgl_single, 6), "bgl_mixed": round(bgl_mixed, 6)}


def main(seeds, rounds):
    rows = []
    for mode in ("scalar", "embedding"):
        for s in seeds:
            rows.append(run(s, mode, rounds))

    os.makedirs("results", exist_ok=True)
    out = "results/transformer_mechanism.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def ms(v):
        return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)

    print("\n" + "=" * 72)
    print(f"TRANSFORMER MECHANISM SUMMARY (n={len(seeds)} seeds: {seeds}, rounds={rounds})")
    print("=" * 72)
    for mode in ("scalar", "embedding"):
        sub = [r for r in rows if r["input_mode"] == mode]
        fm, fs = ms([r["forgetting"] for r in sub])
        hd, hds = ms([r["hdfs_single"] - r["hdfs_mixed"] for r in sub])
        bd, bds = ms([r["bgl_single"] - r["bgl_mixed"] for r in sub])
        af, afs = ms([r["hdfs_after_seq"] for r in sub])
        print(f"{mode:9s}  H3 forgetting={fm:+.4f}+/-{fs:.4f} (HDFS after {af:.3f}+/-{afs:.3f})  "
              f"|  H2 HDFS drop={hd:+.4f}+/-{hds:.4f}  BGL drop={bd:+.4f}+/-{bds:.4f}")
    print("\nEXPECTED (if the mechanism generalizes): embedding forgets (large +), scalar does not (~0);")
    print("both H2 drops are ~0. Whatever the run shows is the real result.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()
    main(args.seeds, args.rounds)
