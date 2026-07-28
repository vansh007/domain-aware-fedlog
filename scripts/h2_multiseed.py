#!/usr/bin/env python
"""
h2_multiseed.py — multi-seed robustness for H2 (DeepLog single-domain vs mixed).

Runs the H2 experiment across seeds through the same code path as a single run
(scripts.run_h2_deeplog.main) and aggregates the per-domain single/mixed F1 and the drop
(single - mixed) into mean +/- std. A negative result (no degradation under mixing) needs
replication just as much as a positive one.

Writes results/h2_multiseed.csv (per-domain mean/std of single, mixed, drop).

Usage:
    python scripts/h2_multiseed.py                 # seeds 0..2 (deep runs are ~4min each)
    python scripts/h2_multiseed.py --seeds 0 1 2 3 4
"""

from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_h2_deeplog import main as run_h2


def main(seeds, rounds):
    per_seed = []
    for s in seeds:
        print(f"\n########## H2 seed {s} ##########")
        res = run_h2(rounds=rounds, local_epochs=1, seed=s, smoke=False,
                     out_path=f"results/h2_deeplog_seed{s}.csv")
        per_seed.append(res)

    rows = []
    print("\n" + "=" * 64)
    print(f"H2 MULTI-SEED SUMMARY (n={len(seeds)} seeds: {seeds}, rounds={rounds})")
    print("=" * 64)
    print(f"{'domain':6s} {'single mean':>12s} {'mixed mean':>12s} {'drop mean':>11s} {'drop std':>9s}")
    for domain in ["hdfs", "bgl"]:
        singles = [r[f"{domain}_single"] for r in per_seed]
        mixeds = [r[f"{domain}_mixed"] for r in per_seed]
        drops = [s - m for s, m in zip(singles, mixeds)]
        sm, mm = statistics.fmean(singles), statistics.fmean(mixeds)
        dm = statistics.fmean(drops)
        ds = statistics.pstdev(drops) if len(drops) > 1 else 0.0
        print(f"{domain:6s} {sm:12.4f} {mm:12.4f} {dm:+11.4f} {ds:9.4f}")
        rows.append({"domain": domain, "single_mean": round(sm, 4), "mixed_mean": round(mm, 4),
                     "drop_mean": round(dm, 4), "drop_std": round(ds, 4), "n": len(seeds),
                     "single_values": " ".join(f"{v:.4f}" for v in singles),
                     "mixed_values": " ".join(f"{v:.4f}" for v in mixeds)})

    out = "results/h2_multiseed.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()
    main(args.seeds, args.rounds)
