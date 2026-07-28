#!/usr/bin/env python
"""
mechanism_multiseed.py — replicate the representation mechanism across seeds.

Runs, for each seed:
  - H3 sequential arrival with EMBEDDING input (the catastrophic-forgetting condition), and
  - H2 simultaneous mixing with EMBEDDING input (the robust condition),
through the same code paths as the single runs, then aggregates:
  - DeepLog HDFS forgetting after BGL arrival (H3 embedding) -> mean +/- std
  - DeepLog HDFS/BGL single-vs-mixed drop (H2 embedding)     -> mean +/- std

Writes results/mechanism_multiseed.csv.

Usage:
    python scripts/mechanism_multiseed.py                 # seeds 0..2
    python scripts/mechanism_multiseed.py --seeds 0 1 2 3 4
"""

from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_h3_sequential import main as run_h3
from scripts.run_h2_deeplog import main as run_h2


def main(seeds, rounds):
    h3, h2 = [], []
    for s in seeds:
        print(f"\n########## MECHANISM seed {s} — H3 embedding ##########")
        h3.append(run_h3(rounds, s, do_deeplog=True, input_mode="embedding",
                         out_path=f"results/h3_sequential_embedding_seed{s}.csv"))
        print(f"\n########## MECHANISM seed {s} — H2 embedding ##########")
        h2.append(run_h2(rounds=rounds, local_epochs=1, seed=s, smoke=False,
                         input_mode="embedding",
                         out_path=f"results/h2_deeplog_embedding_seed{s}.csv"))

    def ms(vals):
        return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    # H3 embedding: DeepLog HDFS forgetting
    forget_vals = [r["dl_forget"] for r in h3]
    after_vals = [r["dl_hdfs_after"] for r in h3]
    fmean, fstd = ms(forget_vals)
    amean, astd = ms(after_vals)

    # H2 embedding: per-domain drop
    hdfs_drop = [r["hdfs_single"] - r["hdfs_mixed"] for r in h2]
    bgl_drop = [r["bgl_single"] - r["bgl_mixed"] for r in h2]
    hdm, hds = ms(hdfs_drop)
    bdm, bds = ms(bgl_drop)

    rows = [
        {"experiment": "H3_embedding", "metric": "deeplog_hdfs_forgetting",
         "mean": round(fmean, 4), "std": round(fstd, 4), "n": len(seeds),
         "values": " ".join(f"{v:.4f}" for v in forget_vals)},
        {"experiment": "H3_embedding", "metric": "deeplog_hdfs_f1_after_bgl",
         "mean": round(amean, 4), "std": round(astd, 4), "n": len(seeds),
         "values": " ".join(f"{v:.4f}" for v in after_vals)},
        {"experiment": "H2_embedding", "metric": "deeplog_hdfs_drop",
         "mean": round(hdm, 4), "std": round(hds, 4), "n": len(seeds),
         "values": " ".join(f"{v:+.4f}" for v in hdfs_drop)},
        {"experiment": "H2_embedding", "metric": "deeplog_bgl_drop",
         "mean": round(bdm, 4), "std": round(bds, 4), "n": len(seeds),
         "values": " ".join(f"{v:+.4f}" for v in bgl_drop)},
    ]
    out = "results/mechanism_multiseed.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "metric", "mean", "std", "n", "values"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 68)
    print(f"MECHANISM MULTI-SEED SUMMARY (n={len(seeds)} seeds: {seeds})")
    print("=" * 68)
    print(f"H3 embedding — DeepLog HDFS forgetting : {fmean:+.4f} +/- {fstd:.4f}  "
          f"(HDFS F1 after BGL = {amean:.4f} +/- {astd:.4f})")
    print(f"H2 embedding — DeepLog HDFS drop       : {hdm:+.4f} +/- {hds:.4f}")
    print(f"H2 embedding — DeepLog BGL drop        : {bdm:+.4f} +/- {bds:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()
    main(args.seeds, args.rounds)
