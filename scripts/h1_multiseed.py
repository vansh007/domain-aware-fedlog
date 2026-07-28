#!/usr/bin/env python
"""
h1_multiseed.py — robustness run for H1 across multiple seeds.

Runs configs/hdfs_bgl_mixed.yaml at seeds 0..N-1 through the SAME code path as a normal
single run (scripts.run_experiment.run), then aggregates the length_global vs
length_domain_aware F1 (per domain) into mean +/- std.

This is the "measured an effect" version of the single-seed H1 demonstration: the
reference authors averaged three seeds; we match that discipline with five.

Writes:
  - results/hdfs_bgl_mixed_seed{k}.csv  (one per seed, full method table)
  - results/h1_multiseed.csv            (tidy: method,domain,metric,mean,std,n,values)

Usage:
    python scripts/h1_multiseed.py                 # seeds 0..4
    python scripts/h1_multiseed.py --seeds 0 1 2   # explicit seeds
"""

from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_experiment import run

CONFIG = "configs/hdfs_bgl_mixed.yaml"

# The rows we care about for H1: (method, domain).
TRACKED = [
    ("length_global", "hdfs"),
    ("length_domain_aware", "hdfs"),
    ("length_global", "bgl"),
    ("length_domain_aware", "bgl"),
    ("known_events", "hdfs"),
    ("known_events", "bgl"),
]


def _read_f1(csv_path: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["method"], row["domain"])] = float(row["f1"])
    return out


def main(seeds: list[int], split: str | None = None, tag: str = "") -> None:
    # tag distinguishes runs (e.g. '_lognormal') so per-seed CSVs and the summary
    # don't clobber the IID run.
    per_seed: list[dict[tuple[str, str], float]] = []
    for s in seeds:
        print(f"\n===== seed {s}{(' split=' + split) if split else ''} =====")
        csv_path = run(CONFIG, seed=s, suffix=f"{tag}_seed{s}", split=split)
        per_seed.append(_read_f1(csv_path))

    # aggregate
    print("\n" + "=" * 68)
    print(f"H1 MULTI-SEED SUMMARY  (n={len(seeds)} seeds: {seeds})")
    print("=" * 68)
    print(f"{'method':22s} {'domain':5s} {'mean F1':>9s} {'std':>8s}   values")
    summary_rows = []
    for method, domain in TRACKED:
        vals = [ps[(method, domain)] for ps in per_seed if (method, domain) in ps]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        vals_str = " ".join(f"{v:.4f}" for v in vals)
        print(f"{method:22s} {domain:5s} {mean:9.4f} {std:8.4f}   {vals_str}")
        summary_rows.append({
            "method": method, "domain": domain, "metric": "f1",
            "mean": round(mean, 6), "std": round(std, 6),
            "n": len(vals), "values": vals_str,
        })

    out = os.path.join("results", f"h1_multiseed{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "domain", "metric", "mean", "std", "n", "values"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwrote summary -> {out}")

    # the headline contrast
    lg = next((r for r in summary_rows if r["method"] == "length_global" and r["domain"] == "hdfs"), None)
    da = next((r for r in summary_rows if r["method"] == "length_domain_aware" and r["domain"] == "hdfs"), None)
    if lg and da:
        print("\nHEADLINE (HDFS Length): "
              f"global {lg['mean']:.4f}+/-{lg['std']:.4f}  ->  "
              f"domain-aware {da['mean']:.4f}+/-{da['std']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default=None, choices=["iid", "lognormal"],
                    help="override the config split; also sets the output tag")
    ap.add_argument("--tag", default=None,
                    help="output filename tag (default: derived from --split)")
    args = ap.parse_args()
    tag = args.tag if args.tag is not None else (f"_{args.split}" if args.split else "")
    main(args.seeds, split=args.split, tag=tag)
