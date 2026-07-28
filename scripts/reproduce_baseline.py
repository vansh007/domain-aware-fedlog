#!/usr/bin/env python
"""
reproduce_baseline.py — the Week-1 gate.

Purpose: confirm we can reproduce the reference paper's single-domain numbers BEFORE
writing any mixed-domain conclusions. If this doesn't roughly match, stop and report —
that is itself a finding, not a bug to hide (see CLAUDE.md rule 2).

This wraps run_experiment.py on the single-domain configs and prints the reference
targets next to what we got, so the comparison is explicit.

Usage:
    python scripts/reproduce_baseline.py --dataset hdfs
    python scripts/reproduce_baseline.py --dataset bgl
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reference targets (see docs/REFERENCE_PAPER_NOTES.md). Approximate — their Table II.
REFERENCE_TARGETS = {
    "hdfs": {
        "note": "ECVC & Events+Length+ECVC ~0.95; DeepLog ~0.91-0.93; Length alone ~0.57",
        "length_expected_low": True,   # Length is known-weak on HDFS alone
    },
    "bgl": {
        "note": "Events & Events+Length+Edit ~0.99; Length & n-grams weak",
        "length_expected_low": True,
    },
}


def main(dataset: str) -> None:
    cfg = f"configs/{dataset}_iid.yaml"
    if not os.path.exists(cfg):
        sys.exit(f"missing config {cfg}")

    print("=" * 70)
    print(f"WEEK-1 GATE — reproducing single-domain baseline for {dataset.upper()}")
    print("Reference target:", REFERENCE_TARGETS[dataset]["note"])
    print("=" * 70)

    from scripts.run_experiment import run
    run(cfg)

    print("\nNEXT: compare results/%s_baseline.csv against the reference targets above." % dataset)
    print("If the picture matches (right methods strong/weak on the right dataset),")
    print("the gate is PASSED — proceed to configs/hdfs_bgl_mixed.yaml.")
    print("If it does NOT match, STOP and report to your supervisor before continuing.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hdfs", "bgl"], required=True)
    args = ap.parse_args()
    main(args.dataset)
