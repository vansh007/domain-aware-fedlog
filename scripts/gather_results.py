#!/usr/bin/env python
"""
gather_results.py — collate every results/*.csv into one summary table.

Read-only over results/. Prints a combined table and writes results/_summary.csv.
Never invents rows.
"""
from __future__ import annotations
import csv
import glob
import os
import sys


def main() -> None:
    files = sorted(glob.glob("results/*.csv"))
    files = [f for f in files if not f.endswith("_summary.csv")]
    if not files:
        sys.exit("no results/*.csv yet — run experiments first")

    all_rows = []
    for path in files:
        exp = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for r in csv.DictReader(f):
                r = {"experiment": exp, **r}
                all_rows.append(r)

    if not all_rows:
        sys.exit("result files are empty")

    fields = list(all_rows[0].keys())
    out = "results/_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # pretty print
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in all_rows)) for k in fields}
    print(" | ".join(k.ljust(widths[k]) for k in fields))
    print("-+-".join("-" * widths[k] for k in fields))
    for r in all_rows:
        print(" | ".join(str(r.get(k, "")).ljust(widths[k]) for k in fields))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
