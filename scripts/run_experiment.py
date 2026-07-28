#!/usr/bin/env python
"""
run_experiment.py — main entry point for a mixed / sequential experiment.

Reads a YAML config, builds the federation, runs the requested methods, and writes real
results to results/<experiment_name>.csv. Lightweight methods run end-to-end once the
dataloader is wired; DeepLog runs once deeplog_stub is wired to the reference repo.

This script deliberately does NOT invent results. If the dataloader isn't wired yet it
will fail loudly at data-loading — that's correct: you cannot have results without data.

Usage:
    python scripts/run_experiment.py --config configs/hdfs_bgl_mixed.yaml
"""

from __future__ import annotations
import argparse
import csv
import os
import sys

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # PyYAML

from src.dataloader import build_federation
from src.methods.length import LengthDetector
from src.methods.known_events import KnownEventsDetector
from src.aggregation import DomainAwareLength
from src.metrics import prf


def run(config_path: str, seed: int | None = None, suffix: str = "",
        split: str | None = None) -> str:
    """Run one experiment. Returns the path to the results CSV written.

    seed   : if given, overrides the config's seed (for multi-seed robustness runs).
    suffix : appended to the output filename stem so per-seed runs don't clobber each
             other, e.g. suffix='_seed3' -> results/<name>_seed3.csv.
    split  : if given, overrides the config's split ('iid' or 'lognormal') — used to run
             the non-IID robustness check without editing the config.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg["experiment_name"]
    domain_specs = [(d["name"], d["clients"]) for d in cfg["domains"]]
    split = cfg.get("split", "iid") if split is None else split
    seed = cfg.get("seed", 0) if seed is None else seed
    data_root = cfg.get("data_root", "data")

    print(f"[{name}] building federation: {domain_specs} split={split} seed={seed}")
    fed = build_federation(domain_specs, data_root=data_root, split=split, seed=seed)
    print(fed.summary())

    rows = []

    # ---- Lightweight: Length (domain-blind, the H1 target) ----
    if "length" in cfg["methods"]:
        det = LengthDetector()
        local_ranges = [det.fit_local(c.normal_train) for c in fed.clients]
        det.aggregate(local_ranges)
        for domain in fed.vocab.domains:
            dc = fed.clients_of(domain)[0]           # shared per-domain test set
            pred = det.predict(dc.test_sequences)
            m = prf(dc.test_labels, pred)
            rows.append({"method": "length_global", "domain": domain, **m.as_row()})
            print(f"  length_global   {domain}: F1={m.f1:.4f}")

    # ---- Lightweight: Known Events (set-union) ----
    if "known_events" in cfg["methods"]:
        det = KnownEventsDetector()
        local_sets = [det.fit_local(c.normal_train) for c in fed.clients]
        det.aggregate(local_sets)
        for domain in fed.vocab.domains:
            dc = fed.clients_of(domain)[0]
            pred = det.predict(dc.test_sequences)
            m = prf(dc.test_labels, pred)
            rows.append({"method": "known_events", "domain": domain, **m.as_row()})
            print(f"  known_events    {domain}: F1={m.f1:.4f}")

    # ---- OUR method: domain-aware Length (should fix H1) ----
    if cfg.get("run_domain_aware", False):
        da = DomainAwareLength()
        by_domain: dict[str, list] = {}
        base = LengthDetector()
        for c in fed.clients:
            by_domain.setdefault(c.domain, []).append(base.fit_local(c.normal_train))
        da.aggregate_per_domain(by_domain)
        for domain in fed.vocab.domains:
            dc = fed.clients_of(domain)[0]
            pred = da.predict(dc.test_sequences, [domain] * len(dc.test_sequences))
            m = prf(dc.test_labels, pred)
            rows.append({"method": "length_domain_aware", "domain": domain, **m.as_row()})
            print(f"  length_da       {domain}: F1={m.f1:.4f}")

    # ---- DeepLog (needs the reference repo wired) ----
    if "deeplog" in cfg["methods"]:
        print("  deeplog: SKIPPED — wire src/methods/deeplog_stub.py to the reference "
              "repo first (Week 1). Not writing a fake number.")

    # ---- write results ----
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"{name}{suffix}.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{name}] wrote {len(rows)} rows -> {out}")
    else:
        print(f"[{name}] no rows produced (check config methods / wiring)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed (for a single reseeded run)")
    ap.add_argument("--suffix", default="",
                    help="append to the output filename stem, e.g. _seed3")
    ap.add_argument("--split", default=None, choices=["iid", "lognormal"],
                    help="override the config split (iid or lognormal quantity-skew)")
    args = ap.parse_args()
    run(args.config, seed=args.seed, suffix=args.suffix, split=args.split)
