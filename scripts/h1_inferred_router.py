#!/usr/bin/env python
"""
h1_inferred_router.py — remove the oracle-domain-tag assumption from the H1 fix.

The H1 domain-aware Length fix routes each test sequence to its domain's [min,max] range.
In `run_experiment.py` the domain tag is supplied by an oracle (`[domain]*len(...)`). A real
deployment may not tag sequences with their domain. This script closes that open problem by
INFERRING the domain from the sequence's event ids: HDFS owns global id block [0,33) and BGL
owns [33,427), so a sequence's domain is the majority block of its event ids. We then route
by the INFERRED tag and compare F1 to the oracle-tag routing, and we report routing accuracy.

If inference is (near-)perfect, inferred-router F1 == oracle F1, so the fix needs no tag and
is deployable. This is measured, not assumed.

Writes results/h1_inferred_router.csv:
  columns: seed, domain, oracle_f1, inferred_f1, route_acc, n_test

Usage:
    python scripts/h1_inferred_router.py                 # seeds 0..4
    python scripts/h1_inferred_router.py --seeds 0 1 2
"""

from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import build_federation
from src.methods.length import LengthDetector
from src.aggregation import DomainAwareLength
from src.metrics import prf

HDFS_CLIENTS = 3
BGL_CLIENTS = 2
DATA_ROOT = "data"


def infer_domains(sequences, vocab):
    """Infer each sequence's domain by majority vote over vocab.domain_of(event_id).

    Empty sequences (no events) fall back to the first registered domain; they carry no
    length signal anyway. Returns a list of inferred domain names, one per sequence.
    """
    domains = vocab.domains
    fallback = domains[0]
    inferred = []
    for s in sequences:
        if not s:
            inferred.append(fallback)
            continue
        counts = {}
        for gid in s:
            d = vocab.domain_of(gid)
            counts[d] = counts.get(d, 0) + 1
        inferred.append(max(counts, key=counts.get))
    return inferred


def run_seed(seed):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    vocab = fed.vocab

    # per-domain aggregated ranges (the H1 fix)
    da = DomainAwareLength()
    by_domain = {}
    base = LengthDetector()
    for c in fed.clients:
        by_domain.setdefault(c.domain, []).append(base.fit_local(c.normal_train))
    da.aggregate_per_domain(by_domain)

    out_rows = []
    for domain in vocab.domains:
        dc = fed.clients_of(domain)[0]
        seqs, labs = dc.test_sequences, dc.test_labels
        true_tags = [domain] * len(seqs)

        # oracle routing (true tag)
        oracle_pred = da.predict(seqs, true_tags)
        oracle_f1 = prf(labs, oracle_pred).f1

        # inferred routing (tag from vocabulary block)
        inf_tags = infer_domains(seqs, vocab)
        inferred_pred = da.predict(seqs, inf_tags)
        inferred_f1 = prf(labs, inferred_pred).f1
        route_acc = sum(1 for a, b in zip(inf_tags, true_tags) if a == b) / max(len(seqs), 1)

        out_rows.append({"seed": seed, "domain": domain,
                         "oracle_f1": round(oracle_f1, 6),
                         "inferred_f1": round(inferred_f1, 6),
                         "route_acc": round(route_acc, 6),
                         "n_test": len(seqs)})
        print(f"[seed {seed}] {domain:5s} oracle_f1={oracle_f1:.4f}  "
              f"inferred_f1={inferred_f1:.4f}  route_acc={route_acc:.4f}  n={len(seqs)}")
    return out_rows


def main(seeds):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(s))

    os.makedirs("results", exist_ok=True)
    out = "results/h1_inferred_router.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "domain", "oracle_f1", "inferred_f1",
                                          "route_acc", "n_test"])
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 68)
    print(f"H1 INFERRED-ROUTER SUMMARY (n={len(seeds)} seeds: {seeds})")
    print("=" * 68)
    for domain in sorted({r["domain"] for r in all_rows}):
        orc = [r["oracle_f1"] for r in all_rows if r["domain"] == domain]
        inf = [r["inferred_f1"] for r in all_rows if r["domain"] == domain]
        acc = [r["route_acc"] for r in all_rows if r["domain"] == domain]

        def ms(v):
            return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
        om, os_ = ms(orc); im, is_ = ms(inf); am, as_ = ms(acc)
        print(f"{domain:5s}  oracle F1={om:.4f}+/-{os_:.4f}   "
              f"inferred F1={im:.4f}+/-{is_:.4f}   route_acc={am:.4f}+/-{as_:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    main(args.seeds)
