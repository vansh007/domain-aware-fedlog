#!/usr/bin/env python
"""
run_h1_ndomain.py — H1 (range collapse / domain-aware fix) and tag-free routing for an ARBITRARY
set of domains. Generalizes run_h1_3domain.py; default is the FOUR-domain federation
{HDFS, BGL, OpenStack, Hadoop}, the strongest generality test of Proposition 1.

Hadoop is interesting because its normal-length max (3,666) exceeds BGL's (900): adding Hadoop makes
it the new "widener", so the containment condition predicts the global range now also swallows BGL's
and Hadoop's own short anomalies, not just HDFS/OpenStack. Whatever the run shows is the real result.

Writes results/h1_<N>domain.csv. Lightweight (no training); runs in a couple of minutes.

Usage:
    python scripts/run_h1_ndomain.py                                  # 4 domains, seeds 0-4
    python scripts/run_h1_ndomain.py --domains hdfs:3 bgl:2 openstack:2 hadoop:2 --seeds 0 1 2
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
from src.methods.known_events import KnownEventsDetector
from src.aggregation import DomainAwareLength
from src.metrics import prf

DATA_ROOT = "data"


# small domains need a larger training fraction than the 1% default (too few normal sequences)
TRAIN_FRAC = {"hadoop": 0.5}


def run_seed(fed_spec, seed):
    fed = build_federation(fed_spec, data_root=DATA_ROOT, split="iid", seed=seed,
                           train_frac=TRAIN_FRAC)
    vocab = fed.vocab
    domains = vocab.domains
    id_blocks = {d: vocab.domain_range(d) for d in domains}
    base = LengthDetector()

    global_len = LengthDetector()
    global_len.aggregate([base.fit_local(c.normal_train) for c in fed.clients])

    da = DomainAwareLength()
    by_dom = {}
    for c in fed.clients:
        by_dom.setdefault(c.domain, []).append(base.fit_local(c.normal_train))
    da.aggregate_per_domain(by_dom)

    ke = KnownEventsDetector()
    ke.aggregate([KnownEventsDetector().fit_local(c.normal_train) for c in fed.clients])

    rows = []
    for domain in domains:
        dc = fed.clients_of(domain)[0]
        seqs, labs = dc.test_sequences, dc.test_labels
        tags = [domain] * len(seqs)
        f1_blind = prf(labs, global_len.predict(seqs)).f1
        f1_da = prf(labs, da.predict(seqs, tags)).f1
        f1_vb = prf(labs, da.predict_domain_unknown(seqs, "vocab_block", id_blocks=id_blocks)).f1
        inf = da.infer_domains(seqs, id_blocks)
        route_acc = sum(1 for t in inf if t == domain) / max(len(seqs), 1)
        f1_ke = prf(labs, ke.predict(seqs)).f1
        rows.append({"seed": seed, "domain": domain,
                     "length_blind_f1": round(f1_blind, 6), "length_da_f1": round(f1_da, 6),
                     "length_da_vocabblock_f1": round(f1_vb, 6), "known_events_f1": round(f1_ke, 6),
                     "route_acc": round(route_acc, 6), "n_test": len(seqs)})
        print(f"[seed {seed}] {domain:10s} blind={f1_blind:.4f} da={f1_da:.4f} vb={f1_vb:.4f} "
              f"KE={f1_ke:.4f} route={route_acc:.4f} n={len(seqs)}")
    return rows


def main(fed_spec, seeds):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(fed_spec, s))
    n = len(fed_spec)
    os.makedirs("results", exist_ok=True)
    out = f"results/h1_{n}domain.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    def ms(v):
        return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
    names = [d for d, _ in fed_spec]
    print("\n" + "=" * 80)
    print(f"H1 {n}-DOMAIN SUMMARY (n={len(seeds)} seeds) — {{{', '.join(names)}}}")
    print("=" * 80)
    for domain in sorted({r["domain"] for r in all_rows}):
        sub = [r for r in all_rows if r["domain"] == domain]
        lb, lbs = ms([r["length_blind_f1"] for r in sub])
        ld, lds = ms([r["length_da_f1"] for r in sub])
        ke, kes = ms([r["known_events_f1"] for r in sub])
        ra, _ = ms([r["route_acc"] for r in sub])
        print(f"{domain:10s} Length blind={lb:.4f}+/-{lbs:.4f}  da={ld:.4f}+/-{lds:.4f}  "
              f"KnownEvents={ke:.4f}+/-{kes:.4f}  route_acc={ra:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["hdfs:3", "bgl:2", "openstack:2", "hadoop:2"],
                    help="space-separated name:nclients")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    fed_spec = [(d.split(":")[0], int(d.split(":")[1])) for d in args.domains]
    main(fed_spec, args.seeds)
