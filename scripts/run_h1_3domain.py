#!/usr/bin/env python
"""
run_h1_3domain.py — generalize H1 (and the containment condition + tag-free routing) from the
HDFS+BGL pair to a THREE-domain federation {HDFS, BGL, OpenStack}.

This directly answers the main reviewer objection ("only two domains"). We add OpenStack, parsed
from the LogHub/DeepLog raw logs (scripts/parse_openstack.py), as a third disjoint vocabulary block
[427,443). OpenStack is a NARROW, tight-length domain (16 events, normal length 24.9+/-0.8) whose
few labelled anomalies are event-based, not length-based --- a different profile from HDFS and BGL,
which is exactly what a generality test wants.

For each seed we run, in the mixed 3-domain federation, per domain:
  - Length, domain-blind (one global min-max over ALL clients of ALL domains)
  - Length, domain-aware (per-domain min-max) with oracle routing AND vocab-block routing
  - Known Events (global set union)
and report precision/recall/F1, plus the 3-domain vocab-block routing accuracy.

Predictions from the paper's containment condition: HDFS Length collapses (BGL widens the global
range) and domain-aware recovers it; BGL is the widener and is unchanged; OpenStack's anomalies are
not length-outliers so Length is weak for it either way, while Known Events (event-based) detects
them --- so the robust method stays robust across all three domains, and vocab-block routing is exact
on three disjoint blocks. Writes results/h1_3domain.csv.

Usage: python scripts/run_h1_3domain.py [--seeds 0 1 2 3 4]
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

FED = [("hdfs", 3), ("bgl", 2), ("openstack", 2)]
DATA_ROOT = "data"


def run_seed(seed):
    fed = build_federation(FED, data_root=DATA_ROOT, split="iid", seed=seed)
    vocab = fed.vocab
    domains = vocab.domains
    id_blocks = {d: vocab.domain_range(d) for d in domains}

    base = LengthDetector()

    # --- domain-blind global Length: one range over ALL clients ---
    global_len = LengthDetector()
    global_len.aggregate([base.fit_local(c.normal_train) for c in fed.clients])

    # --- domain-aware Length: per-domain ranges ---
    da = DomainAwareLength()
    by_dom = {}
    for c in fed.clients:
        by_dom.setdefault(c.domain, []).append(base.fit_local(c.normal_train))
    da.aggregate_per_domain(by_dom)

    # --- Known Events: global union over all clients ---
    ke = KnownEventsDetector()
    ke.aggregate([KnownEventsDetector().fit_local(c.normal_train) for c in fed.clients])

    rows = []
    for domain in domains:
        dc = fed.clients_of(domain)[0]
        seqs, labs = dc.test_sequences, dc.test_labels
        tags = [domain] * len(seqs)

        f1_blind = prf(labs, global_len.predict(seqs)).f1
        f1_da = prf(labs, da.predict(seqs, tags)).f1
        inf_tags = da.infer_domains(seqs, id_blocks)
        f1_vb = prf(labs, da.predict_domain_unknown(seqs, "vocab_block", id_blocks=id_blocks)).f1
        route_acc = sum(1 for t in inf_tags if t == domain) / max(len(seqs), 1)
        f1_ke = prf(labs, ke.predict(seqs)).f1

        rows.append({"seed": seed, "domain": domain,
                     "length_blind_f1": round(f1_blind, 6),
                     "length_da_f1": round(f1_da, 6),
                     "length_da_vocabblock_f1": round(f1_vb, 6),
                     "known_events_f1": round(f1_ke, 6),
                     "route_acc": round(route_acc, 6), "n_test": len(seqs)})
        print(f"[seed {seed}] {domain:10s} Length blind={f1_blind:.4f} da={f1_da:.4f} "
              f"vb={f1_vb:.4f} | KnownEvents={f1_ke:.4f} | route_acc={route_acc:.4f} n={len(seqs)}")
    return rows


def main(seeds):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(s))
    os.makedirs("results", exist_ok=True)
    out = "results/h1_3domain.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    def ms(v):
        return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)

    print("\n" + "=" * 78)
    print(f"H1 THREE-DOMAIN SUMMARY (n={len(seeds)} seeds) — {{HDFS, BGL, OpenStack}}")
    print("=" * 78)
    for domain in sorted({r["domain"] for r in all_rows}):
        sub = [r for r in all_rows if r["domain"] == domain]
        lb, lbs = ms([r["length_blind_f1"] for r in sub])
        ld, lds = ms([r["length_da_f1"] for r in sub])
        vb, vbs = ms([r["length_da_vocabblock_f1"] for r in sub])
        ke, kes = ms([r["known_events_f1"] for r in sub])
        ra, ras = ms([r["route_acc"] for r in sub])
        print(f"{domain:10s} Length blind={lb:.4f}+/-{lbs:.4f}  da={ld:.4f}+/-{lds:.4f}  "
              f"vb={vb:.4f}  | KnownEvents={ke:.4f}+/-{kes:.4f}  route_acc={ra:.4f}+/-{ras:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    main(args.seeds)
