#!/usr/bin/env python
"""
run_domain_routing.py — close the "domain-unknown routing" open problem in aggregation.py.

Domain-aware Length (the H1 fix) routes each test sequence to its domain's [min,max] range.
In run_experiment.py the domain tag is supplied by an oracle. A real deployment may not tag
sequences with their domain, so we evaluate TWO tag-free routing strategies and compare them
to the oracle:

  (a) oracle          — the domain tag we already have (upper bound / reference).
  (b) vocab_block     — Strategy 1: infer the domain from the sequence's event-id block
                        (HDFS ids [0,33), BGL ids [33,427)); majority vote; ids outside all
                        blocks -> "unknown domain" -> flagged anomalous.
  (c) conservative    — Strategy 2: no inference; flag a sequence anomalous only if its
                        length is outside EVERY domain's range.

We report per-domain F1 for all three, plus the vocab_block routing accuracy (inferred vs
true domain) and the number of "unknown-domain" sequences. Writes results/domain_routing.csv.

This is deliberately NOT an MoE / adapter / ML router. The point is that the disjoint
vocabulary structure already encodes domain identity; we read it back off the event ids,
measure it, and report it. For disjoint HDFS/BGL vocabularies we expect vocab_block to match
oracle exactly; the interesting open case is OVERLAPPING vocabularies (future work).

Usage:
    python scripts/run_domain_routing.py                 # seeds 0..4
    python scripts/run_domain_routing.py --seeds 0 1 2
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


def run_seed(seed):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    vocab = fed.vocab

    # per-domain aggregated ranges (the H1 fix), aggregated only across same-domain clients
    da = DomainAwareLength()
    base = LengthDetector()
    by_domain: dict[str, list[tuple[int, int]]] = {}
    for c in fed.clients:
        by_domain.setdefault(c.domain, []).append(base.fit_local(c.normal_train))
    da.aggregate_per_domain(by_domain)

    # id blocks come straight from the vocabulary structure — no ML, no training
    id_blocks = {d: vocab.domain_range(d) for d in vocab.domains}

    out_rows = []
    for domain in vocab.domains:
        dc = fed.clients_of(domain)[0]
        seqs, labs = dc.test_sequences, dc.test_labels
        true_tags = [domain] * len(seqs)

        # (a) oracle routing
        oracle_pred = da.predict(seqs, true_tags)
        oracle_f1 = prf(labs, oracle_pred).f1

        # (b) vocab-block routing (Strategy 1)
        vb_pred = da.predict_domain_unknown(seqs, "vocab_block", id_blocks=id_blocks)
        vb_f1 = prf(labs, vb_pred).f1
        inferred = da.infer_domains(seqs, id_blocks)
        n_unknown = sum(1 for d in inferred if d is None)
        route_acc = sum(1 for d in inferred if d == domain) / max(len(seqs), 1)

        # (c) conservative routing (Strategy 2)
        cons_pred = da.predict_domain_unknown(seqs, "conservative")
        cons_f1 = prf(labs, cons_pred).f1

        exact = (vb_pred == oracle_pred)
        out_rows.append({"seed": seed, "domain": domain,
                         "oracle_f1": round(oracle_f1, 6),
                         "vocab_block_f1": round(vb_f1, 6),
                         "conservative_f1": round(cons_f1, 6),
                         "route_acc": round(route_acc, 6),
                         "n_unknown": n_unknown,
                         "vocab_block_eq_oracle": int(exact),
                         "n_test": len(seqs)})
        print(f"[seed {seed}] {domain:5s} oracle={oracle_f1:.4f}  vocab_block={vb_f1:.4f}"
              f"  conservative={cons_f1:.4f}  route_acc={route_acc:.4f}"
              f"  unknown={n_unknown}  vb==oracle={exact}  n={len(seqs)}")
    return out_rows


def main(seeds):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(s))

    os.makedirs("results", exist_ok=True)
    out = "results/domain_routing.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "domain", "oracle_f1", "vocab_block_f1",
                                          "conservative_f1", "route_acc", "n_unknown",
                                          "vocab_block_eq_oracle", "n_test"])
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 74)
    print(f"DOMAIN-ROUTING SUMMARY (n={len(seeds)} seeds: {seeds})")
    print("=" * 74)

    def ms(vals):
        return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    all_exact = True
    for domain in sorted({r["domain"] for r in all_rows}):
        sub = [r for r in all_rows if r["domain"] == domain]
        om, os_ = ms([r["oracle_f1"] for r in sub])
        vm, vs = ms([r["vocab_block_f1"] for r in sub])
        cm, cs = ms([r["conservative_f1"] for r in sub])
        am, as_ = ms([r["route_acc"] for r in sub])
        exact = all(r["vocab_block_eq_oracle"] == 1 for r in sub)
        all_exact = all_exact and exact
        print(f"{domain:5s}  oracle F1={om:.4f}+/-{os_:.4f}   "
              f"vocab_block F1={vm:.4f}+/-{vs:.4f}   conservative F1={cm:.4f}+/-{cs:.4f}")
        print(f"       route_acc={am:.4f}+/-{as_:.4f}   vocab_block==oracle (all seeds): {exact}")

    print("-" * 74)
    if all_exact:
        print("RESULT: vocab-block routing reproduces oracle routing EXACTLY on every test")
        print("sequence and seed. For disjoint HDFS/BGL vocabularies the event-id block IS the")
        print("domain label, so no manual domain tag is needed at inference.")
    else:
        print("RESULT: vocab-block routing did NOT match oracle exactly (see per-seed rows).")
    print("NOTE: this holds because HDFS ids [0,33) and BGL ids [33,427) are DISJOINT. The")
    print("open case is OVERLAPPING vocabularies (a shared event shared across domains), where")
    print("the block no longer identifies the domain uniquely — left for future work.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    main(args.seeds)
