#!/usr/bin/env python
"""
run_ensemble.py — does domain-aware Length lift the Events+Length+ECVC ensemble (§5.5)?

The reference paper's best HDFS method is the UNION ensemble Events OR Length OR ECVC (~0.95).
H1 showed the Length *component* collapses under domain mixing. This asks the follow-up: inside
the full ensemble, does swapping domain-blind Length for domain-aware Length lift the mixed
ensemble's per-domain F1 — and toward the single-domain ~0.95?

Because the ensemble is a UNION, a collapsed Length cannot drag recall down (OR only adds
detections); the question is whether restoring Length adds back detections the ensemble was
missing, and whether it changes precision. We report member F1s and three ensembles:
  - single-domain ensemble  (reference target)
  - mixed, Length global    (domain-blind)
  - mixed, Length domain-aware (ours)

Writes results/ensemble.csv. CPU-only; ECVC uses a supervised (transductive) threshold, as in
the reference. Use --cap N for a fast smoke on a capped test set.

Usage:
    python scripts/run_ensemble.py
    python scripts/run_ensemble.py --cap 3000
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import build_federation
from src.methods.length import LengthDetector
from src.methods.known_events import KnownEventsDetector
from src.methods.ecvc import ECVCDetector
from src.aggregation import DomainAwareLength
from src.metrics import prf

DATA_ROOT = "data"


def _or(*preds):
    n = len(preds[0])
    return [1 if any(p[i] for p in preds) else 0 for i in range(n)]


def _cap(seqs, labs, cap):
    if cap is None:
        return seqs, labs
    norm = [(s, l) for s, l in zip(seqs, labs) if l == 0][:cap]
    abn = [(s, l) for s, l in zip(seqs, labs) if l == 1][:cap]
    both = norm + abn
    return [s for s, _ in both], [l for _, l in both]


def _fit_members(fed, domain_aware: bool):
    """Fit Events (union), Length (global + domain-aware), ECVC (union) over the federation."""
    events = KnownEventsDetector()
    events.aggregate([events.fit_local(c.normal_train) for c in fed.clients])

    length = LengthDetector()
    length.aggregate([length.fit_local(c.normal_train) for c in fed.clients])

    da = DomainAwareLength()
    by_dom = {}
    base = LengthDetector()
    for c in fed.clients:
        by_dom.setdefault(c.domain, []).append(base.fit_local(c.normal_train))
    da.aggregate_per_domain(by_dom)

    ecvc = ECVCDetector(vocab_size=fed.vocab.size)
    ecvc.aggregate([ecvc.fit_local(c.normal_train) for c in fed.clients])
    return events, length, da, ecvc


def _eval_domain(fed, domain, events, length, da, ecvc, cap, rows, tag):
    dc = fed.clients_of(domain)[0]
    seqs, labs = _cap(dc.test_sequences, dc.test_labels, cap)

    p_events = events.predict(seqs)
    p_len_g = length.predict(seqs)
    p_len_da = da.predict(seqs, [domain] * len(seqs))
    # ECVC: supervised threshold on THIS domain's (capped) test scores, then predict
    scores = ecvc.score(seqs)
    ecvc.set_threshold_supervised(scores, labs)
    p_ecvc = (scores >= ecvc.threshold).int().tolist()

    def rec(method, pred):
        m = prf(labs, pred)
        rows.append({"setting": tag, "domain": domain, "method": method,
                     "precision": round(m.precision, 4), "recall": round(m.recall, 4),
                     "f1": round(m.f1, 4)})
        return m.f1

    f_events = rec("events", p_events)
    f_len_g = rec("length_global", p_len_g)
    f_len_da = rec("length_domain_aware", p_len_da)
    f_ecvc = rec("ecvc", p_ecvc)
    f_ens_g = rec("ensemble_length_global", _or(p_events, p_len_g, p_ecvc))
    f_ens_da = rec("ensemble_length_domain_aware", _or(p_events, p_len_da, p_ecvc))
    print(f"  [{tag} {domain}] events={f_events:.3f} len_g={f_len_g:.3f} len_da={f_len_da:.3f} "
          f"ecvc={f_ecvc:.3f} | ENS_global={f_ens_g:.3f} ENS_da={f_ens_da:.3f}")
    return f_ens_g, f_ens_da


def main(cap):
    t0 = time.time()
    rows = []

    # ---- single-domain ensembles (reference ~0.95 target) ----
    print("[single-domain ensembles]")
    for domain, n in [("hdfs", 3), ("bgl", 2)]:
        fed = build_federation([(domain, n)], data_root=DATA_ROOT, split="iid", seed=0)
        members = _fit_members(fed, domain_aware=False)
        _eval_domain(fed, domain, *members, cap, rows, tag="single")

    # ---- mixed federation ensemble ----
    print("[mixed federation ensemble]")
    fed = build_federation([("hdfs", 3), ("bgl", 2)], data_root=DATA_ROOT, split="iid", seed=0)
    members = _fit_members(fed, domain_aware=True)
    for domain in fed.vocab.domains:
        _eval_domain(fed, domain, *members, cap, rows, tag="mixed")

    os.makedirs("results", exist_ok=True)
    out = "results/ensemble_smoke.csv" if cap else "results/ensemble.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "domain", "method", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(rows)

    # headline
    def get(tag, dom, method):
        return next((r["f1"] for r in rows if r["setting"] == tag and r["domain"] == dom
                     and r["method"] == method), None)
    print("\n" + "=" * 64)
    print("ENSEMBLE SUMMARY — HDFS ensemble F1")
    print("=" * 64)
    print(f"  single-domain           : {get('single','hdfs','ensemble_length_global')}")
    print(f"  mixed, Length global     : {get('mixed','hdfs','ensemble_length_global')}")
    print(f"  mixed, Length domain-aware: {get('mixed','hdfs','ensemble_length_domain_aware')}")
    print(f"\nwrote {out}   (elapsed {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=None, help="cap test set per class (fast smoke)")
    args = ap.parse_args()
    main(args.cap)
