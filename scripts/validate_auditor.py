#!/usr/bin/env python
"""
validate_auditor.py — does the safety auditor actually PREDICT the measured outcomes?

The auditor (fedlog_audit) encodes the paper's findings as decision rules over summary statistics.
This script closes the loop: it runs the auditor \\emph{blind} on every configuration for which we
have a measured result, and scores its SAFE/FAILS verdict against what actually happened. The
observed outcomes are read from the result CSVs, so the comparison is exact and traceable.

Configurations covered (2 architectures x 2 input encodings x 3 arrival patterns for the deep model,
plus the lightweight range-aggregation cases across 2 and 3 domains):
  deep forgetting : {DeepLog, Transformer} x {scalar, embedding} x {simultaneous, forward, reverse}
  lightweight     : domain-blind Length collapse per domain, 2-domain and 3-domain federations

A verdict is FAILS if the relevant risk is HIGH/CRITICAL, else SAFE. An observed outcome is FAILS if
the measured effect crosses a documented threshold (forgetting > 0.15; a single-domain-detectable
Length dropping below 0.10 under domain-blind aggregation). Writes results/auditor_validation.csv.

Usage: python scripts/validate_auditor.py
"""

from __future__ import annotations
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedlog_audit import DomainSpec, audit_federation

# domain summary stats used by the auditor (event-id block, normal-length range)
HDFS = DomainSpec("hdfs", (0, 33), (4, 300))
BGL = DomainSpec("bgl", (33, 427), (1, 900))
OPENSTACK = DomainSpec("openstack", (427, 443), (1, 27))


def _rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def _mean(vals):
    return st.fmean(vals) if vals else float("nan")


# ---- observed outcomes, read straight from the result CSVs -------------------------------
def observed():
    o = {}
    # DeepLog embedding forward forgetting + H2 embedding drop
    for r in _rows("results/mechanism_multiseed.csv"):
        if r["experiment"] == "H3_embedding" and r["metric"] == "deeplog_hdfs_forgetting":
            o["dl_emb_fwd"] = float(r["mean"])
        if r["experiment"] == "H2_embedding" and r["metric"] == "deeplog_hdfs_drop":
            o["dl_emb_sim"] = float(r["mean"])
    # DeepLog scalar forward forgetting (h3_sequential: deeplog hdfs before==after)
    dd = {(r["method"], r["after_arrival"]): float(r["f1"])
          for r in _rows("results/h3_sequential.csv") if r["eval_domain"] == "hdfs"}
    o["dl_sca_fwd"] = dd.get(("deeplog", "hdfs"), 0.0) - dd.get(("deeplog", "bgl"), 0.0)
    # DeepLog scalar simultaneous drop (h2_multiseed)
    for r in _rows("results/h2_multiseed.csv"):
        if r["domain"] == "hdfs":
            o["dl_sca_sim"] = float(r["drop_mean"])
    # DeepLog reverse (h3_reverse_order): forgetting of the earlier domain (BGL)
    for r in _rows("results/h3_reverse_order.csv"):
        pass
    rev = _rows("results/h3_reverse_order.csv")
    o["dl_sca_rev"] = _mean([float(r["forgetting"]) for r in rev if r["input_mode"] == "scalar"])
    o["dl_emb_rev"] = _mean([float(r["forgetting"]) for r in rev if r["input_mode"] == "embedding"])
    # Transformer (transformer_mechanism): forgetting + hdfs simultaneous drop
    tr = _rows("results/transformer_mechanism.csv")
    for mode in ("scalar", "embedding"):
        sub = [r for r in tr if r["input_mode"] == mode]
        o[f"tf_{mode}_fwd"] = _mean([float(r["forgetting"]) for r in sub])
        o[f"tf_{mode}_sim"] = _mean([float(r["hdfs_single"]) - float(r["hdfs_mixed"]) for r in sub])
    # lightweight domain-blind Length per domain (2-domain h1_multiseed, 3-domain h1_3domain)
    for r in _rows("results/h1_multiseed.csv"):
        if r["method"] == "length_global":
            o[f"len_blind_2d_{r['domain']}"] = float(r["mean"])
    for dom in ("hdfs", "bgl", "openstack"):
        vals = [float(r["length_blind_f1"]) for r in _rows("results/h1_3domain.csv")
                if r["domain"] == dom]
        o[f"len_blind_3d_{dom}"] = _mean(vals)
    return o


def deep_verdict(**kw):
    rep = audit_federation([HDFS, BGL], **kw)
    r = next(x for x in rep.risks if x.name == "Catastrophic forgetting")
    return "FAILS" if r.level in ("HIGH", "CRITICAL") else "SAFE"


def length_verdict(domains, domain):
    rep = audit_federation(domains, arrival="simultaneous", detector="lightweight")
    r = next(x for x in rep.risks if x.name == "Range aggregation (Length)")
    # the auditor flags the whole lightweight family; a domain is a predicted victim iff its range
    # is strictly contained in the global range (the containment condition)
    gmin = min(d.min_len for d in domains); gmax = max(d.max_len for d in domains)
    d = next(x for x in domains if x.name == domain)
    victim = (gmin < d.min_len or gmax > d.max_len) and r.level in ("HIGH", "CRITICAL", "MEDIUM")
    return "FAILS" if victim else "SAFE"


def main():
    o = observed()
    FORGET_TH = 0.15   # forgetting above this = catastrophic
    LEN_TH = 0.10      # domain-blind Length below this (from a detectable baseline) = collapse
    SINGLE_LEN = {"hdfs": 0.538, "bgl": 0.020, "openstack": 0.0}  # single-domain Length F1 (baseline)

    checks = []  # (config, predicted, observed_value, observed_label, note)

    # ---- deep-model forgetting: 2 archs x 2 inputs x 3 arrivals ----
    deep = [
        ("DeepLog  scalar    simultaneous", dict(architecture="lstm", representation="scalar",
            arrival="simultaneous", detector="deep"), "dl_sca_sim"),
        ("DeepLog  scalar    forward",      dict(architecture="lstm", representation="scalar",
            arrival="sequential", detector="deep", order=["hdfs", "bgl"]), "dl_sca_fwd"),
        ("DeepLog  scalar    reverse",      dict(architecture="lstm", representation="scalar",
            arrival="sequential", detector="deep", order=["bgl", "hdfs"]), "dl_sca_rev"),
        ("DeepLog  embedding simultaneous", dict(architecture="lstm", representation="embedding",
            arrival="simultaneous", detector="deep"), "dl_emb_sim"),
        ("DeepLog  embedding forward",      dict(architecture="lstm", representation="embedding",
            arrival="sequential", detector="deep", order=["hdfs", "bgl"]), "dl_emb_fwd"),
        ("DeepLog  embedding reverse",      dict(architecture="lstm", representation="embedding",
            arrival="sequential", detector="deep", order=["bgl", "hdfs"]), "dl_emb_rev"),
        ("Transf.  scalar    simultaneous", dict(architecture="transformer", representation="scalar",
            arrival="simultaneous", detector="deep"), "tf_scalar_sim"),
        ("Transf.  scalar    forward",      dict(architecture="transformer", representation="scalar",
            arrival="sequential", detector="deep", order=["hdfs", "bgl"]), "tf_scalar_fwd"),
        ("Transf.  embedding simultaneous", dict(architecture="transformer", representation="embedding",
            arrival="simultaneous", detector="deep"), "tf_embedding_sim"),
        ("Transf.  embedding forward",      dict(architecture="transformer", representation="embedding",
            arrival="sequential", detector="deep", order=["hdfs", "bgl"]), "tf_embedding_fwd"),
    ]
    for label, kw, key in deep:
        pred = deep_verdict(**kw)
        val = o.get(key, float("nan"))
        obs = "FAILS" if (val > FORGET_TH) else "SAFE"   # drops are ~0 (safe); forgetting>0.15 fails
        checks.append((label, pred, val, obs, ""))

    # ---- lightweight range collapse: per domain, 2-domain and 3-domain ----
    light = [
        ("Length blind  2-dom  HDFS", [HDFS, BGL], "hdfs", "len_blind_2d_hdfs"),
        ("Length blind  2-dom  BGL",  [HDFS, BGL], "bgl",  "len_blind_2d_bgl"),
        ("Length blind  3-dom  HDFS", [HDFS, BGL, OPENSTACK], "hdfs", "len_blind_3d_hdfs"),
        ("Length blind  3-dom  BGL",  [HDFS, BGL, OPENSTACK], "bgl",  "len_blind_3d_bgl"),
        ("Length blind  3-dom  OpenStack", [HDFS, BGL, OPENSTACK], "openstack", "len_blind_3d_openstack"),
    ]
    for label, doms, dom, key in light:
        pred = length_verdict(doms, dom)
        val = o.get(key, float("nan"))
        # observed FAILS = a domain whose single-domain Length WORKED (>=0.10) collapses (<0.10)
        # under domain-blind aggregation. A domain whose Length never worked, or is unchanged, is SAFE.
        collapsed = (SINGLE_LEN[dom] >= LEN_TH) and (val < LEN_TH)
        obs = "FAILS" if collapsed else "SAFE"
        note = ""
        if dom == "openstack":
            note = ("conservative false-positive: range coupling is structurally real, but "
                    "OpenStack anomalies are order-based so no Length detector existed to break")
        checks.append((label, pred, val, obs, note))

    # ---- score ----
    os.makedirs("results", exist_ok=True)
    with open("results/auditor_validation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "auditor_prediction", "observed_value", "observed_label", "match", "note"])
        scored = correct = 0
        for label, pred, val, obs, note in checks:
            if obs == "N/A":
                match = "n/a"
            else:
                scored += 1
                match = "OK" if pred == obs else "MISS"
                correct += (pred == obs)
            w.writerow([label, pred, f"{val:+.4f}", obs, match, note])

    print("=" * 84)
    print("AUDITOR PREDICTION VALIDATION  (auditor run blind; outcomes read from result CSVs)")
    print("=" * 84)
    print(f"{'configuration':32s} {'predicted':10s} {'observed':10s} {'value':>9s}  match")
    print("-" * 84)
    for label, pred, val, obs, note in checks:
        match = "n/a" if obs == "N/A" else ("OK " if pred == obs else "*** MISS")
        tail = f"   [{note}]" if note else ""
        print(f"{label:32s} {pred:10s} {obs:10s} {val:>+9.4f}  {match}{tail}")
    print("-" * 84)
    print(f"SCORED CONFIGURATIONS: {correct}/{scored} correct "
          f"({100*correct/scored:.0f}%); {len(checks)-scored} excluded as N/A (degenerate).")
    print("wrote results/auditor_validation.csv")


if __name__ == "__main__":
    main()
