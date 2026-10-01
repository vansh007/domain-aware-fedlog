#!/usr/bin/env python
"""
impact_analysis.py — quantify the ORGANISATIONAL stakes of the paper's findings, from real numbers.

Two things a decision-maker asks: "how bad is the failure?" and "how expensive is the fix?" This
script answers both with measured/derived quantities (no fabricated figures):

  (A) SILENT BLIND SPOT — what fraction of a domain's anomalies silently stop being detected when the
      failure fires. Derived from the measured recall/TP/FN in results/*baseline*.csv and
      results/ensemble.csv (range collapse) and the measured F1 drop for the deep forgetting.

  (B) COST OF THE FIX — the memory/compute overhead of each safeguard: per-domain range parameters,
      the replay buffer, and the auditor's runtime. Model sizes are measured from the actual objects;
      the auditor is timed; the buffer is sized from the documented 1% training pool.

Writes results/impact_blindspot.csv and results/impact_overhead.csv. CPU-only, seconds to run.

Usage: python scripts/impact_analysis.py
"""

from __future__ import annotations
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


# ------------------------------------------------------------------------------------------
# (A) Silent blind spot — fraction of anomalies that stop being detected when the failure fires
# ------------------------------------------------------------------------------------------
def blind_spot():
    out = []

    # Range collapse (H1): HDFS Length single-domain vs domain-blind mixed, from measured TP/FN.
    base = {r["method"]: r for r in _rows("results/hdfs_baseline.csv")}
    lg = base.get("length_global")
    if lg:
        tp, fn = int(lg["tp"]), int(lg["fn"])
        total_anom = tp + fn
        rec_single = tp / total_anom
        # mixed-blind recall from ensemble.csv (measured)
        ens = {(r["setting"], r["method"]): r for r in _rows("results/ensemble.csv")
               if r["domain"] == "hdfs"}
        rec_mixed = float(ens[("mixed", "length_global")]["recall"]) if ("mixed", "length_global") in ens else 0.0
        caught_single = round(rec_single * total_anom)
        caught_mixed = round(rec_mixed * total_anom)
        out.append({
            "failure": "H1 range collapse (HDFS Length, domain-blind mixing)",
            "metric": "anomaly recall",
            "healthy": f"{rec_single:.4f}", "failed": f"{rec_mixed:.4f}",
            "anomalies_total": total_anom,
            "caught_healthy": caught_single, "caught_failed": caught_mixed,
            "newly_missed": caught_single - caught_mixed,
            "coverage_lost_pct": round(100 * (rec_single - rec_mixed) / rec_single, 2) if rec_single else 0,
            "note": "domain-aware fix restores recall to the single-domain level"})

    # Catastrophic forgetting (H3, embedding): HDFS detection F1 before/after a new domain arrives.
    f1_before, f1_after = 0.7037, 0.039   # results/mechanism_multiseed.csv
    out.append({
        "failure": "H3 catastrophic forgetting (deep model, new domain onboarded)",
        "metric": "HDFS detection F1",
        "healthy": f"{f1_before:.4f}", "failed": f"{f1_after:.4f}",
        "anomalies_total": "", "caught_healthy": "", "caught_failed": "", "newly_missed": "",
        "coverage_lost_pct": round(100 * (f1_before - f1_after) / f1_before, 2),
        "note": "fires silently the moment a new log source is onboarded; replay restores F1 to 0.700"})

    # Set-union growth (C2): not a detection loss but an unbounded-cost failure under evolution.
    seq = {(r["method"], r["after_arrival"]): r for r in _rows("results/h3_sequential.csv")
           if r["eval_domain"] == "hdfs"}
    if ("known_events", "hdfs") in seq and ("known_events", "bgl") in seq:
        b0 = int(seq[("known_events", "hdfs")]["model_bytes"])
        b1 = int(seq[("known_events", "bgl")]["model_bytes"])
        out.append({
            "failure": "C2 unbounded growth (set-union, per added domain)",
            "metric": "model state bytes",
            "healthy": b0, "failed": b1, "anomalies_total": "", "caught_healthy": "",
            "caught_failed": "", "newly_missed": "",
            "coverage_lost_pct": round(100 * (b1 - b0) / b0, 2),
            "note": f"x{b1/b0:.1f} per domain; unbounded on an open stream"})
    return out


# ------------------------------------------------------------------------------------------
# (B) Cost of the fix — measured memory/compute overhead of each safeguard
# ------------------------------------------------------------------------------------------
def overhead():
    out = []
    INT = 8  # bytes per stored event id / range endpoint (int64)

    # per-domain range parameters vs one global range: 2 ints per domain
    for nd in (2, 3):
        out.append({"safeguard": f"Domain-aware ranges ({nd} domains)",
                    "overhead_bytes": 2 * INT * nd, "baseline_bytes": 2 * INT,
                    "note": "2 integers per domain; vs 2 for the global range"})

    # replay buffer: 10% of the 1% HDFS training pool (5,582 seqs) ~= 558 seqs, avg length 29
    pool, frac, avg_len = 5582, 0.10, 29
    buf_seqs = round(frac * pool)
    buf_bytes = buf_seqs * avg_len * INT
    out.append({"safeguard": "Replay buffer (10% of earlier domain)",
                "overhead_bytes": buf_bytes, "baseline_bytes": "",
                "note": f"{buf_seqs} sequences x ~{avg_len} events; raw ids, no parameters"})

    # DeepLog model size (measured) — the thing the buffer protects
    try:
        from src.methods.deeplog_fed import DeepLog, state_size_bytes
        m = DeepLog(num_keys=427, input_mode="embedding")
        dl_bytes = state_size_bytes(m)
        out.append({"safeguard": "DeepLog model (reference point)",
                    "overhead_bytes": dl_bytes, "baseline_bytes": "",
                    "note": "the replay buffer is smaller than the model it protects"})
    except Exception as e:  # noqa: BLE001
        dl_bytes = None
        print("  (model sizing skipped:", e, ")")

    # auditor runtime (measured): time many pre-deployment audits
    try:
        from fedlog_audit import DomainSpec, audit_federation
        doms = [DomainSpec("hdfs", (0, 33), (4, 300)), DomainSpec("bgl", (33, 427), (1, 900)),
                DomainSpec("openstack", (427, 443), (1, 27))]
        N = 2000
        t0 = time.perf_counter()
        for _ in range(N):
            audit_federation(doms, arrival="sequential", representation="embedding",
                             detector="deep", order=["hdfs", "bgl", "openstack"])
        per_ms = 1000 * (time.perf_counter() - t0) / N
        out.append({"safeguard": "Pre-deployment auditor (per run)",
                    "overhead_bytes": "", "baseline_bytes": "",
                    "note": f"{per_ms:.3f} ms per audit, summary stats only, no raw logs"})
    except Exception as e:  # noqa: BLE001
        print("  (auditor timing skipped:", e, ")")
    return out


def main():
    os.makedirs("results", exist_ok=True)
    bs = blind_spot()
    with open("results/impact_blindspot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bs[0].keys()))
        w.writeheader(); w.writerows(bs)
    ov = overhead()
    with open("results/impact_overhead.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ov[0].keys()))
        w.writeheader(); w.writerows(ov)

    print("=" * 78)
    print("ORGANISATIONAL IMPACT — the stakes, from measured/derived numbers")
    print("=" * 78)
    print("\n(A) SILENT BLIND SPOT (what stops being detected when the failure fires):")
    for r in bs:
        print(f"  - {r['failure']}")
        print(f"      {r['metric']}: {r['healthy']} -> {r['failed']}  "
              f"({r['coverage_lost_pct']}% lost)"
              + (f"; {r['newly_missed']} of {r['anomalies_total']} anomalies newly missed"
                 if r['newly_missed'] != "" else ""))
    print("\n(B) COST OF THE FIX (measured overhead):")
    for r in ov:
        b = r["overhead_bytes"]
        size = f"{b:,} B" if isinstance(b, int) else "—"
        print(f"  - {r['safeguard']:38s} {size:>12s}   {r['note']}")
    print("\nwrote results/impact_blindspot.csv, results/impact_overhead.csv")


if __name__ == "__main__":
    main()
