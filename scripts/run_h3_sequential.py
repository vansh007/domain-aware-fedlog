#!/usr/bin/env python
"""
run_h3_sequential.py — the H3 experiment: sequential domain arrival (HDFS then BGL).

H3: neither family is deployable as-is under sequential arrival.
  - Set-union (Known Events): NEVER forgets (union is monotone) but GROWS unbounded — the
    known-event set expands with every new domain's vocabulary.
  - DeepLog: stays FIXED-size but FORGETS the earlier domain when adapted to the new one.

We measure, after each arrival:
  (a) forgetting  = F1 on HDFS *before* BGL arrives minus F1 on HDFS *after* BGL arrives.
  (b) model size in bytes (the growth curve).

Merged vocabulary (427) is fixed from the start so DeepLog has a stable output space; the
research point is temporal (domains arrive over time), not that the vocabulary is unknown.

Writes results/h3_sequential.csv and prints a forgetting + size summary. CPU-first.

Usage:
    python scripts/run_h3_sequential.py                 # full (deeplog rounds=12)
    python scripts/run_h3_sequential.py --rounds 6
    python scripts/run_h3_sequential.py --no-deeplog    # only the set-union curve (fast)
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import build_federation
from src.methods.known_events import KnownEventsDetector
from src.metrics import prf, forgetting

ARRIVALS = ["hdfs", "bgl"]
HDFS_CLIENTS = 3
BGL_CLIENTS = 2
DATA_ROOT = "data"


def _test_of(fed, domain):
    c = fed.clients_of(domain)[0]
    return c.test_sequences, c.test_labels


def run_known_events(fed, rows):
    """Set-union sequential: accumulate known events as domains arrive; never forgets, grows."""
    det = KnownEventsDetector()
    seen_local_sets = []
    f1_hdfs = {}
    for stage, domain in enumerate(ARRIVALS):
        # this domain's clients contribute their local event sets
        for c in fed.clients_of(domain):
            seen_local_sets.append(det.fit_local(c.normal_train))
        det.aggregate(seen_local_sets)
        size = det.state_size_bytes()
        # evaluate on HDFS (the earlier domain) at every stage to measure forgetting
        seqs, labs = _test_of(fed, "hdfs")
        f1h = prf(labs, det.predict(seqs)).f1
        f1_hdfs[domain] = f1h
        rows.append({"method": "known_events", "after_arrival": domain, "eval_domain": "hdfs",
                     "f1": round(f1h, 4), "model_bytes": size})
        # also evaluate the just-arrived domain
        if domain != "hdfs":
            seqs, labs = _test_of(fed, domain)
            f1d = prf(labs, det.predict(seqs)).f1
            rows.append({"method": "known_events", "after_arrival": domain, "eval_domain": domain,
                         "f1": round(f1d, 4), "model_bytes": size})
        print(f"[known_events after {domain}] HDFS F1={f1h:.4f}  size={size} bytes")
    forget = forgetting(f1_hdfs["hdfs"], f1_hdfs["bgl"])
    print(f"[known_events] HDFS forgetting after BGL arrival = {forget:+.4f}")
    return forget


def run_deeplog(fed, rows, rounds, seed, input_mode="scalar"):
    """Returns (f1_hdfs_before, f1_hdfs_after, forgetting)."""
    from src.methods.deeplog_fed import FedConfig, train_federated, evaluate, state_size_bytes
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=rounds, local_epochs=1, seed=seed,
                    input_mode=input_mode)

    # stage 1: train on HDFS clients only
    print(f"\n[deeplog] stage 1 — train on HDFS ({HDFS_CLIENTS} clients), vocab={fed.vocab.size}")
    hdfs_clients = [c.normal_train for c in fed.clients_of("hdfs")]
    model = train_federated(hdfs_clients, cfg)
    seqs, labs = _test_of(fed, "hdfs")
    f1_hdfs_before = evaluate(model, seqs, labs, cfg).f1
    size1 = state_size_bytes(model)
    rows.append({"method": "deeplog", "after_arrival": "hdfs", "eval_domain": "hdfs",
                 "f1": round(f1_hdfs_before, 4), "model_bytes": size1})
    print(f"[deeplog after hdfs] HDFS F1={f1_hdfs_before:.4f}  size={size1} bytes")

    # stage 2: CONTINUE training the same model on BGL clients only
    print(f"[deeplog] stage 2 — continue on BGL ({BGL_CLIENTS} clients)")
    bgl_clients = [c.normal_train for c in fed.clients_of("bgl")]
    model = train_federated(bgl_clients, cfg, init_model=model)
    seqs, labs = _test_of(fed, "hdfs")
    f1_hdfs_after = evaluate(model, seqs, labs, cfg).f1
    seqs_b, labs_b = _test_of(fed, "bgl")
    f1_bgl_after = evaluate(model, seqs_b, labs_b, cfg).f1
    size2 = state_size_bytes(model)
    rows.append({"method": "deeplog", "after_arrival": "bgl", "eval_domain": "hdfs",
                 "f1": round(f1_hdfs_after, 4), "model_bytes": size2})
    rows.append({"method": "deeplog", "after_arrival": "bgl", "eval_domain": "bgl",
                 "f1": round(f1_bgl_after, 4), "model_bytes": size2})
    print(f"[deeplog after bgl] HDFS F1={f1_hdfs_after:.4f}  BGL F1={f1_bgl_after:.4f}  size={size2} bytes")
    forget = forgetting(f1_hdfs_before, f1_hdfs_after)
    print(f"[deeplog] HDFS forgetting after BGL arrival = {forget:+.4f}")
    return f1_hdfs_before, f1_hdfs_after, forget


def main(rounds, seed, do_deeplog, input_mode="scalar", out_path=None):
    t0 = time.time()
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    rows = []
    print("=" * 64)
    print(f"H3 — sequential arrival: HDFS then BGL  (deeplog input_mode={input_mode})")
    print("=" * 64)
    ke_forget = run_known_events(fed, rows)
    dl = run_deeplog(fed, rows, rounds, seed, input_mode) if do_deeplog else None
    dl_forget = dl[2] if dl else None

    os.makedirs("results", exist_ok=True)
    if out_path is not None:
        out = out_path
    else:
        out = "results/h3_sequential_embedding.csv" if input_mode == "embedding" else "results/h3_sequential.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "after_arrival", "eval_domain", "f1", "model_bytes"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 64)
    print("H3 SUMMARY")
    print("=" * 64)
    print(f"known_events : HDFS forgetting = {ke_forget:+.4f}  (expect ~0, union never forgets)")
    ke_sizes = [r["model_bytes"] for r in rows if r["method"] == "known_events" and r["eval_domain"] == "hdfs"]
    if len(ke_sizes) >= 2:
        print(f"known_events : size {ke_sizes[0]} -> {ke_sizes[-1]} bytes "
              f"(x{ke_sizes[-1] / max(ke_sizes[0], 1):.1f} growth across arrivals)")
    if dl_forget is not None:
        print(f"deeplog      : HDFS forgetting = {dl_forget:+.4f}  (expect >0, adapts and forgets)")
        dl_sizes = [r["model_bytes"] for r in rows if r["method"] == "deeplog" and r["eval_domain"] == "hdfs"]
        if len(dl_sizes) >= 2:
            print(f"deeplog      : size {dl_sizes[0]} -> {dl_sizes[-1]} bytes (fixed)")
    print(f"\nwrote {out}   (elapsed {time.time() - t0:.1f}s)")

    return {"seed": seed, "ke_forget": ke_forget,
            "dl_hdfs_before": dl[0] if dl else None,
            "dl_hdfs_after": dl[1] if dl else None,
            "dl_forget": dl_forget}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-deeplog", action="store_true", help="skip the deep model (fast set-union-only)")
    ap.add_argument("--embed", action="store_true",
                    help="use a learned embedding input for DeepLog (mechanism test)")
    args = ap.parse_args()
    main(args.rounds, args.seed, do_deeplog=not args.no_deeplog,
         input_mode="embedding" if args.embed else "scalar")
