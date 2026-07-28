#!/usr/bin/env python
"""
run_h2_deeplog.py — the H2 experiment: does FedAvg on DeepLog degrade under domain mixing?

For each domain we compare:
  - SINGLE : DeepLog trained by FedAvg over ONLY that domain's clients, its own vocab.
  - MIXED  : DeepLog trained by FedAvg over ALL clients (HDFS+BGL), merged 427 vocab.
Both are evaluated on the SAME per-domain test set. H2 predicts MIXED < SINGLE.

Writes results/h2_deeplog.csv (one row per method/domain/setting) plus a printed summary
with the per-domain F1 drop (single - mixed).

CPU-first. Model is tiny; evaluation is made tractable by deduplicating identical test
sequences (see src/methods/deeplog_fed.evaluate).

Usage:
    python scripts/run_h2_deeplog.py                    # full run (rounds=10)
    python scripts/run_h2_deeplog.py --rounds 5 --seed 0
    python scripts/run_h2_deeplog.py --smoke            # fast sanity: 1 round, capped eval
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import build_federation
from src.methods.deeplog_fed import FedConfig, train_federated, evaluate, state_size_bytes

DATA_ROOT = "data"
HDFS_CLIENTS = 3
BGL_CLIENTS = 2


def _client_train(fed, domain):
    return [c.normal_train for c in fed.clients_of(domain)]


def _cap_test(fed, domain, cap):
    """Optionally cap the test set (smoke mode only) for a fast end-to-end check."""
    c = fed.clients_of(domain)[0]
    seqs, labs = c.test_sequences, c.test_labels
    if cap is None:
        return seqs, labs
    # keep first `cap` normals and first `cap` abnormals (deterministic)
    norm = [(s, l) for s, l in zip(seqs, labs) if l == 0][:cap]
    abn = [(s, l) for s, l in zip(seqs, labs) if l == 1][:cap]
    both = norm + abn
    return [s for s, _ in both], [l for _, l in both]


def main(rounds: int, local_epochs: int, seed: int, smoke: bool, out_path: str | None = None,
         input_mode: str = "scalar") -> dict:
    eval_cap = 400 if smoke else None
    if smoke:
        rounds = min(rounds, 1)

    t0 = time.time()
    rows = []

    # ---------- SINGLE-domain models ----------
    single_f1 = {}
    for domain, n_clients in [("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)]:
        fed = build_federation([(domain, n_clients)], data_root=DATA_ROOT, split="iid", seed=seed)
        cfg = FedConfig(num_keys=fed.vocab.size, rounds=rounds, local_epochs=local_epochs,
                        seed=seed, input_mode=input_mode)
        print(f"\n[SINGLE {domain}] vocab={fed.vocab.size} clients={n_clients} "
              f"rounds={rounds} — training...")
        model = train_federated(_client_train(fed, domain), cfg)
        seqs, labs = _cap_test(fed, domain, eval_cap)
        m = evaluate(model, seqs, labs, cfg)
        single_f1[domain] = m.f1
        rows.append({"method": "deeplog", "domain": domain, "setting": "single",
                     "model_bytes": state_size_bytes(model), "num_keys": fed.vocab.size,
                     **m.as_row()})
        print(f"[SINGLE {domain}] F1={m.f1:.4f} (P={m.precision:.3f} R={m.recall:.3f})")

    # ---------- MIXED-domain model ----------
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=rounds, local_epochs=local_epochs,
                    seed=seed, input_mode=input_mode)
    all_clients = [c.normal_train for c in fed.clients]
    print(f"\n[MIXED] vocab={fed.vocab.size} clients={len(fed.clients)} rounds={rounds} — training...")
    model = train_federated(all_clients, cfg)
    mixed_bytes = state_size_bytes(model)
    for domain in fed.vocab.domains:
        seqs, labs = _cap_test(fed, domain, eval_cap)
        m = evaluate(model, seqs, labs, cfg)
        rows.append({"method": "deeplog", "domain": domain, "setting": "mixed",
                     "model_bytes": mixed_bytes, "num_keys": fed.vocab.size, **m.as_row()})
        drop = single_f1[domain] - m.f1
        print(f"[MIXED {domain}] F1={m.f1:.4f} (P={m.precision:.3f} R={m.recall:.3f})  "
              f"| single={single_f1[domain]:.4f} -> drop={drop:+.4f}")

    # ---------- write ----------
    os.makedirs("results", exist_ok=True)
    if out_path is not None:
        out = out_path
    else:
        out = "results/h2_deeplog_smoke.csv" if smoke else "results/h2_deeplog.csv"
    fields = ["method", "domain", "setting", "precision", "recall", "f1",
              "tp", "fp", "fn", "tn", "model_bytes", "num_keys"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 64)
    print(f"H2 SUMMARY (seed={seed}, rounds={rounds}, smoke={smoke}, input_mode={input_mode})")
    print("=" * 64)
    print(f"{'domain':6s} {'single F1':>10s} {'mixed F1':>10s} {'drop':>8s}")
    for domain in ["hdfs", "bgl"]:
        sm = next((r for r in rows if r["domain"] == domain and r["setting"] == "mixed"), None)
        if sm:
            print(f"{domain:6s} {single_f1[domain]:10.4f} {sm['f1']:10.4f} "
                  f"{single_f1[domain] - sm['f1']:+8.4f}")
    print(f"\nwrote {out}   (elapsed {time.time() - t0:.1f}s)")

    result = {"seed": seed}
    for domain in ["hdfs", "bgl"]:
        sm = next((r for r in rows if r["domain"] == domain and r["setting"] == "mixed"), None)
        result[f"{domain}_single"] = single_f1[domain]
        result[f"{domain}_mixed"] = sm["f1"] if sm else None
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="fast sanity: 1 round, capped eval")
    ap.add_argument("--out", default=None, help="explicit output CSV path (per-seed runs)")
    ap.add_argument("--embed", action="store_true",
                    help="use a learned embedding input (mechanism test) instead of scalar id")
    args = ap.parse_args()
    mode = "embedding" if args.embed else "scalar"
    default_out = "results/h2_deeplog_embedding.csv" if args.embed else None
    main(args.rounds, args.local_epochs, args.seed, args.smoke,
         out_path=args.out or default_out, input_mode=mode)
