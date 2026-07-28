#!/usr/bin/env python
"""
deeplog_convergence.py — does more training push single-domain HDFS DeepLog toward the
reference's reported ~0.93?

Trains single-domain HDFS DeepLog (3 clients, 1% train, scalar id input) with a manual FedAvg
loop, evaluating the FULL HDFS test set at checkpoints so we see the F1 trajectory and whether
it plateaus. Reports honestly whether it approaches 0.93.

Context: the reference config (config_files/hdfs_iid.yaml) uses number_rounds=1, max_epoch=1,
train_per=0.01, batch=2048, lr=1e-3, window=10, num_candidates=9 — i.e. LESS training than our
default 12 rounds. So this sweep tests whether more training helps at all.

Usage:
    python scripts/deeplog_convergence.py
    python scripts/deeplog_convergence.py --local-epochs 5 --batch 2048
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.dataloader import build_federation
from src.methods.deeplog_fed import (
    DeepLog, FedConfig, _build_window_tensors, _train_local, _fedavg, evaluate,
)

CHECKPOINTS = [1, 5, 12, 25, 50, 100]


def main(local_epochs, batch, lr, checkpoints, num_candidates):
    fed = build_federation([("hdfs", 3)], data_root="data", split="iid", seed=0)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=max(checkpoints), local_epochs=local_epochs,
                    lr=lr, batch=batch, num_candidates=num_candidates, seed=0)
    dc = fed.clients_of("hdfs")[0]
    test_seqs, test_labs = dc.test_sequences, dc.test_labels

    torch.manual_seed(cfg.seed)
    model = DeepLog(cfg.num_keys, cfg.hidden_size, cfg.num_layers, input_mode="scalar").to(cfg.device)
    client_tensors = [_build_window_tensors(c.normal_train, cfg.window, cfg.device, "scalar")
                      for c in fed.clients_of("hdfs")]

    print(f"config: local_epochs={local_epochs} batch={batch} lr={lr} "
          f"num_candidates={num_candidates} train_windows={sum(t[0].shape[0] for t in client_tensors)}")
    rows = []
    t0 = time.time()
    for rnd in range(1, max(checkpoints) + 1):
        state_dicts, weights = [], []
        for (X, y) in client_tensors:
            local = DeepLog(cfg.num_keys, cfg.hidden_size, cfg.num_layers, input_mode="scalar").to(cfg.device)
            local.load_state_dict(model.state_dict())
            n = _train_local(local, X, y, cfg.local_epochs, cfg.lr, cfg.batch)
            state_dicts.append(local.state_dict()); weights.append(n)
        model.load_state_dict(_fedavg(state_dicts, weights))
        if rnd in checkpoints:
            m = evaluate(model, test_seqs, test_labs, cfg)
            rows.append({"rounds": rnd, "precision": round(m.precision, 4),
                         "recall": round(m.recall, 4), "f1": round(m.f1, 4)})
            print(f"  round {rnd:3d}: HDFS F1={m.f1:.4f} (P={m.precision:.3f} R={m.recall:.3f})  "
                  f"[{time.time()-t0:.0f}s]")

    os.makedirs("results", exist_ok=True)
    out = f"results/deeplog_convergence_le{local_epochs}_b{batch}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rounds", "precision", "recall", "f1"])
        w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["f1"])
    print(f"\nBEST HDFS F1 = {best['f1']} at {best['rounds']} rounds "
          f"(reference reports ~0.91-0.93). wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-candidates", type=int, default=9)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    args = ap.parse_args()
    main(args.local_epochs, args.batch, args.lr, args.checkpoints, args.num_candidates)
