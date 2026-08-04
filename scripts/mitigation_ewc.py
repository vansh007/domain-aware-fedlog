#!/usr/bin/env python
"""
mitigation_ewc.py — a MEMORY-FREE anti-forgetting defense (EWC) for the
(sequential, shared-representation) failure, compared head-to-head with replay.

Section~\\ref{sec:replay} showed a small experience-replay buffer removes the
embedding-DeepLog catastrophic forgetting (HDFS F1 0.70 -> 0.04 without it). Replay,
however, stores raw sequences of the earlier domain. A reviewer will ask the obvious
question: does the standard *parameter-based* defense, Elastic Weight Consolidation
(EWC, Kirkpatrick et al. 2017), fix it WITHOUT storing any old data? This script answers
that directly, so the paper compares a memory-based (replay) and a memory-free (EWC)
mitigation under the identical sequential-arrival protocol.

EWC recipe (faithful, federated):
  1. Train stage-1 on HDFS -> model0 (the anchor theta*).
  2. Estimate the empirical Fisher information F_i on HDFS training windows at model0
     (how "important" each weight is to HDFS).
  3. When BGL arrives, continue federated training on BGL clients, but each local update
     adds the EWC penalty  (lambda/2) * sum_i F_i (theta_i - theta*_i)^2, which pins the
     HDFS-important weights in place while the rest adapt to BGL.
We sweep lambda (including lambda=0, which must reproduce the no-defense forgetting) over
3 seeds, all in the EMBEDDING condition (the only one that forgets), at the H2/H3 plateau
(rounds=12), CPU-first. Writes results/mitigation_ewc.csv.

This script imports the DeepLog model/eval/windowing READ-ONLY and reimplements only the
local training loop (to add the penalty); it does not modify any existing federated script
or result.

Usage:
    python scripts/mitigation_ewc.py                      # seeds 0..2, default lambda sweep
    python scripts/mitigation_ewc.py --seeds 0 1 2 --lambdas 0 100 10000 1000000
"""

from __future__ import annotations
import argparse
import csv
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.dataloader import build_federation
from src.methods.deeplog_fed import (DeepLog, FedConfig, train_federated, evaluate,
                                      _build_window_tensors, _fedavg)

HDFS_CLIENTS = 3
BGL_CLIENTS = 2
DATA_ROOT = "data"
ROUNDS = 12
INPUT_MODE = "embedding"   # the failure condition


def _test_of(fed, domain):
    c = fed.clients_of(domain)[0]
    return c.test_sequences, c.test_labels


def _new_model(cfg: FedConfig) -> DeepLog:
    return DeepLog(cfg.num_keys, cfg.hidden_size, cfg.num_layers,
                   input_mode=cfg.input_mode, embed_dim=cfg.embed_dim).to(cfg.device)


def compute_fisher(model: DeepLog, X, y, cfg: FedConfig) -> dict:
    """Empirical Fisher: average of squared loss-gradients over the earlier domain's windows.
    A larger F_i means weight i matters more to the earlier (HDFS) task."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    for i in range(0, n, cfg.batch):
        xb, yb = X[i:i + cfg.batch], y[i:i + cfg.batch]
        model.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        bs = xb.shape[0]
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += (p.grad.detach() ** 2) * bs
    for name in fisher:
        fisher[name] /= max(n, 1)
    return fisher


def _train_local_ewc(model, X, y, cfg, anchor, fisher, lam):
    """One client's local training with the EWC quadratic penalty toward `anchor`."""
    if X is None:
        return 0
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            if lam > 0:
                pen = model.fc.weight.new_zeros(())
                for name, p in model.named_parameters():
                    pen = pen + (fisher[name] * (p - anchor[name]) ** 2).sum()
                loss = loss + (lam / 2.0) * pen
            loss.backward()
            opt.step()
    return n


def train_federated_ewc(client_seqs, cfg, init_model, anchor, fisher, lam):
    """FedAvg over BGL clients continuing from init_model, each local step EWC-penalized."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    global_model = _new_model(cfg)
    global_model.load_state_dict(init_model.state_dict())
    client_tensors = [_build_window_tensors(s, cfg.window, cfg.device, cfg.input_mode)
                      for s in client_seqs]
    for _ in range(cfg.rounds):
        sds, ws = [], []
        for (X, y) in client_tensors:
            if X is None:
                continue
            local = _new_model(cfg)
            local.load_state_dict(global_model.state_dict())
            n = _train_local_ewc(local, X, y, cfg, anchor, fisher, lam)
            sds.append(local.state_dict())
            ws.append(n)
        if sds:
            global_model.load_state_dict(_fedavg(sds, ws))
    return global_model


def run_seed(seed, lambdas):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=ROUNDS, local_epochs=1, seed=seed,
                    input_mode=INPUT_MODE)

    hdfs_clients = [c.normal_train for c in fed.clients_of("hdfs")]
    bgl_clients = [c.normal_train for c in fed.clients_of("bgl")]
    hdfs_pool = [s for client in hdfs_clients for s in client]

    hdfs_test = _test_of(fed, "hdfs")
    bgl_test = _test_of(fed, "bgl")

    # ---- stage 1: HDFS only (the anchor task) ----
    model0 = train_federated(hdfs_clients, cfg)
    f1_before = evaluate(model0, *hdfs_test, cfg).f1
    anchor = {n: p.detach().clone() for n, p in model0.named_parameters()}

    # Fisher on the pooled HDFS training windows at model0
    Xh, yh = _build_window_tensors(hdfs_pool, cfg.window, cfg.device, cfg.input_mode)
    fisher = compute_fisher(model0, Xh, yh, cfg)
    print(f"[seed {seed}] stage1 HDFS F1_before={f1_before:.4f}  (pool={len(hdfs_pool)} seqs)")

    rows = []
    for lam in lambdas:
        m = train_federated_ewc(bgl_clients, cfg, model0, anchor, fisher, lam)
        hdfs_after = evaluate(m, *hdfs_test, cfg).f1
        bgl_after = evaluate(m, *bgl_test, cfg).f1
        rows.append({"seed": seed, "lambda": lam,
                     "hdfs_before": round(f1_before, 6), "hdfs_after": round(hdfs_after, 6),
                     "bgl_after": round(bgl_after, 6),
                     "forgetting": round(f1_before - hdfs_after, 6)})
        tag = "no-defense" if lam == 0 else f"lambda={lam:g}"
        print(f"[seed {seed}]   EWC {tag:<16} HDFS_after={hdfs_after:.4f}  "
              f"BGL_after={bgl_after:.4f}  forget={f1_before - hdfs_after:+.4f}")
    return rows


def main(seeds, lambdas):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(s, lambdas))

    os.makedirs("results", exist_ok=True)
    out = "results/mitigation_ewc.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "lambda", "hdfs_before", "hdfs_after",
                                          "bgl_after", "forgetting"])
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 72)
    print(f"EWC MITIGATION SUMMARY (n={len(seeds)} seeds: {seeds}, input={INPUT_MODE})")
    print("=" * 72)
    print(f"{'lambda':>10}  {'HDFS_after':>12} {'BGL_after':>12} {'forgetting':>12}")
    for lam in lambdas:
        sub = [r for r in all_rows if r["lambda"] == lam]
        if not sub:
            continue

        def ms(key):
            v = [r[key] for r in sub]
            return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
        hm, hs = ms("hdfs_after"); bm, bs = ms("bgl_after"); fm, fs = ms("forgetting")
        label = "no-defense" if lam == 0 else f"{lam:g}"
        print(f"{label:>10}  {hm:>6.4f}+/-{hs:.4f} {bm:>6.4f}+/-{bs:.4f} "
              f"{fm:>+6.4f}+/-{fs:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 1e2, 1e4, 1e6, 1e8])
    args = ap.parse_args()
    main(args.seeds, args.lambdas)
