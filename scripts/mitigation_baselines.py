#!/usr/bin/env python
"""
mitigation_baselines.py — compare our replay fix against an established continual-learning
baseline: A-GEM (Averaged Gradient Episodic Memory, Chaudhry et al. 2019).

Replay (Section sec:replay) simply trains on BGL + a 10% HDFS buffer jointly. A-GEM uses the same
size of episodic memory but a different rule: at each step it computes a reference gradient on a
memory batch of the earlier domain and, if the new-domain gradient would increase loss on that
memory (negative dot product), it projects the gradient to remove the interfering component. This
is the standard memory-based CL method, so it is the fair yardstick for "why your simple replay
instead of the established toolkit?".

Matched budget: A-GEM's episodic memory is the SAME 10% HDFS buffer replay uses (558 sequences,
same deterministic sample per seed), embedding-DeepLog, sequential HDFS->BGL, 3 seeds. We also run
no-defense as a sanity check that reproduces the +0.665 forgetting. Writes results/mitigation_agem.csv;
replay and EWC numbers for the comparison come from results/mitigation_{replay,ewc}.csv.

Usage: python scripts/mitigation_baselines.py [--seeds 0 1 2]
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
INPUT_MODE = "embedding"
FRAC = 0.10


def _test_of(fed, domain):
    c = fed.clients_of(domain)[0]
    return c.test_sequences, c.test_labels


def _new_model(cfg):
    return DeepLog(cfg.num_keys, cfg.hidden_size, cfg.num_layers,
                   input_mode=cfg.input_mode, embed_dim=cfg.embed_dim).to(cfg.device)


def _flat_grad(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])


def _assign_grad(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n


def _train_local_agem(model, X, y, cfg, mem_X, mem_y, rng):
    """One client's local training with the A-GEM gradient projection against episodic memory."""
    if X is None:
        return 0
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    m = mem_X.shape[0]
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            # reference gradient on a random memory batch (the earlier domain)
            mb = torch.randint(0, m, (min(cfg.batch, m),))
            opt.zero_grad()
            loss_fn(model(mem_X[mb]), mem_y[mb]).backward()
            g_ref = _flat_grad(model)
            # current gradient on the new-domain batch
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            g = _flat_grad(model)
            dot = torch.dot(g, g_ref)
            if dot < 0:                                   # interference -> project it out
                g = g - (dot / torch.dot(g_ref, g_ref)) * g_ref
                _assign_grad(model, g)
            opt.step()
    return n


def train_federated_agem(client_seqs, cfg, init_model, mem_X, mem_y):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    global_model = _new_model(cfg)
    global_model.load_state_dict(init_model.state_dict())
    tensors = [_build_window_tensors(s, cfg.window, cfg.device, cfg.input_mode) for s in client_seqs]
    for _ in range(cfg.rounds):
        sds, ws = [], []
        for (X, y) in tensors:
            if X is None:
                continue
            local = _new_model(cfg)
            local.load_state_dict(global_model.state_dict())
            nn_ = _train_local_agem(local, X, y, cfg, mem_X, mem_y, rng)
            sds.append(local.state_dict()); ws.append(nn_)
        if sds:
            global_model.load_state_dict(_fedavg(sds, ws))
    return global_model


def run_seed(seed):
    fed = build_federation([("hdfs", HDFS_CLIENTS), ("bgl", BGL_CLIENTS)],
                           data_root=DATA_ROOT, split="iid", seed=seed)
    cfg = FedConfig(num_keys=fed.vocab.size, rounds=ROUNDS, local_epochs=1, seed=seed,
                    input_mode=INPUT_MODE)
    hdfs_clients = [c.normal_train for c in fed.clients_of("hdfs")]
    bgl_clients = [c.normal_train for c in fed.clients_of("bgl")]
    hdfs_pool = [s for client in hdfs_clients for s in client]
    hdfs_test, bgl_test = _test_of(fed, "hdfs"), _test_of(fed, "bgl")

    model0 = train_federated(hdfs_clients, cfg)
    f1_before = evaluate(model0, *hdfs_test, cfg).f1

    # episodic memory = the SAME 10% HDFS buffer replay uses (same deterministic sample)
    rng = random.Random(1000 + seed)
    k = max(1, int(round(FRAC * len(hdfs_pool))))
    buffer = rng.sample(hdfs_pool, k)
    mem_X, mem_y = _build_window_tensors(buffer, cfg.window, cfg.device, cfg.input_mode)

    rows = []
    # no-defense sanity (should reproduce +0.665)
    m_nd = train_federated(bgl_clients, cfg, init_model=model0)
    rows.append(("no-defense", 0,
                 evaluate(m_nd, *hdfs_test, cfg).f1, evaluate(m_nd, *bgl_test, cfg).f1))
    # A-GEM at 10% memory
    m_ag = train_federated_agem(bgl_clients, cfg, model0, mem_X, mem_y)
    rows.append(("agem_10pct", k,
                 evaluate(m_ag, *hdfs_test, cfg).f1, evaluate(m_ag, *bgl_test, cfg).f1))

    out = []
    for method, buf, hdfs_after, bgl_after in rows:
        out.append({"seed": seed, "method": method, "buffer_seqs": buf,
                    "hdfs_before": round(f1_before, 6), "hdfs_after": round(hdfs_after, 6),
                    "bgl_after": round(bgl_after, 6), "forgetting": round(f1_before - hdfs_after, 6)})
        print(f"[seed {seed}] {method:12s} HDFS_after={hdfs_after:.4f} BGL_after={bgl_after:.4f} "
              f"forget={f1_before - hdfs_after:+.4f}")
    return out


def main(seeds):
    all_rows = []
    for s in seeds:
        all_rows.extend(run_seed(s))
    os.makedirs("results", exist_ok=True)
    out = "results/mitigation_agem.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "method", "buffer_seqs", "hdfs_before",
                                          "hdfs_after", "bgl_after", "forgetting"])
        w.writeheader(); w.writerows(all_rows)

    print("\n" + "=" * 72)
    print(f"A-GEM BASELINE SUMMARY (n={len(seeds)} seeds, embedding, HDFS->BGL, 10% memory)")
    print("=" * 72)
    for method in ("no-defense", "agem_10pct"):
        sub = [r for r in all_rows if r["method"] == method]
        def ms(k):
            v = [r[k] for r in sub]
            return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
        hm, hs = ms("hdfs_after"); bm, bs = ms("bgl_after"); fm, fs = ms("forgetting")
        print(f"{method:12s} HDFS={hm:.4f}+/-{hs:.4f} BGL={bm:.4f}+/-{bs:.4f} "
              f"forget={fm:+.4f}+/-{fs:.4f}")
    print("\nCompare against replay (results/mitigation_replay.csv, 10% -> forget +0.004) and "
          "EWC (results/mitigation_ewc.csv, lambda=1e8 -> forget +0.014).")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    main(args.seeds)
