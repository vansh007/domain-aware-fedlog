#!/usr/bin/env python
"""
deeplog_centralized_hdfs.py — STANDALONE centralized DeepLog upper-bound baseline on HDFS.

One-off run to establish the capacity ceiling of our DeepLog under the most favourable
setting: full data, no federation. It does NOT touch any federated script or result file;
it only imports the DeepLog model and evaluate() read-only.

Exact specification (as requested):
  - Training data: the FULL HDFS normal.csv (all 558,223 normal sequences), not 1%.
  - Centralized: a single model on a single machine; no FedAvg.
  - 100 rounds x 1 local epoch per round (= 100 full passes over the training windows).
  - Adam lr=1e-3, batch=1024, window=10, hidden=64, num_layers=2, top-g (num_candidates)=9.
  - Validation (transductive, per the upper-bound framing): all normal (seen in training) as
    negatives + all abnormal as positives; F1/precision/recall computed each round.
  - Writes results/deeplog_centralized_hdfs.csv incrementally with columns
    [round, train_loss, val_f1, val_precision, val_recall].

Prints the final F1/precision/recall and the plateau round at the end.
"""

from __future__ import annotations
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import _load_parsed_domain
from src.vocabulary import GlobalVocabulary
from src.methods.deeplog_fed import DeepLog, FedConfig, evaluate  # read-only reuse

DATA_ROOT = "data"
WINDOW = 10
HIDDEN = 64
LAYERS = 2
LR = 1e-3
BATCH = 1024
NUM_CANDIDATES = 9
ROUNDS = 100
SEED = 0
OUT = "results/deeplog_centralized_hdfs.csv"


def build_windows(seqs, window):
    """Efficient (numpy stride-trick) sliding windows over all sequences."""
    n_total = sum(max(0, len(s) - window) for s in seqs)
    X = np.empty((n_total, window), dtype=np.int64)
    Y = np.empty((n_total,), dtype=np.int64)
    i = 0
    for s in seqs:
        if len(s) <= window:
            continue
        a = np.asarray(s, dtype=np.int64)
        sw = np.lib.stride_tricks.sliding_window_view(a, window + 1)
        n = sw.shape[0]
        X[i:i + n] = sw[:, :window]
        Y[i:i + n] = sw[:, window]
        i += n
    return X, Y


def plateau_round(f1s, tol=0.005):
    """Earliest round after which F1 stays within `tol` of the eventual max (1-indexed)."""
    if not f1s:
        return None
    mx = max(f1s)
    for r, v in enumerate(f1s, start=1):
        if v >= mx - tol:
            return r
    return len(f1s)


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- data (full HDFS) ----
    templates, examples = _load_parsed_domain("hdfs", DATA_ROOT)
    vocab = GlobalVocabulary()
    vocab.add_domain("hdfs", templates)
    normal = [vocab.encode_sequence("hdfs", s) for s, l in examples if l == 0]
    abnormal = [vocab.encode_sequence("hdfs", s) for s, l in examples if l == 1]
    print(f"[data] normal={len(normal)} abnormal={len(abnormal)} vocab={vocab.size}", flush=True)

    X, Y = build_windows(normal, WINDOW)
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)   # [N, W, 1]
    Yt = torch.tensor(Y, dtype=torch.long)
    N = Xt.shape[0]
    print(f"[data] training windows={N:,}  ({Xt.element_size()*Xt.nelement()/1e6:.0f} MB)", flush=True)

    # transductive validation set
    val_seqs = normal + abnormal
    val_labels = [0] * len(normal) + [1] * len(abnormal)
    cfg = FedConfig(num_keys=vocab.size, window=WINDOW, num_candidates=NUM_CANDIDATES,
                    hidden_size=HIDDEN, num_layers=LAYERS)

    # ---- model / optimizer (centralized: one model) ----
    model = DeepLog(vocab.size, HIDDEN, LAYERS, input_mode="scalar")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("results", exist_ok=True)
    f = open(OUT, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=["round", "train_loss", "val_f1",
                                           "val_precision", "val_recall"])
    writer.writeheader()
    f.flush()

    f1s = []
    for rnd in range(1, ROUNDS + 1):
        model.train()
        perm = torch.randperm(N)
        total_loss, nb = 0.0, 0
        for j in range(0, N, BATCH):
            idx = perm[j:j + BATCH]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, Yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        train_loss = total_loss / nb

        m = evaluate(model, val_seqs, val_labels, cfg)
        f1s.append(m.f1)
        writer.writerow({"round": rnd, "train_loss": round(train_loss, 6),
                         "val_f1": round(m.f1, 6), "val_precision": round(m.precision, 6),
                         "val_recall": round(m.recall, 6)})
        f.flush()
        print(f"[round {rnd:3d}] loss={train_loss:.4f}  F1={m.f1:.4f}  "
              f"P={m.precision:.4f}  R={m.recall:.4f}  "
              f"[{(time.time()-t_start)/60:.1f} min]", flush=True)

    f.close()

    final = f1s[-1]
    best = max(f1s)
    best_round = f1s.index(best) + 1
    plat = plateau_round(f1s)
    # final metrics come from the last round's evaluate()
    print("\n" + "=" * 64)
    print("CENTRALIZED DEEPLOG (HDFS, full normal) — FINAL")
    print("=" * 64)
    print(f"final round {ROUNDS}:  F1={m.f1:.4f}  precision={m.precision:.4f}  recall={m.recall:.4f}")
    print(f"best F1={best:.4f} at round {best_round}")
    print(f"plateau round (F1 within 0.005 of max): {plat}")
    print(f"wrote {OUT}  (total {(time.time()-t_start)/3600:.2f} h)")


if __name__ == "__main__":
    main()
