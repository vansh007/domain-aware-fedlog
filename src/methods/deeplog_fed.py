"""
deeplog_fed.py — DeepLog under FedAvg, for H2.

H2: FedAvg on a deep model degrades in a mixed HDFS+BGL federation because the client
event vocabularies are largely disjoint (HDFS global ids 0..32, BGL 33..426), so
weight-averaging blends models that learned different input spaces.

What this file is (and is not):
  - It is a faithful re-implementation of the DeepLog *mechanism* (sliding-window
    next-event prediction; anomaly = true next event not in the model's top-g predictions)
    plus a plain FedAvg weight-averaging loop.
  - It is NOT a copy of the reference repo's GPL code. The LSTM here is the standard,
    minimal DeepLog architecture (2 x 64 LSTM + linear head); we wrote it fresh so our
    federated code stays under our own license (see docs/DECISIONS.md D2). Hyper-parameters
    (window=10, num_candidates=9, hidden=64, layers=2) match the reference config so the
    comparison is fair.

Key adaptation vs the reference: the reference subtracts 1 from every event id (their ids
are 1-indexed). Our merged vocabulary is already 0-indexed global ids in [0, vocab_size),
so we feed ids directly and set num_classes = vocab_size. No -1 shift.

CPU-first (CLAUDE.md rule 4). The model is tiny; the expensive part is evaluation, which we
make tractable by deduplicating identical test sequences and weighting by their count
(exactly the reference's `generate` trick).

No fabricated numbers: every metric returned comes from a real forward pass.
"""

from __future__ import annotations
from dataclasses import dataclass
import random

import torch
import torch.nn as nn

from src.metrics import PRF


# --------------------------------------------------------------------------------------
# Model — standard DeepLog: stacked LSTM over scalar event ids + linear next-event head.
# --------------------------------------------------------------------------------------
class DeepLog(nn.Module):
    """DeepLog LSTM with two selectable input encodings (this choice is the crux of H2/H3):

      - input_mode='scalar'    : feed the raw event id as a single float (the reference's
                                 encoding). Different domains' ids occupy disjoint numeric
                                 ranges, so the model can separate them by magnitude alone.
      - input_mode='embedding' : map each id through a learned embedding. Here the two
                                 vocabularies overlap in feature space, so mixing/adaptation
                                 must actually reconcile them — the condition H2/H3 assume.
    """

    def __init__(self, num_keys: int, hidden_size: int = 64, num_layers: int = 2,
                 input_mode: str = "scalar", embed_dim: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_mode = input_mode
        if input_mode == "embedding":
            self.embed = nn.Embedding(num_keys, embed_dim)
            lstm_input = embed_dim
        elif input_mode == "scalar":
            self.embed = None
            lstm_input = 1
        else:
            raise ValueError(f"unknown input_mode {input_mode!r}")
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scalar: x is [B, window, 1] float; embedding: x is [B, window] long ids
        if self.embed is not None:
            x = self.embed(x)                  # [B, window, embed_dim]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])          # logits over next event, [B, num_keys]


class TransformerLog(nn.Module):
    """A modern, self-attention next-event detector with the SAME interface as DeepLog, so it
    drops into the identical H2/H3/mechanism harness (window in, top-g rule out). Its purpose is
    generality: the paper's forgetting mechanism predicts that ANY semantics-based encoder
    (shared feature space across domains) forgets under sequential arrival, and any disjoint-range
    encoder does not. A Transformer with a learned embedding shares feature space (embedding mode);
    with a scalar-id projection it keeps domains in disjoint magnitude ranges (scalar mode) --- the
    same two conditions we test on DeepLog. Kept tiny (2 layers, 64-d, CPU-first).

    Accepts exactly the tensors DeepLog does: [B, window, 1] float (scalar) or [B, window] long
    (embedding), so evaluate()/_build_window_tensors need no changes."""

    def __init__(self, num_keys: int, hidden_size: int = 64, num_layers: int = 2,
                 input_mode: str = "scalar", embed_dim: int = 16, nhead: int = 4,
                 window: int = 10):
        super().__init__()
        self.input_mode = input_mode
        d_model = hidden_size
        if input_mode == "embedding":
            self.embed = nn.Embedding(num_keys, d_model)   # shared feature space across domains
            self.proj = None
        elif input_mode == "scalar":
            self.embed = None
            self.proj = nn.Linear(1, d_model)              # raw id magnitude -> disjoint ranges
        else:
            raise ValueError(f"unknown input_mode {input_mode!r}")
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))  # learned positional encoding
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=2 * d_model, dropout=0.0,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            h = self.embed(x)                  # [B, window, d_model]
        else:
            h = self.proj(x)                   # [B, window, 1] -> [B, window, d_model]
        h = h + self.pos[:, :h.size(1), :]
        h = self.encoder(h)                    # [B, window, d_model]
        return self.fc(h[:, -1, :])            # logits over next event, [B, num_keys]


# --------------------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------------------
def _windows(seq: list[int], window: int):
    """Yield (window_ids, next_id) pairs for one sequence. Empty if len(seq) <= window."""
    for i in range(len(seq) - window):
        yield seq[i:i + window], seq[i + window]


def _build_window_tensors(sequences: list[list[int]], window: int, device: str,
                          input_mode: str = "scalar"):
    """Flatten all sequences into (X, y) training tensors. y: [N].
    X is [N, window, 1] float for scalar mode, or [N, window] long for embedding mode."""
    xs, ys = [], []
    for s in sequences:
        for win, nxt in _windows(s, window):
            xs.append(win)
            ys.append(nxt)
    if not xs:
        return None, None
    if input_mode == "embedding":
        X = torch.tensor(xs, dtype=torch.long, device=device)                 # [N, window]
    else:
        X = torch.tensor(xs, dtype=torch.float32, device=device).unsqueeze(-1)  # [N, window, 1]
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return X, y


# --------------------------------------------------------------------------------------
# Local training + FedAvg
# --------------------------------------------------------------------------------------
def _train_local(model: DeepLog, X: torch.Tensor, y: torch.Tensor,
                 epochs: int, lr: float, batch: int) -> int:
    """Train `model` in place on one client's windows. Returns #samples (for FedAvg weight)."""
    if X is None:
        return 0
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(X[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
    return n


def _fedavg(state_dicts: list[dict], weights: list[int]) -> dict:
    """Weighted average of client state dicts (weights = #local samples)."""
    total = float(sum(weights)) or 1.0
    avg = {}
    for k in state_dicts[0]:
        acc = None
        for sd, w in zip(state_dicts, weights):
            term = sd[k].float() * (w / total)
            acc = term if acc is None else acc + term
        avg[k] = acc
    return avg


@dataclass
class FedConfig:
    num_keys: int
    window: int = 10
    num_candidates: int = 9
    hidden_size: int = 64
    num_layers: int = 2
    rounds: int = 10
    local_epochs: int = 1
    lr: float = 1e-3
    batch: int = 1024
    device: str = "cpu"
    seed: int = 0
    input_mode: str = "scalar"     # 'scalar' (reference) or 'embedding' (mechanism test)
    embed_dim: int = 16
    model_type: str = "deeplog"    # 'deeplog' (LSTM, default) or 'transformer' (generality test)
    nhead: int = 4                 # attention heads when model_type='transformer'


def train_federated(client_train_seqs: list[list[list[int]]], cfg: FedConfig,
                    init_model: "DeepLog | None" = None) -> DeepLog:
    """FedAvg over clients. `client_train_seqs[i]` = list of encoded normal sequences for
    client i. Returns the trained global model.

    init_model : if given, training CONTINUES from this model's weights instead of a fresh
    init. Used for H3 sequential arrival (train on domain 1, then keep training on domain 2
    with the same fixed-size model — this is where deep models forget)."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = cfg.device

    def _new_model():
        if cfg.model_type == "transformer":
            return TransformerLog(cfg.num_keys, cfg.hidden_size, cfg.num_layers,
                                  input_mode=cfg.input_mode, embed_dim=cfg.embed_dim,
                                  nhead=cfg.nhead, window=cfg.window).to(device)
        return DeepLog(cfg.num_keys, cfg.hidden_size, cfg.num_layers,
                       input_mode=cfg.input_mode, embed_dim=cfg.embed_dim).to(device)

    global_model = _new_model()
    if init_model is not None:
        global_model.load_state_dict(init_model.state_dict())

    # pre-build each client's window tensors once
    client_tensors = [_build_window_tensors(seqs, cfg.window, device, cfg.input_mode)
                      for seqs in client_train_seqs]

    for rnd in range(cfg.rounds):
        state_dicts, weights = [], []
        for (X, y) in client_tensors:
            local = _new_model()
            local.load_state_dict(global_model.state_dict())   # start from global
            n = _train_local(local, X, y, cfg.local_epochs, cfg.lr, cfg.batch)
            if n > 0:
                state_dicts.append(local.state_dict())
                weights.append(n)
        if state_dicts:
            global_model.load_state_dict(_fedavg(state_dicts, weights))
    return global_model


# --------------------------------------------------------------------------------------
# Evaluation — dedup identical sequences, weight by count (reference's tractability trick)
# --------------------------------------------------------------------------------------
def _dedup(sequences: list[list[int]]) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for s in sequences:
        t = tuple(s)
        counts[t] = counts.get(t, 0) + 1
    return counts


@torch.no_grad()
def _flagged(model: DeepLog, seq: tuple, cfg: FedConfig) -> bool:
    """True if ANY window's true next-event is not in the model's top-g predictions.
    Sequences too short to form a window are treated as normal (documented convention)."""
    wins, labels = [], []
    for win, nxt in _windows(list(seq), cfg.window):
        wins.append(win)
        labels.append(nxt)
    if not wins:
        return False
    if cfg.input_mode == "embedding":
        X = torch.tensor(wins, dtype=torch.long, device=cfg.device)          # [W, window]
    else:
        X = torch.tensor(wins, dtype=torch.float32, device=cfg.device).unsqueeze(-1)
    logits = model(X)                                    # [W, num_keys]
    topk = torch.topk(logits, cfg.num_candidates, dim=1).indices  # [W, g]
    y = torch.tensor(labels, device=cfg.device).unsqueeze(1)      # [W, 1]
    in_topk = (topk == y).any(dim=1)                     # [W] true if predicted well
    return bool((~in_topk).any())                        # anomalous if any window mispredicts


def evaluate(model: DeepLog, test_sequences: list[list[int]], test_labels: list[int],
             cfg: FedConfig) -> PRF:
    """Per-domain PRF using the DeepLog top-g rule. Anomaly = positive class (label 1)."""
    model.eval()
    normal = [s for s, l in zip(test_sequences, test_labels) if l == 0]
    abnormal = [s for s, l in zip(test_sequences, test_labels) if l == 1]
    norm_counts = _dedup(normal)
    abn_counts = _dedup(abnormal)

    FP = TN = 0
    for seq, c in norm_counts.items():
        if _flagged(model, seq, cfg):
            FP += c
        else:
            TN += c
    TP = FN = 0
    for seq, c in abn_counts.items():
        if _flagged(model, seq, cfg):
            TP += c
        else:
            FN += c

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, TP, FP, FN, TN)


def state_size_bytes(model: DeepLog) -> int:
    """Fixed-size model footprint (bytes) — the H3 contrast against set-union growth."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
