"""
ecvc.py — ECVC-style event-count-vector detector (our own implementation).

ECVC (Event Count Vector Comparison, from the ADSD paper, ACM 10.1145/3660768) represents
each log sequence as a histogram over event types, and scores a test sequence by its distance
to the *nearest* normal training histogram. Sequences far from every normal count-profile are
anomalous. It is the third member of the reference paper's best HDFS ensemble
(Events + Length + ECVC).

This is a fresh implementation of the published algorithm (not a port of the GPL reference
code — see docs/DECISIONS.md D2):
  - vectorise a sequence as a normalised event-count histogram over the merged vocabulary,
  - score(x) = min over normal training histograms of L1(norm(x), norm(x_i)),
  - flag anomalous if score >= threshold, where the threshold is chosen by supervised
    selection (the F1-maximising split of normal vs abnormal scores) — the same transductive
    calibration the reference uses.

Federated aggregation = union of clients' (deduplicated) training histograms, mirroring the
reference's ECVC update strategy. CPU-only; test histograms are deduplicated for tractability.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch


def _count_vectors(sequences: list[list[int]], vocab_size: int) -> torch.Tensor:
    """[N, vocab_size] raw event-count histograms."""
    if not sequences:
        return torch.zeros((0, vocab_size))
    out = torch.zeros((len(sequences), vocab_size), dtype=torch.float32)
    for i, s in enumerate(sequences):
        if s:
            idx = torch.tensor(s, dtype=torch.long)
            out[i].index_add_(0, idx, torch.ones(len(s)))
    return out


def _normalise(v: torch.Tensor) -> torch.Tensor:
    return v / (v.sum(dim=1, keepdim=True) + 1e-10)


@dataclass
class ECVCDetector:
    vocab_size: int
    threshold: float | None = None
    _train_norm: torch.Tensor = field(default=None, repr=False)

    def fit_local(self, normal_sequences: list[list[int]]) -> torch.Tensor:
        """Client-side: deduplicated normalised training histograms."""
        vecs = _count_vectors(normal_sequences, self.vocab_size)
        vecs = torch.unique(vecs, dim=0)
        return vecs

    def aggregate(self, local_vec_sets: list[torch.Tensor]) -> None:
        """Server-side: union of clients' histograms (dedup)."""
        nonempty = [v for v in local_vec_sets if v.numel() > 0]
        allv = torch.cat(nonempty, dim=0) if nonempty else torch.zeros((0, self.vocab_size))
        allv = torch.unique(allv, dim=0)
        self._train_norm = _normalise(allv)

    def score(self, sequences: list[list[int]], batch: int = 512) -> torch.Tensor:
        """Min L1 distance from each sequence's normalised histogram to the nearest normal one."""
        if self._train_norm is None or self._train_norm.shape[0] == 0:
            return torch.zeros(len(sequences))
        out = torch.empty(len(sequences))
        for i in range(0, len(sequences), batch):
            chunk = sequences[i:i + batch]
            q = _normalise(_count_vectors(chunk, self.vocab_size))
            d = torch.cdist(q, self._train_norm, p=1)          # [chunk, Ntrain]
            out[i:i + len(chunk)] = d.min(dim=1).values
        return out

    def set_threshold_supervised(self, scores: torch.Tensor, labels: list[int],
                                 steps: int = 1000) -> None:
        """Pick the F1-maximising threshold over normal/abnormal scores (transductive)."""
        y = torch.tensor(labels)
        lo, hi = float(scores.min()), float(scores.max())
        best_t, best_f1 = hi, -1.0
        for t in torch.linspace(lo, hi, steps):
            pred = (scores >= t).int()
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t

    def predict(self, sequences: list[list[int]]) -> list[int]:
        if self.threshold is None:
            raise RuntimeError("call set_threshold_supervised() before predict()")
        return (self.score(sequences) >= self.threshold).int().tolist()
