"""
metrics.py — evaluation metrics for the three hypotheses.

Everything here operates on real predictions/labels or real model objects. Nothing here
invents numbers. If you find yourself wanting to "estimate" a metric without a run,
that's the signal to run the experiment instead.

Metrics:
  - precision / recall / f1        : standard detection effectiveness (per domain)
  - forgetting                     : drop on an earlier domain after a later one arrives (H3)
  - model_size_bytes               : size of a lightweight model or DL state (H3)

Run `python -m src.metrics` for a self-test.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_row(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def prf(y_true: list[int], y_pred: list[int]) -> PRF:
    """Binary precision/recall/F1. 1 = anomaly, 0 = normal.

    Convention: anomaly is the positive class (that's what we care about detecting).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn, tn)


def forgetting(f1_before: float, f1_after: float) -> float:
    """H3 forgetting metric: how much F1 on an earlier domain dropped after a new
    domain arrived. Positive = forgot (bad). Negative = improved (rare, fine).

        forgetting = f1_before - f1_after
    """
    return f1_before - f1_after


def model_size_bytes(model) -> int:
    """Best-effort size of a model's learned state, for the H3 growth curve.

    Handles the shapes we actually use:
      - lightweight methods expose `.state_size_bytes()` (implement it per method)
      - a set / list / dict of learned patterns -> recursive rough sizeof
      - a torch module -> sum of parameter buffers (imported lazily; optional)

    The absolute number matters less than the *trend across domain arrivals*, so keep
    the measurement method identical across a run.
    """
    # 1) explicit hook wins
    if hasattr(model, "state_size_bytes"):
        return int(model.state_size_bytes())

    # 2) torch module, if present
    mod = sys.modules.get("torch")
    if mod is not None:
        try:
            import torch  # noqa
            if isinstance(model, torch.nn.Module):
                return sum(p.numel() * p.element_size() for p in model.parameters())
        except Exception:
            pass

    # 3) generic container fallback (rough, but consistent)
    return _rough_sizeof(model)


def _rough_sizeof(obj, _seen=None) -> int:
    """A consistent, dependency-free size estimate for sets/dicts/lists of patterns.
    Not exact bytes-on-disk; used only for relative growth comparisons.
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _rough_sizeof(k, _seen) + _rough_sizeof(v, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _rough_sizeof(item, _seen)
    return size


def _self_test() -> None:
    # perfect prediction
    yt = [0, 0, 1, 1, 0, 1]
    yp = [0, 0, 1, 1, 0, 1]
    m = prf(yt, yp)
    assert m.f1 == 1.0, m

    # the H1 failure shape: Length stops flagging anomalies in the mixed setting,
    # so recall collapses -> F1 collapses even if precision looks fine.
    yt = [1, 1, 1, 1, 0, 0]
    yp = [0, 0, 0, 0, 0, 0]  # flags nothing
    m = prf(yt, yp)
    assert m.recall == 0.0 and m.f1 == 0.0, m

    assert forgetting(0.95, 0.60) == 0.35
    assert _rough_sizeof({"a": [1, 2, 3]}) > 0

    print("metrics self-test: OK")


if __name__ == "__main__":
    _self_test()
