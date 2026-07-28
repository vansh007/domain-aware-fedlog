"""
deeplog_stub.py — thin boundary to the reference DeepLog implementation.

We do NOT reimplement DeepLog. The reference repo (GPL-3.0) already has a validated
DeepLog/LogAnomaly under FedAvg. To keep licenses clean (see docs/DECISIONS.md D2) we
call their code rather than copying it.

This file is intentionally a STUB. Week 1's job is to fill `train_federated` and
`evaluate` by wiring to the cloned reference repo. Until then, calling these raises,
loudly, so nobody mistakes a stub for a result.

H2 lives here: DeepLog under FedAvg averages weights across clients. In a mixed
HDFS+BGL federation the input event-ID spaces are largely disjoint (HDFS 0..32,
BGL 33..426), so weight averaging blends two models that were learning different
vocabularies. We predict per-domain F1 drops versus single-domain training.
"""

from __future__ import annotations
from dataclasses import dataclass


class NotWiredYet(RuntimeError):
    """Raised by stubbed methods so a stub can never be mistaken for a real result."""


@dataclass
class DeepLogFederated:
    reference_repo_path: str            # path to cloned comparison-fed-centr-efficient-ad
    num_candidates: int = 9             # HDFS top-g; BGL uses ~60 (set per config)
    window_size: int = 10               # HDFS=10, BGL=2 in the reference paper
    device: str = "cpu"                 # HARD RULE: CPU-first

    def train_federated(self, clients, rounds: int):
        """Wire to the reference repo's federated DeepLog.

        Expected shape once wired (Week 1):
          - build per-client event-sequence datasets from our dataloader
          - hand them to the reference FedAvg loop (their ml_federated_simulation.py)
          - return the trained global model object

        Reference command for orientation (single dataset):
          python ml_federated_simulation.py \\
            --config config_files/hdfs_iid.yaml --num_clients 2 --model deeplog --device cpu
        """
        raise NotWiredYet(
            "DeepLogFederated.train_federated is a stub. Week 1: wire to the cloned "
            "reference repo at self.reference_repo_path. Do NOT return a fake model."
        )

    def evaluate(self, test_sequences, labels) -> dict:
        raise NotWiredYet(
            "DeepLogFederated.evaluate is a stub. Wire to the reference evaluation, then "
            "return real precision/recall/f1 via src.metrics.prf."
        )

    def state_size_bytes(self) -> int:
        # DL model is fixed-size (2 layers x 64 units) regardless of #domains — that's the
        # H3 contrast against set-union growth. Once wired, sum parameter buffers.
        raise NotWiredYet("wire to the trained torch module, then sum parameter bytes")
