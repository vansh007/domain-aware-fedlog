"""
fedlog_audit — a safety auditor + deployable fixes for heterogeneous / evolving
federated log anomaly detection.

Productizes the paper "Domain-Aware Federated Log Anomaly Detection under Heterogeneous
and Sequentially Arriving Domains". Two layers:

  1. The AUDITOR (pure Python, no deps): from summary statistics only, tell a planned
     federation which of the paper's failure modes it will hit and prescribe the fix.

         from fedlog_audit import DomainSpec, audit_federation
         report = audit_federation(
             [DomainSpec("hdfs", (0, 33), (4, 300)),
              DomainSpec("bgl", (33, 427), (1, 900))],
             arrival="sequential", representation="embedding", detector="deep",
             order=["hdfs", "bgl"])
         print(report.render())

  2. The FIX IMPLEMENTATIONS (re-exported from the research code in `src/`, if importable):
     the domain-aware aggregation + vocab-block router, and the lightweight detectors, so a
     user can act on the audit. These need the research package on the path.
"""

from __future__ import annotations

from fedlog_audit.auditor import (
    DomainSpec,
    Risk,
    AuditReport,
    audit_federation,
    audit_length_aggregation,
    audit_routing,
    audit_bounded_size,
    audit_forgetting,
)

__all__ = [
    "DomainSpec", "Risk", "AuditReport", "audit_federation",
    "audit_length_aggregation", "audit_routing", "audit_bounded_size", "audit_forgetting",
]

# Optional convenience: re-export the actual fix implementations from the research code.
# Kept in a try/except so the auditor works standalone even without the research package.
try:  # pragma: no cover - depends on install layout
    from src.aggregation import DomainAwareLength, fedavg_length_global, fedavg_union
    from src.methods.length import LengthDetector
    from src.methods.known_events import KnownEventsDetector
    from src.vocabulary import GlobalVocabulary
    __all__ += ["DomainAwareLength", "fedavg_length_global", "fedavg_union",
                "LengthDetector", "KnownEventsDetector", "GlobalVocabulary"]
    HAS_RESEARCH_CODE = True
except Exception:  # noqa: BLE001
    HAS_RESEARCH_CODE = False

__version__ = "0.1.0"
