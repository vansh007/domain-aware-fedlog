"""
aggregation.py — federated aggregation strategies.

Two families:
  1. Reference strategies (domain-blind): what the reference paper does. Included so our
     comparison is fair and our failure demonstration is honest.
  2. Domain-aware strategy (OUR contribution): the Week-4 research task. Provided as a
     structured stub with the intended design, not a pretend-finished method.

The whole paper hinges on the contrast:
  - Domain-blind Length aggregation: global min/max -> H1 failure.
  - Domain-aware Length aggregation: per-domain min/max, routed by the sequence's domain
    -> should restore single-domain behaviour without losing the federation.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---- 1. Reference (domain-blind) aggregation -----------------------------------------

def fedavg_length_global(local_ranges: list[tuple[int, int]]) -> tuple[int, int]:
    """Reference Length aggregation: one global [min, max] for all domains.

    This is the operation the reference paper calls distribution-invariant. It is — within
    one dataset. Across datasets it collapses (H1).
    """
    return min(lo for lo, _ in local_ranges), max(hi for _, hi in local_ranges)


def fedavg_union(local_sets: list[set[int]]) -> set[int]:
    """Reference Known-Events aggregation: global union. Never forgets, grows unbounded."""
    return set().union(*local_sets) if local_sets else set()


# ---- 2. Domain-aware aggregation (OUR contribution — STUB) ----------------------------

@dataclass
class DomainAwareLength:
    """Per-domain range parameters instead of one global range.

    Intended design (see docs/DECISIONS.md D5):
      - Keep a separate (min, max) per domain, aggregated only across clients of the SAME
        domain.
      - At inference, route a sequence to its domain's range using the domain tag that
        travels with every client / sequence (see src.dataloader.ClientData.domain).
      - Result: HDFS sequences are judged against HDFS's tight range, so HDFS anomalies
        stay detectable even though BGL is in the federation.

    This preserves federation (clients of a domain still collaborate) while removing the
    cross-domain contamination that causes H1.

    STATUS: The routing + fit are straightforward and implemented below. What remains as
    genuine research (Week 4): the harder case where a sequence's domain is unknown at
    inference (a real deployment may not tag domains), and how to detect/route then.
    That open question is intentionally left as TODO — it is a legitimate contribution to
    reason about in the paper, not something to fake now.
    """

    ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    def aggregate_per_domain(self, local_ranges_by_domain: dict[str, list[tuple[int, int]]]) -> None:
        for domain, ranges in local_ranges_by_domain.items():
            self.ranges[domain] = (
                min(lo for lo, _ in ranges),
                max(hi for _, hi in ranges),
            )

    def predict(self, sequences: list[list[int]], domains: list[str]) -> list[int]:
        """Route each sequence to its domain's range. Requires a domain tag per sequence."""
        if len(sequences) != len(domains):
            raise ValueError("need one domain tag per sequence")
        out = []
        for s, d in zip(sequences, domains):
            lo, hi = self.ranges[d]
            L = len(s)
            out.append(1 if (L < lo or L > hi) else 0)
        return out

    def predict_domain_unknown(self, sequences: list[list[int]]) -> list[int]:
        """The genuinely open Week-4 problem: no domain tag at inference.

        Options to explore (do NOT pick one silently — measure them):
          - infer domain from which vocabulary block the sequence's event IDs fall in
          - flag anomalous only if OUTSIDE every domain's range (conservative)
          - flag anomalous if outside the NEAREST domain's range (by event-ID overlap)
        Leave as TODO until Week 4; each option is an experiment, not a default.
        """
        raise NotImplementedError(
            "Domain-unknown routing is the open Week-4 research question. Implement and "
            "COMPARE the options in the docstring; do not hardcode one as 'the answer'."
        )
