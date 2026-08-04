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

    STATUS: The routing + fit are implemented below, as are TWO tag-free inference
    strategies for the case where a sequence's domain is unknown at inference (a real
    deployment may not tag domains): `predict_domain_unknown(..., strategy=...)` supports
    'vocab_block' (infer the domain from the event-id block) and 'conservative' (flag only
    if outside every domain's range). scripts/run_domain_routing.py evaluates oracle vs.
    both strategies and writes results/domain_routing.csv. For our disjoint HDFS/BGL
    vocabularies the vocab-block router recovers the domain exactly, so it equals oracle
    routing; the open case is OVERLAPPING vocabularies, left as future work.
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

    @staticmethod
    def _domain_of_id(gid: int, id_blocks: dict[str, tuple[int, int]]) -> str | None:
        """Which domain owns global event id `gid`? Blocks are half-open [start, end).
        Returns None if `gid` falls outside every known block (an unseen event id)."""
        for domain, (start, end) in id_blocks.items():
            if start <= gid < end:
                return domain
        return None

    def infer_domains(self, sequences: list[list[int]],
                      id_blocks: dict[str, tuple[int, int]]) -> list[str | None]:
        """Strategy 1 inference only (no anomaly decision): infer each sequence's domain
        from its event ids, using the vocabulary block structure.

        Rule (per the task spec):
          - map every event id to its owning domain block;
          - if any id is outside all known blocks -> domain is None ("unknown domain");
          - otherwise return the majority domain among the ids (handles a sequence whose
            ids span more than one block).
        An empty sequence (no ids) carries no evidence and is reported as None.

        This is deterministic and ML-free: the disjoint vocabulary blocks already encode
        domain identity, so we merely read it back off the ids.
        """
        out: list[str | None] = []
        for s in sequences:
            if not s:
                out.append(None)
                continue
            counts: dict[str, int] = {}
            unknown = False
            for gid in s:
                d = self._domain_of_id(gid, id_blocks)
                if d is None:
                    unknown = True
                    break
                counts[d] = counts.get(d, 0) + 1
            out.append(None if unknown else max(counts, key=counts.get))
        return out

    def predict_domain_unknown(self, sequences: list[list[int]], strategy: str,
                               id_blocks: dict[str, tuple[int, int]] | None = None,
                               ) -> list[int]:
        """Anomaly prediction WITHOUT a manual domain tag at inference.

        Two tag-free strategies (compared, not silently chosen — see
        scripts/run_domain_routing.py):

        strategy='vocab_block' (Strategy 1): infer the domain from the sequence's event-id
            block (via `infer_domains`), then judge the sequence against THAT domain's
            learned [min, max] range. A sequence whose domain cannot be inferred (an event
            id outside every known block) is flagged anomalous — an unseen event is itself
            evidence of anomaly. Requires `id_blocks`.

        strategy='conservative' (Strategy 2): no domain inference at all. Flag a sequence
            anomalous only if its length is outside EVERY domain's range; if it fits any
            domain's range it is called normal. Simpler, but cannot catch an anomaly whose
            length happens to be normal for some OTHER domain.

        Returns a 0/1 list (1 = anomaly), aligned with `sequences`.
        """
        if strategy == "vocab_block":
            if id_blocks is None:
                raise ValueError("strategy='vocab_block' needs id_blocks (domain -> [start,end))")
            inferred = self.infer_domains(sequences, id_blocks)
            out = []
            for s, d in zip(sequences, inferred):
                if d is None:                       # unknown domain -> anomalous
                    out.append(1)
                    continue
                lo, hi = self.ranges[d]
                L = len(s)
                out.append(1 if (L < lo or L > hi) else 0)
            return out

        if strategy == "conservative":
            out = []
            for s in sequences:
                L = len(s)
                # anomalous iff outside every domain's range (inside ANY range -> normal)
                outside_all = all(L < lo or L > hi for (lo, hi) in self.ranges.values())
                out.append(1 if outside_all else 0)
            return out

        raise ValueError(f"unknown strategy {strategy!r}; use 'vocab_block' or 'conservative'")
