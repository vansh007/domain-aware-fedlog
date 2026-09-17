"""
auditor.py — a pre-deployment SAFETY AUDITOR for heterogeneous / evolving federated
log anomaly detection.

This is the productized form of the paper "Domain-Aware Federated Log Anomaly Detection
under Heterogeneous and Sequentially Arriving Domains". Every rule below encodes a MEASURED
finding (each `evidence` string points at the result it came from); the auditor does not run
any model. It only needs each client/domain's SUMMARY STATISTICS:

  - the global event-id block the domain owns (its vocabulary offset+size), and
  - the [min, max] length of its NORMAL sequences.

That is deliberately privacy-preserving: it is exactly the aggregate a federated client can
share without exposing raw logs, so a coordinator can audit a planned federation before any
training happens, and get told which of the paper's failure modes it will hit and the fix.

Pure Python, no third-party dependencies. See fedlog_audit.cli for the command-line form.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------
@dataclass
class DomainSpec:
    """Summary statistics for one log domain in the federation.

    id_range: the half-open global event-id block [start, end) this domain owns (its
              vocabulary lives here). end-start is the domain's vocabulary size.
    normal_length_range: [min, max] sequence length over the domain's NORMAL sequences.
    """
    name: str
    id_range: tuple[int, int]
    normal_length_range: tuple[int, int]

    @property
    def vocab_size(self) -> int:
        return self.id_range[1] - self.id_range[0]

    @property
    def min_len(self) -> int:
        return self.normal_length_range[0]

    @property
    def max_len(self) -> int:
        return self.normal_length_range[1]


# ------------------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------------------
_ORDER = {"SAFE": 0, "INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}


@dataclass
class Risk:
    name: str
    level: str                 # SAFE | INFO | LOW | MEDIUM | HIGH | CRITICAL
    finding: str
    recommendation: str
    evidence: str = ""

    @property
    def rank(self) -> int:
        return _ORDER.get(self.level, 0)


@dataclass
class AuditReport:
    domains: list[DomainSpec]
    arrival: str
    representation: str
    detector: str
    order: list[str]
    risks: list[Risk] = field(default_factory=list)
    cell: str = ""

    @property
    def worst_level(self) -> str:
        return max((r.level for r in self.risks), key=lambda l: _ORDER[l], default="SAFE")

    def render(self) -> str:
        icon = {"SAFE": "OK ", "INFO": "i  ", "LOW": "·  ", "MEDIUM": "!  ",
                "HIGH": "!! ", "CRITICAL": "XX "}
        lines = []
        lines.append("=" * 74)
        lines.append("FEDERATED LOG ANOMALY DETECTION — PRE-DEPLOYMENT SAFETY AUDIT")
        lines.append("=" * 74)
        doms = ", ".join(f"{d.name}(vocab={d.vocab_size}, len[{d.min_len},{d.max_len}])"
                         for d in self.domains)
        lines.append(f"domains      : {doms}")
        lines.append(f"arrival      : {self.arrival}"
                     + (f"  order={' -> '.join(self.order)}" if self.order else ""))
        lines.append(f"representation: {self.representation}")
        lines.append(f"detector     : {self.detector}")
        lines.append(f"2x2 map cell : {self.cell}")
        lines.append(f"overall      : {icon[self.worst_level].strip()} {self.worst_level}")
        lines.append("-" * 74)
        for r in sorted(self.risks, key=lambda r: -r.rank):
            lines.append(f"[{icon[r.level]}{r.level:8s}] {r.name}")
            lines.append(f"    finding: {r.finding}")
            lines.append(f"    fix    : {r.recommendation}")
            if r.evidence:
                lines.append(f"    basis  : {r.evidence}")
        lines.append("=" * 74)
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# Individual audits (each encodes one measured finding)
# ------------------------------------------------------------------------------------------
def audit_length_aggregation(domains: list[DomainSpec], detector: str) -> Risk:
    """H1: domain-blind min-max Length aggregation collapses on tighter domains, because a
    wide-spread domain widens the shared global range until narrow domains' anomalies fit
    inside it. Only relevant when a range/threshold statistic is aggregated globally."""
    if detector == "deep":
        return Risk("Range aggregation (Length)", "SAFE",
                    "Deep detector does not use a global length range.", "n/a")

    gmin = min(d.min_len for d in domains)
    gmax = max(d.max_len for d in domains)
    victims = [d for d in domains if gmin < d.min_len or gmax > d.max_len]
    if not victims:
        return Risk("Range aggregation (Length)", "SAFE",
                    "All domains share the same length range; global range adds no blind spot.",
                    "Use per-domain ranges anyway for robustness.")

    # severity = how much wider the global range is than the worst victim's own range
    def ratio(d):
        own = max(1, d.max_len - d.min_len)
        return (gmax - gmin) / own
    worst = max(victims, key=ratio)
    r = ratio(worst)
    level = "CRITICAL" if r >= 5 else "HIGH" if r >= 2 else "MEDIUM"
    if detector == "ensemble":
        # the union ensemble is carried by ECVC, so a collapsed Length does not lower recall
        level = "LOW"
        note = (" (In a full Events∨Length∨ECVC ensemble this is masked by ECVC and the "
                "ensemble F1 is unchanged — but any lightweight-only Length/Events deployment "
                "collapses.)")
    else:
        note = ""
    names = ", ".join(d.name for d in victims)
    return Risk(
        "Range aggregation (Length)", level,
        f"Global length range [{gmin},{gmax}] is ~{r:.1f}x wider than '{worst.name}''s own "
        f"range [{worst.min_len},{worst.max_len}]; anomalies of {{{names}}} fall inside the "
        f"global band and are missed." + note,
        "Aggregate a SEPARATE [min,max] per domain (domain-aware aggregation) and route each "
        "sequence to its domain's range.",
        "H1: HDFS Length F1 0.538 -> 0.000 (domain-blind) -> 0.561 (domain-aware), 5 seeds.")


def audit_routing(domains: list[DomainSpec]) -> Risk:
    """Whether the domain-aware fix can route WITHOUT a manual domain tag at inference:
    if the id blocks are disjoint, the event id itself identifies the domain."""
    blocks = sorted(((d.id_range[0], d.id_range[1], d.name) for d in domains))
    overlaps = []
    for (a_lo, a_hi, a_name), (b_lo, b_hi, b_name) in zip(blocks, blocks[1:]):
        if b_lo < a_hi:
            overlaps.append(f"{a_name}∩{b_name}")
    if not overlaps:
        return Risk("Tag-free routing", "SAFE",
                    "Domain id-blocks are disjoint, so vocabulary-block routing recovers each "
                    "sequence's domain exactly — no manual domain tag is needed at inference.",
                    "Use vocab-block routing (majority event-id block); flag ids outside all "
                    "blocks as anomalous.",
                    "Routing: vocab-block == oracle exactly, accuracy 1.000, 5 seeds.")
    return Risk("Tag-free routing", "MEDIUM",
                f"Domain vocabularies OVERLAP ({', '.join(overlaps)}); the event id no longer "
                "identifies the domain uniquely.",
                "Provide a domain tag, or resolve overlap; ML-free block routing is exact only "
                "for disjoint vocabularies.",
                "Overlapping vocabularies are the paper's stated open/future-work case.")


def audit_bounded_size(domains: list[DomainSpec], arrival: str, detector: str) -> Risk:
    """C2: set-union (Known-Events) never forgets but grows with every new vocabulary, so its
    state is unbounded on an open stream of domains."""
    if detector == "deep":
        return Risk("Bounded model size", "SAFE",
                    "Deep model is fixed-size; it does not grow as domains arrive.", "n/a")
    total_vocab = sum(d.vocab_size for d in domains)
    if arrival == "simultaneous":
        return Risk("Bounded model size", "INFO",
                    f"Set-union state holds the union of all vocabularies ({total_vocab} events). "
                    "Fixed here because the domain set is fixed.",
                    "Fine for a fixed federation; watch growth if domains will be added later.",
                    "H3: set-union grows x3.7 (1,688 -> 6,300 bytes) when a domain is added.")
    return Risk("Bounded model size", "MEDIUM",
                f"Under sequential/evolving arrival the set-union (Known-Events) state grows with "
                f"each new vocabulary (currently {total_vocab} events) and is UNBOUNDED on an open "
                "stream of domains — contradicting the bounded-size property.",
                "Cap/prune the event set, or use a fixed-size model, if domains keep arriving.",
                "H3: set-union grew x3.7 (1,688 -> 6,300 bytes) for one added domain.")


def audit_forgetting(domains, arrival, representation, detector, order) -> Risk:
    """H3 + mechanism + directionality: a deep model catastrophically forgets an earlier domain
    under sequential arrival ONLY when (a) the input representation makes vocabularies share a
    feature space (embedding), and (b) a BROAD-vocabulary domain is onboarded after a narrow one."""
    if detector != "deep":
        return Risk("Catastrophic forgetting", "SAFE",
                    "Lightweight set/range detectors do not forget (aggregation is monotone).",
                    "n/a")
    if arrival == "simultaneous":
        return Risk("Catastrophic forgetting", "SAFE",
                    "Simultaneous mixed FedAvg does not degrade the deep model; every round sees "
                    "all domains.",
                    "Prefer simultaneous onboarding when possible.",
                    "H2: HDFS drop +0.0002; BGL even improves; holds under embeddings too.")
    if representation in ("scalar", "disjoint"):
        return Risk("Catastrophic forgetting", "SAFE",
                    "Scalar-id input keeps domains in disjoint input ranges, so continued training "
                    "on a new domain barely perturbs the earlier one.",
                    "Safe as-is; note this is what masks the risk in the reference's encoding.",
                    "H3 (scalar): forgetting 0.000.")

    # embedding / shared representation + sequential -> check DIRECTION
    order = order or [d.name for d in domains]
    by_name = {d.name: d for d in domains}
    broad_after_narrow = []
    for i, later in enumerate(order):
        for earlier in order[:i]:
            if later in by_name and earlier in by_name:
                if by_name[later].vocab_size > by_name[earlier].vocab_size:
                    broad_after_narrow.append((earlier, later))
    if broad_after_narrow:
        e, l = broad_after_narrow[0]
        return Risk("Catastrophic forgetting", "CRITICAL",
                    f"Deep model with a shared (embedding) representation, sequential arrival, and a "
                    f"BROAD domain '{l}' (vocab {by_name[l].vocab_size}) onboarded after the NARROW "
                    f"'{e}' (vocab {by_name[e].vocab_size}): '{e}' will be catastrophically forgotten.",
                    "Budget a replay buffer of >=10% of the earlier domain (tuning-free), OR use EWC "
                    "(memory-free but needs a large, carefully-tuned penalty).",
                    "H3 (embedding): forgetting +0.665, HDFS 0.70->0.04. Replay 10% -> +0.004; "
                    "EWC works only at lambda=1e8. Architecture-general: an LSTM (DeepLog) and a "
                    "Transformer both forget under sequential arrival.")
    return Risk("Catastrophic forgetting", "LOW",
                "Deep model with a shared representation under sequential arrival, but domains are "
                "onboarded narrow-after-broad; forgetting is directional and this direction is safe.",
                "Low risk, but keep a small replay buffer as insurance if ordering may change.",
                "Reverse order (broad-first): forgetting -0.103 (earlier domain not forgotten).")


# ------------------------------------------------------------------------------------------
# Top-level entry point
# ------------------------------------------------------------------------------------------
def audit_federation(domains: list[DomainSpec], arrival: str = "simultaneous",
                     representation: str = "scalar", detector: str = "lightweight",
                     order: list[str] | None = None) -> AuditReport:
    """Audit a planned federation from summary statistics only.

    arrival        : 'simultaneous' | 'sequential'
    representation : 'scalar'/'disjoint' | 'embedding'/'shared'   (deep detector only)
    detector       : 'lightweight' | 'deep' | 'ensemble'
    order          : arrival order (list of domain names) for sequential; defaults to given order
    """
    if len(domains) < 1:
        raise ValueError("need at least one domain")
    arrival = arrival.lower()
    representation = representation.lower()
    detector = detector.lower()
    if arrival not in ("simultaneous", "sequential"):
        raise ValueError("arrival must be 'simultaneous' or 'sequential'")

    risks = [
        audit_length_aggregation(domains, detector),
        audit_routing(domains),
        audit_bounded_size(domains, arrival, detector),
        audit_forgetting(domains, arrival, representation, detector, order),
    ]

    # 2x2 map cell (deep-model axis)
    if detector == "deep":
        shared = representation in ("embedding", "shared")
        seq = arrival == "sequential"
        if seq and shared:
            cell = "(sequential, shared representation) — the one FAILING cell"
        else:
            cell = f"({arrival}, {'shared' if shared else 'disjoint'} representation) — safe"
    else:
        cell = "lightweight family (Length collapse / set-union growth axis)"

    return AuditReport(domains=domains, arrival=arrival, representation=representation,
                       detector=detector, order=order or [d.name for d in domains],
                       risks=risks, cell=cell)
