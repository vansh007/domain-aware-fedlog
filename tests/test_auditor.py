"""
test_auditor.py — checks the fedlog_audit safety auditor encodes the paper's findings.

Run:  python tests/test_auditor.py     (or: pytest tests/test_auditor.py)
Each assertion ties to a measured result; the auditor is pure logic, so these are fast.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedlog_audit import DomainSpec, audit_federation

HDFS = DomainSpec("hdfs", (0, 33), (4, 300))
BGL = DomainSpec("bgl", (33, 427), (1, 900))


def _risk(report, name):
    return next(r for r in report.risks if r.name == name)


def test_h1_length_collapse_flagged():
    """H1: mixed lightweight Length aggregation is flagged HIGH (broad BGL widens the range)."""
    rep = audit_federation([HDFS, BGL], arrival="simultaneous", detector="lightweight")
    assert _risk(rep, "Range aggregation (Length)").level == "HIGH"


def test_h1_ensemble_is_scoped_down():
    """Ensemble is carried by ECVC, so a collapsed Length is only LOW there (honest scoping)."""
    rep = audit_federation([HDFS, BGL], arrival="simultaneous", detector="ensemble")
    assert _risk(rep, "Range aggregation (Length)").level == "LOW"


def test_forgetting_is_directional():
    """Mechanism + directionality: forward (broad-after-narrow) is CRITICAL; reverse is LOW."""
    fwd = audit_federation([HDFS, BGL], arrival="sequential", representation="embedding",
                           detector="deep", order=["hdfs", "bgl"])
    rev = audit_federation([HDFS, BGL], arrival="sequential", representation="embedding",
                           detector="deep", order=["bgl", "hdfs"])
    assert _risk(fwd, "Catastrophic forgetting").level == "CRITICAL"
    assert _risk(rev, "Catastrophic forgetting").level == "LOW"


def test_scalar_and_simultaneous_are_safe():
    """H2 + scalar mechanism: simultaneous is safe; scalar sequential is safe."""
    scalar = audit_federation([HDFS, BGL], arrival="sequential", representation="scalar",
                              detector="deep", order=["hdfs", "bgl"])
    simul = audit_federation([HDFS, BGL], arrival="simultaneous", representation="embedding",
                             detector="deep")
    assert _risk(scalar, "Catastrophic forgetting").level == "SAFE"
    assert _risk(simul, "Catastrophic forgetting").level == "SAFE"


def test_disjoint_routing_is_tag_free():
    """Disjoint id-blocks -> vocab-block routing is exact (no oracle tag)."""
    rep = audit_federation([HDFS, BGL])
    assert _risk(rep, "Tag-free routing").level == "SAFE"


def test_overlapping_vocab_warns():
    """Overlapping vocabularies -> routing cannot be inferred exactly (open case)."""
    a = DomainSpec("a", (0, 50), (4, 300))
    b = DomainSpec("b", (30, 80), (1, 900))
    rep = audit_federation([a, b])
    assert _risk(rep, "Tag-free routing").level == "MEDIUM"


def test_setunion_growth_under_evolution():
    """C2: set-union growth is flagged only under sequential/evolving arrival."""
    fixed = audit_federation([HDFS, BGL], arrival="simultaneous", detector="lightweight")
    evolving = audit_federation([HDFS, BGL], arrival="sequential", detector="lightweight")
    assert _risk(fixed, "Bounded model size").level == "INFO"
    assert _risk(evolving, "Bounded model size").level == "MEDIUM"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nauditor tests: OK ({len(fns)} passed)")


if __name__ == "__main__":
    _run_all()
