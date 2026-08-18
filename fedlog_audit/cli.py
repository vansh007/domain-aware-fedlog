"""
cli.py — command-line front-end for the federated-log-AD safety auditor.

Examples:
    fedlog-audit --example
    fedlog-audit --domain hdfs:0-33:4-300 --domain bgl:33-427:1-900 \
                 --arrival sequential --representation embedding --detector deep \
                 --order hdfs,bgl
"""

from __future__ import annotations
import argparse
import sys

from fedlog_audit.auditor import DomainSpec, audit_federation


def _parse_domain(spec: str) -> DomainSpec:
    """Parse 'name:idlo-idhi:lenmin-lenmax', e.g. 'hdfs:0-33:4-300'."""
    try:
        name, ids, lens = spec.split(":")
        ilo, ihi = (int(x) for x in ids.split("-"))
        lmin, lmax = (int(x) for x in lens.split("-"))
    except ValueError:
        raise SystemExit(f"bad --domain {spec!r}; expected name:idlo-idhi:lenmin-lenmax "
                         "(e.g. hdfs:0-33:4-300)")
    return DomainSpec(name, (ilo, ihi), (lmin, lmax))


def _example() -> int:
    """The canonical HDFS+BGL federation from the paper, in the failing configuration."""
    domains = [
        DomainSpec("hdfs", (0, 33), (4, 300)),
        DomainSpec("bgl", (33, 427), (1, 900)),
    ]
    print("### Lightweight detectors, mixed simultaneously (H1 axis):")
    print(audit_federation(domains, arrival="simultaneous", detector="lightweight").render())
    print("\n### Deep detector, SEQUENTIAL embedding onboarding hdfs->bgl (the failing cell):")
    print(audit_federation(domains, arrival="sequential", representation="embedding",
                           detector="deep", order=["hdfs", "bgl"]).render())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="fedlog-audit",
        description="Pre-deployment safety audit for heterogeneous/evolving federated log "
                    "anomaly detection. Uses summary statistics only (no raw logs).")
    ap.add_argument("--domain", action="append", default=[],
                    help="name:idlo-idhi:lenmin-lenmax (repeatable), e.g. hdfs:0-33:4-300")
    ap.add_argument("--arrival", choices=["simultaneous", "sequential"], default="simultaneous")
    ap.add_argument("--representation", choices=["scalar", "disjoint", "embedding", "shared"],
                    default="scalar")
    ap.add_argument("--detector", choices=["lightweight", "deep", "ensemble"],
                    default="lightweight")
    ap.add_argument("--order", default=None,
                    help="comma-separated arrival order for sequential (e.g. hdfs,bgl)")
    ap.add_argument("--example", action="store_true",
                    help="run the canonical HDFS+BGL example from the paper and exit")
    args = ap.parse_args(argv)

    if args.example:
        return _example()
    if not args.domain:
        ap.error("give at least one --domain (or use --example)")

    domains = [_parse_domain(s) for s in args.domain]
    order = args.order.split(",") if args.order else None
    report = audit_federation(domains, arrival=args.arrival, representation=args.representation,
                              detector=args.detector, order=order)
    print(report.render())
    # non-zero exit if any HIGH/CRITICAL risk, so it is usable as a CI gate
    return 1 if report.worst_level in ("HIGH", "CRITICAL") else 0


if __name__ == "__main__":
    sys.exit(main())
