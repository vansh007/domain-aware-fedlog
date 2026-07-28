"""
vocabulary.py — merged global event vocabulary across log domains.

This is the enabling piece for the whole project. The reference paper builds a global
template dictionary across *clients of one dataset* (their pipeline step 3). We build one
across *datasets*, so an HDFS client and a BGL client can live in the same federation.

Design (see docs/DECISIONS.md D3):
  - Each domain contributes its own set of parsed templates (strings).
  - We assign each (domain, template) a unique global integer ID by offsetting:
    HDFS templates -> 0..(n_hdfs-1); BGL templates -> n_hdfs..(n_hdfs+n_bgl-1); etc.
  - A `domain` label travels with every sequence so we can (a) measure per-domain F1
    and (b) let domain-aware aggregation route correctly.

Templates are dataset-specific strings, so cross-domain collisions do not occur; if two
domains ever produced a byte-identical template we would still keep them distinct because
the key is (domain, template), not template alone.

This module has no heavy dependencies and is fully testable on its own — run:
    python -m src.vocabulary
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class GlobalVocabulary:
    """Maps (domain, template) -> global event id, and back."""

    # (domain, template) -> global id
    _fwd: dict[tuple[str, str], int] = field(default_factory=dict)
    # global id -> (domain, template)
    _rev: dict[int, tuple[str, str]] = field(default_factory=dict)
    # domain -> (start_id, end_id_exclusive) — the contiguous block owned by a domain
    _domain_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add_domain(self, domain: str, templates: Iterable[str]) -> None:
        """Register all templates for a domain as one contiguous ID block.

        Call once per domain, in the order domains 'arrive'. Idempotent per domain:
        re-adding a domain that already exists raises, to avoid silent ID drift.
        """
        if domain in self._domain_ranges:
            raise ValueError(f"domain {domain!r} already added; vocabulary is append-only")

        templates = list(dict.fromkeys(templates))  # de-dup, preserve order
        start = len(self._fwd)
        for i, tmpl in enumerate(templates):
            gid = start + i
            key = (domain, tmpl)
            self._fwd[key] = gid
            self._rev[gid] = key
        end = len(self._fwd)
        self._domain_ranges[domain] = (start, end)

    def id_of(self, domain: str, template: str) -> int:
        return self._fwd[(domain, template)]

    def lookup(self, gid: int) -> tuple[str, str]:
        """Return (domain, template) for a global id."""
        return self._rev[gid]

    def domain_of(self, gid: int) -> str:
        return self._rev[gid][0]

    def domain_range(self, domain: str) -> tuple[int, int]:
        return self._domain_ranges[domain]

    @property
    def size(self) -> int:
        return len(self._fwd)

    @property
    def domains(self) -> list[str]:
        return list(self._domain_ranges.keys())

    def encode_sequence(self, domain: str, templates: list[str]) -> list[int]:
        """Turn a list of templates (one log line each) into global event ids."""
        return [self.id_of(domain, t) for t in templates]

    def summary(self) -> str:
        parts = [f"GlobalVocabulary(size={self.size})"]
        for d, (s, e) in self._domain_ranges.items():
            parts.append(f"  {d:12s} ids [{s:4d}, {e:4d})  ({e - s} templates)")
        return "\n".join(parts)


def _self_test() -> None:
    """Tiny sanity check mirroring the real HDFS(33)/BGL(394) setup shape."""
    vocab = GlobalVocabulary()
    vocab.add_domain("hdfs", [f"hdfs_tmpl_{i}" for i in range(33)])
    vocab.add_domain("bgl", [f"bgl_tmpl_{i}" for i in range(394)])

    assert vocab.size == 33 + 394
    assert vocab.domain_range("hdfs") == (0, 33)
    assert vocab.domain_range("bgl") == (33, 427)
    assert vocab.id_of("hdfs", "hdfs_tmpl_0") == 0
    assert vocab.id_of("bgl", "bgl_tmpl_0") == 33
    assert vocab.domain_of(0) == "hdfs"
    assert vocab.domain_of(33) == "bgl"

    seq = vocab.encode_sequence("hdfs", ["hdfs_tmpl_1", "hdfs_tmpl_2", "hdfs_tmpl_1"])
    assert seq == [1, 2, 1]

    print(vocab.summary())
    print("\nvocabulary self-test: OK")


if __name__ == "__main__":
    _self_test()
