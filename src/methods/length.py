"""
length.py — Length Detection method (the H1 target).

Learns the [min, max] length of normal sequences. At inference, a sequence whose length
falls outside that range is flagged anomalous.

Why this is the H1 target:
  Federated aggregation takes global_min = min(local mins), global_max = max(local maxes).
  HDFS normal length ~29±6; BGL ~69±747. In a mixed federation the global range becomes
  roughly [min over both, max over both] — wide enough that HDFS anomalies (which are
  detectable *because* their length is unusual for HDFS) now sit inside the accepted band
  and are missed. We predict Length F1 on HDFS collapses in the mixed setting.

This implementation is deliberately faithful to the reference method so the comparison is
fair. The *domain-aware* fix (per-domain ranges) lives in src/aggregation.py.

Run `python -m src.methods.length` for a self-test that demonstrates the H1 failure.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LengthDetector:
    """Global (domain-blind) length detector — matches the reference method."""

    min_len: int | None = None
    max_len: int | None = None

    def fit_local(self, normal_sequences: list[list[int]]) -> tuple[int, int]:
        """Compute this client's local (min, max) length over its NORMAL sequences."""
        if not normal_sequences:
            raise ValueError("no sequences to fit on")
        lengths = [len(s) for s in normal_sequences]
        return min(lengths), max(lengths)

    def aggregate(self, local_ranges: list[tuple[int, int]]) -> None:
        """Server-side federated aggregation: global min of mins, max of maxes.

        This is the exact operation the reference paper calls 'independent of client
        data distributions'. It is independent *within one dataset*. Across datasets it
        is dominated by the most variable domain — that's the point of H1.
        """
        self.min_len = min(lo for lo, _ in local_ranges)
        self.max_len = max(hi for _, hi in local_ranges)

    def predict(self, sequences: list[list[int]]) -> list[int]:
        if self.min_len is None or self.max_len is None:
            raise RuntimeError("call aggregate() before predict()")
        out = []
        for s in sequences:
            L = len(s)
            out.append(1 if (L < self.min_len or L > self.max_len) else 0)
        return out

    def state_size_bytes(self) -> int:
        # two ints; trivially bounded — Length is not the growth problem (that's H3/set-union)
        return 2 * 8


def _self_test() -> None:
    from src.metrics import prf

    # Simulate HDFS: normal lengths tightly around 29; anomalies are longer.
    hdfs_normal = [[0] * n for n in (24, 26, 29, 31, 34)]           # ~29±6
    hdfs_test_seqs = [[0] * n for n in (28, 30, 55, 60)]           # last two are anomalies
    hdfs_test_labels = [0, 0, 1, 1]

    # Single-domain HDFS: Length works reasonably — anomalies are out of range.
    det_single = LengthDetector()
    lo, hi = det_single.fit_local(hdfs_normal)
    det_single.aggregate([(lo, hi)])
    single_pred = det_single.predict(hdfs_test_seqs)
    single_f1 = prf(hdfs_test_labels, single_pred).f1

    # Mixed federation: add a BGL client with huge max length (747 tail).
    bgl_normal = [[0] * n for n in (10, 40, 69, 200, 800)]         # ~69±747
    det_mixed = LengthDetector()
    hdfs_range = det_mixed.fit_local(hdfs_normal)
    bgl_range = det_mixed.fit_local(bgl_normal)
    det_mixed.aggregate([hdfs_range, bgl_range])                   # global range now [10, 800]
    mixed_pred = det_mixed.predict(hdfs_test_seqs)
    mixed_f1 = prf(hdfs_test_labels, mixed_pred).f1

    print(f"HDFS Length F1 single-domain: {single_f1:.3f}  (range {hdfs_range})")
    print(f"HDFS Length F1 mixed w/ BGL : {mixed_f1:.3f}  (global range [{det_mixed.min_len}, {det_mixed.max_len}])")
    print("H1 illustration: the HDFS anomalies at length 55/60 now sit inside the")
    print("global range [10, 800], so they are no longer flagged. F1 drops.")
    assert mixed_f1 < single_f1, "expected mixed-domain F1 to be worse (H1)"
    print("\nlength self-test: OK (demonstrates the H1 mechanism on toy data)")
    print("NOTE: this is illustrative toy data, NOT a result. Real numbers come from runs.")


if __name__ == "__main__":
    _self_test()
