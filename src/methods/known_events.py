"""
known_events.py — Known Events method (set-union based).

Learns the set of event IDs seen in normal sequences. At inference, any sequence
containing an event ID not in that set is flagged anomalous.

Federated aggregation = union of local event sets. This method:
  - NEVER forgets: union is monotone, so adding a domain only grows the known set (good).
  - GROWS with vocabulary: the set expands with every new domain's events. On an open set
    of arriving domains there is no saturation point — this is the H3 growth mechanism for
    set-union methods.

Because BGL anomalies often contain event types absent from normal training data, Known
Events is strong on BGL in the reference paper. That's expected and fine.

Run `python -m src.methods.known_events` for a self-test.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KnownEventsDetector:
    known: set[int] = field(default_factory=set)

    def fit_local(self, normal_sequences: list[list[int]]) -> set[int]:
        local = set()
        for s in normal_sequences:
            local.update(s)
        return local

    def aggregate(self, local_sets: list[set[int]]) -> None:
        """Federated union. Monotone => never forgets, but grows unbounded across domains."""
        self.known = set().union(*local_sets) if local_sets else set()

    def predict(self, sequences: list[list[int]]) -> list[int]:
        out = []
        for s in sequences:
            out.append(0 if all(e in self.known for e in s) else 1)
        return out

    def state_size_bytes(self) -> int:
        # grows with |known| — this is what the H3 size-growth curve tracks
        import sys
        return sys.getsizeof(self.known) + sum(sys.getsizeof(e) for e in self.known)


def _self_test() -> None:
    from src.metrics import prf

    # Normal HDFS events are ids 0..10; an anomaly introduces an unseen id.
    normal = [[0, 1, 2], [1, 2, 3], [0, 3, 4]]
    test = [[0, 1, 2], [2, 3, 4], [0, 1, 99]]   # last has unseen event 99 -> anomaly
    labels = [0, 0, 1]

    det = KnownEventsDetector()
    local = det.fit_local(normal)
    det.aggregate([local])
    pred = det.predict(test)
    assert prf(labels, pred).f1 == 1.0, pred

    # size grows when a second domain's events are unioned in
    size_before = det.state_size_bytes()
    det.aggregate([local, {33, 34, 35, 36}])   # BGL block ids
    size_after = det.state_size_bytes()
    assert size_after > size_before, "set-union should grow with a new domain (H3)"

    print(f"known set size before 2nd domain: {size_before} bytes")
    print(f"known set size after  2nd domain: {size_after} bytes")
    print("known_events self-test: OK")


if __name__ == "__main__":
    _self_test()
