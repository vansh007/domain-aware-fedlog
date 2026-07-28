"""
dataloader.py — the heart of the contribution.

The reference dataloader assumes ONE dataset per run. To build a mixed federation we
must load multiple domains, encode them into a single merged vocabulary, and hand each
'client' its slice — while remembering which domain each client belongs to.

This file provides:
  - a working orchestration skeleton (domain loading -> merged vocab -> client split)
  - the merged-vocabulary wiring (uses src.vocabulary, already tested)
  - client-partitioning for both IID and quantity-skew (log-normal) splits
  - a clearly marked TODO where real parsed data is read from disk

What is DONE: the merge + split logic (the genuinely new part).
What is a TODO: `_load_parsed_domain` — reading actual parsed sequences from LogHub /
the reference repo's preprocessed output. That is dataset plumbing, best done in Week 1
against the real files, not guessed at now.

Run `python -m src.dataloader --demo` to see the split logic on synthetic data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import argparse
import csv
import math
import os
import random
import sys

from src.vocabulary import GlobalVocabulary


@dataclass
class ClientData:
    client_id: int
    domain: str                          # 'hdfs' or 'bgl' — travels with the client (D3)
    normal_train: list[list[int]]        # encoded normal sequences for local training
    test_sequences: list[list[int]]      # encoded eval sequences
    test_labels: list[int]               # 1 = anomaly, 0 = normal


@dataclass
class Federation:
    vocab: GlobalVocabulary
    clients: list[ClientData]

    def clients_of(self, domain: str) -> list[ClientData]:
        return [c for c in self.clients if c.domain == domain]

    def summary(self) -> str:
        by_dom: dict[str, int] = {}
        for c in self.clients:
            by_dom[c.domain] = by_dom.get(c.domain, 0) + 1
        lines = [f"Federation: {len(self.clients)} clients, vocab size {self.vocab.size}"]
        for d, n in by_dom.items():
            lines.append(f"  {d}: {n} clients")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The part that must be wired to real data in Week 1.
# --------------------------------------------------------------------------------------
def _load_parsed_domain(domain: str, data_root: str) -> tuple[list[str], list[tuple[list[str], int]]]:
    """Return (templates, examples) for a domain, read from the reference repo's CSVs.

    templates : the domain's ordered distinct event strings (for the vocabulary block)
    examples  : list of (sequence_of_templates, label) where label 1 = anomaly

    Data layout (confirmed against reference_repo/datasets/):
        {data_root}/{DOMAIN}/normal.csv    -> label 0
        {data_root}/{DOMAIN}/abnormal.csv  -> label 1
    where DOMAIN is the uppercased domain name (HDFS, BGL). Each CSV has header
    `ID,Event_seq`; column 1 is a block/node id (ignored) and column 2 is a
    space-separated string of integer event ids, e.g. "5 5 22 11 9 26".

    Event-id collision: HDFS and BGL both reuse small integers ("5", "6", "9", ...) for
    *different* templates. To keep them distinct in the merged vocabulary each event is
    represented as f"{domain}:{int_id}" (so HDFS "5" -> "hdfs:5", BGL "5" -> "bgl:5").

    Reads only what is on disk — no synthetic rows are ever produced here.
    """
    domain_dir = os.path.join(data_root, domain.upper())
    files = [
        (os.path.join(domain_dir, "normal.csv"), 0),
        (os.path.join(domain_dir, "abnormal.csv"), 1),
    ]
    for path, _ in files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"expected data file not found: {path!r}. Layout must be "
                f"{{data_root}}/{{DOMAIN}}/normal.csv and abnormal.csv (DOMAIN uppercase)."
            )

    # A single sequence field can be long (BGL length std is large); lift csv's cap.
    _ensure_csv_field_limit()

    examples: list[tuple[list[str], int]] = []
    seen: dict[str, None] = {}  # ordered set of distinct event strings (first-seen order)
    for path, label in files:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # discard the `ID,Event_seq` header row
            for row in reader:
                if len(row) < 2:
                    continue  # skip blank / malformed lines rather than invent data
                seq = []
                for tok in row[1].split():
                    event = f"{domain}:{int(tok)}"  # int() validates it's an integer id
                    seq.append(event)
                    if event not in seen:
                        seen[event] = None
                examples.append((seq, label))

    templates = list(seen.keys())
    return templates, examples


def _ensure_csv_field_limit() -> None:
    """Raise csv's per-field size cap to the platform max (long BGL sequences overflow it)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


# --------------------------------------------------------------------------------------
# The part that is DONE: merge + client split. This is the new logic the paper needs.
# --------------------------------------------------------------------------------------
def build_federation(
    domain_specs: list[tuple[str, int]],     # e.g. [('hdfs', 3), ('bgl', 2)]
    data_root: str,
    split: str = "iid",                       # 'iid' or 'lognormal' (quantity skew)
    seed: int = 0,
    _injected=None,                           # for tests: {domain: (templates, examples)}
) -> Federation:
    """Build one federation spanning multiple domains over a merged vocabulary.

    domain_specs : which domains and how many clients each contributes.
    split        : how to partition a domain's data across its clients.
                   'iid' = even; 'lognormal' = quantity skew (matches the reference
                   paper's non-IID, sigma=0.25).
    _injected    : test hook to bypass disk loading.
    """
    rng = random.Random(seed)
    vocab = GlobalVocabulary()

    # 1) Register each domain's templates as a contiguous ID block (order = arrival order).
    loaded: dict[str, list[tuple[list[str], int]]] = {}
    for domain, _n in domain_specs:
        if _injected is not None:
            templates, examples = _injected[domain]
        else:
            templates, examples = _load_parsed_domain(domain, data_root)
        vocab.add_domain(domain, templates)
        loaded[domain] = examples

    # 2) For each domain, encode examples and split across its clients.
    clients: list[ClientData] = []
    next_client_id = 0
    for domain, n_clients in domain_specs:
        examples = loaded[domain]
        normal = [seq for seq, lab in examples if lab == 0]
        anomalous = [(seq, lab) for seq, lab in examples if lab == 1]

        # encode template-sequences -> global ids
        enc_normal = [vocab.encode_sequence(domain, s) for s in normal]
        # 1% of normal for training (reference convention); rest to test pool
        rng.shuffle(enc_normal)
        n_train = max(1, int(0.01 * len(enc_normal)))
        train_pool = enc_normal[:n_train]
        test_normal = enc_normal[n_train:]

        test_anom = [vocab.encode_sequence(domain, s) for s, _ in anomalous]

        # partition the training pool across this domain's clients
        parts = _partition(train_pool, n_clients, split, rng)

        # every client of a domain shares that domain's test set (per-domain evaluation)
        test_seqs = test_normal + test_anom
        test_labs = [0] * len(test_normal) + [1] * len(test_anom)

        for p in parts:
            clients.append(ClientData(
                client_id=next_client_id,
                domain=domain,
                normal_train=p,
                test_sequences=test_seqs,
                test_labels=test_labs,
            ))
            next_client_id += 1

    return Federation(vocab=vocab, clients=clients)


def _partition(items: list, n: int, split: str, rng: random.Random) -> list[list]:
    """Split a list into n client shards."""
    if n <= 0:
        raise ValueError("need at least one client")
    if split == "iid":
        # even round-robin
        shards: list[list] = [[] for _ in range(n)]
        for i, it in enumerate(items):
            shards[i % n].append(it)
        return shards
    elif split == "lognormal":
        # quantity skew: draw client weights from log-normal(0, 0.25), normalise
        weights = [math.exp(rng.gauss(0.0, 0.25)) for _ in range(n)]
        total = sum(weights)
        cuts = [w / total for w in weights]
        shards = [[] for _ in range(n)]
        idx = 0
        for c in range(n):
            take = round(cuts[c] * len(items)) if c < n - 1 else len(items) - idx
            shards[c] = items[idx: idx + take]
            idx += take
        return shards
    else:
        raise ValueError(f"unknown split {split!r}")


def _demo() -> None:
    """Show the merge+split on synthetic data (no disk, no real numbers)."""
    def fake_domain(prefix, n_templates, n_normal, n_anom):
        templates = [f"{prefix}_t{i}" for i in range(n_templates)]
        rng = random.Random(1)
        examples = []
        for _ in range(n_normal):
            seq = [rng.choice(templates[: n_templates // 2 or 1]) for _ in range(rng.randint(3, 8))]
            examples.append((seq, 0))
        for _ in range(n_anom):
            seq = [rng.choice(templates) for _ in range(rng.randint(3, 20))]
            examples.append((seq, 1))
        return templates, examples

    injected = {
        "hdfs": fake_domain("hdfs", 33, 400, 40),
        "bgl": fake_domain("bgl", 394, 300, 120),
    }
    fed = build_federation(
        [("hdfs", 3), ("bgl", 2)], data_root="(demo)", split="lognormal",
        seed=0, _injected=injected,
    )
    print(fed.summary())
    print(fed.vocab.summary())
    for c in fed.clients:
        print(f"  client {c.client_id} [{c.domain}] "
              f"train={len(c.normal_train)} test={len(c.test_sequences)}")
    print("\ndataloader demo: OK (synthetic — NOT a result)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run the synthetic split demo")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        print("Use --demo for the synthetic split, or wire _load_parsed_domain for real data.")
