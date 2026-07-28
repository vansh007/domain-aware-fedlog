#!/usr/bin/env python
"""smoke_test.py — proves the full experiment pipeline runs end to end on synthetic data.
This is NOT a result. It exists so you know the plumbing works before wiring real data.
Run: python scripts/smoke_test.py
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataloader import build_federation
from src.methods.length import LengthDetector
from src.methods.known_events import KnownEventsDetector
from src.aggregation import DomainAwareLength
from src.metrics import prf

def fake_domain(prefix, n_templates, n_normal, n_anom, normal_len, anom_len, seed):
    rng = random.Random(seed)
    templates = [f"{prefix}_t{i}" for i in range(n_templates)]
    ex = []
    for _ in range(n_normal):
        L = max(1, int(rng.gauss(*normal_len)))
        ex.append(([rng.choice(templates) for _ in range(L)], 0))
    for _ in range(n_anom):
        L = max(1, int(rng.gauss(*anom_len)))
        ex.append(([rng.choice(templates) for _ in range(L)], 1))
    return templates, ex

# HDFS: tight normal length ~29, anomalies longer. BGL: wide, huge tail.
injected = {
    "hdfs": fake_domain("hdfs", 33, 500, 50, (29, 6), (55, 8), 1),
    "bgl":  fake_domain("bgl", 394, 400, 150, (69, 200), (69, 200), 2),
}

print("=== domain-blind (mixed) — expect HDFS Length to suffer (H1 shape) ===")
fed = build_federation([("hdfs",3),("bgl",2)], "(demo)", split="iid", seed=0, _injected=injected)
det = LengthDetector()
det.aggregate([det.fit_local(c.normal_train) for c in fed.clients])
for dom in fed.vocab.domains:
    c = fed.clients_of(dom)[0]
    print(f"  length_global {dom}: F1={prf(c.test_labels, det.predict(c.test_sequences)).f1:.3f}")

print("=== domain-aware — expect HDFS recovered ===")
da = DomainAwareLength()
base = LengthDetector(); by={}
for c in fed.clients: by.setdefault(c.domain,[]).append(base.fit_local(c.normal_train))
da.aggregate_per_domain(by)
for dom in fed.vocab.domains:
    c = fed.clients_of(dom)[0]
    pred = da.predict(c.test_sequences, [dom]*len(c.test_sequences))
    print(f"  length_da     {dom}: F1={prf(c.test_labels, pred).f1:.3f}")

print("\nsmoke test OK — pipeline runs end to end. (synthetic data, NOT a result)")
