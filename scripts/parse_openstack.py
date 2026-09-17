#!/usr/bin/env python
"""
parse_openstack.py — turn the raw LogHub/DeepLog OpenStack logs into the dataloader's
`ID,Event_seq` format, so OpenStack becomes a THIRD federated domain alongside HDFS and BGL.

Pipeline (the standard DeepLog OpenStack treatment):
  - Read openstack_normal1.log, openstack_normal2.log (normal) and openstack_abnormal.log.
  - Each VM instance is a session: group log lines by their [instance: <uuid>] tag (lines with
    no instance tag are OpenStack control-plane noise and are skipped, exactly as DeepLog does).
  - Template each line's human-readable message with Drain (light pre-masking of numbers / IPs /
    UUIDs first), assign each unique template a local 0-indexed event id -> one integer sequence
    per instance.
  - Label: the 4 instance UUIDs in anomaly_labels.txt (injected anomalies) are abnormal; all
    normal1+normal2 instances are normal. Ambiguous non-labelled instances in the abnormal log
    are discarded (they carry no ground-truth label).
  - Write data/OPENSTACK/{normal,abnormal}.csv with header `ID,Event_seq`.

Honest note printed at the end: OpenStack ships only 4 labelled anomalies, a known limitation of
this benchmark; we therefore use OpenStack to test the H1 range-collapse / containment result and
the 3-domain vocabulary+routing, where its rich NORMAL set (1000s of instances) is the signal, and
we report its small positive count openly.

Usage: python scripts/parse_openstack.py
"""

from __future__ import annotations
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RAW = "data/OPENSTACK_raw"
OUT = "data/OPENSTACK"
NORMAL_FILES = ["openstack_normal1.log", "openstack_normal2.log"]
ABNORMAL_FILE = "openstack_abnormal.log"
LABELS = "anomaly_labels.txt"

_INST = re.compile(r"\[instance:\s*([0-9a-f\-]{36})\]")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_IP = re.compile(r"\b\d+\.\d+\.\d+\.\d+\b")
_NUM = re.compile(r"\b\d+\b")
_AFTER_BRACKET = re.compile(r".*\]\s*(.*)$")  # message text after the last ']'


def _message(line: str) -> str:
    m = _AFTER_BRACKET.match(line)
    msg = m.group(1) if m else line
    msg = _UUID.sub("<UUID>", msg)
    msg = _IP.sub("<IP>", msg)
    msg = _NUM.sub("<NUM>", msg)
    return msg.strip()


def _anomaly_ids() -> set[str]:
    ids = set()
    with open(os.path.join(RAW, LABELS)) as f:
        for line in f:
            m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", line)
            if m:
                ids.add(m.group(0))
    return ids


def main():
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.5
    tm = TemplateMiner(config=cfg)

    templ_to_id: dict[str, int] = {}          # drain cluster id -> local event id
    sessions: dict[str, list[int]] = {}       # instance uuid -> event id sequence
    order: list[str] = []                     # first-seen order of instances

    def process(path):
        with open(path, errors="ignore") as f:
            for line in f:
                mi = _INST.search(line)
                if not mi:
                    continue                   # no instance tag -> skip (control-plane noise)
                inst = mi.group(1)
                cid = tm.add_log_message(_message(line))["cluster_id"]
                if cid not in templ_to_id:
                    templ_to_id[cid] = len(templ_to_id)
                eid = templ_to_id[cid]
                if inst not in sessions:
                    sessions[inst] = []
                    order.append(inst)
                sessions[inst].append(eid)

    for nf in NORMAL_FILES:
        process(os.path.join(RAW, nf))
    n_after_normal = len(sessions)
    process(os.path.join(RAW, ABNORMAL_FILE))

    anomalies = _anomaly_ids()
    # normal = every instance first seen in the two normal files (and not an anomaly)
    normal_ids = [i for i in order[:n_after_normal] if i not in anomalies]
    abnormal_ids = [i for i in order if i in anomalies]

    os.makedirs(OUT, exist_ok=True)
    for name, ids in (("normal", normal_ids), ("abnormal", abnormal_ids)):
        with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "Event_seq"])
            for i in ids:
                w.writerow([i, " ".join(str(e) for e in sessions[i])])

    lens = [len(sessions[i]) for i in normal_ids]
    import statistics
    print("=" * 60)
    print("OPENSTACK parsed ->", OUT)
    print("=" * 60)
    print(f"event types (vocabulary): {len(templ_to_id)}")
    print(f"normal instances : {len(normal_ids)}")
    print(f"abnormal instances (labelled): {len(abnormal_ids)}  (uuids: {sorted(anomalies)})")
    if lens:
        print(f"normal seq length: mean {statistics.fmean(lens):.1f} +/- "
              f"{statistics.pstdev(lens):.1f}  (min {min(lens)}, max {max(lens)})")
    ab_lens = [len(sessions[i]) for i in abnormal_ids]
    if ab_lens:
        print(f"abnormal seq length: {ab_lens}")
    print("\nNOTE: OpenStack ships only 4 labelled anomalies (a known benchmark limitation);")
    print("use it for the H1 containment/range result and 3-domain vocab+routing, report N openly.")


if __name__ == "__main__":
    main()
