#!/usr/bin/env python
"""
parse_hadoop.py — turn the LogHub Hadoop logs into the dataloader's `ID,Event_seq` format, adding
Hadoop as a FOURTH federated domain.

Each Hadoop *application* (one WordCount/PageRank run) is a session: we concatenate its container logs
in order, Drain-template each structured log line, and emit one integer event sequence per application.
Labels come from abnormal_label.txt (Normal vs the injected failures Machine down / Network
disconnection / Disk full): 11 normal applications, 44 abnormal. Unlike OpenStack, Hadoop's anomalies
are genuine failure runs with a balanced-ish label set, so it contributes a real (if small) detection
test in addition to a fourth disjoint vocabulary block for routing and a fourth length profile for the
containment condition.

Writes data/HADOOP/{normal,abnormal}.csv. Read-only w.r.t. everything else.

Usage: python scripts/parse_hadoop.py
"""

from __future__ import annotations
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RAW = "data/HADOOP_raw"
OUT = "data/HADOOP"
LABELS = os.path.join(RAW, "abnormal_label.txt")

# "2015-10-17 15:38:05,337 INFO [main] org.apache.hadoop.X.Class: <message>"
_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ \w+ \[[^\]]*\] [\w.$]+: (.*)$")
_HEX = re.compile(r"\b[0-9a-f]{8,}\b")
_NUM = re.compile(r"\b\d+\b")
_PATH = re.compile(r"(/[\w./-]+|hdfs://\S+|\w+://\S+)")
_ID = re.compile(r"(application|container|job|task|attempt|blk)[_\-][\w.\-]+")


def _message(line: str) -> str | None:
    m = _LINE.match(line)
    if not m:
        return None                       # continuation / stack-trace line -> skip (noise)
    msg = m.group(1)
    msg = _ID.sub("<ID>", msg)
    msg = _PATH.sub("<PATH>", msg)
    msg = _HEX.sub("<HEX>", msg)
    msg = _NUM.sub("<NUM>", msg)
    return msg.strip()


def _labels():
    section, normal, abn = None, set(), set()
    for line in open(LABELS):
        s = line.strip()
        if s.endswith(":") and not s.startswith("+"):
            section = s[:-1].strip().lower()
        m = re.match(r"\+\s*(application_\S+)", s)
        if m:
            (normal if section == "normal" else abn).add(m.group(1))
    return normal, abn


def main():
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.5
    tm = TemplateMiner(config=cfg)
    templ_to_id: dict[int, int] = {}
    sessions: dict[str, list[int]] = {}

    apps = sorted(d for d in os.listdir(RAW)
                  if os.path.isdir(os.path.join(RAW, d)) and d.startswith("application_"))
    for app in apps:
        seq: list[int] = []
        app_dir = os.path.join(RAW, app)
        for cf in sorted(os.listdir(app_dir)):
            if not cf.endswith(".log"):
                continue
            with open(os.path.join(app_dir, cf), errors="ignore") as f:
                for line in f:
                    msg = _message(line)
                    if msg is None:
                        continue
                    cid = tm.add_log_message(msg)["cluster_id"]
                    if cid not in templ_to_id:
                        templ_to_id[cid] = len(templ_to_id)
                    seq.append(templ_to_id[cid])
        sessions[app] = seq

    normal_ids, abn_ids = _labels()
    normal = [a for a in apps if a in normal_ids]
    abnormal = [a for a in apps if a in abn_ids]

    os.makedirs(OUT, exist_ok=True)
    for name, ids in (("normal", normal), ("abnormal", abnormal)):
        with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "Event_seq"])
            for a in ids:
                w.writerow([a, " ".join(str(e) for e in sessions[a])])

    import statistics
    nlen = [len(sessions[a]) for a in normal]
    alen = [len(sessions[a]) for a in abnormal]
    print("=" * 60)
    print("HADOOP parsed ->", OUT)
    print("=" * 60)
    print(f"event types (vocabulary): {len(templ_to_id)}")
    print(f"normal apps  : {len(normal)}   length mean {statistics.fmean(nlen):.0f} "
          f"(min {min(nlen)}, max {max(nlen)})")
    print(f"abnormal apps: {len(abnormal)}   length mean {statistics.fmean(alen):.0f} "
          f"(min {min(alen)}, max {max(alen)})")


if __name__ == "__main__":
    main()
