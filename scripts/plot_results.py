#!/usr/bin/env python
"""
plot_results.py — figures for the paper, from REAL result CSVs only.

Produces:
  - per-domain F1 bar chart (single-domain vs mixed) for the H1/H2 story
  - forgetting curve across sequential arrivals (H3)
  - model-size growth curve across arrivals (H3)

Reads only from results/*.csv. If a CSV is missing, it says so and skips — it never
fabricates a curve.

Usage:
    python scripts/plot_results.py --kind f1_compare --single results/hdfs_baseline.csv --mixed results/hdfs_bgl_mixed.csv
"""

from __future__ import annotations
import argparse
import csv
import os
import sys


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"missing results file: {path} — run the experiment first, do not fake it")
    with open(path) as f:
        return list(csv.DictReader(f))


def f1_compare(single_path: str, mixed_path: str, out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — pip install -r requirements.txt")

    single = _load(single_path)
    mixed = _load(mixed_path)

    # index by (method, domain)
    def idx(rows):
        return {(r["method"], r["domain"]): float(r["f1"]) for r in rows}

    si, mi = idx(single), idx(mixed)
    keys = sorted(set(si) | set(mi))
    labels = [f"{m}\n{d}" for (m, d) in keys]
    s_vals = [si.get(k, 0.0) for k in keys]
    m_vals = [mi.get(k, 0.0) for k in keys]

    x = range(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.1), 4.5))
    ax.bar([i - w / 2 for i in x], s_vals, width=w, label="single-domain", color="1C7293")
    ax.bar([i + w / 2 for i in x], m_vals, width=w, label="mixed federation", color="C1440E")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-domain F1: single-domain vs mixed federation")
    ax.legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def h1_collapse_recover(summary_path: str, out: str,
                        baseline_path: str | None = None) -> None:
    """The H1 headline figure: Length collapses under mixing and the domain-aware fix
    recovers it, while Known Events (set-union) is unaffected.

    Reads results/h1_multiseed.csv (columns: method,domain,metric,mean,std,n,values) so the
    bars carry real mean +/- std error bars across seeds. Optionally overlays each domain's
    single-domain Length baseline as a dashed line, to show the recovery lands back at the
    standalone number (not at some new, lower ceiling).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — pip install -r requirements.txt")

    rows = _load(summary_path)
    # (method, domain) -> (mean, std)
    stats = {(r["method"], r["domain"]): (float(r["mean"]), float(r["std"])) for r in rows}

    domains = ["hdfs", "bgl"]
    # the three stories per domain: collapse (global), recover (domain-aware), contrast (union)
    series = [
        ("length_global",      "Length (global / domain-blind)", "#C1440E"),
        ("length_domain_aware", "Length (domain-aware, OURS)",   "#1C7293"),
        ("known_events",       "Known Events (set-union)",       "#8AA29E"),
    ]

    # optional single-domain Length baseline overlay
    base_length = {}
    if baseline_path:
        for r in _load(baseline_path):
            if r["method"] == "length_global":
                base_length[r["domain"]] = float(r["f1"])

    n_series = len(series)
    group_w = 0.8
    bar_w = group_w / n_series
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for si, (method, label, color) in enumerate(series):
        xs, means, errs = [], [], []
        for di, dom in enumerate(domains):
            m, s = stats.get((method, dom), (0.0, 0.0))
            xpos = di + (si - (n_series - 1) / 2) * bar_w
            xs.append(xpos)
            means.append(m)
            errs.append(s)
        bars = ax.bar(xs, means, width=bar_w * 0.92, yerr=errs, capsize=4,
                      label=label, color=color, edgecolor="black", linewidth=0.4)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # single-domain baseline overlay (dashed line spanning each domain's group)
    for di, dom in enumerate(domains):
        if dom in base_length:
            y = base_length[dom]
            ax.plot([di - group_w / 2, di + group_w / 2], [y, y],
                    linestyle="--", color="black", linewidth=1.1, alpha=0.7,
                    label="Length single-domain baseline" if di == 0 else None)
            ax.text(di + group_w / 2, y + 0.01, f"baseline {y:.3f}",
                    ha="right", va="bottom", fontsize=7, style="italic")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.upper() for d in domains])
    ax.set_ylabel("F1 (mean ± std over 5 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("H1: mixed-federation Length collapses on HDFS; domain-aware aggregation recovers it")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def h3_curves(seq_path: str, out: str) -> None:
    """Two-panel H3 figure from results/h3_sequential.csv:
      left  = model size across arrivals (set-union grows, DeepLog fixed)
      right = HDFS F1 across arrivals (set-union keeps it, DeepLog forgets)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — pip install -r requirements.txt")

    rows = _load(seq_path)
    stages = ["hdfs", "bgl"]          # arrival order
    stage_label = ["after HDFS", "after BGL"]
    methods = sorted({r["method"] for r in rows})
    colors = {"known_events": "#8AA29E", "deeplog": "#C1440E"}

    # size across arrivals (use HDFS-eval rows, one per stage per method)
    def series_size(method):
        d = {r["after_arrival"]: int(r["model_bytes"]) for r in rows
             if r["method"] == method and r["eval_domain"] == "hdfs"}
        return [d.get(s) for s in stages]

    def series_f1(method):
        d = {r["after_arrival"]: float(r["f1"]) for r in rows
             if r["method"] == method and r["eval_domain"] == "hdfs"}
        return [d.get(s) for s in stages]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    for m in methods:
        ys = series_size(m)
        if all(v is not None for v in ys):
            axL.plot(stage_label, ys, marker="o", label=m, color=colors.get(m))
    axL.set_ylabel("model state size (bytes)")
    axL.set_title("H3: model-size growth across domain arrivals")
    axL.set_yscale("log")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8)

    for m in methods:
        ys = series_f1(m)
        if all(v is not None for v in ys):
            axR.plot(stage_label, ys, marker="o", label=m, color=colors.get(m))
            for x, y in zip(stage_label, ys):
                axR.text(x, y + 0.015, f"{y:.3f}", ha="center", fontsize=8)
    axR.set_ylabel("HDFS F1 (earlier domain)")
    axR.set_ylim(0, 1.05)
    axR.set_title("H3: HDFS F1 after BGL arrives (higher = less forgetting)")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8)

    fig.suptitle("H3: set-union grows unbounded; DeepLog stays fixed-size — "
                 "neither forgets here (scalar-id input keeps domains separable)")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def mechanism(scalar_h3: str, embed_h3: str, out: str) -> None:
    """The mechanism figure: DeepLog HDFS F1 before/after BGL arrival, scalar vs embedding
    input. Scalar keeps HDFS (disjoint input ranges); embedding forgets it catastrophically.
    Reads results/h3_sequential.csv and results/h3_sequential_embedding.csv."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — pip install -r requirements.txt")

    def hdfs_f1(path):
        rows = _load(path)
        d = {r["after_arrival"]: float(r["f1"]) for r in rows
             if r["method"] == "deeplog" and r["eval_domain"] == "hdfs"}
        return d.get("hdfs"), d.get("bgl")   # (after HDFS, after BGL)

    s_before, s_after = hdfs_f1(scalar_h3)
    e_before, e_after = hdfs_f1(embed_h3)

    stages = ["after HDFS\n(trained)", "after BGL\n(adapted)"]
    x = range(len(stages))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, [s_before, s_after], marker="o", markersize=9, linewidth=2.2,
            color="#1C7293", label="scalar id input (disjoint ranges)")
    ax.plot(x, [e_before, e_after], marker="s", markersize=9, linewidth=2.2,
            color="#C1440E", label="embedding input (shared feature space)")
    for xi, y in zip(x, [s_before, s_after]):
        ax.text(xi, y + 0.02, f"{y:.3f}", ha="center", color="#1C7293", fontsize=9)
    for xi, y in zip(x, [e_before, e_after]):
        ax.text(xi, y - 0.05, f"{y:.3f}", ha="center", color="#C1440E", fontsize=9)
    ax.annotate("catastrophic\nforgetting", xy=(1, e_after), xytext=(0.55, 0.28),
                arrowprops=dict(arrowstyle="->", color="#C1440E"), color="#C1440E", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("HDFS F1 (earlier domain)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Mechanism: DeepLog forgets HDFS under sequential arrival\n"
                 "only when the input representation makes domains share feature space")
    ax.legend(loc="center left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def h2_bars(h2_path: str, out: str) -> None:
    """H2 figure: per-domain DeepLog F1, single-domain vs mixed federation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — pip install -r requirements.txt")

    rows = _load(h2_path)
    f1 = {(r["setting"], r["domain"]): float(r["f1"]) for r in rows}
    domains = ["hdfs", "bgl"]
    w = 0.38
    x = range(len(domains))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    single = [f1.get(("single", d), 0.0) for d in domains]
    mixed = [f1.get(("mixed", d), 0.0) for d in domains]
    b1 = ax.bar([i - w / 2 for i in x], single, width=w, label="single-domain", color="#1C7293")
    b2 = ax.bar([i + w / 2 for i in x], mixed, width=w, label="mixed federation", color="#C1440E")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.upper() for d in domains])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("H2: DeepLog per-domain F1 — single-domain vs mixed federation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["f1_compare", "h1", "h2", "h3", "mech"], required=True)
    ap.add_argument("--single")
    ap.add_argument("--mixed")
    ap.add_argument("--summary", default="results/h1_multiseed.csv",
                    help="h1: the multi-seed summary CSV")
    ap.add_argument("--baseline", default="results/hdfs_baseline.csv",
                    help="h1: optional single-domain baseline overlay (Length)")
    ap.add_argument("--out", default="results/f1_compare.png")
    args = ap.parse_args()
    if args.kind == "f1_compare":
        if not (args.single and args.mixed):
            sys.exit("f1_compare needs --single and --mixed")
        f1_compare(args.single, args.mixed, args.out)
    elif args.kind == "h1":
        base = args.baseline if (args.baseline and os.path.exists(args.baseline)) else None
        h1_collapse_recover(args.summary, args.out, baseline_path=base)
    elif args.kind == "h2":
        h2_bars(args.mixed or "results/h2_deeplog.csv", args.out)
    elif args.kind == "h3":
        h3_curves(args.mixed or "results/h3_sequential.csv", args.out)
    elif args.kind == "mech":
        mechanism("results/h3_sequential.csv", "results/h3_sequential_embedding.csv", args.out)
