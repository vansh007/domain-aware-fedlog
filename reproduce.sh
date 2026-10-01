#!/usr/bin/env bash
# reproduce.sh — regenerate the derived artifacts (analyses, validations, figures) from the
# committed result CSVs, and run the auditor tests. This runs end-to-end from a fresh clone
# WITHOUT the raw datasets, because every result CSV is committed under results/.
#
# To re-run the experiments that PRODUCE those CSVs you need the raw logs (HDFS/BGL from LogHub;
# OpenStack via scripts/parse_openstack.py) and the deep-model runs are CPU-bound — see the
# per-experiment scripts in scripts/ and docs/TILLNOW.md. Those are intentionally not run here.
set -euo pipefail
cd "$(dirname "$0")"

echo "== auditor unit tests =="
python tests/test_auditor.py

echo "== auditor prediction validation (14/15) =="
python scripts/validate_auditor.py | tail -3

echo "== organisational impact analysis (blind-spot + cost) =="
python scripts/impact_analysis.py | tail -14

echo "== regenerate paper figures from result CSVs =="
python scripts/plot_results.py --kind h1    --summary results/h1_multiseed.csv --baseline results/hdfs_baseline.csv --out results/h1_collapse_recover.png
python scripts/plot_results.py --kind h3    --mixed results/h3_sequential.csv  --out results/h3_sequential.png
python scripts/plot_results.py --kind mech  --out results/mechanism_forgetting.png
python scripts/plot_results.py --kind map   --out results/two_by_two_map.png
python scripts/plot_results.py --kind replay --mixed results/mitigation_replay.csv --out results/replay_mitigation.png
python scripts/plot_results.py --kind arch   --out results/architecture_generality.png
python scripts/plot_results.py --kind 3domain --mixed results/h1_3domain.csv --out results/three_domain_h1.png
python scripts/plot_results.py --kind impact --out results/impact_asymmetry.png

echo
echo "== done. Figures + analyses regenerated under results/. Paper: docs/paper/main.tex =="
