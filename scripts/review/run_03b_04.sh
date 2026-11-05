#!/usr/bin/env bash
# Revised exp3 (thumbnail scene clusters) + exp4 (group-held-out CV, all thresholds).
# Run after run_02_04.sh (same CACHE and OUT): 03b replaces scene_clusters.csv with the thumbnail
# clusters and writes exp3b_clusters_by_thr.csv; these two files are the input of run_07_splits.sh.
# Environment: VENV (virtualenv to activate, optional), CACHE (default outputs/review_cache),
# OUT (default outputs/review_out).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
CACHE=${CACHE:-outputs/review_cache}; OUT=${OUT:-outputs/review_out}; mkdir -p "$OUT"
python scripts/review/03b_scene_clusters_thumb.py --cache "$CACHE" --out "$OUT" --chains "$OUT/exp2_chains.csv"
python scripts/review/04_group_holdout.py --cache "$CACHE" --clusters "$OUT/scene_clusters.csv" --chains "$OUT/exp2_chains.csv" --clusters-multi "$OUT/exp3b_clusters_by_thr.csv" --out "$OUT"
echo ALL_DONE
