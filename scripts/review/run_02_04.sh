#!/usr/bin/env bash
# Driver for experiments 2-4 (CPU only). Threshold for exp3 = chain threshold chosen in exp2.
# Needs the full frame cache (01_frame_cache.py without --thumbs-only).
# Environment: VENV (virtualenv to activate, optional), CACHE (default outputs/review_cache),
# OUT (default outputs/review_out).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
CACHE=${CACHE:-outputs/review_cache}; OUT=${OUT:-outputs/review_out}; mkdir -p "$OUT"
python scripts/review/02_near_duplicates.py --cache "$CACHE" --out "$OUT" "$@"
THR=$(python -c "import json;print(json.load(open('$OUT/exp2_summary.json'))['chain_thr'])")
echo "=== exp3 with thr=$THR"
python scripts/review/03_scene_clusters.py --cache "$CACHE" --out "$OUT" --thr "$THR" --chains "$OUT/exp2_chains.csv"
echo "=== exp4"
python scripts/review/04_group_holdout.py --cache "$CACHE" --clusters "$OUT/scene_clusters.csv" --chains "$OUT/exp2_chains.csv" --out "$OUT"
echo ALL_DONE
