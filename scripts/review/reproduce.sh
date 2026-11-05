#!/usr/bin/env bash
# One entry point for the scene-disjoint ("leak-free") experiment: dataset roots whose train,
# validation and test clips come from different scenes, the project's training recipes on them,
# and a comparison with the published numbers.  See scripts/review/README.md.
#
#   bash scripts/review/reproduce.sh                              # published splits, five runs, comparison
#   STAGES="data splits" bash scripts/review/reproduce.sh         # only create the leak-free dataset roots
#   RUNS="R2" bash scripts/review/reproduce.sh                    # one model (ResNet50 on grp27a)
#   STAGES="data rebuild train compare" bash scripts/review/reproduce.sh
#                                                                 # rebuild clusters and splits from the clips first
# Stages (STAGES, default "data splits train compare"):
#   data     fetch the 320p clips with DVC if data/ouc-cge-320/videos is missing (4.4 GB)
#   splits   write data/grp27a, data/grp27b, data/grp17 from the published split files (09, seconds)
#   rebuild  thumbnails of every clip (01 --thumbs-only) -> scene clusters (03b) -> splits (07),
#            each compared with the published files; stops if they differ.  CPU is enough.
#   train    08_run_grouped_training.sh: R1..R5, about 50 min on an RTX 5090 (needs a GPU)
#   compare  published metrics next to this run (10), also written to $OUT/comparison.md
# Environment:
#   VENV          virtualenv to activate first (optional)
#   RUNS          subset of "R1 R2 R3 R4 R5" for the train stage (default: all)
#   OUT           summaries, logs, comparison          (default outputs/review_grouped)
#   CACHE         frame cache of the rebuild stage      (default outputs/review_cache)
#   CLUSTERS_OUT  rebuilt scene clusters                (default outputs/review_out)
# The 01/03b steps use a GPU when one is visible; on a machine whose GPU must not be used set
# CUDA_VISIBLE_DEVICES="" for the data/splits/rebuild stages.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
STAGES=${STAGES:-"data splits train compare"}
export OUT=${OUT:-outputs/review_grouped}
CACHE=${CACHE:-outputs/review_cache}
CLUSTERS_OUT=${CLUSTERS_OUT:-outputs/review_out}
has() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$OUT"

if has data; then
  echo "=== data"
  if [ -d data/ouc-cge-320/videos ] && [ -n "$(ls -A data/ouc-cge-320/videos 2>/dev/null)" ]; then
    echo "data/ouc-cge-320/videos present"
  else
    dvc pull data/ouc-cge-320.dvc
  fi
fi

if has rebuild; then
  echo "=== rebuild: frame thumbnails"
  python scripts/review/01_frame_cache.py --root data/ouc-cge-320 --out "$CACHE" --thumbs-only
  echo "=== rebuild: scene clusters"
  python scripts/review/03b_scene_clusters_thumb.py --cache "$CACHE" --out "$CLUSTERS_OUT"
  python scripts/review/10_compare_with_published.py --clusters "$CLUSTERS_OUT" --out "$OUT/comparison_clusters.md"
  echo "=== rebuild: splits"
  CLUSTERS="$CLUSTERS_OUT" CACHE="$CACHE" REPORT="$OUT/splits" bash scripts/review/run_07_splits.sh
fi

if has splits; then
  echo "=== splits"
  python scripts/review/09_materialize_splits.py
fi

if has train; then
  echo "=== train"
  bash scripts/review/08_run_grouped_training.sh
fi

if has compare; then
  echo "=== compare"
  python scripts/review/10_compare_with_published.py --runs "$OUT" --out "$OUT/comparison.md"
fi
echo REPRODUCE_DONE
