#!/usr/bin/env bash
# Rebuild the three scene-disjoint splits (grp27a, grp27b, grp17) from scene clusters, with the
# arguments of the published splits, and compare the result with the published assignment in
# docs/review/data/grouped/splits/. CPU only, about a minute.
#
#   bash scripts/review/run_07_splits.sh                          # from the published clusters
#   CLUSTERS=outputs/review_out CACHE=outputs/review_cache bash scripts/review/run_07_splits.sh
#                                                                 # from clusters rebuilt by 03b,
#                                                                 # with the thumbnail leakage check
# Environment:
#   VENV      virtualenv to activate first (optional)
#   CLUSTERS  directory with scene_clusters.csv and exp3b_clusters_by_thr.csv (default docs/review/data)
#   CACHE     frame cache of 01_frame_cache.py; enables the nearest-train-clip leakage statistics
#             (needs about 2 GB of RAM)
#   REPORT    where split_<name>.csv and split_<name>_summary.json go (default outputs/review_grouped/splits)
#   OUT_BASE  where the dataset roots go (default data, i.e. data/grp27a, data/grp27b, data/grp17)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
CLUSTERS=${CLUSTERS:-docs/review/data}
REPORT=${REPORT:-outputs/review_grouped/splits}
OUT_BASE=${OUT_BASE:-data}
PUBLISHED=docs/review/data/grouped/splits
common=(--clusters "$CLUSTERS/scene_clusters.csv" --clusters-multi "$CLUSTERS/exp3b_clusters_by_thr.csv"
        --out-base "$OUT_BASE" --report "$REPORT")
if [ -n "${CACHE:-}" ]; then common+=(--cache "$CACHE"); fi

python scripts/review/07_make_grouped_splits.py --name grp27a --thr 27.5 --seed 0 "${common[@]}"
python scripts/review/07_make_grouped_splits.py --name grp27b --thr 27.5 --seed 1 \
    --avoid "$REPORT/split_grp27a.csv" "${common[@]}"
python scripts/review/07_make_grouped_splits.py --name grp17 --thr 17.5 --seed 0 "${common[@]}"

rc=0
for name in grp27a grp27b grp17; do
  if cmp -s "$REPORT/split_$name.csv" "$PUBLISHED/split_$name.csv"; then
    echo "split_$name.csv: identical to $PUBLISHED/split_$name.csv"
  else
    echo "split_$name.csv: DIFFERS from $PUBLISHED/split_$name.csv"; rc=1
  fi
done
if [ $rc -ne 0 ]; then
  echo "The rebuilt splits differ from the published ones. To train on exactly the published splits run:"
  echo "  python scripts/review/09_materialize_splits.py"
fi
echo SPLITS_DONE
exit $rc
