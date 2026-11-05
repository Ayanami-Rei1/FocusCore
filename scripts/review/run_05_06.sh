#!/usr/bin/env bash
# Driver for experiments 5 and 6 (needs a GPU). Runs sequentially (single GPU).
# Evaluates the stored checkpoints of the random-split runs:
#   dvc pull outputs/baseline_ouc_cge.dvc outputs/resnet50_single_frame_best.dvc
# (dvc refuses these two files while the root .gitignore lists "outputs/": remove that line for
# the pull and restore it afterwards).
# Environment: VENV (virtualenv to activate, optional), OUT (default outputs/review_eval).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
OUT=${OUT:-outputs/review_eval}; mkdir -p "$OUT"
SLOW="outputs/baseline_ouc_cge/2026-05-31_15-35-39/checkpoints/epoch=30-val/accuracy=1.0000.ckpt"
R50="outputs/resnet50_single_frame_best/accuracy=0.9961.ckpt"
run() { echo "=== $*"; python scripts/review/05_eval_checkpoints.py --out "$OUT" "$@" 2>&1 | grep -v -E "Warning|warn|UserWarning" ; }
# Experiment 5: reproduce the student's numbers through the project's own path
run --model slow     --ckpt "$SLOW" --split test --mode identity --tag slow_test
run --model slow     --ckpt "$SLOW" --split val  --mode identity --tag slow_val
run --model resnet50 --ckpt "$R50"  --split test --mode identity --tag resnet50_test
# Experiment 6: temporal ablations of the SLOW checkpoint on the test split
run --model slow --ckpt "$SLOW" --split test --mode shuffle       --tag slow_test_shuffle
run --model slow --ckpt "$SLOW" --split test --mode reverse       --tag slow_test_reverse
run --model slow --ckpt "$SLOW" --split test --mode static_center --tag slow_test_static_center
run --model slow --ckpt "$SLOW" --split test --mode static_first  --tag slow_test_static_first
# precision check: fp32 instead of bf16-mixed
run --model slow --ckpt "$SLOW" --split test --mode identity --precision 32 --tag slow_test_fp32
echo ALL_DONE
