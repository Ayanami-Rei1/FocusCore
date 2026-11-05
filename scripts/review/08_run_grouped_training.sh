#!/usr/bin/env bash
# Step 2 of the grouped-split experiment: the project's own training recipes, through the
# project's entry point (python -m src.train), on the random split (control) and on the
# scene-disjoint splits built by 07_make_grouped_splits.py / 09_materialize_splits.py.
# One GPU, strictly sequential.
#
#   R1  resnet50 single frame, random split (control; = the stored run: 10 ep, batch 16, bf16)
#   R2  resnet50, grp27a      R3  resnet50, grp27b      R4  resnet50, grp17
#   R5  SLOW R50, grp27a (15 epochs, early stopping patience 5 after epoch 5)
#
# For every run: training log, MLflow metric files, best checkpoint path, per-clip test and val
# predictions of the best checkpoint and test predictions of last.ckpt through
# scripts/review/05_eval_checkpoints.py (the project's test_step), wall time, the code version
# (git commit, "+local-changes" if src/ or configs/ differ from it), "Skipping" count.
#
# Usage (from anywhere; the script changes to the repository root):
#   bash scripts/review/08_run_grouped_training.sh                 # all five runs, about 50 min on an RTX 5090
#   RUNS="R2 R5" bash scripts/review/08_run_grouped_training.sh    # only these runs
#   nohup bash scripts/review/08_run_grouped_training.sh > 08_grouped.log 2>&1 &
# Environment:
#   VENV        virtualenv to activate first (optional; e.g. /venv/main on the vast.ai image)
#   OUT         where the summaries go            (default outputs/review_grouped)
#   LOGS        where the per-run logs go          (default $OUT/logs)
#   RUNS        subset of "R1 R2 R3 R4 R5"         (default: all)
#   FORCE=1     redo runs that already have an evaluation summary in $OUT/eval
#   BUDGET_MIN  wall-clock budget in minutes; if set and the projection exceeds it, R3 is dropped
#               first, then R5 is shortened to 10 epochs (default 0 = no budget; the published runs
#               had 180 and nothing was dropped)
# Checkpoints are written to outputs/review/<run>/checkpoints/ (best two by val/accuracy + last.ckpt).
# NOTE: last.ckpt is NOT the final epoch.  With Lightning 2.6 `save_last=True` copies the most recently
# saved top-k checkpoint, so here it equals the later of the two best-by-validation epochs.  For the
# final epoch and every other epoch use 11_epoch_curve.sh.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
# MLflow >= 3.17 refuses the file-store backend that the project's `mlflow.tracking_uri: null`
# resolves to ("./mlruns ... in maintenance mode", MlflowException at MLFlowLogger()) unless this is set.
export MLFLOW_ALLOW_FILE_STORE=true
OUT=${OUT:-outputs/review_grouped}
LOGS=${LOGS:-$OUT/logs}
mkdir -p "$OUT/eval" "$OUT/mlflow" "$LOGS"
T_START=$(date +%s)
BUDGET_MIN=${BUDGET_MIN:-0}
SLOW_EPOCHS=15

R50_ARGS="+experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16"
SLOW_ARGS="+experiment=baseline_ouc_cge trainer.early_stopping.patience=5 trainer.early_stopping.min_epochs=5 data.batch_size=32 data.num_workers=16 trainer.precision=bf16-mixed"

code_version() {
  # Commit of the checkout, marked when src/ or configs/ carry uncommitted changes.
  # (The published runs.log says "reviewed-working-tree": that tree is commit 203cfd7.)
  local rev
  rev=$(git rev-parse --short HEAD 2>/dev/null) || { echo "no-git"; return; }
  if git diff --quiet HEAD -- src/ configs/ 2>/dev/null; then echo "$rev"; else echo "$rev+local-changes"; fi
}

want() { [ -z "${RUNS:-}" ] || [[ " ${RUNS} " == *" $1 "* ]]; }

need_root() {  # need_root <root> : the dataset root must exist before a run starts
  local root=$1
  if [ ! -f "$root/data_labels/train.csv" ] || [ ! -d "$root/videos/" ]; then
    echo "MISSING dataset root $root (data_labels/*.csv and videos/)." | tee -a "$OUT/runs.log"
    echo "  clips:  dvc pull data/ouc-cge-320.dvc" | tee -a "$OUT/runs.log"
    echo "  splits: python scripts/review/09_materialize_splits.py" | tee -a "$OUT/runs.log"
    exit 2
  fi
}

elapsed_min() { echo $(( ($(date +%s) - T_START) / 60 )); }

post_run() {  # post_run <name> <model_key> <root> : locate the MLflow run, copy metrics, eval best ckpt
  local name=$1 model=$2 root=$3
  python - "$name" "$OUT" <<'EOF'
import json, shutil, sys
from pathlib import Path
name, out = sys.argv[1], Path(sys.argv[2])
want = f"outputs/review/{name}"
hits = [p.parents[1] for p in Path("mlruns").glob("*/*/params/output_dir") if p.read_text().strip() == want]
if not hits:
    print("MLFLOW_RUN_NOT_FOUND"); sys.exit(0)
hit = max(hits, key=lambda d: (d / "meta.yaml").stat().st_mtime)  # the latest run when a run was repeated
dst = out / "mlflow" / name
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(hit / "metrics", dst / "metrics"); shutil.copytree(hit / "params", dst / "params")
if (hit / "tags").exists(): shutil.copytree(hit / "tags", dst / "tags")
summary = {"run_id": hit.name, "experiment_id": hit.parent.name}
for f in sorted((hit / "metrics").glob("**/*")):
    if f.is_file():
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        key = str(f.relative_to(hit / "metrics"))
        summary[key] = {"last": float(rows[-1][1]), "last_step": int(rows[-1][2]), "n": len(rows),
                        "max": max(float(r[1]) for r in rows), "min": min(float(r[1]) for r in rows)}
for f in (hit / "params").glob("*"):
    summary["param_" + f.name] = f.read_text().strip()
json.dump(summary, open(dst / "summary.json", "w"), indent=2)
print("BEST_CKPT=" + summary.get("param_best_checkpoint_path", ""))
print(f"MLFLOW run={hit.name} best_val={summary.get('best_val/accuracy', {}).get('last')} best_epoch={summary.get('best_epoch', {}).get('last')} "
      f"final_epoch={summary.get('final_epoch', {}).get('last')} test_acc={summary.get('test/accuracy', {}).get('last')} "
      f"test_f1={summary.get('test/f1_macro', {}).get('last')} train_acc_last={summary.get('train/accuracy', {}).get('last')}")
EOF
  local ckpt
  ckpt=$(python - "$OUT/mlflow/$name/summary.json" <<'EOF'
import json, sys
try: print(json.load(open(sys.argv[1])).get("param_best_checkpoint_path", ""))
except Exception: print("")
EOF
)
  if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
    # fall back to the highest val_acc in the checkpoint filenames
    ckpt=$(ls outputs/review/$name/checkpoints/*.ckpt 2>/dev/null | grep -v last.ckpt | sort -t= -k3 -g | tail -1)
    echo "best_checkpoint_path missing in MLflow; using $ckpt"
  fi
  echo "eval $name ckpt=$ckpt"
  for split in test val; do
    python scripts/review/05_eval_checkpoints.py --model "$model" --ckpt "$ckpt" --root "$root" --split $split \
        --mode identity --out "$OUT/eval" --tag "${name}_${split}" > "$LOGS/eval_${name}_${split}.log" 2>&1
    echo "  eval $split rc=$? $(grep -o '"lightning_test_accuracy": [0-9.]*' "$OUT/eval/summary_${name}_${split}.json" 2>/dev/null)"
  done
  # last.ckpt (the later of the two saved top-k epochs, see the note above) on the test split
  local last
  last=$(python - "$OUT/mlflow/$name/summary.json" <<'EOF'
import json, sys
try: print(json.load(open(sys.argv[1])).get("param_last_checkpoint_path", ""))
except Exception: print("")
EOF
)
  if [ -z "$last" ] || [ ! -f "$last" ]; then last="outputs/review/$name/checkpoints/last.ckpt"; fi
  if [ -f "$last" ]; then
    python scripts/review/05_eval_checkpoints.py --model "$model" --ckpt "$last" --root "$root" --split test \
        --mode identity --out "$OUT/eval" --tag "${name}_last_test" > "$LOGS/eval_${name}_last_test.log" 2>&1
    echo "  eval last.ckpt test rc=$? $(grep -o '"lightning_test_accuracy": [0-9.]*' "$OUT/eval/summary_${name}_last_test.json" 2>/dev/null)"
  fi
  echo "  Skipping-count train log: $(grep -c Skipping "$LOGS/train_$name.log")"
}

run() {  # run <name> <model_key> <root> <hydra overrides...>
  local name=$1 model=$2 root=$3; shift 3
  if [ -f "$OUT/eval/summary_${name}_test.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $name already evaluated, skipping (FORCE=1 to redo)" | tee -a "$OUT/runs.log"; return 0
  fi
  need_root "$root"
  local t0=$(date +%s) cv
  cv=$(code_version)
  echo "=== $name model=$model root=$root code=$cv start=$(date -Is) elapsed=$(elapsed_min)min" | tee -a "$OUT/runs.log"
  echo "CMD: python -m src.train $* data.root=$root paths.output_dir=outputs/review/$name mlflow.experiment_name=review_grouped hydra.run.dir=outputs/review/$name/hydra" | tee -a "$OUT/runs.log"
  python -m src.train "$@" data.root="$root" paths.output_dir="outputs/review/$name" \
      mlflow.experiment_name=review_grouped hydra.run.dir="outputs/review/$name/hydra" \
      > "$LOGS/train_$name.log" 2>&1
  local rc=$? t1=$(date +%s)
  echo "$name rc=$rc wall=$((t1 - t0))s code=$cv" | tee -a "$OUT/runs.log"
  if [ $rc -ne 0 ]; then
    echo "TRAINING FAILED ($name); last 40 log lines:"; tail -40 "$LOGS/train_$name.log"
    echo "ABORT after $name" | tee -a "$OUT/runs.log"; exit $rc
  fi
  post_run "$name" "$model" "$root" 2>&1 | tee -a "$OUT/runs.log"
  echo "$name wall_with_eval=$(( $(date +%s) - t0 ))s" | tee -a "$OUT/runs.log"
}

if want R1; then run R1_r50_random resnet50 data/ouc-cge-320 $R50_ARGS; fi
if want R2; then run R2_r50_grp27a resnet50 data/grp27a $R50_ARGS; fi
SKIP_R3=0
if [ "$BUDGET_MIN" -gt 0 ]; then
  T_R50=$(( $(date +%s) - T_START ))   # ~2 x one R50 run (seconds) when R1 and R2 both ran
  PER_R50=$(( T_R50 / 2 ))
  # projected: R3 + R4 (+ R5 ~ 4 min/epoch x 15)
  PROJ=$(( T_R50 + 2 * PER_R50 + 15 * 240 ))
  if [ $PROJ -gt $(( BUDGET_MIN * 60 )) ]; then SKIP_R3=1; fi
  echo "budget: after R1+R2 elapsed=${T_R50}s per_r50=${PER_R50}s projected_total=${PROJ}s budget=$((BUDGET_MIN*60))s skip_R3=$SKIP_R3" | tee -a "$OUT/runs.log"
fi
if want R3; then
  if [ $SKIP_R3 -eq 0 ]; then run R3_r50_grp27b resnet50 data/grp27b $R50_ARGS; else echo "R3 dropped (budget)" | tee -a "$OUT/runs.log"; fi
fi
if want R4; then run R4_r50_grp17 resnet50 data/grp17 $R50_ARGS; fi
if [ "$BUDGET_MIN" -gt 0 ]; then
  REMAIN=$(( BUDGET_MIN * 60 - ($(date +%s) - T_START) ))
  if [ $REMAIN -lt $(( 15 * 240 + 600 )) ]; then SLOW_EPOCHS=10; fi
  echo "budget: before R5 remaining=${REMAIN}s slow_epochs=$SLOW_EPOCHS" | tee -a "$OUT/runs.log"
fi
if want R5; then run R5_slow_grp27a slow data/grp27a $SLOW_ARGS trainer.max_epochs=$SLOW_EPOCHS; fi
echo "ALL_DONE total=$(( $(date +%s) - T_START ))s" | tee -a "$OUT/runs.log"
