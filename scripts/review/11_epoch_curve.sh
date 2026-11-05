#!/usr/bin/env bash
# Test accuracy of EVERY epoch of the scene-disjoint runs (R2..R5), including the final one.
#
# Why this exists: `last.ckpt` is not the final epoch.  With Lightning 2.6, `save_last=True` copies
# the most recently saved top-k checkpoint, so with the project's `save_top_k: 2` it is the later
# of the two best-by-validation epochs (epochs 5 / 3 / 8 / 6 in the published R2 / R3 / R4 / R5),
# and the final-epoch weights of the 08 runs were never written.  This script repeats the same
# recipes with every epoch saved (trainer.checkpoint.save_top_k=-1, nothing else changed) and
# evaluates each epoch on the test split through 05_eval_checkpoints.py.  The checkpoint that the
# project's protocol selects (best validation accuracy) is marked in the table; the other rows
# show what any other stopping point would have given.
#
# Usage:  bash scripts/review/11_epoch_curve.sh            # R2 R3 R4 R5, about 45 min on an RTX 5090
#         RUNS="R2" bash scripts/review/11_epoch_curve.sh
# Environment: VENV (virtualenv to activate), RUNS (default "R2 R3 R4 R5"), EVAL_JOBS (parallel
# evaluations, default 4), OUT (default outputs/review_grouped_epochs).
# Checkpoints go to outputs/review_epochs/<run>/checkpoints/ (one per epoch, 0.3-0.4 GB each).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
export MLFLOW_ALLOW_FILE_STORE=true
OUT=${OUT:-outputs/review_grouped_epochs}
RUNS=${RUNS:-"R2 R3 R4 R5"}
EVAL_JOBS=${EVAL_JOBS:-4}
mkdir -p "$OUT/eval" "$OUT/logs"

R50_ARGS="+experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16"
SLOW_ARGS="+experiment=baseline_ouc_cge trainer.early_stopping.patience=5 trainer.early_stopping.min_epochs=5 data.batch_size=32 data.num_workers=16 trainer.precision=bf16-mixed trainer.max_epochs=15"

want() { [[ " ${RUNS} " == *" $1 "* ]]; }

eval_one() {  # eval_one <ckpt> <model> <root> <name>
  local ckpt=$1 model=$2 root=$3 name=$4 ep
  ep=$(basename "$ckpt" | sed -E 's/^epoch=([0-9]+)-.*/\1/')
  python scripts/review/05_eval_checkpoints.py --model "$model" --ckpt "$ckpt" --root "$root" --split test \
      --mode identity --out "$OUT/eval" --tag "${name}_ep${ep}_test" > "$OUT/logs/eval_${name}_ep${ep}_test.log" 2>&1
  echo "  $name epoch $ep rc=$? $(grep -o '"lightning_test_accuracy": [0-9.]*' "$OUT/eval/summary_${name}_ep${ep}_test.json" 2>/dev/null)"
}
export -f eval_one
export OUT

run() {  # run <name> <model_key> <root> <hydra overrides...>
  local name=$1 model=$2 root=$3; shift 3
  if [ ! -f "$root/data_labels/train.csv" ] || [ ! -d "$root/videos/" ]; then
    echo "MISSING dataset root $root: dvc pull data/ouc-cge-320.dvc; python scripts/review/09_materialize_splits.py"; exit 2
  fi
  local t0=$(date +%s)
  echo "=== $name model=$model root=$root code=$(git rev-parse --short HEAD 2>/dev/null) start=$(date -Is)" | tee -a "$OUT/runs.log"
  python -m src.train "$@" trainer.checkpoint.save_top_k=-1 data.root="$root" \
      paths.output_dir="outputs/review_epochs/$name" mlflow.experiment_name=review_grouped_epochs \
      hydra.run.dir="outputs/review_epochs/$name/hydra" > "$OUT/logs/train_$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "$OUT/runs.log"
  if [ $rc -ne 0 ]; then tail -40 "$OUT/logs/train_$name.log"; echo "ABORT after $name" | tee -a "$OUT/runs.log"; exit $rc; fi
  ls "outputs/review_epochs/$name/checkpoints/"epoch=*.ckpt | \
      xargs -P "$EVAL_JOBS" -I{} bash -c 'eval_one "$@"' _ {} "$model" "$root" "$name" | sort | tee -a "$OUT/runs.log"
}

if want R2; then run R2_r50_grp27a resnet50 data/grp27a $R50_ARGS; fi
if want R3; then run R3_r50_grp27b resnet50 data/grp27b $R50_ARGS; fi
if want R4; then run R4_r50_grp17 resnet50 data/grp17 $R50_ARGS; fi
if want R5; then run R5_slow_grp27a slow data/grp27a $SLOW_ARGS; fi

python - "$OUT" <<'PY'
import glob, json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = {}
for f in glob.glob(str(out / "eval" / "summary_*_ep*_test.json")):
    d = json.load(open(f))
    run, ep = re.match(r"(.*)_ep(\d+)_test$", d["tag"]).groups()
    val = float(re.search(r"val_acc=([0-9.]+)\.ckpt", d["ckpt"]).group(1))
    rows.setdefault(run, []).append((int(ep), val, d["lightning_test_accuracy"], d["lightning_test_f1_macro"], d["evaluated"]))
lines = ["run,epoch,val_acc,test_acc,test_f1_macro,test_clips,selected_by_val,final_epoch"]
md = []
for run in sorted(rows):
    r = sorted(rows[run])
    best = max(r, key=lambda x: (x[1], -x[0]))[0]      # best validation accuracy, earliest on ties
    final = r[-1][0]
    md += [f"### {run}", "", "| epoch | val acc | test acc | test macro-F1 | |", "|---|---|---|---|---|"]
    for ep, val, acc, f1, n in r:
        note = " / ".join(t for t, c in (("selected by validation", ep == best), ("final epoch", ep == final)) if c)
        md.append(f"| {ep} | {val:.4f} | {acc:.4f} | {f1:.4f} | {note} |")
        lines.append(f"{run},{ep},{val:.4f},{acc:.6f},{f1:.6f},{n},{int(ep == best)},{int(ep == final)}")
    accs = [x[2] for x in r]
    md += ["", f"test accuracy over the {len(r)} epochs: min {min(accs):.4f}, max {max(accs):.4f}; "
               f"selected by validation {dict((x[0], x[2]) for x in r)[best]:.4f}, final epoch {accs[-1]:.4f}", ""]
(out / "epoch_curve.csv").write_text("\n".join(lines) + "\n")
(out / "epoch_curve.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
echo "EPOCH_CURVE_DONE" | tee -a "$OUT/runs.log"
