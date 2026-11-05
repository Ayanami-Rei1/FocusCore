# Scene-disjoint OUC-CGE splits

The official OUC-CGE split is a random shuffle of consecutive 10-second clips cut from 21 recorded
sessions, so almost every test clip has a near-identical training clip and a model scores 0.99 by
recognising the scene. The scripts in this directory train the project's own recipes on splits in
which no scene is shared between train, validation and test. Background and results:
`docs/scene_disjoint_split.md`.

## Quick start

```bash
dvc pull data/ouc-cge-320.dvc                              # 7,701 clips at 320p, 4.4 GB
python scripts/scene_split/materialize_splits.py           # data/grp27a, data/grp27b, data/grp17
RUNS="R5" bash scripts/scene_split/train_on_splits.sh      # SLOW R50 on grp27a (the app model)
```

Each dataset root holds `data_labels/{train,val,test}.csv` in the project format and a `videos`
symlink to `../ouc-cge-320/videos`, so `OUCCGEDataset` reads it through `data.root=`:

```bash
export MLFLOW_ALLOW_FILE_STORE=true                        # MLflow >= 3.17
python -m src.train +experiment=baseline_ouc_cge data.root=data/grp27a trainer.max_epochs=15 \
    trainer.early_stopping.patience=5 trainer.early_stopping.min_epochs=5 data.batch_size=32
```

Checkpoints go to `outputs/review/<run>/checkpoints/` (best two by validation accuracy plus
`last.ckpt`, which is a copy of the later of the two, not the final epoch); summaries, per-clip
predictions and logs to `outputs/review_grouped/`. The training needs a GPU.

## The splits

A scene is a connected component of clips linked by near-identical frames: two clips are linked when
the RMSE between 32x32 thumbnails of any of their first / centre / last frames is at most a threshold
(0-255 scale). Whole components go to one split, target 70 / 10 / 20, classes balanced.

| root | threshold | train clips (scenes) | val | test | notes |
|---|---|---|---|---|---|
| `data/grp27a` | 27.5 | 5386 (83) | 771 (9) | 1544 (9) | strict; the test part is dominated by one large mixed-label scene |
| `data/grp27b` | 27.5 | 5187 (84) | 784 (7) | 1730 (10) | strict; different test scenes (2 % overlap with grp27a's test) |
| `data/grp17` | 17.5 | 5391 (437) | 770 (26) | 1540 (67) | looser; 30 % of test clips still have a train clip within 27.5 |
| `data/ouc-cge-320` | - | 6160 | 770 | 771 | the official random split (control) |

The assignment of every clip is `data/splits/split_<name>.csv` (`path,label,split,cluster`), the
leakage statistics of each split are in `data/splits/split_<name>_summary.json`. Four of the 7,705
listed clips failed the 320p transcode and are in no split.

## Runs

| run | model | root | recipe |
|---|---|---|---|
| R1 | ResNet50, 1 frame | `data/ouc-cge-320` | `+experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16` |
| R2 / R3 / R4 | ResNet50, 1 frame | `data/grp27a` / `grp27b` / `grp17` | same |
| R5 | SLOW R50, 8 frames x stride 8 | `data/grp27a` | `+experiment=baseline_ouc_cge trainer.max_epochs=15 trainer.early_stopping.patience=5 trainer.early_stopping.min_epochs=5 data.batch_size=32 trainer.precision=bf16-mixed` |

Seed 42 in both presets. R1-R4 are bit-reproducible on an RTX 5090; R5 is not (`deterministic: false`),
its test accuracy varies by about 0.01 between runs.

## Scripts

| script | purpose |
|---|---|
| `materialize_splits.py` | dataset roots from `data/splits/split_<name>.csv`; no randomness |
| `train_on_splits.sh` | runs R1-R5 through `python -m src.train`, exports the MLflow metrics, evaluates the best checkpoint and `last.ckpt` per clip |
| `evaluate_checkpoint.py` | evaluate a checkpoint through the project's `test_step`, per-clip predictions, temporal ablations |
| `common.py` | manifest parsing, dataset-root writer |

## Environment

- Python 3.12 and `requirements.txt` (or `docker/training/Dockerfile`); `dvc` for the clips.
- MLflow >= 3.17 refuses the file store that `mlflow.tracking_uri: null` resolves to;
  `train_on_splits.sh` exports `MLFLOW_ALLOW_FILE_STORE=true`, do the same for direct calls.
