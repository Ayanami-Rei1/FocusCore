# Scene-disjoint split of OUC-CGE

## Problem

The official OUC-CGE split assigns consecutive 10-second clips of the same recording session to
train, validation and test at random. 84 % of the test clips have a near-identical clip in the
training part (median RMSE between 32x32 thumbnails 2.7 on a 0-255 scale, against 87 for random
pairs). A 1-nearest-neighbour classifier on thumbnails alone reaches 96 % test accuracy on this
split, and the SLOW R50 model trained for the application reaches 0.9987. These numbers measure
scene memorisation, not engagement recognition.

## Method

Three frames (first, centre, last) of every clip are reduced to 32x32 thumbnails. Clips whose
thumbnails differ by at most a threshold RMSE are linked, and the connected components are treated
as scenes. Whole scenes go to one split with a 70 / 10 / 20 target and balanced classes. Two
thresholds were used: 27.5 (`grp27a`, `grp27b`, strict) and 17.5 (`grp17`, looser). The assignment
of every clip is stored in `data/splits/`.

## Results

Best checkpoint by validation accuracy, evaluated on the test part of the same split
(majority-class rate on `grp27a` is 0.39):

| model | split | best epoch | val acc | test acc | test macro-F1 |
|---|---|---|---|---|---|
| ResNet50, 1 frame | official random | 6 | 0.9961 | 0.9935 | 0.9936 |
| ResNet50, 1 frame | grp27a | 2 | 0.6005 | 0.5032 | 0.4592 |
| ResNet50, 1 frame | grp27b | 1 | 0.6314 | 0.5266 | 0.4553 |
| ResNet50, 1 frame | grp17 | 5 | 0.4013 | 0.6903 | 0.6309 |
| SLOW R50, 8 frames | grp27a | 1 | 0.5759 | 0.5180 | 0.4467 |

The SLOW run is not bit-reproducible (`deterministic: false`); repeated runs of the same recipe
score 0.509-0.521 on the test part. The application ships the SLOW R50 checkpoint trained on
`grp27a` (`outputs/review.dvc`, `R5_slow_grp27a/checkpoints/epoch=01-val_acc=0.5759.ckpt`),
exported with `python -m scripts.export_torchscript`.

## Reproducing

```bash
dvc pull data/ouc-cge-320.dvc
python scripts/scene_split/materialize_splits.py --names grp27a
RUNS="R5" bash scripts/scene_split/train_on_splits.sh
```

Details of the scripts and the other runs: `scripts/scene_split/README.md`.

The leakage analysis and the scene clustering were carried out by the project's scientific
consultant; the full study with the near-duplicate statistics, feature-based clustering and
temporal ablations is in https://github.com/dfbakin/focus-core.
