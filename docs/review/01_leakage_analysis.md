# OUC-CGE: is the ~100% test accuracy a leaked / saturated split?

## Summary

**Verdict: yes.** The published train/val/test split of OUC-CGE is a random clip-level split of a
small number of fixed-camera classroom scenes. Almost every test clip has a near pixel-identical
sibling in train, and a trivial nearest-neighbour on 32x32 thumbnails already reaches the student's
accuracy. The SLOW R50 result is reproduced exactly, but the model does not use temporal
information, and any classifier collapses towards chance when whole scenes are held out.
The task *on this split* is memorisation of scenes, not engagement recognition.

The five numbers that matter:

| # | Measurement | Value |
|---|---|---|
| 1 | Nearest train clip of a test clip, 32x32-thumbnail RMSE (0-255 scale): median / p95 | **2.7 / 15.3** (random pairs: median 87) |
| 2 | Test clips with a train clip at RMSE <= 10 (near pixel-identical) / <= 27.5 (same scene) | **84.1% / 98.7%** |
| 3 | 1-NN on raw thumbnails, no learning: test accuracy | **96.1%** (SLOW R50: 99.87%, single-frame ResNet50: 99.35%) |
| 4 | SLOW R50 with frames shuffled / reversed / one static frame repeated 8x | **99.6% / 99.7% / 98.8%** (intact: 99.87%) |
| 5 | Frozen-feature logistic regression, random 5-fold vs scene-held-out 5-fold | **91.5%** vs **47-62%** (majority class 39.3%) |

Everything below was computed on the rented GPU host; nothing heavy ran locally.

## Setup

- Host: vast.ai, RTX 5090 32 GB, 192 CPUs, 1 TB RAM; `source /venv/main/bin/activate`
  (torch 2.11.0+cu128, torchvision 0.26.0, lightning 2.6.6, cv2 5.0.0, sklearn 1.9.1, torchmetrics 1.9.0, scipy 1.18.1).
- Repo copy: `/workspace/focus-core` (rsync of branch `review/ml-validity-and-bugs`; review scripts under `scripts/review/`).
- Data: `/workspace/focus-core/data/ouc-cge-320` (320p short side, 30 fps). Manifests `data_labels/{train,val,test}.csv`:
  6164 / 770 / 771 rows after dropping the malformed `videos/anonymity/a.mp4` row (same parsing as
  `OUCCGEDataset._load_manifest`). 7701 of 7705 clips decode; the 4 unreadable ones are the known transcode
  failures (`videos/high/view{2521,2630,2631,2632}.mp4`, all in train). 117 clips have fewer than 64 frames
  (reported == actual frame count for all clips), i.e. are dropped by the SLOW 64-frame window filter:
  98 train, 10 val, 9 test.
- Checkpoints (DVC-pulled on the host): SLOW R50 `outputs/baseline_ouc_cge/2026-05-31_15-35-39/checkpoints/epoch=30-val/accuracy=1.0000.ckpt`
  (epoch 30, global_step 5921 = MLflow run `f66cb503...`, the run with test acc 0.9974) and single-frame ResNet50
  `outputs/resnet50_single_frame_best/accuracy=0.9961.ckpt` (epoch 6).
- Commands (on the host, from `/workspace/focus-core`; scripts are in this repo under `scripts/review/`):
  ```
  python scripts/review/01_frame_cache.py --root data/ouc-cge-320 --out /workspace/review_cache --workers 64
  bash   scripts/review/run_02_04.sh      # exp 2 (near-duplicates), exp 3 feature-based clustering (superseded), exp 4
  bash   scripts/review/run_03b_04.sh     # exp 3 revised (thumbnail scene clusters) + exp 4 at every threshold
  bash   scripts/review/run_05_06.sh      # exp 5 (reproduce checkpoints) + exp 6 (temporal ablations), GPU, sequential
  python scripts/review/06_report_tables.py --out /workspace/review_out --eval /workspace/review_eval
  ```
  Results were rsynced back into `docs/review/data/` (logs in `docs/review/data/logs/`).
- Seeds: numpy `default_rng(0)` for all pair sampling / cluster picks, `LogisticRegression(random_state=0)`,
  `StratifiedKFold(random_state=0)`, `L.seed_everything(42)` for the checkpoint evaluations.

## Experiment 1: frame cache

Method (`01_frame_cache.py`): every clip is decoded sequentially with cv2 (exact frame count), and the first,
center and last frames are kept. Stored per clip: a 32x32 RGB uint8 thumbnail of each of the three frames
(`cv2.INTER_AREA`), and a 2048-d global-pool feature of each frame from torchvision ResNet50 with
`IMAGENET1K_V1` weights (the same weights the student's `resnet50` backbone starts from), computed with the
ImageNet eval protocol (256 resize, 224 center crop), fp16 autocast on the GPU. One npz per split.

| | train | val | test |
|---|---|---|---|
| manifest rows | 6164 | 770 | 771 |
| readable | 6160 | 770 | 771 |
| < 64 frames (dropped by SLOW window) | 98 | 10 | 9 |

Clip length percentiles (frames; 1/5/25/50/75/95/99): 9 / 77 / 158 / 165 / 300 / 355 / 5068. Most clips are
~5.5 s, not 10 s; a tail of clips is much longer than 30 s (the "truncate to 30 s" step did not apply to them).
Decode: 13 min with 64 workers; features: 17 s.

## Experiment 2: near-duplicate structure

### 2a. Pair similarity distributions (`exp2a_pair_similarity.csv`, `exp3b_random_pair_pcts.csv`)

Three pair populations: same-class ID-adjacent pairs (`viewN`, `viewN+1` in the same class folder; 7696 pairs),
20000 random same-class non-adjacent pairs, 20000 random cross-class pairs.

ImageNet-feature cosine of the center frames:

| pair type | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| adjacent (N, N+1) | 0.773 | 0.803 | 0.849 | **0.907** | 0.955 | 0.983 | 1.000 |
| random same-class | 0.748 | 0.779 | 0.817 | 0.842 | 0.867 | 0.909 | 0.939 |
| random cross-class | 0.752 | 0.779 | 0.816 | 0.840 | 0.863 | 0.899 | 0.925 |

The last-frame(N) to first-frame(N+1) cosine is not higher than center-to-center (median 0.910 vs 0.907), so
consecutive IDs are *not* generally consecutive in time. Note how compressed the whole scale is: random
cross-class pairs have median cosine 0.84. A global ImageNet descriptor sees "a classroom from above" in every
clip and discriminates scenes poorly. This matters for the clustering choice below.

Thumbnail distance is far more discriminative. Clip distance = min over the 3x3 (first/center/last) frame
pairs of the RMSE between 32x32 RGB thumbnails (0-255 scale); "min" so that two clips of one scene link even if
only one of their sampled frames coincides:

| pair type | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|
| random same-class | 20000 | 6.5 | 28.5 | 40.3 | 66.6 | 86.3 | 93.8 | 103.5 |
| random cross-class | 20000 | 29.5 | **35.0** | 46.2 | 72.6 | 86.9 | 94.5 | 104.7 |
| adjacent (N, N+1) | 7696 | 0.4 | 0.5 | 0.6 | **4.4** | **21.4** | 88.5 | 100.9 |
| **val clip to nearest train clip** | 770 | 0.4 | 0.4 | 0.5 | 1.3 | **2.9** | 7.0 | 17.8 |
| **test clip to nearest train clip** | 771 | 0.4 | 0.4 | 0.6 | 1.3 | **2.7** | 7.0 | 15.3 |

Reading: ID-adjacent pairs are bimodal. About half are the same scene (p25 = 4.4, i.e. visually identical),
the other half are different rooms (p75 = 88, the random level). 43.5% of adjacent pairs exceed the p99
cosine of random cross-class pairs; 34% have last(N)->first(N+1) cosine > 0.95. So the supervisor's "N±1
sibling" picture is right for roughly half the IDs, but the real structure is broader: the nearest train
neighbour of a test clip is at RMSE 2.7 (median), closer than the typical adjacent pair, and only 20% of test
clips have their nearest train neighbour at an adjacent ID (29% in the same ID chain). The same scenes recur
across the whole ID range (e.g. test `high/view1118` and train `high/view2203` are the same image, d=1; see
`exp3b_test_nn_pairs.png`). Within a clip, first/center/last frames are near-identical (fixed camera, seated
groups): `exp2a_adjacent_pairs_sheet.png`.

Coverage of the test split by near-duplicates in train (`exp3b_test_nn_dist.csv`):

| nearest-train RMSE <= | 5 | 10 | 17.5 | 27.5 | 35 | 55 |
|---|---|---|---|---|---|---|
| fraction of test clips | 64.5% | **84.1%** | 96.0% | **98.7%** | 99.9% | 100% |

For scale: RMSE 10 on 32x32 thumbnails is the same frame up to compression noise and small motion; 35 is the
1st percentile of random cross-class pairs. 99.9% of test clips have a train clip closer than that, and the
farthest any test clip is from train is 36.3 (median random pair: 87). In other words, for every test clip
there is a train clip from the same scene.

### 2b. Nearest-train-neighbour classifiers (`exp2b_knn.csv`, `exp2b_nearest_train_neighbour.csv`)

Query = each val/test clip; database = train clips; center frames.

| split | descriptor | neighbours allowed | 1-NN acc | 5-NN majority acc | NN score p50 / p95 |
|---|---|---|---|---|---|
| test | 32x32 thumbnail RMSE | any train | **0.961** | 0.946 | 19.3 / 38.4 |
| test | ImageNet cosine | any train | 0.946 | 0.934 | 0.968 / 1.000 |
| test | thumbnail RMSE | different ID chain only | 0.913 | 0.901 | 23.9 / 41.7 |
| test | ImageNet cosine | different ID chain only | 0.889 | 0.874 | 0.959 / 0.997 |
| val | thumbnail RMSE | any train | 0.958 | 0.935 | 19.6 / 38.5 |
| val | ImageNet cosine | any train | 0.948 | 0.934 | 0.967 / 0.999 |
| val | thumbnail RMSE | different ID chain only | 0.927 | 0.897 | 23.1 / 40.5 |

(Here the NN score uses center-to-center thumbnail RMSE, hence the larger medians than the min-over-frames
table above.) A 3072-number thumbnail and `argmin` reproduce 96% of the test labels with no training at all.
The MLflow curve (val acc 0.94 after epoch 0) is exactly what this predicts.

"Different ID chain": chains are maximal runs of consecutive same-folder IDs whose adjacent center-feature
cosine >= 0.9248. The threshold is the 99th percentile of random cross-class cosine (so a kept link is
almost never a chance similarity); sweep in `exp2b_chain_threshold_sweep.csv` (0.5 -> 5 chains of a whole
folder each; 0.9248 -> 4353 chains, 43% of links kept; 0.95 -> 5490 chains). With cosine this compressed the
chains are deliberately conservative (median size 1), so "different chain" removes only the obvious N±1
siblings; accuracy still stays at 89-93%: the leak is not confined to adjacent IDs.

### 2c. Frozen-feature logistic regression on the published split (`exp2_summary.json`)

`StandardScaler` + `LogisticRegression(C=1, max_iter=3000)` on the 2048-d center-frame features of the 6160
readable train clips.

| split | n | accuracy | macro-F1 | confusion (rows = true low/mid/high) |
|---|---|---|---|---|
| train | 6160 | 1.000 | 1.000 | diagonal |
| val | 770 | 0.908 | 0.906 | [289 12 6] [14 177 8] [20 11 233] |
| test | 771 | **0.923** | 0.919 | [310 8 7] [14 168 7] [10 13 234] |

Majority-class baseline: val 0.399, test 0.422. A linear probe on frozen ImageNet features, which have never
seen a video and encode nothing about engagement, gets 92% on this split.

## Experiment 3: scene clustering

### 3a. First attempt, ImageNet-feature cosine (superseded; `exp3_*` files)

Connected components of `cosine >= 0.9248` give one giant component with 7407 of 7701 clips (purity 0.39)
and 163 singletons; at 0.95, 2182 clusters with the largest holding 14% (`exp3_threshold_sweep.csv`). The
descriptor is simply too coarse (see 2a), and single linkage chains everything together. The one
informative cluster it found is id 4 (`exp3_cluster_3_id4_n49.png`): 49 clips, IDs low/545-592, one table,
all "low", split 39/5/5 across train/val/test. I abandoned this descriptor for clustering.

### 3b. Thumbnail scene clusters (`03b_scene_clusters_thumb.py`; `scene_clusters.csv`, `exp3b_*`)

Groups = connected components of `d(i, j) <= thr` with the min-over-frame-pairs thumbnail RMSE from 2a.
Why single linkage: a leakage-free group split must guarantee that **no** cross-group pair is a near
duplicate; single linkage is the only linkage with that guarantee at a given threshold (average/complete
linkage can leave cross-cluster pairs below thr). The price is transitive over-merging, which the sweep exposes.

Threshold sweep (7701 clips; `exp3b_threshold_sweep.csv`):

| thr | clusters | singletons | largest (frac) | median size | label purity (clip-weighted) | clusters 100% pure | clips in pure clusters | clusters spanning >1 split | test clips with a train clip in their cluster |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 1626 | 836 | 246 (3.2%) | 1 | **1.000** | 100% | 100% | 473 | 87.8% |
| 12.5 | 1127 | 490 | 256 (3.3%) | 2 | 1.000 | 100% | 100% | 407 | 93.0% |
| 15 | 784 | 287 | 407 (5.3%) | 2 | 1.000 | 100% | 100% | 337 | 96.2% |
| **17.5** | 530 | 165 | 410 (5.3%) | 4 | **1.000** | 100% | 100% | 273 | **97.9%** |
| 20 | 370 | 104 | 490 (6.4%) | 5 | 0.960 | 99.5% | 92.7% | 209 | 98.6% |
| 22.5 | 250 | 59 | 491 (6.4%) | 8 | 0.922 | 97.6% | 82.0% | 155 | 99.0% |
| 25 | 165 | 39 | 1163 (15.1%) | 11 | 0.869 | 97.0% | 73.9% | 108 | 99.5% |
| **27.5** | 101 | 20 | 1163 (15.1%) | 17 | 0.791 | 89.1% | 45.1% | 69 | **99.6%** |
| 30 | 54 | 11 | 1990 (25.8%) | 24 | 0.655 | 77.8% | 17.1% | 38 | 99.7% |
| 35 | 19 | 1 | 2914 (37.8%) | 108 | 0.518 | 47.4% | 5.4% | 18 | 100% |
| 40 | 8 | 0 | 3423 (44.5%) | 493 | 0.451 | 12.5% | 0.6% | 8 | 100% |

Threshold choice. `scene_clusters.csv` uses **27.5**, chosen by the rule written into the script: the largest
candidate that is below the 1st percentile of random cross-class distances (35.0, so fewer than 1% of
unrelated pairs would be linked directly) and whose largest component holds <= 25% of the clips. At 27.5:
101 clusters, 64 of size >= 5 holding 99.1% of clips, 49 clusters (7386 clips, 95.9%) span all three splits,
and 99.6% of test clips share a cluster with a train clip. Cluster assignments at every threshold are in
`exp3b_clusters_by_thr.csv`, because the conclusion should not hinge on one number.

What the purity column says. Up to thr = 17.5 (near-identical frames) **every** cluster is 100% label-pure:
clips that look the same always carry the same label. Above ~20 impure clusters appear; the contact sheets
show two causes. (i) Same scene, label changing over time: cluster id 11 (318 clips,
`exp3b_cluster_0_id11_n318.png`) is one table with one person in an orange jacket, labelled low in some clips,
medium in others, and high in one. (ii) Transitive merging of distinct but similar tables at the higher
thresholds. So the label is *mostly* but not strictly a property of the scene; some sessions were annotated
with changing engagement. Either way the test clips are covered: a test clip's nearest train clip is at
median RMSE 2.7, far inside the regime where labels are 100% consistent.

Visual sanity checks (all in `docs/review/data/`): `exp3b_test_nn_pairs.png` (16 random test clips beside
their nearest train clip: same table, same people, same lighting, d = 1-13, labels always equal, IDs far
apart), `exp3b_cluster_3_id25_n204.png` (one table, 204 clips, all "high", IDs high/506-2217, 164/23/17 across
train/val/test), `exp3b_cluster_2_id44_n163.png`, `exp3b_cluster_1_id5_n214.png` (merged mid/high scene),
`exp3b_cluster_4..7_*.png` (smaller clusters), plus the two sheets named above.

Chains vs clusters: 99.8% of the ID chains from 2b fall inside a single thumbnail cluster, i.e. the clusters
are a strict coarsening of ID adjacency, as they should be.

## Experiment 4: group-held-out evaluation with frozen features (`04_group_holdout.py`; `exp4_*`)

Same classifier as 2c (scaler + logistic regression on center-frame features), 7701 clips, 5 folds, groups =
thumbnail clusters at each threshold; compared with a stratified random 5-fold (the published protocol) and
ID-chain groups. Majority class (low) = 39.3%; uniform chance = 33.3%.

| scheme | acc mean ± std | macro-F1 mean ± std | pooled acc | pooled confusion (rows = true low/mid/high) |
|---|---|---|---|---|
| stratified random 5-fold (as published) | **0.915 ± 0.004** | 0.912 ± 0.004 | 0.915 | [2825 115 88] [115 1819 107] [103 126 2403] |
| group by ID chain | 0.873 ± 0.022 | 0.868 ± 0.021 | 0.873 | [2717 175 136] [229 1646 166] [160 115 2357] |
| group by scene cluster, thr 10 | 0.781 ± 0.039 | 0.769 ± 0.040 | 0.781 | [2504 319 205] [392 1345 304] [247 220 2165] |
| thr 12.5 | 0.733 ± 0.045 | 0.716 ± 0.040 | 0.733 | |
| thr 15 | 0.696 ± 0.013 | 0.679 ± 0.018 | 0.696 | |
| **thr 17.5** (all clusters label-pure) | **0.621 ± 0.048** | 0.592 ± 0.043 | 0.621 | [2138 474 416] [699 823 519] [425 385 1822] |
| thr 20 | 0.592 ± 0.028 | 0.561 ± 0.041 | 0.592 | |
| thr 22.5 | 0.533 ± 0.074 | 0.506 ± 0.060 | 0.533 | |
| thr 25 | 0.500 ± 0.063 | 0.477 ± 0.068 | 0.500 | |
| **thr 27.5** (`scene_clusters.csv`) | **0.472 ± 0.075** | 0.423 ± 0.078 | 0.472 | [1520 726 782] [820 489 732] [544 466 1622] |
| thr 30 | 0.436 ± 0.067 | 0.382 ± 0.054 | 0.439 | |
| thr 35 | 0.397 ± 0.076 | 0.358 ± 0.039 | 0.402 | |
| majority class | 0.393 | 0.188 | | |

Per-fold numbers (sizes, class counts, per-fold majority rate) are in `exp4_cv_folds.csv`.

Reading. The accuracy falls monotonically with the strictness of the grouping. Merely refusing to train on
near pixel-identical copies of the test clips (thr 10) costs 13 points; holding out label-pure scene groups
(thr 17.5) costs 29 points; holding out whole scenes (thr >= 25) brings the same classifier that scores 91.5%
under the published protocol down to 47-50%, within ~10 points of the majority class and with a confusion
matrix that is close to uniform. The gap between row 1 and the scene-held-out rows is the leak.

Two honest caveats on the strict end of the table. First, at thr >= 25 the groups are large and
label-skewed, so individual folds lose most of one class (e.g. thr 27.5 fold 4 has only 140 "high" clips,
majority rate 9%); that depresses accuracy beyond the pure leakage effect, which is why the whole curve is
shown rather than one row. Second, the classifier is a linear probe on frozen ImageNet features, so these
rows are a lower bound on what a fine-tuned video model could do across scenes, not an estimate of it. The
rows that matter for the leakage question are the comparison within the same classifier.

Implication of the purity result: at every threshold where near-identical clips are grouped (<= 17.5), clusters
are 100% label-pure and 98% of test clips have a train clip in their cluster. The label is, to first order,
a property of the session/table; a within-session random split hands the model the answer.

## Experiment 5: reproducing the student's numbers (`05_eval_checkpoints.py`, `run_05_06.sh`)

Method: `OUCCGEDataset(root="data/ouc-cge-320", split=..., fps=[3.75], num_frames=[8])` with
`get_val_transforms()` (center window, as in the project), backbone via `create_model(..., pretrained=False)`,
`VideoClassificationModule.load_from_checkpoint(ckpt, model=backbone)`, `L.Trainer(precision="bf16-mixed").test`
with batch 16, 16 workers. Per-clip labels/predictions are captured by wrapping `module._shared_step` and the
logits by a forward hook on the backbone, so the numbers below come from the project's own `test_step`.
ResNet50 uses the project's single-frame pathway `fps=[30], num_frames=[1]` (window 1, nothing dropped).

| checkpoint | split | manifest | evaluated | dropped (window 64) | accuracy (Lightning) | macro-F1 | errors | confusion (rows = true low/mid/high) |
|---|---|---|---|---|---|---|---|---|
| SLOW R50 epoch 30 | test | 771 | **762** | 9 | **0.9987** | 0.9984 | 1 | [324 0 0] [0 188 1] [0 0 249] |
| SLOW R50 epoch 30 | val | 770 | 760 | 10 | **1.0000** | 1.0000 | 0 | [305 0 0] [0 199 0] [0 0 256] |
| SLOW R50, fp32 instead of bf16 | test | 771 | 762 | 9 | 0.9987 | 0.9984 | 1 | same |
| ResNet50 single frame, epoch 6 | test | 771 | **771** | 0 | **0.9935** | 0.9936 | 5 | [321 1 3] [0 189 0] [0 1 256] |

The student's MLflow numbers are reproduced: val 1.0000 exactly; test 0.9987 here vs 0.9974 logged. The logged
value equals 767/769, so the training run evaluated 769 clips whereas the current data copy yields 762 after
the window filter (the project caches the filtered manifest in `data_labels/.cache/`, and that cache was
built on the student's copy of the data); the difference is 7 clips and does not change anything. The single
SLOW error is `mid/view1607` predicted "high" (p = 0.93); its nearest train clip is at RMSE 4.6. No duplicate
paths were produced by the dataset's skip-to-next-sample fallback. Mean max-softmax: 0.999 (SLOW), 0.997 (ResNet50).
Per-clip predictions: `docs/review/data/test_predictions_slow.csv`, `docs/review/data/test_predictions_resnet50.csv`
(and `eval/preds_*.csv` for every run).

The single-frame ResNet50 at 99.35% on the full 771-clip test set confirms the student's statement that a
still image suffices.

## Experiment 6: temporal ablations of the SLOW checkpoint (`--mode`, test split, 762 clips)

The dataset wrapper edits the time axis of the (C, T=8, H, W) tensor after the project's transforms.

| input | accuracy | macro-F1 | errors | test loss | mean max-prob |
|---|---|---|---|---|---|
| intact (8 frames, stride 8) | **0.9987** | 0.9984 | 1 | 0.0046 | 0.999 |
| frames in random order (per-clip fixed seed) | 0.9961 | 0.9957 | 3 | 0.0070 | 0.998 |
| frames reversed | 0.9974 | 0.9969 | 2 | 0.0060 | 0.999 |
| center frame repeated 8x (static clip) | 0.9882 | 0.9874 | 9 | 0.0356 | 0.997 |
| first frame repeated 8x (static clip) | 0.9882 | 0.9871 | 9 | 0.0296 | 0.997 |

Destroying temporal order costs 1-2 clips; removing motion entirely (a single frame tiled 8 times, a
distribution the model never saw in training) costs 8 clips out of 762. The 3D model behaves as an image
classifier. Together with experiment 5 (a 2D ResNet50 on one frame is as good) and experiment 2 (1-NN on
thumbnails is nearly as good), the SLOW result carries no evidence that temporal modelling contributes to
engagement recognition on this split.

## Threats to validity

- Scene descriptors measure appearance, not engagement cues. A near-identical thumbnail does not by itself
  prove that the label was obtained "for free"; the argument rests on the conjunction: (a) near-identical
  siblings for 84-99% of test clips, (b) label purity 100% among near-identical clips, (c) a no-learning
  1-NN at 96%, (d) no temporal sensitivity, and (e) the collapse of the same classifier under scene-held-out
  folds. Any one alone would be suggestive; together they leave scene memorisation as the only simple explanation.
- No deep model was trained under a group split (by design: the brief was to avoid deep training). Experiment
  4 shows what a frozen-feature probe does, which is a lower bound for a fine-tuned model. The right next step
  is to retrain SLOW/ResNet50 with `GroupKFold` on `scene_clusters.csv` (or `exp3b_clusters_by_thr.csv` at
  17.5) and report that number; I would expect it to be far below 99% but above the 47-62% of the probe.
- Single linkage over-merges above thr ~20 (some impure clusters are two different tables chained through
  similar-looking clips), and the large groups make the strict folds label-imbalanced. This is why the full
  threshold curve is reported; the leak conclusion holds at every threshold from 10 upwards.
- Labels do change within some sessions (cluster id 11), so "label = property of the session" is not
  universally true in OUC-CGE; it is true at the granularity that matters for the leak (near-identical clips).
- The ID-chain definition uses ImageNet cosine (p99 of random cross-class = 0.9248), a poorly discriminative
  descriptor; chains are therefore conservative (median size 1), and the chain-grouped CV (87.3%) understates the
  effect of ID adjacency. Thumbnail-based chains would be tighter but were not needed once the clusters existed.
- Features and thumbnails are from three frames per clip; a scene that appears only mid-clip could be missed,
  which would make the duplicate counts conservative, not inflated.
- Random pairs are sampled (20000 each, seed 0); the logistic regression uses C = 1 with no tuning; kNN uses
  plain `argmin`. None of these choices favour the leak hypothesis.
- The evaluation set for SLOW differs from the student's run by 7 clips (762 vs 769), see experiment 5.

## Files produced

Scripts (`scripts/review/`): `common.py`, `01_frame_cache.py`, `02_near_duplicates.py`, `03_scene_clusters.py`
(feature-based, superseded; also provides the shared cluster-table / contact-sheet helpers),
`03b_scene_clusters_thumb.py`, `04_group_holdout.py`, `05_eval_checkpoints.py`, `06_report_tables.py`,
`run_02_04.sh`, `run_03b_04.sh`, `run_05_06.sh`.

Data (`docs/review/data/`):
- `scene_clusters.csv` (path, label, split, cluster; thumbnail clusters at thr 27.5), `exp3b_clusters_by_thr.csv`
  (cluster id at every sweep threshold), `exp3b_cluster_table.csv`, `exp3b_threshold_sweep.csv`,
  `exp3b_random_pair_pcts.csv`, `exp3b_summary.json`, `exp3b_test_nn_dist.csv` (+ `.npy`).
- `exp2a_pair_similarity.csv`, `exp2b_knn.csv`, `exp2b_nearest_train_neighbour.csv`, `exp2b_chain_threshold_sweep.csv`,
  `exp2_chains.csv`, `exp2_summary.json`.
- `exp4_cv_results.csv`, `exp4_cv_folds.csv`, `exp4_summary.json`.
- `test_predictions_slow.csv`, `test_predictions_resnet50.csv`; `eval/preds_*.csv` and `eval/summary_*.json`
  for all 8 evaluation runs (SLOW test/val/fp32, ResNet50 test, 4 ablations).
- Contact sheets: `exp2a_adjacent_pairs_sheet.png`, `exp3b_test_nn_pairs.png`, `exp3b_cluster_{0..7}_*.png`,
  and the superseded `exp3_cluster_*.png`; superseded feature-based clustering tables `exp3_*.csv/json`.
- `logs/`: host logs of every run.

Host artefacts not copied (large): `/workspace/review_cache/frames_{train,val,test}.npz` (thumbnails + features, 148 MB).
