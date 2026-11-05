#!/usr/bin/env python
"""Small post-hoc tables for the report (per-clip errors vs nearest-train distance, coverage at thresholds).
Usage: python scripts/review/06_report_tables.py --out outputs/review_out --eval outputs/review_eval"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--eval", type=Path, required=True)
a = ap.parse_args()
multi = pd.read_csv(a.out / "exp3b_clusters_by_thr.csv")
te = multi[multi.split == "test"].reset_index(drop=True)
d = np.load(a.out / "exp3b_nn_test_to_train_dist.npy"); assert len(d) == len(te)
te["nn_dist"] = d
te[["path", "label", "nn_dist"]].to_csv(a.out / "exp3b_test_nn_dist.csv", index=False)
print("test clips:", len(te))
for t in (5, 10, 17.5, 27.5, 35, 55):
    print(f"  frac test with nearest train thumbnail RMSE <= {t}: {(d <= t).mean():.4f}")
print("  nn_dist pcts 50/90/95/99/max:", np.percentile(d, [50, 90, 95, 99]).round(2), round(float(d.max()), 2))
for tag in ["slow_test", "resnet50_test", "slow_test_shuffle", "slow_test_reverse", "slow_test_static_center", "slow_test_static_first"]:
    p = pd.read_csv(a.eval / f"preds_{tag}.csv")
    err = p[p.label != p.pred].merge(te[["path", "nn_dist"]], on="path", how="left")
    print(f"\n{tag}: n={len(p)} errors={len(err)}")
    print(err[["path", "label", "pred", "p0", "p1", "p2", "nn_dist"]].round(3).to_string(index=False))
nn = pd.read_csv(a.out / "exp2b_nearest_train_neighbour.csv")
m = nn.split == "test"
print("\nexp2b feature-NN test rows", int(m.sum()), "label agreement", round(float((nn[m].nn_label == nn[m].label).mean()), 4))
ct = pd.read_csv(a.out / "exp3b_cluster_table.csv")
print("\nclusters thr27.5:", len(ct), "size>=5:", int((ct["size"] >= 5).sum()), "pure:", int((ct.purity == 1).sum()), "clips in pure:", int(ct.loc[ct.purity == 1, "size"].sum()),
      "n_splits==3:", int((ct.n_splits == 3).sum()), "clips in n_splits==3:", int(ct.loc[ct.n_splits == 3, "size"].sum()))
for col in ["cluster_thr10", "cluster_thr17.5", "cluster_thr27.5"]:
    g = multi.groupby(col); sizes = g.size(); trn = g["split"].apply(lambda x: (x == "train").any())
    tcov = multi[multi.split == "test"][col].map(trn).mean()
    pur = g["label"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    print(f"{col}: n={len(sizes)} median={sizes.median()} max={sizes.max()} clusters>=5: {int((sizes >= 5).sum())} "
          f"clips in clusters>=5: {int(sizes[sizes >= 5].sum())} weighted purity={(pur * sizes).sum() / sizes.sum():.4f} test clips with train in cluster={tcov:.4f}")
cv = pd.read_csv(a.out / "exp4_cv_results.csv")
print("\n" + cv[["scheme", "acc_mean", "acc_std", "f1_mean", "f1_std", "pooled_acc"]].round(4).to_string(index=False))
