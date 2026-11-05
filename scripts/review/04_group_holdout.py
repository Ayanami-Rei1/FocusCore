#!/usr/bin/env python
"""Experiment 4: group-held-out evaluation with frozen features (no deep training).

Logistic regression on frozen center-frame ResNet50 features, evaluated with
  * StratifiedKFold(5)            - random split, same protocol as the published split
  * GroupKFold(5) by ID chain     - groups = runs of consecutive same-folder IDs (exp2)
  * GroupKFold(5) by scene cluster- groups = connected components (exp3)
and the majority-class baseline. Reports per-fold accuracy / macro-F1 and pooled confusion.

Usage: python scripts/review/04_group_holdout.py --cache outputs/review_cache --clusters docs/review/data/scene_clusters.csv --chains docs/review/data/exp2_chains.csv --out outputs/review_out
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_cache  # noqa: E402


def run_cv(X, y, splits, seed, name):
    accs, f1s, folds = [], [], []
    pooled = np.zeros((3, 3), dtype=np.int64)
    for k, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        acc = accuracy_score(y[te], pred); f1 = f1_score(y[te], pred, average="macro")
        maj = np.bincount(y[tr], minlength=3).argmax()
        accs.append(acc); f1s.append(f1)
        pooled += confusion_matrix(y[te], pred, labels=[0, 1, 2])
        folds.append(dict(scheme=name, fold=k, n_train=len(tr), n_test=len(te), acc=acc, f1_macro=f1,
                          majority_acc=float((y[te] == maj).mean()),
                          test_class_counts=np.bincount(y[te], minlength=3).tolist(),
                          train_class_counts=np.bincount(y[tr], minlength=3).tolist()))
    return dict(scheme=name, acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)), f1_mean=float(np.mean(f1s)),
                f1_std=float(np.std(f1s)), pooled_acc=float(np.trace(pooled) / pooled.sum()), pooled_confusion=pooled.tolist()), folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--clusters-multi", type=Path, default=None,
                    help="exp3b_clusters_by_thr.csv: run GroupKFold for every cluster_thr* column (robustness)")
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    c = load_cache(args.cache)
    keep = c["ok"]
    df = pd.DataFrame(dict(path=c["path"][keep], label=c["label"][keep], split=c["split"][keep]))
    df = df.merge(pd.read_csv(args.clusters)[["path", "cluster"]], on="path", how="left")
    df = df.merge(pd.read_csv(args.chains)[["path", "chain"]], on="path", how="left")
    assert df.cluster.notna().all() and df.chain.notna().all()
    X = c["feats"][keep][:, 1].astype(np.float32)
    y = df.label.values
    print(f"n={len(df)} class_counts={np.bincount(y).tolist()} clusters={df.cluster.nunique()} chains={df.chain.nunique()}")

    results, folds = [], []
    schemes = [
        ("stratified_random_5fold", list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(X, y))),
        ("group_5fold_by_chain", list(GroupKFold(5).split(X, y, groups=df.chain.values))),
        ("group_5fold_by_cluster", list(GroupKFold(5).split(X, y, groups=df.cluster.values))),
    ]
    if args.clusters_multi and args.clusters_multi.exists():
        mdf = df[["path"]].merge(pd.read_csv(args.clusters_multi), on="path", how="left")
        for col in [c_ for c_ in mdf.columns if c_.startswith("cluster_thr")]:
            schemes.append((f"group_5fold_by_{col}", list(GroupKFold(5).split(X, y, groups=mdf[col].values))))
    for name, sp in schemes:
        r, f = run_cv(X, y, sp, args.seed, name)
        results.append(r); folds += f
        print(f"\n{name}: acc={r['acc_mean']:.4f}±{r['acc_std']:.4f} f1={r['f1_mean']:.4f}±{r['f1_std']:.4f} pooled_acc={r['pooled_acc']:.4f}")
        print("  pooled confusion (rows=true low/mid/high):", r["pooled_confusion"])
        for ff in f:
            print(f"  fold{ff['fold']}: n_test={ff['n_test']} acc={ff['acc']:.4f} f1={ff['f1_macro']:.4f} maj={ff['majority_acc']:.4f} test_counts={ff['test_class_counts']}")
    maj = np.bincount(y).argmax()
    overall_major = float((y == maj).mean())
    pd.DataFrame(results).to_csv(args.out / "exp4_cv_results.csv", index=False)
    pd.DataFrame(folds).to_csv(args.out / "exp4_cv_folds.csv", index=False)
    with open(args.out / "exp4_summary.json", "w") as fh:
        json.dump(dict(majority_class=int(maj), majority_acc_overall=overall_major, results=results), fh, indent=2)
    print(f"\nmajority-class baseline (class {maj}): {overall_major:.4f}")


if __name__ == "__main__":
    main()
