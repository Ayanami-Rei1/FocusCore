#!/usr/bin/env python
"""Experiment 3 (revised): scene clustering from 32x32 thumbnails instead of ImageNet features.

Why: exp3 with ImageNet-feature cosine collapses into one giant component (all classroom clips
look alike to a global ImageNet descriptor: random cross-class pairs have median cosine 0.84).
Raw thumbnail RMSE separates "same scene" from "different scene" far better (exp2a).

Clip-to-clip distance d(i,j) = min over the 3x3 (first/center/last) frame pairs of the RMSE
between 32x32 RGB thumbnails (0-255 scale). "min" so that two clips of the same scene link even if
only one of their sampled frames coincides (e.g. consecutive segments: last(i) ~ first(j)).
Groups = connected components of d <= thr (single linkage; guarantees no cross-group pair is
within thr, which is what a leakage-free split needs). thr is chosen from the sweep: the largest
candidate that is below the 1st percentile of random cross-class pair distances (so <1% of
unrelated pairs would be linked directly) and yields a largest component <= --max-frac of clips.

Outputs: exp3b_random_pair_pcts.csv, exp3b_threshold_sweep.csv, scene_clusters.csv,
exp3b_clusters_by_thr.csv (cluster id per clip for every candidate threshold; with
scene_clusters.csv the input of 07_make_grouped_splits.py), exp3b_cluster_table.csv,
exp3b_summary.json, contact sheets exp3b_cluster_*.png and exp3b_test_nn_pairs.png
(random test clips next to their nearest train clip).

Needs only the thumbnails of the frame cache (01_frame_cache.py, also with --thumbs-only).
The distance matrix is computed with torch on the GPU if one is visible, else on the CPU.

Usage:
  python scripts/review/03b_scene_clusters_thumb.py --cache outputs/review_cache --out outputs/review_out
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_cache  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402
_m = import_module("03_scene_clusters")
cluster_table, sweep_stats, contact_sheet = _m.cluster_table, _m.sweep_stats, _m.contact_sheet


def pairwise_min_rmse(T):
    """T: (N,3,D) float32 thumbnails. Returns (N,N) min-over-frame-pairs RMSE (float32)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(T).to(dev)
    N, F, D = X.shape
    best = None
    for a in range(F):
        for b in range(F):
            d = torch.cdist(X[:, a], X[:, b])  # L2 over D dims
            best = d if best is None else torch.minimum(best, d)
    best = best / np.sqrt(D)  # RMSE
    best = torch.minimum(best, best.T)
    best.fill_diagonal_(0)
    return best.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chains", type=Path, default=None)
    ap.add_argument("--thr", type=float, default=None, help="override the automatic choice")
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    c = load_cache(args.cache)
    keep = c["ok"]
    df = pd.DataFrame(dict(path=c["path"][keep], label=c["label"][keep], split=c["split"][keep],
                           cls_folder=c["cls_folder"][keep], view_id=c["view_id"][keep]))
    if args.chains and args.chains.exists():
        df = df.merge(pd.read_csv(args.chains)[["path", "chain"]], on="path", how="left")
    thumbs = c["thumbs"][keep]                                   # (N,3,32,32,3) uint8
    T = thumbs.reshape(len(df), 3, -1).astype(np.float32)
    D = pairwise_min_rmse(T)
    labels = df.label.values
    N = len(df)

    # random-pair reference distributions for the clip distance
    def rand_pairs(same, n=20000):
        a = rng.integers(0, N, n * 3); b = rng.integers(0, N, n * 3)
        m = (a != b) & ((labels[a] == labels[b]) == same)
        return a[m][:n], b[m][:n]
    rows = []
    for name, same in (("random_same_class", True), ("random_cross_class", False)):
        a, b = rand_pairs(same)
        v = D[a, b]
        rows.append(dict(pair_type=name, n=len(v), mean=float(v.mean()), **{f"p{p}": float(np.percentile(v, p)) for p in (0.1, 1, 5, 25, 50, 75, 95)}))
    key = {(f, v): i for i, (f, v) in enumerate(zip(df.cls_folder, df.view_id))}
    adj = np.array([(i, key[(f, v + 1)]) for i, (f, v) in enumerate(zip(df.cls_folder, df.view_id)) if (f, v + 1) in key])
    v = D[adj[:, 0], adj[:, 1]]
    rows.append(dict(pair_type="adjacent_same_class(N,N+1)", n=len(v), mean=float(v.mean()), **{f"p{p}": float(np.percentile(v, p)) for p in (0.1, 1, 5, 25, 50, 75, 95)}))
    tr = (df.split == "train").values
    for s in ("val", "test"):
        q = (df.split == s).values
        nn = D[np.ix_(q, tr)].min(axis=1)
        rows.append(dict(pair_type=f"{s}_to_nearest_train", n=len(nn), mean=float(nn.mean()), **{f"p{p}": float(np.percentile(nn, p)) for p in (0.1, 1, 5, 25, 50, 75, 95)}))
    pcts = pd.DataFrame(rows); pcts.to_csv(args.out / "exp3b_random_pair_pcts.csv", index=False)
    print("clip distance (min-over-frame-pairs thumbnail RMSE) reference distributions:"); print(pcts.round(2).to_string(index=False))
    p1_cross = float(pcts.loc[pcts.pair_type == "random_cross_class", "p1"].iloc[0])

    cands = [10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40]
    sweep = []
    multi = df[["path", "label", "split"]].copy()
    for t in cands:
        _, lab = components_le(D, t)
        s = sweep_stats(df, lab, t); s["below_p1_random_cross"] = bool(t <= p1_cross)
        sweep.append(s)
        multi[f"cluster_thr{t:g}"] = lab
    multi.to_csv(args.out / "exp3b_clusters_by_thr.csv", index=False)
    sweep = pd.DataFrame(sweep); sweep.to_csv(args.out / "exp3b_threshold_sweep.csv", index=False)
    print(f"\nthreshold sweep (p1 of random cross-class distance = {p1_cross:.2f}):"); print(sweep.round(4).to_string(index=False))

    if args.thr is None:
        ok = sweep[(sweep.thr <= p1_cross) & (sweep.largest_frac <= args.max_frac)]
        thr = float(ok.thr.max()) if len(ok) else float(sweep.thr.min())
    else:
        thr = args.thr
    _, lab = components_le(D, thr)
    df["cluster"] = lab
    table = cluster_table(df, lab); table.to_csv(args.out / "exp3b_cluster_table.csv", index=False)
    df[["path", "label", "split", "cluster"]].to_csv(args.out / "scene_clusters.csv", index=False)
    stats = sweep_stats(df, lab, thr)
    sizes = table["size"].values
    stats.update(p1_random_cross=p1_cross, size_pcts={f"p{p}": float(np.percentile(sizes, p)) for p in (10, 25, 50, 75, 90, 99)},
                 top15_sizes=sizes[:15].tolist(), top15_purity=table["purity"].values[:15].round(3).tolist(),
                 n_clusters_ge5=int((sizes >= 5).sum()), frac_clips_in_clusters_ge5=float(sizes[sizes >= 5].sum() / N))
    # leakage check on the published split: nearest-train distance for test clips, and how many test clips
    # have a train clip within thr (= would be in the same group)
    te = (df.split == "test").values
    nn_te = D[np.ix_(te, tr)].min(axis=1)
    stats["frac_test_with_train_within_thr"] = float((nn_te <= thr).mean())
    stats["frac_test_with_train_within_2thr"] = float((nn_te <= 2 * thr).mean())
    # label purity among non-singleton clusters, and whether impure clusters are "scene with changing label"
    ns = table[table["size"] >= 2]
    stats["nonsingleton_weighted_purity"] = float((ns.purity * ns["size"]).sum() / ns["size"].sum())
    stats["frac_nonsingleton_clusters_pure"] = float((ns.purity == 1).mean())
    stats["frac_clips_in_nonsingleton"] = float(ns["size"].sum() / N)
    stats["frac_clips_in_pure_nonsingleton"] = float(ns.loc[ns.purity == 1, "size"].sum() / N)
    if "chain" in df:
        stats["frac_chains_in_single_cluster"] = float((df.groupby("chain").cluster.nunique() == 1).mean())
    with open(args.out / "exp3b_summary.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nchosen thr={thr}"); print(json.dumps(stats, indent=2))
    print("\nlargest 20 clusters:"); print(table.head(20).to_string(index=False))

    # contact sheets: 4 of the 20 largest clusters + 4 random clusters of size 6..40
    rng2 = np.random.default_rng(args.seed)
    big = table[table["size"] >= 6]
    mid = big[(big["size"] <= 40)]
    pick = list(rng2.choice(big.cluster.values[:20], size=min(4, len(big)), replace=False)) + \
           list(rng2.choice(mid.cluster.values, size=min(4, len(mid)), replace=False))
    for k, cid in enumerate(pick):
        idx = np.where(lab == cid)[0]
        idx = idx[np.argsort(df.cls_folder.values[idx].astype(str) + df.view_id.values[idx].astype(str).str.zfill(5) if False else df.view_id.values[idx] + 100000 * df.label.values[idx])]
        if len(idx) > 32:
            idx = np.sort(rng2.choice(idx, 32, replace=False))
        caps = [f"{df.split[i][:2]} L{df.label[i]} {df.cls_folder[i][0]}{df.view_id[i]}" for i in idx]
        contact_sheet(thumbs[idx, 1], caps, path=args.out / f"exp3b_cluster_{k}_id{cid}_n{int((lab == cid).sum())}.png")
    # test clip vs nearest train clip (center thumbs), 16 random test clips
    ti = np.where(tr)[0]; qi = np.where(te)[0]
    sel = rng2.choice(len(qi), 16, replace=False)
    tiles, caps = [], []
    for s in sel:
        a = qi[s]; b = ti[np.argmin(D[a, tr])]
        tiles += [thumbs[a, 1], thumbs[b, 1]]
        caps += [f"TEST L{labels[a]} {df.cls_folder[a][0]}{df.view_id[a]}", f"NN-tr L{labels[b]} {df.cls_folder[b][0]}{df.view_id[b]} d={D[a, b]:.0f}"]
    contact_sheet(np.stack(tiles), caps, ncol=8, path=args.out / "exp3b_test_nn_pairs.png")
    np.save(args.out / "exp3b_nn_test_to_train_dist.npy", nn_te)
    print("contact sheets written")


def components_le(D, thr):
    A = csr_matrix(D <= thr)
    return connected_components(A, directed=False)


if __name__ == "__main__":
    main()
