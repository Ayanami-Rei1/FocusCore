#!/usr/bin/env python
"""Experiment 3: scene clustering of OUC-CGE clips from frozen center-frame features.

Primary grouping = connected components of the graph "cosine(center feature) >= thr"
(single linkage). Rationale: for a leakage-free group split we need a guarantee that NO pair
of clips across groups is a near-duplicate; single linkage is the only linkage that gives
that guarantee for a threshold (average/complete linkage can leave cross-cluster pairs above
thr). The cost is possible over-merging, which we expose with a threshold sweep.

Outputs (in --out): exp3_threshold_sweep.csv, scene_clusters.csv (path,label,split,cluster),
exp3_cluster_table.csv (per cluster size/purity/splits), exp3_summary.json, contact sheets.

Usage: python scripts/review/03_scene_clusters.py --cache outputs/review_cache --out outputs/review_out --thr <from exp2>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_cache  # noqa: E402


def l2n(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def components(S, thr):
    A = csr_matrix(S >= thr)
    n, lab = connected_components(A, directed=False)
    return n, lab


def cluster_table(df, lab):
    rows = []
    for cid in np.unique(lab):
        m = lab == cid
        counts = np.bincount(df.label.values[m], minlength=3)
        splits = set(df.split.values[m])
        rows.append(dict(cluster=int(cid), size=int(m.sum()), n_low=int(counts[0]), n_mid=int(counts[1]), n_high=int(counts[2]),
                         purity=float(counts.max() / m.sum()), majority_label=int(counts.argmax()),
                         n_train=int((df.split.values[m] == "train").sum()), n_val=int((df.split.values[m] == "val").sum()),
                         n_test=int((df.split.values[m] == "test").sum()), n_splits=len(splits),
                         n_chains=int(len(set(df.chain.values[m]))) if "chain" in df else -1))
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def sweep_stats(df, lab, thr):
    t = cluster_table(df, lab)
    sizes = t["size"].values
    cl_of = dict(zip(t.cluster, t.index))
    tr_clusters = set(lab[(df.split == "train").values])
    te = (df.split == "test").values
    va = (df.split == "val").values
    w_purity = float((t["purity"] * t["size"]).sum() / t["size"].sum())
    return dict(thr=float(thr), n_clusters=int(len(t)), n_singletons=int((sizes == 1).sum()), largest=int(sizes.max()),
                largest_frac=float(sizes.max() / len(df)), median_size=float(np.median(sizes)),
                weighted_purity=w_purity, frac_clusters_pure=float((t["purity"] == 1).mean()),
                frac_clips_in_pure_clusters=float(t.loc[t.purity == 1, "size"].sum() / len(df)),
                n_clusters_multi_split=int((t["n_splits"] > 1).sum()),
                frac_test_with_train_in_cluster=float(np.isin(lab[te], list(tr_clusters)).mean()),
                frac_val_with_train_in_cluster=float(np.isin(lab[va], list(tr_clusters)).mean()))


def contact_sheet(thumbs, captions, ncol=8, tile=96, path=None):
    n = len(thumbs)
    nrow = int(np.ceil(n / ncol))
    cap_h = 14
    img = Image.new("RGB", (ncol * tile, nrow * (tile + cap_h)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, (t, c) in enumerate(zip(thumbs, captions)):
        r, cc = divmod(i, ncol)
        im = Image.fromarray(t).resize((tile, tile), Image.NEAREST)
        img.paste(im, (cc * tile, r * (tile + cap_h)))
        d.text((cc * tile + 2, r * (tile + cap_h) + tile), c, fill=(0, 0, 0))
    if path:
        img.save(path)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thr", type=float, required=True, help="cosine threshold (justified in exp2)")
    ap.add_argument("--chains", type=Path, default=None, help="exp2_chains.csv to annotate clusters with chains")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    c = load_cache(args.cache)
    keep = c["ok"]
    df = pd.DataFrame(dict(path=c["path"][keep], label=c["label"][keep], split=c["split"][keep],
                           cls_folder=c["cls_folder"][keep], view_id=c["view_id"][keep]))
    if args.chains and args.chains.exists():
        ch = pd.read_csv(args.chains)
        df = df.merge(ch[["path", "chain"]], on="path", how="left")
    f = l2n(c["feats"][keep][:, 1])
    thumbs = c["thumbs"][keep][:, 1]
    S = f @ f.T
    np.fill_diagonal(S, 0)

    sweep = []
    for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, args.thr]:
        _, lab = components(S, t)
        sweep.append(sweep_stats(df, lab, t))
    sweep = pd.DataFrame(sweep).sort_values("thr")
    sweep.to_csv(args.out / "exp3_threshold_sweep.csv", index=False)
    print("threshold sweep (connected components of cosine >= thr):")
    print(sweep.round(4).to_string(index=False))

    n, lab = components(S, args.thr)
    df["cluster"] = lab
    table = cluster_table(df, lab)
    table.to_csv(args.out / "exp3_cluster_table.csv", index=False)
    df[["path", "label", "split", "cluster"]].to_csv(args.out / "scene_clusters.csv", index=False)
    stats = sweep_stats(df, lab, args.thr)
    sizes = table["size"].values
    stats["size_pcts"] = {f"p{p}": float(np.percentile(sizes, p)) for p in (10, 25, 50, 75, 90, 99)}
    stats["top10_sizes"] = sizes[:10].tolist()
    stats["top10_purity"] = table["purity"].values[:10].round(3).tolist()
    # chain vs cluster agreement
    if "chain" in df:
        stats["n_chains"] = int(df.chain.nunique())
        stats["clusters_per_chain_mean"] = float(df.groupby("chain").cluster.nunique().mean())
        stats["frac_chains_in_single_cluster"] = float((df.groupby("chain").cluster.nunique() == 1).mean())
    # cross-class near duplicates: pairs above thr with different labels
    ii, jj = np.where(np.triu(S >= args.thr, 1))
    diff = df.label.values[ii] != df.label.values[jj]
    stats["n_pairs_above_thr"] = int(len(ii))
    stats["frac_pairs_above_thr_cross_label"] = float(diff.mean()) if len(ii) else 0.0
    stats["n_cross_label_pairs_above_thr"] = int(diff.sum())
    with open(args.out / "exp3_summary.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\nchosen threshold summary:"); print(json.dumps(stats, indent=2))
    print("\nlargest 15 clusters:"); print(table.head(15).to_string(index=False))

    # contact sheets: 8 random clusters with size >= 6 (stratified: 4 from the largest 20, 4 random others)
    big = table[table["size"] >= 6]
    pick = list(rng.choice(big.cluster.values[:20], size=min(4, len(big)), replace=False)) + \
           list(rng.choice(big.cluster.values[20:], size=min(4, max(0, len(big) - 20)), replace=False))
    for k, cid in enumerate(pick):
        idx = np.where(lab == cid)[0]
        idx = idx[np.argsort(df.view_id.values[idx] + 10000 * df.label.values[idx])]
        if len(idx) > 32:
            idx = np.sort(rng.choice(idx, 32, replace=False))
        caps = [f"{df.split[i][:2]} L{df.label[i]} {df.cls_folder[i][0]}{df.view_id[i]}" for i in idx]
        contact_sheet(thumbs[idx], caps, path=args.out / f"exp3_cluster_{k}_id{cid}_n{int((lab == cid).sum())}.png")
    # adjacent-pair sheet: 6 random same-folder (N,N+1) pairs, showing first/center/last of N then of N+1
    key = {(f_, v): i for i, (f_, v) in enumerate(zip(df.cls_folder, df.view_id))}
    adj = [(i, key[(f_, v + 1)]) for i, (f_, v) in enumerate(zip(df.cls_folder, df.view_id)) if (f_, v + 1) in key]
    sel = rng.choice(len(adj), 6, replace=False)
    tiles, caps = [], []
    allthumbs = c["thumbs"][keep]
    for s in sel:
        a, b = adj[s]
        for who, j in (("N", a), ("N+1", b)):
            for fi, fn in enumerate(("first", "center", "last")):
                tiles.append(allthumbs[j][fi]); caps.append(f"{who} {fn} {df.split[j][:2]} L{df.label[j]} {df.cls_folder[j][0]}{df.view_id[j]}")
    contact_sheet(np.stack(tiles), caps, ncol=6, path=args.out / "exp2a_adjacent_pairs_sheet.png")
    print("contact sheets written")


if __name__ == "__main__":
    main()
