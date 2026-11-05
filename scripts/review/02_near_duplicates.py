#!/usr/bin/env python
"""Experiment 2: near-duplicate structure between splits of OUC-CGE.

Inputs: the npz cache from 01_frame_cache.py.
(a) similarity distributions for same-class ID-adjacent pairs (N, N+1) vs random same-class
    pairs vs random cross-class pairs; center-feature cosine, last(N)->first(N+1) feature cosine,
    and center-thumbnail RMSE (0-255 scale).
(b) nearest-train-neighbour accuracy for val/test clips (1-NN, 5-NN majority) by feature cosine
    and by thumbnail RMSE, plus the same restricted to neighbours from a different "ID chain".
(c) logistic regression on frozen train features -> val/test accuracy on the split as published.

Usage: python scripts/review/02_near_duplicates.py --cache outputs/review_cache --out outputs/review_out
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_cache  # noqa: E402

PCTS = [1, 5, 25, 50, 75, 95, 99]


def l2n(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def pct_row(name, v):
    v = np.asarray(v, dtype=np.float64)
    return dict(pair_type=name, n=int(len(v)), mean=float(v.mean()), **{f"p{p}": float(np.percentile(v, p)) for p in PCTS})


def build_chains(df, sim_adj, thr):
    """Chain id per clip: maximal runs of same-folder consecutive view_ids whose adjacent
    center-feature cosine >= thr. sim_adj[i] = cosine(clip i, clip with view_id+1 in same folder) or nan."""
    chain = np.full(len(df), -1, dtype=np.int64)
    order = df.sort_values(["cls_folder", "view_id"]).index.values
    cid = -1
    prev_idx = None
    for idx in order:
        link = (
            prev_idx is not None
            and df.cls_folder[prev_idx] == df.cls_folder[idx]
            and df.view_id[idx] == df.view_id[prev_idx] + 1
            and np.isfinite(sim_adj[prev_idx]) and sim_adj[prev_idx] >= thr
        )
        if not link:
            cid += 1
        chain[idx] = cid
        prev_idx = idx
    return chain


def knn_eval(S, train_labels, q_labels, k_list=(1, 5), mask=None, larger_is_closer=True):
    """S: (n_query, n_train) similarity (or distance when larger_is_closer=False)."""
    S = S.copy()
    if mask is not None:
        S[~mask] = -np.inf if larger_is_closer else np.inf
    order = np.argsort(-S if larger_is_closer else S, axis=1)
    out = {}
    for k in k_list:
        nn = order[:, :k]
        nl = train_labels[nn]
        if k == 1:
            pred = nl[:, 0]
        else:
            pred = np.array([np.bincount(r, minlength=3).argmax() for r in nl])
        out[f"acc@{k}"] = float((pred == q_labels).mean())
    best = S[np.arange(len(S)), order[:, 0]]
    out["nn_score_pcts"] = {f"p{p}": float(np.percentile(best, p)) for p in PCTS}
    out["nn_score_mean"] = float(best.mean())
    return out, order[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chain-thr", type=float, default=None,
                    help="cosine threshold for ID chains; default = p99 of random cross-class pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    c = load_cache(args.cache)
    df = pd.DataFrame(dict(path=c["path"], label=c["label"], split=c["split"], cls_folder=c["cls_folder"],
                           view_id=c["view_id"], ok=c["ok"], n_reported=c["n_reported"], n_frames=c["n_frames"]))
    print(f"clips={len(df)} ok={int(df.ok.sum())}")
    df = df[df.ok].reset_index(drop=True)
    keep = c["ok"]
    feats = c["feats"][keep].astype(np.float32)          # (N,3,2048): first, center, last
    thumbs = c["thumbs"][keep].astype(np.float32)        # (N,3,32,32,3)
    f_first, f_center, f_last = (l2n(feats[:, i]) for i in range(3))
    t_center = thumbs[:, 1].reshape(len(df), -1)

    # ---------- (a) pair similarity distributions ----------
    key = {(f, v): i for i, (f, v) in enumerate(zip(df.cls_folder, df.view_id))}
    nxt = np.array([key.get((f, v + 1), -1) for f, v in zip(df.cls_folder, df.view_id)])
    has = nxt >= 0
    i_adj, j_adj = np.where(has)[0], nxt[has]
    n_pairs = len(i_adj)
    labels = df.label.values
    # random same-class, non-adjacent pairs and random cross-class pairs
    def sample_pairs(same, n=20000):
        ii, jj = [], []
        while len(ii) < n:
            a = rng.integers(0, len(df), n * 2)
            b = rng.integers(0, len(df), n * 2)
            m = (a != b) & ((labels[a] == labels[b]) == same)
            if same:
                m &= ~((df.cls_folder.values[a] == df.cls_folder.values[b]) & (np.abs(df.view_id.values[a] - df.view_id.values[b]) <= 1))
            ii.extend(a[m].tolist()); jj.extend(b[m].tolist())
        return np.array(ii[:n]), np.array(jj[:n])
    i_same, j_same = sample_pairs(True)
    i_x, j_x = sample_pairs(False)

    def cos(fa, fb, i, j):
        return np.einsum("ij,ij->i", fa[i], fb[j])
    def rmse(i, j):
        return np.sqrt(((t_center[i] - t_center[j]) ** 2).mean(axis=1))

    rows = []
    for name, (i, j) in {"adjacent_same_class(N,N+1)": (i_adj, j_adj), "random_same_class": (i_same, j_same), "random_cross_class": (i_x, j_x)}.items():
        rows.append(dict(metric="center_cosine", **pct_row(name, cos(f_center, f_center, i, j))))
        rows.append(dict(metric="last(N)_to_first(N+1)_cosine", **pct_row(name, cos(f_last, f_first, i, j))))
        rows.append(dict(metric="center_thumb_rmse", **pct_row(name, rmse(i, j))))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.out / "exp2a_pair_similarity.csv", index=False)
    print("\n(a) pair similarity distributions"); print(pairs.round(4).to_string(index=False))

    # also: how often is the adjacent clip more similar than ANY random same-class clip? -> fraction of adjacent
    # pairs with cosine above the p99 of random same-class pairs
    p99_same = float(np.percentile(cos(f_center, f_center, i_same, j_same), 99))
    p99_x = float(np.percentile(cos(f_center, f_center, i_x, j_x), 99))
    adj_c = cos(f_center, f_center, i_adj, j_adj)
    adj_lf = cos(f_last, f_first, i_adj, j_adj)
    summary = dict(n_adjacent_pairs=int(n_pairs), p99_random_same_class_cosine=p99_same, p99_random_cross_class_cosine=p99_x,
                   frac_adjacent_center_above_p99_same=float((adj_c > p99_same).mean()),
                   frac_adjacent_center_above_p99_cross=float((adj_c > p99_x).mean()),
                   frac_adjacent_lastfirst_above_0_9=float((adj_lf > 0.9).mean()),
                   frac_adjacent_lastfirst_above_0_95=float((adj_lf > 0.95).mean()))

    # ---------- chains ----------
    sim_adj = np.full(len(df), np.nan); sim_adj[i_adj] = adj_c
    thr = args.chain_thr if args.chain_thr is not None else p99_x
    summary["chain_thr"] = float(thr)
    sweep = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, thr]:
        ch = build_chains(df, sim_adj, t)
        sizes = np.bincount(ch)
        sweep.append(dict(thr=round(float(t), 4), n_chains=int(len(sizes)), median_size=float(np.median(sizes)), max_size=int(sizes.max()),
                          frac_clips_in_chains_ge2=float((sizes[ch] >= 2).mean()), frac_links_kept=float(np.nanmean(sim_adj >= t))))
    sweep = pd.DataFrame(sweep); sweep.to_csv(args.out / "exp2b_chain_threshold_sweep.csv", index=False)
    print(f"\nchain threshold sweep (chosen thr={thr:.4f} = p99 of random cross-class cosine)"); print(sweep.to_string(index=False))
    chain = build_chains(df, sim_adj, thr)
    df["chain"] = chain

    # ---------- (b) nearest train neighbour ----------
    tr = (df.split == "train").values
    tl = labels[tr]
    res_b = []
    nn_rows = []
    for split in ("val", "test"):
        q = (df.split == split).values
        S_cos = f_center[q] @ f_center[tr].T
        D_thumb = np.sqrt(((t_center[q][:, None, :] - t_center[tr][None, :, :]) ** 2).mean(axis=2)) if q.sum() * tr.sum() * 3072 * 4 < 40e9 else None
        if D_thumb is None:  # chunk
            D_thumb = np.zeros((q.sum(), tr.sum()), dtype=np.float32)
            tq = t_center[q]; tt = t_center[tr]
            for s in range(0, len(tq), 64):
                D_thumb[s:s + 64] = np.sqrt(((tq[s:s + 64, None, :] - tt[None, :, :]) ** 2).mean(axis=2))
        diff_chain = chain[q][:, None] != chain[tr][None, :]
        for metric, S, larger in (("feature_cosine", S_cos, True), ("thumb_rmse", D_thumb, False)):
            r, nn1 = knn_eval(S, tl, labels[q], larger_is_closer=larger)
            r_dc, nn1_dc = knn_eval(S, tl, labels[q], mask=diff_chain, larger_is_closer=larger)
            res_b.append(dict(split=split, metric=metric, restriction="any_train", **{k: v for k, v in r.items() if k != "nn_score_pcts"}, **{f"nn_{k}": v for k, v in r["nn_score_pcts"].items()}))
            res_b.append(dict(split=split, metric=metric, restriction="different_chain", **{k: v for k, v in r_dc.items() if k != "nn_score_pcts"}, **{f"nn_{k}": v for k, v in r_dc["nn_score_pcts"].items()}))
            if metric == "feature_cosine":
                qi = np.where(q)[0]; ti = np.where(tr)[0]
                for a, b, bdc in zip(qi, nn1, nn1_dc):
                    nn_rows.append(dict(path=df.path[a], label=int(labels[a]), split=split, chain=int(chain[a]),
                                        nn_path=df.path[ti[b]], nn_label=int(tl[b]), nn_cos=float(S[a - qi[0] if False else np.where(qi == a)[0][0], b]),
                                        nn_same_chain=bool(chain[a] == chain[ti[b]]),
                                        nn_id_adjacent=bool(df.cls_folder[a] == df.cls_folder[ti[b]] and abs(int(df.view_id[a]) - int(df.view_id[ti[b]])) == 1),
                                        nn_diffchain_path=df.path[ti[bdc]], nn_diffchain_label=int(tl[bdc]), nn_diffchain_cos=float(S[np.where(qi == a)[0][0], bdc])))
    res_b = pd.DataFrame(res_b); res_b.to_csv(args.out / "exp2b_knn.csv", index=False)
    nn_df = pd.DataFrame(nn_rows); nn_df.to_csv(args.out / "exp2b_nearest_train_neighbour.csv", index=False)
    print("\n(b) nearest-train-neighbour accuracy"); print(res_b.round(4).to_string(index=False))
    for split in ("val", "test"):
        m = nn_df.split == split
        summary[f"{split}_frac_nn_is_id_adjacent"] = float(nn_df[m].nn_id_adjacent.mean())
        summary[f"{split}_frac_nn_same_chain"] = float(nn_df[m].nn_same_chain.mean())
        summary[f"{split}_frac_nn_cos_ge_0_95"] = float((nn_df[m].nn_cos >= 0.95).mean())
        summary[f"{split}_frac_nn_cos_ge_0_90"] = float((nn_df[m].nn_cos >= 0.90).mean())

    # ---------- (c) frozen-feature logistic regression ----------
    sc = StandardScaler().fit(feats[tr, 1])
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=args.seed).fit(sc.transform(feats[tr, 1]), tl)
    res_c = {}
    for split in ("train", "val", "test"):
        m = (df.split == split).values
        pred = clf.predict(sc.transform(feats[m, 1]))
        res_c[split] = dict(n=int(m.sum()), acc=float(accuracy_score(labels[m], pred)), f1_macro=float(f1_score(labels[m], pred, average="macro")),
                            confusion=confusion_matrix(labels[m], pred, labels=[0, 1, 2]).tolist())
    print("\n(c) logistic regression on frozen center-frame ResNet50 features (published random split)")
    for k, v in res_c.items():
        print(f"  {k}: n={v['n']} acc={v['acc']:.4f} f1_macro={v['f1_macro']:.4f} confusion={v['confusion']}")
    summary["logreg_random_split"] = res_c
    maj = np.bincount(tl).argmax()
    summary["majority_class_baseline"] = {s: float((labels[(df.split == s).values] == maj).mean()) for s in ("val", "test")}
    summary["class_counts"] = {s: np.bincount(labels[(df.split == s).values], minlength=3).tolist() for s in ("train", "val", "test")}

    df[["path", "label", "split", "cls_folder", "view_id", "chain"]].to_csv(args.out / "exp2_chains.csv", index=False)
    with open(args.out / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsummary:"); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
