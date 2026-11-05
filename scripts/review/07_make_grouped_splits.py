#!/usr/bin/env python
"""Step 1 of the grouped-split experiment: scene-disjoint train/val/test manifests.

Input: the thumbnail scene clusters from 03b (single-linkage connected components of
min-over-frame-pairs 32x32 RMSE, so no near-duplicate pair at RMSE <= thr crosses clusters).
Whole clusters are assigned to splits; a cluster never appears in two splits.

Assignment: greedy, clusters in descending size order, each cluster goes to the split that
minimises the sum over (split, label) cells of ((count - target) / target)^2 with
target = frac[split] * N_label (frac = 70/10/20).  This is the StratifiedGroupKFold idea
with unequal fold sizes.  Because the clusters are large and label-skewed, the pure
size-ordered greedy is not necessarily the best balanced assignment, so the script runs
R random restarts (size order jittered by a seeded log-normal factor), keeps only the
candidates in which every split has >= --min-share of its clips in every class, and takes
the best-scoring one.  A second seed with --avoid <assignment.csv> additionally rejects
candidates whose test clips overlap the avoided assignment's test clusters by more than
--max-test-overlap, which gives a genuinely different scene hold-out.

Outputs per split name:
  <out-base>/<name>/data_labels/{train,val,test}.csv   project manifest format ("path label")
  <out-base>/<name>/videos -> <videos-link>              symlink, so OUCCGEDataset is unchanged
  <report>/split_<name>.csv                              path,label,split,cluster
  <report>/split_<name>_summary.json                     sizes, label counts, clusters, leakage
With --cache the leakage check is computed from the cached thumbnails: for every test (and
val) clip the RMSE to its nearest train clip, versus the same numbers for the random split.

Usage (scripts/review/run_07_splits.sh runs the three published splits with these arguments):
  python scripts/review/07_make_grouped_splits.py --name grp27a --thr 27.5 --seed 0 \
      --cache outputs/review_cache --out-base data --report outputs/review_grouped/splits
  python scripts/review/07_make_grouped_splits.py --name grp27b --thr 27.5 --seed 1 \
      --avoid outputs/review_grouped/splits/split_grp27a.csv ...
  python scripts/review/07_make_grouped_splits.py --name grp17 --thr 17.5 --seed 0 ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SPLITS, load_cache, write_dataset_root  # noqa: E402

FRACS = {"train": 0.70, "val": 0.10, "test": 0.20}
SLOW_WINDOW = 64  # OUCCGEDataset window for the SLOW pathway (8 frames x stride 8)


def greedy_assign(counts: dict, order: list, totals: np.ndarray) -> dict:
    """counts: cluster -> (3,) label counts; order: clusters; totals: (3,) label totals."""
    cur = {s: np.zeros(3) for s in FRACS}
    target = {s: FRACS[s] * totals for s in FRACS}
    assign = {}
    for k in order:
        v = counts[k]
        best, best_score = None, None
        for s in FRACS:
            score = 0.0
            for s2 in FRACS:
                c = cur[s2] + (v if s2 == s else 0)
                score += float((((c - target[s2]) / target[s2]) ** 2).sum())
            if best_score is None or score < best_score:
                best, best_score = s, score
        assign[k] = best
        cur[best] += v
    return assign


def balance_score(assign: dict, counts: dict, totals: np.ndarray) -> float:
    cur = {s: np.zeros(3) for s in FRACS}
    for k, s in assign.items():
        cur[s] += counts[k]
    return float(sum((((cur[s] - FRACS[s] * totals) / (FRACS[s] * totals)) ** 2).sum() for s in FRACS))


def min_class_share(assign: dict, counts: dict) -> float:
    cur = {s: np.zeros(3) for s in FRACS}
    for k, s in assign.items():
        cur[s] += counts[k]
    return float(min((cur[s] / cur[s].sum()).min() for s in FRACS))


def pairwise_min_rmse(T: np.ndarray) -> np.ndarray:
    """T: (N, 3, D) thumbnails. (N, N) min over the 3x3 frame pairs of the RMSE, as in 03b
    (float64 matmul form of the L2 distance; values are on the 0-255 scale)."""
    X = T.astype(np.float64)
    N, F, D = X.shape
    sq = (X ** 2).sum(-1)  # (N, F)
    best = None
    for a in range(F):
        for b in range(F):
            d2 = sq[:, a][:, None] + sq[:, b][None, :] - 2.0 * (X[:, a] @ X[:, b].T)
            np.maximum(d2, 0, out=d2)
            best = d2 if best is None else np.minimum(best, d2)
    best = np.sqrt(best / D)
    best = np.minimum(best, best.T)
    np.fill_diagonal(best, 0)
    return best.astype(np.float32)


def nn_stats(D: np.ndarray, q: np.ndarray, ref: np.ndarray) -> dict:
    nn = D[np.ix_(q, ref)].min(axis=1)
    return dict(n=int(len(nn)), median=float(np.median(nn)), p5=float(np.percentile(nn, 5)),
                p25=float(np.percentile(nn, 25)), p75=float(np.percentile(nn, 75)),
                p95=float(np.percentile(nn, 95)), min=float(nn.min()),
                frac_le_10=float((nn <= 10).mean()), frac_le_17_5=float((nn <= 17.5).mean()),
                frac_le_27_5=float((nn <= 27.5).mean()), frac_le_35=float((nn <= 35).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=Path, default=Path("docs/review/data/scene_clusters.csv"))
    ap.add_argument("--clusters-multi", type=Path, default=Path("docs/review/data/exp3b_clusters_by_thr.csv"))
    ap.add_argument("--thr", type=float, required=True, help="27.5 (scene_clusters.csv) or another cluster_thr* column")
    ap.add_argument("--name", required=True, help="dataset root name, e.g. grp27a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=400)
    ap.add_argument("--jitter", type=float, default=0.6, help="sigma of the log-normal size jitter for restarts > 0")
    ap.add_argument("--min-share", type=float, default=0.10, help="every split must have >= this share of each class")
    ap.add_argument("--avoid", type=Path, default=None, help="split_<name>.csv of another assignment to differ from")
    ap.add_argument("--max-test-overlap", type=float, default=0.34,
                    help="max fraction of this test set's clips whose cluster is a test cluster of --avoid")
    ap.add_argument("--cache", type=Path, default=None, help="review_cache dir; enables the thumbnail leakage check")
    ap.add_argument("--out-base", type=Path, default=Path("data"))
    ap.add_argument("--videos-link", default="../ouc-cge-320/videos")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="only print the assignment statistics")
    args = ap.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.clusters)  # path,label,split,cluster (thr 27.5)
    multi = pd.read_csv(args.clusters_multi)
    col = f"cluster_thr{args.thr:g}"
    if col not in multi.columns:
        raise SystemExit(f"{col} not in {args.clusters_multi}: {list(multi.columns)}")
    df = base.merge(multi[["path", col]], on="path", how="left")
    assert df[col].notna().all(), "cluster ids missing for some clips"
    if args.thr == 27.5:
        assert (df["cluster"] == df[col]).all(), "scene_clusters.csv disagrees with the thr-27.5 column"
    df["grp"] = df[col].astype(int)
    df = df.rename(columns={"split": "random_split"})
    N = len(df)
    totals = np.bincount(df.label.values, minlength=3).astype(float)
    counts = {k: np.bincount(g.label.values, minlength=3).astype(float) for k, g in df.groupby("grp")}
    sizes = {k: int(v.sum()) for k, v in counts.items()}
    print(f"clips={N} labels(low/mid/high)={totals.astype(int).tolist()} clusters@{args.thr:g}={len(counts)} "
          f"largest={max(sizes.values())}")

    avoid_test_clusters = set()
    if args.avoid is not None:
        av = pd.read_csv(args.avoid)
        avoid_test_clusters = set(av.loc[av.split == "test", "cluster"].astype(int))
        print(f"avoiding {len(avoid_test_clusters)} test clusters of {args.avoid}")

    # --- randomised greedy restarts -------------------------------------------------
    rng = np.random.default_rng(args.seed)
    keys = sorted(counts)
    size_arr = np.array([sizes[k] for k in keys], dtype=float)
    cands = []
    for r in range(args.restarts):
        jit = np.ones(len(keys)) if r == 0 else np.exp(rng.normal(0, args.jitter, len(keys)))
        order = [keys[i] for i in np.argsort(-(size_arr * jit), kind="stable")]
        a = greedy_assign(counts, order, totals)
        test_clusters = {k for k, s in a.items() if s == "test"}
        n_test = sum(sizes[k] for k in test_clusters)
        overlap = sum(sizes[k] for k in test_clusters & avoid_test_clusters) / max(n_test, 1)
        cands.append(dict(r=r, assign=a, score=balance_score(a, counts, totals),
                          min_share=min_class_share(a, counts), overlap=overlap))
    cdf = pd.DataFrame([{k: v for k, v in c.items() if k != "assign"} for c in cands])
    print("restart candidates: score p0/p50 = %.3f/%.3f, min_share max = %.3f, overlap min = %.2f"
          % (cdf.score.min(), cdf.score.median(), cdf.min_share.max(), cdf.overlap.min()))
    ok = cdf[(cdf.min_share >= args.min_share) & (cdf.overlap <= args.max_test_overlap)]
    if len(ok) == 0:
        print(f"WARNING no candidate satisfies min_share>={args.min_share} and overlap<={args.max_test_overlap}; "
              "relaxing to best score among overlap-ok candidates")
        ok = cdf[cdf.overlap <= args.max_test_overlap]
        if len(ok) == 0:
            ok = cdf
    best = cands[int(ok.sort_values(["score", "r"]).iloc[0].r)]
    print(f"chosen restart {best['r']}: score={best['score']:.3f} min_share={best['min_share']:.3f} "
          f"test overlap with avoided={best['overlap']:.2f} (pure size-order restart 0: score={cands[0]['score']:.3f} "
          f"min_share={cands[0]['min_share']:.3f})")
    df["split"] = df.grp.map(best["assign"])

    # --- statistics -------------------------------------------------------------------
    summary = dict(name=args.name, thr=args.thr, seed=args.seed, restart=best["r"], score=best["score"],
                   min_class_share=best["min_share"], test_overlap_with_avoided=best["overlap"],
                   avoided=str(args.avoid) if args.avoid else None, n_clips=N, n_clusters=len(counts),
                   fracs=FRACS, splits={})
    for s in SPLITS:
        g = df[df.split == s]
        summary["splits"][s] = dict(n=int(len(g)), frac=float(len(g) / N),
                                    labels=np.bincount(g.label.values, minlength=3).tolist(),
                                    label_frac=(np.bincount(g.label.values, minlength=3) / max(len(g), 1)).round(3).tolist(),
                                    n_clusters=int(g.grp.nunique()),
                                    cluster_sizes_top5=sorted(g.groupby("grp").size().tolist(), reverse=True)[:5])
    for s in SPLITS:
        d = summary["splits"][s]
        print(f"  {s:5s} n={d['n']:4d} ({d['frac']:.3f}) labels={d['labels']} frac={d['label_frac']} "
              f"clusters={d['n_clusters']} top5={d['cluster_sizes_top5']}")
    # no cluster in two splits (by construction; checked anyway)
    assert (df.groupby("grp").split.nunique() == 1).all()

    # --- leakage check from the cached thumbnails ------------------------------------
    if args.cache is not None:
        c = load_cache(args.cache)
        keep = c["ok"]
        cache_df = pd.DataFrame(dict(path=c["path"][keep], n_reported=c["n_reported"][keep]))
        cache_df["ci"] = np.arange(len(cache_df))
        m = df.merge(cache_df, on="path", how="left")
        assert m.ci.notna().all(), "clips missing from the cache"
        T = c["thumbs"][keep].reshape(len(cache_df), 3, -1)
        D = pairwise_min_rmse(T)
        ci = m.ci.values.astype(int)
        leak = {}
        for scheme, colname in (("grouped", "split"), ("random", "random_split")):
            idx = {s: ci[(m[colname] == s).values] for s in SPLITS}
            leak[scheme] = dict(test_vs_train=nn_stats(D, idx["test"], idx["train"]),
                                val_vs_train=nn_stats(D, idx["val"], idx["train"]),
                                test_vs_val=nn_stats(D, idx["test"], idx["val"]))
            for s in SPLITS:
                short = int((m.loc[m[colname] == s, "n_reported"] < SLOW_WINDOW).sum())
                summary["splits"][s][f"under_{SLOW_WINDOW}_frames_{scheme}"] = short
        summary["leakage"] = leak
        for scheme in leak:
            t = leak[scheme]["test_vs_train"]
            print(f"  [{scheme}] test->train nearest RMSE: median={t['median']:.1f} p5={t['p5']:.1f} p95={t['p95']:.1f} "
                  f"min={t['min']:.1f} <=10:{t['frac_le_10']:.3f} <=17.5:{t['frac_le_17_5']:.3f} <=27.5:{t['frac_le_27_5']:.3f}")
        print("  clips under %d frames (grouped) train/val/test: %s" % (
            SLOW_WINDOW, [summary["splits"][s][f"under_{SLOW_WINDOW}_frames_grouped"] for s in SPLITS]))

    # --- write ------------------------------------------------------------------------
    out = df[["path", "label", "split", "grp"]].rename(columns={"grp": "cluster"})
    out.to_csv(args.report / f"split_{args.name}.csv", index=False)
    with open(args.report / f"split_{args.name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.dry_run:
        return
    root = args.out_base / args.name
    write_dataset_root(out, root, args.videos_link)
    print(f"wrote {root} (videos -> {os.readlink(root / 'videos')})")


if __name__ == "__main__":
    main()
