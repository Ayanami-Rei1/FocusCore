#!/usr/bin/env python
"""Compare a reproduction with the published artefacts in docs/review/data/.

--clusters DIR   scene clusters rebuilt by 03b_scene_clusters_thumb.py (scene_clusters.csv,
                 exp3b_clusters_by_thr.csv) against the published ones.  Reports per threshold
                 whether the files describe the same partition of the clips (cluster ids may be
                 numbered differently) and how many clips sit in a differing cluster.
                 Exit code 1 if any partition differs.
--runs DIR       output directory of 08_run_grouped_training.sh (eval/summary_*.json,
                 eval/preds_*.csv, mlflow/<run>/summary.json) against docs/review/data/grouped/.
                 Prints one row per run: selected epoch, validation and test accuracy, macro-F1,
                 last-checkpoint test accuracy, and the share of test clips with the same
                 prediction.  Training is not bit-deterministic (cuDNN autotuning, bf16), so
                 this part never fails; read the differences.
--out FILE       also write the report as Markdown.

Usage:
  python scripts/review/10_compare_with_published.py --clusters outputs/review_out
  python scripts/review/10_compare_with_published.py --runs outputs/review_grouped --out outputs/review_grouped/comparison.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PUBLISHED = REPO / "docs/review/data"
RUNS = ["R1_r50_random", "R2_r50_grp27a", "R3_r50_grp27b", "R4_r50_grp17", "R5_slow_grp27a"]


def same_partition(a: pd.Series, b: pd.Series) -> tuple[bool, int]:
    """(partitions equal, number of clips whose cluster is not matched one-to-one)."""
    pairs = pd.DataFrame({"a": a.values, "b": b.values})
    # a cluster is matched if all its clips share one b-cluster and that b-cluster holds only them
    a_to_b = pairs.groupby("a")["b"].nunique()
    b_to_a = pairs.groupby("b")["a"].nunique()
    bad = pairs["a"].map(a_to_b).gt(1) | pairs["b"].map(b_to_a).gt(1)
    return (not bool(bad.any())), int(bad.sum())


def compare_clusters(new_dir: Path) -> tuple[list[str], bool]:
    lines, all_ok = ["## Scene clusters", "", "| file / threshold | clips | clusters (published / rebuilt) | result |",
                     "|---|---|---|---|"], True
    for fname in ("scene_clusters.csv", "exp3b_clusters_by_thr.csv"):
        pub, new = pd.read_csv(PUBLISHED / fname), pd.read_csv(new_dir / fname)
        identical = (PUBLISHED / fname).read_bytes() == (new_dir / fname).read_bytes()
        if set(pub["path"]) != set(new["path"]):
            lines.append(f"| {fname} | {len(pub)} / {len(new)} | | DIFFERENT CLIP SETS |")
            all_ok = False
            continue
        m = pub.merge(new, on="path", suffixes=("_pub", "_new"))
        cols = [c for c in pub.columns if c.startswith("cluster")]
        for c in cols:
            ok, n_bad = same_partition(m[f"{c}_pub"], m[f"{c}_new"])
            all_ok &= ok
            res = "byte-identical file" if identical else ("same partition" if ok else f"DIFFERS ({n_bad} clips)")
            name = fname if c == "cluster" else c.replace("cluster_thr", "thr ")
            lines.append(f"| {name} | {len(m)} | {m[f'{c}_pub'].nunique()} / {m[f'{c}_new'].nunique()} | {res} |")
        for extra in ("label", "split"):
            if not (m[f"{extra}_pub"] == m[f"{extra}_new"]).all():
                lines.append(f"| {fname} `{extra}` column | | | DIFFERS |")
                all_ok = False
    return lines, all_ok


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _metric(summary, key):
    return None if summary is None or key not in summary else summary[key]["last"]


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def compare_runs(new_dir: Path) -> list[str]:
    pub_dir = PUBLISHED / "grouped"
    lines = ["## Training runs (published / this run)", "",
             "| run | best epoch | val acc (best) | test acc (best ckpt) | test macro-F1 | test acc (last ckpt) | same test prediction |",
             "|---|---|---|---|---|---|---|"]
    for run in RUNS:
        new_test = _load(new_dir / "eval" / f"summary_{run}_test.json")
        if new_test is None:
            continue
        cells = [run]
        pm, nm = _load(pub_dir / "mlflow" / run / "summary.json"), _load(new_dir / "mlflow" / run / "summary.json")
        pe, ne = _metric(pm, "best_epoch"), _metric(nm, "best_epoch")
        cells.append(f"{_fmt(pe, 0)} / {_fmt(ne, 0)}")
        for tag, key in (("val", "lightning_test_accuracy"), ("test", "lightning_test_accuracy"),
                         ("test", "lightning_test_f1_macro"), ("last_test", "lightning_test_accuracy")):
            p, n = _load(pub_dir / "eval" / f"summary_{run}_{tag}.json"), _load(new_dir / "eval" / f"summary_{run}_{tag}.json")
            pv, nv = (p or {}).get(key), (n or {}).get(key)
            delta = "" if pv is None or nv is None else f" ({nv - pv:+.4f})"
            cells.append(f"{_fmt(pv)} / {_fmt(nv)}{delta}")
        pp, np_ = pub_dir / "eval" / f"preds_{run}_test.csv", new_dir / "eval" / f"preds_{run}_test.csv"
        if pp.exists() and np_.exists():
            m = pd.read_csv(pp).merge(pd.read_csv(np_), on="path", suffixes=("_pub", "_new"))
            cells.append(f"{(m.pred_pub == m.pred_new).mean():.3f} of {len(m)}")
        else:
            cells.append("n/a")
        lines.append("| " + " | ".join(cells) + " |")
    if len(lines) == 4:
        lines.append(f"| no eval/summary_*_test.json under {new_dir} | | | | | | |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clusters", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.clusters is None and args.runs is None:
        ap.error("give --clusters and/or --runs")
    lines, ok = [], True
    if args.clusters is not None:
        part, ok = compare_clusters(args.clusters)
        lines += part + [""]
    if args.runs is not None:
        lines += compare_runs(args.runs) + [""]
    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
