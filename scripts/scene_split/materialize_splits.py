#!/usr/bin/env python
"""Write the scene-disjoint dataset roots from the stored split assignment.

data/splits/split_<name>.csv (path,label,split,cluster for all 7701 clips) is the exact assignment
the scene-disjoint runs were trained and evaluated on.  This script turns each of them into a
dataset root the project reads unchanged:

  <out-base>/<name>/data_labels/{train,val,test}.csv   project manifest format ("path label")
  <out-base>/<name>/videos -> ../ouc-cge-320/videos     relative symlink, the clips are not copied

No clustering, no random numbers: the manifests are deterministic (same rows, same order).

Before writing, every split file is checked: paths unique, labels in {0,1,2}, splits in
{train,val,test}, and no scene cluster in more than one split.

Usage:
  python scripts/scene_split/materialize_splits.py                     # grp27a, grp27b, grp17 under data/
  python scripts/scene_split/materialize_splits.py --names grp27a
  python -m src.train +experiment=baseline_ouc_cge data.root=data/grp27a ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SPLITS, write_dataset_root

REPO = Path(__file__).resolve().parents[2]


def check(df: pd.DataFrame, name: str) -> None:
    missing = {"path", "label", "split", "cluster"} - set(df.columns)
    if missing:
        raise SystemExit(f"split_{name}.csv: missing columns {sorted(missing)}")
    if df["path"].duplicated().any():
        raise SystemExit(f"split_{name}.csv: duplicated clip paths")
    if not set(df["label"].unique()) <= {0, 1, 2}:
        raise SystemExit(f"split_{name}.csv: labels outside 0..2")
    if not set(df["split"].unique()) <= set(SPLITS):
        raise SystemExit(f"split_{name}.csv: unknown split names {sorted(set(df['split'].unique()) - set(SPLITS))}")
    crossing = df.groupby("cluster")["split"].nunique()
    if (crossing > 1).any():
        raise SystemExit(f"split_{name}.csv: {int((crossing > 1).sum())} scene clusters appear in more than one split")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", nargs="+", default=["grp27a", "grp27b", "grp17"])
    ap.add_argument("--splits-dir", type=Path, default=REPO / "data/splits")
    ap.add_argument("--out-base", type=Path, default=REPO / "data")
    ap.add_argument("--videos-link", default="../ouc-cge-320/videos",
                    help="target of <root>/videos, relative to the dataset root")
    args = ap.parse_args()

    for name in args.names:
        src = args.splits_dir / f"split_{name}.csv"
        if not src.exists():
            raise SystemExit(f"{src} not found")
        df = pd.read_csv(src)
        check(df, name)
        root = args.out_base / name
        write_dataset_root(df, root, args.videos_link)
        parts = []
        for s in SPLITS:
            g = df[df["split"] == s]
            parts.append(f"{s} {len(g)} clips / {g['cluster'].nunique()} scenes "
                         f"(low/mid/high {np.bincount(g['label'].values, minlength=3).tolist()})")
        print(f"{root}: " + "; ".join(parts))
        if not (root / "videos").is_dir():
            print(f"  note: {root / 'videos'} does not resolve yet; fetch the clips with "
                  f"`dvc pull data/ouc-cge-320.dvc`")


if __name__ == "__main__":
    main()
