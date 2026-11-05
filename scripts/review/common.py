"""Shared helpers for the review scripts (manifest parsing, id parsing, dataset roots)."""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]
CLASS_FOLDERS = {"low": 0, "mid": 1, "high": 2}
_ID_RE = re.compile(r"videos/(low|mid|high)/view(\d+)\.mp4$")


def load_manifest(path: Path) -> pd.DataFrame:
    """Same tolerant parsing as src/data/ouc_cge.py::OUCCGEDataset._load_manifest."""
    df = pd.read_csv(path, sep=" ", names=["path", "label"])
    df["path"] = df["path"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["path", "label"])
    df = df[df["path"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def load_all_manifests(root: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        df = load_manifest(root / "data_labels" / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    parsed = df["path"].str.extract(_ID_RE)
    df["cls_folder"] = parsed[0]
    df["view_id"] = pd.to_numeric(parsed[1], errors="coerce").fillna(-1).astype(int)
    return df


def load_cache(cache_dir: Path, splits=SPLITS) -> dict:
    """Load the per-split npz files written by 01_frame_cache.py into one dict of arrays."""
    # allow_pickle=True: pandas-3 string columns are stored as object arrays; the files are
    # produced by our own 01_frame_cache.py, so this is safe.
    parts = [np.load(cache_dir / f"frames_{s}.npz", allow_pickle=True) for s in splits]
    out = {}
    for key in parts[0].files:
        arr = np.concatenate([p[key] for p in parts], axis=0)
        if arr.dtype == object:
            arr = arr.astype(str)
        out[key] = arr
    return out


def write_dataset_root(df: pd.DataFrame, root: Path, videos_link: str = "../ouc-cge-320/videos") -> None:
    """Write a dataset root that OUCCGEDataset can read unchanged.

    df: one row per clip with columns path ("videos/<class>/viewN.mp4"), label, split.
    Creates <root>/data_labels/{train,val,test}.csv in the project manifest format
    ("path label", space separated, no header, rows in the order of df) and a relative
    symlink <root>/videos -> videos_link, so the clips are not copied.
    """
    root = Path(root)
    (root / "data_labels").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        g = df[df["split"] == split]
        with open(root / "data_labels" / f"{split}.csv", "w") as fh:
            for path, label in zip(g["path"], g["label"]):
                fh.write(f"{path} {int(label)}\n")
    link = root / "videos"
    if not link.exists() and not link.is_symlink():
        os.symlink(videos_link, link)
