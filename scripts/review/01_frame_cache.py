#!/usr/bin/env python
"""Experiment 1: frame cache for OUC-CGE.

For every clip in train/val/test manifests decode the FIRST, CENTER and LAST frame
(full sequential decode with cv2 so the frame count is exact), and store:
  * a 32x32 RGB uint8 thumbnail of each of the three frames
  * an ImageNet-pretrained torchvision ResNet50 global-pool feature (2048-d, fp16)
    of each of the three frames (256-resize + 224 center-crop, ImageNet eval protocol)
One npz per split in --out. Unreadable clips are kept with ok=False and zero rows.

--thumbs-only skips the ResNet50 features (no torch, no GPU, about 100 MB of RAM for the
arrays instead of 3.5 GB). The thumbnails are all that the scene clusters (03b) and the
grouped splits (07) need; the features are only used by 02, 03 and 04.

Usage:
  python scripts/review/01_frame_cache.py --root data/ouc-cge-320 --out outputs/review_cache
  python scripts/review/01_frame_cache.py --root data/ouc-cge-320 --out outputs/review_cache --thumbs-only
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SPLITS, load_all_manifests  # noqa: E402

cv2.setNumThreads(1)


def prep224(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = 256.0 / min(h, w)
    nh, nw = max(256, int(round(h * s))), max(256, int(round(w * s)))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (nh - 224) // 2, (nw - 224) // 2
    return r[y0:y0 + 224, x0:x0 + 224]


def decode(args):
    idx, full_path, want_crops = args
    out = dict(idx=idx, ok=False, n_frames=0, n_reported=0, fps=0.0, w=0, h=0)
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return out
    out["n_reported"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    if n == 0:
        return out
    out.update(ok=True, n_frames=n, h=int(frames[0].shape[0]), w=int(frames[0].shape[1]))
    thumbs, crops = [], []
    for i in (0, n // 2, n - 1):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        thumbs.append(cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA))
        if want_crops:
            crops.append(prep224(rgb))
    out["thumbs"] = np.stack(thumbs)
    if want_crops:
        out["crops"] = np.stack(crops)
    return out


@np.errstate(all="ignore")
def extract_features(crops: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch
    import torchvision

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Same weights the student's resnet50 backbone starts from
    # (torch.hub pytorch/vision resnet50 pretrained=True == IMAGENET1K_V1).
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    feats = np.zeros((len(crops), 2048), dtype=np.float16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            feats[i:i + batch] = model(x).float().cpu().numpy().astype(np.float16)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1),
                    help="decode processes (each holds one fully decoded clip in memory)")
    ap.add_argument("--thumbs-only", action="store_true",
                    help="skip the ResNet50 features: CPU only, enough for 03b and 07")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N clips per split")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_all_manifests(args.root)
    if args.limit:
        df = df.groupby("split").head(args.limit).reset_index(drop=True)
    print(f"manifest rows: {len(df)} " + " ".join(f"{s}={int((df.split == s).sum())}" for s in SPLITS), flush=True)

    t0 = time.time()
    want_crops = not args.thumbs_only
    jobs = [(i, str(args.root / p), want_crops) for i, p in enumerate(df["path"])]
    results = [None] * len(jobs)
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(decode, jobs, chunksize=4)):
            results[r["idx"]] = r
            if (k + 1) % 1000 == 0:
                print(f"decoded {k + 1}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    print(f"decode done in {time.time() - t0:.0f}s", flush=True)

    n = len(df)
    ok = np.array([r["ok"] for r in results])
    thumbs = np.zeros((n, 3, 32, 32, 3), dtype=np.uint8)
    crops = np.zeros((n, 3, 224, 224, 3), dtype=np.uint8) if want_crops else None
    for i, r in enumerate(results):
        if r["ok"]:
            thumbs[i] = r["thumbs"]
            if want_crops:
                crops[i] = r["crops"]
    meta = {k: np.array([r[k] for r in results]) for k in ("n_frames", "n_reported", "fps", "w", "h")}
    print(f"unreadable clips: {int((~ok).sum())}", flush=True)
    for i in np.where(~ok)[0]:
        print("  BAD", df.path[i], df.split[i], flush=True)

    feats = None
    if want_crops:
        t0 = time.time()
        feats = extract_features(crops.reshape(-1, 224, 224, 3)).reshape(n, 3, 2048)
        feats[~ok] = 0
        print(f"features done in {time.time() - t0:.0f}s", flush=True)
    else:
        print("--thumbs-only: no ResNet50 features (02, 03 and 04 need them; 03b and 07 do not)", flush=True)

    for s in SPLITS:
        m = (df.split == s).values
        arrays = dict(
            path=df.path.values[m].astype(str), label=df.label.values[m].astype(np.int64),
            split=df.split.values[m].astype(str), cls_folder=df.cls_folder.values[m].astype(str),
            view_id=df.view_id.values[m].astype(np.int64), ok=ok[m],
            n_frames=meta["n_frames"][m].astype(np.int64), n_reported=meta["n_reported"][m].astype(np.int64),
            fps=meta["fps"][m].astype(np.float32), w=meta["w"][m].astype(np.int64), h=meta["h"][m].astype(np.int64),
            thumbs=thumbs[m],
        )
        if feats is not None:
            arrays["feats"] = feats[m]
        np.savez_compressed(args.out / f"frames_{s}.npz", **arrays)
        print(f"saved {s}: {int(m.sum())} clips, ok={int(ok[m].sum())}", flush=True)
    # Clip-length summary, useful for the window-64 filter in the dataset class.
    nf = meta["n_frames"][ok]
    print("n_frames percentiles (1,5,25,50,75,95,99):", np.percentile(nf, [1, 5, 25, 50, 75, 95, 99]).round(0))
    print("clips with reported frame count < 64 (dropped by OUCCGEDataset window filter):",
          int((meta["n_reported"] < 64).sum()), "of", n)
    print("reported != actual frame count:", int((meta["n_reported"][ok] != nf).sum()))


if __name__ == "__main__":
    main()
