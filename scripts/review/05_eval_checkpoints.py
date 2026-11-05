#!/usr/bin/env python
"""Experiments 5 and 6: evaluate the student's checkpoints through the project's own code path.

Uses OUCCGEDataset(split=..., root=...) + get_val_transforms(), create_model(...) +
VideoClassificationModule.load_from_checkpoint(ckpt, model=backbone), and Lightning
Trainer.test (the same path src/train.py uses after training). Per-clip predictions are
recorded by wrapping module._shared_step (labels/preds) and hooking the backbone output
(logits), so the numbers come from exactly the project's test_step.

--mode implements experiment 6 (temporal ablations of the (C,T,H,W) clip tensor):
  identity | shuffle (random frame order, per-clip fixed seed) | reverse | static_center
  (center frame repeated T times) | static_first

Usage:
  python scripts/review/05_eval_checkpoints.py --model slow --ckpt <ckpt> --split test --mode identity --out <dir> --tag slow_test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data.ouc_cge import OUCCGEDataset  # noqa: E402
from src.data.transforms import get_val_transforms  # noqa: E402
from src.models.base import VideoClassificationModule  # noqa: E402
from src.models.factory import create_model  # noqa: E402
import src.models.backbones  # noqa: E402,F401  (registers slow_r50 / resnet50)

PATHWAYS = {"slow": ([3.75], [8]), "resnet50": ([30], [1])}   # == src/train.py::_pathways_for_model
MODEL_NAME = {"slow": "slow_r50", "resnet50": "resnet50"}


class TemporalAblation(Dataset):
    """Wraps OUCCGEDataset and edits the time axis of flow_num_0, shape (C, T, H, W)."""

    def __init__(self, base, mode, seed=0):
        self.base, self.mode, self.seed = base, mode, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        x = d["flow_num_0"]
        T = x.shape[1]
        if self.mode == "identity":
            pass
        elif self.mode == "shuffle":
            g = torch.Generator().manual_seed(self.seed * 100003 + i)
            x = x[:, torch.randperm(T, generator=g)]
        elif self.mode == "reverse":
            x = x.flip(1)
        elif self.mode == "static_center":
            x = x[:, T // 2:T // 2 + 1].expand(-1, T, -1, -1).contiguous()
        elif self.mode == "static_first":
            x = x[:, :1].expand(-1, T, -1, -1).contiguous()
        else:
            raise ValueError(self.mode)
        d["flow_num_0"] = x
        return d


def build_backbone(model_key):
    try:
        return create_model(MODEL_NAME[model_key], num_classes=3, pretrained=False)
    except Exception as e:  # torch.hub legacy `pretrained` kwarg may be refused by new torchvision
        if model_key != "resnet50":
            raise
        print(f"create_model(resnet50) failed ({e!r}); building torchvision resnet50 directly (same architecture)")
        import torchvision
        m = torchvision.models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 3)
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PATHWAYS), required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/ouc-cge-320"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="identity")
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)

    fps, nf = PATHWAYS[args.model]
    base = OUCCGEDataset(root=args.root, split=args.split, fps=fps, num_frames=nf, transform=get_val_transforms())
    n_manifest = len(OUCCGEDataset._load_manifest(args.root / "data_labels" / f"{args.split}.csv"))
    print(f"split={args.split} manifest_rows={n_manifest} after_window_filter={len(base)} window={base.window_size}")
    ds = TemporalAblation(base, args.mode, seed=args.seed)

    backbone = build_backbone(args.model)
    module = VideoClassificationModule.load_from_checkpoint(str(args.ckpt), model=backbone, map_location="cpu")
    module.eval()

    rows, logits_buf = [], []
    hook = module.model.register_forward_hook(lambda m, i, o: logits_buf.append(o.detach().float().cpu()))
    orig = module._shared_step

    def recording_step(batch):
        loss, preds, labels = orig(batch)
        rows.extend(zip(batch["path"], labels.tolist(), preds.tolist()))
        return loss, preds, labels

    module._shared_step = recording_step
    trainer = L.Trainer(accelerator="gpu", devices=1, precision=args.precision, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, deterministic=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    t0 = time.time()
    res = trainer.test(module, dataloaders=dl, verbose=False)[0]
    dt = time.time() - t0
    hook.remove()

    df = pd.DataFrame(rows, columns=["path", "label", "pred"])
    logits = torch.cat(logits_buf).numpy()
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    for k in range(3):
        df[f"p{k}"] = probs[:, k]
    df["path"] = df["path"].str.replace(str(args.root) + "/", "", regex=False)
    df.to_csv(args.out / f"preds_{args.tag}.csv", index=False)

    y, p = df.label.values, df.pred.values
    summary = dict(tag=args.tag, model=args.model, ckpt=str(args.ckpt), split=args.split, mode=args.mode, precision=args.precision,
                   manifest_rows=int(n_manifest), evaluated=int(len(df)), dropped_by_window_filter=int(n_manifest - len(base)),
                   duplicate_paths_in_eval=int(df.path.duplicated().sum()),
                   lightning_test_accuracy=float(res["test/accuracy"]), lightning_test_f1_macro=float(res["test/f1_macro"]),
                   lightning_test_loss=float(res["test/loss"]),
                   sklearn_accuracy=float(accuracy_score(y, p)), sklearn_f1_macro=float(f1_score(y, p, average="macro")),
                   confusion_rows_true_low_mid_high=confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
                   n_errors=int((y != p).sum()), mean_max_prob=float(probs.max(1).mean()), seconds=round(dt, 1))
    with open(args.out / f"summary_{args.tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
