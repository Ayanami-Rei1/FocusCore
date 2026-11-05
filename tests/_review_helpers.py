"""Shared helpers for the src/ review tests (CPU only, synthetic data).

Synthetic videos are tiny mp4 files written with cv2.VideoWriter. Frame ``i``
is a flat frame whose intensity encodes ``i``; ``decode_index`` recovers it
from a decoded (C, H, W) float tensor, which lets tests assert *which* source
frames the dataset sampled.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

FRAME_LEVELS = 25  # intensity = 5 + 10 * (i % 25) survives mp4v quantisation


def frame_value(i: int) -> int:
    return 5 + 10 * (i % FRAME_LEVELS)


def write_video(path: Path, n_frames: int, size: int = 8, fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    assert writer.isOpened(), "cv2 cannot write mp4 in this environment"
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), frame_value(i), dtype=np.uint8))
    writer.release()
    return path


def write_textured_video(path: Path, n_frames: int, size: int = 32, fps: int = 30) -> Path:
    """Clip whose frames have spatial structure, so crops/flips are visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 200, (size, size, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    assert writer.isOpened(), "cv2 cannot write mp4 in this environment"
    for i in range(n_frames):
        writer.write(np.clip(texture.astype(int) + 2 * i, 0, 255).astype(np.uint8))
    writer.release()
    return path


def decode_index(frame: torch.Tensor) -> int:
    """Inverse of ``frame_value`` for a float (C, H, W) frame in 0..1."""
    mean = float(frame.mean()) * 255.0
    return int(round((mean - 5.0) / 10.0))


def write_manifest(
    root: Path,
    split: str,
    rows: list[tuple[str, int | None]],
    crlf: bool = True,
    trailing_newline: bool = False,
) -> Path:
    """Write ``data_labels/<split>.csv`` in the published format.

    The real manifests use CRLF line endings, have no trailing newline and
    (test split) contain a row without a label; the defaults reproduce that.
    """
    labels_dir = root / "data_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    eol = "\r\n" if crlf else "\n"
    lines = [p if lbl is None else f"{p} {lbl}" for p, lbl in rows]
    text = eol.join(lines) + (eol if trailing_newline else "")
    path = labels_dir / f"{split}.csv"
    path.write_bytes(text.encode())
    return path


def make_root(
    tmp_path: Path, clips: dict[str, list[tuple[str, int, int]]]
) -> Path:
    """Create a dataset root.

    ``clips`` maps split -> list of (relative_path, label, n_frames). A clip
    with ``n_frames <= 0`` is listed in the manifest but not written to disk.
    """
    root = tmp_path / "ouc"
    for split, spec in clips.items():
        rows = []
        for rel, label, n in spec:
            if n > 0:
                write_video(root / rel, n)
            rows.append((rel, label))
        write_manifest(root, split, rows)
    return root


def make_cfg(root: Path, tmp_path: Path, model_name: str):
    """Full structured Config (as train.py/evaluate.py expect) for CPU tests."""
    from omegaconf import OmegaConf

    from src.conf.config import Config, ExperimentConfig, MLflowConfig, PathsConfig
    from src.conf.data import OUCCGEDataConfig
    from src.conf.model import ModelConfig
    from src.conf.trainer import EarlyStoppingConfig, TrainerConfig

    cfg = Config(
        experiment=ExperimentConfig(name="review_synthetic", seed=0),
        mlflow=MLflowConfig(tracking_uri=f"file:{tmp_path}/mlruns"),
        paths=PathsConfig(output_dir=str(tmp_path / "out")),
        data=OUCCGEDataConfig(
            root=str(root), batch_size=2, num_workers=0, pin_memory=False
        ),
        model=ModelConfig(name=model_name, pretrained=False, learning_rate=1e-3),
        trainer=TrainerConfig(
            max_epochs=1, min_epochs=1, accelerator="cpu", devices="1",
            log_every_n_steps=1,
            early_stopping=EarlyStoppingConfig(min_epochs=0, patience=100),
        ),
    )
    return OmegaConf.structured(cfg)


class TinyVideoNet(nn.Module):
    """Minimal (B, 3, T, H, W) -> (B, num_classes) backbone for CPU tests."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=(1, 8, 8), stride=(1, 8, 8))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).mean(dim=(2, 3, 4)))


class ConstantNet(nn.Module):
    """Always predicts ``cls`` (used to make metrics predictable)."""

    def __init__(self, cls: int = 0, num_classes: int = 3):
        super().__init__()
        self.cls = cls
        self.num_classes = num_classes
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        logits = torch.full((x.shape[0], self.num_classes), -10.0, device=x.device)
        logits[:, self.cls] = 10.0
        return logits + 0.0 * self.dummy
