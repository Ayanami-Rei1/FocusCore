"""Export a trained checkpoint (SLOW R50 by default) to TorchScript for CPU inference.

Run from the repository root:
    python -m scripts.export_torchscript --ckpt outputs/.../checkpoints/<best>.ckpt \
        --out app/models/engagement_slow_r50.ts

The traced module takes a (1, 3, 8, 224, 224) float tensor, which is what
app/core/engagement.py builds (NUM_FRAMES=8, CROP_SIZE=224).
"""

import argparse
from pathlib import Path

import torch

from src.models.backbones import *  # noqa: F401,F403 — register models
from src.models.base import VideoClassificationModule
from src.models.factory import create_model

DEFAULT_CKPT = (
    "outputs/baseline_ouc_cge/2026-05-31_15-35-39/checkpoints/"
    "epoch=32-val/accuracy=1.0000.ckpt"
)
DEFAULT_OUT = "engagement_slow_r50.ts"
EXAMPLE_SHAPE = (1, 3, 8, 224, 224)


def export_torchscript(
    ckpt: str,
    out: str,
    model_name: str = "slow_r50",
    num_classes: int = 3,
    example_shape: tuple[int, ...] = EXAMPLE_SHAPE,
    atol: float = 1e-5,
) -> Path:
    """Trace the backbone of a Lightning checkpoint on CPU and save it to ``out``."""
    backbone = create_model(model_name, num_classes=num_classes, pretrained=False)
    module = VideoClassificationModule.load_from_checkpoint(
        checkpoint_path=ckpt, model=backbone, map_location="cpu"
    )
    model = module.model.cpu().eval()

    example = torch.randn(*example_shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        eager_out = model(example)
        traced_out = traced(example)
    if not torch.allclose(eager_out, traced_out, atol=atol):
        raise RuntimeError("traced model output differs from the eager model")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"shape of output: {tuple(eager_out.shape)}")
    print(f"saved TorchScript model to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Lightning checkpoint path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output .ts path")
    parser.add_argument("--model", default="slow_r50", help="registered model name")
    parser.add_argument("--num-classes", type=int, default=3)
    args = parser.parse_args()
    export_torchscript(args.ckpt, args.out, args.model, args.num_classes)


if __name__ == "__main__":
    main()
