"""Tests for scripts/export_torchscript.py.

Builds the real SLOW R50 architecture locally with pytorchvideo
(``pretrained=False``, no download) by mocking ``torch.hub.load``.
"""

import lightning as L
import pytest
import torch

import src.models.backbones  # noqa: F401 — registers slow_r50
from src.models.base import VideoClassificationModule
from src.models.factory import create_model

APP_INPUT_SHAPE = (1, 3, 8, 224, 224)  # app/core/engagement.py: NUM_FRAMES=8, CROP_SIZE=224


@pytest.fixture
def local_hub(monkeypatch):
    import pytorchvideo.models.hub as pv_hub

    def fake_load(repo_or_dir, model, *args, pretrained=False, **kwargs):
        assert pretrained is False, "tests must never download weights"
        return getattr(pv_hub, model)(pretrained=False)

    monkeypatch.setattr(torch.hub, "load", fake_load)


def _save_checkpoint(module, path):
    torch.save({
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": L.__version__,
        "epoch": 0, "global_step": 0,
    }, path)


def test_export_traces_slow_r50_and_matches_eager(local_hub, tmp_path):
    from scripts.export_torchscript import export_torchscript

    torch.manual_seed(0)
    backbone = create_model("slow_r50", num_classes=3, pretrained=False)
    module = VideoClassificationModule(model=backbone, num_classes=3)
    ckpt = tmp_path / "model.ckpt"
    _save_checkpoint(module, ckpt)

    out = export_torchscript(str(ckpt), str(tmp_path / "m.ts"))
    assert out.exists()

    ts = torch.jit.load(str(out))
    x = torch.randn(*APP_INPUT_SHAPE)
    backbone.eval()
    with torch.no_grad():
        eager = backbone(x)
        traced_a, traced_b = ts(x), ts(x)
    assert tuple(eager.shape) == (1, 3)
    assert torch.allclose(eager, traced_a, atol=1e-4)
    assert torch.equal(traced_a, traced_b)  # eval mode: no dropout noise
