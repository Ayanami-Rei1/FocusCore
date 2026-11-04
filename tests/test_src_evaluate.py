"""Tests for src/evaluate.py with a dummy model and synthetic videos (CPU)."""

import lightning as L
import pytest
import torch
from sklearn.metrics import f1_score

from _review_helpers import TinyVideoNet, make_cfg, make_root
from src.evaluate import evaluate
from src.models.base import VideoClassificationModule
from src.models.factory import MODEL_REGISTRY, register_model
from src.train import build_datamodule


def _register_tiny():
    if "review_tiny3d" not in MODEL_REGISTRY:
        register_model("review_tiny3d")(
            lambda num_classes=3, pretrained=True, **kw: TinyVideoNet(num_classes)
        )


@pytest.fixture
def synthetic(tmp_path):
    _register_tiny()
    spec = [("videos/a.mp4", 0, 70), ("videos/b.mp4", 1, 70), ("videos/c.mp4", 2, 70),
            ("videos/d.mp4", 0, 70)]
    root = make_root(tmp_path, {"train": spec, "val": spec, "test": spec})
    return make_cfg(root, tmp_path, model_name="review_tiny3d")


def test_evaluate_requires_checkpoint_path(synthetic):
    with pytest.raises(ValueError, match="checkpoint_path"):
        evaluate(synthetic)


def test_evaluate_runs_test_split_from_checkpoint(synthetic, tmp_path):
    cfg = synthetic
    torch.manual_seed(0)
    module = VideoClassificationModule(model=TinyVideoNet(3), num_classes=3, learning_rate=1e-3)
    trainer = L.Trainer(
        max_epochs=1, limit_train_batches=1, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, datamodule=build_datamodule(cfg))
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(str(ckpt))

    cfg.checkpoint_path = str(ckpt)
    results = evaluate(cfg)
    assert {"test/accuracy", "test/f1_macro", "test/loss"} <= set(results)

    # Expected numbers: run the same weights over the (filtered) test split by hand.
    loaded = VideoClassificationModule.load_from_checkpoint(str(ckpt), model=TinyVideoNet(3))
    loaded.eval()
    ds = build_datamodule(cfg)._test_dataset
    preds, labels = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            preds.append(int(loaded(s["flow_num_0"].unsqueeze(0)).argmax(1)))
            labels.append(int(s["label"]))
    acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
    f1 = f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0)
    assert results["test/accuracy"] == pytest.approx(acc, abs=1e-6)
    assert results["test/f1_macro"] == pytest.approx(f1, abs=1e-5)
