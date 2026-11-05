"""Tests for Hydra config composition and src/train.py (synthetic data, CPU)."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from mlflow import MlflowClient
from omegaconf import OmegaConf

from _training_helpers import TinyVideoNet, make_cfg, make_root
from src.conf.config import register_configs
from src.models.factory import MODEL_REGISTRY, register_model
from src.train import _pathways_for_model, build_datamodule, train

REPO = Path(__file__).resolve().parents[1]


def _compose(overrides):
    register_configs()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_default_composition():
    cfg = _compose([])
    assert cfg.model.name == "slow_r50" and cfg.data.name == "ouc_cge"
    assert cfg.trainer.max_epochs == 50 and cfg.trainer.precision == "32"
    assert cfg.mlflow.experiment_name == "default_experiment"
    assert cfg.paths.checkpoints.endswith("/checkpoints")
    assert cfg.trainer.checkpoint.monitor == cfg.trainer.early_stopping.monitor == "val/accuracy"
    assert cfg.trainer.get("limit_train_batches", None) is None


def test_experiment_presets_compose():
    base = _compose(["+experiment=baseline_ouc_cge"])
    assert base.experiment.name == "baseline_ouc_cge" and base.model.name == "slow_r50"
    assert base.mlflow.experiment_name == "baseline_ouc_cge"

    r50 = _compose(["+experiment=resnet50_ouc_cge_baseline"])
    assert r50.model.name == "resnet50" and r50.trainer.precision == "bf16-mixed"
    assert r50.data.batch_size == 32 and r50.data.num_workers == 16


@pytest.mark.parametrize("override", ["+checkpoint_path=x.ckpt", "checkpoint_path=x.ckpt"])
def test_checkpoint_path_override_forms(override):
    assert _compose([override, "model=slow"]).checkpoint_path == "x.ckpt"


def test_pathways_for_model():
    def cfg(name):
        return SimpleNamespace(model=SimpleNamespace(name=name))

    assert _pathways_for_model(cfg("slowfast_r50")) == ([3.75, 15.0], [8, 32])
    assert _pathways_for_model(cfg("resnet50")) == ([30], [1])
    assert _pathways_for_model(cfg("slow_r50")) == ([3.75], [8])


def _register_tiny():
    if "review_tiny3d" not in MODEL_REGISTRY:
        register_model("review_tiny3d")(
            lambda num_classes=3, pretrained=True, **kw: TinyVideoNet(num_classes)
        )


def _synthetic_root(tmp_path):
    spec = [("videos/a.mp4", 0, 70), ("videos/b.mp4", 1, 70), ("videos/c.mp4", 2, 70)]
    return make_root(tmp_path, {"train": spec, "val": spec, "test": spec})


def test_build_datamodule_on_synthetic_root(tmp_path):
    root = _synthetic_root(tmp_path)
    cfg = OmegaConf.create({
        "model": {"name": "slow_r50"},
        "data": {"root": str(root), "batch_size": 2, "num_workers": 0, "pin_memory": False},
    })
    dm = build_datamodule(cfg)
    assert len(dm._train_dataset) == len(dm._val_dataset) == len(dm._test_dataset) == 3
    batch = next(iter(dm.train_dataloader()))
    assert tuple(batch["flow_num_0"].shape) == (2, 3, 8, 224, 224)
    assert batch["label"].dtype == torch.int64
    val_a = next(iter(dm.val_dataloader()))["flow_num_0"]
    val_b = next(iter(dm.val_dataloader()))["flow_num_0"]
    assert torch.equal(val_a, val_b)


def test_train_end_to_end_on_synthetic_clips(tmp_path):
    """train(): fit 1 epoch, test best ckpt, log to MLflow; checkpoint names are flat."""
    _register_tiny()
    root = _synthetic_root(tmp_path)
    cfg = make_cfg(root, tmp_path, model_name="review_tiny3d")

    result = train(cfg)
    assert isinstance(result, float) and 0.0 <= result <= 1.0

    ckpt_dir = Path(cfg.paths.checkpoints)
    entries = sorted(p.name for p in ckpt_dir.iterdir())
    assert all(p.is_file() for p in ckpt_dir.iterdir()), (
        f"checkpoint callback created nested directories: {entries}"
    )
    assert "last.ckpt" in entries
    assert any(re.fullmatch(r"epoch=\d{2}-val_acc=\d\.\d{4}\.ckpt", e) for e in entries), entries
    assert (Path(cfg.paths.output_dir) / "config.yaml").exists()

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    metrics = runs[0].data.metrics
    for key in ("val/accuracy", "test/accuracy", "test/f1_macro", "best_val/accuracy",
                "best_epoch", "final_epoch"):
        assert key in metrics, key
    assert runs[0].data.params["best_checkpoint_path"].endswith(".ckpt")
