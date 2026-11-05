"""Tests for src/training/callbacks.py (MLflow file store in tmp, CPU only)."""

from types import SimpleNamespace

import lightning as L
import mlflow
import pytest
import torch
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from torch.utils.data import DataLoader, Dataset

from _review_helpers import ConstantNet
from src.models.base import VideoClassificationModule
from src.training.callbacks import EarlyStoppingWithPatience, MLflowModelCheckpoint


class _DictDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"flow_num_0": torch.zeros(3, 2, 8, 8), "label": torch.tensor(self.labels[i])}


def _loaders():
    """Val set where the 2 sanity-check batches score 1.0 but the epoch scores 0.5."""
    train = DataLoader(_DictDataset([0, 0, 0, 0]), batch_size=4)
    val = DataLoader(_DictDataset([0] * 8 + [1] * 8), batch_size=4)
    return train, val


def _trainer(tmp_path, callbacks, logger=False, max_epochs=1):
    return L.Trainer(
        max_epochs=max_epochs, accelerator="cpu", devices=1, logger=logger,
        callbacks=callbacks, num_sanity_val_steps=2, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )


@pytest.fixture
def restore_mlflow_uri():
    old = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(old)


def test_mlflow_best_metric_ignores_sanity_check(tmp_path):
    uri = f"file:{tmp_path}/mlruns"
    logger = MLFlowLogger(experiment_name="review", tracking_uri=uri)
    cb = MLflowModelCheckpoint(monitor="val/accuracy", mode="max")
    module = VideoClassificationModule(model=ConstantNet(0), num_classes=3)
    train, val = _loaders()
    _trainer(tmp_path, [cb], logger=logger).fit(module, train, val)

    hist = MlflowClient(tracking_uri=uri).get_metric_history(logger.run_id, "best_val/accuracy")
    assert [m.value for m in hist] == pytest.approx([0.5]), (
        "best_val/accuracy was taken from the 2-batch sanity check, not a real epoch"
    )
    assert cb.best_value == pytest.approx(0.5)


def test_early_stopping_ignores_sanity_check(tmp_path):
    cb = EarlyStoppingWithPatience(monitor="val/accuracy", patience=5, min_epochs=0, mode="max")
    module = VideoClassificationModule(model=ConstantNet(0), num_classes=3)
    train, val = _loaders()
    _trainer(tmp_path, [cb]).fit(module, train, val)
    assert cb.best_value == pytest.approx(0.5)
    assert cb.wait_count == 0


def test_early_stopping_patience_and_min_epochs_unit():
    cb = EarlyStoppingWithPatience(monitor="val/accuracy", patience=2, min_epochs=2, mode="max")
    seq = [0.9, 0.95, 0.5, 0.4, 0.4]  # epochs 0,1 ignored (min_epochs), best 0.5 at 2
    stops = []
    for epoch, v in enumerate(seq):
        tr = SimpleNamespace(
            current_epoch=epoch, sanity_checking=False, should_stop=False,
            callback_metrics={"val/accuracy": torch.tensor(v)},
        )
        cb.on_validation_epoch_end(tr, None)
        stops.append(tr.should_stop)
    assert cb.best_value == pytest.approx(0.5)
    assert stops == [False, False, False, False, True]


def test_on_train_end_logs_hydra_dir_via_logger_tracking_uri(tmp_path, restore_mlflow_uri):
    """Artifacts must go to the logger's tracking URI, not the global mlflow one."""
    uri = f"file:{tmp_path}/a"
    logger = MLFlowLogger(experiment_name="review", tracking_uri=uri)
    run_id = logger.run_id  # creates the run in store A
    mlflow.set_tracking_uri(f"file:{tmp_path}/b")  # a different global store

    out = tmp_path / "out"
    (out / ".hydra").mkdir(parents=True)
    (out / ".hydra" / "config.yaml").write_text("a: 1\n")
    trainer = SimpleNamespace(
        is_global_zero=True, logger=logger, current_epoch=3,
        checkpoint_callback=None, default_root_dir=str(out),
    )
    MLflowModelCheckpoint().on_train_end(trainer, None)

    client = MlflowClient(tracking_uri=uri)
    paths = [a.path for a in client.list_artifacts(run_id, "hydra_config")]
    assert "hydra_config/config.yaml" in paths
    assert client.get_metric_history(run_id, "final_epoch")[-1].value == 3
    assert client.get_run(run_id).data.params["output_dir"] == str(out)
