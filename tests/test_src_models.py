"""Tests for src/models: factory (torch.hub mocked), LightningModule step logic, metrics."""

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import src.models.backbones  # noqa: F401 — registers slow_r50/slowfast_r50/resnet50
from _review_helpers import ConstantNet
from src.models.base import VideoClassificationModule
from src.models.factory import MODEL_REGISTRY, create_model


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2048, 400)


class _FakePyTorchVideoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity(), _FakeHead()])


class _FakeTorchvisionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, 1000)


@pytest.fixture
def hub_calls(monkeypatch):
    calls = []

    def fake_load(repo_or_dir, model, *args, **kwargs):
        calls.append((repo_or_dir, model, kwargs))
        if model == "resnet50":
            return _FakeTorchvisionResNet()
        return _FakePyTorchVideoNet()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    return calls


def test_registry_contains_expected_models():
    assert {"slow_r50", "slowfast_r50", "resnet50"} <= set(MODEL_REGISTRY)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("does_not_exist")


@pytest.mark.parametrize("name", ["slow_r50", "slowfast_r50"])
def test_pytorchvideo_head_is_replaced(hub_calls, name):
    m = create_model(name, num_classes=3, pretrained=False)
    head = m.blocks[-1].proj
    assert isinstance(head, nn.Linear)
    assert head.in_features == 2048 and head.out_features == 3
    repo, model, kwargs = hub_calls[0]
    assert repo == "facebookresearch/pytorchvideo" and model == name
    assert kwargs["pretrained"] is False


def test_resnet50_fc_is_replaced_and_hub_pinned(hub_calls):
    m = create_model("resnet50", num_classes=3, pretrained=True)
    assert m.fc.out_features == 3
    repo, model, kwargs = hub_calls[0]
    assert repo == "pytorch/vision:v0.10.0" and model == "resnet50"
    assert kwargs["pretrained"] is True


# --------------------------------------------------------------------------- #
# _shared_step: pathway handling
# --------------------------------------------------------------------------- #
class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.seen = x
        b = x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
        return torch.zeros(b, 3) + self.p


def test_single_pathway_is_passed_as_tensor():
    rec = _Recorder()
    module = VideoClassificationModule(model=rec, num_classes=3)
    batch = {"flow_num_0": torch.rand(2, 3, 8, 16, 16), "label": torch.tensor([0, 1])}
    loss, preds, labels = module._shared_step(batch)
    assert isinstance(rec.seen, torch.Tensor) and rec.seen.ndim == 5
    assert preds.shape == (2,) and torch.equal(labels, batch["label"])
    assert loss.ndim == 0


def test_single_frame_pathway_is_squeezed_to_image_batch():
    rec = _Recorder()
    module = VideoClassificationModule(model=rec, num_classes=3)
    batch = {"flow_num_0": torch.rand(2, 3, 1, 16, 16), "label": torch.tensor([0, 1])}
    module._shared_step(batch)
    assert tuple(rec.seen.shape) == (2, 3, 16, 16)


def test_multi_pathway_is_passed_as_ordered_list():
    rec = _Recorder()
    module = VideoClassificationModule(model=rec, num_classes=3)
    slow, fast = torch.rand(2, 3, 8, 16, 16), torch.rand(2, 3, 32, 16, 16)
    batch = {"flow_num_1": fast, "flow_num_0": slow, "label": torch.tensor([0, 1])}
    module._shared_step(batch)
    assert isinstance(rec.seen, list) and len(rec.seen) == 2
    assert rec.seen[0] is slow and rec.seen[1] is fast


def test_batch_without_pathways_raises():
    module = VideoClassificationModule(model=_Recorder(), num_classes=3)
    with pytest.raises(KeyError):
        module._shared_step({"label": torch.tensor([0])})


# --------------------------------------------------------------------------- #
# metrics: logged values match a hand computation
# --------------------------------------------------------------------------- #
class _DictDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"flow_num_0": torch.zeros(3, 2, 8, 8), "label": torch.tensor(self.labels[i])}


def test_val_and_test_metrics_match_hand_computation():
    # always predicts class 0; labels: 3x0, 2x1, 1x2 -> acc 0.5
    labels = [0, 0, 0, 1, 1, 2]
    dl = DataLoader(_DictDataset(labels), batch_size=4)
    module = VideoClassificationModule(model=ConstantNet(0), num_classes=3)
    trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    val = trainer.validate(module, dataloaders=dl)[0]
    test = trainer.test(module, dataloaders=dl)[0]
    # macro F1: class0 P=0.5 R=1 F1=2/3 ; class1 F1=0 ; class2 F1=0 -> 0.2222
    assert val["val/accuracy"] == pytest.approx(0.5)
    assert val["val/f1_macro"] == pytest.approx(2 / 9, abs=1e-4)
    assert test["test/accuracy"] == pytest.approx(0.5)
    assert test["test/f1_macro"] == pytest.approx(2 / 9, abs=1e-4)
