"""Tests for src/data/transforms.py on (T, C, H, W) float clips."""

import torch

from src.data.transforms import get_train_transforms, get_val_transforms

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def _clip(t=8, h=40, w=60, identical=True):
    torch.manual_seed(0)
    if identical:
        return torch.rand(1, 3, h, w).repeat(t, 1, 1, 1)
    return torch.rand(t, 3, h, w)


def test_train_transform_shares_random_params_across_frames():
    """Identical input frames stay identical: the same crop and flip for every frame."""
    x = _clip()
    y = get_train_transforms()(x)
    assert tuple(y.shape) == (3, 8, 224, 224)
    for t in range(1, 8):
        assert torch.equal(y[:, 0], y[:, t])


def test_train_transform_is_actually_random():
    x = _clip()
    tr = get_train_transforms()
    torch.manual_seed(1)
    outs = [tr(x) for _ in range(6)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_val_transform_is_deterministic_and_cthw():
    """The permutation keeps the frame order: frame t of the output is transformed frame t."""
    x = _clip(identical=False)
    tr = get_val_transforms()
    a, b = tr(x), tr(x)
    assert tuple(a.shape) == (3, 8, 224, 224)
    assert torch.equal(a, b)
    single = tr(x[2:3])[:, 0]
    assert torch.allclose(a[:, 2], single)


def test_normalize_is_applied_to_unit_range_floats():
    x = torch.full((2, 3, 40, 60), 0.5)
    y = get_val_transforms()(x)
    expected = (0.5 - MEAN) / STD
    assert torch.allclose(y[:, 0, 0, 0], expected, atol=1e-5)
    assert torch.allclose(y[:, 1, 100, 100], expected, atol=1e-5)
