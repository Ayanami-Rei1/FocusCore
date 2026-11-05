"""Tests for src/data: manifest parsing, filtering, sampling, failure handling.

All tests run on CPU with tiny synthetic mp4 clips (see tests/_review_helpers).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from _review_helpers import (
    decode_index,
    make_root,
    write_manifest,
    write_textured_video,
    write_video,
)
from src.data.ouc_cge import OUCCGEDataset
from src.data.transforms import get_train_transforms, get_val_transforms

REPO = Path(__file__).resolve().parents[1]
REAL_TEST_MANIFEST = REPO / "data" / "ouc-cge" / "data_labels" / "test.csv"


def _ds(root, split, fps, num_frames, transform=None, use_cache=False):
    return OUCCGEDataset(
        root=root, split=split, fps=fps, num_frames=num_frames,
        transform=transform, use_cache=use_cache,
    )


def _indices(clip: torch.Tensor) -> list[int]:
    """Source frame index of every frame in a raw (T, C, H, W) clip."""
    return [decode_index(clip[t]) for t in range(clip.shape[0])]


# --------------------------------------------------------------------------- #
# helper sanity
# --------------------------------------------------------------------------- #
def test_helper_frame_encoding_roundtrip(tmp_path):
    path = write_video(tmp_path / "v.mp4", 25)
    cap = cv2.VideoCapture(str(path))
    got = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = torch.from_numpy(bgr).permute(2, 0, 1).float() / 255.0
        got.append(decode_index(t))
    cap.release()
    assert got == list(range(25))


# --------------------------------------------------------------------------- #
# manifest parsing
# --------------------------------------------------------------------------- #
class TestManifest:
    def test_crlf_no_trailing_newline_and_stray_row(self, tmp_path):
        path = write_manifest(
            tmp_path, "test",
            [("videos/high/a.mp4", 2), ("videos/low/b.mp4", 0),
             ("videos/anonymity/a.mp4", None), ("videos/mid/c.mp4", 1)],
            crlf=True, trailing_newline=False,
        )
        df = OUCCGEDataset._load_manifest(path)
        assert df["path"].tolist() == [
            "videos/high/a.mp4", "videos/low/b.mp4", "videos/mid/c.mp4"
        ]
        assert df["label"].tolist() == [2, 0, 1]
        assert str(df["label"].dtype).startswith("int")
        assert not any("\r" in p for p in df["path"])

    def test_lf_with_trailing_newline(self, tmp_path):
        path = write_manifest(
            tmp_path, "train", [("a.mp4", 0), ("b.mp4", 2)],
            crlf=False, trailing_newline=True,
        )
        df = OUCCGEDataset._load_manifest(path)
        assert len(df) == 2 and df["label"].tolist() == [0, 2]

    @pytest.mark.skipif(not REAL_TEST_MANIFEST.exists(), reason="manifest not pulled")
    def test_real_published_test_manifest(self):
        df = OUCCGEDataset._load_manifest(REAL_TEST_MANIFEST)
        assert len(df) == 771  # 772 rows minus the unlabeled anonymity row
        assert set(df["label"].unique()) == {0, 1, 2}
        assert "videos/anonymity/a.mp4" not in set(df["path"])
        assert not any("\r" in p for p in df["path"])


# --------------------------------------------------------------------------- #
# filtering / cache
# --------------------------------------------------------------------------- #
class TestFiltering:
    def test_drops_short_missing_and_keeps_long(self, tmp_path):
        root = make_root(tmp_path, {"train": [
            ("videos/ok.mp4", 1, 20),
            ("videos/short.mp4", 0, 5),
            ("videos/missing.mp4", 2, 0),
        ]})
        ds = _ds(root, "train", fps=[15], num_frames=[4])  # window 8
        assert ds.window_size == 8
        assert len(ds) == 1
        assert ds.sample["path"].tolist() == ["videos/ok.mp4"]

    def test_filter_cache_is_keyed_by_split_and_window(self, tmp_path):
        root = make_root(tmp_path, {"test": [
            ("videos/a.mp4", 0, 10), ("videos/b.mp4", 1, 70),
        ]})
        ds8 = _ds(root, "test", fps=[15], num_frames=[4], use_cache=True)
        ds64 = _ds(root, "test", fps=[3.75], num_frames=[8], use_cache=True)
        assert len(ds8) == 2 and len(ds64) == 1
        cache_dir = root / "data_labels" / ".cache"
        assert (cache_dir / "test_w8.csv").exists()
        assert (cache_dir / "test_w64.csv").exists()
        # The cache is served verbatim on the next construction (no re-scan).
        (root / "videos" / "a.mp4").unlink()
        assert len(_ds(root, "test", fps=[15], num_frames=[4], use_cache=True)) == 2
        assert len(_ds(root, "test", fps=[15], num_frames=[4], use_cache=False)) == 1


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
class TestSampling:
    def test_rates_and_window_from_fps(self, tmp_path):
        root = make_root(tmp_path, {"val": [("v.mp4", 0, 70)]})
        ds = _ds(root, "val", fps=[3.75, 15, 30], num_frames=[8, 32, 64])
        assert ds.list_of_rates == [8, 2, 1]
        assert ds.window_size == 64

    def test_val_center_window_is_deterministic(self, tmp_path):
        root = make_root(tmp_path, {"val": [("v.mp4", 2, 25)]})
        ds = _ds(root, "val", fps=[30], num_frames=[3])  # window 3
        a, b = ds[0], ds[0]
        assert _indices(a["flow_num_0"]) == [11, 12, 13]  # (25 - 3 + 1) // 2
        assert torch.equal(a["flow_num_0"], b["flow_num_0"])
        assert a["label"].item() == 2 and a["label"].dtype == torch.int64
        assert a["path"].endswith("v.mp4")

    def test_test_split_center_window_with_stride(self, tmp_path):
        root = make_root(tmp_path, {"test": [("v.mp4", 1, 25)]})
        ds = _ds(root, "test", fps=[15], num_frames=[4])  # rate 2, window 8
        assert _indices(ds[0]["flow_num_0"]) == [9, 11, 13, 15]  # start (25-8+1)//2

    def test_train_window_is_random_and_within_bounds(self, tmp_path):
        root = make_root(tmp_path, {"train": [("v.mp4", 1, 25)]})
        ds = _ds(root, "train", fps=[30], num_frames=[3])
        np.random.seed(0)
        starts = set()
        for _ in range(40):
            idx = _indices(ds[0]["flow_num_0"])
            assert idx == [idx[0], idx[0] + 1, idx[0] + 2]
            assert 0 <= idx[0] <= 25 - 3
            starts.add(idx[0])
        assert len(starts) > 1

    def test_clip_of_exactly_window_length_starts_at_zero(self, tmp_path):
        root = make_root(tmp_path, {"train": [("v.mp4", 0, 8)], "val": [("v.mp4", 0, 8)]})
        for split in ("train", "val"):
            ds = _ds(root, split, fps=[15], num_frames=[4])  # window 8 == clip
            assert len(ds) == 1
            assert _indices(ds[0]["flow_num_0"]) == [0, 2, 4, 6]

    def test_padding_when_stream_ends_early(self, tmp_path):
        root = make_root(tmp_path, {"val": [("v.mp4", 0, 25)]})
        ds = _ds(root, "val", fps=[30], num_frames=[4])
        path = root / "v.mp4"
        cap = cv2.VideoCapture(str(path))
        clip = ds._get_frames(cap, path, start_point=22, num_frames=4, rate=1)
        cap.release()
        assert tuple(clip.shape) == (4, 3, 8, 8)
        assert _indices(clip) == [22, 23, 24, 24]  # last decodable frame repeated

    def test_single_frame_pathway_matches_resnet50_config(self, tmp_path):
        root = make_root(tmp_path, {"val": [("v.mp4", 0, 25)]})
        ds = _ds(root, "val", fps=[30], num_frames=[1])
        assert ds.window_size == 1
        raw = ds[0]["flow_num_0"]
        assert tuple(raw.shape) == (1, 3, 8, 8)
        assert _indices(raw) == [12]
        ds_t = _ds(root, "val", fps=[30], num_frames=[1], transform=get_val_transforms())
        assert tuple(ds_t[0]["flow_num_0"].shape) == (3, 1, 224, 224)

    def test_transform_output_layout_is_cthw(self, tmp_path):
        root = make_root(tmp_path, {"train": [("v.mp4", 0, 25)]})
        ds = _ds(root, "train", fps=[15], num_frames=[4], transform=get_train_transforms())
        out = ds[0]["flow_num_0"]
        assert tuple(out.shape) == (3, 4, 224, 224)
        assert out.dtype == torch.float32


# --------------------------------------------------------------------------- #
# failure handling / multi-pathway
# --------------------------------------------------------------------------- #
def _textured_root(tmp_path):
    """One training clip with spatial texture (so crops / flips are visible)."""
    root = tmp_path / "ouc"
    write_textured_video(root / "v.mp4", 25)
    write_manifest(root, "train", [("v.mp4", 0)])
    return root


class TestFailureHandling:
    def test_two_pathways_decode_same_frames_without_transform(self, tmp_path):
        ds = _ds(_textured_root(tmp_path), "train", fps=[30, 30], num_frames=[4, 4])
        s = ds[0]
        assert torch.equal(s["flow_num_0"], s["flow_num_1"])

    def test_multi_pathway_random_transforms_are_consistent(self, tmp_path):
        """SlowFast pathways must get the same random crop / flip.

        With fps=[30, 30], num_frames=[4, 4] both pathways are the *same*
        frames, so after the training transform they must still be equal.
        """
        ds = _ds(_textured_root(tmp_path), "train", fps=[30, 30], num_frames=[4, 4],
                 transform=get_train_transforms())
        torch.manual_seed(0)
        for _ in range(5):
            s = ds[0]
            assert torch.equal(s["flow_num_0"], s["flow_num_1"]), (
                "pathways received different random crop/flip parameters"
            )

    def test_eval_split_raises_instead_of_substituting_neighbour(self, tmp_path, monkeypatch):
        """A corrupt val/test clip must not silently become its neighbour."""
        root = make_root(tmp_path, {"val": [
            ("videos/a.mp4", 0, 20), ("videos/b.mp4", 1, 20),
        ]})
        ds = _ds(root, "val", fps=[30], num_frames=[4])
        orig = OUCCGEDataset._get_frames

        def flaky(self, cap, video_path, start_point, num_frames, rate):
            if str(video_path).endswith("a.mp4"):
                raise RuntimeError("first sampled frame is undecodable (simulated)")
            return orig(self, cap, video_path, start_point, num_frames, rate)

        monkeypatch.setattr(OUCCGEDataset, "_get_frames", flaky)
        assert ds[1]["label"].item() == 1
        with pytest.raises(RuntimeError):
            ds[0]

    def test_train_split_falls_back_to_next_clip(self, tmp_path, monkeypatch):
        root = make_root(tmp_path, {"train": [
            ("videos/a.mp4", 0, 20), ("videos/b.mp4", 1, 20),
        ]})
        ds = _ds(root, "train", fps=[30], num_frames=[4])
        orig = OUCCGEDataset._get_frames

        def flaky(self, cap, video_path, start_point, num_frames, rate):
            if str(video_path).endswith("a.mp4"):
                raise RuntimeError("simulated")
            return orig(self, cap, video_path, start_point, num_frames, rate)

        monkeypatch.setattr(OUCCGEDataset, "_get_frames", flaky)
        s = ds[0]
        assert s["path"].endswith("b.mp4") and s["label"].item() == 1

    def test_train_split_fallback_is_bounded(self, tmp_path, monkeypatch):
        """If every clip fails, raise a clear error instead of recursing forever.

        The recursive fallback keeps every VideoCapture of the chain open (the
        ``finally: cap.release()`` only runs after the recursive call returns),
        so we count live handles to prove the chain is bounded.
        """
        root = make_root(tmp_path, {"train": [
            ("videos/a.mp4", 0, 20), ("videos/b.mp4", 1, 20), ("videos/c.mp4", 2, 20),
        ]})
        ds = _ds(root, "train", fps=[30], num_frames=[4])

        real_capture = cv2.VideoCapture
        live = {"n": 0, "max": 0}

        class TrackingCapture:
            def __init__(self, *a, **k):
                self._cap = real_capture(*a, **k)
                self._released = False
                live["n"] += 1
                live["max"] = max(live["max"], live["n"])

            def release(self):
                if not self._released:
                    self._released = True
                    live["n"] -= 1
                self._cap.release()

            def __getattr__(self, name):
                return getattr(self._cap, name)

        def broken(self, cap, video_path, start_point, num_frames, rate):
            raise RuntimeError("simulated")

        monkeypatch.setattr(cv2, "VideoCapture", TrackingCapture)
        monkeypatch.setattr(OUCCGEDataset, "_get_frames", broken)
        with pytest.raises(RuntimeError, match="consecutive"):
            ds[0]
        assert live["max"] <= 2, f"{live['max']} video handles open at once (unbounded recursion)"
        assert live["n"] == 0
