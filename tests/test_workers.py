"""Tests of the background workers (app/ui/workers.py).

The workers are exercised without a camera and without the TorchScript
model: the capture is a stub and the main loop is called directly in the
test thread, so no Qt event loop and no GPU are involved.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core.offline import TRAINING_CLIP_FPS, frame_stride, iter_clips
from app.core.video_source import CAMERA, VideoSource
from app.ui import workers
from app.ui.workers import MAX_FAILED_READS, NUM_FRAMES, CameraAnalysisWorker

RUNAWAY_READS = 20 * MAX_FAILED_READS


class DeadCapture:
    """A camera that never returns a frame, e.g. unplugged during the lecture."""

    def __init__(self, worker: CameraAnalysisWorker) -> None:
        self._worker = worker
        self.reads = 0

    def get(self, prop: int) -> float:
        return 30.0

    def read(self) -> tuple[bool, None]:
        """Fail every read; stop the worker after RUNAWAY_READS to end a runaway loop."""
        self.reads += 1
        if self.reads >= RUNAWAY_READS:
            self._worker._running = False
        return False, None


def test_paused_camera_worker_reports_a_lost_camera(monkeypatch) -> None:
    """A camera that stops delivering frames is reported even while paused.

    An unplugged camera returns from read() immediately; without the check
    the loop would spin at full CPU and the failure would surface only after
    the user resumes.
    """
    monkeypatch.setattr(CameraAnalysisWorker, "msleep", staticmethod(lambda ms: None))
    worker = CameraAnalysisWorker(VideoSource(CAMERA, camera_index=0), interval_sec=1.0)
    messages: list[str] = []
    worker.failed.connect(messages.append)
    worker.set_paused(True)
    capture = DeadCapture(worker)

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker._process(capture, model=None, executor=executor)

    assert messages, "a camera that stops delivering frames must be reported even while paused"
    assert capture.reads == MAX_FAILED_READS


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0])
def test_clip_sampling_matches_training_at_any_frame_rate(fps: float) -> None:
    """A clip spans the same duration as the 3.75 fps training clips."""
    stride = frame_stride(fps)
    offset, clip = next(iter_clips(range(int(fps * 10)), fps, 1.0, NUM_FRAMES, stride))
    span_sec = (clip[-1] - clip[0]) / fps
    assert span_sec == pytest.approx((NUM_FRAMES - 1) / TRAINING_CLIP_FPS, rel=0.05)


def test_frame_stride_matches_training_sampling() -> None:
    """30 fps video was sampled with stride 8 during training."""
    assert frame_stride(30.0) == 8
    assert frame_stride(25.0) == 7
    assert frame_stride(60.0) == 16
    assert frame_stride(1.0) == 1


def test_module_constants_are_what_the_tests_assume() -> None:
    """Guard: the assertions above are about these exact values."""
    assert workers.NUM_FRAMES == 8
    assert TRAINING_CLIP_FPS == 3.75
