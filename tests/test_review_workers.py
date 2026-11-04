"""Review tests for the background workers (app/ui/workers.py).

The workers are exercised without a camera and without the TorchScript
model: the capture is a stub and the main loop is called directly in the
test thread, so no Qt event loop and no GPU are involved.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import pytest  # noqa: E402

from app.core.offline import iter_clips  # noqa: E402
from app.core.video_source import CAMERA, VideoSource  # noqa: E402
from app.ui import workers  # noqa: E402
from app.ui.workers import FRAME_STRIDE, MAX_FAILED_READS, NUM_FRAMES, CameraAnalysisWorker  # noqa: E402

TRAINING_CLIP_FPS = 3.75  # src/data/ouc_cge.py: 8 frames at 3.75 fps from 30 fps video
RUNAWAY_READS = 20 * MAX_FAILED_READS


class DeadCapture:
    """A camera that never returns a frame, e.g. unplugged during the lecture."""

    def __init__(self, worker: CameraAnalysisWorker) -> None:
        self._worker = worker
        self.reads = 0

    def read(self) -> tuple[bool, None]:
        self.reads += 1
        if self.reads >= RUNAWAY_READS:
            self._worker._running = False  # protect the test from an endless loop
        return False, None


def test_paused_camera_worker_reports_a_lost_camera(monkeypatch) -> None:
    """Bug: while paused, _process reads the camera but ignores failed reads.

    An unplugged camera returns immediately, so the loop spins at 100 % CPU
    and the failure is never reported until the user resumes.
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


def test_clip_sampling_matches_training_at_30_fps() -> None:
    offset, clip = next(iter_clips(range(300), 30.0, 1.0, NUM_FRAMES, FRAME_STRIDE))
    span_sec = (clip[-1] - clip[0]) / 30.0
    assert span_sec == pytest.approx((NUM_FRAMES - 1) / TRAINING_CLIP_FPS)


@pytest.mark.xfail(
    strict=True,
    reason="review: design decision - FRAME_STRIDE is fixed at 8 (correct only for "
    "30 fps sources), so a 25 fps file or a 60 fps camera feeds the model clips "
    "spanning a different duration than the 3.75 fps training clips; the stride "
    "should be derived from the source fps (round(fps / 3.75))",
)
@pytest.mark.parametrize("fps", [25.0, 60.0])
def test_clip_sampling_matches_training_at_other_frame_rates(fps: float) -> None:
    offset, clip = next(iter_clips(range(int(fps * 10)), fps, 1.0, NUM_FRAMES, FRAME_STRIDE))
    span_sec = (clip[-1] - clip[0]) / fps
    assert span_sec == pytest.approx((NUM_FRAMES - 1) / TRAINING_CLIP_FPS, rel=0.05)


def test_module_constants_are_what_the_tests_assume() -> None:
    """Guard: the assertions above are about these exact values."""
    assert workers.FRAME_STRIDE == 8
    assert workers.NUM_FRAMES == 8
