"""Tests of the lecture page logic (app/ui/pages/lecture_page.py).

The page is driven without a camera, a model or a real worker: the session
clock is replaced by a fake one and the worker by a stub, so every test is
deterministic and runs offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from app.core.analysis import ExponentialMovingAverage
from app.core.session import SessionClock, SessionState
from app.core.video_source import CAMERA, FILE
from app.db.local import init_db
from app.db.sessions import create_session, list_sessions, load_engagement_points
from app.ui.pages.lecture_page import LecturePage

START = datetime(2026, 9, 28, 14, 0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class StubWorker:
    """Stands in for the analysis thread; `stop` can take simulated time."""

    def __init__(self, clock: FakeClock, stop_takes_sec: float = 0.0) -> None:
        self._clock = clock
        self._stop_takes = stop_takes_sec
        self.paused: bool | None = None
        self.stopped = False

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def stop(self) -> None:
        self.stopped = True
        self._clock.now += self._stop_takes


@pytest.fixture(scope="module", autouse=True)
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _running_page(
    db_path: Path, clock: FakeClock, source_kind: str = CAMERA, stop_takes_sec: float = 0.0
) -> LecturePage:
    """A page in the RUNNING state with a stored session and a stub worker."""
    page = LecturePage(db_path=db_path)
    page._session_id = create_session(source_kind, 40, db_path, started_at=START)
    page._is_file = source_kind == FILE
    page._smoother = ExponentialMovingAverage(1)
    page._points = []
    page._clock = SessionClock(clock)
    page._clock.start()
    page._worker = StubWorker(clock, stop_takes_sec)
    return page


def test_closing_window_during_lecture_saves_the_session(db_path: Path) -> None:
    """Closing the window during a lecture finishes and stores the session.

    MainWindow.closeEvent calls LecturePage.shutdown(); the session row must
    be completed and the engagement series written, otherwise closing the
    window after a long lecture would lose all of it.
    """
    clock = FakeClock()
    page = _running_page(db_path, clock)
    clock.now = 10.0
    page._record_estimate(page._clock.elapsed_exact(), 55.0)
    clock.now = 20.0

    page.shutdown()

    [session] = list_sessions(db_path)
    assert session.duration_sec == 20
    assert session.avg_engagement == 55.0
    assert load_engagement_points(session.id, db_path) == [(10.0, 55.0)]
    assert page._clock.state == SessionState.IDLE


def test_stored_duration_excludes_time_spent_stopping_the_worker(db_path: Path) -> None:
    """Stopping the worker may block for a running inference.

    _finish_session reads the clock before that wait, so the wait is not
    added to the lecture's active time.
    """
    clock = FakeClock()
    page = _running_page(db_path, clock, stop_takes_sec=2.5)
    clock.now = 10.0

    page._finish_session()

    [session] = list_sessions(db_path)
    assert session.duration_sec == 10

