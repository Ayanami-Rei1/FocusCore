"""Review tests for the profile page state handling (app/ui/pages/profile_page.py).

The page gets a fake AccountService, and the TaskWorker callbacks are called
directly, so no thread and no network are involved.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.cloud.client import CloudError, CloudUnavailableError  # noqa: E402
from app.db.account import StoredAccount  # noqa: E402
from app.ui.pages.profile_page import (  # noqa: E402
    LOGIN_VIEW,
    PROFILE_VIEW,
    WRONG_CREDENTIALS_TEXT,
    ProfilePage,
)

ACCOUNT = StoredAccount("user-1", "a@b.ru", "Анна", "Иванова", "refresh")


class FakeService:
    """AccountService stand-in whose stored account the test controls."""

    def __init__(self) -> None:
        self.account: StoredAccount | None = None
        self.is_configured = True


@pytest.fixture(scope="module", autouse=True)
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    "error",
    [CloudUnavailableError("upload timed out"), CloudError("HTTP 403")],
    ids=["unavailable", "server_error"],
)
def test_sign_in_whose_synchronization_failed_still_unlocks_the_window(error: Exception) -> None:
    """Bug: AccountService.sign_in saves the account first and synchronizes after.

    When the synchronization step fails (network timeout while uploading points,
    a server error), the account is stored but `signed_in` is never emitted, so
    MainWindow keeps the sidebar hidden: the user is signed in (next launch is
    unlocked) yet cannot leave the profile page without restarting.
    """
    service = FakeService()
    page = ProfilePage(service)
    assert page.stack.currentIndex() == LOGIN_VIEW
    unlocked: list[bool] = []
    page.signed_in.connect(lambda: unlocked.append(True))

    service.account = ACCOUNT  # sign_in stored the account, then synchronize raised
    page._on_task_failed(error, WRONG_CREDENTIALS_TEXT)

    assert not page.requires_sign_in
    assert unlocked == [True]
    assert page.stack.currentIndex() == PROFILE_VIEW


def test_background_sync_failure_on_the_profile_view_does_not_emit_signed_in() -> None:
    """Regression guard for the fix: a failed sync of an already signed-in user
    must not re-trigger the sign-in handling (which switches to the lecture page)."""
    service = FakeService()
    service.account = ACCOUNT
    page = ProfilePage(service)
    assert page.stack.currentIndex() == PROFILE_VIEW
    unlocked: list[bool] = []
    page.signed_in.connect(lambda: unlocked.append(True))

    page._on_task_failed(CloudUnavailableError("offline"), WRONG_CREDENTIALS_TEXT)

    assert unlocked == []
    assert page.stack.currentIndex() == PROFILE_VIEW
