"""Review tests for AccountService (app/cloud/account.py) on a fake cloud."""

from pathlib import Path

import pytest

from app.cloud.account import AccountService
from app.cloud.client import AuthSession, InvalidCredentialsError
from app.db.account import load_account
from app.db.local import init_db


class RevokedCloud:
    """The server no longer accepts the saved refresh token (e.g. the password
    was changed on another device and the sessions were revoked)."""

    def __init__(self) -> None:
        self.session: AuthSession | None = None

    def sign_in(self, email: str, password: str) -> AuthSession:
        self.session = AuthSession("user-1", email, "Анна", "Иванова", "access", "refresh")
        return self.session

    def restore(self, refresh_token: str) -> AuthSession:
        raise InvalidCredentialsError("refresh_token_not_found")

    def select(self, table: str, params: dict) -> list[dict]:
        return []

    def upsert(self, table: str, rows: list[dict]) -> None:
        pass

    def delete(self, table: str, ids: list[str]) -> None:
        pass


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_expired_sign_in_found_during_synchronization_forgets_the_account(db_path: Path) -> None:
    """Bug: AccountService.restore() forgets the account when the saved token is
    rejected, but synchronize() does not. After the token is revoked, every
    later sync raises InvalidCredentialsError while the account stays stored,
    so the profile page shows 'wrong email or password' and keeps the dead
    account until the user signs out by hand.
    """
    cloud = RevokedCloud()
    service = AccountService(cloud, db_path)
    service.sign_in("a@b.ru", "secret1")
    assert load_account(db_path) is not None
    cloud.session = None  # a new launch: the client has no live session yet

    with pytest.raises(InvalidCredentialsError):
        service.synchronize()

    assert load_account(db_path) is None
