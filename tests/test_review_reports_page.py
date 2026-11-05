"""Review tests for the PDF export path handling (app/ui/pages/reports_page.py)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from app.core.report import SessionReport  # noqa: E402
from app.db.local import init_db  # noqa: E402
from app.ui.pages.reports_page import ReportsPage  # noqa: E402

REPORT = SessionReport("id", datetime(2026, 9, 28, 14, 0), 600, 40, [(0.0, 80.0)])


@pytest.fixture(scope="module", autouse=True)
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(tmp_path: Path) -> ReportsPage:
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return ReportsPage(db_path)


def _choose(monkeypatch, chosen: str) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (chosen, "PDF (*.pdf)"))
    )


def test_pdf_suffix_is_appended_to_a_name_with_a_dot(page: ReportsPage, tmp_path: Path, monkeypatch) -> None:
    """Bug: Path.with_suffix replaces everything after the last dot, so a file
    name typed as 'Лекция 26.09' is saved as 'Лекция 26.pdf'."""
    _choose(monkeypatch, str(tmp_path / "Лекция 26.09"))
    assert page._ask_export_path(REPORT) == tmp_path / "Лекция 26.09.pdf"


def test_pdf_suffix_is_kept_and_added_in_the_simple_cases(page: ReportsPage, tmp_path: Path, monkeypatch) -> None:
    _choose(monkeypatch, str(tmp_path / "report.PDF"))
    assert page._ask_export_path(REPORT) == tmp_path / "report.PDF"
    _choose(monkeypatch, str(tmp_path / "report"))
    assert page._ask_export_path(REPORT) == tmp_path / "report.pdf"
    _choose(monkeypatch, "")
    assert page._ask_export_path(REPORT) is None
