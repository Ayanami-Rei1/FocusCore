"""Skip test modules whose heavy libraries are not installed.

The CI job installs only requirements-test.txt: the analysis core, the
database and the cloud client. Tests of the interface need PyQt6, tests of
the training code need torch and lightning; they run where those libraries
are installed, for example on the developer's machine, and are skipped in CI
instead of failing on import.
"""

import importlib

QT_TESTS = [
    "test_lecture_page.py",
    "test_profile_page.py",
    "test_reports_page.py",
    "test_workers.py",
]
TRAINING_TESTS = ["test_src_*.py"]


def _missing(*modules: str) -> bool:
    """True if any of the modules cannot be imported.

    The import is attempted, not only looked up: PyQt6 may be installed but
    unusable without the system OpenGL libraries.
    """
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            return True
    return False


collect_ignore_glob = []
if _missing("PyQt6.QtWidgets"):
    collect_ignore_glob += QT_TESTS
if _missing("torch", "lightning"):
    collect_ignore_glob += TRAINING_TESTS
