"""Run one pytest batch without unsafe interpreter-level Qt finalization."""

from __future__ import annotations

import os
import sys

import pytest


def main() -> None:
    """Finish pytest normally, then return its status without re-finalizing Qt.

    Pytest has completed fixture and plugin teardown when ``pytest.main``
    returns.  Some PyQtGraph object graphs can nevertheless segfault during
    the subsequent Python interpreter shutdown, after pytest has already
    reported that every test passed.  Each batch already runs in a disposable
    subprocess, so a direct exit safely avoids that second native teardown
    while preserving pytest's actual exit status.
    """
    exit_code = int(pytest.main(sys.argv[1:]))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
