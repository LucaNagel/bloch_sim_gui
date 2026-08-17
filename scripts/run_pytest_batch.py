"""Run one pytest batch without unsafe native Qt finalization."""

from __future__ import annotations

import os
import sys

import pytest


def main() -> None:
    """Return pytest's status without entering unsafe native Qt cleanup.

    Pytest's final garbage collection and Python's interpreter shutdown can
    both destroy PyQtGraph object graphs after every test has passed.  Each
    batch already runs in a disposable subprocess, so the worker skips those
    native teardown paths while preserving pytest's actual exit status.
    """
    # Pytest's built-in unraisableexception plugin forces cyclic garbage
    # collection while pytest is unconfiguring.  PyQtGraph ViewBox cycles can
    # segfault in that collection after every test has already passed.  These
    # disposable workers intentionally leave native Qt cleanup to os._exit,
    # so disable only that teardown hook here.
    pytest_arguments = ["-p", "no:unraisableexception", *sys.argv[1:]]
    exit_code = int(pytest.main(pytest_arguments))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
