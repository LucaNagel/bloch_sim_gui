"""Project-level pytest entry behavior."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
BATCHED_PYTEST_ENV = "BLOCHSIMULATOR_BATCHED_PYTEST"


def _requests_complete_suite(arguments) -> bool:
    """Return whether a CLI invocation requests the unfiltered test suite."""
    targets = []
    for argument in map(str, arguments):
        if argument in {"-q", "--quiet"}:
            continue
        if argument.startswith("-"):
            return False
        targets.append(argument)
    if not targets:
        return True
    if len(targets) != 1 or "::" in targets[0]:
        return False
    return Path(targets[0]).resolve() == TESTS


def pytest_cmdline_main(config):
    """Delegate full-suite invocations to the Qt-safe batched runner."""
    if os.environ.get(BATCHED_PYTEST_ENV):
        return None
    if not _requests_complete_suite(config.invocation_params.args):
        return None
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_test_suite.py")],
        cwd=ROOT,
        env=os.environ.copy(),
    )
    return int(completed.returncode)
