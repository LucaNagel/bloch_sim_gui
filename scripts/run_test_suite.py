"""Run pytest in bounded processes to avoid long-lived Qt cleanup buildup."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
CHUNKED_GUI_MODULES = (
    TESTS / "test_sequence_gui.py",
    TESTS / "test_sequence_workspace_features.py",
)
# Benchmarks require local phantom assets and are not part of quality validation.
EXCLUDED_MODULES = (TESTS / "test_benchmarks.py",)
GUI_BATCH_SIZE = 2
GUI_NODE_BATCH_SIZE = 10
BATCHED_PYTEST_ENV = "BLOCHSIMULATOR_BATCHED_PYTEST"


def _test_environment():
    env = os.environ.copy()
    env[BATCHED_PYTEST_ENV] = "1"
    if "MPLCONFIGDIR" not in env:
        cache_directory = Path(tempfile.gettempdir()) / "blochsimulator-matplotlib"
        cache_directory.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(cache_directory)
    return env


def _chunks(values, size):
    values = tuple(values)
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _is_gui_module(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return "PyQt5" in source or "pyqtgraph" in source


def _run_pytest(targets) -> int:
    relative_targets = [
        str(target.relative_to(ROOT)) if isinstance(target, Path) else str(target)
        for target in targets
    ]
    print(f"\npytest batch: {' '.join(relative_targets)}", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *relative_targets],
        cwd=ROOT,
        env=_test_environment(),
    )
    return int(completed.returncode)


def _collect_node_ids(module: Path):
    relative_module = str(module.relative_to(ROOT))
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", relative_module],
        cwd=ROOT,
        env=_test_environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"pytest collection failed for {relative_module}")
    node_ids = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(f"{relative_module}::")
    )
    if not node_ids:
        raise RuntimeError(f"pytest collected no tests from {relative_module}")
    return node_ids


def main() -> int:
    excluded = set(EXCLUDED_MODULES)
    modules = tuple(
        module for module in sorted(TESTS.glob("test_*.py")) if module not in excluded
    )
    chunked = set(CHUNKED_GUI_MODULES)
    gui_modules = tuple(
        module for module in modules if module not in chunked and _is_gui_module(module)
    )
    regular_modules = tuple(
        module
        for module in modules
        if module not in chunked and module not in gui_modules
    )

    batches = [regular_modules]
    batches.extend(_chunks(gui_modules, GUI_BATCH_SIZE))
    for module in CHUNKED_GUI_MODULES:
        batches.extend(_chunks(_collect_node_ids(module), GUI_NODE_BATCH_SIZE))

    for batch in batches:
        return_code = _run_pytest(batch)
        if return_code:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
