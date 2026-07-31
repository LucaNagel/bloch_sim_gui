#!/usr/bin/env bash
set -euo pipefail

# Simple helper to build the GUI as a desktop app with PyInstaller.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-packaging}"
PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$ROOT/.pyinstaller}"
export PYINSTALLER_CONFIG_DIR

# shellcheck source=desktop_python.sh
source "$ROOT/scripts/desktop_python.sh"
bloch_prepare_desktop_venv "$ROOT" "$VENV_DIR"

echo "Desktop Python series: $BLOCH_DESKTOP_PYTHON_SERIES"
echo "Build interpreter: $BLOCH_VENV_PYTHON"
echo "Project root: $ROOT"

"$BLOCH_VENV_PYTHON" -m pip install --upgrade pip
"$BLOCH_VENV_PYTHON" -m pip install -r "$ROOT/requirements.txt"
"$BLOCH_VENV_PYTHON" -m pip install pyinstaller
"$BLOCH_VENV_PYTHON" -m pip install \
  --no-build-isolation --no-deps --editable "$ROOT"

echo "Building C extension in place..."
cd "$ROOT"
"$BLOCH_VENV_PYTHON" setup.py build_ext --inplace

echo "Running PyInstaller..."
"$BLOCH_VENV_PYTHON" -m PyInstaller "$ROOT/bloch_gui.spec" --noconfirm

echo "Build complete. Artifacts are under dist/BlochSimulator"
