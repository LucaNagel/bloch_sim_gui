#!/usr/bin/env bash
set -euo pipefail

# Simple helper to build the GUI as a desktop app with PyInstaller.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-packaging}"

echo "Using Python: $PYTHON"
echo "Project root: $ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"
python -m pip install pyinstaller

echo "Building C extension in place..."
python setup.py build_ext --inplace

echo "Running PyInstaller..."
pyinstaller "$ROOT/bloch_gui.spec" --noconfirm

echo "Build complete. Artifacts are under dist/BlochSimulator"
