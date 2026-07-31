#!/usr/bin/env bash
set -euo pipefail

# Run the source GUI with the same Python environment used for the desktop app.

BLOCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCH_VENV_DIR="${VENV_DIR:-$BLOCH_ROOT/.venv-packaging}"

# shellcheck source=desktop_python.sh
source "$BLOCH_ROOT/scripts/desktop_python.sh"
bloch_prepare_desktop_venv "$BLOCH_ROOT" "$BLOCH_VENV_DIR"

if [[ "$BLOCH_VENV_CREATED" == "1" ]] || ! "$BLOCH_VENV_PYTHON" -c \
  'import PyQt5, OpenGL, h5py, imageio, nbformat, numpy, pypulseq, pyqtgraph, scipy' \
  >/dev/null 2>&1; then
  "$BLOCH_VENV_PYTHON" -m pip install --upgrade pip
  "$BLOCH_VENV_PYTHON" -m pip install -r "$BLOCH_ROOT/requirements.txt"
fi

source_package_dir="$(
  "$BLOCH_VENV_PYTHON" -c \
    'import pathlib, blochsimulator; print(pathlib.Path(blochsimulator.__file__).resolve().parent)' \
    2>/dev/null || true
)"
if [[ "$source_package_dir" != "$BLOCH_ROOT/src/blochsimulator" ]]; then
  "$BLOCH_VENV_PYTHON" -m pip install \
    --no-build-isolation --no-deps --editable "$BLOCH_ROOT"
fi

echo "Running source GUI with: $($BLOCH_VENV_PYTHON --version)"
echo "Interpreter: $BLOCH_VENV_PYTHON"

cd "$BLOCH_ROOT"
"$BLOCH_VENV_PYTHON" setup.py build_ext --inplace
exec "$BLOCH_VENV_PYTHON" "$BLOCH_ROOT/bloch_gui.py" "$@"
