#!/usr/bin/env bash

# Shared Python-runtime selection for the source GUI and PyInstaller build.
# This file is intended to be sourced by the launch/build helpers.

bloch_prepare_desktop_venv() {
  local project_root="${1:?project root is required}"
  local venv_dir="${2:?virtual-environment path is required}"
  local version_file="$project_root/.python-version"

  if [[ ! -f "$version_file" ]]; then
    echo "Error: Python version file not found: $version_file" >&2
    return 1
  fi

  local expected_series
  expected_series="$(tr -d '[:space:]' < "$version_file")"
  if [[ ! "$expected_series" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Error: $version_file must contain a major.minor version." >&2
    return 1
  fi

  local requested_python="${BLOCH_PYTHON:-${PYTHON:-}}"
  local -a candidates=()
  if [[ -n "$requested_python" ]]; then
    candidates+=("$requested_python")
  else
    candidates+=("python$expected_series" "python3" "python")
  fi

  local candidate
  local candidate_series
  local selected_python=""
  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    candidate_series="$(
      "$candidate" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"
    if [[ "$candidate_series" == "$expected_series" ]]; then
      selected_python="$(
        "$candidate" -c 'import os, sys; print(os.path.realpath(sys.executable))'
      )"
      break
    fi
  done

  if [[ -z "$selected_python" ]]; then
    echo "Error: BlochSimulator desktop development requires Python $expected_series." >&2
    echo "Set BLOCH_PYTHON to a matching interpreter path." >&2
    return 1
  fi

  BLOCH_VENV_CREATED=0
  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating desktop environment with $selected_python"
    "$selected_python" -m venv "$venv_dir"
    BLOCH_VENV_CREATED=1
  fi

  local venv_python="$venv_dir/bin/python"
  if [[ ! -x "$venv_python" ]]; then
    venv_python="$venv_dir/Scripts/python.exe"
  fi
  if [[ ! -x "$venv_python" ]]; then
    echo "Error: $venv_dir is not a usable Python virtual environment." >&2
    return 1
  fi

  local venv_series
  venv_series="$(
    "$venv_python" -c \
      'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
  )"
  if [[ "$venv_series" != "$expected_series" ]]; then
    echo "Error: $venv_dir uses Python $venv_series; expected $expected_series." >&2
    echo "Move the stale environment aside and run the command again." >&2
    return 1
  fi

  BLOCH_DESKTOP_PYTHON="$selected_python"
  BLOCH_DESKTOP_PYTHON_SERIES="$expected_series"
  BLOCH_VENV_PYTHON="$venv_python"
  export BLOCH_DESKTOP_PYTHON
  export BLOCH_DESKTOP_PYTHON_SERIES
  export BLOCH_VENV_CREATED
  export BLOCH_VENV_PYTHON
}
