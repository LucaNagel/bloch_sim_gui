"""Workspace data-directory helpers used by desktop file dialogs."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return the source workspace or an explicit user data root."""
    configured = os.environ.get("BLOCHSIMULATOR_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.home() / "BlochSimulator"


def workspace_directory(name: str) -> Path:
    """Return and create a named persistent workspace directory."""
    if name not in {"sequences", "phantoms", "exports"}:
        raise ValueError(f"unsupported workspace directory {name!r}")
    path = workspace_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
