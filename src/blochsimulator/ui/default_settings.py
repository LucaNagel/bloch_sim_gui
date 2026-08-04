"""Persistent defaults shared by sequence and phantom creation UIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..units import NUCLEUS_GAMMA_HZ_PER_T


@dataclass(frozen=True)
class WorkspaceDefaults:
    """Values used when a new sequence workspace or phantom is created."""

    sequence_fov_mm: tuple[float, float, float] = (220.0, 220.0, 160.0)
    phantom_fov_mm: tuple[float, float, float] = (220.0, 220.0, 220.0)
    phantom_nucleus: Optional[str] = None
    field_strength_t: float = 3.0

    @classmethod
    def from_settings(cls, settings) -> "WorkspaceDefaults":
        defaults = cls()
        if settings is None:
            return defaults

        def positive_value(key: str, fallback: float) -> float:
            try:
                value = float(settings.value(key, fallback))
            except (TypeError, ValueError):
                return fallback
            return value if np.isfinite(value) and value > 0.0 else fallback

        sequence_fov = tuple(
            positive_value(f"defaults/sequence_fov_{axis}_mm", fallback)
            for axis, fallback in zip("xyz", defaults.sequence_fov_mm)
        )
        phantom_fov = tuple(
            positive_value(f"defaults/phantom_fov_{axis}_mm", fallback)
            for axis, fallback in zip("xyz", defaults.phantom_fov_mm)
        )
        raw_nucleus = settings.value("defaults/phantom_nucleus", "auto")
        nucleus = None if raw_nucleus in (None, "", "auto") else str(raw_nucleus)
        if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            nucleus = defaults.phantom_nucleus
        field_strength_t = positive_value(
            "defaults/field_strength_t", defaults.field_strength_t
        )
        return cls(sequence_fov, phantom_fov, nucleus, field_strength_t)

    def save(self, settings) -> None:
        for axis, value in zip("xyz", self.sequence_fov_mm):
            settings.setValue(f"defaults/sequence_fov_{axis}_mm", float(value))
        for axis, value in zip("xyz", self.phantom_fov_mm):
            settings.setValue(f"defaults/phantom_fov_{axis}_mm", float(value))
        settings.setValue("defaults/phantom_nucleus", self.phantom_nucleus or "auto")
        settings.setValue("defaults/field_strength_t", float(self.field_strength_t))
