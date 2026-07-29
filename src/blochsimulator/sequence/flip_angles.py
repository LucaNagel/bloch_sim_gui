"""Flip-angle schedules for hyperpolarized sequence acquisitions."""

from __future__ import annotations

import numpy as np


VFA_REFERENCE_DOI = "10.1016/j.jmr.2007.10.011"


def variable_flip_angle_schedule(
    num_excitations: int,
    *,
    final_flip_angle_deg: float = 90.0,
) -> np.ndarray:
    """Return a constant-signal hyperpolarized variable-flip-angle schedule.

    The schedule is calculated backwards using
    ``alpha[n] = atan(sin(alpha[n + 1]))``. With a final 90 degree pulse this
    is equivalent to ``atan(1 / sqrt(N - n))`` for one-based excitation index
    ``n`` and total excitation count ``N``. The model assumes negligible T1
    decay during the schedule and complete spoiling between excitations.
    """
    if int(num_excitations) != num_excitations or num_excitations <= 0:
        raise ValueError("num_excitations must be a positive integer")
    final_angle = float(final_flip_angle_deg)
    if not np.isfinite(final_angle) or not 0.0 < final_angle <= 90.0:
        raise ValueError("final_flip_angle_deg must be in the interval (0, 90]")

    angles_rad = np.empty(int(num_excitations), dtype=float)
    angles_rad[-1] = np.deg2rad(final_angle)
    for index in range(angles_rad.size - 2, -1, -1):
        angles_rad[index] = np.arctan(np.sin(angles_rad[index + 1]))
    return np.rad2deg(angles_rad)
