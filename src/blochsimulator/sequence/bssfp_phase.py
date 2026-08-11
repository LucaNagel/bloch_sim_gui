"""Shared phase conventions for balanced-SSFP Pulseq builders."""

from __future__ import annotations

import numpy as np


def wrap_phase_deg(phase_deg: float) -> float:
    """Return a finite phase in the half-open interval [0, 360)."""
    phase = float(phase_deg)
    if not np.isfinite(phase):
        raise ValueError("phase_deg must be finite")
    return float(np.mod(phase, 360.0))


def advance_bssfp_phase_deg(
    phase_deg: float,
    *,
    elapsed_s: float,
    frequency_offset_hz: float = 0.0,
    phase_increment_deg: float = 0.0,
) -> float:
    """Advance an RF/receiver phase in a coherent offset-frequency frame.

    ``phase_increment_deg`` is the user-selected bSSFP phase cycle.  The
    frequency term is the carrier phase accumulated during ``elapsed_s``.
    Keeping both terms here prevents an off-resonant RF carrier from being
    restarted at the same laboratory-frame phase on every TR.
    """
    elapsed = float(elapsed_s)
    frequency = float(frequency_offset_hz)
    increment = float(phase_increment_deg)
    if not np.all(np.isfinite((elapsed, frequency, increment))):
        raise ValueError(
            "elapsed_s, frequency_offset_hz, and phase increment must be finite"
        )
    if elapsed < 0:
        raise ValueError("elapsed_s must be non-negative")
    return wrap_phase_deg(float(phase_deg) + increment + 360.0 * frequency * elapsed)


def pulseq_phase_offset_rad(
    phase_at_center_deg: float,
    *,
    frequency_offset_hz: float,
    event_center_s: float,
) -> float:
    """Convert a desired event-centre phase to a Pulseq phase offset.

    Pulseq applies ``phase_offset + 2*pi*frequency_offset*t`` using time local
    to the RF or ADC event.  Subtracting the local event-centre evolution here
    makes ``phase_at_center_deg`` the unambiguous phase used by all builders.
    """
    frequency = float(frequency_offset_hz)
    center = float(event_center_s)
    if not np.all(np.isfinite((frequency, center))):
        raise ValueError("frequency_offset_hz and event_center_s must be finite")
    if center < 0:
        raise ValueError("event_center_s must be non-negative")
    return float(
        np.deg2rad(wrap_phase_deg(phase_at_center_deg))
        - 2.0 * np.pi * frequency * center
    )
