"""Canonical unit conversions used by sequence simulation.

The legacy Bloch API represents RF in gauss and gradients in gauss/cm.  The
sequence engine uses Pulseq-compatible frequency units instead: RF in Hz and
gradients in Hz/m.  Keeping conversions in one module prevents silent factors
of 100 or 10,000 at subsystem boundaries.
"""

from __future__ import annotations

import numpy as np


PROTON_GAMMA_HZ_PER_T = 42.576e6
GAUSS_PER_TESLA = 1.0e4
CM_PER_M = 100.0
LEGACY_GAMMA_RAD_PER_S_PER_GAUSS = 26753.0
LEGACY_GAMMA_HZ_PER_GAUSS = LEGACY_GAMMA_RAD_PER_S_PER_GAUSS / (2.0 * np.pi)


def rf_gauss_to_hz(values):
    """Convert complex RF amplitude from gauss to nutation frequency in Hz."""
    return np.asarray(values) * LEGACY_GAMMA_HZ_PER_GAUSS


def rf_hz_to_gauss(values):
    """Convert complex RF nutation frequency in Hz to gauss."""
    return np.asarray(values) / LEGACY_GAMMA_HZ_PER_GAUSS


def gradient_g_per_cm_to_hz_per_m(values):
    """Convert physical gradients from G/cm to Pulseq-style Hz/m."""
    return np.asarray(values) * LEGACY_GAMMA_HZ_PER_GAUSS * CM_PER_M


def gradient_hz_per_m_to_g_per_cm(values):
    """Convert Pulseq-style gradients from Hz/m to physical G/cm."""
    return np.asarray(values) / (LEGACY_GAMMA_HZ_PER_GAUSS * CM_PER_M)


def gradient_t_per_m_to_g_per_cm(values):
    """Convert T/m to G/cm (1 T/m = 100 G/cm)."""
    return np.asarray(values) * (GAUSS_PER_TESLA / CM_PER_M)
