"""Canonical unit conversions used by sequence simulation.

The legacy Bloch API represents RF in gauss and gradients in gauss/cm.  The
sequence engine uses Pulseq-compatible frequency units instead: RF in Hz and
gradients in Hz/m.  Keeping conversions in one module prevents silent factors
of 100 or 10,000 at subsystem boundaries.
"""

from __future__ import annotations

import numpy as np


PROTON_GAMMA_HZ_PER_T = 42.576e6
NUCLEUS_GAMMA_HZ_PER_T = {
    "H1": PROTON_GAMMA_HZ_PER_T,
    "C13": 10.705e6,
    "P31": 17.235e6,
    "F19": 40.052e6,
    "Na23": 11.262e6,
}
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


def _nucleus_gamma_hz_per_t(nucleus: str) -> float:
    name = str(nucleus).strip()
    try:
        return float(NUCLEUS_GAMMA_HZ_PER_T[name])
    except KeyError:
        raise ValueError(f"unsupported nucleus {name!r}") from None


def rf_hz_to_gauss_for_nucleus(values, nucleus: str = "H1"):
    """Convert RF nutation frequency in Hz to physical B1 in gauss."""
    gamma_hz_per_t = _nucleus_gamma_hz_per_t(nucleus)
    return np.asarray(values) * GAUSS_PER_TESLA / gamma_hz_per_t


def gradient_hz_per_m_to_t_per_m(values, nucleus: str = "H1"):
    """Convert frequency-encoded gradient amplitude in Hz/m to T/m."""
    return np.asarray(values) / _nucleus_gamma_hz_per_t(nucleus)


def ppm_to_hz(values, field_strength_t: float, nucleus: str = "H1"):
    """Convert a relative frequency offset from ppm to Hz."""
    field = float(field_strength_t)
    if not np.isfinite(field) or field <= 0:
        raise ValueError("field_strength_t must be positive and finite")
    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        raise ValueError(f"unsupported nucleus {nucleus!r}")
    return np.asarray(values) * 1e-6 * NUCLEUS_GAMMA_HZ_PER_T[nucleus] * field


def hz_to_ppm(values, field_strength_t: float, nucleus: str = "H1"):
    """Convert a relative frequency offset from Hz to ppm."""
    scale = ppm_to_hz(1.0, field_strength_t, nucleus)
    return np.asarray(values) / scale
