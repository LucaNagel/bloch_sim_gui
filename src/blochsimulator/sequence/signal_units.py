"""Display units of received signals, images and spectra.

Simulated signals are in simulation units. With a calibrated receiver they are
mol × P/P_th at the calibration's target temperature, so multiplying by that
thermal polarization gives the fully polarized equivalent amount in mol. In
SNR units every ADC sample is divided by its own noise std; an orthonormal
reconstruction then keeps the per-quadrature noise std at 1 in images and
spectra, so every value reads directly as a multiple of the noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .snr_budget import format_amount

#: Settings choices as (label, key).
SIGNAL_UNITS = (
    ("Physical amount (mol, fully polarized)", "physical"),
    ("SNR units (noise σ = 1)", "snr"),
)
DEFAULT_SIGNAL_UNIT = "physical"


def normalize_signal_unit(value) -> str:
    """Return a known unit key; anything else falls back to physical."""
    value = str(value)
    return value if value in {key for _, key in SIGNAL_UNITS} else DEFAULT_SIGNAL_UNIT


def noise_std_per_sample(result) -> Optional[np.ndarray]:
    """Per-quadrature noise std of every ADC sample, or None without noise."""
    noise = result.metadata.get("receiver_noise") or {}
    if noise.get("mode", "off") == "off":
        return None
    event_std = np.asarray(noise.get("noise_std_per_quadrature") or (), dtype=float)
    if (
        not event_std.size
        or np.any(event_std <= 0)
        or not np.all(np.isfinite(event_std))
    ):
        return None
    count = np.asarray(result.signal).shape[-1]
    samples = (result.metadata.get("receiver_acquisition") or {}).get("adc_samples")
    if samples and len(samples) == event_std.size and sum(samples) == count:
        return np.repeat(event_std, samples)
    if np.allclose(event_std, event_std[0], rtol=1e-12, atol=0):
        return np.full(count, event_std[0])
    return None


def _fully_polarized_scale(result) -> Optional[float]:
    """Factor from mol × P/P_th to fully polarized mol of a calibrated run."""
    noise = result.metadata.get("receiver_noise") or {}
    if noise.get("mode") != "calibrated":
        return None
    derived = noise.get("calibration_derived") or {}
    polarization = derived.get("target_thermal_polarization")
    if polarization is None or not np.isfinite(polarization) or polarization <= 0:
        return None
    return float(polarization)


@dataclass(frozen=True)
class SignalDisplay:
    """How one result is scaled for display.

    ``unit`` is ``"physical"`` (mol, fully polarized equivalent), ``"snr"``
    (multiples of the per-quadrature noise std) or ``"simulation"``.
    ``sample_scale`` multiplies the chronological ADC samples; in SNR units it
    differs per sample because the noise std follows each ADC event's
    bandwidth. ``note`` explains a fallback from the requested unit.
    """

    unit: str
    sample_scale: object = 1.0
    note: str = ""

    @property
    def snr(self) -> bool:
        return self.unit == "snr"

    @property
    def image_norm(self) -> Optional[str]:
        """FFT normalization: orthonormal keeps whitened noise at std 1.

        Otherwise the inverse FFT divides by the voxel count, so a voxel holds
        its share of the object amount.
        """
        return "ortho" if self.snr else None

    def spectrum_scale(self, spectral_points: int) -> float:
        """Factor for an unnormalized FFT over ``spectral_points`` samples.

        In SNR units this keeps the noise std at 1 per frequency bin; otherwise
        the bins of one peak add up to its FID amplitude at the first sample.
        """
        points = int(spectral_points)
        return 1.0 / np.sqrt(points) if self.snr else 1.0 / points

    @property
    def axis_label(self) -> str:
        return "Signal / σ" if self.snr else "Signal"

    @property
    def axis_units(self) -> Optional[str]:
        """Units for pyqtgraph, which adds the SI prefix itself."""
        return {"physical": "mol", "simulation": "a.u."}.get(self.unit)

    @property
    def description(self) -> str:
        return {
            "physical": "mol, fully polarized equivalent",
            "snr": "SNR units, noise σ = 1 per quadrature",
            "simulation": "simulation units",
        }[self.unit]

    def format_value(self, value) -> str:
        value = float(value)
        if self.unit == "physical":
            return format_amount(value)
        if self.snr:
            return f"{value:.4g} σ"
        return f"{value:.5g}"

    def scale(self, values):
        """Scale chronological samples (last axis) to the display unit."""
        if values is None:
            return None
        return np.asarray(values) * self.sample_scale


def signal_display(result, unit=DEFAULT_SIGNAL_UNIT) -> SignalDisplay:
    """Choose the display scaling of ``result`` for the requested ``unit``.

    SNR units need receiver noise on the result and physical units a
    calibrated receiver; otherwise the result stays in simulation units.
    """
    unit = normalize_signal_unit(unit)
    if unit == "snr":
        sigma = noise_std_per_sample(result)
        if sigma is not None:
            return SignalDisplay("snr", 1.0 / sigma)
        note = "SNR units need receiver noise on this result"
    else:
        factor = _fully_polarized_scale(result)
        if factor is not None:
            return SignalDisplay("physical", factor)
        note = "Physical units need a calibrated receiver"
    return SignalDisplay("simulation", 1.0, f"{note}; showing simulation units")


def display_result(result, display: SignalDisplay):
    """Copy of ``result`` with all received signals in the display unit."""
    if display.unit == "simulation":
        return result
    return replace(
        result,
        signal=display.scale(result.signal),
        signal_noiseless=display.scale(result.signal_noiseless),
        species_signal=display.scale(result.species_signal),
    )


def noise_only_result(result):
    """Copy of a displayed result that holds only its noise realization."""
    if result.signal_noiseless is None:
        raise ValueError("The result has no noiseless reference signal")
    noise = np.asarray(result.signal) - np.asarray(result.signal_noiseless).reshape(
        np.shape(result.signal)
    )
    return replace(result, signal=noise, signal_noiseless=None, species_signal=None)


def realized_noise_std(image) -> float:
    """Per-quadrature std of a complex noise image (mean over all values)."""
    values = np.asarray(image)
    if not values.size:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(values) ** 2) / 2.0))
