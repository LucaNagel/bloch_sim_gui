"""Probe-oriented sequence simulation result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class SequenceProbeResult:
    """Time-resolved magnetization for explicit position/frequency probes.

    ``magnetization`` has shape ``(time, position, frequency, component)``.
    It is intentionally separate from :class:`SequenceSimulationResult`, which
    represents ADC-centric imaging output.
    """

    time_s: np.ndarray
    positions_m: np.ndarray
    frequency_offsets_hz: np.ndarray
    magnetization: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mx(self) -> np.ndarray:
        return self.magnetization[..., 0]

    @property
    def my(self) -> np.ndarray:
        return self.magnetization[..., 1]

    @property
    def mz(self) -> np.ndarray:
        return self.magnetization[..., 2]

    @property
    def mxy(self) -> np.ndarray:
        return self.mx + 1j * self.my

    @property
    def coherent_mxy(self) -> np.ndarray:
        """Complex ensemble mean over positions, retaining dephasing effects."""
        return np.mean(self.mxy, axis=1)

    @property
    def coherent_mxy_magnitude(self) -> np.ndarray:
        """Magnitude of the complex position ensemble, not mean spin magnitude."""
        return np.abs(self.coherent_mxy)

    def to_xarray(self):
        """Return a coordinate-aware xarray representation."""
        import xarray as xr

        return xr.Dataset(
            data_vars={
                "magnetization": (
                    ("time", "position", "frequency", "component"),
                    self.magnetization,
                ),
                "mx": (("time", "position", "frequency"), self.mx),
                "my": (("time", "position", "frequency"), self.my),
                "mz": (("time", "position", "frequency"), self.mz),
                "mxy_magnitude": (
                    ("time", "position", "frequency"),
                    np.abs(self.mxy),
                ),
                "coherent_mxy_real": (
                    ("time", "frequency"),
                    np.real(self.coherent_mxy),
                ),
                "coherent_mxy_imag": (
                    ("time", "frequency"),
                    np.imag(self.coherent_mxy),
                ),
                "coherent_mxy_magnitude": (
                    ("time", "frequency"),
                    self.coherent_mxy_magnitude,
                ),
            },
            coords={
                "time": ("time", self.time_s, {"units": "s"}),
                "position": np.arange(self.positions_m.shape[0]),
                "frequency": (
                    "frequency",
                    self.frequency_offsets_hz,
                    {"units": "Hz"},
                ),
                "component": ["mx", "my", "mz"],
                "x": ("position", self.positions_m[:, 0], {"units": "m"}),
                "y": ("position", self.positions_m[:, 1], {"units": "m"}),
                "z": ("position", self.positions_m[:, 2], {"units": "m"}),
            },
            attrs=dict(self.metadata),
        )
