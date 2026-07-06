"""Result types for streaming sequence simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SequenceSimulationResult:
    """ADC, final-state, and optional checkpoint output from one simulation.

    ``signal`` has shape ``(adc,)`` for the backward-compatible singleton
    receive channel and ``(coil, adc)`` for multiple receive coils.
    """

    signal: np.ndarray
    adc_times_s: np.ndarray
    final_magnetization: np.ndarray
    checkpoint_magnetization: Optional[np.ndarray]
    checkpoint_times_s: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    adc_gradient_moment_cyc_per_m: Optional[np.ndarray] = None

    @property
    def mx(self) -> np.ndarray:
        return self.final_magnetization[..., 0]

    @property
    def my(self) -> np.ndarray:
        return self.final_magnetization[..., 1]

    @property
    def mz(self) -> np.ndarray:
        return self.final_magnetization[..., 2]

    def to_dict(self) -> Dict[str, Any]:
        """Return a compatibility-friendly dictionary without copying arrays."""
        return {
            "signal": self.signal,
            "adc_times_s": self.adc_times_s,
            "final_magnetization": self.final_magnetization,
            "mx": self.mx,
            "my": self.my,
            "mz": self.mz,
            "checkpoint_magnetization": self.checkpoint_magnetization,
            "checkpoint_times_s": self.checkpoint_times_s,
            "adc_gradient_moment_cyc_per_m": self.adc_gradient_moment_cyc_per_m,
            "metadata": self.metadata,
        }

    def to_xarray(self):
        """Convert sparse output to an xarray Dataset with generic spatial axes."""
        import xarray as xr

        spatial_dims = [
            f"spatial_{index}" for index in range(self.final_magnetization.ndim - 1)
        ]
        signal_dims = ("adc",) if self.signal.ndim == 1 else ("coil", "adc")
        if self.signal.ndim not in (1, 2):
            raise ValueError("signal must have shape (adc,) or (coil, adc)")
        data_vars = {
            "signal": (signal_dims, self.signal),
            "final_magnetization": (
                spatial_dims + ["component"],
                self.final_magnetization,
            ),
        }
        coords = {
            "adc": np.arange(self.adc_times_s.size),
            "adc_time_s": ("adc", self.adc_times_s),
            "component": ["mx", "my", "mz"],
        }
        if self.signal.ndim == 2:
            coords["coil"] = np.arange(self.signal.shape[0])
        if self.adc_gradient_moment_cyc_per_m is not None:
            moments = np.asarray(self.adc_gradient_moment_cyc_per_m)
            if moments.shape != (self.adc_times_s.size, 3):
                raise ValueError(
                    "adc_gradient_moment_cyc_per_m must have shape (adc, 3)"
                )
            coords["gradient_axis"] = ["x", "y", "z"]
            data_vars["adc_gradient_moment_cyc_per_m"] = (
                ("adc", "gradient_axis"),
                moments,
            )
        if self.checkpoint_magnetization is not None:
            coords["checkpoint"] = self.checkpoint_times_s
            data_vars["checkpoint_magnetization"] = (
                ["checkpoint"] + spatial_dims + ["component"],
                self.checkpoint_magnetization,
            )
        attrs = {
            key: value
            for key, value in self.metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def to_cartesian_kspace(self, acquisition, *, validate: bool = True) -> np.ndarray:
        """Reshape chronological ADC data with a Cartesian acquisition layout."""
        if validate:
            acquisition.validate_adc_times(self.adc_times_s)
            if self.adc_gradient_moment_cyc_per_m is not None:
                acquisition.validate_gradient_moments(
                    self.adc_gradient_moment_cyc_per_m
                )
        return acquisition.reshape_signal(self.signal)

    def reconstruct_cartesian(
        self,
        acquisition,
        *,
        validate: bool = True,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Validate, reshape, and inverse-FFT a Cartesian ADC stream."""
        if validate:
            acquisition.validate_adc_times(self.adc_times_s)
            if self.adc_gradient_moment_cyc_per_m is not None:
                acquisition.validate_gradient_moments(
                    self.adc_gradient_moment_cyc_per_m
                )
        return acquisition.reconstruct(
            self.signal,
            norm=norm,
            coil_combine=coil_combine,
            voxel_centered=voxel_centered,
        )
