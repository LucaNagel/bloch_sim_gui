"""Result types for streaming sequence simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
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

    @property
    def acquisition_dimensions(self):
        """Return explicit ADC outer dimensions when present in metadata."""
        metadata = self.metadata.get("acquisition_dimensions")
        if metadata is None:
            return None
        from .acquisition import AcquisitionDimensions

        dimensions = AcquisitionDimensions.from_metadata(metadata)
        if dimensions.num_samples != self.adc_times_s.size:
            raise ValueError(
                "acquisition dimension metadata does not match the ADC stream"
            )
        return dimensions

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
            "t": ("adc", self.adc_times_s),
            "component": ["mx", "my", "mz"],
        }
        if self.signal.ndim == 2:
            coords["coil"] = np.arange(self.signal.shape[0])
        dimensions = self.acquisition_dimensions
        if dimensions is not None:
            event_counts = np.asarray(
                dimensions.adc_event_sample_counts,
                dtype=np.int64,
            )
            coords["adc_event_index"] = (
                "adc",
                np.repeat(np.arange(event_counts.size), event_counts),
            )
            coords["readout_sample_index"] = (
                "adc",
                (
                    np.concatenate(
                        [np.arange(count, dtype=np.int64) for count in event_counts]
                    )
                    if event_counts.size
                    else np.zeros(0, dtype=np.int64)
                ),
            )
            for axis in dimensions.AXIS_NAMES:
                coords[f"{axis}_index"] = (
                    "adc",
                    dimensions.sample_indices(axis),
                )
        if self.adc_gradient_moment_cyc_per_m is not None:
            moments = np.asarray(self.adc_gradient_moment_cyc_per_m)
            if moments.shape != (self.adc_times_s.size, 3):
                raise ValueError(
                    "adc_gradient_moment_cyc_per_m must have shape (adc, 3)"
                )
            coords["gradient_axis"] = ["x", "y", "z"]
            coords["kx"] = ("adc", moments[:, 0])
            coords["ky"] = ("adc", moments[:, 1])
            coords["kz"] = ("adc", moments[:, 2])
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
        attrs["adc_order"] = (
            "chronological; adc_event_index identifies each readout and "
            "readout_sample_index identifies the sample within that readout"
        )
        dataset = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        dataset["t"].attrs.update(long_name="ADC sample time", units="s")
        dataset["adc_time_s"].attrs.update(long_name="ADC sample time", units="s")
        for axis in ("kx", "ky", "kz"):
            if axis in dataset.coords:
                dataset[axis].attrs.update(
                    long_name=f"{axis} gradient moment at ADC sample",
                    units="cycles/m",
                )
        return dataset

    def save(self, filename) -> Path:
        """Save sparse sequence output as NPZ, HDF5, or xarray NetCDF."""
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix == ".nc":
            dataset = self.to_xarray()
            if np.iscomplexobj(dataset["signal"].data):
                signal = dataset["signal"]
                dataset["signal_real"] = signal.real
                dataset["signal_imag"] = signal.imag
                dataset = dataset.drop_vars("signal")
                dataset.attrs["complex_signal"] = "signal_real + 1j*signal_imag"
            dataset.to_netcdf(path)
            return path

        arrays = {
            "signal": self.signal,
            "adc_times_s": self.adc_times_s,
            "final_magnetization": self.final_magnetization,
            "checkpoint_times_s": self.checkpoint_times_s,
        }
        if self.checkpoint_magnetization is not None:
            arrays["checkpoint_magnetization"] = self.checkpoint_magnetization
        if self.adc_gradient_moment_cyc_per_m is not None:
            arrays["adc_gradient_moment_cyc_per_m"] = self.adc_gradient_moment_cyc_per_m
        metadata_json = json.dumps(self.metadata, default=str)
        if suffix == ".npz":
            np.savez_compressed(path, metadata_json=np.asarray(metadata_json), **arrays)
            return path
        if suffix in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "w") as handle:
                for name, values in arrays.items():
                    handle.create_dataset(name, data=values)
                handle.attrs["metadata_json"] = metadata_json
                handle.attrs["format"] = "blochsimulator-sequence-result"
                handle.attrs["version"] = 1
            return path
        raise ValueError("sequence results require a .nc, .npz, .h5, or .hdf5 file")

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
