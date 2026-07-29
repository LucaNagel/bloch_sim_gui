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
    pool_names: tuple = ()
    species_signal: Optional[np.ndarray] = None
    final_pool_magnetization: Optional[np.ndarray] = None
    checkpoint_pool_magnetization: Optional[np.ndarray] = None

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

    @property
    def spectroscopic_acquisition(self):
        """Return an explicit 2D CSI layout when present in result metadata."""
        metadata = self.metadata.get("spectroscopic_acquisition")
        if metadata is None:
            return None
        from .acquisition import SpectroscopicAcquisition

        acquisition = SpectroscopicAcquisition.from_metadata(metadata)
        if acquisition.num_samples != self.adc_times_s.size:
            raise ValueError("CSI acquisition metadata does not match the ADC stream")
        return acquisition

    @property
    def cartesian_acquisition(self):
        """Return an explicit 2D Cartesian layout when present in metadata."""
        metadata = self.metadata.get("cartesian_acquisition")
        if metadata is None:
            return None
        from .acquisition import CartesianAcquisition

        acquisition = CartesianAcquisition.from_metadata(metadata)
        if acquisition.num_samples != self.adc_times_s.size:
            raise ValueError(
                "Cartesian acquisition metadata does not match the ADC stream"
            )
        return acquisition

    @property
    def cartesian_acquisition_frames(self):
        """Return explicit Cartesian frame layouts when present in metadata."""
        metadata = self.metadata.get("cartesian_acquisition_frames")
        if metadata is None:
            return None
        from .acquisition import CartesianAcquisitionFrames

        frames = CartesianAcquisitionFrames.from_metadata(metadata)
        if frames.dimensions.num_samples != self.adc_times_s.size:
            raise ValueError("Cartesian frame metadata does not match the ADC stream")
        return frames

    @property
    def cartesian_acquisition_volumes(self):
        """Return explicit Cartesian 3D volume layouts when present."""
        metadata = self.metadata.get("cartesian_acquisition_volumes")
        if metadata is None:
            return None
        from .acquisition import CartesianAcquisitionVolumes

        volumes = CartesianAcquisitionVolumes.from_metadata(metadata)
        if volumes.frames.dimensions.num_samples != self.adc_times_s.size:
            raise ValueError("Cartesian volume metadata does not match the ADC stream")
        return volumes

    @property
    def spiral_acquisition(self):
        """Return explicit 2D spiral frame metadata when present."""
        metadata = self.metadata.get("spiral_acquisition")
        if metadata is None:
            return None
        from .acquisition import SpiralAcquisition

        acquisition = SpiralAcquisition.from_metadata(metadata)
        if acquisition.num_samples != self.adc_times_s.size:
            raise ValueError(
                "spiral acquisition metadata does not match the ADC stream"
            )
        return acquisition

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
            "pool_names": self.pool_names,
            "species_signal": self.species_signal,
            "final_pool_magnetization": self.final_pool_magnetization,
            "checkpoint_pool_magnetization": self.checkpoint_pool_magnetization,
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
        cartesian = self.cartesian_acquisition
        cartesian_frames = self.cartesian_acquisition_frames
        cartesian_volumes = self.cartesian_acquisition_volumes
        spiral = self.spiral_acquisition
        if cartesian is not None:
            kspace = self.to_cartesian_kspace(cartesian)
            image = self.reconstruct_cartesian(cartesian)
            cartesian_dims = ["phase_y", "read_x"]
            if self.signal.ndim == 2:
                cartesian_dims.insert(0, "coil")
            coords.update(
                {
                    "phase_y": np.arange(cartesian.phase_matrix),
                    "read_x": np.arange(cartesian.read_matrix),
                    "cartesian_kx_cyc_per_m": (
                        "read_x",
                        cartesian.kx_cyc_per_m,
                    ),
                    "cartesian_ky_cyc_per_m": (
                        "phase_y",
                        cartesian.ky_cyc_per_m,
                    ),
                }
            )
            data_vars["cartesian_kspace"] = (cartesian_dims, kspace)
            data_vars["cartesian_image"] = (cartesian_dims, image)
            data_vars["cartesian_image_magnitude"] = (
                cartesian_dims,
                np.abs(image),
            )
        elif cartesian_frames is not None:
            first = cartesian_frames.acquisitions[0]
            same_grid = all(
                acquisition.read_matrix == first.read_matrix
                and acquisition.phase_matrix == first.phase_matrix
                and np.allclose(acquisition.kx_cyc_per_m, first.kx_cyc_per_m)
                and np.allclose(acquisition.ky_cyc_per_m, first.ky_cyc_per_m)
                for acquisition in cartesian_frames.acquisitions
            )
            if same_grid:
                kspace = np.stack(
                    [
                        cartesian_frames.to_cartesian_kspace(self, frame)
                        for frame in range(cartesian_frames.num_frames)
                    ],
                    axis=0,
                )
                image = np.stack(
                    [
                        cartesian_frames.reconstruct(self, frame)
                        for frame in range(cartesian_frames.num_frames)
                    ],
                    axis=0,
                )
                cartesian_dims = ["cartesian_frame", "phase_y", "read_x"]
                if self.signal.ndim == 2:
                    cartesian_dims.insert(1, "coil")
                coords.update(
                    {
                        "cartesian_frame": np.arange(cartesian_frames.num_frames),
                        "phase_y": np.arange(first.phase_matrix),
                        "read_x": np.arange(first.read_matrix),
                        "cartesian_kx_cyc_per_m": (
                            "read_x",
                            first.kx_cyc_per_m,
                        ),
                        "cartesian_ky_cyc_per_m": (
                            "phase_y",
                            first.ky_cyc_per_m,
                        ),
                    }
                )
                for axis_index, axis in enumerate(
                    cartesian_frames.dimensions.AXIS_NAMES
                ):
                    coords[f"cartesian_frame_{axis}_index"] = (
                        "cartesian_frame",
                        [frame[axis_index] for frame in cartesian_frames.frame_indices],
                    )
                data_vars["cartesian_kspace"] = (cartesian_dims, kspace)
                data_vars["cartesian_image"] = (cartesian_dims, image)
                data_vars["cartesian_image_magnitude"] = (
                    cartesian_dims,
                    np.abs(image),
                )
        if cartesian_volumes is not None:
            kspace_3d = cartesian_volumes.dimensioned_kspace(self)
            image_3d = cartesian_volumes.dimensioned_reconstruction(self)
            volume_dims = list(cartesian_volumes.varying_axes)
            if self.signal.ndim == 2:
                volume_dims.append("coil")
            volume_dims.extend(("partition_z", "phase_y", "read_x"))
            coords.update(
                {
                    axis: np.asarray(
                        cartesian_volumes.axis_values(axis), dtype=np.int64
                    )
                    for axis in cartesian_volumes.varying_axes
                }
            )
            coords.update(
                {
                    "partition_z": np.arange(cartesian_volumes.partition_matrix),
                    "phase_y": np.arange(cartesian_volumes.phase_matrix),
                    "read_x": np.arange(cartesian_volumes.read_matrix),
                    "cartesian_kx_cyc_per_m": (
                        "read_x",
                        cartesian_volumes.kx_cyc_per_m,
                    ),
                    "cartesian_ky_cyc_per_m": (
                        "phase_y",
                        cartesian_volumes.ky_cyc_per_m,
                    ),
                    "cartesian_kz_cyc_per_m": (
                        "partition_z",
                        cartesian_volumes.kz_cyc_per_m,
                    ),
                }
            )
            data_vars["cartesian_3d_kspace"] = (volume_dims, kspace_3d)
            data_vars["cartesian_3d_image"] = (volume_dims, image_3d)
            data_vars["cartesian_3d_image_magnitude"] = (
                volume_dims,
                np.abs(image_3d),
            )
        if spiral is not None:
            spiral_kspace = np.stack(
                [spiral.grid_kspace(self, frame) for frame in range(spiral.num_frames)],
                axis=0,
            )
            spiral_image = np.stack(
                [spiral.reconstruct(self, frame) for frame in range(spiral.num_frames)],
                axis=0,
            )
            spiral_dims = ["spiral_frame", "phase_y", "read_x"]
            if self.signal.ndim == 2:
                spiral_dims.insert(1, "coil")
            coords.update(
                {
                    "spiral_frame": np.arange(spiral.num_frames),
                    "phase_y": np.arange(spiral.matrix[1]),
                    "read_x": np.arange(spiral.matrix[0]),
                    "spiral_grid_kx_cyc_per_m": (
                        "read_x",
                        spiral.kx_grid_cyc_per_m,
                    ),
                    "spiral_grid_ky_cyc_per_m": (
                        "phase_y",
                        spiral.ky_grid_cyc_per_m,
                    ),
                }
            )
            for axis_index, axis in enumerate(spiral.dimensions.AXIS_NAMES):
                coords[f"spiral_frame_{axis}_index"] = (
                    "spiral_frame",
                    [frame[axis_index] for frame in spiral.frame_indices],
                )
            data_vars["spiral_gridded_kspace"] = (spiral_dims, spiral_kspace)
            data_vars["spiral_image"] = (spiral_dims, spiral_image)
            data_vars["spiral_image_magnitude"] = (
                spiral_dims,
                np.abs(spiral_image),
            )
        if self.checkpoint_magnetization is not None:
            coords["checkpoint"] = self.checkpoint_times_s
            data_vars["checkpoint_magnetization"] = (
                ["checkpoint"] + spatial_dims + ["component"],
                self.checkpoint_magnetization,
            )
        if self.species_signal is not None:
            coords["pool"] = list(self.pool_names)
            pool_signal_dims = ("pool", "adc")
            if self.signal.ndim == 2:
                pool_signal_dims = ("pool", "coil", "adc")
            data_vars["species_signal"] = (pool_signal_dims, self.species_signal)
            data_vars["final_pool_magnetization"] = (
                ["pool"] + spatial_dims + ["component"],
                self.final_pool_magnetization,
            )
            if self.checkpoint_pool_magnetization is not None:
                data_vars["checkpoint_pool_magnetization"] = (
                    ["checkpoint", "pool"] + spatial_dims + ["component"],
                    self.checkpoint_pool_magnetization,
                )
        spectroscopy = self.spectroscopic_acquisition
        if spectroscopy is not None:
            csi_dims = ["phase_y", "phase_x", "spectral_point"]
            if spectroscopy.num_repetitions > 1:
                csi_dims.insert(0, "repetition")
            if self.signal.ndim == 2:
                csi_dims.insert(0, "coil")
            coords.update(
                {
                    "phase_x": np.arange(spectroscopy.matrix[0]),
                    "phase_y": np.arange(spectroscopy.matrix[1]),
                    "spectral_point": np.arange(spectroscopy.spectral_points),
                    "spatial_kx_cyc_per_m": (
                        "phase_x",
                        spectroscopy.kx_cyc_per_m,
                    ),
                    "spatial_ky_cyc_per_m": (
                        "phase_y",
                        spectroscopy.ky_cyc_per_m,
                    ),
                    "spectral_time_s": (
                        "spectral_point",
                        spectroscopy.spectral_time_s,
                    ),
                    "spectral_frequency_hz": (
                        "spectral_point",
                        spectroscopy.frequency_hz,
                    ),
                }
            )
            if spectroscopy.num_repetitions > 1:
                coords["repetition"] = np.arange(spectroscopy.num_repetitions)
            data_vars["csi_kspace"] = (
                csi_dims,
                spectroscopy.reshape_signal(self.signal),
            )
            data_vars["csi_spatial_fid"] = (
                csi_dims,
                spectroscopy.reconstruct_spatial(self.signal),
            )
            data_vars["csi_spectrum"] = (
                csi_dims,
                spectroscopy.reconstruct_spectra(self.signal),
            )
            if self.species_signal is not None:
                pool_csi_dims = ["pool", "phase_y", "phase_x", "spectral_point"]
                if spectroscopy.num_repetitions > 1:
                    pool_csi_dims.insert(1, "repetition")
                if self.species_signal.ndim == 3:
                    pool_csi_dims.insert(1, "coil")
                data_vars["species_csi_kspace"] = (
                    pool_csi_dims,
                    spectroscopy.reshape_signal(self.species_signal),
                )
                data_vars["species_csi_spatial_fid"] = (
                    pool_csi_dims,
                    spectroscopy.reconstruct_spatial(self.species_signal),
                )
                data_vars["species_csi_spectrum"] = (
                    pool_csi_dims,
                    spectroscopy.reconstruct_spectra(self.species_signal),
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
        for axis in ("spatial_kx_cyc_per_m", "spatial_ky_cyc_per_m"):
            if axis in dataset.coords:
                dataset[axis].attrs.update(units="cycles/m")
        for axis in (
            "cartesian_kx_cyc_per_m",
            "cartesian_ky_cyc_per_m",
            "cartesian_kz_cyc_per_m",
            "spiral_grid_kx_cyc_per_m",
            "spiral_grid_ky_cyc_per_m",
        ):
            if axis in dataset.coords:
                dataset[axis].attrs.update(units="cycles/m")
        if "spectral_time_s" in dataset.coords:
            dataset["spectral_time_s"].attrs.update(units="s")
            dataset["spectral_frequency_hz"].attrs.update(units="Hz")
        return dataset

    def save(self, filename) -> Path:
        """Save sparse sequence output as NPZ, HDF5, or xarray NetCDF."""
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix == ".nc":
            dataset = self.to_xarray()
            complex_names = [
                name
                for name, values in dataset.data_vars.items()
                if np.iscomplexobj(values.data)
            ]
            for name in complex_names:
                values = dataset[name]
                dataset[f"{name}_real"] = values.real
                dataset[f"{name}_imag"] = values.imag
                dataset = dataset.drop_vars(name)
            if complex_names:
                dataset.attrs["complex_variables"] = ", ".join(
                    f"{name}={name}_real+1j*{name}_imag" for name in complex_names
                )
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
            moments = np.asarray(self.adc_gradient_moment_cyc_per_m)
            arrays["kx_cyc_per_m"] = moments[:, 0]
            arrays["ky_cyc_per_m"] = moments[:, 1]
            arrays["kz_cyc_per_m"] = moments[:, 2]
        dimensions = self.acquisition_dimensions
        if dimensions is not None:
            event_counts = np.asarray(
                dimensions.adc_event_sample_counts, dtype=np.int64
            )
            arrays["adc_event_index"] = np.repeat(
                np.arange(event_counts.size, dtype=np.int64), event_counts
            )
            arrays["readout_sample_index"] = np.concatenate(
                [np.arange(count, dtype=np.int64) for count in event_counts]
            )
            for axis in dimensions.AXIS_NAMES:
                arrays[f"{axis}_index"] = dimensions.sample_indices(axis)
        if self.species_signal is not None:
            arrays["species_signal"] = self.species_signal
            arrays["final_pool_magnetization"] = self.final_pool_magnetization
            arrays["pool_names"] = np.asarray(self.pool_names, dtype="S")
            if self.checkpoint_pool_magnetization is not None:
                arrays["checkpoint_pool_magnetization"] = (
                    self.checkpoint_pool_magnetization
                )
        cartesian = self.cartesian_acquisition
        cartesian_frames = self.cartesian_acquisition_frames
        cartesian_volumes = self.cartesian_acquisition_volumes
        spiral = self.spiral_acquisition
        if cartesian is not None:
            image = self.reconstruct_cartesian(cartesian)
            arrays.update(
                {
                    "cartesian_kspace": self.to_cartesian_kspace(cartesian),
                    "cartesian_image": image,
                    "cartesian_image_magnitude": np.abs(image),
                    "cartesian_kx_cyc_per_m": cartesian.kx_cyc_per_m,
                    "cartesian_ky_cyc_per_m": cartesian.ky_cyc_per_m,
                    "cartesian_phase_indices": np.asarray(
                        cartesian.phase_indices, dtype=np.int64
                    ),
                    "cartesian_readout_directions": np.asarray(
                        cartesian.readout_directions, dtype=np.int64
                    ),
                }
            )
        elif cartesian_frames is not None:
            first = cartesian_frames.acquisitions[0]
            same_grid = all(
                acquisition.read_matrix == first.read_matrix
                and acquisition.phase_matrix == first.phase_matrix
                and np.allclose(acquisition.kx_cyc_per_m, first.kx_cyc_per_m)
                and np.allclose(acquisition.ky_cyc_per_m, first.ky_cyc_per_m)
                for acquisition in cartesian_frames.acquisitions
            )
            if same_grid:
                image = np.stack(
                    [
                        cartesian_frames.reconstruct(self, frame)
                        for frame in range(cartesian_frames.num_frames)
                    ],
                    axis=0,
                )
                arrays.update(
                    {
                        "cartesian_kspace": np.stack(
                            [
                                cartesian_frames.to_cartesian_kspace(self, frame)
                                for frame in range(cartesian_frames.num_frames)
                            ],
                            axis=0,
                        ),
                        "cartesian_image": image,
                        "cartesian_image_magnitude": np.abs(image),
                        "cartesian_kx_cyc_per_m": first.kx_cyc_per_m,
                        "cartesian_ky_cyc_per_m": first.ky_cyc_per_m,
                        "cartesian_frame_indices": np.asarray(
                            cartesian_frames.frame_indices, dtype=np.int64
                        ),
                    }
                )
        if cartesian_volumes is not None:
            image_3d = cartesian_volumes.dimensioned_reconstruction(self)
            arrays.update(
                {
                    "cartesian_3d_kspace": cartesian_volumes.dimensioned_kspace(self),
                    "cartesian_3d_image": image_3d,
                    "cartesian_3d_image_magnitude": np.abs(image_3d),
                    "cartesian_kz_cyc_per_m": cartesian_volumes.kz_cyc_per_m,
                    "cartesian_3d_volume_indices": np.asarray(
                        cartesian_volumes.volume_indices, dtype=np.int64
                    ),
                    "cartesian_3d_outer_axis_names": np.asarray(
                        cartesian_volumes.OUTER_AXIS_NAMES, dtype="S"
                    ),
                }
            )
            for axis in cartesian_volumes.varying_axes:
                arrays[f"cartesian_3d_{axis}_index"] = np.asarray(
                    cartesian_volumes.axis_values(axis), dtype=np.int64
                )
        if spiral is not None:
            spiral_image = np.stack(
                [spiral.reconstruct(self, frame) for frame in range(spiral.num_frames)],
                axis=0,
            )
            arrays.update(
                {
                    "spiral_gridded_kspace": np.stack(
                        [
                            spiral.grid_kspace(self, frame)
                            for frame in range(spiral.num_frames)
                        ],
                        axis=0,
                    ),
                    "spiral_image": spiral_image,
                    "spiral_image_magnitude": np.abs(spiral_image),
                    "spiral_grid_kx_cyc_per_m": spiral.kx_grid_cyc_per_m,
                    "spiral_grid_ky_cyc_per_m": spiral.ky_grid_cyc_per_m,
                    "spiral_frame_indices": np.asarray(
                        spiral.frame_indices, dtype=np.int64
                    ),
                }
            )
        spectroscopy = self.spectroscopic_acquisition
        if spectroscopy is not None:
            arrays.update(
                {
                    "csi_kspace": spectroscopy.reshape_signal(self.signal),
                    "csi_spatial_fid": spectroscopy.reconstruct_spatial(self.signal),
                    "csi_spectrum": spectroscopy.reconstruct_spectra(self.signal),
                    "csi_kx_cyc_per_m": spectroscopy.kx_cyc_per_m,
                    "csi_ky_cyc_per_m": spectroscopy.ky_cyc_per_m,
                    "csi_spectral_time_s": spectroscopy.spectral_time_s,
                    "csi_frequency_hz": spectroscopy.frequency_hz,
                }
            )
            if self.species_signal is not None:
                arrays.update(
                    {
                        "species_csi_kspace": spectroscopy.reshape_signal(
                            self.species_signal
                        ),
                        "species_csi_spatial_fid": spectroscopy.reconstruct_spatial(
                            self.species_signal
                        ),
                        "species_csi_spectrum": spectroscopy.reconstruct_spectra(
                            self.species_signal
                        ),
                    }
                )
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

    def save_bruker(self, directory, **kwargs) -> Path:
        """Save the ADC stream as a Bruker-style raw-data directory."""
        from .bruker_export import export_bruker_raw

        return export_bruker_raw(self, directory, **kwargs)

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

    def to_cartesian_3d_kspace(self, acquisition=None) -> np.ndarray:
        """Return automatically dimensioned Cartesian 3D k-space."""
        acquisition = (
            self.cartesian_acquisition_volumes if acquisition is None else acquisition
        )
        if acquisition is None:
            raise ValueError("result has no inferred Cartesian 3D acquisition")
        return acquisition.dimensioned_kspace(self)

    def reconstruct_cartesian_3d(
        self,
        acquisition=None,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Return automatically dimensioned Cartesian 3D reconstructions."""
        acquisition = (
            self.cartesian_acquisition_volumes if acquisition is None else acquisition
        )
        if acquisition is None:
            raise ValueError("result has no inferred Cartesian 3D acquisition")
        return acquisition.dimensioned_reconstruction(
            self,
            norm=norm,
            coil_combine=coil_combine,
            voxel_centered=voxel_centered,
        )
