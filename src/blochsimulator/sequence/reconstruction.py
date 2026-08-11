"""Shared reconstruction preparation and multidimensional result selection.

The desktop explorer and generated result notebooks deliberately operate on
the same xarray schema.  This module keeps GUI code independent of acquisition
ordering details and provides a single place for selecting outer acquisition
dimensions, receive channels, and pool-resolved data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


def _definition_mapping(result) -> dict:
    return {
        str(key).lower(): value
        for key, value in dict(result.metadata.get("sequence_definitions", {})).items()
    }


def is_radial_3d_result(result) -> bool:
    """Return whether a result declares the supported 3D radial trajectory."""
    return str(_definition_mapping(result).get("trajectorytype", "")).lower() == (
        "radial_3d_spiral_phyllotaxis"
    )


def _numeric_definition(value) -> np.ndarray:
    if isinstance(value, str):
        cleaned = value.strip().strip("[](){}").replace(",", " ")
        parsed = np.fromstring(cleaned, sep=" ", dtype=float)
        if parsed.size:
            return parsed
    return np.asarray(value, dtype=float).reshape(-1)


def _trilinear_radial_grid(samples, trajectory, matrix: int, fov_m: float):
    """Density-compensated trilinear gridding onto an isotropic 3D matrix."""
    values = np.asarray(samples)
    coordinates = np.asarray(trajectory, dtype=float)
    if values.shape[-1] != coordinates.shape[0] or coordinates.shape[1:] != (3,):
        raise ValueError("radial samples and trajectory do not match")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("radial trajectory contains non-finite values")

    matrix = int(matrix)
    positions = coordinates * float(fov_m) + matrix / 2.0
    lower = np.floor(positions).astype(np.int64)
    fractional = positions - lower
    radius_squared = np.sum(coordinates**2, axis=1)
    positive = radius_squared[radius_squared > 0]
    floor_weight = float(np.min(positive) * 0.25) if positive.size else 1.0
    density = np.maximum(radius_squared, floor_weight)
    density /= float(np.mean(density))

    flattened = values.reshape((-1, values.shape[-1]))
    output = np.zeros((flattened.shape[0], matrix**3), dtype=values.dtype)
    for dz in (0, 1):
        wz = fractional[:, 2] if dz else 1.0 - fractional[:, 2]
        iz = lower[:, 2] + dz
        for dy in (0, 1):
            wy = fractional[:, 1] if dy else 1.0 - fractional[:, 1]
            iy = lower[:, 1] + dy
            for dx in (0, 1):
                wx = fractional[:, 0] if dx else 1.0 - fractional[:, 0]
                ix = lower[:, 0] + dx
                valid = (
                    (ix >= 0)
                    & (ix < matrix)
                    & (iy >= 0)
                    & (iy < matrix)
                    & (iz >= 0)
                    & (iz < matrix)
                )
                if not np.any(valid):
                    continue
                weight = wx[valid] * wy[valid] * wz[valid] * density[valid]
                flat_index = (iz[valid] * matrix + iy[valid]) * matrix + ix[valid]
                for channel, channel_values in enumerate(flattened):
                    np.add.at(
                        output[channel],
                        flat_index,
                        channel_values[valid] * weight,
                    )
    return output.reshape(values.shape[:-1] + (matrix, matrix, matrix))


def radial_3d_reconstruction_arrays(result, signal=None):
    """Grid supported radial multi-echo results and return named outer axes.

    The implementation is an explicit density-compensated trilinear gridding
    reference.  It is deterministic and dependency-free, while leaving room
    for a later NUFFT backend without changing the exported array schema.
    """
    if not is_radial_3d_result(result):
        raise ValueError("result is not a supported radial 3D acquisition")
    definitions = _definition_mapping(result)
    matrix_values = _numeric_definition(definitions.get("matrixsize", ()))
    fov_values = _numeric_definition(definitions.get("fov", ()))
    if matrix_values.size < 3 or fov_values.size < 3:
        raise ValueError("radial reconstruction requires MatrixSize and FOV")
    if not np.allclose(matrix_values[:3], matrix_values[0]):
        raise ValueError(
            "radial reference gridding currently requires an isotropic matrix"
        )
    if not np.allclose(fov_values[:3], fov_values[0]):
        raise ValueError(
            "radial reference gridding currently requires an isotropic FOV"
        )
    matrix = int(round(float(matrix_values[0])))
    fov_m = float(fov_values[0])

    dimensions = result.acquisition_dimensions
    if dimensions is None:
        raise ValueError("radial reconstruction requires acquisition dimensions")
    moments = np.asarray(result.adc_gradient_moment_cyc_per_m, dtype=float)
    if moments.shape != (dimensions.num_samples, 3):
        raise ValueError("radial reconstruction requires one 3D moment per ADC sample")
    values = np.asarray(result.signal if signal is None else signal)
    if values.shape[-1] != dimensions.num_samples:
        raise ValueError("radial signal does not match acquisition dimensions")

    event_offsets = np.concatenate(
        ([0], np.cumsum(dimensions.adc_event_sample_counts, dtype=np.int64))
    )
    event_labels = {
        axis: np.asarray(dimensions.event_indices(axis), dtype=np.int64)
        for axis in dimensions.AXIS_NAMES
    }
    outer_axes = tuple(
        axis
        for axis in ("slice", "echo", "repetition", "segment")
        if np.unique(event_labels[axis]).size > 1
    )
    outer_values = {
        axis: tuple(int(value) for value in np.sort(np.unique(event_labels[axis])))
        for axis in outer_axes
    }
    outer_shape = tuple(len(outer_values[axis]) for axis in outer_axes)
    leading_shape = values.shape[:-1]
    output_shape = outer_shape + leading_shape + (matrix, matrix, matrix)
    from ..memory import enforce_memory_budget, resolve_memory_budget

    output_elements = int(np.prod(output_shape, dtype=np.int64))
    enforce_memory_budget(
        output_elements * np.dtype(values.dtype).itemsize * 4,
        resolve_memory_budget(),
        description=(f"The radial reconstruction grid would have shape {output_shape}"),
        suggestions=(
            "reduce the radial base resolution, echo/measurement count, pool count, "
            "or receive-coil count"
        ),
    )
    kspace = np.zeros(
        output_shape,
        dtype=values.dtype,
    )

    keys = [()]
    if outer_axes:
        from itertools import product

        keys = list(product(*(outer_values[axis] for axis in outer_axes)))
    for key in keys:
        event_mask = np.ones(dimensions.num_adc_events, dtype=bool)
        for axis, target in zip(outer_axes, key):
            event_mask &= event_labels[axis] == target
        event_indices = np.flatnonzero(event_mask)
        sample_groups = [
            np.arange(event_offsets[event], event_offsets[event + 1], dtype=np.int64)
            for event in event_indices
        ]
        if not sample_groups:
            raise ValueError("radial outer acquisition grid is incomplete")
        selected_samples = np.concatenate(sample_groups)
        relative_trajectory = []
        for indices in sample_groups:
            line = moments[indices]
            # Remove small accumulated balanced-gradient residuals without
            # changing the direction or radial extent of the center-through line.
            origin = 0.5 * (line[0] + line[-1])
            relative_trajectory.append(line - origin)
        trajectory = np.concatenate(relative_trajectory, axis=0)
        selected_values = np.take(values, selected_samples, axis=-1)
        grid = _trilinear_radial_grid(selected_values, trajectory, matrix, fov_m)
        position = tuple(
            outer_values[axis].index(value) for axis, value in zip(outer_axes, key)
        )
        kspace[position] = grid

    spatial_axes = (-3, -2, -1)
    image = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=spatial_axes), axes=spatial_axes),
        axes=spatial_axes,
    )
    return outer_axes, outer_values, kspace, image, matrix, fov_m


def restore_complex_variables(dataset):
    """Restore ``*_real``/``*_imag`` NetCDF pairs as complex variables."""
    ds = dataset.copy()
    for name in list(ds.data_vars):
        if not name.endswith("_real"):
            continue
        base = name[:-5]
        imag = f"{base}_imag"
        if imag in ds:
            ds[base] = ds[name] + 1j * ds[imag]
    return ds


def prepare_reconstruction_dataset(dataset):
    """Return a dataset with all reconstructable Cartesian arrays attached.

    Sequence result exports already contain validated gridded arrays whenever
    acquisition metadata was available during simulation.  For older or
    third-party exports, use the same conservative ADC-coordinate fallback that
    is embedded in generated analysis notebooks.
    """
    import xarray as xr

    ds = restore_complex_variables(dataset)
    reconstructed_variables = {
        "cartesian_image",
        "cartesian_image_magnitude",
        "cartesian_3d_image",
        "cartesian_3d_image_magnitude",
        "spiral_image",
        "spiral_image_magnitude",
        "radial_3d_image",
        "radial_3d_image_magnitude",
        "csi_spatial_fid",
        "notebook_cartesian_image",
        "notebook_cartesian_3d_image",
    }
    if reconstructed_variables.intersection(ds.data_vars):
        return ds
    from ..notebook_exporter import _sequence_result_reconstruction_code

    namespace = {"ds": ds, "np": np, "xr": xr}
    exec(_sequence_result_reconstruction_code(), namespace)
    return namespace["ds"]


def load_reconstruction_dataset(filename):
    """Load a sequence-result NetCDF file for interactive exploration."""
    import xarray as xr

    path = Path(filename)
    if path.suffix.lower() != ".nc":
        raise ValueError("interactive result loading currently requires a .nc file")
    with xr.open_dataset(path) as stored:
        dataset = stored.load()
    return prepare_reconstruction_dataset(dataset)


def ideal_separate(
    echo_images,
    echo_times_s,
    frequency_offsets_hz,
    *,
    echo_axis: int = 0,
    rcond: Optional[float] = None,
):
    """Separate known chemical species by complex multi-echo least squares.

    This is the linear, known-frequency IDEAL signal model without B0 fitting.
    It is useful as a deterministic first reconstruction and intentionally does
    not claim to replace iterative field-map-aware IDEAL implementations.
    """
    values = np.asarray(echo_images)
    times = np.asarray(echo_times_s, dtype=float).reshape(-1)
    frequencies = np.asarray(frequency_offsets_hz, dtype=float).reshape(-1)
    if values.shape[int(echo_axis)] != times.size:
        raise ValueError("echo image axis does not match echo_times_s")
    if times.size < frequencies.size:
        raise ValueError("IDEAL requires at least as many echoes as species")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("IDEAL echo times and frequency offsets must be finite")
    encoding = np.exp(2j * np.pi * times[:, None] * frequencies[None, :])
    moved = np.moveaxis(values, int(echo_axis), 0)
    flattened = moved.reshape(times.size, -1)
    separated = np.linalg.lstsq(encoding, flattened, rcond=rcond)[0]
    return separated.reshape((frequencies.size,) + moved.shape[1:])


@dataclass(frozen=True)
class OuterDimension:
    """One user-selectable non-spatial result dimension."""

    name: str
    values: tuple
    virtual_frame_dimension: Optional[str] = None
    coordinate_name: Optional[str] = None


class SequenceReconstructionModel:
    """Dimension-aware facade over a prepared sequence-result dataset."""

    _SPATIAL_ROLES = ("partition_", "phase_", "read_")

    def __init__(self, dataset):
        self.dataset = prepare_reconstruction_dataset(dataset)
        self.kind = self._detect_kind()
        self.spatial_dims = self._detect_spatial_dims()
        self.outer_dimensions = self._detect_outer_dimensions()

    @classmethod
    def from_result(cls, result):
        return cls(result.to_xarray())

    @classmethod
    def from_file(cls, filename):
        return cls(load_reconstruction_dataset(filename))

    def _detect_kind(self) -> str:
        ds = self.dataset
        if "radial_3d_image" in ds or "radial_3d_image_magnitude" in ds:
            return "radial_3d"
        if (
            "notebook_cartesian_3d_image" in ds
            or "cartesian_3d_image" in ds
            or "cartesian_3d_image_magnitude" in ds
        ):
            return "cartesian_3d"
        if "csi_kspace" in ds:
            return "csi"
        if "spiral_image" in ds or "spiral_image_magnitude" in ds:
            return "spiral_2d"
        if (
            "notebook_cartesian_image" in ds
            or "cartesian_image" in ds
            or "cartesian_image_magnitude" in ds
        ):
            return "cartesian_2d"
        return "raw_signal"

    @staticmethod
    def _role_dimension(data, role: str) -> Optional[str]:
        return next(
            (dimension for dimension in data.dims if dimension.startswith(f"{role}_")),
            None,
        )

    def _detect_spatial_dims(self) -> tuple[str, ...]:
        if self.kind == "cartesian_3d":
            data = self.dataset[self.image_name()]
            return tuple(
                dimension
                for role in ("partition", "phase", "read")
                if (dimension := self._role_dimension(data, role)) is not None
            )
        if self.kind == "radial_3d":
            return ("radial_z", "radial_y", "radial_x")
        if self.kind == "csi":
            return ("phase_y", "phase_x", "spectral_point")
        if self.kind == "spiral_2d":
            return ("phase_y", "read_x")
        if self.kind == "cartesian_2d":
            data = self.dataset[self.image_name()]
            return tuple(
                dimension
                for role in ("phase", "read")
                if (dimension := self._role_dimension(data, role)) is not None
            )
        return ()

    def _frame_outer_dimensions(self, frame_dimension: str, prefix: str):
        dimensions = []
        for axis in ("slice", "echo", "repetition", "segment", "partition"):
            coordinate = f"{prefix}_{axis}_index"
            if coordinate not in self.dataset.coords:
                continue
            values = tuple(
                value.item() if hasattr(value, "item") else value
                for value in np.unique(self.dataset.coords[coordinate].values)
            )
            if len(values) > 1:
                dimensions.append(
                    OuterDimension(axis, values, frame_dimension, coordinate)
                )
        if not dimensions and self.dataset.sizes.get(frame_dimension, 1) > 1:
            dimensions.append(
                OuterDimension(
                    "frame",
                    tuple(range(self.dataset.sizes[frame_dimension])),
                    frame_dimension,
                    None,
                )
            )
        return dimensions

    def _detect_outer_dimensions(self) -> tuple[OuterDimension, ...]:
        ds = self.dataset
        if self.kind == "cartesian_2d" and "cartesian_frame" in ds.dims:
            return tuple(
                self._frame_outer_dimensions("cartesian_frame", "cartesian_frame")
            )
        if self.kind == "spiral_2d" and "spiral_frame" in ds.dims:
            return tuple(self._frame_outer_dimensions("spiral_frame", "spiral_frame"))
        excluded = set(self.spatial_dims) | {"coil", "pool"}
        source = self.dataset[self.image_name()]
        return tuple(
            OuterDimension(
                dimension,
                tuple(
                    value.item() if hasattr(value, "item") else value
                    for value in (
                        ds.coords[dimension].values
                        if dimension in ds.coords
                        else np.arange(ds.sizes[dimension])
                    )
                ),
            )
            for dimension in source.dims
            if dimension not in excluded and ds.sizes[dimension] > 1
        )

    @property
    def pool_names(self) -> tuple[str, ...]:
        if "pool" not in self.dataset.coords:
            return ()
        return tuple(str(value) for value in self.dataset.coords["pool"].values)

    @property
    def coil_count(self) -> int:
        return int(self.dataset.sizes.get("coil", 1))

    def image_name(self, *, pool: bool = False) -> str:
        prefix = "species_" if pool else ""
        candidates = {
            "cartesian_3d": (
                f"{prefix}cartesian_3d_image",
                "notebook_cartesian_3d_image" if not pool else "",
                f"{prefix}cartesian_3d_image_magnitude",
            ),
            "radial_3d": (
                f"{prefix}radial_3d_image",
                f"{prefix}radial_3d_image_magnitude",
            ),
            "cartesian_2d": (
                f"{prefix}cartesian_image",
                "notebook_cartesian_image" if not pool else "",
                f"{prefix}cartesian_image_magnitude",
            ),
            "spiral_2d": (
                f"{prefix}spiral_image",
                f"{prefix}spiral_image_magnitude",
            ),
            "csi": (f"{prefix}csi_spatial_fid",),
        }.get(self.kind, ())
        return next((name for name in candidates if name and name in self.dataset), "")

    def kspace_name(self, *, pool: bool = False) -> str:
        prefix = "species_" if pool else ""
        candidates = {
            "cartesian_3d": (f"{prefix}cartesian_3d_kspace",),
            "radial_3d": (f"{prefix}radial_3d_gridded_kspace",),
            "cartesian_2d": (f"{prefix}cartesian_kspace",),
            "spiral_2d": (f"{prefix}spiral_gridded_kspace",),
            "csi": (f"{prefix}csi_kspace",),
        }.get(self.kind, ())
        return next((name for name in candidates if name in self.dataset), "")

    def spectrum_name(self, *, pool: bool = False) -> str:
        name = f"{'species_' if pool else ''}csi_spectrum"
        return name if name in self.dataset else ""

    def has_pool_data(self) -> bool:
        return bool(self.pool_names and self.image_name(pool=True))

    @staticmethod
    def _numeric_attr(values) -> tuple[float, ...]:
        if values is None:
            return ()
        if isinstance(values, str):
            values = values.split(",")
        try:
            return tuple(float(value) for value in np.asarray(values).reshape(-1))
        except (TypeError, ValueError):
            return ()

    @property
    def ideal_configuration(self):
        """Return ``(echo_dimension, times, offsets)`` when linear IDEAL is valid."""
        echo_dimension = next(
            (item for item in self.outer_dimensions if item.name == "echo"), None
        )
        if echo_dimension is None or not self.image_name():
            return None
        echo_times = self._numeric_attr(self.dataset.attrs.get("echo_times_s"))
        offsets = self._numeric_attr(
            self.dataset.attrs.get("pool_frequency_offsets_hz")
        )
        if (
            len(echo_times) != len(echo_dimension.values)
            or not offsets
            or len(echo_times) < len(offsets)
        ):
            return None
        return echo_dimension, echo_times, offsets

    @property
    def ideal_species_names(self) -> tuple[str, ...]:
        configuration = self.ideal_configuration
        if configuration is None:
            return ()
        count = len(configuration[2])
        if len(self.pool_names) == count:
            return self.pool_names
        return tuple(f"Species {index + 1}" for index in range(count))

    def ideal_images(
        self,
        selections: Optional[Mapping[str, object]] = None,
        *,
        coil_mode: str = "sum",
    ) -> np.ndarray:
        """Return linearly separated complex images for the selected outer state."""
        configuration = self.ideal_configuration
        if configuration is None:
            raise ValueError(
                "IDEAL metadata or a complete echo dimension is unavailable"
            )
        if coil_mode == "rss":
            raise ValueError("IDEAL requires coherent or individual coil data")
        echo_dimension, echo_times, offsets = configuration
        selections = {} if selections is None else dict(selections)
        echoes = []
        for value in echo_dimension.values:
            echo_selection = dict(selections)
            echo_selection[echo_dimension.name] = value
            echoes.append(
                np.asarray(
                    self.select(
                        self.image_name(),
                        echo_selection,
                        coil_mode=coil_mode,
                    )
                )
            )
        return ideal_separate(np.stack(echoes), echo_times, offsets)

    def _frame_index(
        self, selections: Mapping[str, object]
    ) -> Optional[tuple[str, int]]:
        frame_dimensions = {
            item.virtual_frame_dimension
            for item in self.outer_dimensions
            if item.virtual_frame_dimension is not None
        }
        if not frame_dimensions:
            return None
        frame_dimension = next(iter(frame_dimensions))
        candidates = np.arange(self.dataset.sizes[frame_dimension])
        for item in self.outer_dimensions:
            if item.virtual_frame_dimension != frame_dimension:
                continue
            value = selections.get(item.name, item.values[0])
            if item.coordinate_name is None:
                candidates = candidates[candidates == int(value)]
            else:
                coordinate = np.asarray(
                    self.dataset.coords[item.coordinate_name].values
                )
                candidates = candidates[coordinate[candidates] == value]
        if candidates.size != 1:
            raise ValueError("outer dimension selection does not identify one frame")
        return frame_dimension, int(candidates[0])

    def select(
        self,
        variable_name: str,
        selections: Optional[Mapping[str, object]] = None,
        *,
        pool_index: Optional[int] = None,
        coil_mode: str = "rss",
    ):
        """Select outer dimensions and combine receive channels."""
        if not variable_name or variable_name not in self.dataset:
            raise ValueError(f"result variable {variable_name!r} is unavailable")
        selections = {} if selections is None else dict(selections)
        data = self.dataset[variable_name]
        frame = self._frame_index(selections)
        selectors = {}
        if frame is not None and frame[0] in data.dims:
            selectors[frame[0]] = frame[1]
        for item in self.outer_dimensions:
            if item.virtual_frame_dimension is not None:
                continue
            if item.name in data.dims:
                value = selections.get(item.name, item.values[0])
                if item.name in data.coords:
                    matches = np.flatnonzero(
                        np.asarray(data[item.name].values) == value
                    )
                    selectors[item.name] = int(matches[0]) if matches.size else 0
                else:
                    selectors[item.name] = int(value)
        if selectors:
            data = data.isel(selectors)
        if "pool" in data.dims:
            data = data.isel(pool=0 if pool_index is None else int(pool_index))
        if "coil" in data.dims:
            if coil_mode == "rss":
                data = np.sqrt((np.abs(data) ** 2).sum("coil"))
            elif coil_mode == "sum":
                data = data.sum("coil")
            elif coil_mode.startswith("coil:"):
                data = data.isel(coil=int(coil_mode.split(":", 1)[1]))
            else:
                raise ValueError("unknown receive-channel mode")
        return data

    @staticmethod
    def display_values(data, component: str = "magnitude") -> np.ndarray:
        values = np.asarray(data)
        mode = str(component).lower()
        if mode == "magnitude":
            return np.abs(values)
        if mode == "phase":
            return np.angle(values)
        if mode == "real":
            return np.real(values)
        if mode == "imaginary":
            return np.imag(values)
        raise ValueError("component must be magnitude, phase, real, or imaginary")

    def scanner_volume(self, data):
        """Return a 3D result as scanner-ordered ``(x, y, z)`` values and FOV."""
        if self.kind == "radial_3d":
            ordered = data.transpose("radial_z", "radial_y", "radial_x")
            values = np.asarray(ordered).transpose(2, 1, 0)
            fov = self.dataset.attrs.get("radial_fov_m", "1,1,1")
            if isinstance(fov, str):
                fov = tuple(float(value) for value in fov.split(","))
            return values, tuple(fov)
        if self.kind != "cartesian_3d":
            raise ValueError("scanner_volume requires a 3D reconstruction")
        partition_dim, phase_dim, read_dim = self.spatial_dims
        ordered = data.transpose(partition_dim, phase_dim, read_dim)
        logical = np.asarray(ordered)
        axis_codes = str(
            self.dataset.attrs.get("cartesian_encoding_axes", "+x +y +z")
        ).split()
        role_array_axes = {"partition": 0, "phase": 1, "read": 2}
        role_codes = dict(zip(("read", "phase", "partition"), axis_codes))
        role_for_scanner = {code[-1].lower(): role for role, code in role_codes.items()}
        transpose_axes = tuple(
            role_array_axes[role_for_scanner[axis]] for axis in "xyz"
        )
        scanner = np.transpose(logical, transpose_axes)
        for scanner_axis, axis in enumerate("xyz"):
            role = role_for_scanner[axis]
            if role_codes[role].startswith("-"):
                scanner = np.flip(scanner, axis=scanner_axis)
        role_fov = {
            "read": self._fov_from_k_coordinate("cartesian_k_read_cyc_per_m", read_dim),
            "phase": self._fov_from_k_coordinate(
                "cartesian_k_phase_cyc_per_m", phase_dim
            ),
            "partition": self._fov_from_k_coordinate(
                "cartesian_k_partition_cyc_per_m", partition_dim
            ),
        }
        fov = tuple(role_fov[role_for_scanner[axis]] for axis in "xyz")
        return scanner, fov

    def _fov_from_k_coordinate(self, coordinate: str, dimension: str) -> float:
        if coordinate in self.dataset.coords:
            values = np.asarray(self.dataset.coords[coordinate].values, dtype=float)
            if values.size > 1:
                step = float(np.median(np.diff(values)))
                if np.isfinite(step) and step != 0:
                    return abs(1.0 / step)
        return float(self.dataset.sizes.get(dimension, 1))
