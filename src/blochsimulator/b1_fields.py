"""Spatial transmit and receive B1 fields with object-space transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import ellipe, ellipk


_TX_KEYS = (
    "tx_sensitivity_map",
    "tx_sensitivity",
    "b1_plus",
    "b1plus",
    "b1_tx",
    "tx",
    "field",
    "data",
)
_RX_KEYS = (
    "rx_sensitivity_maps",
    "rx_sensitivity_map",
    "rx_sensitivities",
    "b1_minus",
    "b1minus",
    "b1_rx",
    "rx",
    "field",
    "data",
)


_PRESET_LABELS = {
    "uniform": "Uniform",
    "birdcage_cp": "Birdcage CP",
    "surface_loop": "Circular surface loop",
    "circular_array": "8-channel circular array",
    "linear_ramp": "Linear / phase ramp",
}


def b1_preset_options(kind: str) -> Tuple[Tuple[str, str], ...]:
    """Return the available ``(identifier, label)`` pairs for a B1 field kind."""
    normalized_kind = (
        "receive" if str(kind).lower() in {"rx", "receive"} else "transmit"
    )
    identifiers = ["uniform", "birdcage_cp", "surface_loop"]
    if normalized_kind == "receive":
        identifiers.append("circular_array")
    identifiers.append("linear_ramp")
    return tuple((identifier, _PRESET_LABELS[identifier]) for identifier in identifiers)


def _field_grid(
    shape: Sequence[int], fov_m: Sequence[float]
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, ...]]:
    axes = tuple(
        ((np.arange(count, dtype=float) + 0.5) / count - 0.5) * extent
        for count, extent in zip(shape, fov_m)
    )
    spatial = tuple(np.meshgrid(*axes, indexing="ij"))
    if len(spatial) == 2:
        xyz = (spatial[0], spatial[1], np.zeros_like(spatial[0]))
    else:
        xyz = (spatial[0], spatial[1], spatial[2])
    return xyz, axes


def _center_reference(values: np.ndarray, axes: Sequence[np.ndarray]) -> complex:
    center = tuple(int(np.argmin(np.abs(axis))) for axis in axes)
    if values.ndim == len(axes) + 1:
        reference = complex(
            np.sqrt(np.sum(np.abs(values[(slice(None), *center)]) ** 2))
        )
    else:
        reference = complex(values[center])
    if not np.isfinite(reference) or abs(reference) <= np.finfo(float).eps:
        maximum = int(np.argmax(np.abs(values)))
        reference = complex(values.ravel()[maximum])
    if abs(reference) <= np.finfo(float).eps:
        return complex(np.finfo(float).eps)
    return reference


def _finite_wire_birdcage(
    xyz: Sequence[np.ndarray], fov_m: Sequence[float], kind: str, n_rungs: int
) -> np.ndarray:
    """Approximate a quadrature birdcage using finite axial current rungs."""
    x, y, z = xyz
    radius = 0.62 * min(fov_m[:2])
    axial_extent = fov_m[2] if len(fov_m) == 3 else min(fov_m[:2])
    length = 1.35 * axial_extent
    bx = np.zeros_like(x, dtype=np.complex128)
    by = np.zeros_like(x, dtype=np.complex128)
    for rung in range(int(n_rungs)):
        phi = 2.0 * np.pi * rung / n_rungs
        dx = x - radius * np.cos(phi)
        dy = y - radius * np.sin(phi)
        rho2 = np.maximum(dx * dx + dy * dy, (radius * 1e-6) ** 2)
        lower = -0.5 * length - z
        upper = 0.5 * length - z
        factor = (
            upper / np.sqrt(rho2 + upper * upper)
            - lower / np.sqrt(rho2 + lower * lower)
        ) / rho2
        phase_sign = -1.0 if kind == "transmit" else 1.0
        current = np.exp(1j * phase_sign * phi)
        bx += current * (-dy) * factor
        by += current * dx * factor
    return bx + 1j * by if kind == "transmit" else bx - 1j * by


def _circular_loop_vector_field(
    xyz: Sequence[np.ndarray],
    *,
    center: Sequence[float],
    normal: Sequence[float],
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a circular-loop Biot-Savart field in arbitrary orientation."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    helper = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.asarray((0.0, 1.0, 0.0))
    basis_u = np.cross(helper, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)

    relative = tuple(axis - float(offset) for axis, offset in zip(xyz, center))
    local_x = sum(relative[index] * basis_u[index] for index in range(3))
    local_y = sum(relative[index] * basis_v[index] for index in range(3))
    axial = sum(relative[index] * normal[index] for index in range(3))
    radial = np.sqrt(local_x * local_x + local_y * local_y)

    alpha2 = (radius - radial) ** 2 + axial * axial
    beta2 = (radius + radial) ** 2 + axial * axial
    alpha2 = np.maximum(alpha2, (radius * 1e-7) ** 2)
    beta = np.sqrt(np.maximum(beta2, (radius * 1e-7) ** 2))
    parameter = np.clip(4.0 * radius * radial / beta2, 0.0, 1.0 - 1e-12)
    first = ellipk(parameter)
    second = ellipe(parameter)

    axial_field = (
        first + second * (radius * radius - radial * radial - axial * axial) / alpha2
    ) / beta
    safe_radial = np.where(radial > radius * 1e-12, radial, 1.0)
    radial_field = (
        axial
        * (
            second * (radius * radius + radial * radial + axial * axial)
            - first * alpha2
        )
        / (safe_radial * alpha2 * beta)
    )
    radial_field = np.where(radial > radius * 1e-12, radial_field, 0.0)

    fields = []
    for component in range(3):
        radial_direction = (
            local_x * basis_u[component] + local_y * basis_v[component]
        ) / safe_radial
        radial_direction = np.where(radial > radius * 1e-12, radial_direction, 0.0)
        fields.append(radial_field * radial_direction + axial_field * normal[component])
    return tuple(fields)


def _surface_loop_field(
    xyz: Sequence[np.ndarray], fov_m: Sequence[float], kind: str
) -> np.ndarray:
    virtual_z = fov_m[2] if len(fov_m) == 3 else min(fov_m[:2])
    radius = 0.30 * min(fov_m[1], virtual_z)
    field = _circular_loop_vector_field(
        xyz,
        center=(-0.56 * fov_m[0], 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        radius=radius,
    )
    return field[0] + 1j * field[1] if kind == "transmit" else field[0] - 1j * field[1]


def _circular_receive_array(
    xyz: Sequence[np.ndarray], fov_m: Sequence[float], channels: int
) -> np.ndarray:
    virtual_z = fov_m[2] if len(fov_m) == 3 else min(fov_m[:2])
    transverse_extent = min(fov_m[:2])
    center_radius = 0.58 * transverse_extent
    loop_radius = 0.30 * min(transverse_extent, virtual_z)
    fields = []
    for channel in range(int(channels)):
        phi = 2.0 * np.pi * channel / channels
        field = _circular_loop_vector_field(
            xyz,
            center=(center_radius * np.cos(phi), center_radius * np.sin(phi), 0.0),
            normal=(-np.cos(phi), -np.sin(phi), 0.0),
            radius=loop_radius,
        )
        fields.append(field[0] - 1j * field[1])
    return np.stack(fields, axis=0)


def rotation_matrix_xyz(rotation_deg_xyz: Sequence[float]) -> np.ndarray:
    """Return the active XYZ rotation matrix used by B1 field transforms."""
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg_xyz, dtype=float))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rotation_z @ rotation_y @ rotation_x


@dataclass
class B1Field:
    """Complex B1 values on a centered 2D or 3D Cartesian field grid.

    ``data`` is stored canonically as ``(channel, *spatial_shape)``. Transmit
    fields have exactly one channel; receive fields may contain one or more.
    Spatial stretching and rotation map the native field grid into the
    phantom's object-space coordinate system.
    """

    data: np.ndarray
    fov_m: Tuple[float, ...]
    kind: str = "transmit"
    spatial_ndim: Optional[int] = None
    name: str = "B1 field"
    scale_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_deg_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip().lower()
        if self.kind in {"tx", "transmit", "b1+", "b1plus"}:
            self.kind = "transmit"
        elif self.kind in {"rx", "receive", "b1-", "b1minus"}:
            self.kind = "receive"
        else:
            raise ValueError("kind must be 'transmit' or 'receive'")

        values = np.asarray(self.data)
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("B1 field data must be numeric")
        if self.spatial_ndim is None:
            if self.kind == "transmit":
                self.spatial_ndim = values.ndim
            elif values.ndim == 4:
                self.spatial_ndim = 3
            else:
                # A plain 2D/3D receive array is one spatial receive field.
                self.spatial_ndim = values.ndim
        self.spatial_ndim = int(self.spatial_ndim)
        if self.spatial_ndim not in (2, 3):
            raise ValueError("B1 fields must be two- or three-dimensional")

        if values.ndim == self.spatial_ndim:
            values = values[None, ...]
        elif values.ndim != self.spatial_ndim + 1:
            raise ValueError(
                "B1 data must have shape (*spatial) or (channel, *spatial)"
            )
        if values.shape[0] < 1 or any(count < 1 for count in values.shape[1:]):
            raise ValueError("B1 field axes and channel count must be non-empty")
        if self.kind == "transmit" and values.shape[0] != 1:
            raise ValueError("a transmit B1 field must contain exactly one channel")
        values = np.asarray(values, dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("B1 field contains NaN or infinite values")
        self.data = values

        self.fov_m = tuple(float(value) for value in self.fov_m)
        if len(self.fov_m) != self.spatial_ndim:
            raise ValueError("B1 field FOV dimensions must match its spatial data")
        if not np.all(np.isfinite(self.fov_m)) or min(self.fov_m) <= 0:
            raise ValueError("B1 field FOV values must be positive and finite")
        self.set_transform(self.scale_xyz, self.rotation_deg_xyz)

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self.data.shape[1:])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def values(self) -> np.ndarray:
        """Return the single Tx map or all receive-channel maps."""
        return self.data[0] if self.kind == "transmit" else self.data

    @property
    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix_xyz(self.rotation_deg_xyz)

    def set_transform(
        self,
        scale_xyz: Sequence[float],
        rotation_deg_xyz: Sequence[float],
    ) -> None:
        scale = np.asarray(scale_xyz, dtype=float)
        rotation = np.asarray(rotation_deg_xyz, dtype=float)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("B1 spatial scale must contain three positive values")
        if rotation.shape != (3,) or not np.all(np.isfinite(rotation)):
            raise ValueError("B1 rotation must contain three finite angles")
        self.scale_xyz = tuple(float(value) for value in scale)
        self.rotation_deg_xyz = tuple(float(value) for value in rotation)

    def set_fov_m(self, fov_m: Sequence[float]) -> None:
        fov = tuple(float(value) for value in fov_m)
        if (
            len(fov) != self.spatial_ndim
            or min(fov) <= 0
            or not np.all(np.isfinite(fov))
        ):
            raise ValueError("B1 field FOV must match its dimensions and be positive")
        self.fov_m = fov

    @classmethod
    def uniform(
        cls,
        shape: Sequence[int],
        fov_m: Sequence[float],
        *,
        kind: str = "transmit",
        value: complex = 1.0 + 0.0j,
        channels: int = 1,
        name: Optional[str] = None,
    ) -> "B1Field":
        shape = tuple(int(count) for count in shape)
        if len(shape) not in (2, 3) or min(shape) < 1:
            raise ValueError("uniform B1 shape must be 2D or 3D and non-empty")
        normalized_kind = (
            "receive" if str(kind).lower() in {"rx", "receive"} else "transmit"
        )
        channels = int(channels)
        if normalized_kind == "transmit":
            channels = 1
        if channels < 1:
            raise ValueError("receive channel count must be positive")
        data = np.full((channels, *shape), complex(value), dtype=np.complex128)
        return cls(
            data=data,
            fov_m=tuple(fov_m),
            kind=normalized_kind,
            spatial_ndim=len(shape),
            name=name
            or ("Uniform B1+" if normalized_kind == "transmit" else "Uniform B1−"),
        )

    def sample_world(self, positions_m: np.ndarray) -> np.ndarray:
        """Interpolate all channels at object-space positions in metres."""
        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_m must have shape (position, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("sample positions must be finite")

        # For row vectors, world = (local * scale) @ R.T, hence the inverse is
        # local = (world @ R) / scale.
        local = (positions @ self.rotation_matrix) / np.asarray(self.scale_xyz)
        coordinates = []
        for axis, (count, extent) in enumerate(zip(self.spatial_shape, self.fov_m)):
            voxel_size = extent / count
            coordinate = (local[:, axis] + extent / 2.0) / voxel_size - 0.5
            # Affine-derived phantom centres and field centres can differ by a
            # few ulps. Keep exact boundary centres inside without admitting
            # genuinely out-of-FOV sample points.
            coordinate[np.isclose(coordinate, 0.0, rtol=0.0, atol=1e-12)] = 0.0
            coordinate[np.isclose(coordinate, count - 1.0, rtol=0.0, atol=1e-12)] = (
                count - 1.0
            )
            coordinates.append(coordinate)
        coordinates = np.asarray(coordinates)
        sampled = np.empty((self.n_channels, positions.shape[0]), dtype=np.complex128)
        for channel, values in enumerate(self.data):
            real = map_coordinates(
                values.real,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            imag = map_coordinates(
                values.imag,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            sampled[channel] = real + 1j * imag
        return sampled

    def resample_to_phantom(self, phantom) -> np.ndarray:
        """Resample onto a phantom grid and return ``(channel, *shape)``."""
        positions = np.asarray(phantom.positions, dtype=float)
        sampled = self.sample_world(positions)
        return sampled.reshape((self.n_channels, *tuple(phantom.shape)))

    def transformed_voxel_positions(
        self, *, max_points: int = 20000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return transformed voxel centres and matching flat source indices."""
        total = int(np.prod(self.spatial_shape))
        stride = max(1, int(np.ceil(total / max(1, int(max_points)))))
        flat_indices = np.arange(0, total, stride, dtype=np.int64)
        unravelled = np.column_stack(np.unravel_index(flat_indices, self.spatial_shape))
        local = np.zeros((flat_indices.size, 3), dtype=float)
        for axis, (count, extent) in enumerate(zip(self.spatial_shape, self.fov_m)):
            local[:, axis] = ((unravelled[:, axis] + 0.5) / count - 0.5) * extent
        positions = (local * np.asarray(self.scale_xyz)) @ self.rotation_matrix.T
        return positions, flat_indices

    def transformed_corners(self) -> np.ndarray:
        """Return transformed field-boundary corners in object space."""
        half = np.zeros(3, dtype=float)
        half[: self.spatial_ndim] = np.asarray(self.fov_m) / 2.0
        if self.spatial_ndim == 2:
            corners = np.asarray(
                [
                    (-half[0], -half[1], 0),
                    (half[0], -half[1], 0),
                    (half[0], half[1], 0),
                    (-half[0], half[1], 0),
                ],
                dtype=float,
            )
        else:
            corners = np.asarray(
                [
                    (x, y, z)
                    for z in (-half[2], half[2])
                    for y in (-half[1], half[1])
                    for x in (-half[0], half[0])
                ],
                dtype=float,
            )
        return (corners * np.asarray(self.scale_xyz)) @ self.rotation_matrix.T


def create_b1_preset(
    preset: str,
    shape: Sequence[int],
    fov_m: Sequence[float],
    *,
    kind: str = "transmit",
    magnitude: float = 1.0,
    phase_deg: float = 0.0,
    ramp_axis: str = "x",
    ramp_mode: str = "magnitude",
    array_channels: int = 8,
    birdcage_rungs: int = 16,
) -> B1Field:
    """Generate a normalized complex 2D/3D B1 preset.

    Coil fields use a finite-rung quadrature birdcage or circular-loop
    Biot-Savart model. Their central sensitivity is normalized to ``magnitude``;
    multi-channel receive arrays use their central root-sum-of-squares value.
    """
    preset = str(preset).strip().lower()
    kind = "receive" if str(kind).lower() in {"rx", "receive"} else "transmit"
    available = {identifier for identifier, _label in b1_preset_options(kind)}
    if preset not in available:
        raise ValueError(f"preset {preset!r} is not available for {kind} fields")

    shape = tuple(int(count) for count in shape)
    fov_m = tuple(float(value) for value in fov_m)
    if len(shape) not in (2, 3) or min(shape) < 1:
        raise ValueError("B1 preset shape must be 2D or 3D and non-empty")
    if len(fov_m) != len(shape) or min(fov_m) <= 0 or not np.all(np.isfinite(fov_m)):
        raise ValueError("B1 preset FOV must match its shape and be positive")
    magnitude = float(magnitude)
    phase_deg = float(phase_deg)
    if not np.isfinite(magnitude) or magnitude < 0:
        raise ValueError("B1 preset magnitude must be finite and non-negative")
    if not np.isfinite(phase_deg):
        raise ValueError("B1 preset phase must be finite")

    xyz, axes = _field_grid(shape, fov_m)
    global_phase = np.exp(1j * np.deg2rad(phase_deg))
    name = _PRESET_LABELS[preset]
    if preset == "uniform":
        return B1Field.uniform(
            shape,
            fov_m,
            kind=kind,
            value=magnitude * global_phase,
            name=name,
        )

    if preset == "birdcage_cp":
        values = _finite_wire_birdcage(xyz, fov_m, kind, max(4, int(birdcage_rungs)))
    elif preset == "surface_loop":
        values = _surface_loop_field(xyz, fov_m, kind)
    elif preset == "circular_array":
        array_channels = int(array_channels)
        if array_channels < 2:
            raise ValueError("a circular receive array needs at least two channels")
        values = _circular_receive_array(xyz, fov_m, array_channels)
        name = f"{array_channels}-channel circular array"
    else:
        axis_name = str(ramp_axis).strip().lower()
        valid_axes = "xyz"[: len(shape)]
        if axis_name not in valid_axes:
            raise ValueError(
                f"ramp axis must be one of {', '.join(valid_axes.upper())}"
            )
        axis = valid_axes.index(axis_name)
        normalized = 2.0 * xyz[axis] / fov_m[axis]
        mode = str(ramp_mode).strip().lower()
        if mode == "magnitude":
            values = magnitude * (1.0 + 0.5 * normalized) * global_phase
            name = f"Linear magnitude ramp {axis_name.upper()}"
        elif mode == "phase":
            values = magnitude * np.exp(
                1j * (np.deg2rad(phase_deg) + np.pi * normalized)
            )
            name = f"Linear phase ramp {axis_name.upper()}"
        else:
            raise ValueError("ramp mode must be 'magnitude' or 'phase'")
        if kind == "receive":
            values = values[None, ...]
        return B1Field(
            data=values,
            fov_m=fov_m,
            kind=kind,
            spatial_ndim=len(shape),
            name=name,
        )

    values = np.asarray(values, dtype=np.complex128)
    values = values / _center_reference(values, axes)
    values *= magnitude * global_phase
    return B1Field(
        data=values,
        fov_m=fov_m,
        kind=kind,
        spatial_ndim=len(shape),
        name=name,
    )


def _mapping_items(mapping: Mapping) -> Mapping[str, np.ndarray]:
    return {str(key): np.asarray(value) for key, value in mapping.items()}


def _find_case_insensitive(mapping: Mapping[str, np.ndarray], key: str):
    lookup = {name.lower(): name for name in mapping}
    actual = lookup.get(key.lower())
    return None if actual is None else mapping[actual]


def _extract_complex_array(
    mapping: Mapping[str, np.ndarray], kind: str
) -> Tuple[np.ndarray, Optional[str], Optional[int]]:
    preferred = _TX_KEYS if kind == "transmit" else _RX_KEYS
    lower_to_actual = {name.lower(): name for name in mapping}
    for key in preferred:
        actual = lower_to_actual.get(key)
        if actual is not None:
            values = np.asarray(mapping[actual])
            spatial_ndim = None
            if kind == "receive" and "maps" in key and values.ndim in (3, 4):
                spatial_ndim = values.ndim - 1
            return values, actual, spatial_ndim

    prefixes = (
        ("tx_sensitivity", "b1_plus", "b1plus")
        if kind == "transmit"
        else (
            "rx_sensitivity",
            "b1_minus",
            "b1minus",
        )
    )
    for prefix in prefixes:
        real_key = lower_to_actual.get(f"{prefix}_real")
        imag_key = lower_to_actual.get(f"{prefix}_imag")
        if real_key is not None and imag_key is not None:
            values = np.asarray(mapping[real_key]) + 1j * np.asarray(mapping[imag_key])
            spatial_ndim = (
                values.ndim - 1 if kind == "receive" and values.ndim in (3, 4) else None
            )
            return values, f"{real_key} + {imag_key}", spatial_ndim

    candidates = []
    metadata_names = {
        "fov",
        "fov_m",
        "fov_mm",
        "shape",
        "scale_xyz",
        "rotation_deg_xyz",
        "affine_ijk_to_xyz_m",
        "spatial_ndim",
        "voxel_size_m",
        "voxel_size_mm",
    }
    for name, values in mapping.items():
        array = np.asarray(values)
        if name.lower() not in metadata_names and np.issubdtype(array.dtype, np.number):
            if array.ndim in (2, 3, 4):
                candidates.append((name, array))
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0], None
    if not candidates:
        raise ValueError("the file does not contain a numeric 2D/3D B1 array")
    names = ", ".join(name for name, _ in candidates)
    raise ValueError(
        f"multiple possible B1 arrays found ({names}); use a standard B1 key"
    )


def _field_fov(
    mapping: Mapping[str, np.ndarray],
    spatial_shape: Sequence[int],
    spatial_ndim: int,
    default_fov_m: Optional[Sequence[float]],
) -> Tuple[float, ...]:
    for key, factor in (("fov_m", 1.0), ("fov_mm", 1e-3), ("fov", 1.0)):
        values = _find_case_insensitive(mapping, key)
        if values is not None:
            fov = tuple(float(value) * factor for value in np.asarray(values).ravel())
            if len(fov) >= spatial_ndim and min(fov[:spatial_ndim]) > 0:
                return fov[:spatial_ndim]
    for key, factor in (("voxel_size_m", 1.0), ("voxel_size_mm", 1e-3)):
        values = _find_case_insensitive(mapping, key)
        if values is not None:
            sizes = np.asarray(values, dtype=float).ravel() * factor
            if sizes.size >= spatial_ndim and min(sizes[:spatial_ndim]) > 0:
                return tuple(
                    float(sizes[axis]) * int(spatial_shape[axis])
                    for axis in range(spatial_ndim)
                )
    affine = _find_case_insensitive(mapping, "affine_ijk_to_xyz_m")
    if affine is not None:
        affine = np.asarray(affine, dtype=float).reshape(4, 4)
        voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
        if min(voxel_sizes[:spatial_ndim]) > 0:
            return tuple(
                float(voxel_sizes[axis]) * int(spatial_shape[axis])
                for axis in range(spatial_ndim)
            )
    if default_fov_m is not None:
        defaults = tuple(float(value) for value in default_fov_m)
        if len(defaults) >= spatial_ndim and min(defaults[:spatial_ndim]) > 0:
            return defaults[:spatial_ndim]
    return (0.24,) * spatial_ndim


def _load_mapping(path: Path) -> Mapping[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    if suffix in {".h5", ".hdf5"}:
        import h5py

        arrays = {}
        with h5py.File(path, "r") as handle:

            def collect(name, item):
                if isinstance(item, h5py.Dataset):
                    arrays[name] = item[...]

            handle.visititems(collect)
            for key, value in handle.attrs.items():
                arrays.setdefault(str(key), np.asarray(value))
        return arrays
    if suffix == ".mat":
        from scipy.io import loadmat

        return {
            key: value
            for key, value in loadmat(path).items()
            if not key.startswith("__")
        }
    if suffix == ".nc":
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            loaded = dataset.load()
        arrays = {name: np.asarray(value) for name, value in loaded.data_vars.items()}
        arrays.update(
            {str(key): np.asarray(value) for key, value in loaded.attrs.items()}
        )
        return arrays
    raise ValueError(f"unsupported B1 file format: {path.suffix}")


def load_b1_field(
    filename,
    *,
    kind: str,
    default_fov_m: Optional[Sequence[float]] = None,
) -> B1Field:
    """Load a complex 2D/3D B1 field from NumPy, HDF5, MATLAB, or NetCDF."""
    path = Path(filename)
    normalized_kind = (
        "receive" if str(kind).lower() in {"rx", "receive"} else "transmit"
    )
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        mapping = {}
        source_key = None
        value_ndim = np.asarray(values).ndim
        if normalized_kind == "receive" and value_ndim == 4:
            spatial_ndim = 3
        elif (
            normalized_kind == "transmit"
            and value_ndim == 4
            and np.asarray(values).shape[0] == 1
        ):
            spatial_ndim = 3
        else:
            spatial_ndim = value_ndim
    else:
        mapping = _mapping_items(_load_mapping(path))
        values, source_key, spatial_ndim = _extract_complex_array(
            mapping, normalized_kind
        )

    if spatial_ndim is None:
        if normalized_kind == "transmit":
            value_ndim = np.asarray(values).ndim
            spatial_ndim = (
                3
                if value_ndim == 4 and np.asarray(values).shape[0] == 1
                else value_ndim
            )
        elif np.asarray(values).ndim == 4:
            spatial_ndim = 3
        else:
            spatial_ndim = np.asarray(values).ndim
    if spatial_ndim not in (2, 3):
        raise ValueError("loaded B1 data must describe a 2D or 3D field")
    array = np.asarray(values)
    spatial_shape = array.shape[-spatial_ndim:]
    fov_m = _field_fov(mapping, spatial_shape, spatial_ndim, default_fov_m)
    field = B1Field(
        data=array,
        fov_m=fov_m,
        kind=normalized_kind,
        spatial_ndim=spatial_ndim,
        name=path.stem + (f" [{source_key}]" if source_key else ""),
        source_path=str(path),
    )
    scale = _find_case_insensitive(mapping, "scale_xyz")
    rotation = _find_case_insensitive(mapping, "rotation_deg_xyz")
    if scale is not None or rotation is not None:
        field.set_transform(
            field.scale_xyz if scale is None else np.asarray(scale).ravel()[:3],
            (
                field.rotation_deg_xyz
                if rotation is None
                else np.asarray(rotation).ravel()[:3]
            ),
        )
    return field
