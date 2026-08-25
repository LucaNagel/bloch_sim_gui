"""Deterministic intravoxel spin sampling for sequence simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinSampling:
    """Spin points used to integrate magnetization inside each voxel.

    ``counts_xyz`` is always expressed in physical object X/Y/Z order. Missing
    phantom dimensions must retain their singleton count. The default therefore
    reproduces the historical one-spin-per-voxel simulation exactly.

    ``method='midpoint'`` is the regular tensor-product midpoint rule. It is
    symmetric and converges predictably, but repeated integer-cycle spoilers can
    rephase it exactly. ``method='stratified'`` places one deterministic,
    independently jittered point in every tensor-product stratum. It retains
    equal weights and spatial coverage without the short exact recurrence of a
    regular grid.
    """

    counts_xyz: tuple[int, int, int] = (1, 1, 1)
    method: str = "midpoint"
    selected_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        counts = tuple(self.counts_xyz)
        if len(counts) != 3:
            raise ValueError("counts_xyz must contain X, Y, and Z spin counts")
        validated = []
        for axis, value in zip("XYZ", counts):
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(
                    f"subvoxel spin count {axis} must be a positive integer"
                )
            value = int(value)
            if value <= 0:
                raise ValueError(
                    f"subvoxel spin count {axis} must be a positive integer"
                )
            validated.append(value)
        method = str(self.method).strip().lower()
        if method not in {"midpoint", "stratified"}:
            raise ValueError("spin sampling method must be 'midpoint' or 'stratified'")
        object.__setattr__(self, "counts_xyz", tuple(validated))
        object.__setattr__(self, "method", method)
        if self.selected_indices is not None:
            selected = tuple(self.selected_indices)
            if not selected:
                raise ValueError("selected subvoxel spin indices must not be empty")
            grid_count = int(np.prod(validated))
            normalized = []
            for value in selected:
                if isinstance(value, (bool, np.bool_)) or int(value) != value:
                    raise ValueError("selected subvoxel spin indices must be integers")
                value = int(value)
                if value < 0 or value >= grid_count:
                    raise ValueError(
                        "selected subvoxel spin index is outside the full grid"
                    )
                normalized.append(value)
            if len(set(normalized)) != len(normalized):
                raise ValueError("selected subvoxel spin indices must be unique")
            object.__setattr__(self, "selected_indices", tuple(normalized))

    @property
    def grid_spins_per_voxel(self) -> int:
        """Number of points in the complete midpoint grid."""
        count_x, count_y, count_z = self.counts_xyz
        return count_x * count_y * count_z

    @property
    def spins_per_voxel(self) -> int:
        if self.selected_indices is not None:
            return len(self.selected_indices)
        return self.grid_spins_per_voxel

    @property
    def enabled(self) -> bool:
        # A one-point subset of a larger grid still needs its non-central
        # physical offset and its original quadrature weight.
        return self.grid_spins_per_voxel > 1

    def select(self, indices) -> "SpinSampling":
        """Return a partial grid that preserves the complete grid's weights."""
        if self.selected_indices is not None:
            raise ValueError("cannot select again from a partial spin grid")
        return SpinSampling(
            counts_xyz=self.counts_xyz,
            method=self.method,
            selected_indices=tuple(indices),
        )

    def validate_phantom_dimensions(self, ndim: int) -> None:
        """Reject sampling along axes for which a phantom has no cell extent."""
        ndim = int(ndim)
        if ndim not in (1, 2, 3):
            raise ValueError("phantom dimensionality must be 1, 2, or 3")
        for axis_index in range(ndim, 3):
            if self.counts_xyz[axis_index] != 1:
                axis = "XYZ"[axis_index]
                raise ValueError(
                    f"subvoxel sampling along {axis} requires an explicit "
                    f"{axis}-axis phantom extent"
                )

    def normalized_offsets_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fractional voxel offsets and normalized quadrature weights."""
        cell_axes = [np.arange(count, dtype=np.int64) for count in self.counts_xyz]
        cell_x, cell_y, cell_z = np.meshgrid(*cell_axes, indexing="ij")
        cells = np.column_stack((cell_x.ravel(), cell_y.ravel(), cell_z.ravel()))
        if self.method == "midpoint":
            within_cell = np.full(cells.shape, 0.5, dtype=np.float64)
        else:
            point_indices = np.arange(cells.shape[0], dtype=np.uint64)
            within_cell = np.column_stack(
                (
                    _deterministic_unit_interval(point_indices, 0),
                    _deterministic_unit_interval(point_indices, 1),
                    _deterministic_unit_interval(point_indices, 2),
                )
            )
            # A singleton axis has no resolved intravoxel extent. Keeping its
            # point centered avoids a deterministic positional bias.
            for axis_index, count in enumerate(self.counts_xyz):
                if count == 1:
                    within_cell[:, axis_index] = 0.5
        counts = np.asarray(self.counts_xyz, dtype=np.float64)
        offsets = (cells + within_cell) / counts - 0.5
        weights = np.full(
            self.grid_spins_per_voxel,
            1.0 / self.grid_spins_per_voxel,
            dtype=np.float64,
        )
        if self.selected_indices is not None:
            selected = np.asarray(self.selected_indices, dtype=np.intp)
            offsets = offsets[selected]
            weights = weights[selected]
        return offsets, weights

    def offsets_m(self, voxel_basis_m) -> tuple[np.ndarray, np.ndarray]:
        """Return physical offsets for the supplied three voxel basis vectors."""
        basis = np.asarray(voxel_basis_m, dtype=np.float64)
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise ValueError("voxel_basis_m must be a finite 3x3 matrix")
        normalized, weights = self.normalized_offsets_and_weights()
        return np.ascontiguousarray(normalized @ basis.T), weights

    def to_metadata(self) -> dict:
        return {
            "counts_xyz": tuple(int(value) for value in self.counts_xyz),
            "spins_per_voxel": self.spins_per_voxel,
            "grid_spins_per_voxel": self.grid_spins_per_voxel,
            "method": self.method,
            "selected_indices": self.selected_indices,
            "quadrature_weight_sum": (self.spins_per_voxel / self.grid_spins_per_voxel),
        }


def coerce_spin_sampling(value=None) -> SpinSampling:
    """Accept a :class:`SpinSampling`, three counts, one isotropic count, or None."""
    if value is None:
        return SpinSampling()
    if isinstance(value, SpinSampling):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        count = int(value)
        return SpinSampling((count, count, count))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return SpinSampling(tuple(value))
    raise TypeError(
        "spin_sampling must be SpinSampling, three XYZ counts, one integer, or None"
    )


def _deterministic_unit_interval(indices: np.ndarray, axis_index: int) -> np.ndarray:
    """Map integer point IDs to reproducible, decorrelated values in ``(0, 1)``.

    This is a vectorized SplitMix64 finalizer with a distinct stream per axis.
    It is deliberately local and stateless: simulation results never depend on
    NumPy's process-wide random-number generator.
    """
    values = np.asarray(indices, dtype=np.uint64) + np.uint64(
        0x9E3779B97F4A7C15 * (int(axis_index) + 1) & ((1 << 64) - 1)
    )
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    mantissa = values >> np.uint64(11)
    return (mantissa.astype(np.float64) + 0.5) * (1.0 / float(1 << 53))


def phantom_voxel_basis_m(phantom) -> np.ndarray:
    """Return physical X/Y/Z cell vectors for an axis-aligned Phantom grid."""
    basis = np.zeros((3, 3), dtype=np.float64)
    affine = np.asarray(phantom.affine_ijk_to_xyz_m, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("phantom affine must be a finite 4x4 matrix")
    ndim = int(phantom.ndim)
    basis[:, :ndim] = affine[:3, :ndim]
    return basis
