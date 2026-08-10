"""Deterministic intravoxel spin sampling for sequence simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinSampling:
    """Midpoint spin grid used to integrate magnetization inside each voxel.

    ``counts_xyz`` is always expressed in physical object X/Y/Z order. Missing
    phantom dimensions must retain their singleton count. The default therefore
    reproduces the historical one-spin-per-voxel simulation exactly.
    """

    counts_xyz: tuple[int, int, int] = (1, 1, 1)
    method: str = "midpoint"

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
        if method != "midpoint":
            raise ValueError("spin sampling method must be 'midpoint'")
        object.__setattr__(self, "counts_xyz", tuple(validated))
        object.__setattr__(self, "method", method)

    @property
    def spins_per_voxel(self) -> int:
        count_x, count_y, count_z = self.counts_xyz
        return count_x * count_y * count_z

    @property
    def enabled(self) -> bool:
        return self.spins_per_voxel > 1

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
        axes = [
            (np.arange(count, dtype=np.float64) + 0.5) / count - 0.5
            for count in self.counts_xyz
        ]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        offsets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        weights = np.full(
            self.spins_per_voxel,
            1.0 / self.spins_per_voxel,
            dtype=np.float64,
        )
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
            "method": self.method,
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


def phantom_voxel_basis_m(phantom) -> np.ndarray:
    """Return physical X/Y/Z cell vectors for an axis-aligned Phantom grid."""
    basis = np.zeros((3, 3), dtype=np.float64)
    affine = np.asarray(phantom.affine_ijk_to_xyz_m, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("phantom affine must be a finite 4x4 matrix")
    ndim = int(phantom.ndim)
    basis[:, :ndim] = affine[:3, :ndim]
    return basis
