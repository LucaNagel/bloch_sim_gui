"""Non-uniform FFT for reconstructing non-Cartesian k-space data.

The signal model matches the Cartesian reconstruction of this package::

    s(k) = sum_n m(x_n) exp(-2 pi i k . x_n),   x_n = (n - N // 2) * FOV / N

so that ``k = 0`` maps to the image centre and a flat unit k-space covering
the full Cartesian area reconstructs to a unit peak.

:class:`NUFFTOperator` implements the forward (image -> samples) and adjoint
(samples -> image) transforms by Kaiser-Bessel gridding on a ``oversampling``
times larger grid (Beatty et al., IEEE TMI 2005), with analytic
deapodization. Density compensation uses the iterative method of Pipe and
Menon (MRM 1999). Reconstruction is either the density-compensated adjoint
("gridding") or conjugate-gradient least squares. :func:`nudft_forward` and
:func:`nudft_adjoint` evaluate the exact sums for testing small problems.

k-space coordinates are in cycles/m and FOVs in metres. Only numpy and scipy
are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.special import i0

DEFAULT_OVERSAMPLING = 2.0
DEFAULT_KERNEL_WIDTH = 6
# Pipe-Menon handles sample gaps up to about this coarsening by itself; a
# Cartesian 2D grid already has 2 * gap = sqrt(2).
DCF_COARSENING_THRESHOLD = 1.45


def kaiser_bessel_beta(width: float, oversampling: float) -> float:
    """Beatty's near-optimal Kaiser-Bessel shape parameter."""
    value = (width / oversampling) ** 2 * (oversampling - 0.5) ** 2 - 0.8
    return float(np.pi * np.sqrt(max(value, 1e-12)))


def kaiser_bessel_kernel(offset, width: float, beta: float) -> np.ndarray:
    """Kaiser-Bessel kernel evaluated at grid offsets (zero beyond W/2)."""
    offset = np.asarray(offset, dtype=float)
    argument = 1.0 - (2.0 * offset / width) ** 2
    inside = argument >= 0
    values = np.zeros(offset.shape, dtype=float)
    values[inside] = i0(beta * np.sqrt(argument[inside]))
    return values


def kaiser_bessel_transform(frequency, width: float, beta: float) -> np.ndarray:
    """Continuous Fourier transform of :func:`kaiser_bessel_kernel`.

    ``frequency`` is in cycles per grid cell. The result is real; past the
    first zero of ``sinh`` it continues analytically as ``sin``.
    """
    frequency = np.asarray(frequency, dtype=float)
    radicand = beta**2 - (np.pi * width * frequency) ** 2
    root = np.sqrt(np.abs(radicand))
    values = np.empty(frequency.shape, dtype=float)
    small = root < 1e-8
    positive = (radicand > 0) & ~small
    negative = (radicand < 0) & ~small
    values[positive] = np.sinh(root[positive]) / root[positive]
    values[negative] = np.sin(root[negative]) / root[negative]
    values[small] = 1.0
    return width * values


def _as_tuple(values, dims: int, name: str) -> tuple:
    array = np.atleast_1d(np.asarray(values))
    if array.size == 1:
        array = np.repeat(array, dims)
    if array.size != dims:
        raise ValueError(f"{name} needs {dims} entries")
    return tuple(array.tolist())


def image_positions(
    matrix: Sequence[int], fov_m: Sequence[float]
) -> Tuple[np.ndarray, ...]:
    """Voxel centre coordinates in metres per axis, relative to the FOV centre."""
    return tuple(
        (np.arange(int(n)) - int(n) // 2) * (float(fov) / int(n))
        for n, fov in zip(matrix, fov_m)
    )


def nudft_forward(image, k_cyc_per_m, fov_m) -> np.ndarray:
    """Exact non-uniform DFT ``s(k) = sum_x m(x) exp(-2 pi i k.x)``."""
    image = np.asarray(image)
    k = np.atleast_2d(np.asarray(k_cyc_per_m, dtype=float))
    dims = k.shape[1]
    positions = image_positions(image.shape[-dims:], _as_tuple(fov_m, dims, "fov_m"))
    grids = np.meshgrid(*positions, indexing="ij")
    coords = np.stack([grid.ravel() for grid in grids], axis=1)
    phase = np.exp(-2j * np.pi * (k @ coords.T))
    flat = image.reshape(image.shape[:-dims] + (-1,))
    return flat @ phase.T


def nudft_adjoint(samples, k_cyc_per_m, matrix, fov_m) -> np.ndarray:
    """Exact adjoint ``m(x) = sum_k s(k) exp(+2 pi i k.x)`` (unscaled)."""
    samples = np.asarray(samples)
    k = np.atleast_2d(np.asarray(k_cyc_per_m, dtype=float))
    dims = k.shape[1]
    matrix = tuple(int(n) for n in _as_tuple(matrix, dims, "matrix"))
    positions = image_positions(matrix, _as_tuple(fov_m, dims, "fov_m"))
    grids = np.meshgrid(*positions, indexing="ij")
    coords = np.stack([grid.ravel() for grid in grids], axis=1)
    phase = np.exp(2j * np.pi * (coords @ k.T))
    image = samples @ phase.T
    return image.reshape(samples.shape[:-1] + matrix)


def sampling_gap(k_cyc_per_m, fov_m, *, percentile: float = 95.0) -> float:
    """Largest distance from a k-space point inside the sampled region to its
    nearest sample, in units of 1/FOV.

    Lines or turns spaced ``d`` apart give a gap of about ``d / 2``, so
    Nyquist sampling keeps it at or below 0.5 (a Cartesian grid has
    ``sqrt(dims) / 2`` at its cell centres). The probed region is the convex
    hull of the samples, shrunk to 90 % of their extent so that the open
    outer end of a spiral or the corners of a hull do not count as gaps. It
    is probed on a grid of spacing 1/(2 FOV).
    """
    from scipy.spatial import Delaunay, QhullError, cKDTree

    k = np.atleast_2d(np.asarray(k_cyc_per_m, dtype=float))
    k = k[np.all(np.isfinite(k), axis=1)]
    dims = k.shape[1]
    if k.shape[0] <= dims:
        return 0.0
    normalized = k * np.asarray(_as_tuple(fov_m, dims, "fov_m"))
    low, high = normalized.min(axis=0), normalized.max(axis=0)
    axes = [np.arange(np.floor(a), np.ceil(b) + 0.25, 0.5) for a, b in zip(low, high)]
    probes = np.stack([grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")], 1)
    extent = np.maximum(np.abs(normalized).max(axis=0), 1e-12)
    central = np.sum((probes / extent) ** 2, axis=1) <= 0.81
    try:
        inside = central & (Delaunay(normalized).find_simplex(probes) >= 0)
    except QhullError:
        # Collinear samples (a single line) have no area to probe.
        return 0.0
    if not np.any(inside):
        return 0.0
    distance, _ = cKDTree(normalized).query(probes[inside])
    return float(np.percentile(distance, percentile))


@dataclass
class NUFFTOperator:
    """Gridding NUFFT between an image matrix and non-Cartesian samples.

    ``k_cyc_per_m`` has shape ``(samples, dims)`` with ``dims`` equal to the
    length of ``matrix``. Non-finite samples are ignored (zero weight).
    """

    k_cyc_per_m: np.ndarray
    matrix: Tuple[int, ...]
    fov_m: Tuple[float, ...]
    oversampling: float = DEFAULT_OVERSAMPLING
    width: int = DEFAULT_KERNEL_WIDTH
    _interpolation: sparse.csr_matrix = field(init=False, repr=False)
    _deapodization: np.ndarray = field(init=False, repr=False)
    _grid: Tuple[int, ...] = field(init=False, repr=False)
    _valid: np.ndarray = field(init=False, repr=False)
    _density: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _density_key: tuple = field(init=False, default=(), repr=False)
    _gap: Optional[float] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        k = np.atleast_2d(np.asarray(self.k_cyc_per_m, dtype=float))
        dims = k.shape[1]
        self.matrix = tuple(int(n) for n in _as_tuple(self.matrix, dims, "matrix"))
        self.fov_m = tuple(float(v) for v in _as_tuple(self.fov_m, dims, "fov_m"))
        if any(n < 1 for n in self.matrix) or any(v <= 0 for v in self.fov_m):
            raise ValueError("matrix and fov_m must be positive")
        if self.oversampling < 1:
            raise ValueError("oversampling must be at least 1")
        self.k_cyc_per_m = k
        width = int(self.width)
        beta = kaiser_bessel_beta(width, self.oversampling)
        grid = tuple(int(2 * np.ceil(self.oversampling * n / 2.0)) for n in self.matrix)
        self._grid = grid
        self._valid = np.all(np.isfinite(k), axis=1)

        # Sample positions in oversampled grid cells: k * (grid / N) * FOV.
        positions = np.zeros_like(k)
        positions[self._valid] = (
            k[self._valid]
            * np.asarray(self.fov_m)
            * np.asarray(grid)
            / np.asarray(self.matrix)
        )
        per_axis_index = []
        per_axis_weight = []
        for axis in range(dims):
            # Integer grid nodes within W/2 of each sample.
            nodes = (
                np.floor(positions[:, axis] - width / 2.0)
                + 1
                + np.arange(width)[:, None]
            )
            nodes = nodes.T
            weight = kaiser_bessel_kernel(positions[:, axis, None] - nodes, width, beta)
            per_axis_index.append(
                np.mod(nodes.astype(np.int64) + grid[axis] // 2, grid[axis])
            )
            per_axis_weight.append(weight)
        count = k.shape[0]
        index = np.zeros((count,) + (width,) * dims, dtype=np.int64)
        weight = np.ones((count,) + (width,) * dims, dtype=float)
        for axis in range(dims):
            shape = [count] + [1] * dims
            shape[axis + 1] = width
            stride = int(np.prod(grid[axis + 1 :]))
            index = index + per_axis_index[axis].reshape(shape) * stride
            weight = weight * per_axis_weight[axis].reshape(shape)
        weight[~self._valid] = 0.0
        rows = np.repeat(np.arange(count), width**dims)
        self._interpolation = sparse.csr_matrix(
            (weight.ravel(), (rows, index.ravel())),
            shape=(count, int(np.prod(grid))),
        )
        deapodization = np.ones(self.matrix, dtype=float)
        for axis, n in enumerate(self.matrix):
            frequency = (np.arange(n) - n // 2) / grid[axis]
            shape = [1] * dims
            shape[axis] = n
            deapodization = deapodization * kaiser_bessel_transform(
                frequency, width, beta
            ).reshape(shape)
        self._deapodization = deapodization
        self._beta = beta

    @property
    def dims(self) -> int:
        return len(self.matrix)

    @property
    def num_samples(self) -> int:
        return int(self.k_cyc_per_m.shape[0])

    def _pad_slices(self):
        return tuple(
            slice(g // 2 - n // 2, g // 2 - n // 2 + n)
            for g, n in zip(self._grid, self.matrix)
        )

    def forward(self, image) -> np.ndarray:
        """Samples ``s(k)`` of ``image`` (shape ``(..., *matrix)``)."""
        image = np.asarray(image)
        dims = self.dims
        batch = image.shape[:-dims]
        flat = image.reshape((-1,) + self.matrix) / self._deapodization
        padded = np.zeros((flat.shape[0],) + self._grid, dtype=complex)
        padded[(slice(None),) + self._pad_slices()] = flat
        axes = tuple(range(1, dims + 1))
        spectrum = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(padded, axes=axes), axes=axes), axes=axes
        )
        # Grid index j + G//2 holds frequency j, matching the sparse columns.
        samples = self._interpolation @ spectrum.reshape(flat.shape[0], -1).T
        return samples.T.reshape(batch + (self.num_samples,))

    def adjoint(self, samples) -> np.ndarray:
        """Adjoint ``sum_k s(k) exp(+2 pi i k.x)`` on the image matrix."""
        samples = np.asarray(samples)
        dims = self.dims
        batch = samples.shape[:-1]
        flat = samples.reshape(-1, self.num_samples)
        flat = np.where(self._valid, flat, 0.0)
        gridded = (self._interpolation.T @ flat.T).T.reshape(
            (flat.shape[0],) + self._grid
        )
        axes = tuple(range(1, dims + 1))
        image = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(gridded, axes=axes), axes=axes), axes=axes
        ) * float(np.prod(self._grid))
        image = image[(slice(None),) + self._pad_slices()] / self._deapodization
        return image.reshape(batch + self.matrix)

    @property
    def sampling_gap(self) -> float:
        """Largest k-space gap of the samples in units of 1/FOV (see
        :func:`sampling_gap`)."""
        if self._gap is None:
            self._gap = sampling_gap(self.k_cyc_per_m[self._valid], self.fov_m)
        return self._gap

    def density_compensation(
        self, iterations: int = 30, *, coarsening: Optional[float] = None
    ) -> np.ndarray:
        """Pipe-Menon weights, scaled to one per Cartesian k-space cell.

        Pipe-Menon only converges to the sample density when the kernel
        bridges the gaps between samples. Undersampled trajectories (e.g. a
        spiral whose turns are further apart than 1/FOV) therefore use a
        kernel widened by ``coarsening``, by default twice the largest gap
        once that exceeds :data:`DCF_COARSENING_THRESHOLD`.
        """
        if coarsening is None:
            coarsening = 2.0 * self.sampling_gap
            if coarsening <= DCF_COARSENING_THRESHOLD:
                coarsening = 1.0
        coarsening = float(coarsening)
        key = (int(iterations), round(coarsening, 6))
        if self._density is not None and self._density_key == key:
            return self._density
        if coarsening > 1.0 + 1e-6:
            coarse = NUFFTOperator(
                self.k_cyc_per_m,
                tuple(int(np.ceil(n / coarsening)) + 2 for n in self.matrix),
                tuple(fov / coarsening for fov in self.fov_m),
                oversampling=self.oversampling,
                width=self.width,
            )
            weights = coarse.density_compensation(iterations, coarsening=1.0)
            weights = weights * coarsening**self.dims
        else:
            interpolation = self._interpolation
            weights = self._valid.astype(float)
            for _ in range(max(int(iterations), 1)):
                convolved = interpolation @ (interpolation.T @ weights)
                positive = convolved > 0
                weights = np.where(
                    positive, weights / np.where(positive, convolved, 1.0), 0.0
                )
            # At the fixed point the twice-convolved density equals one; a
            # density of rho per grid cell gives rho * psi_hat(0)^(2 dims).
            kernel_area = kaiser_bessel_transform(0.0, self.width, self._beta)
            cells = np.prod(np.asarray(self._grid) / np.asarray(self.matrix))
            weights = weights * kernel_area ** (2 * self.dims) / cells
        self._density = weights
        self._density_key = key
        return weights

    def reconstruct(
        self,
        samples,
        *,
        method: str = "adjoint",
        weights: Optional[np.ndarray] = None,
        iterations: int = 10,
        regularization: float = 0.0,
    ) -> np.ndarray:
        """Reconstruct images from samples of shape ``(..., num_samples)``.

        ``"adjoint"`` is density-compensated gridding. ``"cg"`` solves the
        least-squares problem ``min ||A m - s||^2 + lambda ||m||^2`` by
        conjugate gradients, started from the gridding image and
        preconditioned with the density weights.
        """
        samples = np.asarray(samples)
        if weights is None:
            weights = self.density_compensation()
        scale = 1.0 / float(np.prod(self.matrix))
        initial = self.adjoint(samples * weights) * scale
        if method == "adjoint":
            return initial
        if method != "cg":
            raise ValueError("method must be 'adjoint' or 'cg'")
        return self._conjugate_gradient(
            samples, initial, weights, int(iterations), float(regularization)
        )

    def _conjugate_gradient(
        self, samples, initial, weights, iterations, regularization
    ):
        dims = self.dims
        axes = tuple(range(-dims, 0))

        def normal(image):
            residual = self.forward(image) * weights
            return self.adjoint(residual) + regularization * image

        def dot(a, b):
            return np.sum(np.conj(a) * b, axis=axes, keepdims=True)

        image = initial.astype(complex)
        rhs = self.adjoint(samples * weights)
        residual = rhs - normal(image)
        direction = residual.copy()
        energy = np.real(dot(residual, residual))
        reference = np.real(dot(rhs, rhs))
        for _ in range(iterations):
            mapped = normal(direction)
            curvature = np.real(dot(direction, mapped))
            step = np.where(
                curvature > 0, energy / np.where(curvature > 0, curvature, 1.0), 0.0
            )
            image = image + step * direction
            residual = residual - step * mapped
            updated = np.real(dot(residual, residual))
            if np.all(updated <= 1e-20 * np.maximum(reference, 1e-300)):
                break
            ratio = np.where(
                energy > 0, updated / np.where(energy > 0, energy, 1.0), 0.0
            )
            direction = residual + ratio * direction
            energy = updated
        return image

    def nyquist_fov_m(self) -> Tuple[float, ...]:
        """Approximate FOV free of aliasing given the largest sampling gap
        (line or turn spacing of ``2 * gap / FOV``)."""
        factor = max(1.0, 2.0 * self.sampling_gap)
        return tuple(fov / factor for fov in self.fov_m)

    def point_spread_function(self, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Gridding reconstruction of a unit point source at the FOV centre."""
        samples = np.ones(self.num_samples, dtype=complex)
        return self.reconstruct(samples, weights=weights)
