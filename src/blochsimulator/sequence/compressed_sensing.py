"""Compressed-sensing sampling masks and reconstruction for Cartesian k-space.

Pure numpy, independent of Qt and of the acquisition classes. A k-space grid
uses the same layout and FFT convention as
:meth:`CartesianAcquisition.reconstruct`: the spatial axes are the last two
or three axes, DC sits at index ``n // 2`` and the image is
``fftshift(ifftn(ifftshift(kspace)))``. Masks are boolean arrays that
broadcast against the grid, ``True`` where a sample was acquired.

The reconstruction solves

    min_x 0.5 ||M F x - y||^2 + l_w ||W x||_1 + l_tv TV(x) + l_t ||D_t x||_1

with the primal-dual algorithm of Chambolle and Pock (J Math Imaging Vis
40:120, 2011). ``W`` is a multi-level orthonormal Haar transform, ``TV`` the
isotropic total variation over the spatial axes and ``D_t`` the finite
difference along an optional temporal axis, as in k-t SPARSE. The sparsity
model follows Lustig, Donoho and Pauly (MRM 58:1182, 2007).
The data term has a closed-form proximal step for a masked unitary FFT, so
every iteration costs two FFTs. The weights are relative to the peak of the
zero-filled image, which makes them independent of the signal scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


CS_REFERENCE_DOI = "10.1002/mrm.21391"
PRIMAL_DUAL_REFERENCE_DOI = "10.1007/s10851-010-0251-1"


def _rng(seed):
    return (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )


def _centre_block(size: int, count: int) -> np.ndarray:
    """Return ``count`` indices around the DC index ``size // 2``."""
    start = size // 2 - count // 2
    return np.arange(start, start + count)


def _validate_acceleration(acceleration, samples_available):
    acceleration = float(acceleration)
    if not np.isfinite(acceleration) or acceleration < 1.0:
        raise ValueError("acceleration must be at least 1")
    return max(1, int(round(samples_available / acceleration)))


def variable_density_mask(
    shape: Sequence[int],
    acceleration: float,
    *,
    centre_size: Optional[Sequence[int]] = None,
    density_power: float = 2.0,
    seed=0,
) -> np.ndarray:
    """Draw a variable-density random mask over one or two phase axes.

    ``shape`` is ``(ky,)`` for 2D imaging (one mask entry per phase line) or
    ``(ky, kz)`` for 3D. ``round(prod(shape) / acceleration)`` points are
    drawn without replacement: the central block of ``centre_size`` (default
    about 8 % of each axis, at least 2) always, the rest with probability
    ``(1 - r) ** density_power``, where ``r`` is the normalised distance
    from DC (an elliptical radius for 2D). The draw is reproducible for an
    integer ``seed``.
    """
    shape = tuple(int(value) for value in shape)
    if len(shape) not in (1, 2) or min(shape) < 1:
        raise ValueError("shape must contain one or two positive sizes")
    if float(density_power) < 0 or not np.isfinite(float(density_power)):
        raise ValueError("density_power must be non-negative and finite")
    total = int(np.prod(shape))
    count = _validate_acceleration(acceleration, total)
    if centre_size is None:
        centre_size = tuple(
            min(size, max(2, int(round(0.08 * size)))) for size in shape
        )
    centre_size = tuple(
        int(value) for value in np.broadcast_to(centre_size, len(shape))
    )
    if any(value < 0 or value > size for value, size in zip(centre_size, shape)):
        raise ValueError("centre_size must lie between zero and the axis size")

    mask = np.zeros(shape, dtype=bool)
    mask[np.ix_(*(_centre_block(size, c) for size, c in zip(shape, centre_size)))] = (
        True
    )
    if int(mask.sum()) > count:
        raise ValueError(
            "the fully sampled centre holds more points than the acceleration "
            "allows; reduce centre_size or the acceleration"
        )
    grids = np.meshgrid(
        *((np.arange(size) - size // 2) / max(size / 2.0, 1.0) for size in shape),
        indexing="ij",
    )
    radius = np.sqrt(sum(np.square(grid) for grid in grids))
    density = np.clip(1.0 - radius / max(float(radius.max()), 1e-12), 0.0, 1.0)
    density = np.power(density, float(density_power)) + 1e-6
    density[mask] = 0.0
    remaining = count - int(mask.sum())
    if remaining:
        flat = density.ravel()
        chosen = _rng(seed).choice(
            flat.size, size=remaining, replace=False, p=flat / flat.sum()
        )
        mask.ravel()[chosen] = True
    return mask


def kt_variable_density_mask(
    phase_matrix: int,
    frames: int,
    acceleration: float,
    *,
    centre_lines: Optional[int] = None,
    density_power: float = 2.0,
    seed=0,
) -> np.ndarray:
    """Return a ``(frames, ky)`` mask with an independent draw per frame.

    Varying the random lines between frames spreads the aliasing over time,
    so a temporal sparsity term can separate it from the dynamics.
    """
    frames = int(frames)
    if frames < 1:
        raise ValueError("frames must be positive")
    rng = _rng(seed)
    centre = None if centre_lines is None else (int(centre_lines),)
    return np.stack(
        [
            variable_density_mask(
                (int(phase_matrix),),
                acceleration,
                centre_size=centre,
                density_power=density_power,
                seed=rng,
            )
            for _ in range(frames)
        ]
    )


def expand_phase_mask(mask, read_matrix: int) -> np.ndarray:
    """Repeat a ``(..., ky)`` line mask along a trailing read axis."""
    mask = np.asarray(mask, dtype=bool)
    return np.broadcast_to(mask[..., None], mask.shape + (int(read_matrix),))


def mask_point_spread_function(mask, spatial_ndim: Optional[int] = None):
    """Return the transform point-spread function and its peak sidelobe.

    The PSF is the image of the mask itself, normalised to its peak; the
    largest off-peak magnitude measures the incoherence of the aliasing
    (Lustig et al., MRM 58:1182, 2007). Lower is better.
    """
    mask = np.asarray(mask, dtype=float)
    ndim = mask.ndim if spatial_ndim is None else int(spatial_ndim)
    axes = tuple(range(-ndim, 0))
    psf = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(mask, axes=axes), axes=axes), axes=axes
    )
    peak_index = tuple(size // 2 for size in mask.shape[-ndim:])
    peak = np.abs(psf[(...,) + peak_index])
    psf = psf / np.where(peak > 0, peak, 1.0)[(...,) + (None,) * ndim]
    magnitude = np.abs(psf).copy()
    magnitude[(...,) + peak_index] = 0.0
    sidelobe = magnitude.reshape(magnitude.shape[: mask.ndim - ndim] + (-1,)).max(-1)
    return psf, sidelobe


def _forward_fft(image, axes):
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _inverse_fft(kspace, axes):
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _norm_scale(norm, size):
    """Factor from an orthonormal image to numpy's ``norm`` convention."""
    if norm in (None, "backward"):
        return 1.0 / np.sqrt(size)
    if norm == "ortho":
        return 1.0
    if norm == "forward":
        return np.sqrt(size)
    raise ValueError("norm must be None, 'backward', 'ortho' or 'forward'")


def zero_filled_reconstruction(kspace, mask, *, spatial_ndim: int = 2, norm=None):
    """Return the inverse FFT of the masked grid (unsampled points zero)."""
    kspace = np.asarray(kspace)
    axes = tuple(range(-int(spatial_ndim), 0))
    sampled = np.where(np.broadcast_to(mask, kspace.shape), kspace, 0.0)
    size = int(np.prod([kspace.shape[axis] for axis in axes]))
    return _inverse_fft(sampled, axes) * _norm_scale(norm, size)


class HaarWavelet:
    """Multi-level orthonormal Haar transform over the trailing spatial axes.

    Axes are zero-padded to a multiple of ``2 ** levels``; the padded
    transform is an isometry, so the adjoint is the inverse followed by the
    crop and the operator norm is one. The coefficients use the in-place
    Mallat layout.
    """

    def __init__(self, shape: Sequence[int], levels: int = 3):
        self.shape = tuple(int(value) for value in shape)
        smallest = min(self.shape)
        self.levels = max(
            0, min(int(levels), int(np.floor(np.log2(max(smallest, 1)))) - 1)
        )
        block = 2**self.levels
        self.padded_shape = tuple(-(-size // block) * block for size in self.shape)

    def _axes(self, array):
        return tuple(range(array.ndim - len(self.shape), array.ndim))

    def forward(self, image):
        axes = self._axes(image)
        padding = [(0, 0)] * (image.ndim - len(self.shape)) + [
            (0, padded - size) for size, padded in zip(self.shape, self.padded_shape)
        ]
        coefficients = np.pad(image, padding).astype(complex, copy=True)
        sizes = list(self.padded_shape)
        for _ in range(self.levels):
            block = (Ellipsis,) + tuple(slice(0, size) for size in sizes)
            values = coefficients[block]
            for axis in axes:
                even = np.take(values, np.arange(0, values.shape[axis], 2), axis=axis)
                odd = np.take(values, np.arange(1, values.shape[axis], 2), axis=axis)
                values = np.concatenate(
                    ((even + odd) / np.sqrt(2.0), (even - odd) / np.sqrt(2.0)),
                    axis=axis,
                )
            coefficients[block] = values
            sizes = [size // 2 for size in sizes]
        return coefficients

    def adjoint(self, coefficients):
        axes = self._axes(coefficients)
        image = np.array(coefficients, dtype=complex, copy=True)
        for level in reversed(range(self.levels)):
            sizes = [size // 2**level for size in self.padded_shape]
            block = (Ellipsis,) + tuple(slice(0, size) for size in sizes)
            values = image[block]
            for axis in reversed(axes):
                low, high = np.split(values, 2, axis=axis)
                values = _interleave(
                    (low + high) / np.sqrt(2.0), (low - high) / np.sqrt(2.0), axis
                )
            image[block] = values
        crop = (Ellipsis,) + tuple(slice(0, size) for size in self.shape)
        return image[crop]


def _interleave(even, odd, axis):
    """Interleave two equal arrays along ``axis`` (even first)."""
    stacked = np.stack((even, odd), axis=axis + 1)
    shape = list(even.shape)
    shape[axis] *= 2
    return stacked.reshape(shape)


def _difference(image, axis):
    """Forward difference with a zero last entry (Neumann boundary)."""
    zero = np.zeros_like(np.take(image, [0], axis=axis))
    return np.concatenate((np.diff(image, axis=axis), zero), axis=axis)


def _difference_adjoint(values, axis):
    """Adjoint of :func:`_difference`: ``p[i - 1] - p[i]``, last ``p`` zero."""
    size = values.shape[axis]
    inner = np.take(values, np.arange(size - 1), axis=axis)
    zero = np.zeros_like(np.take(values, [0], axis=axis))
    return np.concatenate((zero, inner), axis=axis) - np.concatenate(
        (inner, zero), axis=axis
    )


def _gradient(image, axes):
    return np.stack([_difference(image, axis) for axis in axes])


def _gradient_adjoint(field, axes):
    """Adjoint of :func:`_gradient`, the negative divergence."""
    return sum(
        _difference_adjoint(component, axis) for component, axis in zip(field, axes)
    )


def _project_elementwise(dual, radius=1.0):
    magnitude = np.abs(dual)
    return dual / np.maximum(1.0, magnitude / radius)


def _project_isotropic(dual, radius=1.0):
    magnitude = np.sqrt(np.sum(np.abs(dual) ** 2, axis=0, keepdims=True))
    return dual / np.maximum(1.0, magnitude / radius)


@dataclass(frozen=True)
class CompressedSensingResult:
    """Reconstructed image, zero-filled reference and solver history.

    ``objective`` holds the scaled objective every ``objective_interval``
    iterations; the image follows the ``norm`` convention of the call.
    """

    image: np.ndarray
    zero_filled: np.ndarray
    iterations: int
    objective: tuple[float, ...]
    scale: float


def reconstruct_compressed_sensing(
    kspace,
    mask,
    *,
    spatial_ndim: int = 2,
    wavelet_weight: float = 0.0,
    tv_weight: float = 0.0,
    temporal_weight: float = 0.0,
    temporal_axis: Optional[int] = None,
    wavelet_levels: int = 3,
    iterations: int = 100,
    objective_interval: int = 10,
    norm=None,
) -> CompressedSensingResult:
    """Reconstruct undersampled Cartesian k-space with sparsity priors.

    ``kspace`` holds the full grid ``(..., [kz,] ky, kx)``; entries where
    ``mask`` is False are ignored. ``temporal_axis`` names a leading axis of
    the frames for the temporal finite-difference term; other leading axes
    (coils, slices, echoes) are reconstructed independently. All weights
    zero returns the zero-filled image.
    """
    kspace = np.asarray(kspace, dtype=complex)
    spatial_ndim = int(spatial_ndim)
    if spatial_ndim not in (2, 3) or kspace.ndim < spatial_ndim:
        raise ValueError("spatial_ndim must be 2 or 3 and fit the k-space array")
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), kspace.shape)
    weights = tuple(
        float(value) for value in (wavelet_weight, tv_weight, temporal_weight)
    )
    if any(value < 0 or not np.isfinite(value) for value in weights):
        raise ValueError("regularisation weights must be non-negative and finite")
    if temporal_weight > 0:
        if temporal_axis is None:
            raise ValueError("temporal_weight needs a temporal_axis")
        temporal_axis = int(temporal_axis) % kspace.ndim
        if temporal_axis >= kspace.ndim - spatial_ndim:
            raise ValueError("temporal_axis must be a leading, non-spatial axis")
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("iterations must be positive")

    axes = tuple(range(-spatial_ndim, 0))
    spatial_shape = kspace.shape[-spatial_ndim:]
    size = int(np.prod(spatial_shape))
    output_scale = _norm_scale(norm, size)
    measured = np.where(mask, kspace, 0.0)
    zero_filled = _inverse_fft(measured, axes)
    scale = float(np.max(np.abs(zero_filled))) if zero_filled.size else 0.0
    if scale == 0.0 or not any(weights):
        return CompressedSensingResult(
            zero_filled * output_scale, zero_filled * output_scale, 0, (), scale
        )
    measured = measured / scale
    weight_mask = mask.astype(float)
    wavelet = HaarWavelet(spatial_shape, wavelet_levels)
    spatial_axes = tuple(kspace.ndim + axis for axis in axes)

    operators = []
    if wavelet_weight > 0:
        operators.append(
            (
                wavelet_weight,
                wavelet.forward,
                wavelet.adjoint,
                _project_elementwise,
                1.0,
            )
        )
    if tv_weight > 0:
        operators.append(
            (
                tv_weight,
                lambda x: _gradient(x, spatial_axes),
                lambda p: _gradient_adjoint(p, spatial_axes),
                _project_isotropic,
                4.0 * spatial_ndim,
            )
        )
    if temporal_weight > 0:
        operators.append(
            (
                temporal_weight,
                lambda x: _difference(x, temporal_axis),
                lambda p: _difference_adjoint(p, temporal_axis),
                _project_elementwise,
                4.0,
            )
        )
    lipschitz = np.sqrt(sum(weight**2 * bound for weight, *_, bound in operators))
    tau = sigma = 0.99 / lipschitz

    def objective(image):
        residual = np.where(mask, _forward_fft(image, axes) - measured, 0.0)
        value = 0.5 * float(np.sum(np.abs(residual) ** 2))
        for weight, apply, _, project, _ in operators:
            coefficients = apply(image)
            if project is _project_isotropic:
                value += weight * float(
                    np.sum(np.sqrt(np.sum(np.abs(coefficients) ** 2, axis=0)))
                )
            else:
                value += weight * float(np.sum(np.abs(coefficients)))
        return value

    image = zero_filled / scale
    extrapolated = image.copy()
    duals = [np.zeros_like(apply(image)) for _, apply, *_ in operators]
    history = []
    for iteration in range(iterations):
        back_projection = np.zeros_like(image)
        for index, (weight, apply, adjoint, project, _) in enumerate(operators):
            duals[index] = project(duals[index] + sigma * weight * apply(extrapolated))
            back_projection += weight * adjoint(duals[index])
        # Closed-form data consistency of the masked unitary FFT.
        candidate = _forward_fft(image - tau * back_projection, axes)
        candidate = (candidate + tau * measured) / (1.0 + tau * weight_mask)
        updated = _inverse_fft(candidate, axes)
        extrapolated = 2.0 * updated - image
        image = updated
        if objective_interval and (iteration + 1) % int(objective_interval) == 0:
            history.append(objective(image))
    return CompressedSensingResult(
        image * scale * output_scale,
        zero_filled * output_scale,
        iterations,
        tuple(history),
        scale,
    )


def normalized_rmse(estimate, reference) -> float:
    """Return ``||estimate - reference|| / ||reference||``."""
    reference = np.asarray(reference)
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError("reference must be non-zero")
    return float(np.linalg.norm(np.asarray(estimate) - reference)) / denominator
