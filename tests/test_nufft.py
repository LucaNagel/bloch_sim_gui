import numpy as np
import pytest

from blochsimulator.sequence.nufft import (
    NUFFTOperator,
    kaiser_bessel_kernel,
    kaiser_bessel_transform,
    nudft_adjoint,
    nudft_forward,
    sampling_gap,
)


def _spiral(turn_spacing, radius=16.0, step=0.2):
    angles = [0.0]
    while turn_spacing * angles[-1] / (2 * np.pi) < radius:
        current = max(turn_spacing * angles[-1] / (2 * np.pi), 0.05)
        angles.append(angles[-1] + step / current)
    angles = np.asarray(angles)
    radii = turn_spacing * angles / (2 * np.pi)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def _cartesian(matrix, fov):
    axes = [(np.arange(n) - n // 2) / f for n, f in zip(matrix, fov)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([grid.ravel() for grid in grids])


def test_kaiser_bessel_transform_matches_numerical_integral():
    width, beta = 6, 13.9
    offsets = np.linspace(-width / 2, width / 2, 20001)
    kernel = kaiser_bessel_kernel(offsets, width, beta)
    for frequency in (0.0, 0.1, 0.3, 0.6):
        numeric = np.trapezoid(
            kernel * np.cos(2 * np.pi * offsets * frequency), offsets
        )
        assert kaiser_bessel_transform(frequency, width, beta) == pytest.approx(
            numeric, rel=1e-6
        )


@pytest.mark.parametrize(
    "matrix,fov", [((16, 12), (0.12, 0.1)), ((8, 6, 4), (0.1, 0.08, 0.05))]
)
def test_forward_and_adjoint_match_the_exact_nudft(matrix, fov):
    rng = np.random.default_rng(1)
    k = (rng.random((300, len(matrix))) - 0.5) * np.asarray(matrix) / np.asarray(fov)
    image = rng.standard_normal(matrix) + 1j * rng.standard_normal(matrix)
    samples = rng.standard_normal(300) + 1j * rng.standard_normal(300)
    operator = NUFFTOperator(k, matrix, fov)
    exact_forward = nudft_forward(image, k, fov)
    exact_adjoint = nudft_adjoint(samples, k, matrix, fov)
    assert (
        np.abs(operator.forward(image) - exact_forward).max()
        < 1e-4 * np.abs(exact_forward).max()
    )
    assert (
        np.abs(operator.adjoint(samples) - exact_adjoint).max()
        < 1e-4 * np.abs(exact_adjoint).max()
    )
    # <A x, y> = <x, A^H y> for the gridding implementation itself.
    left = np.vdot(operator.forward(image), samples)
    right = np.vdot(image, operator.adjoint(samples))
    assert left == pytest.approx(right, rel=1e-10)


def test_full_cartesian_sampling_reconstructs_the_image_exactly():
    matrix, fov = (16, 12), (0.12, 0.1)
    k = _cartesian(matrix, fov)
    operator = NUFFTOperator(k, matrix, fov)
    weights = operator.density_compensation()
    assert weights == pytest.approx(np.ones(k.shape[0]), rel=0.02)
    image = np.random.default_rng(2).standard_normal(matrix)
    samples = nudft_forward(image, k, fov)
    reconstruction = operator.reconstruct(samples, weights=np.ones(k.shape[0]))
    assert np.abs(reconstruction - image).max() < 1e-3
    # A flat unit k-space is a unit point at the FOV centre.
    psf = operator.point_spread_function(weights=np.ones(k.shape[0]))
    assert psf[8, 6] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("turn_spacing", (1.0, 1.6, 2.5))
def test_density_weights_integrate_to_the_sampled_area(turn_spacing):
    k = _spiral(turn_spacing)
    operator = NUFFTOperator(k, (32, 32), (1.0, 1.0))
    assert operator.sampling_gap == pytest.approx(turn_spacing / 2, rel=0.15)
    weights = operator.density_compensation()
    inner = np.hypot(*k.T) < 12
    assert weights[inner].sum() == pytest.approx(np.pi * 144, rel=0.05)


def test_sampling_gap_of_a_cartesian_grid_is_half_a_diagonal():
    assert sampling_gap(_cartesian((16, 16), (1.0, 1.0)), 1.0) == pytest.approx(
        np.sqrt(2) / 2
    )


def test_spiral_disc_reconstruction_with_gridding_and_cg():
    matrix, fov = (32, 32), (0.12, 0.12)
    k = _spiral(1.0) / fov[0]
    positions = (np.arange(32) - 16) * fov[0] / 32
    x, y = np.meshgrid(positions, positions, indexing="ij")
    disc = ((x - 0.01) ** 2 + y**2 < 0.035**2).astype(float)
    samples = nudft_forward(disc, k, fov)
    operator = NUFFTOperator(k, matrix, fov)
    gridding = operator.reconstruct(samples)
    iterative = operator.reconstruct(samples, method="cg", iterations=15)
    gridding_error = np.sqrt(np.mean(np.abs(gridding - disc) ** 2))
    iterative_error = np.sqrt(np.mean(np.abs(iterative - disc) ** 2))
    assert abs(gridding[18, 16]) == pytest.approx(1.0, abs=0.1)
    assert iterative_error < gridding_error
    assert iterative_error < 0.15
    with pytest.raises(ValueError, match="method"):
        operator.reconstruct(samples, method="fft")
