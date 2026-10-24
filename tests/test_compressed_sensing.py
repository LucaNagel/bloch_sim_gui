import numpy as np
import pytest

from blochsimulator.sequence import (
    HaarWavelet,
    expand_phase_mask,
    infer_cartesian_acquisition,
    kt_variable_density_mask,
    load_pulseq,
    make_pulseq_flash,
    mask_point_spread_function,
    normalized_rmse,
    reconstruct_compressed_sensing,
    variable_density_mask,
    zero_filled_reconstruction,
)
from blochsimulator.sequence.compressed_sensing import (
    _difference,
    _difference_adjoint,
    _gradient,
    _gradient_adjoint,
)


def _kspace(image):
    axes = (-2, -1)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes)), axes=axes)


def _piecewise_constant(n=64):
    y, x = np.mgrid[:n, :n] / n - 0.5
    image = np.zeros((n, n))
    image[(x / 0.4) ** 2 + (y / 0.45) ** 2 < 1] = 1.0
    image[(x + 0.12) ** 2 + (y + 0.1) ** 2 < 0.12**2] = 0.4
    image[(x - 0.15) ** 2 / 0.08**2 + (y - 0.1) ** 2 / 0.18**2 < 1] = 1.8
    image[(np.abs(x) < 0.05) & (np.abs(y - 0.3) < 0.05)] = 0.6
    return image


def test_variable_density_mask_keeps_count_centre_and_seed():
    mask = variable_density_mask((64,), 3.0, seed=1)
    assert mask.dtype == bool
    assert int(mask.sum()) == round(64 / 3)
    # About 8 % of the lines around DC (index 32) are always sampled.
    assert mask[30:35].all()
    np.testing.assert_array_equal(mask, variable_density_mask((64,), 3.0, seed=1))
    assert not np.array_equal(mask, variable_density_mask((64,), 3.0, seed=2))

    plane = variable_density_mask((40, 24), 4.0, centre_size=(4, 2), seed=0)
    assert int(plane.sum()) == round(40 * 24 / 4)
    assert plane[18:22, 11:13].all()

    # The density falls off with distance from DC.
    draws = np.stack(
        [variable_density_mask((64,), 4.0, centre_size=0, seed=s) for s in range(50)]
    )
    rate = draws.mean(axis=0)
    assert rate[24:40].mean() > 2.0 * np.r_[rate[:8], rate[-8:]].mean()
    assert variable_density_mask((16,), 1.0).all()


def test_variable_density_mask_rejects_invalid_requests():
    with pytest.raises(ValueError, match="acceleration"):
        variable_density_mask((32,), 0.5)
    with pytest.raises(ValueError, match="centre"):
        variable_density_mask((32,), 8.0, centre_size=8)
    with pytest.raises(ValueError, match="centre_size"):
        variable_density_mask((32,), 2.0, centre_size=40)
    with pytest.raises(ValueError, match="shape"):
        variable_density_mask((8, 8, 8), 2.0)
    with pytest.raises(ValueError, match="density_power"):
        variable_density_mask((32,), 2.0, density_power=-1.0)


def test_kt_mask_varies_between_frames_and_keeps_the_centre():
    mask = kt_variable_density_mask(48, 6, 4.0, centre_lines=4, seed=3)
    assert mask.shape == (6, 48)
    assert (mask.sum(axis=1) == 12).all()
    assert mask[:, 22:26].all()
    assert len({row.tobytes() for row in mask}) == 6
    np.testing.assert_array_equal(
        mask, kt_variable_density_mask(48, 6, 4.0, centre_lines=4, seed=3)
    )
    expanded = expand_phase_mask(mask, 32)
    assert expanded.shape == (6, 48, 32)
    assert (expanded == mask[..., None]).all()


def test_random_mask_aliasing_is_incoherent_unlike_regular_undersampling():
    _, full = mask_point_spread_function(np.ones(32, dtype=bool))
    assert float(full) == pytest.approx(0.0, abs=1e-12)

    regular = np.zeros(64, dtype=bool)
    regular[::2] = True
    _, regular_sidelobe = mask_point_spread_function(regular)
    _, random_sidelobe = mask_point_spread_function(
        variable_density_mask((64,), 2.0, seed=0)
    )
    # Every second line folds a full-amplitude replica at FOV / 2.
    assert float(regular_sidelobe) == pytest.approx(1.0)
    assert float(random_sidelobe) < 0.7

    _, per_frame = mask_point_spread_function(
        kt_variable_density_mask(64, 3, 2.0, seed=0), spatial_ndim=1
    )
    assert per_frame.shape == (3,)


@pytest.mark.parametrize(
    ("shape", "levels"), [((64, 64), 3), ((30, 20), 2), ((16, 12, 10), 2)]
)
def test_haar_wavelet_is_an_isometry_with_a_matching_adjoint(shape, levels):
    rng = np.random.default_rng(0)
    wavelet = HaarWavelet(shape, levels)
    image = rng.normal(size=(2,) + shape) + 1j * rng.normal(size=(2,) + shape)
    coefficients = wavelet.forward(image)
    assert coefficients.shape == (2,) + wavelet.padded_shape
    assert all(size % 2**wavelet.levels == 0 for size in wavelet.padded_shape)

    np.testing.assert_allclose(np.linalg.norm(coefficients), np.linalg.norm(image))
    np.testing.assert_allclose(wavelet.adjoint(coefficients), image, atol=1e-12)
    dual = rng.normal(size=coefficients.shape) + 1j * rng.normal(
        size=coefficients.shape
    )
    assert np.vdot(coefficients, dual) == pytest.approx(
        np.vdot(image, wavelet.adjoint(dual))
    )

    # A constant image is a single coarse coefficient per block.
    flat = wavelet.forward(np.ones(wavelet.padded_shape))
    assert np.count_nonzero(np.abs(flat) > 1e-9) == np.prod(
        [size // 2**wavelet.levels for size in wavelet.padded_shape]
    )


def test_difference_and_gradient_adjoints_match():
    rng = np.random.default_rng(1)
    image = rng.normal(size=(3, 9, 7)) + 1j * rng.normal(size=(3, 9, 7))
    for axis in range(3):
        values = rng.normal(size=image.shape) + 1j * rng.normal(size=image.shape)
        assert np.vdot(_difference(image, axis), values) == pytest.approx(
            np.vdot(image, _difference_adjoint(values, axis))
        )
    axes = (1, 2)
    field = rng.normal(size=(2,) + image.shape)
    assert np.vdot(_gradient(image, axes), field) == pytest.approx(
        np.vdot(image, _gradient_adjoint(field, axes))
    )
    # The Neumann boundary keeps a constant image out of the TV term.
    assert np.abs(_gradient(np.ones((5, 5)), (0, 1))).max() == 0.0


@pytest.mark.parametrize("norm", [None, "ortho", "forward"])
def test_fully_sampled_grid_matches_the_cartesian_inverse_fft(norm):
    rng = np.random.default_rng(2)
    kspace = rng.normal(size=(2, 16, 12)) + 1j * rng.normal(size=(2, 16, 12))
    axes = (-2, -1)
    expected = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm),
        axes=axes,
    )
    np.testing.assert_allclose(
        zero_filled_reconstruction(kspace, True, norm=norm), expected, atol=1e-12
    )
    plain = reconstruct_compressed_sensing(kspace, True, norm=norm)
    np.testing.assert_allclose(plain.image, expected, atol=1e-12)
    assert plain.iterations == 0

    # Light regularisation of a fully sampled grid keeps the image.
    regularised = reconstruct_compressed_sensing(
        kspace, True, tv_weight=1e-4, iterations=50, norm=norm
    )
    assert normalized_rmse(regularised.image, expected) < 5e-3


def test_tv_and_wavelet_reconstructions_beat_zero_filling():
    image = _piecewise_constant()
    kspace = _kspace(image)
    lines = expand_phase_mask(variable_density_mask((64,), 3.0, seed=1), 64)
    zero_filled = zero_filled_reconstruction(kspace, lines)
    tv = reconstruct_compressed_sensing(kspace, lines, tv_weight=0.002, iterations=200)

    assert normalized_rmse(zero_filled, image) > 0.15
    assert normalized_rmse(tv.image, image) < 0.02
    np.testing.assert_allclose(tv.zero_filled, zero_filled)
    assert len(tv.objective) == 20
    assert tv.objective[-1] < tv.objective[0]

    # Wavelet sparsity needs incoherence along both axes, as a (ky, kz)
    # point mask of a 3D acquisition provides.
    points = variable_density_mask((64, 64), 3.0, seed=1)
    wavelet = reconstruct_compressed_sensing(
        kspace, points, wavelet_weight=0.003, iterations=300
    )
    assert normalized_rmse(wavelet.image, image) < 0.1 * normalized_rmse(
        zero_filled_reconstruction(kspace, points), image
    )
    # Both priors together still converge on the line mask.
    combined = reconstruct_compressed_sensing(
        kspace, lines, tv_weight=0.003, wavelet_weight=0.002, iterations=200
    )
    assert normalized_rmse(combined.image, image) < 0.03


def test_kt_reconstruction_uses_temporal_sparsity():
    n, frames = 48, 10
    y, x = np.mgrid[:n, :n] / n - 0.5
    body = ((x / 0.4) ** 2 + (y / 0.42) ** 2 < 1).astype(float)
    lesion = ((x - 0.12) ** 2 + (y + 0.08) ** 2 < 0.1**2).astype(float)
    vessel = ((x + 0.15) ** 2 + (y - 0.1) ** 2 < 0.06**2).astype(float)
    t = np.arange(frames)[:, None, None]
    series = (
        0.5 * body + lesion * (1 - np.exp(-t / 3.0)) + vessel * t * np.exp(-t / 2.5)
    )
    kspace = _kspace(series)
    mask = expand_phase_mask(kt_variable_density_mask(n, frames, 3.0, seed=2), n)

    spatial = reconstruct_compressed_sensing(
        kspace, mask, tv_weight=0.002, iterations=300
    )
    temporal = reconstruct_compressed_sensing(
        kspace,
        mask,
        tv_weight=0.002,
        temporal_weight=0.005,
        temporal_axis=0,
        iterations=300,
    )
    zero_filled = normalized_rmse(spatial.zero_filled, series)
    assert normalized_rmse(spatial.image, series) < 0.2 * zero_filled
    assert normalized_rmse(temporal.image, series) < 0.7 * normalized_rmse(
        spatial.image, series
    )

    # The same lines in every frame give the temporal term nothing to separate.
    repeated = np.broadcast_to(mask[:1], mask.shape)
    frozen = reconstruct_compressed_sensing(
        kspace,
        repeated,
        tv_weight=0.002,
        temporal_weight=0.005,
        temporal_axis=0,
        iterations=300,
    )
    assert normalized_rmse(temporal.image, series) < 0.5 * normalized_rmse(
        frozen.image, series
    )


def test_reconstruction_rejects_invalid_arguments():
    kspace = np.zeros((4, 8, 8), dtype=complex)
    with pytest.raises(ValueError, match="temporal_axis"):
        reconstruct_compressed_sensing(kspace, True, temporal_weight=0.1)
    with pytest.raises(ValueError, match="leading"):
        reconstruct_compressed_sensing(
            kspace, True, temporal_weight=0.1, temporal_axis=-1
        )
    with pytest.raises(ValueError, match="non-negative"):
        reconstruct_compressed_sensing(kspace, True, tv_weight=-1.0)
    with pytest.raises(ValueError, match="spatial_ndim"):
        reconstruct_compressed_sensing(kspace, True, spatial_ndim=1)
    with pytest.raises(ValueError, match="iterations"):
        reconstruct_compressed_sensing(kspace, True, tv_weight=0.1, iterations=0)
    # An empty grid has nothing to regularise.
    empty = reconstruct_compressed_sensing(kspace, True, tv_weight=0.1)
    assert empty.iterations == 0 and not empty.image.any()


def test_retrospective_cs_of_a_bloch_simulated_flash(tmp_path):
    from blochsimulator.phantom import Phantom
    from blochsimulator.simulator import BlochSimulator

    n = 32
    sequence = make_pulseq_flash(
        fov_m=(0.12, 0.12),
        matrix=(n, n),
        echo_time_s=4e-3,
        repetition_time_s=12e-3,
        repetitions=1,
    )
    path = tmp_path / "flash.seq"
    sequence.write(path)
    program = load_pulseq(path)
    acquisition = infer_cartesian_acquisition(program)

    centres = (np.arange(n) + 0.5) / n - 0.5
    x, y = np.meshgrid(centres, centres, indexing="ij")
    density = np.zeros((n, n))
    density[(x / 0.4) ** 2 + (y / 0.35) ** 2 < 1] = 1.0
    density[(x - 0.12) ** 2 + (y + 0.05) ** 2 < 0.1**2] = 0.4
    density[(np.abs(x + 0.15) < 0.06) & (np.abs(y - 0.1) < 0.08)] = 1.6
    shape = (n, n, 1)
    phantom = Phantom(
        shape=shape,
        fov=(0.12, 0.12, 3e-3),
        t1_map=np.ones(shape),
        t2_map=np.full(shape, 0.1),
        pd_map=density[..., None],
    )
    signal = (
        BlochSimulator(use_parallel=False).simulate_sequence(program, phantom).signal
    )

    # Undersample the reshaped grid of a fully sampled run retrospectively.
    dx = acquisition.fov_m[0] / acquisition.read_matrix
    dy = acquisition.fov_m[1] / acquisition.phase_matrix
    kspace = acquisition.reshape_signal(signal) * np.exp(
        2j
        * np.pi
        * (
            np.asarray(acquisition.ky_cyc_per_m)[:, None] * dy / 2
            + np.asarray(acquisition.kx_cyc_per_m)[None, :] * dx / 2
        )
    )
    reference = acquisition.reconstruct(signal)
    np.testing.assert_allclose(
        zero_filled_reconstruction(kspace, True), reference, atol=1e-12
    )

    mask = expand_phase_mask(variable_density_mask((n,), 2.5, seed=0), n)
    result = reconstruct_compressed_sensing(
        kspace, mask, tv_weight=0.002, iterations=300
    )
    assert normalized_rmse(result.image, reference) < 0.6 * normalized_rmse(
        result.zero_filled, reference
    )
