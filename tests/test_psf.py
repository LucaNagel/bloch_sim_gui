from pathlib import Path

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.sequence import load_pulseq, make_pulseq_flash
from blochsimulator.sequence.acquisition import CartesianAcquisition, make_cartesian_epi
from blochsimulator.sequence.advanced_bssfp_builders import (
    make_pulseq_spectral_selective_bssfp,
)
from blochsimulator.sequence.kspace_trajectory import (
    GradientImperfections,
    gradient_error,
)
from blochsimulator.sequence.pulseq_builders import make_pulseq_spiral
from blochsimulator.sequence.psf import (
    PointSource,
    cartesian_psf_layouts,
    estimate_psf,
    estimate_slice_profile,
    evaluate_cartesian_image,
    psf_from_result,
    simulate_point_sources,
)

# Continuous FWHM of the Dirichlet kernel sin(pi x) / (N sin(pi x / N)) in
# voxels; it approaches 1.2067 for large N.
IDEAL_FWHM_VOXELS = {4: 1.2350, 8: 1.2136, 16: 1.2083}


def _simulator():
    return BlochSimulator(use_parallel=False)


def _round_trip(sequence, path: Path):
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _epi(read=16, phase=16, fov=(0.08, 0.08), dwell_s=10e-6):
    acquisition = CartesianAcquisition.epi(read, phase, fov, dwell_s)
    program = make_cartesian_epi(
        acquisition,
        rf_duration_s=100e-6,
        prephaser_duration_s=100e-6,
        blip_duration_s=50e-6,
    )
    return acquisition, program


def _spiral(tmp_path, matrix=16, fov_m=0.08):
    sequence = make_pulseq_spiral(
        fov_m=(fov_m, fov_m),
        matrix=(matrix, matrix),
        sampling_bandwidth_hz=50e3,
        read_offset_m=0.0,
    )
    return _round_trip(sequence, tmp_path / "spiral.seq")


def _echo_spacing_s(result):
    acquisition = result.cartesian_acquisition
    times = np.asarray(result.adc_times_s).reshape(
        acquisition.phase_matrix, acquisition.read_matrix
    )
    return float(np.mean(np.diff(times[:, acquisition.read_matrix // 2])))


def test_lorentzian_ensemble_reproduces_reversible_decay():
    source = PointSource(t2_s=0.1, t2_star_s=10e-3, off_resonance_hz=30.0)
    rate = 1.0 / 10e-3 - 1.0 / 0.1
    frequencies, weights = source.frequency_ensemble()
    times = np.linspace(-5.0 / rate, 5.0 / rate, 401)
    signal = weights @ np.exp(2j * np.pi * np.outer(frequencies, times))
    expected = np.exp(-rate * np.abs(times)) * np.exp(2j * np.pi * 30.0 * times)

    assert frequencies.size == 49
    assert 0.95 < weights.sum() < 1.0
    away_from_echo = np.abs(times) >= 0.3 / rate
    np.testing.assert_allclose(
        signal[away_from_echo], expected[away_from_echo], atol=0.012
    )
    np.testing.assert_allclose(
        signal[~away_from_echo], expected[~away_from_echo], atol=0.05
    )
    assert PointSource(t2_s=5e-3, t2_star_s=10e-3).frequency_ensemble()[0].size == 1


def test_point_sources_have_separate_signals_matching_single_runs():
    _, program = _epi(read=8, phase=8)
    sources = (
        PointSource(position_m=(0.01, 0.0, 0.0)),
        PointSource(position_m=(-0.02, 0.015, 0.0), off_resonance_hz=150.0),
        PointSource(t1_s=0.05, t2_s=0.02, t2_star_s=5e-3, t2_star_spins=9),
    )
    combined = simulate_point_sources(program, sources, simulator=_simulator())

    assert combined.signal.shape == (3, 64)
    for index, source in enumerate(sources):
        single = simulate_point_sources(program, (source,), simulator=_simulator())
        np.testing.assert_allclose(
            combined.signal[index], single.signal[0], rtol=1e-6, atol=1e-9
        )


def test_dft_at_voxel_centres_equals_sequence_reconstruction():
    acquisition, program = _epi()
    source = PointSource(position_m=(0.013, -0.007, 0.0), off_resonance_hz=80.0)
    result = simulate_point_sources(program, (source,), simulator=_simulator())
    layout = cartesian_psf_layouts(result)[0]
    centres = [
        (np.arange(count) - count // 2 + 0.5) * size
        for count, size in zip(layout.matrix, layout.voxel_size_m)
    ]

    image = evaluate_cartesian_image(
        layout.kspace(result)[0], layout.k_cyc_per_m, centres
    )
    reference = acquisition.reconstruct(result.signal[0])
    np.testing.assert_allclose(image, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "position_m", [(0.0, 0.0, 0.0), (0.012, 0.0, 0.0), (0.0031, -0.0217, 0.004)]
)
def test_ideal_epi_psf_is_a_dirichlet_kernel_at_the_source(position_m):
    _, program = _epi()
    psf = estimate_psf(
        program, PointSource(position_m=position_m), simulator=_simulator()
    )

    assert psf.roles == ("phase", "read")
    assert psf.psf.shape == (64, 64)
    for role in ("phase", "read"):
        metrics = psf.axis_metrics[role]
        assert metrics.fwhm_voxels == pytest.approx(IDEAL_FWHM_VOXELS[16], abs=2e-3)
        assert metrics.broadening == pytest.approx(1.0, abs=1e-6)
        assert metrics.peak_offset_voxels == pytest.approx(0.0, abs=1e-3)
        assert metrics.peak_sidelobe_ratio == pytest.approx(0.22, abs=0.01)
    assert psf.peak_efficiency == pytest.approx(1.0, abs=1e-6)
    assert psf.voxel_energy_fraction == pytest.approx(
        psf.ideal_voxel_energy_fraction, abs=1e-9
    )
    np.testing.assert_allclose(psf.psf, psf.ideal_psf, atol=1e-9)


def test_epi_off_resonance_shifts_the_psf_along_phase():
    _, program = _epi()
    off_resonance_hz = 200.0
    result = simulate_point_sources(
        program,
        (PointSource(off_resonance_hz=off_resonance_hz),),
        simulator=_simulator(),
    )
    psf = psf_from_result(result, PointSource(off_resonance_hz=off_resonance_hz))

    expected = off_resonance_hz * _echo_spacing_s(result) * 16
    assert psf.axis_metrics["phase"].peak_offset_voxels == pytest.approx(
        expected, abs=0.01
    )
    assert psf.axis_metrics["read"].peak_offset_voxels == pytest.approx(0.0, abs=0.01)
    assert psf.voxel_energy_fraction < psf.ideal_voxel_energy_fraction


def test_epi_t2_star_broadens_only_the_phase_direction():
    _, program = _epi()
    psf = estimate_psf(program, PointSource(t2_star_s=3e-3), simulator=_simulator())

    assert psf.axis_metrics["phase"].broadening > 1.03
    assert psf.axis_metrics["read"].broadening == pytest.approx(1.0, abs=0.005)
    assert psf.axis_metrics["phase"].peak_offset_voxels == pytest.approx(0.0, abs=0.01)
    # The modulation at the k-space centre equals the decayed echo signal.
    modulation = np.abs(psf.modulation[:, 8])
    assert modulation[0] > modulation[8] > modulation[-1]


def test_3d_volume_psf_resolves_all_three_axes(tmp_path):
    sequence = make_pulseq_spectral_selective_bssfp(
        fov_m=(56e-3, 28e-3, 21e-3),
        matrix=(8, 4, 4),
        target_frequency_offsets_hz=(0.0,),
        receiver_frequency_offsets_hz=(0.0,),
        target_metabolite_names=("A",),
        flip_angle_deg=(30.0,),
        repetitions=1,
        rf_phase_increment_deg=180.0,
    )
    program = _round_trip(sequence, tmp_path / "bssfp_3d.seq")
    # One voxel off-centre on every axis, and on the partition FOV edge.
    sources = (
        PointSource(position_m=(7e-3, 7e-3, 5.25e-3)),
        PointSource(position_m=(0.0, 0.0, -10.5e-3)),
    )
    result = simulate_point_sources(program, sources, simulator=_simulator())

    for index, source in enumerate(sources):
        psf = psf_from_result(result, source, source_index=index)
        assert psf.roles == ("partition", "phase", "read")
        assert psf.psf.shape == (16, 16, 32)
        for role, count in zip(psf.roles, (4, 4, 8)):
            metrics = psf.axis_metrics[role]
            assert metrics.fwhm_voxels == pytest.approx(
                IDEAL_FWHM_VOXELS[count], abs=2e-3
            )
            assert metrics.peak_offset_voxels == pytest.approx(0.0, abs=1e-3)
        assert psf.peak_efficiency == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(ValueError, match="3D acquisitions"):
        estimate_slice_profile(program, positions_m=(0.0, 1e-3), simulator=_simulator())


@pytest.mark.parametrize("encoding_axes", [("+x", "+y", "+z"), ("+x", "+z", "-y")])
def test_flash_slice_profiles_match_the_slice_package(tmp_path, encoding_axes):
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        flip_angle_deg=10.0,
        slice_thickness_m=3e-3,
        slice_gap_m=1e-3,
        n_slices=2,
        slice_offset_m=4e-3,
        encoding_axes=encoding_axes,
    )
    program = _round_trip(sequence, tmp_path / "flash.seq")
    source = PointSource(position_m=(0.01, 0.0, 0.0))
    profile = estimate_slice_profile(program, source, simulator=_simulator())

    normal = profile.normal_xyz
    assert abs(normal[2] if encoding_axes[2] == "+z" else normal[1]) == 1.0
    assert profile.frame_labels == ("slice=0", "slice=1")
    assert profile.slice_indices == (0, 1)
    assert profile.profiles.shape == (2, profile.positions_m.size)
    for metrics, centre in zip(profile.metrics, (2e-3, 6e-3)):
        assert metrics.nominal_centre_m == pytest.approx(centre)
        assert metrics.centre_m == pytest.approx(centre, abs=0.05e-3)
        assert metrics.thickness_ratio == pytest.approx(1.0, abs=0.05)
        assert 0.0 < metrics.out_of_slice_fraction < 0.3

    # Each slice image responds to its own slab: little cross-talk at the
    # other slice centre.
    magnitude = np.abs(profile.profiles)
    at_first = int(np.argmin(np.abs(profile.positions_m - 2e-3)))
    at_second = int(np.argmin(np.abs(profile.positions_m - 6e-3)))
    assert magnitude[1, at_first] < 0.05 * magnitude[0, at_first]
    assert magnitude[0, at_second] < 0.05 * magnitude[1, at_second]


@pytest.mark.parametrize("nucleus", ["C13", "H1"])
def test_hyperpolarized_flash_depletion_broadens_the_phase_psf(tmp_path, nucleus):
    flip_angle_deg = 20.0
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.08),
        matrix=(16, 16),
        sampling_bandwidth_hz=10_000.0,
        echo_time_s=3e-3,
        repetition_time_s=10e-3,
        flip_angle_deg=flip_angle_deg,
        nucleus=nucleus,
    )
    program = _round_trip(sequence, tmp_path / "hp_flash.seq")
    source = PointSource(t1_s=30.0, t2_s=1.0, equilibrium_mz=1e-5)
    psf = estimate_psf(program, source, simulator=_simulator())

    # Linear phase order: line n sees sin(a) cos(a)^n of the initial Mz.
    alpha = np.deg2rad(flip_angle_deg)
    line_signal = np.abs(psf.modulation[:, 8])
    expected = np.sin(alpha) * np.cos(alpha) ** np.arange(16)
    np.testing.assert_allclose(line_signal, expected, rtol=0.01)
    assert psf.axis_metrics["phase"].broadening > 1.02
    assert psf.axis_metrics["read"].broadening == pytest.approx(1.0, abs=1e-3)
    assert psf.axis_metrics["phase"].peak_sidelobe_ratio > (
        psf.axis_metrics["phase"].ideal_peak_sidelobe_ratio
    )


def test_slice_profile_requires_geometry_or_positions():
    _, program = _epi(read=8, phase=8)
    with pytest.raises(ValueError, match="SliceThickness"):
        estimate_slice_profile(program, simulator=_simulator())

    profile = estimate_slice_profile(
        program, positions_m=np.linspace(-0.01, 0.01, 5), simulator=_simulator()
    )
    # A hard pulse excites every position equally.
    np.testing.assert_allclose(
        np.abs(profile.profiles[0]), np.abs(profile.profiles[0, 0]), rtol=1e-9
    )


def test_point_source_validation():
    with pytest.raises(ValueError, match="position_m"):
        PointSource(position_m=(0.0, 0.0))
    with pytest.raises(ValueError, match="t2_star_s"):
        PointSource(t2_star_s=0.0)
    with pytest.raises(ValueError, match="t2_star_spins"):
        PointSource(t2_star_spins=0)
    moved = PointSource(t1_s=2.0).moved_to((1e-3, 2e-3, 3e-3))
    assert moved.position_m == (1e-3, 2e-3, 3e-3)
    assert moved.t1_s == 2.0


def test_spiral_psf_matches_the_spiral_reconstruction(tmp_path):
    program = _spiral(tmp_path)
    source = PointSource(position_m=(0.01, -0.005, 0.0))
    result = simulate_point_sources(program, (source,), simulator=_simulator())
    psf = psf_from_result(result, source)

    assert psf.kind == "spiral"
    assert psf.roles == ("y", "x")
    assert psf.modulation.shape == (256,) == psf.k_cyc_per_m[0].shape
    image = np.abs(result.spiral_acquisition.reconstruct(result, 0)[0])
    np.testing.assert_allclose(np.abs(psf.psf).max(), image.max(), rtol=0.02)
    for role in psf.roles:
        metrics = psf.axis_metrics[role]
        assert abs(metrics.peak_offset_voxels) < 0.05
        assert metrics.broadening == pytest.approx(1.0, abs=0.05)
    assert psf.peak_efficiency == pytest.approx(1.0, abs=0.02)


def test_spiral_gradient_delay_shifts_psf_unless_trajectory_is_corrected(tmp_path):
    program = _spiral(tmp_path)
    source = PointSource(position_m=(0.012, -0.007, 0.0))
    result = simulate_point_sources(program, (source,), simulator=_simulator())
    delta = gradient_error(program, GradientImperfections(delay_s=(5e-6, 5e-6, 0.0)))

    uncorrected = psf_from_result(result, source, gradient_error_cyc_per_m=delta)
    corrected = psf_from_result(
        result, source, gradient_error_cyc_per_m=delta, trajectory_correction=True
    )

    assert uncorrected.gradient_error == "uncorrected"
    assert corrected.gradient_error == "corrected"
    shift = max(abs(m.peak_offset_voxels) for m in uncorrected.axis_metrics.values())
    assert shift > 0.02
    for metrics in corrected.axis_metrics.values():
        assert abs(metrics.peak_offset_voxels) < 1e-3
    assert corrected.peak_efficiency > 0.999
    assert uncorrected.peak_efficiency < corrected.peak_efficiency


def test_epi_gradient_delay_ghost_vanishes_on_the_corrected_trajectory():
    acquisition, program = _epi(read=16, phase=16)
    source = PointSource(position_m=(0.01, 0.0, 0.0))
    result = simulate_point_sources(program, (source,), simulator=_simulator())
    delta = gradient_error(program, GradientImperfections(delay_s=(4e-6, 0.0, 0.0)))
    ideal = psf_from_result(result, source)
    uncorrected = psf_from_result(result, source, gradient_error_cyc_per_m=delta)
    corrected = psf_from_result(
        result, source, gradient_error_cyc_per_m=delta, trajectory_correction=True
    )

    def ghost(psf):
        # Peak of the N/2 ghost along phase relative to the main peak.
        magnitude = np.abs(psf.psf)
        axis = psf.roles.index("phase")
        offsets = psf.offsets_m[axis] / psf.voxel_size_m[axis]
        profile = magnitude.max(axis=1 - axis)
        return profile[np.abs(np.abs(offsets) - 8) < 1.5].max() / magnitude.max()

    assert ghost(uncorrected) > 0.3
    assert ghost(corrected) < 0.1
    assert corrected.peak_efficiency == pytest.approx(ideal.peak_efficiency, rel=1e-3)


def test_gradient_error_shape_is_checked():
    _, program = _epi(read=8, phase=8)
    source = PointSource()
    result = simulate_point_sources(program, (source,), simulator=_simulator())
    with pytest.raises(ValueError):
        psf_from_result(result, source, gradient_error_cyc_per_m=np.zeros((3, 3)))
