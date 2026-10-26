from pathlib import Path

import numpy as np
import pytest
from blochsimulator.simulator import design_rf_pulse
from blochsimulator.sequence.rf_pulses import (
    design_rf_envelope,
    make_pulseq_rf_events,
    rf_time_bandwidth_product_from_envelope,
    scale_rf_envelope_to_flip,
    slr_time_bandwidth_product,
)


def test_standalone_sequence_scripts_do_not_bypass_the_global_rf_factory():
    scripts = Path(__file__).parents[1] / "sequences" / "scripts"
    forbidden = (
        "pp.make_sinc_pulse",
        "pp.make_gauss_pulse",
        "pp.make_block_pulse",
        "pp.make_arbitrary_rf",
        "pp.sigpy_n_seq",
    )

    for path in scripts.glob("generate_*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden), path.name


def test_sequence_slr_is_designed_from_parameters_without_loading_a_file(
    monkeypatch,
):
    monkeypatch.setattr(
        np,
        "loadtxt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("SLR design must not load a waveform file")
        ),
    )

    broad, duration_s, tbw, pulse_type = design_rf_envelope(
        pulse_type="slr",
        duration_s=2.5e-3,
        raster_s=10e-6,
        time_bandwidth_product=3.5,
        slr_sharpness=1.0,
    )
    sharp, *_ = design_rf_envelope(
        pulse_type="slr",
        duration_s=2.5e-3,
        raster_s=10e-6,
        time_bandwidth_product=3.5,
        slr_sharpness=5.0,
    )

    assert pulse_type == "slr"
    assert duration_s == pytest.approx(2.5e-3)
    # SLR ignores the generic TBW input; sharpness sets the designed TBW.
    assert tbw == pytest.approx(slr_time_bandwidth_product(1)) == 2.0
    assert broad.size == sharp.size == 250
    assert np.allclose(broad, broad[::-1])
    assert np.allclose(sharp, sharp[::-1])
    assert not np.allclose(broad, sharp)
    broad_zero_crossings = np.count_nonzero(
        np.signbit(broad.real[:-1]) != np.signbit(broad.real[1:])
    )
    sharp_zero_crossings = np.count_nonzero(
        np.signbit(sharp.real[:-1]) != np.signbit(sharp.real[1:])
    )
    assert sharp_zero_crossings > broad_zero_crossings


def test_free_mode_slr_uses_the_global_sequence_envelope():
    duration_s = 2.5e-3
    sample_count = 250
    time_bandwidth_product = 3.5
    sharpness = 4.0
    free_b1, _ = design_rf_pulse(
        "slr",
        duration=duration_s,
        flip_angle=30.0,
        time_bw_product=time_bandwidth_product,
        npoints=sample_count,
        slr_sharpness=sharpness,
    )
    shared, *_ = design_rf_envelope(
        pulse_type="slr",
        duration_s=duration_s,
        raster_s=duration_s / sample_count,
        time_bandwidth_product=time_bandwidth_product,
        slr_sharpness=sharpness,
        flip_angle_deg=30.0,
    )
    assert np.allclose(
        free_b1 / np.max(np.abs(free_b1)), shared / np.max(np.abs(shared))
    )


def test_free_mode_sinc_uses_symmetric_global_sequence_envelope():
    duration_s = 1e-3
    sample_count = 200
    shape_parameter = 4.0
    free_b1, free_time = design_rf_pulse(
        "sinc",
        duration=duration_s,
        flip_angle=30.0,
        time_bw_product=shape_parameter,
        npoints=sample_count,
    )
    shared, *_ = design_rf_envelope(
        pulse_type="sinc",
        duration_s=duration_s,
        raster_s=duration_s / sample_count,
        time_bandwidth_product=shape_parameter,
        apodization=0.0,
    )

    normalized_free = free_b1 / np.max(np.abs(free_b1))
    normalized_shared = shared / np.max(np.abs(shared))
    assert np.allclose(normalized_free, normalized_shared)
    assert np.allclose(normalized_free, normalized_free[::-1])
    assert abs(normalized_free[0]) == pytest.approx(abs(normalized_free[-1]), abs=1e-15)
    assert free_time + free_time[::-1] == pytest.approx(duration_s)


def test_slr_sharpness_does_not_reduce_temporal_lobes():
    zero_crossings = []
    for sharpness in range(1, 6):
        envelope, *_ = design_rf_envelope(
            pulse_type="slr",
            duration_s=2.5e-3,
            raster_s=10e-6,
            time_bandwidth_product=3.5,
            slr_sharpness=sharpness,
        )
        zero_crossings.append(
            np.count_nonzero(
                np.signbit(envelope.real[:-1]) != np.signbit(envelope.real[1:])
            )
        )

    # Adjacent sharpness settings can share a sampled zero-crossing count
    # while the transition still narrows. The progression must not reverse,
    # and the full sharpness range must still add lobes.
    assert all(
        current >= previous
        for previous, current in zip(zero_crossings, zero_crossings[1:])
    ), zero_crossings
    assert zero_crossings[-1] > zero_crossings[0], zero_crossings


def test_slr_sharpness_one_is_one_positive_central_lobe():
    envelope, *_ = design_rf_envelope(
        pulse_type="slr",
        duration_s=2.5e-3,
        raster_s=10e-6,
        time_bandwidth_product=4.0,
        slr_sharpness=1,
    )

    assert np.all(envelope.real >= -1e-15)
    assert np.allclose(envelope.imag, 0.0)
    assert (
        np.count_nonzero(
            np.signbit(envelope.real[:-1]) != np.signbit(envelope.real[1:])
        )
        == 0
    )
    assert abs(int(np.argmax(np.abs(envelope))) - envelope.size // 2) <= 2


@pytest.mark.parametrize("sharpness", (0, 1.5, np.nan, True))
def test_slr_sharpness_requires_a_positive_integer(sharpness):
    with pytest.raises(ValueError, match="positive integer"):
        design_rf_envelope(
            pulse_type="slr",
            duration_s=2.5e-3,
            raster_s=10e-6,
            time_bandwidth_product=4.0,
            slr_sharpness=sharpness,
        )


def test_full_sinc_apodization_suppresses_outer_lobes():
    envelope, *_ = design_rf_envelope(
        pulse_type="sinc",
        duration_s=3e-3,
        raster_s=1e-6,
        time_bandwidth_product=6.0,
        apodization=1.0,
    )
    normalized_time = (np.arange(envelope.size) + 0.5) / envelope.size * 6.0 - 3.0
    inner = np.max(np.abs(envelope[(normalized_time >= 1.0) & (normalized_time < 2.0)]))
    outer = np.max(np.abs(envelope[(normalized_time >= 2.0) & (normalized_time < 3.0)]))

    assert outer < inner
    assert abs(envelope[0]) < 1e-6 * np.max(np.abs(envelope))
    assert abs(envelope[-1]) < 1e-6 * np.max(np.abs(envelope))


@pytest.mark.parametrize("sharpness", [1.0, 5.0])
def test_sequence_slr_has_small_edges_and_centered_main_lobe(sharpness):
    envelope, *_ = design_rf_envelope(
        pulse_type="slr",
        duration_s=2.33e-3,
        raster_s=1e-6,
        time_bandwidth_product=2.1,
        slr_sharpness=sharpness,
    )
    magnitude = np.abs(envelope)
    center = envelope.size // 2

    # A true SLR pulse is not tapered to zero; its 1 % ripple design leaves
    # edges of a few percent of the peak, with no spikes.
    assert magnitude[0] < 0.03 * magnitude[center]
    assert magnitude[-1] < 0.03 * magnitude[center]
    assert abs(int(np.argmax(magnitude)) - center) <= 2
    assert magnitude[center] == pytest.approx(np.max(magnitude), rel=1e-4)
    assert np.max(magnitude[: envelope.size // 10]) < 0.35 * magnitude[center]
    assert np.max(magnitude[-envelope.size // 10 :]) < 0.35 * magnitude[center]


def test_sequence_gaussian_envelope_is_symmetric_and_reports_shape_tbw():
    envelope, duration_s, tbw, pulse_type = design_rf_envelope(
        pulse_type="gauss",
        duration_s=3e-3,
        raster_s=10e-6,
        time_bandwidth_product=4.0,
    )

    assert pulse_type == "gaussian"
    assert duration_s == pytest.approx(3e-3)
    assert tbw == pytest.approx(rf_time_bandwidth_product_from_envelope(envelope))
    assert envelope.size == 300
    assert np.all(envelope.real > 0)
    assert np.allclose(envelope.imag, 0.0)
    assert np.allclose(envelope, envelope[::-1])
    assert abs(envelope[0]) < abs(envelope[envelope.size // 2])


def test_adiabatic_half_passage():
    """Test AHP pulse generation."""
    duration = 1e-3
    flip_angle = 90
    time_bw_product = 4.0
    npoints = 100

    b1, time = design_rf_pulse(
        "adiabatic_half",
        duration=duration,
        flip_angle=flip_angle,
        time_bw_product=time_bw_product,
        npoints=npoints,
    )

    # Check shapes
    assert len(b1) == npoints
    assert len(time) == npoints

    # Check AHP characteristics (sweep from off-resonance to resonance)
    # At start (t=0), amplitude should be small (sech(-beta))
    # At end (t=duration), amplitude should be max (sech(0)=1.0)

    # Peak should be at the end
    max_amp = np.max(np.abs(b1))
    end_amp = np.abs(b1[-1])
    start_amp = np.abs(b1[0])

    # Allow small tolerance
    assert np.isclose(
        end_amp, max_amp, rtol=1e-5
    ), f"AHP should end at max amplitude. End: {end_amp}, Max: {max_amp}"
    assert (
        start_amp < max_amp * 0.1
    ), f"AHP should start at low amplitude. Start: {start_amp}, Max: {max_amp}"


def test_adiabatic_full_passage():
    """Test AFP pulse generation."""
    duration = 1e-3
    flip_angle = 180
    time_bw_product = 4.0
    npoints = 100

    b1, time = design_rf_pulse(
        "adiabatic_full",
        duration=duration,
        flip_angle=flip_angle,
        time_bw_product=time_bw_product,
        npoints=npoints,
    )

    assert len(b1) == npoints

    # AFP is symmetric (sech centered)
    # Peak should be in the middle
    mid_idx = npoints // 2
    max_amp = np.max(np.abs(b1))
    mid_amp = np.abs(b1[mid_idx])

    # It might be slightly off due to even npoints, but close
    assert np.isclose(
        mid_amp, max_amp, rtol=0.05
    ), f"AFP peak should be near center. Mid: {mid_amp}, Max: {max_amp}"

    # Start and end should be low
    assert np.abs(b1[0]) < max_amp * 0.1
    assert np.abs(b1[-1]) < max_amp * 0.1


def test_bir4_pulse():
    """Test BIR-4 pulse generation."""
    duration = 4e-3
    flip_angle = 90
    time_bw_product = 4.0
    npoints = 400

    b1, time = design_rf_pulse(
        "bir4",
        duration=duration,
        flip_angle=flip_angle,
        time_bw_product=time_bw_product,
        npoints=npoints,
    )

    assert len(b1) == npoints
    # BIR-4 is a composite pulse, check it's not all zeros
    assert np.max(np.abs(b1)) > 0


def _hard_pulse_profile(samples_hz, raster_s, offsets_hz):
    """Return Mx, My, Mz from rest after hard-pulse rotations per raster cell."""
    magnetization = np.zeros((offsets_hz.size, 3))
    magnetization[:, 2] = 1.0
    for sample in np.asarray(samples_hz, dtype=np.complex128):
        field = np.empty((offsets_hz.size, 3))
        field[:, 0] = 2.0 * np.pi * raster_s * sample.real
        field[:, 1] = 2.0 * np.pi * raster_s * sample.imag
        field[:, 2] = 2.0 * np.pi * raster_s * offsets_hz
        angle = np.linalg.norm(field, axis=1, keepdims=True)
        axis = field / np.maximum(angle, 1e-30)
        cos, sin = np.cos(angle), np.sin(angle)
        dot = np.sum(axis * magnetization, axis=1, keepdims=True)
        magnetization = (
            magnetization * cos
            - np.cross(axis, magnetization) * sin
            + axis * dot * (1.0 - cos)
        )
    return magnetization


def _sample_slr(sample_count, duration_s, sharpness, flip_angle_deg):
    envelope, *_ = design_rf_envelope(
        pulse_type="slr",
        duration_s=duration_s,
        raster_s=duration_s / sample_count,
        slr_sharpness=sharpness,
        flip_angle_deg=flip_angle_deg,
    )
    return envelope


@pytest.mark.parametrize("sharpness", [1, 2, 5])
@pytest.mark.parametrize("flip_angle_deg", [None, 90.0])
def test_slr_shape_is_identical_for_even_and_odd_sample_counts(
    sharpness, flip_angle_deg
):
    duration_s = 2.5e-3
    shapes = {}
    for sample_count in (250, 251, 1000, 1001):
        envelope = _sample_slr(sample_count, duration_s, sharpness, flip_angle_deg)
        normalized = envelope.real / np.max(np.abs(envelope))
        assert np.allclose(envelope.imag, 0.0)
        assert np.allclose(normalized, normalized[::-1], atol=1e-9)
        # No Nyquist zigzag: neighbouring second differences may not
        # alternate in sign around a smooth pulse. Their size stays at the
        # level a smooth curve has on this raster.
        curvature = np.diff(normalized, 2)
        assert np.max(np.abs(curvature)) < 40.0 / sample_count**2 * (sharpness + 1) ** 2
        shapes[sample_count] = normalized

    # Even and odd sample counts sample the same continuous pulse.
    for coarse, fine in ((250, 251), (1000, 1001), (250, 1001)):
        positions = (np.arange(coarse) + 0.5) / coarse
        fine_positions = (np.arange(fine) + 0.5) / fine
        assert np.allclose(
            shapes[coarse],
            np.interp(positions, fine_positions, shapes[fine]),
            atol=2e-3,
        )


@pytest.mark.parametrize("sample_count", [400, 401])
@pytest.mark.parametrize("flip_angle_deg", [30.0, 90.0])
def test_slr_pulse_excites_its_designed_time_bandwidth_product(
    sample_count, flip_angle_deg
):
    duration_s = 2e-3
    sharpness = 3
    raster_s = duration_s / sample_count
    envelope, _, tbw, _ = design_rf_envelope(
        pulse_type="slr",
        duration_s=duration_s,
        raster_s=raster_s,
        slr_sharpness=sharpness,
        flip_angle_deg=flip_angle_deg,
    )
    assert tbw == slr_time_bandwidth_product(sharpness) == 4.0
    samples_hz = scale_rf_envelope_to_flip(
        envelope, flip_angle_deg=flip_angle_deg, raster_s=raster_s
    )
    bandwidth_hz = tbw / duration_s
    offsets_hz = np.linspace(0.0, 1.5 * bandwidth_hz, 301)
    profile = _hard_pulse_profile(samples_hz, raster_s, offsets_hz)
    transverse = np.hypot(profile[:, 0], profile[:, 1])
    target = np.sin(np.deg2rad(flip_angle_deg))

    passband = offsets_hz <= 0.25 * bandwidth_hz
    stopband = offsets_hz >= 0.85 * bandwidth_hz
    assert transverse[0] == pytest.approx(target, abs=1e-3)
    assert np.all(np.abs(transverse[passband] - target) < 0.03)
    assert np.max(transverse[stopband]) < 0.02
    half_width = offsets_hz[np.argmax(transverse < 0.5 * target)]
    assert 2.0 * half_width == pytest.approx(bandwidth_hz, rel=0.08)


def test_large_tip_slr_differs_from_small_tip_and_keeps_its_area():
    small = _sample_slr(500, 3e-3, 4, None).real
    large = _sample_slr(500, 3e-3, 4, 90.0).real
    small = small / np.max(small)
    large = large / np.max(large)

    assert not np.allclose(small, large, atol=1e-2)
    samples_hz = scale_rf_envelope_to_flip(
        _sample_slr(500, 3e-3, 4, 90.0), flip_angle_deg=90.0, raster_s=6e-6
    )
    on_resonance = _hard_pulse_profile(samples_hz, 6e-6, np.zeros(1))[0]
    assert on_resonance[2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("sharpness", [0, 1.5])
def test_slr_rejects_invalid_sharpness_before_design(sharpness):
    with pytest.raises(ValueError, match="positive integer"):
        slr_time_bandwidth_product(sharpness)


def test_slr_requires_enough_samples_for_its_time_bandwidth_product():
    with pytest.raises(ValueError, match="too high"):
        _sample_slr(10, 1e-3, 5, None)


def test_pulseq_slr_events_use_flip_specific_shapes():
    pp = pytest.importorskip("pypulseq")
    system = pp.Opts(rf_raster_time=1e-6, grad_raster_time=10e-6)
    events, duration_s, tbw, pulse_type = make_pulseq_rf_events(
        pp,
        system,
        flip_angles_deg=(10.0, 90.0),
        pulse_type="slr",
        duration_s=2e-3,
        slr_sharpness=3,
    )
    assert pulse_type == "slr"
    assert tbw == 4.0
    small, large = (np.real(event.signal) for event in events)
    for event, angle in zip(events, (10.0, 90.0)):
        area_deg = np.rad2deg(
            2.0 * np.pi * np.sum(event.signal).real * system.rf_raster_time
        )
        assert area_deg == pytest.approx(angle, rel=1e-6)
    assert not np.allclose(small / small.max(), large / large.max(), atol=1e-2)
    assert np.allclose(
        large / large.max(),
        _sample_slr(2000, 2e-3, 3, 90.0).real
        / np.max(_sample_slr(2000, 2e-3, 3, 90.0).real),
    )
