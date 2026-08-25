from pathlib import Path

import numpy as np
import pytest
from blochsimulator.simulator import design_rf_pulse
from blochsimulator.sequence.rf_pulses import (
    design_rf_envelope,
    rf_time_bandwidth_product_from_envelope,
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
    assert tbw == pytest.approx(rf_time_bandwidth_product_from_envelope(broad))
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
    )
    assert np.allclose(
        free_b1 / np.max(np.abs(free_b1)), shared / np.max(np.abs(shared))
    )


def test_slr_sharpness_adds_temporal_lobes_monotonically():
    zero_crossings = []
    for sharpness in range(1, 6):
        envelope, *_ = design_rf_envelope(
            pulse_type="slr",
            duration_s=2.5e-3,
            raster_s=10e-6,
            time_bandwidth_product=3.5,
            slr_sharpness=float(sharpness),
        )
        zero_crossings.append(
            np.count_nonzero(
                np.signbit(envelope.real[:-1]) != np.signbit(envelope.real[1:])
            )
        )

    assert all(
        current > previous
        for previous, current in zip(zero_crossings, zero_crossings[1:])
    )


@pytest.mark.parametrize("sharpness", [1.0, 5.0])
def test_sequence_slr_has_zero_edges_and_centered_main_lobe(sharpness):
    envelope, *_ = design_rf_envelope(
        pulse_type="slr",
        duration_s=2.33e-3,
        raster_s=1e-6,
        time_bandwidth_product=2.1,
        slr_sharpness=sharpness,
    )
    magnitude = np.abs(envelope)
    center = envelope.size // 2

    assert magnitude[0] == pytest.approx(0.0, abs=1e-15)
    assert magnitude[-1] == pytest.approx(0.0, abs=1e-15)
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
