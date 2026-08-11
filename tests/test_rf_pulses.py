import numpy as np
import pytest
from blochsimulator.simulator import design_rf_pulse
from blochsimulator.sequence.rf_pulses import design_rf_envelope


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
    assert tbw == pytest.approx(3.5)
    assert broad.size == sharp.size == 250
    assert np.allclose(broad, broad[::-1])
    assert np.allclose(sharp, sharp[::-1])
    assert not np.allclose(broad, sharp)


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


def test_sequence_gaussian_envelope_is_symmetric_and_uses_requested_tbw():
    envelope, duration_s, tbw, pulse_type = design_rf_envelope(
        pulse_type="gauss",
        duration_s=3e-3,
        raster_s=10e-6,
        time_bandwidth_product=4.0,
    )

    assert pulse_type == "gaussian"
    assert duration_s == pytest.approx(3e-3)
    assert tbw == pytest.approx(4.0)
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
