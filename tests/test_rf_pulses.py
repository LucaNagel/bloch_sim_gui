import numpy as np
import pytest
from blochsimulator.simulator import design_rf_pulse


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
