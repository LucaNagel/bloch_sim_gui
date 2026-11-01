import numpy as np
import pytest

from blochsimulator.sequence import GradientEvent
from blochsimulator.sequence.gradient_load import GradientLoadEstimate

RASTER_S = 10e-6


def _trapezoid(axis, start_s, ramp_cells, flat_cells, amplitude):
    """Return a Pulseq-like trapezoid sampled at raster-cell centres."""
    ramp = (np.arange(ramp_cells) + 0.5) / ramp_cells * amplitude
    samples = np.concatenate((ramp, np.full(flat_cells, amplitude), ramp[::-1]))
    return GradientEvent(axis, start_s, samples, RASTER_S)


def test_trapezoid_load_matches_its_amplitude_and_slew():
    amplitude = 4000.0
    ramp_cells = 10
    event = _trapezoid("x", 1e-3, ramp_cells, 50, amplitude)
    slew = amplitude / (ramp_cells * RASTER_S)
    estimate = GradientLoadEstimate(
        (event,),
        duration_s=2e-3,
        max_gradient_hz_per_m=8000.0,
        max_slew_hz_per_m_s=2.0 * slew,
    )

    assert estimate.peak_amplitude() == pytest.approx((0.5, "x"))
    assert estimate.peak_slew() == pytest.approx((0.5, "x"))
    flat_s = 50 * RASTER_S
    ramp_area = np.sum(((np.arange(10) + 0.5) / 10) ** 2) * 2 * RASTER_S
    expected_rms = 0.5 * np.sqrt((flat_s + ramp_area) / 2e-3)
    assert estimate.rms_amplitude()[0] == pytest.approx(expected_rms)
    assert estimate.rms_amplitude()[1] == "x"


def test_binned_load_marks_only_the_bins_with_gradients():
    event = _trapezoid("y", 1e-3, 10, 30, 2000.0)
    estimate = GradientLoadEstimate(
        (event,),
        duration_s=4e-3,
        max_gradient_hz_per_m=4000.0,
        max_slew_hz_per_m_s=1e8,
    )

    edges, amplitude, slew = estimate.binned(0.0, 4e-3, 40)

    assert edges.size == 41
    active = amplitude > 0
    assert np.all(edges[:-1][active] >= 1e-3 - 1e-12)
    assert np.all(edges[1:][active] <= event.end_s + 1e-12)
    assert np.max(amplitude) == pytest.approx(0.5)
    assert np.max(slew) == pytest.approx(estimate.peak_slew()[0])
    assert np.all(slew[edges[1:] < 1e-3] == 0)


def test_coarse_bins_keep_short_event_peaks():
    events = tuple(
        _trapezoid("z", start_s, 2, 2, 1000.0 * (index + 1))
        for index, start_s in enumerate(np.arange(10) * 1e-3)
    )
    estimate = GradientLoadEstimate(
        events,
        duration_s=10e-3,
        max_gradient_hz_per_m=10000.0,
        max_slew_hz_per_m_s=1e9,
    )

    _edges, amplitude, _slew = estimate.binned(0.0, 10e-3, 3)

    assert np.max(amplitude) == pytest.approx(1.0)
    assert amplitude[0] == pytest.approx(0.4)


def test_abutting_events_use_the_joint_step_as_slew():
    first = GradientEvent("x", 0.0, np.full(5, 1000.0), RASTER_S)
    second = GradientEvent("x", first.end_s, np.full(5, 1100.0), RASTER_S)
    estimate = GradientLoadEstimate(
        (second, first),
        duration_s=1e-3,
        max_gradient_hz_per_m=2000.0,
        max_slew_hz_per_m_s=1.0,
    )

    # Only the outer edges ramp from zero; the joint steps by 100 Hz/m.
    assert estimate.peak_slew()[0] == pytest.approx(2 * 1100.0 / RASTER_S)
    _edges, _amplitude, slew = estimate.binned(0.4e-4, 0.6e-4, 1)
    assert slew[0] == pytest.approx(100.0 / RASTER_S)


def test_gradient_load_rejects_invalid_limits():
    with pytest.raises(ValueError, match="max_gradient"):
        GradientLoadEstimate(
            (), duration_s=1.0, max_gradient_hz_per_m=0.0, max_slew_hz_per_m_s=1.0
        )
