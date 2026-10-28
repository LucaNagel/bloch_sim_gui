import numpy as np
import pytest

from blochsimulator.sequence.kspace_trajectory import (
    GradientImperfections,
    calculate_kspace,
    classify_readout,
    classify_rf_events,
)
from blochsimulator.sequence.model import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
)

RASTER = 10e-6


def _hard_pulse(start_s, flip_deg, duration_s=100e-6):
    samples = int(round(duration_s / 1e-6))
    amplitude = flip_deg / 360.0 / duration_s
    return RFEvent(start_s, np.full(samples, amplitude), 1e-6)


def _gradient(axis, start_s, amplitude_hz_per_m, duration_s):
    samples = int(round(duration_s / RASTER))
    return GradientEvent(axis, start_s, np.full(samples, amplitude_hz_per_m), RASTER)


def _program(*events, duration_s=0.05, metadata=None):
    return SequenceProgram(
        events=tuple(events), duration_s=duration_s, metadata=metadata or {}
    )


def test_excitation_resets_k_to_zero_at_the_pulse_centre():
    program = _program(
        # A gradient before the excitation must not leak into the readout.
        _gradient("x", 0.0, 5000.0, 1e-3),
        _hard_pulse(2e-3, 90.0),
        _gradient("x", 3e-3, 2000.0, 1e-3),
        ADCEvent(4.5e-3, 4, 100e-6),
    )
    trajectory = calculate_kspace(program)
    assert [pulse.role for pulse in trajectory.rf_pulses] == ["excitation"]
    assert trajectory.rf_pulses[0].flip_angle_deg == pytest.approx(90.0)
    assert trajectory.rf_pulses[0].center_s == pytest.approx(2.05e-3)
    assert trajectory.adc_k_cyc_per_m[:, 0] == pytest.approx(np.full(4, 2.0))
    assert trajectory.adc_time_since_excitation_s[0] == pytest.approx(2.45e-3)
    # Before the first excitation there is no transverse magnetization.
    assert np.all(np.isnan(trajectory.k_at([1.0e-3])))


def test_refocusing_pulse_negates_k_and_inversion_leaves_it():
    program = _program(
        _hard_pulse(0.0, 90.0),
        _gradient("x", 1e-3, 1000.0, 1e-3),
        _hard_pulse(3e-3, 180.0),
        _gradient("x", 4e-3, 1000.0, 1e-3),
        ADCEvent(5.5e-3, 2, 100e-6),
    )
    trajectory = calculate_kspace(program)
    assert [pulse.role for pulse in trajectory.rf_pulses] == [
        "excitation",
        "refocusing",
    ]
    # k = +1 before the 180, -1 right after, back to 0 at the spin echo.
    assert trajectory.k_at([2.5e-3])[0, 0] == pytest.approx(1.0)
    assert trajectory.k_at([3.5e-3])[0, 0] == pytest.approx(-1.0)
    assert trajectory.adc_k_cyc_per_m[:, 0] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert trajectory.conjugation_at([2.5e-3, 5.5e-3]) == pytest.approx([0.0, 1.0])

    inversion = _program(
        _hard_pulse(0.0, 180.0),
        _hard_pulse(5e-3, 90.0),
        _gradient("x", 6e-3, 1000.0, 1e-3),
        ADCEvent(7.5e-3, 1, 100e-6),
    )
    roles = [pulse.role for pulse in classify_rf_events(inversion)]
    assert roles == ["inversion", "excitation"]
    assert calculate_kspace(inversion).adc_k_cyc_per_m[0, 0] == pytest.approx(1.0)


def test_pulseq_use_overrides_the_flip_angle_classification():
    program = _program(
        _hard_pulse(0.0, 90.0),
        ADCEvent(1e-3, 1, 100e-6),
        metadata={
            "rf_timing": [
                {"start_s": 0.0, "center_s": 0.0, "use": "saturation"},
            ]
        },
    )
    trajectory = calculate_kspace(program)
    assert trajectory.rf_pulses[0].role == "saturation"
    assert np.all(np.isnan(trajectory.adc_k_cyc_per_m))


def test_gradient_delay_and_scale_change_the_trajectory():
    program = _program(
        _hard_pulse(0.0, 90.0),
        _gradient("y", 1e-3, 1000.0, 1e-3),
        ADCEvent(1.5e-3, 1, 10e-6),
    )
    ideal = calculate_kspace(program).adc_k_cyc_per_m[0, 1]
    delayed = calculate_kspace(
        program, imperfections=GradientImperfections(delay_s=(0.0, 100e-6, 0.0))
    ).adc_k_cyc_per_m[0, 1]
    scaled = calculate_kspace(
        program, imperfections={"scale": (1.0, 0.9, 1.0)}
    ).adc_k_cyc_per_m[0, 1]
    assert ideal == pytest.approx(0.5)
    assert delayed == pytest.approx(0.4)
    assert scaled == pytest.approx(0.45)


def test_eddy_currents_bend_the_ramp_but_keep_the_final_moment():
    program = _program(
        _hard_pulse(0.0, 90.0),
        _gradient("x", 1e-3, 1000.0, 2e-3),
        ADCEvent(1.2e-3, 1, 10e-6),
        ADCEvent(20e-3, 1, 10e-6),
    )
    eddy = GradientImperfections(
        eddy_amplitudes=(0.05,), eddy_time_constants_s=(0.5e-3,)
    )
    ideal = calculate_kspace(program).adc_k_cyc_per_m[:, 0]
    actual = calculate_kspace(program, imperfections=eddy).adc_k_cyc_per_m[:, 0]
    # The response to the up-ramp lowers the moment early in the lobe ...
    assert actual[0] < ideal[0]
    # ... and the down-ramp response cancels it once both have decayed.
    assert actual[1] == pytest.approx(ideal[1], rel=1e-6)
    with pytest.raises(ValueError, match="pair up"):
        GradientImperfections(eddy_amplitudes=(0.1,))


def test_readout_classification_recognises_line_radial_and_spiral():
    t = np.linspace(0.0, 1.0, 400)
    line = np.column_stack((np.linspace(-1, 1, 400), np.zeros(400), np.zeros(400)))
    spoke = np.column_stack((t, 0.5 * t, np.zeros(400)))
    angle = 2 * np.pi * 8 * t
    spiral = np.column_stack((t * np.cos(angle), t * np.sin(angle), np.zeros(400)))
    assert classify_readout(line) == "cartesian_line"
    assert classify_readout(spoke) == "radial"
    assert classify_readout(spiral) == "spiral"
    assert classify_readout(np.zeros((10, 3))) == "point"


def test_timeline_resolves_the_reset_at_each_excitation():
    program = _program(
        _hard_pulse(0.0, 30.0),
        _gradient("x", 1e-3, 1000.0, 1e-3),
        _hard_pulse(10e-3, 30.0),
        _gradient("x", 11e-3, 1000.0, 1e-3),
        ADCEvent(12e-3, 1, 10e-6),
        duration_s=20e-3,
    )
    trajectory = calculate_kspace(program)
    times, k = trajectory.timeline()
    before = np.searchsorted(times, 10.05e-3) - 1
    assert k[before, 0] == pytest.approx(1.0)
    assert k[before + 1, 0] == pytest.approx(0.0)
    assert trajectory.event_summary(0)["k_end_cyc_per_m"][0] == pytest.approx(1.0)
