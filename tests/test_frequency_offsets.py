import numpy as np
from unittest.mock import MagicMock

from blochsimulator import (
    BlochSimulator,
    TissueParameters,
    apply_rf_carrier,
    design_rf_pulse,
)
from blochsimulator.ui.parameter_sweep import ParameterSweepWidget
from blochsimulator.ui.main_window import BlochSimulatorGUI


def test_rf_carrier_reports_effective_detuning():
    sim = BlochSimulator(use_parallel=False)
    time = np.arange(8) * 1e-5
    result = sim.simulate(
        (np.zeros(8, dtype=complex), np.zeros((8, 3)), time),
        TissueParameters("test", 1.0, 0.1),
        frequencies=np.array([20.0, 30.0, 40.0]),
        rf_carrier_offset=30.0,
    )
    assert np.array_equal(result["effective_frequencies"], [-10.0, 0.0, 10.0])
    assert result["rf_carrier_offset"] == 30.0


def test_absolute_carrier_modulation():
    time = np.array([0.0, 5e-3, 10e-3])
    modulated = apply_rf_carrier(np.ones(3, dtype=complex), time, 37.0)
    assert np.allclose(np.unwrap(np.angle(modulated)), 2 * np.pi * 37.0 * time)


def test_rf_carrier_translates_frequency_response():
    """A global carrier must translate, not distort, an SSFP-like response."""
    dt = 1e-5
    tr_points = 500
    repeats = 20
    time = np.arange(tr_points * repeats) * dt
    baseband = np.zeros(time.size, dtype=complex)
    pulse, _ = design_rf_pulse("rect", 1e-3, 30.0, npoints=100)
    for repetition in range(repeats):
        start = repetition * tr_points
        baseband[start : start + pulse.size] = pulse

    offset = 37.0
    shifted = apply_rf_carrier(baseband, time, offset)
    gradients = np.zeros((time.size, 3))
    frequencies = np.linspace(-100.0, 100.0, 401)
    simulator = BlochSimulator(use_parallel=False)
    tissue = TissueParameters("test", 1.0, 0.1)

    reference = simulator.simulate(
        (baseband, gradients, time), tissue, frequencies=frequencies
    )["signal"][0]
    translated = simulator.simulate(
        (shifted, gradients, time),
        tissue,
        frequencies=frequencies,
        rf_carrier_offset=offset,
    )["signal"][0]
    expected = np.interp(
        frequencies - offset,
        frequencies,
        np.abs(reference),
        left=np.nan,
        right=np.nan,
    )
    assert np.nanmax(np.abs(np.abs(translated) - expected)) < 1e-5


def test_frequency_sweeps_target_distinct_controls():
    dummy = MagicMock()
    dummy.freq_center = MagicMock()
    dummy.rf_designer = MagicMock()

    holder = MagicMock()
    holder.parent_gui = dummy
    ParameterSweepWidget._apply_parameter_value(
        holder, "Spin Offset Center (Hz)", 125.0
    )
    ParameterSweepWidget._apply_parameter_value(holder, "RF Carrier Offset (Hz)", -75.0)

    dummy.freq_center.setValue.assert_called_once_with(125.0)
    dummy.rf_designer.freq_offset.setValue.assert_called_once_with(-75.0)


def test_spin_frequency_axis_has_independent_center_and_span():
    holder = MagicMock()
    holder.freq_spin.value.return_value = 5
    holder.freq_center.value.return_value = 120.0
    holder.freq_range.value.return_value = 40.0
    axis = BlochSimulatorGUI._build_frequency_axis(holder)
    assert np.array_equal(axis, [100.0, 110.0, 120.0, 130.0, 140.0])
