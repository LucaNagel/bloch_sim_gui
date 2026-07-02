import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

# Ensure we import from src to test the local changes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_designer import SequenceDesigner
from blochsimulator.simulator import design_rf_pulse


class MockSequenceDesigner:
    """Mock for SequenceDesigner to provide widget values."""

    def __init__(self):
        self.tr_spin = MagicMock()
        self.tr_spin.value.return_value = 10.0  # 10 ms

        self.ssfp_repeats = MagicMock()
        self.ssfp_repeats.value.return_value = 5

        # Mock parent_gui structure for shared RF designer state
        self.parent_gui = MagicMock()
        self.parent_gui.rf_designer = MagicMock()
        self.parent_gui.rf_designer.duration = MagicMock()
        self.parent_gui.rf_designer.duration.value.return_value = 1.0  # 1.0 ms
        self.parent_gui.rf_designer.flip_angle = MagicMock()
        self.parent_gui.rf_designer.flip_angle.value.return_value = 90.0
        self.parent_gui.rf_designer.freq_offset = MagicMock()
        self.parent_gui.rf_designer.freq_offset.value.return_value = 0.0

        self.ssfp_start_flip = MagicMock()
        self.ssfp_start_flip.value.return_value = 45.0

        self.ssfp_start_phase = MagicMock()
        self.ssfp_start_phase.value.return_value = 180.0

        self.ssfp_start_tr = MagicMock()
        self.ssfp_start_tr.value.return_value = 0.0

        self.ssfp_use_ratios = MagicMock()
        self.ssfp_use_ratios.isChecked.return_value = False

        self.ssfp_tr_ratio = MagicMock()
        self.ssfp_tr_ratio.value.return_value = 0.5

        self.ssfp_flip_ratio = MagicMock()
        self.ssfp_flip_ratio.value.return_value = 0.5

        self.ssfp_alternate_phase = MagicMock()
        self.ssfp_alternate_phase.isChecked.return_value = True


def test_ssfp_custom_pulse_duration():
    """
    Test that custom pulse duration is calculated correctly in SSFP sequence.

    This ensures that a custom pulse with N points and dwell time dt
    results in a duration of N * dt, avoiding off-by-one errors.
    """
    mock_self = MockSequenceDesigner()

    dt = 1e-5  # 0.01 ms
    # Create a custom pulse of 1.0 ms duration (100 points * 0.01 ms)
    n_pts = 100
    t_pulse = np.linspace(0, 1.0e-3, n_pts, endpoint=False)
    b1_pulse = np.ones(n_pts, dtype=complex)
    custom_pulse = (b1_pulse, t_pulse)

    # Verify input pulse properties
    assert len(t_pulse) == n_pts
    # np.diff(t_pulse) should be constant dt
    assert np.allclose(np.diff(t_pulse), dt)

    # Call _build_ssfp using the unbound method technique
    # We pass 'mock_self' as the instance
    b1, gradients, time = SequenceDesigner._build_ssfp(mock_self, custom_pulse, dt)

    # Check the number of non-zero points in the first pulse period
    # The first pulse should be placed at start_delay=0
    # Its duration should be exactly n_pts

    # Extract the first segment corresponding to the pulse
    first_pulse_segment = b1[: n_pts + 10]  # Take a bit more to check boundaries

    non_zero_count = np.sum(np.abs(first_pulse_segment) > 0)

    # If the fix works, we expect 100 non-zero points.
    # If the bug (N-1)*dt was present, the calculated duration would be 0.99 ms,
    # which at dt=0.01ms results in 99 points.

    assert (
        non_zero_count == n_pts
    ), f"Expected {n_pts} points for custom pulse, got {non_zero_count}. Duration calculation might be off."


def test_ssfp_block_pulse_duration():
    """
    Test that standard block pulse duration is respected.
    """
    mock_self = MockSequenceDesigner()
    # Set standard block pulse duration to 1.0 ms via RF designer
    mock_self.parent_gui.rf_designer.duration.value.return_value = 1.0

    dt = 1e-5  # 0.01 ms

    # Call with custom_pulse=None
    b1, gradients, time = SequenceDesigner._build_ssfp(mock_self, None, dt)

    # Expected points = 1.0 ms / 0.01 ms = 100
    n_expected = 100

    first_pulse_segment = b1[: n_expected + 10]
    non_zero_count = np.sum(np.abs(first_pulse_segment) > 0)

    assert (
        non_zero_count == n_expected
    ), f"Expected {n_expected} points for block pulse, got {non_zero_count}."


def test_set_custom_pulse_does_not_reset_ssfp_prep_settings():
    """RF waveform updates must not overwrite the configured first SSFP pulse."""

    class Dummy:
        pass

    dummy = Dummy()
    dummy.current_role = "Pulse"
    dummy.pulse_waveforms = {}
    dummy.custom_pulse = None
    dummy.parent_gui = MagicMock()
    dummy.parent_gui.rf_designer = MagicMock()
    dummy.ssfp_start_flip = MagicMock()
    dummy.ssfp_flip_ratio = MagicMock()
    dummy.ssfp_start_tr = MagicMock()
    dummy.ssfp_tr_ratio = MagicMock()
    dummy.update_diagram = MagicMock()

    pulse = (np.ones(10, dtype=complex), np.arange(10) * 1e-5)

    SequenceDesigner.set_custom_pulse(dummy, pulse)

    assert dummy.pulse_waveforms["Pulse"] == pulse
    assert dummy.custom_pulse == pulse
    dummy.ssfp_start_flip.setValue.assert_not_called()
    dummy.ssfp_flip_ratio.setValue.assert_not_called()
    dummy.ssfp_start_tr.setValue.assert_not_called()
    dummy.ssfp_tr_ratio.setValue.assert_not_called()
    dummy.update_diagram.assert_called_once_with()


def test_auto_update_ssfp_amplitude_preserves_prep_settings():
    """RF parameter changes should refresh the preview without mutating SSFP prep controls."""

    class Dummy:
        pass

    dummy = Dummy()
    dummy.sequence_designer = MagicMock()
    dummy.sequence_designer.sequence_type.currentText.return_value = "SSFP (Loop)"
    dummy.sequence_designer.ssfp_start_flip = MagicMock()
    dummy.sequence_designer.update_diagram = MagicMock()
    dummy.rf_designer = MagicMock()
    pulse = (np.ones(8, dtype=complex), np.arange(8) * 1e-5)
    dummy.rf_designer.get_pulse.return_value = pulse

    BlochSimulatorGUI._auto_update_ssfp_amplitude(dummy)

    dummy.sequence_designer.ssfp_start_flip.setValue.assert_not_called()
    dummy.sequence_designer.update_diagram.assert_called_once_with(pulse)


def test_ssfp_rf_carrier_phase_is_continuous_across_repetitions():
    """A non-zero RF carrier must accumulate phase between SSFP pulses."""
    mock_self = MockSequenceDesigner()
    mock_self.tr_spin.value.return_value = 5.0
    mock_self.ssfp_repeats.value.return_value = 5
    mock_self.ssfp_start_flip.value.return_value = 90.0
    mock_self.ssfp_start_phase.value.return_value = 0.0
    mock_self.ssfp_alternate_phase.isChecked.return_value = False
    mock_self.parent_gui.rf_designer.freq_offset.value.return_value = 37.0
    mock_self._rf_frequency_offset = lambda: 37.0

    dt = 1e-5
    pulse, pulse_time = design_rf_pulse(
        "rect", duration=1e-3, flip_angle=90.0, npoints=100
    )
    baseband = SequenceDesigner._build_ssfp(mock_self, (pulse, pulse_time), dt)
    b1, _, _ = SequenceDesigner._apply_current_rf_carrier(mock_self, baseband)

    starts = np.rint(np.arange(5) * 5e-3 / dt).astype(int)
    measured = np.unwrap(np.angle(b1[starts]))
    expected = 2 * np.pi * 37.0 * np.arange(5) * 5e-3
    assert np.allclose(measured, expected, atol=1e-12)
