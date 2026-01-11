import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

# Ensure we import from src to test the local changes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from blochsimulator.gui import SequenceDesigner


class MockSequenceDesigner:
    """Mock for SequenceDesigner to provide widget values."""

    def __init__(self):
        self.tr_spin = MagicMock()
        self.tr_spin.value.return_value = 10.0  # 10 ms

        self.ssfp_repeats = MagicMock()
        self.ssfp_repeats.value.return_value = 5

        self.ssfp_amp = MagicMock()
        self.ssfp_amp.value.return_value = 0.05

        self.ssfp_phase = MagicMock()
        self.ssfp_phase.value.return_value = 0.0

        self.ssfp_dur = MagicMock()
        self.ssfp_dur.value.return_value = 1.0  # 1.0 ms

        self.ssfp_start_amp = MagicMock()
        self.ssfp_start_amp.value.return_value = 0.025

        self.ssfp_start_phase = MagicMock()
        self.ssfp_start_phase.value.return_value = 180.0

        self.ssfp_start_delay = MagicMock()
        self.ssfp_start_delay.value.return_value = 0.0

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
    # Set standard block pulse duration to 1.0 ms
    mock_self.ssfp_dur.value.return_value = 1.0

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
