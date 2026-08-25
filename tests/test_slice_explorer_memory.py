"""Regression tests for Slice Explorer memory-limit handling."""

from unittest.mock import MagicMock

from blochsimulator.memory import SimulationMemoryError
from blochsimulator.slice_explorer import SliceSelectionExplorer


def test_profile_memory_error_does_not_escape_run():
    explorer = MagicMock()
    explorer._resolve_pulse.return_value = (
        "b1",
        "time",
        90.0,
        0.002,
        4.0,
        1e-5,
        "Sinc",
    )
    error = SimulationMemoryError("Memory limit exceeded: test rejection")
    explorer._simulate_profile.side_effect = error

    SliceSelectionExplorer.run_simulation(explorer)

    explorer._show_simulation_error.assert_called_once_with(error)


def test_profile_memory_error_clears_stale_results_and_plots():
    explorer = MagicMock()
    explorer.last_b1 = "stale b1"
    explorer.last_gradients = "stale gradients"
    explorer.last_time = "stale time"
    error = SimulationMemoryError("Memory limit exceeded: test rejection")

    SliceSelectionExplorer._show_simulation_error(explorer, error)

    assert explorer.last_b1 is None
    assert explorer.last_gradients is None
    assert explorer.last_time is None
    explorer.plot_rf.clear.assert_called_once_with()
    explorer.plot_profile.clear.assert_called_once_with()
    explorer.pulse_status.setStyleSheet.assert_called_once_with("color: #b00020;")
    explorer.pulse_status.setText.assert_called_once_with(str(error))
