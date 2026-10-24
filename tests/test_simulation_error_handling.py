from unittest.mock import MagicMock

from blochsimulator.ui.main_window import BlochSimulatorGUI


def test_cancel_without_simulation_thread_does_not_raise():
    window = MagicMock()
    window.simulation_thread = None

    BlochSimulatorGUI.cancel_simulation(window)

    window.status_cancel_button.setEnabled.assert_called_once_with(False)


def test_cancel_requests_running_thread():
    window = MagicMock()
    window.simulation_thread.isRunning.return_value = True

    BlochSimulatorGUI.cancel_simulation(window)

    window.simulation_thread.request_cancel.assert_called_once_with()


def test_sequence_compile_value_error_is_reported_not_raised():
    window = MagicMock()
    message = "SSFP startup spacing is shorter than the RF pulse"
    window.sequence_designer.compile_sequence.side_effect = ValueError(message)
    window._confirm_adiabatic_free_mode_run.return_value = True
    window.simulation_thread = None
    window.time_step_spin.value.return_value = 1.0

    BlochSimulatorGUI.run_simulation(window)

    window.on_simulation_error.assert_called_once_with(message)
    assert window.simulation_thread is None
