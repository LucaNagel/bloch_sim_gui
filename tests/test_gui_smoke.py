import pytest
import sys
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QAction, QApplication
from blochsimulator.memory import GIB, MemoryPolicy
from blochsimulator.ui.dialogs import MemorySettingsDialog
from blochsimulator.ui.main_window import BlochSimulatorGUI


def test_gui_instantiation():
    """Smoke test to ensure the main window can be instantiated without crashing."""
    # Ensure a QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        window = BlochSimulatorGUI()
        assert window is not None
        assert window.windowTitle() == "Bloch Equation Simulator"

        # Check if tutorial manager is initialized
        assert hasattr(window, "tutorial_manager")
        assert window.tutorial_manager is not None
        assert window.findChild(QAction, "action_memory_settings") is not None

    finally:
        # We don't want to show the window or start the event loop
        pass


def test_memory_settings_dialog_returns_selected_policy():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = MemorySettingsDialog(MemoryPolicy())
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("custom_reserve"))
    dialog.reserve_spin.setValue(3.5)

    policy = dialog.get_policy()
    assert policy.mode == "custom_reserve"
    assert policy.reserve_bytes == int(3.5 * GIB)
    assert dialog.reserve_spin.isEnabled()
    assert not dialog.limit_spin.isEnabled()


def test_memory_limit_uses_warning_instead_of_generic_error():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window._show_memory_limit_warning = MagicMock()
    message = "Memory limit exceeded: test details"

    with patch("blochsimulator.ui.main_window.QMessageBox.critical") as critical:
        window.on_simulation_error(message)

    window._show_memory_limit_warning.assert_called_once_with(message)
    critical.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
