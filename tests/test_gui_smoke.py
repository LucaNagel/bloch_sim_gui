import pytest
import sys
from PyQt5.QtWidgets import QApplication
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

    finally:
        # We don't want to show the window or start the event loop
        pass


if __name__ == "__main__":
    pytest.main([__file__])
