import sys

from PyQt5.QtWidgets import QApplication

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def test_sequence_workspace_is_lazy_and_initializes_on_selection():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    assert window.sequence_simulation_widget is None
    window.tab_widget.setCurrentIndex(window.sequence_simulation_tab_index)
    app.processEvents()
    assert isinstance(window.sequence_simulation_widget, SequenceSimulationWidget)
    assert window.sequence_simulation_widget.program.source == "internal-fid"
    window.close()
    window.deleteLater()
    app.processEvents()
