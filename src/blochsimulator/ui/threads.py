from PyQt5.QtCore import QThread, pyqtSignal


class SimulationThread(QThread):
    """Thread for running simulations without blocking the GUI."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        simulator,
        sequence,
        tissue,
        positions,
        frequencies,
        mode,
        dt=1e-5,
        m_init=None,
    ):
        super().__init__()
        self.simulator = simulator
        self.sequence = sequence
        self.tissue = tissue
        self.positions = positions
        self.frequencies = frequencies
        self.mode = mode
        self.dt = dt
        self.m_init = m_init
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        """Run the simulation."""
        try:
            if self._cancel_requested:
                self.cancelled.emit()
                return
            result = self.simulator.simulate(
                self.sequence,
                self.tissue,
                self.positions,
                self.frequencies,
                initial_magnetization=self.m_init,
                mode=self.mode,
                dt=self.dt,
            )
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
