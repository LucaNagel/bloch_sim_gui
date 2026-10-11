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
        rf_carrier_offset=0.0,
        output_step_s=None,
        adaptive_rf=False,
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
        self.rf_carrier_offset = float(rf_carrier_offset)
        self.output_step_s = output_step_s
        self.adaptive_rf = adaptive_rf
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
                rf_carrier_offset=self.rf_carrier_offset,
                output_step_s=self.output_step_s,
                adaptive_rf=self.adaptive_rf,
                cancel_check=lambda: self._cancel_requested,
                progress_callback=self.progress.emit,
            )
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            from ..adaptive import SimulationCancelled

            if isinstance(e, SimulationCancelled):
                self.cancelled.emit()
            else:
                self.error.emit(str(e))
