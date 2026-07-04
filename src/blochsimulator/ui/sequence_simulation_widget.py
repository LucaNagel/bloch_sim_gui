"""Integrated desktop UI for event-based 3D sequence simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..phantom import Phantom, PhantomFactory
from ..sequence import ADCEvent, RFEvent, SequenceCompiler, SequenceProgram, load_pulseq
from ..simulator import BlochSimulator


class SequenceSimulationThread(QThread):
    """Run chunked sequence simulation without blocking Qt."""

    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, simulator, program, phantom, checkpoints_s):
        super().__init__()
        self.simulator = simulator
        self.program = program
        self.phantom = phantom
        self.checkpoints_s = checkpoints_s
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            result = self.simulator.simulate_sequence(
                self.program,
                self.phantom,
                checkpoints_s=self.checkpoints_s,
                progress_callback=lambda done, total: self.progress.emit(done, total),
                cancel_callback=lambda: self._cancel_requested,
            )
            if not self._cancel_requested:
                self.result_ready.emit(result)
        except Exception as exc:
            if self._cancel_requested:
                self.failed.emit("Simulation cancelled")
            else:
                self.failed.emit(str(exc))


class SequenceSimulationWidget(QWidget):
    """Load/build sequences, configure a 3D object, and inspect sparse output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.program: Optional[SequenceProgram] = None
        self.phantom: Optional[Phantom] = None
        self.result = None
        self.worker = None
        self.simulator = BlochSimulator(use_parallel=True, num_threads=4)
        self._build_ui()
        self._load_internal_sequence()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter()
        root.addWidget(splitter)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        splitter.addWidget(controls)

        sequence_group = QGroupBox("Sequence")
        sequence_layout = QVBoxLayout(sequence_group)
        self.sequence_source = QComboBox()
        self.sequence_source.addItems(["Internal FID", "Pulseq .seq file"])
        self.sequence_source.currentIndexChanged.connect(self._source_changed)
        sequence_layout.addWidget(self.sequence_source)
        load_button = QPushButton("Load Pulseq…")
        load_button.clicked.connect(self._load_pulseq_file)
        sequence_layout.addWidget(load_button)
        self.sequence_info = QLabel()
        self.sequence_info.setWordWrap(True)
        sequence_layout.addWidget(self.sequence_info)
        controls_layout.addWidget(sequence_group)

        object_group = QGroupBox("3D object")
        object_form = QFormLayout(object_group)
        self.object_type = QComboBox()
        self.object_type.addItems(["Uniform cube", "Sphere"])
        object_form.addRow("Type", self.object_type)
        self.matrix_size = QSpinBox()
        self.matrix_size.setRange(2, 128)
        self.matrix_size.setValue(16)
        object_form.addRow("Matrix", self.matrix_size)
        self.fov_cm = QDoubleSpinBox()
        self.fov_cm.setRange(0.1, 100.0)
        self.fov_cm.setValue(20.0)
        self.fov_cm.setSuffix(" cm")
        object_form.addRow("FOV", self.fov_cm)
        self.t1_ms = self._parameter_spin(1.0, 10000.0, 1000.0, " ms")
        self.t2_ms = self._parameter_spin(0.1, 5000.0, 100.0, " ms")
        self.pd = self._parameter_spin(0.0, 10.0, 1.0, "")
        self.b0_hz = self._parameter_spin(-10000.0, 10000.0, 0.0, " Hz")
        self.chemical_hz = self._parameter_spin(-10000.0, 10000.0, 0.0, " Hz")
        object_form.addRow("T1", self.t1_ms)
        object_form.addRow("T2", self.t2_ms)
        object_form.addRow("Proton density", self.pd)
        object_form.addRow("B0 offset", self.b0_hz)
        object_form.addRow("Chemical shift", self.chemical_hz)
        controls_layout.addWidget(object_group)

        output_group = QGroupBox("Sparse output")
        output_form = QFormLayout(output_group)
        self.checkpoints = QLineEdit()
        self.checkpoints.setPlaceholderText("e.g. 1.0, 5.0 (ms)")
        output_form.addRow("Checkpoints", self.checkpoints)
        controls_layout.addWidget(output_group)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run sequence simulation")
        self.run_button.clicked.connect(self._run)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        controls_layout.addLayout(run_row)
        self.progress = QProgressBar()
        controls_layout.addWidget(self.progress)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        controls_layout.addStretch()

        views = QTabWidget()
        splitter.addWidget(views)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        timeline = QWidget()
        timeline_layout = QVBoxLayout(timeline)
        self.rf_plot = pg.PlotWidget(title="RF magnitude")
        self.rf_plot.setLabel("left", "RF", "Hz")
        self.rf_plot.setLabel("bottom", "Time", "ms")
        self.gradient_plot = pg.PlotWidget(title="Gradients and ADC")
        self.gradient_plot.setLabel("left", "Gradient", "kHz/m")
        self.gradient_plot.setLabel("bottom", "Time", "ms")
        self.gradient_plot.addLegend()
        timeline_layout.addWidget(self.rf_plot)
        timeline_layout.addWidget(self.gradient_plot)
        views.addTab(timeline, "Sequence")

        signal_page = QWidget()
        signal_layout = QVBoxLayout(signal_page)
        self.signal_plot = pg.PlotWidget(title="Received ADC signal")
        self.signal_plot.setLabel("left", "Signal", "a.u.")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        self.signal_plot.addLegend()
        signal_layout.addWidget(self.signal_plot)
        views.addTab(signal_page, "ADC signal")

        state_page = QWidget()
        state_layout = QVBoxLayout(state_page)
        self.state_view = pg.ImageView()
        self.state_view.ui.roiBtn.hide()
        self.state_view.ui.menuBtn.hide()
        state_layout.addWidget(self.state_view)
        self.state_info = QLabel("No result")
        state_layout.addWidget(self.state_info)
        views.addTab(state_page, "Final Mz")

    @staticmethod
    def _parameter_spin(minimum, maximum, value, suffix):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(4)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _source_changed(self):
        if self.sequence_source.currentIndex() == 0:
            self._load_internal_sequence()

    def _load_internal_sequence(self):
        rf_duration = 1e-3
        dwell = 100e-6
        sample_count = 256
        adc_start = rf_duration + dwell / 2
        duration = rf_duration + sample_count * dwell
        self.program = SequenceProgram(
            events=(
                RFEvent(0.0, np.array([250.0]), rf_duration),
                ADCEvent(adc_start, sample_count, dwell),
            ),
            duration_s=duration,
            source="internal-fid",
        )
        self._show_program()

    def _load_pulseq_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Pulseq sequence", "", "Pulseq sequence (*.seq);;All files (*)"
        )
        if not filename:
            return
        try:
            self.program = load_pulseq(filename)
            self.sequence_source.setCurrentIndex(1)
            self._show_program()
            self.status.setText(f"Loaded {Path(filename).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Pulseq import failed", str(exc))

    def _show_program(self):
        if self.program is None:
            return
        try:
            compiled = SequenceCompiler().compile(self.program)
        except Exception as exc:
            self.sequence_info.setText(f"Invalid sequence: {exc}")
            return
        self.sequence_info.setText(
            f"{self.program.source}\nDuration: {self.program.duration_s*1000:.3f} ms\n"
            f"Events: {len(self.program.events)}, intervals: {compiled.n_intervals}, "
            f"ADC samples: {compiled.adc_times_s.size}"
        )
        self.rf_plot.clear()
        self.gradient_plot.clear()
        if compiled.n_intervals:
            starts = np.concatenate(([0.0], compiled.interval_end_s[:-1])) * 1000
            ends = compiled.interval_end_s * 1000
            x = np.column_stack((starts, ends)).ravel()
            max_points = 20000
            stride = max(1, int(np.ceil(x.size / max_points)))
            rf_y = np.repeat(np.abs(compiled.rf_hz), 2)
            self.rf_plot.plot(x[::stride], rf_y[::stride], pen=pg.mkPen("m"))
            colors = ("r", "g", "b")
            for axis, color, values in zip(
                "xyz", colors, compiled.gradient_hz_per_m.T / 1000.0
            ):
                y = np.repeat(values, 2)
                self.gradient_plot.plot(
                    x[::stride], y[::stride], pen=pg.mkPen(color), name=f"G{axis}"
                )
        if compiled.adc_times_s.size:
            self.gradient_plot.plot(
                compiled.adc_times_s * 1000,
                np.zeros_like(compiled.adc_times_s),
                pen=None,
                symbol="o",
                symbolSize=4,
                symbolBrush="y",
                name="ADC",
            )

    def _build_phantom(self):
        n = self.matrix_size.value()
        shape = (n, n, n)
        fov = self.fov_cm.value() / 100.0
        t1 = self.t1_ms.value() / 1000.0
        t2 = self.t2_ms.value() / 1000.0
        pd = self.pd.value()
        if self.object_type.currentText() == "Uniform cube":
            mask = np.ones(shape, dtype=bool)
        else:
            coordinate = (np.arange(n) + 0.5) / n - 0.5
            x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
            mask = x * x + y * y + z * z <= 0.4**2
        self.phantom = Phantom(
            shape=shape,
            fov=(fov, fov, fov),
            t1_map=np.where(mask, t1, 0.0),
            t2_map=np.where(mask, t2, 0.0),
            pd_map=np.where(mask, pd, 0.0),
            b0_map=np.where(mask, self.b0_hz.value(), 0.0),
            chemical_shift_map=np.where(mask, self.chemical_hz.value(), 0.0),
            mask=mask,
            name="Sequence simulation object",
        )

    def _checkpoint_seconds(self):
        text = self.checkpoints.text().strip()
        if not text:
            return ()
        values = tuple(float(value.strip()) / 1000.0 for value in text.split(","))
        return values

    def _run(self):
        if self.program is None:
            QMessageBox.warning(self, "No sequence", "Choose or load a sequence first.")
            return
        try:
            self._build_phantom()
            checkpoints = self._checkpoint_seconds()
            SequenceCompiler().compile(self.program, checkpoints_s=checkpoints)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid simulation", str(exc))
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Simulating…")
        self.worker = SequenceSimulationThread(
            self.simulator, self.program, self.phantom, checkpoints
        )
        self.worker.progress.connect(self._progress)
        self.worker.result_ready.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"Chunk {done}/{total}")

    def _cancel(self):
        if self.worker is not None:
            self.worker.request_cancel()
            self.status.setText("Cancelling after current chunk…")

    def _finished(self, result):
        self.result = result
        self._reset_run_controls()
        self.status.setText("Simulation complete")
        self.signal_plot.clear()
        time_ms = result.adc_times_s * 1000
        signal = np.asarray(result.signal)
        if signal.ndim == 1:
            self.signal_plot.plot(time_ms, np.abs(signal), pen="w", name="Magnitude")
            self.signal_plot.plot(time_ms, signal.real, pen="g", name="Real")
            self.signal_plot.plot(time_ms, signal.imag, pen="r", name="Imaginary")
            coil_text = ""
        else:
            for coil, coil_signal in enumerate(signal):
                self.signal_plot.plot(
                    time_ms,
                    np.abs(coil_signal),
                    pen=pg.intColor(coil, hues=signal.shape[0]),
                    name=f"Coil {coil + 1}",
                )
            coil_text = f"; Rx coils={signal.shape[0]}"
        mz = result.mz
        image = mz[:, :, mz.shape[2] // 2] if mz.ndim == 3 else np.squeeze(mz)
        self.state_view.setImage(np.asarray(image).T, autoLevels=True)
        self.state_info.setText(
            f"Final Mz: min={np.min(mz):.5g}, max={np.max(mz):.5g}; "
            f"ADC samples={result.adc_times_s.size}{coil_text}"
        )

    def _failed(self, message):
        self._reset_run_controls()
        self.status.setText(message)
        if message != "Simulation cancelled":
            QMessageBox.critical(self, "Sequence simulation failed", message)

    def _reset_run_controls(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
