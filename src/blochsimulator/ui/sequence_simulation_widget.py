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
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..phantom import Phantom, PhantomFactory
from ..spectral_phantom import SpectralPhantom
from ..sequence import (
    ADCEvent,
    CartesianAcquisition,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
    infer_cartesian_acquisition,
    load_pulseq,
    make_cartesian_epi,
)
from ..simulator import BlochSimulator
from .volume_viewer import SequenceResultVolumeViewer


class SequenceSimulationThread(QThread):
    """Run chunked sequence simulation without blocking Qt."""

    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        simulator,
        program,
        phantom,
        checkpoints_s,
        signal_weighting="voxel",
    ):
        super().__init__()
        self.simulator = simulator
        self.program = program
        self.phantom = phantom
        self.checkpoints_s = checkpoints_s
        self.signal_weighting = signal_weighting
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            simulate = (
                self.simulator.simulate_spectral_sequence
                if isinstance(self.phantom, SpectralPhantom)
                else self.simulator.simulate_sequence
            )
            result = simulate(
                self.program,
                self.phantom,
                checkpoints_s=self.checkpoints_s,
                signal_weighting=self.signal_weighting,
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
        self.acquisition: Optional[CartesianAcquisition] = None
        self.phantom: Optional[Phantom] = None
        self.result = None
        self.worker = None
        self.acquisition_note = ""
        self.simulator = BlochSimulator(use_parallel=True, num_threads=4)
        self._build_ui()
        self._load_internal_sequence()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter()
        root.addWidget(splitter)

        control_column = QWidget()
        control_column_layout = QVBoxLayout(control_column)
        control_column_layout.setContentsMargins(0, 0, 0, 0)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(controls)
        control_column_layout.addWidget(self.controls_scroll, 1)
        splitter.addWidget(control_column)

        sequence_group = QGroupBox("Sequence")
        sequence_layout = QVBoxLayout(sequence_group)
        sequence_layout.addWidget(QLabel("Source / mode"))
        self.sequence_source = QComboBox()
        self.sequence_source.addItems(
            ["Internal FID", "Cartesian EPI", "Pulseq .seq file"]
        )
        self.sequence_source.currentIndexChanged.connect(self._source_changed)
        self.sequence_source.setToolTip(
            "Choose Cartesian EPI to configure matrix and receiver bandwidth"
        )
        sequence_layout.addWidget(self.sequence_source)
        load_button = QPushButton("Load Pulseq…")
        load_button.clicked.connect(self._load_pulseq_file)
        sequence_layout.addWidget(load_button)
        self.sequence_info = QLabel()
        self.sequence_info.setWordWrap(True)
        sequence_layout.addWidget(self.sequence_info)
        controls_layout.addWidget(sequence_group)

        self.acquisition_group = QGroupBox("Cartesian acquisition")
        acquisition_form = QFormLayout(self.acquisition_group)
        self.acquisition_hint = QLabel(
            "Select ‘Cartesian EPI’ under Source / mode to configure a 2D "
            "kx-ky acquisition."
        )
        self.acquisition_hint.setWordWrap(True)
        self.read_matrix = QSpinBox()
        self.read_matrix.setRange(2, 512)
        self.read_matrix.setValue(16)
        self.phase_matrix = QSpinBox()
        self.phase_matrix.setRange(2, 512)
        self.phase_matrix.setValue(16)
        self.sampling_bandwidth_khz = QDoubleSpinBox()
        self.sampling_bandwidth_khz.setRange(0.1, 2000.0)
        self.sampling_bandwidth_khz.setDecimals(3)
        self.sampling_bandwidth_khz.setValue(50.0)
        self.sampling_bandwidth_khz.setSuffix(" kHz")
        self.dwell_info = QLabel()
        self.pixel_bandwidth_info = QLabel()
        acquisition_form.addRow(self.acquisition_hint)
        acquisition_form.addRow("Read matrix", self.read_matrix)
        acquisition_form.addRow("Phase matrix", self.phase_matrix)
        acquisition_form.addRow("Sampling bandwidth", self.sampling_bandwidth_khz)
        acquisition_form.addRow("ADC dwell", self.dwell_info)
        acquisition_form.addRow("Pixel bandwidth", self.pixel_bandwidth_info)
        self.acquisition_group.setEnabled(False)
        controls_layout.addWidget(self.acquisition_group)

        object_group = QGroupBox("Simulation object")
        object_form = QFormLayout(object_group)
        self.object_source = QComboBox()
        self.object_source.addItems(["Phantom tab / designer", "Built-in quick object"])
        object_form.addRow("Source", self.object_source)
        self.object_type = QComboBox()
        self.object_type.addItems(
            ["None — defined in Phantom tab", "Uniform cube", "Sphere"]
        )
        object_form.addRow("Type", self.object_type)
        self.matrix_size = QSpinBox()
        self.matrix_size.setRange(2, 128)
        self.matrix_size.setValue(16)
        object_form.addRow("In-plane matrix", self.matrix_size)
        self.z_matrix_size = QSpinBox()
        self.z_matrix_size.setRange(1, 128)
        self.z_matrix_size.setValue(16)
        object_form.addRow("Through-plane matrix", self.z_matrix_size)
        self.fov_cm = QDoubleSpinBox()
        self.fov_cm.setRange(0.1, 100.0)
        self.fov_cm.setValue(20.0)
        self.fov_cm.setSuffix(" cm")
        object_form.addRow("In-plane FOV", self.fov_cm)
        self.fov_z_cm = QDoubleSpinBox()
        self.fov_z_cm.setRange(0.001, 100.0)
        self.fov_z_cm.setDecimals(4)
        self.fov_z_cm.setValue(20.0)
        self.fov_z_cm.setSuffix(" cm")
        object_form.addRow("Through-plane FOV", self.fov_z_cm)
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
        self.phantom_summary = QLabel()
        self.phantom_summary.setWordWrap(True)
        object_form.addRow("Selected phantom", self.phantom_summary)
        self.open_phantom_button = QPushButton("Open Phantom tab…")
        self.open_phantom_button.clicked.connect(self._open_phantom_tab)
        object_form.addRow(self.open_phantom_button)
        controls_layout.addWidget(object_group)

        self._built_in_object_widgets = (
            self.object_type,
            self.matrix_size,
            self.z_matrix_size,
            self.fov_cm,
            self.fov_z_cm,
            self.t1_ms,
            self.t2_ms,
            self.pd,
            self.b0_hz,
            self.chemical_hz,
        )
        self.object_source.currentIndexChanged.connect(self._object_source_changed)

        output_group = QGroupBox("Sparse output")
        output_form = QFormLayout(output_group)
        self.checkpoints = QLineEdit()
        self.checkpoints.setPlaceholderText("e.g. 1.0, 5.0 (ms)")
        output_form.addRow("Checkpoints", self.checkpoints)
        self.signal_weighting = QComboBox()
        self.signal_weighting.addItems(
            ["Relative voxel sum", "Physical voxel volume (3D)"]
        )
        output_form.addRow("Signal weighting", self.signal_weighting)
        controls_layout.addWidget(output_group)
        controls_layout.addStretch()

        run_panel = QGroupBox("Run")
        run_panel_layout = QVBoxLayout(run_panel)
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run sequence simulation")
        self.run_button.clicked.connect(self._run)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_panel_layout.addLayout(run_row)
        self.progress = QProgressBar()
        run_panel_layout.addWidget(self.progress)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        run_panel_layout.addWidget(self.status)
        control_column_layout.addWidget(run_panel)

        self.read_matrix.valueChanged.connect(self._acquisition_changed)
        self.phase_matrix.valueChanged.connect(self._acquisition_changed)
        self.sampling_bandwidth_khz.valueChanged.connect(self._acquisition_changed)
        self.fov_cm.valueChanged.connect(self._acquisition_changed)
        self._update_bandwidth_labels()
        self._object_source_changed()

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

        kspace_page = QWidget()
        kspace_layout = QVBoxLayout(kspace_page)
        self.kspace_view = pg.ImageView()
        self.kspace_view.ui.roiBtn.hide()
        self.kspace_view.ui.menuBtn.hide()
        kspace_layout.addWidget(self.kspace_view)
        self.kspace_info = QLabel("No 2D Cartesian result")
        kspace_layout.addWidget(self.kspace_info)
        views.addTab(kspace_page, "2D k-space")

        reconstruction_page = QWidget()
        reconstruction_layout = QVBoxLayout(reconstruction_page)
        self.reconstruction_view = pg.ImageView()
        self.reconstruction_view.ui.roiBtn.hide()
        self.reconstruction_view.ui.menuBtn.hide()
        reconstruction_layout.addWidget(self.reconstruction_view)
        self.reconstruction_info = QLabel("No 2D Cartesian result")
        reconstruction_layout.addWidget(self.reconstruction_info)
        views.addTab(reconstruction_page, "2D Reconstruction")

        state_page = QWidget()
        state_layout = QVBoxLayout(state_page)
        self.state_view = pg.ImageView()
        self.state_view.ui.roiBtn.hide()
        self.state_view.ui.menuBtn.hide()
        state_layout.addWidget(self.state_view)
        self.state_info = QLabel("No result")
        state_layout.addWidget(self.state_info)
        views.addTab(state_page, "Final Mz")

        self.result_volume_viewer = SequenceResultVolumeViewer()
        views.addTab(self.result_volume_viewer, "Spatial Magnetization")

    @staticmethod
    def _parameter_spin(minimum, maximum, value, suffix):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(4)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _object_source_changed(self, *_):
        """Expose controls only for the object source that actually owns them."""
        phantom_selected = self.object_source.currentIndex() == 0
        self.object_type.blockSignals(True)
        if phantom_selected:
            self.object_type.setCurrentIndex(0)
        elif self.object_type.currentIndex() == 0:
            self.object_type.setCurrentIndex(1)
        self.object_type.blockSignals(False)
        for widget in self._built_in_object_widgets:
            widget.setEnabled(not phantom_selected)
        self.phantom_summary.setVisible(phantom_selected)
        self.open_phantom_button.setVisible(phantom_selected)
        self.refresh_object_summary()
        if not phantom_selected and self.sequence_source.currentIndex() == 1:
            self._load_cartesian_epi()

    def _selected_designed_phantom(self):
        main_window = self.window()
        phantom_widget = getattr(main_window, "phantom_widget", None)
        return getattr(phantom_widget, "current_phantom", None)

    def refresh_object_summary(self, *_):
        """Refresh the read-only summary of the shared Phantom-tab object."""
        if self.object_source.currentIndex() != 0:
            return
        phantom = self._selected_designed_phantom()
        if phantom is None:
            self.phantom_summary.setText(
                "No phantom selected. Create or load one in the Phantom tab."
            )
            return
        active = np.asarray(phantom.mask, dtype=bool)
        t1 = np.asarray(phantom.t1_map)[active] * 1000.0
        t2 = np.asarray(phantom.t2_map)[active] * 1000.0
        fov_cm = " × ".join(f"{value * 100:.4g}" for value in phantom.fov)
        relaxation_text = (
            f"T1 {t1.min():.4g}–{t1.max():.4g} ms; "
            f"T2/T2* {t2.min():.4g}–{t2.max():.4g} ms"
            if t1.size and t2.size
            else "No active tissue voxels"
        )
        self.phantom_summary.setText(
            f"{phantom.name}\n"
            f"{phantom.ndim}D, matrix {tuple(phantom.shape)}, "
            f"FOV {fov_cm} cm, {phantom.n_active} active voxels\n"
            f"{relaxation_text}"
        )
        if self.sequence_source.currentIndex() == 1:
            self._load_cartesian_epi()

    def _open_phantom_tab(self):
        main_window = self.window()
        tab_widget = getattr(main_window, "tab_widget", None)
        tab_index = getattr(main_window, "phantom_tab_index", -1)
        if tab_widget is not None and tab_index >= 0:
            tab_widget.setCurrentIndex(tab_index)
        else:
            QMessageBox.information(
                self,
                "Phantom workspace unavailable",
                "Open this widget inside the main application to use the Phantom tab.",
            )

    def _source_changed(self, *_):
        source_index = self.sequence_source.currentIndex()
        epi_selected = source_index == 1
        self.acquisition_group.setEnabled(epi_selected)
        self.acquisition_hint.setText(
            "Read/phase matrix and sampling bandwidth define a 2D kx-ky "
            "acquisition. No kz encoding is performed."
            if epi_selected
            else "Select ‘Cartesian EPI’ under Source / mode to enable these settings."
        )
        if source_index == 0:
            self._load_internal_sequence()
        elif source_index == 1:
            self._load_cartesian_epi()

    def _acquisition_changed(self, *_):
        self._update_bandwidth_labels()
        if self.sequence_source.currentIndex() == 1:
            self._load_cartesian_epi()

    def _update_bandwidth_labels(self):
        bandwidth_hz = self.sampling_bandwidth_khz.value() * 1000.0
        dwell_us = 1e6 / bandwidth_hz
        pixel_bandwidth_hz = bandwidth_hz / self.read_matrix.value()
        self.dwell_info.setText(f"{dwell_us:.3f} µs")
        self.pixel_bandwidth_info.setText(f"{pixel_bandwidth_hz:.3f} Hz/px")

    def _load_internal_sequence(self):
        self.acquisition = None
        self.acquisition_note = ""
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

    def _load_cartesian_epi(self):
        bandwidth_hz = self.sampling_bandwidth_khz.value() * 1000.0
        designed = (
            self._selected_designed_phantom()
            if self.object_source.currentIndex() == 0
            else None
        )
        if designed is not None and designed.ndim >= 2:
            fov_m = tuple(float(value) for value in designed.fov[:2])
        else:
            local_fov = self.fov_cm.value() / 100.0
            fov_m = (local_fov, local_fov)
        try:
            self.acquisition = CartesianAcquisition.epi(
                read_matrix=self.read_matrix.value(),
                phase_matrix=self.phase_matrix.value(),
                fov_m=fov_m,
                dwell_s=1.0 / bandwidth_hz,
            )
            self.program = make_cartesian_epi(self.acquisition)
            self.acquisition_note = ""
            self._show_program()
        except Exception as exc:
            self.acquisition = None
            self.program = None
            self.sequence_info.setText(f"Invalid Cartesian acquisition: {exc}")

    def _load_pulseq_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Pulseq sequence", "", "Pulseq sequence (*.seq);;All files (*)"
        )
        if not filename:
            return
        try:
            self._load_pulseq_path(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Pulseq import failed", str(exc))

    def _load_pulseq_path(self, filename):
        """Load a Pulseq file and attach a validated 2D Cartesian layout."""
        self.program = load_pulseq(filename)
        compiled = SequenceCompiler().compile(self.program)
        try:
            self.acquisition = infer_cartesian_acquisition(
                self.program, compiled=compiled
            )
            self.acquisition_note = "Cartesian acquisition inferred from ADC moments"
        except ValueError as exc:
            self.acquisition = None
            self.acquisition_note = f"Cartesian inference unavailable: {exc}"
        self._apply_pulseq_fov()
        self.sequence_source.setCurrentIndex(2)
        self._show_program()
        status = f"Loaded {Path(filename).name}"
        if self.acquisition_note:
            status += f"; {self.acquisition_note}"
        self.status.setText(status)

    def _apply_pulseq_fov(self):
        definitions = dict(self.program.metadata.get("definitions", {}))
        fov_value = next(
            (value for key, value in definitions.items() if str(key).lower() == "fov"),
            None,
        )
        if fov_value is None:
            return
        fov = np.asarray(fov_value, dtype=float).reshape(-1)
        if fov.size >= 2 and np.all(np.isfinite(fov[:2])) and np.all(fov[:2] > 0):
            if np.isclose(fov[0], fov[1], rtol=1e-6, atol=1e-12):
                self.fov_cm.setValue(float(fov[0]) * 100.0)
        if fov.size >= 3 and np.isfinite(fov[2]) and fov[2] > 0:
            self.fov_z_cm.setValue(float(fov[2]) * 100.0)

    def _show_program(self):
        if self.program is None:
            return
        try:
            compiled = SequenceCompiler().compile(self.program)
        except Exception as exc:
            self.sequence_info.setText(f"Invalid sequence: {exc}")
            return
        acquisition_text = ""
        if self.acquisition is not None:
            offsets = ""
            if self.acquisition.kx_offset_cells or self.acquisition.ky_offset_cells:
                offsets = (
                    f"; offsets=({self.acquisition.kx_offset_cells:.3g}, "
                    f"{self.acquisition.ky_offset_cells:.3g}) cells"
                )
            acquisition_text = (
                f"\nGrid: {self.acquisition.phase_matrix}×"
                f"{self.acquisition.read_matrix}; "
                f"BW: {self.acquisition.sampling_bandwidth_hz/1000:.3f} kHz"
                f"{offsets}"
            )
        elif self.acquisition_note:
            acquisition_text = f"\n{self.acquisition_note}"
        self.sequence_info.setText(
            f"{self.program.source}\nDuration: {self.program.duration_s*1000:.3f} ms\n"
            f"Events: {len(self.program.events)}, intervals: {compiled.n_intervals}, "
            f"ADC samples: {compiled.adc_times_s.size}{acquisition_text}"
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
        if self.object_source.currentIndex() == 0:
            designed = self._selected_designed_phantom()
            if designed is None:
                raise ValueError(
                    "No phantom is loaded in the Phantom tab. Create or load one first."
                )
            self.phantom = designed
            return
        n = self.matrix_size.value()
        nz = self.z_matrix_size.value()
        shape = (n, n, nz)
        fov = self.fov_cm.value() / 100.0
        fov_z = self.fov_z_cm.value() / 100.0
        t1 = self.t1_ms.value() / 1000.0
        t2 = self.t2_ms.value() / 1000.0
        pd = self.pd.value()
        if self.object_type.currentIndex() == 1:
            mask = np.ones(shape, dtype=bool)
        else:
            coordinate = (np.arange(n) + 0.5) / n - 0.5
            coordinate_z = (np.arange(nz) + 0.5) / nz - 0.5
            x, y, z = np.meshgrid(coordinate, coordinate, coordinate_z, indexing="ij")
            mask = x * x + y * y + z * z <= 0.4**2
        self.phantom = Phantom(
            shape=shape,
            fov=(fov, fov, fov_z),
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

    def _signal_weighting_mode(self):
        return "voxel_volume" if self.signal_weighting.currentIndex() == 1 else "voxel"

    def _run(self):
        if self.sequence_source.currentIndex() == 1:
            self._load_cartesian_epi()
        if self.program is None:
            QMessageBox.warning(self, "No sequence", "Choose or load a sequence first.")
            return
        try:
            self._build_phantom()
            checkpoints = self._checkpoint_seconds()
            compiled = SequenceCompiler().compile(
                self.program, checkpoints_s=checkpoints
            )
            if self.acquisition is not None:
                self.acquisition.validate_adc_times(compiled.adc_times_s)
                self.acquisition.validate_gradient_moments(
                    compiled.adc_gradient_moment_cyc_per_m
                )
        except Exception as exc:
            QMessageBox.critical(self, "Invalid simulation", str(exc))
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Simulating…")
        self.worker = SequenceSimulationThread(
            self.simulator,
            self.program,
            self.phantom,
            checkpoints,
            self._signal_weighting_mode(),
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
        if mz.ndim == 3:
            z_index = mz.shape[2] // 2
            image = mz[:, :, z_index]
            z_position_mm = (
                ((z_index + 0.5) / mz.shape[2] - 0.5) * self.phantom.fov[2] * 1000.0
            )
            slice_text = f"; displayed z={z_position_mm:.3g} mm"
        else:
            image = np.squeeze(mz)
            slice_text = ""
        mz_min = float(np.min(mz))
        mz_max = float(np.max(mz))
        if mz_max - mz_min <= max(1e-12, 1e-9 * max(abs(mz_min), abs(mz_max))):
            centre = 0.5 * (mz_min + mz_max)
            levels = (centre - 1e-6, centre + 1e-6)
        else:
            levels = (mz_min, mz_max)
        self.state_view.setImage(np.asarray(image).T, autoLevels=False, levels=levels)
        self.state_info.setText(
            f"Final Mz: min={mz_min:.5g}, max={mz_max:.5g}; "
            f"ADC samples={result.adc_times_s.size}{coil_text}{slice_text}"
        )
        self.result_volume_viewer.set_result(result, self.phantom)
        self._show_cartesian_result(result)

    def _show_cartesian_result(self, result):
        if self.acquisition is None:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            self.kspace_info.setText("No Cartesian acquisition metadata")
            self.reconstruction_info.setText("No Cartesian acquisition metadata")
            return
        try:
            kspace = result.to_cartesian_kspace(self.acquisition)
            if kspace.ndim == 3:
                kspace_magnitude = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
                image = result.reconstruct_cartesian(
                    self.acquisition, coil_combine="rss"
                )
                coil_text = f", {kspace.shape[0]} coils (RSS)"
            else:
                kspace_magnitude = np.abs(kspace)
                image = np.abs(result.reconstruct_cartesian(self.acquisition))
                coil_text = ""
            self.kspace_view.setImage(np.log1p(kspace_magnitude).T, autoLevels=True)
            self.reconstruction_view.setImage(
                np.asarray(np.abs(image)).T, autoLevels=True
            )
            self.kspace_info.setText(
                f"2D log(1+|k|), grid={self.acquisition.phase_matrix}×"
                f"{self.acquisition.read_matrix}{coil_text}"
            )
            z_note = (
                "; z-integrated signal (no kz encoding)"
                if self.phantom.ndim == 3 and self.phantom.shape[2] > 1
                else "; no kz encoding"
            )
            self.reconstruction_info.setText(
                f"2D |IFFT2|, min={np.min(np.abs(image)):.5g}, "
                f"max={np.max(np.abs(image)):.5g}{coil_text}{z_note}"
            )
        except Exception as exc:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            message = f"Cartesian reconstruction unavailable: {exc}"
            self.kspace_info.setText(message)
            self.reconstruction_info.setText(message)

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
