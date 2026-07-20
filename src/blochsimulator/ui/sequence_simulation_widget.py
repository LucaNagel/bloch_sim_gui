"""Integrated desktop UI for event-based 3D sequence simulation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..phantom import Phantom, PhantomFactory
from ..notebook_exporter import export_sequence_result_notebook
from ..paths import workspace_directory
from ..spectral_phantom import SpectralPhantom
from ..dynamic_phantom import DynamicSpectralPhantom
from ..sequence import (
    ADCEvent,
    BrukerExportOptions,
    CartesianAcquisition,
    CartesianAcquisitionFrames,
    SpectroscopicAcquisition,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
    export_bruker_raw,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    infer_spectroscopic_acquisition,
    load_pulseq,
    make_cartesian_epi,
)
from ..simulator import BlochSimulator, resolve_num_threads
from ..units import NUCLEUS_GAMMA_HZ_PER_T, ppm_to_hz
from .controls import UniversalTimeControl
from .magnetization_viewer import MagnetizationViewer
from .probe_viewers import SequenceProbeSpatialViewer, SequenceProbeSpectrumViewer
from .volume_viewer import SequenceResultVolumeViewer


def _representative_sample_indices(values, max_samples):
    """Select raster cells without losing event endpoints or extrema."""
    values = np.asarray(values)
    count = int(values.size)
    max_samples = max(2, int(max_samples))
    if count <= max_samples:
        return np.arange(count, dtype=int)
    base_count = max(2, max_samples - 2)
    selected = set(np.linspace(0, count - 1, base_count, dtype=int))
    selected.add(int(np.argmin(values)))
    selected.add(int(np.argmax(values)))
    return np.asarray(sorted(selected), dtype=int)


def _event_step_plot_data(
    events,
    *,
    samples_attribute,
    start_s,
    end_s,
    scale=1.0,
    magnitude=False,
    max_vertices=50000,
):
    """Build zoom-aware connected waveforms from canonical sequence events.

    Events are separated by NaNs so silent gaps remain silent. Representative
    raster samples are connected inside each event, making short RF and
    gradient events visible even in a long-sequence overview. Per-event extrema
    are retained, while zoomed ranges expose many more native raster samples.
    """
    start_s = float(start_s)
    end_s = float(end_s)
    visible = [
        event for event in events if event.end_s > start_s and event.start_s < end_s
    ]
    if not visible:
        return np.empty(0), np.empty(0)
    samples_per_event = max(4, int(max_vertices // len(visible)) - 5)
    x_parts = []
    y_parts = []
    for event in visible:
        raw = np.asarray(getattr(event, samples_attribute))
        values = np.abs(raw) if magnitude else np.asarray(raw, dtype=float)
        first = max(0, int(np.floor((start_s - event.start_s) / event.raster_s)))
        stop = min(
            values.size,
            int(np.ceil((end_s - event.start_s) / event.raster_s)),
        )
        if stop <= first:
            continue
        window = values[first:stop] * float(scale)
        local_indices = _representative_sample_indices(window, samples_per_event)
        indices = first + local_indices
        visible_start = max(event.start_s, start_s)
        visible_end = min(event.end_s, end_s)
        sample_centres = np.clip(
            event.start_s + (indices + 0.5) * event.raster_s,
            visible_start,
            visible_end,
        )
        x = (
            np.concatenate(
                (
                    [visible_start, visible_start],
                    sample_centres,
                    [visible_end, visible_end, np.nan],
                )
            )
            * 1000.0
        )
        selected_values = window[local_indices]
        y = np.concatenate(
            (
                [0.0, selected_values[0]],
                selected_values,
                [selected_values[-1], 0.0, np.nan],
            )
        )
        x_parts.append(x)
        y_parts.append(y)
    if not x_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(x_parts), np.concatenate(y_parts)


class SequenceSimulationThread(QThread):
    """Run chunked sequence simulation without blocking Qt."""

    progress = pyqtSignal(int, int)
    stage = pyqtSignal(str)
    preview = pyqtSignal(float, object)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        simulator,
        program,
        phantom,
        checkpoints_s,
        signal_weighting="voxel",
        field_strength_t=7.0,
        nucleus="C13",
        live_preview=True,
        chunk_voxels=None,
        simulation_timestep_s=1e-6,
    ):
        super().__init__()
        self.simulator = simulator
        self.program = program
        self.phantom = phantom
        self.checkpoints_s = checkpoints_s
        self.signal_weighting = signal_weighting
        self.field_strength_t = field_strength_t
        self.nucleus = nucleus
        self.live_preview = bool(live_preview)
        self.chunk_voxels = chunk_voxels
        self.simulation_timestep_s = simulation_timestep_s
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            if isinstance(self.phantom, DynamicSpectralPhantom):
                simulate = self.simulator.simulate_dynamic_sequence
            elif isinstance(self.phantom, SpectralPhantom):
                simulate = self.simulator.simulate_spectral_sequence
            else:
                simulate = self.simulator.simulate_sequence
            kwargs = {
                "checkpoints_s": self.checkpoints_s,
                "signal_weighting": self.signal_weighting,
                "progress_callback": lambda done, total: self.progress.emit(
                    done, total
                ),
                "chunk_voxels": self.chunk_voxels,
                "cancel_callback": lambda: self._cancel_requested,
                "status_callback": lambda message: self.stage.emit(message),
                "simulation_timestep_s": self.simulation_timestep_s,
            }
            if self.live_preview:
                kwargs["preview_callback"] = lambda fraction, signal: (
                    self.preview.emit(fraction, signal)
                )
            if isinstance(self.phantom, (SpectralPhantom, DynamicSpectralPhantom)):
                kwargs.update(
                    field_strength_t=self.field_strength_t,
                    nucleus=self.nucleus,
                )
            result = simulate(self.program, self.phantom, **kwargs)
            if not self._cancel_requested:
                self.result_ready.emit(result)
        except Exception as exc:
            if self._cancel_requested:
                self.failed.emit("Simulation cancelled")
            else:
                self.failed.emit(str(exc))


class SequenceProbeThread(QThread):
    """Run explicit spin probes without blocking Qt."""

    progress = pyqtSignal(int, int)
    stage = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        simulator,
        program,
        positions_m,
        frequency_offsets_hz,
        checkpoints_s,
        t1_s,
        t2_s,
        initial_magnetization=(0.0, 0.0, 1.0),
        simulation_timestep_s=1e-6,
    ):
        super().__init__()
        self.simulator = simulator
        self.program = program
        self.positions_m = positions_m
        self.frequency_offsets_hz = frequency_offsets_hz
        self.checkpoints_s = checkpoints_s
        self.t1_s = t1_s
        self.t2_s = t2_s
        self.initial_magnetization = initial_magnetization
        self.simulation_timestep_s = simulation_timestep_s
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            result = self.simulator.simulate_sequence_probes(
                self.program,
                self.positions_m,
                self.frequency_offsets_hz,
                checkpoints_s=self.checkpoints_s,
                t1_s=self.t1_s,
                t2_s=self.t2_s,
                initial_magnetization=self.initial_magnetization,
                progress_callback=lambda done, total: self.progress.emit(done, total),
                cancel_callback=lambda: self._cancel_requested,
                status_callback=lambda message: self.stage.emit(message),
                simulation_timestep_s=self.simulation_timestep_s,
            )
            if not self._cancel_requested:
                self.result_ready.emit(result)
        except Exception as exc:
            if self._cancel_requested:
                self.failed.emit("Probe simulation cancelled")
            else:
                self.failed.emit(str(exc))


class SequenceSimulationWidget(QWidget):
    """Load/build sequences, configure a 3D object, and inspect sparse output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.program: Optional[SequenceProgram] = None
        self.acquisition: Optional[CartesianAcquisition] = None
        self.acquisition_frames: Optional[CartesianAcquisitionFrames] = None
        self.spectroscopic_acquisition: Optional[SpectroscopicAcquisition] = None
        self.phantom: Optional[Phantom] = None
        self.result = None
        self.probe_result = None
        self._split_csi_data = None
        self.worker = None
        self.probe_worker = None
        self._probe_playback_anchor_wall = None
        self._probe_playback_anchor_time_ms = None
        self._sequence_plot_window_s = None
        self._sequence_plot_pending_window_s = None
        self._rf_waveform_item = None
        self._gradient_waveform_items = {}
        self._sequence_spoiler_markers = []
        self.probe_playback_timer = QTimer(self)
        self.probe_playback_timer.setInterval(16)
        self.probe_playback_timer.timeout.connect(self._advance_probe_playback)
        self.sequence_plot_refresh_timer = QTimer(self)
        self.sequence_plot_refresh_timer.setSingleShot(True)
        self.sequence_plot_refresh_timer.setInterval(60)
        self.sequence_plot_refresh_timer.timeout.connect(
            self._refresh_pending_sequence_plot
        )
        settings = getattr(parent, "app_settings", None)
        self.live_preview_enabled = (
            bool(settings.value("sequence/live_progress_enabled", True, type=bool))
            if settings is not None
            else True
        )
        sequence_kernel = (
            str(settings.value("sequence/kernel", "optimized"))
            if settings is not None
            else "optimized"
        )
        if sequence_kernel not in {"optimized", "reference"}:
            sequence_kernel = "optimized"
        try:
            sequence_timestep_us = float(
                settings.value("sequence/timestep_us", 5.0)
                if settings is not None
                else 5.0
            )
        except (TypeError, ValueError):
            sequence_timestep_us = 5.0
        if (
            not np.isfinite(sequence_timestep_us)
            or not 0.1 <= sequence_timestep_us <= 1000.0
        ):
            sequence_timestep_us = 5.0
        thread_mode = (
            str(settings.value("simulation/thread_mode", "automatic"))
            if settings is not None
            else "automatic"
        )
        try:
            manual_threads = int(
                settings.value("simulation/manual_threads", 4)
                if settings is not None
                else 4
            )
        except (TypeError, ValueError):
            manual_threads = 4
        if thread_mode not in {"automatic", "manual"}:
            thread_mode = "automatic"
        configured_threads = (
            None if thread_mode == "automatic" else max(1, manual_threads)
        )
        self.acquisition_note = ""
        self.simulator = BlochSimulator(
            use_parallel=True,
            num_threads=configured_threads,
            sequence_kernel=sequence_kernel,
        )
        self._initial_sequence_timestep_us = sequence_timestep_us
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
        self.sequence_source.addItems(["Internal FID", "EPI", "Pulseq .seq file"])
        self.sequence_source.currentIndexChanged.connect(self._source_changed)
        self.sequence_source.setToolTip(
            "Choose EPI to configure matrix and receiver bandwidth"
        )
        sequence_layout.addWidget(self.sequence_source)
        load_button = QPushButton("Load Pulseq…")
        load_button.clicked.connect(self._load_pulseq_file)
        sequence_layout.addWidget(load_button)
        self.sequence_info = QLabel()
        self.sequence_info.setWordWrap(True)
        sequence_layout.addWidget(self.sequence_info)
        controls_layout.addWidget(sequence_group)

        self.acquisition_group = QGroupBox("EPI acquisition")
        acquisition_form = QFormLayout(self.acquisition_group)
        self.acquisition_hint = QLabel(
            "EPI uses an x readout and y phase encoding on a Cartesian grid."
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
        self.acquisition_group.setVisible(False)
        controls_layout.addWidget(self.acquisition_group)

        object_group = QGroupBox("Simulation object")
        object_form = QFormLayout(object_group)
        self.object_source = QComboBox()
        self.object_source.addItems(["Phantom tab / designer", "Built-in quick object"])
        object_form.addRow("Source", self.object_source)
        self.field_strength_t = QDoubleSpinBox()
        self.field_strength_t.setRange(0.01, 30.0)
        self.field_strength_t.setDecimals(4)
        self.field_strength_t.setValue(3.0)
        self.field_strength_t.setSuffix(" T")
        self.field_strength_t.setToolTip(
            "Converts field-independent phantom frequency offsets from ppm to Hz"
        )
        object_form.addRow("Field strength B0", self.field_strength_t)
        self.nucleus = QComboBox()
        self.nucleus.addItems(list(NUCLEUS_GAMMA_HZ_PER_T))
        self.nucleus.setToolTip("Reference nucleus used for ppm-to-Hz conversion")
        object_form.addRow("Nucleus", self.nucleus)
        self.frequency_reference_info = QLabel()
        self.frequency_reference_info.setWordWrap(True)
        object_form.addRow("Frequency model", self.frequency_reference_info)

        self.built_in_properties_group = QGroupBox("Built-in phantom properties")
        built_in_form = QFormLayout(self.built_in_properties_group)
        self.object_type = QComboBox()
        self.object_type.addItems(
            ["None — defined in Phantom tab", "Uniform cube", "Sphere"]
        )
        built_in_form.addRow("Type", self.object_type)
        self.matrix_size = QSpinBox()
        self.matrix_size.setRange(2, 128)
        self.matrix_size.setValue(16)
        built_in_form.addRow("In-plane matrix", self.matrix_size)
        self.z_matrix_size = QSpinBox()
        self.z_matrix_size.setRange(1, 128)
        self.z_matrix_size.setValue(16)
        built_in_form.addRow("Through-plane matrix", self.z_matrix_size)
        self.fov_cm = QDoubleSpinBox()
        self.fov_cm.setRange(0.1, 100.0)
        self.fov_cm.setValue(20.0)
        self.fov_cm.setSuffix(" cm")
        built_in_form.addRow("In-plane FOV", self.fov_cm)
        self.fov_z_cm = QDoubleSpinBox()
        self.fov_z_cm.setRange(0.001, 100.0)
        self.fov_z_cm.setDecimals(4)
        self.fov_z_cm.setValue(20.0)
        self.fov_z_cm.setSuffix(" cm")
        built_in_form.addRow("Through-plane FOV", self.fov_z_cm)
        self.t1_ms = self._parameter_spin(1.0, 10000.0, 1000.0, " ms")
        self.t2_ms = self._parameter_spin(0.1, 5000.0, 100.0, " ms")
        self.pd = self._parameter_spin(0.0, 10.0, 1.0, "")
        self.b0_ppm = self._parameter_spin(-1000.0, 1000.0, 0.0, " ppm")
        self.chemical_ppm = self._parameter_spin(-1000.0, 1000.0, 0.0, " ppm")
        built_in_form.addRow("T1", self.t1_ms)
        built_in_form.addRow("T2", self.t2_ms)
        built_in_form.addRow("Proton density", self.pd)
        built_in_form.addRow("B0 inhomogeneity", self.b0_ppm)
        built_in_form.addRow("Chemical shift", self.chemical_ppm)
        object_form.addRow(self.built_in_properties_group)
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
            self.b0_ppm,
            self.chemical_ppm,
        )
        self.object_source.currentIndexChanged.connect(self._object_source_changed)

        self.output_group = QGroupBox("Sparse output")
        output_form = QFormLayout(self.output_group)
        self.checkpoints = QLineEdit()
        self.checkpoints.setPlaceholderText("e.g. 1.0, 5.0 (ms)")
        output_form.addRow("Checkpoints", self.checkpoints)
        self.signal_weighting = QComboBox()
        self.signal_weighting.addItems(
            ["Relative voxel sum", "Physical voxel volume (3D)"]
        )
        output_form.addRow("Signal weighting", self.signal_weighting)
        self.frame_selector = QComboBox()
        self.frame_selector.setEnabled(False)
        self.frame_selector.currentIndexChanged.connect(self._frame_changed)
        output_form.addRow("2D acquisition frame", self.frame_selector)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setTracking(True)
        self.frame_slider.setToolTip(
            "Browse inferred slice, repetition, echo, segment, or partition frames"
        )
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)
        output_form.addRow("Frame index", self.frame_slider)
        self.spectral_point_selector = QSpinBox()
        self.spectral_point_selector.setRange(0, 0)
        self.spectral_point_selector.setEnabled(False)
        self.spectral_point_selector.valueChanged.connect(self._spectral_view_changed)
        self.spectral_point_selector.setToolTip(
            "FID sample shown in spatial k-space and the spatial reconstruction"
        )
        self.spectral_point_slider = QSlider(Qt.Horizontal)
        self.spectral_point_slider.setRange(0, 0)
        self.spectral_point_slider.setEnabled(False)
        self.spectral_point_slider.setTracking(True)
        self.spectral_point_slider.setToolTip(
            "Browse the FID time dimension shown in k-space and image space"
        )
        self.spectral_point_slider.valueChanged.connect(
            self.spectral_point_selector.setValue
        )
        self.spectral_point_selector.valueChanged.connect(
            self.spectral_point_slider.setValue
        )
        spectral_point_control = QWidget()
        spectral_point_layout = QHBoxLayout(spectral_point_control)
        spectral_point_layout.setContentsMargins(0, 0, 0, 0)
        spectral_point_layout.addWidget(self.spectral_point_slider, 1)
        spectral_point_layout.addWidget(self.spectral_point_selector)
        output_form.addRow("CSI FID sample", spectral_point_control)
        controls_layout.addWidget(self.output_group)
        self.output_group.hide()

        self.probe_group = QGroupBox("Spin probes")
        self.probe_group.setCheckable(True)
        self.probe_group.setChecked(False)
        probe_group_layout = QVBoxLayout(self.probe_group)
        self.probe_controls = QWidget()
        probe_form = QFormLayout(self.probe_controls)
        self.probe_points = QSpinBox()
        self.probe_points.setRange(2, 65536)
        self.probe_points.setValue(1024)
        probe_form.addRow("Spectral points", self.probe_points)
        self.probe_frequency_units = QComboBox()
        self.probe_frequency_units.addItems(["Hz", "ppm"])
        self.probe_frequency_units.setCurrentText("Hz")
        self._probe_frequency_unit = "Hz"
        self.probe_frequency_units.currentTextChanged.connect(
            self._probe_frequency_unit_changed
        )
        probe_form.addRow("Frequency units", self.probe_frequency_units)
        self.probe_ppm_min = QDoubleSpinBox()
        self.probe_ppm_min.setRange(-1e7, 1e7)
        self.probe_ppm_min.setDecimals(4)
        self.probe_ppm_min.setValue(-2000.0)
        self.probe_ppm_min.setSuffix(" Hz")
        probe_form.addRow("Frequency min", self.probe_ppm_min)
        self.probe_ppm_max = QDoubleSpinBox()
        self.probe_ppm_max.setRange(-1e7, 1e7)
        self.probe_ppm_max.setDecimals(4)
        self.probe_ppm_max.setValue(2000.0)
        self.probe_ppm_max.setSuffix(" Hz")
        probe_form.addRow("Frequency max", self.probe_ppm_max)
        self.probe_frequency_ppm = QDoubleSpinBox()
        self.probe_frequency_ppm.setRange(-1e7, 1e7)
        self.probe_frequency_ppm.setDecimals(4)
        self.probe_frequency_ppm.setValue(0.0)
        self.probe_frequency_ppm.setSuffix(" Hz")
        self.probe_frequency_ppm.setToolTip("Frequency offset used for geometry probes")
        probe_form.addRow("Single frequency", self.probe_frequency_ppm)
        self.probe_position_x_mm = QDoubleSpinBox()
        self.probe_position_y_mm = QDoubleSpinBox()
        self.probe_position_z_mm = QDoubleSpinBox()
        for spin in (
            self.probe_position_x_mm,
            self.probe_position_y_mm,
            self.probe_position_z_mm,
        ):
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(4)
            spin.setSuffix(" mm")
        probe_form.addRow("Position x", self.probe_position_x_mm)
        probe_form.addRow("Position y", self.probe_position_y_mm)
        probe_form.addRow("Position z", self.probe_position_z_mm)
        self.probe_t1_ms = QDoubleSpinBox()
        self.probe_t1_ms.setRange(0.001, 1e9)
        self.probe_t1_ms.setDecimals(4)
        self.probe_t1_ms.setValue(25000.0)
        self.probe_t1_ms.setSuffix(" ms")
        probe_form.addRow("Probe T1", self.probe_t1_ms)
        self.probe_t2_ms = QDoubleSpinBox()
        self.probe_t2_ms.setRange(0.001, 1e9)
        self.probe_t2_ms.setDecimals(4)
        self.probe_t2_ms.setValue(300.0)
        self.probe_t2_ms.setSuffix(" ms")
        probe_form.addRow("Probe T2/T2*", self.probe_t2_ms)
        self.probe_initial_mz = QDoubleSpinBox()
        self.probe_initial_mz.setRange(0.0, 1e7)
        self.probe_initial_mz.setDecimals(4)
        self.probe_initial_mz.setValue(1.0)
        self.probe_initial_mz.setToolTip(
            "Initial longitudinal magnetization Mz. Values above one can model "
            "hyperpolarized starting magnetization."
        )
        probe_form.addRow("Initial Mz", self.probe_initial_mz)
        self.probe_time_points = QSpinBox()
        self.probe_time_points.setRange(2, 20000)
        self.probe_time_points.setValue(512)
        probe_form.addRow("Time samples", self.probe_time_points)
        self.probe_time_sampling = QComboBox()
        self.probe_time_sampling.addItems(["RF pulse ends", "Uniform timeline"])
        self.probe_time_sampling.setToolTip(
            "RF pulse ends shows the response after each individual RF pulse; "
            "uniform sampling is intended for continuous playback."
        )
        self.probe_time_sampling.currentIndexChanged.connect(
            lambda index: self.probe_time_points.setEnabled(index == 1)
        )
        self.probe_time_points.setEnabled(False)
        probe_form.addRow("Time sampling", self.probe_time_sampling)
        self.probe_max_positions = QSpinBox()
        self.probe_max_positions.setRange(1, 200000)
        self.probe_max_positions.setValue(8192)
        self.probe_max_positions.setToolTip(
            "Maximum active phantom positions used by a geometry probe"
        )
        probe_form.addRow("Max geometry positions", self.probe_max_positions)
        probe_buttons = QWidget()
        probe_button_layout = QHBoxLayout(probe_buttons)
        probe_button_layout.setContentsMargins(0, 0, 0, 0)
        self.run_probe_button = QPushButton("Run spectral probe")
        self.run_probe_button.clicked.connect(self._run_spectral_probe)
        self.run_geometry_probe_button = QPushButton("Run geometry probe")
        self.run_geometry_probe_button.clicked.connect(self._run_geometry_probe)
        self.cancel_probe_button = QPushButton("Cancel")
        self.cancel_probe_button.setEnabled(False)
        self.cancel_probe_button.clicked.connect(self._cancel_probe)
        probe_button_layout.addWidget(self.run_probe_button)
        probe_button_layout.addWidget(self.run_geometry_probe_button)
        probe_button_layout.addWidget(self.cancel_probe_button)
        probe_form.addRow(probe_buttons)
        self.probe_status = QLabel("No spin probe result")
        self.probe_status.setWordWrap(True)
        probe_form.addRow(self.probe_status)
        probe_group_layout.addWidget(self.probe_controls)
        self.probe_controls.setVisible(False)
        self.probe_group.toggled.connect(self.probe_controls.setVisible)
        controls_layout.addWidget(self.probe_group)
        controls_layout.addStretch()

        run_panel = QGroupBox("Run")
        run_panel_layout = QVBoxLayout(run_panel)
        timestep_form = QFormLayout()
        self.simulation_timestep_us = QDoubleSpinBox()
        self.simulation_timestep_us.setObjectName("sequence_simulation_timestep_us")
        self.simulation_timestep_us.setRange(0.1, 1000.0)
        self.simulation_timestep_us.setDecimals(2)
        self.simulation_timestep_us.setSingleStep(0.1)
        self.simulation_timestep_us.setValue(self._initial_sequence_timestep_us)
        self.simulation_timestep_us.setSuffix(" µs")
        self.simulation_timestep_us.setToolTip(
            "Time step while RF is active. Larger values average RF and "
            "simultaneous gradients over fewer intervals and can substantially "
            "reduce runtime. ADC times and event boundaries remain exact."
        )
        timestep_form.addRow("Simulation time step", self.simulation_timestep_us)
        run_panel_layout.addLayout(timestep_form)
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
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Not started")
        run_panel_layout.addWidget(self.progress)
        self.export_button = QPushButton("Export results…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_results)
        run_panel_layout.addWidget(self.export_button)
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

        viewer_column = QWidget()
        viewer_column_layout = QVBoxLayout(viewer_column)
        viewer_column_layout.setContentsMargins(0, 0, 0, 0)
        view_mode_row = QHBoxLayout()
        self.split_view_checkbox = QCheckBox("Split view")
        self.split_view_checkbox.setEnabled(False)
        self.split_view_checkbox.setToolTip(
            "Show image or k-space beside the selected voxel FID or spectrum"
        )
        self.split_view_checkbox.toggled.connect(self._toggle_split_view)
        view_mode_row.addWidget(self.split_view_checkbox)
        view_mode_row.addWidget(QLabel("Left"))
        self.split_image_source = QComboBox()
        self.split_image_source.addItems(["Reconstruction", "K-space"])
        self.split_image_source.setEnabled(False)
        self.split_image_source.currentIndexChanged.connect(self._refresh_split_view)
        view_mode_row.addWidget(self.split_image_source)
        view_mode_row.addWidget(QLabel("Right"))
        self.split_signal_source = QComboBox()
        self.split_signal_source.addItems(["Spectrum", "FID"])
        self.split_signal_source.setEnabled(False)
        self.split_signal_source.currentIndexChanged.connect(self._refresh_split_view)
        view_mode_row.addWidget(self.split_signal_source)
        view_mode_row.addStretch()
        viewer_column_layout.addLayout(view_mode_row)

        self.view_stack = QStackedWidget()
        viewer_column_layout.addWidget(self.view_stack, 1)
        views = QTabWidget()
        self.views = views
        self.view_stack.addWidget(views)
        splitter.addWidget(viewer_column)
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
        self.gradient_plot.setXLink(self.rf_plot)
        self.rf_plot.sigXRangeChanged.connect(self._sequence_plot_range_changed)
        self.rf_progress_cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False, pen=pg.mkPen("y", width=2)
        )
        self.gradient_progress_cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False, pen=pg.mkPen("y", width=2)
        )
        self.rf_progress_cursor.setVisible(self.live_preview_enabled)
        self.gradient_progress_cursor.setVisible(self.live_preview_enabled)
        timeline_layout.addWidget(self.rf_plot)
        timeline_layout.addWidget(self.gradient_plot)
        views.addTab(timeline, "Sequence")

        signal_page = QWidget()
        signal_layout = QVBoxLayout(signal_page)
        spectrum_controls = QHBoxLayout()
        self.spectrum_x_selector = QSpinBox()
        self.spectrum_y_selector = QSpinBox()
        self.spectrum_x_slider = QSlider(Qt.Horizontal)
        self.spectrum_y_slider = QSlider(Qt.Horizontal)
        for axis, selector, slider in (
            ("x", self.spectrum_x_selector, self.spectrum_x_slider),
            ("y", self.spectrum_y_selector, self.spectrum_y_slider),
        ):
            selector.setRange(0, 0)
            selector.setEnabled(False)
            selector.valueChanged.connect(self._spectral_view_changed)
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.setTracking(True)
            slider.setMinimumWidth(100)
            tooltip = f"Browse the reconstructed CSI voxel {axis} coordinate"
            selector.setToolTip(tooltip)
            slider.setToolTip(tooltip)
            slider.valueChanged.connect(selector.setValue)
            selector.valueChanged.connect(slider.setValue)
        spectrum_controls.addWidget(QLabel("CSI voxel x"))
        spectrum_controls.addWidget(self.spectrum_x_slider, 1)
        spectrum_controls.addWidget(self.spectrum_x_selector)
        spectrum_controls.addWidget(QLabel("y"))
        spectrum_controls.addWidget(self.spectrum_y_slider, 1)
        spectrum_controls.addWidget(self.spectrum_y_selector)
        spectrum_controls.addStretch()
        signal_layout.addLayout(spectrum_controls)
        self.signal_plot = pg.PlotWidget(title="Received ADC signal")
        self.signal_plot.setLabel("left", "Signal", "a.u.")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        self.signal_plot.addLegend()
        signal_layout.addWidget(self.signal_plot)
        self.spectrum_info = QLabel("No spectroscopic result")
        signal_layout.addWidget(self.spectrum_info)
        views.addTab(signal_page, "Signal / CSI spectrum")

        kspace_page = QWidget()
        kspace_layout = QVBoxLayout(kspace_page)
        self.kspace_view = pg.ImageView()
        self.kspace_view.ui.roiBtn.hide()
        self.kspace_view.ui.menuBtn.hide()
        self._format_colorbar(self.kspace_view)
        kspace_layout.addWidget(self.kspace_view)
        self.kspace_zoom_info = QLabel("Zoom: —")
        kspace_layout.addWidget(self.kspace_zoom_info)
        self.kspace_info = QLabel("No 2D Cartesian result")
        kspace_layout.addWidget(self.kspace_info)
        views.addTab(kspace_page, "2D k-space")

        reconstruction_page = QWidget()
        reconstruction_layout = QVBoxLayout(reconstruction_page)
        self.reconstruction_view = pg.ImageView()
        self.reconstruction_view.ui.roiBtn.hide()
        self.reconstruction_view.ui.menuBtn.hide()
        self._format_colorbar(self.reconstruction_view)
        reconstruction_layout.addWidget(self.reconstruction_view)
        self.reconstruction_zoom_info = QLabel("Zoom: —")
        reconstruction_layout.addWidget(self.reconstruction_zoom_info)
        self.reconstruction_info = QLabel("No 2D Cartesian result")
        reconstruction_layout.addWidget(self.reconstruction_info)
        views.addTab(reconstruction_page, "2D Reconstruction")

        state_page = QWidget()
        state_layout = QVBoxLayout(state_page)
        self.state_view = pg.ImageView()
        self.state_view.ui.roiBtn.hide()
        self.state_view.ui.menuBtn.hide()
        self._format_colorbar(self.state_view)
        state_layout.addWidget(self.state_view)
        self.state_zoom_info = QLabel("Zoom: —")
        state_layout.addWidget(self.state_zoom_info)
        self.state_info = QLabel("No result")
        state_layout.addWidget(self.state_info)
        views.addTab(state_page, "Final Mz")

        self.result_volume_viewer = SequenceResultVolumeViewer()
        views.addTab(self.result_volume_viewer, "Spatial Magnetization")

        probe_page = QWidget()
        probe_layout = QVBoxLayout(probe_page)
        self.probe_info = QLabel("Run a spin probe to populate these views")
        self.probe_info.setWordWrap(True)
        probe_layout.addWidget(self.probe_info)
        self.probe_coherence_info = QLabel()
        self.probe_coherence_info.setWordWrap(True)
        probe_layout.addWidget(self.probe_coherence_info)
        self.probe_time_control = UniversalTimeControl()
        self.probe_time_control.setObjectName("sequence_probe_playback_control")
        self.probe_time_control.setEnabled(False)
        self.probe_time_control.time_changed.connect(self._set_probe_time_index)
        self.probe_time_control.play_pause_button.toggled.connect(
            self._probe_playback_toggled
        )
        self.probe_time_control.reset_button.clicked.connect(self._reset_probe_playback)
        self.probe_time_control.speed_spin.valueChanged.connect(
            self._probe_playback_speed_changed
        )
        probe_layout.addWidget(self.probe_time_control)
        self.probe_views = QTabWidget()
        self.probe_spectrum_viewer = SequenceProbeSpectrumViewer()
        self.probe_spatial_viewer = SequenceProbeSpatialViewer()
        self.probe_magnetization_viewer = MagnetizationViewer()
        self.probe_magnetization_viewer.export_3d_btn.setVisible(False)
        self.probe_magnetization_viewer.position_changed.connect(
            self._update_probe_vector
        )
        self.probe_magnetization_viewer.view_filter_changed.connect(
            self._update_probe_vector
        )
        self.probe_views.addTab(self.probe_spectrum_viewer, "Spectrum")
        self.probe_views.addTab(self.probe_spatial_viewer, "Spatial")
        self.probe_views.addTab(self.probe_magnetization_viewer, "3D Vector")
        self.probe_views.currentChanged.connect(self._probe_view_changed)
        probe_layout.addWidget(self.probe_views, 1)
        views.addTab(probe_page, "Spin Probe")

        split_page = QWidget()
        split_page_layout = QVBoxLayout(split_page)
        split_views = QSplitter(Qt.Horizontal)
        split_page_layout.addWidget(split_views, 1)

        split_image_panel = QWidget()
        split_image_layout = QVBoxLayout(split_image_panel)
        self.split_image_plot = pg.PlotWidget(title="CSI reconstruction")
        self.split_image_plot.setAspectLocked(True)
        self.split_image_plot.setLabel("bottom", "x index")
        self.split_image_plot.setLabel("left", "y index")
        self.split_image_item = pg.ImageItem()
        self.split_image_plot.addItem(self.split_image_item)
        self.split_voxel_marker = pg.ScatterPlotItem(
            size=15,
            symbol="s",
            pen=pg.mkPen("y", width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.split_image_plot.addItem(self.split_voxel_marker)
        self.split_image_plot.scene().sigMouseClicked.connect(self._split_image_clicked)
        split_image_layout.addWidget(self.split_image_plot, 1)
        self.split_image_info = QLabel("Run a CSI simulation to populate this view")
        self.split_image_info.setWordWrap(True)
        split_image_layout.addWidget(self.split_image_info)
        split_views.addWidget(split_image_panel)

        split_signal_panel = QWidget()
        split_signal_layout = QVBoxLayout(split_signal_panel)
        self.split_signal_plot = pg.PlotWidget(title="Voxel spectrum")
        self.split_signal_plot.setLabel("left", "Signal", "a.u.")
        self.split_signal_plot.setLabel("bottom", "Frequency", "Hz")
        self.split_signal_plot.addLegend()
        split_signal_layout.addWidget(self.split_signal_plot, 1)
        self.split_signal_info = QLabel("Select a voxel in the image")
        self.split_signal_info.setWordWrap(True)
        split_signal_layout.addWidget(self.split_signal_info)
        split_views.addWidget(split_signal_panel)
        split_views.setSizes([1, 1])
        self.view_stack.addWidget(split_page)

        for view, label in (
            (self.kspace_view, self.kspace_zoom_info),
            (self.reconstruction_view, self.reconstruction_zoom_info),
            (self.state_view, self.state_zoom_info),
        ):
            view.getView().sigRangeChanged.connect(
                lambda *_, image_view=view, zoom_label=label: self._update_zoom_label(
                    image_view, zoom_label
                )
            )

    @staticmethod
    def _format_colorbar(view):
        view.ui.histogram.axis.tickStrings = lambda values, scale, spacing: [
            f"{value * scale:.2f}" for value in values
        ]

    @staticmethod
    def _update_zoom_label(view, label):
        image = getattr(view, "image", None)
        if image is None or np.asarray(image).ndim < 2:
            label.setText("Zoom: —")
            return
        x_range, y_range = view.getView().viewRange()
        visible_x = max(float(x_range[1] - x_range[0]), 1e-12)
        visible_y = max(float(y_range[1] - y_range[0]), 1e-12)
        shape = np.asarray(image).shape
        zoom = max(1.0, min(shape[0] / visible_x, shape[1] / visible_y))
        label.setText(f"Zoom: {zoom:.2f}×")

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
        self.built_in_properties_group.setVisible(not phantom_selected)
        self.built_in_properties_group.setEnabled(not phantom_selected)
        self.phantom_summary.setVisible(phantom_selected)
        self.open_phantom_button.setVisible(phantom_selected)
        self.refresh_object_summary()
        self._update_frequency_reference_info()
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
        if isinstance(phantom, DynamicSpectralPhantom):
            self.field_strength_t.setValue(phantom.field_strength)
            nucleus_index = self.nucleus.findText(phantom.nucleus)
            if nucleus_index >= 0:
                self.nucleus.setCurrentIndex(nucleus_index)
        if isinstance(phantom, (SpectralPhantom, DynamicSpectralPhantom)):
            self._apply_probe_defaults_from_phantom(phantom)
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
        self._update_frequency_reference_info()
        if self.sequence_source.currentIndex() == 1:
            self._load_cartesian_epi()

    def _update_frequency_reference_info(self):
        if self.object_source.currentIndex() != 0:
            text = "Built-in B0 and chemical-shift values are entered in ppm."
            conversion_enabled = True
        else:
            phantom = self._selected_designed_phantom()
            if isinstance(phantom, (SpectralPhantom, DynamicSpectralPhantom)):
                text = (
                    "Spectral B0 and peak offsets are converted from ppm at run time; "
                    "the sequence carrier is 0 ppm and phantom peaks are offsets "
                    "from the phantom spectral reference."
                )
                conversion_enabled = True
            elif phantom is not None:
                text = (
                    "This conventional phantom stores fixed frequency maps in Hz; "
                    "field/nucleus conversion applies only to ppm spectral designs."
                )
                conversion_enabled = False
            else:
                text = "ppm conversion applies after a spectral phantom is selected."
                conversion_enabled = False
        self.field_strength_t.setEnabled(conversion_enabled)
        self.nucleus.setEnabled(conversion_enabled)
        self.frequency_reference_info.setText(text)

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
        self.acquisition_group.setVisible(epi_selected)
        self.acquisition_group.setEnabled(epi_selected)
        self.acquisition_hint.setText(
            "Read/phase matrix and sampling bandwidth define a 2D kx-ky "
            "acquisition. No kz encoding is performed."
            if epi_selected
            else "Select EPI under Source / mode to enable these settings."
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
        self.acquisition_frames = None
        self.spectroscopic_acquisition = None
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
        self._apply_probe_defaults_from_program()
        self._configure_frame_selector()
        self._configure_spectroscopy_selectors()
        self._show_program()

    def _load_cartesian_epi(self):
        self.acquisition_frames = None
        self.spectroscopic_acquisition = None
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
            self._configure_frame_selector()
            self._configure_spectroscopy_selectors()
            self._show_program()
        except Exception as exc:
            self.acquisition = None
            self.program = None
            self.sequence_info.setText(f"Invalid Cartesian acquisition: {exc}")

    def _load_pulseq_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pulseq sequence",
            str(workspace_directory("sequences")),
            "Pulseq sequence (*.seq);;All files (*)",
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
        self.acquisition_frames = None
        self.spectroscopic_acquisition = None
        try:
            self.spectroscopic_acquisition = infer_spectroscopic_acquisition(
                self.program, compiled=compiled
            )
            self.acquisition = None
            csi = self.spectroscopic_acquisition
            self.acquisition_note = (
                f"2D CSI {csi.matrix[0]}×{csi.matrix[1]}×"
                f"{csi.spectral_points}; BW={csi.spectral_bandwidth_hz:.6g} Hz"
            )
        except ValueError as spectroscopy_error:
            try:
                self.acquisition = infer_cartesian_acquisition(
                    self.program, compiled=compiled
                )
                self.acquisition_note = (
                    "Cartesian acquisition inferred from ADC moments"
                )
            except ValueError as single_error:
                try:
                    self.acquisition_frames = infer_cartesian_acquisition_frames(
                        self.program, compiled=compiled
                    )
                    self.acquisition = self.acquisition_frames.acquisitions[0]
                    metadata = dict(self.program.metadata)
                    metadata["acquisition_dimensions"] = (
                        self.acquisition_frames.dimensions.to_metadata()
                    )
                    self.program = SequenceProgram(
                        events=self.program.events,
                        duration_s=self.program.duration_s,
                        source=self.program.source,
                        version=self.program.version,
                        metadata=metadata,
                    )
                    axes = ", ".join(self.acquisition_frames.varying_axes)
                    self.acquisition_note = (
                        f"{self.acquisition_frames.num_frames} Cartesian 2D frames "
                        f"inferred ({axes})"
                    )
                except ValueError as frame_error:
                    self.acquisition = None
                    self.acquisition_note = (
                        f"Acquisition inference unavailable: CSI: {spectroscopy_error}; "
                        f"Cartesian: {single_error}; frames: {frame_error}"
                    )
        self._apply_pulseq_fov()
        self._apply_pulseq_frequency_reference()
        self._apply_probe_defaults_from_program()
        self.sequence_source.setCurrentIndex(2)
        self._show_program()
        self._configure_frame_selector()
        self._configure_spectroscopy_selectors()
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

    def _apply_pulseq_frequency_reference(self):
        definitions = {
            str(key).lower(): value
            for key, value in self.program.metadata.get("definitions", {}).items()
        }
        field_value = definitions.get("fieldstrengtht")
        try:
            field_strength = float(np.asarray(field_value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            field_strength = np.nan
        if np.isfinite(field_strength) and field_strength > 0:
            self.field_strength_t.setValue(field_strength)
        nucleus_value = definitions.get("nucleus")
        if nucleus_value is not None:
            nucleus = str(nucleus_value).strip()
            nucleus_index = self.nucleus.findText(nucleus)
            if nucleus_index >= 0:
                self.nucleus.setCurrentIndex(nucleus_index)

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
            if self.acquisition_frames is not None:
                acquisition_text += (
                    f"; frames={self.acquisition_frames.num_frames} "
                    f"({', '.join(self.acquisition_frames.varying_axes)})"
                )
        elif self.spectroscopic_acquisition is not None:
            csi = self.spectroscopic_acquisition
            acquisition_text = (
                f"\nCSI grid: {csi.matrix[1]}×{csi.matrix[0]}; "
                f"spectral points={csi.spectral_points}; "
                f"BW={csi.spectral_bandwidth_hz:.6g} Hz; "
                f"resolution={csi.spectral_resolution_hz:.6g} Hz"
            )
        elif self.acquisition_note:
            acquisition_text = f"\n{self.acquisition_note}"
        definitions = dict(self.program.metadata.get("definitions", {}))
        spoiler_end_times = np.asarray(
            definitions.get("EndImageSpoilerEndTimes", ()), dtype=float
        ).reshape(-1)
        spoiler_end_times = spoiler_end_times[np.isfinite(spoiler_end_times)]
        spoiler_text = ""
        if spoiler_end_times.size:
            cycles = definitions.get("EndImageSpoilerCyclesPerFOV", "?")
            axes = definitions.get("EndImageSpoilerAxes", "xyz")
            spoiler_text = (
                f"\nEnd-image spoilers: {spoiler_end_times.size}; "
                f"{cycles} cycles/FOV on {axes}"
            )
        self.sequence_info.setText(
            f"{self.program.source}\nDuration: {self.program.duration_s*1000:.3f} ms\n"
            f"Events: {len(self.program.events)}, intervals: {compiled.n_intervals}, "
            f"ADC samples: {compiled.adc_times_s.size}{acquisition_text}{spoiler_text}"
        )
        self.rf_plot.clear()
        self.gradient_plot.clear()
        self._rf_waveform_item = self.rf_plot.plot(
            np.empty(0), np.empty(0), pen=pg.mkPen("m", width=1.5)
        )
        self._gradient_waveform_items = {
            axis: self.gradient_plot.plot(
                np.empty(0),
                np.empty(0),
                pen=pg.mkPen(color, width=1.25),
                name=f"G{axis}",
            )
            for axis, color in zip("xyz", ("r", "g", "b"))
        }
        if compiled.adc_times_s.size:
            self.gradient_plot.plot(
                compiled.adc_times_s * 1000,
                np.zeros_like(compiled.adc_times_s),
                pen=None,
                symbol="o",
                symbolSize=3,
                symbolBrush=pg.mkBrush(255, 230, 0, 180),
                name="ADC",
            )
        self._sequence_spoiler_markers = []
        for spoiler_end in spoiler_end_times:
            marker = pg.InfiniteLine(
                pos=float(spoiler_end) * 1000.0,
                angle=90,
                movable=False,
                pen=pg.mkPen("#ff9800", width=1.5, style=Qt.DashLine),
            )
            marker.setToolTip("End-image spoiler")
            self.gradient_plot.addItem(marker)
            self._sequence_spoiler_markers.append(marker)
        self.rf_plot.addItem(self.rf_progress_cursor)
        self.gradient_plot.addItem(self.gradient_progress_cursor)
        self._sequence_plot_window_s = None
        duration = max(float(self.program.duration_s), 1e-9)
        self.rf_plot.setXRange(0.0, duration * 1000.0, padding=0)
        self._refresh_sequence_waveforms(0.0, duration)
        if self.program.rf_events:
            rf_peak = max(
                float(np.max(np.abs(event.samples_hz)))
                for event in self.program.rf_events
            )
            self.rf_plot.setYRange(0.0, max(rf_peak * 1.05, 1e-9), padding=0)
        gradient_values = [
            np.asarray(event.samples_hz_per_m, dtype=float) * 1e-3
            for event in self.program.gradient_events
        ]
        if gradient_values:
            gradient_limit = max(
                float(np.max(np.abs(values))) for values in gradient_values
            )
            self.gradient_plot.setYRange(
                -max(gradient_limit * 1.05, 1e-9),
                max(gradient_limit * 1.05, 1e-9),
                padding=0,
            )
        self._set_sequence_cursor(0.0)

    def _sequence_plot_range_changed(self, _view_box, x_range):
        if self.program is None or x_range is None or len(x_range) != 2:
            return
        start_s = max(0.0, float(x_range[0]) / 1000.0)
        end_s = min(float(self.program.duration_s), float(x_range[1]) / 1000.0)
        if end_s <= start_s:
            return
        self._sequence_plot_pending_window_s = (start_s, end_s)
        self.sequence_plot_refresh_timer.start()

    def _refresh_pending_sequence_plot(self):
        if self._sequence_plot_pending_window_s is None:
            return
        start_s, end_s = self._sequence_plot_pending_window_s
        self._sequence_plot_pending_window_s = None
        self._refresh_sequence_waveforms(start_s, end_s)

    def _refresh_sequence_waveforms(self, start_s, end_s):
        if self.program is None or self._rf_waveform_item is None:
            return
        window = (float(start_s), float(end_s))
        if self._sequence_plot_window_s is not None and np.allclose(
            window, self._sequence_plot_window_s, rtol=0.0, atol=1e-12
        ):
            return
        self._sequence_plot_window_s = window
        rf_x, rf_y = _event_step_plot_data(
            self.program.rf_events,
            samples_attribute="samples_hz",
            start_s=window[0],
            end_s=window[1],
            magnitude=True,
        )
        self._rf_waveform_item.setData(rf_x, rf_y, connect="finite")
        gradients = self.program.gradient_events
        for axis in "xyz":
            events = tuple(event for event in gradients if event.axis == axis)
            grad_x, grad_y = _event_step_plot_data(
                events,
                samples_attribute="samples_hz_per_m",
                start_s=window[0],
                end_s=window[1],
                scale=1e-3,
            )
            self._gradient_waveform_items[axis].setData(
                grad_x, grad_y, connect="finite"
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
            b0_map=np.where(
                mask,
                ppm_to_hz(
                    self.b0_ppm.value(),
                    self.field_strength_t.value(),
                    self.nucleus.currentText(),
                ),
                0.0,
            ),
            chemical_shift_map=np.where(
                mask,
                ppm_to_hz(
                    self.chemical_ppm.value(),
                    self.field_strength_t.value(),
                    self.nucleus.currentText(),
                ),
                0.0,
            ),
            mask=mask,
            name="Sequence simulation object",
            metadata={
                "field_strength_t": self.field_strength_t.value(),
                "nucleus": self.nucleus.currentText(),
                "b0_inhomogeneity_ppm": self.b0_ppm.value(),
                "chemical_shift_ppm": self.chemical_ppm.value(),
            },
        )

    def _checkpoint_seconds(self):
        text = self.checkpoints.text().strip()
        if not text:
            return ()
        values = tuple(float(value.strip()) / 1000.0 for value in text.split(","))
        return values

    def _probe_hz_per_ppm(self):
        return float(
            ppm_to_hz(
                1.0,
                self.field_strength_t.value(),
                self.nucleus.currentText(),
            )
        )

    def _probe_frequency_unit_changed(self, unit):
        unit = str(unit)
        previous = getattr(self, "_probe_frequency_unit", unit)
        if unit == previous:
            return
        factor = self._probe_hz_per_ppm()
        conversion = 1.0 / factor if previous == "Hz" and unit == "ppm" else factor
        widgets = (self.probe_ppm_min, self.probe_ppm_max, self.probe_frequency_ppm)
        for spin in widgets:
            spin.blockSignals(True)
            spin.setValue(float(spin.value()) * conversion)
            spin.setSuffix(f" {unit}")
            spin.blockSignals(False)
        self._probe_frequency_unit = unit

    def _apply_probe_defaults_from_program(self):
        if self.program is None or not self.program.rf_events:
            return
        definitions = dict(self.program.metadata.get("definitions", {}))
        centres = np.asarray(
            [event.frequency_offset_hz for event in self.program.rf_events],
            dtype=float,
        )
        bandwidth_candidates = []
        for key in ("SpectralRFFWHM", "SpectralRFBandwidthHz"):
            value = definitions.get(key)
            try:
                bandwidth = float(np.asarray(value).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(bandwidth) and bandwidth > 0:
                bandwidth_candidates.append(bandwidth)
        margin = max([500.0, *bandwidth_candidates])
        self.probe_frequency_units.setCurrentText("Hz")
        self.probe_ppm_min.setValue(float(np.min(centres) - margin))
        self.probe_ppm_max.setValue(float(np.max(centres) + margin))

    def _probe_frequency_axis_hz(self):
        points = int(self.probe_points.value())
        frequency_min = float(self.probe_ppm_min.value())
        frequency_max = float(self.probe_ppm_max.value())
        if frequency_min >= frequency_max:
            raise ValueError("probe frequency min must be smaller than max")
        display_axis = np.linspace(frequency_min, frequency_max, points)
        if self.probe_frequency_units.currentText() == "Hz":
            hz_axis = display_axis
        else:
            hz_axis = ppm_to_hz(
                display_axis,
                self.field_strength_t.value(),
                self.nucleus.currentText(),
            )
        return display_axis, np.asarray(hz_axis, dtype=float)

    def _probe_single_frequency_axis_hz(self):
        display_axis = np.asarray(
            [float(self.probe_frequency_ppm.value())], dtype=float
        )
        if self.probe_frequency_units.currentText() == "Hz":
            hz_axis = display_axis
        else:
            hz_axis = ppm_to_hz(
                display_axis,
                self.field_strength_t.value(),
                self.nucleus.currentText(),
            )
        return display_axis, np.asarray(hz_axis, dtype=float)

    def _probe_checkpoints_s(self):
        if self.program is None:
            raise ValueError("Choose or load a sequence first")
        if self.probe_time_sampling.currentIndex() == 0:
            checkpoints = [0.0]
            checkpoints.extend(event.end_s for event in self.program.rf_events)
            definitions = dict(self.program.metadata.get("definitions", {}))
            spoiler_times = definitions.get("EndImageSpoilerEndTimes", ())
            try:
                checkpoints.extend(np.asarray(spoiler_times, dtype=float).reshape(-1))
            except (TypeError, ValueError):
                pass
            checkpoints.append(float(self.program.duration_s))
            return np.unique(
                np.clip(
                    np.asarray(checkpoints, dtype=float), 0.0, self.program.duration_s
                )
            )
        return np.linspace(
            0.0,
            float(self.program.duration_s),
            int(self.probe_time_points.value()),
        )

    def _probe_position_m(self):
        return np.asarray(
            [
                [
                    self.probe_position_x_mm.value() / 1000.0,
                    self.probe_position_y_mm.value() / 1000.0,
                    self.probe_position_z_mm.value() / 1000.0,
                ]
            ],
            dtype=float,
        )

    def _probe_geometry_positions_m(self):
        if self.phantom is None:
            raise ValueError(
                "Choose or build a phantom before running a geometry probe"
            )
        positions = np.asarray(getattr(self.phantom, "positions", ()), dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("current phantom does not expose 3D spin positions")
        mask = np.asarray(
            getattr(self.phantom, "mask", np.ones(positions.shape[0], dtype=bool)),
            dtype=bool,
        ).ravel()
        if mask.size == positions.shape[0]:
            positions = positions[mask]
        if positions.shape[0] == 0:
            raise ValueError("current phantom contains no active spin positions")
        max_positions = int(self.probe_max_positions.value())
        if positions.shape[0] > max_positions:
            indices = np.linspace(0, positions.shape[0] - 1, max_positions, dtype=int)
            positions = positions[indices]
        return positions

    def _apply_probe_defaults_from_phantom(self, phantom):
        if not isinstance(phantom, (SpectralPhantom, DynamicSpectralPhantom)):
            return
        half_bandwidth = float(phantom.spectral_bandwidth_ppm) / 2.0
        if np.isfinite(half_bandwidth) and half_bandwidth > 0:
            if self.probe_frequency_units.currentText() == "Hz":
                half_bandwidth *= self._probe_hz_per_ppm()
            self.probe_ppm_min.setValue(-half_bandwidth)
            self.probe_ppm_max.setValue(half_bandwidth)
        points = int(getattr(phantom, "spectral_points", self.probe_points.value()))
        if self.probe_points.minimum() <= points <= self.probe_points.maximum():
            self.probe_points.setValue(points)

    def _can_start_probe(self):
        if self.program is None:
            QMessageBox.warning(self, "No sequence", "Choose or load a sequence first.")
            return False
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Simulation running",
                "Wait for the sequence simulation to finish before running a probe.",
            )
            return False
        if self.probe_worker is not None and self.probe_worker.isRunning():
            QMessageBox.warning(
                self, "Probe running", "Cancel the current probe first."
            )
            return False
        return True

    def _run_spectral_probe(self):
        if not self._can_start_probe():
            return
        try:
            display_axis, hz_axis = self._probe_frequency_axis_hz()
            checkpoints = self._probe_checkpoints_s()
            positions = self._probe_position_m()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid spin probe", str(exc))
            return
        self._start_probe(positions, hz_axis, display_axis, checkpoints, "spectral")

    def _run_geometry_probe(self):
        if not self._can_start_probe():
            return
        try:
            display_axis, hz_axis = self._probe_single_frequency_axis_hz()
            checkpoints = self._probe_checkpoints_s()
            positions = self._probe_geometry_positions_m()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid geometry probe", str(exc))
            return
        self._start_probe(positions, hz_axis, display_axis, checkpoints, "geometry")

    def _start_probe(self, positions, hz_axis, display_axis, checkpoints, label):
        self._stop_probe_playback()
        self.probe_time_control.setEnabled(False)
        self.probe_time_control.set_time_range(None)
        self.run_probe_button.setEnabled(False)
        self.run_geometry_probe_button.setEnabled(False)
        self.cancel_probe_button.setEnabled(True)
        self.probe_status.setText(f"Preparing {label} probe…")
        self.probe_result = None
        self.probe_worker = SequenceProbeThread(
            self.simulator,
            self.program,
            positions,
            hz_axis,
            checkpoints,
            self.probe_t1_ms.value() / 1000.0,
            self.probe_t2_ms.value() / 1000.0,
            initial_magnetization=(0.0, 0.0, self.probe_initial_mz.value()),
            simulation_timestep_s=self.simulation_timestep_us.value() * 1e-6,
        )
        self.probe_worker.stage.connect(self._probe_status_update)
        self.probe_worker.progress.connect(self._probe_progress)
        self.probe_worker.result_ready.connect(
            lambda result, axis=display_axis, unit=self.probe_frequency_units.currentText(), mode=label: self._probe_finished(
                result, axis, unit, mode
            )
        )
        self.probe_worker.failed.connect(self._probe_failed)
        self.probe_worker.start()

    def _cancel_probe(self):
        if self.probe_worker is not None:
            self.probe_worker.request_cancel()
            self.probe_status.setText("Cancelling spin probe…")

    def _probe_progress(self, done, total):
        self.probe_status.setText(f"Probe chunk {done}/{total}")

    def _probe_status_update(self, message):
        message = str(message)
        self.probe_status.setText(message)
        logger = getattr(self.window(), "log_message", None)
        if callable(logger):
            logger(f"Sequence probe: {message}")

    def _probe_failed(self, message):
        self.run_probe_button.setEnabled(True)
        self.run_geometry_probe_button.setEnabled(True)
        self.cancel_probe_button.setEnabled(False)
        self.probe_status.setText(message)
        if message != "Probe simulation cancelled":
            QMessageBox.critical(self, "Spin probe failed", message)

    def _probe_finished(self, result, frequency_axis, frequency_unit, mode=None):
        self._stop_probe_playback()
        self.probe_result = result
        frequency_axis = np.asarray(frequency_axis, dtype=float)
        result.metadata["frequency_axis_unit"] = str(frequency_unit)
        result.metadata[f"frequency_offsets_{str(frequency_unit).lower()}"] = (
            frequency_axis
        )
        if mode is not None:
            result.metadata["ui_probe_mode"] = str(mode)
        self.run_probe_button.setEnabled(True)
        self.run_geometry_probe_button.setEnabled(True)
        self.cancel_probe_button.setEnabled(False)
        self.probe_status.setText("Spin probe complete")
        self._show_probe_result()

    def _show_probe_result(self):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            return
        time_ms = result.time_s * 1000.0
        probe_type = str(result.metadata.get("probe_type", "grid"))
        if probe_type == "spectral":
            info = (
                f"Spectral probe: {result.frequency_offsets_hz.size} frequencies, "
                f"{result.time_s.size} time samples, "
                f"{self.field_strength_t.value():.4g} T {self.nucleus.currentText()}"
            )
        elif probe_type == "geometry":
            info = (
                f"Geometry probe: {result.positions_m.shape[0]} positions, "
                f"{result.time_s.size} time samples, "
                f"frequency {result.frequency_offsets_hz[0]:.5g} Hz"
            )
        else:
            info = (
                f"Spin probe grid: {result.positions_m.shape[0]} positions, "
                f"{result.frequency_offsets_hz.size} frequencies, "
                f"{result.time_s.size} time samples"
            )
        self.probe_info.setText(info)
        self.probe_time_control.setEnabled(True)
        self.probe_time_control.set_time_range(result.time_s)
        self.probe_spectrum_viewer.set_result(result)
        self.probe_spatial_viewer.set_result(result)
        self.probe_magnetization_viewer.last_positions = result.positions_m
        self.probe_magnetization_viewer.last_frequencies = result.frequency_offsets_hz
        self.probe_magnetization_viewer.set_selector_limits(
            result.positions_m.shape[0],
            result.frequency_offsets_hz.size,
            disable=False,
        )
        mean = result.magnetization.mean(axis=(1, 2))
        self.probe_magnetization_viewer.set_length_scale(
            max(1e-9, float(np.nanmax(np.linalg.norm(result.magnetization, axis=-1))))
        )
        self.probe_magnetization_viewer.set_preview_data(
            time_ms,
            mean[:, 0],
            mean[:, 1],
            mean[:, 2],
        )
        initial_index = 1 if result.time_s.size > 1 else 0
        self.probe_time_control.set_time_index(initial_index)
        self._update_probe_vector(initial_index)

    def _set_probe_time_index(self, time_index):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            return
        time_index = int(np.clip(time_index, 0, result.time_s.size - 1))
        self.probe_time_control.set_time_index(time_index)
        self._update_probe_vector(time_index)
        if self.probe_playback_timer.isActive():
            self._reset_probe_playback_anchor(time_index)

    def _reset_probe_playback_anchor(self, time_index=None):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            self._probe_playback_anchor_wall = None
            self._probe_playback_anchor_time_ms = None
            return
        if time_index is None:
            time_index = self.probe_time_control.time_slider.value()
        time_index = int(np.clip(time_index, 0, result.time_s.size - 1))
        self._probe_playback_anchor_wall = time.monotonic()
        self._probe_playback_anchor_time_ms = float(result.time_s[time_index] * 1000.0)

    def _probe_playback_toggled(self, playing):
        result = self.probe_result
        if not playing:
            self._stop_probe_playback()
            return
        if result is None or result.time_s.size < 2:
            self._stop_probe_playback()
            return

        time_index = self.probe_time_control.time_slider.value()
        if time_index >= result.time_s.size - 1:
            time_index = 0
            self.probe_magnetization_viewer._clear_path()
            self._set_probe_time_index(time_index)
        self._reset_probe_playback_anchor(time_index)
        self.probe_playback_timer.start()

    def _stop_probe_playback(self):
        self.probe_playback_timer.stop()
        self._probe_playback_anchor_wall = None
        self._probe_playback_anchor_time_ms = None
        if hasattr(self, "probe_time_control"):
            self.probe_time_control.sync_play_state(False)

    def _reset_probe_playback(self):
        self._stop_probe_playback()
        if self.probe_result is None or self.probe_result.time_s.size == 0:
            return
        self.probe_magnetization_viewer._clear_path()
        self._set_probe_time_index(0)

    def _probe_playback_speed_changed(self, _speed):
        if self.probe_playback_timer.isActive():
            self._reset_probe_playback_anchor()

    def _advance_probe_playback(self):
        result = self.probe_result
        if result is None or result.time_s.size < 2:
            self._stop_probe_playback()
            return
        if (
            self._probe_playback_anchor_wall is None
            or self._probe_playback_anchor_time_ms is None
        ):
            self._reset_probe_playback_anchor()
            return

        time_ms = np.asarray(result.time_s, dtype=float) * 1000.0
        start_ms = float(time_ms[0])
        end_ms = float(time_ms[-1])
        duration_ms = end_ms - start_ms
        if not np.isfinite(duration_ms) or duration_ms <= 0:
            self._stop_probe_playback()
            return

        now = time.monotonic()
        elapsed_s = max(0.0, now - self._probe_playback_anchor_wall)
        target_ms = self._probe_playback_anchor_time_ms + elapsed_s * max(
            float(self.probe_time_control.speed_spin.value()), 0.001
        )
        if target_ms > end_ms:
            target_ms = start_ms + (target_ms - start_ms) % duration_ms
            self._probe_playback_anchor_wall = now
            self._probe_playback_anchor_time_ms = target_ms
            self.probe_magnetization_viewer._clear_path()

        time_index = int(np.searchsorted(time_ms, target_ms, side="left"))
        time_index = min(max(time_index, 0), time_ms.size - 1)
        if time_index > 0 and abs(target_ms - time_ms[time_index - 1]) < abs(
            time_ms[time_index] - target_ms
        ):
            time_index -= 1
        if time_index == self.probe_time_control.time_slider.value():
            return
        self.probe_time_control.set_time_index(time_index)
        self._update_probe_vector(time_index)

    def _probe_view_changed(self, _index):
        if self.probe_result is None or self.probe_result.time_s.size == 0:
            return
        self._update_probe_vector(self.probe_time_control.time_slider.value())

    def _update_probe_vector(self, time_index=None):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            return
        if time_index is None:
            time_index = self.probe_magnetization_viewer.time_slider.value()
        time_index = int(np.clip(time_index, 0, result.time_s.size - 1))
        coherent = result.coherent_mxy_magnitude[time_index]
        mean_spin_magnitude = np.mean(np.abs(result.mxy[time_index]), axis=0)
        if result.positions_m.shape[0] == 1:
            self.probe_coherence_info.setText(
                "Single-position probe: a gradient spoiler changes phase but cannot "
                "reduce the spin's |Mxy|; at position (0, 0, 0) even the gradient "
                "phase is zero. Use Run geometry probe to inspect coherent spoiling."
            )
        else:
            self.probe_coherence_info.setText(
                "Coherent ensemble |mean(Mxy)|: "
                f"{float(np.mean(coherent)):.5g}; mean individual |Mxy|: "
                f"{float(np.mean(mean_spin_magnitude)):.5g}. "
                "A working spoiler lowers the coherent value, not each spin magnitude."
            )
        self.probe_magnetization_viewer.set_cursor_index(time_index)
        self.probe_spectrum_viewer.time_index = time_index
        self.probe_spatial_viewer.time_index = time_index

        current_view = self.probe_views.currentIndex()
        if current_view == 0:
            self.probe_spectrum_viewer.refresh()
            return
        if current_view == 1:
            self.probe_spatial_viewer.refresh()
            return

        frame = result.magnetization[time_index]
        mode = self.probe_magnetization_viewer.get_view_mode()
        selector = self.probe_magnetization_viewer.get_selector_index()
        if mode == "Positions @ freq":
            index = min(selector, frame.shape[1] - 1)
            vectors = frame[:, index, :]
        elif mode == "Freqs @ position":
            index = min(selector, frame.shape[0] - 1)
            vectors = frame[index, :, :]
        else:
            vectors = frame.reshape(-1, 3)
        self.probe_magnetization_viewer.update_magnetization(vectors)

    def _signal_weighting_mode(self):
        return "voxel_volume" if self.signal_weighting.currentIndex() == 1 else "voxel"

    def _run(self):
        if self.probe_worker is not None and self.probe_worker.isRunning():
            QMessageBox.warning(
                self,
                "Probe running",
                "Cancel the spin probe before running a sequence simulation.",
            )
            return
        if self.sequence_source.currentIndex() == 1:
            self._load_cartesian_epi()
        if self.program is None:
            QMessageBox.warning(self, "No sequence", "Choose or load a sequence first.")
            return
        try:
            self._build_phantom()
            checkpoints = self._checkpoint_seconds()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid simulation", str(exc))
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        work_units = self._estimated_work_units()
        self.progress.setRange(0, work_units)
        self.progress.setValue(0)
        self.progress.setFormat("Simulation %v/%m")
        self._status_update("Preparing and compiling sequence…")
        chunk_voxels = self._preview_chunk_voxels()
        self.worker = SequenceSimulationThread(
            self.simulator,
            self.program,
            self.phantom,
            checkpoints,
            self._signal_weighting_mode(),
            self.field_strength_t.value(),
            self.nucleus.currentText(),
            self.live_preview_enabled,
            chunk_voxels,
            simulation_timestep_s=self.simulation_timestep_us.value() * 1e-6,
        )
        self.worker.progress.connect(self._progress)
        self.worker.stage.connect(self._status_update)
        self.worker.preview.connect(self._preview)
        self.worker.result_ready.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _status_update(self, message):
        """Show worker stages in both the workspace and the main log."""
        message = str(message)
        self.status.setText(message)
        logger = getattr(self.window(), "log_message", None)
        if callable(logger):
            logger(f"Sequence simulation: {message}")

    def _progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat("Simulation %v/%m")
        if isinstance(self.phantom, DynamicSpectralPhantom):
            unit = "Interval"
        elif isinstance(self.phantom, SpectralPhantom):
            unit = "Component"
        else:
            unit = "Chunk"
        self.status.setText(f"{unit} {done}/{total}")

    def _estimated_work_units(self):
        if isinstance(self.phantom, DynamicSpectralPhantom):
            return max(1, SequenceCompiler().compile(self.program).n_intervals)
        if isinstance(self.phantom, SpectralPhantom):
            return max(
                1,
                sum(
                    bool(np.any(values > 0))
                    for values in self.phantom.concentration_maps.values()
                ),
            )
        chunk_voxels = self._preview_chunk_voxels()
        if chunk_voxels is None:
            chunk_voxels = 65536
        return max(1, int(np.ceil(self.phantom.n_active / chunk_voxels)))

    def _preview_chunk_voxels(self):
        if not self.live_preview_enabled or isinstance(
            self.phantom, (SpectralPhantom, DynamicSpectralPhantom)
        ):
            return None
        return min(65536, max(256, int(np.ceil(self.phantom.n_active / 32))))

    def set_live_preview_enabled(self, enabled):
        """Apply the persisted live sequence visualization preference."""
        self.live_preview_enabled = bool(enabled)
        self.rf_progress_cursor.setVisible(self.live_preview_enabled)
        self.gradient_progress_cursor.setVisible(self.live_preview_enabled)

    def set_sequence_kernel(self, kernel):
        """Apply the persistent sequence-kernel setting to future runs."""
        if kernel not in {"optimized", "reference"}:
            raise ValueError("sequence kernel must be 'optimized' or 'reference'")
        self.simulator.sequence_kernel = kernel

    def set_sequence_timestep_us(self, value):
        """Apply the persistent RF-active time-step default."""
        value = float(value)
        if not np.isfinite(value) or not 0.1 <= value <= 1000.0:
            raise ValueError("sequence time step must be between 0.1 and 1000 µs")
        self.simulation_timestep_us.setValue(value)

    def set_thread_configuration(self, mode, manual_thread_count):
        """Apply automatic or manual native worker selection."""
        if mode not in {"automatic", "manual"}:
            raise ValueError("thread mode must be 'automatic' or 'manual'")
        requested = None if mode == "automatic" else manual_thread_count
        self.simulator.num_threads = resolve_num_threads(requested)

    def _set_sequence_cursor(self, fraction):
        duration_ms = self.program.duration_s * 1000.0 if self.program else 0.0
        position = float(np.clip(fraction, 0.0, 1.0)) * duration_ms
        self.rf_progress_cursor.setPos(position)
        self.gradient_progress_cursor.setPos(position)

    def _preview(self, fraction, signal):
        """Render a throttled intermediate timeline, k-space and image state."""
        if not self.live_preview_enabled or self.program is None:
            return
        fraction = float(np.clip(fraction, 0.0, 1.0))
        self._set_sequence_cursor(fraction)
        if self.acquisition is None and self.spectroscopic_acquisition is None:
            return
        adc_times = (
            np.concatenate([event.sample_times_s for event in self.program.adc_events])
            if self.program.adc_events
            else np.empty(0)
        )
        acquired = int(
            np.count_nonzero(adc_times <= fraction * self.program.duration_s)
        )
        partial_signal = np.array(signal, copy=True)
        partial_signal[..., acquired:] = 0.0
        if self.spectroscopic_acquisition is not None:
            self._show_live_spectroscopy(partial_signal, acquired, adc_times.size)
        else:
            self._show_live_cartesian(partial_signal, acquired, adc_times.size)

    def _show_live_spectroscopy(self, signal, acquired, total):
        csi = self.spectroscopic_acquisition
        point = min(self.spectral_point_selector.value(), csi.spectral_points - 1)
        kspace = csi.reshape_signal(signal)
        fid = csi.reconstruct_spatial(signal)
        if kspace.ndim == 4:
            kspace_image = np.sqrt(np.sum(np.abs(kspace[..., point]) ** 2, axis=0))
            spatial_image = np.sqrt(np.sum(np.abs(fid[..., point]) ** 2, axis=0))
        else:
            kspace_image = np.abs(kspace[..., point])
            spatial_image = np.abs(fid[..., point])
        self.kspace_view.setImage(np.log1p(kspace_image).T, autoLevels=True)
        self.reconstruction_view.setImage(spatial_image.T, autoLevels=True)
        self.kspace_info.setText(
            f"Live CSI k-space, FID sample {point + 1}/{csi.spectral_points}; "
            f"{acquired}/{total} ADC samples"
        )
        self.reconstruction_info.setText("Live spatial 2D IFFT of CSI FID")

    def _show_live_cartesian(self, signal, acquired, total):
        selected = self.frame_selector.currentData()
        frame = max(0, 0 if selected is None or int(selected) < 0 else int(selected))
        if self.acquisition_frames is not None:
            acquisition = self.acquisition_frames.acquisitions[frame]
            frame_signal = self.acquisition_frames._frame_values(signal, frame)
        else:
            acquisition = self.acquisition
            frame_signal = signal
        kspace = acquisition.reshape_signal(frame_signal)
        if kspace.ndim == 3:
            kspace_magnitude = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
            image = acquisition.reconstruct(frame_signal, coil_combine="rss")
        else:
            kspace_magnitude = np.abs(kspace)
            image = acquisition.reconstruct(frame_signal)
        self.kspace_view.setImage(np.log1p(kspace_magnitude).T, autoLevels=True)
        self.reconstruction_view.setImage(np.abs(image).T, autoLevels=True)
        self.kspace_info.setText(f"Live k-space: {acquired}/{total} ADC samples")
        self.reconstruction_info.setText(
            f"Live |IFFT2| from {acquired}/{total} ADC samples"
        )

    def _cancel(self):
        if self.worker is not None:
            self.worker.request_cancel()
            self.status.setText("Cancelling after current chunk…")

    def _finished(self, result):
        self.result = result
        self._reset_run_controls(completed=True)
        self.export_button.setEnabled(True)
        self._status_update("Simulation complete")
        self._set_sequence_cursor(1.0)
        self._configure_frame_selector()
        self._configure_spectroscopy_selectors()
        self.signal_plot.clear()
        self.signal_plot.setTitle("Received ADC signal")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        time_ms = result.adc_times_s * 1000
        signal = np.asarray(result.signal)
        if self.spectroscopic_acquisition is not None:
            csi = self.spectroscopic_acquisition
            center_event = csi.encoding_indices.index(
                (csi.matrix[0] // 2, csi.matrix[1] // 2)
            )
            start = center_event * csi.spectral_points
            stop = start + csi.spectral_points
            plot_signal = signal[..., start:stop]
            plot_time = csi.spectral_time_s * 1000.0
        else:
            plot_signal = signal
            plot_time = time_ms
        if plot_signal.ndim == 1:
            self.signal_plot.plot(
                plot_time, np.abs(plot_signal), pen="w", name="Magnitude"
            )
            self.signal_plot.plot(plot_time, plot_signal.real, pen="g", name="Real")
            self.signal_plot.plot(
                plot_time, plot_signal.imag, pen="r", name="Imaginary"
            )
            coil_text = ""
        else:
            for coil, coil_signal in enumerate(plot_signal):
                self.signal_plot.plot(
                    plot_time,
                    np.abs(coil_signal),
                    pen=pg.intColor(coil, hues=plot_signal.shape[0]),
                    name=f"Coil {coil + 1}",
                )
            coil_text = f"; Rx coils={plot_signal.shape[0]}"
        if result.species_signal is not None:
            pool_signal = np.asarray(result.species_signal)
            if self.spectroscopic_acquisition is not None:
                pool_signal = pool_signal[..., start:stop]
            colors = ("#ffb000", "#00b7ff", "#d95fef", "#7ad151")
            for pool_index, name in enumerate(result.pool_names):
                self.signal_plot.plot(
                    plot_time,
                    np.abs(pool_signal[pool_index]),
                    pen=pg.mkPen(colors[pool_index % len(colors)], width=2),
                    name=f"|{name}|",
                )
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
        self._update_zoom_label(self.state_view, self.state_zoom_info)
        self.state_info.setText(
            f"Final Mz: min={mz_min:.5g}, max={mz_max:.5g}; "
            f"ADC samples={result.adc_times_s.size}{coil_text}{slice_text}"
        )
        self.result_volume_viewer.set_result(result, self.phantom)
        if self.spectroscopic_acquisition is not None:
            self._show_spectroscopic_result(result)
        else:
            self._show_cartesian_result(result)

    def _configure_frame_selector(self):
        self.frame_selector.blockSignals(True)
        self.frame_slider.blockSignals(True)
        self.frame_selector.clear()
        if self.acquisition_frames is None:
            self.frame_selector.addItem("Single 2D frame", 0)
            self.frame_selector.setEnabled(False)
            self.frame_slider.setRange(0, 0)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(False)
        else:
            self.frame_selector.addItem(
                f"All {self.acquisition_frames.num_frames} frames (montage)", -1
            )
            for frame in range(self.acquisition_frames.num_frames):
                self.frame_selector.addItem(
                    self.acquisition_frames.frame_label(frame), frame
                )
            self.frame_selector.setEnabled(True)
            self.frame_slider.setRange(-1, self.acquisition_frames.num_frames - 1)
            self.frame_slider.setValue(-1)
            self.frame_slider.setEnabled(True)
        self.frame_selector.blockSignals(False)
        self.frame_slider.blockSignals(False)

    def _configure_spectroscopy_selectors(self):
        csi = self.spectroscopic_acquisition
        selectors = (
            self.spectral_point_selector,
            self.spectral_point_slider,
            self.spectrum_x_selector,
            self.spectrum_x_slider,
            self.spectrum_y_selector,
            self.spectrum_y_slider,
        )
        for selector in selectors:
            selector.blockSignals(True)
        try:
            if csi is None:
                self._split_csi_data = None
                self.split_view_checkbox.setChecked(False)
                self.split_view_checkbox.setEnabled(False)
                self.split_image_source.setEnabled(False)
                self.split_signal_source.setEnabled(False)
                for selector in selectors:
                    selector.setRange(0, 0)
                    selector.setValue(0)
                    selector.setEnabled(False)
                return
            self.split_view_checkbox.setEnabled(True)
            self.split_image_source.setEnabled(self.split_view_checkbox.isChecked())
            self.split_signal_source.setEnabled(self.split_view_checkbox.isChecked())
            self.spectral_point_selector.setRange(0, csi.spectral_points - 1)
            self.spectral_point_selector.setValue(0)
            self.spectral_point_slider.setRange(0, csi.spectral_points - 1)
            self.spectral_point_slider.setValue(0)
            self.spectrum_x_selector.setRange(0, csi.matrix[0] - 1)
            self.spectrum_x_selector.setValue(csi.matrix[0] // 2)
            self.spectrum_x_slider.setRange(0, csi.matrix[0] - 1)
            self.spectrum_x_slider.setValue(csi.matrix[0] // 2)
            self.spectrum_y_selector.setRange(0, csi.matrix[1] - 1)
            self.spectrum_y_selector.setValue(csi.matrix[1] // 2)
            self.spectrum_y_slider.setRange(0, csi.matrix[1] - 1)
            self.spectrum_y_slider.setValue(csi.matrix[1] // 2)
            for selector in selectors:
                selector.setEnabled(True)
        finally:
            for selector in selectors:
                selector.blockSignals(False)

    def _spectral_view_changed(self, *_):
        if self.result is not None and self.spectroscopic_acquisition is not None:
            self._show_spectroscopic_result(self.result)

    def _toggle_split_view(self, enabled):
        enabled = bool(enabled)
        self.view_stack.setCurrentIndex(1 if enabled else 0)
        active = enabled and self.spectroscopic_acquisition is not None
        self.split_image_source.setEnabled(active)
        self.split_signal_source.setEnabled(active)
        if active:
            self._refresh_split_view()

    def _refresh_split_view(self, *_):
        if self._split_csi_data is None:
            return
        self._update_split_view(*self._split_csi_data)

    def _set_csi_voxel(self, x_index, y_index):
        csi = self.spectroscopic_acquisition
        if csi is None:
            return
        x_index = int(np.clip(x_index, 0, csi.matrix[0] - 1))
        y_index = int(np.clip(y_index, 0, csi.matrix[1] - 1))
        controls = (
            (self.spectrum_x_selector, x_index),
            (self.spectrum_x_slider, x_index),
            (self.spectrum_y_selector, y_index),
            (self.spectrum_y_slider, y_index),
        )
        for control, value in controls:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        if self.result is not None:
            self._show_spectroscopic_result(self.result)
        else:
            self._refresh_split_view()

    def _split_image_clicked(self, event):
        if self._split_csi_data is None or event.button() != Qt.LeftButton:
            return
        view_box = self.split_image_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view_box.mapSceneToView(event.scenePos())
        self._set_csi_voxel(round(point.x()), round(point.y()))

    def _update_split_view(self, csi, kspace, spatial_fid, spectra):
        self._split_csi_data = (csi, kspace, spatial_fid, spectra)
        point = min(self.spectral_point_selector.value(), csi.spectral_points - 1)
        x_index = min(self.spectrum_x_selector.value(), csi.matrix[0] - 1)
        y_index = min(self.spectrum_y_selector.value(), csi.matrix[1] - 1)

        if self.split_image_source.currentText() == "K-space":
            values = kspace[..., point]
            title = "CSI k-space"
            image_note = (
                "Click maps the displayed grid index to the corresponding image "
                "voxel selector. K-space coordinates themselves are not voxels."
            )
        else:
            values = spatial_fid[..., point]
            title = "CSI spatial reconstruction"
            image_note = "Click a voxel to update the FID or spectrum."
        if values.ndim == 3:
            image = np.sqrt(np.sum(np.abs(values) ** 2, axis=0))
        else:
            image = np.abs(values)
        self.split_image_item.setImage(np.asarray(image).T, autoLevels=True)
        self.split_image_item.setRect(QRectF(-0.5, -0.5, csi.matrix[0], csi.matrix[1]))
        self.split_image_plot.setTitle(
            f"{title} — FID sample {point}/{csi.spectral_points - 1}"
        )
        self.split_voxel_marker.setData([x_index], [y_index])
        self.split_image_info.setText(image_note)

        self.split_signal_plot.clear()
        if self.split_signal_source.currentText() == "FID":
            voxel_signal = spatial_fid[..., y_index, x_index, :]
            x_values = csi.spectral_time_s * 1000.0
            self.split_signal_plot.setTitle("Spatially reconstructed voxel FID")
            self.split_signal_plot.setLabel("bottom", "Time", "ms")
        else:
            voxel_signal = spectra[..., y_index, x_index, :]
            x_values = csi.frequency_hz
            self.split_signal_plot.setTitle("Spatially reconstructed voxel spectrum")
            self.split_signal_plot.setLabel("bottom", "Frequency", "Hz")
        if voxel_signal.ndim == 2:
            magnitude = np.sqrt(np.sum(np.abs(voxel_signal) ** 2, axis=0))
            self.split_signal_plot.plot(
                x_values, magnitude, pen="w", name="Magnitude (RSS)"
            )
        else:
            self.split_signal_plot.plot(
                x_values, np.abs(voxel_signal), pen="w", name="Magnitude"
            )
            self.split_signal_plot.plot(
                x_values, voxel_signal.real, pen="g", name="Real"
            )
            self.split_signal_plot.plot(
                x_values, voxel_signal.imag, pen="r", name="Imaginary"
            )
        x_mm = ((x_index + 0.5) / csi.matrix[0] - 0.5) * csi.fov_m[0] * 1000
        y_mm = ((y_index + 0.5) / csi.matrix[1] - 0.5) * csi.fov_m[1] * 1000
        self.split_signal_info.setText(
            f"Voxel (x={x_index}, y={y_index}) at " f"({x_mm:.4g}, {y_mm:.4g}) mm"
        )

    def _show_spectroscopic_result(self, result):
        csi = self.spectroscopic_acquisition
        try:
            csi.validate_adc_times(result.adc_times_s)
            if result.adc_gradient_moment_cyc_per_m is not None:
                csi.validate_gradient_moments(result.adc_gradient_moment_cyc_per_m)
            kspace = csi.reshape_signal(result.signal)
            spatial_fid = csi.reconstruct_spatial(result.signal)
            spectra = csi.reconstruct_spectra(result.signal)
            point = min(self.spectral_point_selector.value(), csi.spectral_points - 1)
            if kspace.ndim == 4:
                kspace_image = np.sqrt(np.sum(np.abs(kspace[..., point]) ** 2, axis=0))
                spatial_image = np.sqrt(
                    np.sum(np.abs(spatial_fid[..., point]) ** 2, axis=0)
                )
                coil_text = f", {kspace.shape[0]} coils (RSS)"
            else:
                kspace_image = np.abs(kspace[..., point])
                spatial_image = np.abs(spatial_fid[..., point])
                coil_text = ""
            self.kspace_view.setImage(np.log1p(kspace_image).T, autoLevels=True)
            self.reconstruction_view.setImage(spatial_image.T, autoLevels=True)
            self._update_zoom_label(self.kspace_view, self.kspace_zoom_info)
            self._update_zoom_label(
                self.reconstruction_view, self.reconstruction_zoom_info
            )
            time_ms = csi.spectral_time_s[point] * 1000.0
            self.kspace_info.setText(
                f"CSI log(1+|k|), grid={csi.matrix[1]}×{csi.matrix[0]}, "
                f"FID sample={point} ({time_ms:.4g} ms){coil_text}"
            )
            self.reconstruction_info.setText(
                f"CSI spatial |IFFT2| at FID sample {point}; "
                f"min={spatial_image.min():.5g}, max={spatial_image.max():.5g}"
                f"{coil_text}; spectral FFT is shown in the signal tab"
            )

            x_index = min(self.spectrum_x_selector.value(), csi.matrix[0] - 1)
            y_index = min(self.spectrum_y_selector.value(), csi.matrix[1] - 1)
            voxel_spectrum = spectra[..., y_index, x_index, :]
            self.signal_plot.clear()
            self.signal_plot.setTitle("Spatially reconstructed CSI spectrum")
            self.signal_plot.setLabel("bottom", "Frequency", "Hz")
            if voxel_spectrum.ndim == 2:
                magnitude = np.sqrt(np.sum(np.abs(voxel_spectrum) ** 2, axis=0))
                self.signal_plot.plot(
                    csi.frequency_hz, magnitude, pen="w", name="Magnitude (RSS)"
                )
            else:
                magnitude = np.abs(voxel_spectrum)
                self.signal_plot.plot(
                    csi.frequency_hz, magnitude, pen="w", name="Magnitude"
                )
                self.signal_plot.plot(
                    csi.frequency_hz, voxel_spectrum.real, pen="g", name="Real"
                )
                self.signal_plot.plot(
                    csi.frequency_hz, voxel_spectrum.imag, pen="r", name="Imaginary"
                )
            x_mm = ((x_index + 0.5) / csi.matrix[0] - 0.5) * csi.fov_m[0] * 1000
            y_mm = ((y_index + 0.5) / csi.matrix[1] - 0.5) * csi.fov_m[1] * 1000
            self.spectrum_info.setText(
                f"Voxel (x={x_index}, y={y_index}) at ({x_mm:.4g}, {y_mm:.4g}) mm; "
                f"BW={csi.spectral_bandwidth_hz:.6g} Hz; "
                f"resolution={csi.spectral_resolution_hz:.6g} Hz"
            )
            self._update_split_view(csi, kspace, spatial_fid, spectra)
        except Exception as exc:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            self.signal_plot.clear()
            message = f"CSI reconstruction unavailable: {exc}"
            self.kspace_info.setText(message)
            self.reconstruction_info.setText(message)
            self.spectrum_info.setText(message)

    def _frame_changed(self, *_):
        selected = self.frame_selector.currentData()
        if selected is not None:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(int(selected))
            self.frame_slider.blockSignals(False)
        if self.result is not None:
            self._show_cartesian_result(self.result)

    def _frame_slider_changed(self, frame):
        combo_index = self.frame_selector.findData(int(frame))
        if combo_index >= 0:
            self.frame_selector.setCurrentIndex(combo_index)

    def _show_cartesian_result(self, result):
        if self.acquisition is None:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            self.kspace_info.setText("No Cartesian acquisition metadata")
            self.reconstruction_info.setText("No Cartesian acquisition metadata")
            return
        try:
            selected_frame = self.frame_selector.currentData()
            selected_frame = 0 if selected_frame is None else int(selected_frame)
            if self.acquisition_frames is not None and selected_frame < 0:
                frame_views = [
                    self._cartesian_frame_views(result, frame)
                    for frame in range(self.acquisition_frames.num_frames)
                ]
                kspace_magnitude = self._montage([item[0] for item in frame_views])
                image = self._montage([item[1] for item in frame_views])
                acquisition = frame_views[0][2]
                coil_text = frame_views[0][3]
                frame_text = f", montage of {self.acquisition_frames.num_frames} frames"
            else:
                frame = max(0, selected_frame)
                kspace_magnitude, image, acquisition, coil_text = (
                    self._cartesian_frame_views(result, frame)
                )
                frame_text = (
                    f", {self.acquisition_frames.frame_label(frame)}"
                    if self.acquisition_frames is not None
                    else ""
                )
            self.kspace_view.setImage(np.log1p(kspace_magnitude).T, autoLevels=True)
            self.reconstruction_view.setImage(
                np.asarray(np.abs(image)).T, autoLevels=True
            )
            self._update_zoom_label(self.kspace_view, self.kspace_zoom_info)
            self._update_zoom_label(
                self.reconstruction_view, self.reconstruction_zoom_info
            )
            self.kspace_info.setText(
                f"2D log(1+|k|), grid={acquisition.phase_matrix}×"
                f"{acquisition.read_matrix}{coil_text}{frame_text}"
            )
            if (
                self.acquisition_frames is not None
                and "slice" in self.acquisition_frames.varying_axes
            ):
                z_note = "; slice-selective 2D frames (no kz encoding)"
            elif self.phantom.ndim == 3 and self.phantom.shape[2] > 1:
                z_note = "; z-integrated signal (no kz encoding)"
            else:
                z_note = "; no kz encoding"
            self.reconstruction_info.setText(
                f"2D |IFFT2|, min={np.min(np.abs(image)):.5g}, "
                f"max={np.max(np.abs(image)):.5g}{coil_text}{frame_text}{z_note}"
            )
        except Exception as exc:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            message = f"Cartesian reconstruction unavailable: {exc}"
            self.kspace_info.setText(message)
            self.reconstruction_info.setText(message)

    def _cartesian_frame_views(self, result, frame):
        if self.acquisition_frames is not None:
            acquisition = self.acquisition_frames.acquisitions[frame]
            kspace = self.acquisition_frames.to_cartesian_kspace(result, frame)
        else:
            acquisition = self.acquisition
            kspace = result.to_cartesian_kspace(acquisition)
        if kspace.ndim == 3:
            kspace_magnitude = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
            if self.acquisition_frames is not None:
                image = self.acquisition_frames.reconstruct(
                    result, frame, coil_combine="rss"
                )
            else:
                image = result.reconstruct_cartesian(acquisition, coil_combine="rss")
            coil_text = f", {kspace.shape[0]} coils (RSS)"
        else:
            kspace_magnitude = np.abs(kspace)
            if self.acquisition_frames is not None:
                image = np.abs(self.acquisition_frames.reconstruct(result, frame))
            else:
                image = np.abs(result.reconstruct_cartesian(acquisition))
            coil_text = ""
        return (
            np.asarray(kspace_magnitude),
            np.asarray(np.abs(image)),
            acquisition,
            coil_text,
        )

    @staticmethod
    def _montage(images, *, maximum_columns=4, gap=1):
        arrays = [np.asarray(image) for image in images]
        if not arrays or any(array.ndim != 2 for array in arrays):
            raise ValueError("montage requires at least one 2D image")
        columns = min(len(arrays), int(maximum_columns))
        rows = int(np.ceil(len(arrays) / columns))
        height = max(array.shape[0] for array in arrays)
        width = max(array.shape[1] for array in arrays)
        canvas = np.zeros(
            (
                rows * height + (rows - 1) * gap,
                columns * width + (columns - 1) * gap,
            ),
            dtype=np.result_type(*arrays),
        )
        for index, array in enumerate(arrays):
            row, column = divmod(index, columns)
            y = row * (height + gap)
            x = column * (width + gap)
            canvas[y : y + array.shape[0], x : x + array.shape[1]] = array
        return canvas

    def _failed(self, message):
        self._reset_run_controls()
        self.status.setText(message)
        if message != "Simulation cancelled":
            QMessageBox.critical(self, "Sequence simulation failed", message)

    def _reset_run_controls(self, *, completed=False):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if completed:
            self.progress.setValue(self.progress.maximum())
            self.progress.setFormat("Complete")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Stopped")

    def _export_results(self):
        if self.result is None:
            QMessageBox.warning(self, "No result", "Run a simulation first.")
            return
        bruker_filter = "Bruker raw dataset (directory)"
        default_path = workspace_directory("exports") / "sequence_result.nc"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export sequence simulation result",
            str(default_path),
            (
                "xarray NetCDF (*.nc);;HDF5 (*.h5);;NumPy archive (*.npz);;"
                "Jupyter notebook (*.ipynb);;Bruker raw dataset (directory)"
            ),
        )
        if not filename:
            return
        path = Path(filename)
        if selected_filter != bruker_filter and not path.suffix:
            suffixes = {
                "xarray NetCDF (*.nc)": ".nc",
                "HDF5 (*.h5)": ".h5",
                "NumPy archive (*.npz)": ".npz",
                "Jupyter notebook (*.ipynb)": ".ipynb",
            }
            path = path.with_suffix(suffixes.get(selected_filter, ".nc"))
        try:
            if selected_filter == bruker_filter:
                options = self._prompt_bruker_export_options(path)
                if options is None:
                    return
                export_bruker_raw(
                    self.result,
                    path,
                    program=self.program,
                    phantom=self.phantom,
                    acquisition=self.acquisition,
                    acquisition_frames=self.acquisition_frames,
                    options=options,
                )
                exported = "fid\nacqp\nmethod\nvisu_pars\npulseprogram"
            elif path.suffix.lower() == ".ipynb":
                data_path = path.with_suffix(".nc")
                self.result.save(data_path)
                export_sequence_result_notebook(str(path), str(data_path))
                exported = f"{path.name}\n{data_path.name}"
            else:
                self.result.save(path)
                exported = path.name
            QMessageBox.information(
                self,
                "Export complete",
                f"Exported to {path.parent}:\n{exported}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _prompt_bruker_export_options(self, path: Path):
        matrix, fov_m = self._default_bruker_spatial_metadata()
        dialog = QDialog(self)
        dialog.setWindowTitle("Bruker export metadata")
        layout = QFormLayout(dialog)

        method_name = QComboBox()
        method_name.setEditable(True)
        method_name.addItems(
            [
                "Bruker:RARE",
                "Bruker:FLASH",
                "Bruker:CSI",
                "User:pulseq",
                "BlochSimulator:SequenceSimulation",
            ]
        )
        method_name.setCurrentText("Bruker:RARE")
        layout.addRow("Method", method_name)

        scan_name = QLineEdit()
        default_scan_name = f"BlochSimulator {path.name or 'sequence'}"
        if self.program is not None and self.program.source:
            default_scan_name = f"BlochSimulator {Path(self.program.source).name}"
        scan_name.setText(default_scan_name)
        layout.addRow("Scan name", scan_name)

        matrix_widget = QWidget()
        matrix_layout = QHBoxLayout(matrix_widget)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_read = QSpinBox()
        matrix_read.setRange(1, 8192)
        matrix_read.setValue(int(matrix[0]))
        matrix_phase = QSpinBox()
        matrix_phase.setRange(1, 8192)
        matrix_phase.setValue(int(matrix[1]))
        matrix_layout.addWidget(matrix_read)
        matrix_layout.addWidget(QLabel("read x phase"))
        matrix_layout.addWidget(matrix_phase)
        layout.addRow("PVM_Matrix", matrix_widget)

        fov_widget = QWidget()
        fov_layout = QHBoxLayout(fov_widget)
        fov_layout.setContentsMargins(0, 0, 0, 0)
        fov_read = QDoubleSpinBox()
        fov_read.setRange(0.001, 10000.0)
        fov_read.setDecimals(4)
        fov_read.setValue(float(fov_m[0]) * 1000.0)
        fov_read.setSuffix(" mm")
        fov_phase = QDoubleSpinBox()
        fov_phase.setRange(0.001, 10000.0)
        fov_phase.setDecimals(4)
        fov_phase.setValue(float(fov_m[1]) * 1000.0)
        fov_phase.setSuffix(" mm")
        fov_layout.addWidget(fov_read)
        fov_layout.addWidget(QLabel("read x phase"))
        fov_layout.addWidget(fov_phase)
        layout.addRow("PVM_Fov", fov_widget)

        slice_thickness = QDoubleSpinBox()
        slice_thickness.setRange(0.001, 10000.0)
        slice_thickness.setDecimals(4)
        slice_thickness.setValue(1.0)
        slice_thickness.setSuffix(" mm")
        layout.addRow("PVM_SliceThick", slice_thickness)

        raw_data_files = QComboBox()
        raw_data_files.addItems(["fid", "rawdata.job0", "both"])
        raw_data_files.setCurrentText("fid")
        layout.addRow("Raw data file", raw_data_files)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return BrukerExportOptions(
            method_name=method_name.currentText(),
            scan_name=scan_name.text(),
            matrix=(matrix_read.value(), matrix_phase.value()),
            fov_m=(fov_read.value() / 1000.0, fov_phase.value() / 1000.0),
            slice_thickness_mm=slice_thickness.value(),
            raw_data_files=raw_data_files.currentText(),
        )

    def _default_bruker_spatial_metadata(self):
        acquisition = self.acquisition
        if acquisition is not None:
            return (
                (acquisition.read_matrix, acquisition.phase_matrix),
                acquisition.fov_m,
            )
        spectroscopy = self.spectroscopic_acquisition
        if spectroscopy is None and self.result is not None:
            try:
                spectroscopy = self.result.spectroscopic_acquisition
            except Exception:
                spectroscopy = None
        if spectroscopy is not None:
            return (spectroscopy.matrix, spectroscopy.fov_m)
        if self.phantom is not None and hasattr(self.phantom, "fov"):
            fov = tuple(float(value) for value in self.phantom.fov)
            if len(fov) >= 2:
                return ((self.result.signal.shape[-1], 1), (fov[0], fov[1]))
            if len(fov) == 1:
                return ((self.result.signal.shape[-1], 1), (fov[0], fov[0]))
        return ((self.result.signal.shape[-1], 1), (1.0, 1.0))
