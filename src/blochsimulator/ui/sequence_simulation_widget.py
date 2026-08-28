"""Integrated desktop UI for event-based 3D sequence simulation."""

from __future__ import annotations

import html
import math
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QProcess, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..memory import MEMORY_ERROR_PREFIX
from ..phantom import Phantom, PhantomFactory
from ..notebook_exporter import (
    export_pulseq_generation_notebook,
    export_sequence_result_notebook,
)
from ..paths import workspace_directory
from ..spectral_phantom import SpectralPhantom
from ..dynamic_phantom import DynamicSpectralPhantom
from ..sequence import (
    ADCEvent,
    BrukerExportOptions,
    CartesianAcquisition,
    CartesianAcquisitionFrames,
    CartesianAcquisitionVolumes,
    EncodingFrame,
    SpiralAcquisition,
    SpectroscopicAcquisition,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
    ScannerParameters,
    SpinSampling,
    analyze_adc_moment_train,
    analyze_repeated_spoiler_train,
    ernst_angle_deg,
    export_bruker_raw,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    infer_spectroscopic_acquisition,
    infer_spiral_acquisition,
    load_pulseq,
    load_scanner_parameters,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_epi,
    make_pulseq_flash,
    make_pulseq_me_bssfp,
    make_pulseq_radial_me_bssfp,
    make_pulseq_spectral_selective_bssfp,
    make_pulseq_spiral,
    recommend_spin_grid,
    recommend_spin_grid_for_phase_train,
    variable_flip_angle_schedule,
)
from ..sequence.spin_sampling import phantom_voxel_basis_m
from ..sequence.rf_pulses import (
    RF_PULSE_TYPE_LABELS,
    analytic_rf_shape_parameter,
    design_rf_envelope,
    rf_time_bandwidth_product_from_envelope,
)
from ..simulator import BlochSimulator, resolve_num_threads
from ..units import (
    NUCLEUS_GAMMA_HZ_PER_T,
    gradient_hz_per_m_to_t_per_m,
    ppm_to_hz,
    rf_gauss_to_hz,
    rf_hz_to_gauss_for_nucleus,
)
from .controls import UniversalTimeControl
from .dialogs import PulseImportDialog
from .magnetization_viewer import MagnetizationViewer
from .plot_interaction import AXIS_ZOOM_TOOLTIP
from .probe_viewers import SequenceProbeSpatialViewer, SequenceProbeSpectrumViewer
from .reconstruction_explorer import SequenceReconstructionExplorer
from .simulation_explorer import SessionSimulationRun
from .volume_viewer import (
    SequenceMagnetizationAnimationViewer,
    SequenceResultVolumeViewer,
)
from .widgets import (
    IMAGE_CANVAS_BACKGROUND,
    compact_image_histogram,
    style_image_item,
)
from .default_settings import WorkspaceDefaults


def _format_duration(seconds):
    """Format a short, stable duration for progress and completion messages."""
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return "<1s"
    total_seconds = int(math.ceil(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def _left_aligned_form(parent=None):
    """Create a form whose parameter names read from the left edge."""
    form = QFormLayout(parent)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return form


def _add_form_section(form: QFormLayout, title: str) -> None:
    """Add a compact horizontal separator and bold section title to a form."""
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    label = QLabel(str(title))
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    form.addRow(separator)
    form.addRow(label)


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
    minimum_vertices_per_event = 9
    maximum_events = max(1, int(max_vertices) // minimum_vertices_per_event)
    if len(visible) > maximum_events:
        event_indices = np.linspace(0, len(visible) - 1, maximum_events, dtype=int)
        visible = [visible[index] for index in event_indices]
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


def _rf_phase_plot_data(events, *, start_s, end_s, max_vertices=50000):
    """Build the effective RF phase waveform in wrapped degrees.

    The displayed phase includes the complex RF envelope, the programmed
    phase offset, and the event-local carrier evolution used by the compiler.
    Wrap discontinuities are separated so the plot does not draw misleading
    vertical lines between +180 and -180 degrees.
    """
    start_s = float(start_s)
    end_s = float(end_s)
    visible = [
        event for event in events if event.end_s > start_s and event.start_s < end_s
    ]
    if not visible:
        return np.empty(0), np.empty(0)
    maximum_events = max(1, int(max_vertices) // 6)
    if len(visible) > maximum_events:
        event_indices = np.linspace(0, len(visible) - 1, maximum_events, dtype=int)
        visible = [visible[index] for index in event_indices]
    samples_per_event = max(2, int(max_vertices // len(visible)) - 2)
    x_parts = []
    y_parts = []
    for event in visible:
        samples = np.asarray(event.samples_hz, dtype=np.complex128)
        first = max(0, int(np.floor((start_s - event.start_s) / event.raster_s)))
        stop = min(
            samples.size,
            int(np.ceil((end_s - event.start_s) / event.raster_s)),
        )
        if stop <= first:
            continue
        indices = first + _representative_sample_indices(
            np.abs(samples[first:stop]), samples_per_event
        )
        relative_times = indices.astype(float) * event.raster_s
        effective = samples[indices] * np.exp(
            1j
            * (
                event.phase_offset_rad
                + 2.0 * np.pi * event.frequency_offset_hz * relative_times
            )
        )
        phase_deg = (np.rad2deg(np.angle(effective)) + 180.0) % 360.0 - 180.0
        phase_deg[np.abs(effective) <= np.finfo(float).eps] = np.nan
        times_ms = (event.start_s + (indices + 0.5) * event.raster_s) * 1000.0
        if phase_deg.size > 1:
            wrap_indices = np.flatnonzero(np.abs(np.diff(phase_deg)) > 180.0) + 1
            times_ms = np.insert(times_ms, wrap_indices, np.nan)
            phase_deg = np.insert(phase_deg, wrap_indices, np.nan)
        x_parts.append(np.concatenate((times_ms, [np.nan])))
        y_parts.append(np.concatenate((phase_deg, [np.nan])))
    if not x_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(x_parts), np.concatenate(y_parts)


@dataclass(frozen=True)
class _SequenceLoadPayload:
    program: SequenceProgram
    compiled: object
    acquisition: Optional[CartesianAcquisition]
    acquisition_frames: Optional[CartesianAcquisitionFrames]
    acquisition_volumes: Optional[CartesianAcquisitionVolumes]
    spectroscopic_acquisition: Optional[SpectroscopicAcquisition]
    spiral_acquisition: Optional[SpiralAcquisition]
    acquisition_note: str


@dataclass(frozen=True)
class _ErnstAngleContext:
    angle_deg: float
    angle_range_deg: tuple[float, float]
    repetition_time_s: float
    effective_t1_s: float
    t1_range_s: tuple[float, float]
    source: str


@dataclass(frozen=True)
class _MagnetizationAnimationData:
    """Reduced-precision spatial states kept only for post-run playback."""

    time_s: np.ndarray
    magnetization: np.ndarray
    pool_magnetization: Optional[np.ndarray]
    pool_names: tuple
    storage_dtype: str
    storage_note: str = ""


@dataclass(frozen=True)
class _SequenceSimulationPayload:
    """Separate the scientific result from optional UI-only animation data."""

    result: object
    animation: Optional[_MagnetizationAnimationData]
    animation_message: str = ""


def _animation_checkpoint_times(
    program,
    *,
    time_resolution_s,
    maximum_frames,
    checkpoints_s=(),
    simulation_timestep_s=None,
):
    """Select time-resolved frames while preserving RF integration boundaries.

    Targets in RF-free intervals remain at the requested times. Targets inside
    an RF-active interval are snapped to the nearest boundary from the original
    compilation so animation capture cannot alter RF integration accuracy.
    """
    time_resolution_s = float(time_resolution_s)
    maximum_frames = int(maximum_frames)
    if (
        not np.isfinite(time_resolution_s)
        or time_resolution_s <= 0.0
        or maximum_frames < 2
    ):
        return np.zeros(0, dtype=np.float64)
    duration_s = float(program.duration_s)
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return np.zeros(0, dtype=np.float64)
    compiled = SequenceCompiler().compile(
        program,
        checkpoints_s=tuple(checkpoints_s),
        simulation_timestep_s=simulation_timestep_s,
    )
    state_times = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.asarray(compiled.interval_end_s, dtype=np.float64),
        )
    )
    complete_steps = int(np.floor(duration_s / time_resolution_s))
    last_regular_s = complete_steps * time_resolution_s
    endpoint_tolerance = max(1e-14, duration_s * 1e-12)
    endpoint_is_regular = abs(last_regular_s - duration_s) <= endpoint_tolerance
    requested_frames = complete_steps + 1 + (0 if endpoint_is_regular else 1)
    if requested_frames > maximum_frames:
        targets = np.linspace(0.0, duration_s, maximum_frames, dtype=np.float64)
    else:
        targets = np.arange(complete_steps + 1, dtype=np.float64)
        targets *= time_resolution_s
        if endpoint_is_regular:
            targets[-1] = duration_s
        else:
            targets = np.concatenate((targets, np.asarray([duration_s])))

    interval_indices = np.searchsorted(
        np.asarray(compiled.interval_end_s), targets, side="left"
    )
    interval_indices = np.clip(interval_indices, 0, compiled.n_intervals - 1)
    rf_active = np.abs(np.asarray(compiled.rf_hz)[interval_indices]) > 0.0
    if np.any(rf_active):
        rf_targets = targets[rf_active]
        right = np.searchsorted(state_times, rf_targets, side="left")
        right = np.clip(right, 0, state_times.size - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(rf_targets - state_times[left]) <= np.abs(
            state_times[right] - rf_targets
        )
        targets[rf_active] = state_times[np.where(choose_left, left, right)]
    targets[0] = 0.0
    targets[-1] = duration_s
    selected = np.unique(targets)
    return selected if selected.size >= 2 else np.zeros(0, dtype=np.float64)


def _checkpoint_indices(available_times, requested_times):
    """Return indices of requested checkpoint times with float-safe matching."""
    available = np.asarray(available_times, dtype=np.float64)
    requested = np.asarray(requested_times, dtype=np.float64)
    if requested.size == 0:
        return np.zeros(0, dtype=np.intp)
    if available.size == 0:
        raise RuntimeError("animation checkpoint times are missing from the result")
    indices = np.searchsorted(available, requested, side="left")
    indices = np.clip(indices, 0, max(0, available.size - 1))
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(available[previous] - requested) < np.abs(
        available[indices] - requested
    )
    indices = np.where(use_previous, previous, indices)
    tolerance = np.maximum(1e-14, np.abs(requested) * 1e-11)
    if np.any(np.abs(available[indices] - requested) > tolerance):
        raise RuntimeError("animation checkpoint times are missing from the result")
    return indices.astype(np.intp, copy=False)


def _split_animation_result(result, user_checkpoints_s, animation_times_s, dtype):
    """Extract UI animation frames and restore the requested result checkpoints."""
    animation_times = np.asarray(animation_times_s, dtype=np.float64)
    if animation_times.size == 0:
        return result, None
    checkpoints = getattr(result, "checkpoint_magnetization", None)
    if checkpoints is None:
        raise RuntimeError("simulation did not return animation checkpoints")
    available_times = np.asarray(result.checkpoint_times_s, dtype=np.float64)
    animation_indices = _checkpoint_indices(available_times, animation_times)
    storage_dtype = np.dtype(dtype)
    pool_source = getattr(result, "checkpoint_pool_magnetization", None)
    all_animation_frames = np.array_equal(
        animation_indices, np.arange(available_times.size, dtype=np.intp)
    )
    selected_magnetization = (
        np.asarray(checkpoints)
        if all_animation_frames
        else np.asarray(checkpoints)[animation_indices]
    )
    selected_pool = None
    if pool_source is not None:
        selected_pool = (
            np.asarray(pool_source)
            if all_animation_frames
            else np.asarray(pool_source)[animation_indices]
        )
    storage_note = ""
    if storage_dtype == np.dtype(np.float16):
        float16_limit = float(np.finfo(np.float16).max)
        maximum = float(np.nanmax(np.abs(selected_magnetization)))
        if selected_pool is not None:
            maximum = max(maximum, float(np.nanmax(np.abs(selected_pool))))
        if not np.isfinite(maximum) or maximum > float16_limit:
            storage_dtype = np.dtype(np.float32)
            if not np.isfinite(maximum):
                reason = "one or more stored values are not finite"
            else:
                reason = (
                    f"the maximum stored magnitude ({maximum:.4g}) exceeds "
                    f"the float16 limit ({float16_limit:.0f})"
                )
            storage_note = (
                "float16 was requested but the animation was stored as float32 "
                f"because {reason}."
            )
    animation_magnetization = selected_magnetization.astype(
        storage_dtype, copy=selected_magnetization.dtype != storage_dtype
    )
    animation_pool = (
        None
        if selected_pool is None
        else selected_pool.astype(
            storage_dtype, copy=selected_pool.dtype != storage_dtype
        )
    )

    user_times = np.asarray(tuple(user_checkpoints_s), dtype=np.float64)
    user_indices = _checkpoint_indices(available_times, user_times)
    user_checkpoints = (
        None
        if user_times.size == 0
        else np.array(np.asarray(checkpoints)[user_indices], copy=True)
    )
    user_pool = (
        None
        if user_times.size == 0 or pool_source is None
        else np.array(np.asarray(pool_source)[user_indices], copy=True)
    )
    clean_result = replace(
        result,
        checkpoint_magnetization=user_checkpoints,
        checkpoint_times_s=user_times,
        checkpoint_pool_magnetization=user_pool,
    )
    animation = _MagnetizationAnimationData(
        time_s=animation_times,
        magnetization=animation_magnetization,
        pool_magnetization=animation_pool,
        pool_names=tuple(getattr(result, "pool_names", ())),
        storage_dtype=storage_dtype.name,
        storage_note=storage_note,
    )
    return clean_result, animation


def _infer_sequence_acquisition(program, compiled):
    """Infer an acquisition layout without touching Qt widget state."""
    acquisition = None
    acquisition_frames = None
    acquisition_volumes = None
    spectroscopic_acquisition = None
    spiral_acquisition = None
    definitions = program.metadata.get("definitions", {})
    if definitions.get("TrajectoryType") == "radial_3d_spiral_phyllotaxis":
        spokes = int(definitions.get("SpokesPerMeasurement", 0))
        measurements = int(definitions.get("Measurements", 0))
        echoes = int(definitions.get("Echoes", 0))
        samples = int(definitions.get("ReadoutSamples", 0))
        note = (
            f"3D radial ME-bSSFP: {measurements} measurement(s) × {spokes} "
            f"spokes × {echoes} echoes × {samples} samples. Non-Cartesian "
            "signal simulation is available; radial gridding/IDEAL "
            "reconstruction is not yet attached."
        )
        return _SequenceLoadPayload(
            program,
            compiled,
            acquisition,
            acquisition_frames,
            acquisition_volumes,
            spectroscopic_acquisition,
            spiral_acquisition,
            note,
        )
    try:
        spiral_acquisition = infer_spiral_acquisition(program, compiled=compiled)
        spiral = spiral_acquisition
        axes = ", ".join(spiral.varying_axes) or "single frame"
        note = (
            f"{spiral.num_frames} spiral 2D frame(s) inferred "
            f"({spiral.matrix[0]}×{spiral.matrix[1]}; {axes})"
        )
        metadata = dict(program.metadata)
        metadata["spiral_acquisition"] = spiral.to_metadata()
        program = SequenceProgram(
            events=program.events,
            duration_s=program.duration_s,
            source=program.source,
            version=program.version,
            metadata=metadata,
        )
        return _SequenceLoadPayload(
            program,
            compiled,
            acquisition,
            acquisition_frames,
            acquisition_volumes,
            spectroscopic_acquisition,
            spiral_acquisition,
            note,
        )
    except ValueError as exc:
        spiral_error = str(exc)
    try:
        spectroscopic_acquisition = infer_spectroscopic_acquisition(
            program, compiled=compiled
        )
        csi = spectroscopic_acquisition
        note = (
            f"2D CSI {csi.matrix[0]}×{csi.matrix[1]}×{csi.spectral_points}; "
            f"repetitions={csi.num_repetitions}; "
            f"BW={csi.spectral_bandwidth_hz:.6g} Hz"
        )
        return _SequenceLoadPayload(
            program,
            compiled,
            acquisition,
            acquisition_frames,
            acquisition_volumes,
            spectroscopic_acquisition,
            spiral_acquisition,
            note,
        )
    except ValueError as exc:
        spectroscopy_error = str(exc)

    single_error = None
    try:
        acquisition = infer_cartesian_acquisition(program, compiled=compiled)
        return _SequenceLoadPayload(
            program,
            compiled,
            acquisition,
            acquisition_frames,
            acquisition_volumes,
            spectroscopic_acquisition,
            spiral_acquisition,
            "Cartesian acquisition inferred from ADC moments",
        )
    except ValueError as exc:
        single_error = str(exc)

    try:
        acquisition_frames = infer_cartesian_acquisition_frames(
            program, compiled=compiled
        )
        try:
            acquisition_volumes = infer_cartesian_acquisition_volumes(
                program,
                compiled=compiled,
                frames=acquisition_frames,
            )
        except ValueError:
            acquisition_volumes = None
        acquisition = acquisition_frames.acquisitions[0]
        metadata = dict(program.metadata)
        metadata["acquisition_dimensions"] = acquisition_frames.dimensions.to_metadata()
        metadata["cartesian_acquisition_frames"] = acquisition_frames.to_metadata()
        if acquisition_volumes is not None:
            metadata["cartesian_acquisition_volumes"] = (
                acquisition_volumes.to_metadata()
            )
        program = SequenceProgram(
            events=program.events,
            duration_s=program.duration_s,
            source=program.source,
            version=program.version,
            metadata=metadata,
        )
        if acquisition_volumes is not None:
            nx, ny, nz = acquisition_volumes.matrix
            if definitions.get("TrajectoryType") == "cartesian_3d_multi_echo":
                echoes = int(definitions.get("Echoes", 0))
                repetitions = int(definitions.get("Repetitions", 0))
                strategy = str(definitions.get("ReadoutStrategy", "unknown"))
                note = (
                    f"Cartesian 3D ME-bSSFP: {repetitions} dynamic volume(s) × "
                    f"{echoes} echoes, {nx}×{ny}×{nz}, {strategy} readout. "
                    "Echo volumes are available; IDEAL metabolite separation "
                    "is not yet attached."
                )
            else:
                axes = ", ".join(acquisition_volumes.varying_axes) or "single volume"
                note = (
                    f"{acquisition_volumes.num_volumes} Cartesian 3D volume(s) "
                    f"inferred ({nx}×{ny}×{nz}; {axes})"
                )
        else:
            axes = ", ".join(acquisition_frames.varying_axes)
            note = (
                f"{acquisition_frames.num_frames} Cartesian 2D frames inferred ({axes})"
            )
    except ValueError as frame_error:
        acquisition = None
        note = (
            f"Acquisition inference unavailable: CSI: {spectroscopy_error}; "
            f"spiral: {spiral_error}; Cartesian: {single_error}; frames: {frame_error}"
        )
    return _SequenceLoadPayload(
        program,
        compiled,
        acquisition,
        acquisition_frames,
        acquisition_volumes,
        spectroscopic_acquisition,
        spiral_acquisition,
        note,
    )


def _prepare_pulseq_load(filename, status_callback=None):
    """Read and infer one Pulseq file for synchronous or threaded callers."""
    status = status_callback if status_callback is not None else lambda _message: None
    status("Reading Pulseq file…")
    program = load_pulseq(filename)
    status("Compiling ADC timing and gradient moments…")
    compiled = SequenceCompiler().compile_acquisition(program, status_callback=status)
    status("Inferring Cartesian/CSI acquisition layout…")
    return _infer_sequence_acquisition(program, compiled)


class PulseqLoadThread(QThread):
    """Read and validate a Pulseq file without blocking the Qt event loop."""

    stage = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, filename):
        super().__init__()
        self.filename = str(filename)

    def run(self):
        try:
            payload = _prepare_pulseq_load(
                self.filename, status_callback=self.stage.emit
            )
            self.result_ready.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


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
        spin_sampling=None,
        spoiler_mode="ideal",
        animation_time_resolution_s=None,
        animation_maximum_frames=0,
        animation_storage_dtype="float32",
        animation_note="",
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
        self.spin_sampling = spin_sampling
        self.spoiler_mode = spoiler_mode
        self.animation_time_resolution_s = (
            None
            if animation_time_resolution_s is None
            else float(animation_time_resolution_s)
        )
        self.animation_maximum_frames = max(0, int(animation_maximum_frames))
        if animation_storage_dtype not in {"float16", "float32"}:
            raise ValueError("animation storage dtype must be float16 or float32")
        self.animation_storage_dtype = animation_storage_dtype
        self.animation_note = str(animation_note)
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
            animation_times = np.zeros(0, dtype=np.float64)
            animation_allowed = (
                self.animation_time_resolution_s is not None
                and self.animation_time_resolution_s > 0.0
                and self.animation_maximum_frames >= 2
            )
            animation_message = self.animation_note if not animation_allowed else ""
            if self.animation_note:
                self.stage.emit(self.animation_note)
            if animation_allowed:
                self.stage.emit("Selecting sparse 3D animation states…")
                animation_times = _animation_checkpoint_times(
                    self.program,
                    time_resolution_s=self.animation_time_resolution_s,
                    maximum_frames=self.animation_maximum_frames,
                    checkpoints_s=self.checkpoints_s,
                    simulation_timestep_s=self.simulation_timestep_s,
                )
            progress_phases = 2 if animation_times.size else 1
            kwargs = {
                "checkpoints_s": tuple(self.checkpoints_s),
                "signal_weighting": self.signal_weighting,
                "progress_callback": lambda done, total: self.progress.emit(
                    done, total * progress_phases
                ),
                "chunk_voxels": self.chunk_voxels,
                "cancel_callback": lambda: self._cancel_requested,
                "status_callback": lambda message: self.stage.emit(message),
                "simulation_timestep_s": self.simulation_timestep_s,
                "spin_sampling": self.spin_sampling,
                "spoiler_mode": self.spoiler_mode,
            }
            if self.live_preview:
                kwargs["preview_callback"] = lambda fraction, signal: self.preview.emit(
                    fraction, signal
                )
            if isinstance(self.phantom, (SpectralPhantom, DynamicSpectralPhantom)):
                kwargs.update(
                    field_strength_t=self.field_strength_t,
                    nucleus=self.nucleus,
                )
            result = simulate(self.program, self.phantom, **kwargs)
            if not self._cancel_requested:
                animation = None
                if animation_times.size:
                    self.stage.emit(
                        "Scientific simulation complete; capturing 3D animation "
                        "states in a separate replay…"
                    )
                    replay_kwargs = dict(kwargs)
                    replay_kwargs["checkpoints_s"] = tuple(animation_times)
                    # Animation replay is UI-only. A float32 staging buffer keeps
                    # its peak memory well below the scientific float64 result.
                    replay_kwargs["checkpoint_dtype"] = "float32"
                    replay_kwargs["progress_callback"] = lambda done, total: (
                        self.progress.emit(total + done, 2 * total)
                    )
                    replay_kwargs.pop("preview_callback", None)
                    try:
                        replay_result = simulate(
                            self.program, self.phantom, **replay_kwargs
                        )
                        _, animation = _split_animation_result(
                            replay_result,
                            (),
                            animation_times,
                            self.animation_storage_dtype,
                        )
                        if animation.storage_note:
                            animation_message = " ".join(
                                part
                                for part in (
                                    self.animation_note,
                                    animation.storage_note,
                                )
                                if part
                            )
                    except Exception as exc:
                        if self._cancel_requested:
                            raise
                        animation_message = (
                            "Scientific result is complete, but the separate 3D "
                            f"animation replay failed: {exc}"
                        )
                        self.stage.emit(animation_message)
                if animation is None:
                    if animation_message:
                        self.result_ready.emit(
                            _SequenceSimulationPayload(
                                result, None, animation_message=animation_message
                            )
                        )
                    else:
                        self.result_ready.emit(result)
                else:
                    self.result_ready.emit(
                        _SequenceSimulationPayload(
                            result,
                            animation,
                            animation_message=animation_message or self.animation_note,
                        )
                    )
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
            adc_events = self.program.adc_events
            result.metadata.update(
                {
                    "stored_timeline": "configured_checkpoints",
                    "configured_playback_times_s": np.asarray(
                        self.checkpoints_s, dtype=float
                    ),
                    "adc_times_s": (
                        np.concatenate([event.sample_times_s for event in adc_events])
                        if adc_events
                        else np.zeros(0, dtype=float)
                    ),
                    "adc_event_indices": (
                        np.concatenate(
                            [
                                np.full(event.num_samples, index, dtype=np.int64)
                                for index, event in enumerate(adc_events)
                            ]
                        )
                        if adc_events
                        else np.zeros(0, dtype=np.int64)
                    ),
                    "adc_sample_dwell_s": (
                        np.concatenate(
                            [
                                np.full(event.num_samples, event.dwell_s, dtype=float)
                                for event in adc_events
                            ]
                        )
                        if adc_events
                        else np.zeros(0, dtype=float)
                    ),
                    "adc_windows_s": np.asarray(
                        [
                            (
                                event.start_s,
                                min(
                                    self.program.duration_s,
                                    event.start_s + event.num_samples * event.dwell_s,
                                ),
                            )
                            for event in adc_events
                        ],
                        dtype=float,
                    ).reshape(-1, 2),
                }
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

    physical_b1_changed = pyqtSignal(object)

    # Generated-sequence forms (notably EPI, CSI, and FLASH) need enough
    # room for a label and its editor alongside the vertical scroll bar.
    # Keeping the focused panel at this width avoids a horizontal scroll bar
    # when switching away from the compact internal FID controls.
    FOCUSED_CONTROL_WIDTH = 600
    # Includes the vertical scrollbar and frame around the controls viewport.
    # At 560 px the viewport is narrower than the controls' minimum size hint,
    # which causes an unnecessary horizontal scrollbar on headless Qt.
    MINIMUM_FOCUSED_CONTROL_WIDTH = 576
    MINIMUM_FOCUSED_VIEWER_WIDTH = 540
    INTERNAL_SOURCE = 0
    EPI_SOURCE = 1
    CSI_SOURCE = 2
    BSSFP_SOURCE = 3
    SS_BSSFP_SOURCE = 4
    RADIAL_ME_BSSFP_SOURCE = 5
    ME_BSSFP_SOURCE = 6
    FLASH_SOURCE = 7
    PULSEQ_SOURCE = 8
    GENERATED_SOURCES = frozenset(range(EPI_SOURCE, FLASH_SOURCE + 1))
    CARTESIAN_3D_SOURCES = frozenset((BSSFP_SOURCE, SS_BSSFP_SOURCE, ME_BSSFP_SOURCE))
    NO_PHANTOM_MESSAGE = (
        "No phantom is loaded in the Phantom tab. Create or load one first."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.program: Optional[SequenceProgram] = None
        self._generated_pulseq_sequence = None
        self._sequence_generation_pending = False
        self._generated_sequence_source_index = None
        self._selected_sequence_source_index = 0
        self._generation_error = ""
        self._probe_frequency_defaults_initialized = False
        self.acquisition: Optional[CartesianAcquisition] = None
        self.acquisition_frames: Optional[CartesianAcquisitionFrames] = None
        self.acquisition_volumes: Optional[CartesianAcquisitionVolumes] = None
        self.spiral_acquisition: Optional[SpiralAcquisition] = None
        self.spectroscopic_acquisition: Optional[SpectroscopicAcquisition] = None
        self.phantom: Optional[Phantom] = None
        self.result = None
        self.magnetization_animation = None
        self._active_session_run_id = None
        self._restoring_session_run = False
        self.probe_result = None
        self._split_csi_data = None
        self._csi_click_view_initialized = False
        self.worker = None
        self.probe_worker = None
        self.pulseq_load_worker = None
        self._pulseq_spoiler_warning_dialog = None
        self.script_process = None
        self.script_output_dialog = None
        self.script_output = None
        self._script_path = None
        self._script_sequence_snapshot = {}
        self._acquisition_compiled = None
        self._simulation_started_at = None
        self._simulation_started_at_utc = None
        self._probe_started_at = None
        self._probe_started_at_utc = None
        self._simulation_progress_started_at = None
        self._simulation_last_progress_at = None
        self._simulation_last_progress_done = 0
        self._simulation_last_progress_total = 1
        self._simulation_progress_rate = None
        self._probe_playback_anchor_wall = None
        self._probe_playback_anchor_time_ms = None
        self._probe_playback_indices = np.zeros(0, dtype=np.int64)
        self._probe_playback_clock_ms = np.zeros(0, dtype=float)
        self._sequence_plot_window_s = None
        self._sequence_plot_pending_window_s = None
        self._preserve_sequence_plot_range_on_next_show = False
        self._rf_waveform_item = None
        self._rf_phase_item = None
        self._gradient_waveform_items = {}
        self._sequence_spoiler_markers = []
        self._rf_designer = None
        self._rf_designer_pulse_data = None
        self._rf_designer_pulse_error = "No RF Pulse Designer waveform is available"
        self.probe_playback_timer = QTimer(self)
        self.probe_playback_timer.setInterval(16)
        self.probe_playback_timer.timeout.connect(self._advance_probe_playback)
        self.sequence_plot_refresh_timer = QTimer(self)
        self.sequence_plot_refresh_timer.setSingleShot(True)
        self.sequence_plot_refresh_timer.setInterval(60)
        self.sequence_plot_refresh_timer.timeout.connect(
            self._refresh_pending_sequence_plot
        )
        self.simulation_time_timer = QTimer(self)
        self.simulation_time_timer.setInterval(1000)
        self.simulation_time_timer.timeout.connect(self._update_simulation_time_label)
        settings = getattr(parent, "app_settings", None)
        self.app_settings = settings
        self.workspace_defaults = WorkspaceDefaults.from_settings(settings)
        self.scanner_parameters = load_scanner_parameters(settings)
        try:
            animation_memory_budget_mib = float(
                settings.value("memory/animation_replay_mib", 512.0)
                if settings is not None
                else 512.0
            )
        except (TypeError, ValueError):
            animation_memory_budget_mib = 512.0
        if not np.isfinite(animation_memory_budget_mib) or not (
            16.0 <= animation_memory_budget_mib <= 1024.0 * 1024.0
        ):
            animation_memory_budget_mib = 512.0
        self.animation_memory_budget_bytes = int(animation_memory_budget_mib * 1024**2)
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
        dynamic_sequence_kernel = (
            str(settings.value("sequence/dynamic_kernel", "optimized"))
            if settings is not None
            else "optimized"
        )
        if dynamic_sequence_kernel not in {
            "optimized",
            "native_parallel",
            "native_serial",
            "metal_hybrid",
            "reference",
        }:
            dynamic_sequence_kernel = "optimized"
        spoiler_mode = (
            str(settings.value("sequence/spoiler_mode", "ideal"))
            if settings is not None
            else "ideal"
        )
        if spoiler_mode not in {"ideal", "gradient"}:
            spoiler_mode = "ideal"
        default_subvoxel_counts = (1, 1, 9)
        subvoxel_counts = []
        for axis, default in zip("xyz", default_subvoxel_counts):
            try:
                count = int(
                    settings.value(f"sequence/subvoxel_spins_{axis}", default)
                    if settings is not None
                    else default
                )
            except (TypeError, ValueError):
                count = default
            subvoxel_counts.append(count if 1 <= count <= 128 else default)
        subvoxel_sampling_method = (
            str(settings.value("sequence/subvoxel_sampling_method", "midpoint"))
            if settings is not None
            else "midpoint"
        )
        if subvoxel_sampling_method not in {"midpoint", "stratified"}:
            subvoxel_sampling_method = "midpoint"
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
            dynamic_sequence_kernel=dynamic_sequence_kernel,
        )
        self._initial_sequence_timestep_us = sequence_timestep_us
        self.spoiler_mode = spoiler_mode
        self.subvoxel_spin_counts = tuple(subvoxel_counts)
        self.subvoxel_sampling_method = subvoxel_sampling_method
        self._build_ui()
        self._connect_rf_designer(parent)
        self._load_internal_sequence()

    def _connect_rf_designer(self, parent) -> None:
        """Subscribe to the main RF designer when hosted by the desktop app."""
        designer = getattr(parent, "rf_designer", None)
        if designer is None:
            return
        self._rf_designer = designer
        self.set_rf_designer_pulse(designer.get_pulse(), reload_sequence=False)
        designer.pulse_changed.connect(self.set_rf_designer_pulse)

    def set_default_fov_mm(self, fov_mm) -> None:
        """Apply newly saved FOV defaults to editable generated-sequence controls."""
        x, y, z = (float(value) for value in fov_mm)
        controls = (
            (self.epi_read_fov_mm, x),
            (self.epi_phase_fov_mm, y),
            (self.csi_read_fov_mm, x),
            (self.csi_phase_fov_mm, y),
            (self.flash_read_fov_mm, x),
            (self.flash_phase_fov_mm, y),
            (self.bssfp_read_fov_mm, x),
            (self.bssfp_phase_fov_mm, y),
            (self.bssfp_partition_fov_mm, z),
            (self.fov_mm, x),
            (self.fov_z_mm, z),
        )
        for control, value in controls:
            control.setValue(value)

    def set_workspace_defaults(self, defaults: WorkspaceDefaults) -> None:
        """Immediately apply newly saved defaults to the active workspace."""
        self.workspace_defaults = defaults
        self.set_default_fov_mm(defaults.sequence_fov_mm)
        self.field_strength_t.setValue(defaults.field_strength_t)
        self.nucleus.setCurrentText(defaults.phantom_nucleus)

    def _export_directory(self) -> Path:
        """Resolve the application-wide configured export directory."""
        provider = getattr(self.window(), "_get_export_directory", None)
        if callable(provider):
            try:
                path = Path(provider())
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                pass
        return workspace_directory("exports")

    def set_rf_designer_pulse(
        self, pulse, state=None, *, reload_sequence: bool = True
    ) -> None:
        """Import the current RF Designer waveform as a reusable excitation shape."""
        try:
            if pulse is None or len(pulse) != 2:
                raise ValueError("design a valid RF pulse in the RF Design tab first")
            b1_gauss, time_s = pulse
            b1_gauss = np.asarray(b1_gauss, dtype=np.complex128).reshape(-1)
            time_s = np.asarray(time_s, dtype=float).reshape(-1)
            if (
                b1_gauss.size == 0
                or b1_gauss.size != time_s.size
                or not np.all(np.isfinite(b1_gauss))
                or not np.all(np.isfinite(time_s))
            ):
                raise ValueError("RF Designer waveform and time axis must be finite")
            if time_s.size > 1 and np.any(np.diff(time_s) <= 0):
                raise ValueError("RF Designer time axis must be strictly increasing")

            if state is None and self._rf_designer is not None:
                state = self._rf_designer.get_state()
            state = dict(state or {})
            duration_s = float(state.get("duration", 0.0)) / 1000.0
            if not np.isfinite(duration_s) or duration_s <= 0:
                duration_s = (
                    float(np.median(np.diff(time_s))) * time_s.size
                    if time_s.size > 1
                    else self.scanner_parameters.rf_raster_time_s
                )
            raster_s = duration_s / b1_gauss.size
            waveform_hz = np.asarray(rf_gauss_to_hz(b1_gauss), np.complex128)
            time_bandwidth_product = state.get("time_bandwidth_product", 0.0)
            time_bandwidth_product = (
                float(time_bandwidth_product)
                if time_bandwidth_product is not None
                else 0.0
            )
            if not np.isfinite(time_bandwidth_product) or time_bandwidth_product <= 0:
                time_bandwidth_product = rf_time_bandwidth_product_from_envelope(
                    waveform_hz
                )
            reference_flip_angle_deg = float(state.get("flip_angle", 0.0))
            if (
                not np.isfinite(reference_flip_angle_deg)
                or reference_flip_angle_deg <= 0
            ):
                reference_flip_angle_deg = 360.0 * abs(np.sum(waveform_hz) * raster_s)
            if reference_flip_angle_deg <= 0:
                raise ValueError(
                    "RF Designer pulse needs a positive reference flip angle"
                )
            waveform_hz.setflags(write=False)
            self._rf_designer_pulse_data = {
                "waveform_hz": waveform_hz,
                "raster_s": raster_s,
                "duration_s": duration_s,
                "flip_angle_deg": reference_flip_angle_deg,
                "name": str(state.get("pulse_type", "custom")),
                "frequency_offset_hz": float(state.get("freq_offset", 0.0)),
                "time_bandwidth_product": time_bandwidth_product,
            }
            self._rf_designer_pulse_error = ""
            for prefix in (
                "epi",
                "csi",
                "flash",
                "bssfp",
                "ss_bssfp",
                "radial_me",
                "me_bssfp",
            ):
                duration_control = getattr(self, f"{prefix}_rf_duration_ms", None)
                if duration_control is None:
                    continue
                previous = duration_control.blockSignals(True)
                if self._selected_shared_rf_pulse_type(prefix) == "designer":
                    duration_control.setValue(duration_s * 1000.0)
                duration_control.blockSignals(previous)
                if self._selected_shared_rf_pulse_type(prefix) == "designer":
                    self._update_shared_rf_controls(prefix)
        except Exception as exc:
            self._rf_designer_pulse_data = None
            self._rf_designer_pulse_error = str(exc)

        if reload_sequence and any(
            hasattr(self, f"{prefix}_rf_pulse_type")
            and self._selected_shared_rf_pulse_type(prefix) == "designer"
            for prefix in (
                "epi",
                "csi",
                "flash",
                "bssfp",
                "ss_bssfp",
                "radial_me",
                "me_bssfp",
            )
        ):
            self._request_generated_sequence_refresh()

    def _load_sequence_rf_pulse(self, prefix):
        """Load a Free-Mode-compatible RF file directly from Sequence Mode."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load RF Pulse",
            "",
            "Pulse Files (*.exc *.dat *.txt *.csv);;All Files (*)",
        )
        if not filename:
            return
        try:
            path = Path(filename)
            if path.suffix.lower() == ".exc":
                from ..pulse_loader import load_pulse_from_file

                b1_gauss, time_s, metadata = load_pulse_from_file(path)
            else:
                dialog = PulseImportDialog(self, str(path))
                if dialog.exec_() != QDialog.Accepted:
                    return
                options = dialog.get_options()
                from ..pulse_loader import load_amp_phase_dat

                b1_gauss, time_s, metadata = load_amp_phase_dat(
                    path,
                    duration_s=options["duration_s"],
                    amplitude_unit=options["amp_unit"],
                    phase_unit=options["phase_unit"],
                    layout=options["layout"],
                )
            duration_s = float(getattr(metadata, "duration", 0.0))
            if duration_s <= 0:
                time_values = np.asarray(time_s, dtype=float)
                duration_s = (
                    float(np.median(np.diff(time_values))) * time_values.size
                    if time_values.size > 1
                    else self.scanner_parameters.rf_raster_time_s
                )
            flip_angle = float(getattr(metadata, "flip_angle", 0.0))
            bandwidth_factor = getattr(metadata, "bwfac", 0.0)
            bandwidth_factor = (
                float(bandwidth_factor) if bandwidth_factor is not None else 0.0
            )
            self.set_rf_designer_pulse(
                (b1_gauss, time_s),
                {
                    "duration": duration_s * 1000.0,
                    "flip_angle": flip_angle,
                    "pulse_type": path.name,
                    "freq_offset": 0.0,
                    "time_bandwidth_product": bandwidth_factor,
                },
                reload_sequence=False,
            )
            if self._rf_designer_pulse_data is None:
                raise ValueError(self._rf_designer_pulse_error)
            getattr(self, f"{prefix}_rf_pulse_type").setCurrentText("RF Pulse Designer")
            self._update_shared_rf_controls(prefix)
        except Exception as exc:
            QMessageBox.critical(self, "RF pulse load failed", str(exc))

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        splitter = QSplitter()
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        control_column = QWidget()
        control_column_layout = QVBoxLayout(control_column)
        control_column_layout.setContentsMargins(0, 0, 0, 0)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(controls)
        self.controls_scroll.setMinimumWidth(self.MINIMUM_FOCUSED_CONTROL_WIDTH)
        self.controls_scroll.setMaximumWidth(680)
        control_column_layout.addWidget(self.controls_scroll, 1)
        splitter.addWidget(control_column)
        self.workspace_splitter = splitter

        sequence_group = QGroupBox()
        sequence_layout = QVBoxLayout(sequence_group)
        sequence_title = QLabel("Sequence")
        sequence_title_font = sequence_title.font()
        sequence_title_font.setBold(True)
        sequence_title_font.setPointSize(max(sequence_title_font.pointSize() + 2, 12))
        sequence_title.setFont(sequence_title_font)
        self.sequence_title = sequence_title
        sequence_layout.addWidget(sequence_title)
        sequence_layout.addWidget(QLabel("Source / mode"))
        self.sequence_source = QComboBox()
        self.sequence_source.addItems(
            [
                "Internal FID",
                "EPI",
                "CSI",
                "bSSFP (3D)",
                "SS-bSSFP (3D)",
                "Radial ME-bSSFP (3D)",
                "ME-bSSFP (3D, Cartesian)",
                "FLASH (2D)",
                "Pulseq .seq file",
            ]
        )
        self.sequence_source.currentIndexChanged.connect(self._source_changed)
        self.sequence_source.setToolTip(
            "Build EPI, CSI, FLASH, Cartesian bSSFP, spectrally selective "
            "bSSFP, Cartesian or radial multi-echo bSSFP interactively, or "
            "load a Pulseq file"
        )
        sequence_layout.addWidget(self.sequence_source)

        generation_grid = QGridLayout()
        generation_grid.setHorizontalSpacing(10)
        generation_grid.setVerticalSpacing(6)
        generation_grid.setContentsMargins(0, 4, 0, 2)
        generation_grid.setColumnStretch(0, 1)
        generation_grid.setColumnStretch(1, 1)

        self.sequence_live_preview = QCheckBox("Live preview")
        self.sequence_live_preview.setChecked(False)
        self.sequence_live_preview.setToolTip(
            "Regenerate the selected sequence after every parameter change. "
            "Leave this off for long sequences and generate once after setup."
        )
        self.sequence_live_preview.toggled.connect(self._sequence_live_preview_changed)
        self.generate_sequence_button = QPushButton("Generate sequence")
        self.generate_sequence_button.setEnabled(False)
        self.generate_sequence_button.setToolTip(
            "Generate the sequence from the current parameters and refresh the timeline"
        )
        self.generate_sequence_button.clicked.connect(self._generate_sequence_clicked)

        time_resolution_row = QHBoxLayout()
        time_resolution_row.addWidget(QLabel("Time resolution"))
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
        time_resolution_row.addWidget(self.simulation_timestep_us, 1)

        self.load_pulseq_button = QPushButton("Load Pulseq…")
        self.load_pulseq_button.clicked.connect(self._load_pulseq_file)

        run_script_button = QPushButton("Run Python script…")
        self.run_script_button = run_script_button
        self.run_script_button.setToolTip(
            "Run a selected Python sequence-generation script inside the GUI. "
            "A newly written Pulseq file is loaded automatically."
        )
        self.run_script_button.clicked.connect(self._run_python_script)

        self.export_pulseq_button = QPushButton("Export Pulseq…")
        self.export_pulseq_button.setEnabled(False)
        self.export_pulseq_button.setToolTip(
            "Export the generated sequence as Pulseq, a reproducing Jupyter "
            "notebook, or both"
        )
        self.export_pulseq_button.clicked.connect(self._export_pulseq)

        # Arrange the sequence workflow from source to output in two columns:
        # choose a source, configure generation, generate, then export.
        generation_grid.addWidget(self.load_pulseq_button, 0, 0)
        generation_grid.addWidget(run_script_button, 0, 1)
        generation_grid.addLayout(time_resolution_row, 1, 0)
        generation_grid.addWidget(self.sequence_live_preview, 1, 1)
        generation_grid.addWidget(self.generate_sequence_button, 2, 0, 1, 2)
        generation_grid.addWidget(self.export_pulseq_button, 3, 0, 1, 2)
        sequence_layout.addLayout(generation_grid)

        self.sequence_info = QLabel()
        self.sequence_info.setWordWrap(True)
        sequence_layout.addWidget(self.sequence_info)
        self.sequence_summary_table = QTableWidget(0, 2)
        self.sequence_summary_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.sequence_summary_table.verticalHeader().setVisible(False)
        self.sequence_summary_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.sequence_summary_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.sequence_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sequence_summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.sequence_summary_table.setAlternatingRowColors(True)
        self.sequence_summary_table.setWordWrap(True)
        self.sequence_summary_table.setShowGrid(False)
        self.sequence_summary_table.setVisible(False)
        self.sequence_summary_table.horizontalHeader().sectionResized.connect(
            self._schedule_sequence_summary_table_fit
        )
        sequence_layout.addWidget(self.sequence_summary_table)
        controls_layout.addWidget(sequence_group)

        self.acquisition_group = QGroupBox("2D acquisition (EPI / spiral)")
        acquisition_form = _left_aligned_form(self.acquisition_group)
        self.acquisition_hint = QLabel(
            "Choose a Cartesian EPI echo train or a continuous centre-out "
            "spiral readout."
        )
        self.acquisition_hint.setWordWrap(True)
        self.epi_readout_trajectory = QComboBox()
        self.epi_readout_trajectory.setObjectName("epi_readout_trajectory")
        self.epi_readout_trajectory.addItems(["Cartesian EPI", "Spiral"])
        self.read_matrix = QSpinBox()
        self.read_matrix.setRange(2, 512)
        self.read_matrix.setValue(16)
        self.phase_matrix = QSpinBox()
        self.phase_matrix.setRange(2, 512)
        self.phase_matrix.setValue(16)
        default_fov_x, default_fov_y, default_fov_z = (
            self.workspace_defaults.sequence_fov_mm
        )
        self.epi_read_fov_mm = self._parameter_spin(0.1, 10000.0, default_fov_x, " mm")
        self.epi_phase_fov_mm = self._parameter_spin(0.1, 10000.0, default_fov_y, " mm")
        self.sampling_bandwidth_khz = self._sampling_bandwidth_spin(50.0)
        self.epi_flip_angle_deg = QDoubleSpinBox()
        self.epi_flip_angle_deg.setRange(0.1, 360.0)
        self.epi_flip_angle_deg.setDecimals(2)
        self.epi_flip_angle_deg.setValue(90.0)
        self.epi_flip_angle_deg.setSuffix("°")
        self.epi_use_ernst_angle = QCheckBox("Use phantom-derived angle")
        self.epi_use_ernst_angle.setToolTip(
            "Set the constant flip angle to acos(exp(-TR/T1)). This ideal "
            "spoiled steady-state model assumes transverse spoiling, but does "
            "not specifically require RF spoiling and does not depend on T2."
        )
        self.epi_ernst_info = QLabel()
        self.epi_ernst_info.setWordWrap(True)
        self.epi_variable_flip_angle = QCheckBox("Enable across repetitions")
        self.epi_variable_flip_angle.setToolTip(
            "Use a hyperpolarized variable-flip-angle schedule that changes "
            "once per complete EPI repetition"
        )
        self.epi_vfa_final_flip_angle_deg = self._parameter_spin(0.1, 90.0, 90.0, "°")
        self.epi_vfa_final_flip_angle_deg.setToolTip(
            "Terminal angle of the backwards-calculated schedule; 90° fully "
            "uses the idealized remaining longitudinal magnetization"
        )
        self.epi_vfa_final_flip_angle_deg.setEnabled(False)
        self.epi_vfa_info = QLabel("Off")
        self.epi_vfa_info.setToolTip(
            "Nagashima VFA schedule without T1-decay compensation "
            "(doi:10.1016/j.jmr.2007.10.011)"
        )
        self.epi_slice_count = QSpinBox()
        self.epi_slice_count.setRange(1, 128)
        self.epi_slice_count.setValue(1)
        self.epi_repetitions = QSpinBox()
        self.epi_repetitions.setRange(1, 10000)
        self.epi_repetitions.setValue(1)
        self.epi_repetition_time_ms = QDoubleSpinBox()
        self.epi_repetition_time_ms.setRange(0.1, 100000.0)
        self.epi_repetition_time_ms.setDecimals(3)
        self.epi_repetition_time_ms.setValue(1000.0)
        self.epi_repetition_time_ms.setSuffix(" ms")
        self.epi_repetition_time_ms.setToolTip(
            "Start-to-start interval between complete EPI or spiral slice-package "
            "acquisitions"
        )
        self.epi_slice_thickness_mm = QDoubleSpinBox()
        self.epi_slice_thickness_mm.setRange(0.05, 100.0)
        self.epi_slice_thickness_mm.setDecimals(3)
        self.epi_slice_thickness_mm.setValue(3.0)
        self.epi_slice_thickness_mm.setSuffix(" mm")
        self.epi_slice_gap_mm = QDoubleSpinBox()
        self.epi_slice_gap_mm.setObjectName("epi_slice_gap_mm")
        self.epi_slice_gap_mm.setRange(0.0, 100.0)
        self.epi_slice_gap_mm.setDecimals(3)
        self.epi_slice_gap_mm.setValue(0.0)
        self.epi_slice_gap_mm.setSuffix(" mm")
        self.epi_slice_gap_mm.setToolTip(
            "Edge-to-edge gap between adjacent slices; slice-centre spacing "
            "is thickness plus gap"
        )
        (
            self.epi_slice_orientation,
            self.epi_read_gradient_axis,
            self.epi_phase_gradient_axis,
            self.epi_slice_gradient_axis,
        ) = self._two_dimensional_orientation_controls("epi")
        self.epi_slice_offset_mm = self._parameter_spin(-10000.0, 10000.0, 0.0, " mm")
        self.epi_slice_offset_mm.setObjectName("epi_slice_offset_mm")
        self.epi_slice_offset_mm.setToolTip(
            "Offset of the centre of the complete slice package along the "
            "selected slice-normal direction"
        )
        self.epi_echo_time_ms = self._parameter_spin(0.1, 10000.0, 20.0, " ms")
        self.epi_echo_time_ms.setObjectName("epi_echo_time_ms")
        self.epi_echo_time_ms.setToolTip(
            "Time from the excitation centre to the centre of EPI k-space; "
            "for centre-out spiral this is the first ADC sample"
        )
        self.epi_spiral_turns = QDoubleSpinBox()
        self.epi_spiral_turns.setObjectName("epi_spiral_turns")
        self.epi_spiral_turns.setRange(0.25, 512.0)
        self.epi_spiral_turns.setDecimals(2)
        self.epi_spiral_turns.setValue(8.0)
        self.epi_spiral_turns.setSuffix(" turns")
        self.epi_spiral_turns.setToolTip(
            "Number of revolutions in the single centre-out spiral interleaf"
        )
        self.epi_spiral_turns.setEnabled(False)
        self.epi_spoil_after_slice = QCheckBox("Enable after each slice")
        self.epi_spoil_after_slice.setChecked(True)
        self.epi_spoil_after_slice.setToolTip(
            "Apply the configured gradient spoiler after every EPI slice readout"
        )
        self.epi_rf_spoiling = QCheckBox("Enable RF spoiling")
        self.epi_rf_spoiling.setChecked(False)
        self.epi_rf_spoiling_increment_deg = self._parameter_spin(
            -360.0, 360.0, 117.0, "°"
        )
        self.epi_spoiler_cycles_per_slice = QDoubleSpinBox()
        self.epi_spoiler_cycles_per_slice.setRange(0.0, 1000.0)
        self.epi_spoiler_cycles_per_slice.setDecimals(3)
        self.epi_spoiler_cycles_per_slice.setValue(8.0)
        self.epi_spoiler_cycles_per_slice.setSuffix(" cycles/slice")
        self.epi_spoiler_cycles_per_slice.setToolTip(
            "Through-slice dephasing across one slice thickness"
        )
        self.epi_spoiler_cycles_per_voxel = QDoubleSpinBox()
        self.epi_spoiler_cycles_per_voxel.setRange(0.0, 1000.0)
        self.epi_spoiler_cycles_per_voxel.setDecimals(3)
        self.epi_spoiler_cycles_per_voxel.setValue(0.0)
        self.epi_spoiler_cycles_per_voxel.setSuffix(" cycles/voxel")
        self.epi_spoiler_cycles_per_voxel.setToolTip(
            "Additional x/y dephasing across one acquired voxel; a non-integer "
            "value avoids exact refocusing on the voxel grid"
        )
        self.epi_spoiler_duration_ms = QDoubleSpinBox()
        self.epi_spoiler_duration_ms.setRange(0.001, 1000.0)
        self.epi_spoiler_duration_ms.setDecimals(3)
        self.epi_spoiler_duration_ms.setValue(4.0)
        self.epi_spoiler_duration_ms.setSuffix(" ms")
        self.dwell_info = QLabel()
        self.pixel_bandwidth_info = QLabel()
        acquisition_form.addRow(self.acquisition_hint)
        _add_form_section(acquisition_form, "Spatial encoding")
        acquisition_form.addRow("Readout trajectory", self.epi_readout_trajectory)
        acquisition_form.addRow("Read matrix", self.read_matrix)
        acquisition_form.addRow("Phase matrix", self.phase_matrix)
        acquisition_form.addRow("Read FOV", self.epi_read_fov_mm)
        acquisition_form.addRow("Phase FOV", self.epi_phase_fov_mm)
        acquisition_form.addRow("Sampling bandwidth", self.sampling_bandwidth_khz)
        acquisition_form.addRow("Spiral revolutions", self.epi_spiral_turns)
        _add_form_section(acquisition_form, "RF pulse")
        acquisition_form.addRow("Flip angle (constant)", self.epi_flip_angle_deg)
        acquisition_form.addRow("Ernst angle", self.epi_use_ernst_angle)
        acquisition_form.addRow("Ernst calculation", self.epi_ernst_info)
        acquisition_form.addRow("Variable flip angle", self.epi_variable_flip_angle)
        acquisition_form.addRow(
            "VFA final flip angle", self.epi_vfa_final_flip_angle_deg
        )
        acquisition_form.addRow("VFA schedule", self.epi_vfa_info)
        self._add_shared_rf_controls(
            acquisition_form,
            "epi",
            pulse_type="Sinc",
            duration_ms=3.0,
            sinc_lobes=3,
        )
        _add_form_section(acquisition_form, "Slice selection")
        acquisition_form.addRow("Plane preset", self.epi_slice_orientation)
        acquisition_form.addRow("Read gradient direction", self.epi_read_gradient_axis)
        acquisition_form.addRow(
            "Phase gradient direction", self.epi_phase_gradient_axis
        )
        acquisition_form.addRow(
            "Slice gradient direction", self.epi_slice_gradient_axis
        )
        acquisition_form.addRow("Slices", self.epi_slice_count)
        acquisition_form.addRow("Slice thickness", self.epi_slice_thickness_mm)
        acquisition_form.addRow("Slice gap", self.epi_slice_gap_mm)
        acquisition_form.addRow("Slice package offset", self.epi_slice_offset_mm)
        _add_form_section(acquisition_form, "Timing")
        acquisition_form.addRow("Echo time (TE)", self.epi_echo_time_ms)
        acquisition_form.addRow(
            "Acquisition interval (start-to-start)",
            self.epi_repetition_time_ms,
        )
        acquisition_form.addRow("Repetitions", self.epi_repetitions)
        _add_form_section(acquisition_form, "Spoiling")
        acquisition_form.addRow("RF spoiling", self.epi_rf_spoiling)
        acquisition_form.addRow(
            "RF spoiling increment", self.epi_rf_spoiling_increment_deg
        )
        acquisition_form.addRow("Gradient spoiler", self.epi_spoil_after_slice)
        acquisition_form.addRow(
            "Through-slice spoiler", self.epi_spoiler_cycles_per_slice
        )
        acquisition_form.addRow("In-plane spoiler", self.epi_spoiler_cycles_per_voxel)
        acquisition_form.addRow("Spoiler duration", self.epi_spoiler_duration_ms)
        _add_form_section(acquisition_form, "Derived sampling")
        acquisition_form.addRow("ADC dwell", self.dwell_info)
        acquisition_form.addRow("Pixel bandwidth", self.pixel_bandwidth_info)
        self.acquisition_group.setVisible(False)
        controls_layout.addWidget(self.acquisition_group)

        self.csi_group = QGroupBox("CSI acquisition")
        csi_form = _left_aligned_form(self.csi_group)
        csi_hint = QLabel(
            "2D phase-encoded chemical-shift imaging with one FID per k-space "
            "location. Spectral bandwidth and points define the FID."
        )
        csi_hint.setWordWrap(True)
        self.csi_read_matrix = QSpinBox()
        self.csi_read_matrix.setRange(1, 128)
        self.csi_read_matrix.setValue(12)
        self.csi_phase_matrix = QSpinBox()
        self.csi_phase_matrix.setRange(1, 128)
        self.csi_phase_matrix.setValue(12)
        self.csi_read_fov_mm = self._parameter_spin(0.1, 10000.0, default_fov_x, " mm")
        self.csi_phase_fov_mm = self._parameter_spin(0.1, 10000.0, default_fov_y, " mm")
        self.csi_spectral_points = QSpinBox()
        self.csi_spectral_points.setRange(2, 8192)
        self.csi_spectral_points.setValue(128)
        self.csi_bandwidth_hz = QDoubleSpinBox()
        self.csi_bandwidth_hz.setRange(10.0, 1_000_000.0)
        self.csi_bandwidth_hz.setDecimals(3)
        self.csi_bandwidth_hz.setValue(4000.0)
        self.csi_bandwidth_hz.setSuffix(" Hz")
        self.csi_encoding_order = QComboBox()
        self.csi_encoding_order.addItems(["linear", "centric", "spiral"])
        self.csi_flip_angle_deg = self._parameter_spin(0.1, 360.0, 15.0, "°")
        self.csi_use_ernst_angle = QCheckBox("Use phantom-derived angle")
        self.csi_use_ernst_angle.setToolTip(
            "Set the constant flip angle to acos(exp(-TR/T1)). This ideal "
            "spoiled steady-state model assumes transverse spoiling, but does "
            "not specifically require RF spoiling and does not depend on T2."
        )
        self.csi_ernst_info = QLabel()
        self.csi_ernst_info.setWordWrap(True)
        self.csi_variable_flip_angle = QCheckBox("Enable across phase encodes")
        self.csi_variable_flip_angle.setToolTip(
            "Change the excitation angle at every CSI phase-encoding step and "
            "restart the schedule for each repetition"
        )
        self.csi_vfa_final_flip_angle_deg = self._parameter_spin(0.1, 90.0, 90.0, "°")
        self.csi_vfa_final_flip_angle_deg.setToolTip(
            "Terminal angle of the backwards-calculated schedule; 90° fully "
            "uses the idealized remaining longitudinal magnetization"
        )
        self.csi_vfa_final_flip_angle_deg.setEnabled(False)
        self.csi_vfa_info = QLabel("Off")
        self.csi_vfa_info.setToolTip(
            "Nagashima VFA schedule without T1-decay compensation "
            "(doi:10.1016/j.jmr.2007.10.011)"
        )
        self.csi_slice_thickness_mm = self._parameter_spin(0.05, 100.0, 10.0, " mm")
        (
            self.csi_slice_orientation,
            self.csi_read_gradient_axis,
            self.csi_phase_gradient_axis,
            self.csi_slice_gradient_axis,
        ) = self._two_dimensional_orientation_controls("csi")
        self.csi_slice_offset_mm = self._parameter_spin(-10000.0, 10000.0, 0.0, " mm")
        self.csi_slice_offset_mm.setObjectName("csi_slice_offset_mm")
        self.csi_echo_time_ms = self._parameter_spin(0.1, 10000.0, 6.0, " ms")
        self.csi_repetition_time_ms = self._parameter_spin(0.1, 100000.0, 100.0, " ms")
        self.csi_repetitions = QSpinBox()
        self.csi_repetitions.setRange(1, 10000)
        self.csi_repetitions.setValue(1)
        self.csi_acquisition_interval_ms = self._acquisition_interval_spin()
        self.csi_rf_spoiling = QCheckBox("Enable RF spoiling")
        self.csi_rf_spoiling.setChecked(False)
        self.csi_rf_spoiling_increment_deg = self._parameter_spin(
            -360.0, 360.0, 117.0, "°"
        )
        self.csi_spoil_after_readout = QCheckBox("Enable after each FID")
        self.csi_spoil_after_readout.setChecked(True)
        self.csi_spoiler_cycles_per_slice = self._parameter_spin(
            0.0, 1000.0, 4.0, " cycles/slice"
        )
        self.csi_spoiler_cycles_per_voxel = self._parameter_spin(
            0.0, 1000.0, 0.0, " cycles/voxel"
        )
        self.csi_spoiler_duration_ms = self._parameter_spin(0.001, 1000.0, 2.0, " ms")
        self.csi_dwell_info = QLabel()
        self.csi_resolution_info = QLabel()
        csi_form.addRow(csi_hint)
        _add_form_section(csi_form, "Spatial encoding")
        csi_form.addRow("Read phase matrix", self.csi_read_matrix)
        csi_form.addRow("Phase phase matrix", self.csi_phase_matrix)
        csi_form.addRow("Read FOV", self.csi_read_fov_mm)
        csi_form.addRow("Phase FOV", self.csi_phase_fov_mm)
        csi_form.addRow("Encoding order", self.csi_encoding_order)
        _add_form_section(csi_form, "Spectral sampling")
        csi_form.addRow("Spectral points", self.csi_spectral_points)
        csi_form.addRow("Spectral bandwidth", self.csi_bandwidth_hz)
        _add_form_section(csi_form, "RF pulse")
        csi_form.addRow("Flip angle (constant)", self.csi_flip_angle_deg)
        csi_form.addRow("Ernst angle", self.csi_use_ernst_angle)
        csi_form.addRow("Ernst calculation", self.csi_ernst_info)
        csi_form.addRow("Variable flip angle", self.csi_variable_flip_angle)
        csi_form.addRow("VFA final flip angle", self.csi_vfa_final_flip_angle_deg)
        csi_form.addRow("VFA schedule", self.csi_vfa_info)
        self._add_shared_rf_controls(
            csi_form,
            "csi",
            pulse_type="Sinc",
            duration_ms=3.0,
            sinc_lobes=3,
        )
        _add_form_section(csi_form, "Slice selection")
        csi_form.addRow("Plane preset", self.csi_slice_orientation)
        csi_form.addRow("Read gradient direction", self.csi_read_gradient_axis)
        csi_form.addRow("Phase gradient direction", self.csi_phase_gradient_axis)
        csi_form.addRow("Slice gradient direction", self.csi_slice_gradient_axis)
        csi_form.addRow("Slice thickness", self.csi_slice_thickness_mm)
        csi_form.addRow("Slice offset", self.csi_slice_offset_mm)
        _add_form_section(csi_form, "Timing")
        csi_form.addRow("Echo time (TE)", self.csi_echo_time_ms)
        csi_form.addRow("Repetition time (TR)", self.csi_repetition_time_ms)
        csi_form.addRow("Repetitions", self.csi_repetitions)
        csi_form.addRow(
            "Acquisition interval (start-to-start)",
            self.csi_acquisition_interval_ms,
        )
        _add_form_section(csi_form, "Spoiling")
        csi_form.addRow("RF spoiling", self.csi_rf_spoiling)
        csi_form.addRow("RF spoiling increment", self.csi_rf_spoiling_increment_deg)
        csi_form.addRow("Gradient spoiler", self.csi_spoil_after_readout)
        csi_form.addRow("Through-slice spoiler", self.csi_spoiler_cycles_per_slice)
        csi_form.addRow("In-plane spoiler", self.csi_spoiler_cycles_per_voxel)
        csi_form.addRow("Spoiler duration", self.csi_spoiler_duration_ms)
        _add_form_section(csi_form, "Derived sampling")
        csi_form.addRow("ADC dwell", self.csi_dwell_info)
        csi_form.addRow("Spectral resolution", self.csi_resolution_info)
        self.csi_group.setVisible(False)
        controls_layout.addWidget(self.csi_group)

        self.flash_group = QGroupBox("FLASH acquisition (2D)")
        flash_form = _left_aligned_form(self.flash_group)
        flash_hint = QLabel(
            "Slice-selective Cartesian spoiled gradient echo with configurable "
            "RF and gradient spoiling. One readout line is acquired per TR."
        )
        flash_hint.setWordWrap(True)
        self.flash_read_matrix = QSpinBox()
        self.flash_read_matrix.setRange(2, 512)
        self.flash_read_matrix.setValue(64)
        self.flash_phase_matrix = QSpinBox()
        self.flash_phase_matrix.setRange(1, 512)
        self.flash_phase_matrix.setValue(64)
        self.flash_read_fov_mm = self._parameter_spin(
            0.1, 10000.0, default_fov_x, " mm"
        )
        self.flash_phase_fov_mm = self._parameter_spin(
            0.1, 10000.0, default_fov_y, " mm"
        )
        self.flash_sampling_bandwidth_khz = self._sampling_bandwidth_spin(100.0)
        self.flash_flip_angle_deg = self._parameter_spin(0.1, 360.0, 15.0, "°")
        self.flash_use_ernst_angle = QCheckBox("Use phantom-derived angle")
        self.flash_use_ernst_angle.setToolTip(
            "Set the flip angle to acos(exp(-TR/T1)). This ideal spoiled "
            "steady-state model assumes transverse spoiling, but does not "
            "specifically require RF spoiling and does not depend on T2."
        )
        self.flash_ernst_info = QLabel()
        self.flash_ernst_info.setWordWrap(True)
        (
            self.flash_slice_orientation,
            self.flash_read_gradient_axis,
            self.flash_phase_gradient_axis,
            self.flash_slice_gradient_axis,
        ) = self._two_dimensional_orientation_controls("flash")
        self.flash_slice_count = QSpinBox()
        self.flash_slice_count.setRange(1, 128)
        self.flash_slice_count.setValue(1)
        self.flash_slice_thickness_mm = self._parameter_spin(0.05, 100.0, 3.0, " mm")
        self.flash_slice_gap_mm = self._parameter_spin(0.0, 100.0, 0.0, " mm")
        self.flash_slice_offset_mm = self._parameter_spin(-10000.0, 10000.0, 0.0, " mm")
        self.flash_echo_time_ms = self._parameter_spin(0.1, 10000.0, 5.0, " ms")
        self.flash_repetition_time_ms = self._parameter_spin(0.1, 100000.0, 15.0, " ms")
        self.flash_repetitions = QSpinBox()
        self.flash_repetitions.setRange(1, 10000)
        self.flash_repetitions.setValue(1)
        self.flash_acquisition_interval_ms = self._acquisition_interval_spin()
        self.flash_rf_spoiling = QCheckBox("Enable RF spoiling")
        self.flash_rf_spoiling.setChecked(True)
        self.flash_rf_spoiling_increment_deg = self._parameter_spin(
            -360.0, 360.0, 117.0, "°"
        )
        self.flash_auto_spoiler = QCheckBox("Minimize coherent signal automatically")
        self.flash_auto_spoiler.setChecked(True)
        self.flash_auto_spoiler.setToolTip(
            "Choose the smallest through-slice and in-plane spoiler moments that "
            "reach a first coherence null across the current phantom voxel. "
            "Disable this option to edit both spoiler strengths manually."
        )
        self.flash_spoiler_cycles_per_slice = self._parameter_spin(
            0.0, 1000.0, 4.0, " cycles/slice"
        )
        self.flash_spoiler_cycles_per_voxel = self._parameter_spin(
            0.0, 1000.0, 0.0, " cycles/voxel"
        )
        self.flash_spoiler_cycles_per_slice.setDisabled(True)
        self.flash_spoiler_cycles_per_voxel.setDisabled(True)
        self.flash_spoiler_duration_ms = self._parameter_spin(0.001, 1000.0, 2.0, " ms")
        self.flash_dwell_info = QLabel()
        # Retain the original FLASH-specific objects as a compatibility surface
        # for callers that inspect the per-sequence check directly.  The visible
        # UI uses the shared Spoiling quality unit created below.
        self.flash_spoiler_info = QLabel()
        self.flash_spoiler_info.setWordWrap(True)
        self.flash_apply_recommended_grid = QPushButton(
            "Apply train-safe subvoxel grid"
        )
        self.flash_apply_recommended_grid.setToolTip(
            "Apply the smallest tested regular midpoint grid whose retained "
            "coherence follows the continuous voxel throughout this FLASH train."
        )
        flash_form.addRow(flash_hint)
        _add_form_section(flash_form, "Spatial encoding")
        flash_form.addRow("Read matrix", self.flash_read_matrix)
        flash_form.addRow("Phase matrix", self.flash_phase_matrix)
        flash_form.addRow("Read FOV", self.flash_read_fov_mm)
        flash_form.addRow("Phase FOV", self.flash_phase_fov_mm)
        flash_form.addRow("Sampling bandwidth", self.flash_sampling_bandwidth_khz)
        _add_form_section(flash_form, "RF pulse")
        flash_form.addRow("Flip angle", self.flash_flip_angle_deg)
        flash_form.addRow("Ernst angle", self.flash_use_ernst_angle)
        flash_form.addRow("Ernst calculation", self.flash_ernst_info)
        self._add_shared_rf_controls(
            flash_form,
            "flash",
            pulse_type="Sinc",
            duration_ms=1.0,
            sinc_lobes=3,
        )
        _add_form_section(flash_form, "Slice selection")
        flash_form.addRow("Plane preset", self.flash_slice_orientation)
        flash_form.addRow("Read gradient direction", self.flash_read_gradient_axis)
        flash_form.addRow("Phase gradient direction", self.flash_phase_gradient_axis)
        flash_form.addRow("Slice gradient direction", self.flash_slice_gradient_axis)
        flash_form.addRow("Slices", self.flash_slice_count)
        flash_form.addRow("Slice thickness", self.flash_slice_thickness_mm)
        flash_form.addRow("Slice gap", self.flash_slice_gap_mm)
        flash_form.addRow("Slice package offset", self.flash_slice_offset_mm)
        _add_form_section(flash_form, "Timing")
        flash_form.addRow("Echo time (TE)", self.flash_echo_time_ms)
        flash_form.addRow("Repetition time (TR)", self.flash_repetition_time_ms)
        flash_form.addRow("Dynamic repetitions", self.flash_repetitions)
        flash_form.addRow(
            "Acquisition interval (start-to-start)",
            self.flash_acquisition_interval_ms,
        )
        _add_form_section(flash_form, "Spoiling")
        flash_form.addRow("RF spoiling", self.flash_rf_spoiling)
        flash_form.addRow("RF spoiling increment", self.flash_rf_spoiling_increment_deg)
        flash_form.addRow("Auto spoiler", self.flash_auto_spoiler)
        flash_form.addRow("Through-slice spoiler", self.flash_spoiler_cycles_per_slice)
        flash_form.addRow("In-plane spoiler", self.flash_spoiler_cycles_per_voxel)
        flash_form.addRow("Spoiler duration", self.flash_spoiler_duration_ms)
        _add_form_section(flash_form, "Derived sampling")
        flash_form.addRow("ADC dwell", self.flash_dwell_info)
        self.flash_group.setVisible(False)
        controls_layout.addWidget(self.flash_group)

        self.bssfp_group = QGroupBox("bSSFP acquisition (3D)")
        bssfp_form = _left_aligned_form(self.bssfp_group)
        bssfp_hint = QLabel(
            "Fully balanced non-selective 3D Cartesian bSSFP. Phase and "
            "partition gradients are rewound in every TR."
        )
        bssfp_hint.setWordWrap(True)
        (
            self.bssfp_read_gradient_axis,
            self.bssfp_phase_gradient_axis,
            self.bssfp_partition_gradient_axis,
        ) = self._three_dimensional_orientation_controls("bssfp")
        self.bssfp_read_matrix = QSpinBox()
        self.bssfp_read_matrix.setRange(2, 256)
        self.bssfp_read_matrix.setValue(8)
        self.bssfp_phase_matrix = QSpinBox()
        self.bssfp_phase_matrix.setRange(1, 256)
        self.bssfp_phase_matrix.setValue(8)
        self.bssfp_partition_matrix = QSpinBox()
        self.bssfp_partition_matrix.setRange(1, 256)
        self.bssfp_partition_matrix.setValue(4)
        self.bssfp_read_fov_mm = self._parameter_spin(
            0.1, 10000.0, default_fov_x, " mm"
        )
        self.bssfp_phase_fov_mm = self._parameter_spin(
            0.1, 10000.0, default_fov_y, " mm"
        )
        self.bssfp_partition_fov_mm = self._parameter_spin(
            0.1, 10000.0, default_fov_z, " mm"
        )
        self.bssfp_bandwidth_khz = self._sampling_bandwidth_spin(10.0)
        self.bssfp_flip_angle_deg = self._parameter_spin(0.1, 360.0, 15.0, "°")
        self.bssfp_repetition_time_ms = self._parameter_spin(0.1, 10000.0, 10.0, " ms")
        self.bssfp_phase_start_deg = self._parameter_spin(-360.0, 360.0, 180.0, "°")
        self.bssfp_phase_increment_deg = self._parameter_spin(-360.0, 360.0, 180.0, "°")
        self.bssfp_alpha_half_phase_deg = self._parameter_spin(-360.0, 360.0, 0.0, "°")
        self.bssfp_alpha_half_phase_deg.setToolTip(
            "Absolute phase of the α/2 preparation RF pulse. For constant "
            "0° full pulses at the 180° passband center, use +90°."
        )
        self.bssfp_alpha_half_use_ratios = QCheckBox("Use ratios for startup pulse")
        self.bssfp_alpha_half_use_ratios.setChecked(True)
        self.bssfp_alpha_half_tr_ratio = self._parameter_spin(0.0, 2.0, 0.5, " × TR")
        self.bssfp_alpha_half_tr_ratio.setSingleStep(0.1)
        self.bssfp_alpha_half_tr_ratio.setToolTip(
            "Center-to-center spacing from the startup pulse to the first "
            "regular RF pulse, relative to the regular TR."
        )
        self.bssfp_alpha_half_flip_ratio = self._parameter_spin(0.0, 2.0, 0.5, " × FA")
        self.bssfp_alpha_half_flip_ratio.setSingleStep(0.1)
        self.bssfp_alpha_half_flip_ratio.setToolTip(
            "Startup flip angle relative to the regular bSSFP flip angle."
        )
        self.bssfp_alpha_half_center_spacing_ms = self._parameter_spin(
            0.0, 10000.0, 5.0, " ms"
        )
        self.bssfp_alpha_half_center_spacing_ms.setToolTip(
            "Explicit center-to-center spacing from the startup pulse to the "
            "first regular RF pulse."
        )
        self.bssfp_alpha_half_flip_angle_deg = self._parameter_spin(
            0.0, 720.0, 7.5, "°"
        )
        self.bssfp_alpha_half_flip_angle_deg.setToolTip(
            "Explicit flip angle of the startup preparation pulse."
        )

        self.bssfp_alpha_half_ratio_container = QWidget()
        bssfp_alpha_half_ratio_form = _left_aligned_form(
            self.bssfp_alpha_half_ratio_container
        )
        bssfp_alpha_half_ratio_form.setContentsMargins(0, 0, 0, 0)
        bssfp_alpha_half_ratio_form.addRow(
            "First TR ratio", self.bssfp_alpha_half_tr_ratio
        )
        bssfp_alpha_half_ratio_form.addRow(
            "First flip ratio", self.bssfp_alpha_half_flip_ratio
        )

        self.bssfp_alpha_half_absolute_container = QWidget()
        bssfp_alpha_half_absolute_form = _left_aligned_form(
            self.bssfp_alpha_half_absolute_container
        )
        bssfp_alpha_half_absolute_form.setContentsMargins(0, 0, 0, 0)
        bssfp_alpha_half_absolute_form.addRow(
            "First TR", self.bssfp_alpha_half_center_spacing_ms
        )
        bssfp_alpha_half_absolute_form.addRow(
            "First flip angle", self.bssfp_alpha_half_flip_angle_deg
        )
        self.bssfp_dummy_repetitions = QSpinBox()
        self.bssfp_dummy_repetitions.setRange(0, 10000)
        self.bssfp_dummy_repetitions.setValue(1)
        self.bssfp_repetitions = QSpinBox()
        self.bssfp_repetitions.setRange(1, 10000)
        self.bssfp_repetitions.setValue(1)
        self.bssfp_acquisition_interval_ms = self._acquisition_interval_spin()
        self.bssfp_alpha_half = QCheckBox("Enable startup pulse (α/2 default)")
        self.bssfp_alpha_half.setChecked(True)
        self.bssfp_dwell_info = QLabel()
        bssfp_form.addRow(bssfp_hint)
        _add_form_section(bssfp_form, "Spatial encoding")
        bssfp_form.addRow("Read gradient direction", self.bssfp_read_gradient_axis)
        bssfp_form.addRow("Phase gradient direction", self.bssfp_phase_gradient_axis)
        bssfp_form.addRow(
            "Partition gradient direction", self.bssfp_partition_gradient_axis
        )
        bssfp_form.addRow("Read matrix", self.bssfp_read_matrix)
        bssfp_form.addRow("Phase matrix", self.bssfp_phase_matrix)
        bssfp_form.addRow("Partition matrix", self.bssfp_partition_matrix)
        bssfp_form.addRow("Read FOV", self.bssfp_read_fov_mm)
        bssfp_form.addRow("Phase FOV", self.bssfp_phase_fov_mm)
        bssfp_form.addRow("Partition FOV", self.bssfp_partition_fov_mm)
        bssfp_form.addRow("Sampling bandwidth", self.bssfp_bandwidth_khz)
        _add_form_section(bssfp_form, "RF pulse and timing")
        bssfp_form.addRow("Flip angle", self.bssfp_flip_angle_deg)
        self._add_shared_rf_controls(
            bssfp_form,
            "bssfp",
            pulse_type="Block",
            duration_ms=1.0,
            sinc_lobes=3,
        )
        bssfp_form.addRow("Repetition time (TR)", self.bssfp_repetition_time_ms)
        bssfp_form.addRow("RF phase start", self.bssfp_phase_start_deg)
        bssfp_form.addRow("RF phase increment", self.bssfp_phase_increment_deg)
        _add_form_section(bssfp_form, "Preparation and dynamics")
        bssfp_form.addRow("Dummy repetitions", self.bssfp_dummy_repetitions)
        bssfp_form.addRow("Dynamic volumes", self.bssfp_repetitions)
        bssfp_form.addRow(
            "Volume interval (start-to-start)",
            self.bssfp_acquisition_interval_ms,
        )
        bssfp_form.addRow("Preparation", self.bssfp_alpha_half)
        bssfp_form.addRow("Startup value mode", self.bssfp_alpha_half_use_ratios)
        bssfp_form.addRow(self.bssfp_alpha_half_ratio_container)
        bssfp_form.addRow(self.bssfp_alpha_half_absolute_container)
        bssfp_form.addRow("Startup pulse phase", self.bssfp_alpha_half_phase_deg)
        _add_form_section(bssfp_form, "Derived sampling")
        bssfp_form.addRow("ADC dwell", self.bssfp_dwell_info)
        self.bssfp_group.setVisible(False)
        controls_layout.addWidget(self.bssfp_group)

        self.ss_bssfp_group = QGroupBox("Spectrally selective bSSFP (3D)")
        ss_form = _left_aligned_form(self.ss_bssfp_group)
        ss_hint = QLabel(
            "Alternating-frequency Cartesian 3D SS-bSSFP following Skinner "
            "et al. (doi:10.1002/mrm.29676). One target is acquired per volume."
        )
        ss_hint.setWordWrap(True)
        (
            self.ss_bssfp_read_gradient_axis,
            self.ss_bssfp_phase_gradient_axis,
            self.ss_bssfp_partition_gradient_axis,
        ) = self._three_dimensional_orientation_controls(
            "ss_bssfp", default_read_axis="+z"
        )
        self.ss_bssfp_read_matrix = QSpinBox()
        self.ss_bssfp_read_matrix.setRange(2, 256)
        self.ss_bssfp_read_matrix.setValue(32)
        self.ss_bssfp_phase_matrix = QSpinBox()
        self.ss_bssfp_phase_matrix.setRange(1, 256)
        self.ss_bssfp_phase_matrix.setValue(16)
        self.ss_bssfp_partition_matrix = QSpinBox()
        self.ss_bssfp_partition_matrix.setRange(1, 256)
        self.ss_bssfp_partition_matrix.setValue(12)
        self.ss_bssfp_read_fov_mm = self._parameter_spin(0.1, 10000.0, 56.0, " mm")
        self.ss_bssfp_phase_fov_mm = self._parameter_spin(0.1, 10000.0, 28.0, " mm")
        self.ss_bssfp_partition_fov_mm = self._parameter_spin(0.1, 10000.0, 21.0, " mm")
        self.ss_bssfp_target_names = QLineEdit("Lac, Py")
        self.ss_bssfp_target_names.setToolTip(
            "Comma-separated names; one name per RF/receiver frequency pair"
        )
        self.ss_bssfp_target_offsets_hz = QLineEdit("1655, -245")
        self.ss_bssfp_target_offsets_hz.setToolTip(
            "Comma-separated RF carrier offsets in Hz, relative to sequence centre"
        )
        self.ss_bssfp_receiver_offsets_hz = QLineEdit("925.44725, 0")
        self.ss_bssfp_receiver_offsets_hz.setToolTip(
            "Comma-separated ADC demodulation offsets in Hz"
        )
        self.ss_bssfp_flip_angles_deg = QLineEdit("90, 4")
        self.ss_bssfp_flip_angles_deg.setToolTip(
            "Comma-separated nominal flip angles matching the target list"
        )
        self.ss_bssfp_bandwidth_khz = self._sampling_bandwidth_spin(10.0)
        self.ss_bssfp_encoding_duration_ms = self._parameter_spin(
            0.01, 100.0, 0.2, " ms"
        )
        self.ss_bssfp_encoding_duration_ms.setEnabled(False)
        self.ss_bssfp_encoding_duration_ms.setToolTip(
            "Automatically calculated from FOV, matrix, sampling bandwidth, "
            "and the configured scanner gradient limits"
        )
        self.ss_bssfp_repetition_time_ms = self._parameter_spin(
            0.1, 10000.0, 6.29, " ms"
        )
        self.ss_bssfp_phase_start_deg = self._parameter_spin(-360.0, 360.0, 0.0, "°")
        self.ss_bssfp_phase_start_deg.setToolTip(
            "Sets the absolute phase of the first RF pulse. The receiver reference "
            "follows the RF phase, so magnitude-only results can remain unchanged; "
            "inspect the RF phase trace in the Sequence tab."
        )
        self.ss_bssfp_phase_increment_deg = self._parameter_spin(
            -360.0, 360.0, 0.0, "°"
        )
        self.ss_bssfp_dummy_repetitions = QSpinBox()
        self.ss_bssfp_dummy_repetitions.setRange(0, 10000)
        self.ss_bssfp_dummy_repetitions.setValue(0)
        self.ss_bssfp_repetitions = QSpinBox()
        self.ss_bssfp_repetitions.setRange(1, 10000)
        self.ss_bssfp_repetitions.setValue(2)
        self.ss_bssfp_acquisition_interval_ms = self._acquisition_interval_spin()
        self.ss_bssfp_alpha_half = QCheckBox(
            "Enable startup pulse before each target volume"
        )
        self.ss_bssfp_alpha_half.setChecked(True)
        self.ss_bssfp_alpha_half_use_ratios = QCheckBox("Use ratios for startup pulse")
        self.ss_bssfp_alpha_half_use_ratios.setChecked(False)
        self.ss_bssfp_alpha_half_tr_ratio = self._parameter_spin(
            0.0, 2.0, 4.31 / 6.29, " × TR"
        )
        self.ss_bssfp_alpha_half_tr_ratio.setSingleStep(0.1)
        self.ss_bssfp_alpha_half_flip_ratio = self._parameter_spin(
            0.0, 2.0, 0.5, " × FA"
        )
        self.ss_bssfp_alpha_half_flip_ratio.setSingleStep(0.1)
        self.ss_bssfp_alpha_half_spacing_ms = self._parameter_spin(
            0.01, 10000.0, 4.31, " ms"
        )
        self.ss_bssfp_alpha_half_flip_angles_deg = QLineEdit("45, 2")
        self.ss_bssfp_alpha_half_flip_angles_deg.setToolTip(
            "Explicit startup flip angles matching the spectral target list"
        )
        self.ss_bssfp_alpha_half_ratio_container = QWidget()
        ss_alpha_half_ratio_form = _left_aligned_form(
            self.ss_bssfp_alpha_half_ratio_container
        )
        ss_alpha_half_ratio_form.setContentsMargins(0, 0, 0, 0)
        ss_alpha_half_ratio_form.addRow(
            "First TR ratio", self.ss_bssfp_alpha_half_tr_ratio
        )
        ss_alpha_half_ratio_form.addRow(
            "First flip ratio", self.ss_bssfp_alpha_half_flip_ratio
        )
        self.ss_bssfp_alpha_half_absolute_container = QWidget()
        ss_alpha_half_absolute_form = _left_aligned_form(
            self.ss_bssfp_alpha_half_absolute_container
        )
        ss_alpha_half_absolute_form.setContentsMargins(0, 0, 0, 0)
        ss_alpha_half_absolute_form.addRow(
            "First TR", self.ss_bssfp_alpha_half_spacing_ms
        )
        ss_alpha_half_absolute_form.addRow(
            "First flip angles", self.ss_bssfp_alpha_half_flip_angles_deg
        )
        self.ss_bssfp_spoiler_cycles = self._parameter_spin(
            0.0, 1000.0, 0.0, " cycles/FOV"
        )
        self.ss_bssfp_spoiler_cycles.setToolTip(
            "Optional legacy moment spread over the complete imaging FOV"
        )
        self.ss_bssfp_spoiler_cycles_per_voxel = self._parameter_spin(
            0.0, 1000.0, 1.0, " cycles/voxel"
        )
        self.ss_bssfp_spoiler_cycles_per_voxel.setToolTip(
            "Crusher phase across each actual simulation-phantom voxel. "
            "One cycle per voxel fully cancels a uniform voxel in the ideal limit."
        )
        self.ss_bssfp_spoiler_duration_ms = self._parameter_spin(
            0.001, 1000.0, 1.0, " ms"
        )
        self.ss_bssfp_dwell_info = QLabel()
        self.ss_bssfp_spoiler_info = QLabel()
        self.ss_bssfp_spoiler_info.setWordWrap(True)
        ss_form.addRow(ss_hint)
        _add_form_section(ss_form, "Spatial encoding")
        ss_form.addRow("Read gradient direction", self.ss_bssfp_read_gradient_axis)
        ss_form.addRow("Phase gradient direction", self.ss_bssfp_phase_gradient_axis)
        ss_form.addRow(
            "Partition gradient direction", self.ss_bssfp_partition_gradient_axis
        )
        ss_form.addRow("Read matrix", self.ss_bssfp_read_matrix)
        ss_form.addRow("Phase matrix", self.ss_bssfp_phase_matrix)
        ss_form.addRow("Partition matrix", self.ss_bssfp_partition_matrix)
        ss_form.addRow("Read FOV", self.ss_bssfp_read_fov_mm)
        ss_form.addRow("Phase FOV", self.ss_bssfp_phase_fov_mm)
        ss_form.addRow("Partition FOV", self.ss_bssfp_partition_fov_mm)
        _add_form_section(ss_form, "Spectral targets and RF pulse")
        ss_form.addRow("Target names", self.ss_bssfp_target_names)
        ss_form.addRow("RF target offsets", self.ss_bssfp_target_offsets_hz)
        ss_form.addRow("Receiver offsets", self.ss_bssfp_receiver_offsets_hz)
        ss_form.addRow("Target flip angles", self.ss_bssfp_flip_angles_deg)
        self._add_shared_rf_controls(
            ss_form,
            "ss_bssfp",
            pulse_type="Gaussian",
            duration_ms=2.33,
            sinc_lobes=3,
            apodization=0.0,
            label_prefix="Spectral RF",
        )
        _add_form_section(ss_form, "Readout and timing")
        ss_form.addRow("Sampling bandwidth", self.ss_bssfp_bandwidth_khz)
        ss_form.addRow(
            "Encoding lobe duration (auto)",
            self.ss_bssfp_encoding_duration_ms,
        )
        ss_form.addRow("Repetition time (TR)", self.ss_bssfp_repetition_time_ms)
        ss_form.addRow("RF phase start", self.ss_bssfp_phase_start_deg)
        ss_form.addRow("RF phase increment", self.ss_bssfp_phase_increment_deg)
        _add_form_section(ss_form, "Preparation, dynamics, and spoiling")
        ss_form.addRow("Dummy repetitions", self.ss_bssfp_dummy_repetitions)
        ss_form.addRow("Dynamic volumes", self.ss_bssfp_repetitions)
        ss_form.addRow(
            "Volume interval (start-to-start)",
            self.ss_bssfp_acquisition_interval_ms,
        )
        ss_form.addRow("Preparation", self.ss_bssfp_alpha_half)
        ss_form.addRow("Startup value mode", self.ss_bssfp_alpha_half_use_ratios)
        ss_form.addRow(self.ss_bssfp_alpha_half_ratio_container)
        ss_form.addRow(self.ss_bssfp_alpha_half_absolute_container)
        ss_form.addRow(
            "Voxel-referenced spoiler", self.ss_bssfp_spoiler_cycles_per_voxel
        )
        ss_form.addRow("Additional FOV spoiler", self.ss_bssfp_spoiler_cycles)
        ss_form.addRow("Spoiler duration", self.ss_bssfp_spoiler_duration_ms)
        ss_form.addRow("Spoiler check", self.ss_bssfp_spoiler_info)
        _add_form_section(ss_form, "Derived sampling")
        ss_form.addRow("ADC dwell", self.ss_bssfp_dwell_info)
        self.ss_bssfp_group.setVisible(False)
        controls_layout.addWidget(self.ss_bssfp_group)

        self.radial_me_bssfp_group = QGroupBox("Radial multi-echo bSSFP (3D)")
        radial_form = _left_aligned_form(self.radial_me_bssfp_group)
        radial_hint = QLabel(
            "Center-through 3D radial ME-bSSFP with spiral phyllotaxis and "
            "monopolar echoes following Wang et al. (doi:10.1002/mrm.30614). "
            "The trajectory axes orient the complete phyllotaxis coordinate "
            "system; each individual spoke has its own readout direction."
        )
        radial_hint.setWordWrap(True)
        (
            self.radial_me_read_gradient_axis,
            self.radial_me_phase_gradient_axis,
            self.radial_me_partition_gradient_axis,
        ) = self._three_dimensional_orientation_controls("radial_me")
        self.radial_me_fov_mm = self._parameter_spin(0.1, 10000.0, 356.0, " mm")
        self.radial_me_base_resolution = QSpinBox()
        self.radial_me_base_resolution.setRange(2, 256)
        self.radial_me_base_resolution.setValue(8)
        self.radial_me_readout_oversampling = QSpinBox()
        self.radial_me_readout_oversampling.setRange(1, 16)
        self.radial_me_readout_oversampling.setValue(2)
        self.radial_me_spokes = QSpinBox()
        self.radial_me_spokes.setRange(1, 100000)
        self.radial_me_spokes.setValue(16)
        self.radial_me_measurements = QSpinBox()
        self.radial_me_measurements.setRange(1, 10000)
        self.radial_me_measurements.setValue(1)
        self.radial_me_acquisition_interval_ms = self._acquisition_interval_spin()
        self.radial_me_echoes = QSpinBox()
        self.radial_me_echoes.setRange(1, 31)
        self.radial_me_echoes.setSingleStep(2)
        self.radial_me_echoes.setValue(5)
        self.radial_me_echo_spacing_ms = self._parameter_spin(0.01, 1000.0, 2.0, " ms")
        self.radial_me_pixel_bandwidth_hz = self._parameter_spin(
            0.1, 1_000_000.0, 1000.0, " Hz/px"
        )
        self.radial_me_flip_angle_deg = self._parameter_spin(0.1, 360.0, 10.0, "°")
        self.radial_me_repetition_time_ms = self._parameter_spin(
            0.1, 10000.0, 16.0, " ms"
        )
        self.radial_me_phase_start_deg = self._parameter_spin(-360.0, 360.0, 0.0, "°")
        self.radial_me_phase_increment_deg = self._parameter_spin(
            -360.0, 360.0, 180.0, "°"
        )
        self.radial_me_alpha_half = QCheckBox("Enable startup pulse (α/2 default)")
        self.radial_me_alpha_half.setChecked(True)
        self.radial_me_alpha_half_use_ratios = QCheckBox("Use ratios for startup pulse")
        self.radial_me_alpha_half_use_ratios.setChecked(True)
        self.radial_me_alpha_half_tr_ratio = self._parameter_spin(
            0.0, 2.0, 0.5, " × TR"
        )
        self.radial_me_alpha_half_tr_ratio.setSingleStep(0.1)
        self.radial_me_alpha_half_flip_ratio = self._parameter_spin(
            0.0, 2.0, 0.5, " × FA"
        )
        self.radial_me_alpha_half_flip_ratio.setSingleStep(0.1)
        self.radial_me_alpha_half_center_spacing_ms = self._parameter_spin(
            0.0, 10000.0, 8.0, " ms"
        )
        self.radial_me_alpha_half_flip_angle_deg = self._parameter_spin(
            0.0, 720.0, 5.0, "°"
        )
        self.radial_me_alpha_half_ratio_container = QWidget()
        radial_alpha_half_ratio_form = _left_aligned_form(
            self.radial_me_alpha_half_ratio_container
        )
        radial_alpha_half_ratio_form.setContentsMargins(0, 0, 0, 0)
        radial_alpha_half_ratio_form.addRow(
            "First TR ratio", self.radial_me_alpha_half_tr_ratio
        )
        radial_alpha_half_ratio_form.addRow(
            "First flip ratio", self.radial_me_alpha_half_flip_ratio
        )
        self.radial_me_alpha_half_absolute_container = QWidget()
        radial_alpha_half_absolute_form = _left_aligned_form(
            self.radial_me_alpha_half_absolute_container
        )
        radial_alpha_half_absolute_form.setContentsMargins(0, 0, 0, 0)
        radial_alpha_half_absolute_form.addRow(
            "First TR", self.radial_me_alpha_half_center_spacing_ms
        )
        radial_alpha_half_absolute_form.addRow(
            "First flip angle", self.radial_me_alpha_half_flip_angle_deg
        )
        self.radial_me_tip_back = QCheckBox("Enable −α/2 tip-back")
        self.radial_me_tip_back.setChecked(True)
        self.radial_me_prephaser_duration_ms = self._parameter_spin(
            0.01, 100.0, 0.5, " ms"
        )
        self.radial_me_rotation_deg = self._parameter_spin(
            -360.0, 360.0, 137.507764, "°"
        )
        self.radial_me_sampling_info = QLabel()
        radial_form.addRow(radial_hint)
        _add_form_section(radial_form, "Spatial and radial encoding")
        radial_form.addRow("Trajectory read axis", self.radial_me_read_gradient_axis)
        radial_form.addRow("Trajectory phase axis", self.radial_me_phase_gradient_axis)
        radial_form.addRow(
            "Trajectory partition axis", self.radial_me_partition_gradient_axis
        )
        radial_form.addRow("Isotropic FOV", self.radial_me_fov_mm)
        radial_form.addRow("Base resolution", self.radial_me_base_resolution)
        radial_form.addRow("Readout oversampling", self.radial_me_readout_oversampling)
        radial_form.addRow("Spokes / measurement", self.radial_me_spokes)
        radial_form.addRow("Dynamic measurements", self.radial_me_measurements)
        radial_form.addRow("Echoes", self.radial_me_echoes)
        radial_form.addRow("Echo spacing", self.radial_me_echo_spacing_ms)
        radial_form.addRow("Pixel bandwidth", self.radial_me_pixel_bandwidth_hz)
        _add_form_section(radial_form, "RF pulse and timing")
        radial_form.addRow("Flip angle", self.radial_me_flip_angle_deg)
        self._add_shared_rf_controls(
            radial_form,
            "radial_me",
            pulse_type="Block",
            duration_ms=0.5,
            sinc_lobes=3,
        )
        radial_form.addRow("Repetition time (TR)", self.radial_me_repetition_time_ms)
        radial_form.addRow(
            "Measurement interval (start-to-start)",
            self.radial_me_acquisition_interval_ms,
        )
        radial_form.addRow("RF phase start", self.radial_me_phase_start_deg)
        radial_form.addRow("RF phase increment", self.radial_me_phase_increment_deg)
        _add_form_section(radial_form, "Preparation and trajectory")
        radial_form.addRow("Preparation", self.radial_me_alpha_half)
        radial_form.addRow("Startup value mode", self.radial_me_alpha_half_use_ratios)
        radial_form.addRow(self.radial_me_alpha_half_ratio_container)
        radial_form.addRow(self.radial_me_alpha_half_absolute_container)
        radial_form.addRow("Tip-back", self.radial_me_tip_back)
        radial_form.addRow(
            "Pre-/postphaser duration", self.radial_me_prephaser_duration_ms
        )
        radial_form.addRow("Rotation between measurements", self.radial_me_rotation_deg)
        _add_form_section(radial_form, "Derived sampling")
        radial_form.addRow("Readout", self.radial_me_sampling_info)
        self.radial_me_bssfp_group.setVisible(False)
        controls_layout.addWidget(self.radial_me_bssfp_group)

        self.me_bssfp_group = QGroupBox("Cartesian multi-echo bSSFP (3D)")
        me_form = _left_aligned_form(self.me_bssfp_group)
        me_hint = QLabel(
            "Balanced Cartesian 3D ME-bSSFP following Gaubatz (2023), with "
            "selectable monopolar flyback or symmetric bipolar readout."
        )
        me_hint.setWordWrap(True)
        (
            self.me_bssfp_read_gradient_axis,
            self.me_bssfp_phase_gradient_axis,
            self.me_bssfp_partition_gradient_axis,
        ) = self._three_dimensional_orientation_controls(
            "me_bssfp", default_read_axis="+z"
        )
        self.me_bssfp_read_matrix = QSpinBox()
        self.me_bssfp_read_matrix.setRange(2, 256)
        self.me_bssfp_read_matrix.setValue(8)
        self.me_bssfp_phase_matrix = QSpinBox()
        self.me_bssfp_phase_matrix.setRange(1, 256)
        self.me_bssfp_phase_matrix.setValue(8)
        self.me_bssfp_partition_matrix = QSpinBox()
        self.me_bssfp_partition_matrix.setRange(1, 256)
        self.me_bssfp_partition_matrix.setValue(4)
        self.me_bssfp_read_fov_mm = self._parameter_spin(0.1, 10000.0, 56.0, " mm")
        self.me_bssfp_phase_fov_mm = self._parameter_spin(0.1, 10000.0, 28.0, " mm")
        self.me_bssfp_partition_fov_mm = self._parameter_spin(0.1, 10000.0, 24.5, " mm")
        self.me_bssfp_echoes = QSpinBox()
        self.me_bssfp_echoes.setRange(1, 31)
        self.me_bssfp_echoes.setSingleStep(2)
        self.me_bssfp_echoes.setValue(5)
        self.me_bssfp_echo_spacing_ms = self._parameter_spin(0.01, 1000.0, 1.32, " ms")
        self.me_bssfp_readout_strategy = QComboBox()
        self.me_bssfp_readout_strategy.addItems(["Flyback", "Symmetric bipolar"])
        self.me_bssfp_bandwidth_khz = self._sampling_bandwidth_spin(39.6825)
        self.me_bssfp_flip_angle_deg = self._parameter_spin(0.1, 360.0, 3.5, "°")
        self.me_bssfp_receiver_offset_hz = self._parameter_spin(
            -1_000_000.0, 1_000_000.0, -460.0, " Hz"
        )
        self.me_bssfp_encoding_duration_ms = self._parameter_spin(
            0.01, 100.0, 0.5, " ms"
        )
        self.me_bssfp_repetition_time_ms = self._parameter_spin(
            0.1, 10000.0, 8.696, " ms"
        )
        self.me_bssfp_phase_start_deg = self._parameter_spin(-360.0, 360.0, 0.0, "°")
        self.me_bssfp_phase_increment_deg = self._parameter_spin(
            -360.0, 360.0, 180.0, "°"
        )
        self.me_bssfp_dummy_repetitions = QSpinBox()
        self.me_bssfp_dummy_repetitions.setRange(0, 10000)
        self.me_bssfp_dummy_repetitions.setValue(0)
        self.me_bssfp_repetitions = QSpinBox()
        self.me_bssfp_repetitions.setRange(1, 10000)
        self.me_bssfp_repetitions.setValue(1)
        self.me_bssfp_acquisition_interval_ms = self._acquisition_interval_spin()
        self.me_bssfp_alpha_half = QCheckBox("Enable startup pulse (α/2 default)")
        self.me_bssfp_alpha_half.setChecked(True)
        self.me_bssfp_alpha_half_use_ratios = QCheckBox("Use ratios for startup pulse")
        self.me_bssfp_alpha_half_use_ratios.setChecked(True)
        self.me_bssfp_alpha_half_tr_ratio = self._parameter_spin(0.0, 2.0, 0.5, " × TR")
        self.me_bssfp_alpha_half_tr_ratio.setSingleStep(0.1)
        self.me_bssfp_alpha_half_flip_ratio = self._parameter_spin(
            0.0, 2.0, 0.5, " × FA"
        )
        self.me_bssfp_alpha_half_flip_ratio.setSingleStep(0.1)
        self.me_bssfp_alpha_half_center_spacing_ms = self._parameter_spin(
            0.0, 10000.0, 4.348, " ms"
        )
        self.me_bssfp_alpha_half_flip_angle_deg = self._parameter_spin(
            0.0, 720.0, 1.75, "°"
        )
        self.me_bssfp_alpha_half_ratio_container = QWidget()
        me_alpha_half_ratio_form = _left_aligned_form(
            self.me_bssfp_alpha_half_ratio_container
        )
        me_alpha_half_ratio_form.setContentsMargins(0, 0, 0, 0)
        me_alpha_half_ratio_form.addRow(
            "First TR ratio", self.me_bssfp_alpha_half_tr_ratio
        )
        me_alpha_half_ratio_form.addRow(
            "First flip ratio", self.me_bssfp_alpha_half_flip_ratio
        )
        self.me_bssfp_alpha_half_absolute_container = QWidget()
        me_alpha_half_absolute_form = _left_aligned_form(
            self.me_bssfp_alpha_half_absolute_container
        )
        me_alpha_half_absolute_form.setContentsMargins(0, 0, 0, 0)
        me_alpha_half_absolute_form.addRow(
            "First TR", self.me_bssfp_alpha_half_center_spacing_ms
        )
        me_alpha_half_absolute_form.addRow(
            "First flip angle", self.me_bssfp_alpha_half_flip_angle_deg
        )
        self.me_bssfp_sampling_info = QLabel()
        me_form.addRow(me_hint)
        _add_form_section(me_form, "Spatial encoding")
        me_form.addRow("Read gradient direction", self.me_bssfp_read_gradient_axis)
        me_form.addRow("Phase gradient direction", self.me_bssfp_phase_gradient_axis)
        me_form.addRow(
            "Partition gradient direction", self.me_bssfp_partition_gradient_axis
        )
        me_form.addRow("Read matrix", self.me_bssfp_read_matrix)
        me_form.addRow("Phase matrix", self.me_bssfp_phase_matrix)
        me_form.addRow("Partition matrix", self.me_bssfp_partition_matrix)
        me_form.addRow("Read FOV", self.me_bssfp_read_fov_mm)
        me_form.addRow("Phase FOV", self.me_bssfp_phase_fov_mm)
        me_form.addRow("Partition FOV", self.me_bssfp_partition_fov_mm)
        me_form.addRow("Echoes", self.me_bssfp_echoes)
        me_form.addRow("Echo spacing", self.me_bssfp_echo_spacing_ms)
        me_form.addRow("Readout strategy", self.me_bssfp_readout_strategy)
        me_form.addRow("Sampling bandwidth", self.me_bssfp_bandwidth_khz)
        _add_form_section(me_form, "RF pulse and frequency")
        me_form.addRow("Flip angle", self.me_bssfp_flip_angle_deg)
        self._add_shared_rf_controls(
            me_form,
            "me_bssfp",
            pulse_type="Gaussian",
            duration_ms=0.5,
            sinc_lobes=3,
        )
        me_form.addRow("Receiver offset", self.me_bssfp_receiver_offset_hz)
        _add_form_section(me_form, "Timing")
        me_form.addRow("Encoding lobe duration", self.me_bssfp_encoding_duration_ms)
        me_form.addRow("Repetition time (TR)", self.me_bssfp_repetition_time_ms)
        me_form.addRow("RF phase start", self.me_bssfp_phase_start_deg)
        me_form.addRow("RF phase increment", self.me_bssfp_phase_increment_deg)
        _add_form_section(me_form, "Preparation and dynamics")
        me_form.addRow("Dummy repetitions", self.me_bssfp_dummy_repetitions)
        me_form.addRow("Dynamic volumes", self.me_bssfp_repetitions)
        me_form.addRow(
            "Volume interval (start-to-start)",
            self.me_bssfp_acquisition_interval_ms,
        )
        me_form.addRow("Preparation", self.me_bssfp_alpha_half)
        me_form.addRow("Startup value mode", self.me_bssfp_alpha_half_use_ratios)
        me_form.addRow(self.me_bssfp_alpha_half_ratio_container)
        me_form.addRow(self.me_bssfp_alpha_half_absolute_container)
        _add_form_section(me_form, "Derived sampling")
        me_form.addRow("ADC dwell", self.me_bssfp_sampling_info)
        self.me_bssfp_group.setVisible(False)
        controls_layout.addWidget(self.me_bssfp_group)

        self.spoiling_quality_group = QGroupBox("Spoiling")
        spoiling_quality_layout = QHBoxLayout(self.spoiling_quality_group)
        spoiling_quality_layout.addWidget(QLabel("Quality"))
        self.spoiling_quality_button = QToolButton()
        self.spoiling_quality_button.setObjectName("spoiling_quality_info_button")
        self.spoiling_quality_button.setText("ⓘ")
        self.spoiling_quality_button.setAutoRaise(True)
        self.spoiling_quality_button.setCursor(Qt.WhatsThisCursor)
        self.spoiling_quality_button.setToolTipDuration(30000)
        self.spoiling_quality_button.setAccessibleName("Spoiling quality details")
        spoiling_quality_layout.addWidget(self.spoiling_quality_button)
        self.spoiling_quality_status = QLabel("Details on hover")
        self.spoiling_quality_status.setStyleSheet("color: #475569;")
        spoiling_quality_layout.addWidget(self.spoiling_quality_status)
        spoiling_quality_layout.addStretch()
        # Retain the plain-text value for programmatic access and backwards
        # compatibility; the user-facing presentation is the hover table.
        self.spoiling_quality_info = QLabel()
        self.spoiling_quality_info.setWordWrap(True)
        self.spoiling_quality_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.spoiling_quality_info.hide()
        self.spoiling_apply_recommended_grid = QPushButton(
            "Apply recommended subvoxel grid"
        )
        self.spoiling_apply_recommended_grid.setToolTip(
            "Apply the smallest tested regular midpoint grid whose retained "
            "coherence follows the continuous voxel throughout this spoiler train."
        )
        spoiling_quality_layout.addWidget(self.spoiling_apply_recommended_grid)
        controls_layout.addWidget(self.spoiling_quality_group)

        object_group = QGroupBox("Simulation object")
        object_form = _left_aligned_form(object_group)
        self.object_form = object_form
        self.object_source = QComboBox()
        self.object_source.addItems(
            ["Phantom tab / designer", "Built-in quick object", "Spin probe"]
        )
        object_form.addRow("Source", self.object_source)
        self.field_strength_t = QDoubleSpinBox()
        self.field_strength_t.setRange(0.01, 30.0)
        self.field_strength_t.setDecimals(4)
        self.field_strength_t.setValue(self.workspace_defaults.field_strength_t)
        self.field_strength_t.setSuffix(" T")
        self.field_strength_t.setToolTip(
            "Converts field-independent phantom frequency offsets from ppm to Hz"
        )
        object_form.addRow("Field strength B0", self.field_strength_t)
        self.field_strength_label = object_form.labelForField(self.field_strength_t)
        self.nucleus = QComboBox()
        self.nucleus.addItems(list(NUCLEUS_GAMMA_HZ_PER_T))
        self.nucleus.setCurrentText(self.workspace_defaults.phantom_nucleus)
        self.nucleus.setToolTip("Reference nucleus used for ppm-to-Hz conversion")
        object_form.addRow("Nucleus", self.nucleus)
        self.nucleus_label = object_form.labelForField(self.nucleus)
        self.frequency_reference_info = QLabel()
        self.frequency_reference_info.setWordWrap(True)
        self.frequency_reference_info.setMinimumWidth(0)
        self.frequency_reference_info.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )
        self.frequency_reference_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.frequency_reference_info.setMinimumHeight(
            3 * self.frequency_reference_info.fontMetrics().lineSpacing()
        )
        object_form.addRow("Frequency model", self.frequency_reference_info)
        self.frequency_reference_label = object_form.labelForField(
            self.frequency_reference_info
        )

        self.built_in_properties_group = QGroupBox("Built-in phantom properties")
        built_in_form = _left_aligned_form(self.built_in_properties_group)
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
        self.fov_mm = QDoubleSpinBox()
        self.fov_mm.setRange(1.0, 1000.0)
        self.fov_mm.setDecimals(3)
        self.fov_mm.setValue(default_fov_x)
        self.fov_mm.setSuffix(" mm")
        built_in_form.addRow("In-plane FOV", self.fov_mm)
        self.fov_z_mm = QDoubleSpinBox()
        self.fov_z_mm.setRange(0.01, 1000.0)
        self.fov_z_mm.setDecimals(3)
        self.fov_z_mm.setValue(default_fov_z)
        self.fov_z_mm.setSuffix(" mm")
        built_in_form.addRow("Through-plane FOV", self.fov_z_mm)
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
        self.phantom_summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.phantom_summary.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.phantom_summary.setMinimumHeight(
            3 * self.phantom_summary.fontMetrics().lineSpacing()
        )
        object_form.addRow("Selected phantom", self.phantom_summary)
        self.phantom_summary_label = object_form.labelForField(self.phantom_summary)
        self.simulation_object_table = QTableWidget(0, 2)
        self.simulation_object_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.simulation_object_table.verticalHeader().setVisible(False)
        self.simulation_object_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.simulation_object_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.simulation_object_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.simulation_object_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.simulation_object_table.setAlternatingRowColors(True)
        self.simulation_object_table.setWordWrap(True)
        self.simulation_object_table.setShowGrid(False)
        self.simulation_object_table.horizontalHeader().sectionResized.connect(
            self._schedule_simulation_object_table_fit
        )
        object_form.addRow(self.simulation_object_table)
        # Keep the old labels populated for API/test compatibility, but replace
        # their multiline presentation with the structured table above.
        self.frequency_reference_info.hide()
        self.frequency_reference_label.hide()
        self.phantom_summary.hide()
        self.phantom_summary_label.hide()
        self.open_phantom_button = QPushButton("Open Phantom tab…")
        self.open_phantom_button.clicked.connect(self._open_phantom_tab)
        object_form.addRow(self.open_phantom_button)
        controls_layout.addWidget(object_group)

        self._built_in_object_widgets = (
            self.object_type,
            self.matrix_size,
            self.z_matrix_size,
            self.fov_mm,
            self.fov_z_mm,
            self.t1_ms,
            self.t2_ms,
            self.pd,
            self.b0_ppm,
            self.chemical_ppm,
        )
        self.object_source.currentIndexChanged.connect(self._object_source_changed)

        self.output_group = QGroupBox("Sparse output")
        output_form = _left_aligned_form(self.output_group)
        self.checkpoints = QLineEdit()
        self.checkpoints.setPlaceholderText("e.g. 1.0, 5.0 (ms)")
        output_form.addRow("Checkpoints", self.checkpoints)
        self.signal_weighting = QComboBox()
        self.signal_weighting.addItems(
            ["Relative voxel sum", "Physical voxel volume (3D)"]
        )
        output_form.addRow("Signal weighting", self.signal_weighting)
        # Keep the combo as the internal frame-to-label mapping used by the
        # existing reconstruction paths.  The visible control is the slider
        # below, so multi-slice 2D data can be browsed continuously.
        self.frame_selector = QComboBox(self.output_group)
        self.frame_selector.setEnabled(False)
        self.frame_selector.hide()
        self.frame_selector.currentIndexChanged.connect(self._frame_changed)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setTracking(True)
        self.frame_slider.setToolTip(
            "Browse inferred slice, repetition, echo, segment, or partition frames"
        )
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)
        self.frame_value_label = QLabel("Single frame")
        self.frame_value_label.setWordWrap(True)
        self.frame_value_label.setMinimumWidth(150)
        frame_control = QWidget()
        frame_control_layout = QHBoxLayout(frame_control)
        frame_control_layout.setContentsMargins(0, 0, 0, 0)
        frame_control_layout.addWidget(self.frame_slider, 1)
        frame_control_layout.addWidget(self.frame_value_label)
        output_form.addRow("2D slice / frame", frame_control)
        self.csi_repetition_selector = QSpinBox()
        self.csi_repetition_selector.setRange(0, 0)
        self.csi_repetition_selector.setEnabled(False)
        self.csi_repetition_selector.valueChanged.connect(self._spectral_view_changed)
        self.csi_repetition_selector.setToolTip(
            "CSI repetition shown in k-space, image, FID, and spectrum views"
        )
        output_form.addRow("CSI repetition", self.csi_repetition_selector)
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

        self.probe_group = QGroupBox("Spin probe configuration")
        probe_group_layout = QVBoxLayout(self.probe_group)
        self.probe_controls = QWidget()
        probe_form = _left_aligned_form(self.probe_controls)
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
        self.probe_ppm_min.setValue(-2500.0)
        self.probe_ppm_min.setSuffix(" Hz")
        probe_form.addRow("Frequency min", self.probe_ppm_min)
        self.probe_ppm_max = QDoubleSpinBox()
        self.probe_ppm_max.setRange(-1e7, 1e7)
        self.probe_ppm_max.setDecimals(4)
        self.probe_ppm_max.setValue(2500.0)
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
            "uniform sampling is intended for continuous playback. Every "
            "simulation step is stored independently and remains available in "
            "the playback mode selector."
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

        # Probe buttons
        probe_buttons = QWidget()
        probe_button_layout = QGridLayout(probe_buttons)
        probe_button_layout.setContentsMargins(0, 0, 0, 0)
        for column in range(3):
            probe_button_layout.setColumnStretch(column, 1)
        self.probe_button_layout = probe_button_layout
        self.run_probe_button = QPushButton("Spectral Probe")
        self.run_probe_button.clicked.connect(self._run_spectral_probe)
        self.run_geometry_probe_button = QPushButton("Geometry Probe")
        self.run_geometry_probe_button.clicked.connect(self._run_geometry_probe)
        probe_button_layout.addWidget(self.run_probe_button, 0, 0)
        probe_button_layout.addWidget(self.run_geometry_probe_button, 0, 1)
        # probe_form.addRow(probe_buttons)
        self.probe_status = QLabel("No spin probe result")
        self.probe_status.setWordWrap(True)
        probe_form.addRow(self.probe_status)
        probe_group_layout.addWidget(self.probe_controls)
        self.probe_controls.setVisible(True)
        self.probe_group.setVisible(False)
        controls_layout.addWidget(self.probe_group)
        controls_layout.addStretch()

        run_panel = QGroupBox("Run")
        run_panel_layout = QVBoxLayout(run_panel)
        self.run_button = QPushButton("Seq. Simulation")
        self.run_button.clicked.connect(self._run)
        probe_button_layout.addWidget(self.run_button, 0, 2)
        run_panel_layout.addWidget(probe_buttons)
        inactive_button_style = (
            "QPushButton:disabled { color: rgba(70, 70, 70, 145); "
            "background-color: rgba(210, 210, 210, 150); }"
        )
        for button in (
            self.run_probe_button,
            self.run_geometry_probe_button,
            self.run_button,
        ):
            button.setStyleSheet(inactive_button_style)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_active_run)
        self.cancel_probe_button = self.cancel_button
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Not started")
        run_panel_layout.addWidget(self.progress)
        self.simulation_time_label = QLabel("Elapsed: — · Remaining: —")
        self.simulation_time_label.setWordWrap(True)
        run_panel_layout.addWidget(self.simulation_time_label)
        self.export_button = QPushButton("Export results…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_results)
        self.open_result_button = QPushButton("Open result…")
        self.open_result_button.setToolTip(
            "Open an exported sequence-result NetCDF file in the reconstruction explorer"
        )
        self.open_result_button.clicked.connect(self._open_reconstruction_result)
        export_cancel_row = QHBoxLayout()
        self.export_cancel_layout = export_cancel_row
        export_cancel_row.addWidget(self.export_button, 1)
        export_cancel_row.addWidget(self.open_result_button, 1)
        export_cancel_row.addWidget(self.cancel_button, 1)
        run_panel_layout.addLayout(export_cancel_row)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        run_panel_layout.addWidget(self.status)
        self._sequence_run_widgets = (
            self.run_button,
            self.progress,
            self.simulation_time_label,
            self.export_button,
            self.status,
        )
        control_column_layout.addWidget(run_panel)

        self.read_matrix.valueChanged.connect(self._acquisition_changed)
        self.phase_matrix.valueChanged.connect(self._acquisition_changed)
        self.epi_readout_trajectory.currentIndexChanged.connect(
            self._readout_trajectory_changed
        )
        self.epi_read_fov_mm.valueChanged.connect(self._acquisition_changed)
        self.epi_phase_fov_mm.valueChanged.connect(self._acquisition_changed)
        self.sampling_bandwidth_khz.valueChanged.connect(self._acquisition_changed)
        self.epi_flip_angle_deg.valueChanged.connect(self._acquisition_changed)
        self.epi_vfa_final_flip_angle_deg.valueChanged.connect(
            self._acquisition_changed
        )
        self.epi_variable_flip_angle.toggled.connect(self._acquisition_changed)
        self.epi_use_ernst_angle.toggled.connect(self._acquisition_changed)
        self.epi_rf_spoiling.toggled.connect(self._acquisition_changed)
        self.epi_rf_spoiling_increment_deg.valueChanged.connect(
            self._acquisition_changed
        )
        self.epi_slice_count.valueChanged.connect(self._acquisition_changed)
        self.epi_repetitions.valueChanged.connect(self._acquisition_changed)
        self.epi_repetition_time_ms.valueChanged.connect(self._acquisition_changed)
        self.epi_echo_time_ms.valueChanged.connect(self._acquisition_changed)
        self.epi_slice_thickness_mm.valueChanged.connect(self._acquisition_changed)
        self.epi_slice_gap_mm.valueChanged.connect(self._acquisition_changed)
        self.epi_slice_offset_mm.valueChanged.connect(self._acquisition_changed)
        self._connect_two_dimensional_orientation_controls(
            self.epi_slice_orientation,
            self.epi_read_gradient_axis,
            self.epi_phase_gradient_axis,
            self.epi_slice_gradient_axis,
            self._acquisition_changed,
        )
        self.epi_spiral_turns.valueChanged.connect(self._acquisition_changed)
        self.epi_spoil_after_slice.toggled.connect(self._acquisition_changed)
        self.epi_spoiler_cycles_per_slice.valueChanged.connect(
            self._acquisition_changed
        )
        self.epi_spoiler_cycles_per_voxel.valueChanged.connect(
            self._acquisition_changed
        )
        self.epi_spoiler_duration_ms.valueChanged.connect(self._acquisition_changed)
        self.epi_spoil_after_slice.toggled.connect(
            self.epi_spoiler_cycles_per_slice.setEnabled
        )
        self.epi_spoil_after_slice.toggled.connect(
            self.epi_spoiler_cycles_per_voxel.setEnabled
        )
        self.epi_spoil_after_slice.toggled.connect(
            self.epi_spoiler_duration_ms.setEnabled
        )
        for widget in (
            self.csi_read_matrix,
            self.csi_phase_matrix,
            self.csi_read_fov_mm,
            self.csi_phase_fov_mm,
            self.csi_spectral_points,
            self.csi_bandwidth_hz,
            self.csi_flip_angle_deg,
            self.csi_vfa_final_flip_angle_deg,
            self.csi_rf_duration_ms,
            self.csi_rf_apodization,
            self.csi_rf_slr_sharpness,
            self.csi_slice_thickness_mm,
            self.csi_slice_offset_mm,
            self.csi_echo_time_ms,
            self.csi_repetition_time_ms,
            self.csi_repetitions,
            self.csi_acquisition_interval_ms,
            self.csi_rf_spoiling_increment_deg,
            self.csi_spoiler_cycles_per_slice,
            self.csi_spoiler_cycles_per_voxel,
            self.csi_spoiler_duration_ms,
        ):
            widget.valueChanged.connect(self._csi_changed)
        self.csi_encoding_order.currentIndexChanged.connect(self._csi_changed)
        self.csi_rf_pulse_type.currentIndexChanged.connect(self._csi_changed)
        self._connect_two_dimensional_orientation_controls(
            self.csi_slice_orientation,
            self.csi_read_gradient_axis,
            self.csi_phase_gradient_axis,
            self.csi_slice_gradient_axis,
            self._csi_changed,
        )
        self.csi_variable_flip_angle.toggled.connect(self._csi_changed)
        self.csi_use_ernst_angle.toggled.connect(self._csi_changed)
        self.csi_rf_spoiling.toggled.connect(self._csi_changed)
        self.csi_spoil_after_readout.toggled.connect(self._csi_changed)
        self.csi_spoil_after_readout.toggled.connect(
            self.csi_spoiler_cycles_per_slice.setEnabled
        )
        self.csi_spoil_after_readout.toggled.connect(
            self.csi_spoiler_cycles_per_voxel.setEnabled
        )
        self.csi_spoil_after_readout.toggled.connect(
            self.csi_spoiler_duration_ms.setEnabled
        )
        for widget in (
            self.flash_read_matrix,
            self.flash_phase_matrix,
            self.flash_read_fov_mm,
            self.flash_phase_fov_mm,
            self.flash_sampling_bandwidth_khz,
            self.flash_flip_angle_deg,
            self.flash_rf_duration_ms,
            self.flash_rf_apodization,
            self.flash_rf_slr_sharpness,
            self.flash_slice_count,
            self.flash_slice_thickness_mm,
            self.flash_slice_gap_mm,
            self.flash_slice_offset_mm,
            self.flash_echo_time_ms,
            self.flash_repetition_time_ms,
            self.flash_repetitions,
            self.flash_acquisition_interval_ms,
            self.flash_rf_spoiling_increment_deg,
            self.flash_spoiler_cycles_per_slice,
            self.flash_spoiler_cycles_per_voxel,
            self.flash_spoiler_duration_ms,
        ):
            widget.valueChanged.connect(self._flash_changed)
        self.flash_use_ernst_angle.toggled.connect(self._flash_changed)
        self.flash_rf_spoiling.toggled.connect(self._flash_changed)
        self.flash_auto_spoiler.toggled.connect(self._flash_auto_spoiler_toggled)
        self.flash_apply_recommended_grid.clicked.connect(
            self._apply_flash_recommended_spin_grid
        )
        self.spoiling_apply_recommended_grid.clicked.connect(
            self._apply_current_recommended_spin_grid
        )
        self.flash_rf_pulse_type.currentIndexChanged.connect(self._flash_changed)
        self._connect_two_dimensional_orientation_controls(
            self.flash_slice_orientation,
            self.flash_read_gradient_axis,
            self.flash_phase_gradient_axis,
            self.flash_slice_gradient_axis,
            self._flash_changed,
        )
        for widget in (
            self.bssfp_read_matrix,
            self.bssfp_phase_matrix,
            self.bssfp_partition_matrix,
            self.bssfp_read_fov_mm,
            self.bssfp_phase_fov_mm,
            self.bssfp_partition_fov_mm,
            self.bssfp_bandwidth_khz,
            self.bssfp_flip_angle_deg,
            self.bssfp_rf_duration_ms,
            self.bssfp_repetition_time_ms,
            self.bssfp_phase_start_deg,
            self.bssfp_phase_increment_deg,
            self.bssfp_alpha_half_phase_deg,
            self.bssfp_alpha_half_tr_ratio,
            self.bssfp_alpha_half_flip_ratio,
            self.bssfp_alpha_half_center_spacing_ms,
            self.bssfp_alpha_half_flip_angle_deg,
            self.bssfp_dummy_repetitions,
            self.bssfp_repetitions,
            self.bssfp_acquisition_interval_ms,
        ):
            widget.valueChanged.connect(self._bssfp_changed)
        self.bssfp_alpha_half.toggled.connect(self._bssfp_changed)
        self.bssfp_alpha_half_use_ratios.toggled.connect(self._bssfp_changed)
        for widget in (
            self.ss_bssfp_read_matrix,
            self.ss_bssfp_phase_matrix,
            self.ss_bssfp_partition_matrix,
            self.ss_bssfp_read_fov_mm,
            self.ss_bssfp_phase_fov_mm,
            self.ss_bssfp_partition_fov_mm,
            self.ss_bssfp_rf_duration_ms,
            self.ss_bssfp_rf_sinc_lobes,
            self.ss_bssfp_rf_slr_sharpness,
            self.ss_bssfp_bandwidth_khz,
            self.ss_bssfp_repetition_time_ms,
            self.ss_bssfp_phase_start_deg,
            self.ss_bssfp_phase_increment_deg,
            self.ss_bssfp_dummy_repetitions,
            self.ss_bssfp_repetitions,
            self.ss_bssfp_acquisition_interval_ms,
            self.ss_bssfp_alpha_half_tr_ratio,
            self.ss_bssfp_alpha_half_flip_ratio,
            self.ss_bssfp_alpha_half_spacing_ms,
            self.ss_bssfp_spoiler_cycles,
            self.ss_bssfp_spoiler_cycles_per_voxel,
            self.ss_bssfp_spoiler_duration_ms,
        ):
            widget.valueChanged.connect(self._ss_bssfp_changed)
        for widget in (
            self.ss_bssfp_target_names,
            self.ss_bssfp_target_offsets_hz,
            self.ss_bssfp_receiver_offsets_hz,
            self.ss_bssfp_flip_angles_deg,
            self.ss_bssfp_alpha_half_flip_angles_deg,
        ):
            widget.editingFinished.connect(self._ss_bssfp_changed)
        self.ss_bssfp_rf_pulse_type.currentIndexChanged.connect(self._ss_bssfp_changed)
        self.ss_bssfp_alpha_half.toggled.connect(self._ss_bssfp_changed)
        self.ss_bssfp_alpha_half_use_ratios.toggled.connect(self._ss_bssfp_changed)
        for widget in (
            self.radial_me_fov_mm,
            self.radial_me_base_resolution,
            self.radial_me_readout_oversampling,
            self.radial_me_spokes,
            self.radial_me_measurements,
            self.radial_me_acquisition_interval_ms,
            self.radial_me_echoes,
            self.radial_me_echo_spacing_ms,
            self.radial_me_pixel_bandwidth_hz,
            self.radial_me_flip_angle_deg,
            self.radial_me_rf_duration_ms,
            self.radial_me_repetition_time_ms,
            self.radial_me_phase_start_deg,
            self.radial_me_phase_increment_deg,
            self.radial_me_alpha_half_tr_ratio,
            self.radial_me_alpha_half_flip_ratio,
            self.radial_me_alpha_half_center_spacing_ms,
            self.radial_me_alpha_half_flip_angle_deg,
            self.radial_me_prephaser_duration_ms,
            self.radial_me_rotation_deg,
        ):
            widget.valueChanged.connect(self._radial_me_bssfp_changed)
        self.radial_me_alpha_half.toggled.connect(self._radial_me_bssfp_changed)
        self.radial_me_alpha_half_use_ratios.toggled.connect(
            self._radial_me_bssfp_changed
        )
        self.radial_me_tip_back.toggled.connect(self._radial_me_bssfp_changed)
        for widget in (
            self.me_bssfp_read_matrix,
            self.me_bssfp_phase_matrix,
            self.me_bssfp_partition_matrix,
            self.me_bssfp_read_fov_mm,
            self.me_bssfp_phase_fov_mm,
            self.me_bssfp_partition_fov_mm,
            self.me_bssfp_echoes,
            self.me_bssfp_echo_spacing_ms,
            self.me_bssfp_bandwidth_khz,
            self.me_bssfp_flip_angle_deg,
            self.me_bssfp_rf_duration_ms,
            self.me_bssfp_rf_offset_hz,
            self.me_bssfp_receiver_offset_hz,
            self.me_bssfp_encoding_duration_ms,
            self.me_bssfp_repetition_time_ms,
            self.me_bssfp_phase_start_deg,
            self.me_bssfp_phase_increment_deg,
            self.me_bssfp_alpha_half_tr_ratio,
            self.me_bssfp_alpha_half_flip_ratio,
            self.me_bssfp_alpha_half_center_spacing_ms,
            self.me_bssfp_alpha_half_flip_angle_deg,
            self.me_bssfp_dummy_repetitions,
            self.me_bssfp_repetitions,
            self.me_bssfp_acquisition_interval_ms,
        ):
            widget.valueChanged.connect(self._me_bssfp_changed)
        self.me_bssfp_readout_strategy.currentIndexChanged.connect(
            self._me_bssfp_changed
        )
        self.me_bssfp_rf_pulse_type.currentIndexChanged.connect(self._me_bssfp_changed)
        self.me_bssfp_alpha_half.toggled.connect(self._me_bssfp_changed)
        self.me_bssfp_alpha_half_use_ratios.toggled.connect(self._me_bssfp_changed)
        self._connect_three_dimensional_orientation_controls(
            self.bssfp_read_gradient_axis,
            self.bssfp_phase_gradient_axis,
            self.bssfp_partition_gradient_axis,
            self._bssfp_changed,
        )
        self._connect_three_dimensional_orientation_controls(
            self.ss_bssfp_read_gradient_axis,
            self.ss_bssfp_phase_gradient_axis,
            self.ss_bssfp_partition_gradient_axis,
            self._ss_bssfp_changed,
        )
        self._connect_three_dimensional_orientation_controls(
            self.radial_me_read_gradient_axis,
            self.radial_me_phase_gradient_axis,
            self.radial_me_partition_gradient_axis,
            self._radial_me_bssfp_changed,
        )
        self._connect_three_dimensional_orientation_controls(
            self.me_bssfp_read_gradient_axis,
            self.me_bssfp_phase_gradient_axis,
            self.me_bssfp_partition_gradient_axis,
            self._me_bssfp_changed,
        )
        for prefix, callback in (
            ("epi", self._acquisition_changed),
            ("csi", self._csi_changed),
            ("flash", self._flash_changed),
            ("bssfp", self._bssfp_changed),
            ("ss_bssfp", self._ss_bssfp_changed),
            ("radial_me", self._radial_me_bssfp_changed),
            ("me_bssfp", self._me_bssfp_changed),
        ):
            self._connect_shared_rf_controls(prefix, callback)
        self.fov_mm.valueChanged.connect(self._acquisition_changed)
        self.fov_z_mm.valueChanged.connect(self._acquisition_changed)
        self.t1_ms.valueChanged.connect(self._phantom_relaxation_changed)
        self.probe_t1_ms.valueChanged.connect(self._phantom_relaxation_changed)
        self.object_type.currentIndexChanged.connect(self._phantom_relaxation_changed)
        self.object_type.currentIndexChanged.connect(
            self._update_simulation_object_table
        )
        for widget in self._built_in_object_widgets[1:]:
            widget.valueChanged.connect(self._update_simulation_object_table)
        for widget in (
            self.matrix_size,
            self.z_matrix_size,
            self.fov_mm,
            self.fov_z_mm,
        ):
            widget.valueChanged.connect(self._update_spoiling_quality)
        for widget in (
            self.probe_points,
            self.probe_ppm_min,
            self.probe_ppm_max,
            self.probe_frequency_ppm,
            self.probe_position_x_mm,
            self.probe_position_y_mm,
            self.probe_position_z_mm,
            self.probe_t1_ms,
            self.probe_t2_ms,
        ):
            widget.valueChanged.connect(self._update_simulation_object_table)
        self.probe_frequency_units.currentIndexChanged.connect(
            self._update_simulation_object_table
        )
        self._update_bandwidth_labels()
        self._update_csi_labels()
        self._update_flash_labels()
        self._update_bssfp_labels()
        self._update_ss_bssfp_labels()
        self._update_radial_me_bssfp_labels()
        self._update_me_bssfp_labels()
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
        views = QTabWidget()
        self.views = views
        views_font = views.tabBar().font()
        views_font.setBold(True)
        views.tabBar().setFont(views_font)
        viewer_column_layout.addWidget(views, 1)
        splitter.addWidget(viewer_column)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        timeline = QWidget()
        timeline_layout = QVBoxLayout(timeline)
        waveform_controls = QHBoxLayout()
        waveform_controls.addWidget(QLabel("Waveform units"))
        self.waveform_units = QComboBox()
        self.waveform_units.addItem("Physical (G, T/m)", "physical")
        self.waveform_units.addItem("Simulation (Hz, kHz/m)", "simulation")
        self.waveform_units.setToolTip(
            "Display RF as physical B1 in gauss and gradients in T/m, or show "
            "the canonical simulation frequency units"
        )
        self.waveform_units.currentIndexChanged.connect(self._waveform_units_changed)
        self.nucleus.currentTextChanged.connect(self._waveform_units_changed)
        self.field_strength_t.valueChanged.connect(self._frequency_reference_changed)
        self.nucleus.currentTextChanged.connect(self._frequency_reference_changed)
        waveform_controls.addWidget(self.waveform_units)
        self.waveform_nucleus_label = QLabel("Conversion: H1")
        waveform_controls.addWidget(self.waveform_nucleus_label)
        waveform_controls.addStretch()
        timeline_layout.addLayout(waveform_controls)
        self.waveform_value_summary = QLabel("No sequence waveforms")
        self.waveform_value_summary.setWordWrap(True)
        timeline_layout.addWidget(self.waveform_value_summary)
        self.rf_plot = pg.PlotWidget(title="RF magnitude and phase")
        self.rf_plot.setLabel("left", "B1", "G")
        self.rf_plot.setLabel("bottom", "Time", "ms")
        self.rf_plot.plotItem.showAxis("right")
        self.rf_plot.plotItem.getAxis("right").setLabel(
            "RF phase", units="°", color="#00bcd4"
        )
        self._rf_phase_view = pg.ViewBox()
        self.rf_plot.scene().addItem(self._rf_phase_view)
        self.rf_plot.plotItem.getAxis("right").linkToView(self._rf_phase_view)
        self._rf_phase_view.setXLink(self.rf_plot.plotItem.vb)
        self._rf_phase_view.setYRange(-180.0, 180.0, padding=0)
        self.rf_plot.plotItem.vb.sigResized.connect(self._sync_rf_phase_view)
        self._sync_rf_phase_view()
        self.gradient_plot = pg.PlotWidget(title="Gradients and ADC")
        self.gradient_plot.setLabel("left", "Gradient", "T/m")
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
        self.signal_page = signal_page
        signal_layout = QVBoxLayout(signal_page)
        # Split View is a Signal/CSI-specific inspection mode. Keeping these
        # controls inside this page leaves the result tabs aligned with the
        # Phantom workspace tabs.
        signal_layout.addLayout(view_mode_row)
        self.view_stack = QStackedWidget()
        signal_layout.addWidget(self.view_stack, 1)
        normal_signal_page = QWidget()
        self.normal_signal_page = normal_signal_page
        normal_signal_layout = QVBoxLayout(normal_signal_page)
        self.view_stack.addWidget(normal_signal_page)
        # CSI voxel coordinates remain as internal state controls for the
        # reconstruction code. Voxel selection itself is performed directly
        # in the clickable reconstruction image below, so the result plot no
        # longer loses a full row to two wide sliders.
        self.spectrum_x_selector = QSpinBox(signal_page)
        self.spectrum_y_selector = QSpinBox(signal_page)
        self.spectrum_x_slider = QSlider(Qt.Horizontal, signal_page)
        self.spectrum_y_slider = QSlider(Qt.Horizontal, signal_page)
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
            selector.hide()
            slider.hide()
        self.signal_plot = pg.PlotWidget(title="Received ADC signal")
        self.signal_plot.setLabel("left", "Signal", "a.u.")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        self.signal_plot.addLegend()
        normal_signal_layout.addWidget(self.signal_plot)
        self.spectrum_info = QLabel("No spectroscopic result")
        normal_signal_layout.addWidget(self.spectrum_info)
        self.signal_tab_index = views.addTab(signal_page, "Signal / CSI spectrum")

        two_d_page = QWidget()
        two_d_layout = QVBoxLayout(two_d_page)
        two_d_splitter = QSplitter(Qt.Horizontal)
        two_d_layout.addWidget(two_d_splitter, 1)

        kspace_page = QWidget()
        kspace_layout = QVBoxLayout(kspace_page)
        kspace_title = QLabel("2D k-space")
        kspace_title_font = kspace_title.font()
        kspace_title_font.setBold(True)
        kspace_title.setFont(kspace_title_font)
        kspace_layout.addWidget(kspace_title)
        self.kspace_view = pg.ImageView()
        self.kspace_view.ui.roiBtn.hide()
        self.kspace_view.ui.menuBtn.hide()
        self._format_colorbar(self.kspace_view)
        kspace_layout.addWidget(self.kspace_view)
        self.kspace_zoom_info = QLabel("Zoom: —")
        kspace_layout.addWidget(self.kspace_zoom_info)
        self.kspace_info = QLabel("No 2D Cartesian result")
        kspace_layout.addWidget(self.kspace_info)
        two_d_splitter.addWidget(kspace_page)

        reconstruction_page = QWidget()
        reconstruction_layout = QVBoxLayout(reconstruction_page)
        reconstruction_title = QLabel("2D Reconstruction")
        reconstruction_title_font = reconstruction_title.font()
        reconstruction_title_font.setBold(True)
        reconstruction_title.setFont(reconstruction_title_font)
        reconstruction_layout.addWidget(reconstruction_title)
        self.reconstruction_view = pg.ImageView()
        self.reconstruction_view.ui.roiBtn.hide()
        self.reconstruction_view.ui.menuBtn.hide()
        self._format_colorbar(self.reconstruction_view)
        reconstruction_layout.addWidget(self.reconstruction_view)
        self.reconstruction_zoom_info = QLabel("Zoom: —")
        reconstruction_layout.addWidget(self.reconstruction_zoom_info)
        self.reconstruction_info = QLabel("No 2D Cartesian result")
        reconstruction_layout.addWidget(self.reconstruction_info)
        two_d_splitter.addWidget(reconstruction_page)
        two_d_splitter.setSizes([1, 1])
        self.two_d_result_tab_index = views.addTab(
            two_d_page, "2D k-space / Reconstruction"
        )

        self.reconstruction_explorer = SequenceReconstructionExplorer()
        self.reconstruction_explorer_tab_index = views.addTab(
            self.reconstruction_explorer, "Reconstruction Explorer"
        )

        self.result_volume_viewer = SequenceResultVolumeViewer()
        views.addTab(self.result_volume_viewer, "Spatial Magnetization")

        self.magnetization_animation_viewer = SequenceMagnetizationAnimationViewer()
        # Aliases keep these user-editable next-run settings in the normal
        # project-state capture while their visible controls live in this tab.
        self.animation_enabled = self.magnetization_animation_viewer.capture_enabled
        self.animation_time_resolution_ms = (
            self.magnetization_animation_viewer.time_resolution_ms
        )
        self.animation_storage_dtype = (
            self.magnetization_animation_viewer.storage_dtype_combo
        )
        self.animation_tab_index = views.addTab(
            self.magnetization_animation_viewer, "3D Magnetization Animation"
        )

        probe_page = QWidget()
        probe_layout = QVBoxLayout(probe_page)
        self.probe_info = QLabel("Run a spin probe to populate these views")
        self.probe_info.setWordWrap(True)
        probe_layout.addWidget(self.probe_info)
        self.probe_coherence_info = QLabel()
        self.probe_coherence_info.setWordWrap(True)
        probe_layout.addWidget(self.probe_coherence_info)
        playback_mode_row = QHBoxLayout()
        playback_mode_row.addWidget(QLabel("Playback mode:"))
        self.probe_playback_mode = QComboBox()
        self.probe_playback_mode.setObjectName("sequence_probe_playback_mode")
        self.probe_playback_mode.addItems(
            [
                "Configured checkpoints",
                "ADC only",
                "All simulation steps",
            ]
        )
        self.probe_playback_mode.setToolTip(
            "Use the configured checkpoint view, skip directly between ADC "
            "samples, or inspect every stored simulation state."
        )
        self.probe_playback_mode.setEnabled(False)
        self.probe_playback_mode.currentIndexChanged.connect(
            self._probe_playback_mode_changed
        )
        playback_mode_row.addWidget(self.probe_playback_mode)
        self.probe_adc_status = QLabel("ADC: off")
        self.probe_adc_status.setObjectName("sequence_probe_adc_status")
        self.probe_adc_status.setVisible(False)
        playback_mode_row.addWidget(self.probe_adc_status)
        playback_mode_row.addStretch()
        probe_layout.addLayout(playback_mode_row)
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
        self.probe_magnetization_viewer.header_container.setVisible(False)
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
        self.split_image_plot.setBackground(IMAGE_CANVAS_BACKGROUND)
        self.split_image_plot.setAspectLocked(True)
        self.split_image_plot.setLabel("bottom", "x index")
        self.split_image_plot.setLabel("left", "y index")
        self.split_image_item = pg.ImageItem()
        style_image_item(self.split_image_item)
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
        for plot in (
            self.rf_plot,
            self.gradient_plot,
            self.signal_plot,
            self.split_signal_plot,
        ):
            plot.setToolTip(AXIS_ZOOM_TOOLTIP)
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
        ):
            view.getView().sigRangeChanged.connect(
                lambda *_, image_view=view, zoom_label=label: self._update_zoom_label(
                    image_view, zoom_label
                )
            )

    def apply_focused_workspace_layout(self):
        """Use an adaptive control/viewer split in the focused workspace."""
        available = int(self.workspace_splitter.width())
        if available <= 0:
            available = sum(self.workspace_splitter.sizes())
        if available <= 0:
            self.workspace_splitter.setSizes([self.FOCUSED_CONTROL_WIDTH, 1000])
            return
        preferred_control_width = min(
            self.FOCUSED_CONTROL_WIDTH,
            max(self.MINIMUM_FOCUSED_CONTROL_WIDTH, round(available * 0.30)),
        )
        control_width = min(
            preferred_control_width,
            max(1, available - self.MINIMUM_FOCUSED_VIEWER_WIDTH),
        )
        self.workspace_splitter.setSizes(
            [control_width, max(1, available - control_width)]
        )

    def activate_focused_workspace_layout(self):
        """Apply the focused split now and once more after Qt finishes layout."""
        self.apply_focused_workspace_layout()
        QTimer.singleShot(0, self.apply_focused_workspace_layout)

    @staticmethod
    def _format_colorbar(view):
        compact_image_histogram(view)
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

    @staticmethod
    def _sampling_bandwidth_spin(default_khz):
        """Create the common total ADC-bandwidth field for imaging readouts."""
        widget = QDoubleSpinBox()
        widget.setRange(0.1, 2000.0)
        widget.setDecimals(3)
        widget.setSingleStep(1.0)
        widget.setValue(default_khz)
        widget.setSuffix(" kHz")
        widget.setToolTip(
            "Total ADC sampling bandwidth; dwell is derived as 1 / bandwidth "
            "and rounded to the scanner ADC raster"
        )
        return widget

    def _add_shared_rf_controls(
        self,
        form,
        prefix,
        *,
        pulse_type="Sinc",
        duration_ms=1.0,
        sinc_lobes=3,
        apodization=0.5,
        slr_sharpness=1.0,
        frequency_offset_hz=0.0,
        label_prefix="RF",
    ):
        """Create the common RF field set used by every generated sequence."""
        pulse_type_control = QComboBox()
        pulse_type_control.setObjectName(f"{prefix}_rf_pulse_type")
        pulse_type_control.addItems(RF_PULSE_TYPE_LABELS)
        pulse_type_control.setCurrentText(pulse_type)
        pulse_type_control.setToolTip(
            "All generated sequences use the same global RF envelope designer. "
            "RF Pulse Designer uses the waveform designed or loaded in Free Mode."
        )
        duration = self._parameter_spin(0.001, 100.0, duration_ms, " ms")
        duration.setObjectName(f"{prefix}_rf_duration_ms")
        duration.setDecimals(3)
        tbw = self._parameter_spin(0.001, 1000.0, 1.0, "")
        tbw.setObjectName(f"{prefix}_rf_time_bandwidth_product")
        tbw.setReadOnly(True)
        tbw.setButtonSymbols(QDoubleSpinBox.NoButtons)
        tbw.setEnabled(False)
        tbw.setToolTip("Calculated automatically from the completed RF pulse shape")
        lobes = QSpinBox()
        lobes.setObjectName(f"{prefix}_rf_sinc_lobes")
        lobes.setRange(1, 100)
        lobes.setValue(max(1, int(sinc_lobes)))
        lobes.setToolTip(
            "Displayed Sinc lobe count; changing it changes the pulse shape and "
            "therefore its automatically calculated TBW"
        )
        apod = self._parameter_spin(0.0, 1.0, apodization, "")
        apod.setObjectName(f"{prefix}_rf_apodization")
        apod.setSingleStep(0.05)
        apod.setToolTip("Cosine apodization of the shared Sinc envelope")
        sharpness = self._parameter_spin(0.1, 20.0, slr_sharpness, "")
        sharpness.setObjectName(f"{prefix}_rf_slr_sharpness")
        sharpness.setSingleStep(0.5)
        sharpness.setToolTip(
            "Higher sharpness narrows the SLR transition and produces more "
            "temporal lobes"
        )
        bandwidth = self._parameter_spin(0.1, 1_000_000.0, 1.0, " Hz")
        bandwidth.setObjectName(f"{prefix}_rf_bandwidth_hz")
        bandwidth.setEnabled(False)
        bandwidth.setToolTip("Calculated as automatic TBW divided by RF duration")
        frequency_offset = self._parameter_spin(
            -1_000_000.0, 1_000_000.0, frequency_offset_hz, " Hz"
        )
        frequency_offset.setObjectName(f"{prefix}_rf_offset_hz")
        frequency_offset.setToolTip(
            "RF carrier offset relative to the sequence centre frequency"
        )
        load_button = QPushButton("Load RF pulse…")
        load_button.setObjectName(f"{prefix}_rf_load_button")
        load_button.setToolTip(
            "Load an RF waveform directly into Sequence Mode and select it "
            "for this sequence"
        )

        for suffix, control in {
            "pulse_type": pulse_type_control,
            "duration_ms": duration,
            "time_bandwidth_product": tbw,
            "sinc_lobes": lobes,
            "apodization": apod,
            "slr_sharpness": sharpness,
            "bandwidth_hz": bandwidth,
            "offset_hz": frequency_offset,
            "load_button": load_button,
        }.items():
            setattr(self, f"{prefix}_rf_{suffix}", control)

        form.addRow(f"{label_prefix} pulse type", pulse_type_control)
        form.addRow(f"{label_prefix} duration", duration)
        form.addRow(f"{label_prefix} time-bandwidth product", tbw)
        form.addRow("Sinc lobes", lobes)
        form.addRow("Sinc apodization", apod)
        form.addRow("SLR sharpness", sharpness)
        form.addRow(f"{label_prefix} bandwidth (auto)", bandwidth)
        form.addRow(f"{label_prefix} frequency offset", frequency_offset)
        form.addRow("Loaded waveform", load_button)
        return pulse_type_control

    def _connect_shared_rf_controls(self, prefix, callback):
        pulse_type = getattr(self, f"{prefix}_rf_pulse_type")
        pulse_type.currentTextChanged.connect(
            lambda *_: self._shared_rf_control_changed(prefix, callback)
        )
        for suffix in (
            "duration_ms",
            "sinc_lobes",
            "apodization",
            "slr_sharpness",
            "offset_hz",
        ):
            control = getattr(self, f"{prefix}_rf_{suffix}")
            control.valueChanged.connect(
                lambda *_, p=prefix, cb=callback: self._shared_rf_control_changed(p, cb)
            )
        getattr(self, f"{prefix}_rf_load_button").clicked.connect(
            lambda *_: self._load_sequence_rf_pulse(prefix)
        )
        self._update_shared_rf_controls(prefix)

    def _shared_rf_control_changed(self, prefix, callback):
        self._update_shared_rf_controls(prefix)
        callback()

    def _selected_shared_rf_pulse_type(self, prefix):
        text = getattr(self, f"{prefix}_rf_pulse_type").currentText()
        return "designer" if text == "RF Pulse Designer" else text.lower()

    def _shared_rf_shape_parameter(self, prefix):
        """Return the fixed construction input, distinct from calculated TBW."""
        pulse_type = self._selected_shared_rf_pulse_type(prefix)
        if pulse_type == "designer":
            if self._rf_designer_pulse_data is not None:
                return self._rf_designer_pulse_data["time_bandwidth_product"]
            return 1.0
        return analytic_rf_shape_parameter(
            pulse_type, getattr(self, f"{prefix}_rf_sinc_lobes").value()
        )

    def _update_shared_rf_controls(self, prefix):
        pulse_type = self._selected_shared_rf_pulse_type(prefix)
        loaded = pulse_type == "designer"
        getattr(self, f"{prefix}_rf_duration_ms").setEnabled(not loaded)
        getattr(self, f"{prefix}_rf_time_bandwidth_product").setEnabled(False)
        getattr(self, f"{prefix}_rf_sinc_lobes").setEnabled(pulse_type == "sinc")
        getattr(self, f"{prefix}_rf_apodization").setEnabled(pulse_type == "sinc")
        getattr(self, f"{prefix}_rf_slr_sharpness").setEnabled(pulse_type == "slr")
        offset = getattr(self, f"{prefix}_rf_offset_hz")
        offset.setEnabled(not loaded)
        if loaded and self._rf_designer_pulse_data is not None:
            duration = getattr(self, f"{prefix}_rf_duration_ms")
            previous = duration.blockSignals(True)
            duration.setValue(self._rf_designer_pulse_data["duration_s"] * 1000.0)
            duration.blockSignals(previous)
            previous = offset.blockSignals(True)
            offset.setValue(self._rf_designer_pulse_data["frequency_offset_hz"])
            offset.blockSignals(previous)
        duration_s = getattr(self, f"{prefix}_rf_duration_ms").value() / 1000.0
        if loaded and self._rf_designer_pulse_data is not None:
            tbw = self._rf_designer_pulse_data["time_bandwidth_product"]
        else:
            shape_parameter = self._shared_rf_shape_parameter(prefix)
            try:
                _, _, tbw, _ = design_rf_envelope(
                    pulse_type=pulse_type,
                    duration_s=duration_s,
                    raster_s=self.scanner_parameters.rf_raster_time_s,
                    time_bandwidth_product=shape_parameter,
                    apodization=getattr(self, f"{prefix}_rf_apodization").value(),
                    slr_sharpness=getattr(self, f"{prefix}_rf_slr_sharpness").value(),
                )
            except (TypeError, ValueError):
                # Keep the UI responsive while another invalid RF parameter is
                # still being edited; sequence generation will report details.
                tbw = shape_parameter
        tbw_control = getattr(self, f"{prefix}_rf_time_bandwidth_product")
        previous = tbw_control.blockSignals(True)
        tbw_control.setValue(tbw)
        tbw_control.blockSignals(previous)
        bandwidth = getattr(self, f"{prefix}_rf_bandwidth_hz")
        previous = bandwidth.blockSignals(True)
        bandwidth.setValue(tbw / duration_s)
        bandwidth.blockSignals(previous)

    def _shared_rf_parameters(self, prefix):
        pulse_type = self._selected_shared_rf_pulse_type(prefix)
        parameters = {
            "rf_pulse_type": pulse_type,
            "rf_duration_s": getattr(self, f"{prefix}_rf_duration_ms").value() / 1000.0,
            "rf_time_bandwidth_product": self._shared_rf_shape_parameter(prefix),
            "rf_apodization": getattr(self, f"{prefix}_rf_apodization").value(),
            "rf_slr_sharpness": getattr(self, f"{prefix}_rf_slr_sharpness").value(),
            "rf_frequency_offset_hz": getattr(self, f"{prefix}_rf_offset_hz").value(),
        }
        if pulse_type == "designer":
            if self._rf_designer_pulse_data is None:
                raise ValueError(self._rf_designer_pulse_error)
            data = self._rf_designer_pulse_data
            parameters.update(
                rf_duration_s=data["duration_s"],
                rf_custom_waveform_hz=tuple(
                    complex(value) for value in data["waveform_hz"]
                ),
                rf_custom_raster_s=data["raster_s"],
                rf_custom_flip_angle_deg=data["flip_angle_deg"],
                rf_custom_name=data["name"],
                rf_frequency_offset_hz=data["frequency_offset_hz"],
            )
        return parameters

    @staticmethod
    def _acquisition_interval_spin():
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 100_000_000.0)
        widget.setDecimals(3)
        widget.setSingleStep(100.0)
        widget.setValue(0.0)
        widget.setSuffix(" ms")
        widget.setSpecialValueText("Back-to-back")
        widget.setToolTip(
            "Start-to-start interval between complete images, volumes, or "
            "measurements. Back-to-back uses the shortest possible interval."
        )
        return widget

    @staticmethod
    def _optional_acquisition_interval_s(widget):
        value_ms = float(widget.value())
        return None if value_ms <= 0.0 else value_ms / 1000.0

    @staticmethod
    def _slice_orientation_combo():
        combo = QComboBox()
        combo.addItem("Axial (XY)", ("+x", "+y", "+z"))
        combo.addItem("Coronal (XZ)", ("+x", "+z", "-y"))
        combo.addItem("Sagittal (YZ)", ("+y", "+z", "+x"))
        combo.addItem("Custom", None)
        combo.setToolTip(
            "Choose a common scanner-plane preset. Read and phase directions "
            "remain separately editable below."
        )
        return combo

    @staticmethod
    def _gradient_axis_combo():
        combo = QComboBox()
        for axis in ("x", "y", "z"):
            combo.addItem(f"+{axis.upper()}", f"+{axis}")
            combo.addItem(f"-{axis.upper()}", f"-{axis}")
        return combo

    def _two_dimensional_orientation_controls(self, prefix):
        preset = self._slice_orientation_combo()
        preset.setObjectName(f"{prefix}_slice_orientation")
        read = self._gradient_axis_combo()
        read.setObjectName(f"{prefix}_read_gradient_axis")
        read.setToolTip(
            "Physical scanner axis and polarity used for the readout gradient"
        )
        phase = self._gradient_axis_combo()
        phase.setObjectName(f"{prefix}_phase_gradient_axis")
        phase.setToolTip(
            "Physical scanner axis and polarity used for the phase-encoding gradient"
        )
        phase.setCurrentIndex(2)
        slice_axis = QLabel()
        slice_axis.setObjectName(f"{prefix}_slice_gradient_axis")
        slice_axis.setWordWrap(True)
        slice_axis.setToolTip(
            "Automatically derived as Read × Phase so the encoding frame stays "
            "right-handed"
        )
        return preset, read, phase, slice_axis

    def _three_dimensional_orientation_controls(
        self,
        prefix,
        *,
        default_read_axis="+x",
        default_phase_axis="+y",
    ):
        read = self._gradient_axis_combo()
        read.setObjectName(f"{prefix}_read_gradient_axis")
        self._set_combo_data(read, default_read_axis)
        read.setToolTip(
            "Physical scanner axis and polarity used for the logical read direction"
        )
        phase = self._gradient_axis_combo()
        phase.setObjectName(f"{prefix}_phase_gradient_axis")
        self._set_combo_data(phase, default_phase_axis)
        phase.setToolTip(
            "Physical scanner axis and polarity used for the logical phase direction"
        )
        partition = QLabel()
        partition.setObjectName(f"{prefix}_partition_gradient_axis")
        partition.setWordWrap(True)
        partition.setToolTip(
            "Automatically derived as Read × Phase so the encoding frame stays "
            "right-handed"
        )
        return read, phase, partition

    def _update_three_dimensional_orientation_state(self, read, phase, partition):
        frame = EncodingFrame.from_read_phase_axes(
            str(read.currentData()), str(phase.currentData())
        )
        partition_code = frame.axis_codes[2]
        partition.setText(f"{partition_code.upper()} — derived as Read × Phase")

    def _connect_three_dimensional_orientation_controls(
        self, read, phase, partition, changed_callback
    ):
        def axes_changed(changed):
            read_axis = str(read.currentData())
            phase_axis = str(phase.currentData())
            if read_axis[-1:] == phase_axis[-1:]:
                other = phase if changed is read else read
                occupied_axis = str(changed.currentData())[-1:]
                replacement = next(
                    f"+{axis}" for axis in ("x", "y", "z") if axis != occupied_axis
                )
                previous = other.blockSignals(True)
                self._set_combo_data(other, replacement)
                other.blockSignals(previous)
            self._update_three_dimensional_orientation_state(read, phase, partition)
            changed_callback()

        read.currentIndexChanged.connect(lambda *_: axes_changed(read))
        phase.currentIndexChanged.connect(lambda *_: axes_changed(phase))
        self._update_three_dimensional_orientation_state(read, phase, partition)

    @staticmethod
    def _set_combo_data(combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _plane_name_for_slice_axis(axis_code):
        return {
            "x": "Sagittal (YZ)",
            "y": "Coronal (XZ)",
            "z": "Axial (XY)",
        }[str(axis_code)[-1].lower()]

    def _update_two_dimensional_orientation_state(
        self, preset, read, phase, slice_axis
    ):
        read_axis = str(read.currentData())
        phase_axis = str(phase.currentData())
        try:
            frame = EncodingFrame.from_read_phase_axes(read_axis, phase_axis)
        except ValueError:
            slice_axis.setText("Invalid — Read and Phase must use different axes")
            slice_axis.setStyleSheet("color: #b00020;")
            target_preset = preset.findText("Custom")
        else:
            read_code, phase_code, partition_code = frame.axis_codes
            plane = self._plane_name_for_slice_axis(partition_code)
            slice_axis.setText(
                f"{partition_code.upper()} — {plane} (derived as Read × Phase)"
            )
            slice_axis.setStyleSheet("")
            target_preset = preset.findText("Custom")
            for index in range(preset.count()):
                axes = preset.itemData(index)
                if axes is not None and tuple(axes[:2]) == (read_code, phase_code):
                    target_preset = index
                    break
        previous = preset.blockSignals(True)
        preset.setCurrentIndex(target_preset)
        preset.blockSignals(previous)

    def _connect_two_dimensional_orientation_controls(
        self, preset, read, phase, slice_axis, changed_callback
    ):
        def preset_changed(*_):
            axes = preset.currentData()
            if axes is not None:
                read_previous = read.blockSignals(True)
                phase_previous = phase.blockSignals(True)
                self._set_combo_data(read, axes[0])
                self._set_combo_data(phase, axes[1])
                read.blockSignals(read_previous)
                phase.blockSignals(phase_previous)
            self._update_two_dimensional_orientation_state(
                preset, read, phase, slice_axis
            )
            changed_callback()

        def axes_changed(changed):
            read_axis = str(read.currentData())
            phase_axis = str(phase.currentData())
            if read_axis[-1:] == phase_axis[-1:]:
                other = phase if changed is read else read
                occupied_axis = str(changed.currentData())[-1:]
                replacement = next(
                    f"+{axis}" for axis in ("x", "y", "z") if axis != occupied_axis
                )
                previous = other.blockSignals(True)
                self._set_combo_data(other, replacement)
                other.blockSignals(previous)
            self._update_two_dimensional_orientation_state(
                preset, read, phase, slice_axis
            )
            changed_callback()

        preset.currentIndexChanged.connect(preset_changed)
        read.currentIndexChanged.connect(lambda *_: axes_changed(read))
        phase.currentIndexChanged.connect(lambda *_: axes_changed(phase))
        self._update_two_dimensional_orientation_state(preset, read, phase, slice_axis)

    def _object_source_changed(self, *_):
        """Expose controls only for the object source that actually owns them."""
        source_index = self.object_source.currentIndex()
        phantom_selected = source_index == 0
        built_in_selected = source_index == 1
        probe_selected = source_index == 2
        self.object_type.blockSignals(True)
        if phantom_selected:
            self.object_type.setCurrentIndex(0)
        elif built_in_selected and self.object_type.currentIndex() == 0:
            self.object_type.setCurrentIndex(1)
        self.object_type.blockSignals(False)
        for widget in self._built_in_object_widgets:
            widget.setEnabled(built_in_selected)
        self.built_in_properties_group.setVisible(built_in_selected)
        self.built_in_properties_group.setEnabled(built_in_selected)
        self.phantom_summary.hide()
        self.phantom_summary_label.hide()
        self.frequency_reference_info.hide()
        self.frequency_reference_label.hide()
        self.open_phantom_button.setVisible(phantom_selected)
        self.probe_group.setVisible(probe_selected)
        self._update_run_action_availability()
        self.refresh_object_summary()
        self._update_all_ernst_controls()
        self._update_frequency_reference_info()
        self._update_spoiling_quality()
        if probe_selected:
            phantom = self._selected_designed_phantom()
            if isinstance(phantom, (SpectralPhantom, DynamicSpectralPhantom)):
                self.field_strength_t.setValue(phantom.field_strength)
                nucleus_index = self.nucleus.findText(phantom.nucleus)
                if nucleus_index >= 0:
                    self.nucleus.setCurrentIndex(nucleus_index)
                self._apply_probe_defaults_from_phantom(phantom)

    def _update_run_action_availability(self):
        """Dim actions that do not apply to the selected simulation object."""
        probe_selected = self.object_source.currentIndex() == 2
        probe_running = self.probe_worker is not None and self.probe_worker.isRunning()
        sequence_running = self.worker is not None and self.worker.isRunning()
        idle = not probe_running and not sequence_running
        self.run_probe_button.setEnabled(
            probe_selected and self.program is not None and idle
        )
        self.run_geometry_probe_button.setEnabled(
            probe_selected and self.program is not None and idle
        )
        self.run_button.setEnabled(
            not probe_selected and self.program is not None and idle
        )

    def _selected_designed_phantom(self):
        main_window = self.window()
        phantom_widget = getattr(main_window, "phantom_widget", None)
        return getattr(phantom_widget, "current_phantom", None)

    def refresh_object_summary(self, *_):
        """Refresh the read-only summary of the shared Phantom-tab object."""
        if self.object_source.currentIndex() != 0:
            self._update_simulation_object_table()
            self._update_spoiling_quality()
            return
        phantom = self._selected_designed_phantom()
        if phantom is None:
            self.phantom_summary.setText(
                "No phantom selected. Create or load one in the Phantom tab."
            )
            self._update_simulation_object_table()
            auto_changed = self._apply_flash_auto_spoilers()
            self._update_flash_spoiler_info()
            self._update_spoiling_quality()
            if (
                auto_changed
                and self.sequence_source.currentIndex() == self.FLASH_SOURCE
            ):
                self._request_generated_sequence_refresh()
            self._phantom_relaxation_changed()
            return
        active = np.asarray(phantom.mask, dtype=bool)
        if isinstance(phantom, (SpectralPhantom, DynamicSpectralPhantom)):
            self.field_strength_t.setValue(phantom.field_strength)
            nucleus_index = self.nucleus.findText(phantom.nucleus)
            if nucleus_index >= 0:
                self.nucleus.setCurrentIndex(nucleus_index)
            self._apply_probe_defaults_from_phantom(phantom)
        t1 = np.asarray(phantom.t1_map)[active] * 1000.0
        t2 = np.asarray(phantom.t2_map)[active] * 1000.0
        fov_mm = " × ".join(f"{value * 1000:.4g}" for value in phantom.fov)
        relaxation_text = (
            f"T1 {t1.min():.4g}–{t1.max():.4g} ms; "
            f"T2/T2* {t2.min():.4g}–{t2.max():.4g} ms"
            if t1.size and t2.size
            else "No active tissue voxels"
        )
        b0_text = ""
        if isinstance(phantom, SpectralPhantom):
            b0_hz = phantom.get_b0_offset_map_hz(
                phantom.field_strength, phantom.nucleus
            )
        elif isinstance(phantom, DynamicSpectralPhantom):
            b0_hz = phantom.b0_offset_hz(phantom.field_strength, phantom.nucleus)
        else:
            b0_hz = None
        if b0_hz is not None:
            active_b0 = np.asarray(b0_hz, dtype=float)[active]
            if active_b0.size:
                b0_text = (
                    f"\nB0 {phantom.field_strength:g} T {phantom.nucleus}; "
                    f"offset {active_b0.min():.4g}–{active_b0.max():.4g} Hz"
                )
        tx_map = getattr(phantom, "tx_sensitivity_map", None)
        rx_maps = getattr(phantom, "rx_sensitivity_maps", None)
        b1_text = ""
        if tx_map is not None:
            active_tx = np.abs(np.asarray(tx_map))[active]
            if active_tx.size:
                b1_text += f"\nB1+ |scale| {active_tx.min():.4g}–{active_tx.max():.4g}"
        if rx_maps is not None:
            rx_array = np.asarray(rx_maps)
            b1_text += f"; B1− {rx_array.shape[0]} receive channel(s)"
        self.phantom_summary.setText(
            f"{phantom.name}\n"
            f"{phantom.ndim}D, matrix {tuple(phantom.shape)}, "
            f"FOV {fov_mm} mm, {phantom.n_active} active voxels\n"
            f"{relaxation_text}{b0_text}{b1_text}"
        )
        auto_changed = self._apply_flash_auto_spoilers()
        self._update_flash_spoiler_info()
        if auto_changed and self.sequence_source.currentIndex() == self.FLASH_SOURCE:
            self._request_generated_sequence_refresh()
        self._update_ss_bssfp_spoiler_info()
        self._update_frequency_reference_info()
        self._update_spoiling_quality()
        if hasattr(self, "waveform_value_summary"):
            self._update_waveform_value_summary()
        self._phantom_relaxation_changed()

    def _mark_sequence_generation_pending(self):
        source_index = self.sequence_source.currentIndex()
        if source_index not in self.GENERATED_SOURCES:
            return
        self._sequence_generation_pending = True
        self.generate_sequence_button.setEnabled(True)
        if self.program is None:
            preview_text = "No generated sequence is loaded yet."
        else:
            preview_text = "The timeline still shows the previously loaded sequence."
        self._show_sequence_message(
            "Sequence parameters changed. Click Generate sequence to refresh.\n"
            f"{preview_text}"
        )
        self._emit_physical_b1_changed()

    def _request_generated_sequence_refresh(self):
        if self._restoring_session_run:
            return False
        if self.sequence_source.currentIndex() not in self.GENERATED_SOURCES:
            return False
        self._mark_sequence_generation_pending()
        if self.sequence_live_preview.isChecked():
            return self._reload_selected_generated_sequence()
        return False

    def _sequence_live_preview_changed(self, enabled):
        if enabled and self.sequence_source.currentIndex() in self.GENERATED_SOURCES:
            self._reload_selected_generated_sequence()

    def _generate_sequence_clicked(self):
        self._reload_selected_generated_sequence()

    def _ensure_current_generated_sequence(self):
        source_index = self.sequence_source.currentIndex()
        if source_index not in self.GENERATED_SOURCES:
            return self.program is not None
        if (
            self.program is None
            or self._sequence_generation_pending
            or self._generated_sequence_source_index != source_index
        ):
            return self._reload_selected_generated_sequence()
        return True

    def _reload_selected_generated_sequence(self):
        source_index = self.sequence_source.currentIndex()
        loaders = {
            1: self._load_cartesian_epi,
            2: self._load_csi,
            3: self._load_bssfp,
            4: self._load_ss_bssfp,
            5: self._load_radial_me_bssfp,
            6: self._load_me_bssfp,
            self.FLASH_SOURCE: self._load_flash,
        }
        loader = loaders.get(source_index)
        if loader is None:
            return False
        state_names = (
            "program",
            "_generated_pulseq_sequence",
            "_acquisition_compiled",
            "acquisition",
            "acquisition_frames",
            "acquisition_volumes",
            "spiral_acquisition",
            "spectroscopic_acquisition",
            "acquisition_note",
        )
        previous_state = {name: getattr(self, name) for name in state_names}
        self._generation_error = ""
        self._preserve_sequence_plot_range_on_next_show = bool(
            self.program is not None
            and self._generated_sequence_source_index == source_index
        )
        try:
            success = bool(loader())
        finally:
            self._preserve_sequence_plot_range_on_next_show = False
        if success:
            self._sequence_generation_pending = False
            self._generated_sequence_source_index = source_index
            self._generation_error = ""
            if source_index == self.FLASH_SOURCE:
                self._update_flash_spoiler_info()
            self._update_spoiling_quality()
            self._emit_physical_b1_changed()
            return True
        for name, value in previous_state.items():
            setattr(self, name, value)
        self._sequence_generation_pending = True
        message = self._generation_error or "Sequence generation failed."
        if self.program is None:
            message += "\nNo usable sequence is currently loaded."
        else:
            message += "\nThe timeline still shows the last valid sequence preview."
        self._show_sequence_message(message)
        self._emit_physical_b1_changed()
        return False

    def _update_frequency_reference_info(self):
        source_index = self.object_source.currentIndex()
        if source_index == 1:
            text = "Built-in B0 and chemical-shift values are entered in ppm."
            conversion_enabled = True
        elif source_index == 2:
            text = (
                "Spin-probe frequency axes in ppm use the selected field strength "
                "and nucleus for conversion to Hz."
            )
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
        self.field_strength_t.setVisible(conversion_enabled)
        self.field_strength_label.setVisible(conversion_enabled)
        self.nucleus.setVisible(conversion_enabled)
        self.nucleus_label.setVisible(conversion_enabled)
        self.frequency_reference_info.setText(text)
        self.frequency_reference_info.hide()
        self.frequency_reference_label.hide()
        self._update_simulation_object_table()

    def _simulation_object_summary_rows(self):
        rows = [("Frequency model", self.frequency_reference_info.text())]
        if self.field_strength_t.isEnabled():
            rows.append(
                (
                    "Frequency reference",
                    f"B0 {self.field_strength_t.value():g} T; "
                    f"nucleus {self.nucleus.currentText()}",
                )
            )

        source_index = self.object_source.currentIndex()
        if source_index == 1:
            rows.extend(
                (
                    ("Simulation object", self.object_type.currentText()),
                    (
                        "Geometry",
                        f"matrix ({self.matrix_size.value()}, "
                        f"{self.matrix_size.value()}, {self.z_matrix_size.value()}); "
                        f"FOV {self.fov_mm.value():g} × {self.fov_mm.value():g} × "
                        f"{self.fov_z_mm.value():g} mm",
                    ),
                    (
                        "Relaxation / density",
                        f"T1 {self.t1_ms.value():g} ms; T2/T2* "
                        f"{self.t2_ms.value():g} ms; density {self.pd.value():g}",
                    ),
                    (
                        "Frequency offsets",
                        f"B0 {self.b0_ppm.value():g} ppm; chemical shift "
                        f"{self.chemical_ppm.value():g} ppm",
                    ),
                )
            )
            return rows

        if source_index == 2:
            unit = self.probe_frequency_units.currentText()
            rows.extend(
                (
                    ("Simulation object", "Spin probe"),
                    (
                        "Frequency samples",
                        f"{self.probe_points.value()} points from "
                        f"{self.probe_ppm_min.value():g} to "
                        f"{self.probe_ppm_max.value():g} {unit}",
                    ),
                    (
                        "Position",
                        f"({self.probe_position_x_mm.value():g}, "
                        f"{self.probe_position_y_mm.value():g}, "
                        f"{self.probe_position_z_mm.value():g}) mm",
                    ),
                    (
                        "Relaxation",
                        f"T1 {self.probe_t1_ms.value():g} ms; T2/T2* "
                        f"{self.probe_t2_ms.value():g} ms",
                    ),
                )
            )
            return rows

        phantom = self._selected_designed_phantom()
        if phantom is None:
            rows.append(
                (
                    "Selected phantom",
                    "No phantom selected. Create or load one in the Phantom tab.",
                )
            )
            return rows

        active = np.asarray(phantom.mask, dtype=bool)
        t1 = np.asarray(phantom.t1_map)[active] * 1000.0
        t2 = np.asarray(phantom.t2_map)[active] * 1000.0
        fov_mm = " × ".join(f"{value * 1000:.4g}" for value in phantom.fov)
        rows.extend(
            (
                ("Selected phantom", phantom.name),
                (
                    "Geometry",
                    f"{phantom.ndim}D; matrix {tuple(phantom.shape)}; "
                    f"FOV {fov_mm} mm; {phantom.n_active} active voxels",
                ),
                (
                    "Relaxation",
                    (
                        f"T1 {t1.min():.4g}–{t1.max():.4g} ms; "
                        f"T2/T2* {t2.min():.4g}–{t2.max():.4g} ms"
                        if t1.size and t2.size
                        else "No active tissue voxels"
                    ),
                ),
            )
        )
        if isinstance(phantom, SpectralPhantom):
            b0_hz = phantom.get_b0_offset_map_hz(
                phantom.field_strength, phantom.nucleus
            )
        elif isinstance(phantom, DynamicSpectralPhantom):
            b0_hz = phantom.b0_offset_hz(phantom.field_strength, phantom.nucleus)
        else:
            b0_hz = None
        if b0_hz is not None:
            active_b0 = np.asarray(b0_hz, dtype=float)[active]
            if active_b0.size:
                rows.append(
                    (
                        "B0",
                        f"{phantom.field_strength:g} T {phantom.nucleus}; offset "
                        f"{active_b0.min():.4g}–{active_b0.max():.4g} Hz",
                    )
                )
        tx_map = getattr(phantom, "tx_sensitivity_map", None)
        rx_maps = getattr(phantom, "rx_sensitivity_maps", None)
        sensitivity_parts = []
        if tx_map is not None:
            active_tx = np.abs(np.asarray(tx_map))[active]
            if active_tx.size:
                sensitivity_parts.append(
                    f"B1+ |scale| {active_tx.min():.4g}–{active_tx.max():.4g}"
                )
        if rx_maps is not None:
            sensitivity_parts.append(
                f"B1− {np.asarray(rx_maps).shape[0]} receive channel(s)"
            )
        if sensitivity_parts:
            rows.append(("RF sensitivity", "; ".join(sensitivity_parts)))
        return rows

    def _update_simulation_object_table(self, *_):
        if not hasattr(self, "simulation_object_table"):
            return
        table = self.simulation_object_table
        rows = self._simulation_object_summary_rows()
        table.setRowCount(len(rows))
        for row, (parameter, value) in enumerate(rows):
            parameter_item = QTableWidgetItem(str(parameter))
            value_item = QTableWidgetItem(str(value))
            parameter_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            value_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            table.setItem(row, 0, parameter_item)
            table.setItem(row, 1, value_item)
        self._fit_summary_table(table, 360)

    @staticmethod
    def _fit_summary_table(table, maximum_height):
        table.resizeRowsToContents()
        content_height = table.horizontalHeader().height() + 2 * table.frameWidth()
        content_height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(min(maximum_height, max(80, content_height)))

    def _schedule_simulation_object_table_fit(self, *_):
        QTimer.singleShot(
            0, lambda: self._fit_summary_table(self.simulation_object_table, 360)
        )

    def _schedule_sequence_summary_table_fit(self, *_):
        QTimer.singleShot(
            0, lambda: self._fit_summary_table(self.sequence_summary_table, 340)
        )

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

    def _show_no_phantom_dialog(self):
        """Show the missing-phantom error with an in-app navigation link."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle("Invalid simulation")
        dialog.setTextFormat(Qt.RichText)
        dialog.setText(
            "No phantom is loaded in the Phantom tab. Create or load one first."
            "<br><br><a href='open-phantom'>Open the Phantom tab</a>"
        )
        dialog.setStandardButtons(QMessageBox.Ok)

        def open_phantom(_link):
            dialog.accept()
            self._open_phantom_tab()

        for label in dialog.findChildren(QLabel):
            label.setOpenExternalLinks(False)
            label.linkActivated.connect(open_phantom)
        dialog.exec_()

    def _source_changed(self, *_):
        source_index = self.sequence_source.currentIndex()
        source_changed = source_index != self._selected_sequence_source_index
        self._selected_sequence_source_index = source_index
        epi_selected = source_index == 1
        csi_selected = source_index == 2
        bssfp_selected = source_index == 3
        ss_bssfp_selected = source_index == 4
        radial_me_selected = source_index == 5
        me_bssfp_selected = source_index == 6
        flash_selected = source_index == self.FLASH_SOURCE
        self.acquisition_group.setVisible(epi_selected)
        self.acquisition_group.setEnabled(epi_selected)
        self.csi_group.setVisible(csi_selected)
        self.csi_group.setEnabled(csi_selected)
        self.bssfp_group.setVisible(bssfp_selected)
        self.bssfp_group.setEnabled(bssfp_selected)
        self.ss_bssfp_group.setVisible(ss_bssfp_selected)
        self.ss_bssfp_group.setEnabled(ss_bssfp_selected)
        self.radial_me_bssfp_group.setVisible(radial_me_selected)
        self.radial_me_bssfp_group.setEnabled(radial_me_selected)
        self.me_bssfp_group.setVisible(me_bssfp_selected)
        self.me_bssfp_group.setEnabled(me_bssfp_selected)
        self.flash_group.setVisible(flash_selected)
        self.flash_group.setEnabled(flash_selected)
        generated_selected = source_index in self.GENERATED_SOURCES
        self.sequence_live_preview.setEnabled(generated_selected)
        self.generate_sequence_button.setEnabled(generated_selected)
        self.export_pulseq_button.setEnabled(generated_selected)
        self.acquisition_hint.setText(
            "Read/phase matrix and sampling bandwidth define each 2D frame. "
            "Choose Cartesian EPI or a single-interleaf centre-out spiral; "
            "slices are acquired sequentially without kz encoding."
            if epi_selected
            else "Select EPI under Source / mode to enable these settings."
        )
        if self._restoring_session_run:
            self._update_spoiling_quality()
            return
        if source_index == self.INTERNAL_SOURCE:
            self._sequence_generation_pending = False
            self._generated_sequence_source_index = None
            self._load_internal_sequence()
        elif generated_selected:
            if source_changed or self._generated_sequence_source_index != source_index:
                self._mark_sequence_generation_pending()
            if self.sequence_live_preview.isChecked():
                self._reload_selected_generated_sequence()
        self._update_spoiling_quality()

    def _three_dimensional_encoding_frame(self, source_index=None):
        if source_index is None:
            source_index = self.sequence_source.currentIndex()
        controls = {
            self.BSSFP_SOURCE: (
                self.bssfp_read_gradient_axis,
                self.bssfp_phase_gradient_axis,
            ),
            self.SS_BSSFP_SOURCE: (
                self.ss_bssfp_read_gradient_axis,
                self.ss_bssfp_phase_gradient_axis,
            ),
            self.RADIAL_ME_BSSFP_SOURCE: (
                self.radial_me_read_gradient_axis,
                self.radial_me_phase_gradient_axis,
            ),
            self.ME_BSSFP_SOURCE: (
                self.me_bssfp_read_gradient_axis,
                self.me_bssfp_phase_gradient_axis,
            ),
        }
        try:
            read, phase = controls[source_index]
        except KeyError as exc:
            raise ValueError("Source has no three-dimensional encoding frame") from exc
        return EncodingFrame.from_read_phase_axes(
            str(read.currentData()), str(phase.currentData())
        )

    def _acquisition_changed(self, *_):
        self._update_bandwidth_labels()
        self._update_spoiling_quality()
        self._request_generated_sequence_refresh()

    def _readout_trajectory_changed(self, *_):
        spiral = self.epi_readout_trajectory.currentText() == "Spiral"
        self.epi_spiral_turns.setEnabled(spiral)
        self.pixel_bandwidth_info.setVisible(not spiral)
        label = self.acquisition_group.layout().labelForField(self.pixel_bandwidth_info)
        if label is not None:
            label.setVisible(not spiral)
        self._acquisition_changed()

    def _rf_pulse_type_changed(self, *_):
        self._update_shared_rf_controls("epi")
        self._acquisition_changed()

    def _epi_sinc_lobes_changed(self, *_):
        self._shared_rf_control_changed("epi", self._acquisition_changed)

    def _selected_rf_pulse_type(self) -> str:
        return self._selected_shared_rf_pulse_type("epi")

    def _epi_rf_parameters(self) -> dict:
        return self._shared_rf_parameters("epi")

    def _csi_changed(self, *_):
        self._update_csi_labels()
        self._update_spoiling_quality()
        if self.sequence_source.currentIndex() == 2:
            self._request_generated_sequence_refresh()

    def _flash_changed(self, *_):
        self._update_flash_labels()
        self._update_spoiling_quality()
        if self.sequence_source.currentIndex() == self.FLASH_SOURCE:
            self._request_generated_sequence_refresh()

    def _flash_auto_spoiler_toggled(self, enabled):
        self.flash_spoiler_cycles_per_slice.setDisabled(enabled)
        self.flash_spoiler_cycles_per_voxel.setDisabled(enabled)
        self._flash_changed()

    def _bssfp_changed(self, *_):
        self._update_bssfp_labels()
        self._update_spoiling_quality()
        if self.sequence_source.currentIndex() == 3:
            self._request_generated_sequence_refresh()

    def _ss_bssfp_changed(self, *_):
        self._update_ss_bssfp_labels()
        self._update_spoiling_quality()
        if self.sequence_source.currentIndex() == 4:
            self._request_generated_sequence_refresh()

    @staticmethod
    def _metabolite_key(name):
        normalized = "".join(
            character for character in str(name).lower() if character.isalnum()
        )
        if "lactate" in normalized or normalized.startswith("lac"):
            return "lactate"
        if (
            "pyruvate" in normalized
            or normalized.startswith("pyr")
            or normalized == "py"
        ):
            return "pyruvate"
        return normalized

    def _ss_bssfp_phantom_peak_offsets(self):
        phantom = self._selected_designed_phantom()
        if isinstance(phantom, SpectralPhantom):
            components = phantom.species
        elif isinstance(phantom, DynamicSpectralPhantom):
            components = phantom.pools
        else:
            return None
        by_key = {
            self._metabolite_key(component.name): phantom.get_frequency_offset(
                component.name,
                self.field_strength_t.value(),
                self.nucleus.currentText(),
            )
            for component in components
        }
        target_names = tuple(
            value.strip()
            for value in self.ss_bssfp_target_names.text().split(",")
            if value.strip()
        )
        if not target_names:
            return None
        offsets = []
        for target_name in target_names:
            key = self._metabolite_key(target_name)
            matches = [
                value
                for component_key, value in by_key.items()
                if key == component_key or key in component_key or component_key in key
            ]
            if len(matches) != 1:
                return None
            offsets.append(float(matches[0]))
        return tuple(offsets)

    def _phantom_voxel_sizes_xyz_m(self):
        source_index = self.object_source.currentIndex()
        if source_index == 1:
            return (
                self.fov_mm.value() / (1000.0 * self.matrix_size.value()),
                self.fov_mm.value() / (1000.0 * self.matrix_size.value()),
                self.fov_z_mm.value() / (1000.0 * self.z_matrix_size.value()),
            )
        if source_index == 2:
            return None
        phantom = self._selected_designed_phantom()
        if phantom is None:
            phantom = self.phantom
        if phantom is None or int(getattr(phantom, "ndim", 0)) != 3:
            return None
        affine = np.asarray(phantom.affine_ijk_to_xyz_m, dtype=float)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            return None
        extents = np.sum(np.abs(affine[:3, :3]), axis=1)
        if np.any(extents <= 0):
            return None
        return tuple(float(value) for value in extents)

    def _sync_ss_bssfp_frequencies_from_phantom(self):
        offsets = self._ss_bssfp_phantom_peak_offsets()
        print(f"Syncing SS-BSSFP frequencies from phantom: {offsets}")
        if offsets is None:
            return False
        text = ", ".join(f"{value:.9g}" for value in offsets)
        self.ss_bssfp_target_offsets_hz.setText(text)
        print(f"SS-BSSFP target offsets updated: {text}")
        self.ss_bssfp_receiver_offsets_hz.setText(text)
        return True

    def _ss_bssfp_reference_voxel_sizes_m(self):
        phantom_voxel_sizes = self._phantom_voxel_sizes_xyz_m()
        if phantom_voxel_sizes is not None:
            return phantom_voxel_sizes

        logical_sizes = np.asarray(self._ss_bssfp_fov_m(), dtype=float) / np.asarray(
            (
                self.ss_bssfp_read_matrix.value(),
                self.ss_bssfp_phase_matrix.value(),
                self.ss_bssfp_partition_matrix.value(),
            ),
            dtype=float,
        )
        frame = self._three_dimensional_encoding_frame(self.SS_BSSFP_SOURCE)
        physical_sizes = np.zeros(3, dtype=float)
        for role, size in zip(("read", "phase", "partition"), logical_sizes):
            axis, _ = frame.axis_and_sign(role)
            physical_sizes["xyz".index(axis)] = size
        return tuple(float(value) for value in physical_sizes)

    def _flash_reference_voxel_sizes_m(self):
        phantom_voxel_sizes = self._phantom_voxel_sizes_xyz_m()
        if phantom_voxel_sizes is not None:
            return phantom_voxel_sizes
        frame = self._two_dimensional_encoding_frame(self.FLASH_SOURCE)
        physical_sizes = np.zeros(3, dtype=float)
        for role, size in zip(
            ("read", "phase", "partition"),
            (
                self.flash_read_fov_mm.value()
                / (1000.0 * self.flash_read_matrix.value()),
                self.flash_phase_fov_mm.value()
                / (1000.0 * self.flash_phase_matrix.value()),
                self.flash_slice_thickness_mm.value() / 1000.0,
            ),
        ):
            axis, _ = frame.axis_and_sign(role)
            physical_sizes["xyz".index(axis)] = size
        return tuple(float(value) for value in physical_sizes)

    def _flash_auto_spoiler_values(self):
        """Return first-null FLASH spoiler strengths for the current geometry."""
        voxel_sizes = np.asarray(self._flash_reference_voxel_sizes_m(), dtype=float)
        frame = self._two_dimensional_encoding_frame(self.FLASH_SOURCE)
        role_reference_sizes = {
            "read": self.flash_read_fov_mm.value()
            / (1000.0 * self.flash_read_matrix.value()),
            "phase": self.flash_phase_fov_mm.value()
            / (1000.0 * self.flash_phase_matrix.value()),
            "partition": self.flash_slice_thickness_mm.value() / 1000.0,
        }

        targets = {}
        axis_indices = {}
        for role, reference_size in role_reference_sizes.items():
            axis, _ = frame.axis_and_sign(role)
            axis_index = "xyz".index(axis)
            axis_indices[role] = axis_index
            targets[role] = reference_size / voxel_sizes[axis_index]

        # FLASH exposes one shared in-plane value. Prefer an axis represented by
        # multiple subvoxel spins so its first null is also present on the
        # simulated midpoint grid, then choose the smaller required moment.
        in_plane_role = min(
            ("read", "phase"),
            key=lambda role: (
                self.subvoxel_spin_counts[axis_indices[role]] <= 1,
                targets[role],
            ),
        )
        return targets["partition"], targets[in_plane_role]

    def _apply_flash_auto_spoilers(self):
        if not self.flash_auto_spoiler.isChecked():
            return False
        slice_cycles, in_plane_cycles = self._flash_auto_spoiler_values()
        changed = False
        clipped = False
        for control, value in (
            (self.flash_spoiler_cycles_per_slice, slice_cycles),
            (self.flash_spoiler_cycles_per_voxel, in_plane_cycles),
        ):
            bounded = min(control.maximum(), max(control.minimum(), float(value)))
            clipped = clipped or not np.isclose(
                float(value), bounded, rtol=0.0, atol=5e-5
            )
            if not np.isclose(control.value(), bounded, rtol=0.0, atol=5e-5):
                previous = control.blockSignals(True)
                control.setValue(bounded)
                control.blockSignals(previous)
                changed = True
        self._flash_auto_spoiler_clipped = clipped
        return changed

    def _update_flash_spoiler_info(self):
        voxel_sizes = np.asarray(self._flash_reference_voxel_sizes_m())
        frame = self._two_dimensional_encoding_frame(self.FLASH_SOURCE)
        cycles_xyz = np.zeros(3, dtype=float)
        role_reference_sizes = (
            self.flash_read_fov_mm.value() / (1000.0 * self.flash_read_matrix.value()),
            self.flash_phase_fov_mm.value()
            / (1000.0 * self.flash_phase_matrix.value()),
            self.flash_slice_thickness_mm.value() / 1000.0,
        )
        role_cycles = (
            self.flash_spoiler_cycles_per_voxel.value(),
            self.flash_spoiler_cycles_per_voxel.value(),
            self.flash_spoiler_cycles_per_slice.value(),
        )
        for role, cycles, reference_size in zip(
            ("read", "phase", "partition"), role_cycles, role_reference_sizes
        ):
            axis, _ = frame.axis_and_sign(role)
            axis_index = "xyz".index(axis)
            cycles_xyz[axis_index] = cycles * voxel_sizes[axis_index] / reference_size
        counts = tuple(int(value) for value in self.subvoxel_spin_counts)
        excitation_count = (
            self.flash_phase_matrix.value()
            * self.flash_slice_count.value()
            * self.flash_repetitions.value()
        )
        analyzed_excitation_count = min(excitation_count, 1024)
        sampling = SpinSampling(counts, method=self.subvoxel_sampling_method)
        report = analyze_repeated_spoiler_train(
            tuple(float(value) for value in cycles_xyz),
            sampling,
            analyzed_excitation_count,
        )
        report_source = "configured repeated crusher"
        phase_train_for_recommendation = None
        if (
            not self._sequence_generation_pending
            and self._generated_sequence_source_index == self.FLASH_SOURCE
            and self.acquisition is not None
        ):
            selected_phantom = self._selected_designed_phantom()
            moment_origins = np.asarray(
                self.acquisition.moment_origins_cyc_per_m, dtype=float
            )
            if selected_phantom is not None and moment_origins.shape[0] >= 2:
                limited_moments = moment_origins[: min(moment_origins.shape[0], 1025)]
                report = analyze_adc_moment_train(
                    limited_moments,
                    phantom_voxel_basis_m(selected_phantom),
                    sampling,
                )
                phase_train_for_recommendation = np.asarray(
                    report.phase_cycles_per_voxel, dtype=float
                )
                report_source = "actual ADC moment origins"

        minimum_counts = [1, 1, 1]
        for role, reference_size in zip(
            ("read", "phase", "partition"), role_reference_sizes
        ):
            axis, _ = frame.axis_and_sign(role)
            axis_index = "xyz".index(axis)
            minimum_counts[axis_index] = max(
                1, int(np.ceil(voxel_sizes[axis_index] / reference_size - 1e-12))
            )
        # A centered singleton cannot resolve gradient phase or slice-selective
        # RF evolution along that axis, even when another axis happens to keep
        # the product coherence near zero. Require at least two points on every
        # axis that carries a crusher moment before optimizing train recurrence.
        for axis_index, cycles in enumerate(cycles_xyz):
            if abs(cycles) > 1e-12:
                minimum_counts[axis_index] = max(2, minimum_counts[axis_index])
        maximum_spins = max(512, int(np.prod(minimum_counts)))
        if phase_train_for_recommendation is None:
            recommendation = recommend_spin_grid(
                tuple(float(np.round(value, 12)) for value in cycles_xyz),
                analyzed_excitation_count,
                tuple(minimum_counts),
                "midpoint",
                0.01,
                maximum_spins,
                max(32, max(minimum_counts)),
            )
        else:
            recommendation = recommend_spin_grid_for_phase_train(
                phase_train_for_recommendation,
                tuple(minimum_counts),
                "midpoint",
                0.01,
                maximum_spins,
                max(32, max(minimum_counts)),
            )
        self._flash_spin_grid_recommendation = recommendation
        recommendation_counts = recommendation.counts_xyz
        recommended_text = (
            f"{recommendation_counts[0]}×{recommendation_counts[1]}"
            f"×{recommendation_counts[2]} regular midpoint spins "
            f"({recommendation.spins_per_voxel} total; train error "
            f"{100 * recommendation.maximum_sampling_error:.3g}%)"
        )
        message = (
            "Effective cycles/phantom voxel XYZ: "
            + ", ".join(f"{value:.4g}" for value in cycles_xyz)
            + ". Remaining coherent signal after one crusher: "
            + f"ideal {100 * report.single_continuous_coherence:.3g}%, "
            + f"{counts[0]}×{counts[1]}×{counts[2]} "
            + f"{self.subvoxel_sampling_method} grid "
            + f"{100 * report.single_sampled_coherence:.3g}%. "
            + f"Across {max(1, report.n_observations)} accumulated "
            + f"orders ({report_source}), maximum sampling error is "
            + f"{100 * report.maximum_sampling_error:.3g}%"
        )
        if report.worst_error_observation is not None:
            message += f" at order {report.worst_error_observation}."
        else:
            message += "."
        warning = (
            report.single_continuous_coherence > 0.05
            or report.maximum_sampling_error > 0.05
        )
        if report.first_alias_observation is not None:
            message += (
                " Warning: artificial subvoxel rephasing starts at crusher "
                f"order {report.first_alias_observation}."
            )
        if report.n_observations < excitation_count - 1:
            message += (
                f" The GUI check is limited to the first "
                f"{report.n_observations} orders of {excitation_count - 1}."
            )
        message += f" Recommended: {recommended_text}."
        if not recommendation.meets_target:
            message += " No tested grid reached the 1% train-error target."
        if self.flash_auto_spoiler.isChecked() and getattr(
            self, "_flash_auto_spoiler_clipped", False
        ):
            message += " Auto spoiler reached the control limit before its target."
        elif self.flash_auto_spoiler.isChecked():
            message += " Auto spoiler targets the first coherence null."
        self._flash_spoiling_quality_rows = [
            ("Sequence", "FLASH (2D)"),
            (
                "Cycles / phantom voxel XYZ",
                ", ".join(f"{value:.4g}" for value in cycles_xyz),
            ),
            (
                "Current subvoxel grid",
                f"{counts[0]}×{counts[1]}×{counts[2]} "
                f"{self.subvoxel_sampling_method}",
            ),
            (
                "Ideal coherence after one crusher",
                f"{100 * report.single_continuous_coherence:.3g}%",
            ),
            (
                "Sampled coherence after one crusher",
                f"{100 * report.single_sampled_coherence:.3g}%",
            ),
            (
                "Maximum train sampling error",
                f"{100 * report.maximum_sampling_error:.3g}%",
            ),
            ("Evaluation source", report_source),
            (
                "Recommended grid",
                f"{recommendation_counts[0]}×{recommendation_counts[1]}×"
                f"{recommendation_counts[2]} midpoint "
                f"({recommendation.spins_per_voxel} spins; "
                f"{100 * recommendation.maximum_sampling_error:.3g}% error)",
            ),
        ]
        if report.first_alias_observation is not None:
            self._flash_spoiling_quality_rows.append(
                (
                    "First artificial rephasing",
                    f"Order {report.first_alias_observation}",
                )
            )
        self.flash_spoiler_info.setText(message)
        self.flash_spoiler_info.setStyleSheet(
            "color: #b45309;" if warning else "color: #15803d;"
        )
        recommendation_is_current = (
            self.subvoxel_sampling_method == recommendation.method
            and counts == recommendation.counts_xyz
        )
        self.flash_apply_recommended_grid.setEnabled(not recommendation_is_current)
        self.flash_apply_recommended_grid.setText(
            "Grid already train-safe"
            if recommendation_is_current
            else f"Apply {recommendation_counts[0]}×{recommendation_counts[1]}"
            f"×{recommendation_counts[2]} midpoint grid"
        )
        if (
            hasattr(self, "spoiling_quality_info")
            and self.sequence_source.currentIndex() == self.FLASH_SOURCE
        ):
            self._sync_flash_spoiling_quality()

    def _apply_flash_recommended_spin_grid(self):
        recommendation = getattr(self, "_flash_spin_grid_recommendation", None)
        if recommendation is None:
            return
        if self.app_settings is not None:
            self.app_settings.setValue(
                "sequence/subvoxel_sampling_method", recommendation.method
            )
            for axis, count in zip("xyz", recommendation.counts_xyz):
                self.app_settings.setValue(
                    f"sequence/subvoxel_spins_{axis}", int(count)
                )
            self.app_settings.sync()
        self.set_spoiler_configuration(
            self.spoiler_mode, recommendation.counts_xyz, recommendation.method
        )

    def _apply_current_recommended_spin_grid(self):
        recommendation = getattr(self, "_spoiling_spin_grid_recommendation", None)
        if recommendation is None:
            return
        if self.app_settings is not None:
            self.app_settings.setValue(
                "sequence/subvoxel_sampling_method", recommendation.method
            )
            for axis, count in zip("xyz", recommendation.counts_xyz):
                self.app_settings.setValue(
                    f"sequence/subvoxel_spins_{axis}", int(count)
                )
            self.app_settings.sync()
        self.set_spoiler_configuration(
            self.spoiler_mode, recommendation.counts_xyz, recommendation.method
        )

    def _two_dimensional_spoiler_spec(self, source_index):
        if source_index == self.EPI_SOURCE:
            enabled = self.epi_spoil_after_slice.isChecked()
            role_reference_sizes = (
                self.epi_read_fov_mm.value() / (1000.0 * self.read_matrix.value()),
                self.epi_phase_fov_mm.value() / (1000.0 * self.phase_matrix.value()),
                self.epi_slice_thickness_mm.value() / 1000.0,
            )
            role_cycles = (
                self.epi_spoiler_cycles_per_voxel.value(),
                self.epi_spoiler_cycles_per_voxel.value(),
                self.epi_spoiler_cycles_per_slice.value(),
            )
            excitation_count = (
                self.epi_slice_count.value() * self.epi_repetitions.value()
            )
            name = "EPI / spiral end-of-slice spoiler"
        elif source_index == self.CSI_SOURCE:
            enabled = self.csi_spoil_after_readout.isChecked()
            role_reference_sizes = (
                self.csi_read_fov_mm.value() / (1000.0 * self.csi_read_matrix.value()),
                self.csi_phase_fov_mm.value()
                / (1000.0 * self.csi_phase_matrix.value()),
                self.csi_slice_thickness_mm.value() / 1000.0,
            )
            role_cycles = (
                self.csi_spoiler_cycles_per_voxel.value(),
                self.csi_spoiler_cycles_per_voxel.value(),
                self.csi_spoiler_cycles_per_slice.value(),
            )
            excitation_count = (
                self.csi_read_matrix.value()
                * self.csi_phase_matrix.value()
                * self.csi_repetitions.value()
            )
            name = "CSI end-of-FID spoiler"
        else:
            raise ValueError("Source has no two-dimensional spoiler configuration")

        frame = self._two_dimensional_encoding_frame(source_index)
        phantom_voxel_sizes = self._phantom_voxel_sizes_xyz_m()
        if phantom_voxel_sizes is None:
            voxel_sizes = np.zeros(3, dtype=float)
            for role, size in zip(("read", "phase", "partition"), role_reference_sizes):
                axis, _ = frame.axis_and_sign(role)
                voxel_sizes["xyz".index(axis)] = size
        else:
            voxel_sizes = np.asarray(phantom_voxel_sizes, dtype=float)

        cycles_xyz = np.zeros(3, dtype=float)
        minimum_counts = [1, 1, 1]
        for role, cycles, reference_size in zip(
            ("read", "phase", "partition"), role_cycles, role_reference_sizes
        ):
            axis, _ = frame.axis_and_sign(role)
            axis_index = "xyz".index(axis)
            cycles_xyz[axis_index] = (
                cycles * voxel_sizes[axis_index] / reference_size if enabled else 0.0
            )
            minimum_counts[axis_index] = max(
                1,
                int(np.ceil(voxel_sizes[axis_index] / reference_size - 1e-12)),
            )
        return {
            "name": name,
            "enabled": enabled,
            "cycles_xyz": cycles_xyz,
            "minimum_counts": minimum_counts,
            "excitation_count": excitation_count,
        }

    def _ss_bssfp_spoiler_spec(self):
        voxel_sizes = np.asarray(self._ss_bssfp_reference_voxel_sizes_m())
        frame = self._three_dimensional_encoding_frame(self.SS_BSSFP_SOURCE)
        minimum_counts = [1, 1, 1]
        for role, role_fov, matrix_size in zip(
            ("read", "phase", "partition"),
            self._ss_bssfp_fov_m(),
            (
                self.ss_bssfp_read_matrix.value(),
                self.ss_bssfp_phase_matrix.value(),
                self.ss_bssfp_partition_matrix.value(),
            ),
        ):
            axis, _ = frame.axis_and_sign(role)
            axis_index = "xyz".index(axis)
            reference_size = role_fov / matrix_size
            minimum_counts[axis_index] = max(
                1,
                int(np.ceil(voxel_sizes[axis_index] / reference_size - 1e-12)),
            )
        cycles_xyz = self._ss_bssfp_effective_spoiler_cycles_xyz()
        return {
            "name": "Spectral-spatial bSSFP end-of-volume spoiler",
            "enabled": bool(np.any(np.abs(cycles_xyz) > 1e-12)),
            "cycles_xyz": cycles_xyz,
            "minimum_counts": minimum_counts,
            "excitation_count": self.ss_bssfp_repetitions.value(),
        }

    def _set_spoiling_quality_table(self, rows, *, warning=False):
        """Render spoiling quality as a rich table in the info-button tooltip."""
        if not hasattr(self, "spoiling_quality_button"):
            return
        color = "#b45309" if warning else "#15803d"
        status = "Review recommended" if warning else "Quality available"
        body = "".join(
            "<tr>"
            f"<th align='left' valign='top'>{html.escape(str(parameter))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
            for parameter, value in rows
        )
        self.spoiling_quality_button.setToolTip(
            "<div style='white-space:normal'>"
            "<b>Spoiling quality</b><br><br>"
            "<table cellspacing='0' cellpadding='4'>"
            f"{body}"
            "</table></div>"
        )
        self.spoiling_quality_button.setStyleSheet(f"color: {color};")
        self.spoiling_quality_status.setText(status)
        self.spoiling_quality_status.setStyleSheet(f"color: {color};")

    def _set_spoiling_quality_unavailable(self, message, warning=False):
        if not hasattr(self, "spoiling_quality_info"):
            return
        self._spoiling_spin_grid_recommendation = None
        self.spoiling_quality_info.setText(message)
        self.spoiling_quality_info.setStyleSheet(
            "color: #b45309;" if warning else "color: #475569;"
        )
        self._set_spoiling_quality_table(
            (("Status", "Warning" if warning else "Information"), ("Details", message)),
            warning=warning,
        )
        self.spoiling_apply_recommended_grid.setEnabled(False)
        self.spoiling_apply_recommended_grid.setVisible(False)

    def _set_configured_spoiling_quality(self, spec):
        cycles_xyz = np.asarray(spec["cycles_xyz"], dtype=float)
        counts = tuple(int(value) for value in self.subvoxel_spin_counts)
        if not spec["enabled"] or not np.any(np.abs(cycles_xyz) > 1e-12):
            self._set_spoiling_quality_unavailable(
                f"{spec['name']}: no gradient spoiler is active. Expected "
                "remaining coherent signal is 100% for this spoiler unit.",
                warning=True,
            )
            return

        minimum_counts = [int(value) for value in spec["minimum_counts"]]
        for axis_index, cycles in enumerate(cycles_xyz):
            if abs(cycles) > 1e-12:
                minimum_counts[axis_index] = max(2, minimum_counts[axis_index])
        excitation_count = max(1, int(spec["excitation_count"]))
        analyzed_excitation_count = min(excitation_count, 1024)
        sampling = SpinSampling(counts, method=self.subvoxel_sampling_method)
        report = analyze_repeated_spoiler_train(
            tuple(float(value) for value in cycles_xyz),
            sampling,
            analyzed_excitation_count,
        )
        maximum_spins = max(512, int(np.prod(minimum_counts)))
        recommendation = recommend_spin_grid(
            tuple(float(np.round(value, 12)) for value in cycles_xyz),
            analyzed_excitation_count,
            tuple(minimum_counts),
            "midpoint",
            0.01,
            maximum_spins,
            max(32, max(minimum_counts)),
        )
        self._spoiling_spin_grid_recommendation = recommendation
        recommendation_counts = recommendation.counts_xyz
        message = (
            f"{spec['name']}. Effective cycles/phantom voxel XYZ: "
            + ", ".join(f"{value:.4g}" for value in cycles_xyz)
            + ". Remaining coherent signal after one spoiler: "
            + f"ideal {100 * report.single_continuous_coherence:.3g}%, "
            + f"{counts[0]}×{counts[1]}×{counts[2]} "
            + f"{self.subvoxel_sampling_method} grid "
            + f"{100 * report.single_sampled_coherence:.3g}%. "
            + f"Across {max(1, report.n_observations)} accumulated spoiler "
            + "orders, maximum sampling error is "
            + f"{100 * report.maximum_sampling_error:.3g}%."
        )
        if report.first_alias_observation is not None:
            message += (
                " Warning: artificial subvoxel rephasing starts at spoiler "
                f"order {report.first_alias_observation}."
            )
        if report.n_observations < excitation_count - 1:
            message += (
                f" The check is limited to the first {report.n_observations} "
                f"orders of {excitation_count - 1}."
            )
        message += (
            f" Recommended: {recommendation_counts[0]}×"
            f"{recommendation_counts[1]}×{recommendation_counts[2]} regular "
            f"midpoint spins ({recommendation.spins_per_voxel} total; train "
            f"error {100 * recommendation.maximum_sampling_error:.3g}%)."
        )
        if not recommendation.meets_target:
            message += " No tested grid reached the 1% train-error target."
        warning = (
            report.single_continuous_coherence > 0.05
            or report.maximum_sampling_error > 0.05
        )
        self.spoiling_quality_info.setText(message)
        self.spoiling_quality_info.setStyleSheet(
            "color: #b45309;" if warning else "color: #15803d;"
        )
        quality_rows = [
            ("Sequence", spec["name"]),
            (
                "Cycles / phantom voxel XYZ",
                ", ".join(f"{value:.4g}" for value in cycles_xyz),
            ),
            (
                "Current subvoxel grid",
                f"{counts[0]}×{counts[1]}×{counts[2]} {self.subvoxel_sampling_method}",
            ),
            (
                "Ideal coherence after one spoiler",
                f"{100 * report.single_continuous_coherence:.3g}%",
            ),
            (
                "Sampled coherence after one spoiler",
                f"{100 * report.single_sampled_coherence:.3g}%",
            ),
            (
                "Maximum train sampling error",
                f"{100 * report.maximum_sampling_error:.3g}%",
            ),
            ("Evaluated spoiler orders", str(max(1, report.n_observations))),
            (
                "Recommended grid",
                f"{recommendation_counts[0]}×{recommendation_counts[1]}×"
                f"{recommendation_counts[2]} midpoint "
                f"({recommendation.spins_per_voxel} spins; "
                f"{100 * recommendation.maximum_sampling_error:.3g}% error)",
            ),
        ]
        if report.first_alias_observation is not None:
            quality_rows.append(
                (
                    "First artificial rephasing",
                    f"Order {report.first_alias_observation}",
                )
            )
        if not recommendation.meets_target:
            quality_rows.append(("1% target", "No tested grid reached the target"))
        self._set_spoiling_quality_table(quality_rows, warning=warning)
        recommendation_is_current = (
            self.subvoxel_sampling_method == recommendation.method
            and counts == recommendation.counts_xyz
        )
        self.spoiling_apply_recommended_grid.setVisible(True)
        self.spoiling_apply_recommended_grid.setEnabled(not recommendation_is_current)
        self.spoiling_apply_recommended_grid.setText(
            "Grid already train-safe"
            if recommendation_is_current
            else f"Apply {recommendation_counts[0]}×{recommendation_counts[1]}"
            f"×{recommendation_counts[2]} midpoint grid"
        )

    def _sync_flash_spoiling_quality(self):
        if not hasattr(self, "spoiling_quality_info"):
            return
        self._spoiling_spin_grid_recommendation = getattr(
            self, "_flash_spin_grid_recommendation", None
        )
        self.spoiling_quality_info.setText(
            "FLASH (2D). " + self.flash_spoiler_info.text()
        )
        self.spoiling_quality_info.setStyleSheet(self.flash_spoiler_info.styleSheet())
        flash_rows = getattr(
            self,
            "_flash_spoiling_quality_rows",
            (("Sequence", "FLASH (2D)"), ("Details", self.flash_spoiler_info.text())),
        )
        self._set_spoiling_quality_table(
            flash_rows,
            warning="#b45309" in self.flash_spoiler_info.styleSheet(),
        )
        self.spoiling_apply_recommended_grid.setVisible(True)
        self.spoiling_apply_recommended_grid.setEnabled(
            self.flash_apply_recommended_grid.isEnabled()
        )
        self.spoiling_apply_recommended_grid.setText(
            self.flash_apply_recommended_grid.text()
        )

    def _set_imported_pulseq_spoiling_quality(self):
        acquisition = next(
            (
                candidate
                for candidate in (
                    self.acquisition,
                    self.spectroscopic_acquisition,
                    self.spiral_acquisition,
                )
                if candidate is not None
                and hasattr(candidate, "moment_origins_cyc_per_m")
            ),
            None,
        )
        phantom = None
        voxel_basis = None
        if self.object_source.currentIndex() == 0:
            phantom = self._selected_designed_phantom()
            if phantom is None:
                phantom = self.phantom
        elif self.object_source.currentIndex() == 1:
            voxel_sizes = self._phantom_voxel_sizes_xyz_m()
            if voxel_sizes is not None:
                voxel_basis = np.diag(np.asarray(voxel_sizes, dtype=float))
        if phantom is not None:
            voxel_basis = phantom_voxel_basis_m(phantom)
        if acquisition is None or voxel_basis is None:
            self._set_spoiling_quality_unavailable(
                "Imported Pulseq sequence: select a 3D phantom and load a "
                "sequence with inferable ADC positions to estimate spoiling quality."
            )
            return
        moment_origins = np.asarray(acquisition.moment_origins_cyc_per_m, dtype=float)
        if moment_origins.ndim != 2 or moment_origins.shape[0] < 2:
            self._set_spoiling_quality_unavailable(
                "Imported Pulseq sequence: at least two inferable ADC moment "
                "origins are required to estimate spoiling quality."
            )
            return
        limited_moments = moment_origins[: min(moment_origins.shape[0], 1025)]
        sampling = SpinSampling(
            tuple(int(value) for value in self.subvoxel_spin_counts),
            method=self.subvoxel_sampling_method,
        )
        report = analyze_adc_moment_train(
            limited_moments,
            voxel_basis,
            sampling,
        )
        phase_train = np.asarray(report.phase_cycles_per_voxel, dtype=float)
        minimum_counts = tuple(
            2 if np.any(np.abs(phase_train[:, axis]) > 1e-12) else 1
            for axis in range(3)
        )
        recommendation = recommend_spin_grid_for_phase_train(
            phase_train,
            minimum_counts,
            "midpoint",
            0.01,
            max(512, int(np.prod(minimum_counts))),
            max(32, max(minimum_counts)),
        )
        self._spoiling_spin_grid_recommendation = recommendation
        counts = sampling.counts_xyz
        ideal_maximum = max(report.continuous_coherence, default=1.0)
        sampled_maximum = max(report.sampled_coherence, default=1.0)
        message = (
            f"Imported Pulseq ADC moment train: {report.n_observations} moment "
            f"offsets were evaluated on the selected phantom. Maximum remaining "
            f"coherent signal is ideal {100 * ideal_maximum:.3g}% and "
            f"{counts[0]}×{counts[1]}×{counts[2]} "
            f"{self.subvoxel_sampling_method} grid "
            f"{100 * sampled_maximum:.3g}%; maximum sampling error is "
            f"{100 * report.maximum_sampling_error:.3g}%. Recommended: "
            f"{recommendation.counts_xyz[0]}×{recommendation.counts_xyz[1]}×"
            f"{recommendation.counts_xyz[2]} regular midpoint spins "
            f"({recommendation.spins_per_voxel} total; train error "
            f"{100 * recommendation.maximum_sampling_error:.3g}%)."
        )
        if moment_origins.shape[0] > limited_moments.shape[0]:
            message += (
                f" The check is limited to the first {limited_moments.shape[0]} "
                f"ADC origins of {moment_origins.shape[0]}."
            )
        if report.first_alias_observation is not None:
            message += (
                " Warning: artificial subvoxel rephasing occurs at ADC moment "
                f"offset {report.first_alias_observation}."
            )
        warning = ideal_maximum > 0.05 or report.maximum_sampling_error > 0.05
        self.spoiling_quality_info.setText(message)
        self.spoiling_quality_info.setStyleSheet(
            "color: #b45309;" if warning else "color: #15803d;"
        )
        quality_rows = [
            ("Sequence", "Imported Pulseq ADC moment train"),
            ("Evaluated ADC moment offsets", str(report.n_observations)),
            (
                "Current subvoxel grid",
                f"{counts[0]}×{counts[1]}×{counts[2]} {self.subvoxel_sampling_method}",
            ),
            ("Maximum ideal coherence", f"{100 * ideal_maximum:.3g}%"),
            ("Maximum sampled coherence", f"{100 * sampled_maximum:.3g}%"),
            (
                "Maximum train sampling error",
                f"{100 * report.maximum_sampling_error:.3g}%",
            ),
            (
                "Recommended grid",
                f"{recommendation.counts_xyz[0]}×{recommendation.counts_xyz[1]}×"
                f"{recommendation.counts_xyz[2]} midpoint "
                f"({recommendation.spins_per_voxel} spins; "
                f"{100 * recommendation.maximum_sampling_error:.3g}% error)",
            ),
        ]
        if report.first_alias_observation is not None:
            quality_rows.append(
                (
                    "First artificial rephasing",
                    f"ADC offset {report.first_alias_observation}",
                )
            )
        self._set_spoiling_quality_table(quality_rows, warning=warning)
        recommendation_is_current = (
            self.subvoxel_sampling_method == recommendation.method
            and sampling.counts_xyz == recommendation.counts_xyz
        )
        self.spoiling_apply_recommended_grid.setVisible(True)
        self.spoiling_apply_recommended_grid.setEnabled(not recommendation_is_current)
        self.spoiling_apply_recommended_grid.setText(
            "Grid already train-safe"
            if recommendation_is_current
            else f"Apply {recommendation.counts_xyz[0]}×"
            f"{recommendation.counts_xyz[1]}×{recommendation.counts_xyz[2]} "
            "midpoint grid"
        )

    def _update_spoiling_quality(self, *_):
        if not hasattr(self, "spoiling_quality_info"):
            return
        source_index = self.sequence_source.currentIndex()
        if source_index in {self.EPI_SOURCE, self.CSI_SOURCE}:
            self._set_configured_spoiling_quality(
                self._two_dimensional_spoiler_spec(source_index)
            )
        elif source_index == self.FLASH_SOURCE:
            self._sync_flash_spoiling_quality()
        elif source_index == self.SS_BSSFP_SOURCE:
            self._set_configured_spoiling_quality(self._ss_bssfp_spoiler_spec())
        elif source_index in {
            self.BSSFP_SOURCE,
            self.RADIAL_ME_BSSFP_SOURCE,
            self.ME_BSSFP_SOURCE,
        }:
            self._set_spoiling_quality_unavailable(
                "Effective net spoiler: 0 cycles/voxel. Expected remaining "
                "coherent signal is 100% because this sequence is fully balanced "
                "during acquisition and intentionally retains transverse coherence."
            )
        elif source_index == self.INTERNAL_SOURCE:
            self._set_spoiling_quality_unavailable(
                "Internal FID has no gradient spoiler (0 cycles/voxel). Expected "
                "remaining coherent signal from gradient spoiling is 100%."
            )
        else:
            self._set_imported_pulseq_spoiling_quality()

    def _ss_bssfp_effective_spoiler_cycles_xyz(self):
        voxel_sizes = np.asarray(self._ss_bssfp_reference_voxel_sizes_m())
        cycles_xyz = np.full(
            3, self.ss_bssfp_spoiler_cycles_per_voxel.value(), dtype=float
        )
        fov_cycles = self.ss_bssfp_spoiler_cycles.value()
        frame = self._three_dimensional_encoding_frame(self.SS_BSSFP_SOURCE)
        for role, role_fov in zip(
            ("read", "phase", "partition"), self._ss_bssfp_fov_m()
        ):
            axis, _ = frame.axis_and_sign(role)
            axis_index = "xyz".index(axis)
            cycles_xyz[axis_index] += fov_cycles * voxel_sizes[axis_index] / role_fov
        return cycles_xyz

    def _update_ss_bssfp_spoiler_info(self):
        voxel_sizes = np.asarray(self._ss_bssfp_reference_voxel_sizes_m())
        cycles_xyz = self._ss_bssfp_effective_spoiler_cycles_xyz()
        continuous = float(np.prod(np.abs(np.sinc(cycles_xyz))))
        counts = tuple(int(value) for value in self.subvoxel_spin_counts)
        discrete = 1.0
        for cycles, count in zip(cycles_xyz, counts):
            offsets = (np.arange(count, dtype=float) + 0.5) / count - 0.5
            discrete *= abs(np.mean(np.exp(2j * np.pi * cycles * offsets)))

        messages = [
            "Voxel "
            + " × ".join(f"{value * 1e3:.4g}" for value in voxel_sizes)
            + " mm; effective cycles/voxel "
            + ", ".join(f"{value:.4g}" for value in cycles_xyz)
            + f". Remaining coherent signal: ideal {100 * continuous:.3g}%, "
            + f"{counts[0]}×{counts[1]}×{counts[2]} grid {100 * discrete:.3g}%."
        ]
        warning = continuous > 0.05 or discrete > 0.05
        if continuous < 0.01 and discrete > 0.05:
            messages.append(
                "Warning: this regular subvoxel grid aliases the selected spoiler."
            )

        peak_offsets = self._ss_bssfp_phantom_peak_offsets()
        try:
            receiver_offsets = self._comma_separated_floats(
                self.ss_bssfp_receiver_offsets_hz.text(), "Receiver offsets"
            )
        except ValueError:
            receiver_offsets = ()
        if peak_offsets is None:
            messages.append("Peak matching unavailable for the selected phantom.")
        elif len(receiver_offsets) == len(peak_offsets):
            detuning = np.asarray(receiver_offsets) - np.asarray(peak_offsets)
            if np.max(np.abs(detuning), initial=0.0) > 1e-6:
                warning = True
                messages.append(
                    "Receiver/phantom mismatch: "
                    + ", ".join(f"{value:+.4g} Hz" for value in detuning)
                    + "."
                )
            else:
                messages.append("Receiver offsets match the named phantom peaks.")
        self.ss_bssfp_spoiler_info.setText(" ".join(messages))
        self.ss_bssfp_spoiler_info.setStyleSheet(
            "color: #b45309;" if warning else "color: #15803d;"
        )
        if (
            hasattr(self, "spoiling_quality_info")
            and self.sequence_source.currentIndex() == self.SS_BSSFP_SOURCE
        ):
            self._update_spoiling_quality()

    def _radial_me_bssfp_changed(self, *_):
        self._update_radial_me_bssfp_labels()
        self._update_spoiling_quality()
        if self.sequence_source.currentIndex() == 5:
            self._request_generated_sequence_refresh()

    def _me_bssfp_changed(self, *_):
        self._update_me_bssfp_labels()
        self._update_spoiling_quality()
        if self.sequence_source.currentIndex() == 6:
            self._request_generated_sequence_refresh()

    def _frequency_reference_changed(self, *_):
        self._update_frequency_reference_info()
        generated_sequence = self._generated_pulseq_sequence
        if generated_sequence is not None:
            self._apply_workspace_frequency_reference(generated_sequence)
            if self.program is not None:
                definitions = self.program.metadata.setdefault("definitions", {})
                definitions["FieldStrengthT"] = self.field_strength_t.value()
                definitions["Nucleus"] = self.nucleus.currentText()
        if self.sequence_source.currentIndex() in {4, 5, 6}:
            self._request_generated_sequence_refresh()

    def _ernst_angle_context(self, repetition_time_s):
        """Resolve the representative phantom T1 used by one global RF pulse."""
        source_index = self.object_source.currentIndex()
        if source_index == 1:
            t1_values = np.asarray([self.t1_ms.value() / 1000.0], dtype=float)
            weights = np.ones(1, dtype=float)
            source = "built-in phantom"
        elif source_index == 2:
            t1_values = np.asarray([self.probe_t1_ms.value() / 1000.0], dtype=float)
            weights = np.ones(1, dtype=float)
            source = "spin probe"
        else:
            phantom = self._selected_designed_phantom()
            if phantom is None:
                return None
            mask = np.asarray(phantom.mask, dtype=bool)
            t1_map = np.asarray(phantom.t1_map, dtype=float)
            if t1_map.shape != mask.shape:
                return None
            t1_values = t1_map[mask]
            pd_map = getattr(phantom, "pd_map", None)
            if pd_map is None or np.asarray(pd_map).shape != mask.shape:
                weights = np.ones(t1_values.shape, dtype=float)
            else:
                weights = np.maximum(np.asarray(pd_map, dtype=float)[mask], 0.0)
            source = str(getattr(phantom, "name", "selected phantom"))
        valid = np.isfinite(t1_values) & (t1_values > 0.0)
        t1_values = t1_values[valid]
        weights = np.asarray(weights, dtype=float)[valid]
        if t1_values.size == 0:
            return None
        if not np.any(np.isfinite(weights) & (weights > 0.0)):
            weights = np.ones(t1_values.shape, dtype=float)
        else:
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        effective_t1_s = float(np.average(t1_values, weights=weights))
        angles = np.asarray(ernst_angle_deg(repetition_time_s, t1_values), dtype=float)
        return _ErnstAngleContext(
            angle_deg=float(ernst_angle_deg(repetition_time_s, effective_t1_s)),
            angle_range_deg=(float(np.min(angles)), float(np.max(angles))),
            repetition_time_s=float(repetition_time_s),
            effective_t1_s=effective_t1_s,
            t1_range_s=(float(np.min(t1_values)), float(np.max(t1_values))),
            source=source,
        )

    def _update_ernst_controls(self, prefix, repetition_time_ms, *, has_vfa=False):
        use_control = getattr(self, f"{prefix}_use_ernst_angle")
        rf_spoiling = getattr(self, f"{prefix}_rf_spoiling")
        increment = getattr(self, f"{prefix}_rf_spoiling_increment_deg")
        flip_control = getattr(self, f"{prefix}_flip_angle_deg")
        info = getattr(self, f"{prefix}_ernst_info")
        variable_control = (
            getattr(self, f"{prefix}_variable_flip_angle") if has_vfa else None
        )
        variable_enabled = bool(
            variable_control is not None and variable_control.isChecked()
        )
        context = self._ernst_angle_context(float(repetition_time_ms) / 1000.0)
        rf_requested = rf_spoiling.isChecked()
        can_use = not variable_enabled and context is not None
        if use_control.isChecked() and not can_use:
            previous = use_control.blockSignals(True)
            use_control.setChecked(False)
            use_control.blockSignals(previous)
        use_control.setEnabled(can_use)
        increment.setEnabled(rf_requested)
        use_ernst = can_use and use_control.isChecked()
        flip_control.setEnabled(not variable_enabled and not use_ernst)
        if variable_control is not None:
            variable_control.setEnabled(not use_ernst)
            getattr(self, f"{prefix}_vfa_final_flip_angle_deg").setEnabled(
                variable_enabled
            )
        if variable_enabled:
            info.setText("Unavailable while variable flip angle is enabled.")
        elif context is None:
            info.setText("Unavailable — the selected simulation object has no T1.")
        else:
            t1_min, t1_max = (1000.0 * value for value in context.t1_range_s)
            angle_min, angle_max = context.angle_range_deg
            range_text = ""
            if not np.isclose(t1_min, t1_max):
                range_text = (
                    f"; T1 range {t1_min:.4g}–{t1_max:.4g} ms gives "
                    f"{angle_min:.4g}–{angle_max:.4g}°"
                )
            info.setText(
                f"{context.angle_deg:.4g}° from TR "
                f"{1000.0 * context.repetition_time_s:.4g} ms and "
                f"effective T1 {1000.0 * context.effective_t1_s:.4g} ms "
                f"({context.source}){range_text}. T2 is not used."
            )
            if use_ernst:
                previous = flip_control.blockSignals(True)
                flip_control.setValue(context.angle_deg)
                flip_control.blockSignals(previous)
        return context if use_ernst else None

    def _rf_spoiling_is_effective(self, prefix):
        enabled = getattr(self, f"{prefix}_rf_spoiling").isChecked()
        increment = getattr(self, f"{prefix}_rf_spoiling_increment_deg").value()
        return enabled and not np.isclose(increment % 360.0, 0.0)

    def _update_all_ernst_controls(self):
        return (
            self._update_ernst_controls(
                "epi", self.epi_repetition_time_ms.value(), has_vfa=True
            ),
            self._update_ernst_controls(
                "csi", self.csi_repetition_time_ms.value(), has_vfa=True
            ),
            self._update_ernst_controls("flash", self.flash_repetition_time_ms.value()),
        )

    def _phantom_relaxation_changed(self, *_):
        self._update_all_ernst_controls()
        source = self.sequence_source.currentIndex()
        active = {
            self.EPI_SOURCE: self.epi_use_ernst_angle.isChecked(),
            self.CSI_SOURCE: self.csi_use_ernst_angle.isChecked(),
            self.FLASH_SOURCE: self.flash_use_ernst_angle.isChecked(),
        }
        if active.get(source, False):
            self._request_generated_sequence_refresh()

    def _update_bandwidth_labels(self):
        self._update_shared_rf_controls("epi")
        self._update_ernst_controls(
            "epi", self.epi_repetition_time_ms.value(), has_vfa=True
        )
        bandwidth_hz = self.sampling_bandwidth_khz.value() * 1000.0
        dwell_us = 1e6 / bandwidth_hz
        pixel_bandwidth_hz = bandwidth_hz / self.read_matrix.value()
        self.dwell_info.setText(f"{dwell_us:.3f} µs")
        self.pixel_bandwidth_info.setText(f"{pixel_bandwidth_hz:.3f} Hz/px")
        if self.epi_variable_flip_angle.isChecked():
            schedule = variable_flip_angle_schedule(
                self.epi_repetitions.value(),
                final_flip_angle_deg=self.epi_vfa_final_flip_angle_deg.value(),
            )
            self.epi_vfa_info.setText(
                f"{schedule[0]:.4g}° → {schedule[-1]:.4g}° "
                f"({schedule.size} repetitions)"
            )
        else:
            self.epi_vfa_info.setText("Off")

    def _update_csi_labels(self):
        self._update_shared_rf_controls("csi")
        self._update_ernst_controls(
            "csi", self.csi_repetition_time_ms.value(), has_vfa=True
        )
        bandwidth_hz = self.csi_bandwidth_hz.value()
        points = self.csi_spectral_points.value()
        self.csi_dwell_info.setText(f"{1e6 / bandwidth_hz:.3f} µs")
        self.csi_resolution_info.setText(f"{bandwidth_hz / points:.6g} Hz")
        if self.csi_variable_flip_angle.isChecked():
            phase_encodes = self.csi_read_matrix.value() * self.csi_phase_matrix.value()
            schedule = variable_flip_angle_schedule(
                phase_encodes,
                final_flip_angle_deg=self.csi_vfa_final_flip_angle_deg.value(),
            )
            self.csi_vfa_info.setText(
                f"{schedule[0]:.4g}° → {schedule[-1]:.4g}° "
                f"({phase_encodes} phase encodes; reset per repetition)"
            )
        else:
            self.csi_vfa_info.setText("Off")

    def _update_flash_labels(self):
        self._update_shared_rf_controls("flash")
        self._update_ernst_controls("flash", self.flash_repetition_time_ms.value())
        self._apply_flash_auto_spoilers()
        bandwidth_hz = self.flash_sampling_bandwidth_khz.value() * 1000.0
        self.flash_dwell_info.setText(f"{1e6 / bandwidth_hz:.3f} µs")
        self._update_flash_spoiler_info()

    def _update_bssfp_labels(self):
        self._update_shared_rf_controls("bssfp")
        bandwidth_hz = self.bssfp_bandwidth_khz.value() * 1000.0
        self.bssfp_dwell_info.setText(f"{1e6 / bandwidth_hz:.3f} µs")
        self._update_bssfp_preparation_controls()

    def _update_bssfp_preparation_controls(self):
        enabled = self.bssfp_alpha_half.isChecked()
        self._update_bssfp_startup_value_controls("bssfp")
        self.bssfp_alpha_half_phase_deg.setEnabled(enabled)

    def _update_bssfp_startup_value_controls(self, prefix):
        enabled = getattr(self, f"{prefix}_alpha_half").isChecked()
        use_ratios_control = getattr(self, f"{prefix}_alpha_half_use_ratios")
        use_ratios = use_ratios_control.isChecked()
        ratio_container = getattr(self, f"{prefix}_alpha_half_ratio_container")
        absolute_container = getattr(self, f"{prefix}_alpha_half_absolute_container")
        use_ratios_control.setEnabled(enabled)
        ratio_container.setVisible(use_ratios)
        ratio_container.setEnabled(enabled)
        absolute_container.setVisible(not use_ratios)
        absolute_container.setEnabled(enabled)

    def _update_ss_bssfp_labels(self):
        self._update_shared_rf_controls("ss_bssfp")
        bandwidth_hz = self.ss_bssfp_bandwidth_khz.value() * 1000.0
        self.ss_bssfp_dwell_info.setText(f"{1e6 / bandwidth_hz:.3f} µs")
        self._update_bssfp_startup_value_controls("ss_bssfp")
        self._update_ss_bssfp_spoiler_info()

    def _update_radial_me_bssfp_labels(self):
        self._update_shared_rf_controls("radial_me")
        samples = (
            self.radial_me_base_resolution.value()
            * self.radial_me_readout_oversampling.value()
        )
        dwell_us = 1e6 / (self.radial_me_pixel_bandwidth_hz.value() * samples)
        self.radial_me_sampling_info.setText(
            f"{samples} samples; requested dwell {dwell_us:.3f} µs"
        )
        self._update_bssfp_startup_value_controls("radial_me")

    def _update_me_bssfp_labels(self):
        self._update_shared_rf_controls("me_bssfp")
        bandwidth_hz = self.me_bssfp_bandwidth_khz.value() * 1000.0
        self.me_bssfp_sampling_info.setText(
            f"requested dwell {1e6 / bandwidth_hz:.3f} µs"
        )
        self._update_bssfp_startup_value_controls("me_bssfp")

    def _epi_fov_m(self):
        return (
            self.epi_read_fov_mm.value() / 1000.0,
            self.epi_phase_fov_mm.value() / 1000.0,
        )

    def _csi_fov_m(self):
        return (
            self.csi_read_fov_mm.value() / 1000.0,
            self.csi_phase_fov_mm.value() / 1000.0,
        )

    def _flash_fov_m(self):
        return (
            self.flash_read_fov_mm.value() / 1000.0,
            self.flash_phase_fov_mm.value() / 1000.0,
        )

    def _two_dimensional_encoding_frame(self, source_index=None):
        source_index = (
            self.sequence_source.currentIndex()
            if source_index is None
            else int(source_index)
        )
        controls = {
            self.EPI_SOURCE: (
                self.epi_read_gradient_axis,
                self.epi_phase_gradient_axis,
            ),
            self.CSI_SOURCE: (
                self.csi_read_gradient_axis,
                self.csi_phase_gradient_axis,
            ),
            self.FLASH_SOURCE: (
                self.flash_read_gradient_axis,
                self.flash_phase_gradient_axis,
            ),
        }
        read_phase = controls.get(source_index)
        if read_phase is None:
            return EncodingFrame.identity()
        read, phase = read_phase
        return EncodingFrame.from_read_phase_axes(
            read.currentData(), phase.currentData()
        )

    def _bssfp_fov_m(self):
        return (
            self.bssfp_read_fov_mm.value() / 1000.0,
            self.bssfp_phase_fov_mm.value() / 1000.0,
            self.bssfp_partition_fov_mm.value() / 1000.0,
        )

    def _ss_bssfp_fov_m(self):
        return (
            self.ss_bssfp_read_fov_mm.value() / 1000.0,
            self.ss_bssfp_phase_fov_mm.value() / 1000.0,
            self.ss_bssfp_partition_fov_mm.value() / 1000.0,
        )

    def _me_bssfp_fov_m(self):
        return (
            self.me_bssfp_read_fov_mm.value() / 1000.0,
            self.me_bssfp_phase_fov_mm.value() / 1000.0,
            self.me_bssfp_partition_fov_mm.value() / 1000.0,
        )

    def _generated_sequence_fov_m(self):
        """Return spatial coverage for the currently generated sequence."""
        source_index = self.sequence_source.currentIndex()
        if source_index == 1:
            return (
                *self._epi_fov_m(),
                (
                    self.epi_slice_thickness_mm.value() * self.epi_slice_count.value()
                    + self.epi_slice_gap_mm.value()
                    * max(0, self.epi_slice_count.value() - 1)
                )
                / 1000.0,
            )
        if source_index == 2:
            return (
                *self._csi_fov_m(),
                self.csi_slice_thickness_mm.value() / 1000.0,
            )
        if source_index == 3:
            return self._bssfp_fov_m()
        if source_index == 4:
            return self._ss_bssfp_fov_m()
        if source_index == 5:
            fov = self.radial_me_fov_mm.value() / 1000.0
            return (fov, fov, fov)
        if source_index == 6:
            return self._me_bssfp_fov_m()
        if source_index == self.FLASH_SOURCE:
            return (
                *self._flash_fov_m(),
                (
                    self.flash_slice_thickness_mm.value()
                    * self.flash_slice_count.value()
                    + self.flash_slice_gap_mm.value()
                    * max(0, self.flash_slice_count.value() - 1)
                )
                / 1000.0,
            )
        return None

    def _confirm_generated_sequence_fov(self):
        """Warn before a generated acquisition undersamples in-plane extent."""
        sequence_fov = self._generated_sequence_fov_m()
        if sequence_fov is None or self.phantom is None:
            return True

        phantom_fov = tuple(float(value) for value in self.phantom.fov)
        source_index = self.sequence_source.currentIndex()
        if source_index in self.CARTESIAN_3D_SOURCES:
            scanner_extents = np.zeros(3, dtype=float)
            scanner_extents[: min(3, len(phantom_fov))] = phantom_fov[:3]
            frame = self._three_dimensional_encoding_frame(source_index)
            required_fov = frame.required_encoding_extents(scanner_extents)
            axis_names = tuple(
                f"{role.title()} / {axis}"
                for role, axis in zip(
                    ("read", "phase", "partition"),
                    frame.axis_codes,
                )
            )
            comparisons = zip(sequence_fov[:3], required_fov)
        elif source_index in {
            self.EPI_SOURCE,
            self.CSI_SOURCE,
            self.FLASH_SOURCE,
        }:
            frame = self._two_dimensional_encoding_frame(source_index)
            scanner_extents = np.zeros(3, dtype=float)
            scanner_extents[: min(3, len(phantom_fov))] = phantom_fov[:3]
            required_fov = frame.required_encoding_extents(scanner_extents)
            read, phase, _ = frame.axis_codes
            axis_names = (f"Read / {read[-1]}", f"Phase / {phase[-1]}")
            comparisons = zip(sequence_fov[:2], required_fov[:2])
        else:
            axis_names = ("Read / x", "Phase / y")
            comparisons = zip(sequence_fov[:2], phantom_fov[:2])
        undersized = [
            (axis_names[index], sequence_extent, phantom_extent)
            for index, (sequence_extent, phantom_extent) in enumerate(comparisons)
            if sequence_extent < phantom_extent
            and not np.isclose(sequence_extent, phantom_extent, rtol=1e-9, atol=1e-12)
        ]
        if not undersized:
            return True

        details = "\n".join(
            f"• {axis}: sequence {sequence_extent * 1000:.4g} mm < "
            f"phantom {phantom_extent * 1000:.4g} mm"
            for axis, sequence_extent, phantom_extent in undersized
        )
        sequence_name = self.sequence_source.currentText()
        answer = QMessageBox.warning(
            self,
            "Sequence FOV is smaller than the phantom",
            f"The generated {sequence_name} sequence does not cover the full "
            f"phantom extent:\n\n{details}\n\n"
            "Signal outside this coverage may be aliased or excluded. "
            "Continue the simulation anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _epi_pulseq_parameters(self):
        ernst = self._update_ernst_controls(
            "epi", self.epi_repetition_time_ms.value(), has_vfa=True
        )
        parameters = {
            "fov_m": self._epi_fov_m(),
            "matrix": (self.read_matrix.value(), self.phase_matrix.value()),
            "sampling_bandwidth_hz": self.sampling_bandwidth_khz.value() * 1000.0,
            "flip_angle_deg": (
                ernst.angle_deg
                if ernst is not None
                else self.epi_flip_angle_deg.value()
            ),
            "variable_flip_angle": self.epi_variable_flip_angle.isChecked(),
            "vfa_final_flip_angle_deg": self.epi_vfa_final_flip_angle_deg.value(),
            "slice_thickness_m": self.epi_slice_thickness_mm.value() / 1000.0,
            "slice_gap_m": self.epi_slice_gap_mm.value() / 1000.0,
            "n_slices": self.epi_slice_count.value(),
            "repetitions": self.epi_repetitions.value(),
            "echo_time_s": self.epi_echo_time_ms.value() / 1000.0,
            "repetition_time_s": self.epi_repetition_time_ms.value() / 1000.0,
            "rf_spoiling": self._rf_spoiling_is_effective("epi"),
            "rf_spoiling_increment_deg": (self.epi_rf_spoiling_increment_deg.value()),
            "slice_offset_m": self.epi_slice_offset_mm.value() / 1000.0,
            "encoding_axes": self._two_dimensional_encoding_frame(
                self.EPI_SOURCE
            ).axis_codes,
            "spoil_after_slice": self.epi_spoil_after_slice.isChecked(),
            "spoiler_cycles_per_slice": self.epi_spoiler_cycles_per_slice.value(),
            "spoiler_cycles_per_voxel": self.epi_spoiler_cycles_per_voxel.value(),
            "spoiler_duration_s": self.epi_spoiler_duration_ms.value() / 1000.0,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        parameters.update(self._epi_rf_parameters())
        return parameters

    def _spiral_pulseq_parameters(self):
        parameters = self._epi_pulseq_parameters()
        parameters["spiral_turns"] = self.epi_spiral_turns.value()
        return parameters

    def _csi_pulseq_parameters(self):
        ernst = self._update_ernst_controls(
            "csi", self.csi_repetition_time_ms.value(), has_vfa=True
        )
        parameters = {
            "fov_m": self._csi_fov_m(),
            "matrix": (
                self.csi_read_matrix.value(),
                self.csi_phase_matrix.value(),
            ),
            "slice_thickness_m": self.csi_slice_thickness_mm.value() / 1000.0,
            "spectral_bandwidth_hz": self.csi_bandwidth_hz.value(),
            "spectral_points": self.csi_spectral_points.value(),
            "phase_encoding_order": self.csi_encoding_order.currentText(),
            "flip_angle_deg": (
                ernst.angle_deg
                if ernst is not None
                else self.csi_flip_angle_deg.value()
            ),
            "variable_flip_angle": self.csi_variable_flip_angle.isChecked(),
            "vfa_final_flip_angle_deg": self.csi_vfa_final_flip_angle_deg.value(),
            "echo_time_s": self.csi_echo_time_ms.value() / 1000.0,
            "repetition_time_s": self.csi_repetition_time_ms.value() / 1000.0,
            "repetitions": self.csi_repetitions.value(),
            "rf_spoiling": self._rf_spoiling_is_effective("csi"),
            "rf_spoiling_increment_deg": (self.csi_rf_spoiling_increment_deg.value()),
            "acquisition_interval_s": self._optional_acquisition_interval_s(
                self.csi_acquisition_interval_ms
            ),
            "slice_offset_m": self.csi_slice_offset_mm.value() / 1000.0,
            "encoding_axes": self._two_dimensional_encoding_frame(
                self.CSI_SOURCE
            ).axis_codes,
            "spoil_after_readout": self.csi_spoil_after_readout.isChecked(),
            "spoiler_cycles_per_slice": self.csi_spoiler_cycles_per_slice.value(),
            "spoiler_cycles_per_voxel": self.csi_spoiler_cycles_per_voxel.value(),
            "spoiler_duration_s": self.csi_spoiler_duration_ms.value() / 1000.0,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        parameters.update(self._shared_rf_parameters("csi"))
        return parameters

    def _flash_pulseq_parameters(self):
        ernst = self._update_ernst_controls(
            "flash", self.flash_repetition_time_ms.value()
        )
        parameters = {
            "fov_m": self._flash_fov_m(),
            "matrix": (
                self.flash_read_matrix.value(),
                self.flash_phase_matrix.value(),
            ),
            "sampling_bandwidth_hz": (
                self.flash_sampling_bandwidth_khz.value() * 1000.0
            ),
            "flip_angle_deg": (
                ernst.angle_deg
                if ernst is not None
                else self.flash_flip_angle_deg.value()
            ),
            "slice_thickness_m": self.flash_slice_thickness_mm.value() / 1000.0,
            "slice_gap_m": self.flash_slice_gap_mm.value() / 1000.0,
            "n_slices": self.flash_slice_count.value(),
            "slice_offset_m": self.flash_slice_offset_mm.value() / 1000.0,
            "echo_time_s": self.flash_echo_time_ms.value() / 1000.0,
            "repetition_time_s": (self.flash_repetition_time_ms.value() / 1000.0),
            "repetitions": self.flash_repetitions.value(),
            "acquisition_interval_s": self._optional_acquisition_interval_s(
                self.flash_acquisition_interval_ms
            ),
            "rf_spoiling": self._rf_spoiling_is_effective("flash"),
            "rf_spoiling_increment_deg": (self.flash_rf_spoiling_increment_deg.value()),
            "spoiler_cycles_per_slice": (self.flash_spoiler_cycles_per_slice.value()),
            "spoiler_cycles_per_voxel": (self.flash_spoiler_cycles_per_voxel.value()),
            "spoiler_duration_s": self.flash_spoiler_duration_ms.value() / 1000.0,
            "encoding_axes": self._two_dimensional_encoding_frame(
                self.FLASH_SOURCE
            ).axis_codes,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        parameters.update(self._shared_rf_parameters("flash"))
        return parameters

    def _bssfp_pulseq_parameters(self):
        parameters = {
            "fov_m": self._bssfp_fov_m(),
            "matrix": (
                self.bssfp_read_matrix.value(),
                self.bssfp_phase_matrix.value(),
                self.bssfp_partition_matrix.value(),
            ),
            "sampling_bandwidth_hz": self.bssfp_bandwidth_khz.value() * 1000.0,
            "flip_angle_deg": self.bssfp_flip_angle_deg.value(),
            "repetition_time_s": self.bssfp_repetition_time_ms.value() / 1000.0,
            "rf_phase_start_deg": self.bssfp_phase_start_deg.value(),
            "rf_phase_increment_deg": self.bssfp_phase_increment_deg.value(),
            "alpha_half_phase_deg": self.bssfp_alpha_half_phase_deg.value(),
            "alpha_half_use_ratios": self.bssfp_alpha_half_use_ratios.isChecked(),
            "alpha_half_tr_ratio": self.bssfp_alpha_half_tr_ratio.value(),
            "alpha_half_flip_ratio": self.bssfp_alpha_half_flip_ratio.value(),
            "alpha_half_center_spacing_s": (
                self.bssfp_alpha_half_center_spacing_ms.value() / 1000.0
            ),
            "alpha_half_flip_angle_deg": (self.bssfp_alpha_half_flip_angle_deg.value()),
            "dummy_repetitions": self.bssfp_dummy_repetitions.value(),
            "repetitions": self.bssfp_repetitions.value(),
            "acquisition_interval_s": self._optional_acquisition_interval_s(
                self.bssfp_acquisition_interval_ms
            ),
            "use_alpha_half": self.bssfp_alpha_half.isChecked(),
            "encoding_axes": self._three_dimensional_encoding_frame(
                self.BSSFP_SOURCE
            ).axis_codes,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        parameters.update(self._shared_rf_parameters("bssfp"))
        return parameters

    @staticmethod
    def _comma_separated_floats(text, name):
        try:
            values = tuple(
                float(value.strip()) for value in str(text).split(",") if value.strip()
            )
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated number list") from exc
        if not values or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite numbers")
        return values

    def _ss_bssfp_pulseq_parameters(self):
        target_names = tuple(
            value.strip()
            for value in self.ss_bssfp_target_names.text().split(",")
            if value.strip()
        )
        if not target_names:
            raise ValueError("Target names must not be empty")
        parameters = {
            "fov_m": self._ss_bssfp_fov_m(),
            "matrix": (
                self.ss_bssfp_read_matrix.value(),
                self.ss_bssfp_phase_matrix.value(),
                self.ss_bssfp_partition_matrix.value(),
            ),
            "target_frequency_offsets_hz": self._comma_separated_floats(
                self.ss_bssfp_target_offsets_hz.text(), "RF target offsets"
            ),
            "receiver_frequency_offsets_hz": self._comma_separated_floats(
                self.ss_bssfp_receiver_offsets_hz.text(), "Receiver offsets"
            ),
            "target_metabolite_names": target_names,
            "flip_angle_deg": self._comma_separated_floats(
                self.ss_bssfp_flip_angles_deg.text(), "Target flip angles"
            ),
            "spectral_rf_bandwidth_hz": None,
            "spectral_rf_bandwidth_factor_hz_ms": (
                self._shared_rf_shape_parameter("ss_bssfp") * 1000.0
            ),
            "sampling_bandwidth_hz": self.ss_bssfp_bandwidth_khz.value() * 1000.0,
            "encoding_duration_s": None,
            "repetition_time_s": self.ss_bssfp_repetition_time_ms.value() / 1000.0,
            "rf_phase_start_deg": self.ss_bssfp_phase_start_deg.value(),
            "rf_phase_increment_deg": self.ss_bssfp_phase_increment_deg.value(),
            "dummy_repetitions": self.ss_bssfp_dummy_repetitions.value(),
            "repetitions": self.ss_bssfp_repetitions.value(),
            "acquisition_interval_s": self._optional_acquisition_interval_s(
                self.ss_bssfp_acquisition_interval_ms
            ),
            "use_alpha_half": self.ss_bssfp_alpha_half.isChecked(),
            "alpha_half_use_ratios": (self.ss_bssfp_alpha_half_use_ratios.isChecked()),
            "alpha_half_tr_ratio": self.ss_bssfp_alpha_half_tr_ratio.value(),
            "alpha_half_flip_ratio": self.ss_bssfp_alpha_half_flip_ratio.value(),
            "alpha_half_center_spacing_s": (
                self.ss_bssfp_alpha_half_spacing_ms.value() / 1000.0
            ),
            "alpha_half_flip_angle_deg": (
                None
                if self.ss_bssfp_alpha_half_use_ratios.isChecked()
                else self._comma_separated_floats(
                    self.ss_bssfp_alpha_half_flip_angles_deg.text(),
                    "Startup flip angles",
                )
            ),
            "end_image_spoiler_cycles_per_fov": self.ss_bssfp_spoiler_cycles.value(),
            "end_image_spoiler_cycles_per_voxel": (
                self.ss_bssfp_spoiler_cycles_per_voxel.value()
            ),
            "end_image_spoiler_voxel_size_m": (
                self._ss_bssfp_reference_voxel_sizes_m()
            ),
            "end_image_spoiler_duration_s": (
                self.ss_bssfp_spoiler_duration_ms.value() / 1000.0
            ),
            "field_strength_t": self.field_strength_t.value(),
            "nucleus": self.nucleus.currentText(),
            "encoding_axes": self._three_dimensional_encoding_frame(
                self.SS_BSSFP_SOURCE
            ).axis_codes,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        shared_rf = self._shared_rf_parameters("ss_bssfp")
        parameters.update(
            {
                "spectral_rf_pulse_type": shared_rf.pop("rf_pulse_type"),
                "spectral_rf_duration_s": shared_rf.pop("rf_duration_s"),
                "spectral_rf_sinc_lobes": self.ss_bssfp_rf_sinc_lobes.value(),
                "spectral_rf_apodization": shared_rf.pop("rf_apodization"),
                "spectral_rf_slr_sharpness": shared_rf.pop("rf_slr_sharpness"),
            }
        )
        shared_rf.pop("rf_time_bandwidth_product", None)
        for name, value in shared_rf.items():
            parameters[f"spectral_{name}"] = value
        return parameters

    def _radial_me_bssfp_pulseq_parameters(self):
        parameters = {
            "fov_m": self.radial_me_fov_mm.value() / 1000.0,
            "base_resolution": self.radial_me_base_resolution.value(),
            "readout_oversampling": self.radial_me_readout_oversampling.value(),
            "spokes_per_measurement": self.radial_me_spokes.value(),
            "measurements": self.radial_me_measurements.value(),
            "acquisition_interval_s": self._optional_acquisition_interval_s(
                self.radial_me_acquisition_interval_ms
            ),
            "echoes": self.radial_me_echoes.value(),
            "echo_spacing_s": self.radial_me_echo_spacing_ms.value() / 1000.0,
            "pixel_bandwidth_hz": self.radial_me_pixel_bandwidth_hz.value(),
            "flip_angle_deg": self.radial_me_flip_angle_deg.value(),
            "repetition_time_s": self.radial_me_repetition_time_ms.value() / 1000.0,
            "rf_phase_start_deg": self.radial_me_phase_start_deg.value(),
            "rf_phase_increment_deg": self.radial_me_phase_increment_deg.value(),
            "use_alpha_half": self.radial_me_alpha_half.isChecked(),
            "alpha_half_use_ratios": (self.radial_me_alpha_half_use_ratios.isChecked()),
            "alpha_half_tr_ratio": self.radial_me_alpha_half_tr_ratio.value(),
            "alpha_half_flip_ratio": self.radial_me_alpha_half_flip_ratio.value(),
            "alpha_half_center_spacing_s": (
                self.radial_me_alpha_half_center_spacing_ms.value() / 1000.0
            ),
            "alpha_half_flip_angle_deg": (
                self.radial_me_alpha_half_flip_angle_deg.value()
            ),
            "use_tip_back": self.radial_me_tip_back.isChecked(),
            "prephaser_duration_s": self.radial_me_prephaser_duration_ms.value()
            / 1000.0,
            "inter_measurement_rotation_deg": self.radial_me_rotation_deg.value(),
            "field_strength_t": self.field_strength_t.value(),
            "nucleus": self.nucleus.currentText(),
            "encoding_axes": self._three_dimensional_encoding_frame(
                self.RADIAL_ME_BSSFP_SOURCE
            ).axis_codes,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        parameters.update(self._shared_rf_parameters("radial_me"))
        return parameters

    def _me_bssfp_pulseq_parameters(self):
        parameters = {
            "fov_m": self._me_bssfp_fov_m(),
            "matrix": (
                self.me_bssfp_read_matrix.value(),
                self.me_bssfp_phase_matrix.value(),
                self.me_bssfp_partition_matrix.value(),
            ),
            "echoes": self.me_bssfp_echoes.value(),
            "echo_spacing_s": self.me_bssfp_echo_spacing_ms.value() / 1000.0,
            "readout_strategy": (
                "flyback"
                if self.me_bssfp_readout_strategy.currentIndex() == 0
                else "symmetric"
            ),
            "sampling_bandwidth_hz": self.me_bssfp_bandwidth_khz.value() * 1000.0,
            "flip_angle_deg": self.me_bssfp_flip_angle_deg.value(),
            "rf_bandwidth_hz": self.me_bssfp_rf_bandwidth_hz.value(),
            "receiver_frequency_offset_hz": (self.me_bssfp_receiver_offset_hz.value()),
            "encoding_duration_s": (
                self.me_bssfp_encoding_duration_ms.value() / 1000.0
            ),
            "repetition_time_s": self.me_bssfp_repetition_time_ms.value() / 1000.0,
            "rf_phase_start_deg": self.me_bssfp_phase_start_deg.value(),
            "rf_phase_increment_deg": self.me_bssfp_phase_increment_deg.value(),
            "dummy_repetitions": self.me_bssfp_dummy_repetitions.value(),
            "repetitions": self.me_bssfp_repetitions.value(),
            "acquisition_interval_s": self._optional_acquisition_interval_s(
                self.me_bssfp_acquisition_interval_ms
            ),
            "use_alpha_half": self.me_bssfp_alpha_half.isChecked(),
            "alpha_half_use_ratios": (self.me_bssfp_alpha_half_use_ratios.isChecked()),
            "alpha_half_tr_ratio": self.me_bssfp_alpha_half_tr_ratio.value(),
            "alpha_half_flip_ratio": self.me_bssfp_alpha_half_flip_ratio.value(),
            "alpha_half_center_spacing_s": (
                self.me_bssfp_alpha_half_center_spacing_ms.value() / 1000.0
            ),
            "alpha_half_flip_angle_deg": (
                self.me_bssfp_alpha_half_flip_angle_deg.value()
            ),
            "field_strength_t": self.field_strength_t.value(),
            "nucleus": self.nucleus.currentText(),
            "encoding_axes": self._three_dimensional_encoding_frame(
                self.ME_BSSFP_SOURCE
            ).axis_codes,
            "scanner_parameters": self.scanner_parameters.to_dict(),
        }
        parameters.update(self._shared_rf_parameters("me_bssfp"))
        # ME-bSSFP retains the legacy explicit bandwidth argument for scripts;
        # the builder prefers the shared TBW when both are supplied.
        return parameters

    def _pulseq_export_spec(self):
        source_index = self.sequence_source.currentIndex()
        if source_index == 1:
            if self.epi_readout_trajectory.currentText() == "Spiral":
                return "spiral", self._spiral_pulseq_parameters(), "spiral.seq"
            return "epi", self._epi_pulseq_parameters(), "epi.seq"
        if source_index == 2:
            return "csi", self._csi_pulseq_parameters(), "csi.seq"
        if source_index == 3:
            return "bssfp_3d", self._bssfp_pulseq_parameters(), "bssfp_3d.seq"
        if source_index == 4:
            return (
                "spectral_bssfp_3d",
                self._ss_bssfp_pulseq_parameters(),
                "spectral_bssfp_3d.seq",
            )
        if source_index == 5:
            return (
                "radial_me_bssfp_3d",
                self._radial_me_bssfp_pulseq_parameters(),
                "radial_me_bssfp_3d.seq",
            )
        if source_index == 6:
            return (
                "me_bssfp_3d",
                self._me_bssfp_pulseq_parameters(),
                "me_bssfp_3d.seq",
            )
        if source_index == self.FLASH_SOURCE:
            return "flash", self._flash_pulseq_parameters(), "flash_2d.seq"
        raise ValueError("Select a generated sequence")

    def _load_internal_sequence(self):
        self._generated_pulseq_sequence = None
        self._sequence_generation_pending = False
        self._generated_sequence_source_index = None
        self._generation_error = ""
        self._acquisition_compiled = None
        self.export_pulseq_button.setEnabled(False)
        self.acquisition = None
        self.acquisition_frames = None
        self.acquisition_volumes = None
        self.spiral_acquisition = None
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
        return True

    def _load_cartesian_epi(self):
        if self.epi_readout_trajectory.currentText() == "Spiral":
            return self._load_spiral()
        try:
            sequence = make_pulseq_epi(**self._epi_pulseq_parameters())
            self._set_generated_pulseq_sequence(sequence, "internal-cartesian-epi")
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.acquisition = None
            self.program = None
            self._acquisition_compiled = None
            self._generation_error = f"Invalid Cartesian acquisition: {exc}"
            return False

    def _load_spiral(self):
        self._generated_pulseq_sequence = None
        self._acquisition_compiled = None
        self.export_pulseq_button.setEnabled(True)
        self.acquisition = None
        self.acquisition_frames = None
        self.acquisition_volumes = None
        self.spectroscopic_acquisition = None
        try:
            sequence = make_pulseq_spiral(**self._spiral_pulseq_parameters())
            self._set_generated_pulseq_sequence(sequence, "internal-spiral")
            return True
        except Exception as exc:
            self.spiral_acquisition = None
            self.program = None
            self._acquisition_compiled = None
            self._generation_error = f"Invalid spiral acquisition: {exc}"
            return False

    def _load_csi(self):
        try:
            sequence = make_pulseq_csi(**self._csi_pulseq_parameters())
            self._set_generated_pulseq_sequence(sequence, "internal-csi")
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.program = None
            self._acquisition_compiled = None
            self.acquisition = None
            self.spiral_acquisition = None
            self.spectroscopic_acquisition = None
            self._generation_error = f"Invalid CSI sequence: {exc}"
            return False

    def _load_flash(self):
        try:
            sequence = make_pulseq_flash(**self._flash_pulseq_parameters())
            self._set_generated_pulseq_sequence(sequence, "internal-flash-2d")
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.program = None
            self._acquisition_compiled = None
            self.acquisition = None
            self.acquisition_frames = None
            self.acquisition_volumes = None
            self.spiral_acquisition = None
            self.spectroscopic_acquisition = None
            self._generation_error = f"Invalid FLASH sequence: {exc}"
            return False

    def _load_bssfp(self):
        try:
            sequence = make_pulseq_bssfp(**self._bssfp_pulseq_parameters())
            self._set_generated_pulseq_sequence(sequence, "internal-bssfp-3d")
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.program = None
            self._acquisition_compiled = None
            self.acquisition = None
            self.acquisition_frames = None
            self.acquisition_volumes = None
            self.spiral_acquisition = None
            self._generation_error = f"Invalid bSSFP sequence: {exc}"
            return False

    def _load_ss_bssfp(self):
        try:
            sequence = make_pulseq_spectral_selective_bssfp(
                **self._ss_bssfp_pulseq_parameters()
            )
            self._set_generated_pulseq_sequence(
                sequence, "internal-spectral-selective-bssfp-3d"
            )
            encoding_duration_ms = (
                float(sequence.definitions["EncodingLobeDuration"]) * 1000.0
            )
            previous = self.ss_bssfp_encoding_duration_ms.blockSignals(True)
            self.ss_bssfp_encoding_duration_ms.setValue(encoding_duration_ms)
            self.ss_bssfp_encoding_duration_ms.blockSignals(previous)
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.program = None
            self._acquisition_compiled = None
            self.acquisition = None
            self.acquisition_frames = None
            self.acquisition_volumes = None
            self.spiral_acquisition = None
            self._generation_error = f"Invalid SS-bSSFP sequence: {exc}"
            return False

    def _load_radial_me_bssfp(self):
        try:
            sequence = make_pulseq_radial_me_bssfp(
                **self._radial_me_bssfp_pulseq_parameters()
            )
            self._set_generated_pulseq_sequence(sequence, "internal-radial-me-bssfp-3d")
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.program = None
            self._acquisition_compiled = None
            self.acquisition = None
            self.acquisition_frames = None
            self.acquisition_volumes = None
            self.spiral_acquisition = None
            self._generation_error = f"Invalid radial ME-bSSFP sequence: {exc}"
            return False

    def _load_me_bssfp(self):
        try:
            sequence = make_pulseq_me_bssfp(**self._me_bssfp_pulseq_parameters())
            self._set_generated_pulseq_sequence(sequence, "internal-me-bssfp-3d")
            return True
        except Exception as exc:
            self._generated_pulseq_sequence = None
            self.program = None
            self._acquisition_compiled = None
            self.acquisition = None
            self.acquisition_frames = None
            self.acquisition_volumes = None
            self.spiral_acquisition = None
            self._generation_error = f"Invalid Cartesian ME-bSSFP sequence: {exc}"
            return False

    def _set_generated_pulseq_sequence(self, sequence, source):
        self._apply_workspace_frequency_reference(sequence)
        with tempfile.TemporaryDirectory(prefix="blochsimulator-pulseq-") as directory:
            path = Path(directory) / "generated.seq"
            sequence.write(str(path), v141_compat=True)
            imported = load_pulseq(path)
        self.program = SequenceProgram(
            events=imported.events,
            duration_s=imported.duration_s,
            source=source,
            version=imported.version,
            metadata=imported.metadata,
        )
        self._generated_pulseq_sequence = sequence
        self._infer_current_acquisition()
        self._apply_probe_defaults_from_program()
        self._configure_frame_selector()
        self._configure_spectroscopy_selectors()
        self._show_program()

    def _infer_current_acquisition(self, compiled=None):
        """Attach CSI, 2D-frame, or 3D-volume layout to the current program."""
        if compiled is None:
            compiled = SequenceCompiler().compile_acquisition(self.program)
        payload = _infer_sequence_acquisition(self.program, compiled)
        self._apply_acquisition_payload(payload)
        return compiled

    def _apply_acquisition_payload(self, payload):
        self.program = payload.program
        self._acquisition_compiled = payload.compiled
        self.acquisition = payload.acquisition
        self.acquisition_frames = payload.acquisition_frames
        self.acquisition_volumes = payload.acquisition_volumes
        self.spectroscopic_acquisition = payload.spectroscopic_acquisition
        self.spiral_acquisition = payload.spiral_acquisition
        self.acquisition_note = payload.acquisition_note

    def _run_python_script(self):
        """Select and run a Python sequence-generation script without a shell."""
        if (
            self.script_process is not None
            and self.script_process.state() != QProcess.NotRunning
        ):
            QMessageBox.information(
                self, "Python script", "A Python script is already running."
            )
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Run Python sequence script",
            str(workspace_directory("sequences")),
            "Python script (*.py);;All files (*)",
        )
        if not filename:
            return

        self._script_path = Path(filename).resolve()
        self._script_sequence_snapshot = self._sequence_file_snapshot(self._script_path)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Python script · {self._script_path.name}")
        dialog.resize(760, 480)
        layout = QVBoxLayout(dialog)
        output = QTextEdit()
        output.setReadOnly(True)
        output.setLineWrapMode(QTextEdit.NoWrap)
        output.setPlainText(
            f"Running {self._script_path.name}\n"
            f"Python: {sys.executable}\n"
            f"Working directory: {self._script_path.parent}\n\n"
        )
        layout.addWidget(output, 1)
        buttons = QHBoxLayout()
        stop_button = QPushButton("Stop script")
        close_button = QPushButton("Close")
        close_button.setEnabled(False)
        stop_button.clicked.connect(self._cancel_python_script)
        close_button.clicked.connect(dialog.close)
        buttons.addStretch(1)
        buttons.addWidget(stop_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(self._script_path.parent))
        process.setProgram(sys.executable)
        process.setArguments([str(self._script_path)])
        process.readyReadStandardOutput.connect(self._read_python_script_output)
        process.errorOccurred.connect(self._python_script_error)
        process.finished.connect(self._python_script_finished)
        self.script_process = process
        self.script_output_dialog = dialog
        self.script_output = output
        self.script_stop_button = stop_button
        self.script_close_button = close_button
        self.run_script_button.setEnabled(False)
        self.status.setText(f"Running Python script {self._script_path.name}…")
        process.start()
        dialog.show()

    def _script_sequence_roots(self, script_path):
        roots = {
            Path(script_path).parent,
            workspace_directory("sequences"),
        }
        return tuple(root for root in roots if root.exists() and root.is_dir())

    def _sequence_file_snapshot(self, script_path):
        snapshot = {}
        for root in self._script_sequence_roots(script_path):
            try:
                files = root.rglob("*.seq")
                for path in files:
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    snapshot[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        return snapshot

    def _read_python_script_output(self):
        process = self.script_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data and self.script_output is not None:
            self.script_output.moveCursor(self.script_output.textCursor().End)
            self.script_output.insertPlainText(data)
            self.script_output.ensureCursorVisible()

    def _cancel_python_script(self):
        process = self.script_process
        if process is not None and process.state() != QProcess.NotRunning:
            process.terminate()
            QTimer.singleShot(
                2000,
                lambda active_process=process: (
                    active_process.kill()
                    if active_process.state() != QProcess.NotRunning
                    else None
                ),
            )
            if self.script_output is not None:
                self.script_output.append("\nStopping script…")

    def _python_script_error(self, error):
        if self.script_output is not None:
            self.script_output.append(f"\nProcess error: {error}")
        if error == QProcess.FailedToStart:
            self.run_script_button.setEnabled(True)
            if hasattr(self, "script_stop_button"):
                self.script_stop_button.setEnabled(False)
            if hasattr(self, "script_close_button"):
                self.script_close_button.setEnabled(True)
            self.status.setText("Python script could not be started")

    def _python_script_finished(self, exit_code, exit_status):
        self._read_python_script_output()
        process = self.script_process
        script_path = self._script_path
        self.run_script_button.setEnabled(True)
        if hasattr(self, "script_stop_button"):
            self.script_stop_button.setEnabled(False)
        if hasattr(self, "script_close_button"):
            self.script_close_button.setEnabled(True)
        normal_exit = exit_status == QProcess.NormalExit and int(exit_code) == 0
        if self.script_output is not None:
            self.script_output.append(
                f"\nScript finished with exit code {int(exit_code)}."
            )
        self.script_process = None
        if process is not None:
            process.deleteLater()
        if not normal_exit or script_path is None:
            self.status.setText(f"Python script failed with exit code {int(exit_code)}")
            return

        current = self._sequence_file_snapshot(script_path)
        changed = [
            path
            for path, signature in current.items()
            if self._script_sequence_snapshot.get(path) != signature
        ]
        if not changed:
            self.status.setText(
                f"{script_path.name} completed; no new or updated .seq file found"
            )
            return
        generated = max(changed, key=lambda path: current[path][0])
        if self.script_output is not None:
            self.script_output.append(f"Loading generated sequence: {generated}")
        self.status.setText(f"Loading generated {generated.name}…")
        self._start_pulseq_load(generated)

    def _load_pulseq_file(self):
        if self.pulseq_load_worker is not None and self.pulseq_load_worker.isRunning():
            QMessageBox.information(
                self, "Pulseq import", "A Pulseq file is already being loaded."
            )
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pulseq sequence",
            str(workspace_directory("sequences")),
            "Pulseq sequence (*.seq);;All files (*)",
        )
        if not filename:
            return

        self._start_pulseq_load(filename)
        self._display_pulseq_spoiler_warning()

    def _show_simulation_settings(self):
        """Open the simulation tab."""
        main_window = self.window()

        if not hasattr(main_window, "show_settings"):
            QMessageBox.warning(
                self,
                "Settings unavailable",
                "The main window does not provide a settings dialog.",
            )
            return

        main_window.show_settings(initial_tab="simulation")

    def _display_pulseq_spoiler_warning(self):
        """Warn about imported spoilers without blocking the background import."""
        existing = self._pulseq_spoiler_warning_dialog
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("Pulseq import spoiler settings")
        dialog.setText(
            "For imported Pulseq sequences, please ensure that spoilers are set and subvoxel spins are activated in the settings.\n\n"
            "Set `Spoiler Simulation` to Gradient waveform (subvoxel spins) in the Simulation Settings to avoid incorrect simulation results.\n"
            "Make sure to set `Subvoxel spins` to > 1."
        )

        settings_button = dialog.addButton(
            "Open Simulation Settings...", QMessageBox.ActionRole
        )
        dialog.addButton(QMessageBox.Cancel)
        settings_button.clicked.connect(self._show_simulation_settings)
        dialog.finished.connect(self._pulseq_spoiler_warning_finished)
        self._pulseq_spoiler_warning_dialog = dialog
        dialog.open()

    def _pulseq_spoiler_warning_finished(self, _result):
        self._pulseq_spoiler_warning_dialog = None

    def _start_pulseq_load(self, filename):
        """Start the shared background Pulseq import path for a known file."""
        if self.pulseq_load_worker is not None and self.pulseq_load_worker.isRunning():
            QMessageBox.information(
                self, "Pulseq import", "A Pulseq file is already being loaded."
            )
            return
        self.load_pulseq_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Loading Pulseq…")
        self.status.setText(f"Loading {Path(filename).name}…")
        self.pulseq_load_worker = PulseqLoadThread(str(filename))
        self.pulseq_load_worker.stage.connect(self._pulseq_load_status)
        self.pulseq_load_worker.result_ready.connect(
            lambda payload, source=filename: self._pulseq_load_finished(payload, source)
        )
        self.pulseq_load_worker.failed.connect(self._pulseq_load_failed)
        self.pulseq_load_worker.finished.connect(self._pulseq_load_thread_finished)
        self.pulseq_load_worker.start()

    def _pulseq_load_status(self, message):
        self.status.setText(str(message))
        logger = getattr(self.window(), "log_message", None)
        if callable(logger):
            logger(f"Pulseq import: {message}")

    def _pulseq_load_finished(self, payload, filename):
        self._apply_loaded_pulseq(payload, filename)
        self._reset_pulseq_load_controls("Pulseq loaded")

    def _pulseq_load_failed(self, message):
        self._reset_pulseq_load_controls("Pulseq import failed")
        QMessageBox.critical(self, "Pulseq import failed", str(message))

    def _reset_pulseq_load_controls(self, progress_text):
        self.load_pulseq_button.setEnabled(True)
        self.run_button.setEnabled(
            self.program is not None and self.object_source.currentIndex() != 2
        )
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat(progress_text)

    def _pulseq_load_thread_finished(self):
        worker = self.pulseq_load_worker
        self.pulseq_load_worker = None
        if worker is not None:
            worker.deleteLater()

    def _build_epi_pulseq(self):
        if self.epi_readout_trajectory.currentText() == "Spiral":
            sequence = make_pulseq_spiral(**self._spiral_pulseq_parameters())
        else:
            sequence = make_pulseq_epi(**self._epi_pulseq_parameters())
        return self._apply_workspace_frequency_reference(sequence)

    def _ernst_pulseq_definitions(self):
        source_index = self.sequence_source.currentIndex()
        ernst_specs = {
            self.EPI_SOURCE: (
                "epi",
                self.epi_repetition_time_ms.value(),
                True,
            ),
            self.CSI_SOURCE: (
                "csi",
                self.csi_repetition_time_ms.value(),
                True,
            ),
            self.FLASH_SOURCE: (
                "flash",
                self.flash_repetition_time_ms.value(),
                False,
            ),
        }
        if source_index not in ernst_specs:
            return {}
        prefix, repetition_time_ms, has_vfa = ernst_specs[source_index]
        context = self._update_ernst_controls(
            prefix, repetition_time_ms, has_vfa=has_vfa
        )
        if context is None:
            return {}
        return {
            "UseErnstAngle": True,
            "ErnstUsesT2": False,
            "ErnstAngleDeg": context.angle_deg,
            "ErnstRepetitionTime": context.repetition_time_s,
            "ErnstEffectiveT1": context.effective_t1_s,
            "ErnstT1Range": list(context.t1_range_s),
            "ErnstT1Source": context.source,
        }

    def _apply_workspace_frequency_reference(self, sequence):
        """Attach the active B0/nucleus reference to every generated Pulseq file."""
        sequence.set_definition("FieldStrengthT", self.field_strength_t.value())
        sequence.set_definition("Nucleus", self.nucleus.currentText())
        for name, value in self._ernst_pulseq_definitions().items():
            sequence.set_definition(name, value)
        return sequence

    def _write_pulseq_path(self, filename, *, export_spec=None):
        """Write the selected generated sequence and return the final path."""
        sequence_kind, parameters, _ = export_spec or self._pulseq_export_spec()
        builders = {
            "epi": make_pulseq_epi,
            "spiral": make_pulseq_spiral,
            "csi": make_pulseq_csi,
            "flash": make_pulseq_flash,
            "bssfp_3d": make_pulseq_bssfp,
            "spectral_bssfp_3d": make_pulseq_spectral_selective_bssfp,
            "me_bssfp_3d": make_pulseq_me_bssfp,
            "radial_me_bssfp_3d": make_pulseq_radial_me_bssfp,
        }
        sequence = builders[sequence_kind](**parameters)
        self._apply_workspace_frequency_reference(sequence)
        path = Path(filename)
        if path.suffix.lower() != ".seq":
            path = path.with_suffix(".seq")
        path.parent.mkdir(parents=True, exist_ok=True)
        sequence.write(str(path), v141_compat=True)
        return path

    def _export_pulseq(self):
        try:
            export_spec = self._pulseq_export_spec()
        except ValueError:
            QMessageBox.warning(
                self,
                "No generated sequence",
                "Select a generated sequence before exporting Pulseq.",
            )
            return
        sequence_kind, parameters, default_name = export_spec
        both_filter = "Pulseq + Jupyter notebook (*.seq)"
        sequence_filter = "Pulseq sequence only (*.seq)"
        notebook_filter = "Jupyter notebook only (*.ipynb)"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export generated Pulseq sequence",
            str(self._export_directory() / default_name),
            f"{both_filter};;{sequence_filter};;{notebook_filter}",
        )
        if not filename:
            return
        try:
            requested_path = Path(filename)
            exported_paths = []
            sequence_path = requested_path.with_suffix(".seq")
            notebook_path = requested_path.with_suffix(".ipynb")
            if selected_filter != notebook_filter:
                sequence_path = self._write_pulseq_path(
                    sequence_path,
                    export_spec=export_spec,
                )
                exported_paths.append(sequence_path)
            if selected_filter != sequence_filter:
                notebook_path = export_pulseq_generation_notebook(
                    str(notebook_path),
                    sequence_kind,
                    parameters,
                    seq_filename=sequence_path.name,
                    pulseq_definitions={
                        "FieldStrengthT": self.field_strength_t.value(),
                        "Nucleus": self.nucleus.currentText(),
                        **self._ernst_pulseq_definitions(),
                    },
                )
                exported_paths.append(notebook_path)
            QMessageBox.information(
                self,
                "Pulseq export complete",
                "Exported to "
                f"{requested_path.parent}:\n"
                + "\n".join(path.name for path in exported_paths),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Pulseq export failed", str(exc))

    def _load_pulseq_path(self, filename):
        """Load a Pulseq file and attach a validated 2D Cartesian layout."""
        payload = _prepare_pulseq_load(filename)
        self._apply_loaded_pulseq(payload, filename)

    def _apply_loaded_pulseq(self, payload, filename):
        """Apply a prepared Pulseq payload on the Qt GUI thread."""
        self._generated_pulseq_sequence = None
        self._sequence_generation_pending = False
        self._generated_sequence_source_index = None
        self._generation_error = ""
        self._apply_acquisition_payload(payload)
        self.sequence_source.setCurrentIndex(self.PULSEQ_SOURCE)
        self._apply_pulseq_fov()
        self._apply_pulseq_frequency_reference()
        self._apply_probe_defaults_from_program()
        self._show_program(compiled=payload.compiled)
        self._configure_frame_selector()
        self._configure_spectroscopy_selectors()
        status = f"Loaded {Path(filename).name}"
        if self.acquisition_note:
            status += f"; {self.acquisition_note}"
        self.status.setText(status)

    def _apply_pulseq_fov(self):
        definitions = dict(self.program.metadata.get("definitions", {}))
        fov_value = next(
            (
                value
                for key, value in definitions.items()
                if str(key).lower() == "encodingfov"
            ),
            None,
        )
        if fov_value is None:
            fov_value = next(
                (
                    value
                    for key, value in definitions.items()
                    if str(key).lower() == "fov"
                ),
                None,
            )
        if fov_value is None:
            return
        fov = np.asarray(fov_value, dtype=float).reshape(-1)
        if fov.size >= 3:
            frame = EncodingFrame.from_definitions(definitions)
            fov = np.abs(frame.matrix) @ fov[:3]
        if fov.size >= 2 and np.all(np.isfinite(fov[:2])) and np.all(fov[:2] > 0):
            if np.isclose(fov[0], fov[1], rtol=1e-6, atol=1e-12):
                self.fov_mm.setValue(float(fov[0]) * 1000.0)
        if fov.size >= 3 and np.isfinite(fov[2]) and fov[2] > 0:
            self.fov_z_mm.setValue(float(fov[2]) * 1000.0)

    def _apply_pulseq_frequency_reference(self):
        definitions = {
            str(key).lower(): value
            for key, value in self.program.metadata.get("definitions", {}).items()
        }
        # Inside the desktop workspace B0 is application state shared with
        # Free Mode and the Phantom designer. A Pulseq definition documents the
        # file that was loaded, but must not silently reset that shared value
        # (many files contain the conventional 3 T default). Standalone widget
        # use still adopts a valid file definition as before.
        shared_field_strength = getattr(
            self.window(), "_workspace_field_strength_t", None
        )
        if shared_field_strength is not None:
            self.field_strength_t.setValue(float(shared_field_strength))
        else:
            field_value = definitions.get("fieldstrengtht")
            try:
                field_strength = float(np.asarray(field_value).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                field_strength = np.nan
            if np.isfinite(field_strength) and field_strength > 0:
                self.field_strength_t.setValue(field_strength)
        shared_nucleus = getattr(self.window(), "_workspace_nucleus", None)
        if shared_nucleus in NUCLEUS_GAMMA_HZ_PER_T:
            self.nucleus.setCurrentText(shared_nucleus)
        else:
            nucleus_value = definitions.get("nucleus")
            if nucleus_value is not None:
                nucleus = str(nucleus_value).strip()
                nucleus_index = self.nucleus.findText(nucleus)
                if nucleus_index >= 0:
                    self.nucleus.setCurrentIndex(nucleus_index)

    def _waveform_nucleus(self) -> str:
        nucleus = str(self.nucleus.currentText()).strip()
        return nucleus if nucleus in NUCLEUS_GAMMA_HZ_PER_T else "H1"

    def _sync_rf_phase_view(self):
        """Keep the phase axis aligned with the RF amplitude plot."""
        self._rf_phase_view.setGeometry(self.rf_plot.plotItem.vb.sceneBoundingRect())
        self._rf_phase_view.linkedViewChanged(
            self.rf_plot.plotItem.vb, self._rf_phase_view.XAxis
        )

    def _waveform_scales(self):
        nucleus = self._waveform_nucleus()
        if self.waveform_units.currentData() == "physical":
            rf_scale = float(rf_hz_to_gauss_for_nucleus(1.0, nucleus))
            gradient_scale = float(gradient_hz_per_m_to_t_per_m(1.0, nucleus))
            return rf_scale, gradient_scale, "B1", "G", "Gradient", "T/m"
        return 1.0, 1e-3, "RF", "Hz", "Gradient", "kHz/m"

    def _waveform_units_changed(self, *_):
        rf_scale, gradient_scale, rf_name, rf_unit, grad_name, grad_unit = (
            self._waveform_scales()
        )
        del rf_scale, gradient_scale
        self.rf_plot.setLabel("left", rf_name, rf_unit)
        self.gradient_plot.setLabel("left", grad_name, grad_unit)
        self.waveform_nucleus_label.setText(f"Conversion: {self._waveform_nucleus()}")
        self._sequence_plot_window_s = None
        if self.program is not None:
            x_range = self.rf_plot.getViewBox().viewRange()[0]
            start_s = max(0.0, float(x_range[0]) / 1000.0)
            end_s = min(float(self.program.duration_s), float(x_range[1]) / 1000.0)
            if end_s <= start_s:
                start_s, end_s = 0.0, float(self.program.duration_s)
            self._refresh_sequence_waveforms(start_s, end_s)
            self._update_waveform_y_ranges()
            self._update_waveform_value_summary()
        self._emit_physical_b1_changed()

    def physical_b1_display_context(self):
        """Describe the nominal peak B1 represented by the loaded sequence."""
        nucleus = self._waveform_nucleus()
        peak_b1_gauss = None
        source = ""
        if self.program is not None:
            source = str(self.program.source)
            peak_hz = max(
                (
                    float(np.max(np.abs(event.samples_hz)))
                    for event in self.program.rf_events
                ),
                default=0.0,
            )
            peak_b1_gauss = float(rf_hz_to_gauss_for_nucleus(peak_hz, nucleus))
        return {
            "nominal_peak_b1_gauss": peak_b1_gauss,
            "nucleus": nucleus,
            "sequence_source": source,
            "parameters_pending": bool(self._sequence_generation_pending),
        }

    def _emit_physical_b1_changed(self):
        self.physical_b1_changed.emit(self.physical_b1_display_context())

    def _update_waveform_y_ranges(self):
        if self.program is None:
            return
        rf_scale, gradient_scale, *_labels = self._waveform_scales()
        if self.program.rf_events:
            rf_peak = max(
                float(np.max(np.abs(event.samples_hz)))
                for event in self.program.rf_events
            )
            self.rf_plot.setYRange(
                0.0, max(rf_peak * rf_scale * 1.05, 1e-12), padding=0
            )
        gradient_values = [
            np.asarray(event.samples_hz_per_m, dtype=float) * gradient_scale
            for event in self.program.gradient_events
        ]
        if gradient_values:
            gradient_limit = max(
                float(np.max(np.abs(values))) for values in gradient_values
            )
            self.gradient_plot.setYRange(
                -max(gradient_limit * 1.05, 1e-12),
                max(gradient_limit * 1.05, 1e-12),
                padding=0,
            )

    def _update_waveform_value_summary(self):
        if self.program is None:
            self.waveform_value_summary.setText("No sequence waveforms")
            return
        nucleus = self._waveform_nucleus()
        rf_peak_hz = max(
            (
                float(np.max(np.abs(event.samples_hz)))
                for event in self.program.rf_events
            ),
            default=0.0,
        )
        rf_peak_gauss = float(rf_hz_to_gauss_for_nucleus(rf_peak_hz, nucleus))
        gradient_peaks = {}
        for axis in "xyz":
            peak_hz_per_m = max(
                (
                    float(np.max(np.abs(event.samples_hz_per_m)))
                    for event in self.program.gradient_events
                    if event.axis == axis
                ),
                default=0.0,
            )
            gradient_peaks[axis] = float(
                gradient_hz_per_m_to_t_per_m(peak_hz_per_m, nucleus)
            )
        effective_text = ""
        phantom = getattr(self, "phantom", None)
        if phantom is None and self.object_source.currentIndex() == 0:
            phantom = self._selected_designed_phantom()
        if phantom is not None:
            tx_source = getattr(phantom, "tx_sensitivity_map", None)
            if tx_source is None:
                tx_source = np.ones(phantom.shape)
            tx = np.asarray(tx_source)
            mask = np.asarray(phantom.mask, dtype=bool)
            active_tx = np.abs(tx)[mask]
            if active_tx.size:
                effective = rf_peak_gauss * active_tx
                effective_text = (
                    f"; effective B1+ {effective.min():.5g}–"
                    f"{effective.max():.5g} G in active voxels"
                )
        self.waveform_nucleus_label.setText(f"Conversion: {nucleus}")
        self.waveform_value_summary.setText(
            f"Used physical values ({nucleus}): nominal peak B1 "
            f"{rf_peak_gauss:.5g} G{effective_text}; peak gradients "
            f"Gx {gradient_peaks['x']:.5g}, Gy {gradient_peaks['y']:.5g}, "
            f"Gz {gradient_peaks['z']:.5g} T/m. These waveform arrays and the "
            "per-voxel max-B1 map are included in result exports."
        )

    def _show_sequence_message(self, message):
        self.sequence_summary_table.setVisible(False)
        self.sequence_info.setVisible(True)
        self.sequence_info.setText(str(message))

    def _set_sequence_summary(self, rows, plain_text):
        """Show only the sequence name while retaining the full text for APIs."""
        sequence_rows = [
            ("Sequence name", value)
            for parameter, value in rows
            if str(parameter).strip().lower() in {"sequence", "sequence name"}
        ]
        if not sequence_rows and self.program is not None:
            sequence_rows = [("Sequence name", self.program.source)]
        table = self.sequence_summary_table
        table.setRowCount(len(sequence_rows))
        for row, (parameter, value) in enumerate(sequence_rows):
            parameter_item = QTableWidgetItem(str(parameter))
            value_item = QTableWidgetItem(str(value))
            parameter_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            value_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            table.setItem(row, 0, parameter_item)
            table.setItem(row, 1, value_item)
        self._fit_summary_table(table, 340)
        self.sequence_info.setText(str(plain_text))
        self.sequence_info.setVisible(False)
        table.setVisible(True)

    def _show_program(self, compiled=None):
        if self.program is None:
            return
        preserved_x_range_ms = None
        if self._preserve_sequence_plot_range_on_next_show:
            current_range = self.rf_plot.getViewBox().viewRange()[0]
            if len(current_range) == 2 and np.all(np.isfinite(current_range)):
                preserved_x_range_ms = tuple(float(value) for value in current_range)
        try:
            if compiled is None:
                compiled = self._acquisition_compiled
            if compiled is None:
                compiled = SequenceCompiler().compile_acquisition(self.program)
            self._acquisition_compiled = compiled
        except Exception as exc:
            self._show_sequence_message(f"Invalid sequence: {exc}")
            return
        summary_rows = []
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
                f"BW: {self.acquisition.sampling_bandwidth_hz / 1000:.3f} kHz"
                f"{offsets}"
            )
            summary_rows.extend(
                (
                    (
                        "Cartesian grid",
                        f"{self.acquisition.phase_matrix} × "
                        f"{self.acquisition.read_matrix}",
                    ),
                    (
                        "Sampling bandwidth",
                        f"{self.acquisition.sampling_bandwidth_hz / 1000:.3f} kHz",
                    ),
                )
            )
            if offsets:
                summary_rows.append(
                    (
                        "k-space offset",
                        f"({self.acquisition.kx_offset_cells:.3g}, "
                        f"{self.acquisition.ky_offset_cells:.3g}) cells",
                    )
                )
            if self.acquisition_frames is not None:
                acquisition_text += (
                    f"; frames={self.acquisition_frames.num_frames} "
                    f"({', '.join(self.acquisition_frames.varying_axes)})"
                )
                summary_rows.append(
                    (
                        "2D frames",
                        f"{self.acquisition_frames.num_frames} "
                        f"({', '.join(self.acquisition_frames.varying_axes)})",
                    )
                )
            if self.acquisition_volumes is not None:
                read_axis, phase_axis, partition_axis = (
                    self.acquisition_volumes.encoding_frame.axis_codes
                )
                acquisition_text += (
                    f"; 3D volumes={self.acquisition_volumes.num_volumes}, "
                    f"matrix={self.acquisition_volumes.matrix[0]}×"
                    f"{self.acquisition_volumes.matrix[1]}×"
                    f"{self.acquisition_volumes.matrix[2]}; encoding="
                    f"read {read_axis}, phase {phase_axis}, partition {partition_axis}"
                )
                summary_rows.append(
                    (
                        "3D volumes",
                        f"{self.acquisition_volumes.num_volumes}; matrix "
                        f"{self.acquisition_volumes.matrix[0]} × "
                        f"{self.acquisition_volumes.matrix[1]} × "
                        f"{self.acquisition_volumes.matrix[2]}; read {read_axis}, "
                        f"phase {phase_axis}, partition {partition_axis}",
                    )
                )
        elif self.spectroscopic_acquisition is not None:
            csi = self.spectroscopic_acquisition
            acquisition_text = (
                f"\nCSI grid: {csi.matrix[1]}×{csi.matrix[0]}; "
                f"spectral points={csi.spectral_points}; "
                f"repetitions={csi.num_repetitions}; "
                f"BW={csi.spectral_bandwidth_hz:.6g} Hz; "
                f"resolution={csi.spectral_resolution_hz:.6g} Hz"
            )
            summary_rows.extend(
                (
                    (
                        "CSI grid",
                        f"{csi.matrix[1]} × {csi.matrix[0]}; "
                        f"{csi.num_repetitions} repetition(s)",
                    ),
                    (
                        "Spectral sampling",
                        f"{csi.spectral_points} points; "
                        f"{csi.spectral_bandwidth_hz:.6g} Hz bandwidth; "
                        f"{csi.spectral_resolution_hz:.6g} Hz resolution",
                    ),
                )
            )
        elif self.spiral_acquisition is not None:
            spiral = self.spiral_acquisition
            self.dwell_info.setText(f"{spiral.dwell_s * 1e6:.3f} µs (actual)")
            acquisition_text = (
                f"\nSpiral: {spiral.matrix[1]}×{spiral.matrix[0]} target grid; "
                f"samples/frame={spiral.samples_per_frame}; "
                f"BW={spiral.sampling_bandwidth_hz / 1000:.3f} kHz; "
                f"frames={spiral.num_frames}"
            )
            summary_rows.extend(
                (
                    (
                        "Spiral target grid",
                        f"{spiral.matrix[1]} × {spiral.matrix[0]}; "
                        f"{spiral.num_frames} frame(s)",
                    ),
                    (
                        "Spiral sampling",
                        f"{spiral.samples_per_frame} samples/frame; "
                        f"{spiral.sampling_bandwidth_hz / 1000:.3f} kHz",
                    ),
                )
            )
        elif self.acquisition_note:
            acquisition_text = f"\n{self.acquisition_note}"
            summary_rows.append(("Acquisition", self.acquisition_note))
        definitions = dict(self.program.metadata.get("definitions", {}))
        end_image_spoilers = "EndImageSpoilerEndTimes" in definitions
        spoiler_end_times = np.asarray(
            definitions.get(
                "EndImageSpoilerEndTimes",
                definitions.get("SpoilerEndTimes", ()),
            ),
            dtype=float,
        ).reshape(-1)
        spoiler_end_times = spoiler_end_times[np.isfinite(spoiler_end_times)]
        spoiler_text = ""
        if spoiler_end_times.size:
            if end_image_spoilers:
                fov_cycles = definitions.get("EndImageSpoilerCyclesPerFOV", "?")
                voxel_cycles = definitions.get("EndImageSpoilerCyclesPerVoxel", 0)
                axes = definitions.get("EndImageSpoilerAxes", "xyz")
                spoiler_text = (
                    f"\nEnd-image spoilers: {spoiler_end_times.size}; "
                    f"{voxel_cycles} cycles/voxel + {fov_cycles} cycles/FOV "
                    f"on {axes}"
                )
                summary_rows.append(
                    (
                        "End-image spoilers",
                        f"{spoiler_end_times.size}; {voxel_cycles} cycles/voxel + "
                        f"{fov_cycles} cycles/FOV on {axes}",
                    )
                )
            else:
                slice_cycles = definitions.get("SpoilerCyclesPerSlice", "?")
                voxel_cycles = definitions.get("SpoilerCyclesPerVoxel", "?")
                axes = definitions.get("SpoilerAxes", "?")
                spoiler_text = (
                    f"\nSpoilers: {spoiler_end_times.size}; "
                    f"{slice_cycles} cycles/slice, {voxel_cycles} cycles/voxel "
                    f"on {axes}"
                )
                summary_rows.append(
                    (
                        "Gradient spoilers",
                        f"{spoiler_end_times.size}; {slice_cycles} cycles/slice, "
                        f"{voxel_cycles} cycles/voxel on {axes}",
                    )
                )
        acquisition_interval_text = ""
        if "AcquisitionInterval" in definitions:
            actual_interval_ms = float(definitions["AcquisitionInterval"]) * 1000.0
            minimum_interval_ms = (
                float(definitions.get("MinimumAcquisitionInterval", 0.0)) * 1000.0
            )
            acquisition_interval_text = (
                f"\nAcquisition interval: {actual_interval_ms:.4g} ms "
                f"start-to-start; minimum {minimum_interval_ms:.4g} ms"
            )
            summary_rows.append(
                (
                    "Acquisition interval",
                    f"{actual_interval_ms:.4g} ms start-to-start; "
                    f"minimum {minimum_interval_ms:.4g} ms",
                )
            )
        interval_label = (
            "acquisition intervals"
            if compiled.metadata.get("acquisition_only")
            else "intervals"
        )
        plain_text = (
            f"{self.program.source}\nDuration: {self.program.duration_s * 1000:.3f} ms\n"
            f"Events: {len(self.program.events)}, {interval_label}: "
            f"{compiled.n_intervals}, "
            f"ADC samples: {compiled.adc_times_s.size}{acquisition_text}"
            f"{acquisition_interval_text}{spoiler_text}"
        )
        base_rows = [
            ("Sequence", self.program.source),
            ("Duration", f"{self.program.duration_s * 1000:.3f} ms"),
            (
                "Timeline",
                f"{len(self.program.events)} events; {compiled.n_intervals} "
                f"{interval_label}; {compiled.adc_times_s.size} ADC samples",
            ),
        ]
        if "FlipAngleDeg" in definitions:
            flip_angles = np.asarray(definitions["FlipAngleDeg"], dtype=float).reshape(
                -1
            )
            flip_text = ", ".join(f"{value:.4g}°" for value in flip_angles)
            if bool(definitions.get("UseErnstAngle", False)):
                flip_text += " (Ernst angle)"
            summary_rows.append(("Flip angle", flip_text))
        if "RFSpoiling" in definitions:
            rf_spoiling_text = "Off"
            if bool(definitions["RFSpoiling"]):
                rf_spoiling_text = (
                    f"On; {float(definitions.get('RFSpoilingIncrementDeg', 0.0)):.4g}° "
                    "increment"
                )
            summary_rows.append(("RF spoiling", rf_spoiling_text))
        if bool(definitions.get("UseErnstAngle", False)):
            summary_rows.append(
                (
                    "Ernst model",
                    f"effective T1 "
                    f"{1000.0 * float(definitions['ErnstEffectiveT1']):.4g} ms; "
                    "T2 not used",
                )
            )
        self._set_sequence_summary(base_rows + summary_rows, plain_text)
        self._update_waveform_value_summary()
        self.rf_plot.clear()
        self._rf_phase_view.clear()
        self.gradient_plot.clear()
        self._rf_waveform_item = self.rf_plot.plot(
            np.empty(0), np.empty(0), pen=pg.mkPen("m", width=1.5)
        )
        self._rf_phase_item = pg.PlotDataItem(
            np.empty(0), np.empty(0), pen=pg.mkPen("c", width=1.25)
        )
        self._rf_phase_view.addItem(self._rf_phase_item)
        self._rf_phase_view.setYRange(-180.0, 180.0, padding=0)
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
            adc_indices = _representative_sample_indices(
                compiled.adc_times_s, max_samples=5000
            )
            displayed_adc_times = compiled.adc_times_s[adc_indices]
            self.gradient_plot.plot(
                displayed_adc_times * 1000,
                np.zeros_like(displayed_adc_times),
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
            marker.setToolTip(
                "End-image spoiler" if end_image_spoilers else "Gradient spoiler"
            )
            self.gradient_plot.addItem(marker)
            self._sequence_spoiler_markers.append(marker)
        self.rf_plot.addItem(self.rf_progress_cursor)
        self.gradient_plot.addItem(self.gradient_progress_cursor)
        self._sequence_plot_window_s = None
        duration = max(float(self.program.duration_s), 1e-9)
        duration_ms = duration * 1000.0
        if preserved_x_range_ms is None:
            start_ms, end_ms = 0.0, duration_ms
        else:
            start_ms, end_ms = preserved_x_range_ms
            visible_span_ms = min(max(end_ms - start_ms, 1e-9), duration_ms)
            start_ms = min(max(start_ms, 0.0), duration_ms - visible_span_ms)
            end_ms = start_ms + visible_span_ms
        self.rf_plot.setXRange(start_ms, end_ms, padding=0)
        self._refresh_sequence_waveforms(start_ms / 1000.0, end_ms / 1000.0)
        self._update_waveform_y_ranges()
        self._set_sequence_cursor(0.0)
        self._update_run_action_availability()
        self._emit_physical_b1_changed()

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
        rf_scale, gradient_scale, *_labels = self._waveform_scales()
        rf_x, rf_y = _event_step_plot_data(
            self.program.rf_events,
            samples_attribute="samples_hz",
            start_s=window[0],
            end_s=window[1],
            scale=rf_scale,
            magnitude=True,
        )
        self._rf_waveform_item.setData(rf_x, rf_y, connect="finite")
        phase_x, phase_y = _rf_phase_plot_data(
            self.program.rf_events,
            start_s=window[0],
            end_s=window[1],
        )
        self._rf_phase_item.setData(phase_x, phase_y, connect="finite")
        gradients = self.program.gradient_events
        for axis in "xyz":
            events = tuple(event for event in gradients if event.axis == axis)
            grad_x, grad_y = _event_step_plot_data(
                events,
                samples_attribute="samples_hz_per_m",
                start_s=window[0],
                end_s=window[1],
                scale=gradient_scale,
            )
            self._gradient_waveform_items[axis].setData(
                grad_x, grad_y, connect="finite"
            )

    def _build_phantom(self):
        source_index = self.object_source.currentIndex()
        if source_index == 0:
            designed = self._selected_designed_phantom()
            if designed is None:
                raise ValueError(self.NO_PHANTOM_MESSAGE)
            self.phantom = designed
            if isinstance(self.phantom, (SpectralPhantom, DynamicSpectralPhantom)):
                self.phantom.field_strength = self.field_strength_t.value()
                self.phantom.nucleus = self.nucleus.currentText()
                design_metadata = self.phantom.metadata.get("phantom_design")
                if isinstance(design_metadata, dict):
                    design_metadata["field_strength_t"] = self.field_strength_t.value()
                    design_metadata["nucleus"] = self.nucleus.currentText()
            else:
                # Conventional phantoms store their frequency maps in Hz, but
                # the selected scanner field and nucleus still belong in the
                # run metadata and all exported result formats.
                self.phantom.metadata["field_strength_t"] = (
                    self.field_strength_t.value()
                )
                self.phantom.metadata["field_strength"] = self.field_strength_t.value()
                self.phantom.metadata["nucleus"] = self.nucleus.currentText()
            return
        if source_index == 2:
            raise ValueError(
                "Spin probe mode does not create a simulation phantom. "
                "Use Run spectral probe or Run geometry probe instead."
            )
        n = self.matrix_size.value()
        nz = self.z_matrix_size.value()
        shape = (n, n, nz)
        fov = self.fov_mm.value() / 1000.0
        fov_z = self.fov_z_mm.value() / 1000.0
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

    def _animation_request(self):
        """Return resolution, memory-bounded frame limit, and status text."""
        if (
            not self.animation_enabled.isChecked()
            or self.phantom is None
            or self.program is None
        ):
            return None, 0, ""
        resolution_s = float(self.animation_time_resolution_ms.value()) * 1e-3
        duration_s = float(self.program.duration_s)
        complete_steps = int(np.floor(duration_s / resolution_s))
        last_regular_s = complete_steps * resolution_s
        endpoint_tolerance = max(1e-14, duration_s * 1e-12)
        requested = complete_steps + 1
        if abs(last_regular_s - duration_s) > endpoint_tolerance:
            requested += 1
        requested = max(2, requested)
        # The replay stores checkpoints in float32 before applying the selected
        # final animation dtype. Account for both the replay's overlapping state
        # arrays and the final combined/pool animation arrays.
        state_sets = 2
        stored_state_sets = 1
        if isinstance(self.phantom, DynamicSpectralPhantom):
            # Two pool states plus their combined spatial state coexist when
            # the dynamic result is assembled.
            state_sets = 3
            stored_state_sets = 3
        elif isinstance(self.phantom, SpectralPhantom):
            state_sets = max(2, 2 * int(self.phantom.n_species) + 3)
            stored_state_sets = 1 + int(self.phantom.n_species)
        storage_itemsize = np.dtype(self.animation_storage_dtype.currentText()).itemsize
        bytes_per_frame = max(
            1,
            int(self.phantom.nvoxels)
            * 3
            * (4 * state_sets + storage_itemsize * stored_state_sets),
        )
        temporary_budget_bytes = self.animation_memory_budget_bytes
        permitted = int(temporary_budget_bytes // bytes_per_frame)
        if permitted < 2:
            return (
                None,
                0,
                (
                    "3D animation disabled for this object because even two "
                    "temporary checkpoint states would exceed the 512 MiB "
                    "animation budget."
                ),
            )
        actual = min(requested, int(permitted))
        note = ""
        if actual < requested:
            effective_resolution_ms = duration_s * 1000.0 / max(1, actual - 1)
            note = (
                f"3D animation limited to {actual} states for this object "
                f"(effective time resolution about {effective_resolution_ms:.4g} "
                f"ms; {requested} requested) to bound temporary checkpoint RAM "
                f"({temporary_budget_bytes / 1024**2:.0f} MiB limit)."
            )
        return resolution_s, actual, note

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
        if (
            self._probe_frequency_defaults_initialized
            or self.program is None
            or not self.program.rf_events
        ):
            return
        # Use a stable, symmetric default that is immediately comparable
        # across sequences. Users can still narrow or expand it afterwards.
        self.probe_frequency_units.setCurrentText("Hz")
        self.probe_ppm_min.setValue(-2500.0)
        self.probe_ppm_max.setValue(2500.0)
        self._probe_frequency_defaults_initialized = True

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
            spoiler_times = definitions.get(
                "EndImageSpoilerEndTimes",
                definitions.get("SpoilerEndTimes", ()),
            )
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
            window_center = float(
                getattr(
                    phantom,
                    "spectral_window_center_ppm",
                    phantom.spectral_reference_ppm,
                )
            )
            relative_center = window_center - float(phantom.spectral_reference_ppm)
            frequency_min = relative_center - half_bandwidth
            frequency_max = relative_center + half_bandwidth
            if self.probe_frequency_units.currentText() == "Hz":
                factor = self._probe_hz_per_ppm()
                frequency_min *= factor
                frequency_max *= factor
            self.probe_ppm_min.setValue(frequency_min)
            self.probe_ppm_max.setValue(frequency_max)
        points = int(getattr(phantom, "spectral_points", self.probe_points.value()))
        if self.probe_points.minimum() <= points <= self.probe_points.maximum():
            self.probe_points.setValue(points)

    def _can_start_probe(self):
        if self.sequence_source.currentIndex() in self.GENERATED_SOURCES:
            if not self._ensure_current_generated_sequence():
                QMessageBox.warning(
                    self,
                    "Sequence generation failed",
                    self._generation_error
                    or "Generate a valid sequence before running a spin probe.",
                )
                return False
        if self.program is None:
            QMessageBox.warning(
                self,
                "No sequence",
                "Choose or load a sequence, or click Generate sequence first.",
            )
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
            if self.object_source.currentIndex() == 2:
                self.phantom = self._selected_designed_phantom()
                if self.phantom is None:
                    raise ValueError(
                        "Create or load a phantom in the Phantom tab before "
                        "running a geometry probe"
                    )
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
        self.probe_playback_mode.setEnabled(False)
        self.run_probe_button.setEnabled(False)
        self.run_geometry_probe_button.setEnabled(False)
        self.cancel_probe_button.setEnabled(True)
        self.probe_status.setText(f"Preparing {label} probe…")
        self.probe_result = None
        self._probe_started_at = time.monotonic()
        self._probe_started_at_utc = datetime.now(timezone.utc)
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
            lambda result, axis=display_axis, unit=self.probe_frequency_units.currentText(), mode=label: (
                self._probe_finished(result, axis, unit, mode)
            )
        )
        self.probe_worker.failed.connect(self._probe_failed)
        self.probe_worker.start()

    def _cancel_probe(self):
        if self.probe_worker is not None:
            self.probe_worker.request_cancel()
            self.probe_status.setText("Cancelling spin probe…")

    def _cancel_active_run(self):
        """Cancel whichever sequence or probe worker currently owns the run area."""
        probe_running = self.probe_worker is not None and self.probe_worker.isRunning()
        sequence_running = self.worker is not None and self.worker.isRunning()
        if probe_running:
            self._cancel_probe()
        if sequence_running:
            self._cancel()

    def _probe_progress(self, done, total):
        self.probe_status.setText(f"Probe chunk {done}/{total}")

    def _probe_status_update(self, message):
        message = str(message)
        self.probe_status.setText(message)
        logger = getattr(self.window(), "log_message", None)
        if callable(logger):
            logger(f"Sequence probe: {message}")

    def _probe_failed(self, message):
        probe_selected = self.object_source.currentIndex() == 2
        self.run_probe_button.setEnabled(probe_selected)
        self.run_geometry_probe_button.setEnabled(probe_selected)
        self.cancel_probe_button.setEnabled(False)
        self.probe_status.setText(message)
        if message != "Probe simulation cancelled":
            self._show_simulation_failure("Spin probe failed", message)

    def _probe_finished(self, result, frequency_axis, frequency_unit, mode=None):
        self._stop_probe_playback()
        self.probe_result = result
        finished_at = datetime.now(timezone.utc)
        if self._probe_started_at is not None:
            result.metadata.setdefault(
                "simulation_wall_time_s",
                max(0.0, time.monotonic() - self._probe_started_at),
            )
        result.metadata.setdefault(
            "simulation_finished_at_utc", finished_at.isoformat()
        )
        frequency_axis = np.asarray(frequency_axis, dtype=float)
        result.metadata["frequency_axis_unit"] = str(frequency_unit)
        result.metadata[f"frequency_offsets_{str(frequency_unit).lower()}"] = (
            frequency_axis
        )
        if mode is not None:
            result.metadata["ui_probe_mode"] = str(mode)
        probe_selected = self.object_source.currentIndex() == 2
        self.run_probe_button.setEnabled(probe_selected)
        self.run_geometry_probe_button.setEnabled(probe_selected)
        self.cancel_probe_button.setEnabled(False)
        self.probe_status.setText("Spin probe complete")
        self._show_probe_result()
        self._register_session_probe_run(result, mode=mode)

    def _show_probe_result(self):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            return
        previous_result = self.probe_spectrum_viewer.result
        previous_time_s = None
        if previous_result is not None and previous_result.time_s.size:
            previous_index = int(
                np.clip(
                    self.probe_spectrum_viewer.time_index,
                    0,
                    previous_result.time_s.size - 1,
                )
            )
            previous_time_s = float(previous_result.time_s[previous_index])
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
        has_complete_timeline = (
            result.metadata.get("stored_timeline") != "configured_checkpoints"
        )
        model = self.probe_playback_mode.model()
        for index in (1, 2):
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(has_complete_timeline)
        if not has_complete_timeline:
            self.probe_playback_mode.setCurrentIndex(0)
            self.probe_playback_mode.setToolTip(
                "This memory-efficient probe stores only the configured checkpoints."
            )
        else:
            self.probe_playback_mode.setToolTip(
                "Use the configured checkpoint view, skip directly between ADC "
                "samples, or inspect every stored simulation state."
            )
        self.probe_playback_mode.setEnabled(True)
        if previous_time_s is None:
            self._configure_probe_playback_mode()
            initial_index = 1 if self._probe_playback_indices.size > 1 else 0
            self._set_probe_time_index(initial_index)
        else:
            nearest_result_index = int(
                np.argmin(np.abs(np.asarray(result.time_s) - previous_time_s))
            )
            self._configure_probe_playback_mode(nearest_result_index)

    @staticmethod
    def _matching_probe_time_indices(result_times_s, selected_times_s):
        result_times = np.asarray(result_times_s, dtype=float)
        selected_times = np.asarray(selected_times_s, dtype=float).reshape(-1)
        if result_times.size == 0 or selected_times.size == 0:
            return np.zeros(0, dtype=np.int64)
        right = np.searchsorted(result_times, selected_times, side="left")
        right = np.clip(right, 0, result_times.size - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(selected_times - result_times[left]) < np.abs(
            result_times[right] - selected_times
        )
        indices = np.where(choose_left, left, right)
        return np.unique(indices.astype(np.int64, copy=False))

    def _configured_probe_playback_indices(self):
        result = self.probe_result
        if result is None:
            return np.zeros(0, dtype=np.int64)
        configured_times = result.metadata.get("configured_playback_times_s")
        if configured_times is None:
            return np.arange(result.time_s.size, dtype=np.int64)
        return self._matching_probe_time_indices(result.time_s, configured_times)

    def _adc_probe_playback_indices(self):
        result = self.probe_result
        if result is None:
            return np.zeros(0, dtype=np.int64)
        return self._matching_probe_time_indices(
            result.time_s,
            result.metadata.get("adc_times_s", ()),
        )

    def _adc_probe_playback_clock_ms(self, indices):
        result = self.probe_result
        if result is None or len(indices) == 0:
            return np.zeros(0, dtype=float)
        actual_times = np.asarray(result.time_s, dtype=float)[indices]
        if actual_times.size < 2:
            return np.zeros(actual_times.size, dtype=float)
        adc_times = np.asarray(result.metadata.get("adc_times_s", ()), dtype=float)
        event_indices = np.asarray(
            result.metadata.get("adc_event_indices", ()), dtype=np.int64
        )
        dwell_s = np.asarray(result.metadata.get("adc_sample_dwell_s", ()), dtype=float)
        if not (
            adc_times.size == event_indices.size == dwell_s.size == actual_times.size
        ):
            return (actual_times - actual_times[0]) * 1000.0
        delta_s = np.diff(actual_times)
        event_changed = np.diff(event_indices) != 0
        transition_s = np.minimum(dwell_s[:-1], dwell_s[1:])
        delta_s[event_changed] = transition_s[event_changed]
        return np.concatenate(([0.0], np.cumsum(delta_s))) * 1000.0

    def _configure_probe_playback_mode(self, preserve_result_index=None):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            self._probe_playback_indices = np.zeros(0, dtype=np.int64)
            self._probe_playback_clock_ms = np.zeros(0, dtype=float)
            self.probe_time_control.setEnabled(False)
            self.probe_time_control.set_time_range(None)
            return
        mode = self.probe_playback_mode.currentIndex()
        if mode == 0:
            indices = self._configured_probe_playback_indices()
        elif mode == 1:
            indices = self._adc_probe_playback_indices()
        else:
            indices = np.arange(result.time_s.size, dtype=np.int64)
        self._probe_playback_indices = np.asarray(indices, dtype=np.int64)
        display_times = np.asarray(result.time_s, dtype=float)[indices]
        if mode == 1:
            self._probe_playback_clock_ms = self._adc_probe_playback_clock_ms(indices)
        else:
            self._probe_playback_clock_ms = display_times * 1000.0
        has_times = indices.size > 0
        self.probe_time_control.setEnabled(has_times)
        self.probe_time_control.set_time_range(display_times if has_times else None)
        self.probe_adc_status.setVisible(mode == 2)
        if not has_times:
            self.probe_adc_status.setText("ADC: no samples")
            return
        if preserve_result_index is None:
            display_index = 0
        else:
            display_index = int(np.argmin(np.abs(indices - preserve_result_index)))
        self.probe_time_control.set_time_index(display_index)
        self._update_probe_vector(int(indices[display_index]))

    def _probe_playback_mode_changed(self, _mode):
        result_index = None
        if self._probe_playback_indices.size:
            display_index = int(
                np.clip(
                    self.probe_time_control.time_slider.value(),
                    0,
                    self._probe_playback_indices.size - 1,
                )
            )
            result_index = int(self._probe_playback_indices[display_index])
        self._stop_probe_playback()
        self._configure_probe_playback_mode(result_index)

    def _probe_result_index(self, playback_index=None):
        if self._probe_playback_indices.size == 0:
            return None
        if playback_index is None:
            playback_index = self.probe_time_control.time_slider.value()
        playback_index = int(
            np.clip(playback_index, 0, self._probe_playback_indices.size - 1)
        )
        return int(self._probe_playback_indices[playback_index])

    def _set_probe_time_index(self, time_index):
        result = self.probe_result
        if (
            result is None
            or result.time_s.size == 0
            or self._probe_playback_indices.size == 0
        ):
            return
        time_index = int(np.clip(time_index, 0, self._probe_playback_indices.size - 1))
        self.probe_time_control.set_time_index(time_index)
        self._update_probe_vector(int(self._probe_playback_indices[time_index]))
        if self.probe_playback_timer.isActive():
            self._reset_probe_playback_anchor(time_index)

    def _reset_probe_playback_anchor(self, time_index=None):
        if self._probe_playback_indices.size == 0:
            self._probe_playback_anchor_wall = None
            self._probe_playback_anchor_time_ms = None
            return
        if time_index is None:
            time_index = self.probe_time_control.time_slider.value()
        time_index = int(np.clip(time_index, 0, self._probe_playback_indices.size - 1))
        self._probe_playback_anchor_wall = time.monotonic()
        self._probe_playback_anchor_time_ms = float(
            self._probe_playback_clock_ms[time_index]
        )

    def _probe_playback_toggled(self, playing):
        result = self.probe_result
        if not playing:
            self._stop_probe_playback()
            return
        if result is None or self._probe_playback_indices.size < 2:
            self._stop_probe_playback()
            return

        time_index = self.probe_time_control.time_slider.value()
        if time_index >= self._probe_playback_indices.size - 1:
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
        if self.probe_result is None or self._probe_playback_indices.size == 0:
            return
        self.probe_magnetization_viewer._clear_path()
        self._set_probe_time_index(0)

    def _probe_playback_speed_changed(self, _speed):
        if self.probe_playback_timer.isActive():
            self._reset_probe_playback_anchor()

    def _advance_probe_playback(self):
        result = self.probe_result
        if result is None or self._probe_playback_indices.size < 2:
            self._stop_probe_playback()
            return
        if (
            self._probe_playback_anchor_wall is None
            or self._probe_playback_anchor_time_ms is None
        ):
            self._reset_probe_playback_anchor()
            return

        time_ms = self._probe_playback_clock_ms
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
        self._set_probe_time_index(time_index)

    def _probe_view_changed(self, _index):
        if self.probe_result is None or self.probe_result.time_s.size == 0:
            return
        time_index = self._probe_result_index()
        if time_index is not None:
            self._update_probe_vector(time_index)

    def _update_probe_vector(self, time_index=None):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            return
        if time_index is None:
            time_index = self.probe_magnetization_viewer.time_slider.value()
        time_index = int(np.clip(time_index, 0, result.time_s.size - 1))
        self._update_probe_adc_status(time_index)
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

    def _update_probe_adc_status(self, time_index):
        result = self.probe_result
        if result is None or result.time_s.size == 0:
            self.probe_adc_status.setText("ADC: off")
            return
        time_s = float(result.time_s[time_index])
        windows = np.asarray(result.metadata.get("adc_windows_s", ()), dtype=float)
        active = False
        if windows.size:
            windows = windows.reshape(-1, 2)
            tolerance = max(1e-15, abs(time_s) * 1e-12)
            active = bool(
                np.any(
                    (windows[:, 0] - tolerance <= time_s)
                    & (time_s <= windows[:, 1] + tolerance)
                )
            )
        self.probe_adc_status.setText("ADC: on" if active else "ADC: off")

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
        if self.object_source.currentIndex() == 2:
            QMessageBox.information(
                self,
                "Spin probe mode",
                "Use Run spectral probe or Run geometry probe in the spin-probe "
                "configuration panel.",
            )
            return
        if self.sequence_source.currentIndex() in self.GENERATED_SOURCES:
            if not self._ensure_current_generated_sequence():
                QMessageBox.warning(
                    self,
                    "Sequence generation failed",
                    self._generation_error
                    or "Generate a valid sequence before running the simulation.",
                )
                return
        if self.program is None:
            QMessageBox.warning(
                self,
                "No sequence",
                "Choose or load a sequence, or click Generate sequence first.",
            )
            return
        try:
            self._build_phantom()
            checkpoints = self._checkpoint_seconds()
            (
                animation_time_resolution_s,
                animation_maximum_frames,
                animation_note,
            ) = self._animation_request()
        except Exception as exc:
            if str(exc) == self.NO_PHANTOM_MESSAGE:
                self._show_no_phantom_dialog()
            else:
                QMessageBox.critical(self, "Invalid simulation", str(exc))
            return
        if not self._confirm_generated_sequence_fov():
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        work_units = self._estimated_work_units()
        self._clear_previous_simulation_views()
        self.view_stack.setCurrentWidget(self.normal_signal_page)
        self.views.setCurrentIndex(self.signal_tab_index)
        self.progress.setRange(0, work_units)
        self.progress.setValue(0)
        self.progress.setFormat("0% · Estimating remaining time…")
        self._simulation_started_at = time.monotonic()
        self._simulation_started_at_utc = datetime.now(timezone.utc).isoformat()
        self._simulation_progress_started_at = None
        self._simulation_last_progress_at = None
        self._simulation_last_progress_done = 0
        self._simulation_last_progress_total = work_units
        self._simulation_progress_rate = None
        self.simulation_time_timer.start()
        self._update_simulation_time_label()
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
            spin_sampling=self._configured_spin_sampling(),
            spoiler_mode=self.spoiler_mode,
            animation_time_resolution_s=animation_time_resolution_s,
            animation_maximum_frames=animation_maximum_frames,
            animation_storage_dtype=self.animation_storage_dtype.currentText(),
            animation_note=animation_note,
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
        total = max(1, int(total))
        done = min(total, max(0, int(done)))
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self._simulation_last_progress_total = total
        percent = min(100, int(round(100.0 * done / total)))
        now = time.monotonic()
        if self._simulation_started_at is None:
            self._simulation_started_at = now
        if self._simulation_progress_started_at is None:
            self._simulation_progress_started_at = now
            self._simulation_last_progress_at = now
            self._simulation_last_progress_done = done
        else:
            delta_done = done - self._simulation_last_progress_done
            delta_s = now - self._simulation_last_progress_at
            if delta_done > 0 and delta_s > 0.0:
                instantaneous_rate = delta_done / delta_s
                if self._simulation_progress_rate is None:
                    self._simulation_progress_rate = instantaneous_rate
                else:
                    self._simulation_progress_rate = (
                        0.25 * instantaneous_rate
                        + 0.75 * self._simulation_progress_rate
                    )
                self._simulation_last_progress_at = now
                self._simulation_last_progress_done = done
        if done >= total:
            progress_text = "100% · Finishing…"
            eta_text = None
        elif self._simulation_progress_rate is not None:
            remaining_s = (total - done) / self._simulation_progress_rate
            eta_text = _format_duration(remaining_s)
            progress_text = f"{percent}% · ETA {eta_text}"
        else:
            eta_text = None
            progress_text = f"{percent}% · Estimating remaining time…"
        self.progress.setFormat(progress_text)
        if isinstance(self.phantom, DynamicSpectralPhantom):
            unit = "Interval"
        elif isinstance(self.phantom, SpectralPhantom):
            unit = "Component"
        else:
            unit = "Chunk"
        status_text = f"{unit} {done}/{total} · {percent}%"
        if eta_text is not None:
            status_text += f" · approximately {eta_text} remaining"
        self.status.setText(status_text)
        self._update_simulation_time_label(now=now)

    def _estimated_remaining_seconds(self, now=None):
        """Estimate outstanding wall time from the latest measured work rate."""
        if self._simulation_progress_rate is None:
            return None
        now = time.monotonic() if now is None else float(now)
        done = float(self._simulation_last_progress_done)
        total = float(max(1, self._simulation_last_progress_total))
        if self._simulation_last_progress_at is not None:
            done += self._simulation_progress_rate * max(
                0.0, now - self._simulation_last_progress_at
            )
        return max(0.0, total - done) / self._simulation_progress_rate

    def _update_simulation_time_label(self, now=None):
        """Refresh elapsed and remaining wall-clock time while a run is active."""
        if self._simulation_started_at is None:
            return
        now = time.monotonic() if now is None else float(now)
        elapsed_s = max(0.0, now - self._simulation_started_at)
        remaining_s = self._estimated_remaining_seconds(now)
        remaining_text = (
            f"approximately {_format_duration(remaining_s)}"
            if remaining_s is not None
            else "estimating…"
        )
        self.simulation_time_label.setText(
            f"Elapsed: {_format_duration(elapsed_s)} · Remaining: {remaining_text}"
        )

    def _estimated_work_units(self):
        if isinstance(self.phantom, DynamicSpectralPhantom):
            compiled = self._acquisition_compiled
            if compiled is None:
                compiled = SequenceCompiler().compile_acquisition(self.program)
                self._acquisition_compiled = compiled
            return max(1, compiled.n_intervals)
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
            chunk_voxels = max(
                1, 65536 // self._configured_spin_sampling().spins_per_voxel
            )
        return max(1, int(np.ceil(self.phantom.n_active / chunk_voxels)))

    def _preview_chunk_voxels(self):
        if not self.live_preview_enabled or isinstance(
            self.phantom, (SpectralPhantom, DynamicSpectralPhantom)
        ):
            return None
        max_parent_voxels = max(
            1, 65536 // self._configured_spin_sampling().spins_per_voxel
        )
        return min(
            max_parent_voxels,
            max(1, int(np.ceil(self.phantom.n_active / 32))),
        )

    def _configured_spin_sampling(self):
        counts = (
            self.subvoxel_spin_counts
            if self.spoiler_mode == "gradient"
            or self.simulator.dynamic_sequence_kernel == "metal_hybrid"
            else (1, 1, 1)
        )
        return SpinSampling(counts, method=self.subvoxel_sampling_method)

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

    def set_dynamic_sequence_kernel(self, kernel):
        """Apply the dynamic two-pool kernel setting to future runs."""
        if kernel not in {
            "optimized",
            "native_parallel",
            "native_serial",
            "metal_hybrid",
            "reference",
        }:
            raise ValueError("unsupported dynamic sequence kernel")
        self.simulator.dynamic_sequence_kernel = kernel

    def set_sequence_timestep_us(self, value):
        """Apply the persistent RF-active time-step default."""
        value = float(value)
        if not np.isfinite(value) or not 0.1 <= value <= 1000.0:
            raise ValueError("sequence time step must be between 0.1 and 1000 µs")
        self.simulation_timestep_us.setValue(value)

    def set_spoiler_configuration(self, mode, counts_xyz, sampling_method=None):
        """Apply persistent ideal or physical-gradient spoiler settings."""
        mode = str(mode).strip().lower()
        if mode not in {"ideal", "gradient"}:
            raise ValueError("spoiler mode must be 'ideal' or 'gradient'")
        method = (
            self.subvoxel_sampling_method
            if sampling_method is None
            else str(sampling_method).strip().lower()
        )
        sampling = SpinSampling(tuple(counts_xyz), method=method)
        self.spoiler_mode = mode
        self.subvoxel_spin_counts = sampling.counts_xyz
        self.subvoxel_sampling_method = sampling.method
        auto_changed = self._apply_flash_auto_spoilers()
        self._update_flash_spoiler_info()
        if auto_changed and self.sequence_source.currentIndex() == self.FLASH_SOURCE:
            self._request_generated_sequence_refresh()
        self._update_ss_bssfp_spoiler_info()
        self._update_spoiling_quality()

    def set_thread_configuration(self, mode, manual_thread_count):
        """Apply automatic or manual native worker selection."""
        if mode not in {"automatic", "manual"}:
            raise ValueError("thread mode must be 'automatic' or 'manual'")
        requested = None if mode == "automatic" else manual_thread_count
        self.simulator.num_threads = resolve_num_threads(requested)

    def set_animation_memory_budget_bytes(self, value):
        """Apply the persistent temporary RAM cap to future animation replays."""
        value = int(value)
        minimum = 16 * 1024**2
        maximum = 1024 * 1024**3
        if not minimum <= value <= maximum:
            raise ValueError("animation memory budget must be between 16 MiB and 1 TiB")
        self.animation_memory_budget_bytes = value

    def set_scanner_parameters(self, parameters):
        """Apply scanner hardware limits to generated sequences and exports."""
        self.scanner_parameters = ScannerParameters.from_mapping(parameters)
        for prefix in (
            "epi",
            "csi",
            "flash",
            "bssfp",
            "ss_bssfp",
            "radial_me",
            "me_bssfp",
        ):
            if hasattr(self, f"{prefix}_rf_pulse_type"):
                self._update_shared_rf_controls(prefix)
        if hasattr(self, "sequence_source"):
            self._request_generated_sequence_refresh()

    def _set_sequence_cursor(self, fraction):
        duration_ms = self.program.duration_s * 1000.0 if self.program else 0.0
        position = float(np.clip(fraction, 0.0, 1.0)) * duration_ms
        self.rf_progress_cursor.setPos(position)
        self.gradient_progress_cursor.setPos(position)

    def _clear_previous_simulation_views(self):
        """Remove stale result data before the next simulation starts."""
        self._active_session_run_id = None
        self.result = None
        self.magnetization_animation = None
        self._split_csi_data = None
        self.signal_plot.clear()
        self.signal_plot.setTitle("Received ADC signal — current simulation")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        self.kspace_view.clear()
        self.reconstruction_view.clear()
        self.kspace_zoom_info.setText("Zoom: —")
        self.reconstruction_zoom_info.setText("Zoom: —")
        if self.live_preview_enabled:
            message = "Waiting for data from the current simulation…"
        else:
            message = "Current simulation running; live preview is disabled"
        self.spectrum_info.setText(message)
        self.kspace_info.setText(message)
        self.reconstruction_info.setText(message)
        self.reconstruction_explorer.clear(message)
        self.magnetization_animation_viewer.clear(
            "Animation states will be available after the simulation completes."
            if self.animation_enabled.isChecked()
            else "3D animation was disabled for this simulation."
        )
        self._set_sequence_cursor(0.0)

    def _preview(self, fraction, signal):
        """Render a throttled intermediate timeline, k-space and image state."""
        if not self.live_preview_enabled or self.program is None:
            return
        fraction = float(np.clip(fraction, 0.0, 1.0))
        self._set_sequence_cursor(fraction)
        if (
            self.acquisition is None
            and self.spectroscopic_acquisition is None
            and self.spiral_acquisition is None
        ):
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
        if isinstance(self.phantom, DynamicSpectralPhantom):
            visible_samples = acquired
            partial_signal[..., visible_samples:] = 0.0
            preview_summary = f"{visible_samples}/{adc_times.size} ADC samples"
        else:
            # Static and spectral previews accumulate complete-sequence signal
            # contributions one voxel chunk or spectral component at a time.
            visible_samples = adc_times.size
            preview_summary = f"{int(round(fraction * 100.0))}% of simulation work"
        self._show_live_signal(
            partial_signal,
            adc_times,
            visible_samples,
            preview_summary,
        )
        if self.spectroscopic_acquisition is not None:
            self._show_live_spectroscopy(partial_signal, preview_summary)
        elif self.spiral_acquisition is not None:
            self._show_live_spiral(partial_signal, preview_summary)
        else:
            self._show_live_cartesian(partial_signal, preview_summary)

    def _show_live_signal(self, signal, adc_times, visible_samples, preview_summary):
        """Plot only ADC data produced by the simulation currently running."""
        self.signal_plot.clear()
        self.signal_plot.setTitle("Live received ADC signal")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        visible_samples = min(int(visible_samples), int(adc_times.size))
        if visible_samples > 0:
            live_signal = np.asarray(signal)[..., :visible_samples]
            live_time_ms = np.asarray(adc_times[:visible_samples]) * 1000.0
            if live_signal.ndim == 1:
                self.signal_plot.plot(
                    live_time_ms, np.abs(live_signal), pen="w", name="Magnitude"
                )
                self.signal_plot.plot(
                    live_time_ms, live_signal.real, pen="g", name="Real"
                )
                self.signal_plot.plot(
                    live_time_ms, live_signal.imag, pen="r", name="Imaginary"
                )
            else:
                for coil, coil_signal in enumerate(live_signal):
                    self.signal_plot.plot(
                        live_time_ms,
                        np.abs(coil_signal),
                        pen=pg.intColor(coil, hues=live_signal.shape[0]),
                        name=f"Coil {coil + 1}",
                    )
        self.spectrum_info.setText(f"Current simulation: {preview_summary}")

    def _show_live_spectroscopy(self, signal, preview_summary):
        csi = self.spectroscopic_acquisition
        point = min(self.spectral_point_selector.value(), csi.spectral_points - 1)
        kspace = self._selected_csi_repetition(csi.reshape_signal(signal))
        fid = self._selected_csi_repetition(csi.reconstruct_spatial(signal))
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
            f"{preview_summary}"
        )
        self.reconstruction_info.setText("Live spatial 2D IFFT of CSI FID")

    def _show_live_cartesian(self, signal, preview_summary):
        selected = self.frame_selector.currentData()
        selected_index = max(
            0, 0 if selected is None or int(selected) < 0 else int(selected)
        )
        if self.acquisition_volumes is not None:
            volume = min(selected_index, self.acquisition_volumes.num_volumes - 1)
            frame = self.acquisition_volumes.volume_frame_indices[volume][
                self.acquisition_volumes.partition_matrix // 2
            ]
        else:
            frame = selected_index
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
        self.kspace_info.setText(f"Live k-space: {preview_summary}")
        reconstruction_kind = (
            "central-kz hybrid |IFFT2|"
            if self.acquisition_volumes is not None
            else "|IFFT2|"
        )
        self.reconstruction_info.setText(
            f"Live {reconstruction_kind} from {preview_summary}"
        )

    def _show_live_spiral(self, signal, preview_summary):
        spiral = self.spiral_acquisition
        selected = self.frame_selector.currentData()
        frame = max(0, 0 if selected is None or int(selected) < 0 else int(selected))
        result = SimpleNamespace(
            signal=signal,
            adc_gradient_moment_cyc_per_m=(
                self._acquisition_compiled.adc_gradient_moment_cyc_per_m
            ),
        )
        kspace = spiral.grid_kspace(result, frame)
        image = spiral.reconstruct(result, frame)
        if kspace.ndim == 3:
            kspace_display = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
            image_display = np.sqrt(np.sum(np.abs(image) ** 2, axis=0))
        else:
            kspace_display = np.abs(kspace)
            image_display = np.abs(image)
        self.kspace_view.setImage(np.log1p(kspace_display).T, autoLevels=True)
        self.reconstruction_view.setImage(image_display.T, autoLevels=True)
        self.kspace_info.setText(
            f"Live linearly gridded spiral k-space: {preview_summary}"
        )
        self.reconstruction_info.setText(
            f"Live spiral gridding + |IFFT2| from {preview_summary}"
        )

    def _cancel(self):
        if self.worker is not None:
            self.worker.request_cancel()
            self.status.setText("Cancelling after current chunk…")

    def closeEvent(self, event):
        process = self.script_process
        if process is not None and process.state() != QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1000)
        super().closeEvent(event)

    def _finished(self, result, *, record_run=True):
        animation = None
        animation_message = ""
        if isinstance(result, _SequenceSimulationPayload):
            animation = result.animation
            animation_message = result.animation_message
            result = result.result
        if record_run:
            elapsed_s = self._record_simulation_timing(result)
        else:
            elapsed_s = result.metadata.get("simulation_wall_time_s")
        self.result = result
        self.magnetization_animation = animation
        self._reset_run_controls(completed=True, elapsed_s=elapsed_s)
        self.export_button.setEnabled(True)
        if result.metadata.get("requested_sequence_kernel") == "metal_hybrid":
            if result.metadata.get("hybrid_fallback_used"):
                completion_message = (
                    "Simulation complete — GPU check not used; exact CPU result shown"
                )
            else:
                completion_message = (
                    "Simulation complete — checked CPU + GPU result shown"
                )
        else:
            completion_message = "Simulation complete"
        self._status_update(completion_message)
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
            center_event = csi.encoding_event_index(
                self.csi_repetition_selector.value(),
                csi.matrix[0] // 2,
                csi.matrix[1] // 2,
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
        else:
            for coil, coil_signal in enumerate(plot_signal):
                self.signal_plot.plot(
                    plot_time,
                    np.abs(coil_signal),
                    pen=pg.intColor(coil, hues=plot_signal.shape[0]),
                    name=f"Coil {coil + 1}",
                )
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
        self.result_volume_viewer.set_result(result, self.phantom)
        if animation is None:
            self.magnetization_animation_viewer.clear(
                animation_message
                or "No post-run animation data are available for this result."
            )
        else:
            self.magnetization_animation_viewer.set_animation(
                animation.time_s,
                animation.magnetization,
                phantom=self.phantom,
                pool_magnetization=animation.pool_magnetization,
                pool_names=animation.pool_names,
                storage_dtype=animation.storage_dtype,
                capture_note=animation_message,
            )
        try:
            self.reconstruction_explorer.set_result(result, self.phantom)
        except Exception as exc:
            self.reconstruction_explorer.clear(
                f"Interactive reconstruction unavailable: {exc}"
            )
        if self.spectroscopic_acquisition is not None:
            self._show_spectroscopic_result(result)
        elif self.spiral_acquisition is not None:
            self._show_spiral_result(result)
        else:
            self._show_cartesian_result(result)
        if record_run:
            self._register_session_simulation_run(
                result,
                animation=animation,
                animation_message=animation_message,
            )

    @staticmethod
    def _session_sequence_name(program):
        source = str(getattr(program, "source", "Sequence") or "Sequence")
        if "/" in source or "\\" in source:
            return Path(source).name
        return source

    def _register_session_simulation_run(
        self, result, *, animation=None, animation_message=""
    ):
        """Hand one completed run to the main window's session explorer."""
        if self.program is None or self.phantom is None:
            return None
        metadata = dict(getattr(result, "metadata", {}))
        run = SessionSimulationRun(
            run_id=uuid4().hex,
            created_at_utc=str(
                metadata.get("simulation_finished_at_utc")
                or datetime.now(timezone.utc).isoformat()
            ),
            sequence_name=self._session_sequence_name(self.program),
            phantom_name=str(getattr(self.phantom, "name", "Phantom")),
            phantom_shape=tuple(getattr(self.phantom, "shape", ())),
            sequence_duration_s=float(self.program.duration_s),
            adc_samples=int(np.asarray(getattr(result, "adc_times_s", ())).size),
            runtime_s=metadata.get("simulation_wall_time_s"),
            kernel=str(
                metadata.get("sequence_kernel")
                or metadata.get("requested_sequence_kernel")
                or ""
            ),
            result=result,
            state=self._session_run_state(
                phantom=self.phantom,
                animation=animation,
                animation_message=animation_message,
            ),
        )
        register = getattr(self.window(), "register_sequence_simulation_run", None)
        if callable(register):
            register(run)
            self._active_session_run_id = run.run_id
        return run

    def _session_run_state(self, *, phantom, **extra):
        state = {
            "program": self.program,
            "phantom": phantom,
            "generated_pulseq_sequence": self._generated_pulseq_sequence,
            "generated_sequence_source_index": self._generated_sequence_source_index,
            "sequence_generation_pending": self._sequence_generation_pending,
            "sequence_source_index": self.sequence_source.currentIndex(),
            "acquisition": self.acquisition,
            "acquisition_frames": self.acquisition_frames,
            "acquisition_volumes": self.acquisition_volumes,
            "spectroscopic_acquisition": self.spectroscopic_acquisition,
            "spiral_acquisition": self.spiral_acquisition,
            "acquisition_note": self.acquisition_note,
            "acquisition_compiled": self._acquisition_compiled,
        }
        state.update(extra)
        return state

    def _register_session_probe_run(self, result, *, mode=None):
        """Hand one completed spin-probe run to the session explorer."""
        if self.program is None:
            return None
        metadata = dict(getattr(result, "metadata", {}))
        probe_mode = str(mode or metadata.get("probe_type") or "spin")
        probe_phantom = self.phantom if probe_mode == "geometry" else None
        if probe_phantom is not None:
            context_name = str(getattr(probe_phantom, "name", "Geometry phantom"))
        else:
            context_name = f"{probe_mode.title()} spin probe"
        run = SessionSimulationRun(
            run_id=uuid4().hex,
            created_at_utc=str(
                metadata.get("simulation_finished_at_utc")
                or datetime.now(timezone.utc).isoformat()
            ),
            sequence_name=self._session_sequence_name(self.program),
            phantom_name=context_name,
            phantom_shape=(
                int(result.positions_m.shape[0]),
                int(result.frequency_offsets_hz.size),
            ),
            sequence_duration_s=float(self.program.duration_s),
            adc_samples=int(np.asarray(metadata.get("adc_times_s", ())).size),
            runtime_s=metadata.get("simulation_wall_time_s"),
            kernel=str(metadata.get("sequence_kernel") or ""),
            result=result,
            state=self._session_run_state(
                phantom=probe_phantom,
                probe_mode=probe_mode,
            ),
            run_type="spin_probe",
        )
        register = getattr(self.window(), "register_sequence_simulation_run", None)
        if callable(register):
            register(run)
            self._active_session_run_id = run.run_id
        return run

    def restore_session_simulation_run(self, run):
        """Restore the scientific objects and result views for a retained run."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Simulation running",
                "Cancel or finish the current simulation before switching runs.",
            )
            return False
        state = dict(run.state)
        self.program = state["program"]
        self.phantom = state["phantom"]
        self._generated_pulseq_sequence = state.get("generated_pulseq_sequence")
        self._generated_sequence_source_index = state.get(
            "generated_sequence_source_index"
        )
        self._sequence_generation_pending = bool(
            state.get("sequence_generation_pending", False)
        )
        self.acquisition = state.get("acquisition")
        self.acquisition_frames = state.get("acquisition_frames")
        self.acquisition_volumes = state.get("acquisition_volumes")
        self.spectroscopic_acquisition = state.get("spectroscopic_acquisition")
        self.spiral_acquisition = state.get("spiral_acquisition")
        self.acquisition_note = state.get("acquisition_note", "")
        self._acquisition_compiled = state.get("acquisition_compiled")
        self._active_session_run_id = run.run_id
        self._show_program(compiled=self._acquisition_compiled)
        if run.run_type == "spin_probe":
            self.result = None
            self.probe_result = run.result
            self._show_probe_result()
            self.probe_status.setText(f"Showing {run.display_name} from this session")
            for index in range(self.views.count()):
                if self.views.tabText(index) == "Spin Probe":
                    self.views.setCurrentIndex(index)
                    break
            self._status_update(f"Showing {run.display_name} from this session")
            return True
        self._finished(
            _SequenceSimulationPayload(
                run.result,
                state.get("animation"),
                animation_message=state.get("animation_message", ""),
            ),
            record_run=False,
        )
        self._active_session_run_id = run.run_id
        self._status_update(f"Showing {run.display_name} from this session")
        return True

    def refresh_after_session_control_restore(self):
        """Refresh dependent labels and panel visibility without regenerating."""
        self._source_changed()
        self._object_source_changed()
        for update in (
            self._update_bandwidth_labels,
            self._update_csi_labels,
            self._update_flash_labels,
            self._update_bssfp_labels,
            self._update_ss_bssfp_labels,
            self._update_radial_me_bssfp_labels,
            self._update_me_bssfp_labels,
        ):
            update()
        self._update_simulation_object_table()
        self._update_spoiling_quality()

    def forget_session_simulation_run(self, run):
        """Clear result views when the run currently being shown is deleted."""
        if self._active_session_run_id != run.run_id:
            return
        if run.run_type == "spin_probe":
            self._stop_probe_playback()
            self.probe_result = None
            self.probe_spectrum_viewer.result = None
            self.probe_spatial_viewer.result = None
            self.probe_info.setText("The displayed spin-probe run was deleted")
            self.probe_status.setText("Spin-probe run deleted")
            self._active_session_run_id = None
            return
        self._clear_previous_simulation_views()
        self.export_button.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Run deleted")
        self.simulation_time_label.setText("Elapsed: — · Remaining: —")
        self.status.setText("The displayed session simulation was deleted")

    def _record_simulation_timing(self, result, *, now=None, finished_at_utc=None):
        """Attach the UI-observed wall-clock timing to a completed result."""
        now = time.monotonic() if now is None else float(now)
        elapsed_s = (
            None
            if self._simulation_started_at is None
            else max(0.0, now - self._simulation_started_at)
        )
        if elapsed_s is not None:
            if finished_at_utc is None:
                finished_at_utc = datetime.now(timezone.utc).isoformat()
            result.metadata.update(
                {
                    "simulation_wall_time_s": float(elapsed_s),
                    "simulation_started_at_utc": self._simulation_started_at_utc,
                    "simulation_finished_at_utc": str(finished_at_utc),
                    "simulation_time_measurement": "wall_clock",
                }
            )
        return elapsed_s

    def _configure_frame_selector(self):
        self.frame_selector.blockSignals(True)
        self.frame_slider.blockSignals(True)
        self.frame_selector.clear()
        if self.acquisition_volumes is not None:
            self.frame_selector.addItem(
                f"All {self.acquisition_volumes.num_volumes} volumes (montage)", -1
            )
            for volume in range(self.acquisition_volumes.num_volumes):
                self.frame_selector.addItem(
                    self.acquisition_volumes.volume_label(volume), volume
                )
            self.frame_selector.setEnabled(True)
            self.frame_slider.setRange(-1, self.acquisition_volumes.num_volumes - 1)
            self.frame_slider.setValue(-1)
            self.frame_slider.setEnabled(True)
        elif self.spiral_acquisition is not None:
            spiral = self.spiral_acquisition
            if spiral.num_frames == 1:
                self.frame_selector.addItem("Single spiral frame", 0)
                self.frame_selector.setEnabled(False)
                self.frame_slider.setRange(0, 0)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(False)
            else:
                self.frame_selector.addItem(
                    f"All {spiral.num_frames} spiral frames (montage)", -1
                )
                for frame in range(spiral.num_frames):
                    self.frame_selector.addItem(spiral.frame_label(frame), frame)
                self.frame_selector.setEnabled(True)
                self.frame_slider.setRange(-1, spiral.num_frames - 1)
                self.frame_slider.setValue(-1)
                self.frame_slider.setEnabled(True)
        elif self.acquisition_frames is None:
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
        selected_index = self.frame_selector.findData(self.frame_slider.value())
        self.frame_value_label.setText(
            self.frame_selector.itemText(selected_index)
            if selected_index >= 0
            else "Single frame"
        )

    def _configure_spectroscopy_selectors(self):
        csi = self.spectroscopic_acquisition
        selectors = (
            self.csi_repetition_selector,
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
                self._csi_click_view_initialized = False
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
            self.csi_repetition_selector.setRange(0, csi.num_repetitions - 1)
            self.csi_repetition_selector.setValue(0)
            self.csi_repetition_selector.setEnabled(csi.num_repetitions > 1)
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

    def _selected_csi_repetition(self, values):
        csi = self.spectroscopic_acquisition
        if csi is None or csi.num_repetitions == 1:
            return values
        repetition = min(self.csi_repetition_selector.value(), csi.num_repetitions - 1)
        return np.take(values, repetition, axis=-4)

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
            f"Voxel (x={x_index}, y={y_index}) at ({x_mm:.4g}, {y_mm:.4g}) mm"
        )

    def _show_spectroscopic_result(self, result):
        csi = self.spectroscopic_acquisition
        try:
            csi.validate_adc_times(result.adc_times_s)
            if result.adc_gradient_moment_cyc_per_m is not None:
                csi.validate_gradient_moments(result.adc_gradient_moment_cyc_per_m)
            kspace = self._selected_csi_repetition(csi.reshape_signal(result.signal))
            spatial_fid = self._selected_csi_repetition(
                csi.reconstruct_spatial(result.signal)
            )
            spectra = self._selected_csi_repetition(
                csi.reconstruct_spectra(result.signal)
            )
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
                f"repetition={self.csi_repetition_selector.value()}; "
                f"BW={csi.spectral_bandwidth_hz:.6g} Hz; "
                f"resolution={csi.spectral_resolution_hz:.6g} Hz"
            )
            self._update_split_view(csi, kspace, spatial_fid, spectra)
            if not self._csi_click_view_initialized:
                self._csi_click_view_initialized = True
                self.split_view_checkbox.setChecked(True)
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
            self.frame_value_label.setText(self.frame_selector.currentText())
        if self.result is not None:
            if self.spiral_acquisition is not None:
                self._show_spiral_result(self.result)
            else:
                self._show_cartesian_result(self.result)

    def _frame_slider_changed(self, frame):
        combo_index = self.frame_selector.findData(int(frame))
        if combo_index >= 0:
            self.frame_selector.setCurrentIndex(combo_index)
            self.frame_value_label.setText(self.frame_selector.itemText(combo_index))

    def _show_spiral_result(self, result):
        spiral = self.spiral_acquisition
        if spiral is None:
            return
        try:
            selected = self.frame_selector.currentData()
            selected = 0 if selected is None else int(selected)
            frames = (
                range(spiral.num_frames)
                if selected < 0
                else (min(selected, spiral.num_frames - 1),)
            )
            views = []
            coil_text = ""
            for frame in frames:
                kspace = spiral.grid_kspace(result, frame)
                if kspace.ndim == 3:
                    kspace_magnitude = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
                    image = spiral.reconstruct(result, frame, coil_combine="rss")
                    coil_text = f", {kspace.shape[0]} coils (RSS)"
                else:
                    kspace_magnitude = np.abs(kspace)
                    image = np.abs(spiral.reconstruct(result, frame))
                views.append((np.asarray(kspace_magnitude), np.asarray(image)))
            kspace_display = self._montage([view[0] for view in views])
            image_display = self._montage([view[1] for view in views])
            frame_text = (
                f", montage of {spiral.num_frames} frames"
                if selected < 0
                else f", {spiral.frame_label(selected)}"
            )
            self.kspace_view.setImage(np.log1p(kspace_display).T, autoLevels=True)
            self.reconstruction_view.setImage(np.abs(image_display).T, autoLevels=True)
            self._update_zoom_label(self.kspace_view, self.kspace_zoom_info)
            self._update_zoom_label(
                self.reconstruction_view, self.reconstruction_zoom_info
            )
            self.kspace_info.setText(
                f"2D spiral samples linearly gridded to "
                f"{spiral.matrix[1]}×{spiral.matrix[0]}{coil_text}{frame_text}"
            )
            self.reconstruction_info.setText(
                f"Spiral gridding + |IFFT2|, min={np.min(np.abs(image_display)):.5g}, "
                f"max={np.max(np.abs(image_display)):.5g}{coil_text}{frame_text}; "
                "slice-selective 2D frames (no kz encoding)"
            )
        except Exception as exc:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            message = f"Spiral reconstruction unavailable: {exc}"
            self.kspace_info.setText(message)
            self.reconstruction_info.setText(message)

    def _show_cartesian_result(self, result):
        if self.acquisition is None:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            self.kspace_info.setText("No Cartesian acquisition metadata")
            self.reconstruction_info.setText("No Cartesian acquisition metadata")
            return
        if self.acquisition_volumes is not None:
            self._show_cartesian_volume_result(result)
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
            if self.acquisition_volumes is not None:
                z_note = (
                    "; xy-IFFT hybrid-space plane I(x,y,kz); "
                    "validated 3D IFFT available in export"
                )
            elif (
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

    def _show_cartesian_volume_result(self, result):
        try:
            selected = self.frame_selector.currentData()
            selected = 0 if selected is None else int(selected)
            if selected < 0:
                views = [
                    self._cartesian_volume_views(result, volume)
                    for volume in range(self.acquisition_volumes.num_volumes)
                ]
                kspace_magnitude = self._montage([item[0] for item in views])
                image = self._montage([item[1] for item in views])
                coil_text = views[0][2]
                volume_text = (
                    f", montage of {self.acquisition_volumes.num_volumes} volumes"
                )
            else:
                volume = min(selected, self.acquisition_volumes.num_volumes - 1)
                kspace_magnitude, image, coil_text = self._cartesian_volume_views(
                    result, volume
                )
                volume_text = f", {self.acquisition_volumes.volume_label(volume)}"
            self.kspace_view.setImage(np.log1p(kspace_magnitude).T, autoLevels=True)
            self.reconstruction_view.setImage(np.asarray(image).T, autoLevels=True)
            self._update_zoom_label(self.kspace_view, self.kspace_zoom_info)
            self._update_zoom_label(
                self.reconstruction_view, self.reconstruction_zoom_info
            )
            nx, ny, nz = self.acquisition_volumes.matrix
            read_axis, phase_axis, partition_axis = (
                self.acquisition_volumes.encoding_frame.axis_codes
            )
            self.kspace_info.setText(
                f"3D log(1+|k|), central partition ({partition_axis}) plane, "
                f"grid={nz}×{ny}×{nx}; read={read_axis}, phase={phase_axis}"
                f"{coil_text}{volume_text}"
            )
            z_index = nz // 2
            z_mm = ((z_index + 0.5) / nz - 0.5) * self.acquisition_volumes.fov_z_m * 1e3
            partition_scanner_axis, partition_sign = (
                self.acquisition_volumes.encoding_frame.axis_and_sign("partition")
            )
            scanner_position_mm = partition_sign * z_mm
            self.reconstruction_info.setText(
                f"3D |IFFT3|, central {partition_scanner_axis}="
                f"{scanner_position_mm:.4g} mm, "
                f"min={np.min(image):.5g}, max={np.max(image):.5g}"
                f"{coil_text}{volume_text}"
            )
        except Exception as exc:
            self.kspace_view.clear()
            self.reconstruction_view.clear()
            message = f"Cartesian 3D reconstruction unavailable: {exc}"
            self.kspace_info.setText(message)
            self.reconstruction_info.setText(message)

    def _cartesian_volume_views(self, result, volume):
        kspace = self.acquisition_volumes.to_cartesian_kspace(result, volume)
        if kspace.ndim == 4:
            kspace_magnitude = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
            image = self.acquisition_volumes.reconstruct(
                result, volume, coil_combine="rss"
            )
            coil_text = f", {kspace.shape[0]} coils (RSS)"
        else:
            kspace_magnitude = np.abs(kspace)
            image = np.abs(self.acquisition_volumes.reconstruct(result, volume))
            coil_text = ""
        centre = self.acquisition_volumes.partition_matrix // 2
        return (
            np.asarray(kspace_magnitude[centre]),
            np.asarray(np.abs(image[centre])),
            coil_text,
        )

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
            self._show_simulation_failure("Sequence simulation failed", message)

    def _show_simulation_failure(self, title, message):
        """Route RAM rejections to the application's actionable warning dialog."""
        message = str(message)
        window = self.window()
        show_memory_warning = getattr(window, "_show_memory_limit_warning", None)
        if message.startswith(MEMORY_ERROR_PREFIX) and callable(show_memory_warning):
            show_memory_warning(message)
            return
        QMessageBox.critical(self, str(title), message)

    def _reset_run_controls(self, *, completed=False, elapsed_s=None):
        self.run_button.setEnabled(
            self.program is not None and self.object_source.currentIndex() != 2
        )
        self.cancel_button.setEnabled(False)
        self.simulation_time_timer.stop()
        if completed:
            self.progress.setValue(self.progress.maximum())
            if elapsed_s is None and self._simulation_started_at is not None:
                elapsed_s = max(0.0, time.monotonic() - self._simulation_started_at)
            if elapsed_s is None:
                self.progress.setFormat("100% · Complete")
                self.simulation_time_label.setText(
                    "Total runtime: unavailable · Remaining: 0s"
                )
            else:
                self.progress.setFormat(
                    f"100% · Complete in {_format_duration(elapsed_s)}"
                )
                self.simulation_time_label.setText(
                    f"Total runtime: {_format_duration(elapsed_s)} · Remaining: 0s"
                )
        else:
            elapsed_s = (
                None
                if self._simulation_started_at is None
                else max(0.0, time.monotonic() - self._simulation_started_at)
            )
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Stopped")
            self.simulation_time_label.setText(
                "Elapsed: — · Remaining: —"
                if elapsed_s is None
                else f"Stopped after {_format_duration(elapsed_s)}"
            )
        self._simulation_started_at = None
        self._simulation_started_at_utc = None
        self._simulation_progress_started_at = None
        self._simulation_last_progress_at = None
        self._simulation_last_progress_done = 0
        self._simulation_last_progress_total = 1
        self._simulation_progress_rate = None

    def _export_results(self):
        if self.result is None:
            QMessageBox.warning(self, "No result", "Run a simulation first.")
            return
        data_notebook_filter = "xarray NetCDF + Jupyter notebook (*.nc)"
        bruker_filter = "Bruker raw dataset (directory)"
        default_path = self._export_directory() / "sequence_result.nc"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export sequence simulation result",
            str(default_path),
            (
                f"{data_notebook_filter};;xarray NetCDF only (*.nc);;"
                "HDF5 only (*.h5);;NumPy archive only (*.npz);;"
                "Jupyter notebook + NetCDF data (*.ipynb);;"
                "Bruker raw dataset (directory)"
            ),
        )
        if not filename:
            return
        path = Path(filename)
        if selected_filter != bruker_filter and not path.suffix:
            suffixes = {
                data_notebook_filter: ".nc",
                "xarray NetCDF only (*.nc)": ".nc",
                "HDF5 only (*.h5)": ".h5",
                "NumPy archive only (*.npz)": ".npz",
                "Jupyter notebook + NetCDF data (*.ipynb)": ".ipynb",
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
                raw_names = {
                    "fid": ["fid"],
                    "rawdata.job0": ["rawdata.job0"],
                    "both": ["fid", "rawdata.job0"],
                }[options.raw_data_files]
                exported = "\n".join(
                    raw_names
                    + ["acqp", "method", "visu_pars", "pulseprogram", "pdata/1"]
                )
            elif selected_filter == data_notebook_filter:
                data_path = path.with_suffix(".nc")
                notebook_path = path.with_suffix(".ipynb")
                self.result.save(data_path)
                export_sequence_result_notebook(
                    str(notebook_path),
                    str(data_path),
                )
                exported = f"{data_path.name}\n{notebook_path.name}"
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

    def _open_reconstruction_result(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open sequence reconstruction result",
            str(self._export_directory()),
            "xarray NetCDF (*.nc)",
        )
        if not filename:
            return
        try:
            from ..sequence.reconstruction import load_reconstruction_dataset

            dataset = load_reconstruction_dataset(filename)
            self.reconstruction_explorer.set_dataset(dataset, source=filename)
            self.views.setCurrentIndex(self.reconstruction_explorer_tab_index)
            self._status_update(f"Opened result {Path(filename).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Open result failed", str(exc))

    def _prompt_bruker_export_options(self, path: Path):
        matrix, fov_m = self._default_bruker_spatial_metadata()
        default_method, default_slice_thickness, default_raw_files = (
            self._default_bruker_export_profile()
        )
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
                "User:lucaCSI4",
                "User:pulseq",
                "BlochSimulator:SequenceSimulation",
            ]
        )
        method_name.setCurrentText(default_method)
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
        slice_thickness.setValue(default_slice_thickness)
        slice_thickness.setSuffix(" mm")
        layout.addRow("PVM_SliceThick", slice_thickness)

        raw_data_files = QComboBox()
        raw_data_files.addItems(["fid", "rawdata.job0", "both"])
        raw_data_files.setCurrentText(default_raw_files)
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
        spiral = self.spiral_acquisition
        if spiral is None and self.result is not None:
            try:
                spiral = self.result.spiral_acquisition
            except Exception:
                spiral = None
        if spiral is not None:
            return (spiral.matrix, spiral.fov_m)
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

    def _default_bruker_export_profile(self):
        spectroscopy = self.spectroscopic_acquisition
        if spectroscopy is None and self.result is not None:
            try:
                spectroscopy = self.result.spectroscopic_acquisition
            except Exception:
                spectroscopy = None

        definitions = {}
        if self.program is not None:
            candidate = self.program.metadata.get("definitions", {})
            if isinstance(candidate, dict):
                definitions = candidate

        slice_thickness_mm = 1.0
        try:
            fov = np.asarray(definitions.get("FOV", ()), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            fov = np.empty(0, dtype=float)
        if fov.size >= 3 and np.isfinite(fov[2]) and fov[2] > 0:
            slice_thickness_mm = float(fov[2]) * 1000.0
        elif spectroscopy is not None and hasattr(self, "csi_slice_thickness_mm"):
            slice_thickness_mm = float(self.csi_slice_thickness_mm.value())

        if spectroscopy is None:
            return ("Bruker:RARE", slice_thickness_mm, "fid")

        order = str(definitions.get("PhaseEncodingOrder", "")).strip().lower()
        if not order:
            n_x, n_y = spectroscopy.matrix
            indices = tuple(
                index
                for index, repetition in zip(
                    spectroscopy.encoding_indices,
                    spectroscopy.repetition_indices,
                )
                if repetition == 0
            )
            radii = np.asarray(
                [
                    np.hypot(
                        (x - n_x // 2) / spectroscopy.fov_m[0],
                        (y - n_y // 2) / spectroscopy.fov_m[1],
                    )
                    for x, y in indices
                ],
                dtype=float,
            )
            if radii.size and np.all(np.diff(radii) >= -1e-12):
                order = "centric"

        method = "User:lucaCSI4" if order == "centric" else "Bruker:CSI"
        return (method, slice_thickness_mm, "both")
