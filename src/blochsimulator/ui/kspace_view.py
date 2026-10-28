"""k-Space tab: trajectory of a sequence, its timing, and its sampling PSF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..sequence.encoding import numeric_definition_array
from ..sequence.kspace_trajectory import (
    AXES,
    GradientImperfections,
    KSpaceTrajectory,
    calculate_kspace,
    classify_readout,
)
from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .plot_interaction import AXIS_ZOOM_TOOLTIP

PLANES = {"kx × ky": (0, 1), "kx × kz": (0, 2), "ky × kz": (1, 2)}
SHOW_MODES = ("Readout", "Frame", "All readouts")
# Animation speed as a fraction of real time.
SPEEDS = (
    ("1/1000 × real time", 1e-3),
    ("1/100 × real time", 1e-2),
    ("1/10 × real time", 1e-1),
    ("Real time", 1.0),
)
DEFAULT_SPEED_INDEX = 1
ANIMATION_TICK_MS = 30
SLIDER_STEPS = 1000
AXIS_COLORS = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb"}
RF_COLOR = "#7c3aed"
ADC_BRUSH = (250, 204, 21, 60)
CURSOR_COLOR = "#e11d48"
TRAIL_COLOR = (225, 29, 72, 170)
IDEAL_COLOR = "#94a3b8"
NOMINAL_COLOR = "#0f766e"
ROLE_LINE_COLORS = {
    "excitation": "#15803d",
    "refocusing": "#d97706",
    "inversion": "#7c3aed",
}
# Readouts drawn at once; more are thinned evenly.
MAX_DRAWN_READOUTS = 400
# Readouts are coloured in this many groups from first to last.
COLOR_BINS = 32
# Larger sample sets compute their sampling PSF only on request.
AUTO_PSF_SAMPLES = 40_000
# Sample markers are drawn while the shown samples stay below this count.
MAX_MARKED_SAMPLES = 6000
PLOT_MINIMUM_HEIGHT = 320
IMAGE_FLOOR_DB = -40.0
EDDY_COMPONENTS = 2


@dataclass(frozen=True)
class KSpaceContext:
    """The sequence whose trajectory is shown, supplied by Sequence Mode.

    ``token`` identifies the sequence revision; a different token makes the
    tab recalculate when it is shown again.
    """

    program: object
    title: str
    token: object
    nucleus: str = "H1"
    field_strength_t: float = 3.0


@dataclass(frozen=True)
class ReadoutLayout:
    """How the ADC events of a sequence form image frames.

    ``frames`` pairs a label with the ADC event indices of every frame.
    ``fov_m`` and ``matrix`` describe the in-plane image when known.
    """

    kind: str = ""
    frames: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    fov_m: Optional[Tuple[float, float]] = None
    matrix: Optional[Tuple[int, int]] = None
    description: str = ""


def _definition_geometry(program):
    definitions = {
        str(key).lower(): value
        for key, value in dict(program.metadata.get("definitions", {})).items()
    }
    fov = matrix = None
    try:
        values = numeric_definition_array(definitions.get("fov", ()), "FOV")
        if values.size >= 2 and np.all(values[:2] > 0):
            fov = (float(values[0]), float(values[1]))
    except (TypeError, ValueError):
        pass
    for key in ("matrixsize", "reconmatrix"):
        try:
            values = numeric_definition_array(definitions.get(key, ()), key)
        except (TypeError, ValueError):
            continue
        if values.size >= 2 and np.all(values[:2] >= 1):
            matrix = (int(values[0]), int(values[1]))
            break
    return fov, matrix


def readout_layout(program, kspace: KSpaceTrajectory) -> ReadoutLayout:
    """Group the ADC events of ``program`` into the frames reconstruction uses.

    Spiral, single Cartesian and multi-frame Cartesian inference are tried in
    turn; otherwise the layout has no frames and takes FOV and matrix from the
    definitions.
    """
    from ..sequence.acquisition import (
        infer_cartesian_acquisition,
        infer_cartesian_acquisition_frames,
        infer_spiral_acquisition,
    )
    from ..sequence.compiler import SequenceCompiler

    events = kspace.adc_event_index
    if events.size == 0:
        return ReadoutLayout(description="The sequence has no ADC events.")

    def event_groups(sample_indices):
        return tuple(
            tuple(int(value) for value in np.unique(events[list(indices)]))
            for indices in sample_indices
        )

    try:
        spiral = infer_spiral_acquisition(program, kspace=kspace)
    except Exception:
        spiral = None
    if spiral is not None:
        groups = event_groups(spiral.sample_indices)
        frames = tuple(
            (f"{index + 1}: {spiral.frame_label(index)}", group)
            for index, group in enumerate(groups)
        )
        return ReadoutLayout(
            kind="spiral",
            frames=frames,
            fov_m=tuple(spiral.fov_m),
            matrix=tuple(spiral.matrix),
            description=(
                f"Spiral, {spiral.num_frames} frame(s), reconstructed by "
                f"{spiral.reconstruction_label}"
            ),
        )
    try:
        compiled = SequenceCompiler().compile(program)
    except Exception:
        compiled = None
    if compiled is not None:
        try:
            single = infer_cartesian_acquisition(program, compiled=compiled)
        except Exception:
            single = None
        if single is not None:
            return ReadoutLayout(
                kind="cartesian",
                frames=(("1: single frame", tuple(range(kspace.num_adc_events))),),
                fov_m=tuple(single.fov_m),
                matrix=(int(single.read_matrix), int(single.phase_matrix)),
                description="Cartesian, 1 frame",
            )
        try:
            cartesian = infer_cartesian_acquisition_frames(program, compiled=compiled)
        except Exception:
            cartesian = None
        if cartesian is not None:
            groups = event_groups(cartesian.sample_indices)
            frames = tuple(
                (f"{index + 1}: {cartesian.frame_label(index)}", group)
                for index, group in enumerate(groups)
            )
            first = cartesian.acquisitions[0]
            return ReadoutLayout(
                kind="cartesian",
                frames=frames,
                fov_m=tuple(first.fov_m),
                matrix=(int(first.read_matrix), int(first.phase_matrix)),
                description=f"Cartesian, {cartesian.num_frames} frame(s)",
            )
    fov, matrix = _definition_geometry(program)
    return ReadoutLayout(
        fov_m=fov,
        matrix=matrix,
        description="No image frames recognised; readouts are shown one by one",
    )


@dataclass(frozen=True)
class SamplingPSF:
    """Gridding image of a unit point source sampled along the trajectory."""

    image: np.ndarray
    fov_m: Tuple[float, float]
    matrix: Tuple[int, int]
    sampling_gap: float
    nyquist_fov_m: Tuple[float, float]

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.image))) if self.image.size else 0.0


def sampling_psf(k_cyc_per_m, fov_m, matrix) -> SamplingPSF:
    """Density-compensated NUFFT image of a point source at the FOV centre.

    ``k_cyc_per_m`` holds two in-plane coordinates per sample. The peak is
    relative to a fully sampled Cartesian k-space (1 for uniform weighting);
    a spiral covers only the disc inside ``k_max``, about pi/4 of it.
    """
    from ..sequence.nufft import NUFFTOperator

    k = np.asarray(k_cyc_per_m, dtype=float)
    k = k[np.all(np.isfinite(k), axis=1)]
    if k.shape[0] < 3:
        raise ValueError("at least three finite k-space samples are needed")
    operator = NUFFTOperator(k, tuple(matrix), tuple(fov_m))
    image = operator.point_spread_function()
    return SamplingPSF(
        image=np.asarray(image),
        fov_m=tuple(float(value) for value in fov_m),
        matrix=tuple(int(value) for value in matrix),
        sampling_gap=float(operator.sampling_gap),
        nyquist_fov_m=tuple(float(value) for value in operator.nyquist_fov_m()),
    )


def step_waveform(kspace: KSpaceTrajectory, start_s: float, stop_s: float):
    """Times and played gradients (Hz/m) for a step plot of ``[start, stop]``.

    Returns ``(edges, values)`` with one more edge than values.
    """
    times, _ = kspace.timeline(start_s, stop_s)
    values = (
        kspace.gradient_hz_per_m(times[:-1]) if times.size > 1 else np.zeros((0, 3))
    )
    return times, values


def peak_slew_t_per_m_per_s(edges, values, gamma_hz_per_t) -> np.ndarray:
    """Largest per-axis slew rate of a step waveform, in T/m/s.

    Steps are taken between the centres of neighbouring raster cells; cells
    shorter than 1 ns only mark RF centres and are skipped.
    """
    durations = np.diff(edges)
    keep = durations > 1e-9
    durations, values = durations[keep], values[keep]
    if values.shape[0] < 2:
        return np.zeros(3)
    spacing = 0.5 * (durations[1:] + durations[:-1])
    slew = np.abs(np.diff(values, axis=0)) / spacing[:, None]
    return np.max(slew, axis=0) / gamma_hz_per_t


def _joined(arrays, horizontal, vertical):
    """Concatenate readouts into one NaN-separated curve of a plane."""
    gap = np.full((1, 3), np.nan)
    joined = np.concatenate([part for k in arrays for part in (k, gap)])
    return joined[:, horizontal], joined[:, vertical]


def _format(value, pattern):
    return "—" if value is None or not np.isfinite(value) else pattern.format(value)


class KSpaceView(QWidget):
    """k-space trajectory of the current sequence.

    ``context_provider`` returns a :class:`KSpaceContext` or raises
    ``ValueError`` with a message for the user; ``token_provider`` returns
    the current sequence token. ``psf_analysis_requested`` asks Sequence Mode
    to open the PSF Analysis tab.
    """

    psf_analysis_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context_provider: Optional[Callable[[], KSpaceContext]] = None
        self.token_provider: Optional[Callable[[], object]] = None
        self.context: Optional[KSpaceContext] = None
        self.kspace: Optional[KSpaceTrajectory] = None
        self.ideal_kspace: Optional[KSpaceTrajectory] = None
        self.layout_info = ReadoutLayout()
        self.psf: Optional[SamplingPSF] = None
        self.cursor_s = 0.0
        self.window_s = (0.0, 0.0)
        self._dirty = False
        self._updating = False
        self._psf_key = None
        self._psf_requested_key = None
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(ANIMATION_TICK_MS)
        self.animation_timer.timeout.connect(self._advance_animation)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 10, 12, 10)

        self.title_label = QLabel("k-space trajectory")
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize() + 2, 12))
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        self.summary_label = QLabel(
            "The trajectory k(t) = ∫G dt is integrated from the gradient "
            "waveforms and referenced to the pulses: an excitation resets k "
            "to 0 at its centre, a refocusing pulse mirrors k through 0. "
            "Gradients are stored in Hz/m for the sequence nucleus, so k does "
            "not depend on B0; the gradient imperfections below model the "
            "scanner's actual gradient. "
            "<a href='psf-analysis'>Simulate the full PSF</a> including "
            "relaxation and off-resonance."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.summary_label.linkActivated.connect(
            lambda _link: self.psf_analysis_requested.emit()
        )
        layout.addWidget(self.summary_label)

        layout.addWidget(self._imperfection_group())
        layout.addLayout(self._display_controls())
        layout.addLayout(self._animation_controls())

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.kspace_plot = self._plot(
            "1 · k-space",
            "kx [1/m]",
            "ky [1/m]",
            "ADC samples of the shown readouts in the chosen plane, coloured "
            "from the first to the last readout. Dashed teal: nominal k_max "
            "of the image matrix. Dashed grey: the ideal trajectory when "
            "gradient imperfections are set. Red: the k-space position at the "
            "time cursor, with its path since the last excitation.",
            legend=False,
        )
        self.kspace_plot.setAspectLocked(True)
        self.sequence_plot = self._plot(
            "2 · Gradients and RF",
            "Time [ms]",
            "Gradient [mT/m]",
            "Played gradients per axis, RF magnitude scaled to the gradient "
            "range, and ADC windows (yellow). Drag the red line to move the "
            "time cursor.",
        )
        self.k_time_plot = self._plot(
            "3 · k(t)",
            "Time [ms]",
            "k [1/m]",
            "k-space position per axis over time. Green, orange and violet "
            "lines mark excitation, refocusing and inversion pulse centres.",
        )
        self.k_time_plot.setXLink(self.sequence_plot)
        self.psf_plot = self._plot(
            "4 · Sampling PSF",
            "Horizontal [mm]",
            "Vertical [mm]",
            "Density-compensated NUFFT image of a point source at the FOV "
            "centre, sampled at the shown readouts in the chosen plane. It "
            "shows the blurring and aliasing caused by the trajectory alone, "
            "relative to the peak of fully sampled Cartesian k-space.",
            legend=False,
        )
        self.psf_plot.setAspectLocked(True)
        self.psf_image = pg.ImageItem(axisOrder="row-major")
        self.psf_image.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(nPts=256)
        )
        self.psf_plot.addItem(self.psf_image)
        self.psf_message = pg.TextItem(color="#475569", anchor=(0.5, 0.5))
        self.psf_plot.addItem(self.psf_message)
        grid.addWidget(self.kspace_plot, 0, 0)
        grid.addWidget(self.sequence_plot, 0, 1)
        grid.addWidget(self.k_time_plot, 1, 1)
        grid.addWidget(self.psf_plot, 1, 0)
        layout.addLayout(grid, 1)

        self.cursor_lines = []
        for plot in (self.sequence_plot, self.k_time_plot):
            line = pg.InfiniteLine(
                angle=90, movable=True, pen=pg.mkPen(CURSOR_COLOR, width=2)
            )
            line.setZValue(20)
            line.sigPositionChanged.connect(self._cursor_line_moved)
            self.cursor_lines.append(line)

        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.info_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.info_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.info_table.setAlternatingRowColors(True)
        layout.addWidget(self.info_table)
        self._set_controls_enabled(False)
        self.render()

    # Construction ---------------------------------------------------------

    @staticmethod
    def _spin(minimum, maximum, decimals, suffix, tooltip, value=0.0):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setValue(value)
        box.setToolTip(tooltip)
        box.setKeyboardTracking(False)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return box

    def _imperfection_group(self):
        group = QGroupBox("Gradient imperfections")
        grid = QGridLayout(group)
        self.delay_us = [
            self._spin(
                -1000.0,
                1000.0,
                2,
                " µs",
                f"Delay of the {axis} gradient relative to RF and ADC. A "
                "delay shifts the k-space samples along the readout.",
            )
            for axis in AXES
        ]
        self.scale_percent = [
            self._spin(
                50.0,
                150.0,
                2,
                " %",
                f"Amplitude of the played {axis} gradient relative to the "
                "programmed one (gradient calibration).",
                100.0,
            )
            for axis in AXES
        ]
        grid.addWidget(QLabel("Delay x / y / z"), 0, 0)
        grid.addWidget(QLabel("Scale x / y / z"), 1, 0)
        for column, (delay, scale) in enumerate(zip(self.delay_us, self.scale_percent)):
            grid.addWidget(delay, 0, column + 1)
            grid.addWidget(scale, 1, column + 1)
        self.eddy_amplitude_percent = []
        self.eddy_time_constant_ms = []
        for component in range(EDDY_COMPONENTS):
            amplitude = self._spin(
                0.0,
                50.0,
                3,
                " %",
                "Eddy-current amplitude relative to the gradient change; 0 "
                "switches the component off. Eddy currents act on all axes.",
            )
            constant = self._spin(
                0.001,
                1000.0,
                3,
                " ms",
                "Decay time constant of the eddy-current component",
                0.5 if component == 0 else 5.0,
            )
            self.eddy_amplitude_percent.append(amplitude)
            self.eddy_time_constant_ms.append(constant)
            grid.addWidget(QLabel(f"Eddy current {component + 1}"), 2 + component, 0)
            grid.addWidget(amplitude, 2 + component, 1)
            grid.addWidget(QLabel("τ"), 2 + component, 2, Qt.AlignRight)
            grid.addWidget(constant, 2 + component, 3)
        self.show_ideal = QCheckBox("Show ideal trajectory")
        self.show_ideal.setChecked(True)
        self.show_ideal.setToolTip(
            "Overlay the trajectory without imperfections (dashed grey)"
        )
        self.show_ideal.toggled.connect(self._display_changed)
        self.reset_imperfections_button = QPushButton("Reset")
        self.reset_imperfections_button.setToolTip("Restore the ideal gradients")
        self.reset_imperfections_button.clicked.connect(self.reset_imperfections)
        grid.addWidget(self.show_ideal, 4, 1, 1, 2)
        grid.addWidget(self.reset_imperfections_button, 4, 3)
        for box in (
            self.delay_us
            + self.scale_percent
            + self.eddy_amplitude_percent
            + self.eddy_time_constant_ms
        ):
            box.valueChanged.connect(self._imperfections_changed)
        return group

    def _display_controls(self):
        controls = QHBoxLayout()
        self.recalculate_button = QPushButton("Recalculate")
        self.recalculate_button.setToolTip(
            "Calculate the trajectory of the current sequence"
        )
        self.recalculate_button.clicked.connect(self.recalculate)
        controls.addWidget(self.recalculate_button)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #475569;")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, 1)
        controls.addWidget(QLabel("Show"))
        self.show_mode = QComboBox()
        self.show_mode.addItems(list(SHOW_MODES))
        self.show_mode.setToolTip(
            "Readout: one ADC event. Frame: all readouts of one image frame. "
            "All readouts: the whole sequence."
        )
        self.show_mode.currentIndexChanged.connect(self._selection_changed)
        controls.addWidget(self.show_mode)
        controls.addWidget(QLabel("Readout"))
        self.readout_selector = QSpinBox()
        self.readout_selector.setRange(1, 1)
        self.readout_selector.setKeyboardTracking(False)
        self.readout_selector.setToolTip("ADC event, in chronological order")
        self.readout_selector.valueChanged.connect(self._readout_changed)
        controls.addWidget(self.readout_selector)
        controls.addWidget(QLabel("Frame"))
        self.frame_selector = QComboBox()
        self.frame_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.frame_selector.setToolTip("Image frame as grouped for reconstruction")
        self.frame_selector.currentIndexChanged.connect(self._frame_changed)
        controls.addWidget(self.frame_selector)
        controls.addWidget(QLabel("Plane"))
        self.plane_selector = QComboBox()
        self.plane_selector.addItems(list(PLANES))
        self.plane_selector.setToolTip("Scanner axes shown in plots 1 and 4")
        self.plane_selector.currentIndexChanged.connect(self._display_changed)
        controls.addWidget(self.plane_selector)
        self.psf_scale = QComboBox()
        self.psf_scale.addItems(["Linear", "dB"])
        self.psf_scale.setToolTip(
            "Colour scale of the sampling PSF; dB reaches down to "
            f"{IMAGE_FLOOR_DB:g} dB of its peak"
        )
        self.psf_scale.currentIndexChanged.connect(self._display_changed)
        controls.addWidget(self.psf_scale)
        self.compute_psf_button = QPushButton("Compute sampling PSF")
        self.compute_psf_button.setToolTip(
            f"Shown readouts with more than {AUTO_PSF_SAMPLES:,} samples are "
            "gridded only on request, which takes a few seconds"
        )
        self.compute_psf_button.clicked.connect(self._compute_psf_requested)
        self.compute_psf_button.hide()
        controls.addWidget(self.compute_psf_button)
        return controls

    def _animation_controls(self):
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setToolTip(
            "Animate the k-space position through the shown time window"
        )
        self.play_button.toggled.connect(self._play_toggled)
        controls.addWidget(self.play_button)
        self.speed_selector = QComboBox()
        self.speed_selector.addItems([label for label, _ in SPEEDS])
        self.speed_selector.setCurrentIndex(DEFAULT_SPEED_INDEX)
        self.speed_selector.setToolTip("Animation speed")
        controls.addWidget(self.speed_selector)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, SLIDER_STEPS)
        self.time_slider.setToolTip("Time cursor within the shown window")
        self.time_slider.valueChanged.connect(self._slider_moved)
        controls.addWidget(self.time_slider, 1)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(260)
        controls.addWidget(self.time_label)
        return controls

    @staticmethod
    def _plot(title, x_label, y_label, tooltip, legend=True):
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.setTitle(title, color="#0f172a")
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.showGrid(x=True, y=True, alpha=0.2)
        if legend:
            plot.addLegend(
                offset=(-10, 10),
                brush=pg.mkBrush(255, 255, 255, 180),
                pen=pg.mkPen("#cbd5e1"),
            )
        for axis in ("bottom", "left"):
            plot.getAxis(axis).enableAutoSIPrefix(False)
        plot.setToolTip(f"{tooltip}\n\n{AXIS_ZOOM_TOOLTIP}")
        plot.setMinimumHeight(PLOT_MINIMUM_HEIGHT)
        return plot

    # Calculation ----------------------------------------------------------

    def imperfections(self) -> GradientImperfections:
        """Return the gradient imperfections described by the controls."""
        components = [
            (amplitude.value() / 100.0, constant.value() / 1000.0)
            for amplitude, constant in zip(
                self.eddy_amplitude_percent, self.eddy_time_constant_ms
            )
            if amplitude.value() > 0
        ]
        return GradientImperfections(
            delay_s=tuple(box.value() * 1e-6 for box in self.delay_us),
            scale=tuple(box.value() / 100.0 for box in self.scale_percent),
            eddy_amplitudes=tuple(amplitude for amplitude, _ in components),
            eddy_time_constants_s=tuple(constant for _, constant in components),
        )

    def reset_imperfections(self):
        self._updating = True
        try:
            for box in self.delay_us + self.eddy_amplitude_percent:
                box.setValue(0.0)
            for box in self.scale_percent:
                box.setValue(100.0)
        finally:
            self._updating = False
        self._imperfections_changed()

    def recalculate(self):
        """Fetch the current sequence and calculate its trajectory."""
        self.play_button.setChecked(False)
        if self.context_provider is None:
            self._show_status("No sequence source is connected.")
            return
        try:
            context = self.context_provider()
        except ValueError as exc:
            self._show_status(str(exc))
            return
        self.set_context(context)

    def set_context(self, context: KSpaceContext):
        """Calculate and show the trajectory of ``context.program``."""
        try:
            ideal = calculate_kspace(context.program)
        except Exception as exc:
            self._show_status(f"k-space calculation failed: {exc}")
            return
        self.context = context
        self.ideal_kspace = ideal
        self.layout_info = readout_layout(context.program, ideal)
        self.kspace = None
        self._psf_key = None
        self._apply_imperfections()
        self._configure_selectors()
        self._show_status("")
        self._selection_changed()

    def _apply_imperfections(self):
        if self.ideal_kspace is None:
            return
        imperfections = self.imperfections()
        if imperfections.is_ideal:
            self.kspace = self.ideal_kspace
            return
        try:
            self.kspace = calculate_kspace(
                self.context.program, imperfections=imperfections
            )
        except Exception as exc:
            self.kspace = self.ideal_kspace
            self._show_status(f"Imperfect gradients failed: {exc}")

    def _imperfections_changed(self, *_):
        if self._updating:
            return
        self._apply_imperfections()
        self._psf_key = None
        self._display_changed()

    def _configure_selectors(self):
        kspace = self.kspace
        events = 0 if kspace is None else kspace.num_adc_events
        self._updating = True
        try:
            self.readout_selector.setRange(1, max(events, 1))
            self.readout_selector.setValue(1)
            self.frame_selector.clear()
            self.frame_selector.addItems(
                [label for label, _ in self.layout_info.frames]
            )
            frames_model = self.show_mode.model()
            frames_model.item(1).setEnabled(bool(self.layout_info.frames))
            if self.layout_info.frames:
                self.show_mode.setCurrentIndex(1)
            elif self.show_mode.currentIndex() == 1:
                self.show_mode.setCurrentIndex(0)
            plane = 0
            if kspace is not None and kspace.adc_k_cyc_per_m.size:
                extent = np.nan_to_num(np.nanmax(np.abs(kspace.adc_k_cyc_per_m), 0))
                plane = {
                    (0, 1): 0,
                    (0, 2): 1,
                    (1, 2): 2,
                }[tuple(sorted(np.argsort(extent)[::-1][:2]))]
            self.plane_selector.setCurrentIndex(plane)
        finally:
            self._updating = False
        self._set_controls_enabled(events > 0)

    def _set_controls_enabled(self, enabled):
        mode = self.show_mode.currentIndex()
        for widget in (
            self.show_mode,
            self.plane_selector,
            self.psf_scale,
            self.play_button,
            self.speed_selector,
            self.time_slider,
        ):
            widget.setEnabled(enabled)
        self.readout_selector.setEnabled(enabled and mode == 0)
        self.frame_selector.setEnabled(
            enabled and mode == 1 and len(self.layout_info.frames) > 1
        )

    # Selection ------------------------------------------------------------

    def selected_events(self) -> Tuple[int, ...]:
        """ADC events shown in plot 1 and used for the sampling PSF."""
        kspace = self.kspace
        if kspace is None or kspace.num_adc_events == 0:
            return ()
        mode = self.show_mode.currentIndex()
        if mode == 1 and self.layout_info.frames:
            index = max(self.frame_selector.currentIndex(), 0)
            return self.layout_info.frames[index][1]
        if mode == 2:
            return tuple(range(kspace.num_adc_events))
        return (self.readout_selector.value() - 1,)

    def _time_window(self, events):
        kspace = self.kspace
        if not events:
            return 0.0, float(kspace.duration_s) if kspace is not None else 0.0
        first = kspace.adc_event_slice(events[0])
        last = kspace.adc_event_slice(events[-1])
        start = float(kspace.adc_times_s[first.start])
        stop = float(kspace.adc_times_s[last.stop - 1])
        excitations = [
            pulse for pulse in kspace.rf_pulses if pulse.role == "excitation"
        ]
        before = [pulse for pulse in excitations if pulse.center_s <= start]
        if before:
            start = before[-1].start_s
        margin = 0.05 * max(stop - start, 1e-4)
        return max(start - margin, 0.0), min(stop + margin, kspace.duration_s)

    def _selection_changed(self, *_):
        if self._updating:
            return
        self._set_controls_enabled(self.kspace is not None)
        events = self.selected_events()
        self.window_s = self._time_window(events)
        if events:
            first = self.kspace.adc_event_slice(events[0])
            self.cursor_s = float(self.kspace.adc_times_s[first.start])
        else:
            self.cursor_s = self.window_s[0]
        self._display_changed()

    def _readout_changed(self, *_):
        self._selection_changed()

    def _frame_changed(self, *_):
        self._selection_changed()

    def _display_changed(self, *_):
        if self._updating:
            return
        self._dirty = True
        if self.isVisible():
            self.render()

    # Time cursor ----------------------------------------------------------

    def set_cursor(self, time_s: float):
        start, stop = self.window_s
        self.cursor_s = float(np.clip(time_s, start, max(stop, start)))
        self._render_cursor()

    def _slider_moved(self, value):
        if self._updating:
            return
        start, stop = self.window_s
        self.set_cursor(start + (stop - start) * value / SLIDER_STEPS)

    def _cursor_line_moved(self, line):
        if self._updating:
            return
        self.set_cursor(line.value() / 1000.0)

    def _play_toggled(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            start, stop = self.window_s
            if self.cursor_s >= stop:
                self.cursor_s = start
            self.animation_timer.start()
        else:
            self.animation_timer.stop()

    def _advance_animation(self):
        start, stop = self.window_s
        if stop <= start:
            self.play_button.setChecked(False)
            return
        speed = SPEEDS[self.speed_selector.currentIndex()][1]
        step = ANIMATION_TICK_MS / 1000.0 * speed
        time_s = self.cursor_s + step
        if time_s > stop:
            time_s = start
        self.set_cursor(time_s)

    def hideEvent(self, event):
        self.play_button.setChecked(False)
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self._stale() or self.context is None:
            self.recalculate()
        if self._dirty:
            self.render()

    def _stale(self) -> bool:
        if self.context is None or self.token_provider is None:
            return False
        return self.token_provider() != self.context.token

    def _show_status(self, text):
        self.status_label.setText(text)

    # Rendering ------------------------------------------------------------

    @property
    def gamma_hz_per_t(self) -> float:
        nucleus = "H1" if self.context is None else self.context.nucleus
        return NUCLEUS_GAMMA_HZ_PER_T.get(nucleus, NUCLEUS_GAMMA_HZ_PER_T["H1"])

    def render(self):
        """Redraw all plots for the current selection."""
        self._dirty = False
        for plot in (self.kspace_plot, self.sequence_plot, self.k_time_plot):
            plot.clear()
            legend = plot.plotItem.legend
            if legend is not None:
                legend.clear()
        self.cursor_trail = None
        self.cursor_dot = None
        kspace = self.kspace
        if kspace is None or kspace.num_adc_events == 0:
            self.psf_image.clear()
            message = (
                "Open a sequence in Sequence Mode to see its trajectory."
                if kspace is None
                else "The sequence has no ADC events."
            )
            self.psf_message.setText(message)
            self.info_table.setRowCount(0)
            self._fit_info_table()
            return
        events = self.selected_events()
        self._render_kspace(events)
        self._render_sequence()
        self._render_k_time()
        self._render_psf(events)
        self._render_info(events)
        self._render_cursor()

    def _plane(self):
        return PLANES[self.plane_selector.currentText()]

    def _drawn_events(self, events):
        if len(events) <= MAX_DRAWN_READOUTS:
            return tuple(events)
        keep = np.linspace(0, len(events) - 1, MAX_DRAWN_READOUTS).astype(int)
        return tuple(events[index] for index in np.unique(keep))

    def _render_kspace(self, events):
        plot = self.kspace_plot
        horizontal, vertical = self._plane()
        plot.setLabel("bottom", f"k{AXES[horizontal]} [1/m]")
        plot.setLabel("left", f"k{AXES[vertical]} [1/m]")
        drawn = self._drawn_events(events)
        samples = sum(self.kspace.event_k(event).shape[0] for event in drawn)
        mark = samples <= MAX_MARKED_SAMPLES
        colormap = pg.colormap.get("viridis")
        ideal = self.ideal_kspace
        show_ideal = (
            self.show_ideal.isChecked()
            and ideal is not None
            and ideal is not self.kspace
        )
        self.show_ideal.setEnabled(ideal is not self.kspace)
        if show_ideal:
            x, y = _joined(
                [ideal.event_k(event) for event in drawn], horizontal, vertical
            )
            plot.plot(
                x,
                y,
                pen=pg.mkPen(IDEAL_COLOR, width=1, style=Qt.DashLine),
                connect="finite",
            )
        # One curve per colour keeps many readouts cheap to draw.
        bins = np.array_split(np.arange(len(drawn)), min(len(drawn), COLOR_BINS))
        for order, members in enumerate(bins):
            fraction = order / (len(bins) - 1) if len(bins) > 1 else 0.5
            color = colormap.map(0.1 + 0.8 * fraction, mode="qcolor")
            x, y = _joined(
                [self.kspace.event_k(drawn[index]) for index in members],
                horizontal,
                vertical,
            )
            plot.plot(
                x,
                y,
                pen=pg.mkPen(color, width=1.2),
                connect="finite",
                symbol="o" if mark else None,
                symbolSize=3,
                symbolPen=None,
                symbolBrush=pg.mkBrush(color),
            )
        nominal = self._nominal_kmax(horizontal, vertical)
        if nominal is not None:
            half_h, half_v = nominal
            plot.plot(
                [-half_h, half_h, half_h, -half_h, -half_h],
                [-half_v, -half_v, half_v, half_v, -half_v],
                pen=pg.mkPen(NOMINAL_COLOR, width=1, style=Qt.DashLine),
            )
        self.cursor_trail = plot.plot(pen=pg.mkPen(TRAIL_COLOR, width=2.5))
        self.cursor_dot = plot.plot(
            symbol="o",
            symbolSize=10,
            symbolBrush=pg.mkBrush(CURSOR_COLOR),
            symbolPen=pg.mkPen("w"),
        )
        self.cursor_trail.setZValue(10)
        self.cursor_dot.setZValue(11)
        if len(drawn) < len(events):
            self._show_status(
                f"Plot 1 shows {len(drawn)} of {len(events)} readouts, evenly "
                "spaced; the sampling PSF uses all of them."
            )

    def _image_geometry(self, horizontal, vertical):
        """FOV and matrix of the chosen plane, from the detected image."""
        layout = self.layout_info
        if layout.fov_m is None or layout.matrix is None:
            return None
        if (horizontal, vertical) != (0, 1):
            return None
        return tuple(layout.fov_m), tuple(layout.matrix)

    def _nominal_kmax(self, horizontal, vertical):
        geometry = self._image_geometry(horizontal, vertical)
        if geometry is None:
            return None
        fov, matrix = geometry
        return tuple(n / (2.0 * width) for n, width in zip(matrix, fov))

    def _render_sequence(self):
        plot = self.sequence_plot
        start, stop = self.window_s
        edges, values = step_waveform(self.kspace, start, stop)
        self._waveform = (edges, values)
        to_mt = 1e3 / self.gamma_hz_per_t
        peak = 0.0
        for index, axis in enumerate(AXES):
            amplitude = values[:, index] * to_mt
            if amplitude.size and np.any(amplitude != 0):
                peak = max(peak, float(np.max(np.abs(amplitude))))
            plot.plot(
                edges * 1e3,
                amplitude,
                stepMode="center",
                pen=pg.mkPen(AXIS_COLORS[axis], width=1.3),
                name=f"G{axis}",
            )
        scale = peak if peak > 0 else 1.0
        program = self.context.program
        rf_events = [
            event
            for event in program.rf_events
            if event.end_s >= start and event.start_s <= stop
        ]
        if rf_events:
            rf_peak = max(
                float(np.max(np.abs(event.samples_hz))) for event in rf_events
            )
            # One step curve with zero cells between the pulses.
            rf_edges = []
            rf_values = []
            for order, event in enumerate(rf_events):
                if order:
                    rf_values.append(np.zeros(1))
                rf_edges.append(
                    event.start_s
                    + np.arange(event.samples_hz.size + 1) * event.raster_s
                )
                rf_values.append(np.abs(event.samples_hz) / max(rf_peak, 1e-12) * scale)
            plot.plot(
                np.concatenate(rf_edges) * 1e3,
                np.concatenate(rf_values),
                stepMode="center",
                fillLevel=0,
                brush=pg.mkBrush(124, 58, 237, 40),
                pen=pg.mkPen(RF_COLOR, width=1),
                name="RF |B1| (scaled)",
            )
        windows = [
            (event.start_s, event.end_s)
            for event in program.adc_events
            if event.end_s >= start and event.start_s <= stop
        ]
        if windows:
            # A step curve filled between -scale and +scale inside the ADC
            # windows shades them without one item per readout.
            adc_edges = [start]
            adc_values = []
            for begin, end in windows:
                adc_edges += [max(begin, start), min(end, stop)]
                adc_values += [-scale, scale]
            adc_edges.append(stop)
            adc_values.append(-scale)
            adc = plot.plot(
                np.asarray(adc_edges) * 1e3,
                np.asarray(adc_values),
                stepMode="center",
                fillLevel=-scale,
                brush=pg.mkBrush(*ADC_BRUSH),
                pen=None,
                name="ADC",
            )
            adc.setZValue(-10)
        plot.addItem(self.cursor_lines[0])
        plot.setXRange(start * 1e3, stop * 1e3, padding=0.0)

    def _render_k_time(self):
        plot = self.k_time_plot
        start, stop = self.window_s
        times, k = self.kspace.timeline(start, stop)
        for index, axis in enumerate(AXES):
            plot.plot(
                times * 1e3,
                k[:, index],
                pen=pg.mkPen(AXIS_COLORS[axis], width=1.3),
                name=f"k{axis}",
                connect="finite",
            )
        for pulse in self.kspace.rf_pulses:
            color = ROLE_LINE_COLORS.get(pulse.role)
            if color is None or not start <= pulse.center_s <= stop:
                continue
            plot.addItem(
                pg.InfiniteLine(
                    pos=pulse.center_s * 1e3,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(color, width=1, style=Qt.DashLine),
                )
            )
        plot.addItem(self.cursor_lines[1])

    def _render_psf(self, events):
        plot = self.psf_plot
        self.compute_psf_button.hide()
        horizontal, vertical = self._plane()
        geometry = self._image_geometry(horizontal, vertical)
        if geometry is None:
            self.psf = None
            self._psf_key = None
            self.psf_image.clear()
            self.psf_message.setText(
                "The sampling PSF needs the in-plane FOV and matrix of a "
                "detected image (kx × ky)."
            )
            self.psf_message.setPos(0, 0)
            plot.setTitle("4 · Sampling PSF", color="#0f172a")
            return
        fov, matrix = geometry
        key = self._psf_selection_key(events)
        if key != self._psf_key:
            k = np.concatenate([self.kspace.event_k(event) for event in events])
            if k.shape[0] > AUTO_PSF_SAMPLES and key != self._psf_requested_key:
                self.psf = None
                self._psf_key = None
                self.psf_image.clear()
                self.psf_message.setText(
                    f"{k.shape[0]:,} samples: click Compute sampling PSF"
                )
                self.psf_message.setPos(0, 0)
                self.compute_psf_button.show()
                plot.setTitle("4 · Sampling PSF", color="#0f172a")
                return
            try:
                self.psf = sampling_psf(k[:, [horizontal, vertical]], fov, matrix)
            except Exception as exc:
                self.psf = None
                self._psf_key = None
                self.psf_image.clear()
                self.psf_message.setText(f"No sampling PSF: {exc}")
                self.psf_message.setPos(0, 0)
                return
            self._psf_key = key
        psf = self.psf
        self.psf_message.setText("")
        magnitude = np.abs(psf.image)
        if self.psf_scale.currentText() == "dB":
            peak = max(psf.peak, 1e-30)
            data = 20 * np.log10(
                np.maximum(magnitude / peak, 10 ** (IMAGE_FLOOR_DB / 20))
            )
            levels = (IMAGE_FLOOR_DB, 0.0)
        else:
            data = magnitude
            levels = (0.0, max(psf.peak, 1e-12))
        width_mm, height_mm = (value * 1e3 for value in fov)
        self.psf_image.setImage(data.T, autoLevels=False, levels=levels)
        self.psf_image.setRect(
            QRectF(-width_mm / 2, -height_mm / 2, width_mm, height_mm)
        )
        plot.setLabel("bottom", f"{AXES[horizontal]} [mm]")
        plot.setLabel("left", f"{AXES[vertical]} [mm]")
        title = f"4 · Sampling PSF · peak {psf.peak:.2f}"
        if self._report_sampling_gap():
            title += (
                f" · gap {psf.sampling_gap:.2f}/FOV · alias-free FOV "
                f"{psf.nyquist_fov_m[0] * 1e3:.0f} mm"
            )
        plot.setTitle(title, color="#0f172a")

    def _psf_selection_key(self, events):
        return (id(self.kspace), tuple(events), self._plane())

    def _compute_psf_requested(self):
        self._psf_requested_key = self._psf_selection_key(self.selected_events())
        self.compute_psf_button.hide()
        self.render()

    def _report_sampling_gap(self):
        # The gap metric reads 0.5 for Nyquist-spaced lines or spiral turns
        # but sqrt(2)/2 at the cell centres of a full Cartesian grid.
        return self.psf is not None and self.layout_info.kind != "cartesian"

    def _render_info(self, events):
        kspace = self.kspace
        rows = []
        layout = self.layout_info
        rows.append(("Detected acquisition", layout.description or "—"))
        if layout.fov_m is not None and layout.matrix is not None:
            rows.append(
                (
                    "Image FOV / matrix",
                    f"{layout.fov_m[0] * 1e3:.1f} × {layout.fov_m[1] * 1e3:.1f} mm, "
                    f"{layout.matrix[0]} × {layout.matrix[1]}",
                )
            )
        roles = [pulse.role for pulse in kspace.rf_pulses]
        rows.append(
            (
                "RF pulses",
                ", ".join(
                    f"{roles.count(role)} {role}" for role in dict.fromkeys(roles)
                )
                or "none",
            )
        )
        rows.append(("ADC events (shown)", f"{kspace.num_adc_events} ({len(events)})"))
        event = events[0]
        summary = kspace.event_summary(event)
        k = kspace.event_k(event)
        shape = classify_readout(k).replace("_", " ")
        rows.append((f"Readout {event + 1} shape", shape))
        dwell = (
            summary["readout_duration_s"] / max(summary["samples"] - 1, 1)
            if summary["samples"] > 1
            else float("nan")
        )
        rows.append(
            (
                f"Readout {event + 1} samples",
                f"{summary['samples']} × {_format(dwell * 1e6, '{:.2f}')} µs = "
                f"{summary['readout_duration_s'] * 1e3:.2f} ms",
            )
        )
        rows.append(
            (
                f"Readout {event + 1} timing",
                f"starts {_format(summary['readout_start_s'] * 1e3, '{:.2f}')} ms "
                "after excitation, k-centre at "
                f"{_format(summary['echo_time_s'] * 1e3, '{:.2f}')} ms",
            )
        )
        selected = np.concatenate([kspace.event_k(index) for index in events])
        extent = np.nanmax(np.abs(selected), axis=0)
        text = ", ".join(f"k{axis} {value:.1f}" for axis, value in zip(AXES, extent))
        horizontal, vertical = self._plane()
        nominal = self._nominal_kmax(horizontal, vertical)
        if nominal is not None:
            text += (
                f" (nominal {nominal[0]:.1f}, {nominal[1]:.1f} in "
                f"k{AXES[horizontal]}, k{AXES[vertical]})"
            )
        rows.append(("Max |k| shown [1/m]", text))
        edges, values = self._waveform
        to_mt = 1e3 / self.gamma_hz_per_t
        peak = np.max(np.abs(values), axis=0) * to_mt if values.size else np.zeros(3)
        slew = peak_slew_t_per_m_per_s(edges, values, self.gamma_hz_per_t)
        rows.append(
            (
                "Peak gradient in window",
                " / ".join(f"{value:.2f}" for value in peak) + " mT/m (x / y / z)",
            )
        )
        rows.append(
            (
                "Peak slew in window",
                " / ".join(f"{value:.1f}" for value in slew) + " T/m/s (x / y / z)",
            )
        )
        rows.append(
            (
                "Nucleus / B0",
                f"{self.context.nucleus}, γ = {self.gamma_hz_per_t / 1e6:.3f} MHz/T, "
                f"B0 = {self.context.field_strength_t:g} T (k is independent of B0)",
            )
        )
        if self._report_sampling_gap():
            psf = self.psf
            rows.append(
                (
                    "Sampling",
                    f"largest gap {psf.sampling_gap:.2f}/FOV (Nyquist ≤ 0.5), "
                    "alias-free FOV ≈ "
                    f"{psf.nyquist_fov_m[0] * 1e3:.0f} × "
                    f"{psf.nyquist_fov_m[1] * 1e3:.0f} mm",
                )
            )
        imperfections = self.imperfections()
        if not imperfections.is_ideal and self.ideal_kspace is not None:
            ideal = np.concatenate([self.ideal_kspace.event_k(i) for i in events])
            deviation = np.nanmax(np.linalg.norm(selected - ideal, axis=1))
            rows.append(
                (
                    "Deviation from ideal",
                    f"max {deviation:.2f} 1/m"
                    + (
                        f" = {deviation * layout.fov_m[0]:.2f} k-space cells"
                        if layout.fov_m is not None
                        else ""
                    ),
                )
            )
        self.info_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.info_table.setItem(row, 0, QTableWidgetItem(name))
            self.info_table.setItem(row, 1, QTableWidgetItem(value))
        self._fit_info_table()

    def _fit_info_table(self):
        table = self.info_table
        height = table.horizontalHeader().height() + 2 * table.frameWidth()
        height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(height)

    def _render_cursor(self):
        kspace = self.kspace
        start, stop = self.window_s
        self._updating = True
        try:
            for line in self.cursor_lines:
                line.setValue(self.cursor_s * 1e3)
            position = 0
            if stop > start:
                position = int(
                    round((self.cursor_s - start) / (stop - start) * SLIDER_STEPS)
                )
            self.time_slider.setValue(position)
        finally:
            self._updating = False
        if kspace is None or self.cursor_dot is None:
            self.time_label.setText("")
            return
        excitations = kspace.excitation_times_s
        before = excitations[excitations <= self.cursor_s]
        horizontal, vertical = self._plane()
        if before.size:
            times, k = kspace.timeline(float(before[-1]), self.cursor_s)
            self.cursor_trail.setData(
                k[:, horizontal], k[:, vertical], connect="finite"
            )
            since = f", {1e3 * (self.cursor_s - before[-1]):.3f} ms after excitation"
        else:
            self.cursor_trail.setData([], [])
            since = ", before the first excitation"
        current = kspace.k_at(np.asarray([self.cursor_s]))[0]
        if np.all(np.isfinite(current)):
            self.cursor_dot.setData([current[horizontal]], [current[vertical]])
        else:
            self.cursor_dot.setData([], [])
        self.time_label.setText(f"t = {self.cursor_s * 1e3:.3f} ms{since}")
