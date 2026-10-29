"""PSF Analysis tab: point-spread function and slice profile of a sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QFrame,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..sequence.psf import (
    ROLE_INDEX,
    PointSource,
    PSFResult,
    SliceProfileResult,
    estimate_slice_profile,
    psf_from_result,
    simulate_point_sources,
)
from .plot_interaction import AXIS_ZOOM_TOOLTIP

ROLE_COLORS = {
    "read": "#2563eb",
    "phase": "#d97706",
    "partition": "#7c3aed",
    "x": "#2563eb",
    "y": "#d97706",
}
SPIRAL_COLOR = "#0f766e"
SLICE_COLOR = "#15803d"
OTHER_SLICE_COLOR = "#94a3b8"
NOMINAL_FILL = (21, 128, 61, 28)
# Profiles open on the main lobe and first side lobes; zoom out for the FOV.
PROFILE_VIEW_VOXELS = 4.0
# The page scrolls, so the plots keep a readable size in small windows.
PLOT_MINIMUM_HEIGHT = 340
# Floor of the dB image, relative to the ideal peak.
IMAGE_FLOOR_DB = -60.0
PLANES = {
    "Read × phase": ("phase", "read"),
    "Read × partition": ("partition", "read"),
    "Phase × partition": ("partition", "phase"),
}


@dataclass(frozen=True)
class PSFSourcePreset:
    label: str
    t1_ms: float
    t2_ms: float
    t2_star_ms: float
    hyperpolarized: bool


# Zero relaxation times mean "no relaxation"; see PSFAnalysisView._source().
PRESETS = (
    PSFSourcePreset("No relaxation", 0.0, 0.0, 0.0, False),
    PSFSourcePreset("1H tissue water (7 T)", 1900.0, 45.0, 25.0, False),
    PSFSourcePreset("HP [1-13C]pyruvate in vivo", 30000.0, 300.0, 20.0, True),
)
NUCLEUS_PRESETS = {"H1": 1, "C13": 2}
# Gradient-error display modes: label, (apply errors, correct trajectory).
GRADIENT_ERROR_MODES = {
    "Ideal gradients": (False, False),
    "Errors, uncorrected": (True, False),
    "Errors, trajectory-corrected": (True, True),
}


@dataclass(frozen=True)
class PSFAnalysisContext:
    """The sequence a PSF is computed for, supplied by Sequence Mode.

    ``token`` identifies the sequence revision; a different token later
    marks the computed PSF as stale.
    """

    program: object
    simulator: object
    simulation_timestep_s: float
    title: str
    token: object


@dataclass(frozen=True)
class PSFComputation:
    """Simulation and reconstructions of one point source."""

    context: PSFAnalysisContext
    source: PointSource
    simulation: object
    psf: PSFResult
    slice_profile: Optional[SliceProfileResult]
    slice_message: str


class PSFAnalysisThread(QThread):
    """Simulate the point source and its slice sweep without blocking Qt."""

    progress = pyqtSignal(int, int)
    stage = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, context, source, slice_profile):
        super().__init__()
        self.context = context
        self.source = source
        self.slice_profile = bool(slice_profile)
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        context = self.context
        options = dict(
            simulator=context.simulator,
            simulation_timestep_s=context.simulation_timestep_s,
            cancel_callback=lambda: self._cancel_requested,
        )
        try:
            self.stage.emit("Simulating the point source…")
            simulation = simulate_point_sources(
                context.program, (self.source,), **options
            )
            psf = psf_from_result(simulation, self.source)
            profile = None
            message = ""
            if not self.slice_profile:
                message = "Slice profile not requested."
            elif len(psf.roles) == 3:
                message = (
                    "3D acquisitions encode the partition direction; its "
                    "profile is shown in plot 2."
                )
            else:
                self.stage.emit("Sweeping the source through the slices…")
                try:
                    profile = estimate_slice_profile(
                        context.program,
                        self.source,
                        progress_callback=lambda done, total: self.progress.emit(
                            done, total
                        ),
                        **options,
                    )
                except ValueError as exc:
                    message = f"No slice profile: {exc}."
            if not self._cancel_requested:
                self.result_ready.emit(
                    PSFComputation(
                        context=context,
                        source=self.source,
                        simulation=simulation,
                        psf=psf,
                        slice_profile=profile,
                        slice_message=message,
                    )
                )
        except Exception as exc:
            if self._cancel_requested:
                self.failed.emit("PSF computation cancelled")
            else:
                self.failed.emit(str(exc))


def _role_label(role, frame):
    if role in ("x", "y", "z"):
        return f"Scanner {role}"
    if frame is None or not frame.is_axis_aligned:
        return role.capitalize()
    axis = frame.axis_codes[ROLE_INDEX[role]]
    return f"{role.capitalize()} ({axis})"


def _format(value, pattern):
    return "—" if value is None or not np.isfinite(value) else pattern.format(value)


class PSFAnalysisView(QWidget):
    """Point-spread function of the current sequence for one point source.

    ``context_provider`` returns a :class:`PSFAnalysisContext` or raises
    ``ValueError`` with a message for the user; ``token_provider`` returns the
    current sequence token, which marks older results stale; and
    ``nucleus_provider`` returns the Sequence Mode nucleus and chooses the
    initial source preset. ``imperfections_provider`` returns the
    :class:`GradientImperfections` of the k-space tab for the gradient-error
    modes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context_provider: Optional[Callable[[], PSFAnalysisContext]] = None
        self.nucleus_provider: Optional[Callable[[], str]] = None
        self.token_provider: Optional[Callable[[], object]] = None
        self.imperfections_provider: Optional[Callable[[], object]] = None
        self._gradient_error_cache = None
        self.computation: Optional[PSFComputation] = None
        self.psf: Optional[PSFResult] = None
        self.worker: Optional[PSFAnalysisThread] = None
        self._dirty = False
        self._preset_chosen = False
        # The plots keep their size; the metrics table follows below them.
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

        self.title_label = QLabel("PSF analysis")
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize() + 2, 12))
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        self.summary_label = QLabel(
            "The point-spread function is the reconstructed image of one "
            "isochromat simulated through the sequence. It includes "
            "relaxation during the readout, flip-angle and hyperpolarized "
            "depletion along the k-space order, off-resonance, and T2* through "
            "a Lorentzian frequency ensemble. Supported are Cartesian 2D "
            "frames and 3D volumes and 2D spiral frames, which are "
            "reconstructed by density-compensated gridding. Gradient errors "
            "come from the imperfections set in the k-space tab."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        layout.addWidget(self._source_group())

        controls = QHBoxLayout()
        self.slice_profile_enabled = QCheckBox("Slice profile")
        self.slice_profile_enabled.setChecked(True)
        self.slice_profile_enabled.setToolTip(
            "For 2D sequences, also sweep the source through the slice package "
            "along the slice normal. This simulates a few hundred sources."
        )
        controls.addWidget(self.slice_profile_enabled)
        self.compute_button = QPushButton("Compute PSF")
        self.compute_button.setToolTip(
            "Simulate the point source through the current sequence"
        )
        self.compute_button.clicked.connect(self.compute)
        controls.addWidget(self.compute_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        controls.addWidget(self.cancel_button)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #475569;")
        controls.addWidget(self.status_label, 1)
        controls.addWidget(QLabel("Frame"))
        self.frame_selector = QComboBox()
        self.frame_selector.setToolTip(
            "Reconstructed frame or volume whose PSF is shown"
        )
        self.frame_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.frame_selector.currentIndexChanged.connect(self._frame_changed)
        controls.addWidget(self.frame_selector)
        controls.addWidget(QLabel("Plane"))
        self.plane_selector = QComboBox()
        self.plane_selector.addItems(list(PLANES))
        self.plane_selector.setToolTip("Plane of a 3D PSF shown in plot 1")
        self.plane_selector.currentIndexChanged.connect(self._display_changed)
        controls.addWidget(self.plane_selector)
        self.image_scale = QComboBox()
        self.image_scale.addItems(["Linear", "dB"])
        self.image_scale.setToolTip(
            "Colour scale of the PSF image; dB shows side lobes down to "
            f"{IMAGE_FLOOR_DB:g} dB of the ideal peak"
        )
        self.image_scale.currentIndexChanged.connect(self._display_changed)
        controls.addWidget(self.image_scale)
        self.modulation_part = QComboBox()
        self.modulation_part.addItems(["Magnitude", "Phase"])
        self.modulation_part.setToolTip("Part of the k-space modulation in plot 3")
        self.modulation_part.currentIndexChanged.connect(self._display_changed)
        controls.addWidget(self.modulation_part)
        self.gradient_error_mode = QComboBox()
        self.gradient_error_mode.addItems(list(GRADIENT_ERROR_MODES))
        self.gradient_error_mode.setToolTip(
            "Gradient imperfections of the k-space tab (delay, scaling, eddy "
            "currents) shift the actual k-space positions by Δk, which adds "
            "exp(-2πi Δk·r0) to the source signal. Uncorrected: reconstruct "
            "on the nominal trajectory. Trajectory-corrected: reconstruct on "
            "the actual (measured) trajectory; 2D only. The simulation itself "
            "is not repeated."
        )
        self.gradient_error_mode.currentIndexChanged.connect(
            self._gradient_error_changed
        )
        controls.addWidget(self.gradient_error_mode)
        layout.addLayout(controls)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.image_plot = self._plot(
            "1 · PSF image",
            "Read offset [voxel]",
            "Phase offset [voxel]",
            "Magnitude of the reconstructed point source, relative to the peak "
            "of the ideal PSF (uniform k-space weighting). The axes are the "
            "displacement from the source in voxels over the whole field of "
            "view. For 3D, the plane passes through the peak.",
            legend=False,
        )
        self.image_plot.setAspectLocked(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(nPts=256)
        )
        self.image_plot.addItem(self.image_item)
        self.profile_plot = self._plot(
            "2 · Profiles through the peak",
            "Offset [voxel]",
            "|PSF| / ideal peak",
            "Solid: simulated PSF along each encoding direction through its "
            "peak. Dashed: ideal PSF with uniform k-space weighting. A wider "
            "main lobe is blurring; a shifted peak is a displacement.",
        )
        self.modulation_plot = self._plot(
            "3 · k-space modulation",
            "k [Δk]",
            "Weight / k-space centre",
            "The acquired k-space of the source with its encoding phase "
            "removed, relative to the k-space centre, along each encoding "
            "direction through the centre. Ideal weighting is flat at 1; decay "
            "blurs, a linear phase shifts, and steps cause ghosts.",
        )
        self.slice_plot = self._plot(
            "4 · Slice profile",
            "Position along slice normal [mm]",
            "|Signal| / peak",
            "Image value at the source's in-plane position while it is swept "
            "along the slice normal. Green band: the nominal slice. Grey: the "
            "same sweep seen in the other slices (cross-talk).",
        )
        self.slice_message = pg.TextItem(color="#475569", anchor=(0.5, 0.5))
        grid.addWidget(self.image_plot, 0, 0)
        grid.addWidget(self.profile_plot, 0, 1)
        grid.addWidget(self.modulation_plot, 1, 0)
        grid.addWidget(self.slice_plot, 1, 1)
        layout.addLayout(grid, 1)

        self.metrics_table = QTableWidget(0, 7)
        self.metrics_table.setHorizontalHeaderLabels(
            [
                "Direction",
                "Nominal [mm]",
                "FWHM [mm]",
                "FWHM / nominal",
                "Broadening vs ideal",
                "Peak shift [voxel]",
                "Side lobe (ideal) [%]",
            ]
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        # The page scrolls instead; _fit_metrics_table sizes it to its rows.
        self.metrics_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.metrics_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setToolTip(
            "Nominal: voxel size, or slice thickness for the slice row. "
            "Broadening: FWHM relative to the ideal PSF of the same matrix "
            "or trajectory (about 1.21 voxels for Cartesian k-space). Peak shift: for the slice, in slice "
            "thicknesses from the nominal centre. Side lobe: largest side lobe "
            "relative to the peak, with the ideal value in brackets."
        )
        self.metrics_summary = QLabel()
        self.metrics_summary.setWordWrap(True)
        self.metrics_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.metrics_summary)
        layout.addWidget(self.metrics_table)
        self._fit_metrics_table()
        self._set_display_controls_enabled(False)
        self.render()

    def _source_group(self):
        group = QGroupBox("Point source")
        grid = QGridLayout(group)
        self.preset_selector = QComboBox()
        self.preset_selector.addItems([preset.label for preset in PRESETS])
        self.preset_selector.setToolTip(
            "Fill the relaxation times and polarization with typical values"
        )
        self.preset_selector.activated.connect(self._preset_activated)
        grid.addWidget(QLabel("Preset"), 0, 0)
        grid.addWidget(self.preset_selector, 0, 1, 1, 3)

        def spin(minimum, maximum, decimals, suffix, tooltip, special=None):
            box = QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setDecimals(decimals)
            box.setSuffix(suffix)
            box.setToolTip(tooltip)
            box.setKeyboardTracking(False)
            if special is not None:
                box.setSpecialValueText(special)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return box

        self.position_mm = [
            spin(
                -1000.0,
                1000.0,
                2,
                " mm",
                f"Source position along scanner {axis}. Off-centre sources "
                "show the shift-variant PSF, e.g. near the slice edge.",
            )
            for axis in "xyz"
        ]
        grid.addWidget(QLabel("Position x / y / z"), 1, 0)
        for column, box in enumerate(self.position_mm):
            grid.addWidget(box, 1, column + 1)
        self.t1_ms = spin(0.0, 1e7, 1, " ms", "T1; ∞ = no recovery", "∞")
        self.t2_ms = spin(0.0, 1e7, 1, " ms", "T2; ∞ = no decay", "∞")
        self.t2_star_ms = spin(
            0.0,
            1e7,
            2,
            " ms",
            "T2* including T2. Off = no reversible dephasing. It must not "
            "exceed T2.",
            "off",
        )
        grid.addWidget(QLabel("T1 / T2 / T2*"), 2, 0)
        grid.addWidget(self.t1_ms, 2, 1)
        grid.addWidget(self.t2_ms, 2, 2)
        grid.addWidget(self.t2_star_ms, 2, 3)
        self.off_resonance_hz = spin(
            -1e6,
            1e6,
            1,
            " Hz",
            "Frequency offset of the source; EPI shifts it along the phase "
            "direction by Δf · echo spacing · N voxels.",
        )
        self.hyperpolarized = QCheckBox("Hyperpolarized")
        self.hyperpolarized.setToolTip(
            "Mz starts at the hyperpolarized level and relaxes toward the "
            "thermal level 1/ε, so every excitation depletes the signal."
        )
        self.hp_enhancement = spin(
            1.0,
            1e9,
            1,
            "",
            "Polarization enhancement ε over thermal equilibrium",
        )
        self.hp_enhancement.setPrefix("ε = ")
        self.hp_enhancement.setValue(10000.0)
        self.hyperpolarized.toggled.connect(self.hp_enhancement.setEnabled)
        grid.addWidget(QLabel("Δf / polarization"), 3, 0)
        grid.addWidget(self.off_resonance_hz, 3, 1)
        grid.addWidget(self.hyperpolarized, 3, 2)
        grid.addWidget(self.hp_enhancement, 3, 3)
        self._apply_preset(0)
        return group

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

    # Source ---------------------------------------------------------------

    def _preset_activated(self, index):
        self._preset_chosen = True
        self._apply_preset(index)

    def _apply_preset(self, index):
        preset = PRESETS[index]
        self.preset_selector.setCurrentIndex(index)
        self.t1_ms.setValue(preset.t1_ms)
        self.t2_ms.setValue(preset.t2_ms)
        self.t2_star_ms.setValue(preset.t2_star_ms)
        self.hyperpolarized.setChecked(preset.hyperpolarized)
        self.hp_enhancement.setEnabled(preset.hyperpolarized)

    def _choose_nucleus_preset(self):
        if self._preset_chosen or self.nucleus_provider is None:
            return
        index = NUCLEUS_PRESETS.get(str(self.nucleus_provider()))
        if index is not None and index != self.preset_selector.currentIndex():
            self._apply_preset(index)

    def source(self) -> PointSource:
        """Return the point source described by the controls."""

        def seconds(box, unbounded):
            return unbounded if box.value() == 0 else box.value() / 1000.0

        t2_s = seconds(self.t2_ms, 1e6)
        t2_star_s = seconds(self.t2_star_ms, None)
        if t2_star_s is not None and t2_star_s > t2_s:
            raise ValueError("T2* must not exceed T2")
        return PointSource(
            position_m=tuple(box.value() / 1000.0 for box in self.position_mm),
            t1_s=seconds(self.t1_ms, 1e6),
            t2_s=t2_s,
            t2_star_s=t2_star_s,
            off_resonance_hz=self.off_resonance_hz.value(),
            equilibrium_mz=(
                1.0 / self.hp_enhancement.value()
                if self.hyperpolarized.isChecked()
                else 1.0
            ),
        )

    # Computation ----------------------------------------------------------

    def compute(self):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            if self.context_provider is None:
                raise ValueError("No sequence is connected to the PSF analysis")
            source = self.source()
            context = self.context_provider()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            self.status_label.setStyleSheet("color: #b91c1c;")
            return
        self.status_label.setStyleSheet("color: #475569;")
        self.status_label.setText("Preparing…")
        self.compute_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = PSFAnalysisThread(
            context, source, self.slice_profile_enabled.isChecked()
        )
        self.worker.stage.connect(self.status_label.setText)
        self.worker.progress.connect(self._progress)
        self.worker.result_ready.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.request_cancel()
            self.status_label.setText("Cancelling…")

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _progress(self, done, total):
        self.status_label.setText(f"Slice profile: {done:,}/{total:,} positions")

    def _finished(self, computation):
        self.compute_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("")
        self.set_computation(computation)
        if self.gradient_error_mode.currentIndex() > 0:
            self._gradient_error_changed()

    def _failed(self, message):
        self.compute_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #475569;"
            if message == "PSF computation cancelled"
            else "color: #b91c1c;"
        )

    def set_computation(self, computation: Optional[PSFComputation]):
        """Show ``computation``, or the introduction when it is ``None``."""
        self.computation = computation
        self.psf = None if computation is None else computation.psf
        blocked = self.frame_selector.blockSignals(True)
        self.frame_selector.clear()
        if computation is not None:
            self.frame_selector.addItems(list(computation.psf.frame_labels))
            self.frame_selector.setCurrentIndex(computation.psf.frame)
        self.frame_selector.blockSignals(blocked)
        self._set_display_controls_enabled(computation is not None)
        self._dirty = True
        if self.isVisible():
            self.render()

    def _set_display_controls_enabled(self, enabled):
        psf = self.psf
        self.frame_selector.setEnabled(
            enabled and psf is not None and len(psf.frame_labels) > 1
        )
        self.plane_selector.setEnabled(
            enabled and psf is not None and len(psf.roles) == 3
        )
        self.image_scale.setEnabled(enabled)
        self.modulation_part.setEnabled(enabled)
        self.gradient_error_mode.setEnabled(enabled)

    def _gradient_error(self, program):
        """Return Δk per ADC sample for the k-space tab's imperfections."""
        if self.imperfections_provider is None:
            raise ValueError("No gradient imperfections are connected")
        imperfections = self.imperfections_provider()
        if imperfections.is_ideal:
            raise ValueError(
                "No gradient imperfections are set in the k-space tab; the "
                "PSF uses ideal gradients"
            )
        key = (id(program), repr(imperfections.to_dict()))
        if self._gradient_error_cache is None or self._gradient_error_cache[0] != key:
            from ..sequence.kspace_trajectory import gradient_error

            self._gradient_error_cache = (key, gradient_error(program, imperfections))
        return self._gradient_error_cache[1]

    def _psf_for(self, frame):
        """PSF of ``frame`` under the selected gradient-error mode."""
        computation = self.computation
        apply_errors, correct = GRADIENT_ERROR_MODES[
            self.gradient_error_mode.currentText()
        ]
        options = {}
        message = ""
        if apply_errors:
            try:
                options = dict(
                    gradient_error_cyc_per_m=self._gradient_error(
                        computation.context.program
                    ),
                    trajectory_correction=correct,
                )
            except ValueError as exc:
                message = str(exc)
        try:
            psf = psf_from_result(
                computation.simulation, computation.source, frame=frame, **options
            )
        except ValueError as exc:
            if not options:
                raise
            message = f"{exc}; showing the uncorrected PSF"
            options["trajectory_correction"] = False
            psf = psf_from_result(
                computation.simulation, computation.source, frame=frame, **options
            )
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #b45309;" if message else "color: #475569;"
        )
        return psf

    def _frame_changed(self, index):
        computation = self.computation
        if computation is None or index < 0 or index == self.psf.frame:
            return
        self.psf = self._psf_for(index)
        self._display_changed()

    def _gradient_error_changed(self, *_):
        if self.computation is None:
            return
        self.psf = self._psf_for(self.psf.frame)
        self._display_changed()

    def _display_changed(self, *_):
        self._dirty = True
        if self.isVisible():
            self.render()

    def showEvent(self, event):
        super().showEvent(event)
        self._choose_nucleus_preset()
        if self._dirty:
            self.render()
        else:
            self._render_summary()

    # Rendering ------------------------------------------------------------

    def _stale(self) -> bool:
        computation = self.computation
        if computation is None or self.token_provider is None:
            return False
        return self.token_provider() != computation.context.token

    def render(self):
        """Redraw all plots from the stored computation."""
        self._dirty = False
        psf = self.psf
        for plot in (self.profile_plot, self.modulation_plot, self.slice_plot):
            plot.clear()
        self.metrics_table.setRowCount(0)
        self._fit_metrics_table()
        if psf is None:
            self.image_item.clear()
            self.title_label.setText("PSF analysis")
            self.metrics_summary.setText("Choose a point source and click Compute PSF.")
            return
        self._render_summary()
        self._render_image(psf)
        self._render_profiles(psf)
        self._render_modulation(psf)
        self._render_slice_profile(psf)
        self._render_metrics(psf)

    def _render_summary(self):
        computation = self.computation
        if computation is None:
            return
        psf = self.psf
        source = computation.source
        title = f"{computation.context.title} · "
        if psf.kind == "spiral":
            title += "2D spiral frame"
        else:
            title += "3D volume" if len(psf.roles) == 3 else "2D frame"
        if len(psf.frame_labels) > 1:
            title += f" {psf.label}"
        self.title_label.setText(title)
        if self._stale():
            self.summary_label.setText(
                "The sequence changed after this PSF was computed. Click "
                "Compute PSF to update it."
            )
            self.summary_label.setStyleSheet("color: #b45309;")
            return
        position = ", ".join(f"{1000 * value:g}" for value in source.position_m)
        relaxation = []
        for name, value in (("T1", source.t1_s), ("T2", source.t2_s)):
            if value < 1e5:
                relaxation.append(f"{name} {1000 * value:g} ms")
        if source.t2_star_s is not None:
            relaxation.append(f"T2* {1000 * source.t2_star_s:g} ms")
        polarization = (
            f"hyperpolarized (ε = {1.0 / source.equilibrium_mz:g})"
            if source.equilibrium_mz < 1.0
            else "thermal"
        )
        gradients = {
            "none": "ideal gradients",
            "uncorrected": "gradient errors reconstructed on the nominal trajectory",
            "corrected": "gradient errors reconstructed on the actual trajectory",
        }[psf.gradient_error]
        self.summary_label.setText(
            f"Point source at ({position}) mm, "
            f"{', '.join(relaxation) or 'no relaxation'}, "
            f"Δf {source.off_resonance_hz:g} Hz, {polarization}, {gradients}."
        )
        self.summary_label.setStyleSheet("color: #15803d;")

    def _render_image(self, psf):
        roles = psf.roles
        if len(roles) == 3:
            row_role, column_role = PLANES[self.plane_selector.currentText()]
        else:
            row_role, column_role = roles
        row_axis, column_axis = roles.index(row_role), roles.index(column_role)
        magnitude = np.abs(psf.psf)
        if magnitude.ndim == 3:
            peak = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
            index = [slice(None)] * 3
            (other,) = {0, 1, 2} - {row_axis, column_axis}
            index[other] = peak[other]
            magnitude = magnitude[tuple(index)]
            # The remaining axes keep their order; rows come first.
            if row_axis > column_axis:
                magnitude = magnitude.T
        scale = float(np.abs(psf.ideal_psf).max()) or 1.0
        values = magnitude / scale
        if self.image_scale.currentText() == "dB":
            values = np.maximum(
                20.0 * np.log10(np.maximum(values, 1e-12)), IMAGE_FLOOR_DB
            )
            levels = (IMAGE_FLOOR_DB, max(0.0, float(values.max())))
        else:
            levels = (0.0, max(1.0, float(values.max())))
        self.image_item.setImage(values, levels=levels, autoLevels=False)
        rows = psf.offsets_m[row_axis] / psf.voxel_size_m[row_axis]
        columns = psf.offsets_m[column_axis] / psf.voxel_size_m[column_axis]
        row_step = rows[1] - rows[0] if rows.size > 1 else 1.0
        column_step = columns[1] - columns[0] if columns.size > 1 else 1.0
        rect = QRectF(
            columns[0] - column_step / 2.0,
            rows[0] - row_step / 2.0,
            column_step * columns.size,
            row_step * rows.size,
        )
        self.image_item.setRect(rect)
        frame = psf.encoding_frame
        self.image_plot.setLabel(
            "bottom", f"{_role_label(column_role, frame)} offset [voxel]"
        )
        self.image_plot.setLabel(
            "left", f"{_role_label(row_role, frame)} offset [voxel]"
        )
        # The whole field of view; the profiles zoom in on the main lobe.
        self.image_plot.getViewBox().setRange(rect=rect, padding=0.0)

    def _render_profiles(self, psf):
        plot = self.profile_plot
        scale = float(np.abs(psf.ideal_psf).max()) or 1.0
        for role in reversed(psf.roles):
            line, values, ideal = psf.profiles[role]
            size = psf.voxel_size_m[psf.roles.index(role)]
            color = ROLE_COLORS[role]
            label = _role_label(role, psf.encoding_frame)
            plot.plot(
                line / size,
                np.abs(values) / scale,
                pen=pg.mkPen(color, width=2),
                name=label,
            )
            plot.plot(
                line / size,
                np.abs(ideal) / scale,
                pen=pg.mkPen(color, width=1, style=Qt.DashLine),
            )
        plot.plot([], [], pen=pg.mkPen("#334155", style=Qt.DashLine), name="ideal")
        plot.setXRange(-PROFILE_VIEW_VOXELS, PROFILE_VIEW_VOXELS, padding=0.0)

    def _modulation_lines(self, psf):
        """Yield ``(role, k in Δk, values)`` through the k-space centre."""
        centre = tuple(int(np.argmin(np.abs(k))) for k in psf.k_cyc_per_m)
        reference = psf.modulation[centre]
        if reference == 0:
            reference = 1.0
        for axis, role in enumerate(psf.roles):
            k = np.asarray(psf.k_cyc_per_m[axis], dtype=float)
            index = list(centre)
            index[axis] = slice(None)
            values = psf.modulation[tuple(index)] / reference
            step = np.median(np.abs(np.diff(k))) if k.size > 1 else 1.0
            yield role, k / (step or 1.0), values

    @staticmethod
    def _spiral_modulation(psf):
        """Return ``(|k| in Δk, values)`` of all samples in acquisition order.

        ``offsets_m`` spans one FOV period per axis, so Δk = 1/FOV.
        """
        fov = [
            offsets.size * (offsets[1] - offsets[0]) if offsets.size > 1 else 1.0
            for offsets in psf.offsets_m
        ]
        radius = np.sqrt(
            sum((np.asarray(k) * size) ** 2 for k, size in zip(psf.k_cyc_per_m, fov))
        )
        modulation = np.asarray(psf.modulation)
        reference = modulation[int(np.argmin(radius))] if modulation.size else 1.0
        return radius, modulation / (reference if reference != 0 else 1.0)

    def _render_modulation(self, psf):
        plot = self.modulation_plot
        phase = self.modulation_part.currentText() == "Phase"
        plot.setLabel(
            "left",
            "Phase vs k-space centre [°]" if phase else "Weight / k-space centre",
        )
        if psf.kind == "spiral":
            plot.setLabel("bottom", "|k| [Δk], samples in acquisition order")
            radius, values = self._spiral_modulation(psf)
            y = np.degrees(np.unwrap(np.angle(values))) if phase else np.abs(values)
            plot.plot(
                radius,
                y,
                pen=pg.mkPen(SPIRAL_COLOR, width=1),
                symbol="o",
                symbolSize=3,
                symbolPen=None,
                symbolBrush=pg.mkBrush(SPIRAL_COLOR),
                name="spiral samples",
            )
            if not phase:
                plot.addItem(
                    pg.InfiniteLine(
                        1.0, angle=0, pen=pg.mkPen("#334155", style=Qt.DashLine)
                    )
                )
            self._limit_zoom(plot, [y], 5.0 if phase else 0.05)
            return
        plot.setLabel("bottom", "k [Δk]")
        drawn = []
        for role, k, values in self._modulation_lines(psf):
            order = np.argsort(k)
            y = (
                np.degrees(np.unwrap(np.angle(values[order])))
                if phase
                else np.abs(values[order])
            )
            plot.plot(
                k[order],
                y,
                pen=pg.mkPen(ROLE_COLORS[role], width=2),
                symbol="o",
                symbolSize=4,
                symbolPen=None,
                symbolBrush=pg.mkBrush(ROLE_COLORS[role]),
                name=_role_label(role, psf.encoding_frame),
            )
            drawn.append(y)
        if not phase:
            plot.addItem(
                pg.InfiniteLine(
                    1.0, angle=0, pen=pg.mkPen("#334155", style=Qt.DashLine)
                )
            )
        self._limit_zoom(plot, drawn, 5.0 if phase else 0.05)

    @staticmethod
    def _limit_zoom(plot, values, half_span):
        """Keep at least ``2 half_span`` of y range around nearly flat data.

        Auto-range would otherwise zoom into rounding noise, e.g. the
        constant weight of a source without relaxation.
        """
        values = np.concatenate([np.ravel(value) for value in values] or [[]])
        values = values[np.isfinite(values)]
        if values.size and np.ptp(values) < half_span:
            centre = 0.5 * (values.min() + values.max())
            plot.setYRange(centre - half_span, centre + half_span, padding=0.0)
        else:
            plot.enableAutoRange(axis="y")

    def _render_slice_profile(self, psf):
        plot = self.slice_plot
        computation = self.computation
        profile = computation.slice_profile
        if profile is None:
            self.slice_message.setText(computation.slice_message)
            plot.addItem(self.slice_message)
            self.slice_message.setPos(0.0, 0.5)
            plot.setXRange(-1.0, 1.0)
            plot.setYRange(0.0, 1.0)
            return
        frame = min(psf.frame, profile.profiles.shape[0] - 1)
        magnitude = np.abs(profile.profiles)
        scale = float(magnitude[frame].max()) or 1.0
        positions_mm = 1000.0 * profile.positions_m
        metrics = profile.metrics[frame]
        if metrics.nominal_thickness_m:
            half = 500.0 * metrics.nominal_thickness_m
            centre = 1000.0 * metrics.nominal_centre_m
            plot.addItem(
                pg.LinearRegionItem(
                    (centre - half, centre + half),
                    movable=False,
                    brush=pg.mkBrush(*NOMINAL_FILL),
                    pen=pg.mkPen(None),
                )
            )
        shown = {profile.slice_indices[frame]}
        other_named = False
        for index, slice_index in enumerate(profile.slice_indices):
            if slice_index in shown:
                continue
            shown.add(slice_index)
            plot.plot(
                positions_mm,
                magnitude[index] / scale,
                pen=pg.mkPen(OTHER_SLICE_COLOR, width=1),
                name=None if other_named else "other slices",
            )
            other_named = True
        plot.plot(
            positions_mm,
            magnitude[frame] / scale,
            pen=pg.mkPen(SLICE_COLOR, width=2),
            name=f"slice {profile.slice_indices[frame] + 1}",
        )
        axis = "".join(
            f"{value:+.2g}{name}"
            for value, name in zip(profile.normal_xyz, "xyz")
            if abs(value) > 1e-9
        )
        plot.setLabel("bottom", f"Position along slice normal ({axis}) [mm]")
        plot.enableAutoRange()

    def _render_metrics(self, psf):
        rows = []
        for role in psf.roles[::-1]:
            metrics = psf.axis_metrics[role]
            rows.append(
                (
                    _role_label(role, psf.encoding_frame),
                    _format(1000.0 * metrics.voxel_size_m, "{:.3g}"),
                    _format(1000.0 * metrics.fwhm_m, "{:.3g}"),
                    _format(metrics.fwhm_voxels, "{:.3f}"),
                    _format(metrics.broadening, "{:.3f}"),
                    _format(metrics.peak_offset_voxels, "{:+.3f}"),
                    _format(100.0 * metrics.peak_sidelobe_ratio, "{:.1f}")
                    + _format(100.0 * metrics.ideal_peak_sidelobe_ratio, " ({:.1f})"),
                )
            )
        lines = [
            f"Peak {psf.peak_efficiency:.3f} × the ideal PSF peak · "
            f"{100 * psf.voxel_energy_fraction:.1f} % of the PSF energy in the "
            f"source voxel (ideal {100 * psf.ideal_voxel_energy_fraction:.1f} %)."
        ]
        profile = self.computation.slice_profile
        if profile is not None:
            frame = min(psf.frame, len(profile.metrics) - 1)
            metrics = profile.metrics[frame]
            thickness = metrics.nominal_thickness_m
            rows.append(
                (
                    "Slice",
                    _format(None if not thickness else 1000.0 * thickness, "{:.3g}"),
                    _format(1000.0 * metrics.fwhm_m, "{:.3g}"),
                    _format(metrics.thickness_ratio, "{:.3f}"),
                    "—",
                    _format(
                        (
                            (metrics.centre_m - metrics.nominal_centre_m) / thickness
                            if thickness
                            else None
                        ),
                        "{:+.3f}",
                    ),
                    "—",
                )
            )
            if np.isfinite(metrics.out_of_slice_fraction):
                lines.append(
                    f"{100 * metrics.out_of_slice_fraction:.1f} % of the slice "
                    "profile area lies outside the nominal slice."
                )
        self.metrics_summary.setText(" ".join(lines))
        self.metrics_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.metrics_table.setItem(row, column, item)
        self._fit_metrics_table()

    def _fit_metrics_table(self):
        table = self.metrics_table
        rows = max(table.rowCount(), 1)
        table.setFixedHeight(
            rows * table.verticalHeader().defaultSectionSize()
            + table.horizontalHeader().sizeHint().height()
            + 2 * table.frameWidth()
        )
