"""Spiral trajectory designer: k-space, gradients, slew and sampling PSF."""

from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..sequence.scanner import ScannerParameters
from ..sequence.spiral_design import SpiralDesign, design_spiral
from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .plot_interaction import AXIS_ZOOM_TOOLTIP

AXIS_COLORS = {"x": "#dc2626", "y": "#16a34a"}
MAGNITUDE_COLOR = "#0f172a"
LIMIT_COLOR = "#e11d48"
NYQUIST_COLOR = "#0f766e"
WARNING_STYLE = "color: #b45309;"
PLOT_MINIMUM_HEIGHT = 240
# Above this many samples the PSF image uses the voxel grid only.
PSF_OVERSAMPLING_SAMPLE_LIMIT = 4096
PSF_EXTENTS = {"1 FOV": 1.0, "2 FOV (show aliasing)": 2.0}


class SpiralTrajectoryDesigner(QDialog):
    """Design the single-shot spiral of Sequence Mode within scanner limits.

    The design is recalculated on every change. ``applied`` emits the
    parameters of the spiral sequence, ``fov_m``, ``matrix``, ``turns`` and
    the ``sampling_bandwidth_hz`` that the limits allow, so that the
    generated sequence plays the designed readout.
    """

    applied = pyqtSignal(dict)

    def __init__(
        self,
        *,
        fov_m=(0.12, 0.12),
        matrix=(32, 32),
        sampling_bandwidth_hz: float = 50e3,
        turns: Optional[float] = None,
        nucleus: str = "H1",
        scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spiral trajectory designer")
        self.resize(1180, 820)
        self.scanner_parameters = ScannerParameters.from_mapping(scanner_parameters)
        self.design: Optional[SpiralDesign] = None
        self.error: Optional[str] = None
        self.recalculate_timer = QTimer(self)
        self.recalculate_timer.setSingleShot(True)
        self.recalculate_timer.setInterval(150)
        self.recalculate_timer.timeout.connect(self.recalculate)

        layout = QHBoxLayout(self)
        layout.addWidget(self._input_panel(), 0)
        layout.addWidget(self._result_panel(), 1)

        self.nucleus.setCurrentText(
            nucleus if nucleus in NUCLEUS_GAMMA_HZ_PER_T else "H1"
        )
        for spin, value in zip((self.fov_x, self.fov_y), fov_m):
            spin.setValue(float(value) * 1e3)
        for spin, value in zip((self.matrix_x, self.matrix_y), matrix):
            spin.setValue(int(value))
        self.bandwidth.setValue(float(sampling_bandwidth_hz) / 1e3)
        self.turns.setValue(
            float(turns) if turns is not None else max(1.0, min(matrix) / 2.0)
        )
        self.max_grad.setValue(self.scanner_parameters.max_grad_mtm)
        self.max_slew.setValue(self.scanner_parameters.max_slew_tms)
        for widget in self._inputs():
            signal = (
                widget.currentIndexChanged
                if isinstance(widget, QComboBox)
                else widget.valueChanged
            )
            signal.connect(self._schedule)
        self.recalculate()

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
        return box

    @staticmethod
    def _matrix_spin(tooltip):
        box = QSpinBox()
        box.setRange(2, 256)
        box.setToolTip(tooltip)
        box.setKeyboardTracking(False)
        return box

    def _input_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(380)
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)

        trajectory = QGroupBox("Trajectory")
        form = QFormLayout(trajectory)
        self.nucleus = QComboBox()
        self.nucleus.addItems(list(NUCLEUS_GAMMA_HZ_PER_T))
        self.nucleus.setToolTip(
            "Nucleus of the readout. k-space depends only on ∫γG dt, so a "
            "nucleus with a smaller γ needs a stronger, faster gradient for "
            "the same trajectory."
        )
        form.addRow("Nucleus", self.nucleus)
        self.fov_x = self._spin(1.0, 10000.0, 1, " mm", "Field of view along x")
        self.fov_y = self._spin(1.0, 10000.0, 1, " mm", "Field of view along y")
        form.addRow("FOV x", self.fov_x)
        form.addRow("FOV y", self.fov_y)
        self.matrix_x = self._matrix_spin("Image matrix along x; k_max = Nx / 2FOVx")
        self.matrix_y = self._matrix_spin("Image matrix along y; k_max = Ny / 2FOVy")
        form.addRow("Matrix x", self.matrix_x)
        form.addRow("Matrix y", self.matrix_y)
        self.bandwidth = self._spin(
            0.1,
            2000.0,
            3,
            " kHz",
            "Requested receiver bandwidth. The readout has Nx·Ny samples; "
            "the bandwidth is lowered when the spiral would exceed the "
            "gradient amplitude or slew limit.",
            50.0,
        )
        form.addRow("Requested bandwidth", self.bandwidth)
        turns_row = QHBoxLayout()
        self.turns = self._spin(
            0.25,
            512.0,
            2,
            "",
            "Revolutions of the single centre-out interleaf. Fewer than "
            "max(N)/2 turns undersample k-space radially and "
            "alias the image at a smaller FOV.",
            8.0,
        )
        turns_row.addWidget(self.turns, 1)
        self.nyquist_button = QPushButton("Nyquist")
        self.nyquist_button.setToolTip(
            "Set the fewest turns that sample k-space at the Nyquist rate of "
            "the FOV: max(Nx, Ny) / 2"
        )
        self.nyquist_button.clicked.connect(self._set_nyquist_turns)
        turns_row.addWidget(self.nyquist_button)
        form.addRow("Revolutions", turns_row)
        column.addWidget(trajectory)

        limits = QGroupBox("Gradient limits")
        form = QFormLayout(limits)
        self.max_grad = self._spin(
            0.1,
            10000.0,
            1,
            " mT/m",
            "Maximum gradient amplitude per axis, initialised from the "
            "scanner profile. Changing it here does not change the profile.",
        )
        self.max_slew = self._spin(
            1.0,
            100000.0,
            1,
            " T/m/s",
            "Maximum slew rate per axis, initialised from the scanner profile. "
            "Changing it here does not change the profile.",
        )
        form.addRow("Max gradient", self.max_grad)
        form.addRow("Max slew rate", self.max_slew)
        column.addWidget(limits)

        psf = QGroupBox("Sampling PSF")
        form = QFormLayout(psf)
        self.t2_star = self._spin(
            0.0,
            10000.0,
            2,
            " ms",
            "T2* decay during the readout; 0 disables it. Decay weights the "
            "outer k-space and broadens the PSF.",
        )
        self.t2_star.setSpecialValueText("off")
        self.off_resonance = self._spin(
            -10000.0,
            10000.0,
            1,
            " Hz",
            "Off-resonance of the point source. The phase accrued during the "
            "readout blurs a spiral PSF into a ring instead of shifting it.",
        )
        self.psf_extent = QComboBox()
        self.psf_extent.addItems(list(PSF_EXTENTS))
        self.psf_extent.setToolTip(
            "Image extent. With 2 FOV the aliasing ring of an undersampled "
            "spiral becomes visible."
        )
        form.addRow("T2*", self.t2_star)
        form.addRow("Off-resonance", self.off_resonance)
        form.addRow("Image extent", self.psf_extent)
        column.addWidget(psf)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet(WARNING_STYLE)
        column.addWidget(self.message)
        column.addStretch(1)

        buttons = QDialogButtonBox()
        self.export_button = buttons.addButton("Export…", QDialogButtonBox.ActionRole)
        self.export_button.setToolTip(
            "Save k-space, sample times and gradient waveforms as .npz"
        )
        self.export_button.clicked.connect(self._export_requested)
        self.apply_button = buttons.addButton("Apply", QDialogButtonBox.ApplyRole)
        self.apply_button.setToolTip(
            "Set FOV, matrix, revolutions and the achievable bandwidth in the "
            "spiral sequence"
        )
        self.apply_button.clicked.connect(self.apply)
        close = buttons.addButton(QDialogButtonBox.Close)
        close.clicked.connect(self.reject)
        column.addWidget(buttons)
        return panel

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

    def _result_panel(self):
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        self.kspace_plot = self._plot(
            "1 · k-space",
            "kx [1/m]",
            "ky [1/m]",
            "ADC samples of the spiral, coloured from the first to the last "
            "sample. Dashed teal: k_max of the image matrix.",
            legend=False,
        )
        self.kspace_plot.setAspectLocked(True)
        self.gradient_plot = self._plot(
            "2 · Gradients",
            "Time [ms]",
            "Gradient [mT/m]",
            "Physical gradient per axis for the chosen nucleus; red dashed: "
            "amplitude limit.",
        )
        self.slew_plot = self._plot(
            "3 · Slew rate",
            "Time [ms]",
            "Slew [T/m/s]",
            "Slew rate per axis including the ramps to zero at both ends; "
            "red dashed: slew limit.",
        )
        self.slew_plot.setXLink(self.gradient_plot)
        self.psf_plot = self._plot(
            "4 · Sampling PSF",
            "x [mm]",
            "y [mm]",
            "Magnitude of the density-compensated gridding image of a point "
            "source at the FOV centre, with the T2* and off-resonance set on "
            "the left, relative to the peak of fully sampled Cartesian "
            "k-space.",
            legend=False,
        )
        self.psf_plot.setAspectLocked(True)
        self.psf_image = pg.ImageItem(axisOrder="row-major")
        self.psf_image.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(nPts=256)
        )
        self.psf_plot.addItem(self.psf_image)
        grid.addWidget(self.kspace_plot, 0, 0)
        grid.addWidget(self.gradient_plot, 0, 1)
        grid.addWidget(self.psf_plot, 1, 0)
        grid.addWidget(self.slew_plot, 1, 1)
        column.addLayout(grid, 1)

        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.info_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.info_table.setAlternatingRowColors(True)
        self.info_table.setFixedHeight(275)
        column.addWidget(self.info_table)
        return panel

    def _inputs(self):
        return (
            self.nucleus,
            self.fov_x,
            self.fov_y,
            self.matrix_x,
            self.matrix_y,
            self.bandwidth,
            self.turns,
            self.max_grad,
            self.max_slew,
            self.t2_star,
            self.off_resonance,
            self.psf_extent,
        )

    # Design ---------------------------------------------------------------

    def _schedule(self, *_):
        self.recalculate_timer.start()

    def _set_nyquist_turns(self):
        self.turns.setValue(max(self.matrix_x.value(), self.matrix_y.value()) / 2.0)

    def parameters(self) -> dict:
        """Design inputs as keyword arguments of :func:`design_spiral`."""
        scanner = self.scanner_parameters.to_dict()
        scanner["max_grad_mtm"] = self.max_grad.value()
        scanner["max_slew_tms"] = self.max_slew.value()
        return {
            "fov_m": (self.fov_x.value() / 1e3, self.fov_y.value() / 1e3),
            "matrix": (self.matrix_x.value(), self.matrix_y.value()),
            "sampling_bandwidth_hz": self.bandwidth.value() * 1e3,
            "spiral_turns": self.turns.value(),
            "nucleus": self.nucleus.currentText(),
            "scanner_parameters": scanner,
        }

    def recalculate(self):
        self.recalculate_timer.stop()
        try:
            self.design = design_spiral(**self.parameters())
            self.error = None
        except ValueError as exc:
            self.design = None
            self.error = str(exc)
        self.psf = None
        if self.design is not None:
            samples = self.design.k_cyc_per_m.shape[0]
            t2_star_ms = self.t2_star.value()
            self.psf = self.design.sampling_psf(
                t2_star_s=t2_star_ms / 1e3 if t2_star_ms > 0 else None,
                off_resonance_hz=self.off_resonance.value(),
                field_of_views=PSF_EXTENTS[self.psf_extent.currentText()],
                oversampling=2 if samples <= PSF_OVERSAMPLING_SAMPLE_LIMIT else 1,
            )
        self.apply_button.setEnabled(self.design is not None)
        self.export_button.setEnabled(self.design is not None)
        self.render()

    def warnings(self) -> list:
        design = self.design
        if design is None:
            return [self.error] if self.error else []
        notes = []
        if design.bandwidth_limited:
            notes.append(
                f"The gradient limits lower the bandwidth from "
                f"{design.requested_bandwidth_hz / 1e3:.3f} kHz to "
                f"{design.sampling_bandwidth_hz / 1e3:.3f} kHz."
            )
        if design.turn_spacing > 1.0 + 1e-9:
            fov = design.alias_free_fov_m
            notes.append(
                f"{design.turns:g} turns undersample k-space "
                f"{design.turn_spacing:.2f}×: the image aliases outside "
                f"{fov[0] * 1e3:.1f} × {fov[1] * 1e3:.1f} mm. Use at least "
                f"{max(design.matrix) / 2:g} turns."
            )
        if design.nucleus != "H1":
            notes.append(
                "The sequence builder checks the gradient limits with the 1H "
                "gyromagnetic ratio. Apply this design so that the "
                f"{design.nucleus} gradients stay within the limits."
            )
        return notes

    # Apply and export -----------------------------------------------------

    def applied_parameters(self) -> dict:
        """Spiral sequence parameters that reproduce the design.

        The bandwidth is rounded up to the 1 Hz resolution of the sequence
        form, so that the builder chooses the designed dwell time.
        """
        design = self.design
        if design is None:
            raise ValueError(self.error or "no valid spiral design")
        return {
            "fov_m": design.fov_m,
            "matrix": design.matrix,
            "turns": design.turns,
            "sampling_bandwidth_hz": float(math.ceil(design.sampling_bandwidth_hz)),
        }

    def apply(self):
        if self.design is not None:
            self.applied.emit(self.applied_parameters())

    def export_design(self, path):
        if self.design is None:
            raise ValueError(self.error or "no valid spiral design")
        np.savez(path, **self.design.to_dict())

    def _export_requested(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export spiral trajectory", "spiral_trajectory.npz", "NumPy (*.npz)"
        )
        if path:
            self.export_design(path)

    # Rendering ------------------------------------------------------------

    def render(self):
        for plot in (self.kspace_plot, self.gradient_plot, self.slew_plot):
            plot.clear()
        self.psf_image.clear()
        self.message.setText("\n\n".join(self.warnings()))
        design = self.design
        if design is None:
            self.info_table.setRowCount(0)
            return
        self._render_kspace(design)
        self._render_waveforms(design)
        self._render_psf()
        self._render_table(design)

    def _render_kspace(self, design):
        k = design.k_cyc_per_m
        colors = pg.colormap.get("viridis").map(
            np.linspace(0.0, 1.0, k.shape[0]), mode="qcolor"
        )
        self.kspace_plot.plot(k[:, 0], k[:, 1], pen=pg.mkPen("#94a3b8", width=0.8))
        self.kspace_plot.addItem(
            pg.ScatterPlotItem(
                k[:, 0],
                k[:, 1],
                size=3,
                pen=None,
                brush=[pg.mkBrush(color) for color in colors],
            )
        )
        k_max = np.asarray(design.matrix) / (2.0 * np.asarray(design.fov_m))
        angle = np.linspace(0.0, 2.0 * np.pi, 361)
        self.kspace_plot.plot(
            k_max[0] * np.cos(angle),
            k_max[1] * np.sin(angle),
            pen=pg.mkPen(NYQUIST_COLOR, width=1, style=Qt.DashLine),
        )

    def _render_waveforms(self, design):
        edges = np.arange(design.gradient_hz_per_m.shape[0] + 1)
        edges = edges * design.gradient_raster_s * 1e3
        gradient = design.gradient_mtm
        for axis, role in enumerate(("x", "y")):
            self.gradient_plot.plot(
                edges,
                gradient[:, axis],
                stepMode="center",
                pen=pg.mkPen(AXIS_COLORS[role], width=1.2),
                name=f"G{role}",
            )
        self.gradient_plot.plot(
            edges,
            np.hypot(gradient[:, 0], gradient[:, 1]),
            stepMode="center",
            pen=pg.mkPen(MAGNITUDE_COLOR, width=1, style=Qt.DotLine),
            name="|G|",
        )
        slew = design.slew_tms
        times = np.arange(slew.shape[0]) * design.gradient_raster_s * 1e3
        for axis, role in enumerate(("x", "y")):
            self.slew_plot.plot(
                times,
                slew[:, axis],
                pen=pg.mkPen(AXIS_COLORS[role], width=1.2),
                name=f"S{role}",
            )
        for plot, limit in (
            (self.gradient_plot, design.max_grad_mtm),
            (self.slew_plot, design.max_slew_tms),
        ):
            for sign in (1.0, -1.0):
                plot.addItem(
                    pg.InfiniteLine(
                        pos=sign * limit,
                        angle=0,
                        pen=pg.mkPen(LIMIT_COLOR, width=1, style=Qt.DashLine),
                    )
                )

    def _render_psf(self):
        psf = self.psf
        if psf is None:
            return
        magnitude = np.abs(psf.image) / max(psf.peak, 1e-30)
        self.psf_image.setImage(magnitude, levels=(0.0, 1.0), autoLevels=False)
        y, x = (offset * 1e3 for offset in psf.offsets_m)
        step_x = x[1] - x[0] if x.size > 1 else 1.0
        step_y = y[1] - y[0] if y.size > 1 else 1.0
        self.psf_image.setRect(
            x[0] - step_x / 2, y[0] - step_y / 2, step_x * x.size, step_y * y.size
        )
        self.psf_plot.autoRange()

    def _render_table(self, design):
        psf = self.psf
        rows = [
            (
                "Bandwidth",
                f"{design.sampling_bandwidth_hz / 1e3:.3f} kHz "
                f"(dwell {design.dwell_s * 1e6:.2f} µs)",
            ),
            (
                "Readout",
                f"{design.readout_duration_s * 1e3:.2f} ms, "
                f"{design.k_cyc_per_m.shape[0]} samples",
            ),
            (
                "Peak gradient",
                f"{design.peak_gradient_mtm:.2f} of "
                f"{design.max_grad_mtm:.1f} mT/m ({design.nucleus})",
            ),
            (
                "Peak slew rate",
                f"{design.peak_slew_tms:.1f} of {design.max_slew_tms:.1f} T/m/s",
            ),
            (
                "Turn spacing",
                f"{design.turn_spacing:.2f} / FOV (Nyquist ≤ 1), alias-free FOV "
                f"{design.alias_free_fov_m[0] * 1e3:.1f} × "
                f"{design.alias_free_fov_m[1] * 1e3:.1f} mm",
            ),
        ]
        if psf is not None:
            rows += [
                (
                    "PSF FWHM",
                    f"x {psf.fwhm_voxels['x']:.2f}, y {psf.fwhm_voxels['y']:.2f} "
                    "voxels",
                ),
                (
                    "PSF peak",
                    f"{psf.peak:.3f} of Cartesian, shift x "
                    f"{psf.peak_offset_voxels['x']:.2f}, y "
                    f"{psf.peak_offset_voxels['y']:.2f} voxels",
                ),
                (
                    "Peak side lobe",
                    f"{100 * psf.peak_sidelobe_ratio:.1f} % beyond 2 voxels",
                ),
            ]
        self.info_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.info_table.setItem(row, 0, QTableWidgetItem(name))
            self.info_table.setItem(row, 1, QTableWidgetItem(value))
