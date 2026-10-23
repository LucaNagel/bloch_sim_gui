"""Spoiling Analysis tab: how the subvoxel grid recommendation is derived."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..sequence.spin_sampling import SpinSampling
from ..sequence.spoiling import SpinGridRecommendation, pathway_grid_coherence
from .plot_interaction import AXIS_ZOOM_TOOLTIP

VOXEL_COLOR = "#334155"
CURRENT_COLOR = "#d97706"
RECOMMENDED_COLOR = "#15803d"
PATHWAY_COLOR = (37, 99, 235)
SECOND_VOXEL_COLOR = "#7c3aed"
TARGET_COLOR = "#dc2626"
# Plot no more rows than pyqtgraph scatters smoothly; the strongest are kept.
MAX_PLOTTED_ROWS = 20000
# Cell lines stop being readable beyond this count.
MAX_GUIDE_LINES = 40
# The log error axis needs a floor for exactly zero errors.
ERROR_FLOOR_PERCENT = 1e-4


@dataclass(frozen=True)
class SpoilingAnalysisSnapshot:
    """Inputs and result of one spoiling-quality evaluation.

    ``phases`` holds one row per observed pathway in cycles across one
    phantom voxel along X/Y/Z, ``weights`` its amplitude relative to the
    wanted signal, and ``observation`` the x position used in the plots
    (crusher order or ADC number).
    """

    title: str
    summary: str
    phases: np.ndarray
    weights: np.ndarray
    observation: np.ndarray
    observation_label: str
    sampling: SpinSampling
    recommendation: Optional[SpinGridRecommendation]
    weighted: bool
    warning: bool = False
    target_error: float = 0.01


def _grid_text(counts, method):
    label = "midpoint" if method == "midpoint" else "jittered stratified"
    return f"{counts[0]}×{counts[1]}×{counts[2]} {label}"


class SpoilingAnalysisView(QWidget):
    """Plots that explain the gradient-spoiling subvoxel analysis.

    The snapshot is rendered lazily: while the tab is hidden, updates only
    store it, so editing sequence parameters stays responsive.
    """

    apply_recommendation_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot: Optional[SpoilingAnalysisSnapshot] = None
        self._dirty = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        self.title_label = QLabel("Spoiling analysis")
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize() + 2, 12))
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Axis"))
        self.axis_selector = QComboBox()
        self.axis_selector.addItems(["X", "Y", "Z"])
        self.axis_selector.setToolTip(
            "Voxel axis shown in the dephasing, coherence and position plots. "
            "It starts at the axis with the largest pathway dephasing."
        )
        self.axis_selector.currentIndexChanged.connect(self._axis_changed)
        controls.addWidget(self.axis_selector)
        controls.addStretch()
        self.apply_button = QPushButton("Apply recommended grid")
        self.apply_button.setToolTip(
            "Use the recommended subvoxel grid for the next simulation."
        )
        self.apply_button.clicked.connect(self.apply_recommendation_requested)
        controls.addWidget(self.apply_button)
        layout.addLayout(controls)

        self.stack = QStackedWidget()
        self.empty_label = QLabel()
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.empty_label.setStyleSheet("color: #475569;")
        self.stack.addWidget(self.empty_label)
        plots = QWidget()
        grid = QGridLayout(plots)
        grid.setContentsMargins(0, 0, 0, 0)
        self.pathway_plot = self._plot(
            "1 · Pathway dephasing",
            "Observation",
            "Dephasing [cycles/voxel]",
            "Every point is one coherence pathway at an observation. Darker and "
            "larger points carry more signal. Ringed points exceed the error "
            "target with the current grid. Dashed lines mark the first "
            "dephasing at which a regular grid on this axis rephases completely.",
            # Dephasing usually grows with the observation, so the upper
            # left corner is the emptiest place for the legend.
            legend_offset=(10, 10),
        )
        self.coherence_plot = self._plot(
            "2 · Signal of one voxel versus dephasing",
            "Dephasing [cycles/voxel]",
            "Signal / M0",
            "Grey: a uniform voxel (|sinc|). Coloured: the current and the "
            "recommended grid along the selected axis. Blue points: pathways "
            "on the current-grid curve, darker for larger weights. A pathway "
            "on a grid peak is an artificial echo.",
        )
        self.error_plot = self._plot(
            "3 · Weighted sampling error per pathway",
            "Observation",
            "Weight × error [%]",
            "Pathway weight times the deviation of the grid signal from the "
            "uniform voxel. The recommendation is the cheapest grid whose "
            "largest value stays below the red target line.",
        )
        self.error_plot.setLogMode(y=True)
        self.positions_plot = self._plot(
            "4 · Subvoxel spin positions in one voxel",
            "Offset [voxel]",
            "Offset [voxel]",
            "Positions of the current grid inside one phantom voxel, in the "
            "plane of its two most finely sampled axes. Jittered stratified "
            "points are shown for two neighbouring voxels.",
        )
        self.positions_plot.setAspectLocked(True)
        grid.addWidget(self.pathway_plot, 0, 0)
        grid.addWidget(self.coherence_plot, 0, 1)
        grid.addWidget(self.error_plot, 1, 0)
        grid.addWidget(self.positions_plot, 1, 1)
        self.stack.addWidget(plots)
        layout.addWidget(self.stack, 1)
        self.set_snapshot(None, "No spoiling analysis available yet.")

    @staticmethod
    def _plot(title, x_label, y_label, tooltip, legend_offset=(-10, 10)):
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.setTitle(title, color="#0f172a")
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.addLegend(
            offset=legend_offset,
            brush=pg.mkBrush(255, 255, 255, 180),
            pen=pg.mkPen("#cbd5e1"),
        )
        for axis in ("bottom", "left"):
            # Voxel offsets and cycles are dimensionless; no "x0.001" scaling.
            plot.getAxis(axis).enableAutoSIPrefix(False)
        plot.setToolTip(f"{tooltip}\n\n{AXIS_ZOOM_TOOLTIP}")
        plot.setMinimumHeight(220)
        return plot

    def set_snapshot(self, snapshot, message=None):
        """Show ``snapshot``, or ``message`` when no analysis is available."""
        self.snapshot = snapshot
        self._message = message or ""
        if snapshot is not None and snapshot.phases.size:
            magnitudes = np.max(np.abs(snapshot.phases), axis=0)
            blocked = self.axis_selector.blockSignals(True)
            self.axis_selector.setCurrentIndex(int(np.argmax(magnitudes)))
            self.axis_selector.blockSignals(blocked)
        self._dirty = True
        if self.isVisible():
            self.render()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self.render()

    def _axis_changed(self, _index):
        self._dirty = True
        if self.isVisible():
            self.render()

    def render(self):
        """Redraw all plots from the stored snapshot."""
        self._dirty = False
        snapshot = self.snapshot
        recommendation = None if snapshot is None else snapshot.recommendation
        is_current = recommendation is not None and (
            recommendation.counts_xyz == snapshot.sampling.counts_xyz
            and recommendation.method == snapshot.sampling.method
        )
        self.apply_button.setVisible(recommendation is not None)
        self.apply_button.setEnabled(recommendation is not None and not is_current)
        if recommendation is not None:
            self.apply_button.setText(
                "Recommended grid already selected"
                if is_current
                else "Apply "
                + _grid_text(recommendation.counts_xyz, recommendation.method)
                + " grid"
            )
        if snapshot is None or snapshot.phases.shape[0] == 0:
            self.title_label.setText(
                "Spoiling analysis" if snapshot is None else snapshot.title
            )
            self.summary_label.setText("")
            self.empty_label.setText(
                self._message
                if snapshot is None
                else snapshot.summary + "\n\nNo dephased pathway to plot."
            )
            self.stack.setCurrentIndex(0)
            return
        self.title_label.setText(snapshot.title)
        self.summary_label.setText(snapshot.summary)
        self.summary_label.setStyleSheet(
            "color: #b45309;" if snapshot.warning else "color: #15803d;"
        )
        self.stack.setCurrentIndex(1)

        phases, weights, observation = self._plotted_rows(snapshot)
        axis = self.axis_selector.currentIndex()
        axis_name = "XYZ"[axis]
        grids = [("current", snapshot.sampling.counts_xyz, snapshot.sampling.method)]
        if recommendation is not None and not is_current:
            grids.append(
                ("recommended", recommendation.counts_xyz, recommendation.method)
            )
        self._render_pathways(snapshot, phases, weights, observation, axis, grids)
        self._render_coherence(snapshot, phases, weights, axis, axis_name, grids)
        self._render_errors(snapshot, phases, weights, observation, grids)
        self._render_positions(snapshot.sampling, axis)

    @staticmethod
    def _plotted_rows(snapshot):
        phases = np.asarray(snapshot.phases, dtype=float)
        weights = np.asarray(snapshot.weights, dtype=float)
        observation = np.asarray(snapshot.observation, dtype=float)
        if weights.size > MAX_PLOTTED_ROWS:
            keep = np.argpartition(weights, -MAX_PLOTTED_ROWS)[-MAX_PLOTTED_ROWS:]
            keep.sort()
            phases, weights, observation = (
                phases[keep],
                weights[keep],
                observation[keep],
            )
        return phases, weights, observation

    @staticmethod
    def _weight_brushes(weights):
        # Quantized alpha keeps the number of distinct brush objects small.
        levels = np.clip(np.rint(np.clip(weights, 0.0, 1.0) * 8), 1, 8)
        brushes = {
            level: pg.mkBrush(*PATHWAY_COLOR, int(40 + 26 * level))
            for level in np.unique(levels)
        }
        return [brushes[level] for level in levels]

    def _render_pathways(self, snapshot, phases, weights, observation, axis, grids):
        plot = self.pathway_plot
        plot.clear()
        plot.setLabel("bottom", snapshot.observation_label)
        plot.setLabel("left", f"Dephasing along {'XYZ'[axis]} [cycles/voxel]")
        values = phases[:, axis]
        sizes = 4.0 + 6.0 * np.clip(weights, 0.0, 1.0)
        plot.addItem(
            pg.ScatterPlotItem(
                observation,
                values,
                size=sizes,
                pen=None,
                brush=self._weight_brushes(weights),
                name="pathway (size/opacity = weight)",
            )
        )
        _, _, errors = pathway_grid_coherence(phases, *grids[0][1:])
        failing = weights * errors > snapshot.target_error
        if np.any(failing):
            plot.addItem(
                pg.ScatterPlotItem(
                    observation[failing],
                    values[failing],
                    size=13,
                    pen=pg.mkPen(CURRENT_COLOR, width=1.5),
                    brush=None,
                    name=f"> {100 * snapshot.target_error:.3g}% error, current grid",
                )
            )
        for label, counts, method in grids:
            count = int(counts[axis])
            if method != "midpoint" or count < 2:
                continue
            color = CURRENT_COLOR if label == "current" else RECOMMENDED_COLOR
            pen = pg.mkPen(color, width=1.2, style=Qt.DashLine)
            # Only the first alias is drawn; its multiples rephase as well.
            for level in (count, -count):
                if np.any(values * np.sign(level) > 0.5 * count):
                    plot.addItem(pg.InfiniteLine(level, angle=0, pen=pen))
            plot.plot([], [], pen=pen, name=f"{label} grid first rephases (±{count})")
        # Keep zero dephasing in view so the slope of the train is readable.
        plot.setYRange(min(0.0, values.min()), max(0.0, values.max()), padding=0.08)

    def _render_coherence(self, snapshot, phases, weights, axis, axis_name, grids):
        plot = self.coherence_plot
        plot.clear()
        plot.setLabel("bottom", f"Dephasing along {axis_name} [cycles/voxel]")
        extent = float(np.max(np.abs(phases[:, axis]), initial=0.0))
        counts = [int(grid[1][axis]) for grid in grids]
        upper = min(max(1.1 * extent, 1.2 * max(counts), 2.0), 200.0)
        phi = np.linspace(0.0, upper, 4001)
        curve_phases = np.zeros((phi.size, 3))
        curve_phases[:, axis] = phi
        continuous = None
        for label, grid_counts, method in grids:
            axis_counts = [1, 1, 1]
            axis_counts[axis] = int(grid_counts[axis])
            continuous, sampled, _ = pathway_grid_coherence(
                curve_phases, axis_counts, method
            )
            color = CURRENT_COLOR if label == "current" else RECOMMENDED_COLOR
            suffix = " (RMS)" if method == "stratified" else ""
            plot.plot(
                phi,
                np.abs(sampled),
                pen=pg.mkPen(color, width=2),
                name=f"{label}: {axis_counts[axis]} {method} points{suffix}",
            )
        plot.plot(
            phi,
            np.abs(continuous),
            pen=pg.mkPen(VOXEL_COLOR, width=1.5),
            name="uniform voxel |sinc|",
        )
        on_axis = np.abs(phases[:, axis]) > 1e-9
        if np.any(on_axis):
            pathway_phases = np.zeros((int(np.count_nonzero(on_axis)), 3))
            pathway_phases[:, axis] = np.abs(phases[on_axis, axis])
            current_counts = [1, 1, 1]
            current_counts[axis] = int(grids[0][1][axis])
            _, on_curve, _ = pathway_grid_coherence(
                pathway_phases, current_counts, grids[0][2]
            )
            plot.addItem(
                pg.ScatterPlotItem(
                    pathway_phases[:, axis],
                    np.abs(on_curve),
                    size=7,
                    pen=None,
                    brush=self._weight_brushes(weights[on_axis]),
                    name="pathways on the current grid",
                )
            )
        plot.setYRange(0.0, 1.05, padding=0.0)

    def _render_errors(self, snapshot, phases, weights, observation, grids):
        plot = self.error_plot
        plot.clear()
        plot.setLabel("bottom", snapshot.observation_label)
        for label, counts, method in grids:
            _, _, errors = pathway_grid_coherence(phases, counts, method)
            values = np.maximum(100.0 * weights * errors, ERROR_FLOOR_PERCENT)
            color = CURRENT_COLOR if label == "current" else RECOMMENDED_COLOR
            suffix = " (RMS)" if method == "stratified" else ""
            # A PlotDataItem follows the log y axis; a bare scatter would not.
            plot.plot(
                observation,
                values,
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolPen=None,
                symbolBrush=pg.mkBrush(color),
                name=f"{label}: {_grid_text(counts, method)}{suffix}",
            )
        target = pg.mkPen(TARGET_COLOR, width=1.5, style=Qt.DashLine)
        # The y axis is logarithmic, so the line position is log10(percent).
        plot.addItem(
            pg.InfiniteLine(
                np.log10(100.0 * snapshot.target_error), angle=0, pen=target
            )
        )
        plot.plot([], [], pen=target, name=f"{100 * snapshot.target_error:.3g}% target")

    def _render_positions(self, sampling, axis):
        """Plot the grid in the plane of its two most finely sampled axes."""
        plot = self.positions_plot
        plot.clear()
        counts = sampling.counts_xyz
        # Show the two most finely sampled axes, the selected one on ties,
        # so a singleton axis does not collapse the plot onto a line.
        ranked = sorted(range(3), key=lambda index: (-counts[index], index != axis))
        first, second = sorted(ranked[:2])
        plot.setLabel("bottom", f"{'XYZ'[first]} offset [voxel]")
        plot.setLabel("left", f"{'XYZ'[second]} offset [voxel]")
        cell_pen = pg.mkPen("#cbd5e1", width=1)
        for plot_axis, data_axis in ((90, first), (0, second)):
            count = int(counts[data_axis])
            if count + 1 > MAX_GUIDE_LINES:
                continue
            for edge in np.linspace(-0.5, 0.5, count + 1):
                plot.addItem(pg.InfiniteLine(edge, angle=plot_axis, pen=cell_pen))
        voxels = [0, 1] if sampling.varies_per_voxel else [0]
        offsets = sampling.normalized_voxel_offsets(np.asarray(voxels))
        for voxel, color in zip(voxels, (CURRENT_COLOR, SECOND_VOXEL_COLOR)):
            name = (
                f"voxel {voxel}" if sampling.varies_per_voxel else "every voxel"
            ) + f" ({sampling.spins_per_voxel} spins)"
            plot.addItem(
                pg.ScatterPlotItem(
                    offsets[voxel, :, first],
                    offsets[voxel, :, second],
                    size=7,
                    pen=pg.mkPen("w", width=0.5),
                    brush=pg.mkBrush(color),
                    name=name,
                )
            )
        plot.setXRange(-0.5, 0.5, padding=0.05)
        plot.setYRange(-0.5, 0.5, padding=0.05)
