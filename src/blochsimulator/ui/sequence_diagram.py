"""Stacked per-channel sequence diagram for the Free Mode sequence designer."""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

GAMMA_1H_HZ_PER_G = 4257.7
B1_UT_PER_G = 100.0
GRADIENT_MT_M_PER_G_CM = 10.0

RF_COLOR = (0, 200, 220)
PHASE_COLOR = (120, 220, 235)
GRADIENT_PEN_COLOR = (200, 200, 200)
POSITIVE_FILL = (220, 60, 60, 150)
NEGATIVE_FILL = (60, 110, 230, 150)
TIMING_COLOR = (230, 170, 40)
MUTED_COLOR = (130, 130, 130)
PLAYHEAD_COLOR = (255, 220, 0)


@dataclass
class PulseSegment:
    """One contiguous RF event found in a B1 waveform."""

    start_s: float
    end_s: float
    peak_g: float
    flip_deg: Optional[float]
    phase_deg: Optional[float]

    @property
    def center_s(self) -> float:
        return 0.5 * (self.start_s + self.end_s)


@dataclass
class TimingMarker:
    """Bracket on the timing row, e.g. TE or TR, in seconds."""

    label: str
    start_s: float
    end_s: float
    level: int = 0


@dataclass
class DiagramAnnotations:
    """Sequence timing information drawn above the waveform lanes."""

    brackets: List[TimingMarker] = field(default_factory=list)
    ticks: List[Tuple[str, float]] = field(default_factory=list)
    first_tr_window_s: Optional[Tuple[float, float]] = None


def find_pulse_segments(
    b1: np.ndarray, time_s: np.ndarray, max_gap_samples: int = 3
) -> List[PulseSegment]:
    """Split a B1 waveform (gauss) into RF events and estimate flip angles.

    Flip angle and phase come from the coherent RF area, which matches the
    nominal flip of amplitude-modulated pulses.  Pulses whose phase varies so
    much that the coherent area is far below the magnitude area (adiabatic or
    frequency-shifted pulses) keep only their peak amplitude.
    """
    b1 = np.asarray(b1, dtype=complex)
    time_s = np.asarray(time_s, dtype=float)
    if b1.size == 0 or b1.size != time_s.size:
        return []
    magnitude = np.abs(b1)
    peak = float(magnitude.max())
    if peak <= 0:
        return []
    active = np.flatnonzero(magnitude > peak * 1e-6)
    if active.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(active) > max_gap_samples + 1)
    starts = np.concatenate(([active[0]], active[breaks + 1]))
    ends = np.concatenate((active[breaks], [active[-1]]))
    if time_s.size > 1:
        dt = np.diff(time_s)
        dt = np.append(dt, dt[-1])
    else:
        dt = np.ones(1)
    segments = []
    for first, last in zip(starts, ends):
        window = slice(first, last + 1)
        coherent = np.sum(b1[window] * dt[window])
        magnitude_area = np.sum(magnitude[window] * dt[window])
        flip = phase = None
        if magnitude_area > 0 and abs(coherent) >= 0.9 * magnitude_area:
            flip = float(np.degrees(2 * np.pi * GAMMA_1H_HZ_PER_G * abs(coherent)))
            phase = float(np.degrees(np.angle(coherent)))
        segments.append(
            PulseSegment(
                start_s=float(time_s[first]),
                end_s=float(time_s[last] + dt[last]),
                peak_g=float(magnitude[window].max()),
                flip_deg=flip,
                phase_deg=phase,
            )
        )
    return segments


def pulse_label_html(segment: PulseSegment) -> str:
    """Return a compact label such as 90°<sub>x</sub> for an RF event."""
    if segment.flip_deg is None:
        return f"{segment.peak_g * B1_UT_PER_G:.3g} µT"
    flip = (
        f"{segment.flip_deg:.0f}°"
        if segment.flip_deg >= 9.5
        else (f"{segment.flip_deg:.2g}°")
    )
    phase = segment.phase_deg % 360.0
    for axis_phase, axis in ((0, "x"), (90, "y"), (180, "-x"), (270, "-y"), (360, "x")):
        if abs(phase - axis_phase) <= 2.0:
            return f"{flip}<sub>{axis}</sub>"
    return f"{flip}<sub>{segment.phase_deg:.0f}°</sub>"


def _peak_decimate(time_ms, values, max_points):
    """Keep min and max of each bin so short gradient blips stay visible."""
    n = values.size
    if n <= max_points:
        return time_ms, values
    bins = max_points // 2
    width = n // bins
    usable = bins * width
    shaped = values[:usable].reshape(bins, width)
    times = time_ms[:usable].reshape(bins, width)
    lo_idx = np.nanargmin(np.where(np.isnan(shaped), np.inf, shaped), axis=1)
    hi_idx = np.nanargmax(np.where(np.isnan(shaped), -np.inf, shaped), axis=1)
    first = np.minimum(lo_idx, hi_idx)
    second = np.maximum(lo_idx, hi_idx)
    rows = np.arange(bins)
    out_t = np.empty(2 * bins)
    out_v = np.empty(2 * bins)
    out_t[0::2] = times[rows, first]
    out_t[1::2] = times[rows, second]
    out_v[0::2] = shaped[rows, first]
    out_v[1::2] = shaped[rows, second]
    return out_t, out_v


class LinkedPlayhead:
    """InfiniteLine-like handle that moves the playhead in every lane."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._value = 0.0
        self._visible = False

    def setValue(self, value):
        self._value = float(value)
        for line in self._lines:
            line.setValue(self._value)

    def value(self):
        return self._value

    def show(self):
        self.setVisible(True)

    def hide(self):
        self.setVisible(False)

    def setVisible(self, visible):
        self._visible = bool(visible)
        for line in self._lines:
            line.setVisible(self._visible)

    def isVisible(self):
        return self._visible


class SequenceDiagramWidget(pg.GraphicsLayoutWidget):
    """RF, RF phase and gradient waveforms on separate lanes with a shared time axis."""

    time_clicked = pyqtSignal(float)

    MAX_PLOT_POINTS = 20000
    MAX_PULSE_LABELS = 40
    MIN_LABEL_SPACING_PX = 38
    TIME_PAD_FRACTION = 0.03
    GRADIENT_ZOOM_FRACTION = 0.1
    LABEL_WIDTH = 72
    MIN_BRACKET_PX = 12
    AXIS_WIDTH = 46
    COLLAPSED_HEIGHT = 22
    TIMING_HEIGHT = 44
    # (key, name, unit, tooltip)
    LANES = (
        ("rf", "RF |B1|", "µT", "RF amplitude in the rotating frame."),
        ("phase", "RF phase", "°", "RF phase, drawn only while RF is on."),
        ("gz", "Gz", "mT/m", "Gradient along z (slice-select axis)."),
        ("gy", "Gy", "mT/m", "Gradient along y."),
        ("gx", "Gx", "mT/m", "Gradient along x."),
    )
    GRADIENT_COLUMNS = {"gx": 0, "gy": 1, "gz": 2}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.ci.layout.setContentsMargins(2, 2, 6, 2)
        self.ci.layout.setVerticalSpacing(0)
        self.ci.layout.setHorizontalSpacing(0)
        self.total_ms = 0.0
        self.first_tr_window_ms = None
        self.pulse_segments: List[PulseSegment] = []
        self.pulse_labels = []
        self.collapsed = {}
        self.lane_peaks = {}
        self.gradient_zoomed = {}
        self._timing_texts = []

        self.timing_label = pg.LabelItem("")
        self.timing_label.setFixedWidth(self.LABEL_WIDTH)
        self.addItem(self.timing_label, row=0, col=0)
        self.timing_plot = self._make_plot(row=0)
        self.timing_plot.getAxis("left").setStyle(showValues=False)
        self.timing_plot.getAxis("left").setPen(pg.mkPen(None))
        self.timing_plot.setYRange(0, 2.4, padding=0)
        self.timing_plot.setFixedHeight(self.TIMING_HEIGHT)

        self.lane_labels = {}
        self.plots = {}
        for row, (key, name, unit, tooltip) in enumerate(self.LANES, start=1):
            label = pg.LabelItem(self._label_html(name, unit), justify="right")
            label.setFixedWidth(self.LABEL_WIDTH)
            label.setToolTip(tooltip)
            self.addItem(label, row=row, col=0)
            self.lane_labels[key] = label
            plot = self._make_plot(row=row)
            plot.setToolTip(tooltip)
            plot.addItem(
                pg.InfiniteLine(
                    pos=0, angle=0, pen=pg.mkPen((110, 110, 110, 160), width=1)
                ),
                ignoreBounds=True,
            )
            self.plots[key] = plot
        last = self.plots[self.LANES[-1][0]]
        last.showAxis("bottom")
        last.setLabel("bottom", "Time", "ms")
        for plot in [self.timing_plot, *self.plots.values()]:
            if plot is not last:
                plot.setXLink(last)
        self.time_axis_plot = last

        lines = []
        for plot in [self.timing_plot, *self.plots.values()]:
            line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen(PLAYHEAD_COLOR, width=2)
            )
            line.setZValue(100)
            plot.addItem(line, ignoreBounds=True)
            lines.append(line)
        self.playhead = LinkedPlayhead(lines)
        self.playhead.hide()

        for _ in range(self.MAX_PULSE_LABELS):
            label = pg.TextItem(anchor=(0.5, 1.0))
            label.setZValue(50)
            label.hide()
            self.plots["rf"].addItem(label, ignoreBounds=True)
            self.pulse_labels.append(label)
        self.time_axis_plot.sigXRangeChanged.connect(self._update_pulse_labels)
        self.time_axis_plot.getViewBox().sigResized.connect(self._update_pulse_labels)

        self.error_item = pg.TextItem("", color=(255, 90, 90), anchor=(0.5, 0.5))
        self.error_item.setZValue(200)
        self.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._dynamic_items = {key: [] for key in ["timing", *self.plots]}

    # ------------------------------------------------------------------ layout
    @staticmethod
    def _label_html(name, unit, muted=False):
        if muted:
            return f"<span style='color:#777; font-size:8pt'>{name} · off</span>"
        return (
            f"<span style='color:#ddd; font-size:9pt'>{name}</span>"
            f"<br><span style='color:#999; font-size:8pt'>{unit}</span>"
        )

    def _make_plot(self, row):
        plot = self.addPlot(row=row, col=1)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=True, y=False)
        plot.hideAxis("bottom")
        plot.getAxis("left").setWidth(self.AXIS_WIDTH)
        tick_font = QFont()
        tick_font.setPointSize(8)
        plot.getAxis("left").setStyle(tickLength=-4, tickFont=tick_font)
        plot.getAxis("bottom").setStyle(tickFont=tick_font)
        plot.showGrid(x=True, y=False, alpha=0.15)
        plot.getViewBox().setDefaultPadding(0)
        return plot

    def _clear_dynamic(self):
        for key, items in self._dynamic_items.items():
            plot = self.timing_plot if key == "timing" else self.plots[key]
            for item in items:
                if isinstance(item, pg.PlotDataItem):
                    # pyqtgraph looks up the view range of clipped curves while
                    # they are detached, which fails outside a ViewBox.
                    item.setClipToView(False)
                    item.setDownsampling(auto=False)
                plot.removeItem(item)
            items.clear()
        for label in self.pulse_labels:
            label.hide()
        self._timing_texts = []
        if self.error_item.scene() is not None:
            self.plots["rf"].removeItem(self.error_item)

    def _add(self, key, item, ignore_bounds=True, fast=False):
        plot = self.timing_plot if key == "timing" else self.plots[key]
        plot.addItem(item, ignoreBounds=ignore_bounds)
        if fast:
            # Enable only once inside the ViewBox: pyqtgraph queries the view
            # range while the item is being added to the scene.
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
        self._dynamic_items[key].append(item)
        return item

    def _set_collapsed(self, key, collapsed):
        name, unit = next((n, u) for k, n, u, _ in self.LANES if k == key)
        plot = self.plots[key]
        self.collapsed[key] = collapsed
        self.lane_labels[key].setText(self._label_html(name, unit, muted=collapsed))
        axis = plot.getAxis("left")
        if collapsed:
            plot.setFixedHeight(self.COLLAPSED_HEIGHT)
            axis.setTicks([[(0.0, "0")], []])
            plot.setYRange(-1, 1, padding=0)
        else:
            plot.setMinimumHeight(40)
            plot.setMaximumHeight(16777215)
            plot.setPreferredHeight(80)
        self.lane_labels[key].setMaximumHeight(
            self.COLLAPSED_HEIGHT if collapsed else 16777215
        )
        is_last = key == self.LANES[-1][0]
        if is_last and collapsed:
            # The time axis lives on this lane, so keep room for it.
            plot.setFixedHeight(self.COLLAPSED_HEIGHT + 34)
            self.lane_labels[key].setMaximumHeight(self.COLLAPSED_HEIGHT + 34)

    def _apply_stretch(self):
        layout = self.ci.layout
        layout.setRowStretchFactor(0, 0)
        stretch = {"rf": 4, "phase": 1, "gz": 3, "gy": 3, "gx": 3}
        for row, (key, *_rest) in enumerate(self.LANES, start=1):
            layout.setRowStretchFactor(
                row, 0 if self.collapsed.get(key) else stretch[key]
            )

    # ----------------------------------------------------------------- drawing
    def show_error(self, message: str):
        """Replace the waveforms by an error message."""
        self._clear_dynamic()
        self.total_ms = 0.0
        self.first_tr_window_ms = None
        self.pulse_segments = []
        self.playhead.hide()
        for key, *_rest in self.LANES:
            self._set_collapsed(key, key != "rf")
        self._set_collapsed("rf", False)
        self._apply_stretch()
        rf = self.plots["rf"]
        rf.getAxis("left").setTicks([[], []])
        self.error_item.setText(f"Invalid sequence:\n{message}")
        rf.addItem(self.error_item, ignoreBounds=True)
        self.error_item.setPos(0.5, 0.5)
        self.time_axis_plot.setLimits(xMin=None, xMax=None)
        self.time_axis_plot.setXRange(0, 1, padding=0)
        rf.setYRange(0, 1, padding=0)

    def error_text(self) -> str:
        return self.error_item.toPlainText() if self.error_item.scene() else ""

    def set_sequence(
        self,
        b1: np.ndarray,
        gradients: np.ndarray,
        time_s: np.ndarray,
        annotations: Optional[DiagramAnnotations] = None,
    ):
        """Draw a compiled sequence; B1 in gauss, gradients in G/cm, time in s."""
        self._clear_dynamic()
        b1 = np.asarray(b1, dtype=complex).ravel()
        time_s = np.asarray(time_s, dtype=float).ravel()
        gradients = np.asarray(gradients, dtype=float)
        if gradients.ndim == 1:
            gradients = gradients[:, None]
        if time_s.size == 0:
            self.total_ms = 0.0
            return
        t0 = time_s[0]
        rel_s = time_s - t0
        time_ms = rel_s * 1e3
        dt_ms = float(np.median(np.diff(time_ms))) if time_ms.size > 1 else 0.01
        self.total_ms = float(time_ms[-1] + dt_ms)
        self.pulse_segments = find_pulse_segments(b1, rel_s)

        self._draw_rf(b1, time_ms)
        self._draw_phase(b1, time_ms)
        lanes = {
            key: (
                gradients[:, column] * GRADIENT_MT_M_PER_G_CM
                if column < gradients.shape[1]
                else np.zeros(time_ms.size)
            )
            for key, column in self.GRADIENT_COLUMNS.items()
        }
        shared_peak = max(
            float(np.max(np.abs(g))) if g.size else 0.0 for g in lanes.values()
        )
        for key, g in lanes.items():
            self._draw_gradient(key, g, time_ms, shared_peak)
        self._apply_stretch()
        self._draw_timing(annotations or DiagramAnnotations())

        window = annotations.first_tr_window_s if annotations else None
        self.first_tr_window_ms = (
            (window[0] * 1e3, min(window[1] * 1e3, self.total_ms)) if window else None
        )
        min_range = max(5 * dt_ms, 1e-3)
        pad = self.TIME_PAD_FRACTION * self.total_ms
        self.time_axis_plot.setLimits(
            xMin=-pad,
            xMax=self.total_ms + pad,
            minXRange=min(min_range, self.total_ms),
        )
        self.show_full_sequence()
        self._update_pulse_labels()
        self.playhead.setValue(0.0)
        self.playhead.show()

    def _draw_rf(self, b1, time_ms):
        plot = self.plots["rf"]
        magnitude = np.abs(b1) * B1_UT_PER_G
        peak = float(magnitude.max()) if magnitude.size else 0.0
        self.lane_peaks["rf"] = peak
        if peak <= 0:
            self._set_collapsed("rf", True)
            return
        self._set_collapsed("rf", False)
        t, y = _peak_decimate(time_ms, magnitude, self.MAX_PLOT_POINTS)
        self._add(
            "rf",
            pg.PlotDataItem(
                t,
                y,
                pen=pg.mkPen(RF_COLOR, width=1.5),
                fillLevel=0.0,
                brush=pg.mkBrush(*RF_COLOR, 90),
                name="RF |B1|",
            ),
            fast=True,
        )
        plot.getAxis("left").setTicks([[(0.0, "0"), (peak, f"{peak:.3g}")], []])
        plot.setYRange(-0.04 * peak, 1.5 * peak, padding=0)

    @staticmethod
    def _pulse_html(segment):
        return (
            "<span style='color:#e8f8fb; font-size:8pt'>"
            f"{pulse_label_html(segment)}</span>"
        )

    def _draw_phase(self, b1, time_ms):
        plot = self.plots["phase"]
        magnitude = np.abs(b1)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        active = magnitude > 0.01 * peak if peak > 0 else np.zeros(b1.size, bool)
        if not np.any(active):
            self._set_collapsed("phase", True)
            return
        self._set_collapsed("phase", False)
        phase = np.degrees(np.angle(b1))
        phase[~active] = np.nan
        # Break the line where the phase wraps so no vertical jumps are drawn.
        wraps = np.flatnonzero(np.abs(np.diff(phase)) > 180.0)
        phase[wraps + 1] = np.nan
        if phase.size > self.MAX_PLOT_POINTS:
            stride = int(np.ceil(phase.size / self.MAX_PLOT_POINTS))
            t, phase = time_ms[::stride], phase[::stride]
        else:
            t = time_ms
        self._add(
            "phase",
            pg.PlotDataItem(
                t,
                phase,
                pen=pg.mkPen(PHASE_COLOR, width=1.5),
                connect="finite",
                name="RF phase",
            ),
        )
        plot.getAxis("left").setTicks(
            [[(-180.0, "-180"), (0.0, "0"), (180.0, "180")], []]
        )
        plot.setYRange(-215, 215, padding=0)

    def _draw_gradient(self, key, g, time_ms, shared_peak):
        plot = self.plots[key]
        peak = float(np.max(np.abs(g))) if g.size else 0.0
        self.lane_peaks[key] = peak
        name = next(n for k, n, *_ in self.LANES if k == key)
        if peak <= 0:
            self._set_collapsed(key, True)
            self._add(
                key,
                pg.PlotDataItem(time_ms[[0, -1]], [0.0, 0.0], pen=None, name=name),
            )
            return
        self._set_collapsed(key, False)
        t, y = _peak_decimate(time_ms, g, self.MAX_PLOT_POINTS)
        for part, brush in (
            (np.maximum(y, 0.0), POSITIVE_FILL),
            (np.minimum(y, 0.0), NEGATIVE_FILL),
        ):
            if np.any(part):
                self._add(
                    key,
                    pg.PlotDataItem(
                        t,
                        part,
                        pen=None,
                        fillLevel=0.0,
                        brush=pg.mkBrush(*brush),
                    ),
                    fast=True,
                )
        self._add(
            key,
            pg.PlotDataItem(
                t,
                y,
                pen=pg.mkPen(GRADIENT_PEN_COLOR, width=1.2),
                name=name,
            ),
            fast=True,
        )
        plot.getAxis("left").setTicks(
            [[(-peak, f"{-peak:.3g}"), (0.0, "0"), (peak, f"{peak:.3g}")], []]
        )
        # All gradient lanes share one scale so their amplitudes compare
        # directly; a lane far below the others gets its own, marked scale.
        zoomed = peak < self.GRADIENT_ZOOM_FRACTION * shared_peak
        span = peak if zoomed else shared_peak
        self.gradient_zoomed[key] = zoomed
        unit = "mT/m<br>own scale" if zoomed else "mT/m"
        self.lane_labels[key].setText(self._label_html(name, unit))
        plot.setYRange(-1.18 * span, 1.18 * span, padding=0)

    @staticmethod
    def _timing_text(text, anchor):
        item = pg.TextItem(anchor=anchor)
        item.setHtml(
            f"<span style='color:rgb{TIMING_COLOR}; font-size:8pt'>{text}</span>"
        )
        return item

    def _draw_timing(self, annotations: DiagramAnnotations):
        pen = pg.mkPen(TIMING_COLOR, width=1.2)
        for marker in annotations.brackets:
            x0, x1 = marker.start_s * 1e3, marker.end_s * 1e3
            if x1 <= x0 or x0 > self.total_ms:
                continue
            y = 0.35 + 1.15 * marker.level
            bracket = pg.PlotDataItem(
                [x0, x0, x0, x1, x1, x1],
                [y - 0.2, y + 0.2, y, y, y + 0.2, y - 0.2],
                pen=pen,
                connect=np.array([1, 0, 1, 0, 1, 0], dtype=bool),
            )
            self._add("timing", bracket)
            text = self._timing_text(marker.label, anchor=(0.5, 1.0))
            text.setPos(0.5 * (x0 + x1), y + 0.08)
            self._add("timing", text)
            self._timing_texts.append((text, bracket, x0, x1))
        dotted = pg.mkPen(TIMING_COLOR, width=1, style=Qt.DotLine)
        for label, t_s in annotations.ticks:
            x = t_s * 1e3
            if x > self.total_ms:
                continue
            self._add("timing", pg.PlotDataItem([x, x], [0.0, 0.9], pen=dotted))
            text = self._timing_text(label, anchor=(0.0, 1.0))
            text.setPos(x, 1.0)
            self._add("timing", text)

    def _update_pulse_labels(self, *_args):
        """Label the visible pulses when there is room for every label."""
        segments = []
        if self.total_ms > 0 and not self.collapsed.get("rf", True):
            lo, hi = self.visible_time_range_ms()
            segments = [
                seg
                for seg in self.pulse_segments
                if seg.end_s * 1e3 >= lo and seg.start_s * 1e3 <= hi
            ]
            width_px = max(self.time_axis_plot.getViewBox().width(), 1.0)
            px_per_ms = width_px / max(hi - lo, 1e-12)
            centers = np.array([seg.center_s * 1e3 for seg in segments])
            too_dense = centers.size > 1 and (
                np.min(np.diff(centers)) * px_per_ms < self.MIN_LABEL_SPACING_PX
            )
            if too_dense or len(segments) > len(self.pulse_labels):
                segments = []
        for label, segment in zip(self.pulse_labels, segments):
            label.setHtml(self._pulse_html(segment))
            # Keep labels of pulses at the view edges fully inside the lane.
            half = (0.5 * label.boundingRect().width() + 2) / px_per_ms
            x = segment.center_s * 1e3
            x = min(max(x, lo + half), hi - half) if hi - lo > 2 * half else x
            label.setPos(x, segment.peak_g * B1_UT_PER_G * 1.03)
            label.show()
        for label in self.pulse_labels[len(segments) :]:
            label.hide()
        if self.total_ms > 0:
            lo, hi = self.visible_time_range_ms()
            width_px = max(self.time_axis_plot.getViewBox().width(), 1.0)
            px_per_ms = width_px / max(hi - lo, 1e-12)
            for text, bracket, x0, x1 in self._timing_texts:
                width = (x1 - x0) * px_per_ms
                text.setVisible(width >= text.boundingRect().width() + 4)
                # A bracket squeezed to a few pixels reads as noise.
                bracket.setVisible(width >= self.MIN_BRACKET_PX)

    def visible_pulse_labels(self) -> List[str]:
        return [label.toPlainText() for label in self.pulse_labels if label.isVisible()]

    # -------------------------------------------------------------- navigation
    def show_full_sequence(self):
        if self.total_ms > 0:
            pad = self.TIME_PAD_FRACTION * self.total_ms
            self.time_axis_plot.setXRange(-pad, self.total_ms + pad, padding=0)

    def show_first_tr(self):
        if self.first_tr_window_ms is not None:
            lo, hi = self.first_tr_window_ms
            pad = self.TIME_PAD_FRACTION * (hi - lo)
            self.time_axis_plot.setXRange(lo - pad, hi + pad, padding=0)

    def visible_time_range_ms(self) -> Tuple[float, float]:
        return tuple(self.time_axis_plot.getViewBox().viewRange()[0])

    def _on_scene_clicked(self, event):
        if event.button() != Qt.LeftButton or self.total_ms <= 0:
            return
        if event.double():
            self.show_full_sequence()
            return
        pos = event.scenePos()
        for plot in [self.timing_plot, *self.plots.values()]:
            view = plot.getViewBox()
            if view.sceneBoundingRect().contains(pos):
                time_ms = float(np.clip(view.mapSceneToView(pos).x(), 0, self.total_ms))
                self.playhead.setValue(time_ms)
                self.playhead.show()
                self.time_clicked.emit(time_ms / 1e3)
                return

    def lane_data(self, key) -> Sequence[np.ndarray]:
        """Return (time_ms, values) of the outline curve drawn in a lane."""
        plot = self.plots[key]
        for item in reversed(plot.listDataItems()):
            if item.name():
                return item.xData, item.yData
        return np.array([]), np.array([])
