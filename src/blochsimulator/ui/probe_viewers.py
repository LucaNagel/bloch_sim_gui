"""Free-mode style viewers for sequence spin-probe results."""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .widgets import CheckableComboBox

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"
except Exception:
    gl = None
    HAS_OPENGL = False


def _clear_plot(plot_widget: pg.PlotWidget) -> None:
    plot_widget.clear()


def _safe_range(values, *, fallback=(0.0, 1.0), pad_fraction=0.05):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _set_colorbar_levels(colorbar, data):
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if np.isfinite(lo) and np.isfinite(hi) and not np.isclose(lo, hi):
        colorbar.setLevels((lo, hi))


def _position_axis_mm(positions_m: np.ndarray):
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        return np.zeros(1), 0
    spans = np.ptp(positions, axis=0)
    axis = int(np.argmax(spans))
    values = positions[:, axis] * 1000.0
    if np.allclose(values, values[0]):
        values = np.arange(positions.shape[0], dtype=float)
    return values, axis


class SequenceProbeSpectrumViewer(QWidget):
    """Spectrum viewer mirroring the Free Mode spectrum controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.time_index = 0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Frequency Spectrum"))
        header.addStretch()
        self.export_button = QPushButton("Export Results")
        self.export_button.setVisible(False)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        controls = QHBoxLayout()
        self.spectrum_3d_toggle = QCheckBox("3D View")
        self.spectrum_3d_toggle.setEnabled(HAS_OPENGL)
        self.spectrum_3d_toggle.toggled.connect(self.refresh)
        controls.addWidget(self.spectrum_3d_toggle)
        controls.addWidget(QLabel("Plot type:"))
        self.plot_type = QComboBox()
        self.plot_type.setObjectName("sequence_probe_spectrum_plot_type")
        self.plot_type.addItems(["Line", "Heatmap"])
        self.plot_type.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.plot_type)
        controls.addWidget(QLabel("Spectrum view:"))
        self.view_mode = QComboBox()
        self.view_mode.setObjectName("sequence_probe_spectrum_view_mode")
        self.view_mode.addItems(["Individual position", "Coherent mean over positions"])
        self.view_mode.setToolTip(
            "The coherent mean averages complex Mxy and therefore reveals "
            "gradient-induced dephasing."
        )
        self.view_mode.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.view_mode)
        self.position_label = QLabel("Pos: 0.000 mm")
        controls.addWidget(self.position_label)
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setObjectName("sequence_probe_spectrum_position_slider")
        self.position_slider.setRange(0, 0)
        self.position_slider.valueChanged.connect(self.refresh)
        controls.addWidget(self.position_slider, 1)
        layout.addLayout(controls)

        component_row = QHBoxLayout()
        self.component_label = QLabel("Component:")
        component_row.addWidget(self.component_label)
        self.component_combo = CheckableComboBox()
        self.component_combo.add_items(
            ["Magnitude", "Phase", "Phase (unwrapped)", "Real", "Imaginary", "Mz"]
        )
        self.component_combo.set_selected_items(["Magnitude"])
        self.component_combo.selection_changed.connect(self.refresh)
        component_row.addWidget(self.component_combo, 1)
        self.heatmap_mode_label = QLabel("Heatmap mode:")
        self.heatmap_mode = QComboBox()
        self.heatmap_mode.setObjectName("sequence_probe_spectrum_heatmap_mode")
        self.heatmap_mode.addItems(
            ["Spin vs Time (Evolution)", "Spin vs Frequency (FFT)"]
        )
        self.heatmap_mode.currentTextChanged.connect(self.refresh)
        self.heatmap_mode_label.setVisible(False)
        self.heatmap_mode.setVisible(False)
        component_row.addWidget(self.heatmap_mode_label)
        component_row.addWidget(self.heatmap_mode)
        layout.addLayout(component_row)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Magnitude")
        self.plot.setLabel("bottom", "Spin offset", "Hz")
        self.plot.setDownsampling(mode="peak")
        self.plot.setClipToView(True)
        layout.addWidget(self.plot, 1)

        self.heatmap_layout = pg.GraphicsLayoutWidget()
        self.heatmap_plot = self.heatmap_layout.addPlot(row=0, col=0)
        self.heatmap_plot.setLabel("left", "Spin Index")
        self.heatmap_plot.setLabel("bottom", "Time", "ms")
        self.heatmap_item = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_item)
        self.heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.heatmap_layout.addItem(self.heatmap_colorbar, row=0, col=1)
        self.heatmap_colorbar.setImageItem(self.heatmap_item)
        self.heatmap_layout.setVisible(False)
        layout.addWidget(self.heatmap_layout, 1)

        self.plot_3d = None
        if HAS_OPENGL:
            self.plot_3d = gl.GLViewWidget()
            self.plot_3d.opts["distance"] = 40
            self.plot_3d.setVisible(False)
            layout.addWidget(self.plot_3d, 1)

    def set_result(self, result):
        self.result = result
        self.time_index = max(0, result.time_s.size - 1)
        npos = int(result.positions_m.shape[0])
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, max(0, npos - 1))
        self.position_slider.setValue(0)
        self.position_slider.blockSignals(False)
        self.refresh()

    def set_time_index(self, index: int):
        if self.result is None or self.result.time_s.size == 0:
            return
        self.time_index = int(np.clip(index, 0, self.result.time_s.size - 1))
        self.refresh()

    def _frequency_axis_hz(self):
        if self.result is None:
            return np.zeros(1)
        return np.asarray(self.result.frequency_offsets_hz, dtype=float)

    def _position_label_text(self, pos_index):
        if self.result is None or self.result.positions_m.shape[0] == 0:
            return "Pos: 0.000 mm"
        axis_values, axis = _position_axis_mm(self.result.positions_m)
        if pos_index < axis_values.size:
            axis_name = "xyz"[axis]
            return f"Pos {axis_name}: {axis_values[pos_index]:.3f} mm"
        return f"Pos idx: {pos_index}"

    def _component(self, signal, mz, name):
        if name == "Magnitude":
            return np.abs(signal)
        if name == "Phase":
            return np.angle(signal) / np.pi
        if name == "Phase (unwrapped)":
            return np.unwrap(np.angle(signal)) / np.pi
        if name == "Real":
            return np.real(signal)
        if name == "Imaginary":
            return np.imag(signal)
        if name == "Mz":
            return mz
        return np.abs(signal)

    def refresh(self, *_):
        if self.result is None:
            return
        if self.plot_3d is not None and self.spectrum_3d_toggle.isChecked():
            self.plot.setVisible(False)
            self.heatmap_layout.setVisible(False)
            self.plot_3d.setVisible(True)
            self._render_3d()
            return
        if self.plot_3d is not None:
            self.plot_3d.setVisible(False)

        is_heatmap = self.plot_type.currentText() == "Heatmap"
        self.plot.setVisible(not is_heatmap)
        self.heatmap_layout.setVisible(is_heatmap)
        self.component_label.setVisible(not is_heatmap)
        self.component_combo.setVisible(not is_heatmap)
        self.heatmap_mode_label.setVisible(is_heatmap)
        self.heatmap_mode.setVisible(is_heatmap)
        if is_heatmap:
            self._render_heatmap()
        else:
            self._render_line()

    def _render_line(self):
        result = self.result
        _clear_plot(self.plot)
        freq = self._frequency_axis_hz()
        pos_count = result.positions_m.shape[0]
        pos_index = min(self.position_slider.value(), max(0, pos_count - 1))
        individual = self.view_mode.currentText() == "Individual position"
        self.position_slider.setEnabled(individual and pos_count > 1)
        self.position_label.setText(self._position_label_text(pos_index))

        snapshot_signal = result.mxy[self.time_index]
        snapshot_mz = result.mz[self.time_index]
        if individual and pos_count > 0:
            signal = snapshot_signal[pos_index]
            mz = snapshot_mz[pos_index]
        else:
            signal = np.mean(snapshot_signal, axis=0)
            mz = np.mean(snapshot_mz, axis=0)

        selected = self.component_combo.get_selected_items() or ["Magnitude"]
        visible = []
        colors = {
            "Magnitude": "c",
            "Phase": "y",
            "Phase (unwrapped)": "y",
            "Real": "r",
            "Imaginary": "g",
            "Mz": "m",
        }
        for component in selected:
            values = self._component(signal, mz, component)
            visible.append(values)
            pen = pg.mkPen(colors.get(component, "w"), width=2)
            if component == "Real":
                pen.setStyle(Qt.DashLine)
            elif component == "Imaginary":
                pen.setStyle(Qt.DotLine)
            self.plot.plot(freq, values, pen=pen, name=component)

        self.plot.setLabel("bottom", "Spin offset", "Hz")
        ylabel = selected[0] if len(selected) == 1 else "Signal"
        if selected == ["Phase"]:
            ylabel = "Phase (units of pi)"
        self.plot.setLabel("left", ylabel)
        self.plot.setXRange(*_safe_range(freq, fallback=(-1.0, 1.0)), padding=0)
        if visible:
            self.plot.setYRange(
                *_safe_range(np.concatenate([np.ravel(v) for v in visible])), padding=0
            )

    def _render_heatmap(self):
        result = self.result
        signal = result.mxy
        time_ms = result.time_s * 1000.0
        if signal.shape[0] == 0:
            return
        mode = self.heatmap_mode.currentText()
        if mode == "Spin vs Time (Evolution)" or signal.shape[0] < 2:
            data = np.abs(signal).reshape(signal.shape[0], -1).T
            x_axis = time_ms
            x_label = ("Time", "ms")
            title = "Temporal Evolution (|Mxy| over time)"
        else:
            signal_slice = signal[: self.time_index + 1]
            dt = float(np.mean(np.diff(result.time_s[: self.time_index + 1])))
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
            n_fft = int(2 ** np.ceil(np.log2(max(2, signal_slice.shape[0]))))
            flat = signal_slice.reshape(signal_slice.shape[0], -1)
            spectrum = np.fft.fftshift(np.fft.fft(flat, n=n_fft, axis=0), axes=0)
            data = np.abs(spectrum).T
            x_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))
            x_label = ("Frequency from signal FFT", "Hz")
            title = "Spectra Stack (FFT of signal per spin)"

        x_min, x_max = _safe_range(x_axis, fallback=(0.0, 1.0), pad_fraction=0.0)
        x_span = max(float(x_max - x_min), 1e-9)
        self.heatmap_item.setImage(data, autoLevels=True, axisOrder="row-major")
        self.heatmap_item.setRect(float(x_min), 0.0, x_span, float(data.shape[0]))
        self.heatmap_plot.setLabel("bottom", *x_label)
        self.heatmap_plot.setLabel("left", "Spin Index")
        self.heatmap_plot.setTitle(title)
        self.heatmap_plot.setXRange(float(x_min), float(x_max), padding=0)
        self.heatmap_plot.setYRange(0, data.shape[0], padding=0)
        _set_colorbar_levels(self.heatmap_colorbar, data)

    def _render_3d(self):
        if self.plot_3d is None or self.result is None:
            return
        self.plot_3d.clear()
        freq = self._frequency_axis_hz()
        if freq.size == 0:
            return
        pos_count = self.result.positions_m.shape[0]
        pos_index = min(self.position_slider.value(), max(0, pos_count - 1))
        snapshot = self.result.mxy[self.time_index]
        data = snapshot[pos_index] if pos_count else snapshot.reshape(-1)
        if data.size != freq.size:
            data = np.ravel(data)[: freq.size]
        freq_min, freq_max = float(np.nanmin(freq)), float(np.nanmax(freq))
        span = freq_max - freq_min if not np.isclose(freq_min, freq_max) else 1.0
        freq_norm = (freq - freq_min) / span * 20.0 - 10.0
        pts = np.vstack([freq_norm, np.real(data) * 5.0, np.imag(data) * 5.0]).T
        line = gl.GLLinePlotItem(pos=pts, color=(0, 1, 1, 1), width=2, antialias=True)
        self.plot_3d.addItem(line)


class SequenceProbeSpatialViewer(QWidget):
    """Spatial viewer mirroring the Free Mode spatial controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.time_index = 0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Spatial Profile"))
        header.addStretch()
        self.export_button = QPushButton("Export Results")
        self.export_button.setVisible(False)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        self.mean_only = QCheckBox("Mean only (Mag/Signal/3D)")
        self.mean_only.setObjectName("sequence_probe_spatial_mean_only")
        self.mean_only.toggled.connect(self.refresh)
        layout.addWidget(self.mean_only)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plot type:"))
        self.plot_type = QComboBox()
        self.plot_type.setObjectName("sequence_probe_spatial_plot_type")
        self.plot_type.addItems(["Line", "Heatmap"])
        self.plot_type.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.plot_type)
        controls.addWidget(QLabel("Heatmap mode:"))
        self.heatmap_mode = QComboBox()
        self.heatmap_mode.setObjectName("sequence_probe_spatial_heatmap_mode")
        self.heatmap_mode.addItems(["Position vs Frequency", "Position vs Time"])
        self.heatmap_mode.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.heatmap_mode)
        controls.addWidget(QLabel("View:"))
        self.view_mode = QComboBox()
        self.view_mode.setObjectName("sequence_probe_spatial_view_mode")
        self.view_mode.addItems(["Individual freq", "Mean over freqs"])
        self.view_mode.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.view_mode)
        self.freq_label = QLabel("Freq: 0.0 Hz")
        controls.addWidget(self.freq_label)
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setObjectName("sequence_probe_spatial_freq_slider")
        self.freq_slider.setRange(0, 0)
        self.freq_slider.valueChanged.connect(self.refresh)
        controls.addWidget(self.freq_slider, 1)
        layout.addLayout(controls)

        self.markers = QCheckBox("Show colored position/frequency markers")
        self.markers.setObjectName("sequence_probe_spatial_markers")
        self.markers.toggled.connect(self.refresh)
        layout.addWidget(self.markers)

        component_row = QHBoxLayout()
        component_row.addWidget(QLabel("Component:"))
        self.component_combo = CheckableComboBox()
        self.component_combo.add_items(
            ["Magnitude", "Phase", "Phase (unwrapped)", "Real", "Imaginary"]
        )
        self.component_combo.set_selected_items(["Magnitude", "Real", "Imaginary"])
        self.component_combo.selection_changed.connect(self.refresh)
        component_row.addWidget(self.component_combo, 1)
        layout.addLayout(component_row)

        self.line_container = QWidget()
        line_layout = QHBoxLayout(self.line_container)
        line_layout.setContentsMargins(0, 0, 0, 0)
        self.mxy_plot = pg.PlotWidget()
        self.mxy_plot.setLabel("left", "Mxy (transverse)")
        self.mxy_plot.setLabel("bottom", "Position", "mm")
        self.mxy_plot.setDownsampling(mode="peak")
        self.mxy_plot.setClipToView(True)
        self.mz_plot = pg.PlotWidget()
        self.mz_plot.setLabel("left", "Mz (longitudinal)")
        self.mz_plot.setLabel("bottom", "Position", "mm")
        self.mz_plot.setDownsampling(mode="peak")
        self.mz_plot.setClipToView(True)
        line_layout.addWidget(self.mxy_plot)
        line_layout.addWidget(self.mz_plot)
        layout.addWidget(self.line_container, 1)

        self.heatmap_container = QWidget()
        heatmap_layout = QVBoxLayout(self.heatmap_container)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)
        self.mxy_heatmap_layout = pg.GraphicsLayoutWidget()
        self.mxy_heatmap = self.mxy_heatmap_layout.addPlot(row=0, col=0)
        self.mxy_heatmap.setLabel("bottom", "Position", "mm")
        self.mxy_heatmap.setLabel("left", "Frequency", "Hz")
        self.mxy_heatmap.setTitle("Mxy magnitude (|Mxy|)")
        self.mxy_heatmap_item = pg.ImageItem()
        self.mxy_heatmap.addItem(self.mxy_heatmap_item)
        self.mxy_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.mxy_heatmap_layout.addItem(self.mxy_heatmap_colorbar, row=0, col=1)
        self.mxy_heatmap_colorbar.setImageItem(self.mxy_heatmap_item)
        splitter.addWidget(self.mxy_heatmap_layout)

        self.mz_heatmap_layout = pg.GraphicsLayoutWidget()
        self.mz_heatmap = self.mz_heatmap_layout.addPlot(row=0, col=0)
        self.mz_heatmap.setLabel("bottom", "Position", "mm")
        self.mz_heatmap.setLabel("left", "Frequency", "Hz")
        self.mz_heatmap.setTitle("Mz")
        self.mz_heatmap_item = pg.ImageItem()
        self.mz_heatmap.addItem(self.mz_heatmap_item)
        self.mz_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.mz_heatmap_layout.addItem(self.mz_heatmap_colorbar, row=0, col=1)
        self.mz_heatmap_colorbar.setImageItem(self.mz_heatmap_item)
        splitter.addWidget(self.mz_heatmap_layout)
        heatmap_layout.addWidget(splitter)
        self.heatmap_container.setVisible(False)
        layout.addWidget(self.heatmap_container, 1)

    def set_result(self, result):
        self.result = result
        self.time_index = max(0, result.time_s.size - 1)
        nfreq = int(result.frequency_offsets_hz.size)
        self.freq_slider.blockSignals(True)
        self.freq_slider.setRange(0, max(0, nfreq - 1))
        self.freq_slider.setValue(0)
        self.freq_slider.blockSignals(False)
        self.refresh()

    def set_time_index(self, index: int):
        if self.result is None or self.result.time_s.size == 0:
            return
        self.time_index = int(np.clip(index, 0, self.result.time_s.size - 1))
        self.refresh()

    def _freq_axis(self):
        if self.result is None:
            return np.zeros(1)
        return np.asarray(self.result.frequency_offsets_hz, dtype=float)

    def refresh(self, *_):
        if self.result is None:
            return
        show_heatmap = self.plot_type.currentText() == "Heatmap"
        self.line_container.setVisible(not show_heatmap)
        self.heatmap_container.setVisible(show_heatmap)
        self.heatmap_mode.setEnabled(show_heatmap)
        if show_heatmap:
            self._render_heatmap()
        else:
            self._render_line()

    def _snapshot(self):
        result = self.result
        time_index = int(np.clip(self.time_index, 0, result.time_s.size - 1))
        return result.mx[time_index], result.my[time_index], result.mz[time_index]

    def _render_line(self):
        result = self.result
        mx, my, mz = self._snapshot()
        position, axis = _position_axis_mm(result.positions_m)
        freq_axis = self._freq_axis()
        nfreq = freq_axis.size
        freq_index = min(self.freq_slider.value(), max(0, nfreq - 1))
        individual = self.view_mode.currentText() == "Individual freq"
        self.freq_slider.setEnabled(individual and nfreq > 1)
        if freq_axis.size:
            self.freq_label.setText(f"Freq: {freq_axis[freq_index]:.3g} Hz")
        else:
            self.freq_label.setText("Freq: 0.0 Hz")

        if individual and nfreq > 0:
            mx_line = mx[:, freq_index]
            my_line = my[:, freq_index]
            mz_line = mz[:, freq_index]
        else:
            mx_line = np.mean(mx, axis=1)
            my_line = np.mean(my, axis=1)
            mz_line = np.mean(mz, axis=1)
        mxy_line = np.hypot(mx_line, my_line)

        _clear_plot(self.mxy_plot)
        _clear_plot(self.mz_plot)
        selected = self.component_combo.get_selected_items() or ["Magnitude"]
        visible = []
        if "Phase" in selected:
            phase = np.angle(mx_line + 1j * my_line) / np.pi
            self.mxy_plot.plot(
                position, phase, pen=pg.mkPen("c", width=2), name="Phase"
            )
            visible.append(phase)
        if "Phase (unwrapped)" in selected:
            phase = np.unwrap(np.angle(mx_line + 1j * my_line)) / np.pi
            self.mxy_plot.plot(
                position,
                phase,
                pen=pg.mkPen("y", width=2),
                name="Phase (unwrapped)",
            )
            visible.append(phase)
        if "Magnitude" in selected:
            self.mxy_plot.plot(
                position, mxy_line, pen=pg.mkPen("b", width=2), name="|Mxy|"
            )
            visible.append(mxy_line)
        if "Real" in selected:
            self.mxy_plot.plot(
                position,
                mx_line,
                pen=pg.mkPen("r", style=Qt.DashLine, width=2),
                name="Mx",
            )
            visible.append(mx_line)
        if "Imaginary" in selected:
            self.mxy_plot.plot(
                position,
                my_line,
                pen=pg.mkPen("g", style=Qt.DotLine, width=2),
                name="My",
            )
            visible.append(my_line)
        self.mz_plot.plot(position, mz_line, pen=pg.mkPen("m", width=2), name="Mz")
        self.mxy_plot.setTitle("Transverse Magnetization")
        self.mz_plot.setTitle("Longitudinal Magnetization")
        self.mxy_plot.setLabel("bottom", f"Position {'xyz'[axis]}", "mm")
        self.mz_plot.setLabel("bottom", f"Position {'xyz'[axis]}", "mm")
        self.mxy_plot.setXRange(*_safe_range(position), padding=0)
        self.mz_plot.setXRange(*_safe_range(position), padding=0)
        if visible:
            self.mxy_plot.setYRange(
                *_safe_range(
                    np.concatenate([np.ravel(v) for v in visible]), fallback=(-1.0, 1.0)
                ),
                padding=0,
            )
        self.mz_plot.setYRange(*_safe_range(mz_line, fallback=(-1.0, 1.0)), padding=0)

    def _render_heatmap(self):
        result = self.result
        position, axis = _position_axis_mm(result.positions_m)
        freq_axis = self._freq_axis()
        mode = self.heatmap_mode.currentText()
        freq_index = min(self.freq_slider.value(), max(0, freq_axis.size - 1))
        if mode == "Position vs Time":
            mxy = np.abs(result.mxy[:, :, freq_index])
            mz = result.mz[:, :, freq_index]
            y_axis = result.time_s * 1000.0
            y_label = ("Time", "ms")
            title_mxy = f"|Mxy| vs time @ freq {freq_axis[freq_index]:.3g} Hz"
            title_mz = f"Mz vs time @ freq {freq_axis[freq_index]:.3g} Hz"
        else:
            snap_mxy = np.abs(result.mxy[self.time_index])
            snap_mz = result.mz[self.time_index]
            mxy = snap_mxy.T
            mz = snap_mz.T
            y_axis = freq_axis
            y_label = ("Frequency", "Hz")
            title_mxy = "Mxy magnitude (|Mxy|)"
            title_mz = "Mz"

        def set_map(plot, item, colorbar, data, title):
            x_min, x_max = _safe_range(position, pad_fraction=0.0)
            y_min, y_max = _safe_range(y_axis, pad_fraction=0.0)
            item.setImage(data, autoLevels=True, axisOrder="row-major")
            item.setRect(
                float(x_min),
                float(y_min),
                max(float(x_max - x_min), 1e-9),
                max(float(y_max - y_min), 1e-9),
            )
            plot.setLabel("bottom", f"Position {'xyz'[axis]}", "mm")
            plot.setLabel("left", *y_label)
            plot.setTitle(title)
            plot.setXRange(float(x_min), float(x_max), padding=0)
            plot.setYRange(float(y_min), float(y_max), padding=0)
            _set_colorbar_levels(colorbar, data)

        set_map(
            self.mxy_heatmap,
            self.mxy_heatmap_item,
            self.mxy_heatmap_colorbar,
            mxy,
            title_mxy,
        )
        set_map(
            self.mz_heatmap,
            self.mz_heatmap_item,
            self.mz_heatmap_colorbar,
            mz,
            title_mz,
        )
