from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMenu,
    QSlider,
    QComboBox,
    QCheckBox,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
from pathlib import Path
from ..visualization import ImageExporter


class MagnetizationViewer(QWidget):
    """3D visualization of magnetization vector."""

    position_changed = pyqtSignal(int)
    view_filter_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._export_dir_provider = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.playhead_line = None

        # Add export header for 3D view
        header_3d = QHBoxLayout()
        header_3d.addWidget(QLabel("3D Magnetization Vector"))
        header_3d.addStretch()

        self.export_3d_btn = QPushButton("Export ▼")
        export_3d_menu = QMenu()
        export_3d_menu.addAction(
            "Image (PNG)...", lambda: self._export_3d_screenshot("png")
        )
        export_3d_menu.addAction(
            "Image (SVG)...", lambda: self._export_3d_screenshot("svg")
        )
        export_3d_menu.addSeparator()
        export_3d_menu.addAction("Animation (GIF/MP4)...", self._export_3d_animation)
        export_3d_menu.addAction(
            "Sequence diagram (GIF/MP4)...", self._export_sequence_animation
        )
        self.export_3d_btn.setMenu(export_3d_menu)
        header_3d.addWidget(self.export_3d_btn)
        layout.addLayout(header_3d)

        # 3D view
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=5)
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gl_widget.setMinimumHeight(300)

        # Add coordinate axes
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.gl_widget.addItem(axis)

        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(1, 1)
        self.gl_widget.addItem(grid)

        # Initialize magnetization vectors (one per frequency)
        self.vector_plot = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)), color=(1, 0, 0, 1), width=3, mode="lines"
        )
        self.gl_widget.addItem(self.vector_plot)
        self.vector_colors = None

        layout.addWidget(self.gl_widget, stretch=5)

        # Preview plot for time cursor
        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setLabel("left", "M")
        self.preview_plot.setLabel("bottom", "Time", "ms")
        self.preview_plot.enableAutoRange(x=False, y=False)
        self.preview_plot.setMaximumHeight(180)
        self.preview_mx = self.preview_plot.plot(pen="r")
        self.preview_my = self.preview_plot.plot(pen="g")
        self.preview_mz = self.preview_plot.plot(pen="b")
        self.preview_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y"))
        self.preview_plot.addItem(self.preview_line)
        layout.addWidget(self.preview_plot, stretch=1)

        # Time slider for scrubbing animation
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(self._slider_moved)
        self.time_slider.setVisible(
            False
        )  # Hide in favor of the universal time control
        layout.addWidget(self.time_slider)

        # B1 indicator arrow (optional overlay)
        self.b1_arrow = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(0, 1, 1, 0.9),
            width=3,
            mode="lines",
        )
        self.b1_arrow.setVisible(False)
        self.gl_widget.addItem(self.b1_arrow)
        self.b1_scale = 1.0

        # Control buttons and view mode selectors
        control_container = QWidget()
        controls_v = QVBoxLayout()
        controls_v.setContentsMargins(0, 0, 0, 0)
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            ["All positions x freqs", "Positions @ freq", "Freqs @ position"]
        )
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        view_layout.addWidget(self.view_mode_combo)
        self.selector_label = QLabel("All spins")
        view_layout.addWidget(self.selector_label)
        self.selector_slider = QSlider(Qt.Horizontal)
        self.selector_slider.setRange(0, 0)
        self.selector_slider.valueChanged.connect(self._on_selector_changed)
        view_layout.addWidget(self.selector_slider)
        controls_v.addLayout(view_layout)

        # Initialize tracking state and path storage BEFORE checkbox initialization
        self.length_scale = 1.0  # Reference magnitude used to normalize vectors in view
        self.preview_time = None
        self.track_path = False
        self.path_points = []
        self.path_item = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)), color=(1, 1, 0, 0.8), width=2, mode="line_strip"
        )
        self.gl_widget.addItem(self.path_item)
        self.mean_vector = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(1, 1, 0, 1), width=5, mode="lines"
        )
        self.gl_widget.addItem(self.mean_vector)

        # Now create checkboxes that depend on these variables
        self.track_checkbox = QCheckBox("Track tip path")
        self.track_checkbox.setChecked(True)
        self.track_checkbox.toggled.connect(self._toggle_track_path)
        # Sync internal flag to the initial checkbox state so tracking is active on first playback
        self._toggle_track_path(self.track_checkbox.isChecked())
        controls_v.addWidget(self.track_checkbox)
        self.mean_checkbox = QCheckBox("Show mean magnetization")
        self.mean_checkbox.setChecked(False)
        self.mean_checkbox.toggled.connect(self._toggle_mean_vector)
        controls_v.addWidget(self.mean_checkbox)
        control_container.setLayout(controls_v)
        self.control_container = control_container
        layout.addWidget(control_container)

        self.last_positions = None
        self.last_frequencies = None
        # Track available position/frequency counts for selector range updates
        self._npos = 1
        self._nfreq = 1
        self._update_selector_range()

        self.setLayout(layout)

    def _ensure_vectors(self, count: int, colors=None):
        """Cache colors for vector updates."""
        if colors is not None:
            self.vector_colors = colors

    def set_length_scale(self, scale: float):
        """Set reference magnitude to normalize displayed vectors."""
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        self.length_scale = float(scale)

    def update_magnetization(self, mx, my=None, mz=None, colors=None):
        """
        Update the magnetization vector display.
        Accepts either separate mx/my/mz scalars/arrays or an array of shape (nfreq, 3).
        """
        if my is None and mz is None:
            arr = np.asarray(mx)
            if arr.ndim == 1:
                arr = arr.reshape(1, 3)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                pass
            else:
                return
            vecs = arr
        else:
            mx_arr = np.atleast_1d(mx)
            my_arr = np.atleast_1d(my)
            mz_arr = np.atleast_1d(mz)
            if mx_arr.shape != my_arr.shape or mx_arr.shape != mz_arr.shape:
                return
            if mx_arr.ndim == 0:
                vecs = np.array([[float(mx_arr), float(my_arr), float(mz_arr)]])
            else:
                vecs = np.stack([mx_arr, my_arr, mz_arr], axis=-1)

        count = vecs.shape[0]
        self._ensure_vectors(count, colors=colors)

        norm = 1.0 / max(self.length_scale, 1e-9)
        vecs_scaled = vecs * norm

        # Construct interleaved array for single draw call: (Origin, Tip, Origin, Tip...)
        pos = np.zeros((count * 2, 3), dtype=np.float32)
        pos[1::2] = vecs_scaled

        # Handle colors
        use_colors = (
            self.vector_colors
            if self.vector_colors is not None and len(self.vector_colors) >= count
            else None
        )
        if use_colors is not None:
            # Repeat colors for each vertex (2 per line)
            c_arr = np.asarray(use_colors)
            if c_arr.ndim == 1 and c_arr.shape[0] == 4:
                self.vector_plot.setData(pos=pos, color=c_arr, mode="lines")
            else:
                c_expanded = np.repeat(c_arr[:count], 2, axis=0)
                self.vector_plot.setData(pos=pos, color=c_expanded, mode="lines")
        else:
            self.vector_plot.setData(pos=pos, color=(1, 0, 0, 1), mode="lines")

        mean_vec = np.mean(vecs_scaled, axis=0) if vecs.size else None
        if self.track_path and mean_vec is not None:
            self._append_path_point(mean_vec)
        # Mean vector (over all components)
        if mean_vec is not None and (self.mean_checkbox.isChecked() or self.track_path):
            self.mean_vector.setData(pos=np.array([[0, 0, 0], mean_vec]))
            self.mean_vector.setVisible(True)
        else:
            self.mean_vector.setVisible(False)

    def set_preview_data(self, time_ms, mx, my, mz):
        """Update preview plot and slider for scrubbing."""
        if time_ms is None or len(time_ms) == 0:
            self.preview_time = None
            self.preview_mx.clear()
            self.preview_my.clear()
            self.preview_mz.clear()
            self.time_slider.setRange(0, 0)
            return
        self.preview_time = np.asarray(time_ms)
        self.preview_mx.setData(time_ms, mx)
        self.preview_my.setData(time_ms, my)
        self.preview_mz.setData(time_ms, mz)
        x_min, x_max = float(time_ms[0]), float(time_ms[-1])
        # Clamp preview to expected magnetization range
        max_abs = 1.0
        for arr in (mx, my, mz):
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.size:
                with np.errstate(invalid="ignore"):
                    current = np.nanmax(np.abs(arr_np))
                if np.isfinite(current):
                    max_abs = max(max_abs, float(current))
        y_min, y_max = -1.1 * max_abs, 1.1 * max_abs
        self.preview_plot.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        self.preview_plot.setXRange(x_min, x_max, padding=0)
        self.preview_plot.setYRange(y_min, y_max, padding=0)
        max_idx = max(len(time_ms) - 1, 0)
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max_idx)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self._update_cursor_line(0)

    def set_cursor_index(self, idx: int):
        """Move cursor/slider without emitting position change."""
        if self.preview_time is None:
            return
        idx = int(max(0, min(idx, len(self.preview_time) - 1)))
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(idx)
        self.time_slider.blockSignals(False)
        self._update_cursor_line(idx)

    def _slider_moved(self, idx: int):
        self._update_cursor_line(idx)
        self.position_changed.emit(idx)

    def _update_cursor_line(self, idx: int):
        if self.preview_time is None or len(self.preview_time) == 0:
            return
        idx = int(max(0, min(idx, len(self.preview_time) - 1)))
        self.preview_line.setValue(self.preview_time[idx])
        if not self.track_path:
            self._clear_path()
        # Move sequence playhead if visible
        if self.playhead_line is not None:
            try:
                self.playhead_line.setValue(self.preview_time[idx])
            except Exception:
                pass
        # Time cursor lines removed for performance
        pass

    def _on_view_mode_changed(self, *args):
        self._update_selector_range()
        self.view_filter_changed.emit()

    def _on_selector_changed(self, value: int):
        # Update label to reflect the new selection, then notify listeners
        self._update_selector_range()
        self.view_filter_changed.emit()

    def _update_selector_range(self):
        """Update selector slider/label based on current mode and data availability."""
        mode = (
            self.view_mode_combo.currentText()
            if hasattr(self, "view_mode_combo")
            else "All positions x freqs"
        )
        if mode == "Positions @ freq":
            max_idx = max(0, self._nfreq - 1)
            idx = min(self.selector_slider.value(), max_idx)
            freq_hz_val = (
                self.last_frequencies[idx]
                if self.last_frequencies is not None
                and idx < len(self.last_frequencies)
                else idx
            )
            label_text = f"Freq: {freq_hz_val:.1f} Hz"
        elif mode == "Freqs @ position":
            max_idx = max(0, self._npos - 1)
            idx = min(self.selector_slider.value(), max_idx)
            pos_val = (
                self.last_positions[idx, 2] * 100
                if self.last_positions is not None and idx < len(self.last_positions)
                else idx
            )
            label_text = f"Pos: {pos_val:.2f} cm"
        else:
            max_idx = 0
            label_text = "All spins"

        self.selector_slider.blockSignals(True)
        self.selector_slider.setMaximum(max_idx)
        self.selector_slider.setValue(
            min(self.selector_slider.value(), max_idx) if max_idx > 0 else 0
        )
        self.selector_slider.setVisible(max_idx > 0)
        self.selector_slider.blockSignals(False)
        self.selector_label.setText(label_text)

    def set_selector_limits(self, npos: int, nfreq: int, disable: bool = False):
        """Set available position/frequency counts for the selector control."""
        self._npos = max(1, int(npos)) if np.isfinite(npos) else 1
        self._nfreq = max(1, int(nfreq)) if np.isfinite(nfreq) else 1
        enabled = not disable
        self.view_mode_combo.setEnabled(enabled)
        self.selector_slider.setEnabled(enabled)
        self._update_selector_range()

    def get_view_mode(self) -> str:
        """Return the current 3D view mode selection."""
        return (
            self.view_mode_combo.currentText()
            if hasattr(self, "view_mode_combo")
            else "All positions x freqs"
        )

    def get_selector_index(self) -> int:
        """Return the currently selected index for the active view mode."""
        return (
            int(self.selector_slider.value()) if hasattr(self, "selector_slider") else 0
        )

    def _toggle_track_path(self, enabled: bool):
        self.track_path = enabled
        if not enabled:
            self._clear_path()

    def _toggle_mean_vector(self, enabled: bool):
        if not enabled:
            self.mean_vector.setVisible(False)

    def _clear_path(self):
        self.path_points = []
        self.path_item.setData(pos=np.zeros((0, 3)))

    def _append_path_point(self, vec):
        """Append a point to the tracked tip path."""
        if not self.track_path:
            return
        vec = np.asarray(vec, dtype=float).ravel()
        if vec.shape[0] != 3:
            return
        self.path_points.append(vec)
        if len(self.path_points) > 5000:
            self.path_points = self.path_points[-5000:]
        self.path_item.setData(pos=np.asarray(self.path_points))

    def _export_3d_screenshot(self, format="png"):
        """Export 3D view as screenshot."""
        exporter = ImageExporter()

        export_dir = self._resolve_export_directory()
        default_path = export_dir / f"3d_view.{format}"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D View",
            str(default_path),
            f"{format.upper()} Images (*.{format})",
        )

        if filename:
            try:
                result = exporter.export_widget_screenshot(
                    self.gl_widget, filename, format=format
                )

                if result:
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"3D view exported to:\n{Path(result).name}",
                    )
                else:
                    QMessageBox.warning(
                        self, "Export Failed", "Could not export 3D view."
                    )

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _resolve_export_directory(self) -> Path:
        """Resolve an export directory via provider, window hook, or cwd."""
        # External provider takes precedence
        if callable(self._export_dir_provider):
            try:
                path = Path(self._export_dir_provider())
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                pass
        # Ask the top-level window if it exposes an export directory helper
        win = self.window()
        if win and hasattr(win, "_get_export_directory"):
            try:
                path = Path(win._get_export_directory())
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                pass
        path = Path.cwd()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def set_export_directory_provider(self, provider):
        """Provide a callable returning a directory for exports."""
        self._export_dir_provider = provider

    def _show_not_implemented_3d(self, feature_name):
        """Show a message for features not yet implemented in 3D viewer."""
        QMessageBox.information(
            self,
            "Coming Soon",
            f"{feature_name} export will be available in a future update.",
        )

    def _export_3d_animation(self):
        """
        Delegate animation export to parent window (BlochSimulatorGUI) if available.
        """
        win = self.window()
        if win and hasattr(win, "_export_3d_animation"):
            try:
                win._export_3d_animation()
                return
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))
                return
        # Fallback message if parent handler not found
        self._show_not_implemented_3d("Animation")

    def _export_sequence_animation(self):
        """Delegate sequence diagram animation export to parent window."""
        win = self.window()
        if win and hasattr(win, "_export_sequence_diagram_animation"):
            try:
                win._export_sequence_diagram_animation()
                return
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))
                return
        self._show_not_implemented_3d("Sequence animation")
