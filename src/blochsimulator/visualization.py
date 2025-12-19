"""
visualization_export.py - Export functionality for GUI visualizations

This module provides classes for exporting visualizations from the Bloch simulator GUI:
- Static images (PNG, SVG, PDF)
- Animations (GIF, MP4) for time-resolved simulations
- Dataset exports (CSV, DAT, NPY) for external analysis

Author: Luca Nagel
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, List, Callable
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter as PgImageExporter, SVGExporter as PgSVGExporter
from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QMessageBox,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QBuffer
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from . import __version__

try:
    import imageio
except ImportError:  # pragma: no cover - dependency is optional at import time
    imageio = None


class ImageExporter:
    """
    Export PyQtGraph plots and Qt widgets as static images.

    Supports PNG, SVG, and PDF formats with configurable quality settings.
    """

    def __init__(self):
        self.default_dpi = 300  # For print quality

    def export_pyqtgraph_plot(self, plot_widget: pg.PlotWidget, filename: str,
                              format: str = 'png', width: Optional[int] = None,
                              height: Optional[int] = None) -> bool:
        """
        Export a PyQtGraph PlotWidget to an image file.

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            The plot widget to export
        filename : str
            Output filename (extension determines format if not specified)
        format : str
            Export format ('png', 'svg', 'pdf')
        width : int, optional
            Image width in pixels (defaults to widget width * DPI scaling)
        height : int, optional
            Image height in pixels (defaults to widget height * DPI scaling)

        Returns
        -------
        bool
            True if export successful, False otherwise
        """
        try:
            format = format.lower()

            # Ensure filename has correct extension
            filepath = Path(filename)
            if filepath.suffix.lower() != f'.{format}':
                filepath = filepath.with_suffix(f'.{format}')

            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if format == 'png':
                # Use ImageExporter for PNG
                exporter = PgImageExporter(plot_widget.plotItem)

                # Set resolution
                if width is None:
                    # Scale by DPI for high quality (default 96 DPI -> 300 DPI)
                    scale_factor = self.default_dpi / 96.0
                    width = int(plot_widget.width() * scale_factor)

                exporter.parameters()['width'] = width
                exporter.export(str(filepath))

            elif format == 'svg':
                # Use SVGExporter for vector format
                exporter = PgSVGExporter(plot_widget.plotItem)
                exporter.export(str(filepath))

            elif format == 'pdf':
                # PyQtGraph doesn't have native PDF export, convert SVG to PDF
                # For now, we'll use the SVG exporter and let the user convert
                # A future enhancement could use reportlab or similar
                exporter = PgSVGExporter(plot_widget.plotItem)
                svg_path = filepath.with_suffix('.svg')
                exporter.export(str(svg_path))

                # Inform user that PDF export created SVG instead
                return str(svg_path)

            else:
                raise ValueError(f"Unsupported format: {format}. Use 'png', 'svg', or 'pdf'.")

            # Verify the file was actually created
            if not filepath.exists():
                print(f"Error: File was not created at {filepath}")
                return False

            return str(filepath)

        except Exception as e:
            import traceback
            print(f"Error exporting plot: {e}")
            traceback.print_exc()
            return False

    def export_widget_screenshot(self, widget: QWidget, filename: str,
                                 format: str = 'png') -> bool:
        """
        Export a Qt widget as a screenshot image.

        Useful for 3D OpenGL views and custom widgets.

        Parameters
        ----------
        widget : QWidget
            The widget to capture
        filename : str
            Output filename
        format : str
            Image format ('png', 'jpg', 'bmp')

        Returns
        -------
        bool
            True if export successful, False otherwise
        """
        try:
            format = format.lower()

            # Ensure filename has correct extension
            filepath = Path(filename)
            if filepath.suffix.lower() != f'.{format}':
                filepath = filepath.with_suffix(f'.{format}')

            # Capture widget as pixmap
            pixmap = widget.grab()
            if format == 'svg':
                # Embed PNG in simple SVG wrapper for compatibility
                image = pixmap.toImage().convertToFormat(pixmap.toImage().Format_RGBA8888)
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                image.save(buffer, "PNG")
                b64 = buffer.data().toBase64().data().decode('ascii')
                width = image.width()
                height = image.height()
                svg_content = (
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">'
                    f'<image href="data:image/png;base64,{b64}" '
                    f'width="{width}" height="{height}" /></svg>'
                )
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(svg_content)
                return str(filepath)

            # Save to file (PNG/JPG/BMP)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            success = pixmap.save(str(filepath), format.upper())

            if success:
                return str(filepath)
            else:
                return False

        except Exception as e:
            print(f"Error exporting widget screenshot: {e}")
            return False

    def export_multiple_plots(self, plot_widgets: list, filename: str,
                             format: str = 'png', layout: str = 'vertical') -> bool:
        """
        Export multiple plots to a single image file.

        Parameters
        ----------
        plot_widgets : list of pg.PlotWidget
            List of plot widgets to export
        filename : str
            Output filename
        format : str
            Export format ('png', 'svg')
        layout : str
            Layout arrangement ('vertical', 'horizontal', 'grid')

        Returns
        -------
        bool
            True if export successful, False otherwise
        """
        # This is a placeholder for future multi-plot export
        # For now, we'll export plots individually
        raise NotImplementedError("Multi-plot export not yet implemented. "
                                 "Export plots individually for now.")


class ExportImageDialog(QDialog):
    """
    Dialog for configuring image export options.

    Allows user to select format, resolution, and output file.
    """

    def __init__(self, parent=None, default_filename: str = "export", default_directory: Path = None):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self.default_filename = default_filename
        self.default_directory = default_directory if default_directory else Path.cwd()
        self.selected_format = 'png'
        self.selected_width = None
        self.output_path = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['PNG (Raster)', 'SVG (Vector)', 'PDF (Vector)'])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # Resolution setting (for PNG only)
        self.resolution_layout = QHBoxLayout()
        self.resolution_layout.addWidget(QLabel("Width (pixels):"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 10000)
        self.width_spin.setValue(2400)  # 300 DPI at ~8 inches
        self.width_spin.setSingleStep(100)
        self.resolution_layout.addWidget(self.width_spin)
        layout.addLayout(self.resolution_layout)

        # Info label
        self.info_label = QLabel("High quality (300 DPI) - suitable for publications")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setMinimumWidth(400)

    def _on_format_changed(self, index):
        """Handle format selection change."""
        formats = ['png', 'svg', 'pdf']
        self.selected_format = formats[index]

        # Show/hide resolution controls based on format
        is_raster = (self.selected_format == 'png')
        self.width_spin.setEnabled(is_raster)

        if is_raster:
            self.info_label.setText("High quality (300 DPI) - suitable for publications")
        else:
            self.info_label.setText("Vector format - infinitely scalable, ideal for publications")

    def _on_export_clicked(self):
        """Open file dialog and export."""
        format_filters = {
            'png': "PNG Images (*.png)",
            'svg': "SVG Images (*.svg)",
            'pdf': "PDF Documents (*.pdf)"
        }

        # Use default directory for initial location
        default_path = self.default_directory / f"{self.default_filename}.{self.selected_format}"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(default_path),
            format_filters[self.selected_format]
        )

        if filename:
            self.output_path = filename
            self.selected_width = self.width_spin.value() if self.selected_format == 'png' else None
            self.accept()

    def get_export_params(self) -> Dict:
        """
        Get the selected export parameters.

        Returns
        -------
        dict
            Dictionary with 'filename', 'format', and 'width' keys
        """
        return {
            'filename': self.output_path,
            'format': self.selected_format,
            'width': self.selected_width
        }


class ExportAnimationDialog(QDialog):
    """
    Dialog for configuring animation export options.
    """

    def __init__(self, parent=None, total_frames: int = 0, default_filename: str = "animation",
                 default_directory: Path = None):
        super().__init__(parent)
        self.setWindowTitle("Export Animation")
        self.default_directory = default_directory if default_directory else Path.cwd()
        self.default_filename = default_filename
        self.total_frames = max(total_frames, 0)
        self.selected_format = 'mp4'
        self.output_path = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['MP4', 'GIF'])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame rate (FPS):"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 500)
        self.fps_spin.setValue(30)
        fps_layout.addWidget(self.fps_spin)
        layout.addLayout(fps_layout)

        # Max frames
        max_frames_layout = QHBoxLayout()
        max_frames_layout.addWidget(QLabel("Max frames (0 = all):"))
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 15000)
        self.max_frames_spin.setValue(400)
        max_frames_layout.addWidget(self.max_frames_spin)
        layout.addLayout(max_frames_layout)

        # Frame range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start frame:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max(self.total_frames - 1, 0))
        self.start_spin.setValue(0)
        range_layout.addWidget(self.start_spin)
        range_layout.addWidget(QLabel("End frame:"))
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, max(self.total_frames - 1, 0))
        self.end_spin.setValue(max(self.total_frames - 1, 0))
        range_layout.addWidget(self.end_spin)
        layout.addLayout(range_layout)
        self.start_spin.valueChanged.connect(self._sync_range)
        self.end_spin.valueChanged.connect(self._sync_range)

        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Width (px):"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(200, 10000)  # allow >999 via typing
        self.width_spin.setValue(2560)
        self.width_spin.setSingleStep(80)
        res_layout.addWidget(self.width_spin)
        res_layout.addWidget(QLabel("Height (px):"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(200, 10000)
        self.height_spin.setValue(1440)
        self.height_spin.setSingleStep(60)
        res_layout.addWidget(self.height_spin)
        layout.addLayout(res_layout)

        # Simplified mean option for data-heavy exports
        self.mean_only_checkbox = QCheckBox("Use mean traces (if available)")
        self.mean_only_checkbox.setChecked(True)
        layout.addWidget(self.mean_only_checkbox)

        # Include sequence diagram option
        self.include_sequence_checkbox = QCheckBox("Export sequence diagram as separate file")
        self.include_sequence_checkbox.setChecked(False)
        self.include_sequence_checkbox.setToolTip("If checked, also export the sequence diagram as its own animation.")
        layout.addWidget(self.include_sequence_checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setMinimumWidth(420)

    def _sync_range(self):
        """Keep frame range sane."""
        start = self.start_spin.value()
        end = self.end_spin.value()
        if start > end:
            self.end_spin.setValue(start)

    def _on_export_clicked(self):
        format_filters = {
            'mp4': "MP4 Video (*.mp4)",
            'gif': "GIF Animation (*.gif)"
        }
        formats = ['mp4', 'gif']
        self.selected_format = formats[self.format_combo.currentIndex()]
        default_path = self.default_directory / f"{self.default_filename}.{self.selected_format}"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Animation",
            str(default_path),
            format_filters[self.selected_format]
        )

        if filename:
            self.output_path = filename
            self.accept()

    def get_export_params(self) -> Dict:
        """Return the export configuration."""
        return {
            'filename': self.output_path,
            'format': self.selected_format,
            'fps': self.fps_spin.value(),
            'max_frames': self.max_frames_spin.value(),
            'start_idx': self.start_spin.value(),
            'end_idx': self.end_spin.value(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'mean_only': self.mean_only_checkbox.isChecked(),
            'include_sequence': self.include_sequence_checkbox.isChecked()
        }


class ExportDataDialog(QDialog):
    """
    Dialog for configuring data export options (HDF5, Notebooks, etc.).
    Allows selecting multiple formats simultaneously.
    """

    def __init__(self, parent=None, default_filename: str = "simulation_data", default_directory: Path = None):
        super().__init__(parent)
        self.setWindowTitle("Export Data")
        self.default_filename = default_filename
        self.default_directory = default_directory if default_directory else Path.cwd()
        self.selected_options = {}
        self.base_path = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select export formats:"))

        # Checkboxes
        self.chk_hdf5 = QCheckBox("HDF5 Data File (.h5)")
        self.chk_hdf5.setChecked(True)
        self.chk_hdf5.setToolTip("Full simulation data including magnetization, signal, and parameters.")
        layout.addWidget(self.chk_hdf5)

        self.chk_nb_analysis = QCheckBox("Jupyter Notebook: Analysis (Mode A)")
        self.chk_nb_analysis.setChecked(False)
        self.chk_nb_analysis.setToolTip("Notebook that loads the HDF5 file for visualization and analysis.")
        layout.addWidget(self.chk_nb_analysis)

        self.chk_nb_repro = QCheckBox("Jupyter Notebook: Reproduce (Mode B)")
        self.chk_nb_repro.setChecked(False)
        self.chk_nb_repro.setToolTip("Notebook that contains all parameters to re-run this simulation.")
        layout.addWidget(self.chk_nb_repro)

        self.chk_csv = QCheckBox("CSV/Text Data")
        self.chk_csv.setChecked(False)
        self.chk_csv.setToolTip("Magnetization and signal traces in text format.")
        layout.addWidget(self.chk_csv)

        # Options for CSV
        self.csv_opts = QWidget()
        csv_layout = QHBoxLayout()
        csv_layout.setContentsMargins(20, 0, 0, 0)
        csv_layout.addWidget(QLabel("Format:"))
        self.csv_fmt = QComboBox()
        self.csv_fmt.addItems(["csv", "tsv", "dat", "npy"])
        csv_layout.addWidget(self.csv_fmt)
        self.csv_opts.setLayout(csv_layout)
        self.csv_opts.setVisible(False)
        self.chk_csv.toggled.connect(self.csv_opts.setVisible)
        layout.addWidget(self.csv_opts)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setMinimumWidth(350)

    def _on_export_clicked(self):
        # Validate selection
        if not any([self.chk_hdf5.isChecked(), self.chk_nb_analysis.isChecked(), 
                   self.chk_nb_repro.isChecked(), self.chk_csv.isChecked()]):
            QMessageBox.warning(self, "No Selection", "Please select at least one export format.")
            return

        # Mode A notebook requires HDF5
        if self.chk_nb_analysis.isChecked() and not self.chk_hdf5.isChecked():
            ret = QMessageBox.question(self, "Dependency", 
                                     "Analysis Notebook requires HDF5 data.\nEnable HDF5 export also?",
                                     QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes:
                self.chk_hdf5.setChecked(True)
            else:
                return

        # Get base filename
        default_path = self.default_directory / self.default_filename
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data (Base Filename)",
            str(default_path),
            "All Files (*)"
        )

        if filename:
            # Strip extension if user typed one, we'll append based on formats
            p = Path(filename)
            if p.suffix.lower() in ['.h5', '.ipynb', '.csv', '.dat', '.npy', '.txt']:
                self.base_path = str(p.with_suffix(''))
            else:
                self.base_path = str(p)
            
            self.accept()

    def get_export_options(self) -> Dict:
        """Return selected options and base filename."""
        return {
            'base_path': self.base_path,
            'hdf5': self.chk_hdf5.isChecked(),
            'notebook_analysis': self.chk_nb_analysis.isChecked(),
            'notebook_repro': self.chk_nb_repro.isChecked(),
            'csv': self.chk_csv.isChecked(),
            'csv_format': self.csv_fmt.currentText()
        }


class AnimationExporter:
    """
    Export time-resolved simulations as animated GIF or MP4 videos.
    """

    def __init__(self):
        self.default_width = 960
        self.default_height = 540
        self.default_bitrate = "5000k"

    def _ensure_imageio(self):
        if imageio is None:
            raise ImportError("Animation export requires 'imageio'. Install with: pip install imageio imageio-ffmpeg")

    def _infer_format(self, filename: str, format_hint: Optional[str] = None) -> str:
        fmt = format_hint or Path(filename).suffix.lower().lstrip('.')
        fmt = fmt.lower()
        if fmt in ('gif', 'mp4'):
            return fmt
        raise ValueError(f"Unsupported animation format '{fmt}'. Use GIF or MP4.")

    @staticmethod
    def _even_size(value: int) -> int:
        """Ensure dimension is divisible by 2 for libx264."""
        if value % 2 != 0:
            return max(2, value + 1)
        return value

    def _compute_indices(self, total_frames: int, max_frames: Optional[int] = None,
                         start_idx: int = 0, end_idx: Optional[int] = None) -> np.ndarray:
        end_idx = total_frames - 1 if end_idx is None else min(end_idx, total_frames - 1)
        start_idx = max(0, min(start_idx, end_idx))
        count = end_idx - start_idx + 1
        if max_frames is None or max_frames <= 0 or count <= max_frames:
            return np.arange(start_idx, end_idx + 1, dtype=int)
        step = int(np.ceil(count / max_frames))
        indices = np.arange(start_idx, end_idx + 1, step, dtype=int)
        if indices[-1] != end_idx:
            indices = np.append(indices, end_idx)
        return indices

    def _compute_group_limits(self, groups: List[Dict]) -> List[Tuple[float, float]]:
        limits = []
        for group in groups:
            ymin, ymax = None, None
            for series in group.get('series', []):
                data = np.asarray(series['data'])
                if data.size == 0:
                    continue
                smin = np.nanmin(data)
                smax = np.nanmax(data)
                ymin = smin if ymin is None else min(ymin, smin)
                ymax = smax if ymax is None else max(ymax, smax)
            if ymin is None or ymax is None:
                ymin, ymax = -1.0, 1.0
            if np.isclose(ymin, ymax):
                pad = 0.5 if ymin == 0 else abs(ymin) * 0.1
                ymin -= pad
                ymax += pad
            limits.append((float(ymin), float(ymax)))
        return limits

    def export_time_series_animation(
        self,
        time_s: Union[List[float], np.ndarray],
        groups: List[Dict],
        filename: str,
        fps: int = 15,
        max_frames: int = 400,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
        indices: Optional[np.ndarray] = None,
        frame_hook: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> Optional[str]:
        """
        Export a list of time-series traces as an animation.

        Parameters
        ----------
        time_s : array-like
            Time array in seconds (length = nframes)
        groups : list of dict
            Each dict: {'title': str, 'ylabel': str, 'series': [{'data': arr, 'label': str, 'color': str, 'style': str}]}
        filename : str
            Output filename
        fps : int
            Frames per second
        max_frames : int
            Maximum frames to render (0 or None = all)
        width : int
            Output width in pixels
        height : int
            Output height in pixels
        format : str
            'gif' or 'mp4'. If None, inferred from filename.
        progress_cb : callable
            progress_cb(done_frames, total_frames)
        cancel_cb : callable
            Returns True if export should abort.
        """
        self._ensure_imageio()

        time_s = np.asarray(time_s, dtype=float)
        if time_s.ndim != 1 or len(time_s) == 0:
            raise ValueError("time_s must be a 1D array with at least one element.")
        fmt = self._infer_format(filename, format_hint=format)

        filepath = Path(filename)
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        width_px = self._even_size(int(width)) if width else self._even_size(self.default_width)
        height_px = self._even_size(int(height)) if height else self._even_size(self.default_height)

        if indices is None:
            indices = self._compute_indices(len(time_s), max_frames=max_frames, start_idx=start_idx, end_idx=end_idx)
        else:
            indices = np.asarray(indices, dtype=int)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(time_s)):
                raise ValueError("indices must be a 1D array within time bounds.")
        
        # Subsample data to match indices for efficiency
        time_ds = time_s[indices]
        
        # Prepare data structures
        groups_ds = []
        for group in groups:
            new_group = dict(group)
            new_series = []
            for series in group.get('series', []):
                data = np.asarray(series['data'])
                if data.ndim == 0:
                    data = np.asarray([float(data)] * len(time_s))
                if len(data) != len(time_s):
                    raise ValueError("Series length must match time array length.")
                # We keep the full data for plotting the background trace, 
                # but we'll need efficient access for the "current point" marker
                new_series.append({
                    'full_data': data,      # Full trace
                    'data': data[indices],  # Subsampled for dot marker
                    'label': series.get('label', ''),
                    'color': series.get('color', None),
                    'style': series.get('style', '-')
                })
            new_group['series'] = new_series
            groups_ds.append(new_group)

        # Initialize Plot (One-time setup)
        nrows = max(len(groups_ds), 1)
        fig = Figure(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
        canvas = FigureCanvasAgg(fig)
        axes = []
        
        # Store artist references for updating
        time_lines = []     # Vertical cursor lines
        dot_artists = []    # Current value dots (list of lists)

        # Compute limits first
        limits = self._compute_group_limits(groups_ds)

        for i, group in enumerate(groups_ds):
            ax = fig.add_subplot(nrows, 1, i + 1)
            group_dots = []
            
            # Plot full static traces
            for series in group.get('series', []):
                color = series.get('color', None)
                style = series.get('style', '-')
                # Background trace (static)
                ax.plot(time_s, series['full_data'], style, label=series['label'], color=color, linewidth=1.8)
                
                # Current point marker (dynamic) - initialize at start
                dot, = ax.plot([], [], 'o', color=color or 'k', markersize=5, alpha=0.9)
                group_dots.append(dot)
            
            dot_artists.append(group_dots)

            # Vertical time cursor (dynamic)
            vline = ax.axvline(time_ds[0], color='k', linestyle='--', alpha=0.25)
            time_lines.append(vline)

            ax.set_title(group.get('title', ''), fontsize=10)
            ax.set_ylabel(group.get('ylabel', ''))
            ax.set_xlim(time_s[0], time_s[-1])
            ymin, ymax = limits[i]
            ax.set_ylim(ymin, ymax)
            
            if group.get('series'):
                ax.legend(loc='upper right', fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel("Time (s)")
            axes.append(ax)

        fig.tight_layout()

        # Initialize Video Writer
        if fmt == 'gif':
            writer = imageio.get_writer(str(filepath), mode='I', fps=fps, format='GIF', loop=0)
        else:
            writer = imageio.get_writer(
                str(filepath),
                fps=fps,
                format='FFMPEG',
                codec='libx264',
                bitrate=self.default_bitrate,
                quality=8,
                macro_block_size=None,
                ffmpeg_params=['-metadata', f'comment=BlochSimulator {__version__}']
            )

        total_frames = len(time_ds)
        
        try:
            for i in range(total_frames):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("Animation export cancelled")

                current_time = time_ds[i]

                # Update dynamic artists
                for ax_idx, group in enumerate(groups_ds):
                    # Update vertical line
                    time_lines[ax_idx].set_xdata([current_time, current_time])
                    
                    # Update dots
                    for ser_idx, series in enumerate(group['series']):
                        # Get pre-subsampled value for efficiency
                        val = series['data'][i]
                        dot_artists[ax_idx][ser_idx].set_data([current_time], [val])

                # Render frame
                canvas.draw()
                buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                w, h = canvas.get_width_height()
                image = buf.reshape((h, w, 4))[:, :, :3]  # Drop alpha

                if frame_hook:
                    image = frame_hook(image, int(indices[i]))
                
                writer.append_data(image)
                
                if progress_cb and (i % 5 == 0 or i == total_frames - 1):
                    progress_cb(i + 1, total_frames)
        finally:
            writer.close()
            # Clean up matplotlib figures
            import matplotlib.pyplot as plt
            plt.close(fig)

        if progress_cb:
            progress_cb(total_frames, total_frames)

        return str(filepath)


class DatasetExporter:
    """
    Export simulation data as CSV, TSV (DAT), or NPY files for external analysis.
    """

    def __init__(self):
        # No stateful resources required
        self.supported_formats = ('csv', 'dat', 'tsv', 'npy')

    def _write_columns(self, columns: Dict[str, np.ndarray], filename: str, format: str = 'csv') -> str:
        """Write a dictionary of named 1D arrays to disk."""
        if not columns:
            raise ValueError("No columns provided for export.")
        lengths = {len(np.asarray(col).ravel()) for col in columns.values()}
        if len(lengths) != 1:
            raise ValueError("All columns must have the same length for export.")
        length = lengths.pop()
        keys = list(columns.keys())
        data = np.column_stack([np.asarray(columns[k]).ravel() for k in keys])

        fmt = format.lower()
        filepath = Path(filename)
        version_header = f"# BlochSimulator {__version__}\n"
        
        if fmt == 'csv':
            if filepath.suffix.lower() != '.csv':
                filepath = filepath.with_suffix('.csv')
            header = version_header + ",".join(keys)
            np.savetxt(filepath, data, delimiter=',', header=header, comments='')
        elif fmt in ('dat', 'tsv'):
            if filepath.suffix.lower() not in ('.dat', '.tsv'):
                filepath = filepath.with_suffix('.dat')
            header = version_header + "\t".join(keys)
            np.savetxt(filepath, data, delimiter='\t', header=header, comments='')
        elif fmt == 'npy':
            if filepath.suffix.lower() != '.npy':
                filepath = filepath.with_suffix('.npy')
            dtype = [(key, float) for key in keys]
            structured = np.zeros(length, dtype=dtype)
            for key in keys:
                structured[key] = np.asarray(columns[key]).ravel()
            np.save(filepath, structured)
        else:
            raise ValueError(f"Unsupported export format '{format}'. Use one of: {self.supported_formats}")

        return str(filepath)

    def export_magnetization(self, time_s: np.ndarray, mx: np.ndarray, my: np.ndarray, mz: np.ndarray,
                             positions: Optional[np.ndarray], frequencies: Optional[np.ndarray],
                             filename: str, format: str = 'csv') -> str:
        """Export magnetization time series for all positions/frequencies."""
        time_ms = np.asarray(time_s).ravel() * 1000.0
        mx = np.asarray(mx)
        my = np.asarray(my)
        mz = np.asarray(mz)
        # Normalize shapes to (ntime, npos, nfreq)
        if mx.ndim == 2:
            mx = mx[None, ...]
            my = my[None, ...]
            mz = mz[None, ...]
        ntime, npos, nfreq = mx.shape

        columns: Dict[str, np.ndarray] = {'time_ms': time_ms}
        for pi in range(npos):
            for fi in range(nfreq):
                label = f"p{pi}_f{fi}"
                columns[f"mx_{label}"] = mx[:, pi, fi]
                columns[f"my_{label}"] = my[:, pi, fi]
                columns[f"mz_{label}"] = mz[:, pi, fi]
                mxy_mag = np.sqrt(mx[:, pi, fi] ** 2 + my[:, pi, fi] ** 2)
                columns[f"mxy_mag_{label}"] = mxy_mag

        return self._write_columns(columns, filename, format=format)

    def export_signal(self, time_s: np.ndarray, signal: np.ndarray, filename: str, format: str = 'csv') -> str:
        """Export complex signal traces for all positions/frequencies."""
        time_ms = np.asarray(time_s).ravel() * 1000.0
        sig = np.asarray(signal)
        if sig.ndim == 1:
            sig = sig[:, None, None]
        elif sig.ndim == 2:
            sig = sig[:, :, None]
        ntime, npos, nfreq = sig.shape

        columns: Dict[str, np.ndarray] = {'time_ms': time_ms}
        for pi in range(npos):
            for fi in range(nfreq):
                label = f"p{pi}_f{fi}"
                trace = sig[:, pi, fi]
                columns[f"signal_real_{label}"] = np.real(trace)
                columns[f"signal_imag_{label}"] = np.imag(trace)
                columns[f"signal_mag_{label}"] = np.abs(trace)
                columns[f"signal_phase_rad_{label}"] = np.angle(trace)

        return self._write_columns(columns, filename, format=format)

    def export_spectrum(self, frequency_hz: np.ndarray, series: Dict[str, np.ndarray],
                        filename: str, format: str = 'csv') -> str:
        """Export spectrum data with one or more series."""
        freq = np.asarray(frequency_hz).ravel()
        columns: Dict[str, np.ndarray] = {'frequency_hz': freq}
        for name, arr in series.items():
            if arr is None:
                continue
            data = np.asarray(arr).ravel()
            if len(data) != len(freq):
                continue
            columns[name] = data
        return self._write_columns(columns, filename, format=format)

    def export_spatial(self, position_axis: np.ndarray, mxy: np.ndarray, mz: np.ndarray,
                       filename: str, format: str = 'csv',
                       mxy_per_freq: Optional[np.ndarray] = None,
                       mz_per_freq: Optional[np.ndarray] = None) -> str:
        """Export spatial profiles along the chosen axis."""
        pos = np.asarray(position_axis).ravel()
        columns: Dict[str, np.ndarray] = {
            'position_m': pos,
            'mxy': np.asarray(mxy).ravel(),
            'mz': np.asarray(mz).ravel()
        }
        if mxy_per_freq is not None:
            mxy_arr = np.asarray(mxy_per_freq)
            for fi in range(mxy_arr.shape[1]):
                columns[f"mxy_f{fi}"] = mxy_arr[:, fi]
        if mz_per_freq is not None:
            mz_arr = np.asarray(mz_per_freq)
            for fi in range(mz_arr.shape[1]):
                columns[f"mz_f{fi}"] = mz_arr[:, fi]
        return self._write_columns(columns, filename, format=format)
