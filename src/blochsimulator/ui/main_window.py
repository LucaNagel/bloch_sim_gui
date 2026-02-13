import sys
import os
import time
import math
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QComboBox,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QTabWidget,
    QTextEdit,
    QSplitter,
    QProgressBar,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QMenu,
    QDialog,
    QProgressDialog,
    QToolBar,
    QListWidget,
    QToolButton,
    QFrame,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QImage, QPalette, QColor

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from .. import __version__
from ..simulator import BlochSimulator, TissueParameters
from ..visualization import (
    ImageExporter,
    ExportImageDialog,
    AnimationExporter,
    ExportAnimationDialog,
    ExportDataDialog,
    DatasetExporter,
    imageio as vz_imageio,
)
from ..slice_explorer import SliceSelectionExplorer

# Import phantom/kspace if available
try:
    from ..phantom import Phantom, PhantomFactory
    from ..phantom_widget import PhantomWidget

    PHANTOM_AVAILABLE = False  # Disabled per user request
except ImportError:
    PHANTOM_AVAILABLE = False

try:
    from ..kspace_widget import KSpaceWidget
    from ..kspace import KSpaceSimulator

    KSPACE_AVAILABLE = False  # Disabled per user request
except ImportError:
    KSPACE_AVAILABLE = False

from .widgets import CheckableComboBox
from .threads import SimulationThread
from .tissue_parameters import TissueParameterWidget
from .rf_pulse_designer import RFPulseDesigner
from .sequence_designer import SequenceDesigner
from .controls import UniversalTimeControl
from .magnetization_viewer import MagnetizationViewer
from .parameter_sweep import ParameterSweepWidget
from .tutorial_manager import TutorialManager
from .tutorial_overlay import TutorialOverlay


def get_app_data_dir() -> Path:
    """Return a writable per-user application directory."""
    override = os.environ.get("BLOCH_APP_DIR")
    if override:
        return Path(override).expanduser()

    system = sys.platform
    if system.startswith("win"):
        root = Path(os.environ.get("APPDATA", Path.home()))
    elif system == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return root / "BlochSimulator"


class BlochSimulatorGUI(QMainWindow):
    """Main GUI window for the Bloch simulator."""

    def __init__(self):
        super().__init__()
        self.simulator = BlochSimulator(use_parallel=True, num_threads=4)
        self.simulation_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Bloch Equation Simulator")
        self.setGeometry(100, 100, 1400, 900)
        self.last_pulse_range = None
        self.mxy_region = None
        self.mz_region = None
        self.signal_region = None
        self.spatial_plot = None
        self.last_result = None
        self.last_positions = None
        self.last_frequencies = None
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._animate_vector)
        self.anim_interval_ms = 30
        self.anim_data = None
        self.anim_time = None
        self.playback_indices = (
            None  # Mapping from playback frame to full-resolution time index
        )
        self.playback_time = None  # Time array aligned to playback indices (seconds)
        self.playback_time_ms = (
            None  # Same as playback_time but in ms for plot previews
        )
        self.anim_index = 0
        self._playback_anchor_wall = None  # monotonic timestamp for playback pacing
        self._playback_anchor_time_ms = None  # simulation time (ms) at anchor
        self._frame_step = 1
        self._min_anim_interval_ms = (
            1000.0 / 120.0
        )  # cap display rate to avoid event loop overload
        self._target_render_fps = 60.0
        self._min_render_interval_ms = 1000.0 / self._target_render_fps
        self._last_render_wall = None
        self._suppress_heavy_updates = False
        self._heavy_update_every = (
            3  # update heavy plots every N frames during playback
        )
        self._playback_frame_counter = 0
        self.anim_b1 = None
        self.anim_b1_scale = 1.0
        self.anim_vectors_full = None  # (ntime, npos, nfreq, 3) before flattening
        self.mxy_legend = None
        self.mz_legend = None
        self.signal_legend = None
        self.initial_mz = 1.0  # Track initial Mz to scale plot limits
        self._last_spatial_export = None
        self._last_spectrum_export = None
        self._spectrum_final_range = None
        self.dataset_exporter = DatasetExporter()
        self._plotting_in_progress = False
        self.tutorial_manager = TutorialManager(self)
        self.tutorial_overlay = None
        self.tutorial_manager.step_reached.connect(self._on_tutorial_step)
        self.tutorial_manager.playback_finished.connect(self._on_tutorial_finished)
        self._sweep_mode = False
        self.spectrum_y_max = 1.1  # Constant maximum for spectrum Y-axis

        # Pre-calculated plot ranges for stability during animation
        self.spatial_mxy_yrange = None
        self.spatial_mz_yrange = None
        self.spectrum_yrange = None
        # Dirty flags for expensive computations during animation
        self._spectrum_needs_update = False
        self._spatial_needs_update = False

        # Cache for persistent plot items (avoids clear+replot cycle)
        self._mxy_plot_items = {}  # key: (pi, fi, component) -> PlotDataItem
        self._mz_plot_items = {}  # key: (pi, fi) -> PlotDataItem
        self._signal_plot_items = {}  # key: (pi, fi, component) -> PlotDataItem
        self._plot_items_initialized = False

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(6, 6, 6, 18)
        left_layout.setSpacing(8)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(480)
        left_panel.setMinimumWidth(340)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Tissue parameters
        self.tissue_widget = TissueParameterWidget()
        left_layout.addWidget(self.tissue_widget)

        # RF Pulse designers
        # 1. Compact panel view
        self.rf_designer_panel = RFPulseDesigner(compact=True)
        left_layout.addWidget(self.rf_designer_panel)

        # 2. Full view for main tab (created here, added to tab widget later)
        self.rf_designer_tab = RFPulseDesigner(compact=False)
        self.rf_designer = self.rf_designer_tab  # Alias for compatibility

        # Connect Panel and Tab designers for synchronization
        self.rf_designer_panel.parameters_changed.connect(
            self.rf_designer_tab.set_state
        )
        self.rf_designer_tab.parameters_changed.connect(
            self.rf_designer_panel.set_state
        )

        # Sync initial state
        self.rf_designer_panel.set_state(self.rf_designer_tab.get_state())

        # Sequence designer
        self.sequence_designer = SequenceDesigner()
        left_layout.addWidget(self.sequence_designer)
        self.sequence_designer.parent_gui = self
        self.rf_designer.pulse_changed.connect(self.sequence_designer.set_custom_pulse)
        self.rf_designer.pulse_changed.connect(
            lambda _: self._auto_update_ssfp_amplitude()
        )
        self.sequence_designer.set_custom_pulse(self.rf_designer.get_pulse())
        # Connect sequence type changes to preset loader
        self.sequence_designer.sequence_type.currentTextChanged.connect(
            self._load_sequence_presets
        )
        self.sequence_designer.sequence_type.currentTextChanged.connect(
            self._update_tab_highlights
        )
        self.sequence_designer.sequence_type.currentTextChanged.connect(
            lambda _: self._auto_update_ssfp_amplitude()
        )
        # self.sequence_designer.ssfp_dur.valueChanged.connect(...) # Removed
        self.rf_designer.flip_angle.valueChanged.connect(
            lambda _: self._auto_update_ssfp_amplitude()
        )
        # Link 3D viewer playhead to sequence diagram
        self.sequence_designer.update_diagram()

        # Simulation controls
        control_group = QGroupBox("Simulation Controls")
        control_layout = QVBoxLayout()

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("mode_combo")
        self.mode_combo.addItems(["Endpoint", "Time-resolved"])
        # Default to time-resolved so users see waveforms/animation without changing anything
        self.mode_combo.setCurrentText("Time-resolved")
        mode_layout.addWidget(self.mode_combo)
        control_layout.addLayout(mode_layout)

        # Positions
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Positions:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setObjectName("pos_spin")
        self.pos_spin.setRange(1, 1100)
        self.pos_spin.setValue(1)
        pos_layout.addWidget(self.pos_spin)
        pos_layout.addWidget(QLabel("Range (cm):"))
        self.pos_range = QDoubleSpinBox()
        self.pos_range.setObjectName("pos_range")
        self.pos_range.setRange(0.01, 9999.0)
        self.pos_range.setValue(2.0)
        self.pos_range.setSingleStep(1.0)
        pos_layout.addWidget(self.pos_range)
        control_layout.addLayout(pos_layout)

        # Frequencies
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequencies:"))
        self.freq_spin = QSpinBox()
        self.freq_spin.setObjectName("freq_spin")
        self.freq_spin.setRange(1, 10000)
        self.freq_spin.setValue(31)
        freq_layout.addWidget(self.freq_spin)
        freq_layout.addWidget(QLabel("Range (Hz):"))
        self.freq_range = QDoubleSpinBox()
        self.freq_range.setObjectName("freq_range")
        # Avoid zero-span (forces unique frequencies)
        self.freq_range.setRange(0.01, 1e4)
        self.freq_range.setValue(100.0)
        freq_layout.addWidget(self.freq_range)
        control_layout.addLayout(freq_layout)
        # Frequency helper text
        self.freq_label = QLabel("Frequencies: [0]")
        self.freq_label.setWordWrap(True)
        control_layout.addWidget(self.freq_label)

        # Time resolution control
        time_res_layout = QHBoxLayout()
        time_res_layout.addWidget(QLabel("Time step (us):"))
        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setObjectName("time_step_spin")
        self.time_step_spin.setRange(0.1, 5000)
        self.time_step_spin.setValue(10.0)
        self.time_step_spin.setDecimals(2)
        self.time_step_spin.setSingleStep(0.1)
        self.time_step_spin.valueChanged.connect(self._update_time_step)
        time_res_layout.addWidget(self.time_step_spin)
        control_layout.addLayout(time_res_layout)

        # Extra post-sequence simulation time
        tail_layout = QHBoxLayout()
        tail_layout.addWidget(QLabel("Extra tail (ms):"))
        self.extra_tail_spin = QDoubleSpinBox()
        self.extra_tail_spin.setObjectName("extra_tail_spin")
        self.extra_tail_spin.setRange(0.0, 1e6)
        self.extra_tail_spin.setValue(5.0)
        self.extra_tail_spin.setDecimals(3)
        self.extra_tail_spin.setSingleStep(1.0)
        tail_layout.addWidget(self.extra_tail_spin)
        control_layout.addLayout(tail_layout)

        # Max traces control for performance
        max_traces_layout = QHBoxLayout()
        max_traces_layout.addWidget(QLabel("Max plot traces:"))
        self.max_traces_spin = QSpinBox()
        self.max_traces_spin.setRange(1, 500)
        self.max_traces_spin.setValue(50)
        self.max_traces_spin.setSingleStep(5)
        self.max_traces_spin.setToolTip(
            "Maximum number of individual traces to plot (for performance)"
        )
        max_traces_layout.addWidget(self.max_traces_spin)
        control_layout.addLayout(max_traces_layout)

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        left_layout.addStretch()

        # Make the left panel scrollable so controls remain reachable on smaller screens
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_scroll.setMinimumHeight(0)
        left_scroll.setWidget(left_panel)
        # Ensure the scroll area knows the full height so bottom controls are reachable
        left_panel.adjustSize()
        left_panel.setMinimumHeight(left_panel.sizeHint().height() + 32)

        left_container = QWidget()
        left_container_layout = QVBoxLayout()
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(6)
        left_container_layout.addWidget(left_scroll)
        left_container_layout.setStretch(0, 1)
        left_container.setLayout(left_container_layout)
        left_container.setMinimumWidth(left_panel.minimumWidth())
        left_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Build run bars early so references are available if needed
        self._build_status_run_bar()
        self._build_toolbar_run_bar()

        # Alias the primary controls to the status bar versions for compatibility
        self.simulate_button = self.status_run_button
        self.progress_bar = self.status_progress
        self.cancel_button = self.status_cancel_button
        self.preview_checkbox = self.status_preview_checkbox

        # Right panel - Visualization + log
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Shared heatmap colormap selector for all tabs
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Heatmap colormap:"))
        self.heatmap_colormap = QComboBox()
        self.heatmap_colormap.addItems(
            ["viridis", "plasma", "magma", "cividis", "inferno", "gray"]
        )
        self.heatmap_colormap.setCurrentText("viridis")
        self.heatmap_colormap.currentTextChanged.connect(self._apply_heatmap_colormap)
        colormap_layout.addWidget(self.heatmap_colormap)

        # Universal time control - controls all time-resolved views
        self.time_control = UniversalTimeControl()
        self.time_control.setEnabled(False)
        colormap_layout.addWidget(self.time_control, 1)
        right_layout.addLayout(colormap_layout)

        # Tab widget for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("main_tabs")

        # Magnetization plots
        mag_widget = QWidget()
        mag_layout = QVBoxLayout()
        mag_widget.setLayout(mag_layout)

        # Add export header
        mag_header = QHBoxLayout()
        mag_header.addWidget(QLabel("Magnetization Evolution"))
        mag_header.addStretch()

        mag_export_btn = QPushButton("Export Results")
        mag_export_btn.clicked.connect(self.export_results)
        mag_header.addWidget(mag_export_btn)
        mag_layout.addLayout(mag_header)

        # Magnetization view filter controls (align with 3D view options)
        mag_view_layout = QHBoxLayout()

        # Add plot type selector (Line vs Heatmap) - default to Heatmap
        mag_view_layout.addWidget(QLabel("Plot type:"))
        self.mag_plot_type = QComboBox()
        self.mag_plot_type.setObjectName("mag_plot_type")
        self.mag_plot_type.addItems(["Heatmap", "Line"])
        self.mag_plot_type.currentTextChanged.connect(
            lambda _: self._refresh_mag_plots()
        )
        mag_view_layout.addWidget(self.mag_plot_type)

        # Add component selector for heatmap
        mag_view_layout.addWidget(QLabel("Component:"))
        self.mag_component = QComboBox()
        self.mag_component.setObjectName("mag_component")
        self.mag_component.addItems(
            ["Magnitude", "Real (Mx/Re)", "Imaginary (My/Im)", "Phase", "Mz"]
        )
        self.mag_component.currentTextChanged.connect(
            lambda _: self._refresh_mag_plots()
        )
        mag_view_layout.addWidget(self.mag_component)

        mag_view_layout.addWidget(QLabel("View mode:"))
        self.mag_view_mode = QComboBox()
        self.mag_view_mode.setObjectName("mag_view_mode")
        self.mag_view_mode.addItems(
            ["All positions x freqs", "Positions @ freq", "Freqs @ position"]
        )
        self.mag_view_mode.currentTextChanged.connect(
            lambda _: self._refresh_mag_plots()
        )
        mag_view_layout.addWidget(self.mag_view_mode)
        self.mag_view_selector_label = QLabel("All spins")
        mag_view_layout.addWidget(self.mag_view_selector_label)
        self.mag_view_selector = QSlider(Qt.Horizontal)
        self.mag_view_selector.setRange(0, 0)
        self.mag_view_selector.setValue(0)
        self.mag_view_selector.valueChanged.connect(lambda _: self._refresh_mag_plots())
        mag_view_layout.addWidget(self.mag_view_selector)
        mag_layout.addLayout(mag_view_layout)

        # Create both line plot and heatmap widgets, but only show one at a time
        self.mxy_plot = pg.PlotWidget()
        self.mxy_plot.setLabel("left", "Mx / My")
        self.mxy_plot.setLabel("bottom", "Time", "ms")
        self.mxy_plot.setDownsampling(mode="peak")
        self.mxy_plot.setClipToView(True)
        self.mxy_plot.hide()  # Hide by default (heatmap is default)

        self.mz_plot = pg.PlotWidget()
        self.mz_plot.setLabel("left", "Mz")
        self.mz_plot.setLabel("bottom", "Time", "ms")
        self.mz_plot.setDownsampling(mode="peak")
        self.mz_plot.setClipToView(True)
        self.mz_plot.hide()  # Hide by default (heatmap is default)

        # Create heatmap widgets using GraphicsLayoutWidget for proper colorbar alignment
        self.mxy_heatmap_layout = pg.GraphicsLayoutWidget()
        self.mxy_heatmap = self.mxy_heatmap_layout.addPlot(row=0, col=0)
        self.mxy_heatmap.setLabel("left", "Spin Index")
        self.mxy_heatmap.setLabel("bottom", "Time", "ms")
        self.mxy_heatmap_item = pg.ImageItem()
        self.mxy_heatmap.addItem(self.mxy_heatmap_item)
        self.mxy_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.mxy_heatmap_layout.addItem(self.mxy_heatmap_colorbar, row=0, col=1)
        self.mxy_heatmap_colorbar.setImageItem(self.mxy_heatmap_item)
        # Show by default (heatmap is default view)

        self.mz_heatmap_layout = pg.GraphicsLayoutWidget()
        self.mz_heatmap = self.mz_heatmap_layout.addPlot(row=0, col=0)
        self.mz_heatmap.setLabel("left", "Spin Index")
        self.mz_heatmap.setLabel("bottom", "Time", "ms")
        self.mz_heatmap_item = pg.ImageItem()
        self.mz_heatmap.addItem(self.mz_heatmap_item)
        self.mz_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.mz_heatmap_layout.addItem(self.mz_heatmap_colorbar, row=0, col=1)
        self.mz_heatmap_colorbar.setImageItem(self.mz_heatmap_item)
        # Show by default (heatmap is default view)

        # Allow resizing between stacked plots so lower plots stay visible
        mag_splitter = QSplitter(Qt.Vertical)
        mag_splitter.addWidget(self.mxy_plot)
        mag_splitter.addWidget(self.mz_plot)
        mag_splitter.addWidget(self.mxy_heatmap_layout)
        mag_splitter.addWidget(self.mz_heatmap_layout)
        mag_splitter.setStretchFactor(0, 1)
        mag_splitter.setStretchFactor(1, 1)
        mag_splitter.setStretchFactor(2, 1)
        mag_splitter.setStretchFactor(3, 1)
        mag_layout.addWidget(mag_splitter)

        # Disable autorange so manual ranges stick
        for plt in (self.mxy_plot, self.mz_plot, self.mxy_heatmap, self.mz_heatmap):
            plt.getViewBox().disableAutoRange()

        self.tab_widget.addTab(self._wrap_in_scroll_area(mag_widget), "Magnetization")

        # 3D visualization
        self.mag_3d = MagnetizationViewer()
        self.mag_3d.playhead_line = self.sequence_designer.playhead_line
        self.mag_3d.set_export_directory_provider(self._get_export_directory)
        self.mag_3d.position_changed.connect(
            lambda idx: self._set_animation_index_from_slider(idx, reset_anchor=True)
        )
        self.mag_3d.view_filter_changed.connect(lambda: self._refresh_vector_view())
        # Disable selector until data is available
        self.mag_3d.set_selector_limits(1, 1, disable=True)
        # Show controls so track/mean toggles are available
        if hasattr(self.mag_3d, "control_container"):
            self.mag_3d.control_container.setVisible(True)
        self.tab_widget.addTab(self._wrap_in_scroll_area(self.mag_3d), "3D Vector")

        # Signal plot
        signal_widget = QWidget()
        signal_layout = QVBoxLayout()
        signal_widget.setLayout(signal_layout)

        # Add export header
        signal_header = QHBoxLayout()
        signal_header.addWidget(QLabel("Signal Evolution"))
        signal_header.addStretch()

        signal_export_btn = QPushButton("Export Results")
        signal_export_btn.clicked.connect(self.export_results)
        signal_header.addWidget(signal_export_btn)
        signal_layout.addLayout(signal_header)

        # Add signal view controls - default to Heatmap
        signal_view_layout = QHBoxLayout()
        signal_view_layout.addWidget(QLabel("Plot type:"))
        self.signal_plot_type = QComboBox()
        self.signal_plot_type.setObjectName("signal_plot_type")
        self.signal_plot_type.addItems(["Heatmap", "Line"])
        self.signal_plot_type.currentTextChanged.connect(
            lambda _: self._refresh_signal_plots()
        )
        signal_view_layout.addWidget(self.signal_plot_type)

        # Add component selector for signal heatmap
        signal_view_layout.addWidget(QLabel("Component:"))
        self.signal_component = QComboBox()
        self.signal_component.setObjectName("signal_component")
        self.signal_component.addItems(["Magnitude", "Real", "Imaginary", "Phase"])
        self.signal_component.currentTextChanged.connect(
            lambda _: self._refresh_signal_plots()
        )
        signal_view_layout.addWidget(self.signal_component)

        signal_view_layout.addWidget(QLabel("View mode:"))
        self.signal_view_mode = QComboBox()
        self.signal_view_mode.setObjectName("signal_view_mode")
        self.signal_view_mode.addItems(
            ["All positions x freqs", "Positions @ freq", "Freqs @ position"]
        )
        self.signal_view_mode.currentTextChanged.connect(
            lambda _: self._refresh_signal_plots()
        )
        signal_view_layout.addWidget(self.signal_view_mode)
        self.signal_view_selector_label = QLabel("All spins")
        signal_view_layout.addWidget(self.signal_view_selector_label)
        self.signal_view_selector = QSlider(Qt.Horizontal)
        self.signal_view_selector.setRange(0, 0)
        self.signal_view_selector.setValue(0)
        self.signal_view_selector.valueChanged.connect(
            lambda _: self._refresh_signal_plots()
        )
        signal_view_layout.addWidget(self.signal_view_selector)
        signal_layout.addLayout(signal_view_layout)

        # Create line plot
        self.signal_plot = pg.PlotWidget()
        self.signal_plot.setLabel("left", "Signal")
        self.signal_plot.setLabel("bottom", "Time", "ms")
        self.signal_plot.setDownsampling(mode="peak")
        self.signal_plot.setClipToView(True)
        self.signal_plot.enableAutoRange(x=False, y=False)
        self.signal_plot.hide()  # Hide by default (heatmap is default)

        # Create heatmap using GraphicsLayoutWidget for proper colorbar alignment
        self.signal_heatmap_layout = pg.GraphicsLayoutWidget()
        self.signal_heatmap = self.signal_heatmap_layout.addPlot(row=0, col=0)
        self.signal_heatmap.setLabel("left", "Spin Index")
        self.signal_heatmap.setLabel("bottom", "Time", "ms")
        self.signal_heatmap_item = pg.ImageItem()
        self.signal_heatmap.addItem(self.signal_heatmap_item)
        self.signal_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.signal_heatmap_layout.addItem(self.signal_heatmap_colorbar, row=0, col=1)
        self.signal_heatmap_colorbar.setImageItem(self.signal_heatmap_item)
        # Show by default (heatmap is default view)
        self.signal_heatmap.getViewBox().disableAutoRange()

        signal_splitter = QSplitter(Qt.Vertical)
        signal_splitter.addWidget(self.signal_plot)
        signal_splitter.addWidget(self.signal_heatmap_layout)
        signal_layout.addWidget(signal_splitter)

        self.tab_widget.addTab(self._wrap_in_scroll_area(signal_widget), "Signal")

        # Time cursor lines removed for performance

        # Frequency spectrum
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel("left", "Magnitude")
        self.spectrum_plot.setLabel("bottom", "Frequency", "Hz")
        self.spectrum_plot.setDownsampling(mode="peak")
        self.spectrum_plot.setClipToView(True)

        # 3D Spectrum Plot (Hidden by default)
        self.spectrum_plot_3d = gl.GLViewWidget()
        self.spectrum_plot_3d.opts["distance"] = 40
        self.spectrum_plot_3d.hide()

        spectrum_container = QWidget()
        spectrum_layout = QVBoxLayout()

        # Add export header for spectrum
        spectrum_header = QHBoxLayout()
        spectrum_header.addWidget(QLabel("Frequency Spectrum"))
        spectrum_header.addStretch()

        spectrum_export_btn = QPushButton("Export Results")
        spectrum_export_btn.clicked.connect(self.export_results)
        spectrum_header.addWidget(spectrum_export_btn)
        spectrum_layout.addLayout(spectrum_header)

        spectrum_controls = QHBoxLayout()

        # 3D View Toggle
        self.spectrum_3d_toggle = QCheckBox("3D View")
        self.spectrum_3d_toggle.toggled.connect(self._toggle_spectrum_3d_mode)
        spectrum_controls.addWidget(self.spectrum_3d_toggle)

        spectrum_controls.addWidget(QLabel("Plot type:"))
        self.spectrum_plot_type = QComboBox()
        self.spectrum_plot_type.setObjectName("spectrum_plot_type")
        self.spectrum_plot_type.addItems(["Line", "Heatmap"])
        self.spectrum_plot_type.setCurrentText("Line")
        self.spectrum_plot_type.currentTextChanged.connect(
            lambda _: self._refresh_spectrum(
                time_idx=(
                    self._current_playback_index()
                    if hasattr(self, "_current_playback_index")
                    else None
                )
            )
        )
        spectrum_controls.addWidget(self.spectrum_plot_type)

        spectrum_controls.addWidget(QLabel("Spectrum view:"))
        self.spectrum_view_mode = QComboBox()
        self.spectrum_view_mode.setObjectName("spectrum_view_mode")
        self.spectrum_view_mode.addItems(["Individual position", "Mean over positions"])
        self.spectrum_view_mode.currentIndexChanged.connect(
            lambda _: (
                self._refresh_spectrum(time_idx=self._current_playback_index())
                if hasattr(self, "_current_playback_index")
                else None
            )
        )
        spectrum_controls.addWidget(self.spectrum_view_mode)
        self.spectrum_pos_slider = QSlider(Qt.Horizontal)
        self.spectrum_pos_slider.setObjectName("spectrum_pos_slider")
        self.spectrum_pos_slider.setRange(0, 0)
        self.spectrum_pos_slider.valueChanged.connect(
            lambda _: (
                self._refresh_spectrum(time_idx=self._current_playback_index())
                if hasattr(self, "_current_playback_index")
                else None
            )
        )
        self.spectrum_pos_label = QLabel("Pos: 0.00 cm")
        spectrum_controls.addWidget(self.spectrum_pos_label)
        spectrum_controls.addWidget(self.spectrum_pos_slider)
        spectrum_layout.addLayout(spectrum_controls)

        # Component selector for spectrum
        spectrum_comp_layout = QHBoxLayout()
        self.spectrum_component_label = QLabel("Component:")
        spectrum_comp_layout.addWidget(self.spectrum_component_label)
        self.spectrum_component_combo = CheckableComboBox()
        self.spectrum_component_combo.add_items(
            ["Magnitude", "Phase", "Phase (unwrapped)", "Real", "Imaginary", "Mz"]
        )
        self.spectrum_component_combo.set_selected_items(["Magnitude"])
        self.spectrum_component_combo.selection_changed.connect(
            lambda: (
                self._refresh_spectrum(time_idx=self._current_playback_index())
                if hasattr(self, "_current_playback_index")
                else None
            )
        )
        spectrum_comp_layout.addWidget(self.spectrum_component_combo)

        # Heatmap specific mode selector
        self.spectrum_heatmap_mode_label = QLabel("Heatmap mode:")
        self.spectrum_heatmap_mode = QComboBox()
        self.spectrum_heatmap_mode.setObjectName("spectrum_heatmap_mode")
        self.spectrum_heatmap_mode.addItems(
            ["Spin vs Time (Evolution)", "Spin vs Frequency (FFT)"]
        )
        self.spectrum_heatmap_mode.currentTextChanged.connect(
            lambda _: (
                self._refresh_spectrum(time_idx=self._current_playback_index())
                if hasattr(self, "_current_playback_index")
                else None
            )
        )
        self.spectrum_heatmap_mode_label.setVisible(False)
        self.spectrum_heatmap_mode.setVisible(False)
        spectrum_comp_layout.addWidget(self.spectrum_heatmap_mode_label)
        spectrum_comp_layout.addWidget(self.spectrum_heatmap_mode)

        spectrum_layout.addLayout(spectrum_comp_layout)

        spectrum_layout.addWidget(self.spectrum_plot)
        spectrum_layout.addWidget(self.spectrum_plot_3d)
        # Spectrum heatmap using GraphicsLayoutWidget
        self.spectrum_heatmap_layout = pg.GraphicsLayoutWidget()
        self.spectrum_heatmap = self.spectrum_heatmap_layout.addPlot(row=0, col=0)
        self.spectrum_heatmap.setLabel("left", "Spin Index")
        self.spectrum_heatmap.setLabel("bottom", "Frequency", "Hz")
        self.spectrum_heatmap_item = pg.ImageItem()
        self.spectrum_heatmap.addItem(self.spectrum_heatmap_item)
        self.spectrum_heatmap_colorbar = pg.ColorBarItem(
            values=(0, 1), interactive=False
        )
        self.spectrum_heatmap_layout.addItem(
            self.spectrum_heatmap_colorbar, row=0, col=1
        )
        self.spectrum_heatmap_colorbar.setImageItem(self.spectrum_heatmap_item)
        spectrum_layout.addWidget(self.spectrum_heatmap_layout)
        spectrum_container.setLayout(spectrum_layout)
        self.tab_widget.addTab(
            self._wrap_in_scroll_area(spectrum_container), "Spectrum"
        )

        # Spatial profile plot (Mxy and Mz vs position at selected time)
        # Note: Time control is now unified in the universal control below the tabs
        spatial_container = QWidget()
        spatial_layout = QVBoxLayout()

        # Add export header for spatial
        spatial_header = QHBoxLayout()
        spatial_header.addWidget(QLabel("Spatial Profile"))
        spatial_header.addStretch()

        spatial_export_btn = QPushButton("Export Results")
        spatial_export_btn.clicked.connect(self.export_results)
        spatial_header.addWidget(spatial_export_btn)
        spatial_layout.addLayout(spatial_header)

        # Display controls
        self.mean_only_checkbox = QCheckBox("Mean only (Mag/Signal/3D)")
        self.mean_only_checkbox.setObjectName("mean_only_checkbox")
        self.mean_only_checkbox.stateChanged.connect(
            lambda _: self.update_plots(self.last_result) if self.last_result else None
        )
        spatial_layout.addWidget(self.mean_only_checkbox)

        spatial_controls = QHBoxLayout()
        spatial_controls.addWidget(QLabel("Plot type:"))
        self.spatial_plot_type = QComboBox()
        self.spatial_plot_type.setObjectName("spatial_plot_type")
        self.spatial_plot_type.addItems(["Line", "Heatmap"])

        spatial_controls.addWidget(self.spatial_plot_type)
        spatial_controls.addWidget(QLabel("Heatmap mode:"))
        self.spatial_heatmap_mode = QComboBox()
        self.spatial_heatmap_mode.setObjectName("spatial_heatmap_mode")
        self.spatial_heatmap_mode.addItems(
            ["Position vs Frequency", "Position vs Time"]
        )
        self.spatial_heatmap_mode.currentTextChanged.connect(
            lambda _: self.update_spatial_plot_from_last_result()
        )
        spatial_controls.addWidget(self.spatial_heatmap_mode)

        self.spatial_plot_type.currentTextChanged.connect(
            lambda text: self._update_spatial_controls_visibility(text)
        )
        self.spatial_plot_type.currentTextChanged.connect(
            lambda _: self.update_spatial_plot_from_last_result()
        )

        spatial_controls.addWidget(QLabel("View:"))
        self.spatial_mode = QComboBox()  # Renamed from spatial_mode to avoid confusion
        self.spatial_mode.setObjectName("spatial_mode")
        self.spatial_mode.addItems(["Individual freq", "Mean over freqs"])
        self.spatial_mode.currentIndexChanged.connect(
            self.update_spatial_plot_from_last_result
        )
        spatial_controls.addWidget(self.spatial_mode)
        self.spatial_freq_slider = QSlider(Qt.Horizontal)
        self.spatial_freq_slider.setObjectName("spatial_freq_slider")
        self.spatial_freq_slider.setRange(0, 0)
        self.spatial_freq_slider.valueChanged.connect(
            lambda: self.update_spatial_plot_from_last_result()
        )
        self.spatial_freq_label = QLabel("Freq: 0.0 Hz")
        spatial_controls.addWidget(self.spatial_freq_label)
        spatial_controls.addWidget(self.spatial_freq_slider)
        spatial_layout.addLayout(spatial_controls)

        # Toggle for colored position/frequency markers
        self.spatial_markers_checkbox = QCheckBox(
            "Show colored position/frequency markers"
        )
        self.spatial_markers_checkbox.setObjectName("spatial_markers_checkbox")
        self.spatial_markers_checkbox.setChecked(False)
        self.spatial_markers_checkbox.setToolTip(
            "Display vertical lines at each position/frequency with 3D-view colors"
        )
        self.spatial_markers_checkbox.toggled.connect(
            lambda _: self.update_spatial_plot_from_last_result()
        )
        spatial_layout.addWidget(self.spatial_markers_checkbox)

        # Component selector for spatial plot
        spatial_comp_layout = QHBoxLayout()
        self.spatial_component_label = QLabel("Component:")
        spatial_comp_layout.addWidget(self.spatial_component_label)
        self.spatial_component_combo = CheckableComboBox()
        self.spatial_component_combo.add_items(
            ["Magnitude", "Phase", "Phase (unwrapped)", "Real", "Imaginary"]
        )
        self.spatial_component_combo.set_selected_items(
            ["Magnitude", "Real", "Imaginary"]
        )
        self.spatial_component_combo.selection_changed.connect(
            lambda: self.update_spatial_plot_from_last_result()
        )
        spatial_comp_layout.addWidget(self.spatial_component_combo)
        spatial_layout.addLayout(spatial_comp_layout)

        # Mxy and Mz plots side by side
        spatial_plots_layout = QHBoxLayout()

        # Mxy vs position plot
        self.spatial_mxy_plot = pg.PlotWidget()
        self.spatial_mxy_plot.setLabel("left", "Mxy (transverse)")
        self.spatial_mxy_plot.setLabel("bottom", "Position", "m")
        self.spatial_mxy_plot.enableAutoRange(x=False, y=False)
        self.spatial_mxy_plot.setDownsampling(mode="peak")
        self.spatial_mxy_plot.setClipToView(True)
        # Slice thickness guides (added once and reused)
        slice_pen = pg.mkPen((180, 180, 180), style=Qt.DashLine)
        self.spatial_slice_lines = {
            "mxy": [
                pg.InfiniteLine(angle=90, pen=slice_pen),
                pg.InfiniteLine(angle=90, pen=slice_pen),
            ],
            "mz": [
                pg.InfiniteLine(angle=90, pen=slice_pen),
                pg.InfiniteLine(angle=90, pen=slice_pen),
            ],
        }
        for ln in self.spatial_slice_lines["mxy"]:
            self.spatial_mxy_plot.addItem(ln)
        spatial_plots_layout.addWidget(self.spatial_mxy_plot)

        # Mz vs position plot
        self.spatial_mz_plot = pg.PlotWidget()
        self.spatial_mz_plot.setLabel("left", "Mz (longitudinal)")
        self.spatial_mz_plot.setLabel("bottom", "Position", "m")
        self.spatial_mz_plot.enableAutoRange(x=False, y=False)
        self.spatial_mz_plot.setDownsampling(mode="peak")
        self.spatial_mz_plot.setClipToView(True)
        for ln in self.spatial_slice_lines["mz"]:
            self.spatial_mz_plot.addItem(ln)
        spatial_plots_layout.addWidget(self.spatial_mz_plot)

        spatial_layout.addLayout(spatial_plots_layout)

        # Heatmap container (hidden by default)
        self.spatial_heatmap_container = QWidget()
        spatial_heatmap_splitter = QSplitter(Qt.Vertical)
        spatial_heatmap_splitter.setContentsMargins(0, 0, 0, 0)

        self.spatial_heatmap_mxy_layout = pg.GraphicsLayoutWidget()
        self.spatial_heatmap_mxy = self.spatial_heatmap_mxy_layout.addPlot(row=0, col=0)
        self.spatial_heatmap_mxy.setLabel("bottom", "Position", "m")
        self.spatial_heatmap_mxy.setLabel("left", "Frequency", "Hz")
        self.spatial_heatmap_mxy.setTitle("Mxy magnitude (|Mxy|)")
        self.spatial_heatmap_mxy_item = pg.ImageItem()
        self.spatial_heatmap_mxy.addItem(self.spatial_heatmap_mxy_item)
        self.spatial_heatmap_mxy_colorbar = pg.ColorBarItem(
            values=(0, 1), interactive=False
        )
        self.spatial_heatmap_mxy_layout.addItem(
            self.spatial_heatmap_mxy_colorbar, row=0, col=1
        )
        self.spatial_heatmap_mxy_colorbar.setImageItem(self.spatial_heatmap_mxy_item)
        spatial_heatmap_splitter.addWidget(self.spatial_heatmap_mxy_layout)

        self.spatial_heatmap_mz_layout = pg.GraphicsLayoutWidget()
        self.spatial_heatmap_mz = self.spatial_heatmap_mz_layout.addPlot(row=0, col=0)
        self.spatial_heatmap_mz.setLabel("bottom", "Position", "m")
        self.spatial_heatmap_mz.setLabel("left", "Frequency", "Hz")
        self.spatial_heatmap_mz.setTitle("Mz")
        self.spatial_heatmap_mz_item = pg.ImageItem()
        self.spatial_heatmap_mz.addItem(self.spatial_heatmap_mz_item)
        self.spatial_heatmap_mz_colorbar = pg.ColorBarItem(
            values=(0, 1), interactive=False
        )
        self.spatial_heatmap_mz_layout.addItem(
            self.spatial_heatmap_mz_colorbar, row=0, col=1
        )
        self.spatial_heatmap_mz_colorbar.setImageItem(self.spatial_heatmap_mz_item)
        spatial_heatmap_splitter.addWidget(self.spatial_heatmap_mz_layout)

        self.spatial_heatmap_container.setLayout(QVBoxLayout())
        self.spatial_heatmap_container.layout().addWidget(spatial_heatmap_splitter)
        self.spatial_heatmap_container.hide()
        spatial_layout.addWidget(self.spatial_heatmap_container)

        spatial_container.setLayout(spatial_layout)
        self.tab_widget.addTab(self._wrap_in_scroll_area(spatial_container), "Spatial")

        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(self)
            self.tab_widget.addTab(
                self._wrap_in_scroll_area(self.phantom_widget), "🔬 Phantom"
            )
        else:
            self.phantom_widget = None

        # === K-SPACE TAB (Signal-based simulation) ===
        if KSPACE_AVAILABLE:

            def get_phantom_for_kspace():
                """Get current phantom from PhantomWidget."""
                if self.phantom_widget is not None:
                    if hasattr(self.phantom_widget, "creator"):
                        return self.phantom_widget.creator.current_phantom
                return None

            def get_magnetization_for_kspace():
                """Get magnetization from last Bloch simulation."""
                if self.last_result is not None:
                    return {
                        "mx": self.last_result.get("mx"),
                        "my": self.last_result.get("my"),
                        "mz": self.last_result.get("mz"),
                    }
                return None

            self.kspace_widget = KSpaceWidget(
                self,
                get_phantom_callback=get_phantom_for_kspace,
                get_magnetization_callback=get_magnetization_for_kspace,
            )
            self.tab_widget.addTab(
                self._wrap_in_scroll_area(self.kspace_widget), "📡 K-Space"
            )
        else:
            self.kspace_widget = None

        # === PARAMETER SWEEP TAB ===
        self.param_sweep_widget = ParameterSweepWidget(self)
        self.tab_widget.addTab(
            self._wrap_in_scroll_area(self.param_sweep_widget), "📊 Parameter Sweep"
        )

        # === RF PULSE DESIGN TAB ===
        self.tab_widget.addTab(
            self._wrap_in_scroll_area(self.rf_designer_tab), "RF Design"
        )

        # === SLICE EXPLORER TAB ===
        self.slice_explorer = SliceSelectionExplorer(self)
        self.tab_widget.addTab(
            self._wrap_in_scroll_area(self.slice_explorer), "Slice Explorer"
        )

        # Wire up signals for the panel instance (tab instance is wired via self.rf_designer alias earlier)
        self.rf_designer_panel.pulse_changed.connect(
            self.sequence_designer.set_custom_pulse
        )
        self.rf_designer_panel.pulse_changed.connect(
            lambda _: self._auto_update_ssfp_amplitude()
        )

        # Log console lives in its own tab to save vertical space
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(6, 6, 6, 6)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        log_layout.addWidget(self.log_widget)
        log_tab.setLayout(log_layout)
        self.tab_widget.addTab(self._wrap_in_scroll_area(log_tab), "Log")

        # Add time cursor lines to spatial plots for synchronization
        self.spatial_mxy_time_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("y", width=2)
        )
        self.spatial_mxy_time_line.hide()
        self.spatial_mxy_plot.addItem(self.spatial_mxy_time_line)

        self.spatial_mz_time_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("y", width=2)
        )
        self.spatial_mz_time_line.hide()
        self.spatial_mz_plot.addItem(self.spatial_mz_time_line)

        # Share spatial time lines with the 3D viewer for synchronized scrubbing
        self.mag_3d.spatial_mxy_time_line = self.spatial_mxy_time_line
        self.mag_3d.spatial_mz_time_line = self.spatial_mz_time_line

        right_layout.addWidget(self.tab_widget, 1)

        # Apply initial colormap selection now that heatmaps are constructed
        default_cmap = "viridis"
        if hasattr(self, "heatmap_colormap") and self.heatmap_colormap is not None:
            default_cmap = self.heatmap_colormap.currentText()
        self._apply_heatmap_colormap(default_cmap)

        # Default to the 3D Vector tab so users immediately see the vector view
        if hasattr(self, "mag_3d"):
            # Find the tab that contains the 3D viewer (or its scroll wrapper)
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "3D Vector":
                    self.tab_widget.setCurrentIndex(i)
                    break

        # Connect tab change to optimize rendering
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 1000])
        main_layout.addWidget(splitter)

        # Push initial time-step into designers
        self._update_time_step(self.time_step_spin.value())

        # Menu bar
        self.create_menu()

        # Status bar
        self.statusBar().showMessage("Ready")

        # Connect universal time control to all views
        self._setup_time_synchronization()

        # Connect view mode controls for synchronization
        self.mag_3d.view_mode_combo.currentTextChanged.connect(self._sync_view_modes)
        self.mag_view_mode.currentTextChanged.connect(self._sync_view_modes)
        self.mag_3d.selector_slider.valueChanged.connect(self._sync_selectors)
        self.mag_view_selector.valueChanged.connect(self._sync_selectors)

        # Initialize spatial controls visibility
        self._update_spatial_controls_visibility(self.spatial_plot_type.currentText())

        # Initialize tab highlights
        self._update_tab_highlights()

    def _update_tab_highlights(self):
        """Update tab colors based on current sequence selection."""
        if not hasattr(self, "tab_widget"):
            return

        seq_type = self.sequence_designer.sequence_type.currentText()
        bar = self.tab_widget.tabBar()

        # Reset all tabs to default first
        for i in range(self.tab_widget.count()):
            bar.setTabTextColor(i, self.palette().color(QPalette.WindowText))

        # Highlight specific tabs based on sequence
        highlight_color = QColor("#0078D7")  # Professional blue

        for i in range(self.tab_widget.count()):
            text = self.tab_widget.tabText(i)
            if seq_type == "Free Induction Decay" and "Spectrum" in text:
                bar.setTabTextColor(i, highlight_color)
            elif seq_type == "Slice Select + Rephase" and "Spatial" in text:
                bar.setTabTextColor(i, highlight_color)
            elif seq_type == "SSFP (Loop)" and "Signal" in text:
                bar.setTabTextColor(i, highlight_color)

    def _wrap_in_scroll_area(self, widget):
        """Wrap a widget in a QScrollArea with no frame."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _update_spatial_controls_visibility(self, plot_type: str):
        """Enable/disable spatial controls based on plot type."""
        is_heatmap = plot_type == "Heatmap"
        # Hide/Disable heatmap mode selector if not in heatmap mode
        if hasattr(self, "spatial_heatmap_mode"):
            self.spatial_heatmap_mode.setEnabled(is_heatmap)

        # Hide markers checkbox in heatmap mode (as it applies to line plots)
        if hasattr(self, "spatial_markers_checkbox"):
            self.spatial_markers_checkbox.setVisible(not is_heatmap)

        # Show/hide component selector (only for line plots)
        if hasattr(self, "spatial_component_combo"):
            self.spatial_component_combo.setVisible(not is_heatmap)
            if hasattr(self, "spatial_component_label"):
                self.spatial_component_label.setVisible(not is_heatmap)

    def _color_for_index(self, idx: int, total: int):
        """Consistent color cycling for multiple frequencies."""
        return pg.intColor(idx, hues=max(total, 1), values=1.0, maxValue=255)

    def _get_trace_indices_to_plot(self, total_traces: int) -> list:
        """
        Get indices of traces to plot, respecting max_traces limit.

        Returns evenly-spaced subset if total exceeds limit, otherwise all indices.
        Always includes first and last trace for boundary visualization.

        Parameters
        ----------
        total_traces : int
            Total number of available traces

        Returns
        -------
        list of int
            Indices to plot
        """
        max_traces = self.max_traces_spin.value()

        if total_traces <= max_traces:
            return list(range(total_traces))

        # Evenly space the indices
        indices = np.linspace(0, total_traces - 1, max_traces, dtype=int)
        return sorted(list(set(indices)))  # Remove duplicates and sort

    def _sync_view_modes(self, text: str):
        """Synchronize the view mode across all relevant tabs."""
        # Prevent recursive signals
        if getattr(self, "_syncing_views", False):
            return
        self._syncing_views = True

        try:
            # Update 3D Viewer
            if self.mag_3d.view_mode_combo.currentText() != text:
                self.mag_3d.view_mode_combo.setCurrentText(text)

            # Update Magnetization Plot
            if self.mag_view_mode.currentText() != text:
                self.mag_view_mode.setCurrentText(text)

            # Update Signal Plot
            if self.signal_view_mode.currentText() != text:
                self.signal_view_mode.setCurrentText(text)

            # Trigger a refresh of the plots with the new mode
            if self.last_result:
                self.update_plots(self.last_result)
                self._refresh_vector_view()

        finally:
            self._syncing_views = False

    def _sync_selectors(self, value: int):
        """Synchronize the view selector sliders across all relevant tabs."""
        if getattr(self, "_syncing_views", False):
            return
        self._syncing_views = True

        try:
            # Update 3D Viewer
            if self.mag_3d.selector_slider.value() != value:
                self.mag_3d.selector_slider.setValue(value)

            # Update Magnetization Plot
            if self.mag_view_selector.value() != value:
                self.mag_view_selector.setValue(value)

            # Update Signal Plot
            if self.signal_view_selector.value() != value:
                self.signal_view_selector.setValue(value)

        finally:
            self._syncing_views = False

    def _safe_clear_plot(self, plot_widget, persistent_items=None):
        """
        Safely clear a plot widget and re-add persistent items.

        This prevents Qt warnings about items being removed from the wrong scene.

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            The plot widget to clear
        persistent_items : list, optional
            List of items to re-add after clearing (e.g., cursor lines, guide lines)
        """
        plot_widget.clear()
        if persistent_items:
            for item in persistent_items:
                if item is not None:
                    # Check if item is actually in the scene before adding
                    # After clear(), items should not be in the plot
                    try:
                        if item.scene() is None:
                            plot_widget.addItem(item)
                    except (AttributeError, RuntimeError):
                        # Item doesn't have scene() method or was deleted, skip it
                        pass

    def _update_or_create_plot_item(
        self, plot_widget, cache_dict, key, x_data, y_data, pen, name=None
    ):
        """Update existing plot item or create new one if needed."""
        if key in cache_dict and cache_dict[key] is not None:
            # Update existing item
            try:
                cache_dict[key].setData(x_data, y_data)
                cache_dict[key].setPen(pen)
                return cache_dict[key]
            except (AttributeError, RuntimeError):
                # Item was deleted or invalid, recreate
                pass

        # Create new item
        item = plot_widget.plot(x_data, y_data, pen=pen, name=name)
        cache_dict[key] = item
        return item

    def _invalidate_plot_caches(self):
        """Clear all cached plot items (e.g., when loading new simulation)."""
        # Remove items from plots
        for cache, plot in [
            (self._mxy_plot_items, self.mxy_plot),
            (self._mz_plot_items, self.mz_plot),
            (self._signal_plot_items, self.signal_plot),
        ]:
            for item in cache.values():
                if item is not None:
                    try:
                        plot.removeItem(item)
                    except (AttributeError, RuntimeError):
                        pass
            cache.clear()
        self._plot_items_initialized = False

    def _reshape_to_tpf(self, arr: np.ndarray, pos_len: int, freq_len: int):
        """
        Ensure array shape is (ntime, npos, nfreq).
        Tries to infer axis order using known pos/freq lengths.
        """
        if arr is None:
            return arr
        if arr.ndim == 2:
            # Heuristics for 2D: assume either (time, freq) or (time, pos)
            if arr.shape[1] == freq_len and pos_len == 1:
                return arr[:, None, :]
            if arr.shape[1] == pos_len and freq_len == 1:
                return arr[:, :, None]
            if arr.shape[0] == freq_len and pos_len == 1:
                return arr.T[:, None, :]
            if arr.shape[0] == pos_len and freq_len == 1:
                return arr[:, :, None]
            return arr
        if arr.ndim != 3:
            return arr
        shape = arr.shape
        # Already correct
        if shape[1] == pos_len and shape[2] == freq_len:
            return arr
        # Try permutations
        for perm in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            if shape[perm[1]] == pos_len and shape[perm[2]] == freq_len:
                return np.transpose(arr, perm)
        # Heuristic: if first axis equals freq_len, assume (freq, time, pos) or (freq, pos, time)
        if shape[0] == freq_len:
            if shape[1] == pos_len:
                return np.transpose(arr, (2, 1, 0))  # (time, pos, freq)
            elif shape[2] == pos_len:
                return np.transpose(arr, (1, 2, 0))  # (time, pos, freq)
        # Heuristic: if first axis equals pos_len, assume (pos, time, freq) or (pos, freq, time)
        if shape[0] == pos_len:
            if shape[1] == freq_len:
                return np.transpose(arr, (2, 0, 1))
            elif shape[2] == freq_len:
                return np.transpose(arr, (1, 0, 2))
        return arr  # fallback

    def _normalize_time_length(self, time: np.ndarray, ntime: int):
        """Match time array length to ntime if simulator returns a different length."""
        if time is None:
            return np.arange(ntime)
        if len(time) == ntime:
            return time
        if len(time) > 1:
            dt = (time[-1] - time[0]) / (len(time) - 1)
        else:
            dt = 1.0
        return np.arange(ntime) * dt

    def _spectrum_fft_len(self, n: int) -> int:
        """Choose an FFT length for smoother spectra."""
        # Next power of two with a minimum to avoid too few bins
        n = max(n, 8)
        n_fft = 1 << (n - 1).bit_length()
        # Apply a mild zero-padding to improve resolution without overkill
        n_fft = min(max(n_fft * 2, 512), 262144)
        return int(n_fft)

    def _playback_to_full_index(self, playback_idx: int) -> int:
        """Map a playback index to the corresponding full-resolution time index."""
        if self.playback_indices is None or len(self.playback_indices) == 0:
            return int(playback_idx)
        playback_idx = int(max(0, min(playback_idx, len(self.playback_indices) - 1)))
        return int(self.playback_indices[playback_idx])

    def _color_tuple(self, pg_color):
        """Convert a pyqtgraph color to an RGBA tuple for OpenGL items."""
        try:
            return pg_color.getRgbF()
        except Exception:
            try:
                return pg.mkColor(pg_color).getRgbF()
            except Exception:
                return (1, 0, 0, 1)

    def _build_playback_indices(self, total_frames: int) -> np.ndarray:
        """Construct (optionally downsampled) playback indices for animation."""
        # Return all indices to ensure every simulated point is accessible.
        # The animation loop handles frame skipping automatically to maintain speed.
        return np.arange(total_frames, dtype=int)

    def _reset_playback_anchor(self, idx: Optional[int] = None):
        """Record the wall-clock anchor for the current playback position."""
        if self.playback_time_ms is None or len(self.playback_time_ms) == 0:
            self._playback_anchor_wall = None
            self._playback_anchor_time_ms = None
            return
        if idx is None:
            idx = getattr(self, "anim_index", 0)
        idx = int(max(0, min(idx, len(self.playback_time_ms) - 1)))
        self._playback_anchor_wall = time.monotonic()
        self._playback_anchor_time_ms = float(self.playback_time_ms[idx])
        self.anim_index = idx
        self._last_render_wall = None
        self._playback_frame_counter = 0

    def _refresh_vector_view(self, mean_only: bool = None, restart: bool = True):
        """Apply the current vector filter (all/pos/freq) to the 3D view."""
        if (
            self.anim_vectors_full is None
            or self.playback_indices is None
            or self.playback_time is None
        ):
            return
        if mean_only is None:
            mean_only = self.mean_only_checkbox.isChecked()

        base_vectors = self.anim_vectors_full
        # Downsample to playback timeline if needed
        if base_vectors.shape[0] != len(self.playback_indices):
            base_vectors = base_vectors[self.playback_indices]

        nframes, npos, nfreq, _ = base_vectors.shape
        mode = self.mag_3d.get_view_mode()
        selector = self.mag_3d.get_selector_index()

        if mean_only or (npos == 1 and nfreq == 1):
            anim = np.mean(base_vectors, axis=(1, 2), keepdims=True)
            colors = [self._color_tuple(pg.mkColor("c"))]
        elif mode == "Positions @ freq":
            fi = min(max(selector, 0), nfreq - 1)
            anim = base_vectors[:, :, fi, :]
            colors = [
                self._color_tuple(self._color_for_index(i, npos)) for i in range(npos)
            ]
        elif mode == "Freqs @ position":
            pi = min(max(selector, 0), npos - 1)
            anim = base_vectors[:, pi, :, :]
            colors = [
                self._color_tuple(self._color_for_index(i, nfreq)) for i in range(nfreq)
            ]
        else:
            anim = base_vectors.reshape(nframes, npos * nfreq, 3)
            total = npos * nfreq
            colors = [
                self._color_tuple(self._color_for_index(i, total)) for i in range(total)
            ]

        self.anim_data = anim
        self.anim_colors = colors
        self.anim_time = self.playback_time

        # Update preview plot with mean vectors
        mean_vectors = np.mean(anim, axis=1)
        self.mag_3d.set_preview_data(
            (
                self.playback_time_ms
                if self.playback_time_ms is not None
                else self.playback_time
            ),
            mean_vectors[:, 0],
            mean_vectors[:, 1],
            mean_vectors[:, 2],
        )
        # Ensure vector count matches filter
        self.mag_3d._ensure_vectors(anim.shape[1], colors=colors)
        if restart:
            self._start_vector_animation()
        else:
            if self.anim_index >= len(anim):
                self.anim_index = 0
            self._set_animation_index_from_slider(self.anim_index)

    def _extend_sequence_with_tail(self, sequence_tuple, tail_ms: float, dt: float):
        """Append a zero-B1/gradient tail after the sequence to continue acquisition."""
        tail_s = max(0.0, tail_ms) * 1e-3
        if tail_s <= 0:
            return sequence_tuple
        b1, gradients, time = sequence_tuple
        b1 = np.asarray(b1, dtype=complex)
        gradients = np.asarray(gradients, dtype=float)
        time = np.asarray(time, dtype=float)
        if time.size == 0:
            return (b1, gradients, time)
        if gradients.ndim == 1:
            gradients = gradients.reshape(-1, 1)
        if gradients.shape[1] < 3:
            gradients = np.pad(
                gradients, ((0, 0), (0, 3 - gradients.shape[1])), mode="constant"
            )
        elif gradients.shape[1] > 3:
            gradients = gradients[:, :3]

        dt_use = max(dt, 1e-9)
        if len(time) > 1:
            diffs = np.diff(time)
            with np.errstate(invalid="ignore"):
                mean_dt = float(np.nanmean(diffs))
            if np.isfinite(mean_dt) and mean_dt > 0:
                dt_use = mean_dt

        n_tail = int(math.ceil(tail_s / dt_use))
        if n_tail <= 0:
            return (b1, gradients, time)

        tail_time = time[-1] + np.arange(1, n_tail + 1) * dt_use
        b1_tail = np.zeros(n_tail, dtype=complex)
        grad_tail = np.zeros((n_tail, gradients.shape[1]), dtype=float)

        b1_ext = np.concatenate([b1, b1_tail])
        gradients_ext = np.vstack([gradients, grad_tail])
        time_ext = np.concatenate([time, tail_time])
        return (b1_ext, gradients_ext, time_ext)

    def log_message(self, message: str):
        """Append a message to the log console."""
        self.log_widget.append(message)
        self.log_widget.moveCursor(self.log_widget.textCursor().End)

    def _update_time_step(self, us_value: float):
        """Propagate desired time resolution (microseconds) to designers."""
        dt_s = max(us_value, 0.1) * 1e-6
        self.rf_designer.set_time_step(dt_s)
        self.sequence_designer.set_time_step(dt_s)
        self.sequence_designer.update_diagram(self.rf_designer.get_pulse())

    def _auto_update_ssfp_amplitude(self):
        """Auto-calculate SSFP pulse amplitude from flip angle, duration, and integration factor."""
        try:
            if self.sequence_designer.sequence_type.currentText() != "SSFP (Loop)":
                return
            # Get params
            flip_deg = self.rf_designer.flip_angle.value()
            flip_rad = np.deg2rad(flip_deg)
            duration_s = max(self.rf_designer.duration.value() / 1000.0, 1e-9)

            gmr_1h_rad_Ts = 267522187.43999997
            integfac = max(self.rf_designer.get_integration_factor(), 1e-6)

            # Required amplitude (Tesla); convert to Gauss
            amp_gauss = float(flip_rad / (gmr_1h_rad_Ts * integfac * duration_s)) * 1e4
            if not np.isfinite(amp_gauss) or amp_gauss <= 0:
                return
            # Update SSFP start flip angle (default to alpha/2)
            self.sequence_designer.ssfp_start_flip.blockSignals(True)
            self.sequence_designer.ssfp_start_flip.setValue(flip_deg * 0.5)
            self.sequence_designer.ssfp_start_flip.blockSignals(False)

            # Keep duration/phase-driven diagram in sync
            self.sequence_designer.update_diagram(self.rf_designer.get_pulse())
        except Exception:
            # Fail silently to avoid interrupting UI flow
            return

    def set_sweep_mode(self, enabled: bool):
        """Enable/disable sweep mode (skip heavy plotting during sweeps)."""
        self._sweep_mode = bool(enabled)
        if enabled:
            # Stop any running animation to save resources
            if hasattr(self, "anim_timer") and self.anim_timer.isActive():
                self.anim_timer.stop()
                self._sync_play_toggle(False)

    def _setup_time_synchronization(self):
        """Setup connections for universal time control synchronization."""
        # Connect universal time control to update all views
        self.time_control.time_changed.connect(self._on_universal_time_changed)

        # Connect play/pause/reset buttons
        self.time_control.play_pause_button.toggled.connect(
            self._handle_play_pause_toggle
        )
        self.time_control.reset_button.clicked.connect(self._handle_reset_clicked)
        self.time_control.speed_spin.valueChanged.connect(self._update_playback_speed)

        # Connect 3D vector position changes to universal control
        self.mag_3d.position_changed.connect(self._on_3d_vector_position_changed)

    def _on_universal_time_changed(
        self, time_idx: int, skip_expensive_updates=False, reset_anchor=True
    ):
        """Central handler for universal time control changes - updates all views.

        Parameters
        ----------
        time_idx : int
            The time index to display
        skip_expensive_updates : bool
            If True, only update time cursors and skip expensive plot redraws.
            Use this during animation playback for better performance.
        reset_anchor : bool
            If True, reset the playback timing anchor (for manual scrubbing).
        """
        if not hasattr(self, "last_time") or self.last_time is None:
            return
        actual_idx = self._playback_to_full_index(time_idx)
        # Convert to ms for sequence diagram alignment
        if hasattr(self.sequence_designer, "set_cursor_index"):
            self.sequence_designer.set_cursor_index(actual_idx)

        # Update sequence diagram cursor
        # Update 3D vector view
        self.mag_3d.set_cursor_index(time_idx)
        self._set_animation_index_from_slider(time_idx, reset_anchor=reset_anchor)
        self._update_b1_arrow(time_idx)

        # Get current visible tab to optimize updates (define early for all code paths)
        current_tab_index = self.tab_widget.currentIndex()

        # Check which plots are actually visible and need updates
        # Tab indices: 0=Magnetization, 1=3D Vector, 2=Signal, 3=Spectrum, 4=Spatial, ...
        mag_tab_visible = current_tab_index == 0
        signal_tab_visible = current_tab_index == 2
        spectrum_tab_visible = current_tab_index == 3
        spatial_tab_visible = current_tab_index == 4

        # Update time cursors on plots
        # PyQt/pyqtgraph still processes updates even for hidden tabs, causing lag.
        # Disable updates for plots that aren't visible to improve animation performance.
        if self.last_time is not None and 0 <= actual_idx < len(self.last_time):
            time_ms = self.last_time[actual_idx] * 1000

            # Only update time cursors when NOT animating (during scrubbing/pause)
            # During animation, skip time cursor updates for Magnetization and Signal plots
            # to improve performance - only 3D vector animates
            # Time lines removed for performance

        # Always update visible spectrum/spatial views, even during playback
        if spatial_tab_visible:
            self.update_spatial_plot_from_last_result(time_idx=actual_idx)
            self._spatial_needs_update = False
        else:
            self._spatial_needs_update = True

        if spectrum_tab_visible:
            self._refresh_spectrum(time_idx=actual_idx, skip_fft=skip_expensive_updates)
            self._spectrum_needs_update = False
        else:
            self._spectrum_needs_update = True

    def _on_tab_changed(self, index: int):
        """Handle tab changes to optimize rendering.

        Disable updates on plots that aren't visible to speed up tab switching.
        """
        # Enable updates on all plot widgets first
        all_plot_widgets = [
            self.mxy_plot,
            self.mz_plot,
            self.mxy_heatmap_layout,
            self.mz_heatmap_layout,
            self.signal_plot,
            self.signal_heatmap_layout,
            self.spectrum_plot,
            self.spectrum_heatmap_layout,
            self.spatial_mxy_plot,
            self.spatial_mz_plot,
            self.spatial_heatmap_container,
        ]
        for widget in all_plot_widgets:
            if widget is not None:
                widget.setUpdatesEnabled(True)

        # Now disable updates on plots not in the current tab
        # Tab indices: 0=Magnetization, 1=3D Vector, 2=Signal, 3=Spectrum, 4=Spatial
        if index != 0:  # Not Magnetization tab
            self.mxy_plot.setUpdatesEnabled(False)
            self.mz_plot.setUpdatesEnabled(False)
        if index != 2:  # Not Signal tab
            self.signal_plot.setUpdatesEnabled(False)
            self.signal_heatmap_layout.setUpdatesEnabled(False)
        if index != 3:  # Not Spectrum tab
            self.spectrum_plot.setUpdatesEnabled(False)
            if (
                hasattr(self, "spectrum_heatmap_layout")
                and self.spectrum_heatmap_layout is not None
            ):
                self.spectrum_heatmap_layout.setUpdatesEnabled(False)
        else:  # Switching TO Spectrum tab
            # Update spectrum if it's dirty
            if (
                self._spectrum_needs_update
                and hasattr(self, "last_result")
                and self.last_result is not None
            ):
                current_idx = (
                    self.time_control.time_slider.value()
                    if hasattr(self, "time_control")
                    else 0
                )
                actual_idx = self._playback_to_full_index(current_idx)
                self._refresh_spectrum(time_idx=actual_idx, skip_fft=False)
                self._spectrum_needs_update = False
        if index != 4:  # Not Spatial tab
            self.spatial_mxy_plot.setUpdatesEnabled(False)
            self.spatial_mz_plot.setUpdatesEnabled(False)
            if hasattr(self, "spatial_heatmap_container"):
                self.spatial_heatmap_container.setUpdatesEnabled(False)
        else:  # Switching TO Spatial tab
            # Update spatial if it's dirty
            if (
                self._spatial_needs_update
                and hasattr(self, "last_result")
                and self.last_result is not None
            ):
                current_idx = (
                    self.time_control.time_slider.value()
                    if hasattr(self, "time_control")
                    else 0
                )
                actual_idx = self._playback_to_full_index(current_idx)
                self.update_spatial_plot_from_last_result(time_idx=actual_idx)
                self._spatial_needs_update = False

    def _start_vector_animation(self):
        """Start or restart the 3D vector animation if data exists."""
        # Prevent animation during parameter sweeps
        if getattr(self, "_sweep_mode", False):
            self._sync_play_toggle(False)
            return

        if self.anim_data is None or len(self.anim_data) == 0:
            self.anim_timer.stop()
            self._sync_play_toggle(False)
            return
        if self.mag_3d.track_path:
            self.mag_3d._clear_path()
        if self.anim_index >= len(self.anim_data):
            self.anim_index = 0
        self._reset_playback_anchor(self.anim_index)

        # Disable updates on Magnetization and Signal plots during animation
        self.mxy_plot.setUpdatesEnabled(False)
        self.mz_plot.setUpdatesEnabled(False)
        self.signal_plot.setUpdatesEnabled(False)

        # Always recompute interval using current speed control
        self._recompute_anim_interval(
            self.time_control.speed_spin.value()
            if hasattr(self, "time_control")
            else None
        )
        self.anim_timer.start(self.anim_interval_ms)
        self._sync_play_toggle(True)

    def _resume_vector_animation(self):
        """Resume playback of the 3D vector."""
        self._start_vector_animation()

    def _pause_vector_animation(self):
        """Pause the 3D vector animation."""
        self.anim_timer.stop()
        self._sync_play_toggle(False)

        # Re-enable updates on Magnetization and Signal plots after animation stops
        self.mxy_plot.setUpdatesEnabled(True)
        self.mz_plot.setUpdatesEnabled(True)
        self.signal_plot.setUpdatesEnabled(True)

        # When paused, refresh plots with current frame (full update)
        if hasattr(self, "anim_index"):
            self._on_universal_time_changed(
                self.anim_index, skip_expensive_updates=False
            )

    def _reset_vector_animation(self):
        """Reset the 3D vector to the first available frame."""
        self.anim_timer.stop()
        self._sync_play_toggle(False)

        # Re-enable updates on Magnetization and Signal plots after animation stops
        self.mxy_plot.setUpdatesEnabled(True)
        self.mz_plot.setUpdatesEnabled(True)
        self.signal_plot.setUpdatesEnabled(True)

        self.anim_index = 0
        self.mag_3d._clear_path()
        self._reset_playback_anchor(0)
        if self.anim_data is not None and len(self.anim_data) > 0:
            vectors = self.anim_data[0]
            colors = [
                self._color_tuple(self._color_for_index(i, vectors.shape[0]))
                for i in range(vectors.shape[0])
            ]
            self.mag_3d.update_magnetization(vectors, colors=colors)
            self.mag_3d.set_cursor_index(0)
        if self.playback_time is not None:
            self.time_control.set_time_index(0)
            self._on_universal_time_changed(0)
        if self.mag_3d.b1_arrow is not None:
            self.mag_3d.b1_arrow.setVisible(False)

    def _recompute_anim_interval(self, sim_ms_per_s: float = None):
        """Compute animation interval so that wall time matches simulation time scaling."""
        if sim_ms_per_s is None:
            sim_ms_per_s = self.time_control.speed_spin.value()
        if sim_ms_per_s <= 0:
            sim_ms_per_s = 50.0  # fallback to reasonable speed
        total_frames = len(self.playback_time) if self.playback_time is not None else 0
        if total_frames < 2:
            self.anim_interval_ms = 30
            self._frame_step = 1
            return

        # Calculate time per frame in the simulation data
        duration_ms = max(
            float(self.playback_time[-1] - self.playback_time[0]) * 1000.0, 1e-6
        )
        time_per_frame_ms = duration_ms / (total_frames - 1)

        desired_interval_ms = time_per_frame_ms / sim_ms_per_s * 1000.0

        min_interval = getattr(self, "_min_anim_interval_ms", 2.0)
        interval_ms = max(min_interval, desired_interval_ms)
        self.anim_interval_ms = max(1, int(round(interval_ms)))
        self._frame_step = 1

    def _update_playback_speed(self, sim_ms_per_s: float):
        """Adjust playback speed (simulation ms per real second)."""
        self._recompute_anim_interval(sim_ms_per_s)
        self._reset_playback_anchor(self.anim_index)
        if self.anim_timer.isActive():
            self.anim_timer.start(self.anim_interval_ms)

    def _set_animation_index_from_slider(self, idx: int, reset_anchor=True):
        """Scrub animation position from the preview slider."""
        if self.anim_data is None or len(self.anim_data) == 0:
            return
        idx = int(max(0, min(idx, len(self.anim_data) - 1)))

        if idx == 0 and self.mag_3d.track_path and len(self.mag_3d.path_points) > 0:
            self.mag_3d._clear_path()

        self.anim_index = idx
        vectors = self.anim_data[idx]
        self.mag_3d.update_magnetization(vectors, colors=self.anim_colors)
        self.mag_3d.set_cursor_index(idx)
        self._update_b1_arrow(idx)
        if reset_anchor:
            self._reset_playback_anchor(idx)

    def _on_3d_vector_position_changed(self, time_idx: int):
        """Handle 3D vector view position changes."""
        if not self.time_control._updating:
            self.time_control.set_time_index(time_idx)
        # Propagate the change to all synchronized views
        # When user is manually scrubbing, do full update (skip_expensive_updates=False)
        is_playing = (
            self.anim_timer.isActive() if hasattr(self, "anim_timer") else False
        )
        self._on_universal_time_changed(time_idx, skip_expensive_updates=is_playing)

    def _handle_play_pause_toggle(self, playing: bool):
        """Unified handler for play/pause toggle."""
        if playing:
            self._on_universal_play()
        else:
            self._on_universal_pause()

    def _handle_reset_clicked(self):
        """Reset playback and return to paused state."""
        self._on_universal_reset()
        self._sync_play_toggle(False)

    def _sync_play_toggle(self, playing: bool):
        """Keep the play/pause toggle in sync with actual playback."""
        if hasattr(self, "time_control"):
            self.time_control.sync_play_state(playing)

    def _on_universal_play(self):
        """Handle universal play button."""
        # Use latest speed setting
        self._recompute_anim_interval(self.time_control.speed_spin.value())
        self._resume_vector_animation()

    def _on_universal_pause(self):
        """Handle universal pause button."""
        self._pause_vector_animation()
        self._sync_play_toggle(False)

    def _on_universal_reset(self):
        """Handle universal reset button."""
        self._reset_vector_animation()
        self._sync_play_toggle(False)

    def _set_plot_ranges(self, plot_widget, x_min, x_max, y_min=None, y_max=None):
        """Apply consistent axis ranges."""
        x_min = max(0, x_min)
        # Avoid zero span which can break setRange/limits
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        plot_widget.enableAutoRange(x=False)
        plot_widget.setXRange(x_min, x_max, padding=0)
        limits = {"xMin": 0, "xMax": x_max}
        if y_min is not None and y_max is not None:
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            plot_widget.enableAutoRange(y=False)
            plot_widget.setYRange(y_min, y_max, padding=0)
            limits.update({"yMin": y_min, "yMax": y_max})
        plot_widget.setLimits(**limits)

    def _refresh_mag_plots(self):
        """Re-render magnetization plots using the current filter selection."""
        if self.last_result is not None:
            # Check plot type and switch visibility
            plot_type = (
                self.mag_plot_type.currentText()
                if hasattr(self, "mag_plot_type")
                else "Line"
            )
            if plot_type == "Heatmap":
                self.mxy_plot.hide()
                self.mz_plot.hide()
                self.mxy_heatmap_layout.show()
                self.mz_heatmap_layout.show()
                self._update_mag_heatmaps()
            else:
                self.mxy_plot.show()
                self.mz_plot.show()
                self.mxy_heatmap_layout.hide()
                self.mz_heatmap_layout.hide()
                self._render_mag_lines()

    def _render_mag_lines(self):
        """Render magnetization as lines."""
        if self.last_result is None:
            return

        mx_all = self.last_result["mx"]
        my_all = self.last_result["my"]
        mz_all = self.last_result["mz"]

        time_arr = (
            self.last_time
            if self.last_time is not None
            else self.last_result.get("time", None)
        )
        if time_arr is None:
            return
        time_ms = time_arr * 1000

        if mx_all.ndim == 2:
            # Steady-state / endpoint
            t_plot = (
                np.array([time_ms[0], time_ms[-1]])
                if len(time_ms) > 1
                else np.array([0, 1])
            )
            self._safe_clear_plot(self.mxy_plot)
            total_series = mx_all.shape[0] * mx_all.shape[1]
            self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
            idx = 0
            for pi in range(mx_all.shape[0]):
                for fi in range(mx_all.shape[1]):
                    color = self._color_for_index(idx, total_series)
                    self.mxy_plot.plot(
                        t_plot,
                        [mx_all[pi, fi], mx_all[pi, fi]],
                        pen=pg.mkPen(color, width=4),
                    )
                    self.mxy_plot.plot(
                        t_plot,
                        [my_all[pi, fi], my_all[pi, fi]],
                        pen=pg.mkPen(color, style=Qt.DashLine, width=2),
                    )
                    idx += 1

            self._safe_clear_plot(self.mz_plot)
            self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
            idx = 0
            for pi in range(mz_all.shape[0]):
                for fi in range(mz_all.shape[1]):
                    color = self._color_for_index(idx, total_series)
                    self.mz_plot.plot(
                        t_plot,
                        [mz_all[pi, fi], mz_all[pi, fi]],
                        pen=pg.mkPen(color, width=2),
                    )
                    idx += 1
        else:
            # Time-resolved
            ntime, npos, nfreq = mx_all.shape
            mean_only = self.mean_only_checkbox.isChecked()

            self._safe_clear_plot(self.mxy_plot)
            if mean_only:
                mean_mx = np.mean(mx_all, axis=(1, 2))
                mean_my = np.mean(my_all, axis=(1, 2))
                self.mxy_plot.plot(time_ms, mean_mx, pen=pg.mkPen("c", width=4))
                self.mxy_plot.plot(
                    time_ms, mean_my, pen=pg.mkPen("c", width=4, style=Qt.DashLine)
                )
            else:
                total_series = npos * nfreq
                indices_to_plot = self._get_trace_indices_to_plot(total_series)
                for linear_idx in indices_to_plot:
                    pi = linear_idx // nfreq
                    fi = linear_idx % nfreq
                    color = self._color_for_index(linear_idx, total_series)
                    self.mxy_plot.plot(
                        time_ms, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2)
                    )
                    self.mxy_plot.plot(
                        time_ms,
                        my_all[:, pi, fi],
                        pen=pg.mkPen(color, style=Qt.DashLine, width=2),
                    )

            self._safe_clear_plot(self.mz_plot)
            if mean_only:
                mean_mz = np.mean(mz_all, axis=(1, 2))
                self.mz_plot.plot(time_ms, mean_mz, pen=pg.mkPen("c", width=4))
            else:
                indices_to_plot = self._get_trace_indices_to_plot(npos * nfreq)
                for linear_idx in indices_to_plot:
                    pi = linear_idx // nfreq
                    fi = linear_idx % nfreq
                    color = self._color_for_index(linear_idx, npos * nfreq)
                    self.mz_plot.plot(
                        time_ms, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2)
                    )

    def _refresh_signal_plots(self):
        """Re-render signal plots using the current filter selection."""
        if self.last_result is None:
            return
        # Check plot type and switch visibility
        plot_type = (
            self.signal_plot_type.currentText()
            if hasattr(self, "signal_plot_type")
            else "Heatmap"
        )
        if plot_type == "Heatmap":
            self.signal_plot.hide()
            self.signal_heatmap_layout.show()
            self._update_signal_heatmaps()
        else:
            self.signal_plot.show()
            self.signal_heatmap_layout.hide()
            self._render_signal_lines()

    def _render_signal_lines(self):
        """Render received signal as lines."""
        if self.last_result is None:
            return

        signal_all = self.last_result["signal"]
        time_arr = (
            self.last_time
            if self.last_time is not None
            else self.last_result.get("time", None)
        )
        if time_arr is None:
            return
        time_ms = time_arr * 1000

        self._safe_clear_plot(self.signal_plot)

        if signal_all.ndim == 1:
            # Simple 1D signal
            self.signal_plot.plot(
                time_ms,
                np.abs(signal_all),
                pen=pg.mkPen("w", width=2),
                name="Magnitude",
            )
            self.signal_plot.plot(
                time_ms, np.real(signal_all), pen=pg.mkPen("r", width=1), name="Real"
            )
            self.signal_plot.plot(
                time_ms,
                np.imag(signal_all),
                pen=pg.mkPen("g", width=1),
                name="Imaginary",
            )
        elif signal_all.ndim == 3:
            # Time-resolved 3D data (ntime, npos, nfreq)
            ntime, npos, nfreq = signal_all.shape
            mean_only = self.mean_only_checkbox.isChecked()

            if mean_only:
                mean_sig = np.mean(signal_all, axis=(1, 2))
                self.signal_plot.plot(
                    time_ms,
                    np.abs(mean_sig),
                    pen=pg.mkPen("w", width=3),
                    name="Mean Mag",
                )
            else:
                total_series = npos * nfreq
                indices_to_plot = self._get_trace_indices_to_plot(total_series)
                for linear_idx in indices_to_plot:
                    pi = linear_idx // nfreq
                    fi = linear_idx % nfreq
                    color = self._color_for_index(linear_idx, total_series)
                    sig_trace = signal_all[:, pi, fi]
                    self.signal_plot.plot(
                        time_ms, np.abs(sig_trace), pen=pg.mkPen(color, width=1.5)
                    )

    def _calc_symmetric_limits(self, *arrays, base=1.0, pad=1.1):
        """Compute symmetric y-limits with padding based on provided arrays."""
        max_abs = 0.0
        for arr in arrays:
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.size == 0:
                continue
            with np.errstate(invalid="ignore"):
                current = np.nanmax(np.abs(arr_np))
            if np.isfinite(current):
                max_abs = max(max_abs, float(current))
        max_abs = max(base, max_abs)
        return -pad * max_abs, pad * max_abs

    def _reset_legend(self, plot_widget, attr_name: str, enable: bool):
        """Reset/add a legend on the given plot."""
        existing = getattr(self, attr_name, None)
        if existing is not None:
            try:
                # Check if the item is actually in this scene before removing
                if existing.scene() == plot_widget.scene():
                    plot_widget.scene().removeItem(existing)
            except (RuntimeError, AttributeError):
                # Item already deleted or scene mismatch
                pass
            setattr(self, attr_name, None)
        if enable:
            legend = plot_widget.addLegend(offset=(6, 6))
            legend.layout.setSpacing(4)
            setattr(self, attr_name, legend)
            return legend
        return None

    def _apply_heatmap_colormap(self, cmap_name: Optional[str] = None):
        """Apply a shared colormap to all heatmaps/colorbars."""
        if cmap_name is None and hasattr(self, "heatmap_colormap"):
            cmap_name = self.heatmap_colormap.currentText()
        cmap_name = cmap_name or "viridis"
        try:
            cmap = pg.colormap.get(cmap_name)
        except Exception:
            cmap = cmap_name  # fall back to string name if lookup fails

        def _set_cb(cb_item):
            if cb_item is None:
                return
            setter = getattr(cb_item, "setColorMap", None)
            if callable(setter):
                try:
                    setter(cmap)
                except Exception:
                    pass

        for cb in [
            getattr(self, "mxy_heatmap_colorbar", None),
            getattr(self, "mz_heatmap_colorbar", None),
            getattr(self, "signal_heatmap_colorbar", None),
            getattr(self, "spectrum_heatmap_colorbar", None),
            getattr(self, "spatial_heatmap_mxy_colorbar", None),
            getattr(self, "spatial_heatmap_mz_colorbar", None),
        ]:
            _set_cb(cb)

        lut = None
        if hasattr(cmap, "getLookupTable"):
            try:
                lut = cmap.getLookupTable()
            except Exception:
                lut = None
        if lut is not None:
            for img in [
                getattr(self, "mxy_heatmap_item", None),
                getattr(self, "mz_heatmap_item", None),
                getattr(self, "signal_heatmap_item", None),
                getattr(self, "spectrum_heatmap_item", None),
                getattr(self, "spatial_heatmap_mxy_item", None),
                getattr(self, "spatial_heatmap_mz_item", None),
            ]:
                if img is not None and hasattr(img, "setLookupTable"):
                    try:
                        img.setLookupTable(lut)
                    except Exception:
                        pass

        # Keep a consistent palette for the status bar progress as well
        if hasattr(self, "status_progress"):
            self.status_progress.setStyleSheet("")

    def _update_mag_selector_limits(self, npos: int, nfreq: int, disable: bool = False):
        """Sync magnetization view selector with available pos/freq counts."""
        if not hasattr(self, "mag_view_mode"):
            return
        mode = self.mag_view_mode.currentText()
        slider = self.mag_view_selector

        # Helper to find index closest to 0
        def _get_zero_idx(arr):
            if arr is None or len(arr) == 0:
                return 0
            return int(np.argmin(np.abs(arr)))

        if disable:
            slider.setRange(0, 0)
            slider.setEnabled(False)
            self.mag_view_selector_label.setText("All spins")
        elif mode == "Positions @ freq":
            max_idx = max(0, nfreq - 1)
            slider.setRange(0, max_idx)
            slider.setEnabled(nfreq > 1)

            # If current value is out of range or we just switched, try to set to 0 Hz
            if slider.value() > max_idx:
                target = (
                    _get_zero_idx(self.last_frequencies)
                    if self.last_frequencies is not None
                    else 0
                )
                slider.setValue(target)

            idx = slider.value()
            freq_hz_val = (
                self.last_frequencies[idx]
                if self.last_frequencies is not None
                and idx < len(self.last_frequencies)
                else idx
            )
            self.mag_view_selector_label.setText(f"Freq: {freq_hz_val:.1f} Hz")
        elif mode == "Freqs @ position":
            max_idx = max(0, npos - 1)
            slider.setRange(0, max_idx)
            slider.setEnabled(npos > 1)

            # If current value is out of range or we just switched, try to set to 0 cm
            if slider.value() > max_idx:
                target = (
                    _get_zero_idx(self.last_positions[:, 2] * 100)
                    if self.last_positions is not None
                    else 0
                )
                slider.setValue(target)

            idx = slider.value()
            pos_val = (
                self.last_positions[idx, 2] * 100
                if self.last_positions is not None and idx < len(self.last_positions)
                else idx
            )
            self.mag_view_selector_label.setText(f"Pos: {pos_val:.2f} cm")
        else:
            slider.setRange(0, 0)
            slider.setEnabled(False)
            self.mag_view_selector_label.setText("All spins")

    def _current_mag_filter(self, npos: int, nfreq: int):
        """Return the active magnetization view filter selection."""
        mode = (
            self.mag_view_mode.currentText()
            if hasattr(self, "mag_view_mode")
            else "All positions x freqs"
        )
        if mode == "Positions @ freq":
            idx = min(max(self.mag_view_selector.value(), 0), max(0, nfreq - 1))
        elif mode == "Freqs @ position":
            idx = min(max(self.mag_view_selector.value(), 0), max(0, npos - 1))
        else:
            idx = 0
        return mode, idx

    def _update_signal_selector_limits(
        self, npos: int, nfreq: int, disable: bool = False
    ):
        """Sync signal view selector with available pos/freq counts."""
        if not hasattr(self, "signal_view_mode"):
            return
        mode = self.signal_view_mode.currentText()
        slider = self.signal_view_selector

        if disable:
            max_idx = 0
            prefix = "All"
            label_text = "All spins"
        elif mode == "Positions @ freq":
            max_idx = max(0, nfreq - 1)
            prefix = "Freq"
            idx = min(slider.value(), max_idx)
            freq_hz_val = (
                self.last_frequencies[idx]
                if self.last_frequencies is not None
                and idx < len(self.last_frequencies)
                else idx
            )
            label_text = f"{prefix}: {freq_hz_val:.1f} Hz"
        elif mode == "Freqs @ position":
            max_idx = max(0, npos - 1)
            prefix = "Pos"
            idx = min(slider.value(), max_idx)
            pos_val = (
                self.last_positions[idx, 2] * 100
                if self.last_positions is not None and idx < len(self.last_positions)
                else idx
            )
            label_text = f"{prefix}: {pos_val:.2f} cm"
        else:
            max_idx = 0
            prefix = "All"
            label_text = "All spins"

        slider.blockSignals(True)
        slider.setMaximum(max_idx)
        slider.setValue(min(slider.value(), max_idx) if max_idx > 0 else 0)
        slider.setEnabled(not disable and max_idx > 0)
        slider.setVisible(max_idx > 0)
        slider.blockSignals(False)

        self.signal_view_selector_label.setText(label_text)

    def _apply_pulse_region(self, plot_widget, attr_name):
        """Highlight pulse duration on a plot."""
        existing = getattr(self, attr_name, None)
        if existing is not None:
            plot_widget.removeItem(existing)
        if self.last_pulse_range is None:
            setattr(self, attr_name, None)
            return
        start_ms = self.last_pulse_range[0] * 1000
        end_ms = self.last_pulse_range[1] * 1000
        region = pg.LinearRegionItem(
            values=[start_ms, end_ms],
            brush=pg.mkBrush(100, 100, 255, 40),
            movable=False,
        )
        region.setZValue(-10)
        plot_widget.addItem(region)
        setattr(self, attr_name, region)

    def update_spatial_plot_from_last_result(self, time_idx=None):
        """Update spatial plot with Mxy and Mz profiles across positions at selected time.

        For a given time point (or endpoint), plots:
        - Mxy (transverse magnetization) vs position
        - Mz (longitudinal magnetization) vs position

        Parameters
        ----------
        time_idx : int, optional
            Time index to display. If None, uses last time point for time-resolved data.
        """
        if self.last_result is None or self.last_positions is None:
            self.log_message("Spatial plot: missing result or positions")
            return

        result = self.last_result
        mx = result.get("mx")
        my = result.get("my")
        mz = result.get("mz")

        # Validate data
        if mx is None or my is None or mz is None:
            self.log_message("Spatial plot: missing mx, my, or mz data")
            return

        self.log_message(
            f"Spatial plot: mx shape = {mx.shape}, my shape = {my.shape}, mz shape = {mz.shape}"
        )

        # Handle time-resolved vs endpoint data
        is_time_resolved = len(mz.shape) == 3  # (ntime, npos, nfreq)

        if is_time_resolved:
            # Store the full time-resolved data
            self.spatial_mx_time_series = mx
            self.spatial_my_time_series = my
            self.spatial_mz_time_series = mz
            ntime = mz.shape[0]

            # Use provided time index or default to last time point
            if time_idx is None:
                time_idx = ntime - 1
            time_idx = int(min(max(0, time_idx), ntime - 1))

            mx_display = mx[time_idx, :, :]
            my_display = my[time_idx, :, :]
            mz_display = mz[time_idx, :, :]
        else:
            # Endpoint mode: (npos, nfreq)
            mx_display = mx
            my_display = my
            mz_display = mz
            self.spatial_mx_time_series = None
            self.spatial_my_time_series = None
            self.spatial_mz_time_series = None
            time_idx = 0

        # Frequency selection/averaging for spatial view
        freq_count = mx_display.shape[1]
        max_freq_idx = max(0, freq_count - 1)

        # Update slider range
        self.spatial_freq_slider.blockSignals(True)
        self.spatial_freq_slider.setMaximum(max_freq_idx)
        if self.spatial_freq_slider.value() > max_freq_idx:
            self.spatial_freq_slider.setValue(max_freq_idx)
        self.spatial_freq_slider.blockSignals(False)

        freq_sel = min(self.spatial_freq_slider.value(), max_freq_idx)

        # Update label
        actual_freq = 0.0
        if self.last_frequencies is not None and freq_sel < len(self.last_frequencies):
            actual_freq = self.last_frequencies[freq_sel]
            self.spatial_freq_label.setText(f"Freq: {actual_freq:.1f} Hz")
        else:
            self.spatial_freq_label.setText(f"Freq idx: {freq_sel}")

        spatial_view_mode = self.spatial_mode.currentText()
        if spatial_view_mode == "Mean over freqs":
            mxy_pos = np.sqrt(mx_display**2 + my_display**2).mean(axis=1)
            mz_pos = mz_display.mean(axis=1)
        elif (
            spatial_view_mode == "Mean + individuals"
        ):  # This mode is no longer in the dropdown, but keeping for safety
            mxy_pos = np.sqrt(mx_display**2 + my_display**2).mean(axis=1)
            mz_pos = mz_display.mean(axis=1)
        else:  # Individual freq
            mxy_pos = np.sqrt(
                mx_display[:, freq_sel] ** 2 + my_display[:, freq_sel] ** 2
            )
            mz_pos = mz_display[:, freq_sel]

        # Choose a signed position axis (prefer the axis with largest span)
        pos_axis = self.last_positions
        spans = np.ptp(pos_axis, axis=0)  # max - min per axis
        axis_idx = int(np.argmax(spans))
        pos_distance = pos_axis[:, axis_idx]

        self.log_message(
            f"Spatial plot: mxy_pos shape = {mxy_pos.shape}, mz_pos shape = {mz_pos.shape}, pos_distance shape = {pos_distance.shape}"
        )

        freq_axis = (
            np.asarray(self.last_frequencies)
            if self.last_frequencies is not None
            else np.arange(freq_count)
        )
        if freq_axis.shape[0] != freq_count:
            freq_axis = np.linspace(
                freq_axis.min() if freq_axis.size else 0.0,
                freq_axis.max() if freq_axis.size else float(freq_count - 1),
                freq_count,
            )

        # Cache data for export and heatmap updates
        self._last_spatial_export = {
            "position_m": pos_distance,
            "mxy": mxy_pos,
            "mz": mz_pos,
            "freq_index": freq_sel,
            "time_idx": time_idx,
            "time_s": (
                self.last_time[time_idx]
                if self.last_time is not None and len(self.last_time) > time_idx
                else None
            ),
            "mxy_per_freq": np.sqrt(mx_display**2 + my_display**2),
            "mz_per_freq": mz_display,
            "frequency_axis": freq_axis,
            "heatmap_mode": None,
        }

        # Update plots
        plot_type = (
            self.spatial_plot_type.currentText()
            if hasattr(self, "spatial_plot_type")
            else "Line"
        )
        heatmap_mode = (
            self.spatial_heatmap_mode.currentText()
            if hasattr(self, "spatial_heatmap_mode")
            else "Position vs Frequency"
        )
        show_heatmap = plot_type == "Heatmap"
        self._set_spatial_plot_visibility(show_heatmap)
        if show_heatmap:
            # Heatmap mode
            if heatmap_mode == "Position vs Time" and is_time_resolved:
                mxy_time = np.sqrt(
                    self.spatial_mx_time_series[:, :, freq_sel] ** 2
                    + self.spatial_my_time_series[:, :, freq_sel] ** 2
                )
                mz_time = self.spatial_mz_time_series[:, :, freq_sel]
                self._last_spatial_export.update(
                    {
                        "heatmap_mode": "time",
                        "mxy_time": mxy_time,
                        "mz_time": mz_time,
                    }
                )
                self._update_spatial_time_heatmaps(
                    pos_distance, self.last_time, mxy_time, mz_time, freq_sel
                )
            elif heatmap_mode == "Position vs Time" and not is_time_resolved:
                self.log_message(
                    "Spatial heatmap time view requires time-resolved simulation; showing frequency view instead."
                )
                self._last_spatial_export["heatmap_mode"] = "frequency"
                self._update_spatial_heatmaps(
                    pos_distance,
                    self._last_spatial_export["mxy_per_freq"],
                    mz_display,
                    freq_axis,
                )
            else:
                self._last_spatial_export["heatmap_mode"] = "frequency"
                self._update_spatial_heatmaps(
                    pos_distance,
                    self._last_spatial_export["mxy_per_freq"],
                    mz_display,
                    freq_axis,
                )
        else:
            # Line plot mode
            self._update_spatial_line_plots(
                pos_distance,
                mxy_pos,
                mz_pos,
                mx_display,
                my_display,
                mz_display,
                freq_sel,
                spatial_view_mode,
            )

        # Keep sequence diagram in sync but avoid time cursors on spatial (position) axes
        if hasattr(self, "spatial_mxy_time_line"):
            self.spatial_mxy_time_line.hide()
        if hasattr(self, "spatial_mz_time_line"):
            self.spatial_mz_time_line.hide()
        if is_time_resolved and time_idx < len(self.last_time):
            current_time = self.last_time[time_idx]
            # Synchronize sequence diagram playhead
            if hasattr(self, "sequence_designer") and hasattr(
                self.sequence_designer, "playhead_line"
            ):
                if self.sequence_designer.playhead_line is not None:
                    self.sequence_designer.playhead_line.setValue(current_time * 1000.0)
                    if not self.sequence_designer.playhead_line.isVisible():
                        self.sequence_designer.playhead_line.show()

    def _set_spatial_plot_visibility(self, show_heatmap: bool):
        """Toggle between line plots and heatmaps in the Spatial view."""
        if hasattr(self, "spatial_heatmap_container"):
            self.spatial_heatmap_container.setVisible(show_heatmap)
        if hasattr(self, "spatial_mxy_plot"):
            self.spatial_mxy_plot.setVisible(not show_heatmap)
        if hasattr(self, "spatial_mz_plot"):
            self.spatial_mz_plot.setVisible(not show_heatmap)
        # Hide Mz plot when showing ONLY phase (wrapped or unwrapped)
        selected = self.spatial_component_combo.get_selected_items()
        if len(selected) > 0 and all(
            c in ["Phase", "Phase (unwrapped)"] for c in selected
        ):
            self.spatial_mz_plot.setVisible(False)

    def _update_spatial_line_plots(
        self,
        position,
        mxy,
        mz,
        mx_display=None,
        my_display=None,
        mz_display=None,
        freq_sel=0,
        spatial_mode="Mean only",
    ):
        """Update the Mxy and Mz line plots."""
        try:
            # Safely clear plots while preserving persistent items
            persistent_mxy = [self.spatial_mxy_time_line] + self.spatial_slice_lines[
                "mxy"
            ]
            persistent_mz = [self.spatial_mz_time_line] + self.spatial_slice_lines["mz"]
            self._safe_clear_plot(self.spatial_mxy_plot, persistent_mxy)
            self._safe_clear_plot(self.spatial_mz_plot, persistent_mz)

            selected_components = self.spatial_component_combo.get_selected_items()

            # Plot Mxy vs position
            if (
                spatial_mode == "Mean + individuals"
                and mx_display is not None
                and my_display is not None
                and mz_display is not None
            ):
                total_series = mx_display.shape[1]
                self._reset_legend(
                    self.spatial_mxy_plot, "spatial_mxy_legend", total_series > 1
                )
                self._reset_legend(
                    self.spatial_mz_plot, "spatial_mz_legend", total_series > 1
                )
                for fi in range(total_series):
                    color = self._color_for_index(fi, total_series)
                    mxy_ind = np.sqrt(mx_display[:, fi] ** 2 + my_display[:, fi] ** 2)
                    self.spatial_mxy_plot.plot(
                        position, mxy_ind, pen=pg.mkPen(color, width=1), name=f"f{fi}"
                    )
                    self.spatial_mz_plot.plot(
                        position,
                        mz_display[:, fi],
                        pen=pg.mkPen(color, width=1),
                        name=f"f{fi}",
                    )

                if "Magnitude" in selected_components:
                    self.spatial_mxy_plot.plot(
                        position, mxy, pen=pg.mkPen("b", width=3), name="|Mxy| mean"
                    )
                if "Real" in selected_components:
                    mx_mean = np.mean(mx_display, axis=1)
                    self.spatial_mxy_plot.plot(
                        position,
                        mx_mean,
                        pen=pg.mkPen("r", style=Qt.DashLine, width=2),
                        name="Mx mean",
                    )
                if "Imaginary" in selected_components:
                    my_mean = np.mean(my_display, axis=1)
                    self.spatial_mxy_plot.plot(
                        position,
                        my_mean,
                        pen=pg.mkPen("g", style=Qt.DotLine, width=2),
                        name="My mean",
                    )
                if "Phase" in selected_components:
                    mx_mean = np.mean(mx_display, axis=1)
                    my_mean = np.mean(my_display, axis=1)
                    phase = np.angle(mx_mean + 1j * my_mean) / np.pi
                    self.spatial_mxy_plot.plot(
                        position, phase, pen=pg.mkPen("c", width=2), name="Phase mean"
                    )

                if "Phase (unwrapped)" in selected_components:
                    mx_mean = np.mean(mx_display, axis=1)
                    my_mean = np.mean(my_display, axis=1)
                    phase_unwrapped = (
                        np.unwrap(np.angle(mx_mean + 1j * my_mean)) / np.pi
                    )
                    self.spatial_mxy_plot.plot(
                        position,
                        phase_unwrapped,
                        pen=pg.mkPen("y", width=2),
                        name="Phase (unwrapped) mean",
                    )

                self.spatial_mz_plot.plot(
                    position, mz, pen=pg.mkPen("m", width=3), name="Mz mean"
                )
            else:
                self._reset_legend(self.spatial_mxy_plot, "spatial_mxy_legend", True)
                self._reset_legend(self.spatial_mz_plot, "spatial_mz_legend", False)

                self.spatial_mxy_plot.setLabel("left", "Mxy (transverse)")

                if "Phase" in selected_components:
                    if mx_display is not None and my_display is not None:
                        if spatial_mode == "Individual freq" and mx_display.ndim == 2:
                            phase = (
                                np.angle(
                                    mx_display[:, freq_sel]
                                    + 1j * my_display[:, freq_sel]
                                )
                                / np.pi
                                if freq_sel < mx_display.shape[1]
                                else np.zeros(mx_display.shape[0])
                            )
                        else:
                            mx_mean = np.mean(mx_display, axis=1)
                            my_mean = np.mean(my_display, axis=1)
                            phase = np.angle(mx_mean + 1j * my_mean) / np.pi
                        self.spatial_mxy_plot.plot(
                            position, phase, pen=pg.mkPen("c", width=2), name="Phase"
                        )

                if "Phase (unwrapped)" in selected_components:
                    if mx_display is not None and my_display is not None:
                        if spatial_mode == "Individual freq" and mx_display.ndim == 2:
                            phase_unwrapped = (
                                np.unwrap(
                                    np.angle(
                                        mx_display[:, freq_sel]
                                        + 1j * my_display[:, freq_sel]
                                    )
                                )
                                / np.pi
                                if freq_sel < mx_display.shape[1]
                                else np.zeros(mx_display.shape[0])
                            )
                        else:
                            mx_mean = np.mean(mx_display, axis=1)
                            my_mean = np.mean(my_display, axis=1)
                            phase_unwrapped = (
                                np.unwrap(np.angle(mx_mean + 1j * my_mean)) / np.pi
                            )
                        self.spatial_mxy_plot.plot(
                            position,
                            phase_unwrapped,
                            pen=pg.mkPen("y", width=2),
                            name="Phase (unwrapped)",
                        )

                if "Magnitude" in selected_components:
                    self.spatial_mxy_plot.plot(
                        position, mxy, pen=pg.mkPen("b", width=2), name="|Mxy|"
                    )

                if "Real" in selected_components:
                    if mx_display is not None:
                        if (
                            spatial_mode == "Individual freq"
                            and mx_display.ndim == 2
                            and freq_sel < mx_display.shape[1]
                        ):
                            mx_line = mx_display[:, freq_sel]
                        elif mx_display.ndim == 2:
                            mx_line = np.mean(mx_display, axis=1)
                        else:
                            mx_line = mx_display
                        self.spatial_mxy_plot.plot(
                            position,
                            mx_line,
                            pen=pg.mkPen("r", style=Qt.DashLine, width=2),
                            name="Mx",
                        )

                if "Imaginary" in selected_components:
                    if my_display is not None:
                        if (
                            spatial_mode == "Individual freq"
                            and my_display.ndim == 2
                            and freq_sel < my_display.shape[1]
                        ):
                            my_line = my_display[:, freq_sel]
                        elif my_display.ndim == 2:
                            my_line = np.mean(my_display, axis=1)
                        else:
                            my_line = my_display
                        self.spatial_mxy_plot.plot(
                            position,
                            my_line,
                            pen=pg.mkPen("g", style=Qt.DotLine, width=2),
                            name="My",
                        )

                self.spatial_mz_plot.plot(position, mz, pen=pg.mkPen("m", width=2))
            self.spatial_mxy_plot.setTitle("Transverse Magnetization")
            self.spatial_mz_plot.setTitle("Longitudinal Magnetization")

            # Set consistent axis ranges based on full series if available
            pos_min, pos_max = position.min(), position.max()
            pos_pad = (pos_max - pos_min) * 0.1 if pos_max > pos_min else 0.1

            if (
                self.spatial_mx_time_series is not None
                and self.spatial_my_time_series is not None
            ):
                mxy_series = np.sqrt(
                    self.spatial_mx_time_series**2 + self.spatial_my_time_series**2
                )
                mxy_min_all = float(np.nanmin(mxy_series))
                mxy_max_all = float(np.nanmax(mxy_series))
                mx_all = self.spatial_mx_time_series
                my_all = self.spatial_my_time_series
            else:
                mxy_min_all = float(np.nanmin(mxy))
                mxy_max_all = float(np.nanmax(mxy))
                mx_all = mx_display if mx_display is not None else None
                my_all = my_display if my_display is not None else None
            if self.spatial_mz_time_series is not None:
                mz_min_all = float(np.nanmin(self.spatial_mz_time_series))
                mz_max_all = float(np.nanmax(self.spatial_mz_time_series))
            else:
                mz_min_all = float(np.nanmin(mz))
                mz_max_all = float(np.nanmax(mz))

            # Expand transverse range to include real/imag components
            if mx_all is not None:
                mxy_min_all = min(mxy_min_all, float(np.nanmin(mx_all)))
                mxy_max_all = max(mxy_max_all, float(np.nanmax(mx_all)))
            if my_all is not None:
                mxy_min_all = min(mxy_min_all, float(np.nanmin(my_all)))
                mxy_max_all = max(mxy_max_all, float(np.nanmax(my_all)))

            def padded_range(vmin, vmax, scale=1.1):
                if np.isclose(vmin, vmax):
                    pad = max(abs(vmin) * 0.1, 0.1)
                    return vmin - pad, vmax + pad
                span = vmax - vmin
                mid = (vmax + vmin) / 2.0
                half = (span * scale) / 2.0
                return mid - half, mid + half

            is_only_phase = (
                all(c in ["Phase", "Phase (unwrapped)"] for c in selected_components)
                and len(selected_components) > 0
            )
            has_magnitude = any(
                c in selected_components for c in ["Magnitude", "Real", "Imaginary"]
            )
            has_unwrapped = "Phase (unwrapped)" in selected_components

            # Use pre-calculated stable ranges, adapted for the current selection
            if has_unwrapped:
                self.spatial_mxy_plot.enableAutoRange(
                    axis=pg.ViewBox.YAxis, enable=True
                )
            elif is_only_phase:
                mxy_ymin, mxy_ymax = -1.1, 1.1
                self.spatial_mxy_plot.setYRange(mxy_ymin, mxy_ymax, padding=0)
            elif hasattr(self, "spatial_mxy_yrange"):
                # If only "Magnitude" is selected (no Real/Imag), use [0, max]
                if (
                    len(selected_components) == 1
                    and selected_components[0] == "Magnitude"
                ):
                    mxy_ymin, mxy_ymax = (
                        -0.05 * self.spatial_mxy_yrange[1],
                        self.spatial_mxy_yrange[1],
                    )
                else:
                    mxy_ymin, mxy_ymax = self.spatial_mxy_yrange
                self.spatial_mxy_plot.setYRange(mxy_ymin, mxy_ymax, padding=0)
            else:
                mxy_ymin, mxy_ymax = padded_range(mxy_min_all, mxy_max_all, scale=1.1)
                self.spatial_mxy_plot.setYRange(mxy_ymin, mxy_ymax)

            if hasattr(self, "spatial_mz_yrange") and not is_only_phase:
                mz_ymin, mz_ymax = self.spatial_mz_yrange
                self.spatial_mz_plot.setYRange(mz_ymin, mz_ymax, padding=0)
            elif not is_only_phase:
                mz_ymin, mz_ymax = padded_range(mz_min_all, mz_max_all, scale=1.1)
                self.spatial_mz_plot.setYRange(mz_ymin, mz_ymax)

            self.spatial_mxy_plot.setXRange(pos_min - pos_pad, pos_max + pos_pad)
            self.spatial_mz_plot.setXRange(pos_min - pos_pad, pos_max + pos_pad)

            # Update slice thickness guides
            slice_thk = None
            try:
                slice_thk = self.sequence_designer._slice_thickness_m()
            except Exception:
                slice_thk = None
            if slice_thk is not None and slice_thk > 0 and np.isfinite(slice_thk):
                center = float(np.median(position))
                half = slice_thk / 2.0
                positions = [center - half, center + half]
                for line, pos in zip(self.spatial_slice_lines["mxy"], positions):
                    line.setValue(pos)
                    line.setVisible(True)
                for line, pos in zip(self.spatial_slice_lines["mz"], positions):
                    line.setValue(pos)
                    line.setVisible(True)
            else:
                for line in (
                    self.spatial_slice_lines["mxy"] + self.spatial_slice_lines["mz"]
                ):
                    line.setVisible(False)

            # Add colored vertical markers if enabled
            if (
                self.spatial_markers_checkbox.isChecked()
                and mx_display is not None
                and my_display is not None
            ):
                # Determine what we're marking: frequencies or positions
                nfreq = mx_display.shape[1] if mx_display.ndim == 2 else 1
                npos = len(position)
                max_markers = min(
                    self.max_traces_spin.value(), 50
                )  # Limit for spatial markers

                # Broaden condition to show markers even in "Mean" mode if user enabled them
                if nfreq > 0:
                    # Downsample which frequencies to mark
                    if nfreq <= max_markers:
                        freq_indices_to_mark = list(range(nfreq))
                    else:
                        step = nfreq / max_markers
                        freq_indices_to_mark = [
                            int(i * step) for i in range(max_markers)
                        ]

                    # Downsample position points to draw stems at
                    if npos <= max_markers:
                        pos_indices_to_mark = list(range(npos))
                    else:
                        step_pos = npos / max_markers
                        pos_indices_to_mark = [
                            int(i * step_pos) for i in range(max_markers)
                        ]

                    # Mark selected frequencies with colored stem plots
                    for fi in freq_indices_to_mark:
                        # Ensure color is an RGBA tuple for consistent rendering
                        color = self._color_tuple(self._color_for_index(fi, nfreq))
                        mxy_val = (
                            np.sqrt(mx_display[:, fi] ** 2 + my_display[:, fi] ** 2)
                            if mx_display.ndim == 2
                            else mxy
                        )
                        mz_val = mz_display[:, fi] if mz_display.ndim == 2 else mz_pos

                        # Downsample position points to draw stems at
                        if npos <= max_markers:
                            pos_indices_to_mark = list(range(npos))
                        else:
                            step_pos = npos / max_markers
                            pos_indices_to_mark = [
                                int(i * step_pos) for i in range(max_markers)
                            ]

                        # Draw stem lines from 0 to current value at selected positions
                        for pi in pos_indices_to_mark:
                            pos_val = position[pi]
                            # Mxy markers
                            line_mxy = pg.PlotCurveItem(
                                [pos_val, pos_val],
                                [0, mxy_val[pi]],
                                pen=pg.mkPen(color=color, width=1.5),
                            )
                            self.spatial_mxy_plot.addItem(line_mxy)
                            # Mz markers
                            line_mz = pg.PlotCurveItem(
                                [pos_val, pos_val],
                                [0, mz_val[pi]],
                                pen=pg.mkPen(color=color, width=1.5),
                            )
                            self.spatial_mz_plot.addItem(line_mz)

            self.log_message("Spatial plot: updated successfully")
        except Exception as e:
            self.log_message(f"Spatial plot: error updating plots: {e}")
            import traceback

            self.log_message(f"Spatial plot: traceback: {traceback.format_exc()}")

    def _update_spatial_heatmaps(self, position, mxy_per_freq, mz_per_freq, freq_axis):
        """Render spatial heatmaps (position vs frequency) for |Mxy| and Mz."""
        try:
            if mxy_per_freq is None or mz_per_freq is None:
                return
            pos = np.asarray(position)
            freq_axis = np.asarray(freq_axis)
            if pos.size == 0 or freq_axis.size == 0:
                return
            mxy_arr = np.abs(np.asarray(mxy_per_freq))
            mz_arr = np.asarray(mz_per_freq)
            if mxy_arr.ndim != 2 or mz_arr.ndim != 2:
                return
            # Ensure axis lengths match the data
            npos, nfreq = mxy_arr.shape
            if freq_axis.size != nfreq:
                freq_axis = np.linspace(
                    freq_axis.min() if freq_axis.size else 0.0,
                    freq_axis.max() if freq_axis.size else float(nfreq - 1),
                    nfreq,
                )

            pos_min, pos_max = float(np.nanmin(pos)), float(np.nanmax(pos))
            if (
                not np.isfinite(pos_min)
                or not np.isfinite(pos_max)
                or np.isclose(pos_min, pos_max)
            ):
                pos_min, pos_max = 0.0, float(max(npos - 1, 1))
            x_span = pos_max - pos_min if pos_max != pos_min else 1.0
            freq_min, freq_max = float(np.nanmin(freq_axis)), float(
                np.nanmax(freq_axis)
            )
            if (
                not np.isfinite(freq_min)
                or not np.isfinite(freq_max)
                or np.isclose(freq_min, freq_max)
            ):
                freq_min, freq_max = 0.0, float(max(nfreq - 1, 1))
            y_span = freq_max - freq_min if freq_max != freq_min else 1.0

            def _set_heatmap(
                plot_widget, img_item, colorbar, data, pos_min, y_min, x_span, y_span
            ):
                img_item.setImage(data, autoLevels=True, axisOrder="row-major")
                img_item.setRect(pos_min, y_min, x_span, y_span)
                plot_widget.setXRange(pos_min, pos_max, padding=0)
                plot_widget.setYRange(y_min, y_max, padding=0)
                if colorbar is not None:
                    finite_vals = data[np.isfinite(data)]
                    if finite_vals.size:
                        with np.errstate(invalid="ignore"):
                            vmin, vmax = float(np.nanmin(finite_vals)), float(
                                np.nanmax(finite_vals)
                            )
                        if (
                            np.isfinite(vmin)
                            and np.isfinite(vmax)
                            and not np.isclose(vmax, vmin)
                        ):
                            colorbar.setLevels([vmin, vmax])

            # ImageItem expects (rows, cols) = (y, x)
            _set_heatmap(
                self.spatial_heatmap_mxy,
                self.spatial_heatmap_mxy_item,
                getattr(self, "spatial_heatmap_mxy_colorbar", None),
                mxy_arr.T,
                pos_min,
                freq_min,
                x_span,
                y_span,
            )
            _set_heatmap(
                self.spatial_heatmap_mz,
                self.spatial_heatmap_mz_item,
                getattr(self, "spatial_heatmap_mz_colorbar", None),
                mz_arr.T,
                pos_min,
                freq_min,
                x_span,
                y_span,
            )
        except Exception as exc:
            self.log_message(f"Spatial heatmap update failed: {exc}")

    def _update_spatial_time_heatmaps(
        self, position, time_axis, mxy_time, mz_time, freq_sel
    ):
        """Render spatial heatmaps (position vs time) for a selected frequency."""
        try:
            pos = np.asarray(position)
            time_axis = (
                np.asarray(time_axis)
                if time_axis is not None
                else np.arange(mxy_time.shape[0])
            )
            if pos.size == 0 or time_axis.size == 0:
                return
            if mxy_time.ndim != 2 or mz_time.ndim != 2:
                return
            ntime, npos = mxy_time.shape
            if time_axis.size != ntime:
                time_axis = np.linspace(
                    time_axis.min() if time_axis.size else 0.0,
                    time_axis.max() if time_axis.size else float(ntime - 1),
                    ntime,
                )
            time_ms = time_axis * 1000.0

            pos_min, pos_max = float(np.nanmin(pos)), float(np.nanmax(pos))
            if (
                not np.isfinite(pos_min)
                or not np.isfinite(pos_max)
                or np.isclose(pos_min, pos_max)
            ):
                pos_min, pos_max = 0.0, float(max(npos - 1, 1))
            x_span = pos_max - pos_min if pos_max != pos_min else 1.0
            t_min, t_max = float(np.nanmin(time_ms)), float(np.nanmax(time_ms))
            if (
                not np.isfinite(t_min)
                or not np.isfinite(t_max)
                or np.isclose(t_min, t_max)
            ):
                t_min, t_max = 0.0, float(max(ntime - 1, 1))
            t_span = t_max - t_min if t_max != t_min else 1.0

            def _set_heatmap(plot_widget, img_item, colorbar, data, title):
                img_item.setImage(data, autoLevels=True, axisOrder="row-major")
                img_item.setRect(pos_min, t_min, x_span, t_span)
                plot_widget.setXRange(pos_min, pos_max, padding=0)
                plot_widget.setYRange(t_min, t_max, padding=0)
                plot_widget.setLabel("bottom", "Position", "m")
                plot_widget.setLabel("left", "Time", "ms")
                plot_widget.setTitle(title)
                if colorbar is not None:
                    finite_vals = data[np.isfinite(data)]
                    if finite_vals.size:
                        vmin = float(finite_vals.min())
                        vmax = float(finite_vals.max())
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
                            colorbar.setLevels((vmin, vmax))

            _set_heatmap(
                self.spatial_heatmap_mxy,
                self.spatial_heatmap_mxy_item,
                getattr(self, "spatial_heatmap_mxy_colorbar", None),
                np.abs(mxy_time),
                f"|Mxy| vs time @ freq {freq_sel}",
            )
            _set_heatmap(
                self.spatial_heatmap_mz,
                self.spatial_heatmap_mz_item,
                getattr(self, "spatial_heatmap_mz_colorbar", None),
                mz_time,
                f"Mz vs time @ freq {freq_sel}",
            )
        except Exception as exc:
            self.log_message(f"Spatial time heatmap update failed: {exc}")

    def _load_sequence_presets(self, seq_type: str):
        """Load sequence-specific parameter presets if enabled."""
        if not self.tissue_widget.sequence_presets_enabled:
            return

        presets = self.sequence_designer.get_sequence_preset_params(seq_type)
        if not presets:
            return

        # List of all widgets that might be updated
        widgets_to_block = [
            self.sequence_designer.te_spin,
            self.sequence_designer.tr_spin,
            self.sequence_designer.ti_spin,
            self.rf_designer.flip_angle,
            self.rf_designer.b1_amplitude,
            self.sequence_designer.ssfp_repeats,
            self.sequence_designer.ssfp_start_tr,
            self.sequence_designer.ssfp_use_ratios,
            self.sequence_designer.ssfp_tr_ratio,
            self.sequence_designer.ssfp_flip_ratio,
            self.sequence_designer.ssfp_start_flip,
            self.sequence_designer.ssfp_start_phase,
            self.sequence_designer.ssfp_alternate_phase,
            self.rf_designer.pulse_type,
            self.pos_spin,
            self.pos_range,
            self.freq_spin,
            self.freq_range,
            self.rf_designer.duration,
            self.rf_designer.phase,
            self.time_step_spin,
        ]

        # Block signals temporarily to avoid triggering diagram updates multiple times
        for widget in widgets_to_block:
            widget.blockSignals(True)

        # Apply presets
        if "pulse_type" in presets:
            self.rf_designer.pulse_type.setCurrentText(presets["pulse_type"])
        if "te_ms" in presets:
            self.sequence_designer.te_spin.setValue(presets["te_ms"])
        if "tr_ms" in presets:
            self.sequence_designer.tr_spin.setValue(presets["tr_ms"])
        if "ti_ms" in presets:
            self.sequence_designer.ti_spin.setValue(presets["ti_ms"])
        if "flip_angle" in presets:
            self.rf_designer.flip_angle.setValue(presets["flip_angle"])
        if "phase" in presets:
            self.rf_designer.phase.setValue(presets["phase"])
        if "b1_amplitude" in presets:
            self.rf_designer.b1_amplitude.setValue(presets["b1_amplitude"])
        if "duration" in presets:
            self.rf_designer.duration.setValue(presets["duration"])

        # SSFP-specific parameters
        if "ssfp_repeats" in presets:
            self.sequence_designer.ssfp_repeats.setValue(presets["ssfp_repeats"])
        if "ssfp_use_ratios" in presets:
            self.sequence_designer.ssfp_use_ratios.setChecked(
                presets["ssfp_use_ratios"]
            )
        if "ssfp_tr_ratio" in presets:
            self.sequence_designer.ssfp_tr_ratio.setValue(presets["ssfp_tr_ratio"])
        if "ssfp_flip_ratio" in presets:
            self.sequence_designer.ssfp_flip_ratio.setValue(presets["ssfp_flip_ratio"])
        if "ssfp_start_tr" in presets:
            self.sequence_designer.ssfp_start_tr.setValue(presets["ssfp_start_tr"])
        elif "ssfp_start_delay" in presets:  # Backward compatibility
            self.sequence_designer.ssfp_start_tr.setValue(presets["ssfp_start_delay"])
        if "ssfp_start_flip" in presets:
            self.sequence_designer.ssfp_start_flip.setValue(presets["ssfp_start_flip"])
        if "ssfp_start_phase" in presets:
            self.sequence_designer.ssfp_start_phase.setValue(
                presets["ssfp_start_phase"]
            )
        if "ssfp_alternate_phase" in presets:
            self.sequence_designer.ssfp_alternate_phase.setChecked(
                presets["ssfp_alternate_phase"]
            )
        # Optional simulation grid presets
        if "num_positions" in presets:
            self.pos_spin.setValue(int(presets["num_positions"]))
        if "position_range_cm" in presets:
            self.pos_range.setValue(float(presets["position_range_cm"]))
        if "num_frequencies" in presets:
            self.freq_spin.setValue(int(presets["num_frequencies"]))
        if "frequency_range_hz" in presets:
            self.freq_range.setValue(float(presets["frequency_range_hz"]))
        if "time_step" in presets:
            self.time_step_spin.setValue(float(presets["time_step"]))

        # Re-enable signals
        for widget in widgets_to_block:
            widget.blockSignals(False)

        # Force regeneration of the current pulse (Excitation) to match the new presets
        self.rf_designer.update_pulse()

        # Update diagram once with all new values
        self.sequence_designer.update_diagram()

        self.log_message(f"Loaded presets for {seq_type}: {presets}")

    def create_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.setObjectName("menu_file")

        load_action = file_menu.addAction("Load Parameters")
        load_action.setObjectName("action_load_params")
        load_action.triggered.connect(self.load_parameters)

        save_action = file_menu.addAction("Save Parameters")
        save_action.setObjectName("action_save_params")
        save_action.triggered.connect(self.save_parameters)

        file_menu.addSeparator()

        export_action = file_menu.addAction("Export Results")
        export_action.setObjectName("action_export_results")
        export_action.triggered.connect(self.export_results)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.setObjectName("action_exit")
        exit_action.triggered.connect(self.close)

        # Tools/Export menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.setObjectName("menu_tools")

        export_results_action = tools_menu.addAction("Export Results...")
        export_results_action.setObjectName("action_export_results_tools")
        export_results_action.triggered.connect(self.export_results)

        # Tutorials menu
        tut_menu = menubar.addMenu("Tutorials")
        tut_menu.setObjectName("menu_tutorials")

        record_action = tut_menu.addAction("Record New Tutorial...")
        record_action.setObjectName("action_record_tutorial")
        record_action.triggered.connect(self.record_tutorial)

        load_tut_action = tut_menu.addAction("Load Tutorial...")
        load_tut_action.setObjectName("action_load_tutorial")
        load_tut_action.triggered.connect(self.load_tutorial_dialog)

        tut_menu.addSeparator()

        self.stop_tut_action = tut_menu.addAction("Stop Recording/Playback")
        self.stop_tut_action.setObjectName("action_stop_tutorial")
        self.stop_tut_action.setEnabled(False)
        self.stop_tut_action.triggered.connect(self.stop_tutorial)

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.setObjectName("menu_help")

        about_action = help_menu.addAction("About")
        about_action.setObjectName("action_about")
        about_action.triggered.connect(self.show_about)

    def _sync_preview_checkboxes(self, checked: bool):
        """Keep preview checkboxes in sync between footer and status bar."""
        if getattr(self, "_syncing_preview", False):
            return
        self._syncing_preview = True
        try:
            if (
                hasattr(self, "preview_checkbox")
                and self.preview_checkbox.isChecked() != checked
            ):
                self.preview_checkbox.setChecked(checked)
            if (
                hasattr(self, "status_preview_checkbox")
                and self.status_preview_checkbox.isChecked() != checked
            ):
                self.status_preview_checkbox.setChecked(checked)
            if (
                hasattr(self, "toolbar_preview_action")
                and self.toolbar_preview_action.isChecked() != checked
            ):
                was_blocked = self.toolbar_preview_action.blockSignals(True)
                self.toolbar_preview_action.setChecked(checked)
                self.toolbar_preview_action.blockSignals(was_blocked)
        finally:
            self._syncing_preview = False

    def _build_status_run_bar(self):
        """Add always-visible run controls to the status bar."""
        bar = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self.status_run_button = QPushButton("Run Simulation")
        self.status_run_button.setObjectName("run_btn")
        self.status_run_button.clicked.connect(self.run_simulation)
        self.status_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.status_run_button)

        self.status_cancel_button = QPushButton("Cancel")
        self.status_cancel_button.setObjectName("cancel_btn")
        self.status_cancel_button.clicked.connect(self.cancel_simulation)
        self.status_cancel_button.setEnabled(False)
        self.status_cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.status_cancel_button)

        self.status_preview_checkbox = QCheckBox("Preview")
        initial_preview = False
        if hasattr(self, "preview_checkbox"):
            initial_preview = self.preview_checkbox.isChecked()
        self.status_preview_checkbox.setChecked(initial_preview)
        self.status_preview_checkbox.toggled.connect(
            lambda val: self._sync_preview_checkboxes(val)
        )
        layout.addWidget(self.status_preview_checkbox)

        self.status_progress = QProgressBar()
        self.status_progress.setFixedWidth(180)
        initial_progress = 0
        if hasattr(self, "progress_bar"):
            initial_progress = self.progress_bar.value()
        self.status_progress.setValue(initial_progress)
        layout.addWidget(self.status_progress)

        layout.addStretch()
        bar.setLayout(layout)
        self.statusBar().addPermanentWidget(bar, 1)
        self.status_run_bar = bar

    def _build_toolbar_run_bar(self):
        """Add a top toolbar with run controls to keep them visible."""
        tb = QToolBar("Run Controls")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setFloatable(False)

        self.toolbar_run_action = tb.addAction("Run Simulation")
        self.toolbar_run_action.setObjectName("action_toolbar_run")
        self.toolbar_run_action.triggered.connect(self.run_simulation)

        self.toolbar_cancel_action = tb.addAction("Cancel")
        self.toolbar_cancel_action.setObjectName("action_toolbar_cancel")
        self.toolbar_cancel_action.triggered.connect(self.cancel_simulation)
        self.toolbar_cancel_action.setEnabled(False)

        self.toolbar_preview_action = tb.addAction("Preview")
        self.toolbar_preview_action.setObjectName("action_toolbar_preview")
        self.toolbar_preview_action.setCheckable(True)
        initial_preview = False
        if hasattr(self, "preview_checkbox"):
            initial_preview = self.preview_checkbox.isChecked()
        self.toolbar_preview_action.setChecked(initial_preview)
        self.toolbar_preview_action.toggled.connect(
            lambda val: self._sync_preview_checkboxes(val)
        )

        tb.addSeparator()

        self.toolbar_export_button = QToolButton()
        self.toolbar_export_button.setObjectName("toolbar_export_btn")
        self.toolbar_export_button.setText("Export Results")
        self.toolbar_export_button.clicked.connect(self.export_results)
        tb.addWidget(self.toolbar_export_button)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tb.addWidget(spacer)

        self.toolbar_progress = QProgressBar()
        self.toolbar_progress.setFixedWidth(160)
        initial_progress = 0
        if hasattr(self, "progress_bar"):
            initial_progress = self.progress_bar.value()
        self.toolbar_progress.setValue(initial_progress)
        tb.addWidget(self.toolbar_progress)

        self.addToolBar(tb)
        self.toolbar_run_bar = tb

    def run_simulation(self):
        """Run the Bloch simulation."""
        self.statusBar().showMessage("Running simulation...")
        self.simulate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(False)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(True)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(False)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(True)
        self.progress_bar.setValue(0)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(0)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(0)
        if hasattr(self, "status_preview_checkbox"):
            self._sync_preview_checkboxes(self.status_preview_checkbox.isChecked())

        self.log_message("Starting simulation...")

        tissue = self.tissue_widget.get_parameters()
        pulse = self.rf_designer.get_pulse()
        dt_s = max(self.time_step_spin.value(), 0.1) * 1e-6
        sequence_tuple = self.sequence_designer.compile_sequence(
            custom_pulse=pulse, dt=dt_s, log_info=True
        )
        tail_ms = self.extra_tail_spin.value()
        b1_orig_len = len(sequence_tuple[0])
        sequence_tuple = self._extend_sequence_with_tail(sequence_tuple, tail_ms, dt_s)
        if len(sequence_tuple[0]) > b1_orig_len:
            added = len(sequence_tuple[0]) - b1_orig_len
            self.log_message(
                f"Appended {tail_ms:.3f} ms tail ({added} pts) after sequence."
            )

        b1_arr, gradients_arr, time_arr = sequence_tuple
        if not self._sweep_mode:
            self.sequence_designer._render_sequence_diagram(
                b1_arr, gradients_arr, time_arr
            )
        self.last_b1 = np.asarray(b1_arr)
        self.last_gradients = np.asarray(gradients_arr)
        self.last_time = np.asarray(time_arr)
        b1_abs = np.abs(self.last_b1)
        thr = b1_abs.max() * 1e-3 if b1_abs.size else 0.0
        mask = b1_abs > thr
        if mask.any():
            idx = np.where(mask)[0]
            self.last_pulse_range = (self.last_time[idx[0]], self.last_time[idx[-1]])
        else:
            self.last_pulse_range = None

        npos = self.pos_spin.value()
        pos_span_cm = self.pos_range.value()
        span_m = pos_span_cm / 100.0
        half_span = span_m / 2.0
        positions = np.zeros((npos, 3))
        if npos > 1:
            positions[:, 2] = np.linspace(-half_span, half_span, npos)

        nfreq = self.freq_spin.value()
        freq_range = self.freq_range.value()
        if nfreq > 1 and freq_range <= 0:
            freq_range = max(1.0, nfreq - 1)
            self.freq_range.setValue(freq_range)
        if nfreq > 1:
            frequencies = np.linspace(-freq_range / 2, freq_range / 2, nfreq)
        else:
            frequencies = np.array([0.0])
        mode = 2 if self.mode_combo.currentText() == "Time-resolved" else 0

        preview_on = self.preview_checkbox.isChecked() or (
            hasattr(self, "status_preview_checkbox")
            and self.status_preview_checkbox.isChecked()
        )
        if preview_on:
            prev_stride = max(1, int(np.ceil(npos / 64)))
            freq_stride = max(1, int(np.ceil(nfreq / 16)))
            if prev_stride > 1:
                positions = positions[::prev_stride]
                npos = positions.shape[0]
            if freq_stride > 1:
                frequencies = frequencies[::freq_stride]
                nfreq = frequencies.shape[0]
            mode = 0
            dt_s *= 4

        m0 = self.tissue_widget.get_initial_mz()
        self.initial_mz = abs(m0) if np.isfinite(m0) else 1.0
        nfnpos = nfreq * npos
        m_init = np.zeros((3, nfnpos))
        m_init[2, :] = m0
        self.mag_3d.set_length_scale(self.initial_mz)

        self.last_positions = positions
        self.last_frequencies = frequencies

        self.simulation_thread = SimulationThread(
            self.simulator,
            sequence_tuple,
            tissue,
            positions,
            frequencies,
            mode,
            dt=dt_s,
            m_init=m_init,
        )
        self.simulation_thread.finished.connect(self.on_simulation_finished)
        self.simulation_thread.cancelled.connect(self.on_simulation_cancelled)
        self.simulation_thread.error.connect(self.on_simulation_error)
        self.simulation_thread.progress.connect(self.progress_bar.setValue)
        if hasattr(self, "status_progress"):
            self.simulation_thread.progress.connect(self.status_progress.setValue)
        self.simulation_thread.start()

    def cancel_simulation(self):
        """Request cancellation of the current simulation."""
        if hasattr(self, "simulation_thread") and self.simulation_thread.isRunning():
            self.simulation_thread.request_cancel()
            self.statusBar().showMessage("Cancellation requested...")
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)

    def on_simulation_finished(self, result):
        """Handle simulation completion."""
        self.statusBar().showMessage("Simulation complete")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(100)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(True)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(100)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(True)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(False)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(100)

        if result.get("mx", {}).ndim > 2:
            self._precompute_plot_ranges(result)
        self.mag_3d.last_positions = self.last_positions
        self.mag_3d.last_frequencies = self.last_frequencies

        # Prepare B1 for playback
        if hasattr(self, "last_b1") and self.last_b1 is not None:
            self.anim_b1 = np.asarray(self.last_b1)
            max_b1 = (
                float(np.nanmax(np.abs(self.anim_b1))) if self.anim_b1.size else 0.0
            )
            self.anim_b1_scale = 1.5 / max(
                max_b1, 1e-6
            )  # Scale to be slightly longer than unit sphere
        else:
            self.anim_b1 = None
            self.anim_b1_scale = 1.0

        self.update_plots(result)

    def on_simulation_error(self, error_msg):
        """Handle simulation error."""
        self.statusBar().showMessage("Simulation failed")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(True)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(0)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(True)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(False)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(0)
        QMessageBox.critical(self, "Simulation Error", error_msg)

    def on_simulation_cancelled(self):
        """Handle user cancellation."""
        self.statusBar().showMessage("Simulation cancelled")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(True)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(0)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(True)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(False)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(0)

    def update_plots(self, result):
        """Update all visualization plots."""
        if self._plotting_in_progress:
            return
        self._plotting_in_progress = True

        try:
            self._invalidate_plot_caches()

            raw_time = result["time"]
            pos_len = (
                self.last_positions.shape[0]
                if self.last_positions is not None
                else result["mx"].shape[-2] if result["mx"].ndim == 3 else 1
            )
            freq_len = (
                self.last_frequencies.shape[0]
                if self.last_frequencies is not None
                else result["mx"].shape[-1] if result["mx"].ndim == 3 else 1
            )

            mx_arr = self._reshape_to_tpf(result["mx"], pos_len, freq_len)
            my_arr = self._reshape_to_tpf(result["my"], pos_len, freq_len)
            mz_arr = self._reshape_to_tpf(result["mz"], pos_len, freq_len)
            signal_arr = result["signal"]
            if signal_arr.ndim == 3:
                signal_arr = self._reshape_to_tpf(signal_arr, pos_len, freq_len)

            self.last_result = result.copy()
            self.last_result["mx"] = mx_arr
            self.last_result["my"] = my_arr
            self.last_result["mz"] = mz_arr
            self.last_result["signal"] = signal_arr
            try:
                self._compute_final_spectrum_range(signal_arr, self.last_time)
            except Exception:
                self._spectrum_final_range = None

            def _get_zero_idx(arr, length):
                if arr is None or len(arr) == 0:
                    return 0
                if len(arr) != length:
                    return 0
                return int(np.argmin(np.abs(arr)))

            f_idx = _get_zero_idx(self.last_frequencies, freq_len)
            if hasattr(self, "spatial_freq_slider"):
                self.spatial_freq_slider.setValue(f_idx)

            p_idx = _get_zero_idx(
                (
                    self.last_positions[:, 2] * 100
                    if self.last_positions is not None
                    else None
                ),
                pos_len,
            )
            if hasattr(self, "spectrum_pos_slider"):
                self.spectrum_pos_slider.setValue(p_idx)

            if hasattr(self, "mag_view_selector"):
                self.mag_view_selector.setValue(0)

            if hasattr(self, "signal_view_selector"):
                self.signal_view_selector.setValue(0)

            ntime = mx_arr.shape[0]
            time_ms = self._normalize_time_length(raw_time, ntime) * 1000
            self.last_time = self._normalize_time_length(raw_time, ntime)
            if len(time_ms) == 0:
                return
            x_min, x_max = time_ms[0], time_ms[-1]

            if getattr(self, "_sweep_mode", False):
                self._spectrum_needs_update = True
                self._spatial_needs_update = True
                return

            if mx_arr.ndim == 2:
                # Use common line rendering logic but extracted to method
                self._render_mag_lines()
                self._render_signal_lines()

                self.mag_3d.update_magnetization(
                    mx_arr.flatten(), my_arr.flatten(), mz_arr.flatten()
                )
                self.anim_timer.stop()

                self.update_spatial_plot_from_last_result()
                self.time_control.setEnabled(False)
            else:
                mx_all = mx_arr
                my_all = my_arr
                mz_all = mz_arr
                ntime, npos, nfreq = mx_all.shape
                mean_only = self.mean_only_checkbox.isChecked()

                self._update_mag_selector_limits(npos, nfreq, disable=mean_only)
                self._update_signal_selector_limits(npos, nfreq, disable=mean_only)

                # These methods now handle their own Line vs Heatmap logic
                self._refresh_mag_plots()
                self._refresh_signal_plots()

                self.anim_vectors_full = np.stack([mx_all, my_all, mz_all], axis=3)
                self.playback_indices = self._build_playback_indices(ntime)
                self.playback_time = self.last_time[self.playback_indices]
                self.playback_time_ms = self.playback_time * 1000.0

                # Prepare B1 for playback
                if hasattr(self, "last_b1") and self.last_b1 is not None:
                    self.anim_b1 = np.asarray(self.last_b1)[self.playback_indices]
                    max_b1 = (
                        float(np.nanmax(np.abs(self.anim_b1)))
                        if self.anim_b1.size
                        else 0.0
                    )
                    self.anim_b1_scale = 1.5 / max(max_b1, 1e-6)
                else:
                    self.anim_b1 = None
                    self.anim_b1_scale = 1.0

                self.mag_3d.set_selector_limits(npos, nfreq, disable=mean_only)
                self.anim_index = 0
                self._refresh_vector_view(mean_only=mean_only)
                self.time_control.set_time_range(self.playback_time)
                self.time_control.setEnabled(True)

            self.update_spatial_plot_from_last_result()
        finally:
            self._plotting_in_progress = False

    def _precompute_plot_ranges(self, result):
        """Pre-calculate stable Y-ranges for plots based on the full dataset."""
        mx = result.get("mx")
        my = result.get("my")
        mz = result.get("mz")
        signal = result.get("signal")
        initial_mag = self.initial_mz

        if mx is not None and my is not None and mz is not None:
            max_abs_mxy = float(np.nanmax(np.abs(np.sqrt(mx**2 + my**2))))
            mxy_limit = max(initial_mag, max_abs_mxy) * 1.2

            # Spatial transverse components extrema
            mx_min = float(np.nanmin(mx))
            mx_max = float(np.nanmax(mx))
            my_min = float(np.nanmin(my))
            my_max = float(np.nanmax(my))

            spatial_trans_min = min(mx_min, my_min, -0.05 * mxy_limit)
            spatial_trans_max = max(mx_max, my_max, mxy_limit)
            self.spatial_mxy_yrange = (spatial_trans_min, spatial_trans_max)

            # Spatial longitudinal extrema
            mz_min = float(np.nanmin(mz))
            mz_max = float(np.nanmax(mz))
            mz_limit_min = min(mz_min, -initial_mag) * 1.1
            mz_limit_max = max(mz_max, initial_mag) * 1.1
            self.spatial_mz_yrange = (mz_limit_min, mz_limit_max)

        # Precompute extrema for Spectrum view (constant limits)
        if signal is not None and mz is not None:
            # We want global min/max across all positions and frequencies
            self.spectrum_global_extrema = {
                "Magnitude": (0.0, max(initial_mag, float(np.nanmax(np.abs(signal))))),
                "Real": (
                    float(np.nanmin(np.real(signal))),
                    float(np.nanmax(np.real(signal))),
                ),
                "Imaginary": (
                    float(np.nanmin(np.imag(signal))),
                    float(np.nanmax(np.imag(signal))),
                ),
                "Mz": (float(np.nanmin(mz)), float(np.nanmax(mz))),
            }
            # Add initial Mz to Mz range to ensure it's visible
            mz_min, mz_max = self.spectrum_global_extrema["Mz"]
            self.spectrum_global_extrema["Mz"] = (
                min(mz_min, -initial_mag),
                max(mz_max, initial_mag),
            )

    def load_parameters(self):
        """Load simulation parameters from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Parameters",
            str(self._get_export_directory()),
            "JSON Files (*.json)",
        )
        if not filename:
            return

        try:
            with open(filename, "r") as f:
                state = json.load(f)

            if "tissue" in state:
                if hasattr(self.tissue_widget, "set_state"):
                    self.tissue_widget.set_state(state["tissue"])
                else:
                    # Fallback for old version or missing method
                    tw = self.tissue_widget
                    t_state = state["tissue"]
                    if "preset" in t_state:
                        tw.preset_combo.setCurrentText(t_state["preset"])
                    if "field" in t_state:
                        tw.field_combo.setCurrentText(t_state["field"])
                    if "t1_ms" in t_state:
                        tw.t1_spin.setValue(t_state["t1_ms"])
                    if "t2_ms" in t_state:
                        tw.t2_spin.setValue(t_state["t2_ms"])
                    if "m0" in t_state:
                        tw.m0_spin.setValue(t_state["m0"])

            if "rf" in state:
                self.rf_designer.set_state(state["rf"])

            if "sequence" in state:
                if hasattr(self.sequence_designer, "set_state"):
                    self.sequence_designer.set_state(state["sequence"])
                else:
                    # Fallback
                    sd = self.sequence_designer
                    s_state = state["sequence"]
                    if "type" in s_state:
                        sd.sequence_type.setCurrentText(s_state["type"])
                    if "te" in s_state:
                        sd.te_spin.setValue(s_state["te"])
                    if "tr" in s_state:
                        sd.tr_spin.setValue(s_state["tr"])
                    if "ti" in s_state:
                        sd.ti_spin.setValue(s_state["ti"])
                    if "echo_count" in s_state:
                        sd.spin_echo_echoes.setValue(s_state["echo_count"])
                    if "slice_thickness" in s_state:
                        sd.slice_thickness_spin.setValue(s_state["slice_thickness"])
                    if "slice_gradient" in s_state:
                        sd.slice_gradient_spin.setValue(s_state["slice_gradient"])

            if "simulation" in state:
                sim = state["simulation"]
                if "mode" in sim:
                    self.mode_combo.setCurrentText(sim["mode"])
                if "num_pos" in sim:
                    self.pos_spin.setValue(sim["num_pos"])
                if "pos_range" in sim:
                    self.pos_range.setValue(sim["pos_range"])
                if "num_freq" in sim:
                    self.freq_spin.setValue(sim["num_freq"])
                if "freq_range" in sim:
                    self.freq_range.setValue(sim["freq_range"])
                if "time_step" in sim:
                    self.time_step_spin.setValue(sim["time_step"])
                if "extra_tail" in sim:
                    self.extra_tail_spin.setValue(sim["extra_tail"])
                if "max_traces" in sim:
                    self.max_traces_spin.setValue(sim["max_traces"])

            self.log_message(f"Parameters loaded from {Path(filename).name}")
            self.statusBar().showMessage(
                f"Parameters loaded from {Path(filename).name}"
            )

            # Trigger diagram update
            self.sequence_designer.update_diagram()

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Failed to load parameters:\n{str(e)}"
            )
            self.log_message(f"Error loading parameters: {e}")

    def _animate_vector(self):
        """Advance the 3D vector animation if data is available."""
        if (
            self.anim_data is None
            or self.playback_time_ms is None
            or len(self.playback_time_ms) == 0
        ):
            return
        now = time.monotonic()
        if self._last_render_wall is not None:
            delta_ms = (now - self._last_render_wall) * 1000.0
            if delta_ms < self._min_render_interval_ms:
                return
        if self._playback_anchor_wall is None or self._playback_anchor_time_ms is None:
            self._reset_playback_anchor(self.anim_index)
        if len(self.playback_time_ms) == 1:
            self.time_control.set_time_index(0)
            self._on_universal_time_changed(0)
            return

        elapsed_s = max(0.0, now - (self._playback_anchor_wall or now))
        sim_ms_per_s = max(self.time_control.speed_spin.value(), 0.001)

        start_ms = float(self.playback_time_ms[0])
        end_ms = float(self.playback_time_ms[-1])
        duration_ms = max(end_ms - start_ms, 1e-9)

        target_ms = (
            float(self._playback_anchor_time_ms or start_ms) + elapsed_s * sim_ms_per_s
        )
        wrapped = False
        if target_ms > end_ms:
            # Loop seamlessly: wrap to start while preserving speed
            target_rel = (target_ms - start_ms) % duration_ms
            target_ms = start_ms + target_rel
            wrapped = True
            self._playback_anchor_wall = now
            self._playback_anchor_time_ms = target_ms

        idx = int(np.searchsorted(self.playback_time_ms, target_ms, side="left"))
        idx = min(max(idx, 0), len(self.playback_time_ms) - 1)
        if idx > 0:
            prev_ms = float(self.playback_time_ms[idx - 1])
            curr_ms = float(self.playback_time_ms[idx])
            if abs(target_ms - prev_ms) < abs(curr_ms - target_ms):
                idx -= 1

        if wrapped and self.mag_3d.track_path:
            self.mag_3d._clear_path()

        # Move universal time control (label/slider) then propagate to all views
        self.time_control.set_time_index(idx)
        # During animation, skip expensive plot redraws (spatial FFT, spectrum FFT)
        # Only update time cursors and 3D view for smooth playback
        self._on_universal_time_changed(
            idx, skip_expensive_updates=True, reset_anchor=False
        )
        self._last_render_wall = now

    def _update_b1_arrow(self, playback_idx: int):
        """Update the B1 direction/length indicator in the 3D view."""
        if self.anim_b1 is None or self.mag_3d is None:
            return
        idx = int(max(0, min(playback_idx, len(self.anim_b1) - 1)))
        b1_val = self.anim_b1[idx]
        mag = abs(b1_val)
        if not np.isfinite(mag) or mag < 1e-9:
            self.mag_3d.b1_arrow.setVisible(False)
            return
        phase = np.angle(b1_val)
        tip = np.array(
            [
                self.anim_b1_scale * mag * np.cos(phase),
                self.anim_b1_scale * mag * np.sin(phase),
                0.0,
            ]
        )
        pos = np.array([[0.0, 0.0, 0.0], tip])
        self.mag_3d.b1_arrow.setData(pos=pos)
        self.mag_3d.b1_arrow.setVisible(True)

    def save_parameters(self):
        """Save simulation parameters to file."""
        export_dir = self._get_export_directory()
        seq_type = (
            self.sequence_designer.sequence_type.currentText()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "")
            .lower()
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_filename = f"bloch_params_{seq_type}_{timestamp}.json"
        default_path = export_dir / default_filename

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", str(default_path), "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            state = {
                "version": "1.1",
                "timestamp": timestamp,
                "tissue": self.tissue_widget.get_state(),
                "rf": self.rf_designer.get_state(),
                "sequence": self.sequence_designer.get_state(),
                "simulation": self._collect_simulation_parameters(internal_format=True),
            }

            with open(filename, "w") as f:
                json.dump(state, f, indent=2)

            self.log_message(f"Parameters saved to {Path(filename).name}")
            self.statusBar().showMessage(f"Parameters saved to {Path(filename).name}")

        except Exception as e:
            QMessageBox.critical(
                self, "Save Error", f"Failed to save parameters:\n{str(e)}"
            )
            self.log_message(f"Error saving parameters: {e}")

    def export_results(self):
        """Export simulation results and parameters using the new multi-format dialog."""
        if self.last_result is None and not hasattr(self, "last_b1"):
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        # Create the dialog
        dialog = ExportDataDialog(
            self,
            default_filename="simulation_results",
            default_directory=self._get_export_directory(),
            has_time_resolved=(self.last_time is not None and len(self.last_time) > 1),
        )

        if dialog.exec_() != QDialog.Accepted:
            return

        options = dialog.get_export_options()
        base_path = options["base_path"]

        # Collect complete metadata
        all_metadata = self._collect_all_parameters()
        sequence_params = all_metadata["sequence"]
        simulation_params = all_metadata["simulation"]

        # 1. Image Export (Current Tab)
        if options["image"]:
            fmt = options["image_format"]
            current_tab_text = self.tab_widget.tabText(self.tab_widget.currentIndex())
            img_path = f"{base_path}_view.{fmt}"
            try:
                if current_tab_text == "Magnetization":
                    self._export_magnetization_image(default_format=fmt)
                elif current_tab_text == "3D Vector":
                    self.mag_3d._export_3d_screenshot(format=fmt)
                elif current_tab_text == "Signal":
                    self._export_signal_image(default_format=fmt)
                elif current_tab_text == "Spectrum":
                    self._export_spectrum_image(default_format=fmt)
                elif current_tab_text == "Spatial":
                    self._export_spatial_image(default_format=fmt)
                else:
                    # Generic widget grab if not a specific plot tab
                    exporter = ImageExporter()
                    exporter.export_widget_screenshot(
                        self.tab_widget.currentWidget(), img_path, format=fmt
                    )
                    self.log_message(f"Exported view image: {img_path}")
            except Exception as e:
                self.log_message(f"Failed to export image: {e}")

        # 2. Animation Export (Current Tab)
        if options["animation"]:
            current_tab_text = self.tab_widget.tabText(self.tab_widget.currentIndex())
            try:
                # We use the existing specialized animation methods
                if current_tab_text == "Magnetization":
                    self._export_magnetization_animation()
                elif current_tab_text == "3D Vector":
                    # Note: include_sequence is passed through the specialized dialog usually
                    # but here we'll just trigger the existing method
                    self._export_3d_animation()
                elif current_tab_text == "Signal":
                    self._export_signal_animation()
                elif current_tab_text == "Spectrum":
                    self._export_spectrum_animation()
                elif current_tab_text == "Spatial":
                    self._export_spatial_animation()
                else:
                    self.log_message(
                        f"Animation export not supported for tab: {current_tab_text}"
                    )
            except Exception as e:
                self.log_message(f"Failed to export animation: {e}")

        # 3. HDF5 Export
        h5_path = f"{base_path}.h5"
        if options["hdf5"]:
            try:
                self.simulator.save_results(h5_path, sequence_params, simulation_params)
                self.log_message(f"Exported HDF5: {h5_path}")
            except Exception as e:
                QMessageBox.critical(
                    self, "HDF5 Export Error", f"Failed to export HDF5:\n{str(e)}"
                )

        # 4. Notebook Exports
        if options["notebook_analysis"] or options["notebook_repro"]:
            try:
                from ..notebook_exporter import export_notebook
            except ImportError:
                QMessageBox.warning(
                    self,
                    "Missing Dependency",
                    "Notebook export requires 'nbformat'.\nPlease install it: pip install nbformat",
                )
            else:
                tissue_params = all_metadata["tissue"]

                if options["notebook_analysis"]:
                    nb_path = f"{base_path}_analysis.ipynb"
                    try:
                        # Ensure HDF5 exists or was created
                        export_notebook(
                            mode="load_data",
                            filename=nb_path,
                            sequence_params=sequence_params,
                            simulation_params=simulation_params,
                            tissue_params=tissue_params,
                            h5_filename=Path(h5_path).name,
                            title="Bloch Simulation Analysis",
                        )
                        self.log_message(f"Exported Analysis Notebook: {nb_path}")
                    except Exception as e:
                        self.log_message(f"Failed to export analysis notebook: {e}")

                if options["notebook_repro"]:
                    nb_path = f"{base_path}_repro.ipynb"
                    wf_path = f"{base_path}_waveforms.npz"
                    rf_waveform = None
                    if hasattr(self, "last_b1") and self.last_b1 is not None:
                        rf_waveform = (self.last_b1, self.last_time)

                    try:
                        export_notebook(
                            mode="resimulate",
                            filename=nb_path,
                            sequence_params=sequence_params,
                            simulation_params=simulation_params,
                            tissue_params=tissue_params,
                            rf_waveform=rf_waveform,
                            title="Bloch Simulation - Reproducible",
                            waveform_filename=wf_path,
                        )
                        self.log_message(f"Exported Repro Notebook: {nb_path}")
                        self.log_message(f"Exported Waveforms: {wf_path}")
                    except Exception as e:
                        self.log_message(f"Failed to export repro notebook: {e}")

        # 5. CSV/Text Export
        if options["csv"]:
            fmt = options["csv_format"]
            csv_path = (
                f"{base_path}_data.{fmt}" if fmt != "npy" else f"{base_path}_data.npy"
            )
            try:
                if self.last_result:
                    time_arr = self.last_result.get("time")
                    mx = self.last_result.get("mx")
                    my = self.last_result.get("my")
                    mz = self.last_result.get("mz")
                    pos = self.last_result.get("positions")
                    freq = self.last_result.get("frequencies")

                    if time_arr is not None and mx is not None:
                        self.dataset_exporter.export_magnetization(
                            time_arr,
                            mx,
                            my,
                            mz,
                            pos,
                            freq,
                            csv_path,
                            format=fmt,
                            metadata=all_metadata,
                        )
                        self.log_message(f"Exported Data ({fmt}): {csv_path}")
            except Exception as e:
                self.log_message(f"Failed to export CSV/Text data: {e}")

        self.statusBar().showMessage("Export completed")

    def _export_final_state_data(self):
        """Export the final magnetization state (Mx, My, Mz) for all positions/frequencies."""
        if self.last_result is None:
            return

        # Get data
        mx = self.last_result.get("mx")
        my = self.last_result.get("my")
        mz = self.last_result.get("mz")

        if mx is None:
            return

        # Extract final state
        if mx.ndim == 3:  # (ntime, npos, nfreq)
            mx_final = mx[-1]
            my_final = my[-1]
            mz_final = mz[-1]
        else:
            mx_final = mx
            my_final = my
            mz_final = mz

        # Prompt for file
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Final State", "", "CSV Files (*.csv);;NumPy Archive (*.npz)"
        )
        if not filename:
            return

        path = Path(filename)
        if "csv" in selected_filter.lower():
            if path.suffix.lower() != ".csv":
                path = path.with_suffix(".csv")

            import csv

            try:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Pos_Index",
                            "Freq_Index",
                            "Position_m",
                            "Frequency_Hz",
                            "Mx",
                            "My",
                            "Mz",
                            "Mxy_Complex",
                        ]
                    )

                    npos, nfreq = mx_final.shape
                    positions = (
                        self.last_positions[:, 2]
                        if self.last_positions is not None
                        else np.zeros(npos)
                    )
                    frequencies = (
                        self.last_frequencies
                        if self.last_frequencies is not None
                        else np.zeros(nfreq)
                    )

                    for p in range(npos):
                        pos_val = positions[p] if p < len(positions) else 0
                        for f_idx in range(nfreq):
                            freq_hz_val = (
                                frequencies[f_idx] if f_idx < len(frequencies) else 0
                            )
                            val_mx = float(mx_final[p, f_idx])
                            val_my = float(my_final[p, f_idx])
                            val_mz = float(mz_final[p, f_idx])
                            val_mxy_complex = complex(val_mx, val_my)

                            writer.writerow(
                                [
                                    p,
                                    f_idx,
                                    pos_val,
                                    freq_hz_val,
                                    val_mx,
                                    val_my,
                                    val_mz,
                                    str(val_mxy_complex),
                                ]
                            )

                self.statusBar().showMessage(f"Final state exported to {path}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Final state data saved to:\n{path.name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export CSV:\n{str(e)}"
                )

        else:  # NPZ
            if path.suffix.lower() != ".npz":
                path = path.with_suffix(".npz")

            mxy_complex = mx_final + 1j * my_final
            try:
                np.savez(
                    path,
                    mx=mx_final,
                    my=my_final,
                    mz=mz_final,
                    mxy=mxy_complex,
                    positions=self.last_positions,
                    frequencies=self.last_frequencies,
                )
                self.statusBar().showMessage(f"Final state exported to {path}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Final state data saved to:\n{path.name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export NPZ:\n{str(e)}"
                )

    def _export_full_simulation_data(self):
        """Export full simulation data arrays."""
        if self.last_result is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Full Simulation Data", "", "NumPy Archive (*.npz)"
        )
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")

        mx = self.last_result.get("mx")
        my = self.last_result.get("my")
        mxy = None
        if mx is not None and my is not None:
            mxy = mx + 1j * my

        try:
            # Save all arrays
            np.savez(
                path,
                time=self.last_time,
                mx=mx,
                my=my,
                mz=self.last_result.get("mz"),
                mxy=mxy,
                signal=self.last_result.get("signal"),
                positions=self.last_positions,
                frequencies=self.last_frequencies,
            )

            self.statusBar().showMessage(f"Full simulation data exported to {path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Full simulation data saved to:\n{path.name}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export NPZ:\n{str(e)}"
            )

    def _collect_sequence_parameters(self):
        """Collect all pulse sequence parameters from GUI."""
        # Use the state from sequence designer as base
        params = self.sequence_designer.get_state()

        # Add additional computed/internal parameters for export
        params["te_s"] = params["te"] / 1000.0
        params["tr_s"] = params["tr"] / 1000.0
        params["ti_s"] = params["ti"] / 1000.0

        # RF pulse parameters from designer
        rf_state = self.rf_designer.get_state()
        params["rf_pulse_type"] = rf_state.get("pulse_type", "sinc")
        params["rf_flip_angle"] = rf_state.get("flip_angle", 90.0)
        params["rf_duration_s"] = rf_state.get("duration", 1.0) / 1000.0
        params["rf_time_bw_product"] = rf_state.get("time_bw_product", 4.0)
        params["rf_phase"] = rf_state.get("phase", 0.0)
        params["rf_freq_offset"] = rf_state.get("freq_offset", 0.0)

        # Store RF waveform if available
        if hasattr(self, "last_b1") and self.last_b1 is not None:
            params["b1_waveform"] = self.last_b1
            params["time_waveform"] = self.last_time
            if hasattr(self, "last_gradients"):
                params["gradients_waveform"] = self.last_gradients

        return params

    def _collect_simulation_parameters(self, internal_format=False):
        """Collect all simulation parameters from GUI."""
        if internal_format:
            return {
                "mode": self.mode_combo.currentText(),
                "num_pos": self.pos_spin.value(),
                "pos_range": self.pos_range.value(),
                "num_freq": self.freq_spin.value(),
                "freq_range": self.freq_range.value(),
                "time_step": self.time_step_spin.value(),
                "extra_tail": self.extra_tail_spin.value(),
                "max_traces": self.max_traces_spin.value(),
            }

        params = {
            "mode": (
                "time-resolved"
                if self.mode_combo.currentText() == "Time-resolved"
                else "endpoint"
            ),
            "time_step_us": self.time_step_spin.value(),
            "num_positions": self.pos_spin.value(),
            "position_range_cm": self.pos_range.value(),
            "num_frequencies": self.freq_spin.value(),
            "frequency_range_hz": self.freq_range.value(),
            "extra_tail_ms": self.extra_tail_spin.value(),
            "use_parallel": self.simulator.use_parallel,
            "num_threads": self.simulator.num_threads,
            "preview_mode": (
                self.preview_checkbox.isChecked()
                if hasattr(self, "preview_checkbox")
                else False
            ),
        }

        # Include explicit axes if available
        if hasattr(self, "last_positions") and self.last_positions is not None:
            params["position_axis"] = self.last_positions
        else:
            # Reconstruct from range if missing but requested
            npos = self.pos_spin.value()
            pos_span_cm = self.pos_range.value()
            span_m = pos_span_cm / 100.0
            half_span = span_m / 2.0
            positions = np.zeros((npos, 3))
            if npos > 1:
                positions[:, 2] = np.linspace(-half_span, half_span, npos)
            params["position_axis"] = positions

        if hasattr(self, "last_frequencies") and self.last_frequencies is not None:
            params["frequency_axis"] = self.last_frequencies
        else:
            nfreq = self.freq_spin.value()
            freq_range = self.freq_range.value()
            if nfreq > 1:
                frequencies = np.linspace(-freq_range / 2, freq_range / 2, nfreq)
            else:
                frequencies = np.array([0.0])
            params["frequency_axis"] = frequencies

        # Store initial magnetization
        params["initial_mz"] = self.tissue_widget.get_initial_mz()

        return params

    def _collect_all_parameters(self) -> Dict:
        """Collect all parameters (Tissue, Sequence, Simulation) for metadata export."""
        return {
            "tissue": self.tissue_widget.get_state(),
            "sequence": self._collect_sequence_parameters(),
            "simulation": self._collect_simulation_parameters(),
        }

    def _get_export_directory(self):
        """Get or create the default export directory."""
        override = os.environ.get("BLOCH_EXPORT_DIR")
        if override:
            export_dir = Path(override).expanduser()
        else:
            export_dir = get_app_data_dir() / "exports"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to working directory if preferred location is unavailable
            export_dir = Path.cwd()
            export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def _show_not_implemented(self, feature_name):
        """Show a message for features not yet implemented."""
        QMessageBox.information(
            self,
            "Coming Soon",
            f"{feature_name} export will be available in a future update.\n\n"
            "Current available exports:\n"
            "- Static images (PNG, SVG)",
        )

    def _prompt_data_export_path(self, default_name: str):
        """Open a save dialog and return the chosen path and format."""
        export_dir = self._get_export_directory()
        default_path = export_dir / f"{default_name}.csv"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            str(default_path),
            "CSV (*.csv);;NumPy (*.npy);;DAT/TSV (*.dat *.tsv)",
        )
        if not filename:
            return None, None
        fmt = "csv"
        sel = (selected_filter or "").lower()
        path = Path(filename)
        suffix = path.suffix.lower()
        if "npy" in sel or suffix == ".npy":
            fmt = "npy"
            path = path.with_suffix(".npy")
        elif "dat" in sel or "tsv" in sel or suffix in (".dat", ".tsv"):
            fmt = "dat"
            path = path.with_suffix(".dat")
        else:
            fmt = "csv"
            path = path.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, fmt

    def _current_playback_index(self) -> int:
        """Return the current full-resolution time index based on the universal slider."""
        if hasattr(self, "time_control") and self.time_control.time_array is not None:
            playback_idx = int(self.time_control.time_slider.value())
            return int(self._playback_to_full_index(playback_idx))
        if self.last_time is not None:
            return len(self.last_time) - 1
        return 0

    def _calculate_spectrum_data(self, time_idx=None, compute_fft=True):
        """Compute spectrum arrays used for plotting or export."""
        if self.last_result is None:
            return None
        signal = self.last_result.get("signal")
        if signal is None:
            return None
        time_arr = (
            self.last_time
            if self.last_time is not None
            else self.last_result.get("time", None)
        )

        time_arr = np.asarray(time_arr)
        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]
        if time_idx is None:
            time_idx = sig_arr.shape[0] - 1
        time_idx = int(max(0, min(time_idx, sig_arr.shape[0] - 1)))
        sig_slice = sig_arr[: time_idx + 1]

        if time_arr is None or len(time_arr) < 2:
            time_slice = np.arange(sig_slice.shape[0])
            dt = 1.0
        else:
            time_slice = time_arr[: time_idx + 1]
            if len(time_slice) < 2:
                return None
        dt = time_slice[1] - time_slice[0]  # seconds per sample

        spectrum_mode = (
            self.spectrum_view_mode.currentText()
            if hasattr(self, "spectrum_view_mode")
            else "Mean over positions"
        )
        pos_count = sig_slice.shape[1]

        # Fallback for original slider
        pos_sel = (
            min(self.spectrum_pos_slider.value(), pos_count - 1) if pos_count > 0 else 0
        )

        actual_pos_cm = 0.0
        if self.last_positions is not None and pos_sel < len(self.last_positions):
            actual_pos_cm = self.last_positions[pos_sel, 2] * 100
            self.spectrum_pos_label.setText(f"Pos: {actual_pos_cm:.3f} cm")
        else:
            self.spectrum_pos_label.setText(f"Pos idx: {pos_sel}")

        spectrum = None
        spec_mean = None
        freq = None

        if compute_fft:
            if spectrum_mode == "Mean over positions":
                sig_for_fft = np.mean(sig_slice, axis=tuple(range(1, sig_slice.ndim)))
                spec_mean = np.fft.fftshift(
                    np.fft.fft(sig_for_fft, n=self._spectrum_fft_len(len(sig_for_fft)))
                )
            elif spectrum_mode == "Mean + individuals":
                sig_for_fft = np.mean(sig_slice, axis=tuple(range(1, sig_slice.ndim)))
                n_fft = self._spectrum_fft_len(len(sig_for_fft))
                spec_mean = np.fft.fftshift(np.fft.fft(sig_for_fft, n=n_fft))
            else:
                sig_for_fft = np.mean(sig_slice[:, pos_sel, :], axis=1)
                spec_mean = None

            n_fft = self._spectrum_fft_len(len(sig_for_fft))
            spectrum = np.fft.fftshift(np.fft.fft(sig_for_fft, n=n_fft))
            freq = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))

        return {
            "freq": freq,
            "spectrum": spectrum,
            "spec_mean": spec_mean,
            "mode": spectrum_mode,
            "pos_count": pos_count,
            "pos_sel": pos_sel,
            "time_idx": time_idx,
            "time_slice": time_slice,
            "signal_slice": sig_slice,
        }

    def _update_spectrum_heatmap(self, time_idx=None):
        """Render a heatmap of spectra across all position/frequency spins."""
        if self.last_result is None:
            return
        signal = self.last_result.get("signal")
        time_arr = (
            self.last_time
            if self.last_time is not None
            else self.last_result.get("time", None)
        )
        if signal is None or time_arr is None:
            self.log_message("Spectrum heatmap: missing signal or time axis")
            return

        # Determine heatmap mode
        mode_text = (
            self.spectrum_heatmap_mode.currentText()
            if hasattr(self, "spectrum_heatmap_mode")
            else "Spin vs Frequency (FFT)"
        )
        show_evolution = mode_text == "Spin vs Time (Evolution)"

        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]

        ntime = sig_arr.shape[0]
        if ntime < 2:
            self.log_message("Spectrum heatmap: need at least two time points")
            return

        time_arr = np.asarray(time_arr)
        if time_arr.size < 2:
            self.log_message("Spectrum heatmap: invalid time array")
            return

        if time_idx is None:
            time_idx = ntime - 1
        time_idx = int(max(1, min(time_idx, ntime - 1)))

        sig_slice = sig_arr[: time_idx + 1]  # (ntime, npos, nfreq)
        npos, nfreq = sig_arr.shape[1], sig_arr.shape[2]

        if npos > 1 and nfreq == 1:
            y_label = "Position index"
        elif npos == 1 and nfreq > 1:
            y_label = "Frequency index (spin off-res)"
        else:
            y_label = "Spin index (pos×freq)"

        if show_evolution:
            # Direct spin magnitudes over time (no FFT)
            time_ms = time_arr[: time_idx + 1] * 1000.0

            # Reshape to (ntime, spin_count)
            mags = np.abs(sig_slice).reshape(sig_slice.shape[0], -1)
            data = mags.T  # (spin_count, ntime)

            spin_count = data.shape[0]
            try:
                self.spectrum_heatmap_item.setImage(
                    data, autoLevels=True, axisOrder="row-major"
                )
                time_span = float(time_ms[-1] - time_ms[0]) if len(time_ms) > 1 else 1.0
                self.spectrum_heatmap_item.setRect(
                    float(time_ms[0]),
                    0,
                    time_span if time_span != 0 else 1.0,
                    spin_count,
                )
                self.spectrum_heatmap.setLabel("left", y_label)
                self.spectrum_heatmap.setLabel("bottom", "Time", "ms")
                self.spectrum_heatmap.setTitle("Temporal Evolution (|Mxy| over time)")
                self.spectrum_heatmap.setXRange(
                    float(time_ms[0]), float(time_ms[-1]), padding=0
                )
                self.spectrum_heatmap.setYRange(0, spin_count, padding=0)
                if (
                    hasattr(self, "spectrum_heatmap_colorbar")
                    and self.spectrum_heatmap_colorbar is not None
                ):
                    finite_mag = data[np.isfinite(data)]
                    if finite_mag.size:
                        vmin = float(finite_mag.min())
                        vmax = float(finite_mag.max())
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
                            self.spectrum_heatmap_colorbar.setLevels((vmin, vmax))
            except Exception as exc:
                self.log_message(f"Spectrum heatmap update failed: {exc}")
        else:
            # FFT Mode: Stack of spectra
            dt = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 1e-3
            n_fft = self._spectrum_fft_len(sig_slice.shape[0])
            freq_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))

            sig_flat = sig_slice.reshape(sig_slice.shape[0], -1)  # (ntime, spin)
            spec = np.fft.fftshift(
                np.fft.fft(sig_flat, n=n_fft, axis=0), axes=0
            )  # (nfreqbins, spin)
            magnitude = np.abs(spec).T  # (spin, nfreqbins)

            spin_count = magnitude.shape[0]
            try:
                self.spectrum_heatmap_item.setImage(
                    magnitude, autoLevels=True, axisOrder="row-major"
                )
                span = float(freq_axis[-1] - freq_axis[0])
                self.spectrum_heatmap_item.setRect(
                    float(freq_axis[0]), 0, span if span != 0 else 1.0, spin_count
                )
                self.spectrum_heatmap.setLabel("left", y_label)
                self.spectrum_heatmap.setLabel(
                    "bottom", "Frequency (from signal FFT)", "Hz"
                )
                self.spectrum_heatmap.setTitle("Spectra Stack (FFT of signal per spin)")
                self.spectrum_heatmap.setXRange(
                    float(freq_axis[0]), float(freq_axis[-1]), padding=0
                )
                self.spectrum_heatmap.setYRange(0, spin_count, padding=0)
                if (
                    hasattr(self, "spectrum_heatmap_colorbar")
                    and self.spectrum_heatmap_colorbar is not None
                ):
                    finite_mag = magnitude[np.isfinite(magnitude)]
                    if finite_mag.size:
                        vmin = float(finite_mag.min())
                        vmax = float(finite_mag.max())
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
                            self.spectrum_heatmap_colorbar.setLevels((vmin, vmax))
            except Exception as exc:
                self.log_message(f"Spectrum heatmap update failed: {exc}")

    def _toggle_spectrum_3d_mode(self, checked):
        """Toggle between 2D and 3D spectrum visualization."""
        if checked:
            self.spectrum_plot.hide()
            self.spectrum_heatmap_layout.hide()
            self.spectrum_plot_3d.show()
            # Disable other controls that might not apply
            if hasattr(self, "spectrum_plot_type"):
                self.spectrum_plot_type.setEnabled(False)
        else:
            self.spectrum_plot_3d.hide()
            if hasattr(self, "spectrum_plot_type"):
                self.spectrum_plot_type.setEnabled(True)

            # Restore based on plot type
            is_heatmap = self.spectrum_plot_type.currentText() == "Heatmap"
            self.spectrum_plot.setVisible(not is_heatmap)
            self.spectrum_heatmap_layout.setVisible(is_heatmap)

        self._refresh_spectrum()

    def _update_spectrum_3d(self, time_idx=None):
        """Update the 3D spectrum plot."""
        self.spectrum_plot_3d.clear()

        # Add simple grid
        gx = gl.GLGridItem()
        gx.setSize(x=20, y=20, z=20)
        gx.rotate(90, 0, 1, 0)
        self.spectrum_plot_3d.addItem(gx)

        gy = gl.GLGridItem()
        gy.setSize(x=20, y=20, z=20)
        gy.rotate(90, 1, 0, 0)
        self.spectrum_plot_3d.addItem(gy)

        gz = gl.GLGridItem()
        gz.setSize(x=20, y=20, z=20)
        self.spectrum_plot_3d.addItem(gz)

        freqs = None
        data = None

        # 1. Try Off-Resonant (Direct Frequency Axis)
        if self.last_result is not None and self.last_frequencies is not None:
            sig = self.last_result.get("signal")
            if sig is not None:
                sig_arr = np.asarray(sig)
                ntime = sig_arr.shape[0]
                if ntime > 0:
                    if sig_arr.ndim == 1:
                        sig_arr = sig_arr[:, None, None]
                    elif sig_arr.ndim == 2:
                        # Assume (time, freq) or (time, pos)
                        if (
                            self.last_positions is not None
                            and sig_arr.shape[1] == self.last_positions.shape[0]
                        ):
                            sig_arr = sig_arr[:, :, None]  # (time, pos, freq=1)
                        else:
                            sig_arr = sig_arr[:, None, :]  # (time, pos=1, freq)

                    if time_idx is None:
                        time_idx = ntime - 1
                    t_idx = int(max(0, min(time_idx, ntime - 1)))

                    pos_count = sig_arr.shape[1]
                    pos_sel = (
                        min(self.spectrum_pos_slider.value(), pos_count - 1)
                        if pos_count > 0
                        else 0
                    )

                    snapshot = sig_arr[t_idx]  # (npos, nfreq)

                    if pos_count > 0:
                        data = snapshot[pos_sel]
                    else:
                        data = snapshot[0]

                    freqs = np.asarray(self.last_frequencies)

        # 2. If not found, Try FFT
        if data is None:
            # We must force compute_fft=True here to get data for 3D plot
            spec_data = self._calculate_spectrum_data(time_idx, compute_fft=True)
            if spec_data and spec_data.get("spectrum") is not None:
                data = spec_data["spectrum"]
                freqs = spec_data["freq"]

        if data is None or freqs is None or data.size != freqs.size:
            return

        # Prepare Points
        # Normalize Frequency for display [-10, 10]
        f_min, f_max = freqs.min(), freqs.max()
        f_range = f_max - f_min
        if f_range == 0:
            f_range = 1.0

        freq_norm = (freqs - f_min) / f_range * 20.0 - 10.0

        # Scale Magnitude for display
        mag_scale = 5.0

        # x=freq, y=real, z=imag
        pts = np.vstack(
            [freq_norm, np.real(data) * mag_scale, np.imag(data) * mag_scale]
        ).transpose()

        # Main signal line
        line = gl.GLLinePlotItem(pos=pts, color=(0, 1, 1, 1), width=2, antialias=True)
        self.spectrum_plot_3d.addItem(line)

        # Frequency baseline (Real/Imag = 0)
        baseline_pts = np.vstack(
            [freq_norm, np.zeros_like(freqs), np.zeros_like(freqs)]
        ).transpose()
        baseline = gl.GLLinePlotItem(
            pos=baseline_pts, color=(0.5, 0.5, 0.5, 0.5), width=1
        )
        self.spectrum_plot_3d.addItem(baseline)

    def _refresh_spectrum(self, time_idx=None, skip_fft=False):
        """Update spectrum plot using data up to the specified time index."""
        # 3D Mode Check
        if hasattr(self, "spectrum_3d_toggle") and self.spectrum_3d_toggle.isChecked():
            self._update_spectrum_3d(time_idx)
            return

        spec_data = self._calculate_spectrum_data(time_idx, compute_fft=not skip_fft)
        if spec_data is None:
            return

        spectrum_mode = spec_data["mode"]
        pos_count = spec_data["pos_count"]
        pos_sel = spec_data["pos_sel"]
        time_idx = spec_data.get("time_idx", time_idx)

        self.spectrum_pos_slider.setMaximum(max(0, pos_count - 1))
        # Disable slider based on mode and position count instead of hiding
        if hasattr(self, "spectrum_pos_slider"):
            is_individual = spectrum_mode == "Individual position"
            self.spectrum_pos_slider.setEnabled(is_individual and pos_count > 1)
            # Update tooltip to explain why disabled
            if not is_individual:
                self.spectrum_pos_slider.setToolTip(
                    "Switch to 'Individual position' view to select position"
                )
            elif pos_count <= 1:
                self.spectrum_pos_slider.setToolTip("Only one position available")
            else:
                self.spectrum_pos_slider.setToolTip("Select position to view")

        self.spectrum_pos_slider.blockSignals(True)
        self.spectrum_pos_slider.setValue(pos_sel)
        self.spectrum_pos_slider.blockSignals(False)
        plot_type = (
            self.spectrum_plot_type.currentText()
            if hasattr(self, "spectrum_plot_type")
            else "Line"
        )
        is_heatmap = plot_type == "Heatmap"

        # Show/hide heatmap mode selector
        if hasattr(self, "spectrum_heatmap_mode"):
            self.spectrum_heatmap_mode.setVisible(is_heatmap)
            self.spectrum_heatmap_mode_label.setVisible(is_heatmap)

        # Show/hide component selector (only for line plots)
        if hasattr(self, "spectrum_component_combo"):
            self.spectrum_component_combo.setVisible(not is_heatmap)
            if hasattr(self, "spectrum_component_label"):
                self.spectrum_component_label.setVisible(not is_heatmap)

        self.spectrum_plot.setVisible(not is_heatmap)
        if hasattr(self, "spectrum_heatmap"):
            self.spectrum_heatmap_layout.setVisible(is_heatmap)

        # Add colorbar to heatmap
        if is_heatmap and hasattr(self, "spectrum_heatmap_colorbar"):
            self.spectrum_heatmap_layout.addItem(
                self.spectrum_heatmap_colorbar, row=0, col=1
            )

        if is_heatmap:
            self._update_spectrum_heatmap(time_idx=time_idx)
        else:
            self._plot_off_resonant_spins(time_idx=time_idx)

    def _compute_final_spectrum_range(self, signal, time_arr):
        """Compute final-spectrum magnitude range for consistent y-limits."""
        self._spectrum_final_range = None
        if signal is None or time_arr is None:
            return
        time_arr = np.asarray(time_arr)
        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]
        if len(time_arr) < 2 or sig_arr.shape[0] != len(time_arr):
            return
        try:
            # Use mean across positions/frequencies
            sig_for_fft = np.mean(sig_arr, axis=tuple(range(1, sig_arr.ndim)))
            dt = float(time_arr[1] - time_arr[0])
            n_fft = self._spectrum_fft_len(len(sig_for_fft))
            spectrum = np.fft.fftshift(np.fft.fft(sig_for_fft, n=n_fft))
            mag = np.abs(spectrum)
            if mag.size:
                mag_min = float(np.nanmin(mag))
                mag_max = float(np.nanmax(mag))
                if not np.isfinite(mag_min):
                    mag_min = 0.0
                if not np.isfinite(mag_max) or mag_max <= 0:
                    mag_max = 1.0
                self._spectrum_final_range = (mag_min, mag_max * 1.05)
        except Exception:
            self._spectrum_final_range = None

    def _plot_off_resonant_spins(self, time_idx=None) -> bool:
        """Plot a spectrum built directly from the simulated off-resonant spins (no FFT)."""
        if self.last_result is None or self.last_frequencies is None:
            return False
        sig = self.last_result.get("signal")
        mz = self.last_result.get("mz")
        if sig is None or mz is None:
            return False
        sig_arr = np.asarray(sig)
        mz_arr = np.asarray(mz)
        ntime = sig_arr.shape[0]
        if ntime == 0:
            return False

        # Ensure consistent shapes (time, pos, freq)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            if (
                self.last_positions is not None
                and sig_arr.shape[1] == self.last_positions.shape[0]
            ):
                sig_arr = sig_arr[:, :, None]
            else:
                sig_arr = sig_arr[:, None, :]

        if mz_arr.ndim == 1:
            mz_arr = mz_arr[:, None, None]
        elif mz_arr.ndim == 2:
            if (
                self.last_positions is not None
                and mz_arr.shape[1] == self.last_positions.shape[0]
            ):
                mz_arr = mz_arr[:, :, None]
            else:
                mz_arr = mz_arr[:, None, :]

        time_axis = (
            self.last_time
            if self.last_time is not None
            else np.arange(ntime, dtype=float)
        )
        if time_idx is None:
            time_idx = ntime - 1
        t_idx = int(max(0, min(time_idx, ntime - 1)))

        freq_axis = np.asarray(self.last_frequencies)
        if freq_axis.shape[0] != sig_arr.shape[2]:
            freq_axis = np.linspace(-0.5, 0.5, sig_arr.shape[2])

        pos_count = sig_arr.shape[1]
        spectrum_mode = (
            self.spectrum_mode.currentText()
            if hasattr(self, "spectrum_mode")
            else "Mean only"
        )
        pos_sel = (
            min(self.spectrum_pos_slider.value(), pos_count - 1) if pos_count > 0 else 0
        )

        # Snapshot spectrum at the selected time index
        snapshot_sig = sig_arr[t_idx]  # (npos, nfreq)
        snapshot_mz = mz_arr[t_idx]

        mean_sig = np.mean(snapshot_sig, axis=0) if pos_count > 0 else snapshot_sig
        mean_mz = np.mean(snapshot_mz, axis=0) if pos_count > 0 else snapshot_mz

        selected_sig = mean_sig
        selected_mz = mean_mz
        selected_label = "Mean"

        if spectrum_mode == "Mean + individuals":
            selected_sig = snapshot_sig[pos_sel] if pos_count else mean_sig
            selected_mz = snapshot_mz[pos_sel] if pos_count else mean_mz
            selected_label = f"Pos {pos_sel}"
        elif spectrum_mode == "Individual (select pos)":
            selected_sig = snapshot_sig[pos_sel] if pos_count else mean_sig
            selected_mz = snapshot_mz[pos_sel] if pos_count else mean_mz
            selected_label = f"Pos {pos_sel}"
            mean_sig = None
            mean_mz = None

        self.spectrum_plot.clear()
        selected_components = self.spectrum_component_combo.get_selected_items()

        def get_component(sig_data, mz_data, comp):
            if comp == "Magnitude":
                return np.abs(sig_data)
            if comp == "Phase":
                return np.angle(sig_data) / np.pi
            if comp == "Phase (unwrapped)":
                return np.unwrap(np.angle(sig_data)) / np.pi
            if comp == "Real":
                return np.real(sig_data)
            if comp == "Imaginary":
                return np.imag(sig_data)
            if comp == "Mz":
                return mz_data
            return np.abs(sig_data)

        # Track global limits for scaling
        all_visible_data = []

        # Plot each selected component
        for component in selected_components:
            if mean_sig is not None:
                # Use fixed colors for components in the Mean plot
                color = "c"  # Default Mean Magnitude
                if component == "Real":
                    color = "r"
                elif component == "Imaginary":
                    color = "g"
                elif component == "Phase" or component == "Phase (unwrapped)":
                    color = "y"
                elif component == "Mz":
                    color = "m"

                comp_data = get_component(mean_sig, mean_mz, component)
                all_visible_data.append(comp_data)

                self.spectrum_plot.plot(
                    freq_axis,
                    comp_data,
                    pen=pg.mkPen(color, width=3),
                    name=f"Mean {component}",
                )

            # Selected position plot
            if pos_count > 0:
                sel_color = (
                    self._color_for_index(pos_sel, max(pos_count, 1))
                    if pos_count > 1
                    else "w"
                )

                if len(selected_components) == 1:
                    pen = pg.mkPen(sel_color, width=2)
                else:
                    color = "b"  # Selected Magnitude
                    if component == "Real":
                        color = "r"
                    elif component == "Imaginary":
                        color = "g"
                    elif component == "Phase" or component == "Phase (unwrapped)":
                        color = "y"
                    elif component == "Mz":
                        color = "m"
                    pen = pg.mkPen(color, width=2)

                if component == "Real":
                    pen.setStyle(Qt.DashLine)
                elif component == "Imaginary":
                    pen.setStyle(Qt.DotLine)
                elif component == "Phase (unwrapped)":
                    pen.setStyle(Qt.SolidLine)

                comp_data = get_component(selected_sig, selected_mz, component)
                all_visible_data.append(comp_data)

                self.spectrum_plot.plot(
                    freq_axis,
                    comp_data,
                    pen=pen,
                    name=f"{selected_label} {component}",
                )

        self.spectrum_plot.setLabel("bottom", "Frequency", "Hz")
        y_label = (
            "Signal"
            if len(selected_components) > 1
            else selected_components[0] if selected_components else ""
        )
        if (
            "Phase" in selected_components or "Phase (unwrapped)" in selected_components
        ) and len(selected_components) == 1:
            y_label = f"{selected_components[0]} (units of π)"
        self.spectrum_plot.setLabel("left", y_label)

        if freq_axis is not None and freq_axis.size > 0:
            self.spectrum_plot.setXRange(
                float(np.nanmin(freq_axis)), float(np.nanmax(freq_axis)), padding=0.05
            )

        # Constant Y-limit calculation based on global extrema
        if hasattr(self, "spectrum_global_extrema"):
            selected_extrema_min = []
            selected_extrema_max = []

            for component in selected_components:
                # Use Magnitude extrema for Phase/Phase (unwrapped) if specific ones aren't stored
                lookup_comp = component
                if "Phase" in component:
                    # Special handling for phase is done below
                    continue

                if lookup_comp in self.spectrum_global_extrema:
                    c_min, c_max = self.spectrum_global_extrema[lookup_comp]
                    selected_extrema_min.append(c_min)
                    selected_extrema_max.append(c_max)

            if "Phase (unwrapped)" in selected_components:
                self.spectrum_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            elif "Phase" in selected_components and len(selected_components) == 1:
                self.spectrum_plot.setYRange(-1.1, 1.1, padding=0)
            elif selected_extrema_min or selected_extrema_max:
                data_min = min(selected_extrema_min) if selected_extrema_min else 0.0
                data_max = max(selected_extrema_max) if selected_extrema_max else 1.0

                # If Magnitude is present and no "negative" components, min should be 0
                has_negative_possible = any(
                    c in selected_components for c in ["Real", "Imaginary", "Mz"]
                )

                y_min = 0.0
                if has_negative_possible:
                    y_min = data_min * 1.1 if data_min < 0 else 0.0

                y_max = max(abs(data_max), abs(data_min)) * 1.2
                if y_max <= 0:
                    y_max = 1.2

                self.spectrum_plot.setYRange(y_min, y_max, padding=0)

        export_entry = {
            "frequency": freq_axis,
            "selected_magnitude": np.abs(selected_sig),
            "selected_phase_rad": np.angle(selected_sig),
            "selected_mz": selected_mz,
            "mode": "off_res_spins_freq",
            "time_idx": t_idx,
            "time_s": float(time_axis[t_idx]) if t_idx < len(time_axis) else None,
        }
        if mean_sig is not None:
            export_entry["mean_magnitude"] = np.abs(mean_sig)
            export_entry["mean_phase_rad"] = np.angle(mean_sig)
            export_entry["mean_mz"] = mean_mz
        self._last_spectrum_export = export_entry
        return True

        export_entry = {
            "frequency": freq_axis,
            "selected_magnitude": np.abs(selected_series),
            "selected_phase_rad": np.angle(selected_series),
            "mode": "off_res_spins_freq",
            "time_idx": t_idx,
            "time_s": float(time_axis[t_idx]) if t_idx < len(time_axis) else None,
        }
        if mean_series is not None:
            export_entry["mean_magnitude"] = np.abs(mean_series)
            export_entry["mean_phase_rad"] = np.angle(mean_series)
        self._last_spectrum_export = export_entry
        return True

    def _grab_widget_array(
        self, widget: QWidget, target_height: int = None
    ) -> np.ndarray:
        """Grab a Qt widget as an RGB numpy array, optionally scaling height."""
        pixmap = widget.grab()
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        if target_height and target_height > 0:
            image = image.scaledToHeight(target_height, Qt.SmoothTransformation)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (image.height(), image.width(), 4)
        )[:, :, :3]
        return arr

    def _grab_pyqtgraph_array(
        self, plot_widget: pg.PlotWidget, width: int = None, height: int = None
    ) -> np.ndarray:
        """Grab a pyqtgraph plot as an RGB numpy array using its faster internal exporter."""
        import pyqtgraph.exporters

        exporter = pg.exporters.ImageExporter(plot_widget.getPlotItem())
        if width:
            exporter.parameters()["width"] = width
        if height:
            exporter.parameters()["height"] = height

        image = exporter.export(toBytes=True)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (image.height(), image.width(), 4)
        )[:, :, :3]
        return arr

    def _ensure_even_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to even width/height for video encoders."""
        h, w = frame.shape[:2]
        if h % 2 != 0:
            frame = frame[:-1, :, :]
        if w % 2 != 0:
            frame = frame[:, :-1, :]
        return frame

    def _compute_playback_indices(self, time_array, start_idx, end_idx, fps):
        """Compute indices to match the current playback speed setting."""
        speed_ms_per_s = self.time_control.speed_spin.value()
        if speed_ms_per_s <= 1e-6:
            speed_ms_per_s = 50.0

        t_start = time_array[start_idx]
        t_end = time_array[end_idx]

        # Simulation duration in seconds
        sim_dur_s = t_end - t_start

        # Real duration in seconds = (Sim ms) / (ms/s)
        real_dur_s = (sim_dur_s * 1000.0) / speed_ms_per_s

        # Total frames
        n_frames = int(max(2, np.ceil(real_dur_s * fps)))

        # Target times
        target_times = np.linspace(t_start, t_end, n_frames)

        # Find indices
        indices = np.searchsorted(time_array, target_times)
        indices = np.clip(indices, start_idx, end_idx)

        self.log_message(
            f"Exporting animation: {n_frames} frames to match {speed_ms_per_s} ms/s at {fps} FPS."
        )

        return indices

    def _export_widget_animation(
        self, widgets: list, default_filename: str, before_grab=None
    ):
        """Generic widget-grab animation exporter (GIF/MP4) for plot tabs."""
        if self.last_time is None:
            QMessageBox.warning(
                self, "No Data", "Please run a time-resolved simulation first."
            )
            return
        total_frames = (
            len(self.playback_time)
            if self.playback_time is not None
            else len(self.last_time)
        )
        if total_frames < 2:
            QMessageBox.warning(
                self,
                "No Time Series",
                "Need at least two time points to export animation.",
            )
            return

        dialog = ExportAnimationDialog(
            self,
            total_frames=total_frames,
            default_filename=default_filename,
            default_directory=self._get_export_directory(),
        )
        dialog.mean_only_checkbox.setVisible(False)
        dialog.include_sequence_checkbox.setVisible(False)

        if dialog.exec_() != QDialog.Accepted:
            return
        params = dialog.get_export_params()

        time_array = (
            self.playback_time if self.playback_time is not None else self.last_time
        )
        indices = self._compute_playback_indices(
            time_array, params["start_idx"], params["end_idx"], params["fps"]
        )

        fmt = params["format"]
        filepath = Path(params["filename"])
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if vz_imageio is None:
            QMessageBox.warning(
                self,
                "Missing Dependency",
                "Animation export requires the 'imageio' package.",
            )
            return

        exporter = AnimationExporter()
        if fmt == "gif":
            # ImageIO v3 deprecated 'fps' for GIFs, use 'duration' (in ms)
            duration_ms = 1000.0 / params["fps"]
            writer = vz_imageio.get_writer(
                str(filepath), mode="I", duration=duration_ms, format="GIF"
            )
        else:
            writer = vz_imageio.get_writer(
                str(filepath),
                fps=params["fps"],
                format="FFMPEG",
                codec="libx264",
                bitrate=exporter.default_bitrate,
                quality=8,
                macro_block_size=None,
            )

        progress = QProgressDialog(
            "Exporting animation...", "Cancel", 0, len(indices), self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        was_running = self.anim_timer.isActive()
        current_idx = self.anim_index
        self.anim_timer.stop()

        try:
            ui_update_interval = max(
                1, len(indices) // 50
            )  # keep UI responsive without slowing loop
            for i, idx in enumerate(indices):
                if progress.wasCanceled():
                    raise RuntimeError("Animation export cancelled")
                # Sync all plots/time lines
                self._set_animation_index_from_slider(int(idx))
                if before_grab:
                    try:
                        before_grab(int(idx))
                    except Exception:
                        pass
                QApplication.processEvents()
                frames = []
                target_h = params["height"] if params["height"] else None
                target_w = params["width"] if params["width"] else None
                for w in widgets:
                    if isinstance(w, pg.PlotWidget):
                        frames.append(
                            self._grab_pyqtgraph_array(
                                w, width=target_w, height=target_h
                            )
                        )
                    else:
                        frames.append(
                            self._grab_widget_array(w, target_height=target_h)
                        )
                # Normalize heights to smallest to stack horizontally
                min_h = min(f.shape[0] for f in frames)
                frames = [f if f.shape[0] == min_h else f[:min_h, :, :] for f in frames]
                combined = np.hstack(frames)
                target_w = params["width"] if params["width"] else combined.shape[1]
                target_h_final = (
                    params["height"] if params["height"] else combined.shape[0]
                )
                if target_w != combined.shape[1] or target_h_final != combined.shape[0]:
                    qimg = QImage(
                        combined.data,
                        combined.shape[1],
                        combined.shape[0],
                        combined.strides[0],
                        QImage.Format_RGB888,
                    )
                    # If both width/height are provided, honor exact resolution; otherwise keep aspect
                    aspect_mode = (
                        Qt.IgnoreAspectRatio
                        if (params["width"] and params["height"])
                        else Qt.KeepAspectRatio
                    )
                    qimg = qimg.copy().scaled(
                        target_w, target_h_final, aspect_mode, Qt.SmoothTransformation
                    )
                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    # QImage bits are often 32-bit aligned/padded or RGBA even if Format_RGB888 was requested
                    # Check size to determine channels
                    n_bytes = qimg.byteCount()
                    n_pixels = qimg.width() * qimg.height()
                    n_channels = n_bytes // n_pixels

                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                        (qimg.height(), qimg.width(), n_channels)
                    )
                    # Keep only RGB channels (drop alpha if present)
                    combined = arr[:, :, :3]
                combined = self._ensure_even_frame(combined)
                writer.append_data(combined)
                progress.setValue(i + 1)
                if (i % ui_update_interval) == 0:
                    QApplication.processEvents()
        finally:
            writer.close()
            self._set_animation_index_from_slider(current_idx)
            if was_running:
                self._resume_vector_animation()
            progress.setValue(progress.maximum())

        QMessageBox.information(
            self,
            "Export Successful",
            f"Animation exported successfully:\n{filepath.name}",
        )
        self.log_message(f"Animation exported to {filepath}")

    def _export_magnetization_image(self, default_format="png"):
        """Export magnetization plots as an image."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        # Create export dialog with default directory
        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(
            self, default_filename="magnetization", default_directory=export_dir
        )
        dialog.format_combo.setCurrentIndex(["png", "svg", "pdf"].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                # Export both plots
                # For now, export them separately (multi-plot export is future work)
                base_path = Path(params["filename"])

                # Export Mxy plot
                mxy_path = base_path.parent / f"{base_path.stem}_mxy{base_path.suffix}"
                result_mxy = exporter.export_pyqtgraph_plot(
                    self.mxy_plot,
                    str(mxy_path),
                    format=params["format"],
                    width=params["width"],
                )

                # Export Mz plot
                mz_path = base_path.parent / f"{base_path.stem}_mz{base_path.suffix}"
                result_mz = exporter.export_pyqtgraph_plot(
                    self.mz_plot,
                    str(mz_path),
                    format=params["format"],
                    width=params["width"],
                )

                if result_mxy and result_mz:
                    self.log_message(
                        f"Exported magnetization plots to:\n  {result_mxy}\n  {result_mz}"
                    )
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Magnetization plots exported successfully:\n\n"
                        f"Mxy: {Path(result_mxy).name}\n"
                        f"Mz: {Path(result_mz).name}",
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Could not export plots. Check the log for details.",
                    )

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_magnetization_animation(self):
        """Export magnetization time-series as GIF/MP4."""
        if (
            not self.last_result
            or "mx" not in self.last_result
            or self.last_result["mx"] is None
        ):
            QMessageBox.warning(
                self, "No Data", "Please run a time-resolved simulation first."
            )
            return

        mx = self.last_result["mx"]
        my = self.last_result["my"]
        mz = self.last_result["mz"]
        if mx is None or mx.ndim != 3:
            QMessageBox.warning(
                self, "No Time Series", "Animation export requires time-resolved data."
            )
            return

        time_s = (
            np.asarray(self.last_time)
            if self.last_time is not None
            else np.asarray(self.last_result.get("time", []))
        )
        if time_s is None or len(time_s) == 0 or len(time_s) != mx.shape[0]:
            QMessageBox.warning(
                self,
                "Missing Time",
                "Could not determine time axis for animation export.",
            )
            return

        total_frames = len(time_s)
        dialog = ExportAnimationDialog(
            self,
            total_frames=total_frames,
            default_filename="magnetization",
            default_directory=self._get_export_directory(),
        )
        dialog.mean_only_checkbox.setChecked(self.mean_only_checkbox.isChecked())
        dialog.include_sequence_checkbox.setVisible(False)

        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        start_idx = min(params["start_idx"], total_frames - 1)
        end_idx = max(start_idx, min(params["end_idx"], total_frames - 1))
        mean_only = params["mean_only"]

        def _select_component(arr):
            if arr.ndim == 3:
                if mean_only:
                    return np.mean(arr, axis=(1, 2))
                return arr[:, 0, 0]
            elif arr.ndim == 2:
                return arr[:, 0]
            return np.asarray(arr)

        mx_trace = _select_component(mx)
        my_trace = _select_component(my)
        mz_trace = _select_component(mz)

        groups = [
            {
                "title": "Transverse Magnetization (Mx/My)",
                "ylabel": "Magnetization",
                "series": [
                    {"data": mx_trace, "label": "Mx", "color": "r"},
                    {"data": my_trace, "label": "My", "color": "g", "style": "--"},
                ],
            },
            {
                "title": "Longitudinal Magnetization (Mz)",
                "ylabel": "Magnetization",
                "series": [{"data": mz_trace, "label": "Mz", "color": "b"}],
            },
        ]

        indices = self._compute_playback_indices(
            time_s, start_idx, end_idx, params["fps"]
        )
        progress = QProgressDialog("Exporting animation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        def progress_cb(done, total):
            val = int(done / total * 100) if total else 0
            progress.setValue(val)
            QApplication.processEvents()

        def cancel_cb():
            return progress.wasCanceled()

        exporter = AnimationExporter()
        try:
            result = exporter.export_time_series_animation(
                time_s,
                groups,
                params["filename"],
                fps=params["fps"],
                max_frames=params["max_frames"],
                start_idx=start_idx,
                end_idx=end_idx,
                width=params["width"],
                height=params["height"],
                format=params["format"],
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                indices=indices,
            )
            progress.setValue(100)
            if result:
                self.log_message(f"Animation exported to {result}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Animation exported successfully:\n{Path(result).name}",
                )
        except Exception as e:
            progress.close()
            if isinstance(e, RuntimeError) and "cancelled" in str(e).lower():
                self.log_message("Animation export cancelled by user.")
            else:
                self.log_message(f"Animation export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_magnetization_data(self):
        """Export magnetization time series as CSV/NPY/DAT."""
        if not self.last_result or self.last_result.get("mx") is None:
            QMessageBox.warning(
                self, "No Data", "Please run a time-resolved simulation first."
            )
            return
        mx = self.last_result.get("mx")
        my = self.last_result.get("my")
        mz = self.last_result.get("mz")
        if mx is None or my is None or mz is None or mx.ndim != 3:
            QMessageBox.warning(
                self,
                "No Time Series",
                "Data export requires time-resolved magnetization data.",
            )
            return
        time_s = (
            np.asarray(self.last_time)
            if self.last_time is not None
            else np.asarray(self.last_result.get("time", []))
        )
        if time_s is None or len(time_s) != mx.shape[0]:
            QMessageBox.warning(
                self, "Missing Time", "Could not determine time axis for data export."
            )
            return

        path, fmt = self._prompt_data_export_path("magnetization")
        if not path:
            return
        try:
            all_metadata = self._collect_all_parameters()
            result_path = self.dataset_exporter.export_magnetization(
                time_s,
                mx,
                my,
                mz,
                self.last_positions,
                self.last_frequencies,
                str(path),
                format=fmt,
                metadata=all_metadata,
            )
            self.log_message(f"Magnetization data exported to {result_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Magnetization data saved:\n{Path(result_path).name}",
            )
        except Exception as e:
            self.log_message(f"Magnetization data export failed: {e}")
            QMessageBox.critical(
                self, "Export Error", f"Could not export magnetization data:\n{e}"
            )

    def _export_3d_animation(self):
        """Export the 3D vector view as a GIF/MP4."""
        if (
            self.anim_data is None
            or self.playback_time is None
            or len(self.anim_data) < 2
        ):
            QMessageBox.warning(
                self,
                "No Data",
                "3D animation export requires a time-resolved simulation.",
            )
            return
        # Check dependency - use local variable to avoid UnboundLocalError
        imageio_lib = vz_imageio
        if imageio_lib is None:
            try:
                import imageio

                imageio_lib = imageio
            except ImportError as e:
                QMessageBox.critical(
                    self,
                    "Missing Dependency",
                    f"Animation export requires 'imageio'. Install with: pip install imageio imageio-ffmpeg\n\nError: {e}",
                )
                return

        total_frames = len(self.anim_data)
        dialog = ExportAnimationDialog(
            self,
            total_frames=total_frames,
            default_filename="vector3d",
            default_directory=self._get_export_directory(),
        )
        # Mean-only not applicable for 3D view (already uses colored vectors); hide toggle
        dialog.mean_only_checkbox.setVisible(False)

        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        indices = self._compute_playback_indices(
            self.playback_time, params["start_idx"], params["end_idx"], params["fps"]
        )

        # Prepare writers (main + optional sequence-only)
        fmt = params["format"]
        filepath = Path(params["filename"])
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        exporter = AnimationExporter()

        def _make_writer(target_path: Path):
            if fmt == "gif":
                # ImageIO v3 deprecated 'fps' for GIFs, use 'duration' (in ms)
                duration_ms = 1000.0 / params["fps"]
                return imageio_lib.get_writer(
                    str(target_path), mode="I", duration=duration_ms, format="GIF"
                )
            return imageio_lib.get_writer(
                str(target_path),
                fps=params["fps"],
                format="FFMPEG",
                codec="libx264",
                bitrate=exporter.default_bitrate,
                quality=8,
                macro_block_size=None,
            )

        writer = _make_writer(filepath)
        seq_writer = None
        seq_filepath = None
        if params.get("include_sequence", False):
            seq_filepath = filepath.with_name(
                f"{filepath.stem}_sequence{filepath.suffix}"
            )
            seq_filepath.parent.mkdir(parents=True, exist_ok=True)
            seq_writer = _make_writer(seq_filepath)

        progress = QProgressDialog(
            "Exporting 3D animation...", "Cancel", 0, len(indices), self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        # Save state and pause playback to avoid interference
        was_running = self.anim_timer.isActive()
        current_idx = self.anim_index
        self.anim_timer.stop()

        def grab_sequence_frame():
            seq_pixmap = self.sequence_designer.diagram_widget.grab()
            seq_image = seq_pixmap.toImage()
            target_w = params["width"] if params["width"] else seq_image.width()
            target_h = params["height"] if params["height"] else seq_image.height()
            return _qimage_to_rgb(seq_image, target_w=target_w, target_h=target_h)

        def _qimage_to_rgb(image, target_w=None, target_h=None):
            if target_w and target_h:
                image = image.scaled(
                    target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            image = image.convertToFormat(QImage.Format_RGB888)
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            return np.frombuffer(ptr, dtype=np.uint8).reshape(
                (image.height(), image.width(), 3)
            )

        try:
            ui_update_interval = max(1, len(indices) // 50)
            for i, idx in enumerate(indices):
                if progress.wasCanceled():
                    raise RuntimeError("Animation export cancelled")
                self._set_animation_index_from_slider(int(idx))
                QApplication.processEvents()  # Allow UI to update with new vector positions

                # Use fast off-screen rendering for the 3D view
                target_w = (
                    params["width"]
                    if params["width"]
                    else self.mag_3d.gl_widget.width()
                )
                target_h = (
                    params["height"]
                    if params["height"]
                    else self.mag_3d.gl_widget.height()
                )
                frame = self.mag_3d.gl_widget.renderToArray((target_w, target_h))
                frame = self._ensure_even_frame(
                    frame
                )  # Ensure dimensions are even for video codecs
                writer.append_data(frame)
                if seq_writer is not None:
                    seq_frame = grab_sequence_frame()
                    seq_writer.append_data(seq_frame)
                progress.setValue(i + 1)
                QApplication.processEvents()
        finally:
            writer.close()
            if seq_writer is not None:
                seq_writer.close()
            # Restore playback state
            self._set_animation_index_from_slider(current_idx)
            if was_running:
                self._resume_vector_animation()
            progress.setValue(progress.maximum())

        msg = f"3D animation exported successfully:\n{filepath.name}"
        if seq_filepath is not None:
            msg += f"\nSequence diagram exported:\n{seq_filepath.name}"
        QMessageBox.information(self, "Export Successful", msg)
        self.log_message(f"3D animation exported to {filepath}")
        if seq_filepath is not None:
            self.log_message(f"Sequence diagram exported to {seq_filepath}")

    def _export_sequence_diagram_animation(self):
        """Export the sequence diagram only (no combined views)."""
        self._export_widget_animation(
            [self.sequence_designer.diagram_widget], default_filename="sequence"
        )

    def _export_signal_image(self, default_format="png"):
        """Export signal plot as an image."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(
            self, default_filename="signal", default_directory=export_dir
        )
        dialog.format_combo.setCurrentIndex(["png", "svg", "pdf"].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                result = exporter.export_pyqtgraph_plot(
                    self.signal_plot,
                    params["filename"],
                    format=params["format"],
                    width=params["width"],
                )

                if result:
                    self.log_message(f"Exported signal plot to: {result}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Signal plot exported successfully:\n{Path(result).name}",
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Could not export plot. Check the log for details.",
                    )

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_signal_animation(self):
        """Export received signal as animation."""
        if not self.last_result or "signal" not in self.last_result:
            QMessageBox.warning(
                self, "No Data", "Please run a time-resolved simulation first."
            )
            return

        signal_arr = self.last_result["signal"]
        if signal_arr is None or signal_arr.ndim < 2:
            QMessageBox.warning(
                self, "No Time Series", "Animation export requires time-resolved data."
            )
            return

        time_s = (
            np.asarray(self.last_time)
            if self.last_time is not None
            else np.asarray(self.last_result.get("time", []))
        )
        if time_s is None or len(time_s) == 0:
            QMessageBox.warning(
                self,
                "Missing Time",
                "Could not determine time axis for animation export.",
            )
            return

        # Ensure alignment between time and signal length
        nframes = min(len(time_s), signal_arr.shape[0])
        if nframes < 2:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "Need at least two time points to export animation.",
            )
            return
        time_s = time_s[:nframes]

        dialog = ExportAnimationDialog(
            self,
            total_frames=nframes,
            default_filename="signal",
            default_directory=self._get_export_directory(),
        )
        dialog.mean_only_checkbox.setChecked(self.mean_only_checkbox.isChecked())
        dialog.include_sequence_checkbox.setVisible(False)
        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        start_idx = min(params["start_idx"], nframes - 1)
        end_idx = max(start_idx, min(params["end_idx"], nframes - 1))
        mean_only = params["mean_only"]
        indices = self._compute_playback_indices(
            time_s, start_idx, end_idx, params["fps"]
        )

        def _select_signal(arr):
            if arr.ndim == 3:
                if mean_only:
                    return np.mean(arr, axis=(1, 2))
                return arr[:, 0, 0]
            if arr.ndim == 2:
                if mean_only:
                    return np.mean(arr, axis=1)
                return arr[:, 0]
            return np.asarray(arr)

        sig_trace = _select_signal(signal_arr)[:nframes]
        groups = [
            {
                "title": "Signal Magnitude",
                "ylabel": "|S|",
                "series": [{"data": np.abs(sig_trace), "label": "|S|", "color": "c"}],
            },
            {
                "title": "Signal Components",
                "ylabel": "Amplitude",
                "series": [
                    {"data": np.real(sig_trace), "label": "Real", "color": "m"},
                    {
                        "data": np.imag(sig_trace),
                        "label": "Imag",
                        "color": "y",
                        "style": "--",
                    },
                ],
            },
        ]

        progress = QProgressDialog("Exporting animation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        def progress_cb(done, total):
            val = int(done / total * 100) if total else 0
            progress.setValue(val)
            QApplication.processEvents()

        def cancel_cb():
            return progress.wasCanceled()

        exporter = AnimationExporter()
        try:
            result = exporter.export_time_series_animation(
                time_s,
                groups,
                params["filename"],
                fps=params["fps"],
                max_frames=params["max_frames"],
                start_idx=start_idx,
                end_idx=end_idx,
                width=params["width"],
                height=params["height"],
                format=params["format"],
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                indices=indices,
            )
            progress.setValue(100)
            if result:
                self.log_message(f"Signal animation exported to {result}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Signal animation exported successfully:\n{Path(result).name}",
                )
        except Exception as e:
            progress.close()
            if isinstance(e, RuntimeError) and "cancelled" in str(e).lower():
                self.log_message("Signal animation export cancelled by user.")
            else:
                self.log_message(f"Signal animation export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_signal_data(self):
        """Export received signal traces as CSV/NPY/DAT."""
        if not self.last_result or self.last_result.get("signal") is None:
            QMessageBox.warning(
                self, "No Data", "Please run a time-resolved simulation first."
            )
            return
        signal_arr = self.last_result.get("signal")
        if signal_arr is None or signal_arr.ndim < 2:
            QMessageBox.warning(
                self,
                "No Time Series",
                "Data export requires time-resolved signal data.",
            )
            return

        time_s = (
            np.asarray(self.last_time)
            if self.last_time is not None
            else np.asarray(self.last_result.get("time", []))
        )
        nframes = min(len(time_s), signal_arr.shape[0])
        if nframes < 1:
            QMessageBox.warning(
                self, "Missing Time", "Could not determine time axis for data export."
            )
            return
        time_s = time_s[:nframes]
        signal_arr = signal_arr[:nframes]

        path, fmt = self._prompt_data_export_path("signal")
        if not path:
            return
        try:
            all_metadata = self._collect_all_parameters()
            result_path = self.dataset_exporter.export_signal(
                time_s, signal_arr, str(path), format=fmt, metadata=all_metadata
            )
            self.log_message(f"Signal data exported to {result_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Signal data saved:\n{Path(result_path).name}",
            )
        except Exception as e:
            self.log_message(f"Signal data export failed: {e}")
            QMessageBox.critical(
                self, "Export Error", f"Could not export signal data:\n{e}"
            )

    def _export_spectrum_animation(self):
        """Export spectrum plot animation via widget grab."""

        def updater(idx):
            actual_idx = self._playback_to_full_index(idx)
            self._refresh_spectrum(time_idx=actual_idx, skip_fft=False)

        self._export_widget_animation(
            [self.spectrum_plot], default_filename="spectrum", before_grab=updater
        )

    def _export_spatial_animation(self):
        """Export spatial plots animation via widget grab."""
        self._export_widget_animation(
            [self.spatial_mxy_plot, self.spatial_mz_plot], default_filename="spatial"
        )

    def _export_spectrum_image(self, default_format="png"):
        """Export spectrum plot as an image."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(
            self, default_filename="spectrum", default_directory=export_dir
        )
        dialog.format_combo.setCurrentIndex(["png", "svg", "pdf"].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                result = exporter.export_pyqtgraph_plot(
                    self.spectrum_plot,
                    params["filename"],
                    format=params["format"],
                    width=params["width"],
                )

                if result:
                    self.log_message(f"Exported spectrum plot to: {result}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Spectrum plot exported successfully:\n{Path(result).name}",
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Could not export plot. Check the log for details.",
                    )

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_spectrum_data(self):
        """Export spectrum data as CSV/NPY/DAT."""
        if self.last_result is None:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        actual_idx = self._current_playback_index()
        self._refresh_spectrum(time_idx=actual_idx, skip_fft=False)
        export_cache = getattr(self, "_last_spectrum_export", None)
        if not export_cache:
            QMessageBox.warning(
                self, "No Spectrum", "Spectrum data is not available for export."
            )
            return

        freq = export_cache.get("frequency")
        if freq is None or len(freq) == 0:
            QMessageBox.warning(
                self, "No Spectrum", "Spectrum data is not available for export."
            )
            return
        series = {
            "selected_magnitude": export_cache.get("selected_magnitude"),
            "selected_phase_rad": export_cache.get("selected_phase_rad"),
        }
        if export_cache.get("mean_magnitude") is not None:
            series["mean_magnitude"] = export_cache.get("mean_magnitude")
        if export_cache.get("mean_phase_rad") is not None:
            series["mean_phase_rad"] = export_cache.get("mean_phase_rad")

        path, fmt = self._prompt_data_export_path("spectrum")
        if not path:
            return
        try:
            result_path = self.dataset_exporter.export_spectrum(
                freq, series, str(path), format=fmt
            )
            self.log_message(f"Spectrum data exported to {result_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Spectrum data saved:\n{Path(result_path).name}",
            )
        except Exception as e:
            self.log_message(f"Spectrum data export failed: {e}")
            QMessageBox.critical(
                self, "Export Error", f"Could not export spectrum data:\n{e}"
            )

    def _export_spatial_image(self, default_format="png"):
        """Export spatial plots as images."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(
            self, default_filename="spatial", default_directory=export_dir
        )
        dialog.format_combo.setCurrentIndex(["png", "svg", "pdf"].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                # Export both spatial plots
                base_path = Path(params["filename"])

                # Export Mxy spatial plot
                mxy_path = base_path.parent / f"{base_path.stem}_mxy{base_path.suffix}"
                result_mxy = exporter.export_pyqtgraph_plot(
                    self.spatial_mxy_plot,
                    str(mxy_path),
                    format=params["format"],
                    width=params["width"],
                )

                # Export Mz spatial plot
                mz_path = base_path.parent / f"{base_path.stem}_mz{base_path.suffix}"
                result_mz = exporter.export_pyqtgraph_plot(
                    self.spatial_mz_plot,
                    str(mz_path),
                    format=params["format"],
                    width=params["width"],
                )

                if result_mxy and result_mz:
                    self.log_message(
                        f"Exported spatial plots to:\n  {result_mxy}\n  {result_mz}"
                    )
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Spatial plots exported successfully:\n\n"
                        f"Mxy: {Path(result_mxy).name}\n"
                        f"Mz: {Path(result_mz).name}",
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Could not export plots. Check the log for details.",
                    )

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_spatial_data(self):
        """Export spatial profiles as CSV/NPY/DAT."""
        if self.last_result is None or self.last_positions is None:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        # Ensure cache reflects current frame
        self.update_spatial_plot_from_last_result(
            time_idx=self._current_playback_index()
        )
        cache = getattr(self, "_last_spatial_export", None)
        if not cache:
            QMessageBox.warning(
                self, "No Spatial Data", "Spatial data is not available for export."
            )
            return

        position = cache.get("position_m")
        mxy = cache.get("mxy")
        mz = cache.get("mz")
        if position is None or mxy is None or mz is None:
            QMessageBox.warning(
                self, "No Spatial Data", "Spatial data is not available for export."
            )
            return

        path, fmt = self._prompt_data_export_path("spatial")
        if not path:
            return
        try:
            result_path = self.dataset_exporter.export_spatial(
                position,
                mxy,
                mz,
                str(path),
                format=fmt,
                mxy_per_freq=cache.get("mxy_per_freq"),
                mz_per_freq=cache.get("mz_per_freq"),
            )
            self.log_message(f"Spatial data exported to {result_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Spatial data saved:\n{Path(result_path).name}",
            )
        except Exception as e:
            self.log_message(f"Spatial data export failed: {e}")
            QMessageBox.critical(
                self, "Export Error", f"Could not export spatial data:\n{e}"
            )

    def _update_mag_heatmaps(self):
        """Update magnetization heatmaps (Time vs Position/Frequency)."""
        if self.last_result is None:
            return

        mx_all = self.last_result["mx"]
        my_all = self.last_result["my"]
        mz_all = self.last_result["mz"]
        time_arr = (
            self.last_time
            if self.last_time is not None
            else self.last_result.get("time", None)
        )

        if time_arr is None or mx_all.ndim != 3:
            self.log_message("Heatmap requires time-resolved 3D data")
            return

        ntime, npos, nfreq = mx_all.shape
        time_ms = time_arr * 1000  # Convert to ms

        # Get view mode
        view_mode = (
            self.mag_view_mode.currentText()
            if hasattr(self, "mag_view_mode")
            else "All positions x freqs"
        )
        selector = (
            self.mag_view_selector.value() if hasattr(self, "mag_view_selector") else 0
        )

        # Determine if view mode selector should be visible
        # Hide selector for single position/frequency cases
        if view_mode == "Positions @ freq" and npos == 1:
            # Only 1 position - no point showing position selector
            if hasattr(self, "mag_view_selector"):
                self.mag_view_selector.setVisible(False)
                self.mag_view_selector_label.setVisible(False)
        elif view_mode == "Freqs @ position" and nfreq == 1:
            # Only 1 frequency - no point showing frequency selector
            if hasattr(self, "mag_view_selector"):
                self.mag_view_selector.setVisible(False)
                self.mag_view_selector_label.setVisible(False)
        elif view_mode == "All positions x freqs" and npos * nfreq == 1:
            # Only 1 spin total
            if hasattr(self, "mag_view_selector"):
                self.mag_view_selector.setVisible(False)
                self.mag_view_selector_label.setVisible(False)

        # Get component selection
        component = (
            self.mag_component.currentText()
            if hasattr(self, "mag_component")
            else "Magnitude"
        )

        # Prepare data slices based on view mode
        y_min, y_max = 0, 0

        if view_mode == "Positions @ freq" and nfreq > 1:
            # Show all positions at selected frequency
            fi = min(selector, nfreq - 1)
            mx_slice = mx_all[:, :, fi]  # (ntime, npos)
            my_slice = my_all[:, :, fi]
            mz_slice = mz_all[:, :, fi]

            y_label = "Position (cm)"
            n_y = npos
            if self.last_positions is not None:
                # Assume Z-axis varying
                pos_vals = self.last_positions[:, 2] * 100
                y_min, y_max = pos_vals[0], pos_vals[-1]
            else:
                y_label = "Position Index"
                y_min, y_max = 0, npos

        elif view_mode == "Freqs @ position" and npos > 1:
            # Show all frequencies at selected position
            pi = min(selector, npos - 1)
            mx_slice = mx_all[:, pi, :]  # (ntime, nfreq)
            my_slice = my_all[:, pi, :]
            mz_slice = mz_all[:, pi, :]

            y_label = "Frequency (Hz)"
            n_y = nfreq
            if self.last_frequencies is not None:
                y_min, y_max = self.last_frequencies[0], self.last_frequencies[-1]
            else:
                y_label = "Frequency Index"
                y_min, y_max = 0, nfreq
        else:
            # Show all spins (flatten position x frequency)
            mx_slice = mx_all.reshape(ntime, -1)  # (ntime, npos*nfreq)
            my_slice = my_all.reshape(ntime, -1)
            mz_slice = mz_all.reshape(ntime, -1)
            y_label = "Spin Index (pos×freq)"
            n_y = npos * nfreq
            y_min, y_max = 0, n_y

        # Handle degenerate range
        if np.isclose(y_min, y_max):
            y_max = y_min + (1.0 if n_y <= 1 else n_y)

        # Compute the selected component
        if component == "Magnitude":
            mxy_data = np.sqrt(mx_slice**2 + my_slice**2)  # (ntime, n_y)
            mz_data = np.abs(mz_slice)  # (ntime, n_y)
            mxy_title = "|Mxy| Heatmap"
            mz_title = "|Mz| Heatmap"
        elif component == "Real (Mx/Re)":
            mxy_data = mx_slice  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = "Mx Heatmap"
            mz_title = "Mz Heatmap"
        elif component == "Imaginary (My/Im)":
            mxy_data = my_slice  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = "My Heatmap"
            mz_title = "Mz Heatmap"
        elif component == "Phase":
            mxy_data = np.angle(mx_slice + 1j * my_slice)  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = "Mxy Phase Heatmap"
            mz_title = "Mz Heatmap"
        elif component == "Mz":
            mxy_data = mz_slice  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = "Mz Heatmap"
            mz_title = "Mz Heatmap"
        else:
            # Default to magnitude
            mxy_data = np.sqrt(mx_slice**2 + my_slice**2)
            mz_data = np.abs(mz_slice)
            mxy_title = "|Mxy| Heatmap"
            mz_title = "|Mz| Heatmap"

        # Update Mxy heatmap
        # pyqtgraph ImageItem convention: data[row, col] where row=Y, col=X
        # We have mxy_data as (ntime, n_y) meaning data[time, spin]
        # We want: X-axis = Time, Y-axis = Spin/Pos/Freq
        # So we need to transpose to get data[spin, time] = data[Y, X]
        self.mxy_heatmap_item.setImage(
            mxy_data.T, autoLevels=True, axisOrder="row-major"
        )
        # Now set the coordinate mapping using setRect(x, y, width, height)
        self.mxy_heatmap_item.setRect(
            time_ms[0], y_min, time_ms[-1] - time_ms[0], y_max - y_min
        )
        self.mxy_heatmap.setLabel("left", y_label)
        self.mxy_heatmap.setLabel("bottom", "Time", "ms")
        self.mxy_heatmap.setTitle(mxy_title)
        # Set view limits to show only actual data
        self.mxy_heatmap.setXRange(time_ms[0], time_ms[-1], padding=0)
        self.mxy_heatmap.setYRange(y_min, y_max, padding=0)

        # Update Mz heatmap
        self.mz_heatmap_item.setImage(mz_data.T, autoLevels=True, axisOrder="row-major")
        self.mz_heatmap_item.setRect(
            time_ms[0], y_min, time_ms[-1] - time_ms[0], y_max - y_min
        )
        self.mz_heatmap.setLabel("left", y_label)
        self.mz_heatmap.setLabel("bottom", "Time", "ms")
        self.mz_heatmap.setTitle(mz_title)
        # Set view limits to show only actual data
        self.mz_heatmap.setXRange(time_ms[0], time_ms[-1], padding=0)
        self.mz_heatmap.setYRange(y_min, y_max, padding=0)

    def _update_signal_heatmaps(self):
        """Update signal heatmaps (Time vs Position/Frequency)."""
        if self.last_result is None:
            return

        signal_all = self.last_result["signal"]
        time_arr = (
            self.last_time
            if self.last_time is not None
            else self.last_result.get("time", None)
        )

        if time_arr is None or signal_all.ndim != 3:
            self.log_message("Heatmap requires time-resolved 3D data")
            return

        ntime, npos, nfreq = signal_all.shape
        time_ms = time_arr * 1000  # Convert to ms

        # Get view mode
        view_mode = (
            self.signal_view_mode.currentText()
            if hasattr(self, "signal_view_mode")
            else "All positions x freqs"
        )
        selector = (
            self.signal_view_selector.value()
            if hasattr(self, "signal_view_selector")
            else 0
        )

        # Determine if view mode selector should be visible
        # Hide selector for single position/frequency cases
        if view_mode == "Positions @ freq" and npos == 1:
            # Only 1 position - no point showing position selector
            if hasattr(self, "signal_view_selector"):
                self.signal_view_selector.setVisible(False)
                self.signal_view_selector_label.setVisible(False)
        elif view_mode == "Freqs @ position" and nfreq == 1:
            # Only 1 frequency - no point showing frequency selector
            if hasattr(self, "signal_view_selector"):
                self.signal_view_selector.setVisible(False)
                self.signal_view_selector_label.setVisible(False)
        elif view_mode == "All positions x freqs" and npos * nfreq == 1:
            # Only 1 spin total
            if hasattr(self, "signal_view_selector"):
                self.signal_view_selector.setVisible(False)
                self.signal_view_selector_label.setVisible(False)

        # Get component selection
        component = (
            self.signal_component.currentText()
            if hasattr(self, "signal_component")
            else "Magnitude"
        )

        # Prepare data slices based on view mode
        y_min, y_max = 0, 0

        if view_mode == "Positions @ freq" and nfreq > 1:
            # Show all positions at selected frequency
            fi = min(selector, nfreq - 1)
            signal_slice = signal_all[:, :, fi]  # (ntime, npos)

            y_label = "Position (cm)"
            n_y = npos
            if self.last_positions is not None:
                pos_vals = self.last_positions[:, 2] * 100
                y_min, y_max = pos_vals[0], pos_vals[-1]
            else:
                y_label = "Position Index"
                y_min, y_max = 0, npos

        elif view_mode == "Freqs @ position" and npos > 1:
            # Show all frequencies at selected position
            pi = min(selector, npos - 1)
            signal_slice = signal_all[:, pi, :]  # (ntime, nfreq)

            y_label = "Frequency (Hz)"
            n_y = nfreq
            if self.last_frequencies is not None:
                y_min, y_max = self.last_frequencies[0], self.last_frequencies[-1]
            else:
                y_label = "Frequency Index"
                y_min, y_max = 0, nfreq
        else:
            # Show all spins (flatten position x frequency)
            signal_slice = signal_all.reshape(ntime, -1)  # (ntime, npos*nfreq)
            y_label = "Spin Index (pos×freq)"
            n_y = npos * nfreq
            y_min, y_max = 0, n_y

        # Handle degenerate range
        if np.isclose(y_min, y_max):
            y_max = y_min + (1.0 if n_y <= 1 else n_y)

        # Compute the selected component
        if component == "Magnitude":
            signal_data = np.abs(signal_slice)  # (ntime, n_y)
            title = "|Signal| Heatmap"
        elif component == "Real":
            signal_data = np.real(signal_slice)  # (ntime, n_y)
            title = "Re(Signal) Heatmap"
        elif component == "Imaginary":
            signal_data = np.imag(signal_slice)  # (ntime, n_y)
            title = "Im(Signal) Heatmap"
        elif component == "Phase":
            signal_data = np.angle(signal_slice)  # (ntime, n_y)
            title = "Phase(Signal) Heatmap"
        else:
            # Default to magnitude
            signal_data = np.abs(signal_slice)
            title = "|Signal| Heatmap"

        # Update signal heatmap
        # pyqtgraph ImageItem convention: data[row, col] where row=Y, col=X
        # We have signal_data as (ntime, n_y) meaning data[time, spin]
        # We want: X-axis = Time, Y-axis = Spin
        # So we need to transpose to get data[spin, time] = data[Y, X]
        self.signal_heatmap_item.setImage(
            signal_data.T, autoLevels=True, axisOrder="row-major"
        )
        # Set proper scale and position
        self.signal_heatmap_item.setRect(
            time_ms[0], y_min, time_ms[-1] - time_ms[0], y_max - y_min
        )
        self.signal_heatmap.setLabel("left", y_label)
        self.signal_heatmap.setLabel("bottom", "Time", "ms")
        self.signal_heatmap.setTitle(title)
        # Set view limits to show only actual data
        self.signal_heatmap.setXRange(time_ms[0], time_ms[-1], padding=0)
        self.signal_heatmap.setYRange(y_min, y_max, padding=0)

    # === TUTORIAL METHODS ===

    def record_tutorial(self):
        from PyQt5.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "New Tutorial", "Enter tutorial name:")
        if ok and name:
            self.current_tutorial_name = name
            self.tutorial_manager.start_recording()
            self.statusBar().showMessage(f"Recording tutorial: {name}...")
            self.stop_tut_action.setEnabled(True)

    def load_tutorial_dialog(self):
        folder = os.path.join(os.getcwd(), "tutorials")
        if not os.path.exists(folder):
            os.makedirs(folder)

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Tutorial", folder, "JSON Files (*.json)"
        )
        if path:
            name = Path(path).stem
            if self.tutorial_manager.load_tutorial(name):
                self._show_tutorial_overlay()
                self.tutorial_manager.start_playback()
                self.stop_tut_action.setEnabled(True)
                self.statusBar().showMessage(f"Playing tutorial: {name}")
            else:
                QMessageBox.warning(self, "Error", f"Could not load tutorial: {name}")

    def _show_tutorial_overlay(self):
        if self.tutorial_overlay is None:
            self.tutorial_overlay = TutorialOverlay(self)
            self.tutorial_overlay.next_clicked.connect(self.tutorial_manager.next_step)
            self.tutorial_overlay.prev_clicked.connect(self.tutorial_manager.prev_step)
            self.tutorial_overlay.stop_clicked.connect(self.stop_tutorial)

        # Position in top-right corner
        geo = self.geometry()
        x = geo.x() + geo.width() - 300
        y = geo.y() + 100
        self.tutorial_overlay.move(x, y)
        self.tutorial_overlay.show()

    def stop_tutorial(self):
        if self.tutorial_manager.is_recording:
            steps = self.tutorial_manager.stop_recording()
            if hasattr(self, "current_tutorial_name"):
                path = self.tutorial_manager.save_tutorial(self.current_tutorial_name)
                QMessageBox.information(
                    self, "Tutorial Saved", f"Saved {len(steps)} steps to:\n{path}"
                )
            self.statusBar().showMessage("Recording stopped.")
        elif self.tutorial_manager.is_playing:
            self.tutorial_manager.stop_playback()
            self.statusBar().showMessage("Tutorial playback stopped.")

        if self.tutorial_overlay:
            self.tutorial_overlay.hide()
            self.tutorial_overlay = None

        self.stop_tut_action.setEnabled(False)

    def _on_tutorial_step(self, current, total):
        msg = f"Tutorial Step {current + 1}/{total}"
        self.statusBar().showMessage(msg)
        if self.tutorial_overlay:
            instruction = self.tutorial_manager.get_current_instruction()
            description = self.tutorial_manager.get_current_description()
            self.tutorial_overlay.update_step(current, total, instruction, description)

    def _on_tutorial_finished(self):
        self.statusBar().showMessage("Tutorial Completed!")
        self.stop_tut_action.setEnabled(False)
        if self.tutorial_overlay:
            self.tutorial_overlay.hide()
            self.tutorial_overlay = None
        QMessageBox.information(self, "Tutorial", "Tutorial Completed!")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Bloch Simulator",
            "Bloch Equation Simulator\n\n"
            "A Python implementation of the Bloch equation solver\n"
            "originally developed by Brian Hargreaves.\n\n"
            "This GUI provides interactive visualization and\n"
            "parameter control for MRI pulse sequence simulation.\n\n"
            f"Version {__version__}",
        )


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = BlochSimulatorGUI()
    window.show()

    sys.exit(app.exec_())
