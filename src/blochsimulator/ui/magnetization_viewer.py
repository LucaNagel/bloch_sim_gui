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
    QSpinBox,
    QDoubleSpinBox,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QVector3D
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
from pathlib import Path
from ..visualization import ImageExporter


def centered_stride_indices(total: int, limit: int) -> np.ndarray:
    """Return at most ``limit`` indices into ``range(total)`` with one stride.

    A constant stride keeps the physical spacing uniform, and the indices are
    aligned to the central element so a symmetric axis stays symmetric.
    """
    total = max(int(total), 0)
    limit = max(int(limit), 1)
    if total <= limit:
        return np.arange(total)
    if limit == 1:
        return np.array([(total - 1) // 2])
    stride = int(np.ceil((total - 1) / (limit - 1)))
    start = ((total - 1) // 2) % stride
    return np.arange(start, total, stride)


class MagnetizationViewer(QWidget):
    """3D visualization of magnetization vector."""

    position_changed = pyqtSignal(int)
    view_filter_changed = pyqtSignal()

    ARRANGEMENT_OVERLAID = "Overlaid"
    ARRANGEMENT_ALONG_POSITION = "Along position"
    # Scene x coordinates of the "Along position" arrangement, in units of the
    # normalized vector length: gradient bars left of the spins, the net
    # vector left of the gradient bars, position labels on the right.
    NET_VECTOR_X = -4.4
    GRADIENT_BAR_X = -2.0
    GRADIENT_BAR_LENGTH = 0.8
    POSITION_LABEL_X = 1.15
    MAX_POSITION_LABELS = 7
    MAX_PATH_FRAMES = 1500

    def __init__(self):
        super().__init__()
        self._export_dir_provider = None
        self.init_ui()

    def init_ui(self):
        # Fill the page with the normal application window color so the raw
        # OpenGL tab matches the scroll-backed result tabs around it.
        self.setBackgroundRole(QPalette.Window)
        self.setAutoFillBackground(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.playhead_line = None

        # Add export header for 3D view
        header_container = QWidget()
        header_3d = QHBoxLayout(header_container)
        header_3d.setContentsMargins(0, 0, 0, 0)
        self.header_container = header_container
        title = QLabel("3D Magnetization Vector")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(max(title_font.pointSize() + 2, 12))
        title.setFont(title_font)
        title.setVisible(False)
        self.page_title = title
        header_3d.addWidget(title)
        header_3d.addStretch()

        self.export_3d_btn = QPushButton("Export Results")
        self.export_3d_btn.setObjectName("export_3d_btn")
        self.export_3d_btn.clicked.connect(self._trigger_parent_export)
        header_3d.addWidget(self.export_3d_btn)
        layout.addWidget(header_container)

        # 3D view
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=5)
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gl_widget.setMinimumHeight(300)
        # Camera preset that is refitted on resize until the user moves it.
        self._camera_preset = None
        self.gl_widget.installEventFilter(self)

        # Add coordinate axes
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.gl_widget.addItem(axis)
        self.axis_item = axis

        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(1, 1)
        self.gl_widget.addItem(grid)
        self.grid_item = grid

        # Initialize magnetization vectors (one per frequency)
        self.vector_plot = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)), color=(1, 0, 0, 1), width=3, mode="lines"
        )
        self.gl_widget.addItem(self.vector_plot)
        self.vector_colors = None

        # Items for the "Along position" arrangement: each position gets its
        # own origin on a vertical display axis, the net vector sits beside it.
        self.position_axis_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(0.6, 0.6, 0.6, 0.9), width=1, mode="lines"
        )
        self.position_mean_plot = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(1, 1, 1, 1), width=4, mode="lines"
        )
        self.tip_curve_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(1, 1, 1, 0.7), width=2, mode="line_strip"
        )
        self.tip_marker_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=7, pxMode=True
        )
        self.position_path_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(1, 1, 1, 0.6), width=1.5, mode="lines"
        )
        self.position_b1_plot = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(0, 1, 1, 0.9), width=2.5, mode="lines"
        )
        self.gradient_axis_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(0.6, 0.6, 0.6, 0.9), width=1, mode="lines"
        )
        self.gradient_bar_plot = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(1, 1, 1, 1), width=4, mode="lines"
        )
        self.field_label_item = gl.GLTextItem(
            pos=(0.0, 0.0, 0.0), text="", color=(200, 200, 200, 255)
        )
        for item in (
            self.position_axis_item,
            self.position_mean_plot,
            self.tip_curve_item,
            self.tip_marker_item,
            self.position_path_item,
            self.position_b1_plot,
            self.gradient_axis_item,
            self.gradient_bar_plot,
            self.field_label_item,
        ):
            item.setVisible(False)
            self.gl_widget.addItem(item)
        self.position_label_items = []
        self._spatial_available = False
        self._spatial_positions_mm = None
        self._spins_per_position = 1
        self._spatial_origins = np.zeros((0, 3))
        self._net_origin = np.zeros(3)
        self._last_spatial_frame = None
        self._position_color_cache = None
        self._position_path_points = []
        self._field_b1 = None
        self._field_offsets = None
        self._field_gradient_along = None
        self._field_frame_idx = None
        self._field_b1_max = 0.0
        self._field_offset_max = 0.0
        self._net_label_item = None

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
        self.view_layout = view_layout
        view_layout.addWidget(QLabel("View mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setObjectName("mag_3d_view_mode")
        self.view_mode_combo.addItems(
            ["All positions x freqs", "Positions @ freq", "Freqs @ position"]
        )
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        view_layout.addWidget(self.view_mode_combo)
        self.selector_label = QLabel("All spins")
        view_layout.addWidget(self.selector_label)
        self.selector_slider = QSlider(Qt.Horizontal)
        self.selector_slider.setObjectName("mag_3d_selector_slider")
        self.selector_slider.setRange(0, 0)
        self.selector_slider.valueChanged.connect(self._on_selector_changed)
        view_layout.addWidget(self.selector_slider, 1)
        view_layout.addWidget(QLabel("Color palette:"))
        self.vector_palette_combo = QComboBox()
        self.vector_palette_combo.setObjectName("mag_3d_vector_palette")
        self.vector_palette_combo.addItems(
            ["Rainbow", "Viridis", "Plasma", "Magma", "Cividis", "Inferno", "Gray"]
        )
        self.vector_palette_combo.setToolTip(
            "Choose the color palette used to distinguish displayed vectors."
        )
        self.vector_palette_combo.currentTextChanged.connect(
            lambda _: self.view_filter_changed.emit()
        )
        view_layout.addWidget(self.vector_palette_combo)
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
        self._has_vector_data = False
        self._has_mean_vector = False

        # Now create checkboxes that depend on these variables
        self.track_checkbox = QCheckBox("Track tip path")
        self.track_checkbox.setObjectName("mag_3d_track_checkbox")
        self.track_checkbox.setChecked(True)
        self.track_checkbox.toggled.connect(self._toggle_track_path)
        # Sync internal flag to the initial checkbox state so tracking is active on first playback
        self._toggle_track_path(self.track_checkbox.isChecked())
        view_layout.addWidget(self.track_checkbox)
        self.spin_display_combo = QComboBox()
        self.spin_display_combo.setObjectName("mag_3d_spin_display_mode")
        self.spin_display_combo.addItems(
            ["Show all spins", "Show all spins + mean", "Show only mean"]
        )
        self.spin_display_combo.setToolTip(
            "Choose whether to display individual spins, their mean, or both."
        )
        self.spin_display_combo.currentTextChanged.connect(
            self._on_spin_display_mode_changed
        )
        view_layout.addWidget(self.spin_display_combo)

        self._init_spatial_controls(controls_v)

        control_container.setLayout(controls_v)
        self.control_container = control_container
        # Keep all view settings above the plots, consistent with the other
        # result tabs and reachable without scrolling past the preview.
        layout.insertWidget(1, control_container)

        self.last_positions = None
        self.last_frequencies = None
        # Track available position/frequency counts for selector range updates
        self._npos = 1
        self._nfreq = 1
        self._update_selector_range()

        self.setLayout(layout)
        self.show_equilibrium_magnetization()

    def _init_spatial_controls(self, parent_layout):
        """Create the controls for arranging spins along their positions."""
        self.spatial_controls = QWidget()
        rows = QVBoxLayout(self.spatial_controls)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        rows.addLayout(row)
        row.addWidget(QLabel("Arrangement:"))
        self.arrangement_combo = QComboBox()
        self.arrangement_combo.setObjectName("mag_3d_arrangement")
        self.arrangement_combo.addItems(
            [self.ARRANGEMENT_OVERLAID, self.ARRANGEMENT_ALONG_POSITION]
        )
        self.arrangement_combo.setToolTip(
            "Draw all vectors from one origin, or give each position its own "
            "origin along a vertical display axis."
        )
        self.arrangement_combo.currentTextChanged.connect(
            lambda _: self.view_filter_changed.emit()
        )
        row.addWidget(self.arrangement_combo)

        # Options that only apply to the "Along position" arrangement.
        self.spatial_option_widgets = []
        max_label = QLabel("Max positions:")
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setObjectName("mag_3d_max_positions")
        self.max_positions_spin.setRange(2, 256)
        self.max_positions_spin.setValue(16)
        self.max_positions_spin.setToolTip(
            "Largest number of positions drawn; more positions are evenly subsampled."
        )
        self.max_positions_spin.valueChanged.connect(
            lambda _: self.view_filter_changed.emit()
        )
        length_label = QLabel("Vector length:")
        self.vector_length_spin = QDoubleSpinBox()
        self.vector_length_spin.setObjectName("mag_3d_vector_length")
        self.vector_length_spin.setRange(0.25, 5.0)
        self.vector_length_spin.setSingleStep(0.1)
        self.vector_length_spin.setValue(0.8)
        self.vector_length_spin.setSuffix(" × spacing")
        self.vector_length_spin.setToolTip(
            "Length of a fully relaxed vector relative to the spacing between "
            "positions; below 1 neighboring vectors along z do not overlap."
        )
        self.vector_length_spin.valueChanged.connect(
            lambda _: self._rebuild_spatial_scene(fit_camera=True)
        )
        color_label = QLabel("Color by:")
        self.color_by_combo = QComboBox()
        self.color_by_combo.setObjectName("mag_3d_color_by")
        self.color_by_combo.addItems(["Position", "Phase"])
        self.color_by_combo.setToolTip(
            "Color vectors by position (palette) or by transverse phase; "
            "with phase, longitudinal vectors fade to white."
        )
        self.color_by_combo.currentTextChanged.connect(
            lambda _: self._redraw_spatial_frame()
        )
        for widget in (
            max_label,
            self.max_positions_spin,
            length_label,
            self.vector_length_spin,
            color_label,
            self.color_by_combo,
        ):
            row.addWidget(widget)
            self.spatial_option_widgets.append(widget)
        row.addStretch()

        camera_label = QLabel("Camera:")
        row.addWidget(camera_label)
        self.spatial_option_widgets.append(camera_label)
        self.camera_buttons = {}
        for preset in ("Front", "Side", "Oblique"):
            button = QPushButton(preset)
            button.setObjectName(f"mag_3d_camera_{preset.lower()}")
            button.setToolTip(f"Point the camera at the scene: {preset.lower()} view.")
            button.clicked.connect(
                lambda _checked=False, name=preset: self.set_camera_preset(name)
            )
            row.addWidget(button)
            self.camera_buttons[preset] = button
            self.spatial_option_widgets.append(button)

        self.spatial_show_row = QWidget()
        show_row = QHBoxLayout(self.spatial_show_row)
        show_row.setContentsMargins(0, 0, 0, 0)
        show_row.addWidget(QLabel("Show:"))
        self.tip_markers_checkbox = QCheckBox("Tip markers")
        self.tip_markers_checkbox.setObjectName("mag_3d_tip_markers")
        self.tip_markers_checkbox.setChecked(True)
        self.tip_markers_checkbox.setToolTip("Mark the tip of each position vector.")
        self.tip_curve_checkbox = QCheckBox("Tip curve")
        self.tip_curve_checkbox.setObjectName("mag_3d_tip_curve")
        self.tip_curve_checkbox.setChecked(False)
        self.tip_curve_checkbox.setToolTip(
            "Connect the tips of neighboring positions, e.g. to show the "
            "phase helix after a gradient."
        )
        self.net_vector_checkbox = QCheckBox("Net vector")
        self.net_vector_checkbox.setObjectName("mag_3d_net_vector")
        self.net_vector_checkbox.setChecked(False)
        self.net_vector_checkbox.setToolTip(
            "Draw the mean over all positions and frequencies beside the column."
        )
        self.b1_field_checkbox = QCheckBox("B1")
        self.b1_field_checkbox.setObjectName("mag_3d_b1_field")
        self.b1_field_checkbox.setChecked(True)
        self.b1_field_checkbox.setToolTip(
            "Draw the RF field (rotating frame) at every position, scaled to "
            "the largest B1 of the sequence."
        )
        self.gradient_field_checkbox = QCheckBox("Gradient")
        self.gradient_field_checkbox.setObjectName("mag_3d_gradient_field")
        self.gradient_field_checkbox.setChecked(True)
        self.gradient_field_checkbox.setToolTip(
            "Draw the gradient field G·r at every position as a bar left of "
            "the column (red positive, blue negative), scaled to the largest "
            "value of the sequence."
        )
        for checkbox in (
            self.tip_markers_checkbox,
            self.tip_curve_checkbox,
            self.net_vector_checkbox,
        ):
            checkbox.toggled.connect(lambda _: self._redraw_spatial_frame())
            show_row.addWidget(checkbox)
        for checkbox in (self.b1_field_checkbox, self.gradient_field_checkbox):
            checkbox.toggled.connect(lambda _: self._draw_field_frame())
            show_row.addWidget(checkbox)
        show_row.addStretch()
        rows.addWidget(self.spatial_show_row)
        self.spatial_option_widgets.append(self.spatial_show_row)

        parent_layout.addWidget(self.spatial_controls)
        self.spatial_controls.setVisible(False)
        self.arrangement_combo.currentTextChanged.connect(
            lambda _: self._update_spatial_option_visibility()
        )
        self._update_spatial_option_visibility()

    def _update_spatial_option_visibility(self):
        along = self.get_vector_arrangement() == self.ARRANGEMENT_ALONG_POSITION
        for widget in self.spatial_option_widgets:
            widget.setVisible(along)

    def set_spatial_layout_available(self, available: bool):
        """Show the arrangement controls for viewers that can provide positions."""
        self._spatial_available = bool(available)
        self.spatial_controls.setVisible(self._spatial_available)
        if not self._spatial_available:
            self.set_spatial_groups(None)

    def get_vector_arrangement(self) -> str:
        """Return the selected arrangement of the displayed vectors."""
        if not getattr(self, "_spatial_available", False):
            return self.ARRANGEMENT_OVERLAID
        return self.arrangement_combo.currentText()

    def get_max_display_positions(self) -> int:
        """Return the largest number of positions drawn along the position axis."""
        return int(self.max_positions_spin.value())

    def is_spatial_layout_active(self) -> bool:
        """Return whether vectors are currently drawn at separate origins."""
        return self._spatial_positions_mm is not None

    def is_net_vector_shown(self) -> bool:
        """Return whether the spatial layout draws the net vector."""
        return self.net_vector_checkbox.isChecked()

    def set_spatial_groups(self, positions_mm, spins_per_position: int = 1):
        """Arrange vectors along their positions, or pass None to overlay them.

        Parameters
        ----------
        positions_mm : array-like or None
            Position (mm) of every displayed position group, in display order.
        spins_per_position : int
            Number of consecutive vectors (isochromats) per position group.
        """
        new = None
        if positions_mm is not None:
            arr = np.asarray(positions_mm, dtype=float).ravel()
            if arr.size >= 2:
                new = arr
        old = self._spatial_positions_mm
        layout_changed = (old is None) != (new is None) or (
            new is not None and (old.size != new.size or not np.allclose(old, new))
        )
        self._spatial_positions_mm = new
        self._spins_per_position = max(1, int(spins_per_position))
        if new is None:
            self.set_spatial_fields(None)
        if layout_changed:
            self._last_spatial_frame = None
            self._clear_path()
            self._rebuild_spatial_scene(fit_camera=True)

    def _spatial_spacing(self) -> float:
        return 1.0 / max(float(self.vector_length_spin.value()), 1e-6)

    def _scene_bounds(self):
        """Return the (x_min, x_max, z_min, z_max) range of the arrangement."""
        if not self.is_spatial_layout_active() or not len(self._spatial_origins):
            return -1.0, 1.0, -1.0, 1.0
        zs = self._spatial_origins[:, 2]
        x_min = self.GRADIENT_BAR_X - self.GRADIENT_BAR_LENGTH - 0.2
        if self.is_net_vector_shown():
            x_min = self.NET_VECTOR_X - 1.1
        x_max = self.POSITION_LABEL_X + 1.2
        return x_min, x_max, float(zs[0]) - 1.1, float(zs[-1]) + 1.4

    def _rebuild_spatial_scene(self, fit_camera: bool = False):
        """Lay out origins, position axis, labels and the net-vector anchor."""
        for item in self.position_label_items:
            self.gl_widget.removeItem(item)
        self.position_label_items = []
        anchored = (self.mean_vector, self.path_item)

        if not self.is_spatial_layout_active():
            self._spatial_origins = np.zeros((0, 3))
            self._net_origin = np.zeros(3)
            for item in (
                self.position_axis_item,
                self.position_mean_plot,
                self.tip_curve_item,
                self.tip_marker_item,
                self.position_path_item,
                self.position_b1_plot,
                self.gradient_axis_item,
                self.gradient_bar_plot,
                self.field_label_item,
            ):
                item.setVisible(False)
            for item in anchored + (self.axis_item, self.grid_item):
                item.resetTransform()
            self.axis_item.setSize(2, 2, 2)
            self.grid_item.setSize(4, 4)
            if fit_camera:
                self.set_camera_preset("Front")
            return

        positions_mm = self._spatial_positions_mm
        count = positions_mm.size
        spacing = self._spatial_spacing()
        zs = (np.arange(count) - 0.5 * (count - 1)) * spacing
        self._spatial_origins = np.column_stack([np.zeros(count), np.zeros(count), zs])
        self._net_origin = np.array([self.NET_VECTOR_X, 0.0, 0.0])

        tick = min(0.08, 0.3 * spacing)
        segments = [
            [0.0, 0.0, zs[0] - 0.5 * spacing],
            [0.0, 0.0, zs[-1] + 0.5 * spacing],
        ]
        for z in zs:
            segments.extend([[-tick, 0.0, z], [tick, 0.0, z]])
        self.position_axis_item.setData(pos=np.asarray(segments, dtype=np.float32))
        self.position_axis_item.setVisible(True)

        gx = self.GRADIENT_BAR_X
        self.gradient_axis_item.setData(
            pos=np.array(
                [[gx, 0.0, zs[0] - 0.5 * spacing], [gx, 0.0, zs[-1] + 0.5 * spacing]],
                dtype=np.float32,
            )
        )

        label_color = (180, 180, 180, 255)
        label_indices = centered_stride_indices(count, self.MAX_POSITION_LABELS)
        for idx in label_indices:
            label = gl.GLTextItem(
                pos=(self.POSITION_LABEL_X, 0.0, float(zs[idx])),
                text=f"{positions_mm[idx]:.1f} mm",
                color=label_color,
            )
            self.gl_widget.addItem(label)
            self.position_label_items.append(label)
        net_label = gl.GLTextItem(
            pos=(self.NET_VECTOR_X - 0.15, 0.0, -1.3), text="net", color=label_color
        )
        net_label.setVisible(self.is_net_vector_shown())
        self.gl_widget.addItem(net_label)
        self.position_label_items.append(net_label)
        self._net_label_item = net_label
        self.field_label_item.setData(
            pos=(self.GRADIENT_BAR_X - self.GRADIENT_BAR_LENGTH, 0.0, zs[-1] + 1.2)
        )

        for item in anchored:
            item.resetTransform()
            item.translate(*self._net_origin)
        # Axis triad and floor grid sit below the column.
        floor = float(zs[0]) - 1.05
        self.axis_item.resetTransform()
        self.axis_item.translate(0.0, 0.0, floor)
        self.axis_item.setSize(1, 1, 1)
        self.grid_item.resetTransform()
        self.grid_item.translate(0.0, 0.0, floor)
        self.grid_item.setSize(4, 4)
        if fit_camera:
            self.set_camera_preset("Front")
        self._redraw_spatial_frame()
        self._draw_field_frame()

    def set_camera_preset(self, preset: str):
        """Point the camera at the whole scene from a fixed direction."""
        x_min, x_max, z_min, z_max = self._scene_bounds()
        width = max(x_max - x_min, 2.0)
        height = max(z_max - z_min, 2.0)
        center = QVector3D(0.5 * (x_min + x_max), 0.0, 0.5 * (z_min + z_max))
        if preset == "Side":
            elevation, azimuth = 10.0, 0.0
            width = 2.6
        elif preset == "Oblique":
            elevation, azimuth = 22.0, -55.0
        else:
            elevation, azimuth = 12.0, -90.0
        # The field of view is horizontal; fit both extents to the widget.
        view_w = max(self.gl_widget.width(), 1)
        view_h = max(self.gl_widget.height(), 1)
        span = 2.0 * np.tan(0.5 * np.radians(self.gl_widget.opts.get("fov", 60)))
        distance = 1.1 * max(width / span, height * view_w / (view_h * span), 4.0)
        self.gl_widget.setCameraPosition(
            pos=center, distance=distance, elevation=elevation, azimuth=azimuth
        )
        self._camera_preset = (preset, self._camera_state())

    def _camera_state(self):
        opts = self.gl_widget.opts
        center = opts["center"]
        return (
            round(float(opts["distance"]), 6),
            round(float(opts["elevation"]), 6),
            round(float(opts["azimuth"]), 6),
            round(center.x(), 6),
            round(center.y(), 6),
            round(center.z(), 6),
        )

    def eventFilter(self, obj, event):
        """Keep a preset camera fitted while the 3D view changes size."""
        if (
            obj is self.gl_widget
            and event.type() == QEvent.Resize
            and self._camera_preset is not None
            and self._camera_preset[1] == self._camera_state()
        ):
            self.set_camera_preset(self._camera_preset[0])
        return super().eventFilter(obj, event)

    def _position_colors(self, count: int) -> np.ndarray:
        """Return palette RGBA colors (0..1) for each position group."""
        key = (self.vector_palette_combo.currentText(), int(count))
        if self._position_color_cache is None or self._position_color_cache[0] != key:
            colors = np.array(
                [self.color_for_index(i, count).getRgbF() for i in range(count)],
                dtype=float,
            )
            self._position_color_cache = (key, colors)
        return self._position_color_cache[1].copy()

    @staticmethod
    def phase_colors(vectors) -> np.ndarray:
        """Return RGBA colors with hue = transverse phase, saturation = |Mxy|/|M|."""
        vecs = np.asarray(vectors, dtype=float).reshape(-1, 3)
        mxy = np.hypot(vecs[:, 0], vecs[:, 1])
        magnitude = np.linalg.norm(vecs, axis=1)
        hue = (np.arctan2(vecs[:, 1], vecs[:, 0]) / (2.0 * np.pi)) % 1.0
        sat = np.clip(mxy / np.maximum(magnitude, 1e-12), 0.0, 1.0)
        sector = np.floor(hue * 6.0).astype(int) % 6
        frac = hue * 6.0 - np.floor(hue * 6.0)
        p = 1.0 - sat
        q = 1.0 - frac * sat
        t = 1.0 - (1.0 - frac) * sat
        one = np.ones_like(hue)
        r = np.choose(sector, [one, q, p, p, t, one])
        g = np.choose(sector, [t, one, one, q, p, p])
        b = np.choose(sector, [p, p, t, one, one, q])
        return np.column_stack([r, g, b, one])

    def _redraw_spatial_frame(self):
        """Redraw the last spatial frame after a display option changed."""
        if not self.is_spatial_layout_active():
            return
        label = getattr(self, "_net_label_item", None)
        if label is not None:
            label.setVisible(self.is_net_vector_shown())
        if self._last_spatial_frame is None:
            return
        vecs, means, net = self._last_spatial_frame
        self._update_spatial_magnetization(vecs, means, net, append_path=False)

    def _update_spatial_magnetization(self, vecs, means, net, append_path=True):
        """Draw spins, position means and the net vector at separate origins."""
        count = self._spatial_positions_mm.size
        per_position = self._spins_per_position
        norm = 1.0 / max(self.length_scale, 1e-9)
        spins = np.asarray(vecs, dtype=float).reshape(count, per_position, 3) * norm
        if means is None:
            means = spins.mean(axis=1)
        else:
            means = np.asarray(means, dtype=float).reshape(count, 3) * norm
        net = spins.mean(axis=(0, 1)) if net is None else np.asarray(net) * norm
        origins = self._spatial_origins

        mode = self.get_spin_display_mode()
        show_fan = per_position > 1 and mode != "Show only mean"
        show_means = per_position == 1 or mode != "Show all spins"
        show_net = self.is_net_vector_shown()
        by_phase = self.color_by_combo.currentText() == "Phase"
        position_colors = self._position_colors(count)

        if show_fan:
            pos = np.empty((count * per_position * 2, 3), dtype=np.float32)
            pos[0::2] = np.repeat(origins, per_position, axis=0)
            pos[1::2] = (origins[:, None, :] + spins).reshape(-1, 3)
            if by_phase:
                fan_colors = self.phase_colors(spins)
            else:
                fan_colors = np.repeat(position_colors, per_position, axis=0)
            if show_means:
                fan_colors[:, 3] = 0.45
            self.vector_plot.setData(
                pos=pos,
                color=np.repeat(fan_colors, 2, axis=0),
                mode="lines",
                width=1.5,
            )
        self.vector_plot.setVisible(show_fan)

        mean_tips = origins + means
        mean_colors = self.phase_colors(means) if by_phase else position_colors
        if show_means:
            pos = np.empty((count * 2, 3), dtype=np.float32)
            pos[0::2] = origins
            pos[1::2] = mean_tips
            self.position_mean_plot.setData(
                pos=pos, color=np.repeat(mean_colors, 2, axis=0), mode="lines"
            )
            if self.tip_markers_checkbox.isChecked():
                self.tip_marker_item.setData(pos=mean_tips, color=mean_colors)
            if self.tip_curve_checkbox.isChecked():
                self.tip_curve_item.setData(pos=mean_tips.astype(np.float32))
        self.position_mean_plot.setVisible(show_means)
        self.tip_marker_item.setVisible(
            show_means and self.tip_markers_checkbox.isChecked()
        )
        self.tip_curve_item.setVisible(
            show_means and self.tip_curve_checkbox.isChecked()
        )

        self._has_vector_data = True
        self._has_mean_vector = True
        if append_path and self.track_path:
            self._position_path_points.append(means.copy())
            if len(self._position_path_points) > self.MAX_PATH_FRAMES:
                del self._position_path_points[: -self.MAX_PATH_FRAMES]
            if show_net:
                self._append_path_point(net)
        self._draw_position_paths(position_colors, by_phase)
        self.path_item.setVisible(show_net)
        self.mean_vector.setData(pos=np.array([[0.0, 0.0, 0.0], net]))
        self.mean_vector.setVisible(show_net)

    def _draw_position_paths(self, position_colors, by_phase):
        """Draw the tracked tip path of every position (one per fan)."""
        frames = len(self._position_path_points)
        count = len(self._spatial_origins)
        if not self.track_path or frames < 2:
            self.position_path_item.setVisible(False)
            return
        history = np.asarray(self._position_path_points, dtype=np.float32)
        if history.shape[1] != count:
            self._position_path_points = []
            self.position_path_item.setVisible(False)
            return
        tips = history + self._spatial_origins[None, :, :].astype(np.float32)
        pos = np.empty((frames - 1, count, 2, 3), dtype=np.float32)
        pos[:, :, 0] = tips[:-1]
        pos[:, :, 1] = tips[1:]
        if by_phase:
            colors = np.tile([0.85, 0.85, 0.85, 0.6], (count, 1))
        else:
            colors = position_colors.copy()
            colors[:, 3] = 0.7
        vertex_colors = np.broadcast_to(
            colors[None, :, None, :], (frames - 1, count, 2, 4)
        ).reshape(-1, 4)
        self.position_path_item.setData(
            pos=pos.reshape(-1, 3), color=np.ascontiguousarray(vertex_colors)
        )
        self.position_path_item.setVisible(True)

    def set_spatial_fields(self, b1=None, gradients=None, positions_m=None):
        """Store the fields seen by the drawn positions for every playback frame.

        Parameters
        ----------
        b1 : array-like or None
            Complex RF field per frame (G).
        gradients : array-like or None
            Gradient (gx, gy, gz) per frame (G/cm).
        positions_m : array-like or None
            Position (x, y, z) in m of every drawn position group.
        """
        self._field_b1 = None
        self._field_offsets = None
        self._field_gradient_along = None
        if b1 is not None:
            b1_arr = np.asarray(b1).ravel()
            if b1_arr.size:
                self._field_b1 = b1_arr
        if gradients is not None and positions_m is not None:
            grads = np.asarray(gradients, dtype=float)
            if grads.ndim == 1:
                grads = grads[:, None]
            padded = np.zeros((len(grads), 3))
            padded[:, : min(grads.shape[1], 3)] = grads[:, :3]
            pos_cm = np.asarray(positions_m, dtype=float).reshape(-1, 3) * 100.0
            if len(padded) and len(pos_cm):
                # Field offset G·r (G) at every drawn position and frame.
                self._field_offsets = padded @ pos_cm.T
                spread = pos_cm[-1] - pos_cm[0]
                length = float(np.linalg.norm(spread))
                direction = spread / length if length > 0 else np.array([0, 0, 1.0])
                self._field_gradient_along = padded @ direction
        self._field_b1_max = (
            float(np.nanmax(np.abs(self._field_b1)))
            if self._field_b1 is not None
            else 0.0
        )
        self._field_offset_max = (
            float(np.nanmax(np.abs(self._field_offsets)))
            if self._field_offsets is not None
            else 0.0
        )
        self._draw_field_frame()

    def show_field_frame(self, idx: int):
        """Draw B1 and gradient fields of playback frame ``idx``."""
        self._field_frame_idx = int(idx)
        self._draw_field_frame()

    def _draw_field_frame(self):
        items = (
            self.position_b1_plot,
            self.gradient_axis_item,
            self.gradient_bar_plot,
            self.field_label_item,
        )
        idx = self._field_frame_idx
        if not self.is_spatial_layout_active() or idx is None:
            for item in items:
                item.setVisible(False)
            return
        origins = self._spatial_origins
        count = len(origins)
        readout = []

        b1_shown = False
        if self.b1_field_checkbox.isChecked() and self._field_b1 is not None:
            b1_val = self._field_b1[min(max(idx, 0), len(self._field_b1) - 1)]
            mag = abs(b1_val)
            if np.isfinite(mag) and self._field_b1_max > 0 and mag > 1e-9:
                # Same orientation as the overlay B1 indicator.
                phase = np.pi - np.angle(b1_val)
                length = mag / self._field_b1_max
                tip = np.array([np.cos(phase), np.sin(phase), 0.0]) * length
                pos = np.empty((count * 2, 3), dtype=np.float32)
                pos[0::2] = origins
                pos[1::2] = origins + tip
                self.position_b1_plot.setData(pos=pos)
                b1_shown = True
            readout.append(f"B1 {mag * 100.0:.2f} µT")
        self.position_b1_plot.setVisible(b1_shown)

        grad_shown = (
            self.gradient_field_checkbox.isChecked()
            and self._field_offsets is not None
            and self._field_offsets.shape[1] == count
        )
        if grad_shown:
            frame = min(max(idx, 0), len(self._field_offsets) - 1)
            offsets = self._field_offsets[frame]
            scale = (
                self.GRADIENT_BAR_LENGTH / self._field_offset_max
                if self._field_offset_max > 0
                else 0.0
            )
            gx = self.GRADIENT_BAR_X
            pos = np.empty((count * 2, 3), dtype=np.float32)
            pos[0::2] = origins + [gx, 0.0, 0.0]
            pos[1::2] = origins + [gx, 0.0, 0.0]
            pos[1::2, 0] += offsets * scale
            colors = np.where(
                (offsets >= 0)[:, None],
                np.array([1.0, 0.35, 0.3, 1.0]),
                np.array([0.35, 0.55, 1.0, 1.0]),
            )
            self.gradient_bar_plot.setData(pos=pos, color=np.repeat(colors, 2, axis=0))
            along = float(self._field_gradient_along[frame])
            readout.append(f"G {along * 10.0:.2f} mT/m")
        self.gradient_bar_plot.setVisible(grad_shown)
        self.gradient_axis_item.setVisible(grad_shown)

        self.field_label_item.setData(text="   ".join(readout))
        self.field_label_item.setVisible(bool(readout))

    def show_equilibrium_magnetization(self):
        """Display normalized longitudinal magnetization before a simulation."""
        self.update_magnetization(np.array([[0.0, 0.0, 1.0]], dtype=float))
        # The static startup state is not part of a simulated trajectory.
        self._clear_path()

    def color_for_index(self, index: int, total: int):
        """Return the selected 3D palette color for one displayed vector."""
        total = max(1, int(total))
        index = max(0, min(int(index), total - 1))
        palette = self.vector_palette_combo.currentText()
        if palette == "Rainbow":
            return pg.intColor(index, hues=total, values=1.0, maxValue=255)
        try:
            cmap = pg.colormap.get(palette.lower())
            position = 0.5 if total == 1 else index / (total - 1)
            return cmap.map(position, mode="qcolor")
        except Exception:
            return pg.intColor(index, hues=total, values=1.0, maxValue=255)

    def showEvent(self, event):
        """Refresh OpenGL after a hidden tab receives its final geometry."""
        super().showEvent(event)
        QTimer.singleShot(0, self.refresh_viewport)

    def refresh_viewport(self):
        """Repaint the 3D surface without changing the user's camera."""
        if not self.isVisible() or not self.gl_widget.isVisible():
            return
        self.updateGeometry()
        self.gl_widget.updateGeometry()
        self.gl_widget.update()

    def _ensure_vectors(self, count: int, colors=None):
        """Cache colors for vector updates."""
        if colors is not None:
            self.vector_colors = colors

    def set_length_scale(self, scale: float):
        """Set reference magnitude to normalize displayed vectors."""
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        self.length_scale = float(scale)

    def update_magnetization(
        self, mx, my=None, mz=None, colors=None, position_means=None, net=None
    ):
        """
        Update the magnetization vector display.
        Accepts either separate mx/my/mz scalars/arrays or an array of shape (nfreq, 3).

        With an active spatial layout (see ``set_spatial_groups``) the vectors
        are grouped per position; ``position_means`` (npos, 3) and ``net`` (3,)
        then supply the per-position and overall means, which may cover more
        isochromats than the drawn ones.
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
        if (
            self.is_spatial_layout_active()
            and count == self._spatial_positions_mm.size * self._spins_per_position
        ):
            self._last_spatial_frame = (vecs, position_means, net)
            self._update_spatial_magnetization(vecs, position_means, net)
            return
        self._has_vector_data = count > 0

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
                self.vector_plot.setData(pos=pos, color=c_arr, mode="lines", width=3)
            else:
                c_expanded = np.repeat(c_arr[:count], 2, axis=0)
                self.vector_plot.setData(
                    pos=pos, color=c_expanded, mode="lines", width=3
                )
        else:
            self.vector_plot.setData(pos=pos, color=(1, 0, 0, 1), mode="lines", width=3)
        self.vector_plot.setVisible(
            self._has_vector_data and self.get_spin_display_mode() != "Show only mean"
        )

        mean_vec = np.mean(vecs_scaled, axis=0) if vecs.size else None
        self._has_mean_vector = mean_vec is not None
        if self.track_path and mean_vec is not None:
            self._append_path_point(mean_vec)
        # Mean vector (over all components)
        if mean_vec is not None:
            self.mean_vector.setData(pos=np.array([[0, 0, 0], mean_vec]))
        self.mean_vector.setVisible(
            self._has_mean_vector
            and self.get_spin_display_mode()
            in {"Show all spins + mean", "Show only mean"}
        )

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
                self.last_positions[idx, 2] * 1000
                if self.last_positions is not None and idx < len(self.last_positions)
                else idx
            )
            label_text = f"Pos: {pos_val:.2f} mm"
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
        self.arrangement_combo.setEnabled(self._npos > 1)
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

    def get_spin_display_mode(self) -> str:
        """Return which individual and mean vectors should be visible."""
        return (
            self.spin_display_combo.currentText()
            if hasattr(self, "spin_display_combo")
            else "Show all spins"
        )

    def _on_spin_display_mode_changed(self, _mode: str):
        if self.is_spatial_layout_active():
            self._redraw_spatial_frame()
            self.view_filter_changed.emit()
            return
        show_spins = self.get_spin_display_mode() != "Show only mean"
        show_mean = self.get_spin_display_mode() in {
            "Show all spins + mean",
            "Show only mean",
        }
        self.vector_plot.setVisible(self._has_vector_data and show_spins)
        self.mean_vector.setVisible(self._has_mean_vector and show_mean)
        self.view_filter_changed.emit()

    def _toggle_track_path(self, enabled: bool):
        self.track_path = enabled
        if not enabled:
            self._clear_path()

    def _clear_path(self):
        self.path_points = []
        self.path_item.setData(pos=np.zeros((0, 3)))
        self._position_path_points = []
        self.position_path_item.setVisible(False)

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

    def _trigger_parent_export(self):
        """Delegate unified export to the parent window."""
        win = self.window()
        if win and hasattr(win, "export_results"):
            win.export_results()
        else:
            # Fallback to current screenshot if parent not found
            self._export_3d_screenshot("png")
