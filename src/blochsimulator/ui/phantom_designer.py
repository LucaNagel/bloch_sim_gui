"""Interactive shape and Lorentz-peak phantom designer."""

from __future__ import annotations

from typing import Optional
import weakref

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..phantom import Phantom
from ..dynamic_phantom import (
    DynamicSpectralPhantom,
    KineticRegionDefinition,
    TimeCurve,
    kinetic_preroll_start_s,
    simulate_two_pool_kinetics,
)
from ..paths import workspace_directory
from ..phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from ..spectral_phantom import SpectralPhantom
from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .volume_viewer import PhantomInspectorWidget


class ShapeDrawingPlotWidget(pg.PlotWidget):
    """Plot widget with a one-shot drag-to-create shape mode."""

    shapeDrawn = pyqtSignal(str, float, float, float, float)
    drawingCancelled = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drawing_kind = None
        self._drawing_start = None
        self._drawing_guide = pg.PlotDataItem(
            pen=pg.mkPen((255, 220, 80), width=3, style=Qt.DashLine)
        )
        self._drawing_guide.setZValue(2000)
        self.addItem(self._drawing_guide)

    @property
    def drawing_kind(self):
        return self._drawing_kind

    def start_shape_drawing(self, kind):
        if kind not in {"ellipsoid", "box"}:
            raise ValueError("drawing kind must be 'ellipsoid' or 'box'")
        self._drawing_kind = kind
        self._drawing_start = None
        self._drawing_guide.setData([], [])
        self.setCursor(Qt.CrossCursor)

    def cancel_shape_drawing(self):
        was_active = self._drawing_kind is not None
        self._drawing_kind = None
        self._drawing_start = None
        self._drawing_guide.setData([], [])
        self.unsetCursor()
        if was_active:
            self.drawingCancelled.emit()

    def _plot_position(self, event):
        scene_position = self.mapToScene(event.pos())
        return self.plotItem.vb.mapSceneToView(scene_position)

    def _update_drawing_guide(self, end):
        start = self._drawing_start
        if start is None:
            return
        left, right = sorted((float(start.x()), float(end.x())))
        bottom, top = sorted((float(start.y()), float(end.y())))
        if self._drawing_kind == "box":
            x = [left, right, right, left, left]
            y = [bottom, bottom, top, top, bottom]
        else:
            angle = np.linspace(0.0, 2.0 * np.pi, 65)
            x = (left + right) / 2.0 + (right - left) / 2.0 * np.cos(angle)
            y = (bottom + top) / 2.0 + (top - bottom) / 2.0 * np.sin(angle)
        self._drawing_guide.setData(x, y)

    def mousePressEvent(self, event):
        if self._drawing_kind is not None and event.button() == Qt.LeftButton:
            self._drawing_start = self._plot_position(event)
            self._update_drawing_guide(self._drawing_start)
            event.accept()
            return
        if self._drawing_kind is not None and event.button() == Qt.RightButton:
            self.cancel_shape_drawing()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing_kind is not None and self._drawing_start is not None:
            self._update_drawing_guide(self._plot_position(event))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._drawing_kind is not None
            and self._drawing_start is not None
            and event.button() == Qt.LeftButton
        ):
            end = self._plot_position(event)
            left, right = sorted((float(self._drawing_start.x()), float(end.x())))
            bottom, top = sorted((float(self._drawing_start.y()), float(end.y())))
            left, right = np.clip((left, right), 0.0, 1.0)
            bottom, top = np.clip((bottom, top), 0.0, 1.0)
            kind = self._drawing_kind
            self._drawing_kind = None
            self._drawing_start = None
            self._drawing_guide.setData([], [])
            self.unsetCursor()
            if right - left >= 1e-4 and top - bottom >= 1e-4:
                self.shapeDrawn.emit(kind, left, bottom, right - left, top - bottom)
            else:
                self.drawingCancelled.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self._drawing_kind is not None and event.key() == Qt.Key_Escape:
            self.cancel_shape_drawing()
            event.accept()
            return
        super().keyPressEvent(event)


def load_any_phantom(filename):
    """Load either a conventional or spectral phantom file."""
    try:
        return DynamicSpectralPhantom.load(filename)
    except ValueError:
        try:
            return SpectralPhantom.load(filename)
        except ValueError as spectral_error:
            try:
                return Phantom.load(filename)
            except Exception:
                raise spectral_error


class SpectralPhantomDesignerDialog(QDialog):
    """Draw multiple extruded XY shapes and assign Lorentzian peak lists."""

    def __init__(self, parent=None, design: Optional[PhantomDesign] = None):
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Spectral Phantom" if design is not None else "New Spectral Phantom"
        )
        self.resize(1250, 850)
        self.design = design or PhantomDesign(shapes=[ShapeDefinition(name="Shape 1")])
        self.phantom = None
        self._updating = False
        self._rois = []
        self._build_ui()
        self._load_design_into_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        draw_page = QWidget()
        draw_layout = QVBoxLayout(draw_page)
        global_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        global_row.addWidget(QLabel("Name"))
        global_row.addWidget(self.name_edit)
        self.matrix_spins = []
        self.fov_spins = []
        for axis in "XYZ":
            matrix = QSpinBox()
            matrix.setRange(1, 256)
            global_row.addWidget(QLabel(f"N{axis.lower()}"))
            global_row.addWidget(matrix)
            self.matrix_spins.append(matrix)
        for axis in "XYZ":
            fov = QDoubleSpinBox()
            fov.setRange(0.01, 1000.0)
            fov.setDecimals(3)
            fov.setSuffix(" mm")
            global_row.addWidget(QLabel(f"FOV {axis}"))
            global_row.addWidget(fov)
            self.fov_spins.append(fov)
        draw_layout.addLayout(global_row)

        spectral_row = QHBoxLayout()
        spectral_row.addWidget(QLabel("B0"))
        self.field_strength_t = self._number_spin(0.001, 1000.0, 3.0, " T")
        self.field_strength_t.setToolTip(
            "Main field used to convert ppm peak and B0 offsets to Hz"
        )
        spectral_row.addWidget(self.field_strength_t)
        spectral_row.addWidget(QLabel("Nucleus"))
        self.nucleus = QComboBox()
        self.nucleus.addItem("Auto (H1 static / C13 dynamic)", None)
        for nucleus in sorted(NUCLEUS_GAMMA_HZ_PER_T):
            self.nucleus.addItem(nucleus, nucleus)
        self.nucleus.setToolTip(
            "Nucleus used together with B0 for ppm-to-Hz conversion; Auto "
            "preserves the H1 static and C13 dynamic defaults"
        )
        spectral_row.addWidget(self.nucleus)
        spectral_row.addWidget(QLabel("Spectral reference"))
        self.spectral_reference_ppm = self._number_spin(-10000.0, 10000.0, 0.0, " ppm")
        self.spectral_reference_ppm.setToolTip(
            "Scanner carrier/reference in absolute ppm. Internally the sequence "
            "is simulated at 0 ppm and peaks are stored as offsets from this value."
        )
        spectral_row.addWidget(self.spectral_reference_ppm)
        spectral_row.addWidget(QLabel("Bandwidth"))
        self.spectral_bandwidth_ppm = self._number_spin(0.0001, 10000.0, 20.0, " ppm")
        self.spectral_bandwidth_ppm.setToolTip(
            "Default frequency span for per-voxel spectral previews, centered on 0 ppm."
        )
        spectral_row.addWidget(self.spectral_bandwidth_ppm)
        spectral_row.addWidget(QLabel("Points"))
        self.spectral_points = QSpinBox()
        self.spectral_points.setRange(2, 65536)
        self.spectral_points.setValue(1024)
        self.spectral_points.setToolTip(
            "Default number of samples in spectral previews"
        )
        spectral_row.addWidget(self.spectral_points)
        self.spectral_resolution_info = QLabel()
        spectral_row.addWidget(self.spectral_resolution_info)
        spectral_row.addStretch()
        self.spectral_reference_ppm.valueChanged.connect(
            self._spectral_settings_changed
        )
        self.field_strength_t.valueChanged.connect(self._spectral_settings_changed)
        self.nucleus.currentIndexChanged.connect(self._spectral_settings_changed)
        self.spectral_bandwidth_ppm.valueChanged.connect(
            self._spectral_settings_changed
        )
        self.spectral_points.valueChanged.connect(self._spectral_settings_changed)
        draw_layout.addLayout(spectral_row)

        b0_row = QHBoxLayout()
        b0_row.addWidget(QLabel("Global B0 inhomogeneity"))
        self.b0_mode_combo = QComboBox()
        self.b0_mode_combo.addItem("None", "none")
        self.b0_mode_combo.addItem("Linear X (2D)", "linear_x")
        self.b0_mode_combo.addItem("Linear Y (2D)", "linear_y")
        self.b0_mode_combo.addItem("Linear Z (3D)", "linear_z")
        self.b0_mode_combo.addItem("Radial XY (2D)", "radial_xy")
        self.b0_mode_combo.addItem("Radial XYZ (3D)", "radial_xyz")
        self.b0_mode_combo.setToolTip(
            "Analytic spatial B0 variation added to each shape's constant offset"
        )
        b0_row.addWidget(self.b0_mode_combo)
        self.b0_inhomogeneity_ppm = self._number_spin(-1000.0, 1000.0, 0.0, " ppm")
        self.b0_inhomogeneity_ppm.setToolTip(
            "Signed maximum deviation at the edge/corners of the FOV"
        )
        b0_row.addWidget(QLabel("Edge amplitude"))
        b0_row.addWidget(self.b0_inhomogeneity_ppm)
        b0_row.addStretch()
        draw_layout.addLayout(b0_row)

        self.shape_splitter = QSplitter(Qt.Horizontal)
        splitter = self.shape_splitter
        draw_layout.addWidget(splitter)

        shape_panel = QWidget()
        shape_panel.setMinimumWidth(170)
        shape_panel.setMaximumWidth(300)
        shape_layout = QVBoxLayout(shape_panel)
        shape_heading = QLabel("Shapes")
        shape_heading.setToolTip("Later shapes overwrite B0 values in overlaps")
        shape_layout.addWidget(shape_heading)
        self.shape_list = QListWidget()
        self.shape_list.currentRowChanged.connect(self._shape_selected)
        shape_layout.addWidget(self.shape_list)
        shape_buttons = QGridLayout()
        add_ellipse = QPushButton("Add ellipsoid")
        add_ellipse.clicked.connect(lambda: self._add_shape("ellipsoid"))
        add_box = QPushButton("Add box")
        add_box.clicked.connect(lambda: self._add_shape("box"))
        draw_ellipse = QPushButton("Draw ellipsoid")
        draw_ellipse.setToolTip("Drag a new ellipsoid directly in the XY canvas")
        draw_ellipse.clicked.connect(lambda: self._start_shape_drawing("ellipsoid"))
        draw_box = QPushButton("Draw box")
        draw_box.setToolTip("Drag a new box directly in the XY canvas")
        draw_box.clicked.connect(lambda: self._start_shape_drawing("box"))
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_shape)
        shape_buttons.addWidget(add_ellipse, 0, 0)
        shape_buttons.addWidget(add_box, 1, 0)
        shape_buttons.addWidget(draw_ellipse, 2, 0)
        shape_buttons.addWidget(draw_box, 3, 0)
        shape_buttons.addWidget(remove, 4, 0)
        shape_layout.addLayout(shape_buttons)
        splitter.addWidget(shape_panel)

        canvas_panel = QWidget()
        canvas_panel.setMinimumWidth(420)
        canvas_layout = QVBoxLayout(canvas_panel)
        self.canvas_instruction = QLabel(
            "Move/resize existing shapes, or choose Draw and drag in the axial XY plane"
        )
        canvas_layout.addWidget(self.canvas_instruction)
        self.canvas = ShapeDrawingPlotWidget()
        self.canvas.shapeDrawn.connect(self._shape_drawn)
        self.canvas.drawingCancelled.connect(self._shape_drawing_cancelled)
        self.canvas.setAspectLocked(True)
        self.canvas.setXRange(0, 1)
        self.canvas.setYRange(0, 1)
        self.canvas.showGrid(x=True, y=True, alpha=0.3)
        self.canvas.setLabel("bottom", "x", units="mm")
        self.canvas.setLabel("left", "y", units="mm")
        self.fov_outline = pg.PlotDataItem(
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            pen=pg.mkPen((240, 240, 240), width=3),
        )
        self.fov_outline.setZValue(1000)
        self.canvas.addItem(self.fov_outline)
        dialog_ref = weakref.ref(self)

        def physical_ticks(axis_index):
            def tick_strings(values, scale, spacing):
                dialog = dialog_ref()
                fov_mm = dialog.fov_spins[axis_index].value() if dialog else 1.0
                return [f"{(value - 0.5) * fov_mm:.3g}" for value in values]

            return tick_strings

        self.canvas.getAxis("bottom").tickStrings = physical_ticks(0)
        self.canvas.getAxis("left").tickStrings = physical_ticks(1)
        for spin in self.fov_spins[:2]:
            spin.valueChanged.connect(self._update_canvas_axes)
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_panel)

        property_panel = QWidget()
        property_layout = QVBoxLayout(property_panel)
        form = QFormLayout()
        self.shape_name = QLineEdit()
        self.shape_name.editingFinished.connect(self._properties_changed)
        form.addRow("Shape name", self.shape_name)
        self.kind_label = QLabel()
        form.addRow("Kind", self.kind_label)
        self.x_center = self._position_percent_spin(50.0)
        self.y_center = self._position_percent_spin(50.0)
        self.z_center = self._position_percent_spin(50.0)
        self.x_size = self._percent_spin(50.0)
        self.y_size = self._percent_spin(50.0)
        self.z_size = self._percent_spin(50.0)
        self.t1_ms = self._number_spin(0.1, 500000.0, 1000.0, " ms")
        self.initial_mz = self._number_spin(0.0, 1e9, 1.0, "")
        self.b0_ppm = self._number_spin(-1000.0, 1000.0, 0.0, " ppm")
        self.t1_ms.setToolTip(
            "Fallback T1 for peaks whose metabolite-specific T1 cell is empty."
        )
        self.initial_mz.setToolTip(
            "Common initial scale for all hyperpolarized pools in this shape. "
            "In the dynamic model this is excess magnetization above thermal "
            "equilibrium and relaxes toward zero."
        )
        for widget in (
            self.x_center,
            self.y_center,
            self.z_center,
            self.x_size,
            self.y_size,
            self.z_size,
            self.t1_ms,
            self.initial_mz,
            self.b0_ppm,
        ):
            widget.valueChanged.connect(self._properties_changed)
        form.addRow("X centre", self.x_center)
        form.addRow("Y centre", self.y_center)
        form.addRow("Z centre", self.z_center)
        form.addRow("X size", self.x_size)
        form.addRow("Y size", self.y_size)
        form.addRow("Z thickness", self.z_size)
        form.addRow("Default T1", self.t1_ms)
        form.addRow("Initial HP Mz scale", self.initial_mz)
        form.addRow("B0 inhomogeneity", self.b0_ppm)
        self.xy_info = QLabel()
        form.addRow("XY geometry", self.xy_info)
        property_layout.addLayout(form)

        peak_explanation = QLabel(
            "Dynamic phantoms: initial HP pool Mz = Initial HP Mz scale × Initial "
            "pool weight. A weight of 0 keeps the pool defined, so conversion can "
            "populate it. Leave T1 empty to use Default T1."
        )
        peak_explanation.setWordWrap(True)
        property_layout.addWidget(peak_explanation)
        self.peak_table = QTableWidget(0, 5)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Initial pool weight (0=empty)",
                "Peak position (ppm)",
                "T1 (ms; blank=default)",
                "T2* (ms)",
            ]
        )
        peak_header = self.peak_table.horizontalHeader()
        peak_header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            peak_header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.peak_table.cellChanged.connect(self._peaks_changed)
        property_layout.addWidget(self.peak_table)
        peak_row = QHBoxLayout()
        add_peak = QPushButton("Add peak")
        add_peak.clicked.connect(self._add_peak)
        remove_peak = QPushButton("Remove peak")
        remove_peak.clicked.connect(self._remove_peak)
        peak_row.addWidget(add_peak)
        peak_row.addWidget(remove_peak)
        property_layout.addLayout(peak_row)
        splitter.addWidget(property_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([210, 450, 590])
        self.tabs.addTab(draw_page, "Draw and edit")

        kinetics_page = QWidget()
        kinetics_page_layout = QHBoxLayout(kinetics_page)
        kinetics_splitter = QSplitter(Qt.Horizontal)
        kinetics_controls = QWidget()
        kinetics_layout = QVBoxLayout(kinetics_controls)
        kinetics_form = QFormLayout()
        self.dynamic_enabled = QCheckBox("Enable pyruvate → lactate conversion")
        kinetics_form.addRow(self.dynamic_enabled)
        self.pyruvate_peak_name = QLineEdit("Pyruvate")
        self.lactate_peak_name = QLineEdit("Lactate")
        self.default_kpl = self._number_spin(0.0, 10.0, 0.0, " s⁻¹")
        self.conversion_start_s = self._number_spin(-10000.0, 10000.0, 0.0, " s")
        self.kinetics_time_offset_s = self._number_spin(-10000.0, 10000.0, 0.0, " s")
        self.default_kpl.setToolTip(
            "kPL used everywhere unless an optional spatial region overrides it. "
            "Zero means no conversion without a positive regional override."
        )
        self.conversion_start_s.setToolTip(
            "kPL is zero before this time on the shared kinetics timeline and active "
            "afterwards."
        )
        self.kinetics_time_offset_s.setToolTip(
            "Selects which time on the shared inflow/conversion timeline coincides "
            "with sequence t=0. For example, +5 s starts the sequence 5 s into both "
            "curves; -5 s starts it 5 s before them."
        )
        self.default_kpl.valueChanged.connect(self._update_kinetics_preview)
        self.conversion_start_s.valueChanged.connect(self._update_kinetics_preview)
        self.kinetics_time_offset_s.valueChanged.connect(self._update_kinetics_preview)
        self.pyruvate_peak_name.textChanged.connect(self._update_kinetics_preview)
        self.lactate_peak_name.textChanged.connect(self._update_kinetics_preview)
        kinetics_form.addRow("Pyruvate peak name", self.pyruvate_peak_name)
        kinetics_form.addRow("Lactate peak name", self.lactate_peak_name)
        kinetics_form.addRow("Default kPL", self.default_kpl)
        kinetics_form.addRow(
            "Conversion starts at (kinetics time)", self.conversion_start_s
        )
        kinetics_form.addRow(
            "Kinetics time at sequence t=0", self.kinetics_time_offset_s
        )
        self.inflow_enabled = QCheckBox(
            "Enable tabulated pyruvate inflow into pyruvate-shape regions"
        )
        self.inflow_enabled.toggled.connect(self._update_kinetics_preview)
        kinetics_form.addRow(self.inflow_enabled)
        self.dynamic_b0_enabled = QCheckBox("Enable uniform time-dependent B0 offset")
        kinetics_form.addRow(self.dynamic_b0_enabled)
        kinetics_layout.addLayout(kinetics_form)
        background_kpl_help = QLabel(
            "Default kPL is used everywhere in the phantom. Optional spatial kPL "
            "regions below override it only inside their geometry. 0 s⁻¹ means no "
            "pyruvate → lactate conversion unless a region supplies a positive kPL."
        )
        background_kpl_help.setWordWrap(True)
        kinetics_layout.addWidget(background_kpl_help)

        inflow_help = QLabel(
            "Pyruvate inflow points define a longitudinal source added to Pz "
            "inside pyruvate shapes. Values are linearly interpolated and zero "
            "outside the listed kinetics-time interval. The global kinetics-time "
            "setting shifts inflow and conversion together relative to sequence "
            "time zero. Any part before sequence t=0 forms a free kinetic pre-roll "
            "that sets the initial Pz/Lz distribution."
        )
        inflow_help.setWordWrap(True)
        kinetics_layout.addWidget(inflow_help)
        self.inflow_curve_table = QTableWidget(0, 2)
        self.inflow_curve_table.setHorizontalHeaderLabels(
            ["Kinetics time (s)", "Source (relative Mz/s)"]
        )
        self.inflow_curve_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.inflow_curve_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.inflow_curve_table.setMaximumHeight(150)
        self.inflow_curve_table.cellChanged.connect(self._inflow_curve_table_changed)
        kinetics_layout.addWidget(self.inflow_curve_table)
        inflow_buttons = QHBoxLayout()
        add_inflow_point = QPushButton("Add inflow point")
        add_inflow_point.clicked.connect(
            lambda: self._add_curve_point(self.inflow_curve_table)
        )
        remove_inflow_point = QPushButton("Remove inflow point")
        remove_inflow_point.clicked.connect(
            lambda: self._remove_curve_point(self.inflow_curve_table)
        )
        inflow_buttons.addWidget(add_inflow_point)
        inflow_buttons.addWidget(remove_inflow_point)
        inflow_buttons.addStretch()
        kinetics_layout.addLayout(inflow_buttons)

        dynamic_b0_help = QLabel(
            "Dynamic B0 curve: time and additional object frequency in Hz. "
            "Linear interpolation; endpoint values are held outside the table."
        )
        dynamic_b0_help.setWordWrap(True)
        kinetics_layout.addWidget(dynamic_b0_help)
        self.dynamic_b0_curve_table = QTableWidget(0, 2)
        self.dynamic_b0_curve_table.setHorizontalHeaderLabels(
            ["Time (s)", "Offset (Hz)"]
        )
        self.dynamic_b0_curve_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.dynamic_b0_curve_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.dynamic_b0_curve_table.setMaximumHeight(130)
        kinetics_layout.addWidget(self.dynamic_b0_curve_table)
        b0_curve_buttons = QHBoxLayout()
        add_b0_point = QPushButton("Add B0 point")
        add_b0_point.clicked.connect(
            lambda: self._add_curve_point(self.dynamic_b0_curve_table)
        )
        remove_b0_point = QPushButton("Remove B0 point")
        remove_b0_point.clicked.connect(
            lambda: self._remove_curve_point(self.dynamic_b0_curve_table)
        )
        b0_curve_buttons.addWidget(add_b0_point)
        b0_curve_buttons.addWidget(remove_b0_point)
        b0_curve_buttons.addStretch()
        kinetics_layout.addLayout(b0_curve_buttons)

        kinetic_regions_help = QLabel(
            "Optional spatial kPL regions: each row assigns a different kPL to "
            "an ellipsoid or box. If regions overlap, the last row wins. Center "
            "and size use percent of the phantom FOV."
        )
        kinetic_regions_help.setWordWrap(True)
        kinetics_layout.addWidget(kinetic_regions_help)
        self.kinetic_table = QTableWidget(0, 9)
        self.kinetic_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Cx %", "Cy %", "Cz %", "Sx %", "Sy %", "Sz %", "kPL s⁻¹"]
        )
        self.kinetic_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.kinetic_table.cellChanged.connect(self._kinetic_table_changed)
        self.kinetic_table.currentCellChanged.connect(
            self._kinetic_region_selected_for_preview
        )
        kinetics_layout.addWidget(self.kinetic_table)
        kinetic_buttons = QHBoxLayout()
        add_kinetic_ellipse = QPushButton("Add kPL ellipsoid region")
        add_kinetic_ellipse.clicked.connect(
            lambda: self._add_kinetic_region("ellipsoid")
        )
        add_kinetic_box = QPushButton("Add kPL box region")
        add_kinetic_box.clicked.connect(lambda: self._add_kinetic_region("box"))
        remove_kinetic = QPushButton("Remove selected kPL region")
        remove_kinetic.clicked.connect(self._remove_kinetic_region)
        kinetic_buttons.addWidget(add_kinetic_ellipse)
        kinetic_buttons.addWidget(add_kinetic_box)
        kinetic_buttons.addWidget(remove_kinetic)
        kinetic_buttons.addStretch()
        kinetics_layout.addLayout(kinetic_buttons)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_title = QLabel(
            "Representative-voxel preview for one selected shape/object. This is "
            "not a spatial average; RF pulses and gradients are not included."
        )
        preview_title.setWordWrap(True)
        preview_layout.addWidget(preview_title)
        hp_mz_help = QLabel(
            "HP Mz is normalized hyperpolarized excess magnetization: HP Mz=1 is "
            "the initial hyperpolarized level, not the thermal equilibrium value. "
            "T1 relaxation therefore drives it toward approximately 0."
        )
        hp_mz_help.setWordWrap(True)
        preview_layout.addWidget(hp_mz_help)
        preview_form = QFormLayout()
        self.kinetics_preview_shape = QComboBox()
        self.kinetics_preview_shape.currentIndexChanged.connect(
            self._kinetics_preview_shape_changed
        )
        preview_form.addRow("Shape / object to preview", self.kinetics_preview_shape)
        self.kinetics_preview_region = QComboBox()
        self.kinetics_preview_region.currentIndexChanged.connect(
            self._update_kinetics_preview
        )
        preview_form.addRow("kPL source for this voxel", self.kinetics_preview_region)
        self.zero_lactate_button = QPushButton("Set selected shape to initial Lz = 0")
        self.zero_lactate_button.setToolTip(
            "Sets the Lactate initial pool weight to zero without removing the "
            "Lactate pool; positive kPL can then create Lactate from Pyruvate."
        )
        self.zero_lactate_button.clicked.connect(self._set_selected_shape_lactate_zero)
        preview_form.addRow("Pyruvate-only start", self.zero_lactate_button)
        self.kinetics_preview_duration = self._number_spin(0.1, 10000.0, 30.0, " s")
        self.kinetics_preview_duration.valueChanged.connect(
            self._update_kinetics_preview
        )
        preview_form.addRow("Duration", self.kinetics_preview_duration)
        preview_layout.addLayout(preview_form)

        self.kinetics_preview_graphics = pg.GraphicsLayoutWidget()
        self.kinetics_preview_graphics.setMinimumWidth(420)
        self.inflow_preview_plot = self.kinetics_preview_graphics.addPlot(row=0, col=0)
        self.inflow_preview_plot.setLabel("left", "Inflow", units="rel. Mz/s")
        self.inflow_preview_plot.showGrid(x=True, y=True, alpha=0.25)
        self.inflow_preview_curve = self.inflow_preview_plot.plot(
            pen=pg.mkPen("y", width=2),
            fillLevel=0.0,
            brush=pg.mkBrush(255, 255, 0, 45),
        )
        self.inflow_sequence_start_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen((180, 180, 180), width=1, style=Qt.DashLine),
        )
        self.inflow_preview_plot.addItem(self.inflow_sequence_start_line)
        self.pool_preview_plot = self.kinetics_preview_graphics.addPlot(row=1, col=0)
        self.pool_preview_plot.setLabel("left", "Hyperpolarized Mz")
        self.pool_preview_plot.setLabel("bottom", "Time", units="s")
        self.pool_preview_plot.showGrid(x=True, y=True, alpha=0.25)
        self.pool_preview_plot.addLegend(offset=(8, 8))
        self.pyruvate_preview_curve = self.pool_preview_plot.plot(
            pen=pg.mkPen("c", width=2), name="Pyruvate Pz"
        )
        self.lactate_preview_curve = self.pool_preview_plot.plot(
            pen=pg.mkPen("m", width=2, style=Qt.DashLine), name="Lactate Lz"
        )
        self.pool_sequence_start_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen((180, 180, 180), width=1, style=Qt.DashLine),
        )
        self.conversion_start_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen((255, 150, 60), width=1, style=Qt.DotLine),
        )
        self.pool_preview_plot.addItem(self.pool_sequence_start_line)
        self.pool_preview_plot.addItem(self.conversion_start_line)
        self.pool_preview_plot.setXLink(self.inflow_preview_plot)
        preview_layout.addWidget(self.kinetics_preview_graphics)
        self.kinetics_preview_info = QLabel()
        self.kinetics_preview_info.setWordWrap(True)
        preview_layout.addWidget(self.kinetics_preview_info)

        kinetics_splitter.addWidget(kinetics_controls)
        kinetics_splitter.addWidget(preview_panel)
        kinetics_splitter.setStretchFactor(0, 1)
        kinetics_splitter.setStretchFactor(1, 1)
        kinetics_page_layout.addWidget(kinetics_splitter)
        self.tabs.addTab(kinetics_page, "Kinetics / kPL")

        self.inspector = PhantomInspectorWidget()
        self.tabs.addTab(self.inspector, "3D / frequency preview")

        action_row = QHBoxLayout()
        preview = QPushButton("Update preview")
        preview.clicked.connect(self._preview)
        save = QPushButton("Save design…")
        save.clicked.connect(self._save)
        load = QPushButton("Load design…")
        load.clicked.connect(self._load)
        action_row.addWidget(preview)
        action_row.addWidget(save)
        action_row.addWidget(load)
        action_row.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        action_row.addWidget(buttons)
        root.addLayout(action_row)

    @staticmethod
    def _percent_spin(value):
        widget = QDoubleSpinBox()
        widget.setRange(0.1, 100.0)
        widget.setDecimals(2)
        widget.setValue(value)
        widget.setSuffix(" %")
        return widget

    @staticmethod
    def _position_percent_spin(value):
        widget = SpectralPhantomDesignerDialog._percent_spin(value)
        widget.setRange(0.0, 100.0)
        return widget

    @staticmethod
    def _number_spin(minimum, maximum, value, suffix):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(4)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _load_design_into_ui(self):
        self._updating = True
        self.name_edit.setText(self.design.name)
        for widget, value in zip(self.matrix_spins, self.design.shape):
            widget.setValue(int(value))
        for widget, value in zip(self.fov_spins, self.design.fov_m):
            widget.setValue(float(value) * 1000.0)
        self.field_strength_t.setValue(float(self.design.field_strength_t))
        nucleus_index = self.nucleus.findData(self.design.nucleus)
        self.nucleus.setCurrentIndex(max(0, nucleus_index))
        self.spectral_reference_ppm.setValue(self.design.spectral_reference_ppm)
        self.spectral_bandwidth_ppm.setValue(self.design.spectral_bandwidth_ppm)
        self.spectral_points.setValue(int(self.design.spectral_points))
        self._update_spectral_resolution_info()
        mode_index = self.b0_mode_combo.findData(self.design.b0_inhomogeneity_mode)
        self.b0_mode_combo.setCurrentIndex(max(0, mode_index))
        self.b0_inhomogeneity_ppm.setValue(self.design.b0_inhomogeneity_ppm)
        self.dynamic_enabled.setChecked(self.design.dynamic_enabled)
        self.pyruvate_peak_name.setText(self.design.pyruvate_peak_name)
        self.lactate_peak_name.setText(self.design.lactate_peak_name)
        self.default_kpl.setValue(self.design.default_kpl_s_inv)
        self.conversion_start_s.setValue(self.design.conversion_start_s)
        self.kinetics_time_offset_s.setValue(self.design.kinetics_time_offset_s)
        self.inflow_enabled.setChecked(self.design.pyruvate_inflow_curve is not None)
        self.dynamic_b0_enabled.setChecked(self.design.dynamic_b0_curve is not None)
        self._populate_time_curve(
            self.inflow_curve_table,
            self.design.pyruvate_inflow_curve,
            default=((0.0, 0.0), (5.0, 0.1), (15.0, 0.0)),
        )
        self._populate_time_curve(
            self.dynamic_b0_curve_table,
            self.design.dynamic_b0_curve,
            default=((0.0, 0.0), (10.0, 0.0)),
        )
        self._populate_kinetic_regions()
        self._refresh_kinetics_preview_regions()
        self.shape_list.clear()
        for roi in self._rois:
            self.canvas.removeItem(roi)
        self._rois.clear()
        for index, item in enumerate(self.design.shapes):
            self.shape_list.addItem(item.name)
            self._create_roi(item, index)
        self._refresh_kinetics_preview_shapes()
        self._updating = False
        if self.design.shapes:
            self.shape_list.setCurrentRow(0)
        else:
            self._update_kinetics_preview()

    def _create_roi(self, item, index):
        position = (
            item.center[0] - item.size[0] / 2,
            item.center[1] - item.size[1] / 2,
        )
        size = item.size[:2]
        roi_class = pg.EllipseROI if item.kind == "ellipsoid" else pg.RectROI
        roi = roi_class(position, size, pen=self._roi_pen(index, False))
        roi.sigRegionChangeFinished.connect(self._roi_changed)
        original_mouse_click = roi.mouseClickEvent

        def select_on_click(event, roi=roi, original_mouse_click=original_mouse_click):
            self._select_roi(roi)
            original_mouse_click(event)

        roi.mouseClickEvent = select_on_click
        self.canvas.addItem(roi)
        self._rois.append(roi)
        self._update_roi_highlights()

    def _roi_pen(self, index, selected):
        color = pg.intColor(index, hues=max(1, len(self.design.shapes)), alpha=245)
        return pg.mkPen(color, width=5 if selected else 2)

    def _select_roi(self, roi):
        try:
            row = self._rois.index(roi)
        except ValueError:
            return
        self.shape_list.setCurrentRow(row)

    def _update_roi_highlights(self, selected_row=None):
        if selected_row is None:
            selected_row = self._current_row()
        for index, roi in enumerate(self._rois):
            roi.setPen(self._roi_pen(index, index == selected_row))

    def _current_row(self):
        row = self.shape_list.currentRow()
        return row if 0 <= row < len(self.design.shapes) else None

    def _shape_selected(self, row):
        if self._updating or not (0 <= row < len(self.design.shapes)):
            return
        self._updating = True
        item = self.design.shapes[row]
        self.shape_name.setText(item.name)
        self.kind_label.setText(item.kind)
        self.x_center.setValue(item.center[0] * 100.0)
        self.y_center.setValue(item.center[1] * 100.0)
        self.z_center.setValue(item.center[2] * 100.0)
        self.x_size.setValue(item.size[0] * 100.0)
        self.y_size.setValue(item.size[1] * 100.0)
        self.z_size.setValue(item.size[2] * 100.0)
        self.t1_ms.setValue(item.t1_s * 1000.0)
        self.initial_mz.setValue(item.initial_mz)
        self.b0_ppm.setValue(item.b0_ppm)
        self._populate_peaks(item)
        self._update_xy_info(row)
        self._updating = False
        self._set_kinetics_preview_shape(row)
        self._update_roi_highlights(row)
        self._update_kinetics_preview()

    def _roi_changed(self):
        if self._updating:
            return
        for row, roi in enumerate(self._rois):
            if roi is self.sender():
                position = roi.pos()
                size = roi.size()
                item = self.design.shapes[row]
                item.center = (
                    float(position.x() + size.x() / 2),
                    float(position.y() + size.y() / 2),
                    item.center[2],
                )
                item.size = (float(size.x()), float(size.y()), item.size[2])
                self.shape_list.setCurrentRow(row)
                self._updating = True
                self.x_center.setValue(item.center[0] * 100.0)
                self.y_center.setValue(item.center[1] * 100.0)
                self.x_size.setValue(item.size[0] * 100.0)
                self.y_size.setValue(item.size[1] * 100.0)
                self._updating = False
                self._update_xy_info(row)
                break

    def _update_xy_info(self, row):
        item = self.design.shapes[row]
        center_mm = tuple(
            (item.center[index] - 0.5) * self.fov_spins[index].value()
            for index in range(2)
        )
        size_mm = tuple(
            item.size[index] * self.fov_spins[index].value() for index in range(2)
        )
        self.xy_info.setText(
            f"centre=({center_mm[0]:.3g}, {center_mm[1]:.3g}) mm; "
            f"size=({size_mm[0]:.3g}, {size_mm[1]:.3g}) mm"
        )

    def _update_canvas_axes(self, *_):
        for name in ("bottom", "left"):
            axis = self.canvas.getAxis(name)
            axis.picture = None
            axis.update()
        row = self._current_row()
        if row is not None:
            self._update_xy_info(row)

    def _spectral_settings_changed(self, *_):
        if self._updating:
            return
        self._update_spectral_resolution_info()

    def _update_spectral_resolution_info(self):
        points = max(2, int(self.spectral_points.value()))
        resolution = self.spectral_bandwidth_ppm.value() / (points - 1)
        self.spectral_resolution_info.setText(f"Resolution {resolution:.5g} ppm/pt")

    def _properties_changed(self):
        row = self._current_row()
        if self._updating or row is None:
            return
        item = self.design.shapes[row]
        previous_name = item.name
        item.name = self.shape_name.text().strip() or item.name
        size = (
            self.x_size.value() / 100.0,
            self.y_size.value() / 100.0,
            self.z_size.value() / 100.0,
        )
        center = (
            self.x_center.value() / 100.0,
            self.y_center.value() / 100.0,
            self.z_center.value() / 100.0,
        )
        item.center = center
        item.size = size
        item.t1_s = self.t1_ms.value() / 1000.0
        item.initial_mz = self.initial_mz.value()
        item.b0_ppm = self.b0_ppm.value()
        item.b0_hz = None
        self.shape_list.item(row).setText(item.name)
        roi = self._rois[row]
        previous = roi.blockSignals(True)
        roi.setPos(
            item.center[0] - item.size[0] / 2,
            item.center[1] - item.size[1] / 2,
        )
        roi.setSize(item.size[:2])
        roi.blockSignals(previous)
        self._update_xy_info(row)
        if item.name != previous_name:
            self._refresh_kinetics_preview_shapes()
        self._update_kinetics_preview()

    def _populate_peaks(self, item):
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(len(item.peaks))
        for row, peak in enumerate(item.peaks):
            for column, value in enumerate(
                (
                    peak.name,
                    peak.amplitude,
                    peak.frequency_ppm + self.spectral_reference_ppm.value(),
                    "" if peak.t1_s is None else peak.t1_s * 1000,
                    peak.t2_star_s * 1000,
                )
            ):
                self.peak_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.peak_table.blockSignals(False)

    def _peaks_changed(self):
        row = self._current_row()
        if self._updating or row is None:
            return
        try:
            peaks = self._read_peak_table()
        except (AttributeError, ValueError):
            return
        self.design.shapes[row].peaks = peaks
        self._update_kinetics_preview()

    def _read_peak_table(self):
        peaks = []
        reference_ppm = self.spectral_reference_ppm.value()
        for peak_row in range(self.peak_table.rowCount()):
            t1_text = self.peak_table.item(peak_row, 3).text().strip()
            peak = SpectralPeakDefinition(
                name=self.peak_table.item(peak_row, 0).text(),
                amplitude=float(self.peak_table.item(peak_row, 1).text()),
                frequency_ppm=(
                    float(self.peak_table.item(peak_row, 2).text()) - reference_ppm
                ),
                t2_star_s=float(self.peak_table.item(peak_row, 4).text()) / 1000.0,
                t1_s=None if not t1_text else float(t1_text) / 1000.0,
            )
            peak.validate()
            peaks.append(peak)
        return peaks

    def _populate_kinetic_regions(self):
        previous = self.kinetic_table.blockSignals(True)
        self.kinetic_table.setRowCount(len(self.design.kinetic_regions))
        for row, region in enumerate(self.design.kinetic_regions):
            values = (
                region.name,
                region.kind,
                *(value * 100.0 for value in region.center),
                *(value * 100.0 for value in region.size),
                region.kpl_s_inv,
            )
            for column, value in enumerate(values):
                self.kinetic_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.kinetic_table.blockSignals(previous)

    @staticmethod
    def _populate_time_curve(table, curve, default):
        samples = default if curve is None else zip(curve.times_s, curve.values)
        samples = tuple(samples)
        table.setRowCount(len(samples))
        for row, (time_s, value) in enumerate(samples):
            table.setItem(row, 0, QTableWidgetItem(str(time_s)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))

    @staticmethod
    def _read_time_curve(table, *, outside):
        samples = []
        for row in range(table.rowCount()):
            time_item = table.item(row, 0)
            value_item = table.item(row, 1)
            if time_item is None or value_item is None:
                raise ValueError("time-curve rows require both time and value")
            samples.append((float(time_item.text()), float(value_item.text())))
        if not samples:
            raise ValueError("enabled time curve requires at least one sample")
        return TimeCurve(
            times_s=tuple(item[0] for item in samples),
            values=tuple(item[1] for item in samples),
            interpolation="linear",
            outside=outside,
        )

    def _add_curve_point(self, table):
        row = table.rowCount()
        if row:
            previous_time = float(table.item(row - 1, 0).text())
            previous_value = float(table.item(row - 1, 1).text())
            values = (previous_time + 1.0, previous_value)
        else:
            values = (0.0, 0.0)
        table.insertRow(row)
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(str(value)))
        table.setCurrentCell(row, 0)
        if table is self.inflow_curve_table:
            self._update_kinetics_preview()

    def _inflow_curve_table_changed(self, row, column):
        if column == 0:
            self._sort_time_curve_table(self.inflow_curve_table, row, column)
        self._update_kinetics_preview()

    @staticmethod
    def _sort_time_curve_table(table, edited_row, edited_column):
        rows = []
        for row in range(table.rowCount()):
            time_item = table.item(row, 0)
            value_item = table.item(row, 1)
            if time_item is None or value_item is None:
                return
            try:
                time_s = float(time_item.text())
            except ValueError:
                return
            rows.append((time_s, row, time_item, value_item))

        sorted_rows = sorted(rows, key=lambda item: item[0])
        if all(
            original_row == new_row
            for new_row, (_, original_row, *_) in enumerate(sorted_rows)
        ):
            return

        previous = table.blockSignals(True)
        for row in range(table.rowCount()):
            table.takeItem(row, 0)
            table.takeItem(row, 1)
        new_edited_row = edited_row
        for new_row, (_, original_row, time_item, value_item) in enumerate(sorted_rows):
            table.setItem(new_row, 0, time_item)
            table.setItem(new_row, 1, value_item)
            if original_row == edited_row:
                new_edited_row = new_row
        table.setCurrentCell(new_edited_row, edited_column)
        table.blockSignals(previous)

    def _remove_curve_point(self, table):
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)
        if table is self.inflow_curve_table:
            self._update_kinetics_preview()

    def _refresh_kinetics_preview_regions(self):
        if not hasattr(self, "kinetics_preview_region"):
            return
        selected_row = self.kinetics_preview_region.currentData()
        previous = self.kinetics_preview_region.blockSignals(True)
        self.kinetics_preview_region.clear()
        self.kinetics_preview_region.addItem("Default kPL", -1)
        for row in range(self.kinetic_table.rowCount()):
            name_item = self.kinetic_table.item(row, 0)
            name = name_item.text().strip() if name_item is not None else ""
            self.kinetics_preview_region.addItem(f"Region: {name or row + 1}", row)
        selected_index = self.kinetics_preview_region.findData(selected_row)
        self.kinetics_preview_region.setCurrentIndex(max(0, selected_index))
        self.kinetics_preview_region.blockSignals(previous)

    def _refresh_kinetics_preview_shapes(self):
        if not hasattr(self, "kinetics_preview_shape"):
            return
        selected_row = self.kinetics_preview_shape.currentData()
        if selected_row is None:
            selected_row = self._current_row()
        previous = self.kinetics_preview_shape.blockSignals(True)
        self.kinetics_preview_shape.clear()
        for row, shape in enumerate(self.design.shapes):
            self.kinetics_preview_shape.addItem(shape.name, row)
        selected_index = self.kinetics_preview_shape.findData(selected_row)
        self.kinetics_preview_shape.setCurrentIndex(max(0, selected_index))
        self.kinetics_preview_shape.blockSignals(previous)

    def _set_kinetics_preview_shape(self, row):
        if not hasattr(self, "kinetics_preview_shape"):
            return
        index = self.kinetics_preview_shape.findData(row)
        if index < 0 or index == self.kinetics_preview_shape.currentIndex():
            return
        previous = self.kinetics_preview_shape.blockSignals(True)
        self.kinetics_preview_shape.setCurrentIndex(index)
        self.kinetics_preview_shape.blockSignals(previous)

    def _kinetics_preview_shape_changed(self, _index):
        if self._updating:
            return
        shape_row = self.kinetics_preview_shape.currentData()
        if shape_row is None:
            self._update_kinetics_preview()
            return
        shape_row = int(shape_row)
        if self.shape_list.currentRow() != shape_row:
            self.shape_list.setCurrentRow(shape_row)
        else:
            self._update_kinetics_preview()

    def _kinetic_table_changed(self, _row, column):
        if self._updating:
            return
        if column == 0:
            self._refresh_kinetics_preview_regions()
        self._update_kinetics_preview()

    def _kinetic_region_selected_for_preview(
        self, current_row, _current_column, _previous_row, _previous_column
    ):
        if self._updating or current_row < 0:
            return
        preview_index = self.kinetics_preview_region.findData(current_row)
        if preview_index >= 0:
            self.kinetics_preview_region.setCurrentIndex(preview_index)

    def _preview_kpl(self):
        region_row = self.kinetics_preview_region.currentData()
        if region_row is None or int(region_row) < 0:
            return self.default_kpl.value(), "default kPL"
        region_row = int(region_row)
        kpl_item = self.kinetic_table.item(region_row, 8)
        name_item = self.kinetic_table.item(region_row, 0)
        if kpl_item is None:
            raise ValueError("selected kinetic region has no kPL value")
        name = "" if name_item is None else name_item.text().strip()
        return float(kpl_item.text()), f"region {name or region_row + 1}"

    def _set_selected_shape_lactate_zero(self):
        shape_row = self.kinetics_preview_shape.currentData()
        if shape_row is None:
            self.kinetics_preview_info.setText(
                "Preview unavailable: select a shape containing a Lactate pool"
            )
            return
        shape_row = int(shape_row)
        shape = self.design.shapes[shape_row]
        lactate_name = self.lactate_peak_name.text().strip()
        lactate_peak = next(
            (peak for peak in shape.peaks if peak.name == lactate_name), None
        )
        if lactate_peak is None:
            self.kinetics_preview_info.setText(
                f"Preview unavailable: {shape.name} has no {lactate_name!r} pool"
            )
            return
        lactate_peak.amplitude = 0.0
        if self._current_row() == shape_row:
            self._populate_peaks(shape)
        self._update_kinetics_preview()

    def _update_kinetics_preview(self, *_):
        if self._updating or not hasattr(self, "pyruvate_preview_curve"):
            return
        try:
            shape_row = self.kinetics_preview_shape.currentData()
            if shape_row is None:
                shape_row = self._current_row()
            if shape_row is None:
                raise ValueError("select a shape containing both dynamic pools")
            shape_row = int(shape_row)
            shape = self.design.shapes[shape_row]
            peaks = {peak.name: peak for peak in shape.peaks}
            pyruvate_name = self.pyruvate_peak_name.text().strip()
            lactate_name = self.lactate_peak_name.text().strip()
            missing = [
                name for name in (pyruvate_name, lactate_name) if name not in peaks
            ]
            if missing:
                raise ValueError(
                    f"current shape is missing peak(s): {', '.join(missing)}"
                )
            pyruvate_peak = peaks[pyruvate_name]
            lactate_peak = peaks[lactate_name]
            initial_mz = (
                shape.initial_mz * pyruvate_peak.amplitude,
                shape.initial_mz * lactate_peak.amplitude,
            )
            t1_s = (
                pyruvate_peak.effective_t1_s(shape.t1_s),
                lactate_peak.effective_t1_s(shape.t1_s),
            )
            kpl_s_inv, kpl_label = self._preview_kpl()
            inflow_curve = (
                self._read_time_curve(self.inflow_curve_table, outside="zero")
                if self.inflow_enabled.isChecked()
                else None
            )
            conversion_start_s = self.conversion_start_s.value()
            kinetics_time_offset_s = self.kinetics_time_offset_s.value()
            sequence_inflow_curve = (
                None
                if inflow_curve is None
                else inflow_curve.shifted(-kinetics_time_offset_s)
            )
            sequence_conversion_start_s = conversion_start_s - kinetics_time_offset_s
            preview_start_s = kinetic_preroll_start_s(
                inflow_curve,
                conversion_start_s,
                kinetics_time_offset_s,
            )
            duration_s = self.kinetics_preview_duration.value()
            times_s = np.linspace(preview_start_s, duration_s, 601)
            visible_knots = [0.0]
            if sequence_inflow_curve is not None:
                visible_knots.extend(
                    value
                    for value in sequence_inflow_curve.times_s
                    if preview_start_s <= value <= duration_s
                )
            if preview_start_s <= sequence_conversion_start_s <= duration_s:
                visible_knots.append(sequence_conversion_start_s)
            times_s = np.unique(np.concatenate((times_s, visible_knots)))
            pools = simulate_two_pool_kinetics(
                times_s,
                initial_mz,
                t1_s,
                kpl_s_inv,
                inflow_curve=inflow_curve,
                conversion_start_s=conversion_start_s,
                initial_time_s=preview_start_s,
                kinetics_time_offset_s=kinetics_time_offset_s,
            )
            inflow = (
                np.zeros_like(times_s)
                if sequence_inflow_curve is None
                else np.asarray(
                    [sequence_inflow_curve.value_at(value) for value in times_s]
                )
            )
            self.inflow_preview_curve.setData(times_s, inflow)
            self.pyruvate_preview_curve.setData(times_s, pools[0])
            self.lactate_preview_curve.setData(times_s, pools[1])
            self.inflow_preview_plot.setXRange(preview_start_s, duration_s, padding=0.0)
            self.pool_preview_plot.setXRange(preview_start_s, duration_s, padding=0.0)
            has_preroll = preview_start_s < 0.0
            self.inflow_sequence_start_line.setVisible(has_preroll)
            self.pool_sequence_start_line.setVisible(has_preroll)
            self.conversion_start_line.setPos(sequence_conversion_start_s)
            self.conversion_start_line.setVisible(
                not np.isclose(sequence_conversion_start_s, 0.0)
            )
            if np.any(np.abs(inflow) > 1e-15):
                self.inflow_preview_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            else:
                self.inflow_preview_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
                self.inflow_preview_plot.setYRange(0.0, 1.0, padding=0.05)
            self.pool_preview_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            zero_index = int(np.searchsorted(times_s, 0.0))
            details = (
                f"Representative voxel in {shape.name} · kPL source: {kpl_label}. "
                f"kinetics t at sequence t=0: {kinetics_time_offset_s:.4g} s; "
                f"kPL={kpl_s_inv:.4g} s⁻¹ from sequence "
                f"t={sequence_conversion_start_s:.4g} s "
                f"(kinetics t={conversion_start_s:.4g} s), "
                f"T1(P/L)=({t1_s[0]:.4g}/{t1_s[1]:.4g}) s, "
                f"initial HP Mz at sequence t={preview_start_s:.4g} s="
                f"({initial_mz[0]:.4g}/{initial_mz[1]:.4g}), "
                f"sequence-start HP Mz at t=0="
                f"({pools[0, zero_index]:.4g}/{pools[1, zero_index]:.4g}), "
                f"HP Mz({duration_s:.4g} s)=({pools[0, -1]:.4g}/"
                f"{pools[1, -1]:.4g})."
            )
            explanations = []
            if kpl_s_inv == 0.0:
                explanations.append(
                    "kPL=0: no P→L conversion; each existing pool only follows its "
                    "own T1 decay toward zero, while enabled inflow is added only "
                    "to Pz. HP Mz=1 is an initial normalization, not an equilibrium "
                    "target."
                )
                if initial_mz[1] > 0:
                    explanations.append(
                        "Lz starts above zero because the Lactate initial pool "
                        "weight is non-zero; this lactate was initialized, not "
                        "created by conversion."
                    )
                else:
                    explanations.append(
                        "Lz starts at zero and remains zero until kPL is set above "
                        "zero for this voxel."
                    )
            elif initial_mz[1] == 0.0:
                explanations.append(
                    "Lz starts at zero; all subsequent Lactate is created from "
                    "Pyruvate by kPL conversion."
                )
            if np.allclose(pools[0], pools[1], rtol=1e-7, atol=1e-10):
                explanations.append(
                    "Pz and Lz are identical and overlap: cyan is solid, magenta is "
                    "dashed so both curves remain recognizable."
                )
            self.kinetics_preview_info.setText("\n".join((details, *explanations)))
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            self.inflow_preview_curve.setData([], [])
            self.pyruvate_preview_curve.setData([], [])
            self.lactate_preview_curve.setData([], [])
            self.kinetics_preview_info.setText(f"Preview unavailable: {exc}")

    def _read_kinetic_regions(self):
        regions = []
        for row in range(self.kinetic_table.rowCount()):

            def text(column):
                return self.kinetic_table.item(row, column).text()

            region = KineticRegionDefinition(
                name=text(0).strip(),
                kind=text(1).strip().lower(),
                center=tuple(float(text(column)) / 100.0 for column in range(2, 5)),
                size=tuple(float(text(column)) / 100.0 for column in range(5, 8)),
                kpl_s_inv=float(text(8)),
            )
            region.validate()
            regions.append(region)
        return regions

    def _add_kinetic_region(self, kind):
        row = self.kinetic_table.rowCount()
        previous = self.kinetic_table.blockSignals(True)
        self.kinetic_table.insertRow(row)
        values = (f"kPL region {row + 1}", kind, 50, 50, 50, 50, 50, 50, 0.05)
        for column, value in enumerate(values):
            self.kinetic_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.kinetic_table.blockSignals(previous)
        self.dynamic_enabled.setChecked(True)
        self._refresh_kinetics_preview_regions()
        self.kinetics_preview_region.setCurrentIndex(row + 1)
        self._update_kinetics_preview()

    def _remove_kinetic_region(self):
        row = self.kinetic_table.currentRow()
        if row >= 0:
            self.kinetic_table.removeRow(row)
            self._refresh_kinetics_preview_regions()
            self._update_kinetics_preview()

    def _next_shape_name(self):
        existing = {item.name for item in self.design.shapes}
        number = len(existing) + 1
        while f"Shape {number}" in existing:
            number += 1
        return f"Shape {number}"

    def _append_shape(self, item):
        self.design.shapes.append(item)
        self.shape_list.addItem(item.name)
        self._create_roi(item, len(self.design.shapes) - 1)
        self._refresh_kinetics_preview_shapes()
        self.shape_list.setCurrentRow(len(self.design.shapes) - 1)

    def _add_shape(self, kind):
        self._sync_global()
        item = ShapeDefinition(name=self._next_shape_name(), kind=kind)
        self._append_shape(item)

    def _start_shape_drawing(self, kind):
        self._sync_global()
        self.canvas.start_shape_drawing(kind)
        self.canvas_instruction.setText(
            f"Draw {kind}: hold the left mouse button and drag; Esc/right-click cancels"
        )

    def _shape_drawing_cancelled(self):
        self.canvas_instruction.setText(
            "Move/resize existing shapes, or choose Draw and drag in the axial XY plane"
        )

    def _shape_drawn(self, kind, left, bottom, width, height):
        if self.canvas.drawing_kind is not None:
            self.canvas.cancel_shape_drawing()
        item = ShapeDefinition(
            name=self._next_shape_name(),
            kind=kind,
            center=(left + width / 2.0, bottom + height / 2.0, 0.5),
            size=(width, height, 0.5),
        )
        self._append_shape(item)
        self._shape_drawing_cancelled()

    def _remove_shape(self):
        row = self._current_row()
        if row is None:
            return
        self.canvas.removeItem(self._rois.pop(row))
        self.design.shapes.pop(row)
        self.shape_list.takeItem(row)
        self._refresh_kinetics_preview_shapes()
        if self.design.shapes:
            self.shape_list.setCurrentRow(min(row, len(self.design.shapes) - 1))
        self._update_roi_highlights()
        self._update_kinetics_preview()

    def _add_peak(self):
        row = self._current_row()
        if row is None:
            return
        self.design.shapes[row].peaks.append(
            SpectralPeakDefinition(
                name=f"Peak {len(self.design.shapes[row].peaks) + 1}"
            )
        )
        self._populate_peaks(self.design.shapes[row])
        self._update_kinetics_preview()

    def _remove_peak(self):
        shape_row = self._current_row()
        peak_row = self.peak_table.currentRow()
        if shape_row is None or peak_row < 0:
            return
        peaks = self.design.shapes[shape_row].peaks
        if len(peaks) <= 1:
            return
        peaks.pop(peak_row)
        self._populate_peaks(self.design.shapes[shape_row])
        self._update_kinetics_preview()

    def _sync_global(self):
        self._properties_changed()
        row = self._current_row()
        if row is not None:
            self.design.shapes[row].peaks = self._read_peak_table()
        self.design.name = self.name_edit.text().strip() or "Designed spectral phantom"
        self.design.shape = tuple(widget.value() for widget in self.matrix_spins)
        self.design.fov_m = tuple(widget.value() / 1000.0 for widget in self.fov_spins)
        self.design.field_strength_t = self.field_strength_t.value()
        self.design.nucleus = self.nucleus.currentData()
        self.design.spectral_reference_ppm = self.spectral_reference_ppm.value()
        self.design.spectral_bandwidth_ppm = self.spectral_bandwidth_ppm.value()
        self.design.spectral_points = self.spectral_points.value()
        self.design.b0_inhomogeneity_mode = str(self.b0_mode_combo.currentData())
        self.design.b0_inhomogeneity_ppm = self.b0_inhomogeneity_ppm.value()
        dynamic_requested = (
            self.dynamic_enabled.isChecked()
            or self.inflow_enabled.isChecked()
            or self.dynamic_b0_enabled.isChecked()
        )
        self.dynamic_enabled.setChecked(dynamic_requested)
        self.design.dynamic_enabled = dynamic_requested
        self.design.pyruvate_peak_name = self.pyruvate_peak_name.text().strip()
        self.design.lactate_peak_name = self.lactate_peak_name.text().strip()
        self.design.default_kpl_s_inv = self.default_kpl.value()
        self.design.conversion_start_s = self.conversion_start_s.value()
        self.design.kinetics_time_offset_s = self.kinetics_time_offset_s.value()
        self.design.kinetic_regions = self._read_kinetic_regions()
        self.design.pyruvate_inflow_curve = (
            self._read_time_curve(self.inflow_curve_table, outside="zero")
            if self.inflow_enabled.isChecked()
            else None
        )
        self.design.dynamic_b0_curve = (
            self._read_time_curve(self.dynamic_b0_curve_table, outside="hold")
            if self.dynamic_b0_enabled.isChecked()
            else None
        )

    def _preview(self):
        try:
            self._sync_global()
            self.phantom = self.design.build()
            self.inspector.set_phantom(self.phantom)
            if self.design.dynamic_enabled:
                self.inspector.map_combo.setCurrentText("kPL")
            elif self.design.b0_inhomogeneity_mode != "none" or any(
                shape.b0_ppm != 0.0 for shape in self.design.shapes
            ):
                self.inspector.map_combo.setCurrentText("B0")
            self.tabs.setCurrentWidget(self.inspector)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid phantom", str(exc))

    def _save(self):
        default_path = workspace_directory("phantoms") / (
            f"{self.name_edit.text() or 'spectral_phantom'}.npz"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save spectral phantom",
            str(default_path),
            "Spectral phantom (*.npz *.h5 *.nc)",
        )
        if not filename:
            return
        try:
            self._sync_global()
            self.design.build().save(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load spectral phantom",
            str(workspace_directory("phantoms")),
            "Spectral phantom (*.npz *.h5 *.hdf5 *.nc)",
        )
        if not filename:
            return
        try:
            phantom = load_any_phantom(filename)
            self.design = PhantomDesign.from_phantom(phantom)
            self.phantom = phantom
            self._load_design_into_ui()
            self.inspector.set_phantom(phantom)
            self.tabs.setCurrentWidget(self.inspector)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def accept(self):
        try:
            self._sync_global()
            self.phantom = self.design.build()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid phantom", str(exc))
            return
        super().accept()

    def get_phantom(self):
        return self.phantom
