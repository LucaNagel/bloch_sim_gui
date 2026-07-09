"""Interactive shape and Lorentz-peak phantom designer."""

from __future__ import annotations

from typing import Optional
import weakref

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
from ..dynamic_phantom import DynamicSpectralPhantom, KineticRegionDefinition
from ..paths import workspace_directory
from ..phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from ..spectral_phantom import SpectralPhantom
from .volume_viewer import PhantomInspectorWidget


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
            fov.setRange(0.001, 100.0)
            fov.setDecimals(4)
            fov.setSuffix(" cm")
            global_row.addWidget(QLabel(f"FOV {axis}"))
            global_row.addWidget(fov)
            self.fov_spins.append(fov)
        draw_layout.addLayout(global_row)

        spectral_row = QHBoxLayout()
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

        splitter = QSplitter(Qt.Horizontal)
        draw_layout.addWidget(splitter)

        shape_panel = QWidget()
        shape_layout = QVBoxLayout(shape_panel)
        shape_layout.addWidget(QLabel("Shapes (later shapes overwrite B0 in overlaps)"))
        self.shape_list = QListWidget()
        self.shape_list.currentRowChanged.connect(self._shape_selected)
        shape_layout.addWidget(self.shape_list)
        add_row = QHBoxLayout()
        add_ellipse = QPushButton("Add ellipsoid")
        add_ellipse.clicked.connect(lambda: self._add_shape("ellipsoid"))
        add_box = QPushButton("Add box")
        add_box.clicked.connect(lambda: self._add_shape("box"))
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_shape)
        add_row.addWidget(add_ellipse)
        add_row.addWidget(add_box)
        add_row.addWidget(remove)
        shape_layout.addLayout(add_row)
        splitter.addWidget(shape_panel)

        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.addWidget(
            QLabel("Drag and resize shapes in the axial XY plane (physical FOV axes)")
        )
        self.canvas = pg.PlotWidget()
        self.canvas.setAspectLocked(True)
        self.canvas.setXRange(0, 1)
        self.canvas.setYRange(0, 1)
        self.canvas.showGrid(x=True, y=True, alpha=0.3)
        self.canvas.setLabel("bottom", "x", units="cm")
        self.canvas.setLabel("left", "y", units="cm")
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
                fov_cm = dialog.fov_spins[axis_index].value() if dialog else 1.0
                return [f"{(value - 0.5) * fov_cm:.3g}" for value in values]

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
        self.t1_ms = self._number_spin(0.1, 10000.0, 1000.0, " ms")
        self.initial_mz = self._number_spin(0.0, 1e9, 1.0, "")
        self.b0_ppm = self._number_spin(-1000.0, 1000.0, 0.0, " ppm")
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
        form.addRow("T1", self.t1_ms)
        form.addRow("Initial Mz", self.initial_mz)
        form.addRow("B0 inhomogeneity", self.b0_ppm)
        self.xy_info = QLabel()
        form.addRow("XY geometry", self.xy_info)
        property_layout.addLayout(form)

        property_layout.addWidget(
            QLabel(
                "Lorentz peaks: name, amplitude, absolute peak ppm, T2*. "
                "Simulation offsets are peak ppm minus spectral reference."
            )
        )
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(
            ["Name", "Amplitude", "Peak position (ppm)", "T2* (ms)"]
        )
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
        splitter.setStretchFactor(2, 0)
        self.tabs.addTab(draw_page, "Draw and edit")

        kinetics_page = QWidget()
        kinetics_layout = QVBoxLayout(kinetics_page)
        kinetics_form = QFormLayout()
        self.dynamic_enabled = QCheckBox("Enable pyruvate → lactate conversion")
        kinetics_form.addRow(self.dynamic_enabled)
        self.pyruvate_peak_name = QLineEdit("Pyruvate")
        self.lactate_peak_name = QLineEdit("Lactate")
        self.default_kpl = self._number_spin(0.0, 10.0, 0.0, " s⁻¹")
        kinetics_form.addRow("Pyruvate peak name", self.pyruvate_peak_name)
        kinetics_form.addRow("Lactate peak name", self.lactate_peak_name)
        kinetics_form.addRow("Default kPL", self.default_kpl)
        kinetics_layout.addLayout(kinetics_form)
        kinetics_layout.addWidget(
            QLabel(
                "Ordered kinetic regions; later rows overwrite earlier rows. "
                "Center and size use percent of the phantom FOV."
            )
        )
        self.kinetic_table = QTableWidget(0, 9)
        self.kinetic_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Cx %", "Cy %", "Cz %", "Sx %", "Sy %", "Sz %", "kPL s⁻¹"]
        )
        kinetics_layout.addWidget(self.kinetic_table)
        kinetic_buttons = QHBoxLayout()
        add_kinetic_ellipse = QPushButton("Add ellipsoid")
        add_kinetic_ellipse.clicked.connect(
            lambda: self._add_kinetic_region("ellipsoid")
        )
        add_kinetic_box = QPushButton("Add box")
        add_kinetic_box.clicked.connect(lambda: self._add_kinetic_region("box"))
        remove_kinetic = QPushButton("Remove")
        remove_kinetic.clicked.connect(self._remove_kinetic_region)
        kinetic_buttons.addWidget(add_kinetic_ellipse)
        kinetic_buttons.addWidget(add_kinetic_box)
        kinetic_buttons.addWidget(remove_kinetic)
        kinetic_buttons.addStretch()
        kinetics_layout.addLayout(kinetic_buttons)
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
            widget.setValue(float(value) * 100.0)
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
        self._populate_kinetic_regions()
        self.shape_list.clear()
        for roi in self._rois:
            self.canvas.removeItem(roi)
        self._rois.clear()
        for index, item in enumerate(self.design.shapes):
            self.shape_list.addItem(item.name)
            self._create_roi(item, index)
        self._updating = False
        if self.design.shapes:
            self.shape_list.setCurrentRow(0)

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
        self._update_roi_highlights(row)

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
            (item.center[index] - 0.5) * self.fov_spins[index].value() * 10.0
            for index in range(2)
        )
        size_mm = tuple(
            item.size[index] * self.fov_spins[index].value() * 10.0
            for index in range(2)
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

    def _populate_peaks(self, item):
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(len(item.peaks))
        for row, peak in enumerate(item.peaks):
            for column, value in enumerate(
                (
                    peak.name,
                    peak.amplitude,
                    peak.frequency_ppm + self.spectral_reference_ppm.value(),
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

    def _read_peak_table(self):
        peaks = []
        reference_ppm = self.spectral_reference_ppm.value()
        for peak_row in range(self.peak_table.rowCount()):
            peak = SpectralPeakDefinition(
                name=self.peak_table.item(peak_row, 0).text(),
                amplitude=float(self.peak_table.item(peak_row, 1).text()),
                frequency_ppm=(
                    float(self.peak_table.item(peak_row, 2).text()) - reference_ppm
                ),
                t2_star_s=float(self.peak_table.item(peak_row, 3).text()) / 1000.0,
            )
            peak.validate()
            peaks.append(peak)
        return peaks

    def _populate_kinetic_regions(self):
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
        self.kinetic_table.insertRow(row)
        values = (f"kPL region {row + 1}", kind, 50, 50, 50, 50, 50, 50, 0.05)
        for column, value in enumerate(values):
            self.kinetic_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.dynamic_enabled.setChecked(True)

    def _remove_kinetic_region(self):
        row = self.kinetic_table.currentRow()
        if row >= 0:
            self.kinetic_table.removeRow(row)

    def _add_shape(self, kind):
        self._sync_global()
        number = len(self.design.shapes) + 1
        item = ShapeDefinition(name=f"Shape {number}", kind=kind)
        self.design.shapes.append(item)
        self.shape_list.addItem(item.name)
        self._create_roi(item, len(self.design.shapes) - 1)
        self.shape_list.setCurrentRow(len(self.design.shapes) - 1)

    def _remove_shape(self):
        row = self._current_row()
        if row is None:
            return
        self.canvas.removeItem(self._rois.pop(row))
        self.design.shapes.pop(row)
        self.shape_list.takeItem(row)
        if self.design.shapes:
            self.shape_list.setCurrentRow(min(row, len(self.design.shapes) - 1))
        self._update_roi_highlights()

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

    def _sync_global(self):
        self._properties_changed()
        row = self._current_row()
        if row is not None:
            self.design.shapes[row].peaks = self._read_peak_table()
        self.design.name = self.name_edit.text().strip() or "Designed spectral phantom"
        self.design.shape = tuple(widget.value() for widget in self.matrix_spins)
        self.design.fov_m = tuple(widget.value() / 100.0 for widget in self.fov_spins)
        self.design.spectral_reference_ppm = self.spectral_reference_ppm.value()
        self.design.spectral_bandwidth_ppm = self.spectral_bandwidth_ppm.value()
        self.design.spectral_points = self.spectral_points.value()
        self.design.b0_inhomogeneity_mode = str(self.b0_mode_combo.currentData())
        self.design.b0_inhomogeneity_ppm = self.b0_inhomogeneity_ppm.value()
        self.design.dynamic_enabled = self.dynamic_enabled.isChecked()
        self.design.pyruvate_peak_name = self.pyruvate_peak_name.text().strip()
        self.design.lactate_peak_name = self.lactate_peak_name.text().strip()
        self.design.default_kpl_s_inv = self.default_kpl.value()
        self.design.kinetic_regions = self._read_kinetic_regions()

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
