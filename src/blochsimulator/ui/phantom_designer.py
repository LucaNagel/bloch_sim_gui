"""Interactive shape and Lorentz-peak phantom designer."""

from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
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
        self.setWindowTitle("Spectral Phantom Designer")
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
            QLabel("Drag and resize shapes in the normalized axial XY plane")
        )
        self.canvas = pg.PlotWidget()
        self.canvas.setAspectLocked(True)
        self.canvas.setXRange(0, 1)
        self.canvas.setYRange(0, 1)
        self.canvas.showGrid(x=True, y=True, alpha=0.3)
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
        self.z_center = self._percent_spin(50.0)
        self.z_size = self._percent_spin(50.0)
        self.t1_ms = self._number_spin(0.1, 10000.0, 1000.0, " ms")
        self.b0_ppm = self._number_spin(-1000.0, 1000.0, 0.0, " ppm")
        for widget in (self.z_center, self.z_size, self.t1_ms, self.b0_ppm):
            widget.valueChanged.connect(self._properties_changed)
        form.addRow("Z centre", self.z_center)
        form.addRow("Z thickness", self.z_size)
        form.addRow("T1", self.t1_ms)
        form.addRow("B0 inhomogeneity", self.b0_ppm)
        self.xy_info = QLabel()
        form.addRow("XY geometry", self.xy_info)
        property_layout.addLayout(form)

        property_layout.addWidget(
            QLabel("Lorentz peaks: name, amplitude, centre frequency, T2*")
        )
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(
            ["Name", "Amplitude", "Frequency offset (ppm)", "T2* (ms)"]
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
        pen = pg.intColor(index, hues=max(1, len(self.design.shapes)), alpha=220)
        roi_class = pg.EllipseROI if item.kind == "ellipsoid" else pg.RectROI
        roi = roi_class(position, size, pen=pen)
        roi.sigRegionChangeFinished.connect(self._roi_changed)
        self.canvas.addItem(roi)
        self._rois.append(roi)

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
        self.z_center.setValue(item.center[2] * 100.0)
        self.z_size.setValue(item.size[2] * 100.0)
        self.t1_ms.setValue(item.t1_s * 1000.0)
        self.b0_ppm.setValue(item.b0_ppm)
        self._populate_peaks(item)
        self._update_xy_info(row)
        self._updating = False

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
                self._update_xy_info(row)
                break

    def _update_xy_info(self, row):
        item = self.design.shapes[row]
        self.xy_info.setText(
            f"centre=({item.center[0]:.3f}, {item.center[1]:.3f}); "
            f"size=({item.size[0]:.3f}, {item.size[1]:.3f})"
        )

    def _properties_changed(self):
        row = self._current_row()
        if self._updating or row is None:
            return
        item = self.design.shapes[row]
        item.name = self.shape_name.text().strip() or item.name
        item.center = (item.center[0], item.center[1], self.z_center.value() / 100.0)
        item.size = (item.size[0], item.size[1], self.z_size.value() / 100.0)
        item.t1_s = self.t1_ms.value() / 1000.0
        item.b0_ppm = self.b0_ppm.value()
        item.b0_hz = None
        self.shape_list.item(row).setText(item.name)

    def _populate_peaks(self, item):
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(len(item.peaks))
        for row, peak in enumerate(item.peaks):
            for column, value in enumerate(
                (
                    peak.name,
                    peak.amplitude,
                    peak.frequency_ppm,
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
        for peak_row in range(self.peak_table.rowCount()):
            peak = SpectralPeakDefinition(
                name=self.peak_table.item(peak_row, 0).text(),
                amplitude=float(self.peak_table.item(peak_row, 1).text()),
                frequency_ppm=float(self.peak_table.item(peak_row, 2).text()),
                t2_star_s=float(self.peak_table.item(peak_row, 3).text()) / 1000.0,
            )
            peak.validate()
            peaks.append(peak)
        return peaks

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

    def _preview(self):
        try:
            self._sync_global()
            self.phantom = self.design.build()
            self.inspector.set_phantom(self.phantom)
            self.tabs.setCurrentIndex(1)
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
            "Spectral phantom (*.npz *.h5)",
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
            "Spectral phantom (*.npz *.h5 *.hdf5)",
        )
        if not filename:
            return
        try:
            phantom = SpectralPhantom.load(filename)
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
