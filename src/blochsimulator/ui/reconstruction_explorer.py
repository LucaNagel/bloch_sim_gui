"""Interactive, dimension-aware explorer for sequence reconstructions."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import zoom
from PyQt5.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..sequence.reconstruction import SequenceReconstructionModel
from .volume_viewer import VolumeViewerWidget
from .widgets import IMAGE_CANVAS_BACKGROUND, style_image_item


_IMAGE_INTERPOLATION_FACTOR = 8


def _colormap(name: str):
    """Return a pyqtgraph colormap, including a dependency-free gray map."""
    if name == "gray":
        return pg.ColorMap(
            np.asarray([0.0, 1.0]),
            np.asarray([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte),
        )
    return pg.colormap.get(name)


def _display_lut(name: str, strength: float, size: int = 256):
    """Build the lookup table used by both the preview and PNG export."""
    strength = max(float(strength), 1e-6)
    positions = np.linspace(0.0, 1.0, int(size)) ** (1.0 / strength)
    return np.asarray(_colormap(name).map(positions, mode="byte"), dtype=np.ubyte)


def _interpolate_image(values, interpolation: str):
    """Upsample only the displayed pixels while retaining the source extent."""
    data = np.asarray(values, dtype=float)
    order = {"nearest": 0, "linear": 1, "cubic": 3}.get(interpolation, 0)
    if order == 0 or min(data.shape) == 0:
        return data
    return zoom(
        data,
        _IMAGE_INTERPOLATION_FACTOR,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


class _DoubleRangeSlider(QWidget):
    """A compact horizontal float slider with independently draggable ends."""

    valuesChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._domain = (0.0, 1.0)
        self._values = (0.0, 1.0)
        self._active_handle = 0
        self._dragging = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumWidth(220)
        self.setToolTip(
            "Drag the left handle for the minimum and the right handle for the maximum"
        )

    def sizeHint(self):
        return QSize(320, 28)

    def values(self):
        return self._values

    def set_domain(self, low: float, high: float, *, preserve=True):
        low, high = float(low), float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 1.0
        if high <= low:
            high = np.nextafter(low, np.inf)
        previous = self._values
        self._domain = (low, high)
        self.set_values(*(previous if preserve else (low, high)), emit=False)

    def set_values(self, low: float, high: float, *, emit=True):
        domain_low, domain_high = self._domain
        low = float(np.clip(low, domain_low, domain_high))
        high = float(np.clip(high, domain_low, domain_high))
        if high < low:
            low, high = high, low
        changed = (low, high) != self._values
        self._values = (low, high)
        self.update()
        if changed and emit:
            self.valuesChanged.emit(low, high)

    def _handle_x(self, value):
        low, high = self._domain
        span = max(high - low, np.finfo(float).eps)
        fraction = np.clip((value - low) / span, 0.0, 1.0)
        return 10.0 + fraction * max(1.0, self.width() - 20.0)

    def _value_at(self, x):
        fraction = np.clip((float(x) - 10.0) / max(1.0, self.width() - 20.0), 0, 1)
        low, high = self._domain
        return low + fraction * (high - low)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.height() / 2.0
        left, right = 10.0, max(10.0, self.width() - 10.0)
        enabled = self.isEnabled()
        groove = QColor(120, 120, 125, 115 if enabled else 65)
        selected = self.palette().highlight().color()
        if not enabled:
            selected.setAlpha(80)
        painter.setPen(QPen(groove, 4.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(left, center), QPointF(right, center))
        lower_x, upper_x = (self._handle_x(value) for value in self._values)
        painter.setPen(QPen(selected, 5.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(lower_x, center), QPointF(upper_x, center))
        for index, position in enumerate((lower_x, upper_x)):
            fill = (
                selected
                if index == self._active_handle and enabled
                else QColor(245, 245, 245)
            )
            outline = selected if enabled else groove
            painter.setBrush(fill)
            painter.setPen(QPen(outline, 2.0))
            painter.drawEllipse(QPointF(position, center), 7.0, 7.0)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        positions = [self._handle_x(value) for value in self._values]
        self._active_handle = int(
            abs(event.x() - positions[1]) < abs(event.x() - positions[0])
        )
        self._dragging = True
        self.setFocus(Qt.MouseFocusReason)
        self._move_active_handle(event.x())
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._move_active_handle(event.x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        step = (self._domain[1] - self._domain[0]) / 1000.0
        direction = {
            Qt.Key_Left: -1,
            Qt.Key_Down: -1,
            Qt.Key_Right: 1,
            Qt.Key_Up: 1,
        }.get(event.key())
        if direction is not None:
            value = self._values[self._active_handle] + direction * step
            self._set_active_value(value)
            event.accept()
            return
        super().keyPressEvent(event)

    def _move_active_handle(self, x):
        self._set_active_value(self._value_at(x))

    def _set_active_value(self, value):
        low, high = self._values
        if self._active_handle == 0:
            low = min(float(value), high)
        else:
            high = max(float(value), low)
        self.set_values(low, high)


class _SelectableImagePanel(QWidget):
    pixel_selected = pyqtSignal(int, int)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._shape = (0, 0)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title=title)
        self.plot.setBackground(IMAGE_CANVAS_BACKGROUND)
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "x index")
        self.plot.setLabel("left", "y index")
        self.image = pg.ImageItem()
        style_image_item(self.image)
        self.plot.addItem(self.image)
        self.marker = pg.ScatterPlotItem(
            size=15,
            symbol="s",
            pen=pg.mkPen("y", width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.marker.setZValue(100)
        self.plot.addItem(self.marker)
        self.info = QLabel("No data")
        self.info.setWordWrap(True)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.info)
        self.plot.scene().sigMouseClicked.connect(self._clicked)

    def clear(self):
        self._shape = (0, 0)
        self.image.clear()
        self.marker.clear()
        self.info.setText("No data")

    def set_data(
        self,
        values,
        description: str,
        levels=None,
        *,
        lut=None,
        interpolation="nearest",
    ):
        data = np.asarray(values, dtype=float)
        if data.ndim != 2:
            raise ValueError("image panel data must be two-dimensional")
        display = np.nan_to_num(data, copy=True)
        display = _interpolate_image(display, interpolation)
        self._shape = data.shape
        self.image.setLookupTable(lut)
        self.image.setImage(
            display.T,
            autoLevels=levels is None,
            levels=levels,
        )
        self.image.setRect(0.0, 0.0, float(data.shape[1]), float(data.shape[0]))
        self.plot.autoRange()
        self.info.setText(f"{description}; shape={data.shape}")

    def set_selected_pixel(self, x: int, y: int):
        if self._shape == (0, 0):
            self.marker.clear()
            return
        x = int(np.clip(x, 0, self._shape[1] - 1))
        y = int(np.clip(y, 0, self._shape[0] - 1))
        self.marker.setData([x + 0.5], [y + 0.5])

    def _clicked(self, event):
        if event.button() != Qt.LeftButton:
            return
        view = self.plot.getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view.mapSceneToView(event.scenePos())
        x, y = int(np.floor(point.x())), int(np.floor(point.y()))
        if 0 <= y < self._shape[0] and 0 <= x < self._shape[1]:
            self.pixel_selected.emit(x, y)


class SequenceReconstructionExplorer(QWidget):
    """Explore Cartesian, spiral, radial, and CSI sequence result datasets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sequence_reconstruction_explorer")
        self.model = None
        self.source = None
        self.outer_controls = {}
        self.outer_values = {}
        self.outer_value_labels = {}
        self._updating = False
        self._voxel = (0, 0)
        self._current_display = None
        self._pending_state = None
        self._volume_initialized = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.summary = QLabel(
            "Run a sequence simulation or open an exported .nc result to explore it."
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        controls = QGroupBox("Reconstruction controls")
        controls_layout = QGridLayout(controls)
        self.outer_container = QWidget()
        self.outer_layout = QGridLayout(self.outer_container)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setHorizontalSpacing(8)
        self.outer_layout.setVerticalSpacing(2)
        self.outer_layout.setColumnStretch(1, 1)
        self.outer_layout.setColumnStretch(4, 1)
        controls_layout.addWidget(self.outer_container, 0, 0, 1, 8)

        controls_layout.addWidget(QLabel("Data"), 1, 0)
        self.pool_combo = QComboBox()
        self.pool_combo.currentIndexChanged.connect(self._refresh)
        controls_layout.addWidget(self.pool_combo, 1, 1)
        controls_layout.addWidget(QLabel("Receive channels"), 1, 2)
        self.coil_combo = QComboBox()
        self.coil_combo.currentIndexChanged.connect(self._refresh)
        controls_layout.addWidget(self.coil_combo, 1, 3)
        controls_layout.addWidget(QLabel("Display"), 1, 4)
        self.component_combo = QComboBox()
        for label, value in (
            ("Magnitude", "magnitude"),
            ("Phase", "phase"),
            ("Real", "real"),
            ("Imaginary", "imaginary"),
        ):
            self.component_combo.addItem(label, value)
        self.component_combo.currentIndexChanged.connect(self._refresh)
        controls_layout.addWidget(self.component_combo, 1, 5)
        self.spectral_label = QLabel("FID point")
        self.spectral_point = QSlider(Qt.Horizontal)
        self.spectral_point.setTracking(True)
        self.spectral_point.setMinimumWidth(150)
        self.spectral_point.setToolTip(
            "Browse the CSI FID sample used for the spatial reconstruction"
        )
        self.spectral_value_label = QLabel("0")
        self.spectral_value_label.setMinimumWidth(150)
        self.spectral_point.valueChanged.connect(self._spectral_point_changed)
        self.spectral_control = QWidget()
        spectral_layout = QHBoxLayout(self.spectral_control)
        spectral_layout.setContentsMargins(0, 0, 0, 0)
        spectral_layout.setSpacing(6)
        spectral_layout.addWidget(self.spectral_point, 1)
        spectral_layout.addWidget(self.spectral_value_label)
        controls_layout.addWidget(self.spectral_label, 1, 6)
        controls_layout.addWidget(self.spectral_control, 1, 7)

        self.auto_contrast = QCheckBox("Auto contrast")
        self.auto_contrast.setChecked(True)
        self.auto_contrast.toggled.connect(self._contrast_mode_changed)
        controls_layout.addWidget(self.auto_contrast, 2, 0)
        self.contrast_min_label = QLabel("Minimum 0")
        self.contrast_min_label.setMinimumWidth(125)
        self.contrast_min_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls_layout.addWidget(self.contrast_min_label, 2, 1)
        self.contrast_slider = _DoubleRangeSlider()
        self.contrast_slider.setEnabled(False)
        self.contrast_slider.valuesChanged.connect(self._contrast_range_changed)
        controls_layout.addWidget(self.contrast_slider, 2, 2, 1, 4)
        self.contrast_max_label = QLabel("Maximum 1")
        self.contrast_max_label.setMinimumWidth(125)
        controls_layout.addWidget(self.contrast_max_label, 2, 6, 1, 2)

        action_row = QHBoxLayout()
        self.export_button = QPushButton("Export current view…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_current_view)
        action_row.addWidget(self.export_button)
        action_row.addSpacing(12)
        action_row.addWidget(QLabel("Colormap"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["gray", "viridis", "plasma", "magma", "cividis", "inferno", "turbo"]
        )
        self.colormap_combo.setToolTip("Color palette for the reconstructed image")
        self.colormap_combo.currentTextChanged.connect(self._refresh)
        action_row.addWidget(self.colormap_combo)
        action_row.addWidget(QLabel("Intensity γ"))
        self.display_strength = QDoubleSpinBox()
        self.display_strength.setRange(0.1, 5.0)
        self.display_strength.setSingleStep(0.1)
        self.display_strength.setDecimals(2)
        self.display_strength.setValue(1.0)
        self.display_strength.setToolTip(
            "Colormap intensity (gamma): values above 1 brighten mid-tones"
        )
        self.display_strength.valueChanged.connect(self._refresh)
        action_row.addWidget(self.display_strength)
        action_row.addWidget(QLabel("Interpolation"))
        self.interpolation_combo = QComboBox()
        for label, value in (
            ("Nearest", "nearest"),
            ("Linear", "linear"),
            ("Cubic", "cubic"),
        ):
            self.interpolation_combo.addItem(label, value)
        self.interpolation_combo.setToolTip(
            "Image-space interpolation; the reconstructed source data remain unchanged"
        )
        self.interpolation_combo.currentIndexChanged.connect(self._refresh)
        action_row.addWidget(self.interpolation_combo)
        action_row.addStretch()
        controls_layout.addLayout(action_row, 3, 0, 1, 8)
        layout.addWidget(controls)

        self.pages = QStackedWidget()
        self.empty_page = QLabel("No reconstructable image data")
        self.empty_page.setAlignment(Qt.AlignCenter)
        self.pages.addWidget(self.empty_page)

        self.two_d_page = QWidget()
        two_d_layout = QVBoxLayout(self.two_d_page)
        images = QSplitter(Qt.Horizontal)
        self.kspace_panel = _SelectableImagePanel("Gridded k-space")
        self.image_panel = _SelectableImagePanel("Reconstruction")
        self.image_panel.pixel_selected.connect(self._voxel_selected)
        images.addWidget(self.kspace_panel)
        images.addWidget(self.image_panel)
        images.setSizes([1, 1])
        two_d_layout.addWidget(images, 3)
        self.spectrum_plot = pg.PlotWidget(title="Selected voxel")
        self.spectrum_plot.setLabel("left", "Signal", "a.u.")
        self.spectrum_plot.setLabel("bottom", "Frequency", "Hz")
        self.spectrum_plot.addLegend()
        two_d_layout.addWidget(self.spectrum_plot, 2)
        self.pages.addWidget(self.two_d_page)

        self.three_d_page = QWidget()
        three_d_layout = QVBoxLayout(self.three_d_page)
        three_d_layout.setContentsMargins(0, 0, 0, 0)
        self.volume_pages = QTabWidget()
        self.volume_pages.setDocumentMode(True)
        self.image_volume = VolumeViewerWidget()
        self.kspace_volume = VolumeViewerWidget()
        self.volume_pages.addTab(
            self.image_volume, "Reconstruction volume — independent slices"
        )
        self.volume_pages.addTab(
            self.kspace_volume, "Gridded k-space volume — independent slices"
        )
        self.volume_pages.setToolTip(
            "Image and k-space keep separate X/Y/Z slice positions."
        )
        three_d_layout.addWidget(self.volume_pages, 1)
        self.pages.addWidget(self.three_d_page)
        layout.addWidget(self.pages, 1)

    def clear(self, message=None):
        self.model = None
        self.source = None
        self._current_display = None
        self._volume_initialized = False
        self.kspace_panel.clear()
        self.image_panel.clear()
        self.spectrum_plot.clear()
        self.pages.setCurrentWidget(self.empty_page)
        self.empty_page.setText(message or "No reconstructable image data")
        self.summary.setText(
            message
            or "Run a sequence simulation or open an exported .nc result to explore it."
        )
        self.export_button.setEnabled(False)

    def set_result(self, result, phantom=None):
        source = result.metadata.get("sequence_source", "current simulation")
        self.set_model(SequenceReconstructionModel.from_result(result), source=source)

    def set_dataset(self, dataset, *, source=None):
        self.set_model(SequenceReconstructionModel(dataset), source=source)

    def set_model(self, model: SequenceReconstructionModel, *, source=None):
        self.model = model
        self.source = None if source is None else str(source)
        self._populate_controls()
        self._refresh()
        if self._pending_state is not None:
            state, self._pending_state = self._pending_state, None
            self.restore_state(state)

    def _clear_outer_controls(self):
        while self.outer_layout.count():
            item = self.outer_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.outer_controls.clear()
        self.outer_values.clear()
        self.outer_value_labels.clear()

    def _populate_controls(self):
        model = self.model
        self._updating = True
        try:
            self._clear_outer_controls()
            if model.outer_dimensions:
                for position, dimension in enumerate(model.outer_dimensions):
                    row = position // 2
                    column = (position % 2) * 3
                    label = QLabel(dimension.name.replace("_", " ").title())
                    self.outer_layout.addWidget(label, row, column)
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(0, len(dimension.values) - 1)
                    slider.setTracking(True)
                    slider.setMinimumWidth(140)
                    slider.setToolTip(
                        f"Browse reconstruction dimension {dimension.name!r}"
                    )
                    value_label = QLabel()
                    value_label.setMinimumWidth(90)
                    slider.valueChanged.connect(
                        lambda index, name=dimension.name: self._outer_slider_changed(
                            name, index
                        )
                    )
                    self.outer_layout.addWidget(slider, row, column + 1)
                    self.outer_layout.addWidget(value_label, row, column + 2)
                    self.outer_controls[dimension.name] = slider
                    self.outer_values[dimension.name] = tuple(dimension.values)
                    self.outer_value_labels[dimension.name] = value_label
                    self._update_outer_slider_label(dimension.name, 0)
            else:
                self.outer_layout.addWidget(
                    QLabel("Single acquisition frame"), 0, 0, 1, 6
                )

            self.pool_combo.clear()
            self.pool_combo.addItem("Combined signal", ("total", -1))
            if model.has_pool_data():
                for index, name in enumerate(model.pool_names):
                    self.pool_combo.addItem(f"Simulated pool: {name}", ("pool", index))
            for index, name in enumerate(model.ideal_species_names):
                self.pool_combo.addItem(
                    f"Estimated (linear IDEAL): {name}", ("ideal", index)
                )

            self.coil_combo.clear()
            self.coil_combo.addItem("Root-sum-of-squares", "rss")
            self.coil_combo.addItem("Coherent sum", "sum")
            for coil in range(model.coil_count):
                self.coil_combo.addItem(f"Coil {coil + 1}", f"coil:{coil}")
            self.coil_combo.setEnabled(model.coil_count > 1)

            is_csi = model.kind == "csi"
            self.spectral_label.setVisible(is_csi)
            self.spectral_control.setVisible(is_csi)
            count = model.dataset.sizes.get("spectral_point", 1)
            self.spectral_point.setRange(0, max(0, count - 1))
            self._update_spectral_point_label(self.spectral_point.value())
            self.spectrum_plot.setVisible(is_csi)
        finally:
            self._updating = False

    def _selections(self):
        selections = {}
        for name, control in self.outer_controls.items():
            if isinstance(control, QSlider):
                values = self.outer_values[name]
                selections[name] = values[control.value()]
            else:
                selections[name] = control.currentData()
        return selections

    def _outer_slider_changed(self, name, index):
        self._update_outer_slider_label(name, index)
        self._refresh()

    def _spectral_point_changed(self, index):
        self._update_spectral_point_label(index)
        self._refresh()

    def _update_spectral_point_label(self, index):
        if self.model is None:
            self.spectral_value_label.setText(str(index))
            return
        count = int(self.model.dataset.sizes.get("spectral_point", 1))
        index = int(np.clip(index, 0, max(0, count - 1)))
        text = f"{index}  ({index + 1}/{count})"
        frequency = self.model.dataset.coords.get("spectral_frequency_hz")
        if frequency is not None and frequency.size > index:
            value = float(np.asarray(frequency)[index])
            if np.isfinite(value):
                text += f" · {value:.5g} Hz"
        self.spectral_value_label.setText(text)

    def _update_outer_slider_label(self, name, index):
        values = self.outer_values.get(name, ())
        label = self.outer_value_labels.get(name)
        if label is None or not values:
            return
        index = int(np.clip(index, 0, len(values) - 1))
        label.setText(f"{values[index]}  ({index + 1}/{len(values)})")

    def _selected_data_mode(self):
        value = self.pool_combo.currentData()
        if not isinstance(value, tuple) or len(value) != 2:
            return "total", -1
        return str(value[0]), int(value[1])

    def _selected_image(self, selections=None):
        model = self.model
        selections = self._selections() if selections is None else dict(selections)
        mode, index = self._selected_data_mode()
        coil_mode = str(self.coil_combo.currentData() or "rss")
        if mode == "ideal":
            effective_coil_mode = "sum" if coil_mode == "rss" else coil_mode
            separated = model.ideal_images(selections, coil_mode=effective_coil_mode)
            template_selections = dict(selections)
            configuration = model.ideal_configuration
            template_selections[configuration[0].name] = configuration[0].values[0]
            template = model.select(
                model.image_name(),
                template_selections,
                coil_mode=effective_coil_mode,
            )
            return (
                template.copy(data=separated[index]),
                mode,
                index,
                effective_coil_mode,
            )
        pool = mode == "pool"
        return (
            model.select(
                model.image_name(pool=pool),
                selections,
                pool_index=index if pool else None,
                coil_mode=coil_mode,
            ),
            mode,
            index,
            coil_mode,
        )

    def _refresh(self, *_):
        if self._updating or self.model is None:
            return
        try:
            model = self.model
            image_data, mode, pool_index, effective_coil_mode = self._selected_image()
            component = str(self.component_combo.currentData() or "magnitude")
            display = model.display_values(image_data, component)
            display_levels = self._display_levels(
                display,
                domain=self._signal_series_contrast_domain(display, component),
            )
            image_lut = self._image_lut()
            interpolation = self._image_interpolation()
            if model.kind in {"cartesian_3d", "radial_3d"}:
                image_scalar = image_data.copy(data=display)
                image_volume, fov = model.scanner_volume(image_scalar)
                kspace_name = model.kspace_name(pool=mode == "pool")
                kspace = model.select(
                    kspace_name,
                    self._selections(),
                    pool_index=pool_index if mode == "pool" else None,
                    coil_mode=effective_coil_mode,
                )
                kspace_scalar = kspace.copy(data=np.log1p(np.abs(kspace)))
                kspace_volume, _ = model.scanner_volume(kspace_scalar)
                image_indices = (
                    self.image_volume.indices if self._volume_initialized else None
                )
                kspace_indices = (
                    self.kspace_volume.indices if self._volume_initialized else None
                )
                self.image_volume.set_volume(
                    image_volume,
                    fov_m=fov,
                    name=f"{component.title()} reconstruction",
                    levels=display_levels,
                    color_map=_colormap(self.colormap_combo.currentText()),
                    lookup_table=image_lut,
                    interpolation=interpolation,
                )
                self.kspace_volume.set_volume(
                    kspace_volume,
                    fov_m=fov,
                    name="log(1 + |gridded k-space|)",
                )
                if image_indices is not None:
                    self._set_volume_indices(self.image_volume, image_indices)
                if kspace_indices is not None:
                    self._set_volume_indices(self.kspace_volume, kspace_indices)
                self._volume_initialized = True
                self.pages.setCurrentWidget(self.three_d_page)
                self._current_display = image_volume
            else:
                self._refresh_two_dimensional(
                    image_data,
                    display,
                    mode,
                    pool_index,
                    effective_coil_mode,
                    display_levels,
                )
            source = f" · {Path(self.source).name}" if self.source else ""
            ideal_note = (
                " · linear known-frequency IDEAL (no B0 fit)" if mode == "ideal" else ""
            )
            self.summary.setText(
                f"{model.kind.replace('_', ' ').title()} reconstruction{source}{ideal_note}"
            )
            self.export_button.setEnabled(self._current_display is not None)
            echo_control = self.outer_controls.get("echo")
            if echo_control is not None:
                echo_control.setEnabled(mode != "ideal")
        except Exception as exc:
            self.pages.setCurrentWidget(self.empty_page)
            self.empty_page.setText(f"Reconstruction unavailable: {exc}")
            self.summary.setText(f"Reconstruction unavailable: {exc}")
            self._current_display = None
            self.export_button.setEnabled(False)

    def _refresh_two_dimensional(
        self,
        image_data,
        display,
        mode,
        pool_index,
        effective_coil_mode,
        display_levels,
    ):
        model = self.model
        if model.kind == "csi":
            point = self.spectral_point.value()
            image_data = image_data.isel(spectral_point=point)
            display = model.display_values(
                image_data, self.component_combo.currentData()
            )
        image_dims = [
            dimension
            for dimension in image_data.dims
            if dimension in model.spatial_dims
        ]
        if model.kind == "csi":
            image_dims = [
                dimension for dimension in image_dims if dimension != "spectral_point"
            ]
        if len(image_dims) != 2:
            raise ValueError(f"expected two image dimensions, found {image_data.dims}")
        image_values = np.asarray(image_data.copy(data=display).transpose(*image_dims))

        kspace_name = model.kspace_name(pool=mode == "pool")
        kspace = model.select(
            kspace_name,
            self._selections(),
            pool_index=pool_index if mode == "pool" else None,
            coil_mode=effective_coil_mode,
        )
        if model.kind == "csi":
            kspace = kspace.isel(spectral_point=self.spectral_point.value())
        kspace_dims = [
            dimension for dimension in kspace.dims if dimension in image_dims
        ]
        kspace_values = np.asarray(kspace.transpose(*kspace_dims))
        self.image_panel.set_data(
            image_values,
            f"{self.component_combo.currentText()} reconstruction",
            levels=display_levels,
            lut=self._image_lut(),
            interpolation=self._image_interpolation(),
        )
        self.kspace_panel.set_data(
            np.log1p(np.abs(kspace_values)), "log(1 + |gridded k-space|)"
        )
        self._voxel = (
            int(np.clip(self._voxel[0], 0, image_values.shape[1] - 1)),
            int(np.clip(self._voxel[1], 0, image_values.shape[0] - 1)),
        )
        self.image_panel.set_selected_pixel(*self._voxel)
        self.pages.setCurrentWidget(self.two_d_page)
        self._current_display = image_values
        if model.kind == "csi":
            self._refresh_csi_spectrum(mode, pool_index, effective_coil_mode)
        else:
            self.spectrum_plot.clear()

    def _contrast_mode_changed(self, automatic):
        self.contrast_slider.setEnabled(not automatic)
        self._refresh()

    def _contrast_range_changed(self, low, high):
        self._update_contrast_labels(low, high)
        if not self._updating and not self.auto_contrast.isChecked():
            self._refresh()

    def _update_contrast_labels(self, low, high):
        self.contrast_min_label.setText(f"Minimum {low:.7g}")
        self.contrast_max_label.setText(f"Maximum {high:.7g}")

    def _image_lut(self):
        return _display_lut(
            self.colormap_combo.currentText(), self.display_strength.value()
        )

    def _image_interpolation(self):
        return str(self.interpolation_combo.currentData() or "nearest")

    @staticmethod
    def _finite_range(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(finite.min()), float(finite.max())
        return None

    def _signal_series_contrast_domain(self, current_display, component):
        """Return one contrast domain across all slices and repetitions."""
        series_dimensions = tuple(
            item
            for item in self.model.outer_dimensions
            if item.name in {"slice", "repetition"}
        )
        if not series_dimensions:
            finite_range = self._finite_range(current_display)
            return None if finite_range is None else (0.0, max(0.0, finite_range[1]))

        selections = self._selections()
        high = None
        for values in product(*(item.values for item in series_dimensions)):
            series_selections = dict(selections)
            series_selections.update(
                (item.name, value) for item, value in zip(series_dimensions, values)
            )
            image_data, *_ = self._selected_image(series_selections)
            display = self.model.display_values(image_data, component)
            finite_range = self._finite_range(display)
            if finite_range is None:
                continue
            high = finite_range[1] if high is None else max(high, finite_range[1])
        return None if high is None else (0.0, max(0.0, high))

    def _display_levels(self, values, *, domain=None):
        finite_range = self._finite_range(values) if domain is None else domain
        high = 1.0 if finite_range is None else max(0.0, float(finite_range[1]))
        display_high = 1.1 * high
        if display_high <= 0.0:
            display_high = 1e-6
        if self.auto_contrast.isChecked():
            self.contrast_slider.set_domain(0.0, display_high, preserve=False)
            self._update_contrast_labels(0.0, display_high)
            return 0.0, display_high
        self.contrast_slider.set_domain(0.0, display_high, preserve=True)
        selected_low, selected_high = self.contrast_slider.values()
        self._update_contrast_labels(selected_low, selected_high)
        if selected_high <= selected_low:
            selected_high = np.nextafter(selected_low, np.inf)
        return selected_low, selected_high

    def _refresh_csi_spectrum(self, mode, pool_index, coil_mode):
        model = self.model
        spectrum_name = model.spectrum_name(pool=mode == "pool")
        spectrum = model.select(
            spectrum_name,
            self._selections(),
            pool_index=pool_index if mode == "pool" else None,
            coil_mode=coil_mode,
        )
        x, y = self._voxel
        spectrum = spectrum.isel(phase_x=x, phase_y=y)
        values = np.asarray(spectrum)
        frequency = np.asarray(
            model.dataset.coords.get("spectral_frequency_hz", np.arange(values.size))
        )
        self.spectrum_plot.clear()
        self.spectrum_plot.plot(frequency, np.abs(values), pen="w", name="Magnitude")
        self.spectrum_plot.plot(frequency, values.real, pen="g", name="Real")
        self.spectrum_plot.plot(frequency, values.imag, pen="r", name="Imaginary")
        self.spectrum_plot.setTitle(f"Voxel x={x}, y={y}")

    def _voxel_selected(self, x: int, y: int):
        self._voxel = (int(x), int(y))
        self.image_panel.set_selected_pixel(x, y)
        if self.model is not None and self.model.kind == "csi":
            mode, pool_index = self._selected_data_mode()
            coil_mode = str(self.coil_combo.currentData() or "rss")
            self._refresh_csi_spectrum(mode, pool_index, coil_mode)

    @staticmethod
    def _set_volume_indices(viewer, indices):
        if len(indices) != 3:
            return
        states = [slider.blockSignals(True) for slider in viewer.sliders]
        try:
            for slider, value in zip(viewer.sliders, indices):
                slider.setValue(int(np.clip(value, slider.minimum(), slider.maximum())))
        finally:
            for slider, state in zip(viewer.sliders, states):
                slider.blockSignals(state)
        viewer._indices_updated()

    @staticmethod
    def _set_combo_data(combo, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _set_outer_control_value(self, name, value):
        control = self.outer_controls.get(name)
        if control is None:
            return
        if isinstance(control, QSlider):
            values = self.outer_values.get(name, ())
            matches = [
                index for index, candidate in enumerate(values) if candidate == value
            ]
            if matches:
                control.setValue(matches[0])
                self._update_outer_slider_label(name, matches[0])
        else:
            self._set_combo_data(control, value)

    def get_state(self):
        return {
            "outer": self._selections(),
            "pool": list(self._selected_data_mode()),
            "coil": self.coil_combo.currentData(),
            "component": self.component_combo.currentData(),
            "auto_contrast": self.auto_contrast.isChecked(),
            "contrast_range": list(self.contrast_slider.values()),
            "colormap": self.colormap_combo.currentText(),
            "display_strength": self.display_strength.value(),
            "interpolation": self._image_interpolation(),
            "spectral_point": self.spectral_point.value(),
            "voxel": list(self._voxel),
            "image_volume_indices": list(self.image_volume.indices),
            "kspace_volume_indices": list(self.kspace_volume.indices),
            "volume_page": self.volume_pages.currentIndex(),
            "volume_tabs": [
                self.image_volume.tabs.currentIndex(),
                self.kspace_volume.tabs.currentIndex(),
            ],
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            return
        if self.model is None:
            self._pending_state = dict(state)
            return
        self._updating = True
        try:
            for name, value in dict(state.get("outer", {})).items():
                self._set_outer_control_value(name, value)
            pool = state.get("pool")
            if isinstance(pool, (list, tuple)) and len(pool) == 2:
                self._set_combo_data(self.pool_combo, (str(pool[0]), int(pool[1])))
            self._set_combo_data(self.coil_combo, state.get("coil"))
            self._set_combo_data(self.component_combo, state.get("component"))
            contrast_range = state.get("contrast_range", ())
            if isinstance(contrast_range, (list, tuple)) and len(contrast_range) == 2:
                self.contrast_slider.set_values(
                    float(contrast_range[0]), float(contrast_range[1]), emit=False
                )
                self._update_contrast_labels(*self.contrast_slider.values())
            automatic = bool(state.get("auto_contrast", True))
            self.auto_contrast.setChecked(automatic)
            self.contrast_slider.setEnabled(not automatic)
            self.colormap_combo.setCurrentText(str(state.get("colormap", "gray")))
            self.display_strength.setValue(float(state.get("display_strength", 1.0)))
            self._set_combo_data(
                self.interpolation_combo, state.get("interpolation", "nearest")
            )
            self.spectral_point.setValue(int(state.get("spectral_point", 0)))
            voxel = state.get("voxel", (0, 0))
            if isinstance(voxel, (list, tuple)) and len(voxel) == 2:
                self._voxel = (int(voxel[0]), int(voxel[1]))
        finally:
            self._updating = False
        self._refresh()
        legacy_indices = state.get("volume_indices", ())
        image_indices = state.get("image_volume_indices", legacy_indices)
        kspace_indices = state.get("kspace_volume_indices", legacy_indices)
        if isinstance(image_indices, (list, tuple)) and len(image_indices) == 3:
            self._set_volume_indices(self.image_volume, image_indices)
        if isinstance(kspace_indices, (list, tuple)) and len(kspace_indices) == 3:
            self._set_volume_indices(self.kspace_volume, kspace_indices)
        self.volume_pages.setCurrentIndex(
            int(np.clip(state.get("volume_page", 0), 0, self.volume_pages.count() - 1))
        )
        volume_tabs = state.get("volume_tabs", ())
        if isinstance(volume_tabs, (list, tuple)) and len(volume_tabs) == 2:
            for viewer, index in zip(
                (self.image_volume, self.kspace_volume), volume_tabs
            ):
                viewer.tabs.setCurrentIndex(
                    int(np.clip(index, 0, viewer.tabs.count() - 1))
                )

    def _export_current_view(self):
        if self._current_display is None:
            return
        export_directory = None
        window = self.window()
        for provider_name in ("_get_export_directory", "_export_directory"):
            provider = getattr(window, provider_name, None)
            if callable(provider):
                try:
                    export_directory = Path(provider())
                    break
                except Exception:
                    pass
        default_path = (export_directory or Path.cwd()) / "reconstruction_view.npy"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export current reconstruction view",
            str(default_path),
            "NumPy array (*.npy);;NumPy archive (*.npz);;PNG image (*.png)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            if (
                selected_filter.startswith("NumPy archive")
                or path.suffix.lower() == ".npz"
            ):
                path = path.with_suffix(".npz")
                np.savez_compressed(
                    path,
                    image=np.asarray(self._current_display),
                    reconstruction_kind=self.model.kind,
                )
            elif selected_filter.startswith("PNG") or path.suffix.lower() == ".png":
                path = path.with_suffix(".png")
                values = np.asarray(self._current_display)
                if values.ndim == 3:
                    values = values[:, :, values.shape[2] // 2].T
                finite = values[np.isfinite(values)]
                if self.auto_contrast.isChecked():
                    low = float(finite.min()) if finite.size else 0.0
                    high = float(finite.max()) if finite.size else 1.0
                else:
                    low, high = self.contrast_slider.values()
                scale = high - low if high > low else 1.0
                normalized = np.nan_to_num(
                    np.clip((values - low) / scale, 0.0, 1.0),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                normalized = _interpolate_image(normalized, self._image_interpolation())
                lut = self._image_lut()
                indices = np.clip(
                    np.rint(normalized * (len(lut) - 1)), 0, len(lut) - 1
                ).astype(np.intp)
                pixels = np.ascontiguousarray(lut[indices])
                height, width = pixels.shape[:2]
                image = QImage(
                    pixels.data,
                    width,
                    height,
                    int(pixels.strides[0]),
                    QImage.Format_RGBA8888,
                ).copy()
                if not image.save(str(path)):
                    raise ValueError("Qt could not encode the PNG image")
            else:
                path = path.with_suffix(".npy")
                np.save(path, np.asarray(self._current_display))
            QMessageBox.information(self, "Export complete", f"Exported {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
