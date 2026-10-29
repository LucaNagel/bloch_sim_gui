"""Interactive, dimension-aware explorer for sequence reconstructions."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import zoom
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..sequence.reconstruction import SequenceReconstructionModel
from .colormaps import COLORMAP_NAMES, colormap
from .reconstruction_phantom import (
    VOLUME_PLANES,
    contour_segments,
    grid_description,
    legend_html,
    planar_image_grid,
    resample_phantom,
    resampled_rgba,
    volume_display,
    volume_image_grid,
    volume_section_grid,
    voxel_composition,
)
from .sequence_geometry_viewer import (
    AXES,
    SEQUENCE_COLOR,
    VIEW_PLANES,
    automatic_contrast,
    object_contrasts,
    object_sections,
    projected_outline_mm,
)
from .volume_viewer import VolumeViewerWidget
from .widgets import IMAGE_CANVAS_BACKGROUND, style_image_item


_IMAGE_INTERPOLATION_FACTOR = 8
_PHANTOM_CACHE_SIZE = 48
_VOXEL_COLOR = (255, 0, 255)
# Initial width of the control column and smallest phantom overview height.
_CONTROLS_WIDTH = 420
_OVERVIEW_MIN_HEIGHT = 150
# Share of the result height the voxel plot gets when it appears.
_VOXEL_PLOT_SHARE = 0.4
_VOXEL_PLOT_MIN_HEIGHT = 160
# Interval between repetitions while the frame bar plays them.
_PLAY_INTERVAL_MS = 500
# Metabolite colours by name prefix, so one metabolite keeps its colour in
# every sequence; further names take the fallback colours in order.
_METABOLITE_COLORS = (
    (("pyr", "py"), "#3b9cff"),
    (("lac",), "#ff8c1a"),
    (("ala",), "#3ccf5a"),
    (("bic", "hco"), "#e8434b"),
    (("ure",), "#b07cff"),
    (("glu",), "#f2d43d"),
)
_FALLBACK_METABOLITE_COLORS = ("#33d6d6", "#ff6fd8", "#c8c8c8", "#9fd356")


def metabolite_colors(labels) -> dict:
    """Return one display colour per distinct metabolite label."""
    colors = {}
    fallback = iter(_FALLBACK_METABOLITE_COLORS * 4)
    for label in dict.fromkeys(labels):
        key = str(label).strip().lower()
        if not key:
            continue
        # "Pyr hydrate" must not take the pyruvate colour.
        named = (
            None
            if "hyd" in key
            else next(
                (
                    color
                    for prefixes, color in _METABOLITE_COLORS
                    if key.startswith(prefixes) and color not in colors.values()
                ),
                None,
            )
        )
        colors[label] = named or next(fallback)
    return colors


def _colormap(name: str):
    """Return a pyqtgraph colormap, including a dependency-free gray map."""
    return colormap(name)


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
        # The first domain shows its full range; later domains keep the choice.
        self._range_chosen = False
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
        keep = preserve and self._range_chosen
        previous = self._values
        self._domain = (low, high)
        self.set_values(*(previous if keep else (low, high)), emit=False)

    def set_values(self, low: float, high: float, *, emit=True):
        domain_low, domain_high = self._domain
        low = float(np.clip(low, domain_low, domain_high))
        high = float(np.clip(high, domain_low, domain_high))
        if high < low:
            low, high = high, low
        changed = (low, high) != self._values
        self._values = (low, high)
        self._range_chosen = True
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
        # A dark halo under a magenta outline keeps the selected voxel visible
        # on bright objects as well as on the dark background.
        self.marker_halo = pg.ScatterPlotItem(
            size=17,
            symbol="s",
            pen=pg.mkPen(0, 0, 0, 220, width=5),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.marker = pg.ScatterPlotItem(
            size=17,
            symbol="s",
            pen=pg.mkPen(255, 0, 255, width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.marker_halo.setZValue(99)
        self.marker.setZValue(100)
        self.plot.addItem(self.marker_halo)
        self.plot.addItem(self.marker)
        self.info = QLabel("No data")
        self.info.setWordWrap(True)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.info)
        self.plot.scene().sigMouseClicked.connect(self._clicked)

    def clear(self):
        self._shape = (0, 0)
        self.image.clear()
        self.marker_halo.clear()
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

    def set_rgba(self, rgba, shape, description: str):
        """Show colours mapped elsewhere on the pixel grid ``shape``."""
        self._shape = (int(shape[0]), int(shape[1]))
        self.image.setLookupTable(None)
        self.image.setImage(
            np.transpose(np.asarray(rgba), (1, 0, 2)),
            autoLevels=False,
            levels=(0, 255),
        )
        self.image.setRect(0.0, 0.0, float(shape[1]), float(shape[0]))
        self.plot.autoRange()
        self.info.setText(description)

    def set_selected_pixel(self, x: int, y: int):
        if self._shape == (0, 0):
            self.marker_halo.clear()
            self.marker.clear()
            return
        x = int(np.clip(x, 0, self._shape[1] - 1))
        y = int(np.clip(y, 0, self._shape[0] - 1))
        self.marker_halo.setData([x + 0.5], [y + 0.5])
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


class _ContourOverlay:
    """Phantom region boundaries drawn as thin lines on an image view."""

    def __init__(self, view):
        self._view = view
        self._items = []

    @property
    def drawn(self) -> bool:
        return bool(self._items)

    def clear(self):
        for item in self._items:
            self._view.removeItem(item)
        self._items = []

    def set_segments(self, segments, to_view):
        """Draw :func:`contour_segments`; ``to_view`` maps points to ``(x, y)``."""
        self.clear()
        for rgba, points in segments:
            x, y = to_view(points)
            item = pg.PlotCurveItem(
                np.asarray(x, dtype=float),
                np.asarray(y, dtype=float),
                connect="pairs",
                pen=pg.mkPen(rgba, width=2),
            )
            item.setZValue(50)
            self._view.addItem(item, ignoreBounds=True)
            self._items.append(item)


class _PhantomOverview(QWidget):
    """Phantom sections through the selected voxel with FOV and voxel outlines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground(IMAGE_CANVAS_BACKGROUND)
        self.canvas.setFixedHeight(_OVERVIEW_MIN_HEIGHT)
        self.canvas.setToolTip(
            "Phantom sections through the selected voxel centre. Orange outlines "
            "the imaged field of view (and slab), magenta the selected voxel."
        )
        self.plots = []
        self.images = []
        self.fov_outlines = []
        self.voxel_outlines = []
        for horizontal, vertical in VIEW_PLANES:
            plot = self.canvas.addPlot(
                title=f"{AXES[horizontal].upper()}–{AXES[vertical].upper()}"
            )
            plot.setAspectLocked(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(False, False)
            plot.hideButtons()
            image = pg.ImageItem(axisOrder="col-major")
            plot.addItem(image)
            fov = pg.PlotDataItem(pen=pg.mkPen(SEQUENCE_COLOR, width=1.5))
            voxel = pg.PlotDataItem(pen=pg.mkPen(*_VOXEL_COLOR, width=2))
            fov.setZValue(10)
            voxel.setZValue(11)
            plot.addItem(fov)
            plot.addItem(voxel)
            self.plots.append(plot)
            self.images.append(image)
            self.fov_outlines.append(fov)
            self.voxel_outlines.append(voxel)
        layout.addWidget(self.canvas)

    def resizeEvent(self, event):
        # Three square sections side by side grow with the column width.
        super().resizeEvent(event)
        height = int(np.clip(self.width() / 3 + 24, _OVERVIEW_MIN_HEIGHT, 360))
        if height != self.canvas.height():
            self.canvas.setFixedHeight(height)

    def clear(self):
        for image, fov, voxel in zip(
            self.images, self.fov_outlines, self.voxel_outlines
        ):
            image.clear()
            fov.setData([], [])
            voxel.setData([], [])

    def set_view(self, phantom, contrast, image_box, voxel_box):
        sections, _shown_range = object_sections(phantom, contrast, voxel_box.center_m)
        for plot, image, fov, voxel, (rgba, rect), plane in zip(
            self.plots,
            self.images,
            self.fov_outlines,
            self.voxel_outlines,
            sections,
            VIEW_PLANES,
        ):
            image.setImage(rgba, levels=(0, 255), autoLevels=False)
            image.setRect(rect)
            outline = projected_outline_mm(image_box, plane)
            fov.setData(outline[:, 0], outline[:, 1])
            marker = projected_outline_mm(voxel_box, plane)
            voxel.setData(marker[:, 0], marker[:, 1])
            # The phantom sets the view; a much larger FOV is clipped, the
            # voxel always stays inside.
            shown = QRectF(rect)
            low, high = marker[:, :2].min(axis=0), marker[:, :2].max(axis=0)
            shown = shown.united(
                QRectF(
                    low[0],
                    low[1],
                    max(high[0] - low[0], 1e-3),
                    max(high[1] - low[1], 1e-3),
                )
            )
            plot.setRange(rect=shown, padding=0.05)


class SequenceReconstructionExplorer(QWidget):
    """Explore Cartesian, spiral, radial, and CSI sequence result datasets."""

    sequence_settings_visibility_requested = pyqtSignal(bool)

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
        self._contrast_domain_cache = {}
        self._time_course_cache = {}
        self._repetition_labels = {}
        self._metabolite_colors = {}
        # Manual contrast window of each metabolite and the one last shown.
        self._metabolite_windows = {}
        self._window_metabolite = None
        self._refreshing = False
        self._image_dims = ()
        self._session_runs = {}
        self._result = None
        # Returns Δk per ADC sample of a result for the gradient-error option.
        self.gradient_error_provider = None
        self._phantom = None
        self._phantom_contrasts = []
        self._phantom_contrast = None
        self._phantom_grid = None
        self._phantom_cache = {}
        self._syncing_volumes = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        simulation_row = QHBoxLayout()
        simulation_row.addWidget(QLabel("Simulation"))
        self.simulation_combo = QComboBox()
        self.simulation_combo.setObjectName("reconstruction_simulation_selector")
        self.simulation_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.simulation_combo.setMinimumContentsLength(30)
        self.simulation_combo.setPlaceholderText("Run a sequence simulation first")
        self.simulation_combo.setEnabled(False)
        self.simulation_combo.setToolTip(
            "Choose a completed simulation, including individual parameter-loop runs. "
            "This changes the reconstruction view without changing the sequence setup."
        )
        self.simulation_combo.currentIndexChanged.connect(self._simulation_changed)
        simulation_row.addWidget(self.simulation_combo, 1)
        # The sequence parameters are not needed while browsing results, so
        # the host collapses them for this tab; this button brings them back.
        self.sequence_settings_button = QPushButton("Show sequence settings")
        self.sequence_settings_button.setObjectName(
            "reconstruction_sequence_settings_toggle"
        )
        self.sequence_settings_button.setCheckable(True)
        self.sequence_settings_button.setToolTip(
            "Show or hide the sequence parameter column while exploring results"
        )
        self.sequence_settings_button.toggled.connect(self._sequence_settings_toggled)
        simulation_row.addWidget(self.sequence_settings_button)
        layout.addLayout(simulation_row)
        self.summary = QLabel(
            "Run a sequence simulation or open an exported .nc result to explore it."
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        self.body_splitter = body

        self.pages = QStackedWidget()
        self.empty_page = QLabel("No reconstructable image data")
        self.empty_page.setAlignment(Qt.AlignCenter)
        self.pages.addWidget(self.empty_page)

        self.two_d_page = QWidget()
        two_d_layout = QVBoxLayout(self.two_d_page)
        two_d_layout.setContentsMargins(0, 0, 0, 0)
        result_splitter = QSplitter(Qt.Vertical)
        result_splitter.setChildrenCollapsible(False)
        self.result_splitter = result_splitter
        images = QSplitter(Qt.Horizontal)
        self.kspace_panel = _SelectableImagePanel("Gridded k-space")
        self.image_panel = _SelectableImagePanel("Reconstruction")
        self.image_panel.pixel_selected.connect(self._voxel_selected)
        self.phantom_panel = _SelectableImagePanel("Phantom on the image grid")
        self.phantom_panel.setToolTip(
            "The simulated phantom resampled onto the reconstruction grid: the "
            "same pixel shows the object part that the voxel measures. Click to "
            "select a voxel."
        )
        self.phantom_panel.pixel_selected.connect(self._voxel_selected)
        self.phantom_panel.hide()
        # Region outlines of the phantom drawn on top of the reconstruction.
        self.image_contours = _ContourOverlay(self.image_panel.plot)
        images.addWidget(self.kspace_panel)
        images.addWidget(self.image_panel)
        images.addWidget(self.phantom_panel)
        # The reconstruction is the image of interest; k-space stays as a
        # smaller companion view.
        images.setStretchFactor(0, 2)
        images.setStretchFactor(1, 3)
        images.setStretchFactor(2, 3)
        result_splitter.addWidget(images)
        self.spectrum_plot = pg.PlotWidget(title="Selected voxel")
        self.spectrum_plot.setLabel("left", "Signal", "a.u.")
        self.spectrum_plot.setLabel("bottom", "Frequency", "Hz")
        self.spectrum_plot.addLegend()
        self.spectrum_plot.setMinimumHeight(_VOXEL_PLOT_MIN_HEIGHT)
        result_splitter.addWidget(self.spectrum_plot)
        result_splitter.setStretchFactor(0, 3)
        result_splitter.setStretchFactor(1, 2)
        self._voxel_plot_share = _VOXEL_PLOT_SHARE
        result_splitter.splitterMoved.connect(self._voxel_plot_resized)
        two_d_layout.addWidget(result_splitter, 1)
        self.pages.addWidget(self.two_d_page)

        # Acquisitions without spatial encoding, such as a single pulse: the
        # signal of the one voxel is shown as FID and spectrum.
        self.signal_page = QWidget()
        signal_layout = QVBoxLayout(self.signal_page)
        signal_layout.setContentsMargins(0, 0, 0, 0)
        signal_event_row = QHBoxLayout()
        self.signal_event_label = QLabel("Excitation")
        self.signal_event_spin = QSpinBox()
        self.signal_event_spin.setToolTip(
            "Excitation whose spectrum is shown; the FID plot covers all of them"
        )
        self.signal_event_spin.valueChanged.connect(self._refresh)
        signal_event_row.addWidget(self.signal_event_label)
        signal_event_row.addWidget(self.signal_event_spin)
        signal_event_row.addStretch(1)
        signal_layout.addLayout(signal_event_row)
        self.signal_fid_plot = pg.PlotWidget(title="FID")
        self.signal_spectrum_plot = pg.PlotWidget(title="Spectrum")
        for plot, axis, unit in (
            (self.signal_fid_plot, "Time", "ms"),
            (self.signal_spectrum_plot, "Frequency", "Hz"),
        ):
            plot.setLabel("left", "Signal", "a.u.")
            plot.setLabel("bottom", axis, unit)
            plot.addLegend()
            plot.setMinimumHeight(_VOXEL_PLOT_MIN_HEIGHT)
            signal_layout.addWidget(plot, 1)
        self.pages.addWidget(self.signal_page)

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
        self.phantom_volume = VolumeViewerWidget(show_3d=False)
        self.volume_pages.addTab(self.phantom_volume, "Phantom on the image grid")
        self.volume_pages.setToolTip(
            "Image and k-space keep separate X/Y/Z slice positions; the phantom "
            "follows the reconstruction."
        )
        self._volume_contour_items = []
        for viewer in (self.image_volume, self.phantom_volume):
            viewer.indices_changed.connect(
                lambda indices, source=viewer: self._volume_indices_changed(
                    source, indices
                )
            )
            for plane, view in (
                ("xy", viewer.xy_view),
                ("xz", viewer.xz_view),
                ("yz", viewer.yz_view),
            ):
                overlay = _ContourOverlay(view.getView())
                self._volume_contour_items.append((viewer, plane, overlay))
        self.volume_splitter = QSplitter(Qt.Vertical)
        self.volume_splitter.setChildrenCollapsible(False)
        self.volume_splitter.addWidget(self.volume_pages)
        self.volume_plot = pg.PlotWidget(title="Selected voxel")
        self.volume_plot.setObjectName("reconstruction_volume_voxel_plot")
        self.volume_plot.addLegend()
        self.volume_plot.setMinimumHeight(_VOXEL_PLOT_MIN_HEIGHT)
        self.volume_splitter.addWidget(self.volume_plot)
        self.volume_splitter.setStretchFactor(0, 3)
        self.volume_splitter.setStretchFactor(1, 1)
        self.volume_plot.hide()
        self.image_volume.indices_changed.connect(self._volume_voxel_changed)
        three_d_layout.addWidget(self.volume_splitter, 1)
        self.pages.addWidget(self.three_d_page)

        image_area = QWidget()
        image_area_layout = QVBoxLayout(image_area)
        image_area_layout.setContentsMargins(0, 0, 0, 0)
        image_area_layout.addWidget(self.pages, 1)
        image_area_layout.addWidget(self._build_frame_bar())
        body.addWidget(image_area)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(6, 0, 6, 0)

        data_group = QGroupBox("Data")
        data_layout = QFormLayout(data_group)
        self.pool_combo = QComboBox()
        self.pool_combo.currentIndexChanged.connect(self._refresh)
        data_layout.addRow("Data", self.pool_combo)
        self.coil_combo = QComboBox()
        self.coil_combo.currentIndexChanged.connect(self._refresh)
        data_layout.addRow("Receive channels", self.coil_combo)
        self.component_combo = QComboBox()
        for label, value in (
            ("Magnitude", "magnitude"),
            ("Phase", "phase"),
            ("Real", "real"),
            ("Imaginary", "imaginary"),
        ):
            self.component_combo.addItem(label, value)
        self.component_combo.currentIndexChanged.connect(self._refresh)
        data_layout.addRow("Display", self.component_combo)
        self.spectral_label = QLabel("FID point")
        self.spectral_point = QSlider(Qt.Horizontal)
        self.spectral_point.setTracking(True)
        self.spectral_point.setToolTip(
            "Browse the CSI FID sample used for the spatial reconstruction"
        )
        self.spectral_value_label = QLabel("0")
        self.spectral_point.valueChanged.connect(self._spectral_point_changed)
        self.spectral_control = QWidget()
        spectral_layout = QVBoxLayout(self.spectral_control)
        spectral_layout.setContentsMargins(0, 0, 0, 0)
        spectral_layout.setSpacing(2)
        spectral_layout.addWidget(self.spectral_point)
        spectral_layout.addWidget(self.spectral_value_label)
        data_layout.addRow(self.spectral_label, self.spectral_control)
        self.voxel_plot_label = QLabel("Voxel plot")
        self.voxel_plot_combo = QComboBox()
        self.voxel_plot_combo.setObjectName("reconstruction_voxel_plot")
        self.voxel_plot_combo.setToolTip(
            "Signal of the selected voxel shown below the images"
        )
        self.voxel_plot_combo.currentIndexChanged.connect(self._voxel_plot_changed)
        data_layout.addRow(self.voxel_plot_label, self.voxel_plot_combo)
        controls_layout.addWidget(data_group)
        controls_layout.addWidget(self._build_reconstruction_group())

        phantom_group = QGroupBox("Phantom")
        phantom_group.setObjectName("reconstruction_phantom_group")
        phantom_layout = QVBoxLayout(phantom_group)
        self.phantom_show = QCheckBox("Show phantom next to the image")
        self.phantom_show.setObjectName("reconstruction_phantom_show")
        self.phantom_show.setChecked(False)
        self.phantom_show.setToolTip(
            "Resample the simulated phantom onto the reconstruction grid"
        )
        self.phantom_show.toggled.connect(self._phantom_options_changed)
        phantom_layout.addWidget(self.phantom_show)
        self.phantom_contours = QCheckBox("Outline phantom on the reconstruction")
        self.phantom_contours.setObjectName("reconstruction_phantom_contours")
        self.phantom_contours.setToolTip(
            "Draw the object outline (cyan) and region boundaries of the phantom "
            "on the reconstructed image"
        )
        self.phantom_contours.toggled.connect(self._phantom_options_changed)
        phantom_layout.addWidget(self.phantom_contours)
        phantom_form = QFormLayout()
        self.phantom_contrast_combo = QComboBox()
        self.phantom_contrast_combo.setObjectName("reconstruction_phantom_contrast")
        self.phantom_contrast_combo.setToolTip(
            "Phantom property that colours the phantom view and its outlines"
        )
        self.phantom_contrast_combo.currentIndexChanged.connect(
            self._phantom_options_changed
        )
        phantom_form.addRow("Colour by", self.phantom_contrast_combo)
        phantom_layout.addLayout(phantom_form)
        self.phantom_overview = _PhantomOverview()
        phantom_layout.addWidget(self.phantom_overview)
        self.phantom_legend = QLabel()
        self.phantom_legend.setWordWrap(True)
        self.phantom_legend.setTextFormat(Qt.RichText)
        phantom_layout.addWidget(self.phantom_legend)
        self.phantom_composition = QLabel()
        self.phantom_composition.setObjectName("reconstruction_phantom_composition")
        self.phantom_composition.setWordWrap(True)
        self.phantom_composition.setTextFormat(Qt.RichText)
        self.phantom_composition.setTextInteractionFlags(Qt.TextSelectableByMouse)
        phantom_layout.addWidget(self.phantom_composition)
        self.phantom_note = QLabel()
        self.phantom_note.setWordWrap(True)
        phantom_layout.addWidget(self.phantom_note)
        controls_layout.addWidget(phantom_group)
        self._set_phantom(None)

        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout(contrast_group)
        self.auto_contrast = QCheckBox("Auto contrast")
        self.auto_contrast.setChecked(False)
        self.auto_contrast.toggled.connect(self._contrast_mode_changed)
        contrast_layout.addWidget(self.auto_contrast)
        self.contrast_slider = _DoubleRangeSlider()
        self.contrast_slider.valuesChanged.connect(self._contrast_range_changed)
        contrast_layout.addWidget(self.contrast_slider)
        range_row = QHBoxLayout()
        self.contrast_min_label = QLabel("Minimum 0")
        range_row.addWidget(self.contrast_min_label)
        range_row.addStretch()
        self.contrast_max_label = QLabel("Maximum 1")
        self.contrast_max_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        range_row.addWidget(self.contrast_max_label)
        contrast_layout.addLayout(range_row)
        controls_layout.addWidget(contrast_group)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        self.kspace_scale_combo = QComboBox()
        self.kspace_scale_combo.addItem("Magnitude", "magnitude")
        self.kspace_scale_combo.addItem("Log magnitude", "log")
        self.kspace_scale_combo.setCurrentIndex(1)
        self.kspace_scale_combo.setToolTip(
            "Show gridded k-space as linear magnitude or log(1 + magnitude)"
        )
        self.kspace_scale_combo.currentIndexChanged.connect(self._refresh)
        appearance_layout.addRow("K-space view", self.kspace_scale_combo)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(COLORMAP_NAMES)
        self.colormap_combo.setToolTip("Color palette for the reconstructed image")
        self.colormap_combo.currentTextChanged.connect(self._refresh)
        appearance_layout.addRow("Colormap", self.colormap_combo)
        self.display_strength = QDoubleSpinBox()
        self.display_strength.setRange(0.1, 5.0)
        self.display_strength.setSingleStep(0.1)
        self.display_strength.setDecimals(2)
        self.display_strength.setValue(1.0)
        self.display_strength.setToolTip(
            "Colormap intensity (gamma): values above 1 brighten mid-tones"
        )
        self.display_strength.valueChanged.connect(self._refresh)
        appearance_layout.addRow("Intensity γ", self.display_strength)
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
        appearance_layout.addRow("Interpolation", self.interpolation_combo)
        controls_layout.addWidget(appearance_group)

        self.export_button = QPushButton("Export current view…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_current_view)
        controls_layout.addWidget(self.export_button)
        controls_layout.addStretch(1)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("reconstruction_controls")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(controls)
        self.controls_scroll.setMinimumWidth(240)
        body.addWidget(self.controls_scroll)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        # The images take the width the control column does not need; the
        # column has no maximum, so dragging the handle never leaves a gap.
        body.setSizes([3 * _CONTROLS_WIDTH, _CONTROLS_WIDTH])
        layout.addWidget(body, 1)

    def _build_frame_bar(self):
        """Build the frame bar below the images: outer dimensions and playback."""
        self.frame_bar = QFrame()
        self.frame_bar.setObjectName("reconstruction_frame_bar")
        self.frame_bar.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(self.frame_bar)
        layout.setContentsMargins(8, 4, 8, 4)
        self.play_button = QToolButton()
        self.play_button.setObjectName("reconstruction_play_repetitions")
        self.play_button.setCheckable(True)
        self.play_button.setText("▶")
        self.play_button.setToolTip("Play through the repetitions")
        self.play_button.toggled.connect(self._play_toggled)
        layout.addWidget(self.play_button)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(_PLAY_INTERVAL_MS)
        self._play_timer.timeout.connect(self._play_step)
        self.outer_container = QWidget()
        self.outer_layout = QGridLayout(self.outer_container)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setHorizontalSpacing(6)
        self.outer_layout.setVerticalSpacing(2)
        self.outer_layout.setColumnStretch(1, 1)
        layout.addWidget(self.outer_container, 1)
        self.metabolite_filter = QComboBox()
        self.metabolite_filter.setObjectName("reconstruction_metabolite_filter")
        self.metabolite_filter.setToolTip(
            "Browse, average, and contrast-scale only the repetitions that image "
            "one metabolite"
        )
        self.metabolite_filter.currentIndexChanged.connect(
            self._metabolite_filter_changed
        )
        layout.addWidget(self.metabolite_filter)
        self.average_repetitions = QCheckBox("Average repetitions")
        self.average_repetitions.setObjectName("reconstruction_average_repetitions")
        self.average_repetitions.setToolTip(
            "Show the complex mean over all repetitions, whatever component is "
            "displayed. Averaging happens before the receive channels are combined. "
            "With spectrally selective sequences only the repetitions of one "
            "metabolite are averaged."
        )
        self.average_repetitions.toggled.connect(self._average_toggled)
        self.average_repetitions.setVisible(False)
        layout.addWidget(self.average_repetitions)
        self.frame_bar.hide()
        return self.frame_bar

    def _sequence_settings_toggled(self, visible):
        self.sequence_settings_button.setText(
            "Hide sequence settings" if visible else "Show sequence settings"
        )
        self.sequence_settings_visibility_requested.emit(bool(visible))

    def set_sequence_settings_visible(self, visible):
        """Reflect the host's parameter-column state without re-emitting it."""
        previous = self.sequence_settings_button.blockSignals(True)
        self.sequence_settings_button.setChecked(bool(visible))
        self.sequence_settings_button.blockSignals(previous)
        self.sequence_settings_button.setText(
            "Hide sequence settings" if visible else "Show sequence settings"
        )

    def clear(self, message=None):
        self.model = None
        self.source = None
        self._result = None
        self._select_session_run(None)
        self._current_display = None
        self._volume_initialized = False
        self._contrast_domain_cache.clear()
        self._time_course_cache.clear()
        self.play_button.setChecked(False)
        self.frame_bar.hide()
        self.kspace_panel.clear()
        self.image_panel.clear()
        self.spectrum_plot.clear()
        self.signal_fid_plot.clear()
        self.signal_spectrum_plot.clear()
        self.volume_plot.clear()
        self.volume_plot.hide()
        self._set_phantom(None)
        self.pages.setCurrentWidget(self.empty_page)
        self.empty_page.setText(message or "No reconstructable image data")
        self.summary.setText(
            message
            or "Run a sequence simulation or open an exported .nc result to explore it."
        )
        self.export_button.setEnabled(False)

    def _build_reconstruction_group(self):
        """Spiral reconstruction method and gradient-error options."""
        group = QGroupBox("Reconstruction")
        group.setObjectName("reconstruction_method_group")
        layout = QFormLayout(group)
        self.reconstruction_method_combo = QComboBox()
        self.reconstruction_method_combo.setObjectName("reconstruction_method")
        for label, value in (
            ("Density-compensated NUFFT", "nufft"),
            ("Iterative CG-NUFFT", "cg"),
            ("Linear gridding", "linear"),
        ):
            self.reconstruction_method_combo.addItem(label, value)
        self.reconstruction_method_combo.setToolTip(
            "NUFFT: adjoint NUFFT with Pipe-Menon density compensation. "
            "CG-NUFFT: 10 conjugate-gradient iterations of the least-squares "
            "problem, sharper but slower. Linear: interpolation of the samples "
            "onto the Cartesian grid followed by an FFT."
        )
        self.reconstruction_method_combo.currentIndexChanged.connect(
            self._reconstruction_options_changed
        )
        layout.addRow("Method", self.reconstruction_method_combo)
        self.reconstruction_trajectory_combo = QComboBox()
        self.reconstruction_trajectory_combo.setObjectName("reconstruction_trajectory")
        self.reconstruction_trajectory_combo.addItem("Ideal gradients", False)
        self.reconstruction_trajectory_combo.addItem(
            "Gradient errors, uncorrected", True
        )
        self.reconstruction_trajectory_combo.setToolTip(
            "Reconstruct as if the gradient imperfections set in the k-space "
            "tab had acted during the scan while the reconstruction assumes the "
            "nominal trajectory. The simulated signal is reconstructed on "
            "k - Δk, which gives the same trajectory mismatch to first order."
        )
        self.reconstruction_trajectory_combo.currentIndexChanged.connect(
            self._reconstruction_options_changed
        )
        layout.addRow("Trajectory", self.reconstruction_trajectory_combo)
        self.reconstruction_message = QLabel()
        self.reconstruction_message.setWordWrap(True)
        self.reconstruction_message.setStyleSheet("color: #475569;")
        layout.addRow(self.reconstruction_message)
        self.reconstruction_group = group
        group.setVisible(False)
        return group

    def _update_reconstruction_controls(self, result):
        """Show the spiral options for ``result`` and select its own method."""
        try:
            spiral = getattr(result, "spiral_acquisition", None)
        except ValueError:
            spiral = None
        self.reconstruction_group.setVisible(spiral is not None)
        if spiral is None:
            return
        for combo, value in (
            (self.reconstruction_method_combo, spiral.reconstruction_method),
            (self.reconstruction_trajectory_combo, False),
        ):
            previous = combo.blockSignals(True)
            self._set_combo_data(combo, value)
            combo.blockSignals(previous)
        self.reconstruction_message.setText("")
        self.reconstruction_trajectory_combo.setEnabled(
            self.gradient_error_provider is not None
        )

    def reconstruction_variant(self, result, method, gradient_errors=False):
        """Return ``result`` reconstructed with ``method`` (spiral only).

        With ``gradient_errors`` the ADC moments are shifted by -Δk from
        :attr:`gradient_error_provider`, so the reconstruction assumes a
        trajectory that differs from the one that encoded the signal.
        """
        from dataclasses import replace

        spiral = result.spiral_acquisition.with_reconstruction_method(method)
        metadata = dict(result.metadata)
        metadata["spiral_acquisition"] = spiral.to_metadata()
        moments = result.adc_gradient_moment_cyc_per_m
        if gradient_errors:
            delta = np.asarray(self.gradient_error_provider(result), dtype=float)
            moments = np.asarray(moments, dtype=float) - delta
        return replace(result, metadata=metadata, adc_gradient_moment_cyc_per_m=moments)

    def _reconstruction_options_changed(self, *_):
        result = self._result
        if result is None or self.model is None:
            return
        method = self.reconstruction_method_combo.currentData()
        errors = bool(self.reconstruction_trajectory_combo.currentData())
        message = ""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                variant = self.reconstruction_variant(result, method, errors)
            except ValueError as exc:
                message = f"{exc}; reconstructed with ideal gradients"
                variant = self.reconstruction_variant(result, method)
            model = SequenceReconstructionModel.from_result(variant)
        finally:
            QApplication.restoreOverrideCursor()
        state = self.get_state()
        run_id = self.simulation_combo.currentData()
        self.set_model(model, source=self.source, phantom=self._phantom)
        self.restore_state(state)
        self._result = result
        self._select_session_run(run_id)
        self.reconstruction_group.setVisible(True)
        self.reconstruction_message.setText(message)
        self.reconstruction_message.setStyleSheet(
            "color: #b45309;" if message else "color: #475569;"
        )

    def set_result(self, result, phantom=None):
        source = result.metadata.get("sequence_source", "current simulation")
        run = next(
            (run for run in self._session_runs.values() if run.result is result), None
        )
        if run is not None:
            source = run.sequence_name
        self.set_model(
            SequenceReconstructionModel.from_result(result),
            source=source,
            phantom=phantom,
        )
        self._result = result
        self._select_session_run(None if run is None else run.run_id)
        self._update_reconstruction_controls(result)

    def set_dataset(self, dataset, *, source=None):
        self.set_model(SequenceReconstructionModel(dataset), source=source)

    def set_model(
        self, model: SequenceReconstructionModel, *, source=None, phantom=None
    ):
        self._result = None
        self._select_session_run(None)
        self.reconstruction_group.setVisible(False)
        self.play_button.setChecked(False)
        self.model = model
        self.source = None if source is None else str(source)
        self._contrast_domain_cache.clear()
        self._time_course_cache.clear()
        self._metabolite_windows.clear()
        self._window_metabolite = None
        self._set_phantom(phantom)
        self._populate_controls()
        self._refresh()
        if self._pending_state is not None:
            state, self._pending_state = self._pending_state, None
            self.restore_state(state)

    def _session_run_label(self, run):
        number = list(self._session_runs).index(run.run_id) + 1
        return run.display_name or f"Simulation {number} · {run.sequence_name}"

    def _select_session_run(self, run_id):
        previous = self.simulation_combo.blockSignals(True)
        self.simulation_combo.setCurrentIndex(
            -1 if run_id is None else self.simulation_combo.findData(run_id)
        )
        self.simulation_combo.blockSignals(previous)

    def add_session_run(self, run, *, current=False):
        """Expose a retained result without copying its scientific arrays."""
        self._session_runs[run.run_id] = run
        index = self.simulation_combo.findData(run.run_id)
        previous = self.simulation_combo.blockSignals(True)
        if index < 0:
            index = self.simulation_combo.count()
            self.simulation_combo.addItem(self._session_run_label(run), run.run_id)
        else:
            self.simulation_combo.setItemText(index, self._session_run_label(run))
        self.simulation_combo.setItemData(
            index, f"{run.sequence_name} · {run.phantom_name}", Qt.ToolTipRole
        )
        if current or self._result is run.result:
            self.simulation_combo.setCurrentIndex(index)
        elif self._result is None and self.simulation_combo.count() == 1:
            self.simulation_combo.setCurrentIndex(-1)
        self.simulation_combo.blockSignals(previous)
        self.simulation_combo.setEnabled(True)
        self.simulation_combo.setPlaceholderText("Select a completed simulation")

    def remove_session_run(self, run):
        selected = self.simulation_combo.currentData() == run.run_id
        self._session_runs.pop(run.run_id, None)
        index = self.simulation_combo.findData(run.run_id)
        previous = self.simulation_combo.blockSignals(True)
        if index >= 0:
            self.simulation_combo.removeItem(index)
        self.simulation_combo.blockSignals(previous)
        if selected:
            self.clear("The displayed session simulation was deleted")
        self.simulation_combo.setEnabled(bool(self._session_runs))

    def _simulation_changed(self, *_):
        run = self._session_runs.get(self.simulation_combo.currentData())
        if run is None:
            return
        try:
            self.set_result(run.result, run.state.get("phantom"))
        except Exception as exc:
            self.clear(f"Interactive reconstruction unavailable: {exc}")
            self._select_session_run(run.run_id)

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
                    row = position
                    column = 0
                    label = QLabel(dimension.name.replace("_", " ").title())
                    self.outer_layout.addWidget(label, row, column)
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(0, len(dimension.values) - 1)
                    slider.setTracking(True)
                    slider.setMinimumWidth(80)
                    slider.setToolTip(
                        f"Browse reconstruction dimension {dimension.name!r}"
                    )
                    value_label = QLabel()
                    value_label.setMinimumWidth(60)
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
            self._populate_metabolites()
            for name in self.outer_controls:
                self._update_outer_slider_label(name, 0)
            self.frame_bar.setVisible(bool(model.outer_dimensions))

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

            has_repetitions = self._time_course_dimension() is not None
            self.average_repetitions.setVisible(has_repetitions)
            self.play_button.setVisible(has_repetitions)
            if not has_repetitions:
                self.average_repetitions.setChecked(False)
            self._update_average_availability()

            is_csi = model.kind == "csi"
            self.spectral_label.setVisible(is_csi)
            self.spectral_control.setVisible(is_csi)
            count = model.dataset.sizes.get("spectral_point", 1)
            self.spectral_point.setRange(0, max(0, count - 1))
            self._update_spectral_point_label(self.spectral_point.value())
            self._populate_voxel_plot_modes(is_csi)
        finally:
            self._updating = False

    def _populate_metabolites(self):
        """Label repetitions with the metabolite their spectral pulse excites."""
        dimension = self._time_course_dimension()
        labels = () if dimension is None else self.model.repetition_metabolites
        self._repetition_labels = dict(zip(dimension.values, labels)) if labels else {}
        self._metabolite_colors = metabolite_colors(labels)
        previous = self.metabolite_filter.currentData()
        self.metabolite_filter.blockSignals(True)
        try:
            self.metabolite_filter.clear()
            self.metabolite_filter.addItem("All metabolites", None)
            for label, color in self._metabolite_colors.items():
                count = sum(1 for value in labels if value == label)
                self.metabolite_filter.addItem(f"{label} ({count})", label)
                index = self.metabolite_filter.count() - 1
                self.metabolite_filter.setItemData(
                    index, pg.mkColor(color), Qt.ForegroundRole
                )
            self.metabolite_filter.setCurrentIndex(
                max(0, self.metabolite_filter.findData(previous))
            )
        finally:
            self.metabolite_filter.blockSignals(False)
        self.metabolite_filter.setVisible(len(self._metabolite_colors) > 1)
        self._apply_metabolite_filter()

    def _metabolite_filter_value(self):
        if len(self._metabolite_colors) < 2:
            return None
        return self.metabolite_filter.currentData()

    def _apply_metabolite_filter(self):
        """Restrict the repetition slider to the selected metabolite."""
        dimension = self._time_course_dimension()
        slider = self.outer_controls.get("repetition")
        if dimension is None or slider is None:
            return
        label = self._metabolite_filter_value()
        values = tuple(
            value
            for value in dimension.values
            if label is None or self._repetition_labels.get(value) == label
        ) or tuple(dimension.values)
        previous_values = self.outer_values.get("repetition", values)
        current = previous_values[
            int(np.clip(slider.value(), 0, len(previous_values) - 1))
        ]
        # Keep the repetition, or move to the next one of the new metabolite.
        index = next(
            (position for position, value in enumerate(values) if value >= current),
            len(values) - 1,
        )
        self.outer_values["repetition"] = values
        blocked = slider.blockSignals(True)
        try:
            slider.setRange(0, len(values) - 1)
            slider.setValue(index)
        finally:
            slider.blockSignals(blocked)
        self._update_outer_slider_label("repetition", index)
        self._update_average_availability()

    def _metabolite_filter_changed(self, *_):
        if self._updating or self.model is None:
            return
        self._apply_metabolite_filter()
        self._refresh()

    def _average_allowed(self):
        """Average only images of one metabolite, never mixed contrasts."""
        return (
            len(self._metabolite_colors) < 2
            or self._metabolite_filter_value() is not None
        )

    def _update_average_availability(self):
        allowed = self._average_allowed()
        self.average_repetitions.setEnabled(allowed)
        self.average_repetitions.setText(
            "Average repetitions"
            if allowed or not self._metabolite_colors
            else "Average repetitions (choose a metabolite)"
        )

    def _current_metabolite(self):
        """Return the metabolite of the shown image, or None if unlabelled."""
        if not self._repetition_labels:
            return None
        label = self._metabolite_filter_value()
        if label is not None:
            return label
        return self._repetition_labels.get(self._selections().get("repetition"))

    def _metabolite_html(self, label):
        color = self._metabolite_colors.get(label)
        if color is None:
            return ""
        return f'<span style="color:{color}; font-weight:600">● {label}</span>'

    def _update_metabolite_annotation(self):
        label = self._current_metabolite()
        color = self._metabolite_colors.get(label)
        pen = None if color is None else pg.mkPen(color, width=3)
        self.image_panel.plot.getPlotItem().getViewBox().setBorder(pen)
        for view in (
            self.image_volume.xy_view,
            self.image_volume.xz_view,
            self.image_volume.yz_view,
        ):
            view.getView().setBorder(pen)

    def _play_toggled(self, playing):
        self.play_button.setText("⏸" if playing else "▶")
        self.play_button.setToolTip(
            "Pause playback" if playing else "Play through the repetitions"
        )
        if playing:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _play_step(self):
        slider = self.outer_controls.get("repetition")
        if slider is None or not slider.isEnabled() or self.model is None:
            self.play_button.setChecked(False)
            return
        slider.setValue((slider.value() + 1) % (slider.maximum() + 1))

    def _populate_voxel_plot_modes(self, is_csi):
        previous = self.voxel_plot_combo.currentData()
        self.voxel_plot_combo.blockSignals(True)
        try:
            self.voxel_plot_combo.clear()
            if is_csi:
                self.voxel_plot_combo.addItem("Spectrum", "spectrum")
                self.voxel_plot_combo.addItem("FID", "fid")
            if self._time_course_dimension() is not None:
                self.voxel_plot_combo.addItem("Time course", "time_course")
            index = self.voxel_plot_combo.findData(previous)
            self.voxel_plot_combo.setCurrentIndex(max(0, index))
        finally:
            self.voxel_plot_combo.blockSignals(False)
        available = self.voxel_plot_combo.count() > 0
        self.voxel_plot_label.setVisible(available)
        self.voxel_plot_combo.setVisible(available)
        reappearing = available and self.spectrum_plot.isHidden()
        self.spectrum_plot.setVisible(available)
        if reappearing:
            self._restore_voxel_plot_height()

    def _restore_voxel_plot_height(self):
        # A splitter gives a re-shown widget only its minimum height.
        total = sum(self.result_splitter.sizes())
        if total > 0:
            plot = int(round(total * self._voxel_plot_share))
            self.result_splitter.setSizes([total - plot, plot])

    def _voxel_plot_resized(self, *_):
        sizes = self.result_splitter.sizes()
        if self.spectrum_plot.isVisible() and sum(sizes) > 0:
            self._voxel_plot_share = sizes[1] / sum(sizes)

    def _time_course_dimension(self):
        """Return the outer dimension that orders dynamic acquisitions."""
        if self.model is None:
            return None
        return next(
            (item for item in self.model.outer_dimensions if item.name == "repetition"),
            None,
        )

    def _voxel_plot_changed(self, *_):
        if self._updating or self.model is None:
            return
        mode, pool_index = self._selected_data_mode()
        self._refresh_voxel_plot(mode, pool_index, self._effective_coil_mode(mode))

    def _effective_coil_mode(self, mode):
        coil_mode = str(self.coil_combo.currentData() or "rss")
        return "sum" if mode == "ideal" and coil_mode == "rss" else coil_mode

    def _selections(self):
        selections = {}
        for name, control in self.outer_controls.items():
            if isinstance(control, QSlider):
                values = self.outer_values[name]
                selections[name] = values[control.value()]
            else:
                selections[name] = control.currentData()
        return selections

    def _average_dimensions(self):
        """Return the outer dimensions averaged instead of browsed."""
        if self.model is None or not self.average_repetitions.isChecked():
            return ()
        if self._time_course_dimension() is None or not self._average_allowed():
            return ()
        return ("repetition",)

    def _average_kwargs(self):
        """Return the averaging arguments of ``SequenceReconstructionModel.select``."""
        averaged = self._average_dimensions()
        subset = (
            {"repetition": self.outer_values["repetition"]}
            if averaged and self._metabolite_filter_value() is not None
            else None
        )
        return {"average_dimensions": averaged, "average_values": subset}

    def _average_toggled(self, checked):
        if checked:
            self.play_button.setChecked(False)
        if self._updating:
            return
        self._refresh()

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
        text = f"{values[index]}  ({index + 1}/{len(values)})"
        if name == "repetition" and self._repetition_labels:
            metabolite = self._repetition_labels.get(values[index])
            label.setTextFormat(Qt.RichText)
            text = f"{values[index]} {self._metabolite_html(metabolite)}  ({index + 1}/{len(values)})"
        label.setText(text)

    def _selected_data_mode(self):
        value = self.pool_combo.currentData()
        if not isinstance(value, tuple) or len(value) != 2:
            return "total", -1
        return str(value[0]), int(value[1])

    def _selected_image(self, selections=None, *, average=True):
        model = self.model
        selections = self._selections() if selections is None else dict(selections)
        averaging = (
            self._average_kwargs()
            if average
            else {"average_dimensions": (), "average_values": None}
        )
        mode, index = self._selected_data_mode()
        coil_mode = str(self.coil_combo.currentData() or "rss")
        if mode == "ideal":
            effective_coil_mode = "sum" if coil_mode == "rss" else coil_mode
            separated = model.ideal_images(
                selections,
                coil_mode=effective_coil_mode,
                **averaging,
            )
            template_selections = dict(selections)
            configuration = model.ideal_configuration
            template_selections[configuration[0].name] = configuration[0].values[0]
            template = model.select(
                model.image_name(),
                template_selections,
                coil_mode=effective_coil_mode,
                **averaging,
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
                **averaging,
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
            if model.kind == "raw_signal":
                self._refresh_acquisition_signal()
                return
            image_data, mode, pool_index, effective_coil_mode = self._selected_image()
            component = str(self.component_combo.currentData() or "magnitude")
            display = model.display_values(image_data, component)
            contrast_domain = (
                None
                if self.auto_contrast.isChecked()
                else self._dataset_contrast_domain(display, component)
            )
            image_lut = self._image_lut()
            interpolation = self._image_interpolation()
            if model.kind in {"cartesian_3d", "radial_3d"}:
                # The previous phantom grid may not fit the new volume.
                self._phantom_grid = None
                display_levels = self._display_levels(
                    display,
                    domain=contrast_domain,
                )
                image_scalar = image_data.copy(data=display)
                image_volume, fov = model.scanner_volume(image_scalar)
                kspace_name = model.kspace_name(pool=mode == "pool")
                kspace = model.select(
                    kspace_name,
                    self._selections(),
                    pool_index=pool_index if mode == "pool" else None,
                    coil_mode=effective_coil_mode,
                    **self._average_kwargs(),
                )
                kspace_display, kspace_title = self._kspace_display(kspace)
                kspace_scalar = kspace.copy(data=kspace_display)
                kspace_volume, _ = model.scanner_volume(kspace_scalar)
                metabolite = self._current_metabolite()
                image_name = f"{component.title()} reconstruction" + (
                    "" if metabolite is None else f" · {metabolite}"
                )
                image_settings = dict(
                    name=image_name,
                    levels=display_levels,
                    color_map=_colormap(self.colormap_combo.currentText()),
                    lookup_table=image_lut,
                )
                same_grid = (
                    self._volume_initialized
                    and np.shape(self.image_volume.data) == np.shape(image_volume)
                    and np.shape(self.kspace_volume.data) == np.shape(kspace_volume)
                    and self.image_volume.fov_m is not None
                    and np.allclose(self.image_volume.fov_m[: len(fov)], fov)
                )
                self._refreshing = True
                try:
                    if same_grid:
                        # Browsing repetitions keeps the slices and 3D camera.
                        self.image_volume.interpolation = interpolation
                        self.image_volume.update_volume_data(
                            image_volume, **image_settings
                        )
                        self.kspace_volume.update_volume_data(
                            kspace_volume, name=kspace_title
                        )
                    else:
                        image_indices = (
                            self.image_volume.indices
                            if self._volume_initialized
                            else None
                        )
                        kspace_indices = (
                            self.kspace_volume.indices
                            if self._volume_initialized
                            else None
                        )
                        self.image_volume.set_volume(
                            image_volume,
                            fov_m=fov,
                            interpolation=interpolation,
                            **image_settings,
                        )
                        self.kspace_volume.set_volume(
                            kspace_volume,
                            fov_m=fov,
                            name=kspace_title,
                        )
                        if image_indices is not None:
                            self._set_volume_indices(self.image_volume, image_indices)
                        if kspace_indices is not None:
                            self._set_volume_indices(self.kspace_volume, kspace_indices)
                    self._refresh_phantom_volume()
                finally:
                    self._refreshing = False
                self._volume_initialized = True
                self.pages.setCurrentWidget(self.three_d_page)
                self._current_display = image_volume
                self._refresh_volume_plot()
            else:
                self._refresh_two_dimensional(
                    image_data,
                    display,
                    mode,
                    pool_index,
                    effective_coil_mode,
                    contrast_domain,
                )
            source = f" · {Path(self.source).name}" if self.source else ""
            ideal_note = (
                " · linear known-frequency IDEAL (no B0 fit)" if mode == "ideal" else ""
            )
            metabolite = self._current_metabolite()
            metabolite_note = "" if metabolite is None else f" · {metabolite}"
            self.summary.setText(
                f"{model.kind.replace('_', ' ').title()} reconstruction"
                f"{source}{metabolite_note}{ideal_note}"
            )
            self._update_metabolite_annotation()
            self.export_button.setEnabled(self._current_display is not None)
            echo_control = self.outer_controls.get("echo")
            if echo_control is not None:
                echo_control.setEnabled(mode != "ideal")
            repetition_control = self.outer_controls.get("repetition")
            if repetition_control is not None:
                repetition_control.setEnabled(not self._average_dimensions())
        except Exception as exc:
            self.pages.setCurrentWidget(self.empty_page)
            self.empty_page.setText(f"Reconstruction unavailable: {exc}")
            self.summary.setText(f"Reconstruction unavailable: {exc}")
            self._current_display = None
            self.export_button.setEnabled(False)

    def _refresh_acquisition_signal(self):
        """Show the FID and spectrum of an acquisition without image encoding."""
        model = self.model
        mode, index = self._selected_data_mode()
        pool = mode == "pool"
        signal = np.asarray(
            model.acquisition_signal(
                pool_index=index if pool else None,
                coil_mode=str(self.coil_combo.currentData() or "rss"),
            )
        ).reshape(-1)
        times = np.asarray(model.dataset.coords["adc_time_s"].values, dtype=float)
        events = model.acquisition_events()
        event_values = list(dict.fromkeys(events.tolist()))
        self.signal_event_label.setVisible(len(event_values) > 1)
        self.signal_event_spin.setVisible(len(event_values) > 1)
        blocked = self.signal_event_spin.blockSignals(True)
        self.signal_event_spin.setRange(1, max(1, len(event_values)))
        self.signal_event_spin.blockSignals(blocked)
        selected = event_values[self.signal_event_spin.value() - 1]
        rows = np.flatnonzero(events == selected)

        self.signal_fid_plot.clear()
        self.signal_spectrum_plot.clear()
        self.signal_fid_plot.setLimits(xMin=None, xMax=None)
        measurement_ms = 1e3 * float(model.dataset.attrs.get("duration_s") or 0.0)
        time_ms = times * 1e3
        for plot, x, values in (
            (self.signal_fid_plot, time_ms, signal),
            *self._signal_spectrum(times[rows], signal[rows]),
        ):
            plot.plot(x, np.abs(values), pen="w", name="Magnitude")
            plot.plot(x, values.real, pen="g", name="Real")
            plot.plot(x, values.imag, pen="r", name="Imaginary")
        if time_ms.size:
            # Zooming out of the FID stops at the measurement.
            self.signal_fid_plot.setLimits(
                xMin=0.0, xMax=max(measurement_ms, float(time_ms.max()))
            )
        self.signal_fid_plot.setTitle("FID")
        label = "" if not pool else f" · {model.pool_names[index]}"
        self.signal_spectrum_plot.setTitle(f"Spectrum{label}")
        self.pages.setCurrentWidget(self.signal_page)
        source = f" · {Path(self.source).name}" if self.source else ""
        self.summary.setText(
            f"Single-voxel acquisition{source}{label}: no spatial encoding, so the "
            "FID and its spectrum are shown instead of an image"
        )
        self._current_display = None
        self.export_button.setEnabled(False)

    def _signal_spectrum(self, times, values):
        """Return ``[(plot, frequency_hz, spectrum)]`` of one excitation's FID."""
        if values.size < 2:
            return []
        dwell = float(np.median(np.diff(times)))
        if not dwell > 0:
            return []
        frequency = np.fft.fftshift(np.fft.fftfreq(values.size, dwell))
        spectrum = np.fft.fftshift(np.fft.fft(values))
        return [(self.signal_spectrum_plot, frequency, spectrum)]

    def _refresh_two_dimensional(
        self,
        image_data,
        display,
        mode,
        pool_index,
        effective_coil_mode,
        contrast_domain,
    ):
        model = self.model
        if model.kind == "csi":
            point = self.spectral_point.value()
            image_data = image_data.isel(spectral_point=point)
            display = model.display_values(
                image_data, self.component_combo.currentData()
            )
        display_levels = self._display_levels(display, domain=contrast_domain)
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
        self._image_dims = tuple(image_dims)

        kspace_name = model.kspace_name(pool=mode == "pool")
        kspace = model.select(
            kspace_name,
            self._selections(),
            pool_index=pool_index if mode == "pool" else None,
            coil_mode=effective_coil_mode,
            **self._average_kwargs(),
        )
        if model.kind == "csi":
            kspace = kspace.isel(spectral_point=self.spectral_point.value())
        kspace_dims = [
            dimension for dimension in kspace.dims if dimension in image_dims
        ]
        kspace_values = np.asarray(kspace.transpose(*kspace_dims))
        metabolite = self._current_metabolite()
        self.image_panel.set_data(
            image_values,
            f"{self.component_combo.currentText()} reconstruction"
            + ("" if metabolite is None else f" · {metabolite}"),
            levels=display_levels,
            lut=self._image_lut(),
            interpolation=self._image_interpolation(),
        )
        kspace_display, kspace_title = self._kspace_display(kspace_values)
        self.kspace_panel.set_data(kspace_display, kspace_title)
        self._voxel = (
            int(np.clip(self._voxel[0], 0, image_values.shape[1] - 1)),
            int(np.clip(self._voxel[1], 0, image_values.shape[0] - 1)),
        )
        self.image_panel.set_selected_pixel(*self._voxel)
        self.pages.setCurrentWidget(self.two_d_page)
        self._current_display = image_values
        self._refresh_phantom_planar()
        self._refresh_voxel_plot(mode, pool_index, effective_coil_mode)

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

    def _kspace_display(self, values):
        magnitude = np.abs(np.asarray(values))
        if self.kspace_scale_combo.currentData() == "log":
            return np.log1p(magnitude), "log(1 + |gridded k-space|)"
        return magnitude, "|gridded k-space|"

    def _image_interpolation(self):
        return str(self.interpolation_combo.currentData() or "nearest")

    @staticmethod
    def _finite_range(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(finite.min()), float(finite.max())
        return None

    def _dataset_contrast_domain(self, current_display, component):
        """Return one contrast domain across every selectable image frame."""
        mode, pool_index = self._selected_data_mode()
        coil_mode = str(self.coil_combo.currentData() or "rss")
        averaged = self._average_dimensions()
        metabolite = self._current_metabolite()
        # Metabolites differ by orders of magnitude, so each gets its own scale.
        cache_key = (mode, pool_index, coil_mode, component, averaged, metabolite)
        if cache_key in self._contrast_domain_cache:
            return self._contrast_domain_cache[cache_key]

        outer_dimensions = tuple(
            item for item in self.model.outer_dimensions if item.name not in averaged
        )
        if not outer_dimensions:
            finite_range = self._finite_range(current_display)
            domain = None if finite_range is None else (0.0, max(0.0, finite_range[1]))
            self._contrast_domain_cache[cache_key] = domain
            return domain

        selections = self._selections()
        high = None

        def browsed_values(item):
            if item.name != "repetition" or metabolite is None:
                return item.values
            return [
                value
                for value in item.values
                if self._repetition_labels.get(value) == metabolite
            ]

        for values in product(*(browsed_values(item) for item in outer_dimensions)):
            frame_selections = dict(selections)
            frame_selections.update(
                (item.name, value) for item, value in zip(outer_dimensions, values)
            )
            try:
                image_data, *_ = self._selected_image(frame_selections)
            except ValueError:
                # Some imported datasets contain sparse combinations of labelled
                # frame coordinates. Those absent combinations are not images.
                continue
            display = self.model.display_values(image_data, component)
            finite_range = self._finite_range(display)
            if finite_range is None:
                continue
            high = finite_range[1] if high is None else max(high, finite_range[1])
        domain = None if high is None else (0.0, max(0.0, high))
        self._contrast_domain_cache[cache_key] = domain
        return domain

    def _display_levels(self, values, *, domain=None):
        if self.auto_contrast.isChecked():
            finite_range = self._finite_range(values)
            display_low, display_high = (
                (0.0, 1.0) if finite_range is None else finite_range
            )
            if display_high <= display_low:
                display_high = np.nextafter(display_low, np.inf)
            self.contrast_slider.set_domain(display_low, display_high, preserve=False)
            self._update_contrast_labels(display_low, display_high)
            return display_low, display_high

        finite_range = self._finite_range(values) if domain is None else domain
        high = 1.0 if finite_range is None else max(0.0, float(finite_range[1]))
        display_high = 1.1 * high
        if display_high <= 0.0:
            display_high = 1e-6
        metabolite = self._current_metabolite()
        if metabolite != self._window_metabolite:
            # Each metabolite keeps its own window; a new one starts at full range.
            if self._window_metabolite is not None:
                self._metabolite_windows[self._window_metabolite] = (
                    self.contrast_slider.values()
                )
            self._window_metabolite = metabolite
            self.contrast_slider.set_domain(0.0, display_high, preserve=False)
            window = self._metabolite_windows.get(metabolite)
            if window is not None:
                self.contrast_slider.set_values(*window, emit=False)
        self.contrast_slider.set_domain(0.0, display_high, preserve=True)
        selected_low, selected_high = self.contrast_slider.values()
        if selected_high <= selected_low:
            # A much weaker result collapses the kept range; show all of it.
            self.contrast_slider.set_domain(0.0, display_high, preserve=False)
            selected_low, selected_high = self.contrast_slider.values()
        self._update_contrast_labels(selected_low, selected_high)
        if selected_high <= selected_low:
            selected_high = np.nextafter(selected_low, np.inf)
        return selected_low, selected_high

    def _refresh_voxel_plot(self, mode, pool_index, coil_mode):
        plot_mode = self.voxel_plot_combo.currentData()
        self.spectrum_plot.clear()
        if plot_mode is None or self.pages.currentWidget() is not self.two_d_page:
            return
        if plot_mode == "time_course":
            self._plot_time_course(mode, pool_index)
        else:
            self._plot_csi_voxel(plot_mode, mode, pool_index, coil_mode)

    def _plot_csi_voxel(self, plot_mode, mode, pool_index, coil_mode):
        model = self.model
        pool = mode == "pool"
        if plot_mode == "fid":
            name = model.image_name(pool=pool)
            axis_name = "spectral_time_s"
        else:
            name = model.spectrum_name(pool=pool)
            axis_name = "spectral_frequency_hz"
        data = model.select(
            name,
            self._selections(),
            pool_index=pool_index if pool else None,
            coil_mode=coil_mode,
            **self._average_kwargs(),
        )
        x, y = self._voxel
        values = np.asarray(data.isel(phase_x=x, phase_y=y))
        axis = np.asarray(model.dataset.coords.get(axis_name, np.arange(values.size)))
        if plot_mode == "fid":
            axis = axis * 1e3
            self.spectrum_plot.setLabel("bottom", "Time", "ms")
        else:
            self.spectrum_plot.setLabel("bottom", "Frequency", "Hz")
        self.spectrum_plot.setLabel("left", "Signal", "a.u.")
        self.spectrum_plot.plot(axis, np.abs(values), pen="w", name="Magnitude")
        self.spectrum_plot.plot(axis, values.real, pen="g", name="Real")
        self.spectrum_plot.plot(axis, values.imag, pen="r", name="Imaginary")
        title = "FID" if plot_mode == "fid" else "Spectrum"
        self.spectrum_plot.setTitle(f"{title} · voxel x={x}, y={y}")

    def _time_course_planes(self, dimension):
        """Return the displayed image of every repetition, cached per view.

        2D results give planes ordered like the image panel; 3D results give
        scanner-ordered volumes like the volume viewer.
        """
        selections = self._selections()
        component = str(self.component_combo.currentData() or "magnitude")
        spectral_point = (
            self.spectral_point.value() if self.model.kind == "csi" else None
        )
        volumetric = self.model.kind in {"cartesian_3d", "radial_3d"}
        key = (
            self._selected_data_mode(),
            str(self.coil_combo.currentData() or "rss"),
            component,
            spectral_point,
            volumetric,
            tuple(
                sorted(
                    (name, repr(value))
                    for name, value in selections.items()
                    if name != dimension.name
                )
            ),
        )
        cached = self._time_course_cache.get(key)
        if cached is not None:
            return cached
        planes = []
        for value in dimension.values:
            frame_selections = dict(selections)
            frame_selections[dimension.name] = value
            try:
                image_data, *_ = self._selected_image(frame_selections, average=False)
            except ValueError:
                planes.append(None)
                continue
            if spectral_point is not None:
                image_data = image_data.isel(spectral_point=spectral_point)
            display = image_data.copy(
                data=self.model.display_values(image_data, component)
            )
            if volumetric:
                planes.append(np.asarray(self.model.scanner_volume(display)[0]))
            else:
                planes.append(np.asarray(display.transpose(*self._image_dims)))
        self._time_course_cache[key] = planes
        return planes

    @staticmethod
    def _voxel_series(planes, index):
        """Return one voxel of every plane, NaN where it does not exist."""
        values = []
        for plane in planes:
            inside = plane is not None and all(
                0 <= position < size for position, size in zip(index, plane.shape)
            )
            values.append(float(plane[index]) if inside else np.nan)
        return np.asarray(values)

    def _plot_time_course(self, mode=None, pool_index=None, *, plot=None, index=None):
        """Plot the voxel's repetitions, one coloured curve per metabolite."""
        plot = self.spectrum_plot if plot is None else plot
        dimension = self._time_course_dimension()
        if dimension is None:
            return
        planes = self._time_course_planes(dimension)
        if index is None:
            x, y = self._voxel
            index = (y, x)
            location = f"x={x}, y={y}"
        else:
            location = "x={}, y={}, z={}".format(*index)
        values = self._voxel_series(planes, tuple(index))
        times = self.model.outer_dimension_times_s(dimension.name)
        if times is None:
            axis = np.arange(values.size, dtype=float)
            plot.setLabel("bottom", "Repetition")
        else:
            axis = np.asarray(times, dtype=float)
            plot.setLabel("bottom", "Time", "s")
        component = self.component_combo.currentText()
        plot.setLabel("left", component, "a.u.")
        repetitions = list(dimension.values)
        if self._repetition_labels:
            selected = self._metabolite_filter_value()
            for label, color in self._metabolite_colors.items():
                if selected is not None and label != selected:
                    continue
                rows = [
                    position
                    for position, value in enumerate(repetitions)
                    if self._repetition_labels.get(value) == label
                ]
                plot.plot(
                    axis[rows],
                    values[rows],
                    pen=pg.mkPen(color, width=2),
                    symbol="o",
                    symbolSize=5,
                    symbolPen=color,
                    symbolBrush=color,
                    name=label,
                )
        else:
            plot.plot(
                axis,
                values,
                pen="w",
                symbol="o",
                symbolSize=5,
                symbolBrush="w",
                name=component,
            )
        current = self._selections().get(dimension.name)
        if current in repetitions and not self._average_dimensions():
            plot.addItem(
                pg.InfiniteLine(
                    axis[repetitions.index(current)],
                    angle=90,
                    pen=pg.mkPen(255, 0, 255, width=1),
                )
            )
        plot.setTitle(f"Time course · voxel {location}")

    def _refresh_volume_plot(self):
        """Show the time course of the voxel selected in the 3D image."""
        plot = self.volume_plot
        plot.clear()
        available = (
            self.model is not None
            and self._time_course_dimension() is not None
            and self.pages.currentWidget() is self.three_d_page
        )
        plot.setVisible(available)
        if available:
            self._plot_time_course(plot=plot, index=self.image_volume.indices)

    def _volume_voxel_changed(self, *_):
        if self._refreshing or self._updating:
            return
        try:
            self._refresh_volume_plot()
        except Exception:
            self.volume_plot.clear()

    def _set_phantom(self, phantom):
        """Use ``phantom`` as the object behind the shown reconstruction."""
        self._phantom = phantom
        self._phantom_cache.clear()
        self._phantom_grid = None
        self._phantom_contrast = None
        try:
            self._phantom_contrasts = object_contrasts(phantom)
        except Exception:
            self._phantom_contrasts = []
        combo = self.phantom_contrast_combo
        previous_key = combo.currentData() or "auto"
        blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItem("Automatic", "auto")
        combo.addItem("Object density", "density")
        for contrast in self._phantom_contrasts:
            combo.addItem(contrast.title, contrast.key)
        index = combo.findData(previous_key)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(blocked)
        available = phantom is not None
        for widget in (self.phantom_show, self.phantom_contours, combo):
            widget.setEnabled(available)
        self._clear_phantom_views()
        self.phantom_note.setText(
            ""
            if available
            else "No phantom is stored with this result (opened files carry none)."
        )

    def _clear_phantom_views(self):
        self.phantom_panel.clear()
        self.phantom_panel.hide()
        self.image_contours.clear()
        for _viewer, _plane, item in self._volume_contour_items:
            item.clear()
        self.phantom_overview.clear()
        self.phantom_legend.setText("")
        self.phantom_composition.setText("")
        self._set_phantom_tab_visible(False)

    def _set_phantom_tab_visible(self, visible):
        index = self.volume_pages.indexOf(self.phantom_volume)
        if not visible and self.volume_pages.currentIndex() == index:
            self.volume_pages.setCurrentIndex(0)
        self.volume_pages.setTabVisible(index, bool(visible))

    def _phantom_options_changed(self, *_):
        if not self._updating:
            self._refresh()

    def _selected_phantom_contrast(self):
        key = self.phantom_contrast_combo.currentData() or "auto"
        if key == "auto":
            return automatic_contrast(self._phantom_contrasts)
        return next(
            (contrast for contrast in self._phantom_contrasts if contrast.key == key),
            None,
        )

    def _resampled_phantom(self, grid, contrast):
        key = (
            np.asarray(grid.origin_m, dtype=float).tobytes(),
            np.asarray(grid.steps_m, dtype=float).tobytes(),
            tuple(grid.shape),
            bool(grid.planar),
            bool(grid.projection),
            None if contrast is None else contrast.key,
        )
        resampled = self._phantom_cache.get(key)
        if resampled is None:
            resampled = resample_phantom(self._phantom, grid, contrast)
            if len(self._phantom_cache) >= _PHANTOM_CACHE_SIZE:
                self._phantom_cache.pop(next(iter(self._phantom_cache)))
            self._phantom_cache[key] = resampled
        return resampled

    def _phantom_unavailable(self, message):
        self._phantom_grid = None
        self._clear_phantom_views()
        self.phantom_note.setText(message)

    def _refresh_phantom_planar(self):
        self._phantom_grid = None
        self._set_phantom_tab_visible(False)
        if self._phantom is None:
            return
        try:
            grid = planar_image_grid(
                self.model, self._image_dims, self._phantom, self._selections()
            )
            if grid is None:
                self._phantom_unavailable(
                    "The phantom position of this image is not known."
                )
                return
            contrast = self._selected_phantom_contrast()
            resampled = self._resampled_phantom(grid, contrast)
        except Exception as exc:
            self._phantom_unavailable(f"Phantom unavailable for this image: {exc}")
            return
        self.phantom_note.setText("")
        rows, cols = int(grid.shape[0]), int(grid.shape[1])
        if self.phantom_show.isChecked():
            self.phantom_panel.set_rgba(
                resampled_rgba(resampled), (rows, cols), grid_description(grid)
            )
            self.phantom_panel.set_selected_pixel(*self._voxel)
            self.phantom_panel.show()
        else:
            self.phantom_panel.hide()
        if self.phantom_contours.isChecked():
            # Array axis 0 is the image row (y), axis 1 the column (x).
            self.image_contours.set_segments(
                contour_segments(resampled), lambda points: (points[:, 1], points[:, 0])
            )
        else:
            self.image_contours.clear()
        self._phantom_grid = grid
        self._phantom_contrast = contrast
        self._update_phantom_selection()

    def _refresh_phantom_volume(self):
        self._phantom_grid = None
        self.phantom_panel.hide()
        self.image_contours.clear()
        if self._phantom is None:
            return
        try:
            grid = volume_image_grid(self.model)
            if grid is None or tuple(grid.shape) != tuple(
                np.shape(self.image_volume.data)[:3]
            ):
                self._phantom_unavailable(
                    "The phantom position of this volume is not known."
                )
                return
            contrast = self._selected_phantom_contrast()
            resampled = self._resampled_phantom(grid, contrast)
        except Exception as exc:
            self._phantom_unavailable(f"Phantom unavailable for this volume: {exc}")
            return
        self.phantom_note.setText("")
        self._phantom_grid = grid
        self._phantom_contrast = contrast
        show = self.phantom_show.isChecked()
        if show:
            values, levels, lookup = volume_display(resampled)
            title = "object density" if contrast is None else contrast.title
            self.phantom_volume.set_volume(
                values,
                fov_m=self.image_volume.fov_m,
                name=f"Phantom on the image grid · {title}",
                levels=levels,
                lookup_table=lookup,
            )
            self._syncing_volumes = True
            try:
                self.phantom_volume.set_indices(self.image_volume.indices)
            finally:
                self._syncing_volumes = False
        self._set_phantom_tab_visible(show)
        self._update_volume_contours()
        self._update_phantom_selection()

    def _volume_indices_changed(self, source, indices):
        if self._syncing_volumes:
            return
        grid = self._phantom_grid
        if grid is None or grid.planar:
            return
        target = (
            self.phantom_volume if source is self.image_volume else self.image_volume
        )
        if np.shape(target.data) == np.shape(source.data):
            self._syncing_volumes = True
            try:
                target.set_indices(indices)
            finally:
                self._syncing_volumes = False
        self._update_volume_contours()
        self._update_phantom_selection()

    def _update_volume_contours(self):
        grid = self._phantom_grid
        for viewer, plane, item in self._volume_contour_items:
            if viewer is self.phantom_volume:
                visible = self.phantom_show.isChecked()
            else:
                visible = self.phantom_contours.isChecked()
            if grid is None or grid.planar or not visible:
                item.clear()
                continue
            horizontal, vertical, normal = VOLUME_PLANES[plane]
            try:
                section = volume_section_grid(grid, plane, viewer.indices[normal])
                segments = contour_segments(
                    self._resampled_phantom(section, self._phantom_contrast)
                )
            except Exception:
                item.clear()
                continue
            width = float(viewer.fov_m[horizontal]) * 1000.0
            height = float(viewer.fov_m[vertical]) * 1000.0
            columns, rows = section.shape[0], section.shape[1]
            item.set_segments(
                segments,
                lambda points, w=width, h=height, n=columns, m=rows: (
                    -w / 2.0 + points[:, 0] * w / n,
                    -h / 2.0 + points[:, 1] * h / m,
                ),
            )

    def _update_phantom_selection(self):
        grid = self._phantom_grid
        if grid is None or self._phantom is None:
            self.phantom_overview.clear()
            self.phantom_legend.setText("")
            self.phantom_composition.setText("")
            return
        if grid.planar:
            index = (self._voxel[1], self._voxel[0], 0)
            position = f"x={self._voxel[0]}, y={self._voxel[1]}"
        else:
            index = self.image_volume.indices
            position = "(" + ", ".join(str(int(value)) for value in index) + ")"
        index = np.clip(np.asarray(index, dtype=int), 0, np.asarray(grid.shape) - 1)
        box = grid.voxel_box(index)
        contrast = self._phantom_contrast
        try:
            self.phantom_overview.set_view(self._phantom, contrast, grid.box(), box)
            composition = voxel_composition(
                self._phantom, box, self._phantom_contrasts, contrast
            )
        except Exception as exc:
            self.phantom_composition.setText("")
            self.phantom_note.setText(f"Voxel composition unavailable: {exc}")
            return
        center = np.asarray(box.center_m, dtype=float) * 1000.0
        self.phantom_composition.setText(
            f"<b>Selected voxel</b> {position} · centre "
            f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm<br>"
            + composition.html()
        )
        self.phantom_legend.setText(legend_html(contrast))

    def _voxel_selected(self, x: int, y: int):
        self._voxel = (int(x), int(y))
        self.image_panel.set_selected_pixel(x, y)
        if self.phantom_panel.isVisible():
            self.phantom_panel.set_selected_pixel(x, y)
        self._update_phantom_selection()
        if self.model is not None:
            mode, pool_index = self._selected_data_mode()
            self._refresh_voxel_plot(mode, pool_index, self._effective_coil_mode(mode))

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
            "average_repetitions": self.average_repetitions.isChecked(),
            "metabolite_filter": self.metabolite_filter.currentData(),
            "pool": list(self._selected_data_mode()),
            "coil": self.coil_combo.currentData(),
            "component": self.component_combo.currentData(),
            "kspace_scale": self.kspace_scale_combo.currentData(),
            "auto_contrast": self.auto_contrast.isChecked(),
            "contrast_range": list(self.contrast_slider.values()),
            "colormap": self.colormap_combo.currentText(),
            "display_strength": self.display_strength.value(),
            "interpolation": self._image_interpolation(),
            "spectral_point": self.spectral_point.value(),
            "voxel_plot": self.voxel_plot_combo.currentData(),
            "voxel": list(self._voxel),
            "image_volume_indices": list(self.image_volume.indices),
            "kspace_volume_indices": list(self.kspace_volume.indices),
            "volume_page": self.volume_pages.currentIndex(),
            "volume_tabs": [
                self.image_volume.tabs.currentIndex(),
                self.kspace_volume.tabs.currentIndex(),
            ],
            "phantom": {
                "show": self.phantom_show.isChecked(),
                "contours": self.phantom_contours.isChecked(),
                "contrast": self.phantom_contrast_combo.currentData(),
            },
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            return
        if self.model is None:
            self._pending_state = dict(state)
            return
        self._updating = True
        try:
            self._set_combo_data(self.metabolite_filter, state.get("metabolite_filter"))
            self._apply_metabolite_filter()
            for name, value in dict(state.get("outer", {})).items():
                self._set_outer_control_value(name, value)
            self.average_repetitions.setChecked(
                bool(state.get("average_repetitions", False))
                and self._time_course_dimension() is not None
            )
            pool = state.get("pool")
            if isinstance(pool, (list, tuple)) and len(pool) == 2:
                self._set_combo_data(self.pool_combo, (str(pool[0]), int(pool[1])))
            self._set_combo_data(self.coil_combo, state.get("coil"))
            self._set_combo_data(self.component_combo, state.get("component"))
            self._set_combo_data(
                self.kspace_scale_combo, state.get("kspace_scale", "log")
            )
            contrast_range = state.get("contrast_range", ())
            if isinstance(contrast_range, (list, tuple)) and len(contrast_range) == 2:
                # The saved range becomes the domain until the data replaces it.
                self.contrast_slider.set_domain(
                    float(contrast_range[0]), float(contrast_range[1]), preserve=False
                )
                self._update_contrast_labels(*self.contrast_slider.values())
            automatic = bool(state.get("auto_contrast", False))
            self.auto_contrast.setChecked(automatic)
            self.contrast_slider.setEnabled(not automatic)
            self.colormap_combo.setCurrentText(str(state.get("colormap", "gray")))
            self.display_strength.setValue(float(state.get("display_strength", 1.0)))
            self._set_combo_data(
                self.interpolation_combo, state.get("interpolation", "nearest")
            )
            self.spectral_point.setValue(int(state.get("spectral_point", 0)))
            self._set_combo_data(self.voxel_plot_combo, state.get("voxel_plot"))
            voxel = state.get("voxel", (0, 0))
            if isinstance(voxel, (list, tuple)) and len(voxel) == 2:
                self._voxel = (int(voxel[0]), int(voxel[1]))
            phantom = state.get("phantom", {})
            if isinstance(phantom, dict):
                self.phantom_show.setChecked(bool(phantom.get("show", False)))
                self.phantom_contours.setChecked(bool(phantom.get("contours", False)))
                self._set_combo_data(
                    self.phantom_contrast_combo, phantom.get("contrast", "auto")
                )
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
