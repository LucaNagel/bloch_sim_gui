"""Editor for the atlas-based mouse perfusion phantom."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..mouse_phantom import (
    INJECTION_COMPOUNDS,
    MousePerfusionConfig,
    MousePerfusionPhantom,
    build_reference_mouse_anatomy,
)
from .default_settings import WorkspaceDefaults
from .volume_viewer import PhantomInspectorWidget
from .widgets import ExplicitAcceptDialog


class ClickableImageItem(pg.ImageItem):
    """Image item that reports reliable item-local click coordinates."""

    clicked = pyqtSignal(float, float)

    def mouseClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            position = event.pos()
            self.clicked.emit(float(position.x()), float(position.y()))
            event.accept()
            return
        super().mouseClickEvent(event)


class MouseSlicePreviewWidget(QWidget):
    """Three orthogonal anatomy slices with an optional perfusion overlay."""

    voxel_clicked = pyqtSignal(tuple)

    _ANATOMY_COLORS = np.asarray(
        [
            (38, 42, 48, 255),
            (116, 116, 126, 255),
            (218, 132, 91, 255),
            (150, 78, 60, 255),
            (226, 145, 98, 255),
            (157, 83, 64, 255),
            (169, 108, 45, 255),
            (80, 135, 170, 255),
            (210, 73, 80, 255),
            (240, 55, 55, 255),
            (66, 105, 220, 255),
            (202, 174, 126, 255),
            (236, 214, 92, 255),
            (214, 186, 64, 255),
            (214, 186, 64, 255),
        ],
        dtype=np.ubyte,
    )

    def __init__(self, *, show_anatomy=True, parent=None):
        super().__init__(parent)
        self.show_anatomy = bool(show_anatomy)
        self.labels = np.zeros((1, 1, 1), dtype=np.uint8)
        self.perfusion = np.zeros((1, 1, 1), dtype=float)
        self.precomputed_anatomy_slices = None
        self.selected_voxel = None
        self.fov_m = None
        self.display_extent_m = None
        self.display_center_m = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = pg.GraphicsLayoutWidget()
        layout.addWidget(self.canvas, 1)
        self.items = []
        self.plots = []
        self.selection_items = []
        for column, title in enumerate(("Axial", "Coronal", "Sagittal")):
            plot = self.canvas.addPlot(row=0, col=column, title=title)
            plot.setAspectLocked(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            anatomy_item = ClickableImageItem()
            anatomy_item.setLookupTable(self._ANATOMY_COLORS)
            perfusion_item = pg.ImageItem()
            overlay_lut = pg.colormap.get("plasma").getLookupTable(0.0, 1.0, 256)
            overlay_lut = np.asarray(overlay_lut, dtype=np.ubyte)
            alpha = np.concatenate(
                (
                    np.zeros(1, dtype=np.ubyte),
                    np.linspace(70, 235, overlay_lut.shape[0] - 1, dtype=np.ubyte),
                )
            )
            if overlay_lut.shape[1] == 3:
                overlay_lut = np.column_stack((overlay_lut, alpha))
            else:
                overlay_lut[:, 3] = alpha
            perfusion_item.setLookupTable(overlay_lut)
            plot.addItem(anatomy_item)
            plot.addItem(perfusion_item)
            selection_item = pg.ScatterPlotItem(
                size=12,
                pen=pg.mkPen("#ffffff", width=2),
                brush=pg.mkBrush(0, 0, 0, 0),
                symbol="o",
            )
            selection_item.setZValue(30)
            plot.addItem(selection_item)
            self.plots.append(plot)
            self.items.append((anatomy_item, perfusion_item))
            self.selection_items.append(selection_item)
            anatomy_item.clicked.connect(
                lambda image_x, image_y, orientation=column: self._image_clicked(
                    orientation, image_x, image_y
                )
            )
        self.fov_items = []
        for plot in self.plots:
            item = pg.PlotCurveItem(pen=pg.mkPen("#f7f7f7", width=2))
            item.setZValue(20)
            plot.addItem(item)
            self.fov_items.append(item)

    @staticmethod
    def _slices(volume):
        nx, ny, nz = volume.shape
        return (
            np.asarray(volume[:, :, nz // 2]).T,
            np.asarray(volume[:, ny // 2, :]).T,
            np.asarray(volume[nx // 2, :, :]).T,
        )

    def set_anatomy(self, labels):
        values = np.asarray(labels, dtype=np.uint8)
        if values.ndim != 3:
            raise ValueError("mouse anatomy preview requires a 3D label volume")
        self.labels = values
        self.precomputed_anatomy_slices = None
        if self.perfusion.shape != values.shape:
            self.perfusion = np.zeros(values.shape, dtype=float)
        self._render()

    def set_anatomy_slices(self, axial, coronal, sagittal):
        """Display already-rendered high-resolution orthogonal label slices."""

        slices = tuple(
            np.asarray(value, dtype=np.uint8)
            for value in (
                axial,
                coronal,
                sagittal,
            )
        )
        if any(value.ndim != 2 for value in slices):
            raise ValueError("anatomy preview slices must be two-dimensional")
        self.precomputed_anatomy_slices = slices
        self._render()

    def set_perfusion(self, values):
        data = np.asarray(values, dtype=float)
        if data.shape != self.labels.shape:
            raise ValueError("perfusion preview must match anatomy shape")
        self.perfusion = data
        self._render()

    def set_display_geometry(self, fov_m, display_extent_m, display_center_m=None):
        """Set physical FOV and full-mouse preview extents."""

        self.fov_m = np.asarray(fov_m, dtype=float)
        self.display_extent_m = np.asarray(display_extent_m, dtype=float)
        self.display_center_m = np.asarray(
            (0.0, 0.0, 0.0) if display_center_m is None else display_center_m,
            dtype=float,
        )
        self._render()

    @staticmethod
    def _resize_nearest(values, target_shape):
        """Resize a 2D preview with nearest-neighbour sampling."""

        values = np.asarray(values)
        if values.shape == tuple(target_shape):
            return values
        indices = tuple(
            np.minimum(
                (np.arange(target, dtype=float) * source / target).astype(int),
                source - 1,
            )
            for source, target in zip(values.shape, target_shape)
        )
        return values[np.ix_(*indices)]

    def _image_clicked(self, orientation, image_x, image_y):
        voxel = self._voxel_for_image_position(orientation, image_x, image_y)
        if voxel is not None:
            self.set_selected_voxel(voxel)
            self.voxel_clicked.emit(voxel)

    @staticmethod
    def _oriented_extents(extents):
        x, y, z = np.asarray(extents, dtype=float)
        return ((y, x), (z, x), (z, y))

    @staticmethod
    def _oriented_centers(centers):
        x, y, z = np.asarray(centers, dtype=float)
        return ((y, x), (z, x), (z, y))

    def _perfusion_on_display_grid(self, values, orientation, target_shape):
        """Place a simulated-FOV slice into the full-mouse preview grid."""

        if (
            self.fov_m is None
            or self.display_extent_m is None
            or self.display_center_m is None
        ):
            return self._resize_nearest(values, target_shape)
        source_extents = self._oriented_extents(self.fov_m)[orientation]
        target_extents = self._oriented_extents(self.display_extent_m)[orientation]
        target_centers = self._oriented_centers(self.display_center_m)[orientation]
        source = np.asarray(values)
        result = np.zeros(target_shape, dtype=source.dtype)
        target_axes = tuple(
            center
            + (np.arange(count, dtype=float) + 0.5) / count * extent
            - 0.5 * extent
            for count, extent, center in zip(
                target_shape, target_extents, target_centers
            )
        )
        source_indices = []
        valid_axes = []
        for coordinates, source_count, source_extent in zip(
            target_axes, source.shape, source_extents
        ):
            valid = np.abs(coordinates) <= 0.5 * source_extent
            index = np.floor((coordinates / source_extent + 0.5) * source_count).astype(
                int
            )
            source_indices.append(np.clip(index, 0, source_count - 1))
            valid_axes.append(valid)
        sampled = source[np.ix_(*source_indices)]
        valid = np.logical_and.outer(*valid_axes)
        result[valid] = sampled[valid]
        return result

    def _update_fov_boxes(self, anatomy_slices):
        if self.fov_m is None or self.display_extent_m is None:
            for item in self.fov_items:
                item.setData([], [])
            return
        for item, fov_extent in zip(
            self.fov_items,
            self._oriented_extents(self.fov_m),
        ):
            left, right = -0.5 * fov_extent[0], 0.5 * fov_extent[0]
            bottom, top = -0.5 * fov_extent[1], 0.5 * fov_extent[1]
            item.setData(
                [left, right, right, left, left],
                [bottom, bottom, top, top, bottom],
            )

    def _voxel_for_image_position(self, orientation, image_x, image_y):
        """Map an orthogonal preview position to a simulation voxel."""

        nx, ny, nz = self.labels.shape
        anatomy_slices = (
            self._slices(self.labels)
            if self.precomputed_anatomy_slices is None
            else self.precomputed_anatomy_slices
        )
        height, width = anatomy_slices[orientation].shape
        first = int(np.floor(image_x * height / max(height, 1)))
        second = int(np.floor(image_y * width / max(width, 1)))
        if not (0 <= first < height and 0 <= second < width):
            return None
        source_shapes = self._slices(self.labels)
        source_height, source_width = source_shapes[orientation].shape
        if (
            self.fov_m is not None
            and self.display_extent_m is not None
            and self.display_center_m is not None
        ):
            fov_extent = self._oriented_extents(self.fov_m)[orientation]
            display_extent = self._oriented_extents(self.display_extent_m)[orientation]
            display_center = self._oriented_centers(self.display_center_m)[orientation]
            physical_first = (
                display_center[0]
                + (first + 0.5) / height * display_extent[0]
                - 0.5 * display_extent[0]
            )
            physical_second = (
                display_center[1]
                + (second + 0.5) / width * display_extent[1]
                - 0.5 * display_extent[1]
            )
            if (
                abs(physical_first) > 0.5 * fov_extent[0]
                or abs(physical_second) > 0.5 * fov_extent[1]
            ):
                return None
            first = min(
                int((physical_first / fov_extent[0] + 0.5) * source_height),
                source_height - 1,
            )
            second = min(
                int((physical_second / fov_extent[1] + 0.5) * source_width),
                source_width - 1,
            )
        elif self.precomputed_anatomy_slices is not None:
            first = min(int(first * source_height / height), source_height - 1)
            second = min(int(second * source_width / width), source_width - 1)
        if orientation == 0:
            return (second, first, nz // 2)
        if orientation == 1:
            return (second, ny // 2, first)
        return (nx // 2, second, first)

    def set_selected_voxel(self, voxel):
        self.selected_voxel = None if voxel is None else tuple(int(v) for v in voxel)
        if self.selected_voxel is None:
            for marker in self.selection_items:
                marker.setData([], [])
            return
        x, y, z = self.selected_voxel
        points = ((y, x), (z, x), (z, y))
        source_shapes = self._slices(self.labels)
        anatomy_slices = (
            source_shapes
            if self.precomputed_anatomy_slices is None
            else self.precomputed_anatomy_slices
        )
        for orientation, (marker, point, source, displayed) in enumerate(
            zip(self.selection_items, points, source_shapes, anatomy_slices)
        ):
            if self.fov_m is not None:
                fov_extent = self._oriented_extents(self.fov_m)[orientation]
                display_first = ((point[0] + 0.5) / source.shape[0] - 0.5) * fov_extent[
                    0
                ]
                display_second = (
                    (point[1] + 0.5) / source.shape[1] - 0.5
                ) * fov_extent[1]
            else:
                display_first = (point[0] + 0.5) * (
                    displayed.shape[0] / source.shape[0]
                )
                display_second = (point[1] + 0.5) * (
                    displayed.shape[1] / source.shape[1]
                )
            marker.setData(
                pos=np.asarray(
                    [[float(display_first), float(display_second)]], dtype=float
                )
            )

    def _render(self):
        anatomy_slices = (
            self._slices(self.labels)
            if self.precomputed_anatomy_slices is None
            else self.precomputed_anatomy_slices
        )
        perfusion_slices = self._slices(self.perfusion)
        high = float(np.nanmax(self.perfusion)) if self.perfusion.size else 0.0
        high = max(high, np.finfo(float).eps)
        for orientation, (
            (anatomy_item, perfusion_item),
            anatomy,
            perfusion,
        ) in enumerate(zip(self.items, anatomy_slices, perfusion_slices)):
            perfusion = self._perfusion_on_display_grid(
                perfusion, orientation, anatomy.shape
            )
            base = anatomy if self.show_anatomy else np.zeros_like(anatomy)
            anatomy_item.setImage(base, autoLevels=False, levels=(0, 14))
            perfusion_item.setImage(
                np.nan_to_num(perfusion), autoLevels=False, levels=(0.0, high)
            )
            if self.display_extent_m is not None and self.display_center_m is not None:
                extent = self._oriented_extents(self.display_extent_m)[orientation]
                center = self._oriented_centers(self.display_center_m)[orientation]
                rect = QRectF(
                    center[0] - 0.5 * extent[0],
                    center[1] - 0.5 * extent[1],
                    extent[0],
                    extent[1],
                )
                anatomy_item.setRect(rect)
                perfusion_item.setRect(rect)
        self._update_fov_boxes(anatomy_slices)
        if self.selected_voxel is not None:
            self.set_selected_voxel(self.selected_voxel)


class MousePhantomDesignerDialog(ExplicitAcceptDialog):
    """Configure anatomy, injection, renal kinetics, pH, and breathing."""

    def __init__(
        self,
        parent=None,
        config: Optional[MousePerfusionConfig] = None,
        settings=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Mouse Perfusion Phantom"
            if config is not None
            else "New Mouse Perfusion Phantom"
        )
        self.resize(1250, 850)
        if config is None:
            defaults = WorkspaceDefaults.from_settings(settings)
            config = MousePerfusionConfig(field_strength_t=defaults.field_strength_t)
        self.config = MousePerfusionConfig.from_dict(config.to_dict())
        self.phantom = None
        self._selected_voxel = None
        self._animation_bounds_s = (0.0, 1.0)
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(50)
        self._animation_timer.timeout.connect(self._advance_animation)
        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setSingleShot(True)
        self._live_preview_timer.setInterval(300)
        self._live_preview_timer.timeout.connect(self._preview)
        self._build_ui()
        self._load_config()
        self._connect_live_preview_controls()
        self._preview()

    @staticmethod
    def _float_spin(
        minimum,
        maximum,
        *,
        decimals=4,
        suffix="",
        step=None,
    ):
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setSuffix(suffix)
        if step is not None:
            spin.setSingleStep(float(step))
        return spin

    @staticmethod
    def _scrollable_form(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(widget)
        return scroll

    @staticmethod
    def _anatomy_resolution_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("HD anatomy preview (8× slices)", "hd")
        combo.addItem("Simulated resolution", "simulated")
        return combo

    @classmethod
    def _animation_speed_spin(cls) -> QDoubleSpinBox:
        spin = cls._float_spin(0.01, 20.0, decimals=2, suffix=" s/s", step=0.1)
        spin.setValue(0.5)
        spin.setToolTip("Simulated seconds advanced per real second")
        return spin

    def _build_ui(self):
        root = QVBoxLayout(self)
        intro = QLabel(
            "Atlas-based 3D mouse with brain, liver, bilateral renal cortex/"
            "medulla, a directed major-vessel scaffold and dose-normalized "
            "voxelwise bolus arrival. The animation shows the selected compound "
            "accumulating and clearing inside the anatomy."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        anatomy_page = QWidget()
        anatomy_layout = QVBoxLayout(anatomy_page)
        anatomy_form = QFormLayout()
        self.name_edit = QLineEdit()
        anatomy_form.addRow("Name", self.name_edit)
        matrix_widget = QWidget()
        matrix_layout = QGridLayout(matrix_widget)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        self.matrix_spins = []
        for axis, label in enumerate("XYZ"):
            matrix_layout.addWidget(QLabel(label), 0, 2 * axis)
            spin = QSpinBox()
            spin.setRange(8, 256)
            spin.setSingleStep(4)
            self.matrix_spins.append(spin)
            matrix_layout.addWidget(spin, 0, 2 * axis + 1)
            matrix_layout.setColumnStretch(2 * axis + 1, 1)
        anatomy_form.addRow("Matrix", matrix_widget)
        fov_widget = QWidget()
        fov_layout = QGridLayout(fov_widget)
        fov_layout.setContentsMargins(0, 0, 0, 0)
        self.fov_spins = []
        for axis, label in enumerate("XYZ"):
            fov_layout.addWidget(QLabel(label), 0, 2 * axis)
            spin = self._float_spin(5.0, 500.0, decimals=2, suffix=" mm", step=1.0)
            self.fov_spins.append(spin)
            fov_layout.addWidget(spin, 0, 2 * axis + 1)
            fov_layout.setColumnStretch(2 * axis + 1, 1)
        anatomy_form.addRow("FOV", fov_widget)
        mouse_size_widget = QWidget()
        mouse_size_layout = QGridLayout(mouse_size_widget)
        mouse_size_layout.setContentsMargins(0, 0, 0, 0)
        self.mouse_size_spins = []
        for axis, label in enumerate("XYZ"):
            mouse_size_layout.addWidget(QLabel(label), 0, 2 * axis)
            spin = self._float_spin(5.0, 500.0, decimals=2, suffix=" mm", step=1.0)
            self.mouse_size_spins.append(spin)
            mouse_size_layout.addWidget(spin, 0, 2 * axis + 1)
            mouse_size_layout.setColumnStretch(2 * axis + 1, 1)
        anatomy_form.addRow("Mouse size", mouse_size_widget)
        mouse_position_widget = QWidget()
        mouse_position_layout = QGridLayout(mouse_position_widget)
        mouse_position_layout.setContentsMargins(0, 0, 0, 0)
        self.mouse_offset_spins = []
        for axis, label in enumerate("XYZ"):
            mouse_position_layout.addWidget(QLabel(label), 0, 2 * axis)
            spin = self._float_spin(-500.0, 500.0, decimals=2, suffix=" mm", step=0.5)
            spin.setToolTip(f"Move the mouse along {label} relative to the FOV center.")
            self.mouse_offset_spins.append(spin)
            mouse_position_layout.addWidget(spin, 0, 2 * axis + 1)
            mouse_position_layout.setColumnStretch(2 * axis + 1, 1)
        anatomy_form.addRow("Mouse position", mouse_position_widget)
        self.field_strength = self._float_spin(
            0.001, 1000.0, decimals=3, suffix=" T", step=0.1
        )
        anatomy_form.addRow("B0", self.field_strength)
        self.anatomy_resolution = self._anatomy_resolution_combo()
        self.anatomy_resolution.currentIndexChanged.connect(
            self._anatomy_resolution_changed
        )
        anatomy_form.addRow("Anatomy view", self.anatomy_resolution)
        anatomy_layout.addLayout(anatomy_form)
        anatomy_help = QLabel(
            "The current built-in atlas is procedural and deterministic. It "
            "contains soft tissue, brain, liver, lungs, heart, bilateral renal "
            "cortex/medulla and explicitly voxelized arterial/venous paths. HD "
            "changes only this view; simulation uses the matrix above. The complete "
            "mouse remains visible, while the white outline marks the simulated "
            "FOV. Mouse size and position are physical millimeter values and do "
            "not depend on the simulation matrix."
        )
        anatomy_help.setWordWrap(True)
        anatomy_layout.addWidget(anatomy_help)
        self.anatomy_preview = MouseSlicePreviewWidget(show_anatomy=True)
        anatomy_layout.addWidget(self.anatomy_preview, 1)
        self.tabs.addTab(anatomy_page, "Anatomy")

        injection_page = QWidget()
        injection_layout = QHBoxLayout(injection_page)
        injection_controls = QWidget()
        injection_controls_layout = QVBoxLayout(injection_controls)
        injection_controls_layout.setContentsMargins(0, 0, 0, 0)
        injection_figures = QWidget()
        injection_figures_layout = QVBoxLayout(injection_figures)
        injection_figures_layout.setContentsMargins(0, 0, 0, 0)
        injection_controls.setMinimumWidth(410)
        injection_controls.setMaximumWidth(540)
        injection_layout.addWidget(injection_controls, 0)
        injection_layout.addWidget(injection_figures, 1)
        injection_form = QFormLayout()
        self.injection_compound = QComboBox()
        for key, label in INJECTION_COMPOUNDS.items():
            self.injection_compound.addItem(label, key)
        self.injection_compound.currentIndexChanged.connect(
            self._compound_selection_changed
        )
        self.injection_amount = self._float_spin(
            0.0, 1e6, decimals=6, suffix=" µmol", step=0.05
        )
        self.injection_start = self._float_spin(
            -10000.0, 10000.0, decimals=4, suffix=" s", step=0.1
        )
        self.injection_duration = self._float_spin(
            0.001, 10000.0, decimals=4, suffix=" s", step=0.1
        )
        self.circulation_delay = self._float_spin(
            0.0, 10000.0, decimals=4, suffix=" s", step=0.05
        )
        self.perfusion_timestep = self._float_spin(
            0.0001, 100.0, decimals=4, suffix=" s", step=0.005
        )
        self.injection_polarization = self._float_spin(
            0.000001, 1e9, decimals=3, suffix=" × thermal", step=100.0
        )
        self.bolus_alpha = self._float_spin(0.05, 100.0, decimals=3, step=0.1)
        self.bolus_beta = self._float_spin(0.05, 100.0, decimals=3, step=0.1)
        self.bolus_alpha.setToolTip(
            "Controls the rising side of the normalized bolus profile "
            "u^alpha (1-u)^beta. Larger alpha shifts the peak later."
        )
        self.bolus_beta.setToolTip(
            "Controls the falling side of the normalized bolus profile "
            "u^alpha (1-u)^beta. Larger beta shifts the peak earlier and "
            "shortens the tail."
        )
        self.tissue_clearance = self._float_spin(
            0.0001, 100.0, decimals=4, suffix=" s⁻¹", step=0.01
        )
        self.renal_excretion = self._float_spin(
            0.0, 100.0, decimals=4, suffix=" s⁻¹", step=0.01
        )
        self.renal_excretion.setToolTip(
            "Rate at which kidney tissue passes injected compound into the "
            "urine, in addition to tissue clearance. 0 disables excretion. "
            "The Bloch solver removes the excreted compound from the kidneys "
            "and feeds it, T1-relaxed, into the ureters and bladder."
        )
        self.renal_tubular_transit = self._float_spin(
            0.0, 10000.0, decimals=2, suffix=" s", step=5.0
        )
        self.renal_tubular_transit.setToolTip(
            "Mean nephron/pelvis transit; individual transit times are spread "
            "uniformly over 0.5–1.5× this value. Rodent tubular fluid reaches "
            "the end of the proximal tubule after ~10–15 s and the urine after "
            "roughly 1–2 min. Hyperpolarization relaxes with T1 during "
            "transit; tubular content is shown in the preview only."
        )
        self.ureter_transit = self._float_spin(
            0.0, 10000.0, decimals=2, suffix=" s", step=1.0
        )
        self.ureter_transit.setToolTip(
            "Mean residence of urine in each well-mixed ureter before it "
            "drains into the bladder. 0 passes urine straight to the bladder."
        )
        injection_form.addRow("Injected compound", self.injection_compound)
        injection_form.addRow("Injected amount", self.injection_amount)
        injection_form.addRow("Injection starts", self.injection_start)
        injection_form.addRow("Bolus duration", self.injection_duration)
        injection_form.addRow(
            "Representative circulation delay", self.circulation_delay
        )
        injection_form.addRow("Perfusion time step", self.perfusion_timestep)
        injection_form.addRow("Inflow polarization", self.injection_polarization)
        injection_form.addRow("Bolus alpha", self.bolus_alpha)
        injection_form.addRow("Bolus beta", self.bolus_beta)
        injection_form.addRow("Preview tissue clearance", self.tissue_clearance)
        injection_form.addRow("Renal excretion to urine", self.renal_excretion)
        injection_form.addRow("Renal tubular transit", self.renal_tubular_transit)
        injection_form.addRow("Ureter transit", self.ureter_transit)
        injection_controls_layout.addLayout(injection_form)

        perfusion_group = QGroupBox("Relative delivery weights")
        perfusion_form = QFormLayout(perfusion_group)
        self.background_perfusion = self._float_spin(0.0, 1e6, decimals=4)
        self.cortex_perfusion = self._float_spin(0.0, 1e6, decimals=4)
        self.medulla_perfusion = self._float_spin(0.0, 1e6, decimals=4)
        self.liver_perfusion = self._float_spin(0.0, 1e6, decimals=4)
        self.brain_perfusion = self._float_spin(0.0, 1e6, decimals=4)
        self.vessel_perfusion = self._float_spin(0.0, 1e6, decimals=4)
        perfusion_form.addRow("Other perfused tissue", self.background_perfusion)
        perfusion_form.addRow("Renal cortex", self.cortex_perfusion)
        perfusion_form.addRow("Renal medulla", self.medulla_perfusion)
        perfusion_form.addRow("Liver", self.liver_perfusion)
        perfusion_form.addRow("Brain", self.brain_perfusion)
        perfusion_form.addRow("Resolved vessels", self.vessel_perfusion)
        injection_controls_layout.addWidget(perfusion_group)
        injection_help = QLabel(
            "The delivery map is normalized by physical voxel volume, so the "
            "space-time integral of the source equals the entered amount. "
            "Absolute receive signal still requires scanner/coil calibration."
        )
        injection_help.setWordWrap(True)
        injection_controls_layout.addWidget(injection_help)
        self.compound_help = QLabel()
        self.compound_help.setWordWrap(True)
        injection_controls_layout.addWidget(self.compound_help)
        injection_controls_layout.addStretch(1)
        perfusion_animation_row = QHBoxLayout()
        self.perfusion_play_button = QPushButton("Play")
        self.perfusion_play_button.setCheckable(True)
        self.perfusion_play_button.toggled.connect(
            lambda checked: self.play_button.setChecked(checked)
        )
        perfusion_animation_row.addWidget(self.perfusion_play_button)
        perfusion_animation_row.addWidget(QLabel("Perfusion animation time"))
        self.perfusion_animation_time = self._float_spin(
            -10000.0, 10000.0, decimals=3, suffix=" s", step=0.05
        )
        self.perfusion_animation_time.valueChanged.connect(
            lambda value: self.animation_time.setValue(value)
        )
        perfusion_animation_row.addWidget(self.perfusion_animation_time)
        perfusion_animation_row.addWidget(QLabel("Speed"))
        self.perfusion_animation_speed = self._animation_speed_spin()
        perfusion_animation_row.addWidget(self.perfusion_animation_speed)
        self.perfusion_animation_slider = QSlider(Qt.Horizontal)
        self.perfusion_animation_slider.setRange(0, 1000)
        self.perfusion_animation_slider.valueChanged.connect(
            lambda value: self.animation_slider.setValue(value)
        )
        perfusion_animation_row.addWidget(self.perfusion_animation_slider, 1)
        curve_row = QHBoxLayout()
        self.source_curve_plot = pg.PlotWidget(title="Injected bolus")
        self.source_curve_plot.setLabel("bottom", "Time", "s")
        self.source_curve_plot.setLabel("left", "Rate", "µmol/s")
        self.organ_curve_plot = pg.PlotWidget(title="Organ concentration")
        self.organ_curve_plot.setLabel("bottom", "Time", "s")
        self.organ_curve_plot.setLabel("left", "Mean concentration", "mM")
        self.organ_curve_plot.addLegend()
        self.source_curve_plot.setMinimumHeight(150)
        self.organ_curve_plot.setMinimumHeight(150)
        curve_row.addWidget(self.source_curve_plot, 1)
        curve_row.addWidget(self.organ_curve_plot, 1)
        injection_figures_layout.addLayout(curve_row)
        self.perfusion_preview = MouseSlicePreviewWidget(show_anatomy=False)
        injection_figures_layout.addWidget(self.perfusion_preview, 1)
        injection_figures_layout.addLayout(perfusion_animation_row)
        self.tabs.addTab(injection_page, "Injected perfusion")

        kidney_page = QWidget()
        kidney_layout = QVBoxLayout(kidney_page)
        kinetic_group = QGroupBox("Pyruvate → lactate conversion")
        kinetic_grid = QGridLayout(kinetic_group)
        kinetic_grid.setColumnStretch(2, 1)
        kinetic_grid.addWidget(QLabel("Region"), 0, 0)
        kinetic_grid.addWidget(QLabel("kPL"), 0, 1)
        self.kpl_spins = {}
        regions = (
            ("left_cortex", "Left cortex"),
            ("left_medulla", "Left medulla"),
            ("right_cortex", "Right cortex"),
            ("right_medulla", "Right medulla"),
        )
        for row, (key, label) in enumerate(regions, start=1):
            kinetic_grid.addWidget(QLabel(label), row, 0)
            spin = self._float_spin(0.0, 100.0, decimals=5, suffix=" s⁻¹")
            spin.setMaximumWidth(300)
            self.kpl_spins[key] = spin
            kinetic_grid.addWidget(spin, row, 1)
        kidney_layout.addWidget(kinetic_group)

        liver_group = QGroupBox("Liver conversion")
        liver_form = QFormLayout(liver_group)
        self.liver_kpa = self._float_spin(0.0, 100.0, decimals=5, suffix=" s⁻¹")
        self.liver_kpa.setMaximumWidth(300)
        liver_form.addRow("Pyruvate → alanine kPA", self.liver_kpa)
        liver_note = QLabel(
            "The Alanine pool is coupled to Pyruvate through this liver-specific "
            "kPA map. Precursor loss includes both kPL and kPA during sequence "
            "simulation."
        )
        liver_note.setWordWrap(True)
        liver_form.addRow(liver_note)
        kidney_layout.addWidget(liver_group)

        ph_group = QGroupBox("Ground-truth pH regions")
        ph_grid = QGridLayout(ph_group)
        ph_grid.setColumnStretch(2, 1)
        self.ph_spins = {}
        ph_regions = (("background", "Other tissue"),) + regions
        for row, (key, label) in enumerate(ph_regions):
            ph_grid.addWidget(QLabel(label), row, 0)
            spin = self._float_spin(0.0, 14.0, decimals=3, step=0.05)
            spin.setMaximumWidth(300)
            self.ph_spins[key] = spin
            ph_grid.addWidget(spin, row, 1)
        kidney_layout.addWidget(ph_group)
        ph_help = QLabel(
            "pH is stored as ground truth and shown in the preview. It does not "
            "silently shift pyruvate or lactate. A later pH-sensitive compound "
            "model can explicitly connect this map to chemical shift or an "
            "acid/base reaction."
        )
        ph_help.setWordWrap(True)
        kidney_layout.addWidget(ph_help)
        kidney_layout.addStretch(1)
        self.tabs.addTab(self._scrollable_form(kidney_page), "Metabolism & pH")

        motion_page = QWidget()
        motion_layout = QVBoxLayout(motion_page)
        motion_form = QFormLayout()
        self.breathing_enabled = QCheckBox("Enable breathing preview")
        self.breathing_rate = self._float_spin(
            0.01, 100.0, decimals=4, suffix=" Hz", step=0.1
        )
        self.breathing_amplitude = self._float_spin(
            0.0, 100.0, decimals=4, suffix=" mm", step=0.1
        )
        self.breathing_phase = self._float_spin(
            -1e6, 1e6, decimals=4, suffix=" rad", step=0.1
        )
        motion_form.addRow(self.breathing_enabled)
        motion_form.addRow("Rate", self.breathing_rate)
        motion_form.addRow("Peak displacement", self.breathing_amplitude)
        motion_form.addRow("Phase at t=0", self.breathing_phase)
        motion_layout.addLayout(motion_form)
        motion_help = QLabel(
            "The smooth displacement field is available for inspection and is "
            "saved with the phantom. It is not yet applied to gradient phase in "
            "the sequence solver; result metadata reports this explicitly."
        )
        motion_help.setWordWrap(True)
        motion_layout.addWidget(motion_help)
        motion_layout.addStretch(1)
        self.tabs.addTab(motion_page, "Breathing")

        combined_page = QWidget()
        combined_layout = QVBoxLayout(combined_page)
        animation_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._play_toggled)
        animation_row.addWidget(self.play_button)
        animation_row.addWidget(QLabel("Animation time (local bolus concentration)"))
        self.animation_time = self._float_spin(
            -10000.0, 10000.0, decimals=3, suffix=" s", step=0.05
        )
        self.animation_time.valueChanged.connect(self._animation_time_changed)
        animation_row.addWidget(self.animation_time)
        animation_row.addWidget(QLabel("Speed"))
        self.animation_speed = self._animation_speed_spin()
        animation_row.addWidget(self.animation_speed)
        animation_row.addWidget(QLabel("Anatomy view"))
        self.combined_anatomy_resolution = self._anatomy_resolution_combo()
        self.combined_anatomy_resolution.currentIndexChanged.connect(
            self._anatomy_resolution_changed
        )
        animation_row.addWidget(self.combined_anatomy_resolution)
        self.animation_slider = QSlider(Qt.Horizontal)
        self.animation_slider.setRange(0, 1000)
        self.animation_slider.valueChanged.connect(self._animation_slider_changed)
        animation_row.addWidget(self.animation_slider, 1)
        combined_layout.addLayout(animation_row)
        time_help = QLabel(
            "This time selects the locally present injected compound after "
            "voxelwise arrival and tissue clearance—not merely the source peak."
        )
        time_help.setWordWrap(True)
        combined_layout.addWidget(time_help)
        self.combined_preview = MouseSlicePreviewWidget(show_anatomy=True)
        self.combined_preview.voxel_clicked.connect(self._voxel_selected)
        combined_layout.addWidget(self.combined_preview, 1)
        self.voxel_spectrum_label = QLabel(
            "Click an anatomy slice to inspect the local dynamic spectrum."
        )
        combined_layout.addWidget(self.voxel_spectrum_label)
        self.voxel_spectrum_plot = pg.PlotWidget(title="Selected voxel spectrum")
        self.voxel_spectrum_plot.setLabel("bottom", "Chemical shift", "ppm")
        self.voxel_spectrum_plot.setLabel("left", "Relative signal")
        self.voxel_spectrum_plot.setMinimumHeight(170)
        self.voxel_spectrum_plot.addLegend()
        self.voxel_spectrum_plot.getPlotItem().invertX(True)
        combined_layout.addWidget(self.voxel_spectrum_plot)
        self.perfusion_animation_speed.valueChanged.connect(
            self.animation_speed.setValue
        )
        self.animation_speed.valueChanged.connect(
            self.perfusion_animation_speed.setValue
        )
        self.tabs.addTab(combined_page, "Anatomy + perfusion")

        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        self.preview_summary = QLabel()
        self.preview_summary.setWordWrap(True)
        preview_layout.addWidget(self.preview_summary)
        self.inspector = PhantomInspectorWidget()
        preview_layout.addWidget(self.inspector, 1)
        self.tabs.addTab(preview_page, "Maps & spectrum")

        action_row = QHBoxLayout()
        preview_button = QPushButton("Update preview")
        preview_button.clicked.connect(self._preview)
        action_row.addWidget(preview_button)
        action_row.addStretch(1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        action_row.addWidget(self.buttons)
        root.addLayout(action_row)

    def _connect_live_preview_controls(self):
        """Debounce expensive phantom rebuilds while configuration is edited."""

        self.name_edit.textChanged.connect(self._schedule_live_preview)
        self.injection_compound.currentIndexChanged.connect(self._schedule_live_preview)
        controls = [
            *self.matrix_spins,
            *self.fov_spins,
            *self.mouse_size_spins,
            *self.mouse_offset_spins,
            self.field_strength,
            self.injection_amount,
            self.injection_start,
            self.injection_duration,
            self.circulation_delay,
            self.perfusion_timestep,
            self.injection_polarization,
            self.bolus_alpha,
            self.bolus_beta,
            self.tissue_clearance,
            self.renal_excretion,
            self.renal_tubular_transit,
            self.ureter_transit,
            self.background_perfusion,
            self.cortex_perfusion,
            self.medulla_perfusion,
            self.liver_perfusion,
            self.brain_perfusion,
            self.vessel_perfusion,
            *self.kpl_spins.values(),
            self.liver_kpa,
            *self.ph_spins.values(),
            self.breathing_rate,
            self.breathing_amplitude,
            self.breathing_phase,
        ]
        for control in controls:
            control.valueChanged.connect(self._schedule_live_preview)
        self.breathing_enabled.toggled.connect(self._schedule_live_preview)

    def _schedule_live_preview(self, *_):
        self._live_preview_timer.start()

    def _load_config(self):
        config = self.config
        self.name_edit.setText(config.name)
        for spin, value in zip(self.matrix_spins, config.shape):
            spin.setValue(value)
        for spin, value in zip(self.fov_spins, config.fov_m):
            spin.setValue(value * 1000.0)
        for spin, value in zip(self.mouse_size_spins, config.mouse_size_m):
            spin.setValue(value * 1000.0)
        for spin, value in zip(self.mouse_offset_spins, config.mouse_offset_m):
            spin.setValue(value * 1000.0)
        self.anatomy_resolution.setCurrentIndex(self.anatomy_resolution.findData("hd"))
        self.combined_anatomy_resolution.setCurrentIndex(
            self.combined_anatomy_resolution.findData("hd")
        )
        self.field_strength.setValue(config.field_strength_t)
        compound_index = self.injection_compound.findData(config.injection_compound)
        self.injection_compound.setCurrentIndex(max(0, compound_index))
        self.injection_amount.setValue(config.injection_amount_umol)
        self.injection_start.setValue(config.injection_start_s)
        self.injection_duration.setValue(config.injection_duration_s)
        self.circulation_delay.setValue(config.circulation_delay_s)
        self.perfusion_timestep.setValue(config.perfusion_timestep_s)
        self.injection_polarization.setValue(config.injection_polarization)
        self.bolus_alpha.setValue(config.bolus_alpha)
        self.bolus_beta.setValue(config.bolus_beta)
        self.tissue_clearance.setValue(config.tissue_clearance_s_inv)
        self.renal_excretion.setValue(config.renal_excretion_s_inv)
        self.renal_tubular_transit.setValue(config.renal_tubular_transit_s)
        self.ureter_transit.setValue(config.ureter_transit_s)
        self.background_perfusion.setValue(config.background_perfusion_weight)
        self.cortex_perfusion.setValue(config.renal_cortex_perfusion_weight)
        self.medulla_perfusion.setValue(config.renal_medulla_perfusion_weight)
        self.liver_perfusion.setValue(config.liver_perfusion_weight)
        self.brain_perfusion.setValue(config.brain_perfusion_weight)
        self.vessel_perfusion.setValue(config.vessel_perfusion_weight)
        self.kpl_spins["left_cortex"].setValue(config.left_cortex_kpl_s_inv)
        self.kpl_spins["left_medulla"].setValue(config.left_medulla_kpl_s_inv)
        self.kpl_spins["right_cortex"].setValue(config.right_cortex_kpl_s_inv)
        self.kpl_spins["right_medulla"].setValue(config.right_medulla_kpl_s_inv)
        self.liver_kpa.setValue(config.liver_kpa_s_inv)
        self.ph_spins["background"].setValue(config.background_ph)
        self.ph_spins["left_cortex"].setValue(config.left_cortex_ph)
        self.ph_spins["left_medulla"].setValue(config.left_medulla_ph)
        self.ph_spins["right_cortex"].setValue(config.right_cortex_ph)
        self.ph_spins["right_medulla"].setValue(config.right_medulla_ph)
        self.breathing_enabled.setChecked(config.breathing_enabled)
        self.breathing_rate.setValue(config.breathing_rate_hz)
        self.breathing_amplitude.setValue(config.breathing_amplitude_mm)
        self.breathing_phase.setValue(config.breathing_phase_rad)
        self.animation_time.setValue(
            config.injection_start_s - 0.1 * config.injection_duration_s
        )
        self._compound_selection_changed()

    def _read_config(self) -> MousePerfusionConfig:
        values = self.config.to_dict()
        values.update(
            name=self.name_edit.text().strip(),
            shape=tuple(spin.value() for spin in self.matrix_spins),
            fov_m=tuple(spin.value() / 1000.0 for spin in self.fov_spins),
            mouse_size_m=tuple(spin.value() / 1000.0 for spin in self.mouse_size_spins),
            mouse_offset_m=tuple(
                spin.value() / 1000.0 for spin in self.mouse_offset_spins
            ),
            field_strength_t=self.field_strength.value(),
            injection_compound=str(self.injection_compound.currentData()),
            injection_amount_umol=self.injection_amount.value(),
            injection_start_s=self.injection_start.value(),
            injection_duration_s=self.injection_duration.value(),
            circulation_delay_s=self.circulation_delay.value(),
            perfusion_timestep_s=self.perfusion_timestep.value(),
            injection_polarization=self.injection_polarization.value(),
            bolus_alpha=self.bolus_alpha.value(),
            bolus_beta=self.bolus_beta.value(),
            tissue_clearance_s_inv=self.tissue_clearance.value(),
            renal_excretion_s_inv=self.renal_excretion.value(),
            renal_tubular_transit_s=self.renal_tubular_transit.value(),
            ureter_transit_s=self.ureter_transit.value(),
            background_perfusion_weight=self.background_perfusion.value(),
            renal_cortex_perfusion_weight=self.cortex_perfusion.value(),
            renal_medulla_perfusion_weight=self.medulla_perfusion.value(),
            liver_perfusion_weight=self.liver_perfusion.value(),
            brain_perfusion_weight=self.brain_perfusion.value(),
            vessel_perfusion_weight=self.vessel_perfusion.value(),
            left_cortex_kpl_s_inv=self.kpl_spins["left_cortex"].value(),
            left_medulla_kpl_s_inv=self.kpl_spins["left_medulla"].value(),
            right_cortex_kpl_s_inv=self.kpl_spins["right_cortex"].value(),
            right_medulla_kpl_s_inv=self.kpl_spins["right_medulla"].value(),
            liver_kpa_s_inv=self.liver_kpa.value(),
            background_ph=self.ph_spins["background"].value(),
            left_cortex_ph=self.ph_spins["left_cortex"].value(),
            left_medulla_ph=self.ph_spins["left_medulla"].value(),
            right_cortex_ph=self.ph_spins["right_cortex"].value(),
            right_medulla_ph=self.ph_spins["right_medulla"].value(),
            breathing_enabled=self.breathing_enabled.isChecked(),
            breathing_rate_hz=self.breathing_rate.value(),
            breathing_amplitude_mm=self.breathing_amplitude.value(),
            breathing_phase_rad=self.breathing_phase.value(),
        )
        return MousePerfusionConfig.from_dict(values)

    def _build_phantom(self) -> MousePerfusionPhantom:
        config = self._read_config()
        phantom = config.build()
        self.config = config
        return phantom

    def _preview(self):
        try:
            phantom = self._build_phantom()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid mouse phantom", str(exc))
            return
        self.phantom = phantom
        self._voxel_pool_curves = None
        self.voxel_spectrum_plot.setXRange(
            phantom.spectral_window_center_ppm - 0.5 * phantom.spectral_bandwidth_ppm,
            phantom.spectral_window_center_ppm + 0.5 * phantom.spectral_bandwidth_ppm,
            padding=0.0,
        )
        self._animation_bounds_s = phantom.preview_time_bounds_s()
        current_time = float(
            np.clip(
                self.animation_time.value(),
                self._animation_bounds_s[0],
                self._animation_bounds_s[1],
            )
        )
        self.animation_time.blockSignals(True)
        self.animation_time.setValue(current_time)
        self.animation_time.blockSignals(False)
        phantom.preview_bolus_time_s = current_time
        self.anatomy_preview.set_anatomy(phantom.anatomy_labels)
        self.perfusion_preview.set_anatomy(phantom.anatomy_labels)
        self.perfusion_preview.set_display_geometry(phantom.fov, phantom.fov)
        self.combined_preview.set_anatomy(phantom.anatomy_labels)
        self._anatomy_resolution_changed()
        self._update_curve_plots()
        self._animation_time_changed(current_time)
        selection_is_valid = (
            self._selected_voxel is not None
            and all(
                0 <= value < extent
                for value, extent in zip(self._selected_voxel, phantom.shape)
            )
            and phantom.anatomy_labels[self._selected_voxel] > 0
        )
        if not selection_is_valid:
            candidates = np.argwhere(phantom.anatomy_labels == 2)
            if candidates.size == 0:
                candidates = np.argwhere(phantom.anatomy_labels > 0)
            self._selected_voxel = (
                None
                if candidates.size == 0
                else tuple(int(value) for value in candidates[len(candidates) // 2])
            )
        self.combined_preview.set_selected_voxel(self._selected_voxel)
        self._prepare_voxel_spectrum()
        self._update_voxel_spectrum(current_time)
        previous_map = self.inspector.map_combo.currentText()
        self.inspector.set_phantom(phantom)
        target_map = (
            previous_map
            if self.inspector.map_combo.findText(previous_map) >= 0
            else "Anatomy labels"
        )
        if not target_map:
            target_map = "Anatomy labels"
        self.inspector.map_combo.setCurrentText(target_map)
        curve = phantom.pyruvate_inflow.rate_curve_s_inv
        delivered_mol = curve.integral(curve.times_s[0], curve.times_s[-1]) * (
            np.sum(phantom.pyruvate_inflow.delivery_map) * phantom.voxel_volume_m3
        )
        kidney_voxels = int(
            np.count_nonzero(np.isin(phantom.anatomy_labels, (2, 3, 4, 5)))
        )
        self.preview_summary.setText(
            f"{phantom.name} · matrix {phantom.shape} · "
            f"{phantom.injection_compound_label} · "
            f"{phantom.n_active:,} active voxels · {kidney_voxels:,} renal voxels · "
            f"{len(phantom.vascular_graph.edges)} major-vessel segments · "
            f"integrated source {delivered_mol * 1e6:.6g} µmol"
        )

    def _compound_selection_changed(self, *_):
        key = str(self.injection_compound.currentData())
        messages = {
            "pyruvate": (
                "Pyruvate is delivered to the first dynamic pool. Renal kPL is "
                "active in the sequence solver, together with the liver-specific "
                "Alanine pool and kPA conversion."
            ),
            "lactate": (
                "Lactate is transported as the injected pool without forced "
                "pyruvate conversion."
            ),
            "z_ompd": (
                "Z-OMPD is transported as a pH/perfusion tracer. C5 and C1 "
                "spectral pools are created; the current source drives C5 while "
                "the renal pH map remains explicit ground truth."
            ),
            "contrast_agent": (
                "Generic contrast agent animates tracer delivery and clearance. "
                "Agent-specific T1/T2 relaxivity is not inferred automatically."
            ),
        }
        self.compound_help.setText(messages.get(key, ""))
        enabled = key == "pyruvate"
        for spin in self.kpl_spins.values():
            spin.setEnabled(enabled)
        self.liver_kpa.setEnabled(enabled)

    def _anatomy_resolution_changed(self, *_):
        if self.phantom is None:
            return
        fov = np.asarray(self.phantom.fov, dtype=float)
        mouse_size = np.asarray(self.phantom.config.mouse_size_m, dtype=float)
        mouse_offset = np.asarray(self.phantom.config.mouse_offset_m, dtype=float)
        display_lower = np.minimum(-0.5 * fov, mouse_offset - 0.5 * mouse_size)
        display_upper = np.maximum(0.5 * fov, mouse_offset + 0.5 * mouse_size)
        display_extent = tuple(display_upper - display_lower)
        display_center = tuple(0.5 * (display_lower + display_upper))
        self.anatomy_preview.set_display_geometry(
            self.phantom.fov, display_extent, display_center
        )
        self.combined_preview.set_display_geometry(
            self.phantom.fov, display_extent, display_center
        )
        slices_by_scale = {}
        for widget, resolution in (
            (self.anatomy_preview, self.anatomy_resolution.currentData()),
            (self.combined_preview, self.combined_anatomy_resolution.currentData()),
        ):
            scale = 8 if resolution == "hd" else 1
            if scale not in slices_by_scale:
                slices_by_scale[scale] = self._build_anatomy_slices(
                    scale, display_extent, display_center
                )
            widget.set_anatomy_slices(*slices_by_scale[scale])

    def _build_anatomy_slices(self, scale, display_extent, display_center):
        nx, ny, nz = self.phantom.shape
        kwargs = {
            "graph": self.phantom.vascular_graph,
            "mouse_size_m": self.phantom.config.mouse_size_m,
            "mouse_offset_m": self.phantom.config.mouse_offset_m,
        }
        axial, _, _ = build_reference_mouse_anatomy(
            (nx * scale, ny * scale, 1),
            display_extent,
            grid_center_m=(display_center[0], display_center[1], 0.0),
            **kwargs,
        )
        coronal, _, _ = build_reference_mouse_anatomy(
            (nx * scale, 1, nz * scale),
            display_extent,
            grid_center_m=(display_center[0], 0.0, display_center[2]),
            **kwargs,
        )
        sagittal, _, _ = build_reference_mouse_anatomy(
            (1, ny * scale, nz * scale),
            display_extent,
            grid_center_m=(0.0, display_center[1], display_center[2]),
            **kwargs,
        )
        return (
            axial[:, :, 0].T,
            coronal[:, 0, :].T,
            sagittal[0, :, :].T,
        )

    def _update_curve_plots(self):
        if self.phantom is None:
            return
        self.source_curve_plot.clear()
        self.organ_curve_plot.clear()
        curve = self.phantom.pyruvate_inflow.rate_curve_s_inv
        source_times = np.asarray(curve.times_s, dtype=float)
        source_rate = np.asarray(curve.values, dtype=float) * 1e6
        self.source_curve_plot.plot(
            source_times,
            source_rate,
            pen=pg.mkPen("#f4a261", width=2),
            fillLevel=0.0,
            brush=pg.mkBrush(244, 162, 97, 60),
        )
        time_start, time_end = self._animation_bounds_s
        times = np.linspace(time_start, time_end, 120)
        colors = {
            "Brain": "#e9c46a",
            "Liver": "#2a9d8f",
            "Kidneys": "#e76f51",
            "Bladder": "#8ecae6",
        }
        for name, values in self.phantom.organ_bolus_curves(times).items():
            self.organ_curve_plot.plot(
                times,
                values,
                pen=pg.mkPen(colors[name], width=2),
                name=name,
            )
        self.source_time_marker = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("w", width=1)
        )
        self.organ_time_marker = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("w", width=1)
        )
        self.source_curve_plot.addItem(self.source_time_marker)
        self.organ_curve_plot.addItem(self.organ_time_marker)

    def _animation_time_changed(self, value):
        if self.phantom is None:
            return
        self.phantom.preview_bolus_time_s = float(value)
        frame = self.phantom.injected_concentration_map(float(value))
        self.perfusion_preview.set_perfusion(frame)
        self.combined_preview.set_perfusion(frame)
        start, end = self._animation_bounds_s
        fraction = 0.0 if end <= start else (float(value) - start) / (end - start)
        slider_value = int(np.clip(round(1000.0 * fraction), 0, 1000))
        self.animation_slider.blockSignals(True)
        self.animation_slider.setValue(slider_value)
        self.animation_slider.blockSignals(False)
        self.perfusion_animation_time.blockSignals(True)
        self.perfusion_animation_time.setValue(float(value))
        self.perfusion_animation_time.blockSignals(False)
        self.perfusion_animation_slider.blockSignals(True)
        self.perfusion_animation_slider.setValue(slider_value)
        self.perfusion_animation_slider.blockSignals(False)
        if hasattr(self, "source_time_marker"):
            self.source_time_marker.setValue(float(value))
            self.organ_time_marker.setValue(float(value))
        if self.inspector.map_combo.currentText() in {
            "Bolus source",
            "Injected concentration",
        }:
            self.inspector._map_changed()
        self._update_voxel_spectrum(float(value))

    def _voxel_selected(self, voxel):
        if self.phantom is None:
            return
        self._selected_voxel = tuple(int(value) for value in voxel)
        self._prepare_voxel_spectrum()
        self._update_voxel_spectrum(self.animation_time.value())

    def _prepare_voxel_spectrum(self):
        if self.phantom is None or self._selected_voxel is None:
            self._voxel_pool_curves = None
            return
        start, end = self._animation_bounds_s
        step = min(
            self.phantom.config.perfusion_timestep_s,
            self.phantom.config.injection_duration_s / 128.0,
        )
        count = max(2, int(np.ceil((end - start) / step)) + 1)
        times = np.linspace(start, end, count)
        curves = self.phantom.preview_pool_curves_at(self._selected_voxel, times)
        self._voxel_pool_curves = (times, curves)

    def _update_voxel_spectrum(self, time_s):
        if (
            self.phantom is None
            or self._selected_voxel is None
            or getattr(self, "_voxel_pool_curves", None) is None
        ):
            return
        times, curves = self._voxel_pool_curves
        axis = np.linspace(
            self.phantom.spectral_window_center_ppm
            - 0.5 * self.phantom.spectral_bandwidth_ppm,
            self.phantom.spectral_window_center_ppm
            + 0.5 * self.phantom.spectral_bandwidth_ppm,
            max(1025, self.phantom.spectral_points),
        )
        total = np.zeros(axis.shape, dtype=float)
        colors = ("#f4a261", "#2a9d8f", "#e76f51", "#8ecae6")
        components = []
        for pool, color in zip(self.phantom.pools, colors):
            amplitude = float(
                np.interp(float(time_s), times, curves[pool.name], left=0.0, right=0.0)
            )
            center = (
                self.phantom.spectral_reference_ppm
                + self.phantom.get_frequency_offset_ppm(pool.name)
            )
            carbon_frequency_hz = max(
                self.phantom.field_strength * 10.705e6, np.finfo(float).eps
            )
            half_width = max(
                0.5 * (1.0 / (np.pi * pool.t2_star)) / carbon_frequency_hz * 1e6,
                np.finfo(float).eps,
            )
            component = amplitude / (1.0 + ((axis - center) / half_width) ** 2)
            total += component
            components.append((pool.name, component, color, amplitude))
        self.voxel_spectrum_plot.clear()
        for name, component, color, amplitude in components:
            if amplitude > 0.0:
                self.voxel_spectrum_plot.plot(
                    axis,
                    component,
                    pen=pg.mkPen(color, width=1),
                    name=name,
                )
        self.voxel_spectrum_plot.plot(axis, total, pen=pg.mkPen("w", width=2))
        label_value = int(self.phantom.anatomy_labels[self._selected_voxel])
        tissue = self.phantom.tissue_label_names.get(label_value, "Unknown")
        concentrations = ", ".join(
            f"{name} {amplitude:.3g} mM" for name, _, _, amplitude in components
        )
        self.voxel_spectrum_label.setText(
            f"Voxel {self._selected_voxel} · {tissue} · {concentrations}"
        )

    def _animation_slider_changed(self, value):
        start, end = self._animation_bounds_s
        self.animation_time.setValue(start + (end - start) * float(value) / 1000.0)

    def _play_toggled(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        self.perfusion_play_button.blockSignals(True)
        self.perfusion_play_button.setChecked(bool(playing))
        self.perfusion_play_button.setText("Pause" if playing else "Play")
        self.perfusion_play_button.blockSignals(False)
        if playing:
            self._animation_timer.start()
        else:
            self._animation_timer.stop()

    def _advance_animation(self):
        start, end = self._animation_bounds_s
        step = self.animation_speed.value() * self._animation_timer.interval() / 1000.0
        value = self.animation_time.value() + step
        if value > end:
            value = start
        self.animation_time.setValue(value)

    def closeEvent(self, event):
        self._animation_timer.stop()
        self._live_preview_timer.stop()
        super().closeEvent(event)

    def _accept(self):
        try:
            self.phantom = self._build_phantom()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid mouse phantom", str(exc))
            return
        self.accept()

    def get_phantom(self) -> Optional[MousePerfusionPhantom]:
        return self.phantom
