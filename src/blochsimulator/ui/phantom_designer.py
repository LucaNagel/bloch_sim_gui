"""Interactive shape and Lorentz-peak phantom designer."""

from __future__ import annotations

import os
from pathlib import Path
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
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
from .default_settings import WorkspaceDefaults

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    gl = None
    HAS_OPENGL = False


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
        if kind not in {"ellipsoid", "box", "cylinder"}:
            raise ValueError("drawing kind must be 'ellipsoid', 'box', or 'cylinder'")
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


def _shape_preview_mesh(item, fov_mm):
    """Return vertices and triangular faces for a physical-aspect 3D preview."""
    fov = np.asarray(fov_mm, dtype=float)
    scale = max(float(np.max(fov)), 1e-12)
    half_size = np.asarray(item.size, dtype=float) * fov / scale
    center = (np.asarray(item.center, dtype=float) - 0.5) * 2.0 * fov / scale

    if item.kind == "box":
        vertices = np.asarray(
            [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            dtype=float,
        )
        faces = np.asarray(
            [
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            ],
            dtype=np.uint32,
        )
    elif item.kind == "cylinder":
        segments = 36
        angle = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ring = np.column_stack((np.cos(angle), np.sin(angle)))
        vertices = np.vstack(
            (
                np.column_stack((ring, -np.ones(segments))),
                np.column_stack((ring, np.ones(segments))),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
            )
        )
        faces = []
        for index in range(segments):
            nxt = (index + 1) % segments
            faces.extend(
                (
                    (index, nxt, segments + nxt),
                    (index, segments + nxt, segments + index),
                    (2 * segments, nxt, index),
                    (2 * segments + 1, segments + index, segments + nxt),
                )
            )
        faces = np.asarray(faces, dtype=np.uint32)
    else:
        latitude_count, longitude_count = 14, 28
        latitude = np.linspace(0.0, np.pi, latitude_count + 1)
        longitude = np.linspace(0.0, 2.0 * np.pi, longitude_count, endpoint=False)
        vertices = np.asarray(
            [
                (
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                )
                for phi in latitude
                for theta in longitude
            ],
            dtype=float,
        )
        faces = []
        for row in range(latitude_count):
            for column in range(longitude_count):
                nxt = (column + 1) % longitude_count
                lower = row * longitude_count + column
                lower_next = row * longitude_count + nxt
                upper = (row + 1) * longitude_count + column
                upper_next = (row + 1) * longitude_count + nxt
                faces.extend(
                    ((lower, lower_next, upper_next), (lower, upper_next, upper))
                )
        faces = np.asarray(faces, dtype=np.uint32)

    vertices = vertices * half_size
    angles = np.deg2rad(
        np.asarray(getattr(item, "rotation_deg", (0.0, 0.0, 0.0)), dtype=float)
    )
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    rotation = rotation_z @ rotation_y @ rotation_x
    return vertices @ rotation.T + center, faces


def _convex_hull_2d(points):
    """Return counter-clockwise indices of the 2D convex hull."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return np.arange(len(points), dtype=int)
    order = np.lexsort((points[:, 1], points[:, 0]))
    unique = []
    for index in order:
        if not unique or not np.allclose(points[index], points[unique[-1]]):
            unique.append(int(index))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=int)

    def cross(origin, first, second):
        a = points[first] - points[origin]
        b = points[second] - points[origin]
        return a[0] * b[1] - a[1] * b[0]

    lower = []
    for index in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 0.0:
            lower.pop()
        lower.append(index)
    upper = []
    for index in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 0.0:
            upper.pop()
        upper.append(index)
    return np.asarray(lower[:-1] + upper[:-1], dtype=int)


def _shape_xy_projection(item, fov_mm):
    """Return the rotated primitive's orthographic XY silhouette in [0, 1]."""
    fov = np.asarray(fov_mm, dtype=float)
    vertices, _faces = _shape_preview_mesh(item, fov)
    scale = max(float(np.max(fov)), 1e-12)
    projected = np.column_stack(
        (
            0.5 + vertices[:, 0] * scale / (2.0 * fov[0]),
            0.5 + vertices[:, 1] * scale / (2.0 * fov[1]),
        )
    )
    hull = _convex_hull_2d(projected)
    if hull.size:
        hull = np.append(hull, hull[0])
    return projected[hull, 0], projected[hull, 1]


class ShapePreview3DWidget(QWidget):
    """Small orbitable preview of all design primitives without rasterization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = None
        self._shape_items = []
        if HAS_OPENGL and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":
            self.view = gl.GLViewWidget()
            # Match VolumeViewerWidget's default camera so switching to the
            # full 3D/frequency preview does not rotate the phantom.
            self.view.setCameraPosition(distance=4.2, elevation=25, azimuth=35)
            self.view.setBackgroundColor((18, 22, 30))
            self.bounds = gl.GLLinePlotItem(
                pos=np.zeros((0, 3)),
                color=(0.75, 0.8, 0.9, 0.8),
                width=1.2,
                mode="lines",
                antialias=True,
            )
            self.view.addItem(self.bounds)
            layout.addWidget(self.view)
        else:
            unavailable = QLabel("3D preview unavailable (OpenGL)")
            unavailable.setAlignment(Qt.AlignCenter)
            unavailable.setStyleSheet("background: #12161e; color: #c8d0df;")
            layout.addWidget(unavailable)

    @staticmethod
    def _bounds_vertices(fov_mm):
        extent = np.asarray(fov_mm, dtype=float)
        extent = extent / max(float(np.max(extent)), 1e-12)
        corners = np.asarray(
            [
                (x, y, z)
                for x in (-extent[0], extent[0])
                for y in (-extent[1], extent[1])
                for z in (-extent[2], extent[2])
            ]
        )
        edges = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )
        return np.asarray([corners[index] for edge in edges for index in edge])

    def _add_mesh(self, item, fov_mm, color, *, selected=False):
        vertices, faces = _shape_preview_mesh(item, fov_mm)
        red, green, blue, alpha = color
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            color=(red, green, blue, alpha),
            drawEdges=True,
            edgeColor=(red, green, blue, 1.0 if selected else 0.6),
            smooth=item.kind != "box",
            shader="shaded",
            glOptions="translucent",
        )
        self.view.addItem(mesh)
        self._shape_items.append(mesh)

    def set_shapes(
        self,
        shapes,
        fov_mm,
        selected_row=None,
        highlighted_region=None,
    ):
        if self.view is None:
            return
        for mesh in self._shape_items:
            self.view.removeItem(mesh)
        self._shape_items.clear()
        self.bounds.setData(pos=self._bounds_vertices(fov_mm))
        hue_count = max(1, len(shapes))
        for index, item in enumerate(shapes):
            color = pg.intColor(index, hues=hue_count)
            red, green, blue, _ = color.getRgbF()
            selected = index == selected_row
            self._add_mesh(
                item,
                fov_mm,
                (red, green, blue, 0.82 if selected else 0.28),
                selected=selected,
            )
        if highlighted_region is not None:
            self._add_mesh(
                highlighted_region,
                fov_mm,
                (1.0, 0.78, 0.08, 0.38),
                selected=True,
            )


def load_any_phantom(filename):
    """Load either a conventional or spectral phantom file."""
    try:
        phantom = DynamicSpectralPhantom.load(filename)
    except ValueError:
        try:
            phantom = SpectralPhantom.load(filename)
        except ValueError as spectral_error:
            try:
                phantom = Phantom.load(filename)
            except Exception:
                raise spectral_error
    phantom.name = Path(filename).stem
    return phantom


class SpectralPhantomDesignerDialog(QDialog):
    """Compose rotatable 3D primitives and assign Lorentzian peak lists."""

    def __init__(
        self,
        parent=None,
        design: Optional[PhantomDesign] = None,
        settings=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Spectral Phantom" if design is not None else "New Spectral Phantom"
        )
        self.resize(1450, 900)
        if design is None:
            defaults = WorkspaceDefaults.from_settings(settings)
            design = PhantomDesign(
                fov_m=tuple(value / 1000.0 for value in defaults.phantom_fov_mm),
                nucleus=defaults.phantom_nucleus,
                field_strength_t=defaults.field_strength_t,
                shapes=[ShapeDefinition(name="Shape 1", kind="cylinder")],
            )
        self.design = design
        self.phantom = None
        self._updating = False
        self._last_spectral_reference_ppm = float(design.spectral_reference_ppm)
        self._inspector_preview_dirty = True
        self._building_inspector_preview = False
        self._rois = []
        self._projection_items = []
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
            matrix.setRange(1, 1025)
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

        sampling_row = QHBoxLayout()
        self.supersampling_enabled = QCheckBox("Supersampling")
        self.supersampling_enabled.setToolTip(
            "Rasterize every object on a finer subvoxel grid and average it "
            "back to the selected matrix, producing fractional edge voxels"
        )
        sampling_row.addWidget(self.supersampling_enabled)
        sampling_row.addWidget(QLabel("Factor"))
        self.supersampling_factor = QSpinBox()
        self.supersampling_factor.setRange(2, 8)
        self.supersampling_factor.setValue(4)
        self.supersampling_factor.setSuffix("× per axis")
        self.supersampling_factor.setToolTip(
            "Samples per voxel axis; the work grows with the cube of this value"
        )
        self.supersampling_factor.setEnabled(False)
        self.supersampling_enabled.toggled.connect(self.supersampling_factor.setEnabled)
        sampling_row.addWidget(self.supersampling_factor)
        sampling_row.addStretch()
        draw_layout.addLayout(sampling_row)

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
        shape_panel.setMinimumWidth(280)
        shape_panel.setMaximumWidth(360)
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
        add_cylinder = QPushButton("Add cylinder")
        add_cylinder.setToolTip(
            "Add a cylinder whose local axis initially points along Z"
        )
        add_cylinder.clicked.connect(lambda: self._add_shape("cylinder"))
        draw_ellipse = QPushButton("Draw ellipsoid")
        draw_ellipse.setToolTip("Drag a new ellipsoid directly in the XY canvas")
        draw_ellipse.clicked.connect(lambda: self._start_shape_drawing("ellipsoid"))
        draw_box = QPushButton("Draw box")
        draw_box.setToolTip("Drag a new box directly in the XY canvas")
        draw_box.clicked.connect(lambda: self._start_shape_drawing("box"))
        draw_cylinder = QPushButton("Draw cylinder")
        draw_cylinder.setToolTip("Drag the X/Y diameters of a new Z-aligned cylinder")
        draw_cylinder.clicked.connect(lambda: self._start_shape_drawing("cylinder"))
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_shape)
        shape_buttons.addWidget(add_ellipse, 0, 0)
        shape_buttons.addWidget(add_box, 1, 0)
        shape_buttons.addWidget(add_cylinder, 2, 0)
        shape_buttons.addWidget(draw_ellipse, 3, 0)
        shape_buttons.addWidget(draw_box, 4, 0)
        shape_buttons.addWidget(draw_cylinder, 5, 0)
        shape_buttons.addWidget(remove, 6, 0)
        shape_layout.addLayout(shape_buttons)
        splitter.addWidget(shape_panel)

        canvas_panel = QWidget()
        canvas_panel.setMinimumWidth(420)
        canvas_layout = QVBoxLayout(canvas_panel)
        self.canvas_instruction = QLabel(
            "XY drag/resize · values left · live 3D preview"
        )
        self.canvas_instruction.setToolTip(
            "Solid outlines are the rotated 3D shapes projected onto XY. "
            "The faint dashed outline is the selected shape's editable XY frame."
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
        for spin in self.fov_spins:
            spin.valueChanged.connect(self._update_canvas_axes)
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_panel)

        property_panel = QWidget()
        property_layout = QVBoxLayout(property_panel)
        property_layout.addWidget(QLabel("Live 3D preview · drag to orbit"))
        self.shape_preview_3d = ShapePreview3DWidget()
        self.shape_preview_3d.setMinimumHeight(180)
        self.shape_preview_3d.setMaximumHeight(220)
        property_layout.addWidget(self.shape_preview_3d)

        geometry_group = QGroupBox("Selected shape geometry")
        geometry_layout = QVBoxLayout(geometry_group)
        identity_form = QFormLayout()
        identity_form.setContentsMargins(0, 0, 0, 0)
        self.shape_name = QLineEdit()
        self.shape_name.editingFinished.connect(self._properties_changed)
        identity_form.addRow("Name", self.shape_name)
        self.kind_label = QLabel()
        identity_form.addRow("Kind", self.kind_label)
        geometry_layout.addLayout(identity_form)
        self.x_center = self._position_percent_spin(50.0)
        self.y_center = self._position_percent_spin(50.0)
        self.z_center = self._position_percent_spin(50.0)
        self.x_size = self._percent_spin(50.0)
        self.y_size = self._percent_spin(50.0)
        self.z_size = self._percent_spin(50.0)
        self.z_size.setToolTip(
            "Cylinder length along its local Z axis, as a percentage of FOV Z. "
            "Cylinders may extend beyond the FOV and are clipped at its boundary."
        )
        self.rotation_spins = []
        for axis in "XYZ":
            rotation = self._number_spin(-360.0, 360.0, 0.0, "°")
            rotation.setDecimals(1)
            rotation.setToolTip(
                f"Rotate the shape about the physical {axis} axis; rotations are "
                "applied in X, Y, Z order and shown in the 3D preview"
            )
            self.rotation_spins.append(rotation)
        self.t1_ms = self._number_spin(0.1, 500000.0, 1000.0, " ms")
        self.initial_mz = self._number_spin(0.0, 1e9, 1.0, "")
        self.b0_ppm = self._number_spin(-1000.0, 1000.0, 0.0, " ppm")
        self.t1_ms.setMaximumWidth(140)
        self.initial_mz.setMaximumWidth(105)
        self.b0_ppm.setMaximumWidth(120)
        self.t1_ms.setToolTip(
            "Fallback T1 for peaks whose metabolite-specific T1 cell is empty."
        )
        self.initial_mz.setToolTip(
            "Fallback initial longitudinal polarization for peaks whose own "
            "polarization cell is empty. This is independent of spin density. "
            "In both models, polarization 1 is thermal equilibrium; T1 drives "
            "larger or smaller values toward 1."
        )
        for widget in (
            self.x_center,
            self.y_center,
            self.z_center,
            self.x_size,
            self.y_size,
            self.z_size,
            *self.rotation_spins,
            self.t1_ms,
            self.initial_mz,
            self.b0_ppm,
        ):
            widget.valueChanged.connect(self._properties_changed)
        for widget in (
            self.x_center,
            self.y_center,
            self.z_center,
            self.x_size,
            self.y_size,
            self.z_size,
            *self.rotation_spins,
        ):
            widget.setMinimumWidth(70)
            widget.setMaximumWidth(90)
        geometry_grid = QGridLayout()
        geometry_grid.setHorizontalSpacing(10)
        geometry_grid.setVerticalSpacing(6)
        geometry_grid.setContentsMargins(0, 4, 0, 4)
        for column, heading in enumerate(("Centre", "Size", "Rotation"), start=1):
            label = QLabel(heading)
            label.setAlignment(Qt.AlignCenter)
            geometry_grid.addWidget(label, 0, column)
        center_spins = (self.x_center, self.y_center, self.z_center)
        size_spins = (self.x_size, self.y_size, self.z_size)
        for row, (axis, center, size, rotation) in enumerate(
            zip("XYZ", center_spins, size_spins, self.rotation_spins), start=1
        ):
            axis_label = QLabel(axis)
            axis_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            geometry_grid.addWidget(axis_label, row, 0)
            geometry_grid.addWidget(center, row, 1)
            geometry_grid.addWidget(size, row, 2)
            geometry_grid.addWidget(rotation, row, 3)
        geometry_layout.addLayout(geometry_grid)

        # Retained as a semantic label for code inspecting the cylinder control;
        # the visible grid row stays consistently labelled "Z".
        self.z_size_label = QLabel("Z size")
        self.xy_info = QLabel()
        self.xy_info.setWordWrap(True)
        geometry_layout.addWidget(self.xy_info)
        shape_layout.addWidget(geometry_group)

        defaults_row = QHBoxLayout()
        defaults_row.setSpacing(6)
        defaults_row.addWidget(QLabel("Default T1"))
        defaults_row.addWidget(self.t1_ms)
        defaults_row.addWidget(QLabel("Initial polarization"))
        defaults_row.addWidget(self.initial_mz)
        defaults_row.addWidget(QLabel("B0 offset"))
        defaults_row.addWidget(self.b0_ppm)
        property_layout.addLayout(defaults_row)

        peak_explanation = QLabel(
            "Spin density / concentration sets how much signal-producing material "
            "is present. Initial polarization sets its longitudinal start state; "
            "the initial signal is their product. Polarization 1 is thermal "
            "equilibrium; hyperpolarized values can be much larger. Set spin "
            "density to 0 for a region with no initial material. Leave "
            "polarization or T1 empty to use the shape default."
        )
        peak_explanation.setWordWrap(True)
        property_layout.addWidget(peak_explanation)
        self.peak_table = QTableWidget(0, 6)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Spin density",
                "Initial polarization",
                "Peak (ppm)",
                "T1 (ms)",
                "T2* (ms)",
            ]
        )
        peak_header = self.peak_table.horizontalHeader()
        peak_header.setSectionResizeMode(0, QHeaderView.Fixed)
        peak_header.resizeSection(0, 90)
        for column in range(1, 4):
            peak_header.setSectionResizeMode(column, QHeaderView.Stretch)
        for column in (4, 5):
            peak_header.setSectionResizeMode(column, QHeaderView.Fixed)
            peak_header.resizeSection(column, 75)
        peak_header.setMinimumSectionSize(62)
        self.peak_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.peak_table.setWordWrap(False)
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
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([330, 470, 600])
        self.tabs.addTab(draw_page, "Draw and edit")

        kinetics_page = QWidget()
        kinetics_page_layout = QHBoxLayout(kinetics_page)
        kinetics_splitter = QSplitter(Qt.Horizontal)
        kinetics_splitter.setChildrenCollapsible(False)
        kinetics_controls = QWidget()
        kinetics_controls.setMinimumWidth(600)
        kinetics_layout = QVBoxLayout(kinetics_controls)
        kinetics_form = QFormLayout()
        kinetics_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.dynamic_enabled = QCheckBox(
            "Enable hyperpolarized pyruvate/lactate model (polarization → 1)"
        )
        self.dynamic_enabled.setToolTip(
            "Select the hyperpolarized two-pool solver. Conversion itself is "
            "controlled separately by kPL; kPL=0 means T1 relaxation of "
            "polarization toward thermal equilibrium 1 without P→L."
        )
        self.dynamic_enabled.toggled.connect(self._update_kinetics_preview)
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
            "Pyruvate inflow points separately define concentration rate and "
            "polarization. Each row is held until the next time; outside the "
            "listed interval the rate is zero. For example, rate 10 from 5–6 s "
            "adds concentration 10, and polarization 10000 gives the incoming "
            "material that polarization. The global kinetics-time "
            "setting shifts inflow and conversion together relative to sequence "
            "time zero. Any part before sequence t=0 forms a free kinetic pre-roll "
            "that sets the initial Pz/Lz distribution."
        )
        inflow_help.setWordWrap(True)
        kinetics_layout.addWidget(inflow_help)
        self.inflow_curve_table = QTableWidget(0, 3)
        self.inflow_curve_table.setHorizontalHeaderLabels(
            [
                "Kinetics time (s)",
                "Concentration rate (relative/s)",
                "Inflow polarization",
            ]
        )
        self.inflow_curve_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.inflow_curve_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.inflow_curve_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.inflow_curve_table.setMaximumHeight(150)
        self.inflow_curve_table.setMinimumHeight(110)
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
        self.dynamic_b0_curve_table.setMinimumHeight(105)
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
        self.kinetic_table.setMinimumHeight(130)
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
            "This preview is active only for the hyperpolarized model. HP Mz is "
            "spin density × initial polarization (plus inflow/conversion). "
            "Polarization=1 is thermal equilibrium; T1 drives any larger or "
            "smaller polarization toward 1."
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
            "Sets Lactate initial polarization to zero without changing its spin "
            "density; positive kPL can then create Lactate from Pyruvate."
        )
        self.zero_lactate_button.clicked.connect(self._set_selected_shape_lactate_zero)
        preview_form.addRow("Pyruvate-only start", self.zero_lactate_button)
        self.kinetics_preview_duration = self._number_spin(0.1, 10000.0, 30.0, " s")
        self.kinetics_preview_duration.valueChanged.connect(
            self._update_kinetics_preview
        )
        preview_form.addRow("Duration", self.kinetics_preview_duration)
        preview_controls = QHBoxLayout()
        preview_controls.addLayout(preview_form, 1)
        spatial_preview_group = QGroupBox("Spatial selection · drag to orbit")
        spatial_preview_layout = QVBoxLayout(spatial_preview_group)
        self.kinetics_shape_preview_3d = ShapePreview3DWidget()
        self.kinetics_shape_preview_3d.setMinimumSize(300, 155)
        self.kinetics_shape_preview_3d.setMaximumHeight(190)
        spatial_preview_layout.addWidget(self.kinetics_shape_preview_3d)
        self.kinetics_spatial_preview_info = QLabel()
        self.kinetics_spatial_preview_info.setWordWrap(True)
        spatial_preview_layout.addWidget(self.kinetics_spatial_preview_info)
        preview_controls.addWidget(spatial_preview_group, 1)
        preview_layout.addLayout(preview_controls)

        self.kinetics_preview_graphics = pg.GraphicsLayoutWidget()
        self.kinetics_preview_graphics.setMinimumWidth(420)
        self.inflow_preview_plot = self.kinetics_preview_graphics.addPlot(row=0, col=0)
        self.inflow_preview_plot.setLabel(
            "left", "Concentration inflow", units="relative/s"
        )
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
        self.polarization_preview_plot = self.kinetics_preview_graphics.addPlot(
            row=1, col=0
        )
        self.polarization_preview_plot.setLabel("left", "Polarization")
        self.polarization_preview_plot.showGrid(x=True, y=True, alpha=0.25)
        self.polarization_preview_plot.addLegend(offset=(8, 8))
        self.pyruvate_polarization_curve = self.polarization_preview_plot.plot(
            pen=pg.mkPen("c", width=2), name="Pyruvate P"
        )
        self.lactate_polarization_curve = self.polarization_preview_plot.plot(
            pen=pg.mkPen("m", width=2, style=Qt.DashLine), name="Lactate P"
        )
        self.equilibrium_polarization_line = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            pen=pg.mkPen((180, 180, 180), width=1, style=Qt.DotLine),
        )
        self.polarization_preview_plot.addItem(self.equilibrium_polarization_line)
        self.pool_preview_plot = self.kinetics_preview_graphics.addPlot(row=2, col=0)
        self.pool_preview_plot.setLabel("left", "Mz = concentration × P")
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
        self.polarization_preview_plot.setXLink(self.inflow_preview_plot)
        preview_layout.addWidget(self.kinetics_preview_graphics)
        self.kinetics_preview_info = QLabel()
        self.kinetics_preview_info.setWordWrap(True)
        preview_layout.addWidget(self.kinetics_preview_info)

        # Keep explanatory text and fields at their natural height. On shorter
        # displays the controls scroll instead of being compressed until labels
        # and checkboxes overlap.
        kinetics_controls.setMinimumHeight(kinetics_controls.sizeHint().height())
        kinetics_scroll = QScrollArea()
        kinetics_scroll.setWidgetResizable(True)
        kinetics_scroll.setFrameShape(QFrame.NoFrame)
        kinetics_scroll.setMinimumWidth(620)
        kinetics_scroll.setWidget(kinetics_controls)
        self.kinetics_controls_scroll = kinetics_scroll
        self.kinetics_splitter = kinetics_splitter

        kinetics_splitter.addWidget(kinetics_scroll)
        kinetics_splitter.addWidget(preview_panel)
        kinetics_splitter.setStretchFactor(0, 1)
        kinetics_splitter.setStretchFactor(1, 1)
        kinetics_splitter.setSizes([650, 750])
        kinetics_page_layout.addWidget(kinetics_splitter)
        self.tabs.addTab(kinetics_page, "Kinetics / kPL")

        self.inspector = PhantomInspectorWidget()
        self.tabs.addTab(self.inspector, "3D / frequency preview")
        self.tabs.currentChanged.connect(self._tab_changed)

        action_row = QHBoxLayout()
        preview = QPushButton("Update preview")
        preview.setToolTip(
            "Rasterize the current design; use the 3D view to orbit around it"
        )
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
        self.supersampling_enabled.setChecked(self.design.supersampling_enabled)
        self.supersampling_factor.setValue(int(self.design.supersampling_factor))
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
        self._populate_inflow_curve(
            self.design.pyruvate_inflow_curve,
            self.design.pyruvate_inflow_polarization_curve,
            default=((0.0, 0.0, 1.0), (5.0, 10.0, 10000.0), (6.0, 0.0, 1.0)),
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
        for projection in self._projection_items:
            self.canvas.removeItem(projection)
        self._projection_items.clear()
        for index, item in enumerate(self.design.shapes):
            self.shape_list.addItem(item.name)
            self._create_roi(item, index)
        self._refresh_kinetics_preview_shapes()
        self._updating = False
        if self.design.shapes:
            self.shape_list.setCurrentRow(0)
        else:
            self._update_shape_preview()
            self._update_kinetics_preview()

    def _create_roi(self, item, index):
        position = (
            item.center[0] - item.size[0] / 2,
            item.center[1] - item.size[1] / 2,
        )
        size = item.size[:2]
        roi_class = (
            pg.EllipseROI if item.kind in {"ellipsoid", "cylinder"} else pg.RectROI
        )
        roi = roi_class(position, size, pen=self._roi_pen(index, False))
        roi.sigRegionChangeFinished.connect(self._roi_changed)
        original_mouse_click = roi.mouseClickEvent

        def select_on_click(event, roi=roi, original_mouse_click=original_mouse_click):
            self._select_roi(roi)
            original_mouse_click(event)

        roi.mouseClickEvent = select_on_click
        self.canvas.addItem(roi)
        self._rois.append(roi)
        projection = pg.PlotDataItem(pen=self._projection_pen(index, False))
        projection.setZValue(500)
        self.canvas.addItem(projection)
        self._projection_items.append(projection)
        self._update_shape_projection(index)
        self._update_roi_highlights()

    def _roi_pen(self, index, selected):
        color = pg.intColor(index, hues=max(1, len(self.design.shapes)))
        color.setAlpha(145 if selected else 30)
        return pg.mkPen(
            color,
            width=1.5 if selected else 1,
            style=Qt.DashLine,
        )

    def _projection_pen(self, index, selected):
        color = pg.intColor(index, hues=max(1, len(self.design.shapes)), alpha=245)
        return pg.mkPen(color, width=4 if selected else 2)

    def _update_shape_projection(self, row):
        if not (0 <= row < len(self.design.shapes)) or not (
            0 <= row < len(self._projection_items)
        ):
            return
        fov_mm = tuple(widget.value() for widget in self.fov_spins)
        x, y = _shape_xy_projection(self.design.shapes[row], fov_mm)
        self._projection_items[row].setData(x, y)

    def _update_shape_projections(self):
        for row in range(len(self.design.shapes)):
            self._update_shape_projection(row)

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
        for index, projection in enumerate(self._projection_items):
            projection.setPen(self._projection_pen(index, index == selected_row))

    def _current_row(self):
        row = self.shape_list.currentRow()
        return row if 0 <= row < len(self.design.shapes) else None

    def _shape_selected(self, row):
        if self._updating or not (0 <= row < len(self.design.shapes)):
            return
        self._updating = True
        item = self.design.shapes[row]
        self._configure_shape_geometry_controls(item)
        self.shape_name.setText(item.name)
        self.kind_label.setText(item.kind)
        self.x_center.setValue(item.center[0] * 100.0)
        self.y_center.setValue(item.center[1] * 100.0)
        self.z_center.setValue(item.center[2] * 100.0)
        self.x_size.setValue(item.size[0] * 100.0)
        self.y_size.setValue(item.size[1] * 100.0)
        self.z_size.setValue(item.size[2] * 100.0)
        for widget, angle in zip(self.rotation_spins, item.rotation_deg):
            widget.setValue(float(angle))
        self.t1_ms.setValue(item.t1_s * 1000.0)
        self.initial_mz.setValue(item.initial_mz)
        self.b0_ppm.setValue(item.b0_ppm)
        self._populate_peaks(item)
        self._update_xy_info(row)
        self._updating = False
        self._set_kinetics_preview_shape(row)
        self._update_roi_highlights(row)
        self._update_shape_preview()
        self._update_kinetics_preview()

    def _configure_shape_geometry_controls(self, item):
        is_cylinder = item.kind == "cylinder"
        self.z_size_label.setText("Cylinder length" if is_cylinder else "Z size")
        self.z_size.setMaximum(1000.0 if is_cylinder else 100.0)

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
                self._update_shape_projection(row)
                self._update_shape_preview()
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
        z_size_mm = item.size[2] * self.fov_spins[2].value()
        self.xy_info.setText(
            f"centre=({center_mm[0]:.3g}, {center_mm[1]:.3g}) mm; "
            f"size=({size_mm[0]:.3g}, {size_mm[1]:.3g}) mm; "
            f"{'length' if item.kind == 'cylinder' else 'Z size'}="
            f"{z_size_mm:.3g} mm"
        )

    def _update_canvas_axes(self, *_):
        for name in ("bottom", "left"):
            axis = self.canvas.getAxis(name)
            axis.picture = None
            axis.update()
        row = self._current_row()
        if row is not None:
            self._update_xy_info(row)
        self._update_shape_projections()
        self._update_shape_preview()

    def _update_shape_preview(self):
        if not hasattr(self, "shape_preview_3d"):
            return
        fov_mm = tuple(widget.value() for widget in self.fov_spins)
        self.shape_preview_3d.set_shapes(
            self.design.shapes,
            fov_mm,
            selected_row=self._current_row(),
        )
        self._update_kinetics_spatial_preview()

    def _spectral_settings_changed(self, *_):
        if self._updating:
            return
        reference_ppm = self.spectral_reference_ppm.value()
        reference_delta_ppm = reference_ppm - self._last_spectral_reference_ppm
        if reference_delta_ppm:
            # Peak positions shown in the designer are absolute ppm values, while
            # the model stores offsets from the spectral reference. Rebase every
            # stored offset so changing the reference does not move any peak.
            for shape in self.design.shapes:
                for peak in shape.peaks:
                    peak.frequency_ppm -= reference_delta_ppm
            self.design.spectral_reference_ppm = reference_ppm
            self._last_spectral_reference_ppm = reference_ppm
            self._inspector_preview_dirty = True
            self._update_kinetics_preview()
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
        item.rotation_deg = tuple(widget.value() for widget in self.rotation_spins)
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
        self._update_shape_projection(row)
        if item.name != previous_name:
            self._refresh_kinetics_preview_shapes()
        self._update_shape_preview()
        self._update_kinetics_preview()

    def _populate_peaks(self, item):
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(len(item.peaks))
        for row, peak in enumerate(item.peaks):
            for column, value in enumerate(
                (
                    peak.name,
                    peak.amplitude,
                    (
                        ""
                        if peak.initial_polarization is None
                        else peak.initial_polarization
                    ),
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
            polarization_text = self.peak_table.item(peak_row, 2).text().strip()
            t1_text = self.peak_table.item(peak_row, 4).text().strip()
            peak = SpectralPeakDefinition(
                name=self.peak_table.item(peak_row, 0).text(),
                amplitude=float(self.peak_table.item(peak_row, 1).text()),
                frequency_ppm=(
                    float(self.peak_table.item(peak_row, 3).text()) - reference_ppm
                ),
                t2_star_s=float(self.peak_table.item(peak_row, 5).text()) / 1000.0,
                t1_s=None if not t1_text else float(t1_text) / 1000.0,
                initial_polarization=(
                    None if not polarization_text else float(polarization_text)
                ),
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

    def _populate_inflow_curve(self, rate_curve, polarization_curve, default):
        if rate_curve is None:
            samples = tuple(default)
        else:
            samples = tuple(
                (
                    time_s,
                    rate,
                    (
                        1.0
                        if polarization_curve is None
                        else polarization_curve.value_at(time_s)
                    ),
                )
                for time_s, rate in zip(rate_curve.times_s, rate_curve.values)
            )
        self.inflow_curve_table.setRowCount(len(samples))
        for row, values in enumerate(samples):
            for column, value in enumerate(values):
                self.inflow_curve_table.setItem(
                    row, column, QTableWidgetItem(str(value))
                )

    def _read_inflow_curves(self):
        samples = []
        for row in range(self.inflow_curve_table.rowCount()):
            values = tuple(
                float(self.inflow_curve_table.item(row, column).text())
                for column in range(3)
            )
            samples.append(values)
        if not samples:
            raise ValueError("enabled inflow requires at least one sample")
        times = tuple(item[0] for item in samples)
        return (
            TimeCurve(
                times_s=times,
                values=tuple(item[1] for item in samples),
                interpolation="step",
                outside="zero",
            ),
            TimeCurve(
                times_s=times,
                values=tuple(item[2] for item in samples),
                interpolation="step",
                outside="hold",
            ),
        )

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
            values = (
                previous_time + 1.0,
                *(
                    float(table.item(row - 1, column).text())
                    for column in range(1, table.columnCount())
                ),
            )
        else:
            values = (0.0,) + tuple(0.0 for _ in range(table.columnCount() - 1))
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
            value_items = tuple(
                table.item(row, column) for column in range(1, table.columnCount())
            )
            if time_item is None or any(item is None for item in value_items):
                return
            try:
                time_s = float(time_item.text())
            except ValueError:
                return
            rows.append((time_s, row, time_item, value_items))

        sorted_rows = sorted(rows, key=lambda item: item[0])
        if all(
            original_row == new_row
            for new_row, (_, original_row, *_) in enumerate(sorted_rows)
        ):
            return

        previous = table.blockSignals(True)
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                table.takeItem(row, column)
        new_edited_row = edited_row
        for new_row, (_, original_row, time_item, value_items) in enumerate(
            sorted_rows
        ):
            table.setItem(new_row, 0, time_item)
            for column, value_item in enumerate(value_items, start=1):
                table.setItem(new_row, column, value_item)
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

    def _update_kinetics_spatial_preview(self):
        if not hasattr(self, "kinetics_shape_preview_3d"):
            return
        shape_row = self.kinetics_preview_shape.currentData()
        selected_shape_row = None if shape_row is None else int(shape_row)
        region_row = self.kinetics_preview_region.currentData()
        highlighted_region = None
        region_name = "default kPL"
        try:
            if region_row is not None and int(region_row) >= 0:
                region_row = int(region_row)
                regions = self._read_kinetic_regions()
                highlighted_region = regions[region_row]
                region_name = highlighted_region.name or f"region {region_row + 1}"
        except (AttributeError, IndexError, TypeError, ValueError):
            highlighted_region = None
            region_name = "invalid kPL region"

        fov_mm = tuple(widget.value() for widget in self.fov_spins)
        self.kinetics_shape_preview_3d.set_shapes(
            self.design.shapes,
            fov_mm,
            selected_row=selected_shape_row,
            highlighted_region=highlighted_region,
        )
        if selected_shape_row is None or not (
            0 <= selected_shape_row < len(self.design.shapes)
        ):
            shape_name = "no shape selected"
        else:
            shape_name = self.design.shapes[selected_shape_row].name
        self.kinetics_spatial_preview_info.setText(
            f"Shape: {shape_name} · kPL source: {region_name}"
        )

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
        lactate_peak.initial_polarization = 0.0
        if self._current_row() == shape_row:
            self._populate_peaks(shape)
        self._update_kinetics_preview()

    def _update_kinetics_preview(self, *_):
        if self._updating or not hasattr(self, "pyruvate_preview_curve"):
            return
        self._update_kinetics_spatial_preview()
        if not self.dynamic_enabled.isChecked():
            self.inflow_preview_curve.setData([], [])
            self.pyruvate_preview_curve.setData([], [])
            self.lactate_preview_curve.setData([], [])
            self.pyruvate_polarization_curve.setData([], [])
            self.lactate_polarization_curve.setData([], [])
            self.inflow_sequence_start_line.setVisible(False)
            self.pool_sequence_start_line.setVisible(False)
            self.conversion_start_line.setVisible(False)
            self.kinetics_preview_info.setText(
                "Hyperpolarized preview inactive. Enable the hyperpolarized "
                "pyruvate/lactate model above to use concentration-resolved "
                "kinetics. With "
                "the model off, the conventional spectral phantom uses "
                "polarization=1 as thermal equilibrium."
            )
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
            spin_density = (
                pyruvate_peak.amplitude,
                lactate_peak.amplitude,
            )
            initial_polarization = (
                pyruvate_peak.effective_initial_polarization(shape.initial_mz),
                lactate_peak.effective_initial_polarization(shape.initial_mz),
            )
            initial_mz = (
                spin_density[0] * initial_polarization[0],
                spin_density[1] * initial_polarization[1],
            )
            t1_s = (
                pyruvate_peak.effective_t1_s(shape.t1_s),
                lactate_peak.effective_t1_s(shape.t1_s),
            )
            kpl_s_inv, kpl_label = self._preview_kpl()
            if self.inflow_enabled.isChecked():
                inflow_curve, inflow_polarization_curve = self._read_inflow_curves()
            else:
                inflow_curve = inflow_polarization_curve = None
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
            pools, concentrations = simulate_two_pool_kinetics(
                times_s,
                initial_mz,
                t1_s,
                kpl_s_inv,
                inflow_curve=inflow_curve,
                conversion_start_s=conversion_start_s,
                initial_time_s=preview_start_s,
                kinetics_time_offset_s=kinetics_time_offset_s,
                initial_concentration=spin_density,
                inflow_polarization_curve=inflow_polarization_curve,
                equilibrium_polarization=1.0,
                return_concentration=True,
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
            polarization = np.divide(
                pools,
                concentrations,
                out=np.zeros_like(pools),
                where=concentrations > 1e-15,
            )
            self.pyruvate_polarization_curve.setData(times_s, polarization[0])
            self.lactate_polarization_curve.setData(times_s, polarization[1])
            self.inflow_preview_plot.setXRange(preview_start_s, duration_s, padding=0.0)
            self.pool_preview_plot.setXRange(preview_start_s, duration_s, padding=0.0)
            self.polarization_preview_plot.setXRange(
                preview_start_s, duration_s, padding=0.0
            )
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
            self.polarization_preview_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            zero_index = int(np.searchsorted(times_s, 0.0))
            details = (
                f"Representative voxel in {shape.name} · kPL source: {kpl_label}. "
                f"kinetics t at sequence t=0: {kinetics_time_offset_s:.4g} s; "
                f"kPL={kpl_s_inv:.4g} s⁻¹ from sequence "
                f"t={sequence_conversion_start_s:.4g} s "
                f"(kinetics t={conversion_start_s:.4g} s), "
                f"T1(P/L)=({t1_s[0]:.4g}/{t1_s[1]:.4g}) s, "
                f"spin density(P/L)=({spin_density[0]:.4g}/{spin_density[1]:.4g}), "
                f"initial polarization(P/L)=({initial_polarization[0]:.4g}/"
                f"{initial_polarization[1]:.4g}), "
                f"initial Mz=C×P at sequence t={preview_start_s:.4g} s="
                f"({initial_mz[0]:.4g}/{initial_mz[1]:.4g}), "
                f"sequence-start Mz at t=0="
                f"({pools[0, zero_index]:.4g}/{pools[1, zero_index]:.4g}), "
                f"Mz({duration_s:.4g} s)=({pools[0, -1]:.4g}/"
                f"{pools[1, -1]:.4g}), polarization="
                f"({polarization[0, -1]:.4g}/{polarization[1, -1]:.4g})."
            )
            explanations = []
            if kpl_s_inv == 0.0:
                explanations.append(
                    "kPL=0: no P→L conversion; each existing pool only follows its "
                    "own T1 relaxation toward polarization 1, while enabled inflow "
                    "is added only to Pyruvate."
                )
                if initial_mz[1] > 0:
                    explanations.append(
                        "Lz starts above zero because both Lactate spin density and "
                        "initial polarization are non-zero; this lactate was "
                        "initialized, not created by conversion."
                    )
                else:
                    explanations.append(
                        "Lz starts at zero and recovers thermally toward "
                        "Mz=concentration (polarization 1), even when kPL=0."
                    )
            elif initial_mz[1] == 0.0:
                explanations.append(
                    "Lz starts at zero; kPL conversion adds Pyruvate-derived "
                    "Lactate magnetization alongside thermal recovery toward "
                    "polarization 1."
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
            self.pyruvate_polarization_curve.setData([], [])
            self.lactate_polarization_curve.setData([], [])
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
            "XY drag/resize · values left · live 3D preview"
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
        self.canvas.removeItem(self._projection_items.pop(row))
        self.design.shapes.pop(row)
        self.shape_list.takeItem(row)
        self._refresh_kinetics_preview_shapes()
        if self.design.shapes:
            self.shape_list.setCurrentRow(min(row, len(self.design.shapes) - 1))
        self._update_roi_highlights()
        self._update_shape_preview()
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
        self.design.supersampling_enabled = self.supersampling_enabled.isChecked()
        self.design.supersampling_factor = self.supersampling_factor.value()
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
        if self.inflow_enabled.isChecked():
            (
                self.design.pyruvate_inflow_curve,
                self.design.pyruvate_inflow_polarization_curve,
            ) = self._read_inflow_curves()
        else:
            self.design.pyruvate_inflow_curve = None
            self.design.pyruvate_inflow_polarization_curve = None
        self.design.dynamic_b0_curve = (
            self._read_time_curve(self.dynamic_b0_curve_table, outside="hold")
            if self.dynamic_b0_enabled.isChecked()
            else None
        )

    def _tab_changed(self, index):
        if self.tabs.widget(index) is self.inspector:
            if self.phantom is None or self._inspector_preview_dirty:
                self._refresh_inspector_preview()
        else:
            # Any edit is made outside the inspector. Rebuilding on the next
            # visit keeps the large preview current without rasterizing on
            # every spin-box step or ROI drag.
            self._inspector_preview_dirty = True

    def _refresh_inspector_preview(self):
        if self._building_inspector_preview:
            return False
        self._building_inspector_preview = True
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
            self._inspector_preview_dirty = False
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Invalid phantom", str(exc))
            return False
        finally:
            self._building_inspector_preview = False

    def _preview(self):
        if self._refresh_inspector_preview():
            self.tabs.setCurrentWidget(self.inspector)

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
