"""Interactive shape and Lorentz-peak phantom designer."""

from __future__ import annotations

from dataclasses import replace
import html
import os
from pathlib import Path
from typing import Optional
import weakref

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QLocale, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QStyledItemDelegate,
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
from ..default_phantoms import DEFAULT_SPECTRAL_BANDWIDTH_PPM, DEFAULT_SPECTRAL_POINTS
from ..paths import workspace_directory
from ..metabolite_presets import MetabolitePreset, metabolite_presets
from ..phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from ..sequence.snr_budget import format_amount
from ..spectral_phantom import SpectralPhantom
from ..units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm, thermal_polarization
from .volume_viewer import PhantomInspectorWidget
from .sequence_geometry_viewer import (
    OrientedBox,
    projected_outline_mm,
    sequence_overhang_m,
    sequence_overhang_note,
)
from .default_settings import WorkspaceDefaults
from .styles import BOLD_GROUP_TITLES_STYLE
from .widgets import ExplicitAcceptDialog

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    gl = None
    HAS_OPENGL = False


# Typical hyperpolarized [1-13C]pyruvate → lactate rate used when conversion
# is switched on for shapes that do not convert yet.
DEFAULT_ENABLED_KPL_S_INV = 0.05

# Phantom cells per sequence pixel set by the "From seq." geometry buttons.
SEQUENCE_GRID_OVERSAMPLING = 2


def sequence_axis_grids(geometry) -> list:
    """Return the sequence FOV and matrix along scanner x, y and z.

    Each entry is ``(fov_m, matrix)`` when exactly one encoded direction runs
    along that scanner axis, otherwise a sentence saying why the axis has no
    sequence grid: the slice direction of a 2D sequence is not encoded, and
    an oblique direction spans several scanner axes.
    """
    if geometry is None or geometry.fov_m is None or geometry.matrix is None:
        return ["The current sequence has no encoded FOV."] * 3
    frame = np.asarray(geometry.frame, dtype=float)
    encoded = min(len(geometry.fov_m), len(geometry.matrix))
    grids = []
    for axis in range(3):
        directions = np.flatnonzero(np.abs(frame[axis]) > 1e-6)
        if directions.size != 1:
            grids.append("The sequence is oblique to this axis.")
            continue
        direction = int(directions[0])
        if direction >= encoded:
            grids.append(
                "This is the slice direction of the 2D sequence, which is not "
                "encoded. Set it by hand so that a slice spans several cells."
            )
            continue
        grids.append(
            (float(geometry.fov_m[direction]), int(geometry.matrix[direction]))
        )
    return grids


# Shape volume per product of the three extents.
_SHAPE_VOLUME_FACTOR = {"ellipsoid": np.pi / 6.0, "box": 1.0, "cylinder": np.pi / 4.0}
_AMOUNT_GRID_LIMIT = 96


class SuffixNumberDelegate(QStyledItemDelegate):
    """Show numeric table cells with a unit suffix but edit the bare number."""

    def __init__(self, suffix, parent=None, minimum=None):
        super().__init__(parent)
        self._suffix = suffix
        self._minimum = minimum

    def displayText(self, value, locale):
        try:
            return f"{float(value):g}{self._suffix}"
        except (TypeError, ValueError):
            return super().displayText(value, locale)

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator(editor)
        validator.setLocale(QLocale.c())
        validator.setNotation(QDoubleValidator.ScientificNotation)
        if self._minimum is not None:
            validator.setBottom(float(self._minimum))
        editor.setValidator(validator)
        return editor


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


class ScaleOnlyEllipseROI(pg.EllipseROI):
    """Ellipse ROI whose handle edits size without an independent 2D rotation."""

    def _addHandles(self):
        # EllipseROI normally adds both a scale handle and a rotation handle.
        # Shape rotation belongs to the explicit X/Y/Z controls in the designer;
        # an extra ROI-only angle would desynchronise the editable outline from
        # the model geometry.
        diagonal = 0.5 * 2.0**-0.5 + 0.5
        self.addScaleHandle([diagonal, diagonal], [0.5, 0.5])


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
    return _shape_projection(item, fov_mm, (0, 1))


def _shape_projection(item, fov_mm, axes):
    """Return the rotated primitive's silhouette on two axes in [0, 1]."""
    fov = np.asarray(fov_mm, dtype=float)
    vertices, _faces = _shape_preview_mesh(item, fov)
    scale = max(float(np.max(fov)), 1e-12)
    projected = np.column_stack(
        tuple(0.5 + vertices[:, axis] * scale / (2.0 * fov[axis]) for axis in axes)
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
        if phantom.metadata.get("phantom_family") == "mouse_perfusion":
            from ..mouse_phantom import MousePerfusionPhantom

            phantom = MousePerfusionPhantom.from_dynamic_phantom(phantom)
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


class SpectralPhantomDesignerDialog(ExplicitAcceptDialog):
    """Compose rotatable 3D primitives and assign Lorentzian peak lists."""

    def __init__(
        self,
        parent=None,
        design: Optional[PhantomDesign] = None,
        settings=None,
        sequence_reference_ppm: Optional[float] = None,
        sequence_geometry=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Spectral Phantom" if design is not None else "New Spectral Phantom"
        )
        self.resize(1450, 900)
        if design is None:
            defaults = WorkspaceDefaults.from_settings(settings)
            design = PhantomDesign(
                shape=(64, 64, 64),
                fov_m=(
                    defaults.phantom_fov_mm[0] / 1000.0,
                    defaults.phantom_fov_mm[1] / 1000.0,
                    0.032,
                ),
                nucleus=defaults.phantom_nucleus,
                field_strength_t=defaults.field_strength_t,
                spectral_bandwidth_ppm=DEFAULT_SPECTRAL_BANDWIDTH_PPM,
                spectral_points=DEFAULT_SPECTRAL_POINTS,
                shapes=[
                    ShapeDefinition(
                        name="Shape 1",
                        kind="cylinder",
                        peaks=[SpectralPeakDefinition(t2_star_s=0.020)],
                    )
                ],
            )
            if sequence_reference_ppm is not None:
                design.spectral_reference_ppm = float(sequence_reference_ppm)
        self.design = design
        self.sequence_reference_ppm = float(
            design.spectral_reference_ppm
            if sequence_reference_ppm is None
            else sequence_reference_ppm
        )
        self.phantom = None
        self._updating = False
        self._last_spectral_reference_ppm = float(design.spectral_reference_ppm)
        self._inspector_preview_dirty = True
        self._building_inspector_preview = False
        self._suppress_default_kpl = False
        self._b0_shape_cache = None
        self._rois = []
        self._projection_items = []
        self._sequence_geometry = None
        self._build_ui()
        self._load_design_into_ui()
        self.set_sequence_geometry(sequence_geometry)

    def _build_ui(self):
        self.setStyleSheet(
            BOLD_GROUP_TITLES_STYLE
            + "QListWidget#phantomShapeList::item { padding: 5px 7px; }"
            "QListWidget#phantomShapeList::item:selected {"
            " background-color: palette(highlight);"
            " color: palette(highlighted-text);"
            " font-weight: bold;"
            " border: 2px solid palette(highlight);"
            "}"
        )
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        draw_page = QWidget()
        draw_layout = QVBoxLayout(draw_page)
        name_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        name_row.addWidget(QLabel("Name"))
        name_row.addWidget(self.name_edit)
        draw_layout.addLayout(name_row)

        configuration_row = QHBoxLayout()
        configuration_row.setSpacing(10)

        geometry_group = QGroupBox("Phantom geometry")
        geometry_group.setMinimumWidth(350)
        geometry_layout = QVBoxLayout(geometry_group)
        geometry_grid = QGridLayout()
        self.phantom_geometry_grid = geometry_grid
        geometry_grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        geometry_grid.setHorizontalSpacing(10)
        geometry_grid.setVerticalSpacing(6)
        self.matrix_spins = []
        self.fov_spins = []
        self.sequence_grid_buttons = []
        matrix_heading = QLabel("Matrix")
        matrix_heading.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(matrix_heading, 0, 1)
        fov_heading = QLabel("FOV")
        fov_heading.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(fov_heading, 0, 2)
        for row, axis in enumerate("XYZ", start=1):
            matrix = QSpinBox()
            matrix.setRange(1, 1025)
            matrix.setMinimumWidth(70)
            self.matrix_spins.append(matrix)
            fov = QDoubleSpinBox()
            fov.setRange(0.01, 1000.0)
            fov.setDecimals(3)
            fov.setSuffix(" mm")
            fov.setMinimumWidth(115)
            self.fov_spins.append(fov)
            axis_label = QLabel(axis)
            axis_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            from_sequence = QPushButton("From seq.")
            from_sequence.setObjectName(f"phantom_{axis.lower()}_from_sequence")
            from_sequence.clicked.connect(
                lambda _checked=False, index=row - 1: self._copy_sequence_axis(index)
            )
            self.sequence_grid_buttons.append(from_sequence)
            geometry_grid.addWidget(axis_label, row, 0)
            geometry_grid.addWidget(matrix, row, 1)
            geometry_grid.addWidget(fov, row, 2)
            geometry_grid.addWidget(from_sequence, row, 3)
        geometry_layout.addLayout(geometry_grid)

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
        geometry_layout.addLayout(sampling_row)
        self.object_amount_info = QLabel()
        self.object_amount_info.setObjectName("phantom_object_amount_info")
        self.object_amount_info.setWordWrap(True)
        self.object_amount_info.setToolTip(
            "Volume covered by any shape and the observed nuclei it initially "
            "holds: Σ concentration × shape volume × isotope enrichment. "
            "Calibrated receiver noise and physical signal units use this amount."
        )
        geometry_layout.addWidget(self.object_amount_info)
        self.simulation_region_summary = QLabel()
        self.simulation_region_summary.setObjectName(
            "phantom_simulation_region_summary"
        )
        self.simulation_region_summary.setWordWrap(True)
        self.simulation_region_summary.setStyleSheet("color: #2563eb;")
        self.simulation_region_summary.hide()
        geometry_layout.addWidget(self.simulation_region_summary)
        # Keep the compact controls together at the top of the group.  Without
        # an explicit trailing stretch, the configuration row's height is
        # distributed between the matrix and supersampling layouts.
        geometry_layout.addStretch(1)
        configuration_row.addWidget(geometry_group, 0)

        spectral_group = QGroupBox("Spectral settings and preview")
        spectral_group_layout = QHBoxLayout(spectral_group)
        spectral_fields = QGridLayout()
        spectral_fields.setHorizontalSpacing(8)
        spectral_fields.setVerticalSpacing(6)
        self.field_strength_t = self._number_spin(0.001, 1000.0, 3.0, " T")
        self.field_strength_t.setToolTip(
            "Main field used to convert ppm peak and B0 offsets to Hz"
        )
        self.nucleus = QComboBox()
        self.nucleus.addItem("Auto (H1 static / C13 dynamic)", None)
        for nucleus in sorted(NUCLEUS_GAMMA_HZ_PER_T):
            self.nucleus.addItem(nucleus, nucleus)
        self.nucleus.setToolTip(
            "Nucleus used together with B0 for ppm-to-Hz conversion; Auto "
            "preserves the H1 static and C13 dynamic defaults"
        )
        self.spectral_reference_ppm = self._number_spin(-10000.0, 10000.0, 0.0, " ppm")
        self.spectral_reference_ppm.setToolTip(
            "Internal storage origin retained for backwards-compatible phantom files."
        )
        self.spectral_reference_ppm.hide()
        self.sequence_reference_display = self._number_spin(
            -10000.0, 10000.0, self.sequence_reference_ppm, " ppm"
        )
        self.sequence_reference_display.setReadOnly(True)
        self.sequence_reference_display.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.sequence_reference_display.setToolTip(
            "Read-only sequence zero frequency from Sequence Simulation. "
            "The orange spectrum marker uses this value."
        )
        self.spectral_window_center_ppm = self._number_spin(
            -10000.0, 10000.0, 0.0, " ppm"
        )
        self.spectral_window_center_ppm.setToolTip(
            "Absolute ppm center of the sampled spectral window. This is independent "
            "of the sequence reference/carrier."
        )
        self.spectral_bandwidth_ppm = self._number_spin(
            0.0001, 10000.0, DEFAULT_SPECTRAL_BANDWIDTH_PPM, " ppm"
        )
        self.spectral_bandwidth_ppm.setToolTip(
            "Full width of the sampled spectral window around Spectrum centre."
        )
        self.spectral_points = QSpinBox()
        self.spectral_points.setRange(2, 65536)
        self.spectral_points.setValue(DEFAULT_SPECTRAL_POINTS)
        self.spectral_points.setToolTip(
            "Default number of samples in spectral previews"
        )
        self.spectral_resolution_info = QLabel()
        self.temperature_c = self._number_spin(-273.0, 1000.0, 37.0, " °C")
        self.temperature_c.setDecimals(2)
        self.temperature_c.setObjectName("phantom_temperature_c")
        self.temperature_c.setToolTip(
            "Sample temperature. With nucleus and B0 it sets the thermal "
            "polarization that 1 × thermal stands for; receiver calibration "
            "uses it to convert the signal to physical units."
        )
        self.thermal_polarization_info = QLabel()
        self.thermal_polarization_info.setWordWrap(True)
        self.isotope_enrichment = self._number_spin(0.01, 100.0, 100.0, " %")
        self.isotope_enrichment.setDecimals(2)
        self.isotope_enrichment.setObjectName("phantom_isotope_enrichment")
        self.isotope_enrichment.setToolTip(
            "Fraction of the molecules that carry the observed isotope at the "
            "labelled position, e.g. 99 % for [1-13C]pyruvate or 1.1 % for "
            "natural-abundance 13C. Concentrations count molecules; calibrated "
            "receiver noise and physical signal units scale them by this value."
        )

        spectral_fields.addWidget(QLabel("B0"), 0, 0)
        spectral_fields.addWidget(self.field_strength_t, 0, 1)
        spectral_fields.addWidget(QLabel("Nucleus"), 0, 2)
        spectral_fields.addWidget(self.nucleus, 0, 3)
        spectral_fields.addWidget(QLabel("Sequence reference"), 1, 0)
        spectral_fields.addWidget(self.sequence_reference_display, 1, 1)
        spectral_fields.addWidget(QLabel("Spectrum centre"), 1, 2)
        spectral_fields.addWidget(self.spectral_window_center_ppm, 1, 3)
        spectral_fields.addWidget(QLabel("Bandwidth"), 2, 0)
        spectral_fields.addWidget(self.spectral_bandwidth_ppm, 2, 1)
        spectral_fields.addWidget(QLabel("Points"), 2, 2)
        spectral_fields.addWidget(self.spectral_points, 2, 3)

        spectral_fields.addWidget(QLabel("Temperature"), 3, 0)
        spectral_fields.addWidget(self.temperature_c, 3, 1)
        spectral_fields.addWidget(self.thermal_polarization_info, 3, 2, 1, 2)
        spectral_fields.addWidget(QLabel("Isotope enrichment"), 4, 0)
        spectral_fields.addWidget(self.isotope_enrichment, 4, 1)
        spectral_fields.addWidget(self.spectral_resolution_info, 4, 2, 1, 2)
        # Absorb spare vertical space below the resolution instead of between
        # the field rows, so the resolution reads as part of the settings.
        spectral_fields.setRowStretch(5, 1)
        spectral_group_layout.addLayout(spectral_fields, 0)

        spectral_preview_layout = QVBoxLayout()
        self.spectral_preview_plot = pg.PlotWidget(title="Selected shape spectrum")
        self.spectral_preview_plot.setMinimumSize(340, 105)
        self.spectral_preview_plot.setMaximumHeight(135)
        self.spectral_preview_plot.setLabel("bottom", "Frequency", units="ppm")
        self.spectral_preview_plot.setLabel("left", "Amplitude", units="a.u.")
        self.spectral_preview_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spectral_preview_curve = self.spectral_preview_plot.plot(
            pen=pg.mkPen("c", width=2)
        )
        self.spectral_reference_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen((255, 150, 60), width=2, style=Qt.DashLine),
        )
        self.spectral_reference_line.setZValue(100)
        self.spectral_preview_plot.addItem(self.spectral_reference_line)
        spectral_preview_layout.addWidget(self.spectral_preview_plot)
        spectral_preview_layout.addStretch(1)
        spectral_group_layout.addLayout(spectral_preview_layout, 1)

        self.spectral_window_center_ppm.valueChanged.connect(
            self._spectral_settings_changed
        )
        self.field_strength_t.valueChanged.connect(self._spectral_settings_changed)
        self.nucleus.currentIndexChanged.connect(self._spectral_settings_changed)
        self.nucleus.currentIndexChanged.connect(self._refresh_metabolite_presets)
        self.spectral_bandwidth_ppm.valueChanged.connect(
            self._spectral_settings_changed
        )
        self.spectral_points.valueChanged.connect(self._spectral_settings_changed)
        self.temperature_c.valueChanged.connect(self._spectral_settings_changed)
        self.isotope_enrichment.valueChanged.connect(self._update_object_amount_info)
        self.nucleus.currentIndexChanged.connect(self._update_object_amount_info)
        for spin in (*self.matrix_spins, *self.fov_spins):
            spin.valueChanged.connect(self._update_object_amount_info)
        configuration_row.addWidget(spectral_group, 1)
        draw_layout.addLayout(configuration_row)

        self.shape_splitter = QSplitter(Qt.Horizontal)
        splitter = self.shape_splitter
        draw_layout.addWidget(splitter)

        shape_panel = QWidget()
        shape_panel.setMinimumWidth(360)
        shape_panel.setMaximumWidth(440)
        shape_layout = QVBoxLayout(shape_panel)
        shape_heading = QLabel("Shapes")
        shape_layout.addWidget(shape_heading)
        self.shape_list = QListWidget()
        self.shape_list.setObjectName("phantomShapeList")
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
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_shape)
        shape_buttons.addWidget(add_ellipse, 0, 0, 1, 2)
        shape_buttons.addWidget(add_box, 1, 0, 1, 2)
        shape_buttons.addWidget(add_cylinder, 2, 0, 1, 2)
        shape_buttons.addWidget(remove, 3, 0, 1, 2)
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
        geometry_group.setMinimumHeight(275)
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
        for widget in (
            self.x_center,
            self.y_center,
            self.z_center,
            self.x_size,
            self.y_size,
            self.z_size,
            *self.rotation_spins,
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
            widget.setMinimumWidth(88)
            widget.setMaximumWidth(110)
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

        peak_explanation = QLabel(
            "<b>Concentration</b>: how much of the metabolite is present in mM "
            "of molecules (0 = none); the isotope enrichment above sets how many "
            "carry the observed nucleus. Only calibrated receiver noise and "
            "physical signal units use the absolute value; otherwise only the "
            "ratios matter. <b>Initial polarization</b>: its starting polarization in "
            "multiples of thermal equilibrium (1 × thermal = not hyperpolarized, "
            "e.g. 10000 × thermal for hyperpolarized pyruvate). The initial signal "
            "is concentration × polarization; T1 relaxes the polarization back to "
            "1 × thermal. Presets add one Lorentzian peak; add peaks for "
            "multiplets."
        )
        peak_explanation.setWordWrap(True)
        property_layout.addWidget(peak_explanation)
        self.peak_table = QTableWidget(0, 6)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Concentration\n(mM)",
                "Initial\npolarization",
                "Peak\n(ppm)",
                "T1\n(ms)",
                "T2\n(ms)",
            ]
        )
        for column, tooltip in enumerate(
            (
                "Metabolite name; the kinetic model matches pyruvate and lactate "
                "by this name",
                "Concentration of the metabolite in this shape in mM of molecules "
                "(spin density). Overlapping shapes add up; 0 means none is "
                "present initially.",
                "Starting longitudinal polarization in multiples of the thermal "
                "equilibrium value. 1 × thermal is a conventional sample; "
                "hyperpolarized substrates use e.g. 10000 × thermal.",
                "Absolute chemical shift of the peak",
                "Longitudinal relaxation time; polarization relaxes toward "
                "1 × thermal with this time constant",
                "Transverse relaxation time; also sets the Lorentzian line width "
                "1/(π T2)",
            )
        ):
            self.peak_table.horizontalHeaderItem(column).setToolTip(tooltip)
        self._polarization_delegate = SuffixNumberDelegate(
            " × thermal", self.peak_table, minimum=0.0
        )
        self.peak_table.setItemDelegateForColumn(2, self._polarization_delegate)
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
        metabolite_row = QHBoxLayout()
        metabolite_row.addWidget(QLabel("Common metabolite"))
        self.metabolite_preset_combo = QComboBox()
        self.metabolite_preset_combo.setObjectName("metabolitePresetCombo")
        self.metabolite_preset_combo.currentIndexChanged.connect(
            self._update_metabolite_preset_info
        )
        metabolite_row.addWidget(self.metabolite_preset_combo, 1)
        self.add_metabolite_preset_button = QPushButton("Add selected")
        self.add_metabolite_preset_button.setObjectName("addMetabolitePresetButton")
        self.add_metabolite_preset_button.clicked.connect(self._add_metabolite_preset)
        metabolite_row.addWidget(self.add_metabolite_preset_button)
        property_layout.addLayout(metabolite_row)
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
        self.region_page = self._build_region_page()
        self.tabs.addTab(self.region_page, "Simulated region")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.region_page),
            "Simulate only part of the phantom: a smaller FOV around the "
            "isocentre and/or a ppm range of its peaks.",
        )

        self.tabs.addTab(self._build_b0_page(), "B0")
        self.tabs.addTab(self._build_kinetics_page(), "Kinetics / kPL")

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

    def _build_region_page(self):
        """Controls that crop the simulated part of the designed phantom."""
        page = QWidget()
        layout = QHBoxLayout(page)
        controls = QWidget()
        controls.setMinimumWidth(380)
        controls.setMaximumWidth(480)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "'Draw and edit' defines the whole phantom. Here you choose which "
            "part of it the sequence simulates, e.g. a few slices of a long "
            "cylinder. The simulated FOV is centred on the isocentre and keeps "
            "the phantom's voxel size."
        )
        intro.setWordWrap(True)
        controls_layout.addWidget(intro)

        spatial_group = QGroupBox("Spatial region")
        spatial_layout = QVBoxLayout(spatial_group)
        self.simulation_fov_enabled = QCheckBox("Simulate only a region of the phantom")
        self.simulation_fov_enabled.setObjectName("phantom_simulation_fov_enabled")
        spatial_layout.addWidget(self.simulation_fov_enabled)
        region_grid = QGridLayout()
        region_grid.setHorizontalSpacing(10)
        region_grid.addWidget(QLabel("Simulated FOV"), 0, 1)
        position_heading = QLabel("Phantom position")
        position_heading.setToolTip(
            "Centre of the phantom relative to the simulated FOV centre "
            "(the isocentre), like the mouse position of the mouse phantom"
        )
        region_grid.addWidget(position_heading, 0, 2)
        self.simulation_fov_spins = []
        self.simulation_offset_spins = []
        for row, axis in enumerate("XYZ", start=1):
            fov = QDoubleSpinBox()
            fov.setRange(0.01, 1000.0)
            fov.setDecimals(3)
            fov.setSuffix(" mm")
            fov.setObjectName(f"phantom_simulation_fov_{axis.lower()}")
            offset = QDoubleSpinBox()
            offset.setRange(-1000.0, 1000.0)
            offset.setDecimals(3)
            offset.setSuffix(" mm")
            offset.setObjectName(f"phantom_simulation_offset_{axis.lower()}")
            self.simulation_fov_spins.append(fov)
            self.simulation_offset_spins.append(offset)
            label = QLabel(axis)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            region_grid.addWidget(label, row, 0)
            region_grid.addWidget(fov, row, 1)
            region_grid.addWidget(offset, row, 2)
        spatial_layout.addLayout(region_grid)
        self.fit_simulation_fov_button = QPushButton("Fit to sequence FOV")
        self.fit_simulation_fov_button.setObjectName("phantom_fit_simulation_fov")
        self.fit_simulation_fov_button.setToolTip(
            "Set the simulated FOV to the smallest box around the isocentre "
            "that holds the sequence FOV, its offsets and its slice package. "
            "The slice direction keeps half a slice thickness on each side "
            "for the slice-profile transition."
        )
        self.fit_simulation_fov_button.clicked.connect(self._fit_simulation_fov)
        spatial_layout.addWidget(self.fit_simulation_fov_button)
        self.simulation_region_info = QLabel()
        self.simulation_region_info.setObjectName("phantom_simulation_region_info")
        self.simulation_region_info.setWordWrap(True)
        spatial_layout.addWidget(self.simulation_region_info)
        self.simulation_fov_warning = QLabel()
        self.simulation_fov_warning.setObjectName("phantom_simulation_fov_warning")
        self.simulation_fov_warning.setWordWrap(True)
        self.simulation_fov_warning.setStyleSheet(
            "color: #78350f; background: #fef3c7; border: 1px solid #f59e0b; "
            "border-radius: 4px; padding: 4px;"
        )
        self.simulation_fov_warning.hide()
        spatial_layout.addWidget(self.simulation_fov_warning)
        controls_layout.addWidget(spatial_group)

        spectral_group = QGroupBox("Spectral range")
        spectral_layout = QVBoxLayout(spectral_group)
        self.simulation_ppm_enabled = QCheckBox(
            "Simulate only peaks inside a ppm range"
        )
        self.simulation_ppm_enabled.setObjectName("phantom_simulation_ppm_enabled")
        spectral_layout.addWidget(self.simulation_ppm_enabled)
        ppm_row = QHBoxLayout()
        self.simulation_ppm_low = self._number_spin(-10000.0, 10000.0, 0.0, " ppm")
        self.simulation_ppm_high = self._number_spin(-10000.0, 10000.0, 0.0, " ppm")
        self.simulation_ppm_low.setObjectName("phantom_simulation_ppm_low")
        self.simulation_ppm_high.setObjectName("phantom_simulation_ppm_high")
        ppm_row.addWidget(QLabel("From"))
        ppm_row.addWidget(self.simulation_ppm_low, 1)
        ppm_row.addWidget(QLabel("to"))
        ppm_row.addWidget(self.simulation_ppm_high, 1)
        spectral_layout.addLayout(ppm_row)
        self.fit_ppm_range_button = QPushButton("Use spectrum window")
        self.fit_ppm_range_button.setToolTip(
            "Set the range to the sampled spectral window (Spectrum centre ± "
            "bandwidth / 2) of 'Draw and edit'"
        )
        self.fit_ppm_range_button.clicked.connect(self._fit_simulation_ppm_range)
        spectral_layout.addWidget(self.fit_ppm_range_button)
        self.simulation_ppm_info = QLabel()
        self.simulation_ppm_info.setObjectName("phantom_simulation_ppm_info")
        self.simulation_ppm_info.setWordWrap(True)
        self.simulation_ppm_info.setTextFormat(Qt.RichText)
        spectral_layout.addWidget(self.simulation_ppm_info)
        controls_layout.addWidget(spectral_group)
        controls_layout.addStretch(1)
        layout.addWidget(controls, 0)

        previews = QGridLayout()
        self.region_plots = []
        self._region_items = []
        for index, (horizontal, vertical) in enumerate(((0, 1), (0, 2), (1, 2))):
            plot = pg.PlotWidget(title=f"{'XYZ'[horizontal]}–{'XYZ'[vertical]}")
            plot.setAspectLocked(True)
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setLabel("bottom", "xyz"[horizontal], units="mm")
            plot.setLabel("left", "xyz"[vertical], units="mm")
            plot.setMinimumSize(220, 200)
            self.region_plots.append(((horizontal, vertical), plot))
            previews.addWidget(plot, index // 2, index % 2)
        self.region_spectrum_plot = pg.PlotWidget(title="Peaks")
        self.region_spectrum_plot.setLabel("bottom", "Frequency", units="ppm")
        self.region_spectrum_plot.hideAxis("left")
        self.region_spectrum_plot.setMinimumSize(220, 200)
        self.simulation_ppm_region = pg.LinearRegionItem(
            brush=pg.mkBrush(80, 200, 120, 50)
        )
        self.simulation_ppm_region.setZValue(-10)
        self.simulation_ppm_region.sigRegionChangeFinished.connect(
            self._simulation_ppm_region_dragged
        )
        self.region_spectrum_plot.addItem(self.simulation_ppm_region)
        self._region_peak_items = []
        previews.addWidget(self.region_spectrum_plot, 1, 1)
        legend = QLabel(
            "<span style='color:#f0f0f0'>■</span> whole phantom &nbsp; "
            "<span style='color:#50c878'>■</span> simulated FOV &nbsp; "
            "<span style='color:#ff9632'>■</span> sequence FOV"
        )
        legend.setTextFormat(Qt.RichText)
        preview_column = QVBoxLayout()
        preview_column.addLayout(previews, 1)
        preview_column.addWidget(legend)
        layout.addLayout(preview_column, 1)

        self.simulation_fov_enabled.toggled.connect(self._simulation_region_changed)
        self.simulation_ppm_enabled.toggled.connect(self._simulation_region_changed)
        for spin in (*self.simulation_fov_spins, *self.simulation_offset_spins):
            spin.valueChanged.connect(self._simulation_region_changed)
        for spin in (self.simulation_ppm_low, self.simulation_ppm_high):
            spin.valueChanged.connect(self._simulation_region_changed)
        for spin in (*self.matrix_spins, *self.fov_spins):
            spin.valueChanged.connect(self._update_simulation_region_summary)
        return page

    def _build_b0_page(self):
        b0_page = QWidget()
        b0_page_layout = QHBoxLayout(b0_page)
        b0_splitter = QSplitter(Qt.Horizontal)
        b0_splitter.setChildrenCollapsible(False)
        self.b0_splitter = b0_splitter
        b0_controls = QWidget()
        b0_controls.setMinimumWidth(430)
        b0_layout = QVBoxLayout(b0_controls)
        b0_intro = QLabel(
            "The B0 offset of each voxel is the constant offset of the shape it "
            "belongs to plus the spatial inhomogeneity pattern (plus an optional "
            "uniform drift over time). All offsets shift every peak in that voxel."
        )
        b0_intro.setWordWrap(True)
        b0_layout.addWidget(b0_intro)

        inhomogeneity_group = QGroupBox("Spatial B0 inhomogeneity")
        inhomogeneity_layout = QVBoxLayout(inhomogeneity_group)
        inhomogeneity_form = QFormLayout()
        inhomogeneity_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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
        inhomogeneity_form.addRow("Pattern", self.b0_mode_combo)
        self.b0_inhomogeneity_ppm = self._number_spin(-1000.0, 1000.0, 0.0, " ppm")
        self.b0_inhomogeneity_ppm.setToolTip(
            "Strength of the pattern; the text below states the resulting range"
        )
        inhomogeneity_form.addRow("Strength", self.b0_inhomogeneity_ppm)
        center_row = QHBoxLayout()
        self.b0_center_spins = []
        for axis in "XYZ":
            spin = self._number_spin(-50.0, 150.0, 50.0, " %")
            spin.setDecimals(1)
            spin.setPrefix(f"{axis} ")
            spin.setToolTip(
                f"Pattern centre along {axis} in percent of the FOV (50 % = FOV "
                "centre). Drag the crosshair in the maps to move it."
            )
            self.b0_center_spins.append(spin)
            center_row.addWidget(spin)
        self.b0_center_reset = QPushButton("Reset")
        self.b0_center_reset.setToolTip("Move the pattern centre to the FOV centre")
        self.b0_center_reset.clicked.connect(self._reset_b0_center)
        center_row.addWidget(self.b0_center_reset)
        inhomogeneity_form.addRow("Centre", center_row)
        inhomogeneity_layout.addLayout(inhomogeneity_form)
        self.b0_pattern_info = QLabel()
        self.b0_pattern_info.setWordWrap(True)
        inhomogeneity_layout.addWidget(self.b0_pattern_info)
        self.b0_mode_combo.currentIndexChanged.connect(self._b0_settings_changed)
        self.b0_inhomogeneity_ppm.valueChanged.connect(self._b0_settings_changed)
        for spin in self.b0_center_spins:
            spin.valueChanged.connect(self._b0_settings_changed)
        b0_layout.addWidget(inhomogeneity_group)

        offsets_group = QGroupBox("Constant B0 offset per shape")
        offsets_layout = QVBoxLayout(offsets_group)
        offsets_help = QLabel(
            "Added to the inhomogeneity inside each shape, e.g. for susceptibility "
            "differences between compartments. Later shapes overwrite earlier ones "
            "where they overlap."
        )
        offsets_help.setWordWrap(True)
        offsets_layout.addWidget(offsets_help)
        self.b0_offset_table = QTableWidget(0, 2)
        self.b0_offset_table.setHorizontalHeaderLabels(["Shape", "B0 offset (ppm)"])
        self.b0_offset_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.b0_offset_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.b0_offset_table.setMinimumHeight(110)
        self.b0_offset_table.setMaximumHeight(170)
        self.b0_offset_table.cellChanged.connect(self._b0_offset_table_changed)
        self.b0_offset_table.currentCellChanged.connect(self._b0_offset_row_selected)
        offsets_layout.addWidget(self.b0_offset_table)
        b0_layout.addWidget(offsets_group)

        drift_group = QGroupBox("Uniform B0 drift over time (optional)")
        drift_layout = QVBoxLayout(drift_group)
        self.dynamic_b0_enabled = QCheckBox("Enable time-dependent B0 offset")
        self.dynamic_b0_enabled.toggled.connect(self._update_curve_controls_enabled)
        drift_layout.addWidget(self.dynamic_b0_enabled)
        self.dynamic_b0_help = QLabel(
            "Adds the same frequency offset in Hz to every voxel, linearly "
            "interpolated over sequence time; endpoint values are held outside "
            "the table. Requires the time-resolved phantom, so enabling it also "
            "enables the conversion model on the Kinetics tab."
        )
        self.dynamic_b0_help.setWordWrap(True)
        drift_layout.addWidget(self.dynamic_b0_help)
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
        self.dynamic_b0_curve_table.cellChanged.connect(
            self._dynamic_b0_curve_table_changed
        )
        drift_layout.addWidget(self.dynamic_b0_curve_table)
        b0_curve_buttons = QHBoxLayout()
        self.add_b0_point = QPushButton("Add B0 point")
        self.add_b0_point.clicked.connect(
            lambda: self._add_curve_point(self.dynamic_b0_curve_table)
        )
        self.remove_b0_point = QPushButton("Remove B0 point")
        self.remove_b0_point.clicked.connect(
            lambda: self._remove_curve_point(self.dynamic_b0_curve_table)
        )
        b0_curve_buttons.addWidget(self.add_b0_point)
        b0_curve_buttons.addWidget(self.remove_b0_point)
        b0_curve_buttons.addStretch()
        drift_layout.addLayout(b0_curve_buttons)
        b0_layout.addWidget(drift_group)
        b0_layout.addStretch(1)

        b0_scroll = QScrollArea()
        b0_scroll.setWidgetResizable(True)
        b0_scroll.setFrameShape(QFrame.NoFrame)
        b0_scroll.setMinimumWidth(450)
        b0_scroll.setWidget(b0_controls)
        b0_splitter.addWidget(b0_scroll)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_help = QLabel(
            "B0 map through the pattern centre (drag the crosshair to move it). "
            "Outlines show the shapes; the colour scale is symmetric about 0 ppm."
        )
        preview_help.setWordWrap(True)
        preview_layout.addWidget(preview_help)
        self.b0_map_graphics = pg.GraphicsLayoutWidget()
        self.b0_map_graphics.setMinimumSize(420, 300)
        self.b0_map_plots = []
        self.b0_map_images = []
        self.b0_center_targets = []
        self.b0_outline_items = [[], []]
        for column, (title, vertical_axis) in enumerate(
            (("XY slice (axial)", "y"), ("XZ slice", "z"))
        ):
            plot = self.b0_map_graphics.addPlot(row=0, col=column)
            plot.setTitle(title, size="9pt")
            plot.setAspectLocked(True)
            plot.setLabel("bottom", "x", units="mm")
            plot.setLabel("left", vertical_axis, units="mm")
            plot.setMenuEnabled(False)
            image = pg.ImageItem(axisOrder="col-major")
            plot.addItem(image)
            target = pg.TargetItem(
                pos=(0.0, 0.0),
                size=14,
                symbol="crosshair",
                pen=pg.mkPen((255, 220, 80), width=2),
                movable=True,
            )
            target.setZValue(2000)
            target.sigPositionChanged.connect(
                lambda item, column=column: self._b0_target_moved(column, item)
            )
            plot.addItem(target)
            self.b0_map_plots.append(plot)
            self.b0_map_images.append(image)
            self.b0_center_targets.append(target)
        self.b0_colorbar = pg.ColorBarItem(
            values=(-1.0, 1.0),
            colorMap=pg.colormap.get("CET-D1"),
            label="B0 (ppm)",
            interactive=False,
        )
        self.b0_colorbar.setImageItem(
            self.b0_map_images, insert_in=self.b0_map_plots[-1]
        )
        preview_layout.addWidget(self.b0_map_graphics, 3)

        spectrum_row = QHBoxLayout()
        spectrum_row.addWidget(QLabel("Spectrum of shape"))
        self.b0_preview_shape = QComboBox()
        self.b0_preview_shape.currentIndexChanged.connect(
            self._b0_preview_shape_changed
        )
        spectrum_row.addWidget(self.b0_preview_shape, 1)
        preview_layout.addLayout(spectrum_row)
        self.b0_spectrum_plot = pg.PlotWidget(
            title="Spectrum summed over the shape's voxels"
        )
        self.b0_spectrum_plot.setMinimumHeight(180)
        self.b0_spectrum_plot.setLabel("bottom", "Frequency", units="ppm")
        self.b0_spectrum_plot.setLabel("left", "Amplitude", units="a.u.")
        self.b0_spectrum_plot.showGrid(x=True, y=True, alpha=0.25)
        self.b0_spectrum_plot.addLegend(offset=(-8, 8))
        self.b0_spectrum_reference_curve = self.b0_spectrum_plot.plot(
            pen=pg.mkPen((170, 170, 170), width=1.5, style=Qt.DashLine),
            name="shape offset only",
        )
        self.b0_spectrum_curve = self.b0_spectrum_plot.plot(
            pen=pg.mkPen((0, 200, 200), width=2),
            name="with inhomogeneity",
        )
        preview_layout.addWidget(self.b0_spectrum_plot, 2)
        self.b0_preview_info = QLabel()
        self.b0_preview_info.setWordWrap(True)
        preview_layout.addWidget(self.b0_preview_info)
        b0_splitter.addWidget(preview_panel)
        b0_splitter.setStretchFactor(0, 0)
        b0_splitter.setStretchFactor(1, 1)
        b0_splitter.setSizes([480, 900])
        b0_page_layout.addWidget(b0_splitter)
        self.b0_page = b0_page
        self._b0_shape_cache = None
        return b0_page

    def _build_kinetics_page(self):
        kinetics_page = QWidget()
        kinetics_page_layout = QHBoxLayout(kinetics_page)
        kinetics_splitter = QSplitter(Qt.Horizontal)
        kinetics_splitter.setChildrenCollapsible(False)
        kinetics_controls = QWidget()
        kinetics_controls.setMinimumWidth(600)
        kinetics_layout = QVBoxLayout(kinetics_controls)
        self.dynamic_enabled = QCheckBox(
            "Enable pyruvate → lactate conversion model (kPL, inflow)"
        )
        self.dynamic_enabled.setToolTip(
            "Simulate pyruvate and lactate as exchanging pools with "
            "concentration-resolved kinetics. Hyperpolarization itself is set "
            "per peak on the Draw tab and does not require this model."
        )
        self.dynamic_enabled.toggled.connect(self._dynamic_enabled_toggled)
        self.dynamic_enabled.toggled.connect(self._update_kinetics_preview)
        self.dynamic_enabled.toggled.connect(self._update_spectral_preview)
        self.dynamic_enabled.toggled.connect(self._refresh_metabolite_presets)
        kinetics_layout.addWidget(self.dynamic_enabled)
        model_help = QLabel(
            "Hyperpolarization is set per peak on the Draw tab (Initial "
            "polarization, e.g. 10000 × thermal) and relaxes with T1 in every "
            "simulation, with or without this model. Enable the model only to add "
            "metabolic conversion of pyruvate to lactate (kPL) and pyruvate "
            "inflow. It uses the two peaks named below."
        )
        model_help.setWordWrap(True)
        kinetics_layout.addWidget(model_help)

        # Everything below only applies to the conversion model and is greyed
        # out while it is switched off.
        self.kinetics_model_controls = QWidget()
        model_layout = QVBoxLayout(self.kinetics_model_controls)
        model_layout.setContentsMargins(0, 0, 0, 0)
        kinetics_layout.addWidget(self.kinetics_model_controls)

        kinetics_form = QFormLayout()
        kinetics_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pyruvate_peak_name = QLineEdit("Pyruvate")
        self.lactate_peak_name = QLineEdit("Lactate")
        self.conversion_start_s = self._number_spin(-10000.0, 10000.0, 0.0, " s")
        self.kinetics_time_offset_s = self._number_spin(-10000.0, 10000.0, 0.0, " s")
        self.conversion_start_s.setToolTip(
            "kPL is zero before this time on the shared kinetics timeline and active "
            "afterwards."
        )
        self.kinetics_time_offset_s.setToolTip(
            "Selects which time on the shared inflow/conversion timeline coincides "
            "with sequence t=0. For example, +5 s starts the sequence 5 s into both "
            "curves; -5 s starts it 5 s before them."
        )
        self.conversion_start_s.valueChanged.connect(self._update_kinetics_preview)
        self.kinetics_time_offset_s.valueChanged.connect(self._update_kinetics_preview)
        self.pyruvate_peak_name.textChanged.connect(self._update_kinetics_preview)
        self.lactate_peak_name.textChanged.connect(self._update_kinetics_preview)
        kinetics_form.addRow("Pyruvate peak name", self.pyruvate_peak_name)
        kinetics_form.addRow("Lactate peak name", self.lactate_peak_name)
        kinetics_form.addRow(
            "Conversion starts at (kinetics time)", self.conversion_start_s
        )
        kinetics_form.addRow(
            "Kinetics time at sequence t=0", self.kinetics_time_offset_s
        )
        model_layout.addLayout(kinetics_form)

        inflow_group = QGroupBox("Pyruvate inflow")
        inflow_layout = QVBoxLayout(inflow_group)
        self.inflow_enabled = QCheckBox(
            "Add inflowing pyruvate to every shape that contains pyruvate"
        )
        self.inflow_enabled.toggled.connect(self._update_curve_controls_enabled)
        self.inflow_enabled.toggled.connect(self._update_kinetics_preview)
        inflow_layout.addWidget(self.inflow_enabled)
        self.inflow_help = QLabel(
            "<b>Inflow rate</b>: how much pyruvate arrives per second, in the same "
            "relative concentration units as the peak table. <b>Inflow "
            "polarization</b>: how strongly the arriving pyruvate is polarized "
            "(1 × thermal = not hyperpolarized). The added Mz per second is rate × "
            "polarization. Rows are linearly interpolated; outside the table the "
            "rate is zero. Inflow before sequence t=0 is simulated as a pre-roll "
            "that sets the starting state."
        )
        self.inflow_help.setWordWrap(True)
        inflow_layout.addWidget(self.inflow_help)
        self.inflow_curve_table = QTableWidget(0, 3)
        self.inflow_curve_table.setHorizontalHeaderLabels(
            [
                "Kinetics time (s)",
                "Inflow rate (conc./s)",
                "Inflow polarization",
            ]
        )
        self.inflow_curve_table.horizontalHeaderItem(1).setToolTip(
            "Relative pyruvate concentration added per second"
        )
        self.inflow_curve_table.horizontalHeaderItem(2).setToolTip(
            "Polarization of the arriving pyruvate in multiples of thermal "
            "equilibrium"
        )
        self._inflow_polarization_delegate = SuffixNumberDelegate(
            " × thermal", self.inflow_curve_table, minimum=0.0
        )
        self.inflow_curve_table.setItemDelegateForColumn(
            2, self._inflow_polarization_delegate
        )
        inflow_header = self.inflow_curve_table.horizontalHeader()
        inflow_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        inflow_header.setSectionResizeMode(1, QHeaderView.Stretch)
        inflow_header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.inflow_curve_table.setMaximumHeight(150)
        self.inflow_curve_table.setMinimumHeight(110)
        self.inflow_curve_table.cellChanged.connect(self._inflow_curve_table_changed)
        inflow_layout.addWidget(self.inflow_curve_table)
        inflow_buttons = QHBoxLayout()
        self.add_inflow_point = QPushButton("Add inflow point")
        self.add_inflow_point.clicked.connect(
            lambda: self._add_curve_point(self.inflow_curve_table)
        )
        self.remove_inflow_point = QPushButton("Remove inflow point")
        self.remove_inflow_point.clicked.connect(
            lambda: self._remove_curve_point(self.inflow_curve_table)
        )
        inflow_buttons.addWidget(self.add_inflow_point)
        inflow_buttons.addWidget(self.remove_inflow_point)
        inflow_buttons.addStretch()
        inflow_layout.addLayout(inflow_buttons)
        model_layout.addWidget(inflow_group)

        regions_group = QGroupBox("Spatial kPL regions (optional)")
        regions_layout = QVBoxLayout(regions_group)
        kinetic_regions_help = QLabel(
            "Each row overrides the shape kPL inside an ellipsoid or box; the last "
            "row wins where regions overlap. Centre and size are % of the FOV."
        )
        kinetic_regions_help.setWordWrap(True)
        regions_layout.addWidget(kinetic_regions_help)
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
        regions_layout.addWidget(self.kinetic_table)
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
        regions_layout.addLayout(kinetic_buttons)
        model_layout.addWidget(regions_group)
        kinetics_layout.addStretch(1)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_title = QLabel(
            "Preview of one representative voxel of the chosen shape (no RF "
            "pulses or gradients). <span style='color:#00c8c8'>■</span> pyruvate, "
            "<span style='color:#d050d0'>■</span> lactate."
        )
        preview_title.setWordWrap(True)
        preview_layout.addWidget(preview_title)
        preview_form = QFormLayout()
        self.kinetics_preview_shape = QComboBox()
        self.kinetics_preview_shape.currentIndexChanged.connect(
            self._kinetics_preview_shape_changed
        )
        preview_form.addRow("Shape / object to preview", self.kinetics_preview_shape)
        self.shape_kpl = self._number_spin(0.0, 10.0, 0.0, " s⁻¹")
        self.shape_kpl.setToolTip(
            "Pyruvate → lactate conversion rate in the selected shape. 0 s⁻¹ "
            "means no conversion; spatial kPL regions override it."
        )
        self.shape_kpl.valueChanged.connect(self._shape_kpl_changed)
        # Compatibility alias for integrations which used the former global field.
        self.default_kpl = self.shape_kpl
        preview_form.addRow("kPL of this shape", self.shape_kpl)
        # Not shown: selecting a row of the kPL region table previews that region.
        self.kinetics_preview_region = QComboBox()
        self.kinetics_preview_region.currentIndexChanged.connect(
            self._update_kinetics_preview
        )
        self.zero_lactate_button = QPushButton("Start lactate at 0 × thermal")
        self.zero_lactate_button.setToolTip(
            "Sets the lactate initial polarization of the selected shape to 0 "
            "without changing its concentration, so all lactate signal comes from "
            "conversion (and thermal recovery)."
        )
        self.zero_lactate_button.clicked.connect(self._set_selected_shape_lactate_zero)
        preview_form.addRow("Lactate signal", self.zero_lactate_button)
        self.kinetics_preview_duration = self._number_spin(0.1, 10000.0, 30.0, " s")
        self.kinetics_preview_duration.valueChanged.connect(
            self._update_kinetics_preview
        )
        preview_form.addRow("Duration", self.kinetics_preview_duration)
        self.time_curve_display = QComboBox()
        self.time_curve_display.addItem("Interpolated curve", "simulated")
        self.time_curve_display.addItem("Table points only", "set")
        self.time_curve_display.setToolTip(
            "Show the inflow as the interpolated curve passed to the simulator or "
            "only the points entered in the table."
        )
        self.time_curve_display.currentIndexChanged.connect(
            self._update_kinetics_preview
        )
        preview_form.addRow("Inflow display", self.time_curve_display)
        preview_controls = QHBoxLayout()
        preview_controls.addLayout(preview_form, 1)
        spatial_preview_group = QGroupBox("Spatial selection · drag to orbit")
        spatial_preview_layout = QVBoxLayout(spatial_preview_group)
        self.kinetics_shape_preview_3d = ShapePreview3DWidget()
        self.kinetics_shape_preview_3d.setMinimumSize(260, 130)
        self.kinetics_shape_preview_3d.setMaximumHeight(170)
        spatial_preview_layout.addWidget(self.kinetics_shape_preview_3d)
        self.kinetics_spatial_preview_info = QLabel()
        self.kinetics_spatial_preview_info.setWordWrap(True)
        spatial_preview_layout.addWidget(self.kinetics_spatial_preview_info)
        preview_controls.addWidget(spatial_preview_group, 1)
        preview_layout.addLayout(preview_controls)

        self.kinetics_preview_graphics = pg.GraphicsLayoutWidget()
        self.kinetics_preview_graphics.setMinimumWidth(420)
        self.kinetics_preview_graphics.setMinimumHeight(560)
        pyruvate_pen = pg.mkPen((0, 200, 200), width=2)
        lactate_pen = pg.mkPen((208, 80, 208), width=2, style=Qt.DashLine)

        def add_kinetics_plot(row, title, left_label, units=None):
            plot = self.kinetics_preview_graphics.addPlot(row=row, col=0)
            plot.setTitle(title, size="9pt", justify="left")
            plot.setLabel("left", left_label, units=units)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis("left").setWidth(62)
            return plot

        self.inflow_preview_plot = add_kinetics_plot(
            0,
            "Input: pyruvate inflow rate",
            "rate",
            units="conc./s",
        )
        self.inflow_preview_curve = self.inflow_preview_plot.plot(
            pen=pg.mkPen("y", width=2),
            fillLevel=0.0,
            brush=pg.mkBrush(255, 255, 0, 45),
        )
        self.concentration_preview_plot = add_kinetics_plot(
            1,
            "Concentration: how much pyruvate / lactate is present",
            "conc.",
        )
        self.concentration_preview_plot.addLegend(offset=(-8, 4))
        self.pyruvate_concentration_curve = self.concentration_preview_plot.plot(
            pen=pyruvate_pen, name="Pyruvate"
        )
        self.lactate_concentration_curve = self.concentration_preview_plot.plot(
            pen=lactate_pen, name="Lactate"
        )
        self.polarization_preview_plot = add_kinetics_plot(
            2,
            "Polarization: how strongly it is polarized (relaxes to 1 × thermal)",
            "× thermal",
        )
        self.pyruvate_polarization_curve = self.polarization_preview_plot.plot(
            pen=pyruvate_pen
        )
        self.lactate_polarization_curve = self.polarization_preview_plot.plot(
            pen=lactate_pen
        )
        self.equilibrium_polarization_line = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            pen=pg.mkPen((180, 180, 180), width=1, style=Qt.DotLine),
            label="1 × thermal",
            labelOpts={
                "position": 0.02,
                "color": (180, 180, 180),
                "anchors": [(0, 1), (0, 1)],
            },
        )
        self.polarization_preview_plot.addItem(self.equilibrium_polarization_line)
        self.pool_preview_plot = add_kinetics_plot(
            3,
            "Signal: Mz = concentration × polarization",
            "Mz",
        )
        self.pool_preview_plot.setLabel("bottom", "Sequence time", units="s")
        self.pyruvate_preview_curve = self.pool_preview_plot.plot(pen=pyruvate_pen)
        self.lactate_preview_curve = self.pool_preview_plot.plot(pen=lactate_pen)

        self._sequence_start_lines = []
        self._conversion_start_lines = []
        kinetics_plots = (
            self.inflow_preview_plot,
            self.concentration_preview_plot,
            self.polarization_preview_plot,
            self.pool_preview_plot,
        )
        for index, plot in enumerate(kinetics_plots):
            show_label = index == 0
            sequence_line = pg.InfiniteLine(
                pos=0.0,
                angle=90,
                pen=pg.mkPen((180, 180, 180), width=1, style=Qt.DashLine),
                label="sequence start" if show_label else None,
                labelOpts={"position": 0.85, "color": (180, 180, 180)},
            )
            conversion_line = pg.InfiniteLine(
                pos=0.0,
                angle=90,
                pen=pg.mkPen((255, 150, 60), width=1, style=Qt.DotLine),
                label="kPL on" if show_label else None,
                labelOpts={"position": 0.6, "color": (255, 150, 60)},
            )
            plot.addItem(sequence_line)
            plot.addItem(conversion_line)
            self._sequence_start_lines.append(sequence_line)
            self._conversion_start_lines.append(conversion_line)
            if plot is not self.inflow_preview_plot:
                plot.setXLink(self.inflow_preview_plot)
        self.inflow_sequence_start_line = self._sequence_start_lines[0]
        self.pool_sequence_start_line = self._sequence_start_lines[-1]
        self.conversion_start_line = self._conversion_start_lines[-1]
        preview_layout.addWidget(self.kinetics_preview_graphics, 1)
        self.kinetics_preview_info = QLabel()
        self.kinetics_preview_info.setWordWrap(True)
        self.kinetics_preview_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
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

        # The four stacked plots need their height; scroll rather than squeeze
        # them into the explanatory text on short screens.
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setFrameShape(QFrame.NoFrame)
        preview_scroll.setWidget(preview_panel)
        kinetics_splitter.addWidget(kinetics_scroll)
        kinetics_splitter.addWidget(preview_scroll)
        kinetics_splitter.setStretchFactor(0, 1)
        kinetics_splitter.setStretchFactor(1, 1)
        kinetics_splitter.setSizes([650, 750])
        kinetics_page_layout.addWidget(kinetics_splitter)
        return kinetics_page

    def _dynamic_enabled_toggled(self, enabled):
        self.kinetics_model_controls.setEnabled(bool(enabled))
        self._inspector_preview_dirty = True
        self._update_thermal_polarization_info()
        if not enabled or self._updating or self._suppress_default_kpl:
            return
        # Switching conversion on with every rate at zero would preview nothing
        # but T1 decay, so start from a typical non-zero kPL instead.
        fallback = self.design.default_kpl_s_inv
        if any(
            shape.effective_kpl_s_inv(fallback) > 0.0 for shape in self.design.shapes
        ):
            return
        try:
            if any(region.kpl_s_inv > 0.0 for region in self._read_kinetic_regions()):
                return
        except (AttributeError, TypeError, ValueError):
            pass
        for shape in self.design.shapes:
            shape.kpl_s_inv = DEFAULT_ENABLED_KPL_S_INV
        self._updating = True
        try:
            self.shape_kpl.setValue(DEFAULT_ENABLED_KPL_S_INV)
        finally:
            self._updating = False

    def _b0_center_fraction(self):
        return tuple(spin.value() / 100.0 for spin in self.b0_center_spins)

    def _reset_b0_center(self):
        self._set_b0_center((0.5, 0.5, 0.5))

    def _set_b0_center(self, center):
        for spin, value in zip(self.b0_center_spins, center):
            previous = spin.blockSignals(True)
            spin.setValue(float(value) * 100.0)
            spin.blockSignals(previous)
        self._b0_settings_changed()

    def _b0_target_moved(self, column, target):
        if self._updating or getattr(self, "_moving_b0_targets", False):
            return
        fov_mm = [widget.value() for widget in self.fov_spins]
        position = target.pos()
        center = list(self._b0_center_fraction())
        # Column 0 shows x/y, column 1 shows x/z.
        vertical_axis = 1 if column == 0 else 2
        center[0] = 0.5 + position.x() / max(fov_mm[0], 1e-12)
        center[vertical_axis] = 0.5 + position.y() / max(fov_mm[vertical_axis], 1e-12)
        center = [min(max(value, -0.5), 1.5) for value in center]
        self._set_b0_center(center)

    def _b0_settings_changed(self, *_):
        if self._updating:
            return
        mode = str(self.b0_mode_combo.currentData())
        if (
            self.design.b0_inhomogeneity_mode == "none"
            and mode != "none"
            and self.b0_inhomogeneity_ppm.value() == 0.0
        ):
            # A pattern without strength has no effect; start from 1 ppm.
            blocked = self.b0_inhomogeneity_ppm.blockSignals(True)
            self.b0_inhomogeneity_ppm.setValue(1.0)
            self.b0_inhomogeneity_ppm.blockSignals(blocked)
        self.design.b0_inhomogeneity_mode = mode
        self.design.b0_inhomogeneity_ppm = self.b0_inhomogeneity_ppm.value()
        self.design.b0_inhomogeneity_center = self._b0_center_fraction()
        self._inspector_preview_dirty = True
        self._update_b0_pattern_info()
        self._update_b0_preview()

    def _update_b0_pattern_info(self):
        mode = str(self.b0_mode_combo.currentData())
        amplitude = self.b0_inhomogeneity_ppm.value()
        centre_enabled = mode != "none"
        for spin in self.b0_center_spins:
            spin.setEnabled(centre_enabled)
        self.b0_center_reset.setEnabled(centre_enabled)
        self.b0_inhomogeneity_ppm.setEnabled(centre_enabled)
        if mode == "none":
            text = "No spatial variation: only the per-shape offsets below apply."
        elif mode.startswith("linear"):
            axis = mode[-1].upper()
            text = (
                f"Linear gradient along {axis}: 0 ppm at the centre, changing by "
                f"{amplitude:g} ppm per half FOV ({2.0 * amplitude:g} ppm across "
                "the FOV). Moving the centre shifts the zero crossing."
            )
        else:
            plane = "XY plane" if mode == "radial_xy" else "3D"
            text = (
                f"Radial ({plane}): {-amplitude / 2.0:g} ppm at the centre, rising "
                f"to {amplitude / 2.0:g} ppm at the distance of the FOV corner from "
                "the FOV centre. Moving the centre moves the extremum."
            )
        self.b0_pattern_info.setText(text)

    def _populate_b0_offset_table(self):
        if not hasattr(self, "b0_offset_table"):
            return
        previous = self.b0_offset_table.blockSignals(True)
        self.b0_offset_table.setRowCount(len(self.design.shapes))
        for row, shape in enumerate(self.design.shapes):
            name_item = QTableWidgetItem(shape.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.b0_offset_table.setItem(row, 0, name_item)
            if shape.b0_hz is not None:
                value_item = QTableWidgetItem(f"{shape.b0_hz:g} Hz (legacy)")
                value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
                value_item.setToolTip(
                    "Loaded from a design stored in Hz; re-create the shape to use "
                    "a ppm offset"
                )
            else:
                value_item = QTableWidgetItem(f"{shape.b0_ppm:g}")
            self.b0_offset_table.setItem(row, 1, value_item)
        self.b0_offset_table.blockSignals(previous)
        previous = self.b0_preview_shape.blockSignals(True)
        selected = self.b0_preview_shape.currentData()
        if selected is None:
            selected = self._current_row()
        self.b0_preview_shape.clear()
        for row, shape in enumerate(self.design.shapes):
            self.b0_preview_shape.addItem(shape.name, row)
        index = self.b0_preview_shape.findData(selected)
        self.b0_preview_shape.setCurrentIndex(max(0, index))
        self.b0_preview_shape.blockSignals(previous)

    def _b0_offset_table_changed(self, row, column):
        if self._updating or column != 1 or not (0 <= row < len(self.design.shapes)):
            return
        shape = self.design.shapes[row]
        item = self.b0_offset_table.item(row, 1)
        try:
            value = float(item.text())
            if not np.isfinite(value):
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            previous = self.b0_offset_table.blockSignals(True)
            item.setText(f"{shape.b0_ppm:g}")
            self.b0_offset_table.blockSignals(previous)
            return
        shape.b0_ppm = value
        if self._b0_shape_cache is not None:
            self._b0_shape_cache["offset_map"] = None
        self._inspector_preview_dirty = True
        if self._current_row() == row:
            self._update_spectral_preview()
        self._update_b0_preview()

    def _b0_offset_row_selected(self, row, _column, _previous_row, _previous_column):
        if row < 0:
            return
        index = self.b0_preview_shape.findData(row)
        if index >= 0:
            self.b0_preview_shape.setCurrentIndex(index)

    def _b0_preview_shape_changed(self, _index):
        self._update_b0_preview()

    def _refresh_b0_preview(self):
        """Rasterize shape masks once for the B0 tab on a coarse grid."""
        try:
            self._sync_global()
            preview_design = replace(
                self.design,
                shape=tuple(min(int(count), 48) for count in self.design.shape),
                # The B0 tab shows the whole phantom, not the simulated region.
                simulation_fov_m=None,
                supersampling_enabled=False,
                shapes=list(self.design.shapes),
                kinetic_regions=[],
            )
            masks = [
                preview_design.rasterize_mask(shape) for shape in self.design.shapes
            ]
        except (TypeError, ValueError) as exc:
            self._b0_shape_cache = None
            self.b0_preview_info.setText(f"Preview unavailable: {exc}")
            return
        self._b0_shape_cache = {
            "design": preview_design,
            "masks": masks,
            "offset_map": None,
        }
        self._update_b0_preview()

    def _b0_preview_maps(self):
        cache = self._b0_shape_cache
        preview_design = cache["design"]
        preview_design.b0_inhomogeneity_mode = self.design.b0_inhomogeneity_mode
        preview_design.b0_inhomogeneity_ppm = self.design.b0_inhomogeneity_ppm
        preview_design.b0_inhomogeneity_center = self.design.b0_inhomogeneity_center
        if cache["offset_map"] is None:
            offset_map = np.zeros(preview_design.shape, dtype=float)
            for shape, mask in zip(self.design.shapes, cache["masks"]):
                offset_map[mask] = shape.b0_ppm if shape.b0_hz is None else 0.0
            cache["offset_map"] = offset_map
        return (
            preview_design,
            cache["offset_map"],
            cache["offset_map"] + preview_design.rasterize_b0_inhomogeneity(),
        )

    def _update_b0_preview(self):
        if (
            not hasattr(self, "b0_map_images")
            or self.tabs.currentWidget() is not self.b0_page
        ):
            return
        if self._b0_shape_cache is None or len(self._b0_shape_cache["masks"]) != len(
            self.design.shapes
        ):
            self._refresh_b0_preview()
            return
        preview_design, offset_map, b0_map = self._b0_preview_maps()
        fov_mm = np.asarray(self.design.fov_m, dtype=float) * 1000.0
        grid = np.asarray(preview_design.shape)
        center = np.asarray(self.design.b0_inhomogeneity_center, dtype=float)
        center_index = np.clip(np.floor(center * grid).astype(int), 0, grid - 1)
        slices = (
            b0_map[:, :, center_index[2]],
            b0_map[:, center_index[1], :],
        )
        limit = float(np.max(np.abs(b0_map))) if b0_map.size else 0.0
        limit = limit if limit > 1e-12 else 1.0
        self._moving_b0_targets = True
        try:
            for column, (image, data, axes) in enumerate(
                zip(self.b0_map_images, slices, ((0, 1), (0, 2)))
            ):
                image.setImage(data, autoLevels=False)
                image.setRect(
                    QRectF(
                        -fov_mm[axes[0]] / 2.0,
                        -fov_mm[axes[1]] / 2.0,
                        fov_mm[axes[0]],
                        fov_mm[axes[1]],
                    )
                )
                self.b0_center_targets[column].setPos(
                    (center[axes[0]] - 0.5) * fov_mm[axes[0]],
                    (center[axes[1]] - 0.5) * fov_mm[axes[1]],
                )
                self.b0_center_targets[column].setVisible(
                    self.design.b0_inhomogeneity_mode != "none"
                )
                plot = self.b0_map_plots[column]
                for outline in self.b0_outline_items[column]:
                    plot.removeItem(outline)
                self.b0_outline_items[column] = []
                for index, shape in enumerate(self.design.shapes):
                    x, y = _shape_projection(shape, fov_mm, axes)
                    outline = pg.PlotDataItem(
                        (np.asarray(x) - 0.5) * fov_mm[axes[0]],
                        (np.asarray(y) - 0.5) * fov_mm[axes[1]],
                        pen=pg.mkPen(
                            pg.intColor(index, hues=max(1, len(self.design.shapes))),
                            width=1.5,
                        ),
                    )
                    outline.setZValue(1000)
                    plot.addItem(outline)
                    self.b0_outline_items[column].append(outline)
                plot.setXRange(-fov_mm[axes[0]] / 2.0, fov_mm[axes[0]] / 2.0)
                plot.setYRange(-fov_mm[axes[1]] / 2.0, fov_mm[axes[1]] / 2.0)
        finally:
            self._moving_b0_targets = False
        self.b0_colorbar.setLevels((-limit, limit))
        self._update_b0_spectrum(b0_map)

    def _update_b0_spectrum(self, b0_map):
        row = self.b0_preview_shape.currentData()
        if row is None or not (0 <= int(row) < len(self.design.shapes)):
            self.b0_spectrum_curve.setData([], [])
            self.b0_spectrum_reference_curve.setData([], [])
            self.b0_preview_info.setText("Add a shape to preview its spectrum.")
            return
        row = int(row)
        shape = self.design.shapes[row]
        values = b0_map[self._b0_shape_cache["masks"][row]]
        if values.size == 0:
            self.b0_spectrum_curve.setData([], [])
            self.b0_spectrum_reference_curve.setData([], [])
            self.b0_preview_info.setText(
                f"{shape.name} covers no voxel of the preview grid."
            )
            return
        if values.size > 4000:
            values = values[:: int(np.ceil(values.size / 4000))]
        reference_ppm = float(self.spectral_reference_ppm.value())
        window_center_ppm = float(self.spectral_window_center_ppm.value())
        bandwidth_ppm = float(self.spectral_bandwidth_ppm.value())
        points = max(2, int(self.spectral_points.value()))
        frequency_ppm = np.linspace(
            window_center_ppm - bandwidth_ppm / 2.0,
            window_center_ppm + bandwidth_ppm / 2.0,
            points,
        )
        nucleus = self._effective_nucleus()
        field_strength_t = float(self.field_strength_t.value())
        spectrum = np.zeros(points, dtype=float)
        reference = np.zeros(points, dtype=float)
        shape_offset = shape.b0_ppm if shape.b0_hz is None else 0.0
        for peak in shape.peaks:
            fwhm_ppm = abs(
                float(
                    hz_to_ppm(1.0 / (np.pi * peak.t2_star_s), field_strength_t, nucleus)
                )
            )
            half_width_ppm = max(fwhm_ppm / 2.0, np.finfo(float).eps)
            amplitude = peak.amplitude * peak.effective_initial_polarization(
                shape.initial_mz
            )
            peak_ppm = reference_ppm + peak.frequency_ppm
            reference += amplitude / (
                1.0 + ((frequency_ppm - peak_ppm - shape_offset) / half_width_ppm) ** 2
            )
            distance = (
                frequency_ppm[:, None] - peak_ppm - values[None, :]
            ) / half_width_ppm
            spectrum += amplitude * np.mean(1.0 / (1.0 + distance**2), axis=1)
        self.b0_spectrum_reference_curve.setData(frequency_ppm, reference)
        self.b0_spectrum_curve.setData(frequency_ppm, spectrum)
        self.b0_spectrum_plot.setXRange(
            float(frequency_ppm[0]), float(frequency_ppm[-1]), padding=0.02
        )
        self.b0_spectrum_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.b0_preview_info.setText(
            f"B0 inside {shape.name}: mean {np.mean(values):.3g} ppm, range "
            f"{np.min(values):.3g} … {np.max(values):.3g} ppm, standard deviation "
            f"{np.std(values):.3g} ppm (preview grid "
            f"{'×'.join(str(n) for n in self._b0_shape_cache['design'].shape)})."
        )

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

    def _update_thermal_polarization_info(self):
        nucleus = self._effective_nucleus()
        try:
            pth = thermal_polarization(
                nucleus,
                self.field_strength_t.value(),
                self.temperature_c.value() + 273.15,
            )
        except (KeyError, ValueError):
            self.thermal_polarization_info.setText("")
            return
        self.thermal_polarization_info.setText(
            f"1 × thermal = {100 * pth:.4g} % ({nucleus})"
        )

    def _effective_nucleus(self):
        nucleus = self.nucleus.currentData()
        if nucleus is None:
            dynamic = (
                hasattr(self, "dynamic_enabled") and self.dynamic_enabled.isChecked()
            )
            nucleus = "C13" if dynamic else "H1"
        return nucleus

    def _refresh_metabolite_presets(self, *_):
        if not hasattr(self, "metabolite_preset_combo") or not hasattr(
            self, "dynamic_enabled"
        ):
            return
        nucleus = self._effective_nucleus()
        previous = self.metabolite_preset_combo.currentData()
        previous_key = previous.key if isinstance(previous, MetabolitePreset) else None
        presets = metabolite_presets(nucleus)
        blocked = self.metabolite_preset_combo.blockSignals(True)
        self.metabolite_preset_combo.clear()
        if presets:
            for preset in presets:
                self.metabolite_preset_combo.addItem(preset.label, preset)
            matching_index = next(
                (
                    index
                    for index, preset in enumerate(presets)
                    if preset.key == previous_key
                ),
                0,
            )
            self.metabolite_preset_combo.setCurrentIndex(matching_index)
        else:
            self.metabolite_preset_combo.addItem(f"No curated {nucleus} presets", None)
        self.metabolite_preset_combo.blockSignals(blocked)
        self._update_metabolite_preset_info()

    def _update_metabolite_preset_info(self, *_):
        preset = self.metabolite_preset_combo.currentData()
        available = isinstance(preset, MetabolitePreset)
        self.add_metabolite_preset_button.setEnabled(available)
        if available:
            text = preset.relaxation_summary
            tooltip = (
                f"{text}\nRelaxation is context-dependent; edit the table values "
                "for the intended field, tissue, and acquisition."
            )
        else:
            text = (
                "No curated presets for this nucleus. Peaks can still be added "
                "manually."
            )
            tooltip = text
        self.metabolite_preset_combo.setToolTip(tooltip)
        self.add_metabolite_preset_button.setToolTip(tooltip)

    def set_sequence_geometry(self, geometry):
        """Point the "From seq." buttons at the given sequence geometry."""
        self._sequence_geometry = geometry
        self._sequence_axis_grids = sequence_axis_grids(geometry)
        for axis, (button, grid) in enumerate(
            zip(self.sequence_grid_buttons, self._sequence_axis_grids)
        ):
            name = "XYZ"[axis]
            if isinstance(grid, str):
                button.setEnabled(False)
                button.setToolTip(f"No sequence grid along {name}. {grid}")
                continue
            fov_m, matrix = grid
            button.setEnabled(True)
            button.setToolTip(
                f"Copy the sequence FOV along {name} ({fov_m * 1000.0:.4g} mm) "
                f"and set the phantom matrix to {SEQUENCE_GRID_OVERSAMPLING} × "
                f"the sequence matrix ({matrix} → "
                f"{SEQUENCE_GRID_OVERSAMPLING * matrix}). Two phantom cells per "
                "sequence pixel are a good compromise between simulation speed "
                "and image quality. FOV offsets are not copied."
            )
        self._simulation_region_changed()

    def _copy_sequence_axis(self, axis):
        grid = self._sequence_axis_grids[axis]
        if isinstance(grid, str):
            return
        fov_m, matrix = grid
        matrix_spin = self.matrix_spins[axis]
        matrix_spin.setValue(
            min(SEQUENCE_GRID_OVERSAMPLING * matrix, matrix_spin.maximum())
        )
        self.fov_spins[axis].setValue(fov_m * 1000.0)

    def _simulation_region_values(self):
        """Return the simulated FOV, phantom position and ppm range in the UI."""
        simulation_fov_m = None
        offset_m = (0.0, 0.0, 0.0)
        if self.simulation_fov_enabled.isChecked():
            simulation_fov_m = tuple(
                spin.value() / 1000.0 for spin in self.simulation_fov_spins
            )
            offset_m = tuple(
                spin.value() / 1000.0 for spin in self.simulation_offset_spins
            )
        ppm_range = None
        if self.simulation_ppm_enabled.isChecked():
            ppm_range = (
                float(self.simulation_ppm_low.value()),
                float(self.simulation_ppm_high.value()),
            )
        return simulation_fov_m, offset_m, ppm_range

    def _load_simulation_region_into_ui(self):
        design = self.design
        enabled = design.simulation_fov_m is not None
        self.simulation_fov_enabled.setChecked(enabled)
        for spin, value in zip(self.simulation_fov_spins, design.simulated_fov_m):
            spin.setValue(float(value) * 1000.0)
        for spin, value in zip(
            self.simulation_offset_spins,
            design.simulation_offset_m if enabled else (0.0, 0.0, 0.0),
        ):
            spin.setValue(float(value) * 1000.0)
        ppm_range = design.simulation_ppm_range
        self.simulation_ppm_enabled.setChecked(ppm_range is not None)
        if ppm_range is None:
            centre = design.effective_spectral_window_center_ppm
            half = 0.5 * float(design.spectral_bandwidth_ppm)
            ppm_range = (centre - half, centre + half)
        self.simulation_ppm_low.setValue(float(ppm_range[0]))
        self.simulation_ppm_high.setValue(float(ppm_range[1]))

    def _simulation_region_changed(self, *_):
        if self._updating:
            return
        for spin in (*self.simulation_fov_spins, *self.simulation_offset_spins):
            spin.setEnabled(self.simulation_fov_enabled.isChecked())
        self.fit_simulation_fov_button.setEnabled(
            self.simulation_fov_enabled.isChecked()
            and self._sequence_geometry is not None
            and self._sequence_geometry.fov_m is not None
        )
        for spin in (self.simulation_ppm_low, self.simulation_ppm_high):
            spin.setEnabled(self.simulation_ppm_enabled.isChecked())
        self.fit_ppm_range_button.setEnabled(self.simulation_ppm_enabled.isChecked())
        self._inspector_preview_dirty = True
        self._update_simulation_region_summary()
        if self.tabs.currentWidget() is self.region_page:
            self._update_region_preview()

    def _simulation_grid(self):
        """Return the simulated FOV (m) and matrix for the current UI values."""
        fov_m = np.array([spin.value() / 1000.0 for spin in self.fov_spins])
        matrix = np.array([spin.value() for spin in self.matrix_spins])
        simulation_fov_m, _offset, _ppm = self._simulation_region_values()
        if simulation_fov_m is None:
            return fov_m, matrix
        region = np.asarray(simulation_fov_m, dtype=float)
        counts = np.maximum(1, np.rint(matrix * region / fov_m)).astype(int)
        return region, counts

    def _update_simulation_region_summary(self, *_):
        """One line in 'Draw and edit' saying that only a region is simulated."""
        simulation_fov_m, _offset, ppm_range = self._simulation_region_values()
        parts = []
        if simulation_fov_m is not None:
            region, counts = self._simulation_grid()
            parts.append(
                "FOV "
                + " × ".join(f"{value * 1000:.4g}" for value in region)
                + " mm (matrix "
                + " × ".join(str(int(value)) for value in counts)
                + ")"
            )
        if ppm_range is not None:
            parts.append(f"{ppm_range[0]:.4g}–{ppm_range[1]:.4g} ppm")
        self.simulation_region_summary.setText(
            "Simulated region: " + ", ".join(parts) if parts else ""
        )
        self.simulation_region_summary.setVisible(bool(parts))

    def _sequence_fit_extent_m(self):
        """Half-extent around the isocentre that holds the sequence FOV."""
        geometry = self._sequence_geometry
        if geometry is None or geometry.fov_m is None:
            return None
        box = geometry.fov_box(0.0)
        if box is None:
            return None
        corners = box.corners()
        half = np.max(np.abs(corners), axis=0)
        if geometry.is_2d and geometry.slice_thickness_m:
            # Spins just outside the nominal slices still see the slice
            # profile's transition band.
            normal = np.abs(np.asarray(geometry.frame, dtype=float)[:, 2])
            half = half + 0.5 * float(geometry.slice_thickness_m) * normal
        return half

    def _fit_simulation_fov(self):
        half = self._sequence_fit_extent_m()
        if half is None:
            return
        for spin, value in zip(self.simulation_fov_spins, half):
            if value > 0.0:
                spin.setValue(2000.0 * float(value))

    def _fit_simulation_ppm_range(self):
        centre = float(self.spectral_window_center_ppm.value())
        half = 0.5 * float(self.spectral_bandwidth_ppm.value())
        self.simulation_ppm_low.setValue(centre - half)
        self.simulation_ppm_high.setValue(centre + half)

    def _simulation_ppm_region_dragged(self):
        if self._updating:
            return
        low, high = self.simulation_ppm_region.getRegion()
        if not self.simulation_ppm_enabled.isChecked():
            self.simulation_ppm_enabled.setChecked(True)
        self.simulation_ppm_low.setValue(float(low))
        self.simulation_ppm_high.setValue(float(high))

    def _designed_peak_positions(self):
        """Return (name, absolute ppm) of every distinct peak in the design."""
        reference = float(self.spectral_reference_ppm.value())
        peaks = []
        for item in self.design.shapes:
            for peak in item.peaks:
                entry = (peak.name, reference + float(peak.frequency_ppm))
                if not any(
                    name == entry[0] and np.isclose(ppm, entry[1])
                    for name, ppm in peaks
                ):
                    peaks.append(entry)
        return peaks

    def _update_region_preview(self):
        """Draw phantom, simulated FOV and sequence FOV; list kept peaks."""
        for plot, item in self._region_items:
            plot.removeItem(item)
        self._region_items = []
        fov_mm = np.array([spin.value() for spin in self.fov_spins])
        simulation_fov_m, offset_m, ppm_range = self._simulation_region_values()
        offset_mm = np.asarray(offset_m, dtype=float) * 1000.0
        region_m, counts = self._simulation_grid()
        phantom_box = OrientedBox(tuple(offset_m), tuple(fov_mm / 1000.0))
        region_box = OrientedBox((0.0, 0.0, 0.0), tuple(region_m))
        sequence_box = (
            None
            if self._sequence_geometry is None or self._sequence_geometry.fov_m is None
            else self._sequence_geometry.fov_box(0.0)
        )
        for plane, plot in self.region_plots:

            def add(item, plot=plot):
                plot.addItem(item)
                self._region_items.append((plot, item))

            for index, shape in enumerate(self.design.shapes):
                x, y = _shape_projection(shape, tuple(fov_mm), plane)
                add(
                    pg.PlotDataItem(
                        (np.asarray(x) - 0.5) * fov_mm[plane[0]] + offset_mm[plane[0]],
                        (np.asarray(y) - 0.5) * fov_mm[plane[1]] + offset_mm[plane[1]],
                        pen=pg.mkPen(
                            pg.intColor(index, hues=max(1, len(self.design.shapes))),
                            width=2,
                        ),
                    )
                )
            for box, color, style, width in (
                (phantom_box, (240, 240, 240), Qt.SolidLine, 2),
                (region_box, (80, 200, 120), Qt.DashLine, 3),
                (sequence_box, (255, 150, 50), Qt.SolidLine, 2),
            ):
                if box is None:
                    continue
                outline = projected_outline_mm(box, plane)
                add(
                    pg.PlotDataItem(
                        outline[:, 0],
                        outline[:, 1],
                        pen=pg.mkPen(color, width=width, style=style),
                    )
                )
            plot.enableAutoRange()

        voxel = region_m / counts * 1000.0
        whole = " × ".join(str(spin.value()) for spin in self.matrix_spins)
        info = (
            "Simulated matrix "
            + " × ".join(str(int(value)) for value in counts)
            + f" ({int(np.prod(counts))} voxels), voxel "
            + " × ".join(f"{value:.3g}" for value in voxel)
            + f" mm. Whole phantom: {whole}."
        )
        if self._sequence_geometry is None or self._sequence_geometry.fov_m is None:
            info += " The current sequence has no encoded FOV to compare with."
        self.simulation_region_info.setText(info)
        note = sequence_overhang_note(
            sequence_overhang_m(self._sequence_geometry, region_box)
        )
        self.simulation_fov_warning.setText("" if note is None else note)
        self.simulation_fov_warning.setVisible(note is not None)

        for item in self._region_peak_items:
            self.region_spectrum_plot.removeItem(item)
        self._region_peak_items = []
        peaks = self._designed_peak_positions()
        kinetic_names = (
            (
                self.pyruvate_peak_name.text().strip(),
                self.lactate_peak_name.text().strip(),
            )
            if self.dynamic_enabled.isChecked()
            else ()
        )
        kept, dropped = [], []
        for name, ppm in peaks:
            inside = ppm_range is None or ppm_range[0] <= ppm <= ppm_range[1]
            (kept if inside or name in kinetic_names else dropped).append((name, ppm))
            line = pg.InfiniteLine(
                pos=ppm,
                angle=90,
                pen=pg.mkPen((80, 200, 120) if inside else (150, 150, 150), width=2),
                label=name,
                labelOpts={
                    "position": 0.9,
                    "rotateAxis": (1, 0),
                    "anchors": [(0, 0), (0, 0)],
                },
            )
            self.region_spectrum_plot.addItem(line)
            self._region_peak_items.append(line)
        self._updating, updating = True, self._updating
        try:
            self.simulation_ppm_region.setRegion(
                (self.simulation_ppm_low.value(), self.simulation_ppm_high.value())
            )
        finally:
            self._updating = updating
        self.simulation_ppm_region.setVisible(ppm_range is not None)
        positions = [ppm for _name, ppm in peaks]
        if ppm_range is not None:
            positions.extend(ppm_range)
        if positions:
            self.region_spectrum_plot.setXRange(
                min(positions) - 2.0, max(positions) + 2.0, padding=0.05
            )

        def listed(entries):
            return ", ".join(
                f"{html.escape(name)} ({ppm:.4g} ppm)" for name, ppm in entries
            )

        text = f"<b>Simulated:</b> {listed(kept) or 'none'}"
        if dropped:
            text += f"<br><b>Left out:</b> {listed(dropped)}"
        if ppm_range is not None and any(
            name in kinetic_names and not (ppm_range[0] <= ppm <= ppm_range[1])
            for name, ppm in peaks
        ):
            text += (
                "<br>Pyruvate and lactate stay in a dynamic phantom because "
                "the kinetics need both pools."
            )
        if not kept:
            text += (
                "<br><span style='color:#b91c1c'>No peak lies in the range; "
                "the phantom cannot be built.</span>"
            )
        self.simulation_ppm_info.setText(text)

    def _load_design_into_ui(self):
        self._updating = True
        self.name_edit.setText(self.design.name)
        for widget, value in zip(self.matrix_spins, self.design.shape):
            widget.setValue(int(value))
        for widget, value in zip(self.fov_spins, self.design.fov_m):
            widget.setValue(float(value) * 1000.0)
        self._load_simulation_region_into_ui()
        self.field_strength_t.setValue(float(self.design.field_strength_t))
        self.temperature_c.setValue(self.design.temperature_k - 273.15)
        self.isotope_enrichment.setValue(100.0 * self.design.isotope_enrichment)
        nucleus_index = self.nucleus.findData(self.design.nucleus)
        self.nucleus.setCurrentIndex(max(0, nucleus_index))
        self.spectral_reference_ppm.setValue(self.design.spectral_reference_ppm)
        self.spectral_window_center_ppm.setValue(
            self.design.effective_spectral_window_center_ppm
        )
        self.spectral_bandwidth_ppm.setValue(self.design.spectral_bandwidth_ppm)
        self.spectral_points.setValue(int(self.design.spectral_points))
        self.supersampling_enabled.setChecked(self.design.supersampling_enabled)
        self.supersampling_factor.setValue(int(self.design.supersampling_factor))
        self._update_spectral_resolution_info()
        self._update_thermal_polarization_info()
        mode_index = self.b0_mode_combo.findData(self.design.b0_inhomogeneity_mode)
        self.b0_mode_combo.setCurrentIndex(max(0, mode_index))
        self.b0_inhomogeneity_ppm.setValue(self.design.b0_inhomogeneity_ppm)
        for spin, value in zip(
            self.b0_center_spins, self.design.b0_inhomogeneity_center
        ):
            spin.setValue(float(value) * 100.0)
        self._update_b0_pattern_info()
        self.dynamic_enabled.setChecked(self.design.dynamic_enabled)
        self.kinetics_model_controls.setEnabled(self.design.dynamic_enabled)
        self.pyruvate_peak_name.setText(self.design.pyruvate_peak_name)
        self.lactate_peak_name.setText(self.design.lactate_peak_name)
        self.shape_kpl.setValue(self.design.default_kpl_s_inv)
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
        self._update_curve_controls_enabled()
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
        self._populate_b0_offset_table()
        self._b0_shape_cache = None
        self._updating = False
        self._simulation_region_changed()
        self._refresh_metabolite_presets()
        if self.tabs.currentWidget() is self.b0_page:
            self._refresh_b0_preview()
        if self.design.shapes:
            self.shape_list.setCurrentRow(0)
        else:
            self._update_shape_preview()
            self._update_kinetics_preview()
            self._update_spectral_preview()

    def _create_roi(self, item, index):
        position = (
            item.center[0] - item.size[0] / 2,
            item.center[1] - item.size[1] / 2,
        )
        size = item.size[:2]
        roi_class = (
            ScaleOnlyEllipseROI
            if item.kind in {"ellipsoid", "cylinder"}
            else pg.RectROI
        )
        roi = roi_class(
            position,
            size,
            pen=self._roi_pen(index, False),
            rotatable=False,
        )
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
        if selected:
            color = pg.mkColor(255, 220, 80, 235)
        else:
            color.setAlpha(30)
        return pg.mkPen(
            color,
            width=2.5 if selected else 1,
            style=Qt.DashLine,
        )

    def _projection_pen(self, index, selected):
        color = (
            pg.mkColor(255, 220, 80, 255)
            if selected
            else pg.intColor(index, hues=max(1, len(self.design.shapes)), alpha=130)
        )
        return pg.mkPen(color, width=5 if selected else 2)

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
            selected = index == selected_row
            projection.setPen(self._projection_pen(index, selected))
            projection.setZValue(900 if selected else 500)

    def _current_row(self):
        row = self.shape_list.currentRow()
        return row if 0 <= row < len(self.design.shapes) else None

    def _shape_selected(self, row):
        if self._updating or not (0 <= row < len(self.design.shapes)):
            return
        self._updating = True
        try:
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
            self.shape_kpl.setValue(
                item.effective_kpl_s_inv(self.design.default_kpl_s_inv)
            )
            self._populate_peaks(item)
            self._update_xy_info(row)
        finally:
            # A stuck flag would silently drop every later table edit.
            self._updating = False
        self._set_kinetics_preview_shape(row)
        self._update_roi_highlights(row)
        self._update_shape_preview()
        self._update_kinetics_preview()
        self._update_spectral_preview()

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
                try:
                    self.x_center.setValue(item.center[0] * 100.0)
                    self.y_center.setValue(item.center[1] * 100.0)
                    self.x_size.setValue(item.size[0] * 100.0)
                    self.y_size.setValue(item.size[1] * 100.0)
                finally:
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
        volume_ml = (
            size_mm[0] * size_mm[1] * z_size_mm * _SHAPE_VOLUME_FACTOR[item.kind] / 1e3
        )
        self.xy_info.setText(
            f"centre=({center_mm[0]:.3g}, {center_mm[1]:.3g}) mm; "
            f"size=({size_mm[0]:.3g}, {size_mm[1]:.3g}) mm; "
            f"{'length' if item.kind == 'cylinder' else 'Z size'}="
            f"{z_size_mm:.3g} mm; volume {volume_ml:.4g} ml"
        )

    def _update_object_amount_info(self, *_):
        """Show the object volume and its observed nuclei."""
        if self._updating or not hasattr(self, "object_amount_info"):
            return
        fov_m = tuple(widget.value() / 1000.0 for widget in self.fov_spins)
        # A coarse voxel-centre grid keeps this cheap for large matrices; the
        # shapes are smooth, so the volume stays within about a percent.
        grid = tuple(
            min(int(widget.value()), _AMOUNT_GRID_LIMIT) for widget in self.matrix_spins
        )
        probe = PhantomDesign(shape=grid, fov_m=fov_m, supersampling_enabled=False)
        voxel_ml = float(np.prod(fov_m) / np.prod(grid)) * 1e6
        covered = np.zeros(grid, dtype=bool)
        molecules_mol = 0.0
        try:
            for item in self.design.shapes:
                mask = probe.rasterize_mask(item)
                covered |= mask
                concentration_mm = sum(peak.amplitude for peak in item.peaks)
                # mM × ml × 1e-6 = mol
                molecules_mol += concentration_mm * mask.sum() * voxel_ml * 1e-6
        except ValueError:
            self.object_amount_info.setText("")
            return
        enrichment = self.isotope_enrichment.value() / 100.0
        self.object_amount_info.setText(
            f"Object volume ≈ {covered.sum() * voxel_ml:.4g} ml; initially "
            f"{format_amount(molecules_mol * enrichment)} observed "
            f"{self._effective_nucleus()} nuclei "
            f"({enrichment * 100:.4g} % of {format_amount(molecules_mol)})."
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
        self._update_object_amount_info()
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
        self.design.spectral_window_center_ppm = self.spectral_window_center_ppm.value()
        self._inspector_preview_dirty = True
        self._update_spectral_resolution_info()
        self._update_thermal_polarization_info()
        self._update_spectral_preview()

    def _update_spectral_resolution_info(self):
        points = max(2, int(self.spectral_points.value()))
        resolution = self.spectral_bandwidth_ppm.value() / (points - 1)
        self.spectral_resolution_info.setText(f"Resolution {resolution:.5g} ppm/pt")

    def _update_spectral_preview(self):
        if not hasattr(self, "spectral_preview_curve"):
            return
        row = self._current_row()
        if row is None or not (0 <= row < len(self.design.shapes)):
            self.spectral_preview_curve.setData([], [])
            return

        item = self.design.shapes[row]
        phantom_reference_ppm = float(self.spectral_reference_ppm.value())
        sequence_reference_ppm = float(self.sequence_reference_ppm)
        window_center_ppm = float(self.spectral_window_center_ppm.value())
        bandwidth_ppm = float(self.spectral_bandwidth_ppm.value())
        points = max(2, int(self.spectral_points.value()))
        half_bandwidth_ppm = bandwidth_ppm / 2.0
        frequency_ppm = np.linspace(
            window_center_ppm - half_bandwidth_ppm,
            window_center_ppm + half_bandwidth_ppm,
            points,
        )
        spectrum = np.zeros(points, dtype=float)
        nucleus = self._effective_nucleus()
        field_strength_t = float(self.field_strength_t.value())
        for peak in item.peaks:
            fwhm_ppm = abs(
                float(
                    hz_to_ppm(
                        1.0 / (np.pi * peak.t2_star_s),
                        field_strength_t,
                        nucleus,
                    )
                )
            )
            half_width_ppm = max(fwhm_ppm / 2.0, np.finfo(float).eps)
            peak_ppm = phantom_reference_ppm + peak.frequency_ppm + item.b0_ppm
            amplitude = peak.amplitude * peak.effective_initial_polarization(
                item.initial_mz
            )
            spectrum += amplitude / (
                1.0 + ((frequency_ppm - peak_ppm) / half_width_ppm) ** 2
            )

        self.spectral_preview_curve.setData(frequency_ppm, spectrum)
        self.spectral_reference_line.setValue(sequence_reference_ppm)
        display_min = min(float(frequency_ppm[0]), sequence_reference_ppm)
        display_max = max(float(frequency_ppm[-1]), sequence_reference_ppm)
        if np.isclose(display_min, display_max):
            display_min -= 0.5
            display_max += 0.5
        self.spectral_preview_plot.setXRange(display_min, display_max, padding=0.04)
        self.spectral_preview_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.spectral_preview_plot.setToolTip(
            f"{item.name}: {frequency_ppm[0]:g}–{frequency_ppm[-1]:g} ppm · "
            f"orange dashed line: sequence reference {sequence_reference_ppm:g} ppm"
        )

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
        item.kpl_s_inv = self.shape_kpl.value()
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
        self._update_spectral_preview()

    def _populate_peaks(self, item):
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(len(item.peaks))
        for row, peak in enumerate(item.peaks):
            for column, value in enumerate(
                (
                    peak.name,
                    peak.amplitude,
                    # Show the effective values: peaks without their own
                    # entry inherit the shape fallback, which is not edited
                    # separately any more.
                    peak.effective_initial_polarization(item.initial_mz),
                    peak.frequency_ppm + self.spectral_reference_ppm.value(),
                    peak.effective_t1_s(item.t1_s) * 1000,
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
        self._inspector_preview_dirty = True
        self._update_spectral_preview()
        self._update_object_amount_info()
        self._update_kinetics_preview()

    def _read_peak_table(self):
        peaks = []
        reference_ppm = self.spectral_reference_ppm.value()
        row = self._current_row()
        shape = (
            self.design.shapes[row]
            if row is not None and 0 <= row < len(self.design.shapes)
            else None
        )

        def optional_value(text, peak_row, attribute, fallback, scale=1.0):
            # The table shows inherited shape values. Keep them inherited (None)
            # until the user enters a different value, so loaded designs keep
            # their fallbacks.
            if not text:
                return None
            value = float(text) / scale
            if shape is not None and peak_row < len(shape.peaks):
                if getattr(shape.peaks[peak_row], attribute) is None and np.isclose(
                    value, fallback, rtol=1e-9, atol=0.0
                ):
                    return None
            return value

        for peak_row in range(self.peak_table.rowCount()):
            polarization_text = self.peak_table.item(peak_row, 2).text().strip()
            t1_text = self.peak_table.item(peak_row, 4).text().strip()
            t1_s = optional_value(
                t1_text,
                peak_row,
                "t1_s",
                None if shape is None else shape.t1_s,
                scale=1000.0,
            )
            initial_polarization = optional_value(
                polarization_text,
                peak_row,
                "initial_polarization",
                None if shape is None else shape.initial_mz,
            )
            peak = SpectralPeakDefinition(
                name=self.peak_table.item(peak_row, 0).text(),
                amplitude=float(self.peak_table.item(peak_row, 1).text()),
                frequency_ppm=(
                    float(self.peak_table.item(peak_row, 3).text()) - reference_ppm
                ),
                t2_star_s=float(self.peak_table.item(peak_row, 5).text()) / 1000.0,
                t1_s=t1_s,
                initial_polarization=initial_polarization,
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
                interpolation="linear",
                outside="zero",
            ),
            TimeCurve(
                times_s=times,
                values=tuple(item[2] for item in samples),
                interpolation="linear",
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

    def _dynamic_b0_curve_table_changed(self, row, column):
        if column == 0:
            self._sort_time_curve_table(self.dynamic_b0_curve_table, row, column)
        self._inspector_preview_dirty = True

    def _update_curve_controls_enabled(self, *_):
        inflow_enabled = self.inflow_enabled.isChecked()
        for widget in (
            self.inflow_curve_table,
            self.add_inflow_point,
            self.remove_inflow_point,
        ):
            widget.setEnabled(inflow_enabled)
        b0_enabled = self.dynamic_b0_enabled.isChecked()
        for widget in (
            self.dynamic_b0_curve_table,
            self.add_b0_point,
            self.remove_b0_point,
        ):
            widget.setEnabled(b0_enabled)
        self._inspector_preview_dirty = True

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

    def _shape_kpl_changed(self, value):
        if self._updating:
            return
        shape_row = self.kinetics_preview_shape.currentData()
        if shape_row is None:
            shape_row = self._current_row()
        if shape_row is None:
            return
        self.design.shapes[int(shape_row)].kpl_s_inv = float(value)
        self._inspector_preview_dirty = True
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
            shape_row = self.kinetics_preview_shape.currentData()
            if shape_row is None:
                raise ValueError("select a shape to set its kPL")
            shape = self.design.shapes[int(shape_row)]
            return (
                shape.effective_kpl_s_inv(self.design.default_kpl_s_inv),
                "shape-specific kPL",
            )
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
        region_name = None
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
        if selected_shape_row is None:
            kpl_text = "kPL unavailable"
        else:
            shape_kpl = self.design.shapes[selected_shape_row].effective_kpl_s_inv(
                self.design.default_kpl_s_inv
            )
            kpl_text = f"kPL: {shape_kpl:.4g} s⁻¹"
        if highlighted_region is not None:
            kpl_text = (
                f"kPL override: {region_name} "
                f"({highlighted_region.kpl_s_inv:.4g} s⁻¹)"
            )
        self.kinetics_spatial_preview_info.setText(f"Shape: {shape_name} · {kpl_text}")

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
            self._clear_kinetics_preview_curves()
            for line in (*self._sequence_start_lines, *self._conversion_start_lines):
                line.setVisible(False)
            self.kinetics_preview_info.setText(
                "Conversion model off. Peaks still start at their initial "
                "polarization and relax with T1 toward 1 × thermal; enable the "
                "model above to add pyruvate → lactate conversion and inflow."
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
            concentration = (
                pyruvate_peak.amplitude,
                lactate_peak.amplitude,
            )
            initial_polarization = (
                pyruvate_peak.effective_initial_polarization(shape.initial_mz),
                lactate_peak.effective_initial_polarization(shape.initial_mz),
            )
            initial_mz = (
                concentration[0] * initial_polarization[0],
                concentration[1] * initial_polarization[1],
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
                initial_concentration=concentration,
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
            if (
                self.time_curve_display.currentData() == "set"
                and sequence_inflow_curve is not None
            ):
                self.inflow_preview_curve.setData(
                    sequence_inflow_curve.times_s,
                    sequence_inflow_curve.values,
                    pen=None,
                    symbol="o",
                    symbolSize=7,
                    symbolPen=pg.mkPen("y"),
                    symbolBrush=pg.mkBrush("y"),
                    fillLevel=None,
                    brush=None,
                )
            else:
                self.inflow_preview_curve.setData(
                    times_s,
                    inflow,
                    pen=pg.mkPen("y", width=2),
                    symbol=None,
                    fillLevel=0.0,
                    brush=pg.mkBrush(255, 255, 0, 45),
                )
            self.pyruvate_concentration_curve.setData(times_s, concentrations[0])
            self.lactate_concentration_curve.setData(times_s, concentrations[1])
            polarization = np.divide(
                pools,
                concentrations,
                out=np.zeros_like(pools),
                where=concentrations > 1e-15,
            )
            self.pyruvate_polarization_curve.setData(times_s, polarization[0])
            self.lactate_polarization_curve.setData(times_s, polarization[1])
            self.pyruvate_preview_curve.setData(times_s, pools[0])
            self.lactate_preview_curve.setData(times_s, pools[1])
            self.inflow_preview_plot.setXRange(preview_start_s, duration_s, padding=0.0)
            has_preroll = preview_start_s < 0.0
            converts = not np.isclose(sequence_conversion_start_s, 0.0)
            for line in self._sequence_start_lines:
                line.setVisible(has_preroll)
            for line in self._conversion_start_lines:
                line.setPos(sequence_conversion_start_s)
                line.setVisible(converts)
            if np.any(np.abs(inflow) > 1e-15):
                self.inflow_preview_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            else:
                self.inflow_preview_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
                self.inflow_preview_plot.setYRange(0.0, 1.0, padding=0.05)
            for plot in (
                self.concentration_preview_plot,
                self.polarization_preview_plot,
                self.pool_preview_plot,
            ):
                plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            zero_index = int(np.searchsorted(times_s, 0.0))

            def pair(values, suffix=""):
                return f"{values[0]:.4g}{suffix} / {values[1]:.4g}{suffix}"

            lines = [
                f"Representative voxel in {shape.name} · kPL={kpl_s_inv:.4g} s⁻¹ "
                f"({kpl_label}), active from sequence "
                f"t={sequence_conversion_start_s:.4g} s (kinetics "
                f"t={conversion_start_s:.4g} s); kinetics t at sequence t=0: "
                f"{kinetics_time_offset_s:.4g} s.",
                f"Values as pyruvate / lactate. Start (sequence "
                f"t={preview_start_s:.4g} s): concentration {pair(concentration)}, "
                f"polarization {pair(initial_polarization, ' × thermal')}, "
                f"Mz = C × P {pair(initial_mz)}.",
                f"sequence-start Mz at t=0: "
                f"{pair((pools[0, zero_index], pools[1, zero_index]))}; at "
                f"t={duration_s:.4g} s: Mz {pair(pools[:, -1])}, polarization "
                f"{pair(polarization[:, -1], ' × thermal')}. T1 {pair(t1_s, ' s')}.",
            ]
            explanations = []
            if initial_polarization[0] <= 1.0 and inflow_curve is None:
                explanations.append(
                    "Pyruvate starts at thermal polarization (not hyperpolarized); "
                    "set its Initial polarization on the Draw tab, e.g. 10000 × "
                    "thermal."
                )
            if kpl_s_inv == 0.0:
                explanations.append(
                    "kPL=0: no P→L conversion; each existing pool only follows its "
                    "own T1 relaxation toward polarization 1, while enabled inflow "
                    "is added only to Pyruvate."
                )
                if initial_mz[1] > 0:
                    explanations.append(
                        "Lactate starts above zero because both its concentration "
                        "and initial polarization are non-zero; this lactate was "
                        "initialized, not created by conversion."
                    )
                else:
                    explanations.append(
                        "Lactate Mz starts at zero and recovers thermally toward "
                        "Mz = concentration (polarization 1), even when kPL=0."
                    )
            elif initial_mz[1] == 0.0:
                explanations.append(
                    "Lactate Mz starts at zero; kPL conversion adds "
                    "pyruvate-derived lactate magnetization alongside thermal "
                    "recovery toward polarization 1."
                )
            if np.allclose(pools[0], pools[1], rtol=1e-7, atol=1e-10):
                explanations.append(
                    "Pyruvate and lactate Mz are identical and overlap: cyan is "
                    "solid, magenta is dashed so both curves remain recognizable."
                )
            self.kinetics_preview_info.setText("\n".join((*lines, *explanations)))
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            self._clear_kinetics_preview_curves()
            self.kinetics_preview_info.setText(f"Preview unavailable: {exc}")

    def _clear_kinetics_preview_curves(self):
        for curve in (
            self.inflow_preview_curve,
            self.pyruvate_concentration_curve,
            self.lactate_concentration_curve,
            self.pyruvate_polarization_curve,
            self.lactate_polarization_curve,
            self.pyruvate_preview_curve,
            self.lactate_preview_curve,
        ):
            curve.setData([], [])

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
        if self.dynamic_enabled.isChecked() and item.kpl_s_inv is None:
            item.kpl_s_inv = DEFAULT_ENABLED_KPL_S_INV
        self.design.shapes.append(item)
        self.shape_list.addItem(item.name)
        self._create_roi(item, len(self.design.shapes) - 1)
        self._refresh_kinetics_preview_shapes()
        self.shape_list.setCurrentRow(len(self.design.shapes) - 1)

    def _add_shape(self, kind):
        self._sync_global()
        item = ShapeDefinition(
            name=self._next_shape_name(),
            kind=kind,
            peaks=[SpectralPeakDefinition(t2_star_s=0.020)],
        )
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
            peaks=[SpectralPeakDefinition(t2_star_s=0.020)],
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
        else:
            self._update_spectral_preview()
        self._update_roi_highlights()
        self._update_shape_preview()
        self._update_kinetics_preview()

    def _add_peak(self):
        row = self._current_row()
        if row is None:
            return
        self.design.shapes[row].peaks.append(
            SpectralPeakDefinition(
                name=f"Peak {len(self.design.shapes[row].peaks) + 1}",
                t2_star_s=0.020,
            )
        )
        self._populate_peaks(self.design.shapes[row])
        self._update_kinetics_preview()
        self._inspector_preview_dirty = True
        self._update_spectral_preview()

    def _add_metabolite_preset(self):
        shape_row = self._current_row()
        preset = self.metabolite_preset_combo.currentData()
        if shape_row is None or not isinstance(preset, MetabolitePreset):
            return
        try:
            peaks = self._read_peak_table()
        except (AttributeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Invalid peak table",
                f"Correct the current peak values before adding a preset:\n{exc}",
            )
            return
        existing_row = next(
            (index for index, peak in enumerate(peaks) if peak.name == preset.name),
            None,
        )
        if existing_row is not None:
            self.peak_table.setCurrentCell(existing_row, 0)
            QMessageBox.information(
                self,
                "Metabolite already present",
                f"{preset.name} is already defined for this shape.",
            )
            return
        peaks.append(
            SpectralPeakDefinition(
                name=preset.name,
                amplitude=1.0,
                frequency_ppm=(
                    preset.chemical_shift_ppm - self.spectral_reference_ppm.value()
                ),
                t2_star_s=preset.t2_star_s,
                t1_s=preset.t1_s,
            )
        )
        self.design.shapes[shape_row].peaks = peaks
        self._populate_peaks(self.design.shapes[shape_row])
        self.peak_table.setCurrentCell(len(peaks) - 1, 0)
        self._update_kinetics_preview()
        self._inspector_preview_dirty = True
        self._update_spectral_preview()

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
        self._inspector_preview_dirty = True
        self._update_spectral_preview()

    def _sync_global(self):
        self._properties_changed()
        row = self._current_row()
        if row is not None:
            self.design.shapes[row].peaks = self._read_peak_table()
            self._update_spectral_preview()
        self.design.name = self.name_edit.text().strip() or "Designed spectral phantom"
        self.design.shape = tuple(widget.value() for widget in self.matrix_spins)
        self.design.fov_m = tuple(widget.value() / 1000.0 for widget in self.fov_spins)
        self.design.field_strength_t = self.field_strength_t.value()
        self.design.temperature_k = self.temperature_c.value() + 273.15
        self.design.isotope_enrichment = self.isotope_enrichment.value() / 100.0
        (
            self.design.simulation_fov_m,
            self.design.simulation_offset_m,
            self.design.simulation_ppm_range,
        ) = self._simulation_region_values()
        self.design.nucleus = self.nucleus.currentData()
        self.design.spectral_reference_ppm = self.spectral_reference_ppm.value()
        self.design.spectral_window_center_ppm = self.spectral_window_center_ppm.value()
        self.design.spectral_bandwidth_ppm = self.spectral_bandwidth_ppm.value()
        self.design.spectral_points = self.spectral_points.value()
        self.design.supersampling_enabled = self.supersampling_enabled.isChecked()
        self.design.supersampling_factor = self.supersampling_factor.value()
        self.design.b0_inhomogeneity_mode = str(self.b0_mode_combo.currentData())
        self.design.b0_inhomogeneity_ppm = self.b0_inhomogeneity_ppm.value()
        self.design.b0_inhomogeneity_center = self._b0_center_fraction()
        dynamic_requested = (
            self.dynamic_enabled.isChecked()
            or self.inflow_enabled.isChecked()
            or self.dynamic_b0_enabled.isChecked()
        )
        # Inflow or B0 drift need the time-resolved phantom; switching it on
        # for them must not invent a conversion rate the user never asked for.
        self._suppress_default_kpl = True
        try:
            self.dynamic_enabled.setChecked(dynamic_requested)
        finally:
            self._suppress_default_kpl = False
        self.design.dynamic_enabled = dynamic_requested
        self.design.pyruvate_peak_name = self.pyruvate_peak_name.text().strip()
        self.design.lactate_peak_name = self.lactate_peak_name.text().strip()
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
        if self.tabs.widget(index) is self.region_page:
            self._sync_global()
            self._update_region_preview()
        if self.tabs.widget(index) is self.b0_page:
            self._populate_b0_offset_table()
            self._b0_shape_cache = None
            self._refresh_b0_preview()
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
            self.phantom = self._build_phantom()
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
            # The file already contains the rasterized phantom represented by
            # the loaded design. Mark that preview as current so opening the
            # inspector later does not rebuild it (which is especially costly
            # for highly supersampled designs), and leave the user's current
            # editor tab unchanged.
            self._inspector_preview_dirty = False
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def accept(self):
        try:
            self._sync_global()
            self.phantom = self._build_phantom()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid phantom", str(exc))
            return
        super().accept()

    def _build_phantom(self):
        phantom = self.design.build()
        # The orange reference line marks the sequence carrier, which can
        # differ from the design's internal peak-offset reference.
        phantom.metadata["sequence_reference_ppm"] = float(self.sequence_reference_ppm)
        return phantom

    def get_phantom(self):
        return self.phantom
