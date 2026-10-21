"""Pre-run planning view of object, simulated grid and sequence coverage.

Three orthogonal sum projections of the selected object are overlaid with
the simulated phantom grid, the configured object extent (mouse phantoms can
extend beyond or sit off the grid), the encoded sequence FOV and any 2D
slices.  Boxes are oriented by the sequence encoding frame and drawn as the
convex hull of their projected corners, so oblique frames are handled too.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsPolygonItem,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..phantom import Phantom
from .plot_interaction import AXIS_ZOOM_TOOLTIP

AXES = "xyz"
# (horizontal, vertical) scanner axes of each orthogonal view.
VIEW_PLANES = ((0, 1), (0, 2), (1, 2))
MAX_PROJECTION_CELLS = 256

GRID_COLOR = "#9e9e9e"
OBJECT_COLOR = "#4caf50"
SEQUENCE_COLOR = "#ff9800"
SLICE_COLOR = "#26c6da"
PROBE_COLOR = "#e91e63"


@dataclass(frozen=True)
class OrientedBox:
    """A box with its edge directions given as scanner-xyz columns."""

    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))

    def corners(self) -> np.ndarray:
        half = 0.5 * np.asarray(self.size_m, dtype=float)
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        axes = np.asarray(self.axes, dtype=float)
        return np.asarray(self.center_m, dtype=float) + (signs * half) @ axes.T


@dataclass
class SequenceGeometry:
    """Spatial coverage of the selected sequence in scanner coordinates."""

    frame: np.ndarray
    fov_m: Optional[tuple[float, ...]] = None
    matrix: Optional[tuple[int, ...]] = None
    slice_positions_m: tuple[float, ...] = ()
    slice_thickness_m: Optional[float] = None
    name: str = "Sequence"

    @property
    def is_2d(self) -> bool:
        return self.fov_m is not None and len(self.fov_m) == 2

    def fov_box(self, fallback_partition_m: float) -> Optional[OrientedBox]:
        """Return the encoded FOV; 2D packages span their slices."""
        if self.fov_m is None:
            return None
        if len(self.fov_m) >= 3:
            return OrientedBox((0.0, 0.0, 0.0), tuple(self.fov_m[:3]), self.frame)
        if self.slice_positions_m and self.slice_thickness_m:
            low = min(self.slice_positions_m) - 0.5 * self.slice_thickness_m
            high = max(self.slice_positions_m) + 0.5 * self.slice_thickness_m
        else:
            low, high = -0.5 * fallback_partition_m, 0.5 * fallback_partition_m
        center = 0.5 * (low + high) * np.asarray(self.frame, dtype=float)[:, 2]
        return OrientedBox(
            tuple(center), (*self.fov_m[:2], high - low), np.asarray(self.frame)
        )

    def slice_boxes(self, scanner_extent_m: Sequence[float]) -> list[OrientedBox]:
        """Return slice slabs; unencoded slices span the given object extent."""
        if not self.slice_thickness_m:
            return []
        frame = np.asarray(self.frame, dtype=float)
        if self.fov_m is not None:
            in_plane_m = self.fov_m[:2]
        else:
            in_plane_m = np.abs(frame[:, :2]).T @ np.asarray(scanner_extent_m, float)
        normal = frame[:, 2]
        return [
            OrientedBox(
                tuple(position * normal),
                (in_plane_m[0], in_plane_m[1], self.slice_thickness_m),
                np.asarray(self.frame),
            )
            for position in self.slice_positions_m
        ]


def sequence_geometry_from_definitions(
    definitions: dict,
    frame_matrix,
    *,
    fov_m: Optional[Sequence[float]] = None,
    matrix: Optional[Sequence[int]] = None,
    name: str = "Sequence",
) -> SequenceGeometry:
    """Collect FOV and slice placement from Pulseq-style definitions."""
    normalized = {str(key).lower(): value for key, value in dict(definitions).items()}

    def scalar(key):
        try:
            value = float(np.ravel(normalized[key])[0])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    thickness = scalar("slicethickness")
    positions: tuple[float, ...] = ()
    if thickness is not None and thickness > 0.0:
        raw_positions = normalized.get("slicepositions")
        if raw_positions is not None:
            try:
                values = np.ravel(np.asarray(raw_positions, dtype=float))
            except (TypeError, ValueError):
                values = np.empty(0)
            positions = tuple(float(value) for value in values if np.isfinite(value))
        if not positions:
            offset = scalar("sliceoffset")
            positions = (offset if offset is not None else 0.0,)
    else:
        thickness = None
    return SequenceGeometry(
        frame=np.asarray(frame_matrix, dtype=float),
        fov_m=None if fov_m is None else tuple(float(value) for value in fov_m),
        matrix=None if matrix is None else tuple(int(value) for value in matrix),
        slice_positions_m=positions,
        slice_thickness_m=thickness,
        name=name,
    )


def phantom_grid_box(phantom) -> OrientedBox:
    """Return the full simulated grid extent, voxel edges included."""
    shape = tuple(int(value) for value in phantom.shape)
    fov = tuple(float(value) for value in phantom.fov)
    affine = getattr(phantom, "affine_ijk_to_xyz_m", None)
    if affine is None:
        affine = Phantom.default_affine(shape, fov)
    affine = np.asarray(affine, dtype=float)
    ndim = len(shape)
    low_ijk = np.full(3, -0.5)
    high_ijk = np.array([count - 0.5 for count in shape] + [0.5] * (3 - ndim))
    if ndim < 3:
        low_ijk[ndim:] = 0.0
        high_ijk[ndim:] = 0.0
    low = affine[:3, :3] @ low_ijk + affine[:3, 3]
    high = affine[:3, :3] @ high_ijk + affine[:3, 3]
    return OrientedBox(tuple(0.5 * (low + high)), tuple(np.abs(high - low)), np.eye(3))


def configured_object_box(phantom) -> Optional[OrientedBox]:
    """Return a designed object extent that is independent of the grid."""
    config = getattr(phantom, "config", None)
    size = getattr(config, "mouse_size_m", None)
    if size is None:
        return None
    offset = getattr(config, "mouse_offset_m", (0.0, 0.0, 0.0))
    return OrientedBox(
        tuple(float(value) for value in offset),
        tuple(float(value) for value in size),
    )


def _object_density(phantom) -> np.ndarray:
    labels = getattr(phantom, "anatomy_labels", None)
    if labels is not None:
        return (np.asarray(labels) > 0).astype(np.float32)
    pd_map = getattr(phantom, "pd_map", None)
    if pd_map is not None:
        density = np.abs(np.asarray(pd_map)).astype(np.float32)
        if density.shape == tuple(phantom.shape) and np.any(density > 0):
            return density
    return np.asarray(phantom.mask, dtype=np.float32)


def object_projections(phantom) -> list[tuple[np.ndarray, QRectF]]:
    """Return XY, XZ and YZ sum projections with their mm rectangles."""
    density = _object_density(phantom)
    while density.ndim < 3:
        density = density[..., np.newaxis]
    # Projections only need to show the outline; strided sampling keeps
    # large dynamic phantoms responsive.
    steps = tuple(max(1, int(np.ceil(n / MAX_PROJECTION_CELLS))) for n in density.shape)
    density = density[:: steps[0], :: steps[1], :: steps[2]]
    box = phantom_grid_box(phantom)
    low = (np.asarray(box.center_m) - 0.5 * np.asarray(box.size_m)) * 1000.0
    size = np.asarray(box.size_m) * 1000.0
    size = np.where(size > 0.0, size, 1.0)
    low = np.where(np.asarray(box.size_m) > 0.0, low, -0.5)
    projections = []
    for horizontal, vertical in VIEW_PLANES:
        summed_axis = 3 - horizontal - vertical
        image = density.sum(axis=summed_axis)
        peak = float(np.max(image)) if image.size else 0.0
        if peak > 0.0:
            image = image / peak
        projections.append(
            (
                image,
                QRectF(
                    low[horizontal], low[vertical], size[horizontal], size[vertical]
                ),
            )
        )
    return projections


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return the closed convex hull (Andrew's monotone chain) of 2D points."""
    unique = np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)
    if len(unique) <= 2:
        return np.vstack([unique, unique[:1]])
    ordered = sorted(map(tuple, unique))

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return np.vstack([hull, hull[:1]])


def projected_outline_mm(box: OrientedBox, plane: tuple[int, int]) -> np.ndarray:
    corners = box.corners()[:, list(plane)] * 1000.0
    return convex_hull_2d(corners)


def coverage_notes(
    phantom, sequence: Optional[SequenceGeometry], object_box=None
) -> list[str]:
    """Explain where sequence coverage and simulated extent disagree."""
    notes = []
    grid = phantom_grid_box(phantom) if phantom is not None else None
    if object_box is not None and grid is not None:
        grid_low = np.asarray(grid.center_m) - 0.5 * np.asarray(grid.size_m)
        grid_high = np.asarray(grid.center_m) + 0.5 * np.asarray(grid.size_m)
        object_low = np.asarray(object_box.center_m) - 0.5 * np.asarray(
            object_box.size_m
        )
        object_high = np.asarray(object_box.center_m) + 0.5 * np.asarray(
            object_box.size_m
        )
        cut = [
            AXES[axis]
            for axis in range(3)
            if object_low[axis] < grid_low[axis] - 1e-9
            or object_high[axis] > grid_high[axis] + 1e-9
        ]
        if cut:
            notes.append(
                "The configured object extends beyond the simulated grid along "
                f"{', '.join(cut)}; only the part inside the grid is simulated."
            )
    if sequence is None:
        notes.append("No sequence is loaded yet. Generate or load one to see its FOV.")
        return notes
    if sequence.fov_m is None:
        if sequence.slice_thickness_m:
            notes.append(
                "The sequence has no spatial encoding; the selected slice "
                "contributes its whole in-plane extent to one signal."
            )
        else:
            notes.append(
                "The sequence has no spatial encoding; the whole object "
                "contributes to one signal."
            )
        return notes
    if grid is None:
        return notes
    frame = np.asarray(sequence.frame, dtype=float)
    required = np.abs(frame).T @ np.asarray(grid.size_m, dtype=float)
    roles = ("read", "phase", "partition")
    encoded = len(sequence.fov_m)
    for index in range(encoded):
        sequence_fov = sequence.fov_m[index]
        if sequence_fov < required[index] and not np.isclose(
            sequence_fov, required[index], rtol=1e-9, atol=1e-12
        ):
            notes.append(
                f"{roles[index].title()} FOV {sequence_fov * 1000:.4g} mm is smaller "
                f"than the simulated grid ({required[index] * 1000:.4g} mm): signal "
                "outside the FOV aliases."
            )
    if sequence.matrix is not None:
        voxel = np.asarray(grid.size_m) / np.maximum(
            np.asarray(
                tuple(int(value) for value in phantom.shape)
                + (1,) * (3 - len(phantom.shape))
            ),
            1,
        )
        phantom_widths = np.abs(frame).T @ voxel
        for index in range(min(encoded, len(sequence.matrix))):
            pixel = sequence.fov_m[index] / max(int(sequence.matrix[index]), 1)
            if phantom_widths[index] > pixel and not np.isclose(
                phantom_widths[index], pixel, rtol=1e-9, atol=1e-12
            ):
                notes.append(
                    f"Phantom cells ({phantom_widths[index] * 1000:.3g} mm) are "
                    f"coarser than the {roles[index]} pixel "
                    f"({pixel * 1000:.3g} mm)."
                )
    return notes


def _format_mm(values) -> str:
    return " × ".join(f"{float(value) * 1000:.4g}" for value in values) + " mm"


class SequenceGeometryViewer(QWidget):
    """Orthogonal projections of the object with grid and sequence FOV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        grid = QGridLayout()
        self.plots = []
        self.images = []
        self._overlay_items = []
        for index, (horizontal, vertical) in enumerate(VIEW_PLANES):
            plot = pg.PlotWidget()
            plot.setToolTip(AXIS_ZOOM_TOOLTIP)
            plot.setAspectLocked(True)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setTitle(f"{AXES[horizontal].upper()}–{AXES[vertical].upper()}")
            plot.setLabel("bottom", AXES[horizontal], units="mm")
            plot.setLabel("left", AXES[vertical], units="mm")
            image = pg.ImageItem(axisOrder="col-major")
            plot.addItem(image)
            self.plots.append(plot)
            self.images.append(image)
            grid.addWidget(plot, index // 2, index % 2)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.summary, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)
        self.update_geometry(None, None)

    def _clear_overlays(self):
        for plot, item in self._overlay_items:
            plot.removeItem(item)
        self._overlay_items = []

    def _add_outline(
        self, box, color, *, style=Qt.SolidLine, fill_alpha=0, z=10, thin_axis=None
    ):
        for plot, plane in zip(self.plots, VIEW_PLANES):
            if thin_axis is not None:
                # A slab seen face-on would only cover the object; its
                # in-plane extent is already shown by the sequence FOV.
                viewing_axis = 3 - plane[0] - plane[1]
                if abs(np.asarray(box.axes, float)[viewing_axis, thin_axis]) > 0.7:
                    continue
            outline = projected_outline_mm(box, plane)
            item = pg.PlotDataItem(
                outline[:, 0],
                outline[:, 1],
                pen=pg.mkPen(color, width=2, style=style),
            )
            if fill_alpha:
                fill_color = pg.mkColor(color)
                fill_color.setAlpha(fill_alpha)
                fill = QGraphicsPolygonItem(
                    QPolygonF([QPointF(x, y) for x, y in outline[:-1]])
                )
                fill.setBrush(pg.mkBrush(fill_color))
                fill.setPen(pg.mkPen(None))
                fill.setZValue(z - 1)
                plot.addItem(fill)
                self._overlay_items.append((plot, fill))
            item.setZValue(z)
            plot.addItem(item)
            self._overlay_items.append((plot, item))

    def _add_point(self, position_m, color):
        for plot, plane in zip(self.plots, VIEW_PLANES):
            point = np.asarray(position_m, dtype=float)[list(plane)] * 1000.0
            item = pg.ScatterPlotItem(
                [point[0]], [point[1]], size=10, brush=pg.mkBrush(color), pen=None
            )
            item.setZValue(20)
            plot.addItem(item)
            self._overlay_items.append((plot, item))

    def update_geometry(
        self,
        phantom,
        sequence: Optional[SequenceGeometry],
        *,
        object_label: str = "",
        probe_position_m=None,
    ):
        """Redraw the projections and overlays for the current selection."""
        self._clear_overlays()
        rows = []
        object_box = None
        if phantom is None:
            for image in self.images:
                image.clear()
        else:
            for image, (projection, rect) in zip(
                self.images, object_projections(phantom)
            ):
                image.setImage(projection, levels=(0.0, 1.0), autoLevels=False)
                image.setRect(rect)
            grid_box = phantom_grid_box(phantom)
            self._add_outline(grid_box, GRID_COLOR, style=Qt.DashLine, z=11)
            rows.append(
                (
                    GRID_COLOR,
                    "Simulated grid",
                    f"{_format_mm(phantom.fov)}, matrix {tuple(phantom.shape)}",
                )
            )
            object_box = configured_object_box(phantom)
            if object_box is not None:
                self._add_outline(object_box, OBJECT_COLOR, style=Qt.DashLine, z=12)
                rows.append(
                    (
                        OBJECT_COLOR,
                        "Object extent",
                        f"{_format_mm(object_box.size_m)} centred at "
                        f"({', '.join(f'{value * 1000:.3g}' for value in object_box.center_m)}) mm",
                    )
                )
        if probe_position_m is not None:
            self._add_point(probe_position_m, PROBE_COLOR)
            rows.append(
                (
                    PROBE_COLOR,
                    "Spin probe",
                    "("
                    + ", ".join(f"{value * 1000:.3g}" for value in probe_position_m)
                    + ") mm",
                )
            )
        if sequence is not None:
            fallback = (
                float(np.max(phantom_grid_box(phantom).size_m))
                if phantom is not None
                else 0.0
            )
            fov_box = sequence.fov_box(fallback)
            if fov_box is not None:
                self._add_outline(fov_box, SEQUENCE_COLOR, z=13)
                encoded = (
                    f"{_format_mm(sequence.fov_m)}"
                    + (
                        f", matrix {' × '.join(str(n) for n in sequence.matrix)}"
                        if sequence.matrix
                        else ""
                    )
                    + (" (2D, read × phase)" if sequence.is_2d else "")
                )
                rows.append((SEQUENCE_COLOR, "Sequence FOV", encoded))
            extent = (
                phantom_grid_box(phantom).size_m
                if phantom is not None
                else (fallback,) * 3
            )
            slices = sequence.slice_boxes(extent)
            for box in slices:
                self._add_outline(box, SLICE_COLOR, fill_alpha=50, z=14, thin_axis=2)
            if slices:
                rows.append(
                    (
                        SLICE_COLOR,
                        "Slices" if len(slices) > 1 else "Slice",
                        f"{len(slices)} × {sequence.slice_thickness_m * 1000:.3g} mm "
                        "at "
                        + ", ".join(
                            f"{value * 1000:.3g}"
                            for value in sequence.slice_positions_m
                        )
                        + " mm",
                    )
                )
        for plot in self.plots:
            plot.enableAutoRange()
        notes = (
            coverage_notes(phantom, sequence, object_box)
            if (phantom is not None or sequence is not None)
            else []
        )
        self.summary.setText(self._summary_html(object_label, rows, notes))

    @staticmethod
    def _summary_html(object_label, rows, notes) -> str:
        parts = []
        if object_label:
            parts.append(f"<b>{html.escape(object_label)}</b>")
        if rows:
            parts.append(
                "<table cellspacing='2'>"
                + "".join(
                    f"<tr><td><span style='color:{color}'>■</span> "
                    f"{html.escape(label)}</td>"
                    f"<td>{html.escape(value)}</td></tr>"
                    for color, label, value in rows
                )
                + "</table>"
            )
        if notes:
            parts.append(
                "<ul style='margin-left:-20px'>"
                + "".join(f"<li>{html.escape(note)}</li>" for note in notes)
                + "</ul>"
            )
        if not parts:
            parts.append("Select an object and a sequence to inspect their geometry.")
        return "".join(parts)
