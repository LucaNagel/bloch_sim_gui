"""Pre-run planning view of object, simulated grid and sequence coverage.

Three orthogonal sections through the selected object, cut at the sequence
centre and drawn over the faded object silhouette, are overlaid with the
simulated phantom grid, the configured object extent (mouse phantoms can
extend beyond or sit off the grid), the encoded sequence FOV and any 2D
slices.  Boxes are oriented by the sequence encoding frame and drawn as the
convex hull of their projected corners, so oblique frames are handled too.

The object is coloured by the property that differs most across it: anatomy
or species regions when there are several, otherwise the map with the
largest relative spread (kPL, concentration, T1, ...).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import binary_erosion
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QPolygonF
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsPolygonItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..mouse_phantom import MOUSE_TISSUE_COLORS
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
# The last generated FOV and slices while the views preview new parameters.
GENERATED_COLOR = "#ffcc80"
GENERATED_ALPHA = 140

# Region colours for categorical maps without their own palette.
CATEGORY_COLORS = (
    (78, 121, 167),
    (242, 142, 43),
    (225, 87, 89),
    (118, 183, 178),
    (89, 161, 79),
    (237, 201, 72),
    (176, 122, 161),
    (255, 157, 167),
    (156, 117, 95),
    (186, 176, 172),
)
CONTRAST_COLORMAP = "viridis"
# Object parts outside the shown section.
SILHOUETTE_COLOR = (64, 64, 64)
# Maps varying by less than this fraction leave the object uniform.
MIN_CONTRAST_SPREAD = 0.01

STALE_SEQUENCE_MESSAGE = (
    "Sequence parameters changed since the shown sequence was generated."
)

# Logical axes a drag in the views can move, in frame-column order.
OFFSET_ROLES = ("read", "phase", "slice")
# Dragged offsets snap to this step.
OFFSET_SNAP_M = 1e-4
DRAG_TOOLTIP = (
    "Drag to move the FOV centre and slice package; hold Shift to move along "
    "one axis only. Right-click while dragging cancels."
)


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
    # Logical read/phase position of the FOV centre.
    in_plane_center_m: tuple[float, float] = (0.0, 0.0)

    @property
    def is_2d(self) -> bool:
        return self.fov_m is not None and len(self.fov_m) == 2

    def _in_plane_shift(self) -> np.ndarray:
        frame = np.asarray(self.frame, dtype=float)
        read, phase = (float(value) for value in self.in_plane_center_m)
        return read * frame[:, 0] + phase * frame[:, 1]

    def fov_box(self, fallback_partition_m: float) -> Optional[OrientedBox]:
        """Return the encoded FOV; 2D packages span their slices."""
        if self.fov_m is None:
            return None
        if len(self.fov_m) >= 3:
            return OrientedBox(
                tuple(self._in_plane_shift()), tuple(self.fov_m[:3]), self.frame
            )
        if self.slice_positions_m and self.slice_thickness_m:
            low = min(self.slice_positions_m) - 0.5 * self.slice_thickness_m
            high = max(self.slice_positions_m) + 0.5 * self.slice_thickness_m
        else:
            low, high = -0.5 * fallback_partition_m, 0.5 * fallback_partition_m
        center = (
            0.5 * (low + high) * np.asarray(self.frame, dtype=float)[:, 2]
            + self._in_plane_shift()
        )
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
            shift = self._in_plane_shift()
        else:
            in_plane_m = np.abs(frame[:, :2]).T @ np.asarray(scanner_extent_m, float)
            shift = np.zeros(3)
        normal = frame[:, 2]
        return [
            OrientedBox(
                tuple(position * normal + shift),
                (in_plane_m[0], in_plane_m[1], self.slice_thickness_m),
                np.asarray(self.frame),
            )
            for position in self.slice_positions_m
        ]

    def covers_same_region(self, other: "SequenceGeometry") -> bool:
        """Return whether both geometries place the same FOV and slices."""
        return all(
            _same_values(first, second)
            for first, second in (
                (self.frame, other.frame),
                (self.fov_m, other.fov_m),
                (self.in_plane_center_m, other.in_plane_center_m),
                (self.slice_positions_m, other.slice_positions_m),
                (self.slice_thickness_m, other.slice_thickness_m),
            )
        )

    def offsets_m(self) -> np.ndarray:
        """Return the read, phase and slice-package centre offsets."""
        package = (
            float(np.mean(self.slice_positions_m)) if self.slice_positions_m else 0.0
        )
        return np.array([*self.in_plane_center_m, package], dtype=float)

    def dragged_offsets_m(
        self, plane, delta_mm, adjustable, *, single_axis=False
    ) -> np.ndarray:
        """Return the offsets after dragging the placement in one view.

        The view shift ``delta_mm`` is projected onto the read, phase and
        slice axes. Only the ``adjustable`` roles move, with ``single_axis``
        only the one that moves most, and moved offsets snap to 0.1 mm.
        """
        shift = np.zeros(3)
        shift[list(plane)] = np.asarray(delta_mm, dtype=float) / 1000.0
        moved = np.asarray(self.frame, dtype=float).T @ shift
        moved[[role not in adjustable for role in OFFSET_ROLES]] = 0.0
        if single_axis:
            moved[np.arange(3) != int(np.argmax(np.abs(moved)))] = 0.0
        offsets = self.offsets_m()
        changed = np.abs(moved) > 0.0
        offsets[changed] = (
            np.round((offsets[changed] + moved[changed]) / OFFSET_SNAP_M)
            * OFFSET_SNAP_M
        )
        return offsets

    def view_shift_mm(self, plane, offsets_m) -> np.ndarray:
        """Return how far new offsets move the placement in one view."""
        shift = np.asarray(self.frame, dtype=float) @ (
            np.asarray(offsets_m, dtype=float) - self.offsets_m()
        )
        return shift[list(plane)] * 1000.0

    def center_text(self) -> str:
        """Describe a shifted FOV centre, e.g. ``centre read 6, phase -4 mm``."""
        read, phase = (value * 1000 for value in self.in_plane_center_m)
        if self.fov_m is None or np.allclose((read, phase), 0.0, atol=1e-9):
            return ""
        return f"centre read {read:.3g}, phase {phase:.3g} mm"

    def slice_text(self) -> str:
        """Describe the slice package, e.g. ``2 × 3 mm at -1.5, 1.5 mm``."""
        if not self.slice_thickness_m or not self.slice_positions_m:
            return ""
        positions = ", ".join(f"{value * 1000:.3g}" for value in self.slice_positions_m)
        return (
            f"{len(self.slice_positions_m)} × {self.slice_thickness_m * 1000:.3g} mm "
            f"at {positions} mm"
        )


def _same_values(first, second) -> bool:
    if first is None or second is None:
        return first is None and second is None
    first = np.ravel(np.asarray(first, dtype=float))
    second = np.ravel(np.asarray(second, dtype=float))
    return first.shape == second.shape and np.allclose(
        first, second, rtol=1e-9, atol=1e-12
    )


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
        in_plane_center_m=(scalar("readoffset") or 0.0, scalar("phaseoffset") or 0.0),
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
        # Phantom Designer objects simulated only in part keep their full FOV.
        design = getattr(phantom, "metadata", {}).get("phantom_design")
        if not isinstance(design, dict) or design.get("simulation_fov_m") is None:
            return None
        return OrientedBox(
            tuple(
                float(value)
                for value in design.get("simulation_offset_m", (0.0, 0.0, 0.0))
            ),
            tuple(float(value) for value in design["fov_m"]),
        )
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


def _volume(values) -> np.ndarray:
    values = np.asarray(values)
    while values.ndim < 3:
        values = values[..., np.newaxis]
    return values


def _projection_layout(phantom, shape) -> tuple[tuple[slice, ...], list[QRectF]]:
    """Return the sampling slices and mm rectangles of the three views."""
    # Projections only need to show the outline; strided sampling keeps
    # large dynamic phantoms responsive.
    sampling = tuple(
        slice(None, None, max(1, int(np.ceil(n / MAX_PROJECTION_CELLS)))) for n in shape
    )
    box = phantom_grid_box(phantom)
    low = (np.asarray(box.center_m) - 0.5 * np.asarray(box.size_m)) * 1000.0
    size = np.asarray(box.size_m) * 1000.0
    size = np.where(size > 0.0, size, 1.0)
    low = np.where(np.asarray(box.size_m) > 0.0, low, -0.5)
    rects = [
        QRectF(low[horizontal], low[vertical], size[horizontal], size[vertical])
        for horizontal, vertical in VIEW_PLANES
    ]
    return sampling, rects


@dataclass
class ObjectContrast:
    """A voxel property of the object that can colour the projections.

    Categorical maps hold integer region labels and list them in
    ``categories`` as ``label -> (name, rgb)``; ``score`` is then the number
    of regions inside the object.  Continuous maps score their relative
    spread over the object, offsets (B0, shifts, pH) their spread relative to
    ``offset_scale``.
    """

    key: str
    label: str
    values: np.ndarray
    unit: str = ""
    categories: Optional[dict] = None
    offset_scale: Optional[float] = None
    score: float = 0.0
    value_range: tuple[float, float] = (0.0, 0.0)

    @property
    def categorical(self) -> bool:
        return self.categories is not None

    @property
    def title(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


def _attribute(*names):
    def get(phantom):
        value = phantom
        for name in names:
            value = getattr(value, name, None)
            if value is None:
                return None
        return value

    return get


def _concentration_maps(phantom) -> dict:
    maps = getattr(phantom, "concentration_maps", None)
    if maps is None:
        maps = getattr(phantom, "initial_concentration_maps", None)
    return dict(maps or {})


def _total_concentration(phantom):
    maps = _concentration_maps(phantom)
    if not maps:
        return None
    return sum(np.asarray(values, dtype=float) for values in maps.values())


# (key, label, unit, getter, offset scale) of continuous object properties,
# in the order that breaks ties of the automatic choice.
CONTINUOUS_CONTRASTS = (
    ("concentration", "Total concentration", "", _total_concentration, None),
    ("pd", "Proton density", "", _attribute("pd_map"), None),
    ("t1", "T1", "s", _attribute("t1_map"), None),
    ("t2", "T2", "s", _attribute("t2_map"), None),
    ("t2_star", "T2*", "s", _attribute("t2_star_map"), None),
    ("kpl", "kPL", "1/s", _attribute("kpl_map_s_inv"), None),
    ("kpa", "kPA", "1/s", _attribute("kpa_map_s_inv"), None),
    (
        "delivery",
        "Bolus delivery",
        "",
        _attribute("pyruvate_inflow", "delivery_map"),
        None,
    ),
    (
        "arrival",
        "Bolus arrival delay",
        "s",
        _attribute("pyruvate_inflow", "arrival_delay_map_s"),
        None,
    ),
    (
        "washout",
        "Washout rate",
        "1/s",
        _attribute("voxel_washout", "loss_map_s_inv"),
        None,
    ),
    ("ph", "pH", "", _attribute("ph_map"), 1.0),
    ("b0_ppm", "B0 offset", "ppm", _attribute("b0_map_ppm"), 1.0),
    ("b0", "B0 offset", "Hz", _attribute("b0_map"), 50.0),
    # Spectral phantoms derive df_map in ppm from their species mix.
    (
        "df",
        "Off-resonance",
        "Hz",
        lambda phantom: phantom.df_map if isinstance(phantom, Phantom) else None,
        50.0,
    ),
    ("chemical_shift", "Chemical shift", "Hz", _attribute("chemical_shift_map"), 50.0),
)


def _labels_by_size(values) -> list[int]:
    labels, counts = np.unique(values, return_counts=True)
    return [int(labels[index]) for index in np.argsort(-counts, kind="stable")]


def _categorical_contrasts(phantom, inside) -> list[ObjectContrast]:
    contrasts = []
    labels = getattr(phantom, "anatomy_labels", None)
    if labels is not None and np.shape(labels) == inside.shape:
        names = dict(getattr(phantom, "tissue_label_names", {}) or {})
        values = np.asarray(labels).astype(np.int64)
        categories = {}
        for label in _labels_by_size(values[inside]):
            color = (
                MOUSE_TISSUE_COLORS[label]
                if 0 <= label < len(MOUSE_TISSUE_COLORS)
                else CATEGORY_COLORS[label % len(CATEGORY_COLORS)]
            )
            categories[label] = (str(names.get(label, f"Label {label}")), color)
        contrasts.append(
            ObjectContrast("anatomy", "Anatomy", values, categories=categories)
        )
    maps = _concentration_maps(phantom)
    if len(maps) >= 2:
        names = list(maps)
        stacked = np.stack([np.asarray(maps[name], dtype=float) for name in names])
        if stacked.shape[1:] == inside.shape:
            dominant = np.argmax(stacked, axis=0) + 1
            dominant[stacked.max(axis=0) <= 0.0] = 0
            categories = {
                label: (
                    names[label - 1],
                    CATEGORY_COLORS[(label - 1) % len(CATEGORY_COLORS)],
                )
                for label in _labels_by_size(dominant[inside])
                if label > 0
            }
            contrasts.append(
                ObjectContrast(
                    "species", "Dominant species", dominant, categories=categories
                )
            )
    for contrast in contrasts:
        contrast.score = float(len(contrast.categories))
    return [contrast for contrast in contrasts if contrast.score >= 2]


def _continuous_contrasts(phantom, inside) -> list[ObjectContrast]:
    specs = list(CONTINUOUS_CONTRASTS)
    for name, values in dict(
        getattr(phantom, "chemical_shift_maps_ppm", None) or {}
    ).items():
        specs.append(
            (f"shift:{name}", f"Shift {name}", "ppm", lambda _p, v=values: v, 1.0)
        )
    for name, values in _concentration_maps(phantom).items():
        specs.append(
            (
                f"concentration:{name}",
                f"{name} concentration",
                "",
                lambda _p, v=values: v,
                None,
            )
        )
    # Partial-volume voxels at the object edge would make every uniform map
    # look varying, so the spread is scored on the interior only (full
    # connectivity, so diagonal rim voxels are dropped too).
    interior = binary_erosion(inside, structure=np.ones((3,) * inside.ndim, bool))
    if not np.any(interior):
        interior = inside
    contrasts = []
    for key, label, unit, getter, offset_scale in specs:
        try:
            raw = getter(phantom)
        except (AttributeError, TypeError, ValueError):
            continue
        if raw is None:
            continue
        values = np.asarray(raw)
        if values.shape != inside.shape or not np.isrealobj(values):
            continue
        values = values.astype(float, copy=False)
        selected = values[inside]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            continue
        if any(np.array_equal(values, other.values) for other in contrasts):
            continue
        low, high = float(selected.min()), float(selected.max())
        if high <= low:
            continue
        core = values[interior]
        core = core[np.isfinite(core)]
        core_low, core_high = (
            (float(core.min()), float(core.max())) if core.size else (low, high)
        )
        if offset_scale is None:
            scale = max(abs(core_low), abs(core_high))
            score = (core_high - core_low) / scale if scale > 0.0 else 0.0
        else:
            score = (core_high - core_low) / offset_scale
        contrasts.append(
            ObjectContrast(
                key,
                label,
                values,
                unit=unit,
                offset_scale=offset_scale,
                score=min(1.0, score),
                value_range=(low, high),
            )
        )
    return contrasts


def object_contrasts(phantom) -> list[ObjectContrast]:
    """Return the object properties that can colour the projections."""
    if phantom is None:
        return []
    inside = np.asarray(_object_density(phantom)) > 0.0
    if not np.any(inside):
        return []
    return _categorical_contrasts(phantom, inside) + _continuous_contrasts(
        phantom, inside
    )


def automatic_contrast(contrasts) -> Optional[ObjectContrast]:
    """Pick region labels when there are several, else the widest map."""
    for contrast in contrasts:
        if contrast.categorical and contrast.score >= 2:
            return contrast
    varying = [
        contrast
        for contrast in contrasts
        if not contrast.categorical and contrast.score >= MIN_CONTRAST_SPREAD
    ]
    # max() keeps the first of equal scores, i.e. the table order.
    return max(varying, key=lambda contrast: contrast.score) if varying else None


def _section_indices(phantom, shape, center_m) -> list[Optional[int]]:
    """Return the sampled voxel index of ``center_m`` along each grid axis.

    ``None`` marks an axis whose section plane lies outside the grid.
    """
    box = phantom_grid_box(phantom)
    size = np.asarray(box.size_m, dtype=float)
    low = np.asarray(box.center_m, dtype=float) - 0.5 * size
    center = np.asarray(box.center_m if center_m is None else center_m, dtype=float)
    indices = []
    for axis, count in enumerate(shape):
        if count <= 1 or size[axis] <= 0.0:
            indices.append(0)
            continue
        fraction = (center[axis] - low[axis]) / size[axis]
        if not 0.0 <= fraction <= 1.0:
            indices.append(None)
            continue
        indices.append(min(int(np.floor(fraction * count)), count - 1))
    return indices


def object_sections(phantom, contrast: Optional[ObjectContrast] = None, center_m=None):
    """Return RGBA sections through ``center_m`` and the shown value range.

    Each view cuts the object at ``center_m`` (default: the grid centre), so
    a uniform object looks uniform; averaging along the viewing ray would
    turn partial-volume edges into apparent contrast.  The rest of the
    object stays visible as a faded silhouette.  Without ``contrast`` the
    object density is shown in gray.  The range is ``None`` for region
    labels and the density.
    """
    density = _volume(_object_density(phantom))
    sampling, rects = _projection_layout(phantom, density.shape)
    density = density[sampling]
    inside = density > 0.0
    values = density if contrast is None else _volume(contrast.values)[sampling]
    indices = _section_indices(phantom, inside.shape, center_m)
    shown_range = None
    if contrast is None:
        peak = float(np.max(density)) if density.size else 0.0
        lookup = np.repeat(np.arange(256, dtype=np.ubyte)[:, np.newaxis], 3, axis=1)
        low, high = 0.0, peak
    elif not contrast.categorical:
        lookup = pg.colormap.get(CONTRAST_COLORMAP).getLookupTable(
            0.0, 1.0, 256, alpha=False
        )
        low, high = shown_range = contrast.value_range
    images = []
    for horizontal, vertical in VIEW_PLANES:
        axis = 3 - horizontal - vertical
        covered = np.any(inside, axis=axis)
        rgba = np.zeros(covered.shape + (4,), dtype=np.ubyte)
        rgba[covered, :3] = SILHOUETTE_COLOR
        index = indices[axis]
        if index is not None:
            cut = np.take(inside, index, axis=axis)
            cut_values = np.take(values, index, axis=axis)
            if contrast is not None and contrast.categorical:
                for label, (_name, color) in contrast.categories.items():
                    rgba[cut & (cut_values == label), :3] = color
            else:
                finite = cut & np.isfinite(cut_values)
                if high > low:
                    level = np.clip((cut_values - low) / (high - low), 0.0, 1.0)
                else:
                    level = np.full(cut_values.shape, 0.5)
                level = np.where(finite, level, 0.0)
                rgba[finite, :3] = lookup[np.round(level * 255).astype(int)][finite]
        rgba[..., 3] = np.where(covered, 255, 0)
        images.append(rgba)
    return list(zip(images, rects)), shown_range


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


def sequence_overhang_m(
    sequence: Optional[SequenceGeometry], grid_box: Optional[OrientedBox]
) -> np.ndarray:
    """Return how far the sequence FOV reaches beyond a box along x, y and z.

    A 2D FOV spans its slice package. Without slices only its plane counts.
    """
    overhang = np.zeros(3)
    if sequence is None or grid_box is None or sequence.fov_m is None:
        return overhang
    corners = sequence.fov_box(0.0).corners()
    grid_center = np.asarray(grid_box.center_m, dtype=float)
    grid_half = 0.5 * np.asarray(grid_box.size_m, dtype=float)
    below = (grid_center - grid_half) - corners.min(axis=0)
    above = corners.max(axis=0) - (grid_center + grid_half)
    overhang = np.maximum(np.maximum(below, above), 0.0)
    # Rounding of millimetre inputs is not an overhang.
    overhang[overhang < 1e-7] = 0.0
    return overhang


def sequence_overhang_note(overhang_m) -> Optional[str]:
    """Describe a sequence FOV that reaches beyond the simulated phantom."""
    beyond = [
        f"{AXES[axis]} ({value * 1000:.3g} mm)"
        for axis, value in enumerate(np.asarray(overhang_m, dtype=float))
        if value > 0.0
    ]
    if not beyond:
        return None
    return (
        "The sequence FOV reaches beyond the simulated phantom FOV along "
        f"{', '.join(beyond)}. No spins are simulated there, so any object "
        "part outside is missing from the signal."
    )


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
    overhang_note = sequence_overhang_note(sequence_overhang_m(sequence, grid))
    if overhang_note is not None:
        notes.append(overhang_note)
    frame = np.asarray(sequence.frame, dtype=float)
    required = np.abs(frame).T @ np.asarray(grid.size_m, dtype=float)
    grid_centers = frame.T @ np.asarray(grid.center_m, dtype=float)
    fov_centers = (*sequence.in_plane_center_m, 0.0)
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
            continue
        overhang = abs(grid_centers[index] - fov_centers[index]) - 0.5 * (
            sequence_fov - required[index]
        )
        if index < 2 and overhang > 1e-9:
            notes.append(
                f"The simulated grid reaches {overhang * 1000:.3g} mm beyond the "
                f"offset {roles[index]} FOV: signal outside the FOV aliases."
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


def _swatch(color) -> str:
    red, green, blue = (int(value) for value in color[:3])
    return f"<span style='color:rgb({red},{green},{blue})'>■</span>"


def _contrast_legend_html(contrast: Optional[ObjectContrast], shown_range=None) -> str:
    """Return the legend of the projection colours."""
    if contrast is None:
        return ""
    title = f"<b>Contrast:</b> {html.escape(contrast.title)}"
    if contrast.categorical:
        # Non-breaking names keep each swatch next to its whole label.
        entries = " &nbsp; ".join(
            f"{_swatch(color)}&nbsp;{html.escape(name).replace(' ', '&nbsp;')}"
            for name, color in contrast.categories.values()
        )
        return f"<div>{title}<br>{entries}</div>"
    lookup = pg.colormap.get(CONTRAST_COLORMAP).getLookupTable(0.0, 1.0, 5, alpha=False)
    low, high = shown_range if shown_range is not None else contrast.value_range
    gradient = "".join(_swatch(color) for color in lookup)
    return (
        f"<div>{title}, section through the sequence centre<br>"
        f"{low:.3g} {gradient} {high:.3g}</div>"
    )


def _faded(color):
    faded = pg.mkColor(color)
    faded.setAlpha(GENERATED_ALPHA)
    return faded


def _format_mm(values) -> str:
    return " × ".join(f"{float(value) * 1000:.4g}" for value in values) + " mm"


def _outline_rect(outlines) -> QRectF:
    points = np.vstack(outlines)
    low, high = points.min(axis=0), points.max(axis=0)
    return QRectF(low[0], low[1], high[0] - low[0], high[1] - low[1])


class OffsetDragHandle(pg.ROI):
    """Translation-only area over the drawn FOV or slice package of one view.

    ``place(delta_mm, single_axis)`` returns how far the placement may move
    for a drag of ``delta_mm``; the handle follows that, not the cursor.
    """

    def __init__(self, rect: QRectF, plane, color, place):
        super().__init__(
            rect.topLeft(),
            rect.size(),
            movable=True,
            rotatable=False,
            resizable=False,
            removable=False,
            pen=pg.mkPen(None),
            hoverPen=pg.mkPen(color, width=3),
        )
        self.origin = pg.Point(rect.topLeft())
        self.plane = plane
        self._place = place
        self._drag_start = pg.Point(0.0, 0.0)
        self.setCursor(Qt.SizeAllCursor)
        self.setToolTip(DRAG_TOOLTIP)

    def shift_mm(self) -> np.ndarray:
        offset = self.pos() - self.origin
        return np.array([offset.x(), offset.y()], dtype=float)

    def drag_by(self, delta_mm, *, single_axis=False, finish=True):
        """Move by ``delta_mm`` from the drawn position, as far as allowed."""
        shown = self._place(np.asarray(delta_mm, dtype=float), single_axis)
        self.setPos(self.origin + pg.Point(*shown), finish=finish)

    def mouseDragEvent(self, ev):
        # pyqtgraph would read Shift as a scale drag; here it keeps one axis.
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self._moveStarted()
            self._drag_start = self.pos() - self.origin
        if not self.isMoving:  # a right click cancelled the drag
            return
        if ev.isFinish():
            self._moveFinished()
            return
        moved = self.mapToParent(ev.pos()) - self.mapToParent(ev.buttonDownPos())
        delta = self._drag_start + moved
        self.drag_by(
            (delta.x(), delta.y()),
            single_axis=bool(ev.modifiers() & Qt.ShiftModifier),
            finish=False,
        )


class SequenceGeometryViewer(QWidget):
    """Orthogonal projections of the object with grid and sequence FOV."""

    update_requested = pyqtSignal()
    # New read, phase and slice-package offsets (m) dragged in the views.
    geometry_offset_requested = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        # The views redraw on their own when the object changes, but a new
        # sequence FOV needs a regenerated sequence; this notice offers it.
        self.stale_notice = QFrame()
        self.stale_notice.setObjectName("geometryStaleNotice")
        self.stale_notice.setStyleSheet(
            "#geometryStaleNotice { background: #fef3c7; "
            "border: 1px solid #f59e0b; border-radius: 4px; }"
            "#geometryStaleNotice QLabel { color: #78350f; }"
        )
        notice_layout = QHBoxLayout(self.stale_notice)
        notice_layout.setContentsMargins(8, 4, 4, 4)
        self.stale_notice_label = QLabel()
        self.stale_notice_label.setWordWrap(True)
        self.update_button = QPushButton("Update")
        self.update_button.setToolTip(
            "Generate the sequence with the current parameters and redraw "
            "its FOV and slices."
        )
        self.update_button.clicked.connect(self.update_requested)
        notice_layout.addWidget(self.stale_notice_label, 1)
        notice_layout.addWidget(self.update_button)
        self.stale_notice.hide()
        layout.addWidget(self.stale_notice)
        grid = QGridLayout()
        self.plots = []
        self.images = []
        self.drag_handles = []
        self.drag_labels = []
        self._overlay_items = []
        for index, (horizontal, vertical) in enumerate(VIEW_PLANES):
            plot = pg.PlotWidget()
            plot.setToolTip(AXIS_ZOOM_TOOLTIP)
            plot.setAspectLocked(True)
            # Free zoom onto the object and out to the sequence FOV: with the
            # locked aspect, data-range limits would crop the object.
            plot.getViewBox().auto_data_limits = False
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setTitle(f"{AXES[horizontal].upper()}–{AXES[vertical].upper()}")
            plot.setLabel("bottom", AXES[horizontal], units="mm")
            plot.setLabel("left", AXES[vertical], units="mm")
            image = pg.ImageItem(axisOrder="col-major")
            plot.addItem(image)
            label = pg.TextItem(
                color="w", fill=pg.mkBrush(0, 0, 0, 170), anchor=(0.5, 0.5)
            )
            label.setZValue(30)
            label.hide()
            plot.addItem(label, ignoreBounds=True)
            self.plots.append(plot)
            self.images.append(image)
            self.drag_labels.append(label)
            grid.addWidget(plot, index // 2, index % 2)
        self.contrast_combo = QComboBox()
        self.contrast_combo.setToolTip(
            "Object property that colours the projections. Auto picks anatomy "
            "or species regions when there are several, otherwise the map that "
            "varies most across the object."
        )
        self.contrast_combo.currentIndexChanged.connect(self._contrast_changed)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast"))
        contrast_row.addWidget(self.contrast_combo, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addLayout(contrast_row)
        side_layout.addWidget(self.summary, 1)
        grid.addWidget(side_panel, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)
        self._phantom = None
        self._object_label = ""
        self._probe_position_m = None
        self._sequence = None
        self._generated_sequence = None
        self._adjustable_offsets = ()
        self.contrasts = []
        self.contrast = None
        self._contrast_range = None
        self._section_center_m = None
        self._summary_parts = ("", [], [])
        self.update_geometry(None, None)

    def set_stale(
        self,
        stale: bool,
        message: str = STALE_SEQUENCE_MESSAGE,
        question: str = "Update the geometry view?",
    ):
        """Show or hide the notice that the drawn sequence is outdated."""
        if stale:
            self.stale_notice_label.setText(f"{message} {question}")
        self.stale_notice.setVisible(bool(stale))

    def _populate_contrast_combo(self):
        previous = self.contrast_combo.currentData()
        automatic = automatic_contrast(self.contrasts)
        self.contrast_combo.blockSignals(True)
        try:
            self.contrast_combo.clear()
            self.contrast_combo.addItem(
                f"Auto: {automatic.title if automatic else 'object density'}", ""
            )
            for contrast in self.contrasts:
                self.contrast_combo.addItem(contrast.title, contrast.key)
            index = self.contrast_combo.findData(previous) if previous else 0
            self.contrast_combo.setCurrentIndex(max(index, 0))
        finally:
            self.contrast_combo.blockSignals(False)
        self.contrast_combo.setEnabled(bool(self.contrasts))

    def _selected_contrast(self) -> Optional[ObjectContrast]:
        key = self.contrast_combo.currentData()
        for contrast in self.contrasts:
            if key and contrast.key == key:
                return contrast
        return automatic_contrast(self.contrasts)

    def _draw_object(self):
        if self._phantom is None:
            self.contrast = None
            self._contrast_range = None
            for image in self.images:
                image.clear()
            return
        self.contrast = self._selected_contrast()
        sections, self._contrast_range = object_sections(
            self._phantom, self.contrast, self._section_center_m
        )
        for image, (rgba, rect) in zip(self.images, sections):
            image.setImage(rgba, levels=(0, 255), autoLevels=False)
            image.setRect(rect)

    def _contrast_changed(self, _index):
        self._draw_object()
        self._update_summary()

    def _update_summary(self):
        object_label, rows, notes = self._summary_parts
        self.summary.setText(
            self._summary_html(
                object_label,
                rows,
                notes,
                _contrast_legend_html(self.contrast, self._contrast_range),
            )
        )

    def _clear_overlays(self):
        for plot, item in self._overlay_items:
            plot.removeItem(item)
        self._overlay_items = []
        self.drag_handles = []
        for label in self.drag_labels:
            label.hide()

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
        generated_sequence: Optional[SequenceGeometry] = None,
        adjustable_offsets: Sequence[str] = (),
    ):
        """Redraw the projections and overlays for the current selection."""
        self._phantom = phantom
        self._object_label = object_label
        self._probe_position_m = probe_position_m
        self.contrasts = object_contrasts(phantom)
        self._populate_contrast_combo()
        self._section_center_m = self._sequence_center_m(sequence)
        self._draw_object()
        self.set_sequence_geometry(
            sequence,
            generated_sequence=generated_sequence,
            adjustable_offsets=adjustable_offsets,
        )
        for plot in self.plots:
            plot.enableAutoRange()

    def set_sequence_geometry(
        self,
        sequence: Optional[SequenceGeometry],
        *,
        generated_sequence: Optional[SequenceGeometry] = None,
        adjustable_offsets: Sequence[str] = (),
    ):
        """Redraw only the sequence overlays and keep the object projections.

        ``generated_sequence`` is the placement of the last generated
        sequence; while it differs from ``sequence`` it stays visible as a
        dotted outline. ``adjustable_offsets`` names the offsets of
        ``OFFSET_ROLES`` that dragging the FOV or slices may change.
        """
        self._sequence = sequence
        self._generated_sequence = generated_sequence
        self._adjustable_offsets = tuple(adjustable_offsets)
        center = self._sequence_center_m(sequence)
        if not _same_values(center, self._section_center_m):
            self._section_center_m = center
            self._draw_object()
        self._draw_overlays()

    def _sequence_center_m(self, sequence):
        """Return where the views cut the object: the FOV or slice centre."""
        if sequence is None:
            return None
        fallback = (
            float(np.max(phantom_grid_box(self._phantom).size_m))
            if self._phantom is not None
            else 0.0
        )
        box = sequence.fov_box(fallback)
        if box is None:
            slices = sequence.slice_boxes((fallback,) * 3)
            if not slices:
                return None
            return tuple(np.mean([item.center_m for item in slices], axis=0))
        return tuple(box.center_m)

    def _draw_overlays(self):
        phantom = self._phantom
        sequence = self._sequence
        probe_position_m = self._probe_position_m
        self._clear_overlays()
        rows = []
        object_box = None
        if phantom is not None:
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
            extent = (
                phantom_grid_box(phantom).size_m
                if phantom is not None
                else (fallback,) * 3
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
                    + (f", {sequence.center_text()}" if sequence.center_text() else "")
                )
                rows.append((SEQUENCE_COLOR, "Sequence FOV", encoded))
            slices = sequence.slice_boxes(extent)
            for box in slices:
                self._add_outline(box, SLICE_COLOR, fill_alpha=50, z=14, thin_axis=2)
            if slices:
                rows.append(
                    (
                        SLICE_COLOR,
                        "Slices" if len(slices) > 1 else "Slice",
                        sequence.slice_text(),
                    )
                )
            previous = self._generated_sequence
            if previous is not None and not previous.covers_same_region(sequence):
                rows.append(self._draw_generated_outline(previous, fallback, extent))
            self._add_drag_handles(sequence, fov_box, slices)
        notes = (
            coverage_notes(phantom, sequence, object_box)
            if (phantom is not None or sequence is not None)
            else []
        )
        self._summary_parts = (self._object_label, rows, notes)
        self._update_summary()

    def _add_drag_handles(self, sequence, fov_box, slices):
        """Make the FOV, or slices without one, draggable where offsets allow."""
        adjustable = [role in self._adjustable_offsets for role in OFFSET_ROLES]
        frame = np.asarray(sequence.frame, dtype=float)
        for index, (plot, plane) in enumerate(zip(self.plots, VIEW_PLANES)):
            if not np.any(np.abs(frame[list(plane)][:, adjustable]) > 1e-9):
                continue
            if fov_box is not None:
                outlines = [projected_outline_mm(fov_box, plane)]
            else:
                outlines = [projected_outline_mm(box, plane) for box in slices]
            if not outlines:
                continue

            def place(delta_mm, single_axis, plane=plane):
                return sequence.view_shift_mm(
                    plane,
                    sequence.dragged_offsets_m(
                        plane,
                        delta_mm,
                        self._adjustable_offsets,
                        single_axis=single_axis,
                    ),
                )

            handle = OffsetDragHandle(
                _outline_rect(outlines), plane, SEQUENCE_COLOR, place
            )
            handle.setZValue(15)
            handle.sigRegionChanged.connect(
                lambda moved, index=index: self._offset_drag_moved(moved, index)
            )
            handle.sigRegionChangeFinished.connect(self._offset_drag_finished)
            plot.addItem(handle, ignoreBounds=True)
            self._overlay_items.append((plot, handle))
            self.drag_handles.append(handle)

    def _dragged_offsets(self, handle) -> np.ndarray:
        return self._sequence.dragged_offsets_m(
            handle.plane, handle.shift_mm(), self._adjustable_offsets
        )

    def _offset_drag_moved(self, handle, index):
        if self._sequence is None:
            return
        moved_mm = (self._dragged_offsets(handle) - self._sequence.offsets_m()) * 1e3
        text = ", ".join(
            f"{role} {value:+.1f} mm"
            for role, value in zip(OFFSET_ROLES, moved_mm)
            if role in self._adjustable_offsets and abs(value) > 1e-6
        )
        label = self.drag_labels[index]
        label.setText(text or "no shift")
        label.setPos(handle.mapToParent(handle.boundingRect().center()))
        label.show()

    def _offset_drag_finished(self, handle):
        for label in self.drag_labels:
            label.hide()
        if self._sequence is None:
            return
        offsets = self._dragged_offsets(handle)
        if np.allclose(offsets, self._sequence.offsets_m(), rtol=0.0, atol=1e-9):
            return
        self.geometry_offset_requested.emit(*(float(value) for value in offsets))

    def _draw_generated_outline(self, sequence, fallback_partition_m, extent_m):
        """Draw the last generated placement dotted and return its summary row."""
        parts = []
        fov_box = sequence.fov_box(fallback_partition_m)
        if fov_box is not None:
            self._add_outline(fov_box, _faded(SEQUENCE_COLOR), style=Qt.DotLine, z=9)
            parts.append(_format_mm(sequence.fov_m))
            if sequence.center_text():
                parts.append(sequence.center_text())
        for box in sequence.slice_boxes(extent_m):
            self._add_outline(
                box, _faded(SLICE_COLOR), style=Qt.DotLine, z=10, thin_axis=2
            )
        if sequence.slice_text():
            parts.append(sequence.slice_text())
        return (
            GENERATED_COLOR,
            "Generated (dotted)",
            ", ".join(parts) or "no spatial encoding",
        )

    @staticmethod
    def _summary_html(object_label, rows, notes, contrast_html="") -> str:
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
        if contrast_html:
            parts.append(contrast_html)
        if notes:
            parts.append(
                "<ul style='margin-left:-20px'>"
                + "".join(f"<li>{html.escape(note)}</li>" for note in notes)
                + "</ul>"
            )
        if not parts:
            parts.append("Select an object and a sequence to inspect their geometry.")
        return "".join(parts)
