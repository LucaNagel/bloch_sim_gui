"""Place reconstructed images in the simulated phantom.

The reconstruction explorer shows the phantom next to the measured data, so
the image grid of a result is described in scanner coordinates here: every
array axis of the displayed image has a voxel-centre origin and a step
vector.  A 2D image is one layer whose depth is the slice thickness; images
without a slice (unselective excitation) project through the whole phantom.

With that grid the phantom is resampled onto the image plane (averaged over
the slab), the footprint of a selected voxel is turned into an oriented box,
the phantom cells inside it are summarised, and region boundaries become a
contour overlay for the reconstruction.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg

from ..phantom import Phantom
from ..sequence.encoding import EncodingFrame
from .sequence_geometry_viewer import (
    CONTRAST_COLORMAP,
    SILHOUETTE_COLOR,
    ObjectContrast,
    OrientedBox,
    _contrast_legend_html,
    _object_density,
    _swatch,
    _volume,
    phantom_grid_box,
)

# Display samples per image pixel and slab samples are capped so a large
# phantom never makes a selection change slow.
MAX_IN_PLANE_UPSAMPLING = 8
MAX_DISPLAY_SIZE = 512
MAX_DEPTH_SAMPLES = 64
MAX_VOLUME_UPSAMPLING = 4
MAX_SAMPLES = 4_000_000
CONTOUR_LEVELS = 8
OUTLINE_COLOR = (38, 198, 218)  # SLICE_COLOR of the geometry viewer
CONTOUR_ALPHA = 200


@dataclass(frozen=True)
class ImageGrid:
    """Voxel centres of an image as ``origin + sum(index * step)``.

    ``steps_m`` holds one scanner-xyz step per array axis.  2D images have a
    third axis of size one whose step is the slab normal times its depth;
    ``projection`` marks a depth that spans the phantom because the
    sequence selected no slice.
    """

    origin_m: np.ndarray
    steps_m: np.ndarray
    shape: tuple[int, int, int]
    planar: bool = False
    projection: bool = False

    @property
    def spacing_m(self) -> np.ndarray:
        return np.linalg.norm(self.steps_m, axis=1)

    @property
    def axes(self) -> np.ndarray:
        """Unit step directions as columns."""
        return (self.steps_m / self.spacing_m[:, np.newaxis]).T

    def center(self, index) -> np.ndarray:
        values = np.zeros(3)
        values[: len(index)] = np.asarray(index, dtype=float)
        return self.origin_m + values @ self.steps_m

    def voxel_box(self, index) -> OrientedBox:
        return OrientedBox(tuple(self.center(index)), tuple(self.spacing_m), self.axes)

    def box(self) -> OrientedBox:
        middle = 0.5 * (np.asarray(self.shape, dtype=float) - 1.0)
        return OrientedBox(
            tuple(self.center(middle)),
            tuple(self.spacing_m * np.asarray(self.shape)),
            self.axes,
        )

    def index_at(self, point_m) -> np.ndarray:
        """Return the fractional array index of a scanner position."""
        return np.linalg.solve(
            self.steps_m.T, np.asarray(point_m, dtype=float) - self.origin_m
        )

    def sample_points(self, upsampling) -> np.ndarray:
        """Return sub-voxel sample centres, shaped ``(*shape * up, 3)``."""
        axes = []
        for count, factor in zip(self.shape, upsampling):
            fine = (np.arange(count * factor) + 0.5) / factor - 0.5
            axes.append(fine)
        grid = np.meshgrid(*axes, indexing="ij")
        indices = np.stack(grid, axis=-1)
        return self.origin_m + indices @ self.steps_m


def dataset_definitions(dataset) -> dict:
    """Return the sequence definitions stored with a result dataset."""
    raw = dataset.attrs.get("metadata_json")
    if not raw:
        return {}
    try:
        metadata = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    definitions = metadata.get("sequence_definitions") or {}
    return dict(definitions) if isinstance(definitions, dict) else {}


def _normalized(definitions) -> dict:
    return {str(key).lower(): value for key, value in dict(definitions).items()}


def _scalar(definitions, key) -> Optional[float]:
    try:
        value = float(np.ravel(np.asarray(_normalized(definitions)[key], float))[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _values(definitions, key) -> np.ndarray:
    raw = _normalized(definitions).get(key)
    if raw is None:
        return np.empty(0)
    if isinstance(raw, str):
        cleaned = raw.strip().strip("[](){}").replace(",", " ")
        values = np.fromstring(cleaned, sep=" ", dtype=float)
    else:
        try:
            values = np.ravel(np.asarray(raw, dtype=float))
        except (TypeError, ValueError):
            return np.empty(0)
    return values[np.isfinite(values)]


def encoding_frame(dataset, definitions=None) -> np.ndarray:
    """Return the read/phase/partition directions as scanner-xyz columns."""
    basis = dataset.attrs.get("cartesian_encoding_basis_xyz")
    if basis:
        try:
            values = np.asarray(str(basis).split(","), dtype=float)
            if values.size == 9:
                return values.reshape(3, 3)
        except ValueError:
            pass
    definitions = dataset_definitions(dataset) if definitions is None else definitions
    try:
        return EncodingFrame.from_definitions(definitions).matrix
    except (TypeError, ValueError):
        return np.eye(3)


def phantom_affine(phantom) -> np.ndarray:
    shape = tuple(int(value) for value in phantom.shape)
    affine = getattr(phantom, "affine_ijk_to_xyz_m", None)
    if affine is None:
        affine = Phantom.default_affine(shape, tuple(phantom.fov))
    return np.asarray(affine, dtype=float)


def _cell_size_along(phantom, direction) -> float:
    """Return the phantom cell extent along a scanner direction."""
    linear = phantom_affine(phantom)[:3, :3]
    shape = tuple(int(value) for value in phantom.shape) + (1,) * 3
    per_metre = np.abs(np.linalg.solve(linear, np.asarray(direction, dtype=float)))
    # Singleton grid axes do not resolve anything along their direction.
    per_metre = np.where(np.asarray(shape[:3]) > 1, per_metre, 0.0)
    peak = float(np.max(per_metre))
    return 1.0 / peak if peak > 0.0 else np.inf


def _phantom_extent_along(phantom, direction) -> tuple[float, float]:
    """Return the phantom grid's centre and extent along a unit direction."""
    box = phantom_grid_box(phantom)
    corners = box.corners() @ np.asarray(direction, dtype=float)
    low, high = float(corners.min()), float(corners.max())
    if high - low <= 0.0:
        cell = _cell_size_along(phantom, direction)
        span = cell if np.isfinite(cell) else 1e-3
        return 0.5 * (low + high), span
    return 0.5 * (low + high), high - low


def _dimension_positions(dataset, dimension: str):
    """Return logical voxel-centre positions and their role of one image axis."""
    coords = dataset.coords
    for role in ("read", "phase", "partition"):
        name = f"cartesian_{role}_position_m"
        if name in coords and coords[name].dims == (dimension,):
            return np.asarray(coords[name].values, dtype=float), role
    for axis, role in (("x", "read"), ("y", "phase")):
        name = f"spiral_position_{axis}_m"
        if name in coords and coords[name].dims == (dimension,):
            return np.asarray(coords[name].values, dtype=float), role
    return None, None


def _csi_positions(dataset, definitions, dimension: str):
    """CSI is voxel centred on its FOV without position coordinates."""
    axis = {"phase_x": 0, "phase_y": 1}.get(dimension)
    if axis is None:
        return None
    count = int(dataset.sizes[dimension])
    fov = _values(definitions, "fov")
    if fov.size <= axis:
        k_name = f"spatial_k{'xy'[axis]}_cyc_per_m"
        if k_name not in dataset.coords or dataset.coords[k_name].size < 2:
            return None
        step = float(np.median(np.diff(np.asarray(dataset.coords[k_name].values))))
        if not np.isfinite(step) or step == 0.0:
            return None
        extent = count / abs(step)
    else:
        extent = float(fov[axis])
        if extent > 2.0:
            # Pulseq definitions write FOV in metres, older files in mm.
            extent /= 1000.0
    return ((np.arange(count) + 0.5) / count - 0.5) * extent


def _slice_position(definitions, model, selections) -> Optional[float]:
    positions = _values(definitions, "slicepositions")
    if positions.size == 0:
        offset = _scalar(definitions, "sliceoffset")
        positions = np.asarray([0.0 if offset is None else offset])
    index = 0
    slice_dimension = next(
        (item for item in model.outer_dimensions if item.name == "slice"), None
    )
    if slice_dimension is not None:
        value = (selections or {}).get("slice", slice_dimension.values[0])
        try:
            index = int(value)
        except (TypeError, ValueError):
            index = 0
    return float(positions[int(np.clip(index, 0, positions.size - 1))])


def planar_image_grid(model, image_dims, phantom, selections=None):
    """Return the grid of a displayed 2D image (rows, columns) or ``None``."""
    dataset = model.dataset
    definitions = dataset_definitions(dataset)
    frame = encoding_frame(dataset, definitions)
    steps = []
    origin = np.zeros(3)
    for dimension in image_dims:
        if model.kind == "csi":
            positions = _csi_positions(dataset, definitions, dimension)
            direction = np.eye(3)[{"phase_x": 0, "phase_y": 1}.get(dimension, 0)]
            if positions is None:
                return None
            center = _scalar(
                definitions, "readoffset" if dimension == "phase_x" else "phaseoffset"
            )
            positions = positions + (center or 0.0)
        else:
            positions, role = _dimension_positions(dataset, dimension)
            if positions is None:
                return None
            direction = frame[:, ("read", "phase", "partition").index(role)]
        if positions.size == 0:
            return None
        step = float(np.median(np.diff(positions))) if positions.size > 1 else None
        if step is None or not np.isfinite(step) or step == 0.0:
            fov = _values(definitions, "fov")
            step = float(fov[0]) if fov.size else 1e-3
        origin = origin + positions[0] * direction
        steps.append(step * direction)
    if len(steps) != 2:
        return None
    normal = np.cross(steps[1], steps[0])
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return None
    normal /= norm
    thickness = _scalar(definitions, "slicethickness")
    projection = thickness is None or thickness <= 0.0
    if projection:
        depth_center, thickness = _phantom_extent_along(phantom, normal)
    else:
        # The slice normal of the frame can point either way along the
        # image normal; slice positions are measured along the frame axis.
        partition = frame[:, 2]
        sign = 1.0 if float(partition @ normal) >= 0.0 else -1.0
        depth_center = sign * _slice_position(definitions, model, selections)
    origin = origin - (origin @ normal) * normal + depth_center * normal
    return ImageGrid(
        origin_m=origin,
        steps_m=np.vstack([steps[0], steps[1], thickness * normal]),
        shape=(
            int(dataset.sizes[image_dims[0]]),
            int(dataset.sizes[image_dims[1]]),
            1,
        ),
        planar=True,
        projection=projection,
    )


def volume_image_grid(model):
    """Return the scanner-ordered grid of ``model.scanner_volume`` or ``None``."""
    dataset = model.dataset
    positions = []
    for axis in "xyz":
        names = (f"radial_position_{axis}_m", f"cartesian_{axis}_position_m")
        name = next((item for item in names if item in dataset.coords), None)
        if name is None:
            return None
        values = np.sort(np.asarray(dataset.coords[name].values, dtype=float))
        positions.append(values)
    steps = []
    for values in positions:
        if values.size > 1:
            step = float(np.median(np.diff(values)))
        else:
            step = 0.0
        if not np.isfinite(step) or step <= 0.0:
            return None
        steps.append(step)
    return ImageGrid(
        origin_m=np.asarray([values[0] for values in positions]),
        steps_m=np.diag(steps),
        shape=tuple(int(values.size) for values in positions),
    )


# Orthogonal sections of a volume grid: horizontal, vertical and normal axis.
VOLUME_PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def volume_section_grid(grid: ImageGrid, plane: str, index: int) -> ImageGrid:
    """Return one voxel-thick orthogonal section of a volume grid."""
    horizontal, vertical, normal = VOLUME_PLANES[plane]
    return ImageGrid(
        origin_m=grid.origin_m + int(index) * grid.steps_m[normal],
        steps_m=grid.steps_m[[horizontal, vertical, normal]],
        shape=(int(grid.shape[horizontal]), int(grid.shape[vertical]), 1),
        planar=True,
    )


def grid_description(grid: ImageGrid) -> str:
    """Describe which part of the phantom a planar grid covers."""
    center = grid.box().center_m
    # Adding 0.0 turns a rounded -0.0 into 0.0.
    where = "/".join(f"{round(value * 1000, 1) + 0.0:.1f}" for value in center)
    if grid.projection:
        return (
            "Phantom projected through its whole depth (the sequence selects no "
            f"slice); image centre x/y/z = {where} mm"
        )
    return (
        f"Phantom averaged over the {grid.spacing_m[2] * 1000:.3g} mm slab; "
        f"centre x/y/z = {where} mm"
    )


def _phantom_indices(phantom, points_m):
    """Return nearest phantom cell indices of points and whether they exist."""
    affine = phantom_affine(phantom)
    shape = tuple(int(value) for value in phantom.shape) + (1,) * 3
    flat = np.asarray(points_m, dtype=float).reshape(-1, 3)
    ijk = np.linalg.solve(affine[:3, :3], (flat - affine[:3, 3]).T).T
    indices = np.rint(ijk).astype(np.int64)
    valid = np.ones(flat.shape[0], dtype=bool)
    for axis in range(3):
        if shape[axis] <= 1:
            indices[:, axis] = 0
        else:
            valid &= (indices[:, axis] >= 0) & (indices[:, axis] < shape[axis])
    indices[~valid] = 0
    return indices, valid


def _sample(volume, indices, valid):
    values = _volume(volume)[indices[:, 0], indices[:, 1], indices[:, 2]]
    return values, valid


def _planar_upsampling(phantom, grid: ImageGrid) -> tuple[int, int, int]:
    factors = []
    for axis in range(2):
        cell = _cell_size_along(phantom, grid.axes[:, axis])
        wanted = (
            1 if not np.isfinite(cell) else int(np.ceil(grid.spacing_m[axis] / cell))
        )
        limit = max(1, MAX_DISPLAY_SIZE // max(1, grid.shape[axis]))
        factors.append(int(np.clip(wanted, 1, min(MAX_IN_PLANE_UPSAMPLING, limit))))
    cell = _cell_size_along(phantom, grid.axes[:, 2])
    depth = 1 if not np.isfinite(cell) else int(np.ceil(grid.spacing_m[2] / cell))
    depth = int(np.clip(depth, 1, MAX_DEPTH_SAMPLES))
    budget = MAX_SAMPLES // max(
        1, grid.shape[0] * grid.shape[1] * factors[0] * factors[1]
    )
    return factors[0], factors[1], max(1, min(depth, budget))


def _volume_upsampling(phantom, grid: ImageGrid) -> tuple[int, int, int]:
    factors = []
    for axis in range(3):
        cell = _cell_size_along(phantom, grid.axes[:, axis])
        wanted = (
            1 if not np.isfinite(cell) else int(np.ceil(grid.spacing_m[axis] / cell))
        )
        factors.append(int(np.clip(wanted, 1, MAX_VOLUME_UPSAMPLING)))
    while (
        int(np.prod(grid.shape)) * int(np.prod(factors)) > MAX_SAMPLES
        and max(factors) > 1
    ):
        factors[int(np.argmax(factors))] -= 1
    return tuple(factors)


@dataclass
class ResampledPhantom:
    """Phantom contrast on an image grid.

    ``fraction`` is the share of samples inside the object; ``values`` holds
    the category index (``-1`` outside) of a categorical contrast, else the
    mean over the inside samples (``NaN`` outside).  ``upsampling`` is the
    number of display samples per image pixel along each array axis.
    """

    fraction: np.ndarray
    values: np.ndarray
    contrast: Optional[ObjectContrast]
    upsampling: tuple[int, ...]
    labels: tuple = ()


def _reduce(samples_inside, samples_values, block, contrast, labels):
    """Average (or vote) over blocks of samples of shape ``(..., block)``."""
    count = samples_inside.sum(axis=-1)
    fraction = count / samples_inside.shape[-1]
    if contrast is not None and contrast.categorical:
        votes = np.stack(
            [
                np.sum(samples_inside & (samples_values == label), axis=-1)
                for label in labels
            ],
            axis=-1,
        )
        values = np.where(count > 0, np.argmax(votes, axis=-1), -1)
        return fraction, values
    finite = samples_inside & np.isfinite(samples_values)
    total = np.where(finite, samples_values, 0.0).sum(axis=-1)
    used = finite.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(used > 0, total / np.maximum(used, 1), np.nan)
    return fraction, values


def resample_phantom(
    phantom, grid: ImageGrid, contrast: Optional[ObjectContrast] = None
) -> ResampledPhantom:
    """Resample the phantom onto an image grid.

    A planar grid keeps sub-pixel display samples in plane and averages the
    slab; a volume grid averages each voxel.
    """
    density = _object_density(phantom)
    if grid.planar:
        upsampling = _planar_upsampling(phantom, grid)
    else:
        upsampling = _volume_upsampling(phantom, grid)
    points = grid.sample_points(upsampling)
    fine_shape = points.shape[:3]
    indices, valid = _phantom_indices(phantom, points)
    inside_density, _ = _sample(density, indices, valid)
    inside = valid & (inside_density > 0.0)
    if contrast is None:
        values, _ = _sample(density, indices, valid)
        values = values.astype(float)
    else:
        values, _ = _sample(contrast.values, indices, valid)
    labels = (
        tuple(contrast.categories)
        if contrast is not None and contrast.categorical
        else ()
    )
    inside = inside.reshape(fine_shape)
    values = values.reshape(fine_shape)
    if not (contrast is not None and contrast.categorical):
        values = values.astype(float)
    if grid.planar:
        # (rows*u, cols*u, depth) -> reduce the depth samples.
        fraction, reduced = _reduce(inside, values, upsampling[2], contrast, labels)
        resampled_upsampling = upsampling[:2]
    else:
        blocked_shape = []
        for count, factor in zip(grid.shape, upsampling):
            blocked_shape.extend((count, factor))
        inside_blocks = inside.reshape(blocked_shape).transpose(0, 2, 4, 1, 3, 5)
        value_blocks = values.reshape(blocked_shape).transpose(0, 2, 4, 1, 3, 5)
        flat = tuple(grid.shape) + (-1,)
        fraction, reduced = _reduce(
            inside_blocks.reshape(flat),
            value_blocks.reshape(flat),
            int(np.prod(upsampling)),
            contrast,
            labels,
        )
        resampled_upsampling = (1, 1, 1)
    return ResampledPhantom(
        fraction=fraction,
        values=reduced,
        contrast=contrast,
        upsampling=tuple(resampled_upsampling),
        labels=labels,
    )


def contrast_lookup(contrast: Optional[ObjectContrast]):
    """Return ``(lookup table, levels)`` for :func:`volume_display` values."""
    if contrast is not None and contrast.categorical:
        colors = [contrast.categories[label][1] for label in contrast.categories]
        lookup = np.asarray(
            [(0, 0, 0, 0)] + [tuple(color) + (255,) for color in colors],
            dtype=np.ubyte,
        )
        # Category index c (0 = outside) maps to lookup row c.
        return lookup, (-0.5, len(colors) + 0.5)
    if contrast is None:
        gray = np.repeat(np.arange(256, dtype=np.ubyte)[:, np.newaxis], 3, axis=1)
        alpha = np.full((256, 1), 255, dtype=np.ubyte)
        return np.hstack([gray, alpha]), None
    lookup = pg.colormap.get(CONTRAST_COLORMAP).getLookupTable(
        0.0, 1.0, 255, alpha=True
    )
    # Row 0 is black so voxels outside the object differ from the lowest value.
    lookup = np.vstack([np.asarray([[0, 0, 0, 255]]), np.asarray(lookup)])
    low, high = contrast.value_range
    span = high - low if high > low else max(abs(low), 1.0) * 1e-6
    return lookup.astype(np.ubyte), (low - span / 255.0, low + span)


def volume_display(resampled: ResampledPhantom):
    """Return ``(values, levels, lookup table)`` of a volume resampling."""
    contrast = resampled.contrast
    fraction = np.asarray(resampled.fraction, dtype=float)
    lookup, levels = contrast_lookup(contrast)
    if contrast is not None and contrast.categorical:
        values = np.where(resampled.values >= 0, resampled.values + 1, 0).astype(float)
    elif contrast is None:
        values = np.nan_to_num(np.asarray(resampled.values, dtype=float)) * fraction
        peak = float(values.max()) if values.size else 0.0
        levels = (0.0, peak if peak > 0.0 else 1.0)
    else:
        values = np.asarray(resampled.values, dtype=float)
        values = np.where(fraction > 0.0, values, np.nan)
    return values, levels, lookup


def resampled_rgba(resampled: ResampledPhantom) -> np.ndarray:
    """Colour a planar resampling; partial volume fades towards black."""
    contrast = resampled.contrast
    fraction = np.asarray(resampled.fraction, dtype=float)
    values = resampled.values
    rgba = np.zeros(fraction.shape + (4,), dtype=np.ubyte)
    present = fraction > 0.0
    if contrast is not None and contrast.categorical:
        colors = np.asarray(
            [contrast.categories[label][1] for label in resampled.labels], dtype=float
        )
        if colors.size:
            index = np.clip(values, 0, len(colors) - 1)
            rgb = colors[index]
        else:
            rgb = np.zeros(fraction.shape + (3,))
        present &= values >= 0
    else:
        if contrast is None:
            peak = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 0.0
            low, high = 0.0, peak
            lookup = np.repeat(np.arange(256)[:, np.newaxis], 3, axis=1).astype(float)
        else:
            low, high = contrast.value_range
            lookup = (
                pg.colormap.get(CONTRAST_COLORMAP)
                .getLookupTable(0.0, 1.0, 256, alpha=False)
                .astype(float)
            )
        if high > low:
            level = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
        else:
            level = np.full(fraction.shape, 0.5)
        rgb = lookup[np.round(level * 255).astype(int)]
        missing = present & ~np.isfinite(values)
        rgb[missing] = SILHOUETTE_COLOR
    shade = (0.35 + 0.65 * np.clip(fraction, 0.0, 1.0))[..., np.newaxis]
    rgba[..., :3] = np.where(present[..., np.newaxis], rgb * shade, 0).astype(np.ubyte)
    rgba[..., 3] = 255
    return rgba


def legend_html(contrast: Optional[ObjectContrast]) -> str:
    """Explain the phantom colours; dark pixels are partly outside the object."""
    note = "<br><i>Darker pixels lie partly outside the object.</i>"
    if contrast is None:
        return "<b>Colour:</b> object density (gray)" + note
    if contrast.categorical:
        return _contrast_legend_html(contrast) + note
    lookup = pg.colormap.get(CONTRAST_COLORMAP).getLookupTable(0.0, 1.0, 5, alpha=False)
    low, high = contrast.value_range
    gradient = "".join(_swatch(color) for color in lookup)
    return (
        f"<b>Colour:</b> {html.escape(contrast.title)}<br>"
        f"{low:.3g} {gradient} {high:.3g}{note}"
    )


def _region_labels(resampled: ResampledPhantom) -> np.ndarray:
    """Integer regions of a planar resampling; ``-1`` is outside the object."""
    contrast = resampled.contrast
    inside = np.asarray(resampled.fraction) >= 0.5
    if contrast is not None and contrast.categorical:
        labels = np.asarray(resampled.values, dtype=np.int64)
    elif contrast is None:
        labels = np.zeros(inside.shape, dtype=np.int64)
    else:
        low, high = contrast.value_range
        values = np.asarray(resampled.values, dtype=float)
        if high > low:
            level = np.clip((values - low) / (high - low), 0.0, 1.0)
            labels = np.minimum(
                np.floor(np.nan_to_num(level) * CONTOUR_LEVELS), CONTOUR_LEVELS - 1
            ).astype(np.int64)
        else:
            labels = np.zeros(inside.shape, dtype=np.int64)
    return np.where(inside, labels, -1)


def contour_rgba(resampled: ResampledPhantom) -> np.ndarray:
    """Return region boundaries as an RGBA image of the resampled shape.

    The object outline is cyan; boundaries between regions take the colour
    of the region they belong to (categories) or of their level (maps,
    quantised into ``CONTOUR_LEVELS`` bands).
    """
    labels = _region_labels(resampled)
    boundary = np.zeros(labels.shape, dtype=bool)
    outline = np.zeros(labels.shape, dtype=bool)
    for axis in range(labels.ndim):
        for shift in (1, -1):
            neighbour = np.roll(labels, shift, axis=axis)
            edge = [slice(None)] * labels.ndim
            edge[axis] = 0 if shift == 1 else -1
            differs = labels != neighbour
            differs[tuple(edge)] = False
            # Each boundary is drawn on its inner side only.
            boundary |= differs & (labels >= 0)
            outline |= differs & (labels >= 0) & (neighbour < 0)
    rgba = np.zeros(labels.shape + (4,), dtype=np.ubyte)
    contrast = resampled.contrast
    if contrast is not None and contrast.categorical and resampled.labels:
        colors = np.asarray(
            [contrast.categories[label][1] for label in resampled.labels],
            dtype=np.ubyte,
        )
        inner = np.clip(labels, 0, len(colors) - 1)
        rgba[boundary, :3] = colors[inner][boundary]
    elif contrast is not None:
        lookup = pg.colormap.get(CONTRAST_COLORMAP).getLookupTable(
            0.0, 1.0, CONTOUR_LEVELS, alpha=False
        )
        rgba[boundary, :3] = np.asarray(lookup, dtype=np.ubyte)[
            np.clip(labels, 0, CONTOUR_LEVELS - 1)
        ][boundary]
    rgba[outline, :3] = OUTLINE_COLOR
    rgba[boundary | outline, 3] = CONTOUR_ALPHA
    return rgba


@dataclass
class VoxelComposition:
    """Phantom cells whose centres lie inside one image voxel footprint."""

    cell_count: int
    object_fraction: float
    regions: list  # (contrast title, [(name, fraction, rgb), ...])
    means: list  # (title, mean, unit)
    nearest_only: bool = False

    def html(self) -> str:
        if self.cell_count == 0:
            return "<i>Footprint outside the phantom grid</i>"
        cells = "cell" if self.cell_count == 1 else "cells"
        note = (
            " (nearest cell; voxel smaller than a phantom cell)"
            if self.nearest_only
            else ""
        )
        lines = [
            f"<b>{self.cell_count}</b> phantom {cells}{note}, "
            f"<b>{self.object_fraction * 100:.0f} %</b> object"
        ]
        for title, parts in self.regions:
            shown = ", ".join(
                f"<span style='color:rgb{tuple(int(v) for v in rgb)}'>■</span> "
                f"{html.escape(str(name))} {fraction * 100:.0f} %"
                for name, fraction, rgb in parts[:5]
            )
            if len(parts) > 5:
                shown += ", …"
            lines.append(f"{html.escape(title)}: {shown}")
        for title, mean, unit in self.means:
            lines.append(
                f"{html.escape(title)}: {mean:.4g}{(' ' + html.escape(unit)) if unit else ''}"
            )
        return "<br>".join(lines)


def footprint_cells(phantom, box: OrientedBox):
    """Return phantom cell indices whose centres lie inside ``box``."""
    affine = phantom_affine(phantom)
    shape = tuple(int(value) for value in phantom.shape) + (1,) * 3
    shape = shape[:3]
    corners = box.corners()
    ijk = np.linalg.solve(affine[:3, :3], (corners - affine[:3, 3]).T).T
    ranges = []
    for axis in range(3):
        if shape[axis] <= 1:
            ranges.append(np.zeros(1, dtype=np.int64))
            continue
        low = max(0, int(np.floor(ijk[:, axis].min())))
        high = min(shape[axis] - 1, int(np.ceil(ijk[:, axis].max())))
        if high < low:
            return np.empty((0, 3), dtype=np.int64), False
        ranges.append(np.arange(low, high + 1))
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    centers = grid @ affine[:3, :3].T + affine[:3, 3]
    local = (centers - np.asarray(box.center_m)) @ np.asarray(box.axes, dtype=float)
    half = 0.5 * np.asarray(box.size_m, dtype=float) + 1e-12
    # Singleton grid axes are thin sheets: only the in-plane extent counts.
    linear = affine[:3, :3]
    for axis in range(3):
        if shape[axis] <= 1:
            sheet_normal = linear[:, axis] / max(np.linalg.norm(linear[:, axis]), 1e-12)
            spread = np.abs(np.asarray(box.axes, dtype=float).T @ sheet_normal)
            local[:, spread > 0.99] = 0.0
    inside = np.all(np.abs(local) <= half, axis=1)
    if np.any(inside):
        return grid[inside], False
    indices, valid = _phantom_indices(phantom, np.asarray(box.center_m)[np.newaxis])
    if valid[0]:
        return indices, True
    return np.empty((0, 3), dtype=np.int64), False


def voxel_composition(
    phantom, box: OrientedBox, contrasts: Sequence[ObjectContrast], selected=None
) -> VoxelComposition:
    """Summarise the phantom inside a voxel footprint.

    Categorical contrasts list their regions as shares of the object cells;
    the selected continuous contrast and the total concentration report
    their mean over the object cells.
    """
    cells, nearest = footprint_cells(phantom, box)
    if cells.shape[0] == 0:
        return VoxelComposition(0, 0.0, [], [])
    density = _volume(_object_density(phantom))
    inside = density[cells[:, 0], cells[:, 1], cells[:, 2]] > 0.0
    regions = []
    means = []
    count = int(np.count_nonzero(inside))
    shown_means = []
    if selected is not None and not selected.categorical:
        shown_means.append(selected)
    for contrast in contrasts:
        if contrast.categorical:
            if count == 0:
                continue
            values = _volume(contrast.values)[cells[:, 0], cells[:, 1], cells[:, 2]]
            values = values[inside]
            parts = []
            for label, (name, color) in contrast.categories.items():
                share = float(np.count_nonzero(values == label)) / count
                if share > 0.0:
                    parts.append((name, share, color))
            parts.sort(key=lambda item: -item[1])
            if parts:
                regions.append((contrast.label, parts))
        elif contrast.key == "concentration" and contrast is not selected:
            shown_means.append(contrast)
    for contrast in shown_means:
        if count == 0:
            continue
        values = _volume(contrast.values)[cells[:, 0], cells[:, 1], cells[:, 2]]
        values = np.asarray(values[inside], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            means.append((contrast.label, float(values.mean()), contrast.unit))
    return VoxelComposition(
        cell_count=int(cells.shape[0]),
        object_fraction=count / cells.shape[0],
        regions=regions,
        means=means,
        nearest_only=nearest,
    )
