"""Spatial sampling geometry of Free Mode positions.

Positions are ``(N, 3)`` arrays in metres. A 1D line varies one axis. A 2D
plane varies two axes in row-major order: the first plane axis (x for
``"xy"``) is the inner, fast index, so ``positions.reshape(n_second,
n_first, 3)`` is an image indexed ``[second, first]``, e.g. ``[y, x]``. A 3D
volume continues the pattern with x fastest and z slowest, so
``positions.reshape(nz, ny, nx, 3)`` is indexed ``[z, y, x]``.

The layout of any positions array can be recovered with ``infer_layout``, so
saved results need no extra geometry metadata.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

AXIS_NAMES = ("x", "y", "z")
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
VOLUME_AXES = (0, 1, 2)

AxisLike = Union[int, str]


def axis_index(axis: AxisLike) -> int:
    """Return the column index 0/1/2 of an axis given as index or name."""
    if isinstance(axis, str):
        name = axis.strip().lower()
        if name not in AXIS_NAMES:
            raise ValueError(f"Unknown axis {axis!r}; expected x, y or z.")
        return AXIS_NAMES.index(name)
    index = int(axis)
    if index not in (0, 1, 2):
        raise ValueError(f"Axis index must be 0, 1 or 2, got {axis!r}.")
    return index


def plane_axes(plane: str) -> Tuple[int, int]:
    """Return the (first, second) axis indices of a plane name like "xy"."""
    key = str(plane).strip().lower()
    if key not in PLANES:
        raise ValueError(f"Unknown plane {plane!r}; expected xy, xz or yz.")
    return PLANES[key]


def centered_stride_indices(total: int, limit: int) -> np.ndarray:
    """Return at most ``limit`` indices into ``range(total)`` with one stride.

    A constant stride keeps the physical spacing uniform, and the indices are
    aligned to the central element so a symmetric axis stays symmetric.
    """
    total = max(int(total), 0)
    limit = max(int(limit), 1)
    if total <= limit:
        return np.arange(total)
    if limit == 1:
        return np.array([(total - 1) // 2])
    stride = int(np.ceil((total - 1) / (limit - 1)))
    start = ((total - 1) // 2) % stride
    return np.arange(start, total, stride)


def axis_samples(count: int, span_m: float) -> np.ndarray:
    """Return ``count`` evenly spaced samples over ``span_m``, centred on 0."""
    count = max(int(count), 1)
    if count == 1:
        return np.zeros(1)
    half = float(span_m) / 2.0
    return np.linspace(-half, half, count)


def build_positions(
    axes: Sequence[AxisLike],
    counts: Sequence[int],
    spans_m: Sequence[float],
    center_m: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Build a line, plane or volume of positions in row-major order.

    The first axis varies fastest. ``center_m`` shifts the whole object; it
    defaults to the origin.
    """
    axes = [axis_index(axis) for axis in axes]
    if len(axes) not in (1, 2, 3) or len(set(axes)) != len(axes):
        raise ValueError("Positions need one, two or three distinct axes.")
    if len(counts) != len(axes) or len(spans_m) != len(axes):
        raise ValueError("Give one count and one span per axis.")
    samples = [axis_samples(n, span) for n, span in zip(counts, spans_m)]
    # meshgrid over the slowest axis first, so ravel() runs the first fastest.
    grids = np.meshgrid(*reversed(samples), indexing="ij")
    positions = np.zeros((grids[0].size, 3))
    for axis, grid in zip(reversed(axes), grids):
        positions[:, axis] = grid.ravel()
    if center_m is not None:
        positions += np.asarray(center_m, dtype=float).reshape(1, 3)
    return positions


def build_line_positions(count, span_m, axis: AxisLike = "z", center_m=None):
    """Return ``count`` positions along one axis, spanning ``span_m``."""
    return build_positions([axis], [count], [span_m], center_m)


def build_plane_positions(counts, spans_m, plane: str = "xy", center_m=None):
    """Return a row-major grid of ``counts[0] × counts[1]`` positions."""
    return build_positions(plane_axes(plane), counts, spans_m, center_m)


def build_volume_positions(counts, spans_m, center_m=None):
    """Return a row-major ``nx × ny × nz`` grid of positions, x fastest."""
    return build_positions(VOLUME_AXES, counts, spans_m, center_m)


@dataclass(frozen=True)
class PositionLayout:
    """Grid structure of a positions array.

    ``axes`` lists the varying axes, fastest first: none for a single point,
    one for a line, (first, second) for a plane and three for a volume. ``coordinates`` holds the samples of each
    varying axis in metres, in storage order. ``center`` is the coordinate
    of the constant axes (and of the varying ones' first sample).
    """

    axes: Tuple[int, ...]
    coordinates: Tuple[np.ndarray, ...]
    center: np.ndarray
    is_grid: bool = True

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def counts(self) -> Tuple[int, ...]:
        """Samples per varying axis, first axis first."""
        return tuple(int(c.size) for c in self.coordinates)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array shape of per-position values, slowest axis first.

        ``(n_second, n_first)`` for a plane, ``(nz, ny, nx)`` for a volume.
        """
        return tuple(reversed(self.counts))

    @property
    def strides(self) -> Tuple[int, ...]:
        """Flat index step of each varying axis, first axis first."""
        return tuple(int(np.prod(self.counts[:k])) for k in range(self.ndim))

    @property
    def size(self) -> int:
        return int(np.prod(self.counts)) if self.counts else 1

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(AXIS_NAMES[a] for a in self.axes)

    @property
    def primary_axis(self) -> int:
        """Axis to use for 1D plots: the line axis, or z for a single point."""
        return self.axes[0] if self.axes else 2

    @property
    def plane_name(self) -> str:
        return "".join(self.axis_names)

    def describe(self) -> str:
        """Short human-readable description, e.g. "12 × 12 grid in xy"."""
        if self.ndim == 0:
            return "single position"
        if self.ndim == 1:
            return f"{self.counts[0]} positions along {self.axis_names[0]}"
        grid = " × ".join(str(n) for n in self.counts)
        kind = "volume" if self.ndim == 3 else "grid"
        return f"{grid} {kind} in {self.plane_name}"

    def coordinates_mm(self) -> Tuple[np.ndarray, ...]:
        return tuple(np.asarray(c) * 1000.0 for c in self.coordinates)

    def grid_indices(self, max_per_axis: int) -> np.ndarray:
        """Flat indices of an evenly strided sub-grid, at most per axis."""
        if self.ndim == 0:
            return np.arange(1)
        if self.ndim == 1 or not self.is_grid:
            return centered_stride_indices(self.counts[0], max_per_axis)
        picks = [centered_stride_indices(n, max_per_axis) for n in self.counts]
        # Slowest axis first, so ravel() keeps the first axis fastest.
        grids = np.ix_(*reversed(picks))
        flat = sum(g * s for g, s in zip(grids, reversed(self.strides)))
        return np.asarray(flat).ravel()

    def sub_layout(self, indices) -> "PositionLayout":
        """Layout of ``positions[indices]`` for indices from ``grid_indices``."""
        indices = np.asarray(indices, dtype=int)
        if self.ndim == 0:
            return self
        if self.ndim == 1 or not self.is_grid:
            return PositionLayout(
                self.axes,
                (np.asarray(self.coordinates[0])[indices],),
                self.center,
                self.is_grid,
            )
        coordinates = tuple(
            np.asarray(coords)[np.unique((indices // stride) % count)]
            for coords, stride, count in zip(
                self.coordinates, self.strides, self.counts
            )
        )
        return PositionLayout(self.axes, coordinates, self.center)

    def reshape(self, values):
        """Reshape per-position values (leading axis N) to the grid shape."""
        values = np.asarray(values)
        if self.ndim < 2 or not self.is_grid:
            return values
        return values.reshape(self.shape + values.shape[1:])

    def label(self, position_m) -> str:
        """Coordinates of one position on the varying axes, in mm."""
        position_mm = np.asarray(position_m, dtype=float).reshape(3) * 1000.0
        axes = self.axes if self.axes else (2,)
        names = ", ".join(AXIS_NAMES[a] for a in axes)
        values = ", ".join(f"{position_mm[a]:.2f}" for a in axes)
        if len(axes) == 1:
            return f"{names} = {values} mm"
        return f"({names}) = ({values}) mm"


def _row_major_layout(pos, varying, tol) -> Optional[PositionLayout]:
    """Return the grid layout if ``pos`` is row-major over ``varying`` axes."""
    total = pos.shape[0]
    # Each axis first changes after one full sweep of the faster axes.
    first_change = {}
    for axis in varying:
        changed = np.nonzero(np.abs(pos[:, axis] - pos[0, axis]) > tol)[0]
        first_change[axis] = int(changed[0])
    order = sorted(varying, key=first_change.get)
    strides = [first_change[axis] for axis in order] + [total]
    if strides[0] != 1:
        return None
    counts = []
    for inner, outer in zip(strides[:-1], strides[1:]):
        if outer % inner or outer // inner < 2:
            return None
        counts.append(outer // inner)
    grid = pos.reshape(tuple(reversed(counts)) + (3,))
    coordinates = []
    for k, axis in enumerate(order):
        # Axis k is the k-th from the end of the grid shape.
        dim = len(order) - 1 - k
        index = [0] * len(order)
        index[dim] = slice(None)
        coords = grid[tuple(index) + (axis,)]
        shape = [1] * len(order)
        shape[dim] = coords.size
        if np.any(np.abs(grid[..., axis] - coords.reshape(shape)) > tol) or np.any(
            np.abs(np.diff(coords)) <= tol
        ):
            return None
        coordinates.append(coords.copy())
    return PositionLayout(tuple(order), tuple(coordinates), pos[0].copy())


def infer_layout(positions) -> PositionLayout:
    """Recover the line, plane or volume structure of a positions array.

    Arrays that vary along several axes without forming a row-major grid,
    e.g. a diagonal line, are treated as a line along the axis with the
    largest span (``is_grid`` is then False).
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3) if pos.size % 3 == 0 else pos.reshape(-1, 1)
    if pos.shape[1] < 3:
        pos = np.pad(pos, ((0, 0), (0, 3 - pos.shape[1])))
    pos = pos[:, :3]
    if pos.shape[0] == 0:
        return PositionLayout((), (), np.zeros(3))
    span = np.ptp(pos, axis=0)
    tol = max(1e-12, 1e-9 * float(span.max()))
    varying = [axis for axis in range(3) if span[axis] > tol]
    center = pos[0].copy()
    if not varying:
        return PositionLayout((), (), center)
    if len(varying) == 1:
        axis = varying[0]
        return PositionLayout((axis,), (pos[:, axis].copy(),), center)

    if pos.shape[0] >= 2 ** len(varying):
        layout = _row_major_layout(pos, varying, tol)
        if layout is not None:
            return layout

    axis = int(np.argmax(span))
    return PositionLayout((axis,), (pos[:, axis].copy(),), center, is_grid=False)
