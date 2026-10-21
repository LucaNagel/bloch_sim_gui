"""Spatial sampling geometry of Free Mode positions.

Positions are ``(N, 3)`` arrays in metres. A 1D line varies one axis. A 2D
plane varies two axes in row-major order: the first plane axis (x for
``"xy"``) is the inner, fast index, so ``positions.reshape(n_second,
n_first, 3)`` is an image indexed ``[second, first]``, e.g. ``[y, x]``.

The layout of any positions array can be recovered with ``infer_layout``, so
saved results need no extra geometry metadata.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

AXIS_NAMES = ("x", "y", "z")
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

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
    """Build a line (one axis) or row-major plane (two axes) of positions.

    ``center_m`` shifts the whole object; it defaults to the origin.
    """
    axes = [axis_index(axis) for axis in axes]
    if len(axes) not in (1, 2) or len(set(axes)) != len(axes):
        raise ValueError("Positions need one axis or two distinct axes.")
    if len(counts) != len(axes) or len(spans_m) != len(axes):
        raise ValueError("Give one count and one span per axis.")
    samples = [axis_samples(n, span) for n, span in zip(counts, spans_m)]
    if len(axes) == 1:
        positions = np.zeros((samples[0].size, 3))
        positions[:, axes[0]] = samples[0]
    else:
        outer, inner = np.meshgrid(samples[1], samples[0], indexing="ij")
        positions = np.zeros((inner.size, 3))
        positions[:, axes[0]] = inner.ravel()
        positions[:, axes[1]] = outer.ravel()
    if center_m is not None:
        positions += np.asarray(center_m, dtype=float).reshape(1, 3)
    return positions


def build_line_positions(count, span_m, axis: AxisLike = "z", center_m=None):
    """Return ``count`` positions along one axis, spanning ``span_m``."""
    return build_positions([axis], [count], [span_m], center_m)


def build_plane_positions(counts, spans_m, plane: str = "xy", center_m=None):
    """Return a row-major grid of ``counts[0] × counts[1]`` positions."""
    return build_positions(plane_axes(plane), counts, spans_m, center_m)


@dataclass(frozen=True)
class PositionLayout:
    """Grid structure of a positions array.

    ``axes`` lists the varying axes: none for a single point, one for a line
    and (first, second) for a plane. ``coordinates`` holds the samples of each
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
        """Array shape of per-position values: ``(n_second, n_first)``."""
        return tuple(reversed(self.counts))

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
        return f"{self.counts[0]} × {self.counts[1]} grid in {self.plane_name}"

    def coordinates_mm(self) -> Tuple[np.ndarray, ...]:
        return tuple(np.asarray(c) * 1000.0 for c in self.coordinates)

    def grid_indices(self, max_per_axis: int) -> np.ndarray:
        """Flat indices of an evenly strided sub-grid, at most per axis."""
        if self.ndim == 0:
            return np.arange(1)
        if self.ndim == 1 or not self.is_grid:
            return centered_stride_indices(self.counts[0], max_per_axis)
        first = centered_stride_indices(self.counts[0], max_per_axis)
        second = centered_stride_indices(self.counts[1], max_per_axis)
        return (second[:, None] * self.counts[0] + first[None, :]).ravel()

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
        n_first = self.counts[0]
        first = np.unique(indices % n_first)
        second = np.unique(indices // n_first)
        return PositionLayout(
            self.axes,
            (
                np.asarray(self.coordinates[0])[first],
                np.asarray(self.coordinates[1])[second],
            ),
            self.center,
        )

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


def infer_layout(positions) -> PositionLayout:
    """Recover the line or plane structure of a positions array.

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

    if len(varying) == 2 and pos.shape[0] >= 4:
        step = np.abs(pos[1] - pos[0]) > tol
        if np.count_nonzero(step[varying]) == 1:
            first = varying[0] if step[varying[0]] else varying[1]
            second = varying[1] if first == varying[0] else varying[0]
            changed = np.nonzero(np.abs(pos[:, second] - pos[0, second]) > tol)[0]
            n_first = int(changed[0]) if changed.size else pos.shape[0]
            if n_first >= 2 and pos.shape[0] % n_first == 0:
                grid = pos.reshape(-1, n_first, 3)
                first_row = grid[0, :, first]
                second_col = grid[:, 0, second]
                if (
                    np.all(np.abs(grid[:, :, first] - first_row) <= tol)
                    and np.all(np.abs(grid[:, :, second] - second_col[:, None]) <= tol)
                    and np.all(np.abs(np.diff(first_row)) > tol)
                    and np.all(np.abs(np.diff(second_col)) > tol)
                ):
                    return PositionLayout(
                        (first, second),
                        (first_row.copy(), second_col.copy()),
                        center,
                    )

    axis = int(np.argmax(span))
    return PositionLayout((axis,), (pos[:, axis].copy(),), center, is_grid=False)
