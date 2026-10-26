"""Chronological phase-encoding orders for Cartesian sequences.

A phase-encoding order lists the k-space indices of one image in the order
they are acquired. Index ``i`` of an axis with ``n`` steps encodes
``k = (i - n // 2) / FOV``, so the k-space centre is index ``n // 2``.

One axis is either ``"linear"`` (``-n/2, ..., n/2 - 1``) or ``"centric"``
(``0, -1, +1, -2, +2, ...``; an even axis ends with ``-n/2``). Two axes
combine one mode per axis, written ``"<first>-<second>"`` such as
``"centric-linear"``, and nest one axis inside the other: the inner axis
runs through all of its steps before the outer axis advances. The joint
orders ``"elliptical_centric"`` (all index pairs sorted by their distance
from the k-space centre) and ``"spiral"`` (a square spiral) ignore the loop
nesting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

LINEAR = "linear"
CENTRIC = "centric"
ELLIPTICAL_CENTRIC = "elliptical_centric"
SPIRAL = "spiral"

AXIS_ORDERS = (LINEAR, CENTRIC)
PAIRED_ORDERS = tuple(
    f"{first}-{second}" for first in AXIS_ORDERS for second in AXIS_ORDERS
)
JOINT_ORDERS = (ELLIPTICAL_CENTRIC, SPIRAL)
TWO_AXIS_ORDERS = PAIRED_ORDERS + JOINT_ORDERS

_ALIASES = {
    "elliptical-centric": ELLIPTICAL_CENTRIC,
    "elliptical centric": ELLIPTICAL_CENTRIC,
    "elliptic_centric": ELLIPTICAL_CENTRIC,
}


def normalized_axis_order(value) -> str:
    """Return the canonical one-axis order name."""
    text = str(value or LINEAR).strip().lower()
    if text not in AXIS_ORDERS:
        raise ValueError("phase encoding order must be 'linear' or 'centric'")
    return text


def normalized_two_axis_order(
    value, *, allowed: Sequence[str] = TWO_AXIS_ORDERS
) -> str:
    """Return the canonical two-axis order name.

    Accepts ``"<first>-<second>"`` strings, ``(first, second)`` pairs, one
    axis mode that applies to both axes, and the joint orders.
    """
    if value is None:
        value = LINEAR
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("a two-axis phase encoding order needs two modes")
        name = "-".join(normalized_axis_order(item) for item in value)
    else:
        text = str(value).strip().lower()
        name = _ALIASES.get(text, text)
        if name in AXIS_ORDERS:
            name = f"{name}-{name}"
        elif name not in JOINT_ORDERS:
            parts = name.split("-")
            if len(parts) != 2:
                raise ValueError(f"unknown phase encoding order: {value!r}")
            name = "-".join(normalized_axis_order(item) for item in parts)
    if name not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"phase encoding order must be one of {choices}")
    return name


def axis_encoding_order(count: int, order: str = LINEAR) -> tuple[int, ...]:
    """Return the acquisition order of one phase-encoding axis."""
    count = int(count)
    if count < 1:
        raise ValueError("phase encoding axis needs at least one step")
    if normalized_axis_order(order) == LINEAR:
        return tuple(range(count))
    center = count // 2
    result = [center]
    for distance in range(1, count):
        for index in (center - distance, center + distance):
            if 0 <= index < count:
                result.append(index)
    return tuple(result)


def _square_spiral(n_a: int, n_b: int) -> list[tuple[int, int]]:
    center_a, center_b = n_a // 2, n_b // 2
    result = [(center_a, center_b)]
    a, b = center_a, center_b
    step_length = 1
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    direction_index = 0
    while len(result) < n_a * n_b:
        for _ in range(2):
            da, db = directions[direction_index % 4]
            direction_index += 1
            for _ in range(step_length):
                a += da
                b += db
                if 0 <= a < n_a and 0 <= b < n_b:
                    result.append((a, b))
        step_length += 1
    return result


def _elliptical_centric(
    n_a: int, n_b: int, fov_m: tuple[float, float]
) -> list[tuple[int, int]]:
    center_a, center_b = n_a // 2, n_b // 2
    fov_a, fov_b = fov_m

    def key(index: tuple[int, int]) -> tuple[float, float]:
        a, b = index
        k_a = (a - center_a) / fov_a
        k_b = (b - center_b) / fov_b
        return (
            float(np.hypot(k_a, k_b)),
            float((np.arctan2(k_b, k_a) + np.pi) % (2 * np.pi)),
        )

    return sorted(((a, b) for b in range(n_b) for a in range(n_a)), key=key)


def two_axis_encoding_order(
    counts: Sequence[int],
    order=LINEAR,
    *,
    inner_axis: int = 0,
    fov_m: Sequence[float] = (1.0, 1.0),
    allowed: Sequence[str] = TWO_AXIS_ORDERS,
) -> tuple[tuple[int, int], ...]:
    """Return the acquisition order of two phase-encoding axes.

    Each item is an ``(index_a, index_b)`` pair for the first and second
    axis. ``inner_axis`` (0 or 1) selects the axis of the inner loop of the
    paired orders; ``fov_m`` scales the elliptical-centric distance so the
    order follows physical k-space radii.
    """
    n_a, n_b = (int(value) for value in counts)
    if n_a < 1 or n_b < 1:
        raise ValueError("phase encoding axes need at least one step")
    name = normalized_two_axis_order(order, allowed=allowed)
    if name == SPIRAL:
        return tuple(_square_spiral(n_a, n_b))
    if name == ELLIPTICAL_CENTRIC:
        fov_a, fov_b = (float(value) for value in fov_m)
        if not (np.isfinite(fov_a) and np.isfinite(fov_b) and fov_a > 0 and fov_b > 0):
            raise ValueError("elliptical-centric order needs positive FOV values")
        return tuple(_elliptical_centric(n_a, n_b, (fov_a, fov_b)))
    if inner_axis not in (0, 1):
        raise ValueError("inner_axis must be 0 or 1")
    mode_a, mode_b = name.split("-")
    order_a = axis_encoding_order(n_a, mode_a)
    order_b = axis_encoding_order(n_b, mode_b)
    if inner_axis == 0:
        return tuple((a, b) for b in order_b for a in order_a)
    return tuple((a, b) for a in order_a for b in order_b)


def inner_axis_index(value, axis_names: Sequence[str]) -> int:
    """Map an inner-axis name such as ``"ky"`` to 0 or 1 of ``axis_names``."""
    text = str(value).strip().lower()
    names = [str(name).lower() for name in axis_names]
    if text not in names:
        raise ValueError(f"phase encoding inner axis must be one of {', '.join(names)}")
    return names.index(text)


def volume_encoding_order(
    counts: Sequence[int],
    order=LINEAR,
    inner_axis="ky",
    *,
    fov_m: Sequence[float],
    axis_names: Sequence[str] = ("ky", "kz"),
) -> tuple[str, str, tuple[tuple[int, int], ...]]:
    """Resolve the ``(ky, kz)`` order of a Cartesian 3D volume.

    Returns the canonical order name, the canonical inner-axis name and the
    chronological ``(line, partition)`` index pairs. The square spiral is a
    CSI order and is not offered for 3D volumes.
    """
    name = normalized_two_axis_order(
        order, allowed=PAIRED_ORDERS + (ELLIPTICAL_CENTRIC,)
    )
    inner = str(inner_axis).strip().lower()
    indices = two_axis_encoding_order(
        counts,
        name,
        inner_axis=inner_axis_index(inner, axis_names),
        fov_m=fov_m,
    )
    return name, inner, indices
