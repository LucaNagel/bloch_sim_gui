"""Nucleus-specific default phantom used when no phantom has been loaded.

The Sequence workspace falls back to a single sphere so a generated sequence
can be run immediately.  Its resonance sits at the compound's chemical shift
(e.g. [1-13C]pyruvate at 171.076 ppm) or, without a preset, at the sequence
reference.  The sphere sits in a cubic grid
that fits inside the sequence FOV and whose cells are no wider than the
sequence pixels, so neither the aliasing nor the resolution warning fires
for the default object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .metabolite_presets import METABOLITE_PRESETS_BY_NUCLEUS
from .phantom_design import PhantomDesign, ShapeDefinition, SpectralPeakDefinition

DEFAULT_PHANTOM_METADATA_KEY = "default_sequence_phantom"
DEFAULT_SPHERE_DIAMETER_FRACTION = 0.6
DEFAULT_PHANTOM_MATRIX = 64
MIN_DEFAULT_PHANTOM_MATRIX = 32
MAX_DEFAULT_PHANTOM_MATRIX = 128
MAX_DEFAULT_SLICE_AXIS_MATRIX = 256
CELLS_PER_SLICE_THICKNESS = 2
SINGLE_VOXEL_SPHERE_MARGIN = 1.5
CELLS_PER_VOXEL = 8
DEFAULT_PHANTOM_FOV_M = 0.22
DEFAULT_SPECTRAL_BANDWIDTH_PPM = 5.0
DEFAULT_SPECTRAL_POINTS = 257


@dataclass(frozen=True)
class DefaultCompound:
    """The single resonance filling the default sphere of one nucleus."""

    name: str
    t1_s: float
    t2_star_s: float
    # Absolute shift; None places the resonance at the sequence reference.
    chemical_shift_ppm: Optional[float] = None


def _preset_compound(nucleus, key):
    for preset in METABOLITE_PRESETS_BY_NUCLEUS.get(nucleus, ()):
        if preset.key == key:
            return DefaultCompound(
                preset.name,
                preset.t1_s,
                preset.t2_star_s,
                float(preset.chemical_shift_ppm),
            )
    raise KeyError(f"missing {nucleus} metabolite preset {key!r}")


# Relaxation follows the curated presets where one exists; the remaining
# values are generic starting points and only need to be plausible.
DEFAULT_COMPOUNDS = {
    "H1": DefaultCompound("Water", 1.5, 0.080),
    "C13": _preset_compound("C13", "pyruvate"),
    "P31": _preset_compound("P31", "pcr"),
    "F19": DefaultCompound("Fluorine", 1.0, 0.050),
    "Na23": DefaultCompound("Sodium", 0.040, 0.020),
}


def default_compound(nucleus: str) -> DefaultCompound:
    try:
        return DEFAULT_COMPOUNDS[nucleus]
    except KeyError:
        raise ValueError(f"no default phantom for nucleus {nucleus!r}") from None


def _cells(side_m, width_m, maximum=MAX_DEFAULT_PHANTOM_MATRIX):
    cells = 8 * math.ceil(side_m / width_m / 8.0 - 1e-9)
    return int(np.clip(cells, MIN_DEFAULT_PHANTOM_MATRIX, maximum))


def default_phantom_grid(
    sequence_fov_m: Optional[Sequence[float]] = None,
    sequence_matrix: Optional[Sequence[int]] = None,
    encoding_matrix=None,
    *,
    slice_thickness_m: Optional[float] = None,
    slice_normal=None,
) -> tuple[float, tuple[int, int, int]]:
    """Return the cubic grid side length and the x/y/z matrix.

    ``sequence_fov_m`` and ``sequence_matrix`` are the encoded extents and
    sample counts in read/phase(/partition) order.  ``encoding_matrix`` holds
    the encoding directions as scanner-xyz columns; an oblique direction sees
    the cube stretched by the column's L1 norm, so the extents are divided by
    it.  Scanner axes along ``slice_normal`` are refined until a slice spans
    at least two cells, so the slice profile is resolved.  Matrix sizes are
    rounded up to multiples of eight and clamped to keep the object fast.
    """
    side_m, cells = _isotropic_grid(sequence_fov_m, sequence_matrix, encoding_matrix)
    shape = [cells] * 3
    if slice_thickness_m is not None and slice_normal is not None:
        thickness = float(slice_thickness_m)
        if np.isfinite(thickness) and thickness > 0.0:
            normal = np.abs(np.asarray(slice_normal, dtype=float))
            for axis in np.flatnonzero(normal > 1e-9):
                width = thickness / CELLS_PER_SLICE_THICKNESS / normal[axis]
                shape[axis] = max(
                    shape[axis],
                    _cells(side_m, width, MAX_DEFAULT_SLICE_AXIS_MATRIX),
                )
    return side_m, tuple(shape)


def default_single_voxel_grid(voxel_size_m, voxel_center_m=(0.0, 0.0, 0.0)):
    """Return a grid whose centred sphere encloses a single spectroscopy voxel.

    ``voxel_size_m`` and ``voxel_center_m`` are the voxel's edge lengths and
    centre along its own axes.  The sphere radius is
    ``SINGLE_VOXEL_SPHERE_MARGIN`` times the distance from the grid centre to
    the farthest voxel corner, so the whole voxel lies inside the object, and
    each voxel edge spans at least ``CELLS_PER_VOXEL`` cells.
    """
    size = np.abs(np.asarray(tuple(voxel_size_m), dtype=float))
    center = np.abs(np.asarray(tuple(voxel_center_m), dtype=float))
    if (
        size.size == 0
        or center.size != size.size
        or not np.all(np.isfinite(size))
        or not np.all(np.isfinite(center))
        or np.any(size <= 0)
    ):
        return DEFAULT_PHANTOM_FOV_M, (DEFAULT_PHANTOM_MATRIX,) * 3
    corner_m = float(np.linalg.norm(center + 0.5 * size))
    radius_m = SINGLE_VOXEL_SPHERE_MARGIN * corner_m
    side_m = 2.0 * radius_m / DEFAULT_SPHERE_DIAMETER_FRACTION
    cells = _cells(side_m, float(np.min(size)) / CELLS_PER_VOXEL)
    return side_m, (cells,) * 3


def _isotropic_grid(sequence_fov_m, sequence_matrix, encoding_matrix):
    if sequence_fov_m is None or sequence_matrix is None:
        return DEFAULT_PHANTOM_FOV_M, DEFAULT_PHANTOM_MATRIX
    fov = np.asarray(tuple(sequence_fov_m), dtype=float)
    matrix = np.asarray(tuple(sequence_matrix), dtype=float)
    count = min(fov.size, matrix.size)
    fov, matrix = fov[:count], matrix[:count]
    if (
        count == 0
        or not np.all(np.isfinite(fov))
        or not np.all(np.isfinite(matrix))
        or np.any(fov <= 0)
        or np.any(matrix <= 0)
    ):
        return DEFAULT_PHANTOM_FOV_M, DEFAULT_PHANTOM_MATRIX
    if encoding_matrix is not None:
        stretch = np.sum(np.abs(np.asarray(encoding_matrix, dtype=float)), axis=0)
        fov = fov / np.maximum(stretch[:count], 1.0)
    side_m = float(np.min(fov))
    return side_m, _cells(side_m, float(np.min(fov / matrix)))


def default_phantom_design(
    nucleus: str,
    *,
    field_strength_t: float,
    sequence_reference_ppm: float,
    side_m: float = DEFAULT_PHANTOM_FOV_M,
    matrix=DEFAULT_PHANTOM_MATRIX,
) -> PhantomDesign:
    """Design one centred sphere holding the nucleus' default compound."""
    compound = default_compound(nucleus)
    diameter = DEFAULT_SPHERE_DIAMETER_FRACTION
    reference_ppm = float(sequence_reference_ppm)
    shift_ppm = (
        reference_ppm
        if compound.chemical_shift_ppm is None
        else compound.chemical_shift_ppm
    )
    return PhantomDesign(
        name=f"Default sphere ({nucleus} {compound.name.lower()})",
        shape=tuple(int(value) for value in np.broadcast_to(matrix, 3)),
        fov_m=(float(side_m),) * 3,
        field_strength_t=float(field_strength_t),
        nucleus=nucleus,
        spectral_reference_ppm=reference_ppm,
        spectral_window_center_ppm=shift_ppm,
        spectral_bandwidth_ppm=DEFAULT_SPECTRAL_BANDWIDTH_PPM,
        spectral_points=DEFAULT_SPECTRAL_POINTS,
        supersampling_enabled=True,
        supersampling_factor=2,
        shapes=[
            ShapeDefinition(
                name="Sphere",
                kind="ellipsoid",
                center=(0.5, 0.5, 0.5),
                size=(diameter, diameter, diameter),
                t1_s=compound.t1_s,
                peaks=[
                    SpectralPeakDefinition(
                        name=compound.name,
                        amplitude=1.0,
                        frequency_ppm=shift_ppm - reference_ppm,
                        t2_star_s=compound.t2_star_s,
                        t1_s=compound.t1_s,
                    )
                ],
            )
        ],
    )


def build_default_phantom(nucleus: str, **kwargs):
    """Build the default sphere and tag it so the UI can label it."""
    phantom = default_phantom_design(nucleus, **kwargs).build()
    phantom.metadata[DEFAULT_PHANTOM_METADATA_KEY] = True
    return phantom


def is_default_phantom(phantom) -> bool:
    metadata = getattr(phantom, "metadata", None)
    return isinstance(metadata, dict) and bool(
        metadata.get(DEFAULT_PHANTOM_METADATA_KEY)
    )
