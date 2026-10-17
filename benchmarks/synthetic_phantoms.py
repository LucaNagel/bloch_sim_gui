"""Analytic two-container 13C phantom for self-contained accuracy benchmarks.

The bundled ``.npz`` phantoms are local, git-ignored files, so benchmarks that
users should be able to run anywhere build this phantom in code instead. The
object is defined continuously and rasterized with partial-volume fractions,
so the same physical object can be sampled on any grid. That makes it suitable
for comparing phantom grids of 1x, 2x, 4x, ... the acquisition matrix: finer
grids describe the same object more accurately instead of upsampling a fixed
piecewise-constant source.

Geometry (32 mm cube centred on the isocentre):

* container A: cylinder along Z at x = -7 mm, radius 5.5 mm, |z| <= 12 mm;
* container B: cylinder along Z at x = +7 mm, radius 5.5 mm, |z| <= 12 mm;
* a linear B0 offset of +-15 Hz across X so RF pulses see off-resonance.

The static variant holds pyruvate in A and a pyruvate/lactate mixture in B.
The dynamic variant fills both containers with hyperpolarized pyruvate,
converts it to lactate with a different kPL per container, and adds a short
pyruvate inflow.
"""

from __future__ import annotations

import numpy as np

from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom


FOV_M = (0.032, 0.032, 0.032)
FIELD_STRENGTH_T = 7.0
NUCLEUS = "C13"
REFERENCE_PPM = 171.0
LACTATE_SHIFT_PPM = 12.35
CONTAINER_CENTRES_X_M = (-7e-3, 7e-3)
CONTAINER_RADIUS_M = 5.5e-3
CONTAINER_HALF_LENGTH_M = 12e-3
B0_PEAK_HZ = 15.0
KPL_S_INV = (0.05, 0.15)
XY_SUPERSAMPLING = 8
SYNTHETIC_PHANTOM_KINDS = ("static", "dynamic")


def _voxel_edges(count: int, fov: float) -> np.ndarray:
    return np.linspace(-fov / 2.0, fov / 2.0, count + 1)


def _z_fraction(count: int, fov: float) -> np.ndarray:
    """Exact overlap of every Z voxel with the container length."""
    edges = _voxel_edges(count, fov)
    overlap = np.clip(
        np.minimum(edges[1:], CONTAINER_HALF_LENGTH_M)
        - np.maximum(edges[:-1], -CONTAINER_HALF_LENGTH_M),
        0.0,
        None,
    )
    return overlap / np.diff(edges)


def _disc_fraction(shape_xy, fov_xy, centre_x: float) -> np.ndarray:
    """Area fraction of every X/Y voxel inside one container cross-section."""
    fractions = []
    for count, fov in zip(shape_xy, fov_xy):
        edges = _voxel_edges(count, fov)
        offsets = (np.arange(XY_SUPERSAMPLING) + 0.5) / XY_SUPERSAMPLING
        points = edges[:-1, None] + np.diff(edges)[:, None] * offsets[None, :]
        fractions.append(points)
    x_points, y_points = fractions
    distance_sq = (x_points[:, None, :, None] - centre_x) ** 2
    distance_sq = distance_sq + y_points[None, :, None, :] ** 2
    inside = distance_sq <= CONTAINER_RADIUS_M**2
    return inside.mean(axis=(2, 3))


def container_fractions(shape) -> tuple[np.ndarray, np.ndarray]:
    """Return the partial-volume fraction of containers A and B on ``shape``."""
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("synthetic phantom shape requires three positive values")
    z_fraction = _z_fraction(shape[2], FOV_M[2])
    return tuple(
        _disc_fraction(shape[:2], FOV_M[:2], centre)[:, :, None]
        * z_fraction[None, None, :]
        for centre in CONTAINER_CENTRES_X_M
    )


def container_core_masks(shape) -> tuple[np.ndarray, np.ndarray]:
    """In-plane voxels well inside each container, for ROI measurements.

    Voxel centres within 60 % of the radius stay clear of partial-volume edges
    and Gibbs ringing, so the ROI mean reflects the container's content.
    """
    shape = tuple(int(value) for value in shape)
    centres = [
        (_voxel_edges(count, fov)[:-1] + _voxel_edges(count, fov)[1:]) / 2.0
        for count, fov in zip(shape[:2], FOV_M[:2])
    ]
    x_centres, y_centres = centres
    return tuple(
        (x_centres[:, None] - centre) ** 2 + y_centres[None, :] ** 2
        <= (0.6 * CONTAINER_RADIUS_M) ** 2
        for centre in CONTAINER_CENTRES_X_M
    )


def _b0_map_hz(shape) -> np.ndarray:
    x_edges = _voxel_edges(shape[0], FOV_M[0])
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    linear = B0_PEAK_HZ * x_centres / (FOV_M[0] / 2.0)
    return np.broadcast_to(linear[:, None, None], shape).copy()


def _species():
    return (
        ChemicalSpecies("Pyruvate", 0.0, 25.0, 0.3),
        ChemicalSpecies("Lactate", LACTATE_SHIFT_PPM, 25.0, 0.3),
    )


def make_synthetic_phantom(kind: str, shape):
    """Rasterize the analytic two-container phantom on ``shape``."""
    shape = tuple(int(value) for value in shape)
    fraction_a, fraction_b = container_fractions(shape)
    shared = dict(
        shape=shape,
        fov=FOV_M,
        field_strength=FIELD_STRENGTH_T,
        nucleus=NUCLEUS,
        spectral_reference_ppm=REFERENCE_PPM,
        spectral_bandwidth_ppm=30.0,
        metadata={"synthetic_benchmark_phantom": kind},
    )
    if kind == "static":
        return SpectralPhantom(
            species=list(_species()),
            concentration_maps={
                "Pyruvate": fraction_a + 0.4 * fraction_b,
                "Lactate": 0.6 * fraction_b,
            },
            b0_map=_b0_map_hz(shape),
            name=f"Synthetic two-container static {'x'.join(map(str, shape))}",
            **shared,
        )
    if kind == "dynamic":
        occupied = fraction_a + fraction_b
        kpl = np.zeros(shape)
        kpl[fraction_a > 0] = KPL_S_INV[0]
        kpl[fraction_b > 0] = KPL_S_INV[1]
        return DynamicSpectralPhantom(
            pools=_species(),
            initial_concentration_maps={
                "Pyruvate": occupied,
                "Lactate": np.zeros(shape),
            },
            kpl_map_s_inv=kpl,
            b0_map=_b0_map_hz(shape),
            pyruvate_inflow=PyruvateInflow(
                rate_curve_s_inv=TimeCurve(
                    times_s=(0.0, 1.0, 2.0),
                    values=(0.5, 0.25, 0.0),
                    interpolation="step",
                    outside="zero",
                ),
                delivery_map=occupied,
                polarization_curve=TimeCurve(
                    times_s=(0.0,),
                    values=(1.0,),
                    interpolation="step",
                    outside="hold",
                ),
            ),
            name=f"Synthetic two-container dynamic {'x'.join(map(str, shape))}",
            **shared,
        )
    raise ValueError(
        f"unknown synthetic phantom kind {kind!r}; "
        f"expected one of {SYNTHETIC_PHANTOM_KINDS}"
    )
