#!/usr/bin/env python3
"""Benchmark one 2D FLASH acquisition over increasing phantom resolutions.

Goal
----
Measure how phantom discretization affects runtime, received signal, and the
reconstructed 2D FLASH image while the acquisition matrix and physical field of
view stay fixed. Phantom grids may be coarser than, equal to, or finer than the
sequence matrix. The finest requested phantom grid is used as the numerical
reference, not as assumed ground truth.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.benchmark_resolution
    python -m benchmarks.benchmark_resolution --resolution-scales 0.25 0.5 1 2
    python -m benchmarks.benchmark_resolution --flash-matrix 16 16 \\
        --phantom-matrices 16x16 32x32 64x64 128x128

The complete 3D phantom is retained in every simulation. Only its X/Y sampling
is changed; all source Z planes and the complete Z field of view remain present.
FLASH excites the central source Z slice unless ``--resolution-slice-index`` is
given. For a dynamic phantom, FLASH is generated with multiple labelled frames
and the pyruvate/lactate kinetics continue throughout the complete sequence;
control this with ``--dynamic-frames`` and ``--dynamic-frame-interval-s``.

Reading the output
------------------
Image and signal similarity near one indicate convergence to the finest tested
grid. Voxel-volume weighting keeps the physical signal integral comparable
across grid sizes. Upsampling the source refines the simulator discretization
of its piecewise-constant maps but cannot add new anatomical information.
"""

from __future__ import annotations

import argparse
import re
from time import perf_counter

import numpy as np

from blochsimulator.dynamic_phantom import (
    DynamicB0,
    DynamicSpectralPhantom,
    PyruvateInflow,
)
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import make_pulseq_flash
from blochsimulator.spectral_phantom import SpectralPhantom

try:
    from .common import (
        add_common_arguments,
        array_similarity,
        base_result_record,
        load_phantom,
        make_output_directory,
        make_simulator,
        physical_voxel_sizes_xyz_m,
        print_benchmark_header,
        print_phantom_summary,
        print_resolution_comparison,
        print_runtime_row,
        print_saved_results,
        print_sequence_summary,
        print_simulation_start,
        resolve_phantom_paths,
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )
except ImportError:  # Allow: python benchmarks/benchmark_resolution.py
    from common import (  # type: ignore
        add_common_arguments,
        array_similarity,
        base_result_record,
        load_phantom,
        make_output_directory,
        make_simulator,
        physical_voxel_sizes_xyz_m,
        print_benchmark_header,
        print_phantom_summary,
        print_resolution_comparison,
        print_runtime_row,
        print_saved_results,
        print_sequence_summary,
        print_simulation_start,
        resolve_phantom_paths,
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )


DEFAULT_RESOLUTION_SCALES = (0.25, 0.5, 1.0, 2.0)
DEFAULT_FLASH_MATRIX = (16, 16)


def parse_phantom_matrix(value: str) -> tuple[int, int]:
    """Parse an explicit in-plane phantom grid such as ``64x80``."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError("expected a phantom matrix such as 64x80")
    shape = tuple(int(item) for item in match.groups())
    if min(shape) <= 0:
        raise argparse.ArgumentTypeError("phantom matrix values must be positive")
    return shape


def add_resolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resolution-scales",
        nargs="+",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "In-plane phantom resolutions relative to the source X/Y grid "
            "(default when no explicit phantom matrices are supplied: "
            "0.25 0.5 1 2). Values above 1 create a grid finer than the source."
        ),
    )
    parser.add_argument(
        "--phantom-matrices",
        nargs="+",
        type=parse_phantom_matrix,
        metavar="READxPHASE",
        help=(
            "Explicit in-plane phantom grids, independent of --flash-matrix, "
            "for example 16x16 32x32 64x64. May be combined with "
            "--resolution-scales."
        ),
    )
    parser.add_argument(
        "--flash-matrix",
        nargs=2,
        type=int,
        default=DEFAULT_FLASH_MATRIX,
        metavar=("READ", "PHASE"),
        help=(
            "Fixed 2D FLASH acquisition matrix used at every phantom "
            "resolution (default: 16 16)."
        ),
    )
    parser.add_argument(
        "--dynamic-frames",
        type=int,
        default=4,
        help="FLASH frames for dynamic phantoms (default: 4).",
    )
    parser.add_argument(
        "--dynamic-frame-interval-s",
        type=float,
        default=1.0,
        help="Start-to-start interval of dynamic FLASH frames (default: 1 s).",
    )
    parser.add_argument(
        "--resolution-slice-index",
        type=int,
        help="Source Z-slice index; the central slice is used by default.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_resolution_arguments(parser)
    return parser


def _overlap_average_matrix(source_count: int, target_count: int) -> np.ndarray:
    """Map piecewise-constant source cells to target-cell averages."""
    source_edges = np.linspace(0.0, 1.0, source_count + 1)
    target_edges = np.linspace(0.0, 1.0, target_count + 1)
    matrix = np.zeros((target_count, source_count), dtype=np.float64)
    target_width = 1.0 / target_count
    for target in range(target_count):
        left = np.maximum(target_edges[target], source_edges[:-1])
        right = np.minimum(target_edges[target + 1], source_edges[1:])
        matrix[target] = np.maximum(0.0, right - left) / target_width
    return matrix


def conservative_resample_2d(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return area-averaged values while preserving their physical integral."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("conservative_resample_2d requires a two-dimensional map")
    target_shape = tuple(int(value) for value in shape)
    if len(target_shape) != 2 or min(target_shape) <= 0:
        raise ValueError("target shape requires two positive values")
    x_weights = _overlap_average_matrix(source.shape[0], target_shape[0])
    y_weights = _overlap_average_matrix(source.shape[1], target_shape[1])
    return x_weights @ source @ y_weights.T


def _resample_slice(values, slice_index: int, shape: tuple[int, int]) -> np.ndarray:
    return conservative_resample_2d(
        np.asarray(values, dtype=np.float64)[:, :, slice_index], shape
    )[..., None]


def _weighted_resample_slice(
    values,
    weights,
    slice_index: int,
    shape: tuple[int, int],
) -> np.ndarray:
    numerator = _resample_slice(
        np.asarray(values) * np.asarray(weights), slice_index, shape
    )
    denominator = _resample_slice(weights, slice_index, shape)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _resample_volume_xy(values, shape: tuple[int, int]) -> np.ndarray:
    """Area-average every Z plane while retaining the complete volume."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3:
        raise ValueError("volume resampling requires a three-dimensional map")
    return np.stack(
        [
            conservative_resample_2d(source[:, :, z_index], shape)
            for z_index in range(source.shape[2])
        ],
        axis=2,
    )


def _weighted_resample_volume_xy(
    values,
    weights,
    shape: tuple[int, int],
) -> np.ndarray:
    numerator = _resample_volume_xy(np.asarray(values) * np.asarray(weights), shape)
    denominator = _resample_volume_xy(weights, shape)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _slice_affine(source, slice_index: int, shape: tuple[int, int]) -> np.ndarray:
    fov = (
        float(source.fov[0]),
        float(source.fov[1]),
        float(source.fov[2]) / source.shape[2],
    )
    affine = Phantom.default_affine((*shape, 1), fov)
    source_affine = np.asarray(source.affine_ijk_to_xyz_m, dtype=np.float64)
    source_position = source_affine @ np.asarray([0.0, 0.0, float(slice_index), 1.0])
    affine[2, 3] = source_position[2]
    return affine


def _volume_affine(source, shape: tuple[int, int]) -> np.ndarray:
    """Preserve the source volume centre while changing its X/Y sampling."""
    source_affine = np.asarray(source.affine_ijk_to_xyz_m, dtype=np.float64)
    target_shape = np.asarray((*shape, source.shape[2]), dtype=np.float64)
    source_shape = np.asarray(source.shape, dtype=np.float64)
    affine = source_affine.copy()
    affine[:3, 0] *= source_shape[0] / target_shape[0]
    affine[:3, 1] *= source_shape[1] / target_shape[1]
    source_center = source_affine @ np.append((source_shape - 1.0) / 2.0, 1.0)
    affine[:3, 3] = source_center[:3] - affine[:3, :3] @ ((target_shape - 1.0) / 2.0)
    return affine


def _validate_axis_aligned_source(source) -> None:
    affine = np.asarray(source.affine_ijk_to_xyz_m, dtype=np.float64)
    if affine.shape != (4, 4) or not np.allclose(
        affine[:3, :3], np.diag(np.diag(affine[:3, :3])), atol=1e-12
    ):
        raise ValueError(
            "resolution benchmark currently requires axis-aligned phantoms"
        )


def make_resolution_slice(source, shape: tuple[int, int], slice_index: int):
    """Create one physically matched, area-averaged 3D single-slice phantom."""
    _validate_axis_aligned_source(source)
    if not 0 <= slice_index < source.shape[2]:
        raise ValueError(
            f"slice index {slice_index} is outside 0..{source.shape[2] - 1}"
        )
    shape_3d = (*shape, 1)
    fov_3d = (
        float(source.fov[0]),
        float(source.fov[1]),
        float(source.fov[2]) / source.shape[2],
    )
    affine = _slice_affine(source, slice_index, shape)
    metadata = dict(source.metadata)
    metadata.update(
        {
            "resolution_benchmark_source_shape": tuple(source.shape),
            "resolution_benchmark_source_slice": int(slice_index),
        }
    )

    if isinstance(source, DynamicSpectralPhantom):
        spin_density_maps = None
        if source.initial_spin_density_maps is not None:
            spin_density_maps = {
                pool.name: _resample_slice(
                    source.initial_spin_density_maps[pool.name], slice_index, shape
                )
                for pool in source.pools
            }
            initial_maps = {
                pool.name: _weighted_resample_slice(
                    source.initial_concentration_maps[pool.name],
                    source.initial_spin_density_maps[pool.name],
                    slice_index,
                    shape,
                )
                for pool in source.pools
            }
        else:
            initial_maps = {
                pool.name: _resample_slice(
                    source.initial_concentration_maps[pool.name], slice_index, shape
                )
                for pool in source.pools
            }
        inflow = None
        if source.pyruvate_inflow is not None:
            inflow = PyruvateInflow(
                rate_curve_s_inv=source.pyruvate_inflow.rate_curve_s_inv,
                delivery_map=_resample_slice(
                    source.pyruvate_inflow.delivery_map, slice_index, shape
                ),
                polarization_curve=source.pyruvate_inflow.polarization_curve,
            )
        dynamic_b0 = None
        if source.dynamic_b0 is not None:
            dynamic_b0 = DynamicB0(
                offset_curve_hz=source.dynamic_b0.offset_curve_hz,
                spatial_scale_map=_resample_slice(
                    source.dynamic_b0.spatial_scale_map, slice_index, shape
                ),
                pool_scale=source.dynamic_b0.pool_scale,
            )
        return DynamicSpectralPhantom(
            shape=shape_3d,
            fov=fov_3d,
            pools=tuple(source.pools),
            initial_concentration_maps=initial_maps,
            initial_spin_density_maps=spin_density_maps,
            equilibrium_polarization=source.equilibrium_polarization,
            kpl_map_s_inv=_resample_slice(source.kpl_map_s_inv, slice_index, shape),
            b0_map=(
                None
                if source.b0_map is None
                else _resample_slice(source.b0_map, slice_index, shape)
            ),
            b0_map_ppm=(
                None
                if source.b0_map_ppm is None
                else _resample_slice(source.b0_map_ppm, slice_index, shape)
            ),
            field_strength=source.field_strength,
            nucleus=source.nucleus,
            spectral_reference_ppm=source.spectral_reference_ppm,
            spectral_bandwidth_ppm=source.spectral_bandwidth_ppm,
            spectral_points=source.spectral_points,
            name=f"{source.name} {shape[0]}x{shape[1]} resolution slice",
            kinetic_regions=(),
            pyruvate_inflow=inflow,
            dynamic_b0=dynamic_b0,
            conversion_start_s=source.conversion_start_s,
            kinetics_time_offset_s=source.kinetics_time_offset_s,
            metadata=metadata,
            coordinate_system=source.coordinate_system,
            affine_ijk_to_xyz_m=affine,
        )

    if isinstance(source, SpectralPhantom):
        concentrations = {
            species.name: _resample_slice(
                source.concentration_maps[species.name], slice_index, shape
            )
            for species in source.species
        }
        initial_mz = {
            species.name: _weighted_resample_slice(
                source.initial_mz_maps[species.name],
                source.concentration_maps[species.name],
                slice_index,
                shape,
            )
            for species in source.species
        }
        return SpectralPhantom(
            shape=shape_3d,
            fov=fov_3d,
            species=list(source.species),
            concentration_maps=concentrations,
            initial_mz_maps=initial_mz,
            t2_star_map=(
                None
                if source.t2_star_map is None
                else _resample_slice(source.t2_star_map, slice_index, shape)
            ),
            b0_map=(
                None
                if source.b0_map is None
                else _resample_slice(source.b0_map, slice_index, shape)
            ),
            b0_map_ppm=(
                None
                if source.b0_map_ppm is None
                else _resample_slice(source.b0_map_ppm, slice_index, shape)
            ),
            field_strength=source.field_strength,
            nucleus=source.nucleus,
            spectral_reference_ppm=source.spectral_reference_ppm,
            spectral_bandwidth_ppm=source.spectral_bandwidth_ppm,
            spectral_points=source.spectral_points,
            name=f"{source.name} {shape[0]}x{shape[1]} resolution slice",
            metadata=metadata,
            coordinate_system=source.coordinate_system,
            affine_ijk_to_xyz_m=affine,
        )
    raise TypeError("resolution benchmark requires a spectral or dynamic phantom")


def make_resolution_volume(source, shape: tuple[int, int]):
    """Resample X/Y while retaining every Z plane of a 3D phantom."""
    _validate_axis_aligned_source(source)
    if source.ndim != 3:
        raise ValueError("volume resampling requires a three-dimensional source")
    shape_3d = (*shape, source.shape[2])
    fov_3d = tuple(float(value) for value in source.fov)
    affine = _volume_affine(source, shape)
    metadata = dict(source.metadata)
    metadata.update(
        {
            "resolution_benchmark_source_shape": tuple(source.shape),
            "resolution_benchmark_volume_mode": "full",
        }
    )

    if isinstance(source, DynamicSpectralPhantom):
        spin_density_maps = None
        if source.initial_spin_density_maps is not None:
            spin_density_maps = {
                pool.name: _resample_volume_xy(
                    source.initial_spin_density_maps[pool.name], shape
                )
                for pool in source.pools
            }
            initial_maps = {
                pool.name: _weighted_resample_volume_xy(
                    source.initial_concentration_maps[pool.name],
                    source.initial_spin_density_maps[pool.name],
                    shape,
                )
                for pool in source.pools
            }
        else:
            initial_maps = {
                pool.name: _resample_volume_xy(
                    source.initial_concentration_maps[pool.name], shape
                )
                for pool in source.pools
            }
        inflow = None
        if source.pyruvate_inflow is not None:
            inflow = PyruvateInflow(
                rate_curve_s_inv=source.pyruvate_inflow.rate_curve_s_inv,
                delivery_map=_resample_volume_xy(
                    source.pyruvate_inflow.delivery_map, shape
                ),
                polarization_curve=source.pyruvate_inflow.polarization_curve,
            )
        dynamic_b0 = None
        if source.dynamic_b0 is not None:
            dynamic_b0 = DynamicB0(
                offset_curve_hz=source.dynamic_b0.offset_curve_hz,
                spatial_scale_map=_resample_volume_xy(
                    source.dynamic_b0.spatial_scale_map, shape
                ),
                pool_scale=source.dynamic_b0.pool_scale,
            )
        return DynamicSpectralPhantom(
            shape=shape_3d,
            fov=fov_3d,
            pools=tuple(source.pools),
            initial_concentration_maps=initial_maps,
            initial_spin_density_maps=spin_density_maps,
            equilibrium_polarization=source.equilibrium_polarization,
            kpl_map_s_inv=_resample_volume_xy(source.kpl_map_s_inv, shape),
            b0_map=(
                None
                if source.b0_map is None
                else _resample_volume_xy(source.b0_map, shape)
            ),
            b0_map_ppm=(
                None
                if source.b0_map_ppm is None
                else _resample_volume_xy(source.b0_map_ppm, shape)
            ),
            field_strength=source.field_strength,
            nucleus=source.nucleus,
            spectral_reference_ppm=source.spectral_reference_ppm,
            spectral_bandwidth_ppm=source.spectral_bandwidth_ppm,
            spectral_points=source.spectral_points,
            name=f"{source.name} {shape[0]}x{shape[1]}x{source.shape[2]} resolution volume",
            kinetic_regions=(),
            pyruvate_inflow=inflow,
            dynamic_b0=dynamic_b0,
            conversion_start_s=source.conversion_start_s,
            kinetics_time_offset_s=source.kinetics_time_offset_s,
            metadata=metadata,
            coordinate_system=source.coordinate_system,
            affine_ijk_to_xyz_m=affine,
        )

    if isinstance(source, SpectralPhantom):
        concentrations = {
            species.name: _resample_volume_xy(
                source.concentration_maps[species.name], shape
            )
            for species in source.species
        }
        initial_mz = {
            species.name: _weighted_resample_volume_xy(
                source.initial_mz_maps[species.name],
                source.concentration_maps[species.name],
                shape,
            )
            for species in source.species
        }
        return SpectralPhantom(
            shape=shape_3d,
            fov=fov_3d,
            species=list(source.species),
            concentration_maps=concentrations,
            initial_mz_maps=initial_mz,
            t2_star_map=(
                None
                if source.t2_star_map is None
                else _resample_volume_xy(source.t2_star_map, shape)
            ),
            b0_map=(
                None
                if source.b0_map is None
                else _resample_volume_xy(source.b0_map, shape)
            ),
            b0_map_ppm=(
                None
                if source.b0_map_ppm is None
                else _resample_volume_xy(source.b0_map_ppm, shape)
            ),
            field_strength=source.field_strength,
            nucleus=source.nucleus,
            spectral_reference_ppm=source.spectral_reference_ppm,
            spectral_bandwidth_ppm=source.spectral_bandwidth_ppm,
            spectral_points=source.spectral_points,
            name=f"{source.name} {shape[0]}x{shape[1]}x{source.shape[2]} resolution volume",
            metadata=metadata,
            coordinate_system=source.coordinate_system,
            affine_ijk_to_xyz_m=affine,
        )
    raise TypeError("resolution benchmark requires a spectral or dynamic phantom")


def build_resolution_flash_sequence(source, args, slice_index: int):
    dynamic = isinstance(source, DynamicSpectralPhantom)
    frames = args.dynamic_frames if dynamic else 1
    interval = args.dynamic_frame_interval_s if dynamic else None
    slice_position_m = float(
        (
            np.asarray(source.affine_ijk_to_xyz_m, dtype=np.float64)
            @ np.asarray([0.0, 0.0, float(slice_index), 1.0])
        )[2]
    )
    sequence = make_pulseq_flash(
        fov_m=(float(source.fov[0]), float(source.fov[1])),
        matrix=tuple(args.flash_matrix),
        sampling_bandwidth_hz=20_000.0,
        flip_angle_deg=10.0,
        rf_pulse_type="block",
        rf_duration_s=2e-3,
        rf_time_bandwidth_product=1.0,
        encoding_duration_s=2e-3,
        slice_thickness_m=physical_voxel_sizes_xyz_m(source)[2],
        n_slices=1,
        slice_offset_m=slice_position_m,
        repetitions=frames,
        acquisition_interval_s=interval,
        spoiler_cycles_per_slice=1.0,
        spoiler_cycles_per_voxel=0.0,
    )
    if dynamic:
        sequence.set_definition("DynamicFrames", frames)
        sequence.set_definition("DynamicFrameInterval", interval)
    return sequence


def _frame_arrays(result, values) -> list[np.ndarray]:
    frames = result.cartesian_acquisition_frames
    if frames is None:
        acquisition = result.cartesian_acquisition
        if acquisition is None:
            raise ValueError("resolution FLASH result has no inferred 2D acquisition")
        return [np.asarray(values)]
    return [
        np.take(np.asarray(values), indices, axis=-1)
        for indices in frames.sample_indices
    ]


def _reconstructed_frames(result) -> np.ndarray:
    frames = result.cartesian_acquisition_frames
    if frames is None:
        acquisition = result.cartesian_acquisition
        if acquisition is None:
            raise ValueError("resolution FLASH result has no inferred 2D acquisition")
        return np.asarray([result.reconstruct_cartesian(acquisition)])
    return np.stack(
        [frames.reconstruct(result, frame) for frame in range(frames.num_frames)]
    )


def _temporal_record(result) -> dict:
    signal_frames = _frame_arrays(result, result.signal)
    norms = [float(np.linalg.norm(values)) for values in signal_frames]
    record = {
        "frame_count": len(signal_frames),
        "frame_signal_l2_norms": norms,
        "temporal_signal_change_relative_l2": 0.0,
    }
    if len(signal_frames) > 1:
        first_norm = np.linalg.norm(signal_frames[0])
        change = np.linalg.norm(signal_frames[-1] - signal_frames[0])
        record["temporal_signal_change_relative_l2"] = float(
            change / first_norm if first_norm > 0 else 0.0
        )
    if result.species_signal is not None:
        pool_norms = {}
        for pool_index, pool_name in enumerate(result.pool_names):
            values = _frame_arrays(result, result.species_signal[pool_index])
            pool_norms[str(pool_name)] = [
                float(np.linalg.norm(frame)) for frame in values
            ]
        record["pool_frame_signal_l2_norms"] = pool_norms
    return record


def _resolution_cases(source, scales=None, phantom_matrices=None) -> list[dict]:
    """Resolve relative and absolute grids independently of acquisition size."""
    matrices = list(phantom_matrices or ())
    if scales is None and not matrices:
        scales = DEFAULT_RESOLUTION_SCALES

    cases_by_shape = {}
    for scale in sorted(float(value) for value in (scales or ())):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("--resolution-scales values must be positive and finite")
        shape = (
            max(1, int(round(source.shape[0] * scale))),
            max(1, int(round(source.shape[1] * scale))),
        )
        cases_by_shape.setdefault(
            shape,
            {
                "shape": shape,
                "requested_scale": scale,
                "requested_as": f"scale {scale:g}",
            },
        )

    for matrix in matrices:
        shape = tuple(int(value) for value in matrix)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("--phantom-matrices requires positive 2D grids")
        cases_by_shape[shape] = {
            "shape": shape,
            "requested_scale": None,
            "requested_as": f"matrix {shape[0]}x{shape[1]}",
        }

    if not cases_by_shape:
        raise ValueError("at least one phantom resolution must be selected")

    cases = list(cases_by_shape.values())
    for case in cases:
        shape = case["shape"]
        case["scale_x"] = shape[0] / source.shape[0]
        case["scale_y"] = shape[1] / source.shape[1]
    return sorted(
        cases,
        key=lambda case: (
            case["shape"][0] * case["shape"][1],
            case["shape"],
        ),
    )


def _grid_relation(
    phantom_shape: tuple[int, int], acquisition_matrix: tuple[int, int]
) -> str:
    ratios = np.asarray(phantom_shape, dtype=float) / np.asarray(
        acquisition_matrix, dtype=float
    )
    if np.allclose(ratios, 1.0):
        return "matches acquisition matrix"
    if np.all(ratios > 1.0):
        return "finer than acquisition matrix"
    if np.all(ratios < 1.0):
        return "coarser than acquisition matrix"
    return "mixed relative to acquisition matrix"


def _validate_args(args) -> None:
    if len(args.flash_matrix) != 2 or min(args.flash_matrix) <= 0:
        raise ValueError("--flash-matrix requires two positive integers")
    if args.dynamic_frames < 2:
        raise ValueError("--dynamic-frames must be at least 2")
    if (
        not np.isfinite(args.dynamic_frame_interval_s)
        or args.dynamic_frame_interval_s <= 0
    ):
        raise ValueError("--dynamic-frame-interval-s must be positive and finite")
    for scale in args.resolution_scales or ():
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("--resolution-scales values must be positive and finite")


def run(args, output_dir=None) -> list[dict]:
    _validate_args(args)
    output_dir = make_output_directory(
        args.output_dir if output_dir is None else output_dir
    )
    sequence_dir = output_dir / "generated_sequences"
    phantom_dir = output_dir / "generated_phantoms"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    simulator = make_simulator(args)
    timestep_s = args.timestep_us * 1e-6
    all_records = []
    default_scales = (
        list(DEFAULT_RESOLUTION_SCALES)
        if args.resolution_scales is None and not args.phantom_matrices
        else []
    )
    print_benchmark_header(
        "Phantom resolution with 2D FLASH",
        args,
        output_dir,
        selection={
            "Phantoms": args.phantoms,
            "Resolution scales": args.resolution_scales or default_scales or "none",
            "Explicit phantom matrices": args.phantom_matrices or "none",
            "FLASH matrix": args.flash_matrix,
            "Dynamic frames": args.dynamic_frames,
            "Dynamic frame interval": (f"{args.dynamic_frame_interval_s:g} s"),
            "Phantom volume mode": "full 3D; X/Y resolution sweep",
        },
    )

    for phantom_label, phantom_path in resolve_phantom_paths(args.phantoms):
        source = load_phantom(phantom_path)
        if source.ndim != 3:
            raise ValueError("resolution benchmark requires three-dimensional sources")
        slice_index = (
            source.shape[2] // 2
            if args.resolution_slice_index is None
            else args.resolution_slice_index
        )
        print_phantom_summary(phantom_label, phantom_path, source)
        print(
            f"  Full 3D object retained; FLASH selects Z slice {slice_index}",
            flush=True,
        )
        generation_start = perf_counter()
        sequence = build_resolution_flash_sequence(source, args, slice_index)
        sequence_path = sequence_dir / f"flash_resolution_{phantom_label}.seq"
        program = write_and_load_sequence(sequence, sequence_path)
        sequence_generation_time_s = perf_counter() - generation_start
        print_sequence_summary(
            (
                "flash_2d dynamic"
                if isinstance(source, DynamicSpectralPhantom)
                else "flash_2d static"
            ),
            sequence_path,
            program,
            sequence_generation_time_s,
        )

        cases = []
        resolution_cases = _resolution_cases(
            source,
            scales=args.resolution_scales,
            phantom_matrices=args.phantom_matrices,
        )
        for resolution_case in resolution_cases:
            shape = resolution_case["shape"]
            scale_x = float(resolution_case["scale_x"])
            scale_y = float(resolution_case["scale_y"])
            matrix_ratio_x = shape[0] / args.flash_matrix[0]
            matrix_ratio_y = shape[1] / args.flash_matrix[1]
            grid_relation = _grid_relation(shape, tuple(args.flash_matrix))
            shape_3d = (*shape, source.shape[2])
            print(
                f"\nPreparing resolution case: {resolution_case['requested_as']} | "
                f"phantom grid={shape_3d[0]}x{shape_3d[1]}x{shape_3d[2]} | "
                f"acquisition matrix={args.flash_matrix[0]}x{args.flash_matrix[1]}",
                flush=True,
            )
            print(
                f"  Grid relation: {grid_relation} | phantom/acquisition ratio="
                f"{matrix_ratio_x:.3g}x{matrix_ratio_y:.3g}",
                flush=True,
            )
            phantom_start = perf_counter()
            phantom = make_resolution_volume(source, shape)
            phantom_generation_time_s = perf_counter() - phantom_start
            generated_path = phantom_dir / (
                f"{phantom_label}_{shape_3d[0]}x{shape_3d[1]}x{shape_3d[2]}.npz"
            )
            phantom.save(generated_path)
            print_phantom_summary(
                f"{phantom_label} @ {shape_3d[0]}x{shape_3d[1]}x{shape_3d[2]}",
                generated_path,
                phantom,
            )
            print(
                f"  Phantom generation/save: {phantom_generation_time_s:.3f} s",
                flush=True,
            )

            for repeat in range(1, args.repeats + 1):
                dynamic = isinstance(phantom, DynamicSpectralPhantom)
                print_simulation_start(
                    repeat=repeat,
                    repeats=args.repeats,
                    spoiler_mode="ideal",
                    spin_sampling=(1, 1, 1),
                    timestep_us=args.timestep_us,
                    details={
                        "phantom": phantom_label,
                        "grid": shape_3d,
                        "simulated volume": "full 3D",
                        "FLASH slice index": slice_index,
                        "source scale": f"{scale_x:.3g}x{scale_y:.3g}",
                        "grid relation": grid_relation,
                        "frames": args.dynamic_frames if dynamic else 1,
                        "frame interval": (
                            f"{args.dynamic_frame_interval_s:g} s"
                            if dynamic
                            else "static"
                        ),
                    },
                )
                result, simulation_time_s = timed_simulation(
                    simulator,
                    program,
                    phantom,
                    timestep_s=timestep_s,
                    spoiler_mode="ideal",
                    spin_sampling=(1, 1, 1),
                    signal_weighting="voxel_volume",
                )
                record = base_result_record(
                    benchmark="resolution",
                    sequence_name="flash_2d",
                    phantom_label=phantom_label,
                    phantom_path=phantom_path,
                    phantom=phantom,
                    program=program,
                    result=result,
                    repeat=repeat,
                    generation_time_s=sequence_generation_time_s,
                    simulation_time_s=simulation_time_s,
                )
                voxel_size = np.asarray(phantom.fov) / np.asarray(phantom.shape)
                definitions = dict(program.metadata.get("definitions", {}))
                record.update(
                    {
                        "profile": args.profile,
                        "timestep_us": args.timestep_us,
                        "sequence_kernel": args.sequence_kernel,
                        "dynamic_kernel": args.dynamic_kernel,
                        "requested_threads": args.threads,
                        "resolution_scale": resolution_case["requested_scale"],
                        "resolution_scale_x": scale_x,
                        "resolution_scale_y": scale_y,
                        "resolution_x": shape[0],
                        "resolution_y": shape[1],
                        "resolution_z": source.shape[2],
                        "resolution_requested_as": resolution_case["requested_as"],
                        "acquisition_matrix_x": int(args.flash_matrix[0]),
                        "acquisition_matrix_y": int(args.flash_matrix[1]),
                        "phantom_to_acquisition_ratio_x": matrix_ratio_x,
                        "phantom_to_acquisition_ratio_y": matrix_ratio_y,
                        "phantom_grid_relation": grid_relation,
                        "phantom_finer_than_acquisition": bool(
                            shape[0] > args.flash_matrix[0]
                            and shape[1] > args.flash_matrix[1]
                        ),
                        "total_voxels": int(np.prod(phantom.shape)),
                        "voxel_size_x_mm": float(voxel_size[0] * 1e3),
                        "voxel_size_y_mm": float(voxel_size[1] * 1e3),
                        "voxel_size_z_mm": float(voxel_size[2] * 1e3),
                        "phantom_volume_mode": "full_3d",
                        "source_slice_index": slice_index,
                        "phantom_generation_time_s": phantom_generation_time_s,
                        "end_to_end_time_s": (
                            sequence_generation_time_s
                            + phantom_generation_time_s
                            + simulation_time_s
                        ),
                        "signal_weighting": "voxel_volume",
                        "dynamic_sequence": isinstance(phantom, DynamicSpectralPhantom),
                        "dynamic_frames": int(definitions.get("DynamicFrames", 1)),
                        "dynamic_frame_interval_s": (
                            args.dynamic_frame_interval_s
                            if isinstance(phantom, DynamicSpectralPhantom)
                            else 0.0
                        ),
                        "frame_start_times_s": np.asarray(
                            definitions.get("AcquisitionStartTimes", (0.0,)),
                            dtype=float,
                        )
                        .reshape(-1)
                        .tolist(),
                    }
                )
                record.update(_temporal_record(result))
                cases.append((record, result, _reconstructed_frames(result)))
                print_runtime_row(record)

        reference_record, reference_result, reference_images = max(
            cases,
            key=lambda item: (
                int(item[0]["resolution_x"]) * int(item[0]["resolution_y"]),
                int(item[0]["repeat"]),
            ),
        )
        for record, result, images in cases:
            record["reference_resolution"] = (
                f"{reference_record['resolution_x']}x"
                f"{reference_record['resolution_y']}x"
                f"{reference_record['resolution_z']}"
            )
            record["is_reference_resolution"] = (
                record["resolution_x"] == reference_record["resolution_x"]
                and record["resolution_y"] == reference_record["resolution_y"]
            )
            record.update(
                array_similarity(
                    reference_result.signal,
                    result.signal,
                    prefix="signal_vs_finest",
                )
            )
            record.update(
                array_similarity(
                    np.abs(reference_images),
                    np.abs(images),
                    prefix="magnitude_image_vs_finest",
                )
            )
            record.update(
                array_similarity(
                    np.abs(reference_images[-1]),
                    np.abs(images[-1]),
                    prefix="last_frame_magnitude_image_vs_finest",
                )
            )
            if (
                reference_result.species_signal is not None
                and result.species_signal is not None
            ):
                record.update(
                    array_similarity(
                        reference_result.species_signal,
                        result.species_signal,
                        prefix="pool_signal_vs_finest",
                    )
                )
            all_records.append(record)
            print_resolution_comparison(record)
    return all_records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = make_output_directory(args.output_dir)
    records = run(args, output_dir=output_dir)
    csv_path, json_path = write_records(output_dir, "resolution_benchmarks", records)
    print_saved_results(csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
