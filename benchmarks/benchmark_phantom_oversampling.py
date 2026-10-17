#!/usr/bin/env python3
"""Benchmark phantom grids at 1x, 2x, 4x, ... the sequence's spatial resolution.

Goal
----
Measure whether a phantom whose grid is finer than the acquisition matrix
changes the simulated signal and images, and what the finer grid costs. The
same analytic two-container object is rasterized with exact partial-volume
fractions at every ratio, so a finer grid describes the same object more
accurately instead of upsampling a fixed coarse source. The finest requested
ratio (default 8x) is the numerical reference.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.benchmark_phantom_oversampling
    python -m benchmarks.benchmark_phantom_oversampling \\
        --oversampling-ratios 1 2 4 --oversampling-sequences flash
    python -m benchmarks.benchmark_phantom_oversampling \\
        --oversampling-phantoms static --repeats 3

For 2D sequences (FLASH, CSI) the in-plane phantom grid is ``ratio x`` the
acquisition matrix while the Z sampling stays fixed
(``--oversampling-slice-planes``), so only the in-plane resolution changes.
For 3D bSSFP all three axes scale. Spoiling is ideal, so subvoxel spins are
not used and the phantom grid is the only intravoxel sampling. The shared
``--phantoms`` option is not used here; use ``--oversampling-phantoms``.

Reading the output
------------------
``signal_relative_l2_error`` and ``image_relative_l2_error`` compare the
received signal and the complex reconstruction with the finest grid
(0.01 = 1 %). ``max_roi_deviation`` is the largest change of a container's
mean metabolite image intensity, relative to the brightest container ROI;
this is the error a quantitative ROI analysis would see. ``active_voxels``
and ``simulation_time_s`` show the cost, which grows with the number of
phantom voxels (about ratio^2 in 2D and ratio^3 in 3D).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from time import perf_counter

import numpy as np

from blochsimulator.sequence import (
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_flash,
)

try:
    from .common import (
        _slug,
        add_common_arguments,
        array_similarity,
        base_result_record,
        make_output_directory,
        make_simulator,
        parse_positive_int,
        print_saved_results,
        simulate_program,
        write_and_load_sequence,
        write_records,
    )
    from .synthetic_phantoms import (
        CONTAINER_HALF_LENGTH_M,
        FOV_M,
        SYNTHETIC_PHANTOM_KINDS,
        container_core_masks,
        make_synthetic_phantom,
    )
except ImportError:  # Allow: python benchmarks/benchmark_phantom_oversampling.py
    from common import (  # type: ignore
        _slug,
        add_common_arguments,
        array_similarity,
        base_result_record,
        make_output_directory,
        make_simulator,
        parse_positive_int,
        print_saved_results,
        simulate_program,
        write_and_load_sequence,
        write_records,
    )
    from synthetic_phantoms import (  # type: ignore
        CONTAINER_HALF_LENGTH_M,
        FOV_M,
        SYNTHETIC_PHANTOM_KINDS,
        container_core_masks,
        make_synthetic_phantom,
    )


DEFAULT_OVERSAMPLING_RATIOS = (1, 2, 4, 8)
DEFAULT_SLICE_PLANES = 32
DEFAULT_DYNAMIC_FRAMES = 3
SLICE_THICKNESS_M = 8e-3
CONTAINER_NAMES = ("A", "B")
OVERSAMPLING_SEQUENCES = {
    "flash": {
        "description": "2D FLASH 16x16, 2 ms sinc pulse, 8 mm slice",
        "matrix": (16, 16),
    },
    "csi": {
        "description": "2D FID-CSI 8x8, 64 spectral points, 8 mm slice",
        "matrix": (8, 8),
    },
    "bssfp": {
        "description": "3D bSSFP 8x8x8, 1 ms block pulse",
        "matrix": (8, 8, 8),
    },
}


def add_oversampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--oversampling-ratios",
        nargs="+",
        type=parse_positive_int,
        default=DEFAULT_OVERSAMPLING_RATIOS,
        metavar="RATIO",
        help=(
            "Phantom grid per acquisition voxel along each scaled axis "
            "(default: 1 2 4 8; the largest is the reference)."
        ),
    )
    parser.add_argument(
        "--oversampling-sequences",
        nargs="+",
        choices=tuple(OVERSAMPLING_SEQUENCES),
        default=tuple(OVERSAMPLING_SEQUENCES),
        metavar="SEQUENCE",
        help="Sequences to compare: flash, csi, bssfp (default: all).",
    )
    parser.add_argument(
        "--oversampling-phantoms",
        nargs="+",
        choices=SYNTHETIC_PHANTOM_KINDS,
        default=SYNTHETIC_PHANTOM_KINDS,
        metavar="KIND",
        help="Analytic phantom variants: static, dynamic (default: both).",
    )
    parser.add_argument(
        "--oversampling-slice-planes",
        type=parse_positive_int,
        default=DEFAULT_SLICE_PLANES,
        help="Fixed Z planes of the phantom for 2D sequences (default: 32).",
    )
    parser.add_argument(
        "--oversampling-dynamic-frames",
        type=parse_positive_int,
        default=DEFAULT_DYNAMIC_FRAMES,
        help="FLASH frames, 1 s apart, for the dynamic phantom (default: 3).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_oversampling_arguments(parser)
    # The GUI defaults; neither affects the grid comparison itself.
    parser.set_defaults(dynamic_kernel="native_parallel", timestep_us=5.0)
    return parser


def build_oversampling_sequence(name: str, *, dynamic_frames: int = 1):
    """Build one generated sequence of the phantom-oversampling comparison."""
    matrix = OVERSAMPLING_SEQUENCES[name]["matrix"]
    if name == "flash":
        return make_pulseq_flash(
            fov_m=FOV_M[:2],
            matrix=matrix,
            sampling_bandwidth_hz=20_000.0,
            flip_angle_deg=10.0,
            rf_pulse_type="sinc",
            rf_duration_s=2e-3,
            rf_time_bandwidth_product=4.0,
            slice_thickness_m=SLICE_THICKNESS_M,
            spoiler_cycles_per_slice=1.0,
            repetitions=dynamic_frames,
            acquisition_interval_s=1.0 if dynamic_frames > 1 else None,
            nucleus="C13",
        )
    if name == "csi":
        return make_pulseq_csi(
            fov_m=FOV_M[:2],
            matrix=matrix,
            slice_thickness_m=SLICE_THICKNESS_M,
            spectral_bandwidth_hz=4_000.0,
            spectral_points=64,
            repetition_time_s=0.05,
            nucleus="C13",
        )
    if name == "bssfp":
        return make_pulseq_bssfp(
            fov_m=FOV_M,
            matrix=matrix,
            flip_angle_deg=20.0,
            repetitions=1,
            dummy_repetitions=0,
            nucleus="C13",
        )
    raise ValueError(f"unknown oversampling sequence {name!r}")


def oversampled_shape(name: str, ratio: int, slice_planes: int) -> tuple[int, ...]:
    """Phantom grid for ``ratio`` phantom voxels per acquisition voxel."""
    matrix = OVERSAMPLING_SEQUENCES[name]["matrix"]
    if len(matrix) == 3:
        return tuple(int(value) * ratio for value in matrix)
    return (matrix[0] * ratio, matrix[1] * ratio, int(slice_planes))


def _images(result, signal) -> np.ndarray:
    """Return reconstructions with a leading frame axis.

    2D images are ``(frame, y, x)``, CSI amplitude maps ``(frame, y, x, t)``
    and 3D volumes ``(frame, z, y, x)``.
    """
    view = dataclasses.replace(result, signal=np.asarray(signal))
    if view.cartesian_acquisition_volumes is not None:
        volume = np.asarray(view.reconstruct_cartesian_3d())
        return volume.reshape((-1,) + volume.shape[-3:])
    if view.cartesian_acquisition_frames is not None:
        frames = view.cartesian_acquisition_frames
        return np.stack(
            [frames.reconstruct(view, frame) for frame in range(frames.num_frames)]
        )
    if view.cartesian_acquisition is not None:
        return view.cartesian_acquisition.reconstruct(view.signal)[None]
    if view.spectroscopic_acquisition is not None:
        return view.spectroscopic_acquisition.reconstruct_spatial(view.signal)[None]
    raise ValueError("oversampling result has no supported reconstruction")


def _roi_masks(name: str) -> list[np.ndarray]:
    """Container core masks in reconstruction orientation."""
    matrix = OVERSAMPLING_SEQUENCES[name]["matrix"]
    cores = [mask.T for mask in container_core_masks((*matrix[:2], 1))]
    if len(matrix) == 2:
        return cores
    edges = np.linspace(-FOV_M[2] / 2.0, FOV_M[2] / 2.0, matrix[2] + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    inside_z = np.abs(centres) <= 0.5 * CONTAINER_HALF_LENGTH_M
    return [inside_z[:, None, None] & core[None] for core in cores]


def _roi_values(name: str, result) -> dict[tuple[str, str], np.ndarray]:
    """Mean magnitude per (container, metabolite) and frame."""
    masks = _roi_masks(name)
    values = {}
    pools = getattr(result, "pool_names", None) or ("total",)
    signals = (
        result.species_signal if result.species_signal is not None else [result.signal]
    )
    for pool, signal in zip(pools, signals):
        images = _images(result, signal)
        if name == "csi":
            magnitude = np.sqrt(np.sum(np.abs(images) ** 2, axis=-1))
        else:
            magnitude = np.abs(images)
        for container, mask in zip(CONTAINER_NAMES, masks):
            values[(container, str(pool))] = magnitude[:, mask].mean(axis=-1)
    return values


def _roi_metrics(values, reference) -> dict:
    scale = max(float(np.max(item)) for item in reference.values())
    metrics = {}
    worst = 0.0
    for (container, pool), current in values.items():
        deviation = float(np.max(np.abs(current - reference[(container, pool)])))
        deviation = deviation / scale if scale > 0.0 else deviation
        key = f"roi_{container}_{_slug(pool)}"
        metrics[f"{key}_mean"] = float(np.mean(current))
        metrics[f"{key}_deviation"] = deviation
        worst = max(worst, deviation)
    metrics["max_roi_deviation"] = worst
    return metrics


def _print_header(args, output_dir: Path, ratios) -> None:
    threads = "automatic" if args.threads == 0 else str(args.threads)
    print("\n" + "=" * 78, flush=True)
    print("Benchmark: phantom grid relative to the acquisition matrix", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(
        "  Ratios: "
        + ", ".join(f"{value}x" for value in ratios)
        + f" | reference: {max(ratios)}x | RF time step {args.timestep_us:g} us",
        flush=True,
    )
    print(
        f"  Repeats: {args.repeats} (median runtime) | threads: {threads} | "
        f"kernels: sequence={args.sequence_kernel}, dynamic={args.dynamic_kernel}",
        flush=True,
    )
    print("=" * 78, flush=True)


def _print_row(record) -> None:
    marker = " (reference)" if record["is_reference"] else ""
    print(
        f"    {record['oversampling_ratio']:>2d}x {record['phantom_shape']:>12s}"
        f"{marker}: {record['simulation_time_s']:.3f} s | "
        f"active voxels {record['active_voxels']:,} | "
        f"signal error {record['signal_relative_l2_error']:.2e} | "
        f"image error {record['image_relative_l2_error']:.2e} | "
        f"ROI deviation {record['max_roi_deviation']:.2e}",
        flush=True,
    )


def run(args, output_dir=None) -> list[dict]:
    output_dir = make_output_directory(output_dir or args.output_dir)
    sequence_dir = output_dir / "oversampling_sequences"
    ratios = sorted({int(value) for value in args.oversampling_ratios})
    _print_header(args, output_dir, ratios)
    simulator = make_simulator(args)
    records = []
    for kind in args.oversampling_phantoms:
        for name in args.oversampling_sequences:
            frames = (
                args.oversampling_dynamic_frames
                if kind == "dynamic" and name == "flash"
                else 1
            )
            start = perf_counter()
            program = write_and_load_sequence(
                build_oversampling_sequence(name, dynamic_frames=frames),
                sequence_dir / f"{kind}_{name}.seq",
            )
            setup_s = perf_counter() - start
            description = OVERSAMPLING_SEQUENCES[name]["description"]
            if frames > 1:
                description += f", {frames} frames 1 s apart"
            print(f"\n  {kind} phantom | {name}: {description}", flush=True)
            cases = []
            # Finest grid first so every later case is compared immediately.
            for ratio in sorted(ratios, reverse=True):
                shape = oversampled_shape(name, ratio, args.oversampling_slice_planes)
                phantom = make_synthetic_phantom(kind, shape)
                durations = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    result = simulate_program(
                        simulator,
                        program,
                        phantom,
                        timestep_s=args.timestep_us * 1e-6,
                        spoiler_mode="ideal",
                        spin_sampling=(1, 1, 1),
                        signal_weighting="voxel_volume",
                    )
                    durations.append(perf_counter() - start)
                runtime = float(np.median(durations))
                images = _images(result, result.signal)
                roi = _roi_values(name, result)
                if not cases:
                    reference = (result, images, roi, runtime)
                cases.append(ratio)
                record = base_result_record(
                    benchmark="phantom_oversampling",
                    sequence_name=name,
                    phantom_label=f"synthetic_{kind}",
                    phantom_path=Path(f"synthetic:{kind}"),
                    phantom=phantom,
                    program=program,
                    result=result,
                    repeat=args.repeats,
                    generation_time_s=setup_s,
                    simulation_time_s=runtime,
                )
                record.pop("repeat")
                acquisition_matrix = OVERSAMPLING_SEQUENCES[name]["matrix"]
                record.update(
                    {
                        "sequence_description": description,
                        "acquisition_matrix": "x".join(map(str, acquisition_matrix)),
                        "oversampling_ratio": ratio,
                        "reference_ratio": ratios[-1],
                        "is_reference": ratio == ratios[-1],
                        "phantom_voxels": int(np.prod(shape)),
                        "phantom_voxel_mm": "x".join(
                            f"{fov / count * 1e3:g}" for fov, count in zip(FOV_M, shape)
                        ),
                        "repeats": args.repeats,
                        "simulation_time_min_s": float(np.min(durations)),
                        "runtime_vs_1x": float("nan"),
                        "timestep_us": float(args.timestep_us),
                        "dynamic_frames": frames,
                    }
                )
                record.update(
                    array_similarity(
                        reference[0].signal, result.signal, prefix="signal"
                    )
                )
                record.update(array_similarity(reference[1], images, prefix="image"))
                record.update(
                    array_similarity(
                        np.abs(reference[1]), np.abs(images), prefix="magnitude_image"
                    )
                )
                record.update(_roi_metrics(roi, reference[2]))
                records.append(record)
                _print_row(record)
            group = [
                item
                for item in records
                if item["sequence"] == name and item["phantom"] == f"synthetic_{kind}"
            ]
            base = min(group, key=lambda item: item["oversampling_ratio"])
            for item in group:
                item["runtime_vs_1x"] = (
                    item["simulation_time_s"] / base["simulation_time_s"]
                    if base["simulation_time_s"] > 0
                    else float("nan")
                )
    return records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = make_output_directory(args.output_dir)
    records = run(args, output_dir=output_dir)
    csv_path, json_path = write_records(
        output_dir, "phantom_oversampling_benchmarks", records
    )
    print_saved_results(csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
