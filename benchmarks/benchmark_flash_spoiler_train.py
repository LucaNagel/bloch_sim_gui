#!/usr/bin/env python3
"""Long FLASH crusher-convergence benchmark modeled on the debug project.

The default suite deliberately exercises the failure mode of a regular 3x3x3
subvoxel grid. It combines 32, 64, and 128 phase-encoding lines; 0.5, 1, and 2
effective crusher cycles per phantom voxel; regular and stratified spin layouts;
and a 32x32x64 cylindrical phantom with 32x32x32 mm FOV, 1 s T1, and 50 ms T2*.

Run from the repository root::

    python -m benchmarks.benchmark_flash_spoiler_train

The full default can take tens of minutes. Use ``--quick`` for a smoke run. A
CSV and JSON file are written below ``exports/benchmarks`` unless ``--output-dir``
is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    ScannerParameters,
    SpinSampling,
    analyze_adc_moment_train,
    infer_cartesian_acquisition,
    load_pulseq,
    make_pulseq_flash,
)
from blochsimulator.sequence.spin_sampling import phantom_voxel_basis_m

from .common import make_output_directory, write_records


DEFAULT_GRIDS = (
    "midpoint:1x1x1",
    "midpoint:3x3x3",
    "midpoint:3x4x5",
    "midpoint:5x13x1",
    "midpoint:2x3x11",
    "midpoint:4x5x7",
    "stratified:3x3x3",
    "stratified:4x5x7",
)


def make_project_like_phantom(
    shape: tuple[int, int, int] = (32, 32, 64),
) -> Phantom:
    """Return a cylindrical C13-style phantom matching the debug geometry."""
    fov_m = (32e-3, 32e-3, 32e-3)
    axes_mm = [
        ((np.arange(count, dtype=float) + 0.5) / count - 0.5) * fov * 1000.0
        for count, fov in zip(shape, fov_m)
    ]
    x_mm, y_mm, z_mm = np.meshgrid(*axes_mm, indexing="ij")
    mask = (x_mm**2 + y_mm**2 <= 12.0**2) & (np.abs(z_mm) <= 4.0)
    return Phantom(
        shape=shape,
        fov=fov_m,
        t1_map=np.full(shape, 1.0, dtype=np.float64),
        t2_map=np.full(shape, 50e-3, dtype=np.float64),
        pd_map=mask.astype(np.float64),
        mask=mask,
        name="FLASH spoiler convergence cylinder",
        metadata={"field_strength_t": 3.0, "nucleus": "C13"},
    )


def parse_grid(value: str) -> SpinSampling:
    try:
        method, counts_text = str(value).strip().lower().split(":", 1)
        counts = tuple(int(item) for item in counts_text.split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "grid must look like midpoint:3x4x5 or stratified:3x3x3"
        ) from exc
    try:
        return SpinSampling(counts, method=method)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _best_scaled_nrmse(reference, candidate) -> float:
    reference = np.asarray(reference, dtype=np.complex128).ravel()
    candidate = np.asarray(candidate, dtype=np.complex128).ravel()
    denominator = np.vdot(candidate, candidate)
    scale = np.vdot(candidate, reference) / denominator if denominator else 0.0
    reference_norm = np.linalg.norm(reference)
    return float(np.linalg.norm(scale * candidate - reference) / reference_norm)


def _magnitude_nrmse(reference, candidate) -> float:
    reference = np.abs(np.asarray(reference)).ravel()
    candidate = np.abs(np.asarray(candidate)).ravel()
    denominator = float(np.dot(candidate, candidate))
    scale = float(np.dot(candidate, reference) / denominator) if denominator else 0.0
    return float(
        np.linalg.norm(scale * candidate - reference) / np.linalg.norm(reference)
    )


def _write_and_load_flash(
    path: Path,
    *,
    matrix_size: int,
    effective_cycles_per_phantom_voxel: float,
    phantom: Phantom,
):
    voxel_sizes = np.asarray(phantom.fov) / np.asarray(phantom.shape)
    nominal_in_plane_voxel = 32e-3 / matrix_size
    sequence = make_pulseq_flash(
        fov_m=(32e-3, 32e-3),
        matrix=(matrix_size, matrix_size),
        sampling_bandwidth_hz=100_000.0,
        flip_angle_deg=15.0,
        slice_thickness_m=8e-3,
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        rf_spoiling_increment_deg=117.0,
        spoiler_cycles_per_slice=(
            effective_cycles_per_phantom_voxel * 8e-3 / voxel_sizes[2]
        ),
        spoiler_cycles_per_voxel=(
            effective_cycles_per_phantom_voxel * nominal_in_plane_voxel / voxel_sizes[0]
        ),
        spoiler_duration_s=2e-3,
        scanner_parameters=ScannerParameters(
            max_grad_mtm=100.0,
            max_slew_tms=500.0,
        ),
    )
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def run_benchmark(
    *,
    output_dir: Path,
    matrix_sizes=(32, 64, 128),
    spoiler_strengths=(0.5, 1.0, 2.0),
    samplings=tuple(parse_grid(value) for value in DEFAULT_GRIDS),
    phantom_shape=(32, 32, 64),
    timestep_us: float = 10.0,
    threads: int = 0,
) -> list[dict]:
    """Run all ideal references and physical-gradient sampling cases."""
    output_dir = make_output_directory(output_dir)
    sequence_dir = output_dir / "generated_sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    phantom = make_project_like_phantom(tuple(int(value) for value in phantom_shape))
    simulator = BlochSimulator(
        use_parallel=True,
        num_threads=None if int(threads) <= 0 else int(threads),
        sequence_kernel="optimized",
    )
    records: list[dict] = []
    total_cases = (
        len(tuple(matrix_sizes))
        * len(tuple(spoiler_strengths))
        * (1 + len(tuple(samplings)))
    )
    case_index = 0

    for matrix_size in matrix_sizes:
        for strength in spoiler_strengths:
            label = str(float(strength)).replace(".", "p")
            sequence_path = sequence_dir / f"flash_m{matrix_size}_c{label}.seq"
            program = _write_and_load_flash(
                sequence_path,
                matrix_size=int(matrix_size),
                effective_cycles_per_phantom_voxel=float(strength),
                phantom=phantom,
            )
            acquisition = infer_cartesian_acquisition(program)

            case_index += 1
            print(
                f"[{case_index}/{total_cases}] matrix={matrix_size}, cycles={strength}, ideal",
                flush=True,
            )
            start = perf_counter()
            ideal = simulator.simulate_sequence(
                program,
                phantom,
                simulation_timestep_s=float(timestep_us) * 1e-6,
                spin_sampling=SpinSampling(),
                spoiler_mode="ideal",
            )
            ideal_runtime = perf_counter() - start
            ideal_image = acquisition.reconstruct(ideal.signal)
            ideal_mxy = np.hypot(
                ideal.final_magnetization[..., 0],
                ideal.final_magnetization[..., 1],
            )
            records.append(
                {
                    "matrix_size": int(matrix_size),
                    "spoiler_cycles_per_phantom_voxel": float(strength),
                    "sampling_method": "ideal",
                    "spins_x": 1,
                    "spins_y": 1,
                    "spins_z": 1,
                    "spins_per_voxel": 1,
                    "simulation_time_s": ideal_runtime,
                    "runtime_vs_ideal": 1.0,
                    "signal_best_scaled_nrmse": 0.0,
                    "image_magnitude_best_scaled_nrmse": 0.0,
                    "maximum_train_sampling_error": 0.0,
                    "first_alias_observation": "",
                    "final_active_mean_mxy": float(
                        np.mean(ideal_mxy[np.asarray(phantom.mask, dtype=bool)])
                    ),
                    "phantom_shape": "x".join(map(str, phantom.shape)),
                    "phantom_active_voxels": int(phantom.n_active),
                }
            )

            for sampling in samplings:
                case_index += 1
                print(
                    f"[{case_index}/{total_cases}] matrix={matrix_size}, cycles={strength}, "
                    f"{sampling.method}:{'x'.join(map(str, sampling.counts_xyz))}",
                    flush=True,
                )
                start = perf_counter()
                gradient = simulator.simulate_sequence(
                    program,
                    phantom,
                    simulation_timestep_s=float(timestep_us) * 1e-6,
                    spin_sampling=sampling,
                    spoiler_mode="gradient",
                )
                runtime = perf_counter() - start
                gradient_image = acquisition.reconstruct(gradient.signal)
                train = analyze_adc_moment_train(
                    acquisition.moment_origins_cyc_per_m,
                    phantom_voxel_basis_m(phantom),
                    sampling,
                )
                final_mxy = np.hypot(
                    gradient.final_magnetization[..., 0],
                    gradient.final_magnetization[..., 1],
                )
                counts = sampling.counts_xyz
                records.append(
                    {
                        "matrix_size": int(matrix_size),
                        "spoiler_cycles_per_phantom_voxel": float(strength),
                        "sampling_method": sampling.method,
                        "spins_x": counts[0],
                        "spins_y": counts[1],
                        "spins_z": counts[2],
                        "spins_per_voxel": sampling.spins_per_voxel,
                        "simulation_time_s": runtime,
                        "runtime_vs_ideal": runtime / ideal_runtime,
                        "signal_best_scaled_nrmse": _best_scaled_nrmse(
                            ideal.signal, gradient.signal
                        ),
                        "image_magnitude_best_scaled_nrmse": _magnitude_nrmse(
                            ideal_image, gradient_image
                        ),
                        "maximum_train_sampling_error": train.maximum_sampling_error,
                        "first_alias_observation": (
                            train.first_alias_observation or ""
                        ),
                        "final_active_mean_mxy": float(
                            np.mean(final_mxy[np.asarray(phantom.mask, dtype=bool)])
                        ),
                        "phantom_shape": "x".join(map(str, phantom.shape)),
                        "phantom_active_voxels": int(phantom.n_active),
                    }
                )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--matrix-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument(
        "--spoiler-strengths", nargs="+", type=float, default=[0.5, 1.0, 2.0]
    )
    parser.add_argument("--grids", nargs="+", type=parse_grid, default=None)
    parser.add_argument("--phantom-shape", nargs=3, type=int, default=[32, 32, 64])
    parser.add_argument("--timestep-us", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a 12x12x24 phantom, matrices 8/16, two strengths, and three grids.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.phantom_shape = [12, 12, 24]
        args.matrix_sizes = [8, 16]
        args.spoiler_strengths = [0.5, 1.0]
        args.grids = [
            parse_grid("midpoint:3x3x3"),
            parse_grid("midpoint:3x4x5"),
            parse_grid("stratified:3x3x3"),
        ]
    samplings = (
        tuple(args.grids)
        if args.grids is not None
        else tuple(parse_grid(value) for value in DEFAULT_GRIDS)
    )
    output_dir = make_output_directory(args.output_dir)
    records = run_benchmark(
        output_dir=output_dir,
        matrix_sizes=tuple(args.matrix_sizes),
        spoiler_strengths=tuple(args.spoiler_strengths),
        samplings=samplings,
        phantom_shape=tuple(args.phantom_shape),
        timestep_us=args.timestep_us,
        threads=args.threads,
    )
    csv_path, json_path = write_records(
        output_dir, "flash_spoiler_train_benchmarks", records
    )
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
