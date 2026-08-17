#!/usr/bin/env python3
"""Compare Sequence Bloch and dynamic two-pool simulation kernels.

Goal
----
Measure speed and numerical agreement between implementations of two different
physical solvers. Sequence Bloch kernels propagate ordinary independent Bloch
states. Dynamic two-pool kernels additionally retain coupled pyruvate/lactate
states and integrate conversion, polarization, inflow, and optional dynamic B0.
Each optimized, native, or experimental result is compared with the reference
implementation from the same solver family.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.benchmark_kernels
    python -m benchmarks.benchmark_kernels --repeats 3
    python -m benchmarks.benchmark_kernels \\
        --sequence-kernels reference optimized \\
        --dynamic-kernels reference optimized native_serial native_parallel

``--kernel-phantom-matrix`` changes only X/Y sampling; the complete source Z
volume remains in the simulation. Increase it to study realistic large-object
behavior. The default dynamic ``1 x 1 x 9`` subvoxel grid normally crosses the
threshold at which ``native_parallel`` can use multiple CPU workers.

Reading the output
------------------
``speedup_vs_reference`` above one is faster than the reference implementation.
Signal and final-state similarities close to one indicate numerical agreement.
The reference path is a correctness oracle, not the recommended production
kernel. A row with ``fallback_used=true`` does not measure the requested backend;
inspect ``actual_kernel`` and ``fallback_reason`` before drawing a performance
conclusion, especially for the experimental Metal hybrid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_metal_backend import metal_capability
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom

try:
    from .benchmark_resolution import (
        build_resolution_flash_sequence,
        make_resolution_volume,
        parse_phantom_matrix,
    )
    from .common import (
        add_common_arguments,
        array_similarity,
        base_result_record,
        load_phantom,
        make_output_directory,
        print_benchmark_header,
        print_phantom_summary,
        print_saved_results,
        print_sequence_summary,
        print_simulation_start,
        resolve_phantom_paths,
        result_similarity,
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )
except ImportError:  # Allow: python benchmarks/benchmark_kernels.py
    from benchmark_resolution import (  # type: ignore
        build_resolution_flash_sequence,
        make_resolution_volume,
        parse_phantom_matrix,
    )
    from common import (  # type: ignore
        add_common_arguments,
        array_similarity,
        base_result_record,
        load_phantom,
        make_output_directory,
        print_benchmark_header,
        print_phantom_summary,
        print_saved_results,
        print_sequence_summary,
        print_simulation_start,
        resolve_phantom_paths,
        result_similarity,
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )


SEQUENCE_KERNELS = ("reference", "optimized")
DYNAMIC_KERNELS = (
    "reference",
    "optimized",
    "native_serial",
    "native_parallel",
    "metal_hybrid",
)
DEFAULT_KERNEL_PHANTOM_MATRIX = (24, 24)
DEFAULT_KERNEL_FLASH_MATRIX = (16, 16)
DEFAULT_KERNEL_SUBVOXEL_SPINS = (1, 1, 9)

KERNEL_DESCRIPTIONS = {
    "sequence/reference": (
        "General rotation-matrix and per-spin relaxation path. It is the "
        "ordinary Bloch correctness reference and intentionally omits fast paths."
    ),
    "sequence/optimized": (
        "Native streaming ordinary-Bloch path with RF-free, quaternion, and "
        "relaxation-cache fast paths; spins are distributed over CPU threads."
    ),
    "dynamic/reference": (
        "Direct allocation-heavy NumPy two-pool implementation. It is the "
        "correctness oracle for pyruvate/lactate kinetics."
    ),
    "dynamic/optimized": (
        "Production NumPy two-pool path with persistent state, reusable scratch "
        "arrays, and cached kinetic, relaxation, and phase coefficients."
    ),
    "dynamic/native_serial": (
        "Optimized dynamic driver with strict compiled RF and longitudinal "
        "blocks using one worker thread; unsupported pieces fall back explicitly."
    ),
    "dynamic/native_parallel": (
        "Native dynamic blocks with OpenMP across spins. Small objects may use "
        "one thread, and unsupported pieces retain the optimized NumPy path."
    ),
    "dynamic/metal_hybrid": (
        "Experimental Apple-GPU subvoxel calculation checked against independent "
        "Float64 CPU samples; unavailable or failed validation uses exact CPU fallback."
    ),
}


def add_kernel_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sequence-kernels",
        nargs="+",
        choices=SEQUENCE_KERNELS,
        default=list(SEQUENCE_KERNELS),
        help="Ordinary Sequence Bloch kernels to compare (default: both).",
    )
    parser.add_argument(
        "--dynamic-kernels",
        nargs="+",
        choices=DYNAMIC_KERNELS,
        default=list(DYNAMIC_KERNELS),
        help="Dynamic two-pool kernels to compare (default: all).",
    )
    parser.add_argument(
        "--kernel-phantom-matrix",
        type=parse_phantom_matrix,
        default=DEFAULT_KERNEL_PHANTOM_MATRIX,
        metavar="READxPHASE",
        help=(
            "In-plane phantom grid used for kernel comparisons; the complete "
            "source Z volume is retained (default: 24x24)."
        ),
    )
    parser.add_argument(
        "--kernel-flash-matrix",
        nargs=2,
        type=int,
        default=DEFAULT_KERNEL_FLASH_MATRIX,
        metavar=("READ", "PHASE"),
        help="Fixed 2D FLASH acquisition matrix (default: 16 16).",
    )
    parser.add_argument(
        "--kernel-dynamic-frames",
        type=int,
        default=2,
        help="FLASH frames for the dynamic kernel comparison (default: 2).",
    )
    parser.add_argument(
        "--kernel-dynamic-frame-interval-s",
        type=float,
        default=0.25,
        help="Dynamic FLASH start-to-start frame interval (default: 0.25 s).",
    )
    parser.add_argument(
        "--kernel-subvoxel-spins",
        nargs=3,
        type=int,
        default=DEFAULT_KERNEL_SUBVOXEL_SPINS,
        metavar=("X", "Y", "Z"),
        help=(
            "Common dynamic subvoxel grid; at least three total spins are needed "
            "to exercise Metal rather than its CPU fallback (default: 1 1 9)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_kernel_arguments(parser)
    return parser


def print_kernel_descriptions(sequence_kernels, dynamic_kernels) -> None:
    print("\nKernel families and implementations:", flush=True)
    print(
        "  Sequence Bloch kernels simulate ordinary Bloch propagation. A static "
        "spectral phantom runs each metabolite independently and sums its signal.",
        flush=True,
    )
    for kernel in sequence_kernels:
        print(f"    {kernel}: {KERNEL_DESCRIPTIONS[f'sequence/{kernel}']}", flush=True)
    print(
        "  Dynamic two-pool kernels retain coupled pyruvate and lactate states and "
        "also integrate kPL conversion, polarization, inflow, and optional dynamic B0.",
        flush=True,
    )
    for kernel in dynamic_kernels:
        print(f"    {kernel}: {KERNEL_DESCRIPTIONS[f'dynamic/{kernel}']}", flush=True)

    if "metal_hybrid" in dynamic_kernels:
        capability = metal_capability()
        status = (
            f"available ({capability.get('device_name') or 'Apple GPU'})"
            if capability["available"]
            else f"unavailable: {capability.get('reason') or 'unknown reason'}"
        )
        print(f"  Metal capability: {status}", flush=True)


def _validate_args(args) -> None:
    if len(args.kernel_flash_matrix) != 2 or min(args.kernel_flash_matrix) <= 0:
        raise ValueError("--kernel-flash-matrix requires two positive integers")
    if args.kernel_dynamic_frames < 2:
        raise ValueError("--kernel-dynamic-frames must be at least 2")
    if (
        not np.isfinite(args.kernel_dynamic_frame_interval_s)
        or args.kernel_dynamic_frame_interval_s <= 0
    ):
        raise ValueError(
            "--kernel-dynamic-frame-interval-s must be positive and finite"
        )
    if len(args.kernel_subvoxel_spins) != 3 or min(args.kernel_subvoxel_spins) <= 0:
        raise ValueError("--kernel-subvoxel-spins requires three positive integers")
    if "reference" not in args.sequence_kernels:
        raise ValueError("--sequence-kernels must include reference for comparison")
    if "reference" not in args.dynamic_kernels:
        raise ValueError("--dynamic-kernels must include reference for comparison")


def _sequence_arguments(args, dynamic: bool) -> argparse.Namespace:
    return argparse.Namespace(
        flash_matrix=tuple(args.kernel_flash_matrix),
        dynamic_frames=args.kernel_dynamic_frames if dynamic else 1,
        dynamic_frame_interval_s=(
            args.kernel_dynamic_frame_interval_s if dynamic else None
        ),
    )


def _make_simulator(args, *, sequence_kernel: str, dynamic_kernel: str):
    threads = None if args.threads == 0 else args.threads
    return BlochSimulator(
        use_parallel=args.threads != 1,
        num_threads=threads,
        sequence_kernel=sequence_kernel,
        dynamic_sequence_kernel=dynamic_kernel,
    )


def _kernel_similarity(reference, candidate) -> dict:
    metrics = result_similarity(reference, candidate)
    metrics.update(
        array_similarity(
            reference.final_magnetization,
            candidate.final_magnetization,
            prefix="final_state",
        )
    )
    reference_pool = getattr(reference, "final_pool_magnetization", None)
    candidate_pool = getattr(candidate, "final_pool_magnetization", None)
    if reference_pool is not None and candidate_pool is not None:
        metrics.update(
            array_similarity(
                reference_pool,
                candidate_pool,
                prefix="pool_final_state",
            )
        )
    return metrics


def _kernel_record(
    *,
    args,
    family: str,
    kernel: str,
    reference_result,
    reference_time_s: float,
    result,
    simulation_time_s: float,
    phantom_label: str,
    phantom_path: Path,
    phantom,
    program,
    repeat: int,
    generation_time_s: float,
) -> dict:
    record = base_result_record(
        benchmark="kernels",
        sequence_name="flash_2d",
        phantom_label=phantom_label,
        phantom_path=phantom_path,
        phantom=phantom,
        program=program,
        result=result,
        repeat=repeat,
        generation_time_s=generation_time_s,
        simulation_time_s=simulation_time_s,
    )
    metadata = dict(result.metadata)
    actual_kernel = str(metadata.get("sequence_kernel", kernel))
    fallback_reason = metadata.get("hybrid_fallback_reason") or metadata.get(
        "native_fallback_reason"
    )
    record.update(_kernel_similarity(reference_result, result))
    record.update(
        {
            "kernel_family": family,
            "kernel": kernel,
            "requested_kernel": kernel,
            "actual_kernel": actual_kernel,
            "kernel_description": KERNEL_DESCRIPTIONS[f"{family}/{kernel}"],
            "reference_kernel": "reference",
            "runtime_vs_reference": (
                simulation_time_s / reference_time_s
                if reference_time_s > 0
                else float("inf")
            ),
            "speedup_vs_reference": (
                reference_time_s / simulation_time_s
                if simulation_time_s > 0
                else float("inf")
            ),
            "fallback_used": bool(actual_kernel != kernel or fallback_reason),
            "fallback_reason": (
                None if fallback_reason is None else str(fallback_reason)
            ),
            "performance_result_is_requested_kernel": bool(actual_kernel == kernel),
            "profile": args.profile,
            "timestep_us": args.timestep_us,
            "requested_threads": args.threads,
            "dynamic_frames": (
                args.kernel_dynamic_frames if family == "dynamic" else 1
            ),
            "dynamic_frame_interval_s": (
                args.kernel_dynamic_frame_interval_s if family == "dynamic" else 0.0
            ),
            "native_rf_block_enabled": metadata.get("native_rf_block_enabled"),
            "native_rf_fused_block_enabled": metadata.get(
                "native_rf_fused_block_enabled"
            ),
            "native_parallel_threads": metadata.get("native_parallel_threads"),
            "hybrid_validation_passed": metadata.get("hybrid_validation_passed"),
            "actual_backend": metadata.get("actual_backend"),
        }
    )
    return record


def _print_kernel_result(record) -> None:
    requested = record["requested_kernel"]
    actual = record["actual_kernel"]
    print(
        f"  Completed kernel={requested}: {record['simulation_time_s']:.3f} s | "
        f"actual={actual} | speed-up vs reference="
        f"{record['speedup_vs_reference']:.3f}x | signal similarity="
        f"{record['signal_l2_similarity']:.9f} | final-state similarity="
        f"{record['final_state_l2_similarity']:.9f}",
        flush=True,
    )
    if record["fallback_used"]:
        print(
            "    Fallback: this timing is not a valid performance measurement for "
            f"{requested}: {record['fallback_reason'] or f'actual kernel was {actual}'}",
            flush=True,
        )


def _run_family(
    *,
    args,
    family: str,
    kernels,
    phantom_label: str,
    phantom_path: Path,
    phantom,
    program,
    generation_time_s: float,
) -> list[dict]:
    records = []
    spin_sampling = (
        tuple(args.kernel_subvoxel_spins) if family == "dynamic" else (1, 1, 1)
    )
    timestep_s = args.timestep_us * 1e-6
    for repeat in range(1, args.repeats + 1):
        results = {}
        timings = {}
        ordered_kernels = [
            "reference",
            *(item for item in kernels if item != "reference"),
        ]
        for kernel in ordered_kernels:
            simulator = _make_simulator(
                args,
                sequence_kernel=kernel if family == "sequence" else "optimized",
                dynamic_kernel=kernel if family == "dynamic" else "optimized",
            )
            print_simulation_start(
                repeat=repeat,
                repeats=args.repeats,
                spoiler_mode="ideal",
                spin_sampling=spin_sampling,
                timestep_us=args.timestep_us,
                details={
                    "kernel family": family,
                    "kernel": kernel,
                    "phantom": phantom_label,
                },
            )
            result, runtime_s = timed_simulation(
                simulator,
                program,
                phantom,
                timestep_s=timestep_s,
                spoiler_mode="ideal",
                spin_sampling=spin_sampling,
                signal_weighting="voxel_volume",
            )
            results[kernel] = result
            timings[kernel] = runtime_s
            record = _kernel_record(
                args=args,
                family=family,
                kernel=kernel,
                reference_result=results["reference"],
                reference_time_s=timings["reference"],
                result=result,
                simulation_time_s=runtime_s,
                phantom_label=phantom_label,
                phantom_path=phantom_path,
                phantom=phantom,
                program=program,
                repeat=repeat,
                generation_time_s=generation_time_s,
            )
            records.append(record)
            _print_kernel_result(record)
    return records


def run(args, output_dir=None) -> list[dict]:
    _validate_args(args)
    output_dir = make_output_directory(
        args.output_dir if output_dir is None else output_dir
    )
    sequence_dir = output_dir / "generated_sequences"
    phantom_dir = output_dir / "generated_phantoms"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    print_benchmark_header(
        "Simulation kernel comparison",
        args,
        output_dir,
        selection={
            "Sequence Bloch kernels": args.sequence_kernels,
            "Dynamic two-pool kernels": args.dynamic_kernels,
            "Phantom X/Y matrix": args.kernel_phantom_matrix,
            "Phantom volume mode": "full 3D; source Z planes retained",
            "FLASH matrix": args.kernel_flash_matrix,
            "Dynamic subvoxel spins": args.kernel_subvoxel_spins,
        },
        show_configured_kernels=False,
    )
    print_kernel_descriptions(args.sequence_kernels, args.dynamic_kernels)

    records = []
    found_families = set()
    for phantom_label, source_path in resolve_phantom_paths(args.phantoms):
        source = load_phantom(source_path)
        dynamic = isinstance(source, DynamicSpectralPhantom)
        family = "dynamic" if dynamic else "sequence"
        kernels = args.dynamic_kernels if dynamic else args.sequence_kernels
        found_families.add(family)
        if source.ndim != 3:
            raise ValueError("kernel benchmark requires three-dimensional sources")
        slice_index = source.shape[2] // 2
        phantom_start = perf_counter()
        phantom = make_resolution_volume(source, tuple(args.kernel_phantom_matrix))
        generated_phantom_path = phantom_dir / (
            f"{phantom_label}_{family}_"
            f"{phantom.shape[0]}x{phantom.shape[1]}x{phantom.shape[2]}.npz"
        )
        phantom.save(generated_phantom_path)
        print_phantom_summary(phantom_label, generated_phantom_path, phantom)
        print(
            f"  Source phantom: {source_path} | full 3D volume retained | "
            f"FLASH slice index: {slice_index} | "
            f"generation/save: {perf_counter() - phantom_start:.3f} s",
            flush=True,
        )

        generation_start = perf_counter()
        sequence = build_resolution_flash_sequence(
            source, _sequence_arguments(args, dynamic), slice_index
        )
        sequence_path = sequence_dir / f"kernel_flash_{phantom_label}.seq"
        program = write_and_load_sequence(sequence, sequence_path)
        generation_time_s = perf_counter() - generation_start
        print_sequence_summary(
            f"kernel_flash_{family}", sequence_path, program, generation_time_s
        )
        records.extend(
            _run_family(
                args=args,
                family=family,
                kernels=kernels,
                phantom_label=phantom_label,
                phantom_path=generated_phantom_path,
                phantom=phantom,
                program=program,
                generation_time_s=generation_time_s,
            )
        )

    missing = {"sequence", "dynamic"} - found_families
    if missing:
        print(
            "\nNote: no " + " or ".join(sorted(missing)) + " phantom was selected; "
            "that kernel family was not benchmarked.",
            flush=True,
        )
    return records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = make_output_directory(args.output_dir)
    records = run(args, output_dir=output_dir)
    csv_path, json_path = write_records(output_dir, "kernel_benchmarks", records)
    print_saved_results(csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
