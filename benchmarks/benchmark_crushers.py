#!/usr/bin/env python3
"""Compare SS-bSSFP gradient-waveform crushers with the ideal crusher.

Goal
----
Determine how many subvoxel spins are needed for a physical crusher-gradient
waveform to reproduce the signal loss of the simulator's ideal crusher. The
SS-bSSFP sequence contains two acquired volumes so the first end-of-volume
crusher can influence the second volume. The ideal result is the reference for
each phantom, crusher strength, and repeat.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.benchmark_crushers
    python -m benchmarks.benchmark_crushers --gradient-z-spins 1 3 5 9
    python -m benchmarks.benchmark_crushers --crusher-cycles 0.5 1 2 \\
        --phantoms static dynamic --repeats 3

The physical waveform uses a ``1 x 1 x N`` midpoint-spin grid. ``N=1`` is an
intentionally unresolved baseline; increasing N raises runtime approximately
with the number of simulated spins.

Reading the output
------------------
Similarity and complex correlation near one, together with relative L2 error
near zero, indicate convergence to the ideal crusher. Prefer the post-crusher
signal metrics when assessing crusher behavior because samples acquired before
the first crusher should be identical. ``runtime_vs_ideal`` shows the cost of
the physical model relative to the ideal operation.
"""

from __future__ import annotations

import argparse
from time import perf_counter

try:
    from .common import (
        add_common_arguments,
        base_result_record,
        build_sequence,
        ideal_similarity_record,
        load_phantom,
        make_output_directory,
        make_simulator,
        print_benchmark_header,
        print_phantom_summary,
        print_runtime_row,
        print_saved_results,
        print_sequence_summary,
        print_simulation_start,
        resolve_phantom_paths,
        result_similarity,
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )
except ImportError:  # Allow: python benchmarks/benchmark_crushers.py
    from common import (  # type: ignore
        add_common_arguments,
        base_result_record,
        build_sequence,
        ideal_similarity_record,
        load_phantom,
        make_output_directory,
        make_simulator,
        print_benchmark_header,
        print_phantom_summary,
        print_runtime_row,
        print_saved_results,
        print_sequence_summary,
        print_simulation_start,
        resolve_phantom_paths,
        result_similarity,
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )


def add_crusher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--crusher-cycles",
        nargs="+",
        type=float,
        default=[1.0],
        metavar="CYCLES",
        help="Physical crusher cycles per phantom voxel (default: 1).",
    )
    parser.add_argument(
        "--gradient-z-spins",
        nargs="+",
        type=int,
        default=[1, 5, 9],
        metavar="COUNT",
        help=(
            "Z midpoint-spin counts for gradient-waveform runs; X/Y remain one "
            "(default: 1 5 9)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_crusher_arguments(parser)
    return parser


def _validate_args(args) -> None:
    if any(value < 0 for value in args.crusher_cycles):
        raise ValueError("--crusher-cycles values must be non-negative")
    if any(value <= 0 for value in args.gradient_z_spins):
        raise ValueError("--gradient-z-spins values must be positive")


def run(args, output_dir=None) -> list[dict]:
    _validate_args(args)
    output_dir = make_output_directory(
        args.output_dir if output_dir is None else output_dir
    )
    sequence_dir = output_dir / "generated_sequences"
    simulator = make_simulator(args)
    timestep_s = args.timestep_us * 1e-6
    records = []
    print_benchmark_header(
        "Crusher comparison",
        args,
        output_dir,
        selection={
            "Sequence": "ss_bssfp",
            "Phantoms": args.phantoms,
            "Crusher cycles/voxel": args.crusher_cycles,
            "Gradient Z spins": args.gradient_z_spins,
        },
    )

    for phantom_label, phantom_path in resolve_phantom_paths(args.phantoms):
        phantom = load_phantom(phantom_path)
        print_phantom_summary(phantom_label, phantom_path, phantom)
        for crusher_cycles in args.crusher_cycles:
            generation_start = perf_counter()
            sequence = build_sequence(
                "ss_bssfp",
                phantom,
                profile=args.profile,
                crusher_cycles_per_voxel=crusher_cycles,
            )
            cycles_label = str(crusher_cycles).replace(".", "p")
            sequence_path = (
                sequence_dir / f"ss_bssfp_{phantom_label}_{cycles_label}cycles.seq"
            )
            program = write_and_load_sequence(sequence, sequence_path)
            generation_time_s = perf_counter() - generation_start
            print_sequence_summary(
                "ss_bssfp",
                sequence_path,
                program,
                generation_time_s,
            )

            for repeat in range(1, args.repeats + 1):
                print_simulation_start(
                    repeat=repeat,
                    repeats=args.repeats,
                    spoiler_mode="ideal",
                    spin_sampling=(1, 1, 1),
                    timestep_us=args.timestep_us,
                    details={
                        "phantom": phantom_label,
                        "crusher cycles/voxel": crusher_cycles,
                        "reference": "ideal",
                    },
                )
                ideal, ideal_time_s = timed_simulation(
                    simulator,
                    program,
                    phantom,
                    timestep_s=timestep_s,
                    spoiler_mode="ideal",
                    spin_sampling=(1, 1, 1),
                )
                ideal_record = base_result_record(
                    benchmark="crushers",
                    sequence_name="ss_bssfp",
                    phantom_label=phantom_label,
                    phantom_path=phantom_path,
                    phantom=phantom,
                    program=program,
                    result=ideal,
                    repeat=repeat,
                    generation_time_s=generation_time_s,
                    simulation_time_s=ideal_time_s,
                )
                ideal_record.update(ideal_similarity_record(ideal))
                ideal_record.update(
                    {
                        "profile": args.profile,
                        "timestep_us": args.timestep_us,
                        "sequence_kernel": args.sequence_kernel,
                        "dynamic_kernel": args.dynamic_kernel,
                        "requested_threads": args.threads,
                        "crusher": "ideal",
                        "crusher_cycles_per_voxel": crusher_cycles,
                        "gradient_z_spins": 1,
                        "runtime_vs_ideal": 1.0,
                    }
                )
                records.append(ideal_record)
                print_runtime_row(ideal_record)

                for z_spins in args.gradient_z_spins:
                    print_simulation_start(
                        repeat=repeat,
                        repeats=args.repeats,
                        spoiler_mode="gradient",
                        spin_sampling=(1, 1, z_spins),
                        timestep_us=args.timestep_us,
                        details={
                            "phantom": phantom_label,
                            "crusher cycles/voxel": crusher_cycles,
                            "comparison": "against ideal",
                        },
                    )
                    gradient, gradient_time_s = timed_simulation(
                        simulator,
                        program,
                        phantom,
                        timestep_s=timestep_s,
                        spoiler_mode="gradient",
                        spin_sampling=(1, 1, z_spins),
                    )
                    record = base_result_record(
                        benchmark="crushers",
                        sequence_name="ss_bssfp",
                        phantom_label=phantom_label,
                        phantom_path=phantom_path,
                        phantom=phantom,
                        program=program,
                        result=gradient,
                        repeat=repeat,
                        generation_time_s=generation_time_s,
                        simulation_time_s=gradient_time_s,
                    )
                    record.update(result_similarity(ideal, gradient))
                    record.update(
                        {
                            "profile": args.profile,
                            "timestep_us": args.timestep_us,
                            "sequence_kernel": args.sequence_kernel,
                            "dynamic_kernel": args.dynamic_kernel,
                            "requested_threads": args.threads,
                            "crusher": f"gradient_z{z_spins}",
                            "crusher_cycles_per_voxel": crusher_cycles,
                            "gradient_z_spins": z_spins,
                            "runtime_vs_ideal": (
                                gradient_time_s / ideal_time_s
                                if ideal_time_s > 0
                                else float("inf")
                            ),
                        }
                    )
                    records.append(record)
                    print_runtime_row(record)
    return records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = make_output_directory(args.output_dir)
    records = run(args, output_dir=output_dir)
    csv_path, json_path = write_records(output_dir, "crusher_benchmarks", records)
    print_saved_results(csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
