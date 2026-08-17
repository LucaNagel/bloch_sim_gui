#!/usr/bin/env python3
"""Benchmark representative generated sequences on selected phantoms.

Goal
----
Measure end-to-end simulator runtime for the generated SS-bSSFP, bSSFP,
ME-bSSFP, FLASH, and CSI families on the selected static and dynamic phantoms.
This benchmark answers which sequence features and workloads dominate runtime;
it is not an image-quality convergence test. Sequence construction/loading time
and Bloch simulation time are reported separately.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.benchmark_sequences
    python -m benchmarks.benchmark_sequences --sequences ss_bssfp flash
    python -m benchmarks.benchmark_sequences --phantoms dynamic --repeats 3

Use ``--profile quick`` for a short smoke/performance run and ``--profile full``
for larger acquisition matrices. ``--spoiler-mode gradient`` enables physical
gradient-waveform crushing; increase ``--sequence-gradient-z-spins`` to resolve
intravoxel dephasing. SS-bSSFP always uses physical scanner Z as its read axis.

Reading the output
------------------
Compare ``simulation_time_s`` only between rows with the same phantom, sequence
profile, time step, crusher model, subvoxel grid, thread count, and kernel.
``generation_time_s`` covers sequence generation and loading and is not part of
the measured simulator call.
"""

from __future__ import annotations

import argparse
from time import perf_counter

try:
    from .common import (
        SEQUENCE_NAMES,
        add_common_arguments,
        base_result_record,
        build_sequence,
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
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )
except ImportError:  # Allow: python benchmarks/benchmark_sequences.py
    from common import (  # type: ignore
        SEQUENCE_NAMES,
        add_common_arguments,
        base_result_record,
        build_sequence,
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
        timed_simulation,
        write_and_load_sequence,
        write_records,
    )


def add_sequence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=SEQUENCE_NAMES,
        default=list(SEQUENCE_NAMES),
        help="Generated sequence families to benchmark (default: all).",
    )
    parser.add_argument(
        "--spoiler-mode",
        choices=("ideal", "gradient"),
        default="ideal",
        help="Crusher model used for the sequence runtime matrix (default: ideal).",
    )
    parser.add_argument(
        "--sequence-gradient-z-spins",
        type=int,
        default=5,
        help=(
            "Z subvoxel spins when --spoiler-mode=gradient; X/Y remain one "
            "(default: 5)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_sequence_arguments(parser)
    return parser


def run(args, output_dir=None) -> list[dict]:
    if args.sequence_gradient_z_spins <= 0:
        raise ValueError("--sequence-gradient-z-spins must be positive")
    output_dir = make_output_directory(
        args.output_dir if output_dir is None else output_dir
    )
    sequence_dir = output_dir / "generated_sequences"
    simulator = make_simulator(args)
    records = []
    timestep_s = args.timestep_us * 1e-6
    spin_sampling = (
        (1, 1, args.sequence_gradient_z_spins)
        if args.spoiler_mode == "gradient"
        else (1, 1, 1)
    )
    print_benchmark_header(
        "Sequence comparison",
        args,
        output_dir,
        selection={
            "Sequences": args.sequences,
            "Phantoms": args.phantoms,
            "Crusher model": args.spoiler_mode,
            "Subvoxel spins": spin_sampling,
        },
    )

    for phantom_label, phantom_path in resolve_phantom_paths(args.phantoms):
        phantom = load_phantom(phantom_path)
        print_phantom_summary(phantom_label, phantom_path, phantom)
        for sequence_name in args.sequences:
            generation_start = perf_counter()
            sequence = build_sequence(
                sequence_name,
                phantom,
                profile=args.profile,
            )
            sequence_path = sequence_dir / f"{sequence_name}_{phantom_label}.seq"
            program = write_and_load_sequence(sequence, sequence_path)
            generation_time_s = perf_counter() - generation_start
            print_sequence_summary(
                sequence_name,
                sequence_path,
                program,
                generation_time_s,
            )

            for repeat in range(1, args.repeats + 1):
                print_simulation_start(
                    repeat=repeat,
                    repeats=args.repeats,
                    spoiler_mode=args.spoiler_mode,
                    spin_sampling=spin_sampling,
                    timestep_us=args.timestep_us,
                    details={"phantom": phantom_label, "sequence": sequence_name},
                )
                result, simulation_time_s = timed_simulation(
                    simulator,
                    program,
                    phantom,
                    timestep_s=timestep_s,
                    spoiler_mode=args.spoiler_mode,
                    spin_sampling=spin_sampling,
                )
                record = base_result_record(
                    benchmark="sequences",
                    sequence_name=sequence_name,
                    phantom_label=phantom_label,
                    phantom_path=phantom_path,
                    phantom=phantom,
                    program=program,
                    result=result,
                    repeat=repeat,
                    generation_time_s=generation_time_s,
                    simulation_time_s=simulation_time_s,
                )
                record["profile"] = args.profile
                record["timestep_us"] = args.timestep_us
                record["sequence_kernel"] = args.sequence_kernel
                record["dynamic_kernel"] = args.dynamic_kernel
                record["requested_threads"] = args.threads
                records.append(record)
                print_runtime_row(record)
    return records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = make_output_directory(args.output_dir)
    records = run(args, output_dir=output_dir)
    csv_path, json_path = write_records(output_dir, "sequence_benchmarks", records)
    print_saved_results(csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
