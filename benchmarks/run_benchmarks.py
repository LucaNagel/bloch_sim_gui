#!/usr/bin/env python3
"""Run selectable Bloch-simulator benchmark groups.

Goal
----
Provide one reproducible entry point for measuring sequence runtime, crusher
convergence, phantom-resolution convergence, and simulation-kernel performance.
The runner can execute one group or the complete suite and writes each group's
raw records to CSV and JSON.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.run_benchmarks --list
    python -m benchmarks.run_benchmarks --benchmark resolution
    python -m benchmarks.run_benchmarks --benchmark all --repeats 3

The default ``quick`` profile is suitable for a first functional run. Select
``--profile full`` or larger explicit matrices when the goal is to reproduce a
long production workload. Use ``--output-dir PATH`` when result files must have
a stable location.

Reading the output
------------------
``simulation_time_s`` measures the simulator call; sequence setup is recorded
separately. Accuracy fields use the relevant reference for their benchmark:
ideal crushing, the finest requested phantom grid, or the reference kernel.
Values close to one indicate high similarity; relative L2 errors close to zero
indicate close agreement. Always check requested versus actual kernel and any
fallback reason before interpreting accelerator timings.
"""

from __future__ import annotations

import argparse

try:
    from .benchmark_crushers import add_crusher_arguments, run as run_crushers
    from .benchmark_kernels import add_kernel_arguments, run as run_kernels
    from .benchmark_resolution import add_resolution_arguments, run as run_resolution
    from .benchmark_sequences import add_sequence_arguments, run as run_sequences
    from .common import (
        SEQUENCE_NAMES,
        add_common_arguments,
        make_output_directory,
        print_saved_results,
        write_records,
    )
except ImportError:  # Allow: python benchmarks/run_benchmarks.py
    from benchmark_crushers import add_crusher_arguments, run as run_crushers  # type: ignore
    from benchmark_kernels import add_kernel_arguments, run as run_kernels  # type: ignore
    from benchmark_resolution import (  # type: ignore
        add_resolution_arguments,
        run as run_resolution,
    )
    from benchmark_sequences import add_sequence_arguments, run as run_sequences  # type: ignore
    from common import (  # type: ignore
        SEQUENCE_NAMES,
        add_common_arguments,
        make_output_directory,
        print_saved_results,
        write_records,
    )


BENCHMARK_NAMES = ("sequences", "crushers", "resolution", "kernels", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_NAMES,
        default="all",
        help="Benchmark group to run (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List benchmark groups and generated sequence names, then exit.",
    )
    add_common_arguments(parser)
    add_sequence_arguments(parser)
    add_crusher_arguments(parser)
    add_resolution_arguments(parser)
    add_kernel_arguments(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print("Benchmark groups: sequences, crushers, resolution, kernels, all")
        print("Sequences: " + ", ".join(SEQUENCE_NAMES))
        return 0

    output_dir = make_output_directory(args.output_dir)
    if args.benchmark in {"sequences", "all"}:
        sequence_records = run_sequences(args, output_dir=output_dir)
        print_saved_results(
            *write_records(output_dir, "sequence_benchmarks", sequence_records)
        )
    if args.benchmark in {"crushers", "all"}:
        crusher_records = run_crushers(args, output_dir=output_dir)
        print_saved_results(
            *write_records(output_dir, "crusher_benchmarks", crusher_records)
        )
    if args.benchmark in {"resolution", "all"}:
        resolution_records = run_resolution(args, output_dir=output_dir)
        print_saved_results(
            *write_records(output_dir, "resolution_benchmarks", resolution_records)
        )
    if args.benchmark in {"kernels", "all"}:
        kernel_records = run_kernels(args, output_dir=output_dir)
        print_saved_results(
            *write_records(output_dir, "kernel_benchmarks", kernel_records)
        )
    print(f"\nAll selected benchmarks completed. Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
