#!/usr/bin/env python3
"""Run the representative sequence-class time-step convergence suite."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from blochsimulator.sequence import (
    ConvergenceCriteria,
    make_default_sequence_convergence_cases,
    run_sequence_convergence_suite,
)


def _timestep_us(value: str):
    if value.lower() == "native":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "time steps must be 'native' or positive numbers in microseconds"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("time steps must be positive")
    return parsed * 1e-6


def _write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare representative MRI sequence classes with the independent "
            "matrix-exponential Bloch reference."
        )
    )
    parser.add_argument(
        "--rf-raster-us",
        type=float,
        default=1.0,
        help="Native RF raster used to construct the test motifs (default: 1 us).",
    )
    parser.add_argument(
        "--timesteps-us",
        nargs="+",
        type=_timestep_us,
        default=[None, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6],
        metavar="STEP",
        help="Candidates in us; use 'native' for uncoarsened event rasters.",
    )
    parser.add_argument("--max-vector-error", type=float, default=1e-3)
    parser.add_argument("--rms-vector-error", type=float, default=2e-4)
    parser.add_argument("--max-rf-checkpoints", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for detailed per-case records.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional path for one limiting-case record per time step.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.rf_raster_us <= 0:
        raise SystemExit("--rf-raster-us must be positive")
    criteria = ConvergenceCriteria(
        max_vector_error=args.max_vector_error,
        rms_vector_error=args.rms_vector_error,
    )
    cases = make_default_sequence_convergence_cases(
        rf_raster_s=args.rf_raster_us * 1e-6
    )
    result = run_sequence_convergence_suite(
        cases,
        timesteps_s=args.timesteps_us,
        criteria=criteria,
        max_rf_checkpoints=args.max_rf_checkpoints,
    )
    summary = result.summary_records()

    print("timestep | all pass | limiting case | max vector error | largest RMS error")
    for record in summary:
        print(
            f"{record['timestep']:>8} | "
            f"{str(record['all_passed']):>8} | "
            f"{record['limiting_case']:<35} | "
            f"{record['max_vector_error']:.6g} | "
            f"{record['largest_rms_vector_error']:.6g}"
        )
    recommendation = result.coarsest_passing_timestep_s
    if recommendation is None:
        print("No numeric candidate passed every sequence class.")
    else:
        print(
            "Coarsest candidate passing every sequence class: "
            f"{recommendation * 1e6:g} us"
        )

    if args.output_csv is not None:
        _write_csv(args.output_csv, result.to_records())
    if args.summary_csv is not None:
        _write_csv(args.summary_csv, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
