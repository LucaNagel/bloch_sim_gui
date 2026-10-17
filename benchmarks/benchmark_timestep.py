#!/usr/bin/env python3
"""Benchmark how the RF simulation time step trades accuracy for runtime.

Goal
----
Measure how much accuracy is lost, and how much runtime is saved, when RF
pulses are integrated with a coarser simulation time step (for example 1, 5,
10, or 20 us), and show how strongly this depends on the RF pulse and sequence
type. Every candidate is compared with the same simulation at the reference
time step (default 1 us, the native RF raster of the generated sequences).

Only RF-active intervals are subdivided by the time step. Free precession,
gradients outside RF pulses, and ADC sampling are integrated exactly at every
time step, so the error measured here is the RF integration error alone.

Typical use
-----------
Run from the repository root::

    python -m benchmarks.benchmark_timestep
    python -m benchmarks.benchmark_timestep --timesteps-us 1 5 10 \\
        --timestep-sequences flash_sinc ss_bssfp
    python -m benchmarks.benchmark_timestep --sequence-files my_sequence.seq \\
        --timestep-phantom-files my_phantom.npz

Without explicit files the benchmark is self-contained: it generates six
sequences with different RF pulses (block, sinc, spectral Gaussian, and a
lactate-selective spectral-spatial pulse) and simulates them on an analytic
two-container pyruvate/lactate phantom, in a static and a dynamic (kPL and
inflow) variant. ``--timestep-us`` and ``--phantoms`` of the shared options
are not used here; use ``--timesteps-us`` and ``--timestep-phantoms``.

Reading the output
------------------
``signal_relative_l2_error`` is the normalized RMS difference of the complete
received signal from the reference (0.01 = 1 %). For fully sampled Cartesian
acquisitions it equals the image-space error. ``pool_<name>_error_vs_signal``
is a metabolite's own signal error relative to the total reference signal, so
a strongly suppressed metabolite does not produce misleadingly large relative
values. ``speedup_vs_reference`` above one means the time step ran faster
than the reference. Runtimes are the median over ``--repeats``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from blochsimulator.sequence import (
    design_metabolite_selective_pulse,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_flash,
    make_pulseq_spectral_selective_bssfp,
)

try:
    from .common import (
        _metabolite_frequency_hz,
        _slug,
        add_common_arguments,
        array_similarity,
        base_result_record,
        load_phantom,
        make_output_directory,
        make_simulator,
        parse_positive_float,
        print_phantom_summary,
        print_saved_results,
        simulate_program,
        write_records,
    )
    from .synthetic_phantoms import (
        SYNTHETIC_PHANTOM_KINDS,
        make_synthetic_phantom,
    )
except ImportError:  # Allow: python benchmarks/benchmark_timestep.py
    from common import (  # type: ignore
        _metabolite_frequency_hz,
        _slug,
        add_common_arguments,
        array_similarity,
        base_result_record,
        load_phantom,
        make_output_directory,
        make_simulator,
        parse_positive_float,
        print_phantom_summary,
        print_saved_results,
        simulate_program,
        write_records,
    )
    from synthetic_phantoms import (  # type: ignore
        SYNTHETIC_PHANTOM_KINDS,
        make_synthetic_phantom,
    )


DEFAULT_TIMESTEPS_US = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DEFAULT_REFERENCE_TIMESTEP_US = 1.0
DEFAULT_TIMESTEP_PHANTOM_MATRIX = (16, 16, 32)
SLICE_THICKNESS_M = 8e-3
TIMESTEP_SEQUENCES = {
    "flash_block": "2D FLASH, 0.5 ms block (hard) pulse",
    "flash_sinc": "2D FLASH, 2 ms slice-selective sinc pulse (TBW 4)",
    "flash_spsp": "2D FLASH, lactate-selective spectral-spatial pulse",
    "bssfp": "3D bSSFP, 1 ms block pulse",
    "ss_bssfp": "3D spectrally selective bSSFP, Gaussian lactate/pyruvate pulses",
    "csi": "2D FID-CSI, 3 ms slice-selective sinc pulse",
}


def add_timestep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timesteps-us",
        nargs="+",
        type=parse_positive_float,
        default=DEFAULT_TIMESTEPS_US,
        metavar="US",
        help="RF time steps to compare in microseconds (default: 1 2 5 10 20 50).",
    )
    parser.add_argument(
        "--reference-timestep-us",
        type=parse_positive_float,
        default=DEFAULT_REFERENCE_TIMESTEP_US,
        metavar="US",
        help="Reference RF time step in microseconds (default: 1).",
    )
    parser.add_argument(
        "--timestep-sequences",
        nargs="+",
        choices=tuple(TIMESTEP_SEQUENCES),
        default=tuple(TIMESTEP_SEQUENCES),
        metavar="SEQUENCE",
        help=(
            "Generated sequences to compare (default: all of "
            + ", ".join(TIMESTEP_SEQUENCES)
            + ")."
        ),
    )
    parser.add_argument(
        "--sequence-files",
        nargs="+",
        type=Path,
        default=(),
        metavar="SEQ",
        help=(
            "Additional Pulseq files to compare. Combine with "
            "--only-sequence-files to skip the generated sequences."
        ),
    )
    parser.add_argument(
        "--only-sequence-files",
        action="store_true",
        help="Skip the generated sequences and compare only --sequence-files.",
    )
    parser.add_argument(
        "--timestep-phantoms",
        nargs="+",
        choices=SYNTHETIC_PHANTOM_KINDS,
        default=SYNTHETIC_PHANTOM_KINDS,
        metavar="KIND",
        help="Analytic phantom variants: static, dynamic (default: both).",
    )
    parser.add_argument(
        "--timestep-phantom-matrix",
        nargs=3,
        type=int,
        default=DEFAULT_TIMESTEP_PHANTOM_MATRIX,
        metavar=("X", "Y", "Z"),
        help="Grid of the analytic phantom (default: 16 16 32).",
    )
    parser.add_argument(
        "--timestep-phantom-files",
        nargs="+",
        type=Path,
        default=(),
        metavar="NPZ",
        help="Phantom files to use instead of the analytic phantom.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_timestep_arguments(parser)
    # The GUI default; the NumPy kernel would dominate the runtime comparison.
    parser.set_defaults(dynamic_kernel="native_parallel")
    return parser


def _lactate_spsp_pulse(phantom):
    peaks = [
        {
            "name": name,
            "absolute_ppm": float(phantom.spectral_reference_ppm),
            "sequence_offset_hz": offset,
            "carrier_offset_hz": offset,
        }
        for name, offset in (
            ("Pyruvate", _metabolite_frequency_hz(phantom, "pyruvate")),
            ("Lactate", _metabolite_frequency_hz(phantom, "lactate")),
        )
    ]
    design = design_metabolite_selective_pulse(
        "Lactate",
        peaks,
        flip_angle_deg=20.0,
        slice_thickness_m=SLICE_THICKNESS_M,
        # Default scanner limits; 0.7 ms subpulses would exceed 32 mT/m here.
        subpulse_duration_s=1.2e-3,
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        nucleus=str(getattr(phantom, "nucleus", "C13")),
    )
    return design.pulse


def build_timestep_sequence(name: str, phantom):
    """Build one generated sequence of the time-step comparison."""
    fov_xyz = tuple(float(value) for value in phantom.fov)
    nucleus = str(getattr(phantom, "nucleus", "C13"))
    flash = dict(
        fov_m=fov_xyz[:2],
        matrix=(16, 16),
        sampling_bandwidth_hz=20_000.0,
        flip_angle_deg=10.0,
        slice_thickness_m=SLICE_THICKNESS_M,
        spoiler_cycles_per_slice=1.0,
        nucleus=nucleus,
    )
    if name == "flash_block":
        return make_pulseq_flash(rf_pulse_type="block", rf_duration_s=0.5e-3, **flash)
    if name == "flash_sinc":
        return make_pulseq_flash(
            rf_pulse_type="sinc",
            rf_duration_s=2e-3,
            rf_time_bandwidth_product=4.0,
            **flash,
        )
    if name == "flash_spsp":
        pulse = _lactate_spsp_pulse(phantom)
        return make_pulseq_flash(
            rf_pulse_type="spectral_spatial",
            rf_spectral_spatial_pulse=pulse.to_dict(),
            echo_time_s=0.03,
            repetition_time_s=0.05,
            **{**flash, "flip_angle_deg": 20.0},
        )
    if name == "bssfp":
        return make_pulseq_bssfp(
            fov_m=fov_xyz,
            matrix=(16, 8, 6),
            flip_angle_deg=20.0,
            repetitions=1,
            dummy_repetitions=0,
            nucleus=nucleus,
        )
    if name == "ss_bssfp":
        lactate_hz = _metabolite_frequency_hz(phantom, "lactate")
        pyruvate_hz = _metabolite_frequency_hz(phantom, "pyruvate")
        return make_pulseq_spectral_selective_bssfp(
            fov_m=fov_xyz,
            matrix=(16, 8, 6),
            target_frequency_offsets_hz=(lactate_hz, pyruvate_hz),
            receiver_frequency_offsets_hz=(lactate_hz, pyruvate_hz),
            target_metabolite_names=("Lactate", "Pyruvate"),
            dummy_repetitions=0,
            use_alpha_half=False,
            repetitions=1,
            field_strength_t=float(getattr(phantom, "field_strength", 7.0)),
            nucleus=nucleus,
        )
    if name == "csi":
        return make_pulseq_csi(
            fov_m=fov_xyz[:2],
            matrix=(8, 8),
            slice_thickness_m=SLICE_THICKNESS_M,
            spectral_bandwidth_hz=4_000.0,
            spectral_points=64,
            repetition_time_s=0.05,
            nucleus=nucleus,
        )
    raise ValueError(f"unknown time-step sequence {name!r}")


def _phantom_cases(args) -> list[tuple[str, str, object]]:
    if args.timestep_phantom_files:
        cases = []
        for path in args.timestep_phantom_files:
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"phantom not found: {path}")
            cases.append((_slug(path.stem), str(path), load_phantom(path)))
        return cases
    shape = tuple(int(value) for value in args.timestep_phantom_matrix)
    return [
        (
            f"synthetic_{kind}",
            f"synthetic:{kind}",
            make_synthetic_phantom(kind, shape),
        )
        for kind in args.timestep_phantoms
    ]


def _sequence_cases(args, phantom, sequence_dir: Path, phantom_label: str):
    """Yield (name, description, program, setup seconds) for every sequence."""
    names = () if args.only_sequence_files else tuple(args.timestep_sequences)
    for name in names:
        start = perf_counter()
        try:
            sequence = build_timestep_sequence(name, phantom)
        except ValueError as exc:
            print(f"\n  Skipping {name}: {exc}", flush=True)
            continue
        path = sequence_dir / f"{phantom_label}_{name}.seq"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Not write_and_load_sequence: its SS-bSSFP read-axis check is
        # specific to the sequence benchmark.
        sequence.write(str(path), v141_compat=True)
        program = load_pulseq(path)
        yield name, TIMESTEP_SEQUENCES[name], program, perf_counter() - start
    for path in args.sequence_files:
        start = perf_counter()
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"sequence not found: {path}")
        program = load_pulseq(path)
        yield _slug(path.stem), f"Pulseq file {path.name}", program, (
            perf_counter() - start
        )


def _timesteps(args) -> list[float]:
    values = {float(args.reference_timestep_us)}
    values.update(float(value) for value in args.timesteps_us)
    return sorted(values)


def _pool_metrics(reference, candidate) -> dict:
    reference_species = getattr(reference, "species_signal", None)
    candidate_species = getattr(candidate, "species_signal", None)
    if reference_species is None or candidate_species is None:
        return {}
    total = float(np.linalg.norm(reference.signal))
    metrics = {}
    worst = 0.0
    for index, name in enumerate(getattr(reference, "pool_names", ())):
        slug = _slug(str(name))
        difference = float(
            np.linalg.norm(candidate_species[index] - reference_species[index])
        )
        own = float(np.linalg.norm(reference_species[index]))
        versus_total = difference / total if total > 0.0 else difference
        metrics[f"pool_{slug}_error_vs_signal"] = versus_total
        metrics[f"pool_{slug}_relative_l2_error"] = (
            difference / own if own > 0.0 else difference
        )
        metrics[f"pool_{slug}_signal_fraction"] = own / total if total > 0.0 else 0.0
        worst = max(worst, versus_total)
    metrics["max_pool_error_vs_signal"] = worst
    return metrics


def _print_header(args, output_dir: Path, timesteps) -> None:
    threads = "automatic" if args.threads == 0 else str(args.threads)
    print("\n" + "=" * 78, flush=True)
    print("Benchmark: RF simulation time step", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(
        "  Time steps: "
        + ", ".join(f"{value:g}" for value in timesteps)
        + f" us | reference: {args.reference_timestep_us:g} us",
        flush=True,
    )
    print(
        f"  Repeats: {args.repeats} (median runtime) | threads: {threads} | "
        f"kernels: sequence={args.sequence_kernel}, dynamic={args.dynamic_kernel}",
        flush=True,
    )
    print("=" * 78, flush=True)


def _print_row(record) -> None:
    pool = record.get("max_pool_error_vs_signal")
    pool_text = "" if pool is None else f" | worst metabolite {pool:.2e}"
    marker = " (reference)" if record["is_reference"] else ""
    print(
        f"    {record['timestep_us']:>6g} us{marker}: "
        f"{record['simulation_time_s']:.3f} s | "
        f"x{record['speedup_vs_reference']:.2f} | "
        f"intervals {record['compiled_intervals']:,} | "
        f"signal error {record['signal_relative_l2_error']:.2e}{pool_text}",
        flush=True,
    )


def run(args, output_dir=None) -> list[dict]:
    output_dir = make_output_directory(output_dir or args.output_dir)
    sequence_dir = output_dir / "timestep_sequences"
    timesteps = _timesteps(args)
    reference_us = float(args.reference_timestep_us)
    _print_header(args, output_dir, timesteps)
    simulator = make_simulator(args)
    records = []
    for phantom_label, phantom_source, phantom in _phantom_cases(args):
        print_phantom_summary(phantom_label, Path(phantom_source), phantom)
        for name, description, program, setup_s in _sequence_cases(
            args, phantom, sequence_dir, phantom_label
        ):
            print(
                f"\n  {name}: {description} | duration "
                f"{program.duration_s * 1e3:.1f} ms | RF events "
                f"{len(program.rf_events)}",
                flush=True,
            )
            reference = None
            reference_time = None
            for timestep_us in sorted(timesteps, key=lambda v: v != reference_us):
                durations = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    result = simulate_program(
                        simulator,
                        program,
                        phantom,
                        timestep_s=timestep_us * 1e-6,
                        spoiler_mode="ideal",
                        spin_sampling=(1, 1, 1),
                        signal_weighting="voxel_volume",
                    )
                    durations.append(perf_counter() - start)
                runtime = float(np.median(durations))
                if reference is None:
                    reference, reference_time = result, runtime
                record = base_result_record(
                    benchmark="timestep",
                    sequence_name=name,
                    phantom_label=phantom_label,
                    phantom_path=Path(phantom_source),
                    phantom=phantom,
                    program=program,
                    result=result,
                    repeat=args.repeats,
                    generation_time_s=setup_s,
                    simulation_time_s=runtime,
                )
                record.pop("repeat")
                record.update(
                    {
                        "sequence_description": description,
                        "timestep_us": timestep_us,
                        "reference_timestep_us": reference_us,
                        "is_reference": timestep_us == reference_us,
                        "repeats": args.repeats,
                        "simulation_time_min_s": float(np.min(durations)),
                        "speedup_vs_reference": (
                            reference_time / runtime if runtime > 0 else float("nan")
                        ),
                        "sequence_kernel": args.sequence_kernel,
                        "dynamic_kernel": args.dynamic_kernel,
                    }
                )
                record.update(
                    array_similarity(reference.signal, result.signal, prefix="signal")
                )
                record.update(_pool_metrics(reference, result))
                records.append(record)
                _print_row(record)
    if not records:
        raise ValueError("no time-step case could be simulated")
    return records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = make_output_directory(args.output_dir)
    records = run(args, output_dir=output_dir)
    csv_path, json_path = write_records(output_dir, "timestep_benchmarks", records)
    print_saved_results(csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
