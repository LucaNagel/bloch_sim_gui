"""Shared infrastructure for the selectable Bloch-simulator benchmarks.

This module centralizes phantom aliases, common command-line options, generated
sequence profiles, simulation dispatch, timing, similarity metrics, progress
messages, and CSV/JSON serialization. It is an implementation module rather
than a standalone benchmark; use ``python -m benchmarks.run_benchmarks`` or one
of the ``benchmark_*`` modules as the command-line entry point.

Timing deliberately wraps only the simulator call. Sequence construction and
loading are measured separately. Similarity is reported as
``1 / (1 + relative_l2_error)`` so identical arrays score one, while relative
L2 error itself is also retained for unambiguous quantitative interpretation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_flash,
    make_pulseq_me_bssfp,
    make_pulseq_spectral_selective_bssfp,
)
from blochsimulator.spectral_phantom import SpectralPhantom


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHANTOMS = {
    "static": REPOSITORY_ROOT / "phantoms" / "Pyruvate Lactate 2 containers static.npz",
    "dynamic": REPOSITORY_ROOT / "phantoms" / "Pyruvate to lactate with inflow 2.npz",
}
SEQUENCE_NAMES = ("ss_bssfp", "bssfp", "me_bssfp", "flash", "csi")
PROFILE_NAMES = ("quick", "full")
SS_BSSFP_ENCODING_AXES = ("+z", "+x", "+y")


def parse_positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phantoms",
        nargs="+",
        default=["static", "dynamic"],
        metavar="PHANTOM",
        help=(
            "Phantom aliases ('static', 'dynamic') or explicit phantom paths "
            "(default: both bundled pyruvate/lactate phantoms)."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="quick",
        help="Sequence size profile (default: quick).",
    )
    parser.add_argument(
        "--timestep-us",
        type=parse_positive_float,
        default=20.0,
        help="Maximum RF-active simulation step in microseconds (default: 20).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU threads; 0 uses the simulator default/all available (default: 0).",
    )
    parser.add_argument(
        "--sequence-kernel",
        choices=("optimized", "reference"),
        default="optimized",
        help=(
            "Ordinary Bloch kernel used outside the dedicated kernel comparison "
            "(default: optimized)."
        ),
    )
    parser.add_argument(
        "--dynamic-kernel",
        choices=(
            "optimized",
            "native_serial",
            "native_parallel",
            "metal_hybrid",
            "reference",
        ),
        default="optimized",
        help=(
            "Dynamic pyruvate/lactate kernel used outside the dedicated kernel "
            "comparison (default: optimized)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=parse_positive_int,
        default=1,
        help="Measured repetitions per case (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Result directory. By default a timestamped directory is created "
            "below exports/benchmarks/."
        ),
    )


def make_output_directory(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPOSITORY_ROOT / "exports" / "benchmarks" / timestamp
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_phantom_paths(values: Sequence[str]) -> list[tuple[str, Path]]:
    resolved = []
    used_labels: set[str] = set()
    for value in values:
        alias = str(value).strip().lower()
        path = DEFAULT_PHANTOMS.get(alias, Path(value).expanduser())
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"phantom not found: {path}")
        base_label = alias if alias in DEFAULT_PHANTOMS else _slug(path.stem)
        label = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)
        resolved.append((label, path))
    return resolved


def load_phantom(path: Path):
    """Load a dynamic, spectral, or conventional phantom without GUI imports."""
    try:
        return DynamicSpectralPhantom.load(path)
    except ValueError:
        try:
            return SpectralPhantom.load(path)
        except ValueError:
            return Phantom.load(path)


def phantom_kind(phantom) -> str:
    if isinstance(phantom, DynamicSpectralPhantom):
        return "dynamic_spectral"
    if isinstance(phantom, SpectralPhantom):
        return "spectral"
    return "conventional"


def physical_voxel_sizes_xyz_m(phantom) -> tuple[float, float, float]:
    affine = np.asarray(phantom.affine_ijk_to_xyz_m, dtype=float)
    sizes = np.sum(np.abs(affine[:3, :3]), axis=1)
    if sizes.shape != (3,) or np.any(sizes <= 0) or not np.all(np.isfinite(sizes)):
        raise ValueError("benchmarks require a finite three-dimensional phantom")
    return tuple(float(value) for value in sizes)


def physical_fov_xyz_m(phantom) -> tuple[float, float, float]:
    affine = np.asarray(phantom.affine_ijk_to_xyz_m, dtype=float)
    shape = np.asarray(phantom.shape, dtype=float)
    fov = np.abs(affine[:3, :3]) @ shape
    if fov.shape != (3,) or np.any(fov <= 0) or not np.all(np.isfinite(fov)):
        raise ValueError("benchmarks require a finite three-dimensional phantom")
    return tuple(float(value) for value in fov)


def _logical_fov_m(phantom, encoding_axes: Sequence[str]) -> tuple[float, ...]:
    fov_xyz = physical_fov_xyz_m(phantom)
    return tuple(fov_xyz["xyz".index(axis[-1].lower())] for axis in encoding_axes)


def _metabolite_frequency_hz(phantom, metabolite: str) -> float:
    components = getattr(phantom, "species", getattr(phantom, "pools", ()))
    normalized_target = metabolite.lower()
    matches = [item for item in components if normalized_target in item.name.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"phantom must contain exactly one {metabolite} component; "
            f"found {[item.name for item in matches]}"
        )
    return float(
        phantom.get_frequency_offset(
            matches[0].name,
            getattr(phantom, "field_strength", None),
            getattr(phantom, "nucleus", None),
        )
    )


def _sequence_profile(name: str, profile: str) -> Mapping[str, object]:
    profiles = {
        "quick": {
            "ss_bssfp": {"matrix": (16, 8, 6), "repetitions": 2},
            "bssfp": {"matrix": (16, 8, 6)},
            "me_bssfp": {"matrix": (16, 8, 6), "echoes": 3},
            "flash": {"matrix": (32, 32), "repetitions": 2},
            "csi": {"matrix": (10, 10), "spectral_points": 128},
        },
        "full": {
            "ss_bssfp": {"matrix": (32, 16, 12), "repetitions": 10},
            "bssfp": {"matrix": (32, 16, 12)},
            "me_bssfp": {"matrix": (32, 16, 12), "echoes": 5},
            "flash": {"matrix": (64, 64), "repetitions": 10},
            "csi": {"matrix": (16, 16), "spectral_points": 256},
        },
    }
    try:
        return profiles[profile][name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported benchmark sequence/profile: {name}/{profile}"
        ) from exc


def build_sequence(
    name: str,
    phantom,
    *,
    profile: str,
    crusher_cycles_per_voxel: float = 1.0,
):
    """Build one small representative sequence for the supplied phantom."""
    settings = dict(_sequence_profile(name, profile))
    voxel_sizes = physical_voxel_sizes_xyz_m(phantom)
    fov_xyz = physical_fov_xyz_m(phantom)

    if name == "ss_bssfp":
        lactate_hz = _metabolite_frequency_hz(phantom, "lactate")
        pyruvate_hz = _metabolite_frequency_hz(phantom, "pyruvate")
        return make_pulseq_spectral_selective_bssfp(
            fov_m=_logical_fov_m(phantom, SS_BSSFP_ENCODING_AXES),
            target_frequency_offsets_hz=(lactate_hz, pyruvate_hz),
            receiver_frequency_offsets_hz=(lactate_hz, pyruvate_hz),
            target_metabolite_names=("Lactate", "Pyruvate"),
            flip_angle_deg=(20.0, 10.0),
            repetition_time_s=8e-3,
            dummy_repetitions=0,
            use_alpha_half=False,
            end_image_spoiler_cycles_per_fov=0.0,
            end_image_spoiler_cycles_per_voxel=crusher_cycles_per_voxel,
            end_image_spoiler_voxel_size_m=voxel_sizes,
            encoding_axes=SS_BSSFP_ENCODING_AXES,
            field_strength_t=float(getattr(phantom, "field_strength", 7.0)),
            nucleus=str(getattr(phantom, "nucleus", "C13")),
            **settings,
        )
    if name == "bssfp":
        return make_pulseq_bssfp(
            fov_m=fov_xyz,
            repetitions=1,
            dummy_repetitions=0,
            use_alpha_half=False,
            **settings,
        )
    if name == "me_bssfp":
        return make_pulseq_me_bssfp(
            fov_m=fov_xyz,
            repetitions=1,
            dummy_repetitions=0,
            use_alpha_half=False,
            field_strength_t=float(getattr(phantom, "field_strength", 7.0)),
            nucleus=str(getattr(phantom, "nucleus", "C13")),
            **settings,
        )
    if name == "flash":
        return make_pulseq_flash(
            fov_m=fov_xyz[:2],
            sampling_bandwidth_hz=20_000.0,
            slice_thickness_m=fov_xyz[2],
            n_slices=1,
            spoiler_cycles_per_slice=0.0,
            spoiler_cycles_per_voxel=crusher_cycles_per_voxel,
            **settings,
        )
    if name == "csi":
        return make_pulseq_csi(
            fov_m=fov_xyz[:2],
            slice_thickness_m=fov_xyz[2],
            repetitions=1,
            spoil_after_readout=True,
            spoiler_cycles_per_slice=0.0,
            spoiler_cycles_per_voxel=crusher_cycles_per_voxel,
            **settings,
        )
    raise ValueError(f"unknown sequence {name!r}")


def write_and_load_sequence(sequence, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    definitions = dict(program.metadata.get("definitions", {}))
    if definitions.get("Name") == "spectral_selective_bssfp_3d":
        readout_axis = str(definitions.get("ReadoutAxis", "")).lower()
        if readout_axis != "+z":
            raise RuntimeError(
                f"SS-bSSFP benchmark requires read axis +z, got {readout_axis!r}"
            )
    return program


def make_simulator(args) -> BlochSimulator:
    if args.threads < 0:
        raise ValueError("--threads must be zero or positive")
    num_threads = None if args.threads == 0 else args.threads
    return BlochSimulator(
        use_parallel=args.threads != 1,
        num_threads=num_threads,
        sequence_kernel=args.sequence_kernel,
        dynamic_sequence_kernel=args.dynamic_kernel,
    )


def simulate_program(
    simulator: BlochSimulator,
    program,
    phantom,
    *,
    timestep_s: float,
    spoiler_mode: str,
    spin_sampling: Sequence[int],
    signal_weighting: str = "voxel",
):
    kwargs = {
        "simulation_timestep_s": timestep_s,
        "spin_sampling": tuple(int(value) for value in spin_sampling),
        "spoiler_mode": spoiler_mode,
        "signal_weighting": signal_weighting,
    }
    if isinstance(phantom, DynamicSpectralPhantom):
        return simulator.simulate_dynamic_sequence(program, phantom, **kwargs)
    if isinstance(phantom, SpectralPhantom):
        return simulator.simulate_spectral_sequence(program, phantom, **kwargs)
    return simulator.simulate_sequence(program, phantom, **kwargs)


def timed_simulation(*args, **kwargs):
    start = perf_counter()
    result = simulate_program(*args, **kwargs)
    return result, perf_counter() - start


def base_result_record(
    *,
    benchmark: str,
    sequence_name: str,
    phantom_label: str,
    phantom_path: Path,
    phantom,
    program,
    result,
    repeat: int,
    generation_time_s: float,
    simulation_time_s: float,
) -> dict:
    metadata = result.metadata
    definitions = dict(program.metadata.get("definitions", {}))
    signal = np.asarray(result.signal)
    return {
        "benchmark": benchmark,
        "sequence": sequence_name,
        "phantom": phantom_label,
        "phantom_path": str(phantom_path),
        "phantom_kind": phantom_kind(phantom),
        "phantom_shape": "x".join(str(value) for value in phantom.shape),
        "repeat": repeat,
        "generation_time_s": generation_time_s,
        "simulation_time_s": simulation_time_s,
        "generation_plus_simulation_time_s": generation_time_s + simulation_time_s,
        "sequence_duration_s": float(program.duration_s),
        "adc_samples": int(signal.shape[-1] if signal.ndim else signal.size),
        "compiled_intervals": int(
            metadata.get("n_intervals", metadata.get("compiled_interval_count", 0))
        ),
        "active_voxels": int(
            metadata.get("n_active_voxels", getattr(phantom, "n_active", 0))
        ),
        "simulated_spins": int(metadata.get("n_simulated_spins", 0)),
        "readout_axis": str(definitions.get("ReadoutAxis", "")),
        "spoiler_mode": str(metadata.get("spoiler_mode", "")),
        "spin_sampling_xyz": "x".join(
            str(value) for value in metadata.get("subvoxel_spin_counts_xyz", (1, 1, 1))
        ),
        "signal_l2_norm": float(np.linalg.norm(signal)),
    }


def array_similarity(reference, candidate, *, prefix: str) -> dict:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"comparison shape mismatch: {reference.shape} versus {candidate.shape}"
        )
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    difference_norm = float(np.linalg.norm(candidate - reference))
    if reference_norm == 0.0:
        relative_error = 0.0 if candidate_norm == 0.0 else 1.0
        l2_similarity = 1.0 if candidate_norm == 0.0 else 0.0
    else:
        relative_error = difference_norm / reference_norm
        l2_similarity = 1.0 / (1.0 + relative_error)
    if reference_norm == 0.0 or candidate_norm == 0.0:
        correlation = 1.0 if reference_norm == candidate_norm == 0.0 else 0.0
    else:
        correlation = float(
            abs(np.vdot(reference.reshape(-1), candidate.reshape(-1)))
            / (reference_norm * candidate_norm)
        )
    return {
        f"{prefix}_reference_l2_norm": reference_norm,
        f"{prefix}_candidate_l2_norm": candidate_norm,
        f"{prefix}_relative_l2_error": relative_error,
        f"{prefix}_l2_similarity": l2_similarity,
        f"{prefix}_complex_correlation": min(1.0, max(0.0, correlation)),
        f"{prefix}_max_abs_error": float(
            np.max(np.abs(candidate - reference)) if reference.size else 0.0
        ),
    }


def result_similarity(reference, candidate) -> dict:
    metrics = array_similarity(reference.signal, candidate.signal, prefix="signal")
    reference_species = getattr(reference, "species_signal", None)
    candidate_species = getattr(candidate, "species_signal", None)
    if reference_species is not None and candidate_species is not None:
        metrics.update(
            array_similarity(
                reference_species,
                candidate_species,
                prefix="pool_signal",
            )
        )

    reference_final = np.asarray(reference.final_magnetization)
    candidate_final = np.asarray(candidate.final_magnetization)
    metrics.update(
        array_similarity(
            reference_final[..., :2],
            candidate_final[..., :2],
            prefix="final_transverse",
        )
    )

    crusher_times = np.asarray(
        reference.metadata.get("declared_ideal_spoiler_end_times_s", ()), dtype=float
    )
    if crusher_times.size:
        adc_times = np.asarray(reference.adc_times_s, dtype=float)
        after_first_crusher = adc_times > crusher_times[0] + 1e-12
        if np.any(after_first_crusher):
            metrics.update(
                array_similarity(
                    np.asarray(reference.signal)[..., after_first_crusher],
                    np.asarray(candidate.signal)[..., after_first_crusher],
                    prefix="post_crusher_signal",
                )
            )
    return metrics


def ideal_similarity_record(result) -> dict:
    return result_similarity(result, result)


def write_records(
    output_dir: Path, stem: str, records: Iterable[dict]
) -> tuple[Path, Path]:
    rows = list(records)
    if not rows:
        raise ValueError("cannot write an empty benchmark result")
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=True)
        stream.write("\n")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def print_runtime_row(record: Mapping[str, object]) -> None:
    detail = record.get("kernel", record.get("crusher", record.get("spoiler_mode", "")))
    message = (
        "  Completed: "
        f"Simulation {float(record['simulation_time_s']):.3f} s | "
        f"sequence setup {float(record['generation_time_s']):.3f} s | "
        f"ADC {int(record['adc_samples'])} | "
        f"intervals {int(record['compiled_intervals'])} | "
        f"active voxels {int(record['active_voxels'])} | "
        f"simulated spins {int(record['simulated_spins'])}"
    )
    if detail:
        message += f" | crusher {detail}"
    if "post_crusher_signal_l2_similarity" in record:
        message += (
            " | post-crusher similarity "
            f"{float(record['post_crusher_signal_l2_similarity']):.6f}"
        )
    message += "\n" + 39 * "-" + "\n"
    print(message, flush=True)


def print_benchmark_header(
    benchmark: str,
    args,
    output_dir: Path,
    *,
    selection: Mapping[str, object] | None = None,
    show_configured_kernels: bool = True,
) -> None:
    """Print the complete configuration before a benchmark starts."""
    threads = "automatic" if args.threads == 0 else str(args.threads)
    print("\n" + "=" * 78, flush=True)
    print(f"Benchmark: {benchmark}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(
        f"  Profile: {args.profile} | repeats: {args.repeats} | "
        f"time step: {args.timestep_us:g} us | threads: {threads}",
        flush=True,
    )
    if show_configured_kernels:
        print(
            f"  Kernels: sequence={args.sequence_kernel}, "
            f"dynamic={args.dynamic_kernel}",
            flush=True,
        )
    if selection:
        for label, value in selection.items():
            print(f"  {label}: {_format_console_value(value)}", flush=True)
    print("=" * 78, flush=True)


def print_phantom_summary(label: str, path: Path, phantom) -> None:
    """Describe the phantom before sequence generation or simulation."""
    shape = tuple(int(value) for value in phantom.shape)
    fov_mm = tuple(value * 1e3 for value in physical_fov_xyz_m(phantom))
    voxel_mm = tuple(value * 1e3 for value in physical_voxel_sizes_xyz_m(phantom))
    components = getattr(phantom, "species", getattr(phantom, "pools", ()))
    component_names = [str(item.name) for item in components]
    name = str(getattr(phantom, "name", "") or "-")

    print(f"\nPhantom: {label}", flush=True)
    print(f"  File: {path}", flush=True)
    print(f"  Name/type: {name} | {phantom_kind(phantom)}", flush=True)
    print(
        f"  Grid: {_format_shape(shape)} | FOV: {_format_vector(fov_mm)} mm | "
        f"voxel: {_format_vector(voxel_mm)} mm",
        flush=True,
    )
    print(
        f"  Active voxels: {int(getattr(phantom, 'n_active', 0))} | "
        f"nucleus: {getattr(phantom, 'nucleus', '-')} | "
        f"field strength: {float(getattr(phantom, 'field_strength', 0.0)):g} T",
        flush=True,
    )
    if component_names:
        print(
            f"  Pools/metabolites: {len(component_names)} components | "
            f"{', '.join(component_names)}",
            flush=True,
        )
        offsets = []
        for component in components:
            try:
                offset_hz = phantom.get_frequency_offset(
                    component.name,
                    getattr(phantom, "field_strength", None),
                    getattr(phantom, "nucleus", None),
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                offsets = []
                break
            offsets.append(f"{component.name}={float(offset_hz):.6g} Hz")
        if offsets:
            print(
                "  Metabolite centre offsets: " + " | ".join(offsets),
                flush=True,
            )
    spectral_points = getattr(phantom, "spectral_points", None)
    spectral_bandwidth_ppm = getattr(phantom, "spectral_bandwidth_ppm", None)
    spectral_reference_ppm = getattr(phantom, "spectral_reference_ppm", None)
    if spectral_points is not None and spectral_bandwidth_ppm is not None:
        points = int(spectral_points)
        spacing_ppm = float(spectral_bandwidth_ppm) / max(1, points - 1)
        print(
            "  Spectral display grid (not FLASH ADC): "
            f"reference={float(spectral_reference_ppm):.6g} ppm | "
            f"bandwidth={float(spectral_bandwidth_ppm):.6g} ppm | "
            f"points={points} | spacing={spacing_ppm:.6g} ppm",
            flush=True,
        )
    if isinstance(phantom, DynamicSpectralPhantom):
        print(
            "  Dynamics: "
            f"pyruvate inflow={'yes' if phantom.pyruvate_inflow is not None else 'no'}, "
            f"dynamic B0={'yes' if phantom.dynamic_b0 is not None else 'no'}, "
            f"start={float(phantom.conversion_start_s):g} s, "
            f"time offset={float(phantom.kinetics_time_offset_s):g} s",
            flush=True,
        )


def print_sequence_summary(
    sequence_name: str,
    path: Path,
    program,
    generation_time_s: float,
) -> None:
    """Print the relevant Pulseq definitions without dumping large arrays."""
    definitions = dict(program.metadata.get("definitions", {}))
    print(f"\nSequence: {sequence_name}", flush=True)
    print(f"  File: {path}", flush=True)
    print(
        f"  Type: {definitions.get('Name', sequence_name)} | "
        f"duration: {float(program.duration_s):.6g} s | "
        f"setup/load: {generation_time_s:.3f} s",
        flush=True,
    )

    matrix = definitions.get("EncodingMatrixSize", definitions.get("MatrixSize"))
    fov = definitions.get("EncodingFOV", definitions.get("FOV"))
    if matrix is not None or fov is not None:
        fields = []
        if matrix is not None:
            fields.append(f"matrix: {_format_console_value(matrix)}")
        if fov is not None:
            fov_mm = np.asarray(fov, dtype=float).reshape(-1) * 1e3
            fields.append(f"FOV: {_format_vector(fov_mm)} mm")
        print("  " + " | ".join(fields), flush=True)

    axes = [
        str(definitions.get(key, "-"))
        for key in ("ReadoutAxis", "PhaseEncodingAxis", "PartitionEncodingAxis")
    ]
    print(
        f"  Axes: read={axes[0]}, phase={axes[1]}, partition={axes[2]}",
        flush=True,
    )

    _print_definition_group(
        "RF",
        definitions,
        (
            ("FlipAngleDeg", "flip", "deg"),
            ("RFPulseType", "pulse", ""),
            ("SpectralRFPulseType", "pulse", ""),
            ("RFDuration", "duration", "s"),
            ("SpectralRFDuration", "duration", "s"),
            ("RFFrequencyOffsetHz", "offset", "Hz"),
        ),
    )
    _print_definition_group(
        "Timing",
        definitions,
        (
            ("TR", "TR", "s"),
            ("TE", "TE", "s"),
            ("Echoes", "echoes", ""),
            ("EchoTimes", "echo times", "s"),
            ("Repetitions", "repetitions", ""),
        ),
    )
    _print_definition_group(
        "Acquisition",
        definitions,
        (
            ("SamplingBandwidth", "bandwidth", "Hz"),
            ("SpectralBandwidth", "spectral bandwidth", "Hz"),
            ("SpectralPoints", "spectral points", ""),
            ("DynamicFrames", "dynamic frames", ""),
            ("DynamicFrameInterval", "frame interval", "s"),
            ("AcquisitionInterval", "acquisition interval", "s"),
        ),
    )
    _print_definition_group(
        "Spectral",
        definitions,
        (
            ("SpectralTargetNames", "targets", ""),
            ("SpectralTargetOffsetsHz", "target offsets", "Hz"),
            ("SpectralReceiverOffsetsHz", "receiver offsets", "Hz"),
            ("ReceiverFrequencyOffsetHz", "receiver offset", "Hz"),
        ),
    )
    _print_definition_group(
        "Spoiler",
        definitions,
        (
            ("EndImageSpoilerCyclesPerVoxel", "end-image cycles/voxel", ""),
            ("SpoilerCyclesPerVoxel", "cycles/voxel", ""),
            ("SpoilerCyclesPerSlice", "cycles/slice", ""),
            ("EndImageSpoilerDuration", "duration", "s"),
            ("SpoilerDuration", "duration", "s"),
        ),
    )


def print_simulation_start(
    *,
    repeat: int,
    repeats: int,
    spoiler_mode: str,
    spin_sampling: Sequence[int],
    timestep_us: float,
    details: Mapping[str, object] | None = None,
) -> None:
    """Announce a simulation before entering the potentially long call."""
    fields = [
        f"crusher={spoiler_mode}",
        f"subvoxel spins={_format_shape(spin_sampling)}",
        f"time step={timestep_us:g} us",
    ]
    if details:
        fields.extend(
            f"{key}={_format_console_value(value)}" for key, value in details.items()
        )
    print(
        f"\n  Starting simulation {repeat}/{repeats}: " + " | ".join(fields),
        flush=True,
    )


def print_resolution_comparison(record: Mapping[str, object]) -> None:
    """Print the accuracy result once the finest resolution is known."""
    print(
        f"  Comparison {record['phantom_shape']} versus "
        f"{record['reference_resolution']}: "
        "image similarity "
        f"{float(record['magnitude_image_vs_finest_l2_similarity']):.6f} | "
        "signal similarity "
        f"{float(record['signal_vs_finest_l2_similarity']):.6f}",
        flush=True,
    )


def print_saved_results(csv_path: Path, json_path: Path) -> None:
    print("\nResult files:", flush=True)
    print(f"  CSV:  {csv_path}", flush=True)
    print(f"  JSON: {json_path}", flush=True)


def _print_definition_group(
    title: str,
    definitions: Mapping[str, object],
    fields: Sequence[tuple[str, str, str]],
) -> None:
    values = []
    seen_labels = set()
    for key, label, unit in fields:
        if key not in definitions or label in seen_labels:
            continue
        seen_labels.add(label)
        value = _format_console_value(definitions[key])
        values.append(f"{label}={value}{(' ' + unit) if unit else ''}")
    if values:
        print(f"  {title}: " + " | ".join(values), flush=True)


def _format_console_value(value: object) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, Mapping):
        return ", ".join(
            f"{key}={_format_console_value(item)}" for key, item in value.items()
        )
    if isinstance(value, Iterable):
        items = list(value)
        if len(items) > 8:
            return (
                f"[{', '.join(_format_console_value(item) for item in items[:4])}, "
                f"... ({len(items)} values)]"
            )
        return "[" + ", ".join(_format_console_value(item) for item in items) + "]"
    return str(value)


def _format_vector(values: Iterable[object]) -> str:
    return " x ".join(f"{float(value):.4g}" for value in values)


def _format_shape(values: Iterable[object]) -> str:
    return "x".join(str(int(value)) for value in values)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "phantom"
