#!/usr/bin/env python3
"""Run a fixed-sequence FLASH flip-angle sweep from a ``.blochproj`` file.

The saved sequence, phantom, and B1 fields are reused for every run.  Only the
RF waveform amplitude is scaled, so timing, gradients, RF phase spoiling, and
all spatial settings remain unchanged.  Each completed angle is written as an
independent project file, which makes long sweeps resumable.

Example
-------
python scripts/run_flash_flip_angle_sweep.py \
    /path/to/input.blochproj --start 5 --stop 150 --step 5
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.project_io import load_project, save_project
from blochsimulator.sequence import RFEvent, SequenceProgram
from blochsimulator.sequence.spin_sampling import SpinSampling
from blochsimulator.spectral_phantom import SpectralPhantom


DEFAULT_PROJECT = Path(
    "/Users/lucanagel/Library/CloudStorage/"
    "GoogleDrive-luca.sc.nagel@gmail.com/Meine Ablage/phD/05_CryoCoils/"
    "10_publication/02_Results/05_3D_phantom_simulations/"
    "bloch_project_flash_urea_phantom.blochproj"
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def flip_angle_values(start: float, stop: float, step: float) -> np.ndarray:
    """Return an increasing, endpoint-inclusive flip-angle grid."""
    start = float(start)
    stop = float(stop)
    step = float(step)
    if not all(np.isfinite(value) for value in (start, stop, step)):
        raise ValueError("flip-angle limits must be finite")
    if start <= 0.0 or stop <= 0.0 or step <= 0.0:
        raise ValueError("flip angles and step size must be positive")
    if stop < start:
        raise ValueError("stop flip angle must not be smaller than start")
    count = int(np.floor((stop - start) / step + 1e-12))
    values = start + step * np.arange(count + 1, dtype=float)
    tolerance = max(1e-12, abs(stop) * 1e-12)
    if values[-1] < stop - tolerance:
        values = np.concatenate((values, np.asarray([stop], dtype=float)))
    else:
        values[-1] = stop
    return values


def reference_flip_angle_deg(project: dict) -> float:
    """Read the nominal flip angle used to generate the saved RF events."""
    program = project.get("program")
    if program is None:
        raise ValueError("project does not contain a sequence")
    definitions = dict(program.metadata.get("definitions", {}))
    candidates = [definitions.get("FlipAngleDeg")]
    state = project.get("state", {})
    flash_control = state.get("sequence_controls", {}).get("flash_flip_angle_deg", {})
    if isinstance(flash_control, dict):
        candidates.append(flash_control.get("value"))
    for candidate in candidates:
        try:
            angle = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(angle) and angle > 0.0:
            return angle
    raise ValueError("project does not contain a positive nominal FLASH flip angle")


def program_with_flip_angle(
    program: SequenceProgram,
    target_angle_deg: float,
    reference_angle_deg: float,
) -> SequenceProgram:
    """Scale every RF event while preserving its shape and phase history."""
    target_angle_deg = float(target_angle_deg)
    reference_angle_deg = float(reference_angle_deg)
    if not np.isfinite(target_angle_deg) or target_angle_deg <= 0.0:
        raise ValueError("target flip angle must be positive and finite")
    if not np.isfinite(reference_angle_deg) or reference_angle_deg <= 0.0:
        raise ValueError("reference flip angle must be positive and finite")
    if not program.rf_events:
        raise ValueError("sequence does not contain an RF event")

    scale = target_angle_deg / reference_angle_deg
    events = tuple(
        (
            replace(event, samples_hz=np.asarray(event.samples_hz) * scale)
            if isinstance(event, RFEvent)
            else event
        )
        for event in program.events
    )
    metadata = copy.deepcopy(dict(program.metadata))
    definitions = dict(metadata.get("definitions", {}))
    definitions["FlipAngleDeg"] = target_angle_deg
    metadata["definitions"] = definitions
    metadata["flip_angle_sweep"] = {
        "target_flip_angle_deg": target_angle_deg,
        "reference_flip_angle_deg": reference_angle_deg,
        "rf_scale_factor": scale,
    }
    return SequenceProgram(
        events=events,
        duration_s=program.duration_s,
        source=program.source,
        version=program.version,
        metadata=metadata,
    )


def state_with_flip_angle(
    state: dict,
    target_angle_deg: float,
    reference_project: Path,
) -> dict:
    """Update the saved FLASH control without changing the RF Designer reference."""
    updated = copy.deepcopy(dict(state))
    sequence_controls = updated.setdefault("sequence_controls", {})
    control = sequence_controls.setdefault("flash_flip_angle_deg", {"type": "value"})
    if not isinstance(control, dict):
        control = {"type": "value"}
        sequence_controls["flash_flip_angle_deg"] = control
    control["value"] = float(target_angle_deg)
    updated["flip_angle_sweep"] = {
        "target_flip_angle_deg": float(target_angle_deg),
        "reference_project": str(reference_project.resolve()),
    }
    return updated


def _apply_project_b1_fields(project: dict) -> None:
    """Apply the project's independent Tx/Rx fields as the GUI does."""
    phantom = project["phantom"]
    tx_field = project.get("tx_field")
    rx_field = project.get("rx_field")
    if tx_field is not None:
        phantom.tx_sensitivity_map = tx_field.resample_to_phantom(phantom)[0]
    if rx_field is not None:
        phantom.rx_sensitivity_maps = rx_field.resample_to_phantom(phantom)


def _saved_result_metadata(project: dict) -> dict:
    result = project.get("sequence_result")
    return dict(getattr(result, "metadata", {}) or {})


def _state_control_value(project: dict, name: str, default=None):
    control = project.get("state", {}).get("sequence_controls", {}).get(name, {})
    return control.get("value", default) if isinstance(control, dict) else default


def simulation_settings(project: dict, args) -> dict:
    """Resolve reproducible simulation settings, preferring CLI overrides."""
    metadata = _saved_result_metadata(project)
    saved_sampling = metadata.get("spin_sampling", {})
    if not isinstance(saved_sampling, dict):
        saved_sampling = {}

    if args.spin_counts is not None:
        counts = tuple(args.spin_counts)
    else:
        counts = tuple(
            int(value) for value in saved_sampling.get("counts_xyz", (1, 1, 9))
        )
    method = args.spin_method or str(saved_sampling.get("method", "midpoint"))
    selected = saved_sampling.get("selected_indices")
    if args.spin_counts is not None or args.spin_method is not None:
        selected = None
    spin_sampling = SpinSampling(
        counts_xyz=counts,
        method=method,
        selected_indices=(None if selected is None else tuple(selected)),
    )

    saved_timestep_s = metadata.get("simulation_timestep_s")
    if saved_timestep_s is None:
        timestep_us = _state_control_value(project, "simulation_timestep_us", 20.0)
        saved_timestep_s = float(timestep_us) * 1e-6
    timestep_s = (
        float(args.simulation_timestep_us) * 1e-6
        if args.simulation_timestep_us is not None
        else float(saved_timestep_s)
    )

    signal_weighting = args.signal_weighting or str(
        metadata.get("signal_weighting", "voxel")
    )
    spoiler_mode = args.spoiler_mode or str(metadata.get("spoiler_mode", "ideal"))
    sequence_kernel = args.sequence_kernel or str(
        metadata.get("sequence_kernel", "optimized")
    )
    chunk_voxels = (
        args.chunk_voxels
        if args.chunk_voxels is not None
        else metadata.get("chunk_voxels")
    )
    if chunk_voxels is not None:
        chunk_voxels = int(chunk_voxels)

    definitions = project["program"].metadata.get("definitions", {})
    phantom = project["phantom"]
    field_strength_t = float(
        getattr(
            phantom,
            "field_strength",
            definitions.get("FieldStrengthT", 7.0),
        )
    )
    nucleus = str(getattr(phantom, "nucleus", definitions.get("Nucleus", "C13")))
    return {
        "spin_sampling": spin_sampling,
        "simulation_timestep_s": timestep_s,
        "signal_weighting": signal_weighting,
        "spoiler_mode": spoiler_mode,
        "sequence_kernel": sequence_kernel,
        "chunk_voxels": chunk_voxels,
        "field_strength_t": field_strength_t,
        "nucleus": nucleus,
    }


def _angle_label(angle_deg: float) -> str:
    rounded = round(float(angle_deg))
    if np.isclose(angle_deg, rounded, rtol=0.0, atol=1e-9):
        return f"fa_{rounded:03d}deg"
    text = f"{float(angle_deg):08.3f}".rstrip("0").rstrip(".")
    return "fa_" + text.replace("-", "m").replace(".", "p") + "deg"


def _print_settings(
    project_path: Path,
    output_dir: Path,
    angles: np.ndarray,
    reference_angle: float,
    settings: dict,
) -> None:
    sampling = settings["spin_sampling"]
    print(f"Project: {project_path}")
    print(f"Output:  {output_dir}")
    print(
        f"Angles:  {angles[0]:g} to {angles[-1]:g} deg " f"({angles.size} simulations)"
    )
    print(f"Reference flip angle in project: {reference_angle:g} deg")
    print(
        "Simulation settings: "
        f"{settings['simulation_timestep_s'] * 1e6:g} us timestep, "
        f"spin grid {sampling.counts_xyz} {sampling.method}, "
        f"{settings['spoiler_mode']} spoilers, "
        f"{settings['signal_weighting']} signal weighting, "
        f"{settings['sequence_kernel']} kernel"
    )


def _simulate(
    simulator: BlochSimulator,
    program: SequenceProgram,
    phantom,
    settings: dict,
):
    if isinstance(phantom, DynamicSpectralPhantom):
        simulate = simulator.simulate_dynamic_sequence
    elif isinstance(phantom, SpectralPhantom):
        simulate = simulator.simulate_spectral_sequence
    else:
        simulate = simulator.simulate_sequence

    def status(message):
        print(f"    {message}", flush=True)

    kwargs = {
        "checkpoints_s": (),
        "chunk_voxels": settings["chunk_voxels"],
        "signal_weighting": settings["signal_weighting"],
        "status_callback": status,
        "simulation_timestep_s": settings["simulation_timestep_s"],
        "spin_sampling": settings["spin_sampling"],
        "spoiler_mode": settings["spoiler_mode"],
    }
    if isinstance(phantom, (SpectralPhantom, DynamicSpectralPhantom)):
        kwargs.update(
            field_strength_t=settings["field_strength_t"],
            nucleus=settings["nucleus"],
        )
    return simulate(program, phantom, **kwargs)


def _result_record(angle_deg: float, output_path: Path, result) -> dict:
    signal = np.asarray(result.signal)
    magnitude = np.abs(signal)
    metadata = dict(getattr(result, "metadata", {}) or {})
    return {
        "flip_angle_deg": float(angle_deg),
        "project_file": output_path.name,
        "adc_samples": int(np.asarray(result.adc_times_s).size),
        "receive_channels": int(1 if signal.ndim == 1 else signal.shape[0]),
        "peak_abs_signal": float(np.max(magnitude, initial=0.0)),
        "mean_abs_signal": float(np.mean(magnitude)) if magnitude.size else 0.0,
        "rms_abs_signal": (
            float(np.sqrt(np.mean(np.square(magnitude)))) if magnitude.size else 0.0
        ),
        "last_abs_signal": float(magnitude.reshape(-1)[-1]) if magnitude.size else 0.0,
        "simulation_wall_time_s": float(metadata.get("simulation_wall_time_s", np.nan)),
    }


def _write_summary(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _write_signal_archive(path: Path, angles, output_paths, results) -> None:
    signals = [np.asarray(result.signal) for result in results]
    adc_times = [np.asarray(result.adc_times_s) for result in results]
    if not signals:
        return
    reference_shape = signals[0].shape
    if any(signal.shape != reference_shape for signal in signals[1:]):
        raise ValueError("completed simulations have inconsistent signal shapes")
    reference_times = adc_times[0]
    if any(
        times.shape != reference_times.shape
        or not np.allclose(times, reference_times, rtol=0.0, atol=1e-12)
        for times in adc_times[1:]
    ):
        raise ValueError("completed simulations have inconsistent ADC time axes")
    species = [getattr(result, "species_signal", None) for result in results]
    values = {
        "flip_angles_deg": np.asarray(angles, dtype=float),
        "signal": np.stack(signals, axis=0),
        "adc_times_s": reference_times,
        "project_files": np.asarray([path.name for path in output_paths]),
    }
    if all(value is not None for value in species):
        species_arrays = [np.asarray(value) for value in species]
        if all(value.shape == species_arrays[0].shape for value in species_arrays):
            values["species_signal"] = np.stack(species_arrays, axis=0)
            values["pool_names"] = np.asarray(results[0].pool_names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _load_resumable_result(path: Path, expected_angle_deg: float):
    saved = load_project(path)
    program = saved.get("program")
    result = saved.get("sequence_result")
    saved_angle = None
    if program is not None:
        saved_angle = program.metadata.get("definitions", {}).get("FlipAngleDeg")
    try:
        matches = np.isclose(
            float(saved_angle), expected_angle_deg, rtol=0.0, atol=1e-9
        )
    except (TypeError, ValueError):
        matches = False
    if result is None or not matches:
        raise ValueError(
            f"existing output is incomplete or has the wrong angle: {path}; "
            "use --overwrite to replace it"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "project",
        nargs="?",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Input .blochproj file (defaults to the urea FLASH project).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: next to the input project).",
    )
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--stop", type=float, default=150.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument(
        "--num-threads",
        type=_positive_int,
        help="Worker threads (default: all logical CPU cores).",
    )
    parser.add_argument(
        "--simulation-timestep-us",
        type=_positive_float,
        help="Override the project/saved-result simulation time step.",
    )
    parser.add_argument(
        "--spin-counts",
        nargs=3,
        type=_positive_int,
        metavar=("X", "Y", "Z"),
        help="Override the saved subvoxel spin grid.",
    )
    parser.add_argument(
        "--spin-method",
        choices=("midpoint", "stratified"),
        help="Override the saved subvoxel sampling method.",
    )
    parser.add_argument(
        "--spoiler-mode",
        choices=("ideal", "gradient"),
        help="Override the saved spoiler mode.",
    )
    parser.add_argument(
        "--signal-weighting",
        choices=("voxel", "voxel_volume"),
        help="Override the saved signal weighting.",
    )
    parser.add_argument(
        "--sequence-kernel",
        choices=("optimized", "reference"),
        help="Override the saved sequence kernel.",
    )
    parser.add_argument(
        "--chunk-voxels",
        type=_positive_int,
        help="Override the saved/automatic voxel chunk size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-angle project files instead of resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the sweep without running simulations.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    project_path = args.project.expanduser().resolve()
    if not project_path.is_file():
        raise SystemExit(f"project file does not exist: {project_path}")
    if project_path.suffix.lower() != ".blochproj":
        raise SystemExit("input project must have the .blochproj extension")

    try:
        angles = flip_angle_values(args.start, args.stop, args.step)
        project = load_project(project_path)
        if project.get("phantom") is None:
            raise ValueError("project does not contain a phantom")
        if project.get("program") is None:
            raise ValueError("project does not contain a sequence")
        reference_angle = reference_flip_angle_deg(project)
        settings = simulation_settings(project, args)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_path.parent / f"{project_path.stem}_flip_angle_sweep"
    )
    _print_settings(project_path, output_dir, angles, reference_angle, settings)
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_project_b1_fields(project)
    simulator = BlochSimulator(
        use_parallel=True,
        num_threads=args.num_threads,
        sequence_kernel=settings["sequence_kernel"],
    )
    records = []
    results = []
    output_paths = []
    total_started = time.monotonic()
    summary_path = output_dir / "sweep_summary.csv"

    for index, angle in enumerate(angles, start=1):
        output_path = output_dir / f"{_angle_label(angle)}.blochproj"
        print(
            f"[{index}/{angles.size}] Flip angle {angle:g} deg -> "
            f"{output_path.name}",
            flush=True,
        )
        if output_path.exists() and not args.overwrite:
            result = _load_resumable_result(output_path, float(angle))
            print("    Existing completed simulation found; skipping.", flush=True)
        else:
            program = program_with_flip_angle(
                project["program"], float(angle), reference_angle
            )
            started_at = datetime.now(timezone.utc)
            started = time.monotonic()
            result = _simulate(simulator, program, project["phantom"], settings)
            wall_time_s = time.monotonic() - started
            result.metadata.update(
                {
                    "flip_angle_deg": float(angle),
                    "flip_angle_sweep_reference_deg": reference_angle,
                    "flip_angle_sweep_scale_factor": float(angle / reference_angle),
                    "flip_angle_sweep_source_project": str(project_path),
                    "simulation_started_at_utc": started_at.isoformat(),
                    "simulation_finished_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "simulation_wall_time_s": wall_time_s,
                    "simulation_time_measurement": "wall_clock",
                }
            )
            state = state_with_flip_angle(project["state"], float(angle), project_path)
            temporary_path = output_path.with_suffix(".blochproj.tmp")
            save_project(
                temporary_path,
                state,
                phantom=project["phantom"],
                tx_field=project.get("tx_field"),
                rx_field=project.get("rx_field"),
                program=program,
                legacy_result=None,
                sequence_result=result,
            )
            temporary_path.replace(output_path)
            print(f"    Completed in {wall_time_s / 60.0:.1f} min.", flush=True)

        records.append(_result_record(float(angle), output_path, result))
        results.append(result)
        output_paths.append(output_path)
        _write_summary(summary_path, records)

    signal_archive = output_dir / "sweep_signals.npz"
    _write_signal_archive(signal_archive, angles, output_paths, results)
    elapsed_s = time.monotonic() - total_started
    print(
        f"Sweep complete in {elapsed_s / 60.0:.1f} min.\n"
        f"Summary: {summary_path}\n"
        f"Signals: {signal_archive}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
