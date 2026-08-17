"""Validate mixed-precision dynamic execution against the Float64 CPU oracle.

Inputs may be Pulseq ``.seq`` files or self-contained ``.blochproj`` projects.
The validator reports total/species signal, phase, low-signal, time-resolved
growth, final pool state, optional reconstruction, and repeatability metrics.
The private Metal candidate remains a feasibility probe and is never selected
by the normal simulator or GUI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.project_io import load_project
from blochsimulator.sequence import (
    SequenceProgram,
    SequenceSimulationResult,
    load_pulseq,
)
from blochsimulator.spectral_phantom import ChemicalSpecies


DEFAULT_SIGNAL_NRMSE_GATE = 1.0e-3
SIGNIFICANT_MAGNITUDE_FRACTION = 1.0e-3
LOW_SIGNAL_MAGNITUDE_FRACTION = 1.0e-4


def _array_metric(reference, candidate):
    if reference is None or candidate is None:
        return None
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"metric arrays have different shapes: {reference.shape} and "
            f"{candidate.shape}"
        )
    difference = candidate.astype(np.complex128) - reference.astype(np.complex128)
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    difference_norm = float(np.linalg.norm(difference.reshape(-1)))
    peak = float(np.max(np.abs(reference), initial=0.0))
    max_absolute = float(np.max(np.abs(difference), initial=0.0))
    return {
        "shape": list(reference.shape),
        "reference_dtype": reference.dtype.name,
        "candidate_dtype": candidate.dtype.name,
        "nrmse": (
            difference_norm / reference_norm if reference_norm else difference_norm
        ),
        "max_absolute_error": max_absolute,
        "reference_peak": peak,
        "max_error_relative_to_reference_peak": (
            max_absolute / peak if peak else max_absolute
        ),
    }


def _signal_metric(reference, candidate):
    metric = _array_metric(reference, candidate)
    if metric is None:
        return None
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    difference = candidate - reference
    magnitude = np.abs(reference)
    peak = metric["reference_peak"]
    significant = magnitude >= peak * SIGNIFICANT_MAGNITUDE_FRACTION
    low_signal = magnitude <= peak * LOW_SIGNAL_MAGNITUDE_FRACTION
    if np.any(significant):
        phase_error = np.angle(candidate[significant] * np.conj(reference[significant]))
        metric["phase_error_significant_rad"] = {
            "threshold_relative_to_peak": SIGNIFICANT_MAGNITUDE_FRACTION,
            "sample_count": int(np.count_nonzero(significant)),
            "rms": float(np.sqrt(np.mean(np.square(phase_error)))),
            "maximum_absolute": float(np.max(np.abs(phase_error), initial=0.0)),
        }
    else:
        metric["phase_error_significant_rad"] = {
            "threshold_relative_to_peak": SIGNIFICANT_MAGNITUDE_FRACTION,
            "sample_count": 0,
            "rms": 0.0,
            "maximum_absolute": 0.0,
        }
    low_error = np.abs(difference[low_signal])
    metric["low_signal"] = {
        "threshold_relative_to_peak": LOW_SIGNAL_MAGNITUDE_FRACTION,
        "sample_count": int(np.count_nonzero(low_signal)),
        "rms_absolute_error": (
            float(np.sqrt(np.mean(np.square(low_error)))) if low_error.size else 0.0
        ),
        "maximum_absolute_error": float(np.max(low_error, initial=0.0)),
        "maximum_error_relative_to_reference_peak": (
            float(np.max(low_error, initial=0.0)) / peak if peak else 0.0
        ),
    }
    decile_threshold = float(np.quantile(magnitude, 0.1)) if magnitude.size else 0.0
    lowest_decile = magnitude <= decile_threshold
    decile_error = np.abs(difference[lowest_decile])
    metric["lowest_reference_magnitude_decile"] = {
        "magnitude_threshold": decile_threshold,
        "threshold_relative_to_peak": decile_threshold / peak if peak else 0.0,
        "sample_count": int(np.count_nonzero(lowest_decile)),
        "rms_absolute_error": (
            float(np.sqrt(np.mean(np.square(decile_error))))
            if decile_error.size
            else 0.0
        ),
        "maximum_absolute_error": float(np.max(decile_error, initial=0.0)),
        "maximum_error_relative_to_reference_peak": (
            float(np.max(decile_error, initial=0.0)) / peak if peak else 0.0
        ),
    }
    return metric


def _error_growth(reference, candidate, adc_times_s, bin_count=10):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    times = np.asarray(adc_times_s, dtype=float)
    if reference.shape[-1] != times.size:
        raise ValueError("signal and ADC time lengths do not match")
    if times.size == 0:
        return []
    edges = np.linspace(0, times.size, min(int(bin_count), times.size) + 1, dtype=int)
    growth = []
    for bin_index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        window = (..., slice(start, stop))
        cumulative = (..., slice(0, stop))
        growth.append(
            {
                "bin": bin_index,
                "adc_index_start": int(start),
                "adc_index_stop": int(stop),
                "time_start_s": float(times[start]),
                "time_end_s": float(times[stop - 1]),
                "window": _signal_metric(reference[window], candidate[window]),
                "cumulative": _signal_metric(
                    reference[cumulative], candidate[cumulative]
                ),
            }
        )
    return growth


def _truncate_at_adc_event(program, event_count):
    if event_count is None:
        return program
    event_count = int(event_count)
    if event_count < 1 or event_count > len(program.adc_events):
        raise ValueError(
            f"max ADC events must be between 1 and {len(program.adc_events)}"
        )
    last_adc_time = program.adc_events[event_count - 1].end_s
    tolerance = max(1e-15, abs(last_adc_time) * 1e-12)
    events = tuple(
        event for event in program.events if event.start_s <= last_adc_time + tolerance
    )
    duration_s = max((event.end_s for event in events), default=last_adc_time)
    return SequenceProgram(
        events=events,
        duration_s=duration_s,
        source=f"{program.source}:first-{event_count}-adc-events",
        version=program.version,
        # Full-acquisition dimensions and ideal-spoiler declarations can point
        # beyond a prefix. They are intentionally omitted for accumulation runs.
        metadata={},
    )


def _synthetic_phantom(program, shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("synthetic phantom shape requires three positive integers")
    definitions = program.metadata.get("definitions", {})
    fov = np.asarray(definitions.get("FOV", (0.056, 0.028, 0.021)), dtype=float)
    if fov.size != 3 or not np.all(np.isfinite(fov)) or np.any(fov <= 0):
        fov = np.asarray((0.056, 0.028, 0.021), dtype=float)
    coordinates = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=tuple(float(value) for value in fov),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": 0.05 * coordinates,
        },
        kpl_map_s_inv=0.1 * coordinates,
        b0_map=-20.0 + 40.0 * coordinates,
        field_strength=float(definitions.get("FieldStrengthT", 7.0)),
        nucleus=str(definitions.get("Nucleus", "C13")),
    )


def _metal_result(program, reference, raw):
    metadata = dict(reference.metadata)
    metadata.update(raw["metadata"])
    return SequenceSimulationResult(
        signal=np.asarray(raw["signal"]),
        adc_times_s=np.asarray(raw["adc_times_s"]),
        final_magnetization=np.asarray(raw["final_magnetization"]),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.zeros(0, dtype=float),
        metadata=metadata,
        adc_gradient_moment_cyc_per_m=reference.adc_gradient_moment_cyc_per_m,
        pool_names=tuple(reference.pool_names),
        species_signal=np.asarray(raw["species_signal"]),
        final_pool_magnetization=np.asarray(raw["final_pool_magnetization"]),
    )


def _result_metrics(reference, candidate):
    pool_names = tuple(reference.pool_names) or tuple(
        f"pool_{index}" for index in range(reference.species_signal.shape[0])
    )
    metrics = {
        "total_signal": _signal_metric(reference.signal, candidate.signal),
        "species_signal": {
            name: _signal_metric(
                reference.species_signal[index], candidate.species_signal[index]
            )
            for index, name in enumerate(pool_names)
        },
        "final_pool_state": _array_metric(
            reference.final_pool_magnetization,
            candidate.final_pool_magnetization,
        ),
        "final_combined_state": _array_metric(
            reference.final_magnetization,
            candidate.final_magnetization,
        ),
        "error_growth": _error_growth(
            reference.signal,
            candidate.signal,
            reference.adc_times_s,
        ),
    }
    if reference.cartesian_acquisition_volumes is not None:
        metrics["cartesian_3d_kspace"] = _array_metric(
            reference.to_cartesian_3d_kspace(), candidate.to_cartesian_3d_kspace()
        )
        metrics["cartesian_3d_image"] = _array_metric(
            reference.reconstruct_cartesian_3d(),
            candidate.reconstruct_cartesian_3d(),
        )
    return metrics


def _repeatability_metrics(results):
    first = results[0]
    comparisons = []
    for index, candidate in enumerate(results[1:], start=2):
        comparisons.append(
            {
                "run": index,
                "signal": _array_metric(first.signal, candidate.signal),
                "species_signal": _array_metric(
                    first.species_signal, candidate.species_signal
                ),
                "final_pool_state": _array_metric(
                    first.final_pool_magnetization,
                    candidate.final_pool_magnetization,
                ),
                "bitwise_equal": bool(
                    np.array_equal(first.signal, candidate.signal)
                    and np.array_equal(
                        first.final_pool_magnetization,
                        candidate.final_pool_magnetization,
                    )
                ),
            }
        )
    return {"run_count": len(results), "comparisons_to_first": comparisons}


def compare_precision(
    program,
    phantom,
    *,
    simulation_timestep_s,
    candidates=("cpu_float32",),
    metal_repeat_runs=3,
    signal_nrmse_gate=DEFAULT_SIGNAL_NRMSE_GATE,
    spin_sampling=None,
    hybrid_calibration_fraction=0.10,
    hybrid_validation_fraction=0.05,
    hybrid_fallback_to_cpu=True,
    hybrid_run_concurrently=True,
    metal_spin_chunk_size="auto",
):
    from blochsimulator.sequence.spin_sampling import coerce_spin_sampling

    sampling_metadata = coerce_spin_sampling(spin_sampling).to_metadata()
    reference_simulator = BlochSimulator(
        use_parallel=False,
        dynamic_sequence_kernel="optimized",
        dynamic_sequence_precision="float64",
    )
    start = time.perf_counter()
    reference = reference_simulator.simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=simulation_timestep_s,
        spin_sampling=spin_sampling,
    )
    reference_seconds = time.perf_counter() - start
    report = {
        "duration_s": float(program.duration_s),
        "active_voxels": int(phantom.n_active),
        "adc_samples": int(reference.adc_times_s.size),
        "simulation_timestep_s": float(simulation_timestep_s),
        "signal_nrmse_gate": float(signal_nrmse_gate),
        "spin_sampling": sampling_metadata,
        "reference": {
            "backend": "optimized CPU Float64",
            "runtime_seconds": reference_seconds,
            "compiled_interval_count": reference.metadata.get(
                "compiled_interval_count"
            ),
        },
        "candidates": {},
    }
    for candidate_name in candidates:
        if candidate_name == "cpu_float32":
            simulator = BlochSimulator(
                use_parallel=False,
                dynamic_sequence_kernel="optimized",
                dynamic_sequence_precision="float32",
            )
            start = time.perf_counter()
            candidate = simulator.simulate_dynamic_sequence(
                program,
                phantom,
                simulation_timestep_s=simulation_timestep_s,
                spin_sampling=spin_sampling,
            )
            runtime = time.perf_counter() - start
            metrics = _result_metrics(reference, candidate)
            report["candidates"][candidate_name] = {
                "runtime_seconds": runtime,
                "metadata": candidate.metadata,
                "metrics": metrics,
                "gate_passed": bool(
                    metrics["total_signal"]["nrmse"] <= signal_nrmse_gate
                ),
                "repeatability": {
                    "run_count": 1,
                    "note": "the deterministic CPU shadow candidate was run once",
                },
            }
        elif candidate_name in {"metal_probe", "metal_double_single"}:
            from blochsimulator.dynamic_metal_backend import run_metal_precision_probe

            metal_results = []
            runtimes = []
            native_metadata = []
            for _ in range(max(1, int(metal_repeat_runs))):
                start = time.perf_counter()
                raw = run_metal_precision_probe(
                    program,
                    phantom,
                    simulation_timestep_s=simulation_timestep_s,
                    spin_sampling=spin_sampling,
                    precision_strategy=(
                        "double_single"
                        if candidate_name == "metal_double_single"
                        else "float32"
                    ),
                )
                runtimes.append(time.perf_counter() - start)
                native_metadata.append(raw["metadata"])
                metal_results.append(_metal_result(program, reference, raw))
            metrics = _result_metrics(reference, metal_results[0])
            report["candidates"][candidate_name] = {
                "runtime_seconds": runtimes,
                "metadata": native_metadata,
                "metrics": metrics,
                "gate_passed": bool(
                    metrics["total_signal"]["nrmse"] <= signal_nrmse_gate
                ),
                "repeatability": _repeatability_metrics(metal_results),
            }
        elif candidate_name == "metal_hybrid":
            from blochsimulator.dynamic_metal_backend import run_metal_hybrid_probe

            start = time.perf_counter()
            raw = run_metal_hybrid_probe(
                program,
                phantom,
                simulation_timestep_s=simulation_timestep_s,
                spin_sampling=spin_sampling,
                calibration_fraction=hybrid_calibration_fraction,
                validation_fraction=hybrid_validation_fraction,
                signal_nrmse_gate=signal_nrmse_gate,
                fallback_to_cpu=hybrid_fallback_to_cpu,
                run_concurrently=hybrid_run_concurrently,
                spin_chunk_size=metal_spin_chunk_size,
            )
            runtime = time.perf_counter() - start
            candidate = _metal_result(program, reference, raw)
            metrics = _result_metrics(reference, candidate)
            hybrid_validation_passed = bool(raw["metadata"]["hybrid_validation_passed"])
            report["candidates"][candidate_name] = {
                "runtime_seconds": runtime,
                "metadata": raw["metadata"],
                "metrics": metrics,
                "gate_passed": bool(
                    hybrid_validation_passed
                    and metrics["total_signal"]["nrmse"] <= signal_nrmse_gate
                ),
                "hybrid_validation_passed": hybrid_validation_passed,
                "fallback_used": bool(raw["metadata"]["hybrid_fallback_used"]),
                "repeatability": {
                    "run_count": 1,
                    "note": "the CPU-corrected hybrid candidate was run once",
                },
            }
        else:
            raise ValueError(f"unknown precision candidate {candidate_name!r}")
    return report


def _load_inputs(args):
    project = None
    if args.input.suffix.lower() == ".blochproj":
        project = load_project(args.input)
        program = project["program"]
        if program is None:
            raise ValueError("the project does not contain a sequence program")
    else:
        program = load_pulseq(args.input)
    if args.synthetic:
        phantom = _synthetic_phantom(program, args.synthetic_shape)
    elif args.phantom is not None:
        phantom = DynamicSpectralPhantom.load(args.phantom)
    elif project is not None and isinstance(
        project.get("phantom"), DynamicSpectralPhantom
    ):
        phantom = project["phantom"]
    else:
        phantom = _synthetic_phantom(program, args.synthetic_shape)
    saved_reference = None if project is None else project.get("sequence_result")
    return program, phantom, saved_reference


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Pulseq .seq or .blochproj input")
    parser.add_argument(
        "--phantom",
        type=Path,
        help="Saved DynamicSpectralPhantom; project phantoms are used by default",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the small synthetic phantom even when a project contains one",
    )
    parser.add_argument(
        "--synthetic-shape",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--timestep-us", type=float, default=10.0)
    parser.add_argument(
        "--max-adc-events",
        type=int,
        help="Run only through the selected chronological ADC event",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=(
            "cpu_float32",
            "metal_probe",
            "metal_double_single",
            "metal_hybrid",
        ),
        help="Candidate to validate; repeat to test both (default: cpu_float32)",
    )
    parser.add_argument("--metal-repeat-runs", type=int, default=3)
    parser.add_argument(
        "--subvoxel-spins",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help=(
            "Midpoint subvoxel grid. The hybrid candidate requires at least "
            "three total spins, for example 2 2 2."
        ),
    )
    parser.add_argument("--hybrid-calibration-fraction", type=float, default=0.10)
    parser.add_argument("--hybrid-validation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--hybrid-no-fallback",
        action="store_true",
        help="Raise instead of returning the trusted Float64 CPU result on failure",
    )
    parser.add_argument(
        "--hybrid-sequential",
        action="store_true",
        help="Run the GPU and the two CPU samples sequentially for diagnostics",
    )
    parser.add_argument(
        "--metal-spin-chunk-size",
        default="auto",
        help="GPU spin chunk size for the hybrid probe (default: auto)",
    )
    parser.add_argument(
        "--signal-nrmse-gate", type=float, default=DEFAULT_SIGNAL_NRMSE_GATE
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    program, phantom, saved_reference = _load_inputs(args)
    program = _truncate_at_adc_event(program, args.max_adc_events)
    metal_spin_chunk_size = args.metal_spin_chunk_size
    if metal_spin_chunk_size != "auto":
        metal_spin_chunk_size = int(metal_spin_chunk_size)
    report = compare_precision(
        program,
        phantom,
        simulation_timestep_s=args.timestep_us * 1e-6,
        candidates=tuple(args.candidate or ("cpu_float32",)),
        metal_repeat_runs=args.metal_repeat_runs,
        signal_nrmse_gate=args.signal_nrmse_gate,
        spin_sampling=args.subvoxel_spins,
        hybrid_calibration_fraction=args.hybrid_calibration_fraction,
        hybrid_validation_fraction=args.hybrid_validation_fraction,
        hybrid_fallback_to_cpu=not args.hybrid_no_fallback,
        hybrid_run_concurrently=not args.hybrid_sequential,
        metal_spin_chunk_size=metal_spin_chunk_size,
    )
    if saved_reference is not None and args.max_adc_events is None:
        report["saved_project_reference_present"] = True
        report["saved_project_reference_signal_shape"] = list(
            np.asarray(saved_reference.signal).shape
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
