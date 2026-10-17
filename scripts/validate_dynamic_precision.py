"""Validate dynamic sequence kernels against the optimized Float64 oracle.

Inputs may be Pulseq ``.seq`` files or self-contained ``.blochproj`` projects.
The validator reports total/species signal, phase, low-signal, time-resolved
growth, final pool state, and optional reconstruction metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.kernel_capabilities import DYNAMIC_KERNEL_NAMES
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.project_io import load_project
from blochsimulator.sequence import SequenceProgram, load_pulseq
from blochsimulator.spectral_phantom import ChemicalSpecies


DEFAULT_SIGNAL_NRMSE_GATE = 1.0e-10
DEFAULT_RELATIVE_TOLERANCE = 1.0e-10
DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-12
SIGNIFICANT_MAGNITUDE_FRACTION = 1.0e-3
LOW_SIGNAL_MAGNITUDE_FRACTION = 1.0e-4
REFERENCE_KERNEL = "optimized"
CANDIDATE_KERNELS = tuple(
    name for name in DYNAMIC_KERNEL_NAMES if name != REFERENCE_KERNEL
)
DEFAULT_CANDIDATES = ("native_parallel",)


def _array_metric(
    reference,
    candidate,
    *,
    rtol=DEFAULT_RELATIVE_TOLERANCE,
    atol=DEFAULT_ABSOLUTE_TOLERANCE,
):
    if reference is None and candidate is None:
        return None
    if reference is None or candidate is None:
        return {"comparison_available": False, "within_tolerance": False}
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
    absolute_tolerance = atol * max(1.0, peak)
    max_absolute = float(np.max(np.abs(difference), initial=0.0))
    return {
        "within_tolerance": bool(
            np.all(np.isfinite(reference))
            and np.all(np.isfinite(candidate))
            and np.all(
                np.abs(difference) <= absolute_tolerance + rtol * np.abs(reference)
            )
        ),
        "shape": list(reference.shape),
        "reference_dtype": reference.dtype.name,
        "candidate_dtype": candidate.dtype.name,
        "absolute_tolerance": absolute_tolerance,
        "reference_rms": reference_norm / np.sqrt(max(1, reference.size)),
        "rms_absolute_error": difference_norm / np.sqrt(max(1, reference.size)),
        "nrmse": (
            difference_norm / reference_norm if reference_norm else difference_norm
        ),
        "max_absolute_error": max_absolute,
        "reference_peak": peak,
        "max_error_relative_to_reference_peak": (
            max_absolute / peak if peak else max_absolute
        ),
    }


def _signal_metric(reference, candidate, **tolerances):
    metric = _array_metric(reference, candidate, **tolerances)
    if metric is None or reference is None or candidate is None:
        return metric
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


def _error_growth(reference, candidate, adc_times_s, bin_count=10, **tolerances):
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
                "window": _signal_metric(
                    reference[window], candidate[window], **tolerances
                ),
                "cumulative": _signal_metric(
                    reference[cumulative], candidate[cumulative], **tolerances
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


def _result_metrics(reference, candidate, **tolerances):
    pool_names = tuple(reference.pool_names) or tuple(
        f"pool_{index}" for index in range(reference.species_signal.shape[0])
    )
    metrics = {
        "total_signal": _signal_metric(
            reference.signal, candidate.signal, **tolerances
        ),
        "species_signal": {
            name: _signal_metric(
                reference.species_signal[index],
                candidate.species_signal[index],
                **tolerances,
            )
            for index, name in enumerate(pool_names)
        },
        "error_growth": _error_growth(
            reference.signal,
            candidate.signal,
            reference.adc_times_s,
            **tolerances,
        ),
    }
    for label, attribute in (
        ("final_pool_state", "final_pool_magnetization"),
        ("final_combined_state", "final_magnetization"),
        ("checkpoint_pool_state", "checkpoint_pool_magnetization"),
        ("checkpoint_combined_state", "checkpoint_magnetization"),
    ):
        metrics[label] = _array_metric(
            getattr(reference, attribute),
            getattr(candidate, attribute),
            **tolerances,
        )
    metrics["final_species_states"] = {
        name: _array_metric(
            reference.final_pool_magnetization[index],
            candidate.final_pool_magnetization[index],
            **tolerances,
        )
        for index, name in enumerate(pool_names)
    }
    metrics["checkpoint_species_states"] = {
        name: _array_metric(
            (
                None
                if reference.checkpoint_pool_magnetization is None
                else reference.checkpoint_pool_magnetization[:, index]
            ),
            (
                None
                if candidate.checkpoint_pool_magnetization is None
                else candidate.checkpoint_pool_magnetization[:, index]
            ),
            **tolerances,
        )
        for index, name in enumerate(pool_names)
    }
    if reference.cartesian_acquisition_volumes is not None:
        metrics["cartesian_3d_kspace"] = _array_metric(
            reference.to_cartesian_3d_kspace(),
            candidate.to_cartesian_3d_kspace(),
            **tolerances,
        )
        metrics["cartesian_3d_image"] = _array_metric(
            reference.reconstruct_cartesian_3d(),
            candidate.reconstruct_cartesian_3d(),
            **tolerances,
        )
    return metrics


def _failed_metrics(metrics, prefix=""):
    """Collect failed array comparisons, including pools and time windows."""
    failures = []
    if isinstance(metrics, dict):
        if metrics.get("within_tolerance") is False:
            failures.append(prefix)
        for key, value in metrics.items():
            failures.extend(_failed_metrics(value, f"{prefix}.{key}".strip(".")))
    elif isinstance(metrics, list):
        for index, value in enumerate(metrics):
            failures.extend(_failed_metrics(value, f"{prefix}[{index}]"))
    return failures


def compare_kernels(
    program,
    phantom,
    *,
    simulation_timestep_s,
    candidates=DEFAULT_CANDIDATES,
    signal_nrmse_gate=DEFAULT_SIGNAL_NRMSE_GATE,
    spin_sampling=None,
    spoiler_mode=None,
    checkpoints_s=None,
    rtol=DEFAULT_RELATIVE_TOLERANCE,
    atol=DEFAULT_ABSOLUTE_TOLERANCE,
    native_variants=False,
    refined_timestep_s=None,
):
    from blochsimulator.sequence.spin_sampling import coerce_spin_sampling

    for value, name in (
        (rtol, "rtol"),
        (atol, "atol"),
        (signal_nrmse_gate, "signal_nrmse_gate"),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for candidate_name in candidates:
        if candidate_name not in CANDIDATE_KERNELS:
            raise ValueError(f"unknown candidate kernel {candidate_name!r}")
    sampling = coerce_spin_sampling(spin_sampling)
    mode = spoiler_mode or ("gradient" if sampling.enabled else "ideal")
    if sampling.enabled and mode == "ideal":
        raise ValueError("subvoxel sampling requires spoiler_mode='gradient'")
    if refined_timestep_s is not None and not (
        0 < refined_timestep_s < simulation_timestep_s
    ):
        raise ValueError(
            "refined timestep must be positive and smaller than the baseline"
        )
    if checkpoints_s is None:
        checkpoints_s = (program.duration_s / 2, program.duration_s)

    def run(kernel, *, timestep=simulation_timestep_s, **extra):
        simulator = BlochSimulator(
            use_parallel=kernel == "native_parallel",
            dynamic_sequence_kernel=kernel,
        )
        start = time.perf_counter()
        result = simulator.simulate_dynamic_sequence(
            program,
            phantom,
            simulation_timestep_s=timestep,
            spin_sampling=sampling,
            spoiler_mode=mode,
            checkpoints_s=checkpoints_s,
            **extra,
        )
        return result, time.perf_counter() - start

    reference, reference_seconds = run(REFERENCE_KERNEL)
    report = {
        "duration_s": float(program.duration_s),
        "active_voxels": int(phantom.n_active),
        "adc_samples": int(reference.adc_times_s.size),
        "simulation_timestep_s": float(simulation_timestep_s),
        "signal_nrmse_gate": float(signal_nrmse_gate),
        "relative_tolerance": float(rtol),
        "absolute_tolerance": float(atol),
        "spoiler_mode": mode,
        "requested_spin_sampling": sampling.to_metadata(),
        "spin_sampling": reference.metadata["spin_sampling"],
        "reference": {
            "kernel": REFERENCE_KERNEL,
            "runtime_seconds": reference_seconds,
            "execution_plan_fingerprint": reference.metadata.get(
                "execution_plan_fingerprint"
            ),
            "compiled_interval_count": reference.metadata.get(
                "compiled_interval_count"
            ),
        },
        "candidates": {},
    }
    variants = [(name, name, {}) for name in candidates]
    if native_variants:
        for name in candidates:
            if name in {"native_serial", "native_parallel"}:
                variants.extend(
                    (
                        (name + "_previous", name, {"_native_spin_major": False}),
                        (
                            name + "_stepwise",
                            name,
                            {"_native_spin_major_rf_cache": False},
                        ),
                    )
                )
    for label, kernel, extra in variants:
        candidate, runtime = run(kernel, **extra)
        metrics = _result_metrics(reference, candidate, rtol=rtol, atol=atol)
        failures = _failed_metrics(metrics)
        for metric_name, metric in [
            ("total_signal", metrics["total_signal"]),
            *metrics["species_signal"].items(),
        ]:
            # Retain the absolute-error floor for vanishing pool signals.
            if not np.isfinite(metric["rms_absolute_error"]) or (
                metric["rms_absolute_error"]
                > metric["absolute_tolerance"]
                + signal_nrmse_gate * metric["reference_rms"]
            ):
                failures.append(metric_name + ".nrmse")
        actual = candidate.metadata.get("sequence_kernel")
        backend_match = actual == kernel
        spin_major = bool(candidate.metadata.get("native_spin_major_enabled"))
        if label.endswith("_stepwise"):
            backend_match = backend_match and spin_major
        elif label.endswith("_previous"):
            backend_match = backend_match and not spin_major
        fallback_reason = candidate.metadata.get("native_fallback_reason") or (
            candidate.metadata.get("native_spin_major_fallback_reason")
        )
        if fallback_reason == "disabled by request":
            fallback_reason = None
        report["candidates"][label] = {
            "runtime_seconds": runtime,
            "speedup_vs_reference": (
                reference_seconds / runtime if runtime > 0 else None
            ),
            "requested_kernel": kernel,
            "actual_kernel": actual,
            "requested_backend_executed": backend_match,
            "fallback_used": bool(fallback_reason or actual != kernel),
            "fallback_reason": fallback_reason,
            "metadata": candidate.metadata,
            "metrics": metrics,
            "failed_metrics": failures,
            "accuracy_gate_passed": not failures,
            "gate_passed": not failures and backend_match,
        }
    if refined_timestep_s is not None:
        refined, runtime = run(REFERENCE_KERNEL, timestep=refined_timestep_s)
        report["timestep_refinement"] = {
            "timestep_s": float(refined_timestep_s),
            "runtime_seconds": runtime,
            "compiled_interval_count": refined.metadata["compiled_interval_count"],
            "compiled_grid_changed": refined.metadata["execution_plan_fingerprint"]
            != reference.metadata["execution_plan_fingerprint"],
            "metrics": _result_metrics(refined, reference, rtol=rtol, atol=atol),
        }
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
        choices=CANDIDATE_KERNELS,
        help="Dynamic kernel to validate; repeatable (default: native_parallel)",
    )
    parser.add_argument(
        "--subvoxel-spins",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="Midpoint subvoxel grid, for example 2 2 2.",
    )
    parser.add_argument(
        "--signal-nrmse-gate", type=float, default=DEFAULT_SIGNAL_NRMSE_GATE
    )
    parser.add_argument("--rtol", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        help="Absolute tolerance scaled by max(1, each array's reference peak)",
    )
    parser.add_argument("--spoiler-mode", choices=("ideal", "gradient"))
    parser.add_argument("--checkpoints-s", nargs="*", type=float)
    parser.add_argument("--compare-native-variants", action="store_true")
    parser.add_argument("--refined-timestep-us", type=float)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    program, phantom, saved_reference = _load_inputs(args)
    program = _truncate_at_adc_event(program, args.max_adc_events)
    report = compare_kernels(
        program,
        phantom,
        simulation_timestep_s=args.timestep_us * 1e-6,
        candidates=tuple(args.candidate or DEFAULT_CANDIDATES),
        signal_nrmse_gate=args.signal_nrmse_gate,
        spin_sampling=args.subvoxel_spins,
        spoiler_mode=args.spoiler_mode,
        checkpoints_s=args.checkpoints_s,
        rtol=args.rtol,
        atol=args.atol,
        native_variants=args.compare_native_variants,
        refined_timestep_s=(
            None
            if args.refined_timestep_us is None
            else args.refined_timestep_us * 1e-6
        ),
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
    return 0 if all(row["gate_passed"] for row in report["candidates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
