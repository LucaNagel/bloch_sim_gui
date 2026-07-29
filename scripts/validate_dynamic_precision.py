"""Compare dynamic float32 execution with the strict float64 CPU path.

The script can use a saved dynamic phantom or create a small heterogeneous
synthetic phantom from the sequence FOV.  It reports scale-normalized errors
for every public dynamic result array and, when the complete Cartesian metadata
is available, for reconstructed 3D k-space and images as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.sequence import SequenceProgram, load_pulseq
from blochsimulator.spectral_phantom import ChemicalSpecies


RESULT_ARRAYS = (
    "signal",
    "species_signal",
    "final_magnetization",
    "final_pool_magnetization",
    "checkpoint_magnetization",
    "checkpoint_pool_magnetization",
)


def _metric(reference, candidate):
    if reference is None or candidate is None:
        return None
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
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
        "max_error_relative_to_peak": max_absolute / peak if peak else max_absolute,
    }


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


def compare_precision(program, phantom, *, simulation_timestep_s, checkpoints_s=()):
    results = {}
    timings = {}
    for precision in ("float64", "float32"):
        simulator = BlochSimulator(
            use_parallel=False,
            dynamic_sequence_kernel="optimized",
            dynamic_sequence_precision=precision,
        )
        start = time.perf_counter()
        results[precision] = simulator.simulate_dynamic_sequence(
            program,
            phantom,
            checkpoints_s=checkpoints_s,
            simulation_timestep_s=simulation_timestep_s,
        )
        timings[precision] = time.perf_counter() - start

    reference = results["float64"]
    candidate = results["float32"]
    metrics = {
        name: _metric(getattr(reference, name), getattr(candidate, name))
        for name in RESULT_ARRAYS
    }
    if reference.cartesian_acquisition_volumes is not None:
        metrics["cartesian_3d_kspace"] = _metric(
            reference.to_cartesian_3d_kspace(), candidate.to_cartesian_3d_kspace()
        )
        metrics["cartesian_3d_image"] = _metric(
            reference.reconstruct_cartesian_3d(),
            candidate.reconstruct_cartesian_3d(),
        )
    return {
        "duration_s": float(program.duration_s),
        "active_voxels": int(phantom.n_active),
        "adc_samples": int(reference.adc_times_s.size),
        "simulation_timestep_s": float(simulation_timestep_s),
        "timings_s": timings,
        "metrics": metrics,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="Pulseq .seq input")
    parser.add_argument(
        "--phantom",
        type=Path,
        help="Saved DynamicSpectralPhantom; omit for a synthetic phantom",
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
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    program = load_pulseq(args.sequence)
    program = _truncate_at_adc_event(program, args.max_adc_events)
    phantom = (
        DynamicSpectralPhantom.load(args.phantom)
        if args.phantom is not None
        else _synthetic_phantom(program, args.synthetic_shape)
    )
    report = compare_precision(
        program,
        phantom,
        simulation_timestep_s=args.timestep_us * 1e-6,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
