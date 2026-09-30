"""Benchmark the bounded native two-pool execution-plan driver.

Run from the repository root with the project environment::

    python -m benchmarks.benchmark_dynamic_native_plan --repeats 5 \
        --output docs/simulation_acceleration_native_plan_benchmark.json
"""

import argparse
import json
import os
import platform
import resource
import sys
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
import scipy

from benchmarks.common import result_similarity
from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies
import blochsimulator.dynamic_bloch_cy as dynamic_extension


def _fixture(tx_case):
    shape = (16, 16, 16)
    voxel_count = int(np.prod(shape))
    spatial = np.linspace(0.0, 1.0, voxel_count).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.02, 0.02),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.full(shape, 0.05),
        },
        kpl_map_s_inv=0.02 + 0.08 * spatial,
        b0_map=-8.0 + 16.0 * spatial,
        conversion_start_s=0.015,
        nucleus="C13",
    )
    if tx_case == "constant_complex":
        phantom.tx_sensitivity_map = np.full(shape, 0.85 + 0.2j)
    elif tx_case == "spatial":
        phantom.tx_sensitivity_map = (0.75 + 0.4 * spatial) * np.exp(
            1j * (-0.15 + 0.3 * spatial)
        )

    raster = 20e-6
    events = []
    rf = 95.0 * np.exp(-np.linspace(-2.0, 2.0, 16) ** 2)
    for repetition in range(8):
        start = repetition * 2e-3
        events.extend(
            (
                RFEvent(start, rf * np.exp(0.25j * repetition), raster),
                GradientEvent(
                    "x",
                    start,
                    np.linspace(-180.0, 220.0, 80),
                    raster,
                ),
                ADCEvent(start + 20 * raster, 48, raster, phase_offset_rad=0.17),
            )
        )
    program = SequenceProgram(tuple(events), duration_s=0.016)
    return phantom, program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/dynamic_native_plan_benchmark.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    rows = []
    for tx_case in ("unity", "constant_complex", "spatial"):
        phantom, program = _fixture(tx_case)
        reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
            program,
            phantom,
            sequence_kernel="reference",
            simulation_timestep_s=20e-6,
            checkpoints_s=(0.008, 0.016),
        )
        for kernel, threads in (
            ("optimized", 1),
            ("native_serial", 1),
            ("native_parallel", 4),
            ("native_parallel", 8),
        ):
            simulator = BlochSimulator(use_parallel=True, num_threads=threads)
            kwargs = dict(
                sequence_kernel=kernel,
                simulation_timestep_s=20e-6,
                checkpoints_s=(0.008, 0.016),
            )
            simulator.simulate_dynamic_sequence(program, phantom, **kwargs)
            samples = []
            for _ in range(args.repeats):
                started = perf_counter()
                result = simulator.simulate_dynamic_sequence(program, phantom, **kwargs)
                samples.append(perf_counter() - started)
            row = {
                "tx": tx_case,
                "kernel": kernel,
                "threads": threads,
                "seconds": samples,
                "median_s": float(np.median(samples)),
                "mean_s": float(np.mean(samples)),
                "standard_deviation_s": float(np.std(samples, ddof=1)),
                "minimum_s": float(np.min(samples)),
                "maximum_s": float(np.max(samples)),
                "errors_vs_same_tx_reference": result_similarity(reference, result),
                "metadata": {
                    key: result.metadata.get(key)
                    for key in (
                        "sequence_kernel",
                        "numerical_contract",
                        "compiled_interval_count",
                        "n_simulated_spins",
                        "n_rx_coils",
                        "tx_sensitivity",
                        "native_parallel_threads",
                        "native_full_plan_enabled",
                        "native_full_plan_model",
                        "native_full_plan_blocks",
                        "native_full_plan_intervals",
                        "execution_plan_segment_count",
                        "execution_plan_action_count",
                        "execution_plan_bytes",
                        "sequence_compile_s",
                        "execution_plan_build_s",
                    )
                },
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "repeats": args.repeats,
        "dynamic_extension": {
            "path": dynamic_extension.__file__,
            "mtime_ns": os.stat(dynamic_extension.__file__).st_mtime_ns,
            "size_bytes": os.stat(dynamic_extension.__file__).st_size,
            "numerical_contract": "strict-float64",
        },
        "process_max_rss_bytes": int(
            maximum_rss if platform.system() == "Darwin" else maximum_rss * 1024
        ),
        "thread_environment": {
            key: os.environ[key]
            for key in (
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_DYNAMIC",
            )
        },
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"environment": environment, "results": rows}, handle, indent=2)
    print(json.dumps(environment), flush=True)


if __name__ == "__main__":
    main()
