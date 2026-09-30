"""Reproduce the small CPU/B1 comparison in docs/simulation_acceleration_review.md.

Run from the repository root with the bloch-simulator environment:
    python -m benchmarks.benchmark_acceleration_review --output /tmp/bloch_review.json

This measures a synthetic workload, including sequence compilation, and is not
a production speedup estimate. BLAS workers are limited independently of OpenMP.
"""

import argparse
import cProfile
import ctypes
import io
import json
import os
import platform
import pstats
import resource
import sys
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
import scipy
from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies
import blochsimulator.dynamic_bloch_cy as cy
from benchmarks.common import result_similarity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="/tmp/bloch_acceleration_review_bench.json")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
if args.repeats < 3:
    parser.error("--repeats must be at least 3")

shape = (16, 16, 16)
duration = 0.08
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
    kpl_map_s_inv=np.linspace(0.03, 0.1, np.prod(shape)).reshape(shape),
    nucleus="C13",
)
phantom.initial_spin_density_maps = {
    "Pyruvate": np.ones(shape),
    "Lactate": np.full(shape, 0.05),
}
phantom.equilibrium_polarization = 0.01
phantom.pyruvate_inflow = PyruvateInflow(
    TimeCurve((0.0, duration), (0.2, 0.5), "linear", "hold"),
    np.ones(shape),
    TimeCurve((0.0, duration), (0.4, 0.25), "linear", "hold"),
)
raster = 20e-6
events = []
waveform = 110 * np.exp(-np.linspace(-2.5, 2.5, 64) ** 2).astype(complex)
for tr in range(8):
    start = tr * 0.01
    events.extend(
        (
            RFEvent(start, waveform * ((-1) ** tr), raster),
            GradientEvent("z", start, np.full(64, 120.0), raster),
            ADCEvent(start + 0.002, 32, raster),
        )
    )
program = SequenceProgram(tuple(events), duration_s=duration)
kwargs = dict(simulation_timestep_s=20e-6)
metadata_keys = (
    "sequence_kernel",
    "native_fallback_reason",
    "native_rf_block_enabled",
    "native_spatial_rf_block_enabled",
    "tx_sensitivity",
    "numerical_contract",
    "native_rf_fused_block_enabled",
    "native_rf_fused_fallback_reason",
    "native_rf_fused_blocks",
    "native_rf_fused_intervals",
    "native_rf_threads",
    "native_longitudinal_threads",
    "compiled_interval_count",
    "n_simulated_spins",
    "native_full_plan_enabled",
    "native_full_plan_model",
    "native_full_plan_blocks",
    "native_full_plan_intervals",
    "native_spatial_rf_cache_hits",
    "native_spatial_rf_cache_misses",
    "native_spatial_rf_cache_bytes",
    "native_rf_fused_cache_bytes",
    "native_rf_fused_numerical_contract",
    "sequence_compile_s",
    "execution_plan_build_s",
    "execution_plan_bytes",
    "execution_plan_segment_count",
    "execution_plan_action_count",
    "execution_plan_fingerprint",
    "native_spatial_rf_table_limit_bytes",
    "native_spatial_rf_max_groups_per_block",
)
info = dict(
    python=sys.version,
    executable=sys.executable,
    numpy=np.__version__,
    scipy=scipy.__version__,
    platform=platform.platform(),
    shape=shape,
    cpu=platform.processor(),
    logical_cpu_count=os.cpu_count(),
    rf_intervals_nominal=512,
    adc_samples=256,
    repeats=args.repeats,
    thread_environment={
        name: os.environ[name]
        for name in (
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_DYNAMIC",
        )
    },
    blas=np.__config__.show(mode="dicts").get("Build Dependencies", {}).get("blas"),
    dynamic_extension=dict(
        path=cy.__file__,
        mtime_ns=os.stat(cy.__file__).st_mtime_ns,
        size_bytes=os.stat(cy.__file__).st_size,
        numerical_contract="strict-float64",
    ),
)
try:
    info["openmp_max_threads_before_runs"] = ctypes.CDLL(
        cy.__file__
    ).omp_get_max_threads()
except Exception as exc:
    info["openmp_check"] = str(exc)
rows = []
profiles = {}
for tx_case in ("unity", "uniform_0.9", "spatial"):
    phantom.tx_sensitivity_map = (
        np.ones(shape, dtype=complex)
        if tx_case == "unity"
        else (
            np.full(shape, 0.9, dtype=complex)
            if tx_case == "uniform_0.9"
            else (
                np.linspace(0.8, 1.2, np.prod(shape))
                * np.exp(1j * np.linspace(-0.1, 0.1, np.prod(shape)))
            ).reshape(shape)
        )
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=4)
    reference = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="reference", **kwargs
    )
    for kernel, threads in (
        ("optimized", 1),
        ("native_serial", 1),
        ("native_parallel", 4),
        ("native_parallel", 8),
    ):
        simulator = BlochSimulator(use_parallel=True, num_threads=threads)
        simulator.simulate_dynamic_sequence(
            program, phantom, sequence_kernel=kernel, **kwargs
        )
        times = []
        for _ in range(args.repeats):
            start = perf_counter()
            result = simulator.simulate_dynamic_sequence(
                program, phantom, sequence_kernel=kernel, **kwargs
            )
            times.append(perf_counter() - start)
        errors = result_similarity(reference, result)
        row = dict(
            tx=tx_case,
            kernel=kernel,
            threads=threads,
            seconds=times,
            median_s=float(np.median(times)),
            mean_s=float(np.mean(times)),
            standard_deviation_s=float(np.std(times, ddof=1)),
            minimum_s=float(np.min(times)),
            maximum_s=float(np.max(times)),
            errors_vs_same_tx_reference=errors,
            metadata={key: result.metadata.get(key) for key in metadata_keys},
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.profile and kernel == "native_parallel" and threads == 4:
            profiler = cProfile.Profile()
            profiler.runcall(
                simulator.simulate_dynamic_sequence,
                program,
                phantom,
                sequence_kernel=kernel,
                **kwargs,
            )
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(18)
            profiles[tx_case] = stream.getvalue()
max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
info["process_max_rss_bytes"] = int(
    max_rss if platform.system() == "Darwin" else max_rss * 1024
)
output = dict(environment=info, results=rows, profiles=profiles)
with open(args.output, "w") as handle:
    json.dump(output, handle, indent=2)
print(json.dumps(info), flush=True)
for case, profile in profiles.items():
    print(case, profile, flush=True)
