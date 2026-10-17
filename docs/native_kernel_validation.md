# Native kernel integration and validation

Date: 17 October 2026. Integration into the local `pre-release` working tree
uses `feat/spin-major-rf-cache` at `09815a67` and retains the newer
Float64-only API cleanup and bounded ADC-window/preview fixes.

## Implemented changes

- The native dynamic choices automatically use the strict Float64 spin-major
  executor for supported two-pool models. Repeated RF segments can use cached
  affine propagators. Both accelerators retain independent disable switches.
- Spatial complex Tx sensitivity is supported by the stepwise and cached
  executors. Local RF fields use the same complex multiplication and plane-basis
  rotation as the NumPy implementation. Null Tx values, phase variation,
  multiple receive coils, subvoxel sampling, and inflow models are covered.
- Native source/header/Cython files are included in source distributions and
  the new extension is part of wheel import checks. Runtime capability checks
  reject stale extension APIs and safely retain the previous native driver.
- Validation uses the requested subvoxel grid, checks individual pools,
  final states, checkpoints, and available reconstructions, and reports
  requested versus actual execution. Failed checks return exit status 1.

Dynamic B0, spatial inflow delays, additional pools, and smaller checkpoint
dtypes still use the previous execution paths. Integration arithmetic remains
Float64; this work does not restore Metal or Float32 simulation.

## Accuracy criteria

For every compared array, the default elementwise requirement is

$$
|x_{\mathrm{candidate}}-x_{\mathrm{reference}}|
\le a A+r|x_{\mathrm{reference}}|,
\qquad a=10^{-12},\quad r=10^{-10}.
$$

Here $A=\max(1,\max_i|x_{\mathrm{reference},i}|)$ is the reference amplitude
in the stored units. The report records the effective absolute tolerance
$aA$ for every array. Final and checkpoint states also have separate checks
for each pool, preventing a dominant pool from setting another pool's scale.

The total signal and each metabolite also require

$$
\mathrm{RMS}(x_{\mathrm{candidate}}-x_{\mathrm{reference}})
\le a A+g\,\mathrm{RMS}(x_{\mathrm{reference}}),
\qquad g=10^{-10}.
$$

The absolute floor matters for nearly cancelled signals. For example, the
small supplied-sequence prefix produced errors around $8\times10^{-17}$
at a signal peak around $7.5\times10^{-7}$; rejecting such a comparison only
because NRMSE slightly exceeds $10^{-10}$ would ignore the absolute tolerance.
Raw relative errors remain available in the report. The supplied full-object
signal peaks above three million in its stored units. Scaling the absolute
floor is necessary for meaningful comparisons at signal cancellation and
across different concentration or signal-weighting units.

These are implementation-parity criteria. Timestep refinement is reported
separately and does not use this Float64-parity gate. Reports compare execution
plan fingerprints to detect an unchanged integration grid even when smaller
requested timesteps remain below the waveform raster.

## Verification completed

- The final project-defined full test suite passed 1,070 tests and 12 subtests,
  with one skipped test, in the current Conda environment: Python 3.12.15,
  NumPy 2.5.4, and SciPy 1.18.1. An earlier full run passed 1,066 tests in the
  packaging environment before the final validator regression tests were added.
- Tests cover spatial Tx, cache switches, thread and slice determinism,
  capability reporting, and stale-extension fallback. A requested spin-major
  variant cannot pass validation by silently using the previous driver.
- Formatting and focused lint checks passed.
- Source-archive contents match the native working sources. An isolated import
  loaded the built extension from the extracted wheel and confirmed API version
  2, spatial Tx support, and the corresponding capability flags.
- Validator CLI checks exercised actual $2\times2\times2$ sampling, native
  variants, timestep refinement, successful exit, and failed exit with zero
  tolerance. Controlled opposing pool errors and wrong endpoint/checkpoint
  states are rejected.

The bounded ADC-window regression explicitly selects the previous native
driver. It still requires bitwise-identical signals across time windows;
the new spin-major tests separately cover its deterministic reductions.

## Full-object acquisition comparison

Input: the supplied `phantoms/phantoms/Pyruvate to lactate with inflow.npz`,
with shape $64^3$, 39,074 active voxels, and the complete checked-in
`bssfp_3d_spectral_selective_skinner.seq`. The sequence contains 20 dynamic
frames, lasts about 24.2994 s, and compiles to 1,041,538 intervals at a
10 microsecond RF timestep. Checkpoints are at half duration and sequence end.
No object dimensions, voxel positions, pulse events, or acquisition samples
are reduced for this comparison.

Runs use Python from `.venv-packaging`, NumPy 2.3.5, SciPy 1.16.3, and eight
native threads. Wall times include sequence compilation and plan creation;
peak RSS includes the full process. The long NumPy reference and other
verification work overlap these measurements, so they are approximate
observations rather than isolated performance benchmarks.

| Execution | Wall time | Peak RSS |
| --- | ---: | ---: |
| Spin-major with RF propagators | 79.25 s | 0.905 GB |
| Spin-major without RF propagators | 257.26 s | 1.071 GB |
| Previous native driver | 2,421.21 s | 0.889 GB |
| Spin-major with RF propagators, 5 microseconds | 172.41 s | 0.976 GB |
| Spin-major with RF propagators, 1 microsecond | 286.78 s | 1.627 GB |

The four-process NumPy reference took 2,471.82 s. The sum of its individual
worker peak RSS values was 4.218 GB. That sum excludes the coordinator and is
not a simultaneous process-group memory measurement, so it is not directly
comparable with the single-process RSS values above.

The cached run used 18 propagator keys/builds, 140,148,736 cache bytes,
3,820 cached RF occurrences, and 40 stepwise RF occurrences. Its measured
native time was 43.44 s and host-table preparation took 1.70 s.

The NumPy reference uses four independent active-voxel blocks containing
9,769, 9,769, 9,768, and 9,768 voxels. Each block executes the complete sequence
with the original physical coordinates. Signals are combined in fixed block
order; voxel states retain their original global indices. A small heterogeneous
reference case with dynamic B0, spatial Tx, multiple coils, and subvoxel spins
checks this partition against a whole-object NumPy run: endpoint and checkpoint
states have zero numerical difference, and the maximum signal difference is
below $1.15\times10^{-16}$. The initial unpartitioned reference was stopped and
replaced with this complete parallel reference. All 39,074 active voxels are
included.

All full-object comparisons passed the mixed absolute/relative criteria,
including per-pool signals, endpoint states, checkpoints, time windows, and
complex reconstructed images. Phase differences are additionally reported as
diagnostics. Signal NRMSE against the complete NumPy reference at the same
10 microsecond grid is:

| Execution | Pyruvate | Lactate | Combined signal |
| --- | ---: | ---: | ---: |
| Spin-major with RF propagators | $1.05\times10^{-11}$ | $1.03\times10^{-11}$ | $1.05\times10^{-11}$ |
| Spin-major without RF propagators | $2.97\times10^{-14}$ | $1.82\times10^{-13}$ | $3.19\times10^{-14}$ |
| Previous native driver | $1.41\times10^{-13}$ | $2.15\times10^{-13}$ | $1.45\times10^{-13}$ |

The cached endpoint and checkpoint pool-state NRMSE values are
$1.14\times10^{-11}$ and $1.13\times10^{-11}$. The cache introduces a small
additional rounding difference; its disable switch remains available for
diagnosis.

## Timestep accuracy

The full object was also simulated with cached spin-major execution at 5 and
1 microsecond timesteps, producing 1,940,918 and 9,135,958 intervals. Relative
signal RMS differences, expressed as percentages and normalized by the finer
reference, are:

| Comparison | Pyruvate | Lactate |
| --- | ---: | ---: |
| 10 versus 5 microseconds | 0.00528% | 0.07271% |
| 10 versus 1 microsecond | 0.00696% | 0.09601% |
| 5 versus 1 microsecond | 0.00169% | 0.02330% |

The corresponding per-pool complex-image NRMSE values agree with the signal
values at the displayed precision. These differences measure discretization
effects and are much larger than the kernel implementation differences. They
are reported separately from the Float64 parity checks.

For quantitative work on this sequence, prefer the existing **Genau** preset
with a 1 microsecond RF timestep. The application defaults and presets have
not been changed. This refinement study does not establish an exact physical
solution or prove convergence below the sequence's RF waveform raster.

An additional independent NumPy comparison at 1 microsecond used 64 active
voxels spanning the supplied mask and the first 256 ADC events, preserving
their physical coordinates and original waveforms. It exercised 236 cached
RF occurrences and both excitation targets. Every signal and state check
passed; Pyruvate and Lactate signal NRMSE were $2.32\times10^{-12}$ and
$3.30\times10^{-11}$. This supplementary check complements the complete
10 microsecond reference and full-object refinement study.

## Controlled performance comparison

After the long computations ended, separate benchmarks used 4,096 voxels,
64 repeated RF pulses with 64 samples each, two metabolite pools, conversion,
tracked concentrations, polarized inflow, gradients, 256 ADC events, and two
checkpoints. Each method ran a warmup followed by five timed repetitions in
the current Conda environment, with eight native threads. Median times were:

| Execution | Uniform Tx | Spatial complex Tx |
| --- | ---: | ---: |
| Optimized NumPy | 1,691.63 ms | 1,360.87 ms |
| Previous native driver | 351.57 ms | 429.24 ms |
| Spin-major without RF propagators | 126.08 ms | 154.12 ms |
| Spin-major with RF propagators | 91.52 ms | 93.74 ms |

The cached executor is 3.84 times faster than the previous native driver with
uniform Tx and 4.58 times faster with spatial complex Tx in these workloads.
Each cached run built two keys and used 7,340,032 cache bytes. All compared
signals and states passed the Float64 criteria against NumPy. Performance
depends on sequence repetition, object size, and supported model features;
these measurements are not a universal speedup claim.

Exact measurements and environment versions are retained in the
[machine-readable summary](native_kernel_validation_summary.json).

## Reproduction

From the project root, with the built native extensions available:

```bash
python scripts/validate_dynamic_precision.py \
  sequences/sequences/bssfp_3d_spectral_selective_skinner.seq \
  --phantom "phantoms/phantoms/Pyruvate to lactate with inflow.npz" \
  --timestep-us 10 --candidate native_parallel \
  --compare-native-variants --refined-timestep-us 5 \
  --output-json /tmp/dynamic_kernel_validation.json
```

This command uses an unpartitioned optimized NumPy reference and also runs
its refined reference in NumPy. For quick diagnostic runs,
`--max-adc-events` selects an explicit acquisition prefix. The full-object
comparison above uses separate processes to measure peak RSS per variant.
