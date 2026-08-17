# Dynamic sequence simulation performance

This document records the performance paths implemented for the coupled
pyruvate/lactate sequence simulator.  It complements the physical model in
[dynamic_phantom_design.md](dynamic_phantom_design.md) and the general data
flow in
[sequence_simulation_architecture.md](sequence_simulation_architecture.md).

The `optimized` path and the strict single-interval native primitives preserve
the reference implementation bit for bit.  The persistent RF-waveform block
uses the same float64 coefficients and integration topology but groups many
intervals inside one compiled loop.  Its different complex-arithmetic code
path can change the final few float64 bits, so it has a separate
`float64-close` contract with tight regression tolerances.  Result metadata
states which contract and native capabilities were used.

## Available execution paths

`simulate_dynamic_sequence` currently exposes four kernel selections:

- `reference` keeps the direct, allocation-heavy Python/NumPy formulation.  It
  is the correctness oracle for the other CPU paths and is intended mainly for
  tests and small diagnostic simulations.
- `optimized` keeps the same operation and summation order while reusing
  state and coefficient arrays.  It supports static and dynamic B0 as well as
  pyruvate inflow.
- `native_serial` uses the optimized Python driver together with compiled
  Cython primitives for longitudinal two-pool evolution and RF rotation.  The
  common Phantom Designer model with pyruvate inflow, a polarization curve,
  and explicit concentration tracking uses one fused magnetization/
  concentration primitive.  Other unsupported driver combinations retain the
  explicit optimized fallback.  Dynamic B0 does not by itself require a
  longitudinal fallback.
- `native_parallel` adds OpenMP voxel parallelism to the native block
  primitives.  Consecutive RF raster intervals in the common coupled model are
  executed as persistent waveform blocks, amortizing one OpenMP region over
  the complete RF waveform.  ADC summation remains in NumPy and retains the
  original active voxel order.  Small objects use one thread because OpenMP
  startup and synchronization cost more than the available parallel work.

The requested and actually used paths, complete and longitudinal-only fallback
reasons, enabled native capabilities, effective thread counts, and memory-limit
decisions are stored in the simulation result metadata.  The GUI exposes the
dynamic kernel in the performance settings and reports hybrid execution while
the simulation is running.

## NumPy driver optimizations

The first optimization stage reduced Python and allocation overhead without
changing the equations or floating-point evaluation order:

1. The transverse state is retained as a contiguous two-pool `complex128`
   array.  RF-free evolution updates it in place instead of repeatedly
   constructing `Mx + 1j*My` and copying it back to the six-component state.
2. Longitudinal exchange/relaxation coefficients are prepared once per unique
   half-step duration and kept in a bounded cache.  A small fixed scratch set
   replaces interval-sized temporary arrays.
   Concentration-state coefficients use the same cache and their own scratch
   arrays; older builds accidentally recomputed their exponential convolution
   coefficients at every half-step.
3. Transverse T2/phase factors are cached by duration, gradient, and dynamic-B0
   integral.  The cache is bounded so long sequences cannot grow memory in
   proportion to every interval/voxel combination.
4. Static frequencies and gradient projections are only built on cache misses.
   Repeated EPI and bSSFP motifs therefore reuse the already rounded factors.
5. Progress, preview, cancellation, checkpoint, and ADC observation semantics
   remain at the original interval boundaries.
6. With one spin per voxel and a uniform single receive coil, ADC observation
   sums the transverse state directly instead of first multiplying it by an
   all-ones weight array.

On a representative 64x64 EPI benchmark with a 16^3 phantom and 1,472 active
voxels, the reference path took about 0.663 s and `optimized` about 0.127 s
(approximately 5.24x).  A three-repetition dynamic-B0 benchmark improved from
about 2.091 s to 0.779 s (approximately 2.68x).  The relevant result arrays
were bit-identical.  A larger 48^3 three-repetition benchmark ran in about
3.9 s on the development system.

## Strict native block primitives

The dynamic native extension is separate from the legacy Bloch extension.
The legacy extension still uses `-ffast-math` on supported toolchains; the
dynamic extension is compiled with strict floating-point settings, including
disabled fast math and disabled floating-point contraction.  This separation
prevents a performance experiment from silently weakening the established
bit-exact CPU contract.

The Cython layer receives coefficients that NumPy has already calculated and
rounded.  It does not recompute exponentials or trigonometric functions.  The
native operations deliberately reproduce the optimized driver's order of
multiplication and addition.

The largest gain for RF-intensive spectral-selective bSSFP comes from the RF
voxel-block primitive.  It reads the contiguous complex transverse state and
`Mz` directly, applies the Rodrigues rotation to every pool/voxel pair, and
writes both representations in one pass.  This removes two full state
synchronizations, `numpy.cross`, the matrix projection, and several temporary
arrays for every RF interval.

For concentration-tracked pyruvate inflow, a fused longitudinal primitive now
advances magnetization, concentration, exchange, relaxation, equilibrium
polarization, and the linearly varying source in one voxel pass.  A second
primitive retains each voxel's complete two-pool state in registers while it
executes a consecutive RF waveform.  It receives already-rounded NumPy
coefficient tables for the few unique duration/gradient groups in the
waveform, so no exponential or trigonometric functions are evaluated in the
inner loop.  Repeated RF motifs reuse a byte-bounded plan cache.

`native_parallel` additionally groups up to 256 adjacent RF-free intervals for
longitudinal evolution when the longitudinal native primitive supports the
active model.  Only unique half-step coefficient tables are passed to the
block.  The table is capped at 8 MiB and reduced further by the global
simulation memory budget.  Parallel execution is enabled only from 1,024
active voxels.  Unsupported longitudinal cases retain native RF rotation and
use the optimized NumPy kinetics solver.  The persistent RF block is currently
limited to 131,072 simulated spins and static B0 with uniform transmit
sensitivity; larger subvoxel objects keep the safe interval path until the
full-sequence chunked backend is available.

For four representative TRs of the spectral-selective 3D bSSFP simulation
with 19,086 active voxels, measured median runtimes were approximately:

| Kernel | Threads | Runtime | Speed-up over `optimized` |
| --- | ---: | ---: | ---: |
| `optimized` | 1 | 1.209 s | 1.00x |
| `native_serial` | 1 | 0.214 s | 5.64x |
| `native_parallel` | 2 | 0.209 s | 5.79x |
| `native_parallel` | 4 | 0.210 s | 5.77x |
| `native_parallel` | 8 | 0.230 s | 5.26x |

All six public result arrays were bit-identical in these comparisons.  The
modest scaling beyond two threads is expected for this pilot: the dominant
improvement is removal of NumPy RF temporaries, while ADC reduction and much of
the interval driver remain serial.  Higher total CPU utilization is therefore
not itself the optimization target.

### Coupled inflow/concentration benchmark

The current end-to-end benchmark uses the local project and result artifacts
from the original user run:

- shape 42x56x112, with 32,692 active voxels, a 512-point phantom spectral
  grid, and one spin per voxel in ideal-spoiler mode;
- pyruvate inflow, polarization curve, concentration tracking, and kPL;
- the 24.2994 s internally generated Gaussian spectral-selective 3D bSSFP
  sequence;
- 122,880 ADC samples and a 20 us RF simulation step;
- 593,779 compiled intervals after dynamic-curve boundaries on an eight-core
  Apple Silicon Mac.

The complete `native_parallel` simulation, including compilation and result
assembly, takes about 321 s (5 min 21 s).  It fuses 451,622 RF raster intervals
into 3,860 persistent native waveform blocks and uses eight RF/longitudinal
workers.  The bounded fused-plan cache peaks at about 184 MiB.

The old exported result embeds physical RF and gradient waveforms that are
bit-identical to this saved project, so it provides an end-to-end numerical
reference for the original interval driver.  Across the complete acquisition,
signal NRMSE is 3.58e-15, maximum absolute signal error 3.85e-8 (1.52e-15
relative to the reference signal peak), and final-pool-state NRMSE 1.50e-14.
This validates the fused block's `float64-close` contract on the complete
phantom and timeline.  A separate 32-ADC Skinner-sequence prefix measured
15.66 s for `optimized` and 4.23 s for the fused eight-thread path
(approximately 3.7x).

## Compiler and GUI support

The simulation time step is configurable in the GUI and passed to the sequence
compiler.  RF-active intervals are rasterized at that step while RF-free
regions retain their exact event/ADC boundaries.  Coarser time steps trade RF
accuracy for fewer intervals and must be selected using convergence tests.

The progress bar reports percentage and an exponentially smoothed throughput
ETA.  Timing begins with the first solver progress sample, so sequence import
and compilation are not incorrectly extrapolated across every interval.  The
first sample remains in an explicit estimating state.  Live preview updates
clear the previous result at simulation start and show the currently
accumulated signal, inferred 2D k-space, and partial reconstruction.  Preview
work is throttled to roughly one hundred updates per simulation so it does not
run once per interval.

## Current limits and next performance phase

Spatially varying transmit sensitivity still requires the NumPy RF
implementation.  Dynamic B0 and unsupported inflow/concentration combinations
use the strict interval driver instead of the persistent RF-waveform block.
ADC observation and RF-free readout evolution are not yet fused into a native
segment, so ADC-heavy acquisitions still spend substantial time in the Python/
NumPy driver.

The next CPU phase is full-sequence voxel chunking.  The dynamic solver still
expands every subvoxel spin for the full timeline, while a 3x3x3 sampling grid
multiplies state and work by 27.  Chunking must keep a voxel batch through the
complete sequence, accumulate its signal deterministically, and then release
its coefficient/state workspace.  This also removes the current 131,072-spin
limit of the persistent RF plan without allowing coefficient caches to grow
with the complete expanded object.

The next experimental path is `float32`/`complex64` GPU execution.  Unlike the
CPU optimizations, it cannot be bit-identical to the `float64` reference and
must remain an explicit, separately validated backend.  The first gate is a
CPU `float32` shadow execution using the same integration topology; only if
the complete dynamic bSSFP signal and reconstruction remain within the agreed
scientific error limits should a fused Metal kernel become a production path.

The shadow path is selected with `simulation_precision="float32"` or
`BlochSimulator(dynamic_sequence_precision="float32")`.  It stores state in
`float32`, transverse values and ADC output in `complex64`, but prepares the
physical coefficients in `float64` and rounds them once before application.
The default remains `float64`, and the strict native kernels currently reject
the experimental precision instead of silently changing execution.

For a reproducible precision report, including optional 3D k-space and image
metrics, run:

```bash
python scripts/validate_dynamic_precision.py \
  sequences/sequences/bssfp_3d_spectral_selective_skinner.seq \
  --phantom path/to/dynamic_phantom.npz \
  --timestep-us 10 \
  --output-json /tmp/dynamic_precision.json
```

Omit `--phantom` to use a small heterogeneous synthetic two-pool object.  The
`--max-adc-events` option provides a shorter accumulation test while retaining
the original RF/gradient events through the selected acquisition.

The optional CPU/Metal subvoxel experiment is selected only in this validator:

```bash
python scripts/validate_dynamic_precision.py \
  sequences/sequences/bssfp_3d_spectral_selective_skinner.seq \
  --synthetic --max-adc-events 4 --timestep-us 10 \
  --candidate metal_hybrid --subvoxel-spins 2 2 2
```

It runs centre-symmetric CPU calibration and held-out validation subsets while
the GPU predicts the complete grid. A conservative validation failure returns
the normal Float64 CPU answer by default. The unchecked Metal mode remains a
private feasibility probe; the separately checked CPU/Metal hybrid is also
available as an experimental GUI kernel with automatic exact-CPU fallback.

### Initial float32 gate result

The first accumulation test used the checked-in 50-frame Skinner-style
spectral-selective sequence at a 10 us simulation step and an eight-voxel
heterogeneous synthetic phantom.  It intentionally tests the complete RF
waveforms rather than replacing them with ideal flip matrices:

| Simulated extent | Signal NRMSE | Pool-state NRMSE |
| --- | ---: | ---: |
| 4 TR / 128 ADC samples | 4.07e-5 | 7.12e-5 |
| 1 volume / 192 TR / 6,144 samples | 9.01e-4 | 2.53e-3 |
| 5 volumes / 960 TR / 30,720 samples | 3.06e-3 | 1.18e-2 |

This rejects a naive interval-by-interval `float32` port under the provisional
signal-NRMSE target of `1e-3`.  A subsequent private Apple M3 Metal probe kept
the complete prefix state inside one GPU thread, disabled fast math, used
Float64-prepared RF coefficients rounded once, and reduced per-spin ADC output
in a deterministic CPU Float64 tree.  Ordinary Float32 and a two-Float32
double-single state both failed by one volume (about `1.344e-3`) and reached
about `7.00e-3` and `6.80e-3`, respectively, by five volumes.  Lactate error
was substantially larger.  All repeated GPU runs were bitwise identical.

The unchecked probe therefore remains private and no direct Metal kernel is
exposed. The GUI exposes only the checked hybrid, which returns an exact
Float64 CPU result whenever its held-out test fails or Metal cannot be used.
Detailed metrics, timings, memory estimates, error growth, and the next bounded
experiment are in
[dynamic_metal_feasibility.md](dynamic_metal_feasibility.md) and the
machine-readable `dynamic_metal_precision_results_apple_m3.json` artifact.
