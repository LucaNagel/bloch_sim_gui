# Dynamic sequence simulation performance

This document records the performance paths implemented for the coupled
pyruvate/lactate sequence simulator.  It complements the physical model in
[dynamic_phantom_design.md](dynamic_phantom_design.md) and the general data
flow in
[sequence_simulation_architecture.md](sequence_simulation_architecture.md).

The overriding CPU requirement is unchanged: all optimized and native CPU
paths must preserve the results of the reference implementation bit for bit.
The strict regression tests therefore use `numpy.array_equal` for the signal,
pool-resolved signal, final states, and checkpoint states.

## Available execution paths

`simulate_dynamic_sequence` currently exposes four kernel selections:

- `reference` keeps the direct, allocation-heavy Python/NumPy formulation.  It
  is the correctness oracle for the other CPU paths and is intended mainly for
  tests and small diagnostic simulations.
- `optimized` keeps the same operation and summation order while reusing
  state and coefficient arrays.  It supports static and dynamic B0 as well as
  pyruvate inflow.
- `native_serial` uses the optimized Python driver together with strictly
  compiled Cython primitives for longitudinal two-pool evolution and RF
  rotation.  With pyruvate inflow, delayed conversion, or explicit
  concentration tracking it keeps RF rotation native and uses the optimized
  NumPy longitudinal solver as a bit-identical hybrid path.  Dynamic B0 does
  not require a longitudinal fallback.
- `native_parallel` adds OpenMP voxel parallelism to the native block
  primitives.  ADC summation remains in NumPy and retains the original active
  voxel order.  Small objects use one thread because OpenMP startup and
  synchronization cost more than the available parallel work.

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
3. Transverse T2/phase factors are cached by duration, gradient, and dynamic-B0
   integral.  The cache is bounded so long sequences cannot grow memory in
   proportion to every interval/voxel combination.
4. Static frequencies and gradient projections are only built on cache misses.
   Repeated EPI and bSSFP motifs therefore reuse the already rounded factors.
5. Progress, preview, cancellation, checkpoint, and ADC observation semantics
   remain at the original interval boundaries.

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

`native_parallel` additionally groups up to 256 adjacent RF-free intervals for
longitudinal evolution when the longitudinal native primitive supports the
active model.  Only unique half-step coefficient tables are passed to the
block.  The table is capped at 8 MiB and reduced further by the global
simulation memory budget.  Parallel execution is enabled only from 1,024
active voxels.  Unsupported longitudinal cases retain native RF rotation and
use the optimized NumPy kinetics solver.

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

## Compiler and GUI support

The simulation time step is configurable in the GUI and passed to the sequence
compiler.  RF-active intervals are rasterized at that step while RF-free
regions retain their exact event/ADC boundaries.  Coarser time steps trade RF
accuracy for fewer intervals and must be selected using convergence tests.

The progress bar reports percentage and an elapsed-time-based ETA.  Live
preview updates clear the previous result at simulation start and show the
currently accumulated signal, inferred 2D k-space, and partial reconstruction.
Preview work is throttled to roughly one hundred updates per simulation so it
does not run once per interval.

## Current limits and next performance phase

The strict native longitudinal pilot still uses the optimized NumPy solver for
pyruvate inflow, delayed conversion, and explicit concentration tracking.  RF
rotation remains native in those cases and is bit-identical to the complete
optimized path.  Spatially varying transmit sensitivity still requires the
NumPy RF implementation; if it is combined with an unsupported longitudinal
driver, the complete solver falls back to `optimized`.  The native block pilot
also does not yet fuse the complete sequence driver or ADC observation into a
persistent native region, which is why additional OpenMP threads provide
limited benefit for RF-intensive bSSFP.

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
signal-NRMSE target of `1e-3`: storage rounding accumulates across the hundreds
of RF raster points in every spectral pulse.  Preparing factors in `float64`
and using a more stable RF plane rotation improved runtime but did not
materially change the one-volume error.  The next precision experiment should
therefore compose each repeated RF waveform into a per-voxel block propagator
in `float64`, round that propagator once, and apply it once per RF event.  This
matches the required GPU kernel fusion and removes most intermediate state
rounding.  If that still misses the gate, selected state components require a
two-`float32` representation rather than silently accepting the drift.
