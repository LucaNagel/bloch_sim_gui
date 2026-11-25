# Simulation kernels

Technical reference for the kernel selectors under **Tools → Settings → Simulation**. For everyday use, keep the defaults (see the [User Guide](USER_GUIDE.md#9-performance-and-kernels)).

## Sequence-simulation kernels

The **Tools → Settings → Simulation** page exposes two independent kernel
selectors. A kernel is the numerical execution engine; changing it does not
change the phantom, pulse sequence, physical model, ADC times, or requested
checkpoints.

| Simulated object | Setting used |
| --- | --- |
| `Phantom`, sequence probes, and each independent component of a `SpectralPhantom` | **Sequence Bloch kernel** |
| Coupled `DynamicSpectralPhantom`, including additional product pools | **Dynamic spectral kernel** |

Both routes first compile the Pulseq events into propagation intervals. Event,
ADC, kinetic-breakpoint, and checkpoint boundaries remain exact. The configured
**RF-active simulation time step** is only a maximum interval length while RF is
active; making it coarser reduces work but also changes RF integration accuracy.
It is therefore a separate accuracy/performance choice, not another kernel.

## Post-run 3D magnetization animation

Enable **Create post-run animation** in the **3D Animation** tab
to keep a bounded set of spatial magnetization states for the next run. The
**Time resolution** setting specifies the desired interval between frames in
milliseconds. The scientific simulation first runs normally, with only the
manually requested checkpoints. A separate replay then captures the animation:
targets in RF-free sequence intervals are retained at the chosen spacing, while
targets inside an RF pulse are snapped to an existing RF-integration boundary.
Consequently, animation capture cannot change the scientific signal, final
magnetization, checkpoints, or RF integration accuracy. The additional replay
does increase total run time when animation capture is enabled.

The animation is available only after the run has completed. Its map selector
shows `Mz`, coherent `|Mxy|`, `Mx = real(Mxy)`, `My = imag(Mxy)`, or transverse
phase; spectral and dynamic objects additionally expose their individual
pools. Frames are held separately from the scientific result and are not added
to result exports or to manually requested checkpoints. Choose `float32` for
the default display storage or `float16` for smaller animations. Very long
sequences or large objects automatically receive a coarser effective time
resolution to bound the temporary float64 checkpoint memory used during the
replay.

### Sequence Bloch kernel

This kernel is used for the ordinary Bloch equation without coupled
pyruvate-to-lactate kinetics. A `SpectralPhantom` is represented by independent
spectral components; each component is simulated with this kernel and the
received signals are then summed.

| Selection | Implementation and intended use |
| --- | --- |
| **Production (recommended)** — `production` | Default since this release. Runs the static object on the dynamic Float64 spin-major executor (module `blochsimulator.static_spin_major`): spectral components become pools of one run (at most six per pass, more run in passes), per-voxel T1/T2 become relaxation groups (at most 256 per run; objects with more distinct T1/T2 rows run in voxel batches), and coils, Tx maps, subvoxel spins, crushers and checkpoints keep their static semantics. It splits free precession symmetrically around the RF rotation (second order). Anything the executor cannot run falls back to the legacy kernel with a `KernelFallbackWarning` (see below). Use this for normal simulations. |
| **Legacy fast-math (previous default)** — `legacy` | The previous native streaming C kernel (`blochsimulator_cy`, `-ffast-math`, contract `legacy-fast-math`). It propagates one spin through the complete sequence without storing a spin-by-time history, processes active spins in bounded chunks and distributes them over the CPU threads; RF-free intervals avoid the general 3×3 rotation, RF-active intervals use a quaternion rotation, and up to 64 distinct T1/T2 pairs reuse precomputed relaxation factors. It stays selectable for one release to compare old results and will then be removed. |
| **Strict reference (validation)** — `strict_reference` | Separately compiled Float64 extension. It uses the general rotation-matrix path with per-spin/per-interval relaxation and has the same event, ADC, checkpoint, receive-coil, transmit-field, subvoxel, and spoiler semantics as the legacy kernel, but deliberately omits its fast paths. It disables fast math and floating-point contraction and reports the `strict-float64` numerical contract. It checks the legacy kernel for fast-math effects; it is not a more accurate physical model, because it uses the same time discretization. |

The legacy and strict reference paths implement the same Bloch propagation,
but their floating-point operation order is not identical, so rounding-level
differences are possible. The strict reference checks the legacy kernel, not
the spin-major production path; both are checked against the Gold reference
below.

The physical reference for the static kernels is the Python Gold reference
`blochsimulator.sequence.reference.simulate_reference_sequence`. It does not use
the sequence compiler or either C kernel; it expands the events at their own
RF and gradient rasters and solves every interval exactly with a matrix
exponential. It is slow and meant for spin probes and small validation cases.
Its distance to both kernels measures the time-step error of the compiled
symmetric splitting (about `6e-3` at the 20 µs default and `6e-6` at 1 µs for
a FLASH test case, order about 2).

The Gold reference has two methods. `method="exact"` (default) also handles
several spectral pools (`pools=[ReferencePool(...)]`), longitudinal exchange
such as pyruvate-to-lactate conversion (`exchange_rates_per_s`), relaxation to a
per-pool equilibrium and a longitudinal source that is linear between
`knots_s` (inflow), all inside one matrix exponential per interval. The
dynamic kernel agrees with it to `1e-15` without RF and converges to it with
order 2 when RF is on. `method="hargreaves"` repeats the time-stepping of Brian
Hargreaves' `bloch.c`: per raster interval a Cayley-Klein rotation, then
`diag(E2, E2, E1)` decay and recovery; it is first order and has no exchange or
sources. A test compares it with a line-by-line transcription of the C loop.
`steady_state=True` (both methods) starts from the fixed point of the whole
program, like `bloch.c` mode 3.

The exact method also tracks concentrations (`initial_concentration`,
`concentration_source`; Mz then relaxes towards `equilibrium * C`), washout
(`loss_rates_per_s`, acting on all pools, transverse, longitudinal and
concentration states) and contrast agents (`relaxation_rate_increments(t)`
returning `(dR1, dR2)`, held over each interval at its midpoint; Hargreaves
mode supports these too). The dynamic kernel matches all three to `1e-15`
without RF and converges with order 2 with RF. One approximation remains:
washout combined with pyruvate inflow is Strang-split around each free
interval while the inflow is integrated without loss. The error is second
order in the free-interval length and linear in the loss rate (about `1e-8`
for a 240 µs interval with 4/s). Contrast `step_s` epoch boundaries where the
rates change are interval breakpoints, so every interval lies inside one
epoch.

The old name `optimized` is still accepted (in Python, saved settings,
projects, and scripts). In Python it selects `legacy`, so existing scripts and
benchmarks keep bit-identical results; the result metadata reports
`sequence_kernel="legacy"` and keeps the name as passed in
`requested_sequence_kernel`. A GUI setting saved as `optimized` opens with the
new default `production`.

The production result is Float64-close to the legacy kernel, not
bit-identical: the difference is a time-step error (about `5e-4` relative at
20 µs with a 1 µs RF raster, `8e-6` at 1 µs).
`progress_callback(done, total)` counts plan intervals summed over passes
instead of voxels, `chunk_voxels` is ignored, and `metadata["progress_unit"]`
is `"intervals"`; `metadata["static_spin_major"]` is `True`. Cases the
executor cannot run (negative equilibrium Mz, a spatially varying chemical
shift, explicit positions with several pools, more than three axes, an empty
plan) emit
`KernelFallbackWarning`, record `static_spin_major_fallback_reason` and run
the legacy kernel; the GUI completion message names the reason. The private
switch `_static_spin_major` (`None` = by kernel, `True`/`False` = force) is
kept for tests.

Against the gold reference (`sequence/reference.py`, 24 probes on a
phase-cycled sinc slice train, 1 µs RF raster) production is second order and
about half as wrong as the legacy kernel: `2.9e-3` at 20 µs, `1.7e-4` at
5 µs (legacy `5.9e-3`, `3.5e-4`).

The RF-active time step bounds every RF interval. A step finer than the RF
raster divides each RF sample into equal parts no longer than the step, so
sample boundaries remain interval boundaries. This matters for production:
its splitting error depends on the interval length, not on the raster, so a
coarsely rastered pulse (for example a 2 ms sinc with 64 samples, 31 µs)
converges with the step (`3.6e-4` at 20 µs, `2.9e-5` at 5 µs) instead of
keeping a raster-limited error (`1.4e-3`) that the time-step check could not
see. The legacy kernel rotates a constant sample exactly without relaxation,
so subdividing changes its results only by rounding and a smaller relaxation
error. `run_timestep_convergence` and
`estimate_timestep_error` use the simulator's kernel, so the time-step check
measures the kernel that produces the result.

### Dynamic spectral model

The dynamic kernel keeps a separate magnetization vector for every spectral
pool in each simulated spin. The common two-pool model contains pyruvate and
lactate; supported additional product pools use the same timeline and receive
their established precursor-to-product conversion. During each compiled
interval the solver applies a symmetric free-evolution/RF/free-evolution split:

1. half an interval of T2 decay, gradient/static-B0 phase, T1
   relaxation, longitudinal pyruvate-to-lactate conversion, and optional
   pyruvate inflow;
2. the RF rotation for both pools;
3. the second free half-interval, followed by any ADC observation, ideal
   crusher, or checkpoint at that state boundary.

For a segment with constant `kPL` and relaxation rates, the longitudinal
two-pool system is advanced with its analytic exponential solution. Piecewise
linear inflow is integrated analytically within each segment, including the
equal-rate limit. When concentration tracking is enabled, concentration is
advanced separately from polarization so that `Mz = concentration ×
polarization` and T1 relaxation approaches the configured equilibrium
polarization. Dynamic B0 changes transverse phase; it does not alter the
longitudinal conversion equations.

Unlike the ordinary streaming kernel, the dynamic solver retains the complete
pool state of the active object because its state must persist across the
continuous kinetic timeline. Runtime is therefore driven mainly by
`active spins × compiled intervals`, plus ADC reduction and any requested
checkpoints. The phantom's **spectral points** value defines the spectral output
grid; by itself it does not create that many Bloch states. Solver work instead
increases with the number of ADC samples in the sequence, which is a separate
quantity.

### Dynamic spectral kernels

The GUI offers only **Production**. The NumPy and reference paths are no
longer listed in the GUI; they remain available from Python through
`dynamic_sequence_kernel=...` or `sequence_kernel=...` for validation and
benchmarks. Hovering over the choice shows an English summary of its purpose,
current runtime availability, and fallback behavior. The same capability
registry is available from Python through `detect_kernel_capabilities()` and is
used by the kernel benchmark.

A stored GUI selection that is no longer offered, including an earlier
**Optimized NumPy** choice, resolves to Production.

| Name | Old names | Implementation and intended use |
| --- | --- | --- |
| **Production (recommended)** — `production` | `native_parallel`, `native_serial` | Strict Float64 spin-major executor (`dynamic_spin_major_cy`) with OpenMP across spin tiles. Parallel work starts at 1,024 simulated spins; smaller jobs, a simulator with `use_parallel=False`, and the old name `native_serial` use one thread. It supports unity, constant complex, or spatial transmit sensitivity, static B0, delayed conversion, ADC, crushers, receive coils, subvoxel sampling, and checkpoints. It covers direct or polarized pyruvate inflow, with or without concentration tracking, concentration tracking without inflow, voxelwise arrival delays, and up to six coupled pools with the existing precursor-to-product conversions, including pyruvate-to-alanine. Combinations it does not support run on `numpy`. This is the default for dynamic phantoms in the GUI and in `BlochSimulator(...)`. |
| `numpy` (Python only) | `optimized` | Complete CPU path. It keeps the transverse state in a persistent complex array, reuses scratch buffers, caches exact longitudinal coefficients by half-step duration, and caches T2/phase factors by duration and gradient. It supports inflow, delayed conversion, concentration tracking, spatial transmit sensitivity, multiple receive coils, checkpoints, and both spoiler modes. It is the safe fallback for unsupported native combinations. |
| `reference` (Python only) | — | Direct, allocation-heavy Python/NumPy formulation. It recomputes intermediate phase, relaxation, exchange, inflow, and RF arrays at each interval and is the correctness oracle for the other dynamic paths. It supports the complete model but is intended for small validation and diagnostic runs. |

Old names run exactly the path they ran before, with bit-identical results.
The result metadata reports the public name in `sequence_kernel`, the name as
passed in `requested_sequence_kernel`, and the internal paths
(`native_parallel`, `native_serial`, `optimized`, `reference`) in
`dynamic_kernel_path` and `requested_dynamic_kernel_path`.

The spin-major executor is built without fast math or floating-point
contraction. It keeps the Float64 integration topology of the NumPy kernel but
may change the final few bits; its result metadata reports a `float64-close`
contract and it is regression-tested with tight tolerances against `numpy` and
`reference`.

Production runs either completely on the spin-major executor or completely on
`numpy`; there is no mixed native/NumPy path. It falls back to `numpy` when the
extension is missing or too old, when the model is outside the executor's
coverage, or when `_native_spin_major=False` (or
`BLOCHSIM_DYNAMIC_SPIN_MAJOR=0`) switches the executor off on purpose; only the
first two are reported as a kernel fallback. Actions at time zero (ADC,
checkpoint) are taken from the initial state before the executor starts. The
completion status reports the requested and actually used path, effective CPU
thread count, numerical contract, plan model, and any fallback reason. The
same machine-readable fields are retained in result metadata and exports;
`native_full_plan_*` keys are kept for compatibility and describe the
spin-major run.

Large expanded objects are divided only at parent-voxel boundaries. All
subvoxel spins belonging to one parent remain together for the entire sequence,
and signal leaves are reduced in one fixed global parent order. Automatic
chunking starts above the local 131,072-spin limit or when the memory plan
requires it. Python callers can request a smaller deterministic block with
`parent_voxel_chunk_size`; this changes memory use, not the result ordering.

For routine use:

- Keep **Sequence Bloch kernel → Production** for ordinary and independent
  spectral phantoms.
- Keep **Dynamic spectral kernel → Production**. Unsupported parts fall
  back explicitly to `numpy`, and the persistent block reports its
  close-float64 contract. Use `numpy` from Python when bit-identical
  agreement with earlier Optimized NumPy results is required.
- Use the Python-only `reference` kernel and the Gold reference only for a
  controlled comparison.
  They are expected to be slower, and low CPU utilization is not evidence that
  they are more accurate for routine work.
- More CPU threads do not guarantee a proportional speed-up. Sequence driving,
  coefficient preparation, ADC summation, and unsupported longitudinal kinetics
  can remain in NumPy, while frequent small native calls incur OpenMP overhead.
- Dynamic state is always integrated in `float64`.
