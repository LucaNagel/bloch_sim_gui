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
| **Production (recommended)** — `production` | Native streaming C kernel. It propagates one spin through the complete sequence without storing a spin-by-time history, processes active spins in bounded chunks, and distributes spins over the configured CPU threads. RF-free intervals avoid the general 3×3 rotation, RF-active intervals use a quaternion rotation, and phantoms with at most 64 distinct exact T1/T2 pairs reuse precomputed relaxation factors. Continuous T1/T2 maps automatically use per-spin factors instead. Use this for normal simulations. |
| **Strict reference (validation)** — `strict_reference` | Separately compiled Float64 extension. It uses the general rotation-matrix path with per-spin/per-interval relaxation and has the same event, ADC, checkpoint, receive-coil, transmit-field, subvoxel, and spoiler semantics as the production kernel, but deliberately omits its fast paths. It disables fast math and floating-point contraction and reports the `strict-float64` numerical contract. It checks the production kernel for fast-math effects; it is not a more accurate physical model, because it uses the same time discretization. |

The production and strict reference paths implement the same Bloch
propagation, but their floating-point operation order is not identical. Very small rounding-level
differences are therefore possible. OpenMP parallelism is across spins, and
thread-local ADC sums are combined after propagation. This makes the standard
kernel much easier to parallelize than the dynamic solver.

The physical reference for the static kernels is the Python Gold reference
`blochsimulator.sequence.reference.simulate_reference_sequence`. It does not use
the sequence compiler or either C kernel; it expands the events at their own
RF and gradient rasters and solves every interval exactly with a matrix
exponential. It is slow and meant for spin probes and small validation cases.
Its distance to both kernels measures the time-step error of the compiled
symmetric splitting (about `6e-3` at the 20 µs default and `6e-6` at 1 µs for
a FLASH test case, order about 2).

The old name `optimized` is still accepted for `production` (in Python, saved
settings, projects, and scripts). It runs the same kernel with bit-identical
results; the result metadata reports `sequence_kernel="production"` and keeps
the name as passed in `requested_sequence_kernel`.

A private, experimental switch `_static_spin_major=True` on
`simulate_sequence` and `simulate_spectral_sequence` runs the static object on
the dynamic spin-major executor instead (module
`blochsimulator.static_spin_major`). Spectral components become pools of one
run (at most six per pass, more run in passes), per-voxel T1/T2 become
relaxation groups (at most 256), and coils, Tx maps, subvoxel spins, crushers
and checkpoints keep their static semantics. The result is Float64-close, not
bit-identical: the dynamic kernel splits free precession around the RF
rotation, so the difference to the production kernel is a time-step error
(about `5e-4` relative at 20 µs with a 1 µs RF raster, `8e-6` at 1 µs). It
shrinks only where the RF raster is finer than the step, because constant RF
samples are not subdivided. With the switch, `progress_callback(done, total)`
counts plan intervals summed over passes instead of voxels, `chunk_voxels` is
ignored, and `metadata["progress_unit"]` is `"intervals"`. Anything the
executor cannot run (strict reference, more than 256 T1/T2 combinations,
negative equilibrium Mz, a spatially varying chemical shift, a too small
memory budget) emits `KernelFallbackWarning`, records
`static_spin_major_fallback_reason` and runs the static kernel. The default
is unchanged.

Against the gold reference (`sequence/reference.py`, 24 probes on a
phase-cycled sinc slice train, 1 µs RF raster) the switch is second order and
about half as wrong as the static kernel: `2.9e-3` at 20 µs, `1.7e-4` at
5 µs (static `5.9e-3`, `3.5e-4`). At or below the RF raster both stop
changing: the static kernel keeps a first-order relaxation error (exact
without relaxation), the spin-major path a smaller splitting error
(`5e-7` with or without relaxation). `run_timestep_convergence` and
`estimate_timestep_error` take the same private switch, so the time-step
check measures the kernel that produces the result.

### Dynamic spectral model

The dynamic kernel keeps a separate magnetization vector for every spectral
pool in each simulated spin. The common two-pool model contains pyruvate and
lactate; supported additional product pools use the same timeline and receive
their established precursor-to-product conversion. During each compiled
interval the solver applies a symmetric free-evolution/RF/free-evolution split:

1. half an interval of T2 decay, gradient/static/dynamic-B0 phase, T1
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
| **Production (recommended)** — `production` | `native_parallel`, `native_serial` | Adds OpenMP across spins to strict native primitives and bounded plan blocks. Parallel work starts at 1,024 simulated spins; smaller jobs, a simulator with `use_parallel=False`, and the old name `native_serial` use one thread. The complete bounded driver supports unity, constant complex, or spatial transmit sensitivity, static or dynamic B0, delayed conversion, ADC, crushers, receive coils, subvoxel sampling, and checkpoints. It covers direct or polarized pyruvate inflow, with or without concentration tracking, as well as concentration tracking without inflow. A separate multi-pool worker supports the existing precursor-to-product conversions, including pyruvate-to-alanine. Voxelwise arrival delays use bounded local source tables. This is the GUI default for dynamic phantoms. |
| `numpy` (Python only) | `optimized` | Complete CPU path. It keeps the transverse state in a persistent complex array, reuses scratch buffers, caches exact longitudinal coefficients by half-step duration, and caches T2/phase factors by duration, gradient, and dynamic-B0 integral. It supports inflow, delayed conversion, concentration tracking, dynamic B0, spatial transmit sensitivity, multiple receive coils, checkpoints, and both spoiler modes. It is the safe fallback for unsupported native combinations and the default of `BlochSimulator(...)` from Python. |
| `reference` (Python only) | — | Direct, allocation-heavy Python/NumPy formulation. It recomputes intermediate phase, relaxation, exchange, inflow, and RF arrays at each interval and is the correctness oracle for the other dynamic paths. It supports the complete model but is intended for small validation and diagnostic runs. |

Old names run exactly the path they ran before, with bit-identical results.
The result metadata reports the public name in `sequence_kernel`, the name as
passed in `requested_sequence_kernel`, and the internal paths
(`native_parallel`, `native_serial`, `optimized`, `reference`) in
`dynamic_kernel_path` and `requested_dynamic_kernel_path`.

The native extension receives coefficients already evaluated and rounded by
NumPy and is built without fast math or floating-point contraction. Individual
native primitives reproduce the NumPy operation order bit for bit. The
persistent RF waveform block retains the same float64 integration topology but
may change the final few bits through its compiled complex-arithmetic path; its
result metadata reports a `float64-close` contract and it is regression-tested
with tight tolerances.

Native selection is capability-based rather than all-or-nothing:

| Active feature combination | Native RF | Native longitudinal evolution | Actual behavior |
| --- | --- | --- | --- |
| Two pools; no inflow or concentration tracking; unity, constant, or spatial transmit | Yes | Yes | Complete bounded native plan, including delayed conversion and static or dynamic B0 |
| Coupled inflow, polarization curve, and concentration tracking, with or without spatial arrival delays | Yes | Yes | Complete bounded native plan; delayed sources are prepared only for the current time/voxel block |
| Direct or polarized inflow without concentration delivery; concentration tracking with or without inflow | Yes | Yes | Complete bounded native plan with an explicit source mode matching the established model |
| Additional product pools and passive precursor-to-product conversion | Yes | Yes | Separate complete multi-pool plan worker; the specialized two-pool loop remains unchanged |
| Spatial transmit field with otherwise supported kinetics | Yes | Yes | Spatial native RF coefficient planes plus the applicable native longitudinal path |
| Unsupported precision or insufficient memory | Capability-dependent | Capability-dependent | Explicit hybrid or complete fallback to `numpy` |
| Strict native extension unavailable | No | No | Complete fallback to `numpy` |

Dynamic B0 is supported in all rows and does not by itself cause a native
fallback. It is part of the complete native plan both without inflow and with
all existing two-pool source modes, including spatial arrival delays. To retain the established
bitwise result for sequences that sample the unadvanced initial state, an ADC
event exactly at time zero also stays on that interval path. Later ADC events
remain part of the complete native plan. The spin-major executor takes such a
sample, like any action at time zero, from the initial state before it starts
and is not left for it. The completion status reports the
requested and actually used path, effective CPU thread count, numerical
contract, complete-plan model, and any fallback reason. Experimental CPU/GPU
results remain explicitly labelled. The same machine-readable fields are
retained in result metadata and exports, together with enabled native
RF/longitudinal blocks and the memory-limit decision.

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
