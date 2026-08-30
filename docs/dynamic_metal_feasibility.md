# Experimental Metal backend: feasibility design

## Decision gate

The Float64 CPU solver remains the scientific reference.  Metal execution is
mixed precision and must not be exposed as a production sequence kernel unless
the complete signal has NRMSE at or below `1e-3` on increasing extents of the
spectral-selective dynamic bSSFP workload.  Species signals, final pool state,
phase, low-signal samples, error growth, reconstruction metrics (when the
acquisition layout is known), and repeated-run determinism are reported
separately.  A failed gate leaves only an internal precision probe and its
validation evidence.

The already measured interval-by-interval Float32 shadow path is the first
candidate and fails at five volumes (`3.06e-3` signal NRMSE).  The next
candidate keeps each spin's two-pool state in one Metal thread for a complete
bounded sequence prefix, compiles Metal source at runtime with fast math
disabled, writes per-spin ADC contributions only in the isolated small-object
probe, and combines those contributions in fixed spin order using CPU
Float64.  This separates state-propagation error from nondeterministic or
low-precision ADC reduction error.

## Production architecture if the gate passes

The production path will consume a backend-neutral plan segmented at every
event, kinetic-curve, conversion, ADC, crusher, and checkpoint boundary.  The
plan will group consecutive intervals into RF, RF-free, ADC-readout, crusher,
kinetic-boundary, and checkpoint segments without changing any compiled
boundary or sample index.  Float64 CPU coefficient preparation is reused and
values are rounded once at the Metal boundary.

Full-sequence work is chunked by parent voxel.  All subvoxel spins belonging
to a parent stay in the same chunk; state remains in GPU buffers across plan
segments; deterministic GPU partial reductions produce one compact signal per
chunk; and chunk signals are combined in increasing chunk order on the CPU in
Float64.  No allocation scales as expanded spins times the complete timeline
or as expanded spins times all coefficient keys.

The memory planner will account for two aligned pool states per spin,
positions, B0, kPL, inflow delivery, subvoxel weights, parent mapping,
reduction workspace, immutable plan buffers, and a reserve below both the
application budget and Metal's recommended maximum working set.  Command
buffers end at bounded segment groups to retain cancellation and UI
responsiveness.

## Native integration

The feasibility probe uses a private Cython/Objective-C++ bridge and a packaged
Metal source resource.  The extension is compiled only on macOS and links only
Foundation and Metal.  Runtime source compilation is used because the command
line developer tools on the validation host do not provide the offline
`metal` compiler.  Libraries and pipeline states are cached for the process
lifetime.  Non-macOS builds do not define the extension, and ordinary package
imports never import Metal.

If the gate passes, the same ownership and error wrappers can be promoted into
the segmented backend. The unchecked probe stays private to
`validate_dynamic_precision.py`; there is no direct `sequence_kernel="metal"`
option. The separately checked hybrid described below may be selected in the
GUI because it automatically replaces every rejected or unsupported GPU result
with a fresh Float64 CPU result.

## Apple M3 gate result (2026-08-11)

The gate failed, so the production architecture above was not activated.  The
private probe was run on an Apple M3 (Apple GPU family 9) with a 10 us RF time
step and the checked-in Skinner-style spectral-selective bSSFP sequence.  The
phantom was the validator's heterogeneous 2x2x2 two-pool object.  Every run
used the original compiled boundaries and ADC indices, Float64-prepared RF
rotation coefficients rounded once, disabled fast math, complete-prefix state
retention in a GPU thread, and fixed CPU Float64 pairwise ADC reduction.

| Extent | State | Signal NRMSE | Pyruvate | Lactate | Final pool state | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4 TR / 128 ADC | Float32 | 5.39e-5 | 3.51e-5 | 1.29e-3 | 1.42e-4 | total passes; lactate fails |
| 4 TR / 128 ADC | double-single | 5.30e-5 | 3.38e-5 | 1.29e-3 | 1.38e-4 | total passes; lactate fails |
| 1 volume / 6,144 ADC | Float32 | 1.344e-3 | 1.238e-3 | 3.006e-2 | 5.716e-3 | fail |
| 1 volume / 6,144 ADC | double-single | 1.344e-3 | 1.239e-3 | 3.005e-2 | 5.555e-3 | fail |
| 5 volumes / 30,720 ADC | Float32 | 7.003e-3 | 6.801e-3 | 4.785e-2 | 3.055e-2 | fail |
| 5 volumes / 30,720 ADC | double-single | 6.798e-3 | 6.589e-3 | 4.778e-2 | 3.073e-2 | fail |

At one volume, double-single maximum absolute signal error was 1.46e-3
(2.06e-3 of the reference peak), significant-signal phase RMS was 1.21e-3 rad,
and the cumulative NRMSE crossed 1e-3 near 0.9 s.  At five volumes its maximum
absolute signal error was 9.50e-3 (6.35e-3 of the peak) and significant-signal
phase RMS was 2.86e-2 rad.  All repeated Float32 and double-single runs were
bitwise identical.  The validator records both a fixed low-signal threshold
and the lowest reference-magnitude decile; the measured synthetic acquisition
had no samples below 1e-4 of its peak.  In the one-volume lowest-magnitude
decile (615 samples), double-single RMS absolute error was 3.47e-4 and maximum
error was 7.52e-4 (1.06e-3 of the global reference peak).

Double-single state arithmetic changed the one-volume total NRMSE by less than
0.03%, showing that state storage and signal reduction are not the limiting
errors.  Once-rounded Float32 evolution factors, especially phase/trigonometric
coefficients, dominate.  A CPU correction would need to occur more often than
the observed roughly 0.9 s crossing and would require a credible Float64 state
correction, not merely a transfer round trip.  For the 24.2994 s reference that
implies at least about 28 synchronization points while duplicating enough
Float64 propagation to calculate the correction; this is not yet an
acceptable production design.

The smallest credible next experiment is double-single phase evaluation:
represent B0, gradient dot position, pool offset, and the sin/cos argument as
high/low Float32 pairs, then use a validated range-reduced double-single
trigonometric implementation inside one RF block.  It should first be tested
on the one-volume prefix.  The production backend remains gated unless that
reduces total and both species NRMSE below the agreed ceiling without an
unacceptable throughput cost.

## Optional CPU/Metal subvoxel hybrid probe

An opt-in hybrid experiment is available through both the precision validator
and the GUI's dynamic-kernel setting. The GPU predicts the complete subvoxel
grid while two disjoint, centre-symmetric sets of subvoxel offsets are run in
Float64 on the CPU at the same time. One CPU set estimates a species-wise
complex amplitude/phase correction. The other set is held out and cannot
influence that correction. Signals are combined in Float64 on the CPU.

In the GUI the extra option is named **CPU + Apple GPU (experimental)**.
Subvoxel counts are configurable only with gradient-waveform spoiling; ideal
spoiling always uses one spin per voxel. Checkpoints, unavailable Metal
hardware, unsupported field maps, invalid sampling grids, memory limits in the
GPU path, and failed accuracy checks all select the exact CPU path
automatically. The result metadata and completion message report whether the
checked hybrid result or the CPU fallback was used.

The held-out error is scaled conservatively against the complete corrected
signal before the `1e-3` gate is applied.  This matters when large individual
subvoxel signals nearly cancel: a small relative error in one sampled pair can
still be a large error in the much smaller combined signal.  Failure returns a
fresh, unmodified Float64 CPU result by default; `--hybrid-no-fallback` instead
raises and returns no candidate.  The GPU's per-spin ADC output is processed in
bounded chunks, and only CPU reference spins are retained between chunks.

```bash
python scripts/validate_dynamic_precision.py \
  sequences/sequences/bssfp_3d_spectral_selective_skinner.seq \
  --synthetic --max-adc-events 4 --timestep-us 10 \
  --candidate metal_hybrid --subvoxel-spins 2 2 2
```

On the Apple M3 four-TR prefix, the final conservative check correctly rejected
the 2x2x2 hybrid estimate and returned the Float64 CPU fallback.  The two
centre-symmetric CPU sets each represented 25% of the subvoxel weight.  Their
direct held-out total-signal mismatch was only `2.17e-6`, but after accounting
for cancellation in the complete signal the conservative estimate was `0.738`.
This is evidence that one global correction must not be trusted for this
strongly cancelling acquisition.  Larger subvoxel grids and spatially
stratified corrections may recover useful GPU work, but they must pass the same
held-out and full-oracle validation before promotion.
