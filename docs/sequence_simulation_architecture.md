# Sequence simulation architecture

This document is the stable technical reference for the spatially resolved
sequence simulator.  Update it whenever a unit, sign, timing, or public API
decision changes.

See [sequence_simulation_plan.md](sequence_simulation_plan.md) for the original
detailed implementation plan and
[sequence_simulation_roadmap.md](sequence_simulation_roadmap.md) for current
status and next actions.

## Canonical units and signs

- Time is seconds and positions are metres.
- Complex RF is nutation frequency in Hz.
- Gradients are frequency slope in Hz/m.
- B0 and chemical shift are frequency offsets in Hz.
- Transverse signal is `Mx + 1j*My`.
- The legacy solver remains in gauss, G/cm, and cm. Conversions live only in
  `blochsimulator.units` and the legacy adapter. They intentionally use the
  legacy core constant `26753 rad/s/G` so old and new endpoints agree exactly;
  native Pulseq data remains in its original Hz and Hz/m units.

For one interval of duration `dt`, let `RF_eff = RF * tx_sensitivity`. The new
kernel applies the rotation vector

`(-2*pi*Re(RF_eff)*dt, +2*pi*Im(RF_eff)*dt,
  -2*pi*(dot(G, r) + df)*dt)`

followed by T1/T2 relaxation and recovery toward normalized `Mz=1`.
Proton density weights received signal; it does not change normalized state.

## Timing model

`SequenceProgram` stores RF, gradient, and ADC events. `SequenceCompiler`
produces intervals `[t_i, t_(i+1))` with constant effective RF and average
gradient. ADC samples and checkpoints observe the state at an interval end; a
sample at `t=0` observes the initial state.

`ADCEvent.start_s` denotes the centre time of its first sample. Pulseq ADC
delays are imported as `block_start + delay + dwell/2`, matching PyPulseq's
public `adc_times()` result.

RF-active intervals follow RF and gradient raster boundaries. In RF-free
regions, z rotations commute with isotropic transverse relaxation, so arbitrary
gradient evolution can be represented exactly by its interval area. These
regions are split only at ADC samples, checkpoints, event boundaries, or the
end of the sequence.

## Data flow

1. Internal events, legacy arrays, or a Pulseq importer create a
   `SequenceProgram`.
2. `SequenceCompiler` validates overlaps and produces `CompiledSequence`.
3. `BlochSimulator.simulate_sequence` selects active phantom voxels and chunks
   them to respect the configured memory budget.
4. The C/Cython streaming kernel advances each chunk, accumulates ADC signal,
   and returns only final/checkpoint states.
5. Python sums chunk signals and reconstructs spatial result arrays.

Gradient phase is generated only by the Bloch kernel. It must not be applied a
second time by the legacy analytical k-space module.

The compiler accumulates each sparse RF/gradient event only into the intervals
it overlaps. This preserves the timing representation while avoiding an
`Ninterval x Nevent` traversal for full imaging sequences.

## Object model

`Phantom.b0_map` and `Phantom.chemical_shift_map` are separate maps in Hz. The
kernel receives their sum. `df_map` is a deprecated construction-time alias for
legacy total off-resonance files and cannot be combined with either new map.
Coordinates are voxel centres, not FOV edges.

`Phantom.tx_sensitivity_map` is a dimensionless complex single-transmit B1+
map with one value per voxel. The kernel multiplies the compiled complex RF by
this map before constructing the voxel rotation; the compiler timing contract
is unchanged.

`Phantom.rx_sensitivity_maps` has shape `(Ncoil, *phantom.shape)`. With the map
convention `C_c(r)`, each coil receives
`sum_r PD(r) * C_c(r) * (Mx(r) + 1j*My(r))`, followed by ADC demodulation.
The default is one unity map. Consequently `result.signal` remains `(Nadc,)`
for the default/single-coil case and is `(Ncoil, Nadc)` for multiple coils.
Final and checkpoint magnetization remain coil-independent and normalized.

Multiple transmit channels and multiple chemical species per voxel are later
extensions of the same streaming interface.

## Memory and parallelism

The supported large-object path never stores `Ninterval x Nvoxel`
magnetization. Peak storage scales with voxel state, requested checkpoint
states, coil maps, and `Nthread x Ncoil x Nadc` thread-local ADC accumulators.
Chunk boundaries are also the cancellation and progress-reporting boundaries.

## Compatibility

Existing `simulate`, `simulate_phantom`, and legacy sequence classes keep their
current return dictionaries. The new API returns `SequenceSimulationResult`
with an explicit `to_dict()` adapter. Initial Pulseq support targets format
1.5.0 through optional PyPulseq; newer formats are rejected explicitly. Labels
and triggers are retained as metadata, while soft delays and unknown extensions
are rejected.
