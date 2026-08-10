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
`simulate_sequence(signal_weighting="voxel")` preserves the historical
per-voxel sum. The opt-in `"voxel_volume"` mode multiplies PD by the physical
3D voxel volume so absolute signal is invariant to spatial discretization.

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
end of the sequence. Boundaries calculated through different raster arithmetic
are coalesced within a machine-precision tolerance; this prevents zero-duration
intervals without merging physically distinct sequence events.

## Time-step convergence validation

`simulate_reference_sequence` is the independent correctness oracle for small
spin ensembles. It does not call `SequenceCompiler` or a native kernel. Instead,
it expands RF and gradient events at their native rasters and solves the affine
Bloch equation, including relaxation and recovery, with a 4x4 matrix
exponential on every constant interval.

`SpinProbeEnsemble.from_axes` creates a Cartesian product of physical positions,
frequency offsets, complex B1 scales, and explicit T1/T2 pairs.
`run_timestep_convergence` simulates that ensemble with each requested
production time step and reports maximum and RMS local magnetization-vector
errors against the oracle, along with runtime, compiled interval count, and the
worst probe. By default it observes RF event ends and the sequence end. Those
times are already event boundaries, so convergence checkpoints do not
artificially refine the interior of a coarse RF interval.

The default pass limits are an absolute maximum vector error of `1e-3` and RMS
vector error of `2e-4`. Magnetization is normalized to equilibrium `Mz=1`.
These probe tests are intended to select candidate time steps before the more
expensive full-phantom, ADC-signal, and reconstruction validation.

`make_default_sequence_convergence_cases` supplies representative motifs for
non-selective FID/GRE excitation, slice-selective EPI/UTE/CSI excitation, spin
echo, repeated bSSFP, an MPRAGE inversion/readout train, phase-modulated
spectral-selective RF, and adiabatic inversion. The motifs are deliberately
small but retain the RF dynamics that make each class sensitive to temporal
coarsening. `run_sequence_convergence_suite` identifies both the limiting case
at every candidate step and the coarsest candidate passed by every case.

The command-line runner writes detailed and summary CSV files when requested:

```bash
python scripts/run_timestep_convergence.py \
  --rf-raster-us 1 \
  --timesteps-us native 1 2 5 10 20 50 100 \
  --output-csv /tmp/timestep_cases.csv \
  --summary-csv /tmp/timestep_summary.csv
```

With 1 us native RF waveforms and the default strict criteria, the initial
motif baseline is:

| Candidate | All motifs pass | Limiting motif | Maximum error | Largest RMS |
| --- | --- | --- | ---: | ---: |
| Native | yes | Spin echo | 9.92e-6 | 6.17e-6 |
| 1 us | yes | Spin echo | 9.92e-6 | 6.17e-6 |
| 2 us | yes | Adiabatic inversion | 9.73e-5 | 3.52e-5 |
| 5 us | no | Adiabatic inversion | 7.77e-4 | 2.77e-4 |
| 10 us | no | Adiabatic inversion | 3.20e-3 | 1.14e-3 |
| 20 us | no | Adiabatic inversion | 1.29e-2 | 4.60e-3 |
| 50 us | no | Adiabatic inversion | 8.00e-2 | 2.87e-2 |
| 100 us | no | Adiabatic inversion | 3.09e-1 | 1.13e-1 |

This is a regression baseline for the representative motifs, not yet a global
recommendation for arbitrary imported sequences. Full ADC and image validation
must still confirm the surviving candidates on actual sequence programs.
RF-discretization in dynamic hyperpolarized bSSFP is represented by the bSSFP
motif, but the independent oracle does not yet solve coupled exchange or inflow;
those dynamic equations require a separate convergence validation.

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

## Intravoxel spin integration and spoiler models

`SpinSampling` defines a deterministic midpoint grid in physical object X/Y/Z
order. `(1, 1, 1)` is the historical voxel-centre approximation. For larger
grids, each active parent voxel is expanded only inside its current streaming
chunk. T1, T2, off-resonance, Tx/Rx sensitivity, and initial state are repeated;
positions receive physical subvoxel offsets and proton density receives
normalized quadrature weights. The native kernel propagates every subspin for
the complete sequence. Final and checkpoint magnetization are coherently
averaged back onto the parent voxel, so persistent result dimensions do not
change.

`spoiler_mode="ideal"` passes declared crusher state indices to the kernel and
retains the explicit `Mx=My=0` operation. `spoiler_mode="gradient"` passes no
crusher indices; dephasing then follows exclusively from gradient waveforms and
subspin positions. Declared marker times remain in result metadata for audit.
The desktop's ideal mode deliberately uses one spin per voxel for exact
backward compatibility and speed. Static, independent spectral-component, and
dynamic two-pool sequence paths all implement the same two modes.

For a uniform rectangular voxel with initial transverse magnetization and a
gradient moment `k` in cycles/m, the no-relaxation validation oracle is
`exp(-i*2*pi*dot(k, r_center)) * product(sinc(dot(k, voxel_basis_axis)))`,
where `sinc` uses the normalized `sin(pi*x)/(pi*x)` definition. A reversed
gradient must restore the coherent subspin mean; this distinguishes physical
integration from irreversible ideal crushing.

## ADC acquisition and reconstruction

Each ADC value is an instantaneous observation at its sample centre. The
compiler also integrates the gradient waveform to every ADC state and exposes
the raw moment in cycles/m as `adc_gradient_moment_cyc_per_m`. This diagnostic
moment is exact for the event stream, but it is not universally a single
coherence-pathway k-space coordinate after arbitrary refocusing RF pulses.

`CartesianAcquisition` separately describes a 2D ADC layout: read/phase matrix,
FOV, dwell, phase-row order, readout direction per acquired line, and optional
fractional kx/ky cell offsets. It maps the chronological signal to
`(coil, phase, read)` or `(phase, read)`, validates ADC times and gradient
moments, and performs a centred inverse 2D FFT. The FFT applies the half-voxel
phase correction required by the Phantom voxel-centre coordinate convention.
Alternating EPI lines are reversed before the FFT.

`infer_cartesian_acquisition` is deliberately conservative. It currently
accepts one chronological ADC event per phase line, x readout, y phase encoding,
a Pulseq FOV definition, and one common regular kx/ky grid. It derives matrix,
dwell, line direction, phase order, and fractional grid offsets from compiled
ADC gradient moments. Alternating lines on different grids, multiple slices or
repetitions, missing FOV, and non-Cartesian trajectories are rejected instead
of being silently reshaped.

`infer_cartesian_acquisition_volumes` builds on the validated 2D frame mapping
for acquisitions that declare three-element `MatrixSize` and `FOV` definitions.
It subtracts the RF-relative moment origin, sorts planes by their measured kz
coordinate, validates `PAR` labels against that ordering, and retains slice,
echo, repetition, and segment as explicit outer dimensions. Results expose
`cartesian_3d_kspace(..., partition_z, phase_y, read_x)` and a centred 3D IFFT;
the chronological signal and flat 2D frame arrays remain available for audit.
Missing, duplicate, irregular, or inconsistent Cartesian cells are rejected.

`SpectroscopicAcquisition` is a separate model for phase-encoded 2D CSI. One
ADC event is one FID at a fixed `(kx, ky)` point, so its chronological samples
map to `(ky, kx, spectral_point)` rather than to an imaging readout axis.
Inference requires Pulseq `MatrixSize`, `FOV`, `LIN`, and `PAR` metadata and
validates gradient moments relative to the RF event preceding each FID. It
performs a spatial 2D inverse FFT while retaining the FID, followed by an
independent spectral FFT. xarray, NPZ, and HDF5 exports include the sorted CSI
k-space, spatial FID, spectra, k-space coordinates, spectral time, and
frequency axes in addition to the original chronological stream.
The desktop viewer exposes coupled slider/spin controls for the CSI FID point
and reconstructed voxel x/y coordinates. Inferred Cartesian outer dimensions
(slice, repetition, echo, segment, or partition) share a frame slider with the
descriptive frame selector; montage remains the slider's `-1` position.
An optional split view places reconstruction or k-space beside the selected
voxel FID or spectrum. Clicking the reconstruction updates the voxel selectors;
clicking k-space maps its displayed grid index to the same selector as a UI
convenience and explicitly does not reinterpret a k-space coordinate as a
physical voxel.

The main window has two presentation modes. Free Mode retains the single-spin
parameter panel, global run controls, playback controls, and all analysis tabs.
Sequence Mode hides those controls, collapses the legacy left panel, and shows
only Sequence Simulation and Phantom tabs. The workspace selector remains in
the menu-bar corner so the full interface can be restored without reopening
the application.

Sequence-side EPI controls are shown only for the internal EPI source, and the
built-in object property group is shown only when the built-in quick object
owns those values. Shape-designed spectral phantoms retain their editable
`phantom_design` metadata in memory, so the Phantom workspace can reopen and
replace the current design without first saving it to disk.

`make_cartesian_epi` is the first reference builder. It derives read gradient,
prephaser, phase blips/flybacks, and ADC timing from the acquisition object.
The receiver sampling bandwidth is `1/dwell`; nominal readout pixel bandwidth
is `1/(read_matrix*dwell)`. Finite ADC aperture, analogue/digital receiver
filters, oversampling, and polyphase decimation are intentionally separate
future receiver-model operations rather than Bloch-kernel responsibilities.

The desktop `Sequence Simulation` tab exposes this path as a `Cartesian EPI`
source. Simulation matrix and object FOV remain object controls; read matrix,
phase matrix, and sampling bandwidth are independent acquisition controls. For
imported Pulseq, a successfully inferred Cartesian layout enables k-space and
IFFT views; rejected streams continue to show ADC/final state with the concrete
inference error. Pulseq FOV definitions synchronize square in-plane and
through-plane object FOV controls so a thin slice is not accidentally sampled
by a coarse full-volume z grid. In-plane and through-plane matrix sizes remain
independent. Final-Mz levels are based on the complete volume, preventing a
numerically constant displayed slice from being stretched to full contrast.
The complete left control column lives in a vertical scroll area so Run/Cancel
and sparse-output controls remain reachable at reduced window heights.

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

Multiple transmit channels and coupled chemical-species dynamics are later
extensions of the same streaming interface. Independent spectral components
use the composition model below.

## Shape-designed spectral phantoms

`PhantomDesign` stores editable ellipsoid/box geometry in normalized spatial
coordinates. Each shape has T1 and B0 properties plus one or more
`SpectralPeakDefinition` entries containing amplitude, centre frequency in Hz,
and T2*. Rasterization creates a `SpectralPhantom`: every shape/peak pair is an
independent concentration map and `ChemicalSpecies`. Overlapping spectral
components add; for the single shared B0 map, later shapes overwrite earlier
shapes in overlapping voxels.

For event-based simulation, every active spectral component is passed through
the same Bloch sequence independently. Its frequency centre becomes the
chemical-shift offset, its concentration becomes PD, and its T2* is used as the
transverse decay constant. The resulting exponential FID corresponds to a
Lorentzian line with `FWHM = 1/(pi*T2*)`. ADC signals are coherently summed.
Final/checkpoint magnetization is concentration-weighted back onto the spatial
voxel grid. This model supports independent peaks; exchange, J-coupling, and
coupled density-matrix evolution are not implied.

Spectral `.npz` and `.h5` files preserve matrix/FOV, all concentration maps,
peak definitions, B0/T2* maps, and editable designer geometry. Conventional
`Phantom` files remain readable through the same GUI loader.

The phantom inspector displays orthogonal slices, an OpenGL point-cloud volume,
and the Lorentzian spectrum at the selected voxel. The sequence-result viewer
displays spatial Mx/My/Mz, magnitude/phase, and checkpoints. These magnetization
arrays are already in object coordinates and receive no z Fourier transform.
Slice-selective RF determines which z positions are excited. A z-IFFT is only
applied when volume inference verifies an explicitly sampled, regular Cartesian
kz dimension. Individual xy-IFFTs at fixed kz remain hybrid-space planes
`I(x, y, kz)`, not physical z slices.

The designer can add an analytic B0 variation to the constant shape offsets:
linear x/y/z and radial XY/XYZ modes are stored in ppm and converted using the
selected simulation field and nucleus. Slice viewers use centred physical mm
coordinates; the OpenGL point cloud and its FOV bounding box use the same
coordinate system. New boxes and ellipsoids can be created by dragging their XY
bounds directly on the designer canvas; the resulting ROI remains numerically
editable, movable, and resizable.

## Dynamic spectral phantoms

Pyruvate-to-lactate conversion must evolve coupled magnetization states, not
only concentration weights. The required state per active voxel is
`(Mx, My, Mz)` for every species. The first pharmacokinetic model transfers
longitudinal pyruvate magnetization to longitudinal lactate with voxelwise
`kPL`, while transverse states retain species-specific off-resonance and T2
evolution. RF rotations and ADC receive weighting act on the resulting species
states. A fully coherent transverse Bloch-McConnell exchange model is a later,
explicit alternative. Initial magnetization maps and voxelwise rate maps belong
to a dynamic spectral phantom, while the sequence remains unchanged.

Dynamic phantoms may additionally contain a piecewise-linear pyruvate input
curve and a voxelwise delivery map. Their product is a longitudinal source term
inside the coupled rate equations, so newly delivered magnetization retains the
correct RF history instead of resetting `Mz` at acquisition-frame boundaries.
The active simulation support is the union of initially magnetized voxels and
the delivery map.

Inflow samples and the conversion start share a kinetics timeline independent
of Pulseq sequence time. `kinetics_time_offset_s` specifies the kinetics time
that coincides with sequence `t=0`; solver breakpoints are translated by
`t_sequence = t_kinetics - kinetics_time_offset_s`. Any translated interval
before sequence zero is integrated as RF-free longitudinal pre-roll, producing
the pool distribution supplied to the Pulseq simulation. Dynamic B0 remains on
the sequence timeline and is deliberately not shifted by this kinetics offset.

A piecewise-linear dynamic B0 curve in Hz can be combined with a voxelwise
scale map. RF-free transverse evolution uses the exact frequency integral over
each compiled interval. The sequence compiler merges phantom curve knots with
Pulseq event boundaries, ADC samples, and checkpoints. Pulseq RF/ADC carrier
offsets remain sequence properties and are not conflated with this dynamic
object frequency.

The two-pool irreversible model must pass four limiting cases: zero exchange equals the
current independent-component solver, no-RF evolution matches the analytic
rate equations, total pool mass is conserved when relaxation and input are off,
added mass equals the integrated source when relaxation is off, and zero-rate
dynamic output equals a static spectral phantom. A post-hoc
time-dependent scaling of static signals is explicitly rejected because it
cannot model converted longitudinal magnetization or repeated RF depletion.

## Memory and parallelism

The supported large-object path never stores `Ninterval x Nvoxel`
magnetization. Peak storage scales with voxel state, requested checkpoint
states, coil maps, and `Nthread x Ncoil x Nadc` thread-local ADC accumulators.
Chunk boundaries are also the cancellation and progress-reporting boundaries.

The coupled dynamic solver has reference, allocation-reusing NumPy, strict
native serial, and strict native OpenMP execution paths.  The native extension
uses separate non-fast-math compiler flags and consumes NumPy-prepared
coefficients so the optimized CPU outputs remain bit-identical.  Its principal
RF-intensive speed-up comes from applying one Rodrigues rotation directly to
the complete contiguous two-pool voxel block, avoiding repeated NumPy state
synchronization and temporary arrays.  ADC sums retain their original voxel
order.  Detailed kernel behavior, limits, and representative measurements are
recorded in
[dynamic_simulation_performance.md](dynamic_simulation_performance.md).

## Compatibility

Existing `simulate`, `simulate_phantom`, and legacy sequence classes keep their
current return dictionaries. The new API returns `SequenceSimulationResult`
with an explicit `to_dict()` adapter. Initial Pulseq support targets format
1.5.0 through optional PyPulseq; newer formats are rejected explicitly. Labels
and triggers are retained as metadata, while soft delays and unknown extensions
are rejected.
