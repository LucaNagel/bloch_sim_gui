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

Single-voxel PRESS (`make_pulseq_press`) uses the same spectroscopic model.
The sequence declares `MatrixSize` `[1, 1, spectral_points]`, labels every
FID with `LIN = PAR = 0` and `REP`, and has no phase encoding. A 1×1 grid
therefore skips the moment validation: the crusher and slab moments after
the last refocusing pulse are refocused by the echo itself. Inference returns
one FID per repetition as a `SpectroscopicAcquisition` with
`matrix == (1, 1)`. The slice-selective FID (`make_pulseq_slice_fid`) is
labelled the same way and integrates the whole excited slice.

The localisation is physical. Only the intersection of the three slabs forms
the double spin echo, and the crusher pairs dephase everything else across
isochromats. The result therefore depends on the intravoxel spin model
described above:

- **Ideal mode (one spin per voxel):** only adds a phase, so signal outside
  the voxel survives.
- **Gradient-waveform subvoxel spins:** reproduce the suppression.

The refocused profile of a sinc 180° follows `sin²(θ(z)/2)` and is narrower
than its nominal slab. The effective voxel is slightly smaller than
`VoxelSize`.

Inversion recovery is a preparation, not a new acquisition model. The
standalone `make_pulseq_inversion_recovery` and FLASH or EPI with
`inversion_times_s` keep their Cartesian layout. Each repetition and slice is
one frame. A global inversion shifts the TI of later slices in the train, so
every frame's TI is recorded in `SliceInversionTimes`.
`relaxometry.slice_inversion_times` reshapes it, and
`relaxometry.fit_inversion_recovery_t1` fits the reconstructed frame series
along that axis.

Compressed sensing is likewise outside the acquisition models.
`sequence.compressed_sensing` works on the grids that `reshape_signal`
produces and uses the same centred FFT convention. Retrospective masks need
the voxel-centred phase of `CartesianAcquisition.reconstruct` applied to the
k-space first. `CartesianAcquisition` still requires `phase_indices` to be a
full permutation, so a prospectively undersampled grid is rejected. The
staged extension is described in `docs/compressed_sensing_concept.md`.

The main window has two presentation modes. Free Mode retains the single-spin
parameter panel, global run controls, playback controls, and all analysis tabs.
Sequence Mode hides those controls, collapses the legacy left panel, and shows
only Sequence Simulation and Phantom tabs. The workspace selector remains in
the menu-bar corner so the full interface can be restored without reopening
the application.

Sequence-side EPI controls are shown only for the internal EPI source. The EPI
builder supports alternating bipolar lines and monopolar flyback lines with a
combined read rewinder/phase blip. The separate EPSI/MGE source labels echo,
slice, and repetition dimensions explicitly, so the generic Cartesian-frame
pipeline reconstructs each echo without changing the chronological ADC stream.
The built-in object property group is shown only when the built-in quick object
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
independent. The complete left control column lives in a vertical scroll area
so Run/Cancel and sparse-output controls remain reachable at reduced window
heights.

The separate post-run 3D magnetization animation never adds checkpoints to the
scientific simulation. After that simulation completes, the GUI worker performs
a separate replay with sparse checkpoints at the configured temporal
resolution. RF-free targets can add observation boundaries because free
evolution is exact over split intervals; targets inside an RF-active interval
are snapped to an existing integration boundary. Only reduced-precision replay
arrays are retained in memory, so manual checkpoints, final magnetization,
signals, and exports keep their original semantics. This strict separation adds
replay time when animation is enabled. Frame counts are bounded from the object
size before the run, which can coarsen the effective temporal resolution; the
viewer reuses the same spatial geometry while updating scalar values so camera
and slice selection remain stable during playback.

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

## Receiver noise and phantom calibration

The Sequence workspace's **Receiver / Noise** tab, between **B1 Fields**
(whose **Phantom + B1** sub-tab also appears under **Phantom**) and **Simulations**, supports off, relative-SNR and
calibrated-from-phantom-FID modes for one receive channel.
`blochsimulator.sequence.apply_receiver_noise(result, ReceiverNoise(...))`
is the common post-simulation operation for static, independent spectral, and
dynamic phantoms. It acts once on the summed chronological ADC stream, before
the existing spatial or spectral reconstruction. Live streaming previews stay
noiseless. Magnetization, checkpoints, and species-resolved ground truth remain
unchanged.

The SNR and measured-noise inputs have native spin-box arrows and accept
Up/Down keys. Steps adapt to the current decimal order (for example, 50 to 51
or 0.05 to 0.051) while direct typing and scientific notation remain
available. Very small or large values retain double precision instead of being
rounded to a fixed number of displayed decimal places.

For each sample the receiver adds

$$
n = \sigma(\xi_R + i\xi_I), \qquad
\xi_R,\xi_I \sim \mathcal{N}(0,1).
$$

Here $\sigma$ is the standard deviation of each quadrature; complex noise power
is $2\sigma^2$. Measured magnitude standard deviation and complex RMS are not
interchangeable with this input. The local PCG64 generator and saved seed give
reproducible realizations without affecting NumPy's global random state.

### Relative SNR

Relative mode sets the single-scan SNR of the ideal 90-degree FID of the whole
simulated object at a reference dwell $\Delta t_{\mathrm{ref}}$ (the GUI's
reference bandwidth, $1/\Delta t_{\mathrm{ref}}$). Each ADC event with dwell
$\Delta t$ receives

$$
\sigma(\Delta t) = \frac{A_{\mathrm{ideal}}}{\mathrm{SNR}}
\sqrt{\frac{\Delta t_{\mathrm{ref}}}{\Delta t}},
\qquad
A_{\mathrm{ideal}} = w\,\Bigl|\sum_{\mathbf r} m_0(\mathbf r)\,
s_{\mathrm{rx}}(\mathbf r) + D\sum_{\mathbf r} d(\mathbf r)\,s_{\mathrm{rx}}(\mathbf r)\Bigr|,
\qquad
D = \int r(t)\,P(t)\,\mathrm{d}t .
$$

$m_0$ is the initial longitudinal magnetization (proton density times $|M_0|$
for static phantoms, concentration times initial $M_z$ summed over species for
spectral phantoms, initial concentration maps for dynamic phantoms),
$s_{\mathrm{rx}}$ the receive sensitivity and $w$ the voxel volume in m³ for
`signal_weighting="voxel_volume"` (1 for `"voxel"`). A pyruvate inflow adds its
whole delivered dose: the delivery map $d$ times the integral of the rate curve
$r$ and the polarization curve $P$ ($P=1$ for the legacy direct-$M_z$ source).
The integral is exact for linear and step segments and covers the curves'
sampled range, so phantoms that start empty and are filled only by a bolus have
a defined reference.

Because the reference is the whole object at its own signal weighting, the
same SNR gives the same noise relative to the signal for any grid and either
weighting. The reference is ideal and instantaneous: Tx inhomogeneity,
off-resonance, relaxation, metabolism and the actual flip angles are not
included, so a simulated 30-degree FID has half the stated SNR and a
0-degree scan contains only noise of that scale. Use calibrated mode when the
SNR should follow from a phantom measurement and physical amounts.

Runs record the amplitude as
`metadata["receiver_acquisition"]["ideal_fid_amplitude"]` (None when the
object has no initial or delivered magnetization), so **Apply to result** and
reloaded projects work without the phantom. Results from before this field
existed must be simulated again for relative SNR. `metadata["receiver_noise"]`
stores the per-event std and `relative_snr` (SNR, reference dwell and
amplitude). Saved controls from the former direct-std mode restore with noise
off.

### Calibration from a phantom FID

`ReceiverCalibration` turns one measured phantom FID into a scanner
sensitivity. It stores:

- the FID amplitude $A$ at $t=0$ after a flip angle $\alpha$ and the
  per-quadrature noise std $\sigma_{\mathrm{meas}}$ (for example from a
  0-degree scan), in the same scanner units and from scans with the same
  receiver gain, $N_A$ averages and ADC dwell $\Delta t_{\mathrm{ref}}$;
- the concentration $c$ in mM and volume $V$ in ml inside the coil's sensitive
  region, labelled nuclei per molecule $n_l$ and isotope enrichment
  $\varepsilon$;
- the phantom temperature $T_{\mathrm{ref}}$ (thermal polarization), or an
  absolute reference polarization.

A concentration in mM equals mol/m³. With `signal_weighting="voxel_volume"`,
which multiplies by the voxel volume in m³, the simulated signal is therefore
in mol times polarization relative to thermal equilibrium at the simulated
object temperature $T_{\mathrm{sim}}$; $M_z=1$ is thermal. Hyperpolarized
pools enter with $M_z = P_{\mathrm{abs}}/P_{\mathrm{th}}(B_0,T_{\mathrm{sim}})$.

$$
n_{\mathrm{ref}} = c\,V\,10^{-6}\,n_l\,\varepsilon \;[\mathrm{mol}],
\qquad
P_{\mathrm{th}}(B_0,T) = \tanh\frac{h\gamma B_0}{2k_BT}
\quad(\text{spin } 1/2),
$$

$$
\mathrm{SNR}_1 = \frac{A/\sigma_{\mathrm{meas}}}{\sqrt{N_A}},
\qquad
k = \frac{\mathrm{SNR}_1}{n_{\mathrm{ref}}\,P_{\mathrm{ref}}\,\eta},
\qquad
\sigma_{\mathrm{sim}}(\Delta t) =
\frac{f}{k\,P_{\mathrm{th}}(B_0,T_{\mathrm{sim}})}
\sqrt{\frac{\Delta t_{\mathrm{ref}}}{\Delta t}}.
$$

$k$ is the single-scan FID SNR per mol of fully polarized nuclei at the
reference dwell. $\eta$ is the measured amplitude per ideal 90° FID of the
same nuclei: $\sin\alpha$ for a fully relaxed reference at $t=0$ in a
homogeneous coil, or the value of the reference scan model below. Noise power is proportional to the receiver bandwidth
$1/\Delta t$, so each ADC event uses its own dwell. $f$ is a noise factor for
the simulated object, for example extra coil loading in vivo. A target with
$n$ mol and absolute polarization $P$ therefore has an initial single-scan
SNR of $k\,n\,P\sin\beta\sqrt{\Delta t/\Delta t_{\mathrm{ref}}}/f$ after a
flip angle $\beta$ in a homogeneous coil.
`units.thermal_polarization` evaluates the Boltzmann polarization for any spin
with hyperbolic sums that avoid cancellation at polarizations near $10^{-6}$.

Worked example at 7 T. 2 M [13C]urea in 50 ml at 20 °C holds 0.1 mol of
nuclei with $P_{\mathrm{th}} = 6.13\times10^{-6}$, or 0.613 µmol fully
polarized. 80 mM pyruvate in 200 µl at 20 % polarization holds 16 µmol, or
3.2 µmol fully polarized; at 37 °C its $M_z$ is about 34 500. At equal dwell
and single scans the predicted SNR ratio is 5.2. If the urea scan has SNR 200
with 4 averages, its single-scan SNR is 100 and the pyruvate FID starts at
SNR 522 after a 90-degree pulse, before T2*, delivery and metabolism. At 3 T
the thermal reference is 3/7 as large while the hyperpolarized signal is not,
so the ratio becomes about 12.

Practical requirements for the measurement: identical gain, averages and
bandwidth for the FID and noise scans, noise std per quadrature, and the volume
inside the coil's sensitive region. Without the reference scan model the
reference must also be fully relaxed (TR ≥ 5 T1, or a doped phantom) and read
at $t=0$. On Bruker data the digital-filter group delay makes the first stored
points unsuitable as $A(t=0)$; fit or extrapolate the FID, or enter the time
of the measured point as the amplitude delay. **Load FID…** does this from
the raw data (see below).

#### Reference scan model

The optional **Reference scan model** (`ReferenceScan`,
`reference_signal_efficiency`) predicts $\eta$ for a reference that was
saturated, read after a delay or placed in an inhomogeneous coil. With
spoiled repetitions (TR ≫ T2*), a fully relaxed sample before the first of
$D$ dummy scans and instantaneous excitation, the longitudinal magnetization
before repetition $n$ follows

$$
M_{n+1} = E_1\cos\theta\,M_n + (1-E_1)\,M_{\mathrm{eq}},
\qquad E_1 = e^{-\mathrm{TR}/T_1},\quad M_0 = 1,
$$

and the acquired repetitions $D,\dots,D+N_A-1$ average to

$$
\bar M(\theta) = M_{\mathrm{ss}} + (1-M_{\mathrm{ss}})\,
\frac{q^D(1-q^{N_A})}{N_A(1-q)},
\qquad q = E_1\cos\theta,\quad
M_{\mathrm{ss}} = \frac{(1-E_1)M_{\mathrm{eq}}}{1-q}.
$$

Without T1 (full recovery) or with one repetition, $\bar M = 1$. The measured
point at delay $\tau$ after the excitation decays by $e^{-\tau/T_2^*}$. In a
homogeneous coil

$$
\eta = \sin\alpha\;\bar M(\alpha)\;e^{-\tau/T_2^*}.
$$

With **Use B1 workspace coil fields** the transmit and receive fields of the
B1 workspace are sampled on an equal-volume grid inside the reference, a
sphere or a cylinder along z of the reference volume centred at the entered
position in object coordinates. Every point $j$ is excited by
$\theta_j = \alpha\,|B_{1,j}^+|$ and weighted like the simulator's signal:

$$
\eta = \left|\frac{1}{N}\sum_j \sin\theta_j\,\bar M(\theta_j)\,
e^{-i\arg B_{1,j}^+}\,B_{1,j}^-\right| e^{-\tau/T_2^*}.
$$

The model matches the Bloch simulation of the same scan to $10^{-3}$
(`tests/test_reference_scan.py`). Because noise is calibrated per unit
$B_1^-$ weight, a calibration whose $\eta$ depends on the coil only holds for
an object carrying the same B1 maps: runs reject it when the object's maps are
all unity, and **Set calibration** must be repeated after the B1 fields
change. The calibration stores $\eta$ as `signal_efficiency` and the scan as
`reference_scan` (with `coil_model`); the SNR budget lists the factor and its
inputs as **Reference signal fraction**. Only one transmit and one receive
channel are supported.

#### Measured scan import

**Load FID…** and **Load noise scan…** (`load_measured_scan`,
`measured_calibration_inputs` in `sequence.measured_fid`) fill the
calibration inputs from raw data:

- a ParaVision scan folder, or its `fid` or `rawdata.job0` next to `acqp`:
  1-D, one receiver, 16- or 32-bit integer or 32- or 64-bit float in either
  byte order, `fid` rows padded to 1024 bytes;
- a complex array (`.npy`, `.txt`, `.csv`, `.dat`) of complex numbers or
  real/imaginary pairs, one repetition per row, without scanner parameters.

ParaVision parameters give the dwell $1/\mathrm{SW_h}$, the filter group delay
`GRPDLY`, NA, receiver gain (`ACQ_CalibratedRG`, the gain the data were
recorded with, else `RG`), nucleus, field $\mathrm{BF1}/\gamma$, flip
angle, TR and dummy scans. The digital filter delays the FID by `GRPDLY`
points, so sample $k$ lies at $t_k = (k - \mathrm{GRPDLY})\,\Delta t$ after
the acquisition start; that start follows the excitation centre by half the
excitation pulse plus `DeadTime`, the amplitude delay.

The amplitude is fitted to the mean of all repetitions. Starting four points
after the group delay, $\ln|z|$ is fitted with a magnitude-weighted
polynomial (quadratic from 12 points) until the smoothed magnitude falls to
60 % of its first value, using 8 to 256 points. Averaging adjacent samples
first cancels the filter's alternating ringing. The fit at $t=0$ gives $A$,
its slope $T_2^*$.

The noise std is per quadrature and refers to the mean of the FID's
repetitions. A noise scan without excitation must share NA, receiver gain and
bandwidth with the FID; it is used after two group delays, with each row's
mean removed. Without a noise scan, the last quarter of every FID row is used.
While the time-domain std is at most 1.1 times the median of the
Hann-windowed periodogram, the data count as white noise and the time-domain
std is used. Otherwise the spectral median, which ignores a residual line,
spurs or a noisy band edge, is used with a warning. An SNR below 20 also
warns, and so does a receiver gain below 10: there the digitizer rather than
the coil can set the noise floor, so the calibration underestimates the SNR of
scans at a higher gain. On a 2 M [13C]urea phantom at 7 T, an FID at RG 63
had 2.3 times the SNR of the same FID at RG 1. Averages
are NA × repetitions, so the calibration's single-scan SNR is
$A/(\sigma\sqrt{N_A N_R})$. This holds whether the scanner sums or averages
the accumulations, provided FID and noise scan are treated alike.

Loading never sets the calibration or enables the reference scan model.
Concentration, volume, enrichment and temperatures are still entered by hand.
**Set calibration** refuses a loaded FID of another nucleus, or with a field
more than 2 % away from the workspace field strength. On two repeated 90° FIDs
of 2 M [13C]urea at 7 T (2048 points, 2.008 kHz) the import gives
$A = 5.71\times10^4$ and $\sigma = 113$, with the two SNRs (about 505)
agreeing to 1 %.

**Set calibration** freezes the entered values together with the current
nucleus and field strength. Runs and later **Apply to result** calls check the
result's nucleus and field strength against the calibration; another field
needs a new calibration because coil sensitivity and loading change. Different
ADC dwells are allowed and rescaled. Runs record
`metadata["receiver_acquisition"]` with nucleus, field strength, each ADC
event's dwell and sample count in signal order, and the ideal FID amplitude.

Calibrated mode requires `signal_weighting="voxel_volume"` and a 3D phantom.
`metadata["receiver_noise"]` stores the per-event noise std
(`noise_std_scope="per_adc_event"`), the calibration fields, derived values
(reference moles and polarization, target thermal polarization, single-scan
SNR and sensitivity) and the signal unit. Simulated runs are single scans.
Receiver filter shapes, field-strength scaling of coil sensitivity,
multichannel covariance and multi-receiver or multidimensional scanner data
are outside this model.

`SequenceSimulationResult.signal` contains the measured realization and
`signal_noiseless` retains the original ADC data when receiver processing has
been used. Applying noise again starts from `signal_noiseless`, so realizations
never accumulate. **Apply to result** and **New realization** reuse the existing
Bloch result. The generic ADC plot overlays the noiseless magnitude; all
reconstruction paths consume the measured `signal`. NPZ, HDF5, NetCDF and
complete projects preserve the clean signal and receiver metadata. Projects
and retained simulation runs also preserve the controls and frozen calibration.

### Phantom authoring units

The Spectral Phantom Designer keeps initial and inflow polarization in
multiples of thermal equilibrium. Its **Temperature**, with the selected
nucleus and field strength, sets the thermal polarization that 1 × thermal
stands for and shows it in percent. The Mouse Phantom Designer accepts the
injection polarization in absolute percent and shows the thermal multiple
beside it:

$$
P_{\mathrm{rel}} = \frac{P_{\%}/100}{P_{\mathrm{th}}(B_0,T)}.
$$

Shared workspace B0/nucleus changes rebase the temperature-aware phantom's
initial states and inflow, including authoring metadata, while preserving its
concentrations, geometry and absolute polarized amount.

Stored `initial_mz`, `initial_polarization` and inflow polarization curves
remain thermal-relative solver quantities. `PhantomDesign.temperature_k` and
`MousePerfusionConfig.temperature_k` record their thermal reference. Designs
without this field use 310.15 K (37 °C), matching the existing default target
reference, while keeping their stored relative values unchanged. Phantoms
built from these designs include `metadata["temperature_k"]`; receiver run
metadata carries it forward. Calibrated processing and the SNR budget use
this explicit object temperature, while objects without temperature metadata
retain the calibration's configured target-temperature fallback. The measured
reference phantom temperature and sensitivity remain unchanged.

The mouse designer's injection recipe is concentration in mM and volume in µl:

$$
n\,[\mathrm{\mu mol}]
= c\,[\mathrm{mM}]\,V\,[\mathrm{\mu l}]\,10^{-3}.
$$

The derived dose and its fully polarized equivalent are shown beside the
inputs. A new mouse injection starts at 80 mM and 250 µl with 10000 × thermal
polarization (5.8 % at 7 T and 37 °C): 20 µmol nuclei and 1.16 µmol fully
polarized equivalent. The perfusion source
remains normalized to this dose on every spatial grid. Both recipe fields are
saved; `injection_amount_umol` remains the derived amount for existing APIs.
An older file containing only an amount keeps that dose and explicitly reports
that concentration and volume were not recorded. Enabling **Specify
concentration and volume** supplies a new recipe; no unique recipe is inferred
from the old amount. Zero concentration or zero polarization is supported.

### Display units

**Settings → Interface → Signal units** selects physical amounts (the default)
or SNR units. The choice is saved as `sequence/signal_unit` in QSettings and
redraws the current result without another simulation. It applies to the
Sequence signal, Cartesian images (including frames and volumes), CSI FIDs
and spectra, split views, and spiral images. The Reconstruction Explorer
retains its own display conventions for now.

For calibrated runs the physical display multiplies the simulated
$\mathrm{mol}\times P/P_{\mathrm{th}}$ signal by the calibration's target
thermal polarization. Values then represent mol of fully polarized nuclei;
axes choose their SI prefix automatically, and information labels use µmol
where appropriate. Thus 80 mM in 200 µl at 20 % polarization gives 3.2 µmol
fully polarized equivalent. The Cartesian inverse FFT divides by the spatial
voxel count, so each voxel holds its share of the amount. Spectra use
$\mathrm{FFT}/N_{\mathrm{spec}}$, preserving the initial FID amplitude when
summing complex frequency bins.

In SNR units each chronological ADC sample is divided by its own event's
per-quadrature noise standard deviation:

$$
s_i^{\mathrm{display}} = \frac{s_i}{\sigma_i}.
$$

Spatial FFTs use orthonormal normalization and the spectral FFT is divided by
$\sqrt{N_{\mathrm{spec}}}$. Fully sampled Cartesian images, CSI voxel FIDs
and CSI frequency bins therefore retain $\sigma=1$ per complex quadrature.
This refers to the complex noise before taking magnitudes or combining coils
with RSS; magnitude noise does not have standard deviation one. Different
event bandwidths are whitened before reconstruction. Signal and noiseless
species curves use the same sample factors along their last (ADC) axis.

Spiral gridding is not unitary. Each displayed frame is divided by the global
per-quadrature RMS of the reconstructed noise-only realization. Its image
information explicitly identifies this estimate: it is realization-dependent
and does not describe spatial variations or correlations of the gridded noise.

Physical display requires a calibrated run; SNR display requires valid receiver
noise metadata. Otherwise the display falls back to simulation units with an
explanation. Live previews show simulation units because receiver processing
occurs after the Bloch calculation. Stored results, exports, magnetization,
animation and calibration data remain in their original units.

The SNR budget's reference FID also follows the SNR choice: ideal and noisy
curves are divided by the measured reference $\sigma$, with dashed lines at
$\pm1$. In physical display mode this reference plot retains its measured
amplitude and noise units (or the relative mode's amplitude of one).

### SNR budget

Beside the controls, the receiver tab shows an **SNR budget**: the chain from
the receiver setting to the voxel SNR of the current object and sequence. It
needs no Bloch simulation and updates while values are typed and when the
object, sequence, nucleus or field strength changes. Its sections are:

1. **Receiver setting** (relative mode): SNR and reference bandwidth.
   **Reference phantom** (calibrated mode): observed nuclei $n_{\mathrm{ref}}$,
   polarization, fully polarized equivalent $n_{\mathrm{ref}}P_{\mathrm{ref}}$,
   the transverse fraction $\sin\alpha$ or, with the reference scan model, the
   reference signal fraction $\eta$ and its inputs, measured and single-scan
   SNR, sensitivity $k$ per µmol and
   reference bandwidth. Entered values that are not set yet appear as a
   preview for the current nucleus and field. **Simulation units** gives the
   signal unit (mol × $P/P_{\mathrm{th}}$ at $T_{\mathrm{sim}}$), the noise
   factor and $\sigma_{\mathrm{sim}}(\Delta t_{\mathrm{ref}})$.
2. **Simulated object**: $A_{\mathrm{ideal}}$ and the whole-object FID SNR
   $A_{\mathrm{ideal}}/\sigma(\Delta t_{\mathrm{ref}})$, the single-scan ceiling.
   Calibrated mode also converts $A_{\mathrm{ideal}}$ to the fully polarized
   equivalent $A_{\mathrm{ideal}}P_{\mathrm{th}}$ in mol: 3.2 µmol and SNR 521.7
   for the pyruvate example above.
3. **Sequence readout**: one row per ADC dwell with the whole-object SNR per
   sample at that bandwidth, $\mathrm{SNR}_{\mathrm{ideal}}
   \sqrt{\Delta t/\Delta t_{\mathrm{ref}}}$, and the events, samples and σ.
4. **Image voxels**, for sequences that declare a spatial grid (matrix, FOV
   and encoding frame): the ideal amplitude map is binned by voxel centre into
   image voxels. Voxel $n$ of an axis with $N$ cells of width $d$ covers
   $[(n-N/2)d,\,(n-N/2+1)d)$ along read, phase and partition; a 2D grid sums
   through the excited slab. The rows give the grid, voxel size, voxels with
   signal and their equivalent number of equally bright voxels
   $(\sum_v a_v)^2/\sum_v a_v^2$, the brightest voxel's share
   $a_{\max}/A_{\mathrm{ideal}}$ of the whole-object FID, its SNR in one sample
   $a_{\max}/\sigma(\Delta t_{\mathrm{ref}})$, and the fraction of the object
   outside the FOV, which aliases.
5. **Last run**, from the noiseless signal of the current result and therefore
   including flip angles, relaxation, delivery and metabolism: the strongest
   sample $\max_i |s_i|/\sigma_i$, the matched-filter SNR of all samples
   $\mathrm{SNR}_{\mathrm{tot}} = \sqrt{\sum_i |s_i|^2/\sigma_i^2}$, the same
   per species for spectral and dynamic phantoms, and the brightest voxel over
   all samples.

The last value uses a sum rule. An orthonormal reconstruction keeps the noise
std per quadrature in every voxel and frame and preserves signal energy
(Parseval), so

$$
\sum_{v,\,\mathrm{frames}} \mathrm{SNR}_v^2 = \mathrm{SNR}_{\mathrm{tot}}^2 .
$$

If every frame keeps the object's distribution $a_v$, voxel $v$ reaches
$\mathrm{SNR}_{\mathrm{tot}}\,a_v/\sqrt{\sum_u a_u^2}$, combined over frames.
This is exact for fully sampled Cartesian imaging and EPSI. Spiral, radial and
undersampled reconstructions are not unitary, so the value is an upper bound
there. Decay during the readout and species or frames with other spatial
distributions move individual voxels away from it, while the sum still holds.
The NumPy default inverse transform scales signal and noise alike and leaves
voxel SNR unchanged.

Below the table a reference FID draws the receiver setting: relative mode
normalizes the ideal FID to one with $\sigma = 1/\mathrm{SNR}$, calibrated mode
uses the measured amplitude and noise std. Its noise comes from the production
sampler with the current seed, dashed lines mark $\pm\sigma$ per quadrature,
and its 20 ms T2* is illustrative only; no budget value uses it. **New
realization** redraws it and also works without a simulation result; **Apply
to result** processes an existing result.

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
