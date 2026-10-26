# Bloch Simulator – User Guide

This guide covers what is **not obvious from the interface**: how the parts fit
together, which settings mean something different from what you might expect,
and where results can mislead you. Buttons that do what they say are not
described.

**Contents**

1. [Install and launch](#1-install-and-launch)
2. [The two workspaces](#2-the-two-workspaces)
3. [Free Mode](#3-free-mode)
4. [Sequence Mode](#4-sequence-mode)
5. [Phantoms](#5-phantoms)
6. [Receiver noise and SNR](#6-receiver-noise-and-snr)
7. [Analysis tools](#7-analysis-tools)
8. [Saving and exporting](#8-saving-and-exporting)
9. [Performance and kernels](#9-performance-and-kernels)
10. [Troubleshooting](#10-troubleshooting)
11. [Python API](#11-python-api)

---

## 1. Install and launch

```bash
pip install blochsimulator
blochsimulator-gui
```

In a notebook, use `%pip install blochsimulator`. Standalone apps for macOS and
Windows and the macOS first-launch workaround are described in the
[README](../README.md).

From a local Jupyter notebook, `!blochsimulator-gui` starts the GUI (not on
remote servers or Colab).

---

## 2. The two workspaces

Switch with the **Workspace** selector at the top.

| | Free Mode | Sequence Mode |
| --- | --- | --- |
| Purpose | Understand spin behaviour: RF pulses, relaxation, off-resonance, slice profiles | Simulate a complete acquisition as a scanner would |
| Object | A set of isochromats over frequency and/or position | A 3D phantom (shapes, spectra, kinetics) with B1 Tx/Rx fields |
| Sequence | Built-in waveform sequences (FID, SE, GRE, IR, …) | Pulseq `.seq`, generated in the GUI or loaded |
| Output | Magnetization, signal, spectrum, spatial profile, 3D vector | ADC data, k-space, reconstructed images/spectra |

![Free Mode](_static/media/free_mode_preview.png)

The typical Sequence Mode workflow is **Sequence → Phantom → B1 → Run →
Reconstruction Explorer**. Starting a run without a phantom opens the Phantom
Designer.

---

## 3. Free Mode

**Endpoint vs. time-resolved.** *Mode → Endpoint* stores only the final state
and is much faster. The 3D Vector animation, the spatial heatmap over time and
GIF/MP4 export need *Time-resolved*.

**Spectrum and Spatial tabs** show the response over frequency and over
position. A non-selective hard pulse shows ringing in the spectrum; for a slice
profile, use *Slice Select + Rephase* and set *Positions* and *Range* to cover
a few slice thicknesses.

![Spectrum](_static/media/free_mode_spectral_view.png)
![Spatial](_static/media/free_mode_spatial_view.png)

**Custom pulses.** *RF Pulse Design → Load from File* reads `.exc` (Bruker),
`.dat`, `.txt` and `.csv`. For text files you are asked for the column layout.
Select the *Custom* sequence to use the loaded pulse.

**Parameter Sweep** (right panel) varies one parameter (flip angle, TE, TR, T1,
T2, off-resonance, …) and plots the selected metrics. Results export as CSV or
NumPy.

**Live mode** (3D Vector tab) applies 90°/180° pulses and Gx/Gy/Gz gradients
interactively, without running a sequence.

---

## 4. Sequence Mode

![Sequence Mode](_static/media/sequence_mode_basic.png)

### 4.1 Things that are easy to get wrong

**Sequence reference vs. spectrum centre.** The *Sequence reference* (ppm) is
what the sequence treats as 0 Hz; all RF and receiver offsets are relative to
it, and it is stored in the `.seq` file. The phantom's *Spectrum centre* and
*Bandwidth* only decide which frequencies are simulated. If the reference
(orange dashed line in the phantom spectrum) lies outside the simulated window,
you get a warning before the run — usually this means your excitation hits
nothing.

**Generate sequence.** Editing parameters does not update the timeline until
you click **Generate sequence** (or enable *Live preview*, which can be slow for
long sequences). Running a simulation always generates first. If generation
fails, the previous valid timeline stays visible — check the error above the
form.

**Typing `0` in a timing field** (TE, TR, delays) sets the shortest value the
sequence allows.

**TR vs. acquisition interval.** *TR* is between excitations or k-space lines.
*Acquisition interval (start-to-start)* is between complete images, volumes or
radial measurements; *Back-to-back* means no idle time. This matters for
hyperpolarized time series.

**Orientation.** You choose signed *Read* and *Phase* gradient directions; the
slice/partition axis is derived as Read × Phase (right-handed), e.g. Read `+z`,
Phase `+y` → Partition `−x`. FOV and matrix are always given as (read, phase,
partition), not (x, y, z).

**Slice gap** is the edge-to-edge gap; centre spacing = thickness + gap.

**Units.** All geometry fields are in mm. Scanner hardware limits (gradient,
slew, rasters, dead times) are set once under **Tools → Settings → Scanner**
and apply to all *generated* sequences; imported `.seq` files keep their own
timing.

**Loaded RF pulses** keep duration, phase modulation and carrier offset, but
their amplitude is rescaled to the sequence flip angle.

**SLR pulses** are true Shinnar–Le Roux designs for the chosen flip angle, so a
90° pulse has a different shape than a small-tip one. *SLR sharpness* sets the
time-bandwidth product (TBW = sharpness + 1); the TBW field is ignored for SLR.
The pulse edges are a few percent of the peak, not exactly zero.

**The result tab is not switched automatically** after a run. Look in
*Signal* for the ADC signal, FID or CSI spectrum, or in the
*Reconstruction Explorer* (its tab turns green when a new result is ready).

### 4.2 Spoiling: ideal vs. physical

**Settings → Simulation → Spoiler simulation** chooses between two models.
This is the most common source of unexpected results.

| | **Ideal crusher** (default) | **Gradient waveform** |
| --- | --- | --- |
| What happens | At each marked spoiler end, Mxy is set to 0 | Spoiling arises from real gradients acting on subvoxel spins |
| Spins per voxel | 1 | X × Y × Z subvoxel grid |
| Cost | Fast | Scales with spins per voxel |
| Use when | You want the intended spoiled steady state | You study spoiling itself, stimulated echoes, imperfect crushers |

- Imported `.seq` files are treated as ideally spoiled **only** if they carry
  an `IdealSpoilerEndTimes` marker (files exported by this simulator do).
  Other gradients are never guessed to be spoilers.
- In gradient mode, what matters is the phase spread **within one phantom
  voxel**, not cycles across the FOV. A regular midpoint grid with `N` points
  per axis rephases completely at `N` cycles/voxel — so a grid that looks
  perfectly spoiled after one crusher can return to 100 % signal after a few
  TRs. Example: 3 × 3 × 3 at 1 cycle/voxel is fine after one crusher, but fully
  rephased at crusher order 3.
- *Auto adjust subvoxel grid* (on by default) picks the cheapest grid with
  < 1 % weighted error over the whole RF train. The **Spoiling Analysis** tab
  shows why. To verify any grid, run again with twice the spins per axis.
- *Jittered stratified points* avoid grid echoes but leave noise of roughly
  `1/√(spins per voxel)`.
- Through-slice subvoxel sampling needs a 3D phantom with explicit Z extent.

### 4.3 Sequence notes

Only the non-obvious points per sequence:

| Sequence | Notes |
| --- | --- |
| **FLASH (2D)** | *Auto spoiler* sets the gradient moment to the first coherence null, but does not guarantee a finite spin grid stays dephased over the train — check the train-wide warning. |
| **EPI / spiral** | *Flyback* EPI avoids alternating readout polarity at the cost of time. EPI TE targets the k-space centre; centre-out spiral TE targets the first sample. Spiral readout is lengthened automatically if bandwidth exceeds gradient limits. |
| **EPSI / MGE (2D)** | The echo dimension is also the spectral axis: spectral BW = 1/ΔTE, resolution = 1/(N<sub>echo</sub>·ΔTE). |
| **CSI** | Spectral bandwidth is a separate field from imaging readout bandwidth. |
| **SS-bSSFP (3D)** | Alternates volumes between metabolites. Use *Use selected phantom peak frequencies* to avoid RF/receiver vs. phantom frequency mismatch. A flip angle of 0° disables that target but keeps timing. Defaults follow [Skinner et al. 2023](https://doi.org/10.1002/mrm.29676). Check with `examples/validate_ss_bssfp_spoiling.py`. |
| **ME-bSSFP (3D, Cartesian)** | Defaults follow Gaubatz 2023 (the GUI starts with a smaller matrix). The IDEAL estimate in the explorer uses known frequencies and **does not fit B0** — not for quantitative use. |
| **Radial ME-bSSFP (3D)** | Defaults follow [Wang et al.](https://doi.org/10.1002/mrm.30614); set 300 spokes and 4 measurements to match the paper. Reconstruction is simple trilinear gridding for inspection, not a validated NUFFT. |
| **Inversion Recovery (2D)** | With a *global* inversion, later slices see TI plus their slice offset (see *Slice TIs*); *slice-selective* gives every slice the same TI. Per-slice TIs are stored in the `.seq`, so `sequence.relaxometry.fit_inversion_recovery_t1` can fit T1. FLASH and EPI no longer have IR options — use this sequence. |
| **PRESS** | Read as a 1×1 CSI grid. Crushers only work with a subvoxel grid. Empty *TE1* places the first 180° at TE/2. |
| **Slice-selective FID** | No readout gradient; the spectrum sums the whole slice. |
| **Internal FID** | Magnetization is not reset between repetitions. |

**Presets** (beside *Source / mode*) store all sequence, RF, noise and spoiler
settings — not the phantom. A *Startup preset* replaces the FLASH default.

**Run Python script…** runs your own sequence script with the app's Python; a
`.seq` it writes is imported automatically.

### 4.4 Spectral-spatial (SPSP) pulses

Choose *Spectral-spatial Designer* as RF type. Frequencies are offsets from
the RF carrier, and the response **repeats at the alias spacing** — check
where your metabolites fold. *Excitation map* runs a Bloch simulation over
space × frequency; use it rather than trusting the target table.

- Recreate the design after changing the nucleus (gradient units are
  nucleus-specific).
- With a pulse schedule, FLASH advances **one schedule step per image frame**
  by default. *Every TR/RF event* mixes contrasts across k-space.
- *RF amplitude scale* multiplies the RF only, not the gradient.

Details and validation: [Spectral-spatial RF pulse design](SPECTRAL_SPATIAL_RF.md).

---

## 5. Phantoms

![Phantom designer](_static/media/sequence_mode_phantom_design.png)

### 5.1 Spin density vs. polarization

These are separate. *Spin density / concentration* is how much material is
there; *Initial polarization* is in multiples of thermal equilibrium
(**× thermal**), and `Mz = concentration × polarization`. T1 relaxes towards
1 × thermal, **not** towards zero — a hyperpolarized signal never decays below
the thermal level.

For a pyruvate injection, give lactate spin density 1 but polarization 0
(*Start lactate at 0 × thermal*).

*Common metabolite* adds peaks with typical shift, T1 and T2\*; see
[metabolite presets](METABOLITE_PRESETS.md) for the sources.

### 5.2 Kinetics (kPL, inflow, dynamic B0)

Enable *pyruvate → lactate conversion model* in the **Kinetics / kPL** tab.

- All kinetics run on one **kinetics timeline**. *Kinetics time at sequence
  t=0* shifts the whole sequence relative to it (`+5 s` starts the sequence
  5 s into the kinetics), so inflow and conversion do not need re-editing when
  you compare start times.
- Anything before sequence t=0 is a **pre-roll**: inflow, conversion and T1
  are evolved, but no RF, gradients or B0.
- kPL regions override the shape value; in overlaps the last row wins.
- Inflow is interpolated linearly between rows and zero outside them. Set the
  initial pyruvate density to 0 if the region should be empty before arrival.
- Repetitions do **not** reset the phantom state; RF depletion accumulates over
  the whole sequence.
- The *Live conversion preview* shows one voxel without RF depletion; only the
  full simulation includes RF and gradients.

### 5.3 Mouse perfusion phantom

*Phantom Configuration → Type → Mouse Perfusion Phantom…*

- **¹³C**: pyruvate, lactate, Z-OMPD or a tracer, injected via the tail vein
  with voxelwise arrival delays. Dose = concentration (mM) × volume (µl);
  polarization is entered in absolute percent, including losses before
  injection.
- **¹H**: eight tissue-water pools with 7 T PD/T1/T2, optionally a gadolinium
  agent (extended Tofts) that changes R1/R2.
- Mouse size and position are independent of the FOV; the white box in the
  preview is the simulated FOV.

- *Tissue clearance* and *renal excretion* act in the Bloch simulation, not
  only in the preview: excreted compound moves, T1-relaxed, into ureters and
  bladder.
- **Breathing** is a preview only; it does not move spins or affect gradient
  phase.

### 5.4 B0 and B1

Spatial B0 maps and time-dependent B0 offsets (Hz, separate from RF/receiver
carriers) are set in the phantom. Transmit and receive fields (uniform,
birdcage, surface coil) come from the B1 workspace.

---

## 6. Receiver noise and SNR

The **Receiver / Noise** tab adds complex noise either from a relative SNR or
**calibrated from a measured phantom FID**. Calibration lets one thermal
phantom scan predict SNR for any simulated sequence, e.g. hyperpolarized
in vivo.

1. **Measure** an FID of a phantom with known concentration and volume, plus a
   0° noise scan with the same receiver gain, bandwidth and averages (or use
   the FID's signal-free tail).
2. **Load FID…** reads a ParaVision scan folder (`fid` / `rawdata.job0`) or a
   complex `.npy/.txt/.csv/.dat` array and fills the values. The amplitude is
   extrapolated through the digital-filter delay.
3. Enable **Reference scan model** if the phantom was not fully relaxed or the
   amplitude was read late (corrects for T1 saturation and T2\*). With *Use B1
   workspace coil fields*, apply the same coil to the simulated object.
4. Use *Noise factor (loading)* > 1 for the extra noise of a loaded coil.
5. **Set calibration** freezes the values for the current nucleus and field;
   changing either asks for a new calibration.

The **SNR budget** shows the chain from the reference phantom to the image
voxels. Noise scales with each ADC event's bandwidth. **New realization** and
**Apply to result** re-noise without re-simulating. **Settings → Interface →
Signal units** switches plots between physical units and SNR units (σ = 1).

> [!NOTE]
> Calibrated noise only gives meaningful SNR if the run used physical
> voxel-volume weighting; otherwise you are offered to run again with it.

Formulas: [receiver calibration](sequence_simulation_architecture.md#receiver-noise-and-phantom-calibration).

---

## 7. Analysis tools

### Reconstruction Explorer

![Reconstruction Explorer](_static/media/sequence_mode_reconstruction_explorer.png)

- Display settings (colormap, gamma, interpolation, contrast) affect only the
  view and PNG export; NumPy exports stay quantitative.
- *Show phantom next to the image* resamples the phantom onto the image grid,
  including slice offsets and orientation — useful for checking that slice and
  FOV hit what you intended. Clicking a voxel lists its composition.
- Results opened from `.nc` files contain no phantom.

### Parameter Loop

*Param. Loop* runs one simulation per value of any numeric sequence
parameter, from a list (`30, 60, 90`) or a formula over `x` (e.g. `sin(x)` on
`linspace(start, end, N)`; radians, `pi` and `e` available). Values outside the
form's normal range are allowed.

For large loops choose **Save to folder (limit RAM)**: each run is saved as its
own `.blochproj` and released from memory. Runs kept in memory appear in the
explorer's *Simulation* list and under *Simulations*, where several can be
exported at once.

### Spoiling Analysis

Shows, per coherence pathway, the dephasing per voxel against ADC number, the
voxel signal vs. dephasing for continuous and sampled voxels, and the sampling
error against the 1 % target. **Apply** sets the recommended grid. Only visible
with *Gradient waveform* spoiling.

### PSF Analysis

Simulates a point source through the actual sequence (in the background) and
reports PSF FWHM, broadening, peak shift, side lobes and, for 2D, the slice
profile and cross-talk. It includes relaxation and hyperpolarized depletion
along the k-space order, so it shows the *effective* resolution. Cartesian 2D,
EPI and 3D are supported. Results are marked stale when the sequence changes.

### Spin probe

Follows a spectral/spatial spin distribution through the sequence.
*Y scale → Per-spin response* normalizes so an ideal 90° gives |Mxy| = 1; set
*Y max* to compare time points on a fixed scale.

### Post-run 3D animation

Enable *Create post-run animation* before running. The animation is captured
in a separate replay, so it never changes the simulated signal, but it
increases the run time.

---

## 8. Saving and exporting

| What | How | Contains |
| --- | --- | --- |
| **Project** `.blochproj` | *File → Save Project…* | Everything: settings, phantom, B1, sequence, results, explorer state |
| **Results** | *Export results…* | `.nc` (NetCDF, recommended) + analysis notebook; also HDF5/NPZ, Bruker raw (experimental) |
| **Sequence** | *Export Pulseq…* | `.seq` and/or a notebook that regenerates it |
| **Free Mode results** | *File → Export Results* | HDF5 + analysis notebook; optionally a *reproducible* notebook that re-runs the simulation |

*File → Project Explorer…* indexes folders of projects and shows what each
contains without opening it. The default folder for all dialogs is set under
**Tools → Settings → General → Default export directory**.

### Working with exported data

> [!IMPORTANT]
> `signal` is stored **chronologically**. Do not reshape it by length — use the
> exported indices (`adc_event_index`, `readout_sample_index`,
> `repetition_index`, `partition_index`, …) and `kx/ky/kz` (cycles/m).

Ready-made arrays are included where possible:

- 2D CSI: `csi_kspace` `(…, phase_y, phase_x, spectral_point)`, `csi_spectrum`
- Cartesian 3D: `cartesian_3d_kspace`, `cartesian_3d_image`, ordered
  `(partition_*, phase_*, read_*)` with the scanner axis as suffix, e.g.
  `(partition_z, phase_y, read_x)`
- Radial 3D: `radial_3d_gridded_kspace`, `radial_3d_image` (inspection only)
- Dynamic phantoms: `species_signal`, `final_pool_magnetization`

The run time is stored as `simulation_wall_time_s` in the dataset attributes.

---

## 9. Performance and kernels

Keep the defaults (**Sequence Bloch kernel → Optimized**, **Dynamic spectral
kernel → Native automatic**). The *strict reference* and NumPy kernels are for
validation, not speed.

What actually drives runtime:

- **Spins** = phantom voxels × subvoxel spins (× spectral components). The
  subvoxel grid of gradient-waveform spoiling multiplies everything.
- **RF-active time step** (Settings → Simulation): coarser is faster but
  changes RF accuracy.
- Number of ADC samples and intervals. The phantom's *spectral points* setting
  alone does not create extra Bloch states.
- More threads do not scale linearly, especially for dynamic phantoms.

Free Mode: use *Endpoint* and fewer positions/frequencies.

Implementation details, fallbacks and numerical contracts:
[Simulation kernels](simulation_kernels.md).

---

## 10. Troubleshooting

| Problem | Fix |
| --- | --- |
| No signal / image is empty | Sequence reference outside the phantom spectrum window, or RF/receiver offsets miss the peaks. Check the orange line in the phantom spectrum. |
| Signal returns after spoiling | Subvoxel grid rephases (§4.2). Use *Auto adjust* or *Ideal crusher*. |
| Lactate stays zero | Lactate spin density is 0, kPL is 0, or conversion starts after the sequence. |
| Timeline doesn't change | Click *Generate sequence*; check the error above the form. |
| Large `.seq` loads slowly | Loading runs in the background (progress in status bar); RF is rasterized only when you run. |
| DICOM/NIfTI export unavailable | `pip install "blochsimulator[export]"` |

---

## 11. Python API

```python
from blochsimulator import BlochSimulator, TissueParameters, SpinEcho

sim = BlochSimulator()
tissue = TissueParameters.gray_matter(3.0)
result = sim.simulate(SpinEcho(te=0.03, tr=1.0), tissue, mode=2)  # 2 = time-resolved
signal = result["signal"]
```

```python
from blochsimulator.project_io import load_project

project = load_project("simulation.blochproj")
phantom, result = project["phantom"], project["sequence_result"]
```

More examples are in the [README](../README.md#python-api),
in `examples/` and `tutorials/`, and in every exported notebook. Compressed
sensing (masks and reconstruction) is available as
`blochsimulator.sequence.compressed_sensing`; see the
[concept](compressed_sensing_concept.md).

Building, packaging and releases: [Developer Guide](DEVELOPER_GUIDE.md).
