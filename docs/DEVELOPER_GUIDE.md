# Developer Guide

This guide explains how to build, package, and release the BlochSimulator application, as well as how to extend it with new pulse sequences and GUI options.

**Note: This paragraph was fully AI-generated and not yet reviewed**

## 1. Environment Setup

The source GUI and PyInstaller build share the Python 3.12 runtime declared in
`.python-version` and the dedicated `.venv-packaging` environment. This keeps
their Python ABI and GUI dependencies aligned.

To set up the environment (handled automatically by the build script, but good to know):
1.  Ensure Python 3.12 is installed.
2.  Run `./scripts/run_gui.sh` for source development or
    `./scripts/build_pyinstaller.sh` for an app build. Both commands create or
    reuse `.venv-packaging`, install the current repository in editable mode,
    and reject an environment from another Python minor version.
3.  If necessary, select the interpreter explicitly with
    `BLOCH_PYTHON=/path/to/python3.12`.

**Note:** The `.venv-packaging` directory contains many files but is configured to be ignored by `.gitignore`. You should **not** commit it to the repository.

### Conda for notebooks and the Python API

For a separate notebook or Python API environment, run these commands from the
repository root:

```bash
conda create -n bloch-simulator python=3.12 pip
conda activate bloch-simulator
python -m pip install "ipykernel==7.4.0" ".[gui,pulseq]"
```

Use `-e ".[gui,pulseq]"` in place of `".[gui,pulseq]"` for an editable
development installation. Select the `bloch-simulator` environment as the
kernel in your notebook editor. If the editor requires a registered kernel,
register it from the active environment:

```bash
python -m ipykernel install --user --name bloch-simulator --display-name "BlochSimulator (Python 3.12)"
```

Specify `python=3.12` explicitly: creating an environment without a Python
version lets Conda choose one when Python is installed as a dependency of pip.
The version in `.python-version` is used by the project's launcher and build
scripts; it does not constrain `conda create`.
Python 3.12 matches the desktop development runtime, and
[ipykernel 7.4.0 requires Python 3.11 or later](https://pypi.org/project/ipykernel/7.4.0/).

`h5py` is already a required dependency in `pyproject.toml`, so pip installs it
automatically. A compatible pre-built wheel includes the HDF5 libraries needed
at runtime. If pip downloads an `h5py-*.tar.gz` archive instead, it attempts a
source build that requires an installed HDF5 library and development headers.
An error mentioning `Unable to load dependency HDF5` or a missing
`libhdf5.dylib` indicates that this build cannot find HDF5. See the
[h5py installation documentation](https://docs.h5py.org/en/stable/build.html).

To let Conda provide `h5py` and its HDF5 dependency, add `h5py` when creating
the environment, then run the activation and pip commands above:

```bash
conda create -n bloch-simulator python=3.12 pip h5py
```

This is an alternative to installing `h5py` through pip. The desktop launcher
and PyInstaller build continue to use their dedicated `.venv-packaging`
environment.

## 2. Release Workflow (Recommended)

Follow these steps to publish a new version of BlochSimulator. This process is highly automated via GitHub Actions.

### Step 1: Bump Version
Use the included helper script to update version numbers across all files (`pyproject.toml`, `setup.py`, etc.).

```bash
# Replace 2.6.5 with your new version number
python bump_version.py 2.6.5
```

This script will:
*   Update version strings in `pyproject.toml`, `setup.py`, `src/blochsimulator/simulator.py`, `src/blochsimulator/gui.py`, `docs/conf.py`, and `src/blochsimulator/__init__.py`.
*   Print the exact git commands you need to run to commit and tag the release.

### Step 2: Commit and Push the Version Bump

Commit the updated version files and push the commit to the appropriate release
branch. For a stable release, use `main`:

```bash
git add .
git commit -m "Bump version to 2.6.5"
git push origin main
```

### Step 3: Create and Push the Release Tag

Wait for the push-triggered `Tests` workflow to succeed on the exact release
commit. Only then create the tag on that commit and push it explicitly:

```bash
git tag v2.6.5
git push origin v2.6.5
```

Pushing the branch does **not** push a newly created tag automatically. Verify
that `git status` shows the intended branch and that `git show v2.6.5` points to
the version-bump commit before pushing the tag. If you are preparing a
pre-release, use the corresponding development tag (for example,
`v2.6.5.dev1`) and the `pre-release` branch instead.

### Release Paths

#### Pre-release validation path

Use this path to exercise the complete GitHub Actions release workflow before
merging to `main`. Development tags create a GitHub pre-release with the built
wheel and application artifacts, but are never uploaded to PyPI.

1. Start from the `pre-release` branch and update the project to the intended
   final version (for example, `2.3.0`).
   ```bash
   git switch pre-release
   git pull --ff-only origin pre-release
   python bump_version.py 2.3.0
   git add pyproject.toml setup.py src/blochsimulator docs/conf.py web index.html
   git commit -m "Bump version to 2.3.0"
   git push origin pre-release
   ```
2. Wait for the `Tests` workflow triggered by this push to finish successfully.
   The successful `Full test suite` job must belong to this exact commit.
3. Create and push a development tag such as `v2.3.0.dev1`.
   ```bash
   git tag v2.3.0.dev1
   git push origin v2.3.0.dev1
   ```
   The release workflow requires this tagged commit to be reachable from
   `pre-release` and to have a successful push-triggered `Tests` run on that
   branch. It temporarily sets the package version to `2.3.0.dev1` while
   building, creates a GitHub pre-release, and does not publish to PyPI.
4. Inspect the GitHub Actions run and the generated GitHub pre-release assets.
   If another validation build is needed, push a later commit, wait for its
   tests to pass, and use the next development tag (for example,
   `v2.3.0.dev2`). Do not reuse a published tag.

#### Final stable-release path

1. Merge the tested `pre-release` commit into `main`, then wait for the
   `Tests` workflow triggered by the push to `main` to finish successfully.
2. Tag that exact tested `main` commit with the final version and push the tag.
   ```bash
   git switch main
   git pull --ff-only origin main
   git tag v2.3.0
   git push origin v2.3.0
   ```
   A stable tag is accepted only when its commit is reachable from `main` and
   has a successful push-triggered `Tests` workflow on `main`.
3. The workflow builds the PyPI distributions and application artifacts,
   uploads the Python distributions to PyPI, and creates the stable GitHub
   Release with the application artifacts.

---

## 3. Manual Build and Packaging (Optional)

If you need to build the application locally for testing, use these scripts.

### Building the Application
```bash
./scripts/build_pyinstaller.sh
```
**What this does:** Creates the virtual environment, installs dependencies, builds Cython extensions, and runs PyInstaller. Outputs to `dist/BlochSimulator.app` (macOS) or `dist/BlochSimulator` (Windows/Linux).

### Packaging for Release
```bash
./scripts/package_for_release.sh
```
**What this does:** Compresses the app into a `.zip` or `.tar.gz`, automatically naming it based on the current Git tag and architecture.

---

## 4. User Installation Instructions

Since the app is not signed with an Apple Developer ID, users will see a "damaged" or "unidentified developer" warning.

**Instructions for macOS:**
1. Download and unzip the application.
2. Move `BlochSimulator.app` to your Applications folder.
3. **Important:** Run this command in Terminal to allow the app to run:
   ```bash
   xattr -cr /Applications/BlochSimulator.app
   ```
4. Double-click to open.

---

## 5. Automated CI/CD (GitHub Actions)

The repository uses two main workflows:

### A. PyPI Release (`publish.yml`)
Triggered on push to `main` (test only) and on tags `v*` (publish).
- **`test`**: Runs `pytest` on Linux.
- **`build_wheels`**: Builds binary wheels for all OSs.
- **`publish`**: Uploads to PyPI (only on tags).

### B. Standalone App Build (`build_standalone.yml`)
Triggered on tags `v*` or manual dispatch.
- **`build`**: Builds standalone executables on macOS, Windows, and Linux.
- **`release`**: Creates a **Draft Release** on GitHub with the binaries.

### Pre-flight Check
To test your code before tagging:
1.  Push to `main` without a tag.
2.  Check the **Actions** tab on GitHub.
3.  If `test`, `build_wheels`, and `build_sdist` pass, it is safe to proceed with the release.

---

## 6. Package Configuration (MANIFEST.in)

The `MANIFEST.in` file tells `setuptools` which non-Python files to include in the **Source Distribution (`sdist`)**. This is critical for users installing from source who need the C/Cython files and RF pulse data.

Ensure any new asset directories are added here to be bundled with the library.

---

## 7. Extending the Simulator

### Spectral-Spatial Pulse Schedules

`sequence.spectral_spatial` keeps waveform identity separate from ordered use:

- `SpectralSpatialPulseLibrary` maps stable IDs to complete
  `SpectralSpatialPulse` objects.
- `SpectralSpatialSchedule` owns ordered entries and cyclic selection.
- `SpectralSpatialScheduleEntry` references a pulse ID and stores names,
  enabled state, repeat count, and either a flip-angle request or RF amplitude
  scale.

Use `migrate_spectral_spatial_state` when accepting project state. It converts
the old `sequence_spectral_spatial_pulse` singleton to deterministic legacy
IDs. Use `validate_spectral_spatial_schedule` after the final target thickness
and scanner profile are known; this repeats B1, gradient, slew, duration, and
nucleus checks after entry scaling.

`make_pulseq_flash` accepts `rf_spectral_spatial_library` and
`rf_spectral_spatial_schedule`. The default selection level is `frame`, meaning
one complete FLASH image per expanded schedule step. The `tr` level is
explicitly labelled as contrast mixing. Schedule index uses the standard
Pulseq `SET` label, while reproducibility data is stored in
`SPSPSchedule*` definitions. The importer retains both. Simulation-result
metadata copies the definitions, ADC label values, and the schedule-definition
subset.

`make_pulseq_epi` takes the same two arguments. Its `frame` level advances
the schedule once per repetition, so every slice of that repetition shares
one pulse. Its `tr` level advances once per slice excitation, so each
single-shot slice image carries one pulse. Each excitation carries the
schedule index in `SET`, and the same `SPSPSchedule*` definitions are written.
The echo-time minimum is checked for every pulse variant. Variable flip
angles cannot be combined with a schedule and raise `ValueError`. The GUI
enables the multi-step schedule dialog for FLASH and Cartesian EPI
(`_spectral_spatial_schedule_supported`); all other sequences keep one
pulse.

The pure `spectral_spatial_peaks` function is the canonical conversion from
absolute phantom/preset ppm to sequence-relative and carrier-relative Hz.
`simulate_spectral_spatial_schedule_preview` is the UI-independent reduced-grid
preview and propagates complex magnetization through full RF/gradient samples.
Keep future GUI changes in the schedule dialogs and shared plotting helpers;
do not duplicate the frequency or Bloch logic in widgets.

### Inversion Recovery and T1 Fitting

`sequence.inversion_recovery.make_pulseq_inversion_recovery` is the
standalone multi-slice IR sequence. The GUI lists it as the
"Inversion Recovery (2D)" source. `readout` selects the acquisition after
each inversion:

- `epi`: one single-shot EPI image per slice and inversion.
- `gre`: classic IR with one gradient-echo line per slice and inversion. The
  inversion repeats for every phase-encoding line.
- `turboflash`: the whole low-flip FLASH image follows one inversion
  (MP-RAGE-like). TI targets the k-space centre line.

`inversion_mode` places the inversion:

- `global` (default): one non-selective 180° followed by
  TI - 90° slice 1 - 90° slice 2 ... 90° slice N. Slice k sees TI plus the
  k preceding excitation-and-readout durations. Those offsets are recorded in
  `InversionSliceTimeOffsets`.
- `slice_selective`: every slice gets its own inversion right before its
  excitation, so every slice sees the requested TI. The inversion slab is
  `inversion_slab_factor` times the slice thickness. Without the overlap of
  the slice train and the TI delay, the inversion TR grows by roughly the
  number of slices. A block pulse has no slice profile and is rejected. The
  error for a too-steep inversion gradient names the needed mT/m.

`slice_order` is `sequential`, `interleaved` (even then odd slices) or
`cyclic`. The cyclic order rotates the slice train by one position per TI
series, so every slice is acquired at every train position.
`slice_acquisition_order` returns the order for one repetition.
`repetition_time_s` is the inversion-to-inversion time, and
`readout_repetition_time_s` is the gradient-echo line TR. `None` selects the
shortest. `repetitions` defaults to one complete TI series, or one series per
slice with the cyclic order. `timing_only=True` returns an
`InversionRecoveryTimingPlan` with the TE, line TR, inversion TR and TI
minimums. All other keywords are passed to `make_pulseq_epi` or
`make_pulseq_flash`. The arguments the wrapper sets itself raise `TypeError`.

The builders underneath accept the same preparation, so FLASH and EPI with
`inversion_times_s` still work on their own:

- `inversion_pulse_type`, one of `INVERSION_PULSE_TYPES`: `hyperbolic_secant`
  (adiabatic, default), `sinc` or `block`;
- `inversion_duration_s`;
- `inversion_spoiler_cycles_per_slice`;
- `inversion_mode`, `inversion_slab_factor` and `slice_order` as above;
- FLASH only: `inversion_interval` (`image` or `line`), with
  `acquisition_interval_s` and `inversion_repetition_time_s` as the spacing
  of the inversions.

Several TI values cycle across repetitions, so one sequence produces a T1
mapping series. TI is measured from the inversion centre to the excitation
that samples the k-space centre of the first slice in the train. A TI below
the minimum raises `ValueError` and names the minimum. The minimum is the
second half of the inversion pulse, the crusher, and the time up to the
target excitation centre. The definitions record:

- `RequestedInversionTimes` and `InversionTimes`: the requested and actual
  values per repetition;
- `SliceInversionTimes`: the actual TI of every repetition and slice;
- `SliceAcquisitionOrder`: the spatial slice index of every excitation;
- `InversionMode`, `InversionSelectivity` and `InversionSlabThickness`.

The GUI source has readout-specific defaults for flip angle, TE and the TI
list (10° and 250-2500 ms for TurboFLASH). It raises a too-short TE, line TR
or inversion TR to the minimum and lists the per-slice TI offsets after
generation. The FLASH and EPI forms keep their own "Inversion recovery"
section.

`relaxometry.slice_inversion_times(definitions)` reshapes the slice TIs to
(slices, repetitions). They can be passed directly to the fit, because it
broadcasts TI against the signal.

In the three-parameter fit a constant per-slice TI offset `d` does not bias
T1: `a - b exp(-(TI + d)/T1)` is the same model with `b exp(-d/T1)`. A fit
with the nominal TIs then finds the same T1 and only a smaller apparent
inversion efficiency for later slices. The true slice TIs matter for
two-parameter fits, for the inversion efficiency, for the dynamic range
near the zero crossing, and for the cyclic order, whose offset changes per
series.

`sequence.relaxometry.fit_inversion_recovery_t1` fits
`S(TI) = a - b exp(-TI/T1)` voxelwise, with TI on the last axis:

- Complex data is rotated by the phase of the longest-TI sample and then
  fitted on its real part.
- Magnitude data uses polarity restoration (NLS-PR, Barral et al., DOI
  10.1002/mrm.22497).
- The search is a log-spaced T1 grid with closed-form `a` and `b`.

`InversionRecoveryFit` also reports the inversion efficiency `b/a - 1` and
the number of restored samples. `inversion_recovery_signal` is the forward
model. The fit is not yet wired into the Reconstruction Explorer.

### Single-Voxel PRESS

`make_pulseq_press` builds a PRESS sequence:

- **Excitation:** slice-selective 90° on the partition axis.
- **Refocusing:** two slice-selective 180°s, on the phase and read axes, with
  a CPMG phase of 90° (`RefocusingPhaseDeg`).
- **Echo timing:** TE is `2 (t_refocus2 - t_refocus1)`.
  `first_echo_time_s` (TE1) places the first refocusing pulse at TE1/2;
  it defaults to TE/2.
- **Crushers:** each 180° sits between a crusher pair on its own axis.
- **Readout:** the FID is acquired from the echo top without a read
  gradient.

`voxel_size_m` and `voxel_offset_m` are logical (read, phase, partition)
values and follow `encoding_axes`. `timing_only=True` returns a
`PressTimingPlan` with the TE1, TE and TR minimums.
`refocusing_pulse_type` is one of `PRESS_REFOCUSING_PULSE_TYPES`.

The sequence is labelled as a 1×1 CSI grid, so
`infer_spectroscopic_acquisition` returns a single-voxel
`SpectroscopicAcquisition` and the existing CSI spectrum view displays it.

Simulation caveats:

- **Crushers act across isochromats.** Signal outside the voxel only
  cancels when every phantom voxel is represented by many spins along the
  crusher axes. Use gradient-waveform subvoxel spoiling, or check single
  positions with `simulate_sequence_probes` as `tests/test_press.py` does.
  With one spin per voxel, the FIDs of the refocusing pulses and the
  stimulated echo stay visible.
- **A sinc 180° refocuses a narrower profile than its nominal slab.** The
  refocusing efficiency `sin²(θ/2)` falls faster at the slab edges than the
  90° excitation profile. The effective voxel is therefore slightly smaller
  than `voxel_size_m`, more so for `block` and low time-bandwidth products.

### Compressed Sensing

`sequence.compressed_sensing` is a numpy-only module for retrospective CS on
reconstructed Cartesian grids. It provides:

- variable-density and k-t masks;
- PSF incoherence;
- zero-filled reference;
- the primal-dual reconstruction with Haar wavelet, TV and temporal-difference
  terms.

The builders and acquisition models do not accept undersampled grids yet.
`docs/compressed_sensing_concept.md` describes the measured behaviour, the
retrospective workflow and the staged plan for prospective undersampling
(builder masks, subset `phase_indices`, a reconstruction hook and GUI
controls).

### Choosing Subvoxel Sampling for Imported Pulseq Sequences

#### Why imported sequences need gradient-waveform spoiling

Generated sequences write the end of every spoiler block into the
`IdealSpoilerEndTimes` definition. In **Ideal crusher** mode the simulator sets
`Mx` and `My` to zero at these times. This needs only one spin per voxel and is
fast and exact. An arbitrary imported `.seq` file has no such markers, and the
simulator never guesses which gradient is a crusher. Such files must therefore
be simulated with **Gradient waveform (subvoxel spins)**. In that mode every
phantom voxel is represented by several spins at different positions. Spoiling
then follows physically from the phase that the actual gradients give these
spins. A `.seq` file exported by this simulator keeps its
`IdealSpoilerEndTimes` markers, so Ideal crusher mode stays valid for it after
reloading. For such files the GUI shows neither the load-time spoiling warning
nor a subvoxel recommendation.

#### What the counts mean

**Subvoxel spins along X/Y/Z** are counts **per axis and per phantom voxel**.
A setting of `2 / 2 / 2` therefore gives $2 \times 2 \times 2 = 8$ spins in
every phantom voxel. The dialog shows this product in its **Spins per voxel**
row. Runtime and memory grow with the product $N_x N_y N_z$.

The reference length is the **phantom voxel**, not the imaging voxel. A
crusher that winds 4 cycles across an imaging voxel winds only 1 cycle across
a phantom voxel that is four times finer. The number of spins needed depends
on that phase in cycles across one phantom voxel, written $\varphi$ below.

#### How the spin positions are chosen

`SpinSampling` (`sequence/spin_sampling.py`) splits each phantom voxel into
$N_x \times N_y \times N_z$ equal cells and puts exactly one spin in each cell.
All spins have the same weight $1/(N_x N_y N_z)$. The two methods differ only
in where the spin sits inside its cell. Both are reproducible: the same counts
and the same phantom always give the same positions.

- **Regular midpoint grid** (`midpoint`): the spin sits at the cell centre,
  at $-\tfrac12 + (k + \tfrac12)/N$ along each axis. The pattern is the same
  in every voxel.
- **Jittered stratified points (per voxel)** (`stratified`): the spin sits at
  a pseudo-random position inside its cell. The position is a SplitMix64 hash
  of a point ID, with one independent stream per axis. The ID is
  `voxel_index * grid_spins_per_voxel + point_index`, where `voxel_index` is
  the flat index of the voxel in the phantom. Every voxel therefore has its
  own pattern. Because the ID does not depend on chunking or on which other
  voxels are active, the result is identical for any `chunk_voxels`. Axes with
  a count of 1 stay centred.

`normalized_voxel_offsets(voxel_indices)` returns the offsets of many voxels at
once; `simulate_sequence` uses `voxel_offsets_m` for every chunk when
`varies_per_voxel` is true.

#### How the two methods behave

For a linear phase of $\varphi$ cycles across the voxel on one axis, a
continuous, uniform voxel keeps the coherence $|\mathrm{sinc}(\varphi)|$. This
is zero for every non-zero integer $\varphi$. A midpoint grid with $N$ points
keeps

$$
\left|\frac{\sin(\pi\varphi)}{N \sin(\pi\varphi/N)}\right| .
$$

This is also zero for integer $\varphi$, **except at integer multiples of
$N$, where it is exactly 1**. At those phases the grid refocuses completely
and produces an artificial echo that is the same in every voxel.

A jittered stratified grid is an unbiased estimate of the voxel signal: its
mean over all possible jitters is exactly $\mathrm{sinc}(\varphi)$. Each single
voxel deviates from that mean by a random residual with variance

$$
\frac{1 - \mathrm{sinc}^2(\varphi/N)}{N}
$$

per axis (for several axes, $\bigl(1 - \prod_a \mathrm{sinc}^2(\varphi_a/N_a)\bigr)/(N_x N_y N_z)$).
For large phases this approaches $1/N$, i.e. a residual of $1/\sqrt{N}$.
Because the jitter differs per voxel, these residuals have random phases and
partly cancel when many voxels contribute to the same image pixel or spectrum.

Remaining coherence after one crusher along one axis (target: 0 for
$\varphi \ge 1$). For stratified sampling the table gives the RMS coherence
of one voxel, $\sqrt{\mathrm{sinc}^2(\varphi) + \mathrm{variance}}$:

| $\varphi$ [cycles/voxel] | 0.5 | 1 | 2 | 4 | 8 | 9 | 16 |
|---|---|---|---|---|---|---|---|
| continuous voxel | 0.64 | 0 | 0 | 0 | 0 | 0 | 0 |
| midpoint, $N=2$ | 0.71 | 0 | **1** | **1** | **1** | 0 | **1** |
| stratified, $N=2$ (RMS) | 0.71 | 0.55 | 0.71 | 0.71 | 0.71 | 0.71 | 0.71 |
| midpoint, $N=9$ | 0.64 | 0 | 0 | 0 | 0 | **1** | 0 |
| stratified, $N=9$ (RMS) | 0.64 | 0.07 | 0.13 | 0.24 | 0.33 | 0.33 | 0.33 |

Two consequences matter in practice:

- **Midpoint is exact up to its alias.** With $N$ points per axis it is
  correct as long as every relevant phase stays below $N$ cycles across the
  voxel. Choose $N$ above the largest such phase. An $N$ that shares no factor
  with the crusher's cycle count delays the first alias. With 2 cycles per TR,
  $N = 9$ first rephases after 9 repetitions, whereas $N = 8$ already rephases
  after 4.
- **Stratified trades the alias for noise.** It never produces a coherent
  echo, but each voxel keeps a residual of about $1/\sqrt{N_x N_y N_z}$. With
  few spins this is large (0.71 for 2 points), so stratified is useful only
  when the relevant phases are too high for any affordable midpoint grid and
  many voxels average the residual down.

#### Repeated crushers: the train matters, not one crusher

Transverse magnetization that was excited $n$ repetitions ago carries $n$
times the crusher moment. A midpoint grid that looks perfectly spoiled after
one crusher can therefore refocus exactly after a few repetitions. With
hyperpolarized nuclei, long $T_2$, and small flip angles, many such old
coherence pathways still carry signal. The phase that must stay below $N$ is
the one of the **oldest pathway that still carries signal**, not that of one
crusher.

#### How the GUI recommendation is computed

For imported Pulseq files, `_set_imported_pulseq_spoiling_quality` in
`ui/sequence_simulation_widget.py` does the following:

1. **Skip marker files.** If the file declares `IdealSpoilerEndTimes` (or the
   legacy `SpoilerEndTimes` / `EndImageSpoilerEndTimes`), checked with
   `declares_ideal_spoilers`, no analysis is run. The load-time warning and the
   recommendation are suppressed too, because Ideal crusher mode is valid.
2. **Choose the relaxation.** `_spoiling_object_relaxation` takes the longest
   finite $T_1$ and $T_2$ of the selected phantom (or the built-in object's
   values). Long relaxation keeps old pathways alive longest, so this is the
   conservative choice. Phantoms with metabolite pools are treated as
   hyperpolarized: $T_1$ recovery goes towards 0 instead of towards the initial
   magnetization.
3. **Follow the coherence pathways.** `coherence_pathways_from_program`
   (`sequence/coherence_pathways.py`) runs an extended phase graph with
   continuous 3D moments. Each RF pulse is applied instantaneously at its
   $|B_1|$-weighted centre with its flip angle and phase. It keeps
   $\cos^2(\alpha/2)$ of every transverse state, mirrors $\sin^2(\alpha/2)$ of
   it about the current moment (refocusing, so spin echoes and stimulated
   echoes are included), and exchanges amplitude with longitudinal states.
   Transverse states decay with $T_2$ and longitudinal ones with $T_1$. States
   with the same moment are summed with their complex amplitudes. At most
   1024 ADCs and 4096 states are tracked; the GUI notes when either limit was
   reached.
4. **Weight the pathways.** At every ADC the strongest state is the reference
   (normally the signal that is actually wanted). Every other state gets the
   weight $|a_\mathrm{state}| / |a_\mathrm{reference}|$ and a phase equal to its
   moment relative to the reference, in cycles across the phantom voxel.
   States below 1% are dropped, because they cannot exceed a 1% error target.
   `deduplicate_observations` keeps the largest weight for each distinct phase.
5. **Search the grid.** `recommend_spin_grid_for_pathways`
   (`sequence/spoiling.py`) tests grids in order of increasing total spin
   count. The error of a grid is the largest
   $w \cdot |S_\mathrm{grid} - S_\mathrm{voxel}|$ over all pathways. For
   midpoint this is exact; for stratified it is the expected RMS deviation
   in one voxel. At equal cost midpoint wins. Every axis that carries a phase
   gets at least 2 points; the search stops at 32 points per axis and 512
   spins in total. If no grid meets 1%, the grid with the smallest error is
   reported with a note. The pathway train and the recommendation are cached
   per program, voxel size and relaxation, so changing the grid only
   re-evaluates the error of the current grid.

RF spoiling with a quadratic phase increment randomises the phases of old
pathways but leaves their amplitudes unchanged. A strong pathway with a
weight close to 1 is therefore expected for RF-spoiled FLASH with long $T_2$;
it is not a bug.

If no recommendation can be derived (no phantom selected, or no evaluated
ADC), `loaded_pulseq_spoiler_recommendation` falls back to a **midpoint** start
grid of 4 points per phantom axis (`4 × 4 × 4` in 3D). Regular sampling is
used because 2-point jitter keeps a residual of about 0.7 per axis, while a
4-point midpoint grid is exact below 4 cycles per voxel. This grid is only a
starting point.

Generated FLASH sequences use the analogous `recommend_spin_grid`. It
evaluates every accumulated crusher order $1 \ldots n_\mathrm{exc}-1$ of the
known crusher moment.

#### Spoiling Analysis tab

`ui/spoiling_analysis_view.py` draws the analysis above. Each source of the
Spoiling quality panel in `SequenceSimulationWidget` builds a frozen
`SpoilingAnalysisSnapshot` (title, summary, phases in cycles/voxel per
observation and axis, pathway weights, observation numbers and label, current
`SpinSampling`, recommendation, `weighted` flag) and passes it to
`_publish_spoiling_analysis`:

| Source | Phases | Weights | Observation |
| --- | --- | --- | --- |
| `_update_flash_spoiler_info` (FLASH) | `SpoilerTrainAnalysis.phase_cycles_per_voxel` | 1 | ADC number from 2, or crusher order |
| `_set_configured_spoiling_quality` (EPI, spiral, …) | repeated spoiler orders | 1 | spoiler order |
| `_set_imported_pulseq_spoiling_quality` | `CoherencePathwayTrain.phase_cycles_per_voxel` | train weights | `adc_indices + 1` |

`_set_spoiling_quality_unavailable` publishes `None` with the message, which
the view shows instead of the plots. The curves come from
`spoiling.pathway_grid_coherence(phases, counts_xyz, method)`, which returns
the continuous-voxel coherence, the grid coherence (exact Dirichlet product
for midpoint, RMS $\sqrt{(1-\prod_a \mathrm{sinc}^2(\varphi_a/N_a))/N}$ for
stratified, with singleton axes centred) and their difference. It uses the
same formulas as `weighted_pathway_sampling_error`, so the plots match the
recommendation. Rendering is lazy: `set_snapshot` only marks the view dirty,
and it draws on `showEvent` or immediately when visible. Above
`MAX_PLOTTED_ROWS` pathways only the strongest-weighted ones are plotted.

The view widget is owned by `SequenceSimulationWidget` (unparented until
needed). `MainWindow` adds a placeholder tab after **Receiver / Noise**; on the
first `currentChanged` to that tab `_ensure_spoiling_analysis_workspace`
creates the sequence workspace and reparents the view into the page.
`_spoiling_analysis_tab_wanted` is true in Sequence mode with spoiler mode
`"gradient"`, read from the widget, or from the saved setting before the
widget exists. `_update_spoiling_analysis_tab_visibility` runs on the widget's
`spoiler_mode_changed` signal, on workspace changes and after the settings
dialog; if the tab is hidden while current, the Sequence Simulation tab is
selected first. The view's `apply_recommendation_requested` signal reuses
`_apply_current_recommended_spin_grid`.

The **Show Spoiling Analysis** link in the Spoiling group emits
`spoiling_analysis_requested`, which `MainWindow._open_spoiling_analysis_tab`
answers by selecting the tab when it is wanted.

#### Auto adjust of the subvoxel grid

The `spoiling_auto_adjust` checkbox (setting `sequence/subvoxel_auto_adjust`,
default true) makes the recommendation binding. `_publish_spoiling_analysis`
schedules `_auto_adjust_spin_grid` with `QTimer.singleShot(0)` whenever a
snapshot is published, so the caller finishes its update before the grid
changes. `_auto_adjust_spin_grid` does nothing unless the spoiler mode is
`"gradient"`, the box is checked and the recommended method and counts differ
from the current ones; otherwise it calls `_apply_current_recommended_spin_grid`.
FLASH auto spoilers depend on the grid, so a new grid can produce a new
recommendation. A short history of the grids the chain has left (reset after
1 s or when the box is toggled) stops it from returning to one of them, which
prevents oscillation. `_run` flushes a pending adjustment before it checks the
generated sequence, so a simulation started right after an edit uses the
adjusted grid.

`set_spoiler_configuration(..., auto_adjust=...)` sets the box without saving
the setting; `None` keeps it. The project and run states store
`spoiler_configuration["auto_adjust"]`; restoring a state without the key
passes `False`, so older projects keep their stored grid. The settings dialog
switches auto adjust off (and saves that) when the user changes the persisted
gradient-mode grid there, because that is an explicit manual choice.

#### Known limitations of the analysis

- Heterogeneous phantoms are analysed with a single $T_1$/$T_2$ pair, the
  longest in the object. Regions with shorter relaxation need fewer spins
  than recommended.
- Each RF pulse is treated as instantaneous at its centre. Gradients during a
  pulse (slice selection) and the slice profile are not resolved, so pathways
  of selective pulses are estimated with their on-resonance flip angle.
- Diffusion, flow and off-resonance are ignored. They usually weaken old
  echoes, so the recommendation errs towards more spins.
- Pathways after the last evaluated ADC are not tracked.

These limits concern only the **recommendation**. The Bloch simulation itself
evolves every subvoxel spin exactly under the actual waveform.

#### Recommended procedure

1. If the `.seq` file was exported by this simulator, keep **Ideal crusher**.
2. Otherwise select **Gradient waveform** and apply the recommended grid
   (usually a **Regular midpoint grid**).
3. Raise the counts only on axes that carry crusher moments. Choose $N$ above
   the phase in cycles across one phantom voxel of the oldest pathway that
   still carries signal. A finer phantom lowers this phase directly.
4. Check convergence: run the same simulation again with twice the spins per
   axis on the relevant axes. Doubling moves the first alias of a midpoint
   grid from $N$ to $2N$ cycles. If the signal no longer changes within
   your tolerance, the smaller grid is sufficient. The cheapest converged grid
   is optimal, not the largest affordable one.
5. Use **Jittered stratified points** only when no affordable midpoint grid
   meets the target, and check that the residual noise is acceptable.

### Sequence Simulation Backends and Numerical Contracts

The ordinary sequence solver has two separately compiled extensions. The
historical `blochsimulator_cy` target retains its established `-ffast-math`
build and reports `legacy-fast-math`. The validation-only
`blochsimulator_strict_cy` target uses `-fno-fast-math` and
`-ffp-contract=off`; select it with `sequence_kernel="strict_reference"`.
Keep the source files and object targets separate so an incremental build
cannot accidentally reuse fast-math objects in the strict extension.

The dynamic extension `dynamic_bloch_cy` uses the strict compiler flags. Its
backend-neutral input is the immutable Float64/Complex128
`DynamicExecutionPlan` from `dynamic_execution.py`. This plan owns exact
interval boundaries, RF and gradient samples, both dynamic-$B_0$
half-interval integrals, conversion state, source samples, ordered boundary
actions, and a SHA-256 fingerprint. Every backend adapter must consume this
same plan. Dynamic integration remains Float64.

`dynamic_spin_major_cy` is a second strict Float64 extension, selected
automatically by the native dynamic choices for supported two-pool models.
It advances 64-spin tiles through bounded plan slices and reduces ADC sums
over fixed work blocks. Complex spatial Tx fields are multiplied by the RF
waveform per spin; the spatial rotation uses the same plane basis as NumPy.
Repeated RF segments can use simulation-local affine propagators, keyed by
exact interval inputs. Source samples must fit the source basis within a
relative tolerance of $10^{-12}$; other occurrences remain stepwise.
Cached segments containing internal ADC, crusher, or checkpoint actions are
ineligible. Dynamic B0, spatial inflow delays, additional pools, and smaller
checkpoint dtypes retain the established execution paths.

Use `BLOCHSIM_DYNAMIC_SPIN_MAJOR=0` to compare the previous native driver and
`BLOCHSIM_DYNAMIC_SPIN_MAJOR_RF_CACHE=0` to compare uncached spin-major
execution. Metadata beginning with `native_spin_major_` reports selection,
fallback, timings, memory, and RF-propagator coverage. An older extension
without spatial-Tx support safely retains the previous spatial kernel.

The complete native two-pool driver is deliberately capability-gated. It
executes free half-steps, conversion, unity/constant/spatial Tx, static or
dynamic $B_0$, RF, crushers, ADC reduction, and checkpoints in byte-bounded
time blocks. The same worker uses explicit source modes for coupled
concentration/polarization delivery, direct magnetization inflow, polarized
magnetization inflow without concentration delivery, and concentration tracking
without inflow. Voxelwise arrival delays use eight rate/polarization endpoint
planes prepared only for the current bounded time and voxel block; do not
replace them with a global spin-by-timeline table. A separate multi-pool worker
uses the same plan and action order for the existing precursor-to-products
model. Its coefficient tables include the total precursor loss in every cache
entry and apply the individual conversion rate to each product; this avoids
slowing the specialized two-pool loop. Reduced precision or insufficient
memory retain the established NumPy path. Never drop an unsupported field: add
a fallback reason and expose the actual path in result metadata.

An ADC action at state index zero intentionally disables the complete-plan
driver. It samples the unadvanced initial state through the established
interval path so the existing bitwise contract is retained; this is a
compatibility gate, not an unsupported acquisition event.

The outer dynamic driver may stream the object by parent voxel, but never by
subspin or by time. It compiles the sequence and execution plan once, keeps all
subspins of a parent together for the full timeline. The previous native
driver reduces one leaf per global parent with a fixed binary accumulator.
Spin-major instead retains one reduced signal row per object block. Its
results are bitwise stable across thread counts and time-slice/cache sizes
for a fixed object partition. Changing `parent_voxel_chunk_size` can change
the last Float64 bits and retains the `float64-close` numerical contract.
New backends must preserve the same action order: crusher, then ADC, then
checkpoint at each state boundary.

The GUI deliberately reads these machine-readable result fields instead of
inferring execution from the selected combo-box entry. Ordinary and dynamic
results publish `requested_sequence_kernel`, `sequence_kernel`,
`effective_threads`, and `numerical_contract`; dynamic fallbacks additionally
publish their reason and complete-plan metadata. Stable names, numerical
contracts, optional-extension coverage, experiment status, and fallback
capability live in `kernel_capabilities.py`. `BlochSimulator`, the settings
loaders, the GUI tooltips, and `benchmark_kernels.py` consume that registry;
extend it before adding a new label or dispatch branch. `GUI_KERNEL_NAMES` in
the same module lists the subset offered by the settings dialog, with the GUI
default first; the dialog's combo-box constants must match it. Stored settings
outside that subset resolve to the GUI default, and
`ui/default_settings.py:load_kernel_settings` moves the former Optimized NumPy
dynamic default once (`sequence/kernel_settings_revision`). The remaining
kernels stay available from Python for benchmarks and validation.

Run the Qt-safe project test entry point before release changes:

```bash
python scripts/run_test_suite.py
```

The focused backend tests are `tests/test_numerical_contracts.py`,
`tests/test_dynamic_execution_plan.py`, `tests/test_dynamic_native_plan.py`,
and `tests/test_dynamic_chunking.py`. Benchmark claims must include warm-up,
at least five timed repetitions, environment and loaded-extension identity,
plan/build time, error metrics, and RSS; do not infer a speed-up from CPU
utilization alone.

### Choosing the RF Time Step and Phantom Grid

Two settings trade accuracy for runtime: the RF simulation time step
(`simulation_timestep_s`, GUI presets Accurate 1 µs, Balanced 5 µs, Fast 10 µs,
Very fast 20 µs; the default is 20 µs)
and how finely the phantom is sampled relative to the acquisition matrix. Two
self-contained benchmarks measure both on an analytic two-container phantom
(`benchmarks/synthetic_phantoms.py`, pyruvate/lactate, ±15 Hz B0 gradient), so
they need no local phantom files. The tables below were measured on an Apple M3
(8 cores) with the `native_parallel` dynamic kernel. Errors are deterministic;
runtimes are medians of three runs and only indicative.

#### RF time step

The time step only subdivides intervals in which an RF pulse plays. RF and
gradient waveforms are averaged over each step, while free precession,
gradients outside RF pulses, and ADC sampling stay exact. The error therefore
depends almost entirely on the RF pulse, not on the sequence around it.

Relative signal error against 1 µs (the larger value of the static and dynamic
phantom):

| Sequence (RF pulse)                          | 2 µs     | 5 µs     | 10 µs    | 20 µs    | 50 µs  |
|----------------------------------------------|----------|----------|----------|----------|--------|
| 3D bSSFP (1 ms block)                        | < 0.001% | < 0.001% | 0.002%   | 0.003%   | 0.009% |
| 2D FLASH (0.5 ms block)                      | < 0.001% | 0.001%   | 0.006%   | 0.02%    | 0.15%  |
| 2D FID-CSI (3 ms slice-selective sinc)       | < 0.001% | 0.002%   | 0.006%   | 0.02%    | 0.12%  |
| 2D FLASH (2 ms slice-selective sinc)         | < 0.001% | 0.004%   | 0.016%   | 0.06%    | 0.37%  |
| 3D SS-bSSFP (Gaussian spectral pulses)       | 0.03%    | 0.26%    | 1.1%     | 4.4%     | 27%    |
| 2D FLASH (lactate spectral-spatial, 26 ms)   | 1.3%     | 5.1%     | 11%      | (0.4%)\* | 112%   |

Speed-up against 1 µs (dynamic phantom, 16×16×32 grid):

| Sequence       | Runtime at 1 µs | 2 µs | 5 µs | 10 µs | 20 µs | 50 µs |
|----------------|-----------------|------|------|-------|-------|-------|
| 3D bSSFP       | 1.7 s           | ×1.9 | ×3.7 | ×5.4  | ×8.1  | ×11   |
| 2D FLASH block | 0.36 s          | ×1.6 | ×2.5 | ×3.4  | ×4.3  | ×5.5  |
| 2D FID-CSI     | 7.5 s           | ×1.9 | ×4.0 | ×6.5  | ×9.5  | ×13   |
| 2D FLASH sinc  | 1.3 s           | ×1.8 | ×3.6 | ×5.4  | ×8.0  | ×12   |
| 3D SS-bSSFP    | 4.2 s           | ×1.9 | ×4.2 | ×7.2  | ×11   | ×17   |
| 2D FLASH SPSP  | 16 s            | ×1.8 | ×3.3 | ×5.9  | ×11   | ×20   |

\* The small 20 µs error of the spectral-spatial pulse is an accidental
cancellation, not convergence; do not rely on it.

Recommendations:

- Block, sinc, and CSI excitation pulses: **10 µs** ("Fast") keeps the error
  below 0.02% and is 3–7× faster. Even 20 µs stays below 0.1%, which is why
  20 µs ("Very fast") is the GUI default; switch to a finer preset for the
  spectrally selective and SPSP pulses below.
- Spectrally selective pulses (SS-bSSFP, Gaussian): **5 µs** ("Balanced")
  stays below 0.3%; 10 µs costs about 1%.
- Spectral-spatial (SPSP) pulses: **1 µs** ("Accurate"). Their many short
  sub-pulses under an oscillating slice gradient make the error grow roughly
  linearly with the step, so even 2 µs costs about 1%.
- For other pulses, the error grows roughly with the square of the time step:
  halving the step reduces the error about fourfold.

#### Phantom grid relative to the acquisition matrix

The same continuous object is sampled with exact partial-volume fractions on
grids of 1×, 2×, 4×, and 8× the acquisition matrix (2D sequences refine X/Y,
3D bSSFP all three axes). The 8× grid is the reference. Crushing is ideal, so
the phantom grid is the only intravoxel sampling. The "ROI" value is the mean
magnitude in the core of each container, per metabolite, relative to the
brightest ROI.

| Sequence (matrix)   | Phantom grid | Image error | ROI deviation | Runtime (dynamic) |
|---------------------|--------------|-------------|---------------|-------------------|
| 2D FLASH (16×16)    | 1×           | 12%         | 2.6%          | 1.0 s             |
|                     | 2×           | 2.8%        | 0.56%         | 2.4 s             |
|                     | 4×           | 0.44%       | 0.05%         | 10 s              |
|                     | 8× (ref.)    | –           | –             | 86 s              |
| 2D FID-CSI (8×8)    | 1×           | 25%         | 14%           | 1.8 s             |
|                     | 2×           | 3.8%        | 2.0%          | 1.9 s             |
|                     | 4×           | 1.0%        | 0.52%         | 5.0 s             |
|                     | 8× (ref.)    | –           | –             | 20 s              |
| 3D bSSFP (8×8×8)    | 1×           | 20%         | 15%           | 0.33 s            |
|                     | 2×           | 3.3%        | 2.1%          | 0.62 s            |
|                     | 4×           | 0.70%       | 0.55%         | 1.5 s             |
|                     | 8× (ref.)    | –           | –             | 14 s              |

A phantom on the acquisition grid (1×) has one isochromat per voxel. It misses
the signal decay across each voxel, sharp container edges (partial volume), and
the resulting Gibbs ringing, so coarse matrices such as 8×8 CSI change ROI
values by 10–15%. Recommendations:

- Use at least **2×** the acquisition matrix (ROI values within about 2%).
- Use **4×** for quantitative ROI values or metabolite ratios (within about
  0.5%). Finer grids rarely pay off: 8× costs about 4–10× more than 4×.
- The cost grows with the number of phantom voxels: ×4 per refinement step in
  2D and ×8 in 3D. CSI is an exception at small grids because its many RF and
  ADC intervals dominate the runtime.

#### Running the benchmarks yourself

```bash
python -m benchmarks.run_benchmarks --benchmark timestep
python -m benchmarks.run_benchmarks --benchmark oversampling
```

The standalone modules reproduce the tables above and accept your own files:

```bash
python -m benchmarks.benchmark_timestep --repeats 3
python -m benchmarks.benchmark_timestep --sequence-files my.seq \
  --only-sequence-files --timestep-phantom-files my_phantom.npz
python -m benchmarks.benchmark_phantom_oversampling --repeats 3
```

Both write CSV/JSON records to a timestamped directory below
`exports/benchmarks/` (or `--output-dir`); see
[`benchmarks/README.rst`](../benchmarks/README.rst) for all options and
metric definitions.

### Mouse Perfusion: Current Transport Semantics

`MousePerfusionConfig.fov_m` and `MousePerfusionConfig.mouse_size_m` are
independent physical extents. `mouse_size_m` is the visible body bounding box,
not the abstract atlas coordinate cube. Atlas coordinates are obtained by
mapping FOV voxel centers through the fixed procedural body bounds; vessel
centerlines use the same transform. `mouse_offset_m` translates anatomy and
vessels relative to the FOV center. Preview image rectangles use physical
coordinates, so matrix dimensions affect sampling density but not displayed
scale or aspect ratio. Legacy configurations without
`mouse_size_m` convert their saved implicit atlas extent to the corresponding
visible body dimensions.

The mouse perfusion phantom currently implements a spatially delayed source,
not advection of existing spins. Its concentration source is

$$
u(t,\mathbf r)=D(\mathbf r)q\left(t-\tau(\mathbf r)\right),
$$

where $q(t)$ is the dose-normalized injection curve, $D(\mathbf r)$ is the
delivery map, and $\tau(\mathbf r)$ is the voxelwise arrival delay. The arrival
map follows the directed route from the tail vein through the right heart,
lungs, left heart, arteries, and organ beds. This makes the displayed bolus and
the simulated source appear at different positions at different times.

The source is more than a post-hoc signal-amplitude scaling: it adds local
concentration and longitudinal magnetization during sequence simulation. The
inflow concentration and its polarization are tracked separately. Once added,
however, a spin state remains assigned to that voxel. It experiences the local
RF field, gradient phase, off-resonance, $T_1/T_2$ relaxation, and configured
chemical conversion, but it is not transferred to a downstream voxel.

For example, consider a slice-selective RF pulse through the center of the
mouse:

1. Pyruvate already present in that slice is rotated or saturated by the pulse.
2. Its resulting $(M_x,M_y,M_z)$ state remains in the same simulation voxel.
3. Later inflow can add fresh polarized Pyruvate to that voxel and partially
   replenish its signal.
4. A downstream organ receives its own delayed source. It does not receive the
   RF history of the Pyruvate that was excited in the central slice.

The renal $k_{PL}$ and hepatic $k_{PA}$ maps are active reaction terms. They
convert local Pyruvate into Lactate and Alanine, respectively, and precursor
loss includes both product channels where both are present. These are local
reactions and do not imply spatial transport.

Keep the animation model separate from sequence physics. The animated
**Injected perfusion** concentration convolves the delayed source with a local
mono-exponential clearance to make accumulation and wash-out visible. This
clearance is currently preview-only. The Bloch sequence solver has no
perfusion-dependent outflow, vascular exchange, recirculation, or transport of
previously excited magnetization. Breathing is likewise stored as a preview
displacement field and is not yet coupled to gradient phase.

The relevant implementation boundaries are:

- `src/blochsimulator/mouse_phantom.py` builds the anatomy, vascular transit
  map, delayed delivery, preview concentration, and organ curves.
- `src/blochsimulator/dynamic_phantom.py` integrates the local inflow, Bloch
  evolution, relaxation, and reaction terms.
- `arrival_delay_map_s` changes when source material is created in a voxel; it
  does not move a previously created magnetization state.

### Mouse Perfusion: Recommended Transport Roadmap

The next solver should transport concentration and the complete magnetization
state rather than only scheduling independent local sources. A staged
implementation keeps validation and computational cost manageable.

#### Stage 1: Directed vascular-graph transport

Start with the existing directed vessel graph instead of immediately solving a
full 3D flow field.

1. Divide every vessel edge into one-dimensional transport cells with explicit
   volume, velocity, flow rate, and transit-time dispersion.
2. Store concentration and $(M_x,M_y,M_z)$ for every compound in every
   transport cell.
3. Make downstream inflow equal upstream outflow, including the transported RF
   and relaxation history. Do not recreate downstream spins from the original
   injection curve.
4. Add blood and tissue compartments at organ nodes. Exchange should conserve
   compound amount, while $k_{PL}$ and $k_{PA}$ remain reaction terms within
   the selected compartments.
5. Add explicit clearance and venous return before introducing recirculation.
6. Apply RF and gradient evolution according to the current spatial position
   of each transport cell. A saturated bolus passing through a selected slice
   must therefore remain saturated downstream, subject to relaxation and new
   mixing.

This stage can use operator splitting: perform local Bloch/reaction evolution,
then a conservative transport/exchange step. It should expose a transport-model
interface while retaining the existing delayed-source model for backwards
compatibility and fast previews.

#### Stage 2: Spatial advection-reaction Bloch model

For capillary or tissue-scale transport, evolve each pool according to an
advection-reaction Bloch equation such as

$$
\frac{\partial \mathbf{M}_p}{\partial t} + \nabla\cdot\left(\mathbf{v}_p \mathbf{M}_p\right) =
\mathcal{B}_p(\mathbf{M}) + \mathcal{R}_p(\mathbf{M}, C) + \mathbf{S}_p - \mathbf{W}_p,
$$

where $\mathcal{B}_p$ contains RF, gradient phase, off-resonance, and
relaxation; $\mathcal{R}_p$ contains metabolic exchange; $\mathbf{S}_p$ is true
external injection; and $\mathbf{W}_p$ represents clearance or compartment
exchange. Concentration must obey the corresponding conservative transport and
reaction equation.

A finite-volume method is preferable when strict mass conservation is the
priority. A semi-Lagrangian method is easier to stabilize for large time steps
but needs an explicit conservation correction. A Lagrangian moving-spin model
is another useful option for resolved vessels and naturally carries RF and
phase history, but it requires careful particle-to-voxel interpolation and
noise control. Whichever representation is selected, the transport time step
must be validated independently of the Bloch integration time step.

#### Stage 3: Motion and higher-order physiology

After transport is validated, add respiratory deformation of anatomy,
velocity, and magnetization together. Gradient phase must use the time-varying
physical spin position. Later extensions can add pulsatility, portal-hepatic
circulation, renal filtration, recirculation, heterogeneous capillary transit,
and compound-specific permeability or relaxivity.

#### Required validation tests

At minimum, the transport implementation should include:

- dose and concentration conservation with relaxation and reactions disabled;
- a known plug-flow or one-dimensional advection solution;
- a transit-time test on every vascular branch;
- an RF-tagging test in which a saturated slice produces a delayed downstream
  signal reduction;
- recovery of the current local model when velocity and exchange are disabled;
- non-negative concentrations under the supported time-step limits;
- precursor/product balance for $k_{PL}$ and $k_{PA}$;
- agreement between reference and optimized kernels.

### How to Add a New Pulse Sequence

Adding a new sequence involves updates to both the core simulator logic and the GUI.

**1. Define the Sequence Class (Optional but Recommended)**
In `src/blochsimulator/simulator.py`:
Create a new class inheriting from `PulseSequence`. Implement the `compile()` method to return `(b1, gradients, time)`.

```python
class MyNewSequence(PulseSequence):
    def __init__(self, param1, param2, ...):
        # Initialize parameters
        pass

    def compile(self, dt=1e-6):
        # Generate b1 (complex), gradients (N,3), and time arrays
        return b1, gradients, time
```

**2. Register in GUI (`src/blochsimulator/gui.py`)**

*   **Add to List:** In `SequenceDesigner.init_ui()`, add your sequence name to the `self.sequence_type` ComboBox.
    ```python
    self.sequence_type.addItems([..., "My New Sequence"])
    ```

*   **Define Default Parameters:** In `SequenceDesigner.get_sequence_preset_params()`, add a dictionary for your sequence. This sets default TE, TR, and other values when the user selects your sequence.
    ```python
    "My New Sequence": {
        "te_ms": 15,
        "tr_ms": 100,
        "flip_angle": 45,
    },
    ```

*   **Implement Generation Logic:** In `SequenceDesigner.get_sequence()`, handle the new sequence type instantiation.
    ```python
    elif seq_type == "My New Sequence":
        return MyNewSequence(
            te=te,
            tr=tr,
            param1=...,
        )
    ```

### How to Hide/Show Options per Sequence

You can customize which widgets (e.g., TI spinbox, Echo Count, custom checkboxes) are visible for each sequence type.

**1. Create a Container for New Options (if needed)**
In `SequenceDesigner.init_ui()`, create a `QWidget` or `QGroupBox` to hold your specific controls. Add it to `self.options_container`.

```python
# Create widget
self.my_seq_opts = QWidget()
layout = QHBoxLayout()
self.my_param_spin = QSpinBox()
layout.addWidget(QLabel("My Param:"))
layout.addWidget(self.my_param_spin)
self.my_seq_opts.setLayout(layout)

# Add to main container
self.options_container.addWidget(self.my_seq_opts)

# Hide by default
self.my_seq_opts.setVisible(False)
```

**2. Update Visibility Logic**
In `SequenceDesigner._update_sequence_options()`, add your logic to show or hide the widget based on `seq_type`.

```python
def _update_sequence_options(self):
    seq_type = self.sequence_type.currentText()

    # Toggle visibility
    self.spin_echo_opts.setVisible(seq_type in ("Spin Echo", ...))
    self.my_seq_opts.setVisible(seq_type == "My New Sequence")

    # Update pulse list roles if needed (e.g., Excitation only, or Excitation + Refocusing)
    if seq_type == "My New Sequence":
        roles = ["Excitation", "MyPulse"]
    ...
```

This ensures users only see relevant controls for the active sequence.

---

## 8. Web Simulation Extensions

The web version of the Bloch Simulator runs via Pyodide (WASM) and interacts with the DOM. To extend the web simulation options, you need to modify three main components: the HTML view, the JavaScript controller, and the Python simulation logic.

### 1. Update the HTML View
File: `web/partials/rf_explorer.html` (or create a new partial for a new view)

Add the new input control (e.g., a slider or input box). Ensure it has:
*   An `id` (e.g., `id="my_new_param"`).
*   The class `sim-input` (this automatically triggers the `triggerSimulation` event listener).

```html
<div class="control-group">
    <label for="my_new_param">My Parameter</label>
    <input type="number" id="my_new_param" value="1.0" step="0.1" class="sim-input">
</div>
```

### 2. Update the JavaScript Controller
File: `web/static/js/app.js`

You need to update two functions:
*   **`triggerSimulation()`**: Read the value from your new HTML input and pass it to the Python function.
    ```javascript
    const vals = {
        // ... existing params ...
        myParam: parseFloat(document.getElementById("my_new_param").value)
    };

    // Pass to Python
    pyFunc(vals.t1, vals.t2, vals.duration, vals.freq, vals.type, vals.myParam);
    ```

### 3. Update the Python Logic
File: `web/static/js/app.js` (inside the `runPythonAsync` block)

Update the `update_simulation` Python function signature to accept the new argument and use it in the simulation.

```python
def update_simulation(t1_ms, t2_ms, duration_ms, freq_offset_hz, pulse_type, my_param):
    # Use my_param in your simulation logic
    # ...
```

### Testing Changes
Run the local dev server to test your changes without deploying:
```bash
python scripts/dev_server.py
```
This builds the site to `_dev/` and serves it at `http://localhost:8000`.

### Running Real Physics Locally (Docker)
By default, the dev server runs in "Mock Mode" because the C-extension (`bloch_core_modified.c`) isn't compiled for the browser. To run the **real physics engine** locally, you can use Docker to compile the WebAssembly wheel.

1.  **Prerequisites**: Install Docker Desktop.
2.  **Build the WASM Wheel**: Run this command from the project root:
    ```bash
    docker run --rm -v $(pwd):/src -w /src python:3.11 /bin/bash -c "
      pip install pyodide-build &&
      export EMSCRIPTEN=1 &&
      export CFLAGS='-g0 -O3' &&
      pyodide build
    "
    ```
    *Note: This pulls a standard Python image and installs the build tools. The first run takes a few minutes.*

3.  **Start the Dev Server**:
    ```bash
    python scripts/dev_server.py
    ```
    The script will automatically detect the new `.whl` file in `dist/` and switch to "Real Physics Mode". You will see "Installing bloch_simulator... Ready" in the status bar instead of "Dev mode".
