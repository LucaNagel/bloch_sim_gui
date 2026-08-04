# Bloch Equation Simulator - User Guide

This guide covers the installation, usage, and features of the Bloch Equation Simulator. This tool allows you to simulate Magnetic Resonance Imaging (MRI) physics using the Bloch equations, with features for custom pulse design, sequence generation, and interactive visualization.

## 1. Installation

### Method A: Direct Install (PyPI)
*Recommended for most users.*
```bash
pip install blochsimulator
```

For other installation methods see [README](https://github.com/LucaNagel/bloch_sim_gui/blob/main/README.md).

## 2. Launching the Simulator

### From Terminal
Once installed, you can launch the GUI directly:
```bash
blochsimulator-gui
```

### From Jupyter Notebook
You can launch the GUI from within a notebook cell:
```python
!blochsimulator-gui
```

## 3. GUI Overview

The interface is divided into two main areas:

*   **Left Panel (Controls):** Configuration for Tissues, RF Pulses, Sequences, and Simulation Settings.
*   **Right Panel (Visualization):** Interactive tabs for viewing results (Magnetization, 3D Vector, Signal, Spectrum, Spatial).

### Simulation Controls (Bottom Left)
*   **Mode:**
    *   *Time-resolved:* Simulates and stores every time point (required for animations).
    *   *Endpoint:* Only calculates the final state (faster, good for large sweeps).
*   **Positions/Frequencies:** Set the number of spatial spins and off-resonance frequencies to simulate.
*   **Time step:** Simulation temporal resolution (default 10.0 µs).

## 4. Key Use Cases & Tutorials

### Use Case 1: Basic Simulation & Animation Export
**Goal:** Simulate a Spin Echo and create a GIF of the magnetization vector.

1.  **Configure Tissue:** In the **Tissue Parameters** box (top left), select "Gray Matter" from the preset dropdown.
2.  **Select Sequence:** In **Sequence Design**, select "Spin Echo".
    *   *Note:* TE and TR will auto-load standard values (e.g., TE=20ms, TR=500ms).
3.  **Run:** Click **Run Simulation** at the bottom left.
4.  **Visualize:** Switch to the **3D Vector** tab on the right.
    *   Use the **Playback Control** slider (bottom) to scrub through time.
    *   Click **Play** to watch the dynamics.
5.  **Export Animation:**
    *   In the **3D Vector** tab, click the **Export ▼** button at the top right.
    *   Select **Animation (GIF/MP4)...**.
    *   Choose a filename and save.

### Use Case 2: Exporting Data & Jupyter Notebooks
**Goal:** Save simulation results and generate a notebook to reproduce them.

1.  **Run a Simulation** (as above).
2.  **Export:** Go to **File > Export Results** (top menu bar).
3.  **Configure Export Dialog:**
    *   **HDF5 (.h5)** and **Notebook: Analysis** are enabled by default, so the
        raw data and a `.ipynb` file that loads and plots them are exported
        together.
    *   Check **Notebook: Reproducible** to generate a `.ipynb` file that contains all parameters to re-run the simulation from scratch.
4.  **Finish:** Click **Export**. You can now open the generated `.ipynb` files in Jupyter Lab/Notebook.

In the event-based **Sequence Simulation** workspace, **Export Pulseq…**
offers three choices: Pulseq plus a generating notebook, Pulseq only, or
notebook only. Pulseq plus notebook is the default. The notebook records the
exact EPI, spiral, CSI, Cartesian 3D bSSFP, spectrally selective 3D bSSFP,
Cartesian multi-echo 3D bSSFP, or radial multi-echo 3D bSSFP builder function
and all current GUI parameters, then writes the corresponding `.seq` file when
executed.

Sequence generation is explicit by default: finish editing the acquisition
parameters and click **Generate sequence** to refresh the timeline. Enable
**Live preview** beside the button only when the sequence should be regenerated
after every parameter change. Starting a simulation or spin probe also ensures
that a pending generated sequence is built first. If generation fails, the
last valid timeline remains visible and the parameter error is shown above the
controls.

Generated sequences explicitly mark the end of every spoiler block. During
simulation, these markers apply an ideal transverse crusher: `Mx` and `My` are
set to zero for every pool at the spoiler end, while `Mz`, relaxation, inflow,
and chemical exchange continue normally. This applies to all built-in
generators that contain spoilers, including EPI, spiral, CSI, MPRAGE, UTE, and
3D bSSFP. Imported Pulseq files receive the same treatment only when they
contain an explicit `IdealSpoilerEndTimes` marker (or one of the legacy
generated-sequence spoiler definitions); arbitrary gradients are not guessed
to be spoilers.

For Cartesian 3D generation, **Cartesian 3D orientation** separates the logical
acquisition roles from the scanner coordinates. Select the physical **Read**
and **Phase** axes; the right-handed **Partition** axis is derived and shown
immediately. For example, Read `+z` and Phase `+y` produce Partition `-x`.
FOV and matrix values remain ordered as `(read, phase, partition)`, while the
generated Pulseq gradients, phantom coverage check, reconstruction, and export
use the selected scanner axes consistently. The orientation is stored in the
Pulseq definitions, so reloading a generated `.seq` file preserves it.

**SS-bSSFP (3D)** alternates complete Cartesian volumes between the configured
metabolite targets. Enter matching comma-separated target names, RF offsets,
receiver offsets, and flip angles. The defaults follow Skinner et al.
(doi:10.1002/mrm.29676), including the narrow-band SLR pulse, alpha/2
preparation, and end-of-volume spoiler. With the default scanner limits, the
encoding-lobe duration is calculated automatically from FOV, matrix, sampling
bandwidth, and gradient limits. The published 32-point, 10 kHz readout lasts
3.2 ms; that ADC duration is kept separate from the shorter pre-/rephasing
lobes. The published 6.29 ms TR requires a scanner profile capable of producing
the automatically calculated encoding moments within the remaining TR time.

**ME-bSSFP (3D, Cartesian)** acquires an odd number of echo volumes inside each
balanced TR. Choose **Flyback** for monopolar readouts with phase-rewinding
gradients or **Symmetric bipolar** for alternating readout polarity. The
publication controls follow Gaubatz (2023): five echoes centered between RF
pulses, 180° RF phase increments, Gaussian excitation, and α/2 preparation.
The short in-vivo preset uses TR 8.696 ms, echo spacing 1.32 ms, 39.6825 kHz
requested sampling bandwidth, FOV 56 × 28 × 24.5 mm³, and matrix 32 × 16 × 14.
The GUI begins with a smaller matrix for responsive setup. Individual echo
volumes are reconstructed and selectable; IDEAL metabolite separation is not
yet attached.

**Radial ME-bSSFP (3D)** creates monopolar center-through echoes on a spherical
spiral-phyllotaxis trajectory. Its publication preset follows Wang et al.
(doi:10.1002/mrm.30614): TR 16 ms, five echoes at 2 ms spacing, 1000 Hz/px,
and golden-angle rotation between dynamic measurements. The GUI starts with a
small interactive spoke count; set 300 spokes and four measurements to match
the in-vivo acquisition. Bloch signal simulation and Pulseq export are
available, while radial gridding and IDEAL reconstruction are not yet attached
to the result viewer.

For 2D imaging, **Readout trajectory** selects either a Cartesian EPI echo
train or a single-interleaf centre-out spiral. **Slice gap** is the empty
edge-to-edge distance between adjacent slices; the centre spacing is slice
thickness plus gap. Spiral readout duration is extended automatically if the
requested sampling bandwidth would exceed the configured gradient or slew
limits.

All MRI geometry fields in the Sequence workspace, Phantom tools, and K-space
trajectory controls use **millimeters (mm)** consistently. This includes FOV,
slice thickness, slice gap, and spatial probe positions. Values are converted
to SI meters internally for simulation and Pulseq export.

The same 2D acquisition panel configures the slice-selective excitation.
Choose **Sinc**, **SLR**, or **Block** and set RF duration, time-bandwidth
product, Sinc apodization, and (for SLR) the bundled sharpness profile. Choose
**RF Pulse Designer** to use the current complex baseband waveform from the
**RF Design** tab. Its duration, complex phase modulation, and carrier offset
are preserved, while its amplitude is rescaled from the designer's reference
flip angle to the constant or variable flip angle selected for EPI/spiral. The
Sequence-mode TBW still defines the slice-selection bandwidth.

Scanner hardware limits are configured under **Tools → Settings → Scanner**.
Maximum gradient, maximum slew rate, waveform rasters, RF ringdown/dead time,
and ADC dead time are stored persistently and applied to all newly generated
EPI, spiral, CSI, and 3D bSSFP Pulseq sequences. Imported `.seq` files retain
their own event timing.

**Export results…** in the same workspace defaults to exporting both
`sequence_result.nc` and `sequence_result.ipynb`. The notebook loads the
adjacent NetCDF dataset and provides the signal, k-space, reconstruction, and
interactive multidimensional analysis views. Data-only formats and Bruker raw
export remain selectable alternatives.

### Use Case 3: Parameter Sweep
**Goal:** Analyze how simulation metrics change when varying a parameter (e.g., Flip Angle, TE, TR, $T_1$, $T_2$).

1.  **Open Sweep Tab:** Click the **Parameter Sweep** tab on the right panel.
2.  **Configure Sweep:**
    *   **Parameter:** Choose from the dropdown (e.g., "Flip Angle", "TE (ms)", "$T_1$ (ms)", "Frequency Offset").
    *   **Range:** Set the **Start**, **End**, and number of **Steps**.
3.  **Select Metrics:** Check the outputs you want to track (e.g., "Signal Magnitude", "Final $M_z$").
4.  **Run:** Click **Run Sweep**. The simulator will iterate through the range and plot the results.
5.  **Export:** Click **Export Results** to save the sweep data to a CSV or NumPy file for further analysis.

### Use Case 4: Simulating Spatial Profiles (Slice Selection)
**Goal:** Visualize the slice profile of a selective excitation.

1.  **Design Pulse:** In **RF Pulse Design**, select "Sinc" (or "Gaussian").
2.  **Sequence:** In **Sequence Design**, select "Slice Select + Rephase".
3.  **Simulation Grid:**
    *   Set **Positions** to 100 (or more for higher res).
    *   Set **Range (mm)** to cover your slice (e.g., 20 mm).
4.  **Run Simulation.**
5.  **Visualize:**
    *   Go to the **Spatial** tab.
    *   You will see the **$M_{xy}$ (Transverse)** profile showing the excited slice.
    *   Switch **Plot type** to "Heatmap" to see the evolution of the slice profile over time (requires *Time-resolved* mode).

### Use Case 5: Custom RF Pulses
**Goal:** Import a custom waveform defined in a file.

1.  **RF Pulse Design:** Click **Load from File**.
2.  **Format:** Supports `.exc` (Bruker-style), `.dat`, `.txt`, `.csv`.
    *   For text files, a dialog will ask for the data layout (Amplitude/Phase columns vs. Interleaved).
3.  **Run:** The loaded pulse is now used in any sequence set to use the "Custom" pulse role (or standard sequences if compatible).

## 5. Saving & Loading Configurations
You can save the entire state of the GUI (tissue params, sequence settings, pulse design) to a JSON file.
*   **Save:** **File > Save Parameters**.
*   **Load:** **File > Load Parameters**.

## 6. Troubleshooting

*   **"Missing Dependency: nbformat"**:
    *   Install it via pip: `pip install nbformat` to enable notebook export.
*   **Simulation is slow**:
    *   Switch **Mode** to "Endpoint" if you don't need animations.
    *   Reduce **Positions** or **Frequencies**.
    *   Ensure OpenMP is active (check terminal output during installation).
*   **A large Pulseq file is loading**:
    *   Import and Cartesian/CSI inference run in the background. The status bar
        reports the current stage and the rest of the GUI remains responsive.
    *   Full RF rasterization is deferred until a simulation or probe is started;
        the initial load computes only exact ADC timing and gradient moments.
*   **Exported Video is black/empty**:
    *   Ensure you have `ffmpeg` installed on your system if exporting MP4. GIF export usually works out-of-the-box.

## 7. Python API (Quick Reference)

### Sequence-result ADC and k-space ordering

Sequence results store `signal` chronologically. NetCDF exports attach `kx`,
`ky`, and `kz` (cycles/m) to every ADC sample, together with
`adc_event_index`, `readout_sample_index`, and Pulseq outer indices such as
`repetition_index` and `partition_index`. NPZ and HDF5 exports contain the same
arrays as datasets. Do not reshape `signal` solely from its length: first group
by the exported outer indices and verify the k-space coordinates.

For 2D CSI, the export already contains `csi_kspace` ordered as
`(..., phase_y, phase_x, spectral_point)`, plus `csi_spatial_fid` and
`csi_spectrum`. NetCDF is the preferred format because these dimension names
and physical coordinates are retained directly.

Validated Cartesian 3D acquisitions additionally contain
`cartesian_3d_kspace` and `cartesian_3d_image`, ordered with explicit outer
dimensions followed by `(partition_*, phase_*, read_*)`, where each suffix is
the selected scanner axis. The default is
`(partition_z, phase_y, read_x)`; a Read-z/Phase-y acquisition uses
`(partition_x, phase_y, read_z)`. Generic coordinates
`cartesian_k_read_cyc_per_m`, `cartesian_k_phase_cyc_per_m`, and
`cartesian_k_partition_cyc_per_m` are independent of that orientation, while
the physical `cartesian_kx/ky/kz` aliases are retained. Manual reshaping of the
chronological ADC stream is not required.

The analysis notebook generated with **Export results** includes an adaptive
`ipywidgets` explorer. Its `x`, `y`, and `z` sliders move linked orthogonal
reconstruction slices and the k-space crosshair. `Repetition` selects a dynamic
3D volume (or a 2D frame), while `Spectral point` selects the CSI time/frequency
sample. The views update continuously while a slider is being dragged. Sliders
whose dimensions are not present in a particular result are disabled
automatically.

The notebook performs its own centered inverse FFT from the exported Cartesian
k-space. For older NetCDF results that contain only the chronological ADC
stream, it can rebuild the grid after validating the acquisition indices and
projecting physical `kx`, `ky`, and `kz` into the stored logical encoding
frame. Dynamic two-pool exports also produce pool-resolved notebook
reconstructions from `species_signal`.

### Dynamic pyruvate/lactate phantom

Open **Phantom Designer** and define peaks whose names match the
configured pyruvate and lactate pool names. Existing shapes can be moved and
resized through their handles. To create geometry directly with the mouse,
choose **Draw ellipsoid** or **Draw box**, then hold the left mouse button and
drag across the axial XY canvas. Right-click or press Escape to cancel drawing.

For each shape, the initial hyperpolarized longitudinal magnetization of a
metabolite is `Initial HP Mz scale × initial pool weight`. For example, a scale
of 100 with pyruvate weight 1 and lactate weight 0 starts with `Pz=100` and
`Lz=0`. A zero weight keeps the pool and its peak definition present; positive
`kPL` can therefore create lactate from an initially empty lactate pool. The
**Set selected shape to initial Lz = 0** button applies this setup directly.
Each peak can have its own T1; an empty metabolite T1 cell uses the shape's
**Default T1** for compatibility with older designs.

In the hyperpolarized model, `HP Mz=1` means 100% of the initial normalized
hyperpolarized excess magnetization. It is not the thermal equilibrium target.
Because the thermal carbon-13 signal is negligible relative to the
hyperpolarized signal, T1 relaxation drives this state approximately toward
zero. This differs from a conventional normalized Bloch model that recovers
toward equilibrium `Mz=1`.
For a pyruvate-injection experiment, lactate will therefore normally start with
initial pool weight 0. If both weights and both T1 values are equal while
`kPL=0`, `Pz(t)` and `Lz(t)` are identical; the preview draws lactate dashed so
the overlapping curves remain visible.

In the **Kinetics / kPL** tab:

1. Enable pyruvate-to-lactate conversion.
2. Set the default `kPL` in `s⁻¹`. It applies everywhere unless an optional
   spatial region overrides it. Zero means no conversion except in regions with
   a positive override; without any positive default or regional `kPL`, no new
   lactate is produced.
3. Optionally add box or ellipsoid kPL regions with center/size in percent of
   the phantom FOV. A region overrides the default `kPL` in its voxels; if
   regions overlap, the last table row wins.
4. Set **Conversion starts at (kinetics time)** to the point on the shared
   kinetics timeline at which `kPL` becomes active. Before that point `kPL=0`;
   afterwards the default or regional `kPL` applies.
5. Use **Kinetics time at sequence t=0** as a global offset for both inflow and
   conversion. `+5 s` starts the Pulseq sequence five seconds into the defined
   kinetics; `-5 s` starts it five seconds before kinetics time zero. The
   inflow samples and conversion start keep their relative timing and do not
   need to be edited when comparing different sequence start times.
6. Optionally enable pyruvate inflow and enter time/source samples. Every point
   is a `(kinetics time in s, relative Mz per s)` sample of the pyruvate source.
   Rows are sorted numerically when their time is edited. The curve is linearly
   interpolated, is zero outside its listed interval, and adds longitudinal
   pyruvate magnetization to every shape containing the selected pyruvate peak.
   Inflow supplies pyruvate; `kPL` independently determines how much of it is
   converted to lactate.
7. Any inflow or conversion interval shifted before sequence `t=0` defines a
   free longitudinal kinetic pre-roll. Starting from the shape's initial pool
   weights at the earliest pre-roll time, the simulator evolves inflow,
   conversion, and T1 relaxation up to `t=0`. The resulting `Pz/Lz`
   distribution becomes the initial state of the Pulseq simulation. RF,
   gradients, ADC, and dynamic B0 are not executed during this pre-roll.
8. Optionally enable dynamic B0 and enter time/frequency samples in Hz. This is
   an object-frequency offset, separate from Pulseq RF and ADC carrier offsets.
9. Use **Update preview** to inspect the rasterized `kPL` map.

Later kinetic-region rows overwrite earlier rows in overlaps. Run the complete
dynamic sequence from **Sequence Simulation**. The signal plot shows total and
pool-resolved signals; **Spatial Magnetization** can display the sum, pyruvate,
or lactate state. Exports contain `species_signal`,
`final_pool_magnetization`, and pool-resolved CSI arrays when applicable.
Inflow, conversion, T1 decay, RF depletion, and dynamic B0 evolution are
integrated continuously over the Pulseq timeline; repetition labels do not
reset the phantom state.

The right-hand **Live conversion preview** represents one voxel, not a spatial
average. **Shape / object to preview** selects the shape whose initial pool
weights and metabolite T1 values are used. **kPL source for this voxel** then
selects either the default value or one region's override. Selecting a row in
the kPL-region table selects that region automatically. The preview updates
immediately when these values, the conversion start, the global kinetics
offset, or the inflow points change. Its horizontal axis is sequence time. The
upper plot shows the shifted pyruvate source, while the lower plot shows solid
`Pz(t)` and dashed `Lz(t)`. If a pre-roll exists, both plots extend into negative
sequence time and a vertical dashed line marks sequence time zero; an orange
dotted line marks the shifted conversion start. The information line reports
the selected kinetics time and pool distribution at sequence `t=0`. The
preview uses the same free longitudinal two-pool integrator as the sequence
simulation, but deliberately excludes RF depletion and gradients; those
effects remain visible only in the complete Pulseq simulation.

The command-line phantom builder is:

```bash
python sequences/scripts/simulate_dynamic_pyruvate_lactate.py
```

It creates `dynamic_kpl_phantom.npz` only. Load that file in the Phantom tab
and run the desired sequence separately from Sequence Simulation.

For advanced scripting, import the core classes:

```python
from blochsimulator import BlochSimulator, TissueParameters, SpinEcho

# 1. Setup
sim = BlochSimulator()
tissue = TissueParameters.gray_matter(3.0)
seq = SpinEcho(te=0.03, tr=1.0)

# 2. Simulate
result = sim.simulate(seq, tissue, mode=2) # mode 2 = time-resolved

# 3. Access Data
time = result['time']
signal = result['signal']
```

## 8. For Developers

For instructions on building the standalone application, managing versions, and the release workflow, please refer to the **[Developer Guide](https://github.com/LucaNagel/bloch_sim_gui/blob/main/docs/DEVELOPER_GUIDE.md)**.
