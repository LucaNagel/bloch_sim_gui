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
    *   Check **HDF5 (.h5)** to save the raw data.
    *   Check **Notebook: Analysis** to generate a `.ipynb` file that loads the HDF5 data and plots it.
    *   Check **Notebook: Reproducible** to generate a `.ipynb` file that contains all parameters to re-run the simulation from scratch.
4.  **Finish:** Click **Export**. You can now open the generated `.ipynb` files in Jupyter Lab/Notebook.

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
    *   Set **Range (cm)** to cover your slice (e.g., 2.0 cm).
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
dimensions followed by `(partition_z, phase_y, read_x)`. For example, a dynamic
3D acquisition is exported as `(repetition, partition_z, phase_y, read_x)`;
manual reshaping of the chronological ADC stream is not required.

The analysis notebook generated with **Export results** includes an adaptive
`ipywidgets` explorer. Its `x`, `y`, and `z` sliders move linked orthogonal
reconstruction slices and the k-space crosshair. `Repetition` selects a dynamic
3D volume (or a 2D frame), while `Spectral point` selects the CSI time/frequency
sample. The views update continuously while a slider is being dragged. Sliders
whose dimensions are not present in a particular result are disabled
automatically.

### Dynamic pyruvate/lactate phantom

Open **Spectral Shape Designer** and define peaks whose names match the
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
4. Optionally enable pyruvate inflow and enter time/source samples. Every point
   is a `(time in s, relative Mz per s)` sample of the pyruvate source. The curve
   is linearly interpolated, is zero outside its listed interval, and adds
   longitudinal pyruvate magnetization to every shape containing the selected
   pyruvate peak. Inflow supplies pyruvate; `kPL` independently determines how
   much of it is converted to lactate.
5. Optionally enable dynamic B0 and enter time/frequency samples in Hz. This is
   an object-frequency offset, separate from Pulseq RF and ADC carrier offsets.
6. Use **Update preview** to inspect the rasterized `kPL` map.

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
immediately when these values or the inflow points change. Its upper plot shows
the pyruvate source, while the lower plot shows solid `Pz(t)` and dashed
`Lz(t)`. An explanatory message identifies `kPL=0`, initially present lactate,
and exactly overlapping pool curves. The preview uses the same free
longitudinal two-pool integrator as the sequence simulation, but deliberately
excludes RF depletion and gradients; those effects remain visible only in the
complete Pulseq simulation.

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
