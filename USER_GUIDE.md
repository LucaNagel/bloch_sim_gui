# Bloch Equation Simulator - User Guide

This guide covers the installation, usage, and features of the Bloch Equation Simulator. This tool allows you to simulate Magnetic Resonance Imaging (MRI) physics using the Bloch equations, with features for custom pulse design, sequence generation, and interactive visualization.

## 1. Installation

### Method A: Standalone Application (No Python Required)
*Recommended for general users.*

1.  **Download:** Go to the [Releases page](#) (if available) or obtain the `BlochSimulator` executable for your operating system (Windows `.exe`, macOS `.app`/binary, Linux binary).
2.  **Run:** Double-click the application to start.
    *   *Note on macOS/Linux:* You may need to grant execution permissions: `chmod +x BlochSimulator`.

### Method B: Python Package (From Source)
*Recommended for researchers and developers.*

To run from source, you need **Python** and a **C Compiler** installed.

<details>
<summary><strong>👇 Click here for detailed setup instructions (Windows, macOS, Linux)</strong></summary>

#### 1. Install Python 3.9+

*   **Windows:**
    *   Download the installer from [python.org](https://www.python.org/downloads/windows/).
    *   **Important:** During installation, check the box **"Add Python to PATH"**.
*   **macOS:**
    *   Download from [python.org](https://www.python.org/downloads/macos/) OR use Homebrew: `brew install python`.
*   **Linux:**
    *   Usually pre-installed. If not: `sudo apt install python3 python3-pip` (Ubuntu/Debian) or `sudo dnf install python3` (Fedora).

#### 2. Install a C Compiler
Required to build the fast simulation core.

*   **Windows:**
    *   Install **Visual Studio Build Tools** (free).
    *   Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    *   In the installer, select **"Desktop development with C++"**.
*   **macOS:**
    *   Open Terminal and run: `xcode-select --install`.
    *   (Optional but recommended for speed) Install OpenMP: `brew install libomp`.
*   **Linux:**
    *   Install GCC: `sudo apt install build-essential` (Ubuntu/Debian) or `sudo dnf groupinstall "Development Tools"` (Fedora).

</details>

**Steps:**
1.  Clone the repository or download the source code.
2.  Open a terminal in the project folder.
3.  Install in editable mode:
    ```bash
    pip install -e .
    ```

## 2. Launching the Simulator

### From Terminal
Once installed via Method B, you can launch the GUI directly:
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
*   **Time step:** Simulation temporal resolution (default 1.0 µs).

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
