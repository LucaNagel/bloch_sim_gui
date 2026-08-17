# Bloch Equation Simulator for Python


A high-performance Python implementation of the Bloch equation solver originally developed by Brian Hargreaves at Stanford University. This package provides a fast C-based core with Python bindings, parallel processing support, and an interactive GUI with classic waveform simulation and an event-based Sequence mode for [Pulseq](https://github.com/pulseq/pypulseq) workflows.

## Demo

### Sequence mode

![Sequence workspace](docs/_static/media/sequence_mode_demo.gif)

***Sequence Mode:** Demonstration of different EPI sequence modes on a spherical object, including multi-repetition and multi-slice acquisitions with B0 inhomogeneities. Generated EPI, CSI, and bSSFP sequences can be exported as [Pulseq `.seq`](https://github.com/pulseq/pypulseq) files.*

### Classic simulation

![Spin Echo Animation](docs/_static/media/spin_echo_v240.gif)

***Free Mode:** Demonstration of a spin-echo simulation.*

## Features

### Simulation and sequence design

Fast C-based Bloch solver with parallel processing support.
The GUI can be used in 2 modes:

**Free Mode**

Free mode lets you investigate the behaviour of spins over a range of frequencies and spatial positions. Good for education and learning MRI concepts such as off-resonances, relaxation, basic sequences, rf pulses etc.

*Features:*
- Endpoint and **full time-resolved** simulations
- Configurable **tissue properties** including T1, T2, proton density, and initial
  magnetization.
- Parameter sweeps with final-state or full time-resolved result collection.
- **RF pulse design** for rectangular, sinc, Gaussian, adiabatic half/full passage,
  and BIR-4 pulses
- **Sequence support** for FID, spin echo, gradient echo, inversion recovery,
  slice-selective excitation, EPI, and SSFP.
- Live magnetization, signal, spectrum, spatial-profile, heatmap, and **3D-vector
  views**.

**Sequence mode**

A mode that lets you load, generate and simulate [Pulseq
  `.seq`](https://github.com/pulseq/pypulseq) sequences. In addition, an interactive 3D phantom and B1 Tx/Rx  designer is provided.

*Features:*

- **Interactive generation of Pulseq** EPI, centre-out 2D spiral, 2D CSI,
  spoiled 2D FLASH,
  Cartesian 3D bSSFP, alternating-frequency [spectrally selective 3D bSSFP](https://doi.org/10.1002/mrm.29676), and
  Cartesian or [spiral-phyllotaxis radial 3D multi-echo bSSFP sequences](https://doi.org/10.1002/mrm.30614), with
  export to `.seq` files and reproducing Jupyter notebooks.
- Spin Probe mode enables the investigation of the behaviour of spectral/spatial spin distributions during sequences.
- Spectral and dynamic **phantom design** with spatial peak distributions,
  pyruvate-to-lactate kinetics, spatial B0 inhomogeneity maps, and optional
  time-dependent B0 offsets.
- **B1 Transmit Receive design**, letting you choose between uniform, 3D birdcage, 3D surface coil B1 fields.
- A dimension-aware **Reconstruction Explorer** for interactive 2D/3D k-space
  and image views, echo/repetition/slice selection, CSI voxel spectra, receive-coil
  combination, simulated pool comparison, and known-frequency linear IDEAL
  estimates.


### Visualization and analysis and reproducibility

Project files that contain current parameter selection, selected sequence, phantom and B1 can be saved and loaded. The tool has different ways to visualize and export bloch simulations:

**Free Mode**

The time-resolved bevahiour of spins during and after RF Pulse can be visualized in a multitude of ways, including a 3D vector view, heatmaps, spectral and spatial profiles. Additionally, simulation results can be exported as
* figures/animations and `xarray.Dataset` conversion

* **automatically generated Jupyter notebooks** ([analysis](examples/spin_echo_analysis.ipynb) of [spin-echo result data](examples/spin_echo_analysis_data.h5) and [reproduction](examples/spin_echo_reproduction.ipynb)) using the parameters selected in the GUI.


**Sequence Mode**

Loaded and generated sequences can be inspected in a sequence viewer. A 3D phantom design viewer in addition to a 3D B1 design viewer are provided.

- Experimental **export of simulated acquisitions as Bruker raw datasets**,
  including `fid` and/or `rawdata.job0` plus the associated parameter files.
- **Automatically generated Jupyter notebooks** using the parameters selected in
  the GUI.


## Get started

### Desktop application

Download the standalone application for Windows or macOS from
[GitHub Releases](https://github.com/LucaNagel/bloch_sim_gui/releases). This is
the recommended option for interactive simulation and requires no Python
installation. Windows downloads and Python wheels target 64-bit systems. Was mainly tested on macOS 26.5.2.

As I did not pay for the Apple Developer Program, the app is unlicensed and will be put in quarantine after unzipping and installation upon the first run. How to run it anyways:
<details>
<summary><strong>Activation on macOS</strong></summary>

After downloading the application, move `BlochSimulator.app` to your
**Applications** folder and launch it. If macOS blocks the first launch:

1. Dismiss the warning.
2. Open **System Settings > Privacy & Security** and scroll to **Security**.
3. Find the message that `BlochSimulator.app` was blocked and click
   **Open Anyway**.
4. Launch **BlochSimulator** again.

Alternatively, after verifying that you trust the downloaded application,
remove its quarantine flag in Terminal:

```bash
xattr -cr /Applications/BlochSimulator.app
```

</details>

### Python package

Install the full [blochsimulator from PyPI](https://pypi.org/project/blochsimulator/) including GUI and pulseq skills (**recommended**):

```bash
pip install "blochsimulator[gui,pulseq]"
```

or

```bash
pip install blochsimulator
```

The package exposes the full simulation API for Python scripts, Jupyter
notebooks, and custom analysis pipelines, but no graphical user interface or pulseq skulls

## Usage

### GUI application

Once installed, launch the GUI from the applications folder or a terminal:

```bash
blochsimulator-gui
```

### Jupyter Notebook/ Python API

The bloch simulator can be used in both jupyter notebooks or via python api

#### Jupyter Notebook

You can launch the interactive GUI directly from a cell in your local Jupyter
Notebook. You can also export the selected GUI simulation as a notebook. See
the [spin-echo reproduction](examples/spin_echo_reproduction.ipynb) and
[spin-echo analysis](examples/spin_echo_analysis.ipynb) examples.

```python
# Install from PyPI once, if needed
!pip install blochsimulator[gui,pulseq]"

# Launch the GUI
!blochsimulator-gui
```

This requires Jupyter to run on your local machine; it does not work on a
headless remote server or Google Colab.

#### Basic simulation

```python
import numpy as np
from blochsimulator import BlochSimulator, TissueParameters

# Create simulator
sim = BlochSimulator(use_parallel=True, num_threads=4)

# Define tissue parameters
tissue = TissueParameters(
    name="Gray Matter",
    t1=1.33,  # seconds
    t2=0.083  # seconds
)

# Create a simple 90-degree pulse
ntime = 100
dt = 1e-5  # 10 microseconds
time = np.arange(ntime) * dt

b1 = np.zeros(ntime, dtype=complex)
b1[0] = 0.0235  # 90-degree hard pulse

gradients = np.zeros((ntime, 3))  # No gradients

# Run simulation
result = sim.simulate(
    sequence=(b1, gradients, time),
    tissue=tissue,
    mode=2  # Time-resolved output
)

# Plot results
sim.plot_magnetization()
```

<details>
<summary><strong>More Python API examples</strong></summary>

#### Spin echo sequence

```python
from blochsimulator import BlochSimulator, SpinEcho, TissueParameters

sim = BlochSimulator()

# Create spin echo sequence
sequence = SpinEcho(te=20e-3, tr=500e-3)  # 20ms TE, 500ms TR

# Simulate white matter
tissue = TissueParameters.white_matter(3.0)

# Run simulation with multiple frequencies (T2* effects)
frequencies = np.linspace(-50, 50, 11)  # Hz
result = sim.simulate(sequence, tissue, frequencies=frequencies)

# Access magnetization components
mx, my, mz = result['mx'], result['my'], result['mz']
signal = result['signal']
```

#### Custom pulse design

```python
from blochsimulator import design_rf_pulse

# Design a sinc pulse
b1, time = design_rf_pulse(
    pulse_type='sinc',
    duration=2e-3,      # 2 ms
    flip_angle=180,     # degrees
    time_bw_product=4,  # Time-bandwidth product
    npoints=200
)

# Apply phase
phase = np.pi/4  # 45 degrees
b1_phased = b1 * np.exp(1j * phase)
```

#### Parallel simulation

```python
# Simulate multiple positions and frequencies in parallel
positions = np.random.randn(100, 3) * 0.01  # Position scale: 10 mm
frequencies = np.linspace(-200, 200, 41)     # 41 frequencies

result = sim.simulate(
    sequence=sequence,
    tissue=tissue,
    positions=positions,
    frequencies=frequencies,
    mode=0  # Endpoint only (faster)
)

# Result shape: (100 positions, 41 frequencies)
print(f"Signal shape: {result['signal'].shape}")
```

#### Xarray integration

For advanced analysis, you can convert simulation results directly to an
`xarray.Dataset`. This provides named dimensions, coordinates, and automatic
metadata tracking.

```python
# Convert last result to xarray
ds = sim.get_results_as_xarray()

# Access data with named dimensions
# Dimensions: (time, position, frequency)
print(ds.mx.dims)

# Powerful selection and plotting
ds.signal.sel(frequency=0, method='nearest').plot()

# Metadata is preserved in attributes
print(ds.attrs['t1'], ds.attrs['te'])
```

#### Sequence library

Pre-defined sequences are available:

```python
from blochsimulator import SpinEcho, GradientEcho

# Spin Echo
se = SpinEcho(te=30e-3, tr=1.0)

# Gradient Echo
gre = GradientEcho(te=5e-3, tr=10e-3, flip_angle=30)

# Compile to waveforms
b1, gradients, time = se.compile(dt=1e-6)
```

#### Tissue parameter library

Common tissues at different field strengths:

```python
from blochsimulator import TissueParameters

# 3T parameters
gm = TissueParameters.gray_matter(3.0)
wm = TissueParameters.white_matter(3.0)
csf = TissueParameters.csf(3.0)

# 7T parameters
gm_7t = TissueParameters.gray_matter(7.0)

# Custom tissue
liver = TissueParameters(
    name="Liver",
    t1=0.812,
    t2=0.042,
    t2_star=0.028,
    density=0.9
)
```

</details>

## Documentation

For detailed instructions on installation, GUI features, and Python API usage,
see the **[User Guide](https://github.com/LucaNagel/bloch_sim_gui/blob/main/docs/USER_GUIDE.md)**.

## Development

For detailed packaging, release workflows, and CI/CD information, see the
[Developer Guide](docs/DEVELOPER_GUIDE.md).

<details>
<summary><strong>Developer setup and manual desktop build</strong></summary>

### Install from source

The Python package supports Python 3.9 or later. Desktop GUI development and
PyInstaller app builds use the shared Python 3.12 runtime declared in
`.python-version`, so the source GUI and packaged app do not silently use
different interpreters.

- **Windows:** Install Python from [python.org](https://www.python.org/downloads/windows/)
  and select **Add Python to PATH**. Install
  [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  with **Desktop development with C++**.
- **macOS:** Install Python from
  [python.org](https://www.python.org/downloads/macos/) or with
  `brew install python`. Install the compiler with `xcode-select --install`.
  For optional OpenMP acceleration, install `libomp` with Homebrew.
- **Linux:** Install Python and a compiler with `sudo apt install python3
  python3-pip build-essential` on Ubuntu/Debian, or install the corresponding
  Python and Development Tools packages on Fedora.

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/LucaNagel/bloch_sim_gui.git
cd bloch_sim_gui
pip install -e .
```

For desktop GUI development, use the shared launcher instead of invoking an
arbitrary `python` or `python3` from `PATH`:

```bash
./scripts/run_gui.sh
```

Both this launcher and `scripts/build_pyinstaller.sh` use `.venv-packaging`.
The current repository is installed there in editable mode, preventing an old
installed BlochSimulator package from shadowing the working tree.
Set `BLOCH_PYTHON=/path/to/python3.12` if Python 3.12 is not discoverable as
`python3.12`.

Verify the installation:

```python
from blochsimulator import BlochSimulator, TissueParameters

sim = BlochSimulator()
tissue = TissueParameters.gray_matter(3.0)
print(f"T1: {tissue.t1:.3f}s, T2: {tissue.t2:.3f}s")
```

### Build the desktop application

Standalone applications for macOS, Windows, and Linux are automatically built
and attached to GitHub Releases whenever a new version tag is pushed. The
instructions below are for manual local builds. One build per operating system
is required.

Prerequisites:

- macOS: Xcode CLT; `brew install libomp`
- Windows: Python 3.9+ and MSVC Build Tools for the C extension
- Linux: gcc/g++; ensure `libgomp` is available

Quick build:

```bash
./scripts/build_pyinstaller.sh
```

The artifact is written to `dist/BlochSimulator` as a single binary, with an
`.exe` suffix on Windows.

The equivalent explicit commands use the same environment:

```bash
.venv-packaging/bin/python setup.py build_ext --inplace
.venv-packaging/bin/python -m PyInstaller bloch_gui.spec --noconfirm
```

Run the packaged application with `./dist/BlochSimulator` on macOS/Linux or
`dist\\BlochSimulator.exe` on Windows.

Runtime data and exports:

- `rfpulses/` is bundled automatically.
- Exports default to per-user data directories:
  - macOS: `~/Library/Application Support/BlochSimulator/exports`
  - Windows: `%APPDATA%\\BlochSimulator\\exports`
  - Linux: `~/.local/share/BlochSimulator/exports`
- Override the location with `BLOCH_APP_DIR` or `BLOCH_EXPORT_DIR`.

### Project structure

```text
blochsimulator/
├── src/
│   └── blochsimulator/
│       ├── __init__.py
│       ├── simulator.py            # Core Python API
│       ├── gui.py                  # PyQt5 GUI
│       ├── bloch_core_modified.c   # C implementation
│       ├── bloch_core.h            # C header
│       ├── bloch_wrapper.pyx       # Cython wrapper
│       └── ...
├── tests/                          # Unit tests
├── docs/                           # Sphinx documentation
├── pyproject.toml                  # Modern build config
├── setup.py                        # C-extension build config
├── MANIFEST.in                     # Source dist manifest
└── README.md
```

</details>

### Troubleshooting build issues

1. **Missing compiler:** Install gcc (Linux), Xcode (macOS), or Visual Studio
   (Windows).
2. **OpenMP not found:** The code will still work, but without parallelization.
3. **Import error:** Ensure that the `.so` or `.pyd` file is in the expected
   package directory.

### Contributing

Contributions are welcome. Please:

1. Fork the repository.
2. Create a feature branch.
3. Add tests for new features.
4. Submit a pull request.

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{blochsimulator_python,
  title={Python Bloch Equation Simulator GUI and API},
  author={Luca Nagel},
  year={2026},
  url={https://github.com/LucaNagel/bloch_sim_gui}
}
```

## Acknowledgments

This project is based on [code](http://mrsrl.stanford.edu/~brian/blochsim/)
originally developed by Brian Hargreaves at Stanford University. As of July
2026, the original source is unfortunately unavailable. A Python adaptation of
the code is available [here](https://github.com/ZhengguoTan/BlochSim).

- Original Bloch simulator by Brian Hargreaves, Stanford University
- NumPy and SciPy communities
- PyQt/PySide developers
- OpenMP project
- Built partially with [Codex](https://openai.com/codex/),
  [Claude Code](https://claude.ai/), and
  [Gemini CLI](https://github.com/google-gemini/gemini-cli)

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
You may copy, distribute, and modify the software under the terms of GPLv3.
Modified versions distributed to others must also be licensed under GPLv3 and
include the corresponding source code.

## Contact

[Luca Nagel](https://github.com/LucaNagel)
