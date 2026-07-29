# Bloch Equation Simulator for Python

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=github)](https://lucanagel.github.io/bloch_sim_gui/)

A high-performance Python implementation of the Bloch equation solver originally developed by Brian Hargreaves at Stanford University. This package provides a fast C-based core with Python bindings, parallel processing support, and an interactive GUI with classic waveform simulation and an event-based Sequence mode for [Pulseq](https://github.com/pulseq/pypulseq) workflows.

## Demo

### Sequence mode

![Sequence workspace](docs/_static/media/sequence_mode_demo.gif)

*Demonstration of different EPI sequence modes on a spherical object, including multi-repetition and multi-slice acquisitions with B0 inhomogeneities. Generated EPI, CSI, and bSSFP sequences can be exported as [Pulseq `.seq`](https://github.com/pulseq/pypulseq) files.*

### Classic simulation

![Spin Echo Animation](docs/_static/media/spin_echo.gif)

*Demonstration of a spin-echo simulation.*

## Features

### Simulation and sequence design

- Fast C-based Bloch solver with parallel processing support.
- Endpoint and full time-resolved simulations over multiple spatial positions
  and off-resonance frequencies.
- Configurable tissue properties including T1, T2, proton density, and initial
  magnetization.
- **RF pulse design** for rectangular, sinc, Gaussian, adiabatic half/full passage,
  and BIR-4 pulses, including phase and carrier-frequency offsets.
- Sequence support for FID, spin echo, gradient echo, inversion recovery,
  slice-selective excitation, EPI, and SSFP.
- Dedicated event-based **Sequence mode** for **loading and simulating [Pulseq
  `.seq`](https://github.com/pulseq/pypulseq) files**.
- **Interactive generation of Pulseq** EPI, centre-out 2D spiral, 2D CSI, and 3D
  bSSFP sequences, with export to `.seq` files and reproducing Jupyter notebooks.
- Spectral and dynamic **phantom design** with spatial peak distributions,
  pyruvate-to-lactate kinetics, spatial B0 inhomogeneity maps, and optional
  time-dependent B0 offsets.
- Hardware-aware RAM protection for large simulation grids.

### Visualization and analysis

- Live magnetization, signal, spectrum, spatial-profile, heatmap, and 3D-vector
  views.
- Synchronized time controls and animation for time-resolved results.
- Sequence timeline, ADC signal, CSI spectrum, k-space, reconstruction, final
  state, spatial magnetization, and spin-probe views.
- Named dimensions and metadata through direct `xarray.Dataset` conversion.
- Static figures (`.png`, `.svg`) and animations (`.mp4`, `.gif`).

### Export and reproducibility

- Numerical results in Python-compatible NumPy and HDF5 formats.
- Sequence results as `xarray.Dataset` objects or NetCDF (`.nc`) files with
  named acquisition, spatial, spectral, dynamic, and pool dimensions.
- Experimental **export of simulated acquisitions as Bruker raw datasets**,
  including `fid` and/or `rawdata.job0` plus the associated parameter files.
- **Automatically generated Jupyter notebooks** using the parameters selected in
  the GUI.
- Parameter sweeps with final-state or full time-resolved result collection.

#### Jupyter notebook export

The desktop app creates notebooks that match the selected tissue, sequence,
RF, spatial, frequency, and simulation parameters.

| Export mode | Purpose | Spin-echo example |
| --- | --- | --- |
| **Reproduction** | Embeds the selected parameters and re-runs the complete simulation from scratch. | [Open reproduction notebook](examples/spin_echo_reproduction.ipynb) |
| **Analysis** | Loads exported results and prepares `numpy`, `matplotlib`, and `xarray` analyses without re-running the solver. | [Open analysis notebook](examples/spin_echo_analysis.ipynb) |

The analysis example uses the accompanying
[spin-echo result data](examples/spin_echo_analysis_data.h5). The GUI exports
the matching data file together with the analysis workflow.

#### Parameter sweeps

The **Parameter Sweep** panel iterates over a parameter range and runs one
simulation per step. Sweeps can vary flip angle, TE, TR, TI, B1 scaling or
amplitude, T1, T2, spin-offset center, and RF-carrier offset. Results can be
compared directly, exported, and opened in an automatically generated
sweep-analysis notebook.

## Get started

### Desktop application

Download the standalone application for Windows or macOS from
[GitHub Releases](https://github.com/LucaNagel/bloch_sim_gui/releases). This is
the recommended option for interactive simulation and requires no Python
installation.

#### Activation on macOS

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

### Python package

Install [blochsimulator from PyPI](https://pypi.org/project/blochsimulator/):

```bash
pip install blochsimulator
```

The package exposes the full simulation API for Python scripts, Jupyter
notebooks, and custom analysis pipelines.

### Online GUI

Use the [browser-based GUI](https://lucanagel.github.io/bloch_sim_gui/) without
installation. It provides interactive RF-pulse and slice-selection simulations;
the desktop application and Python package provide the complete feature set.

## Sequence mode

The **Sequence Simulation** workspace provides an event-based workflow for
complete Pulseq acquisitions. It keeps RF, gradient, ADC, and label timing from
the sequence and runs the acquisition on a spatial, spectral, or dynamic
phantom without expanding the full sequence into a permanently stored dense
waveform.

### Pulseq import and dynamic sequence generation

- Load Pulseq `.seq` files and inspect their RF, gradient, and ADC timeline
  before simulation.
- Simulate imported Pulseq sequences directly on 1D, 2D, or 3D phantoms.
- Configure Cartesian EPI or spiral readouts, including multi-slice gap and
  spacing and configurable Sinc, SLR, block, or RF-Designer excitation pulses,
  plus 2D CSI and 3D bSSFP acquisitions interactively. The generated sequence
  is updated from the current acquisition parameters and can be exported as a
  Pulseq `.seq` file, a reproducing Jupyter notebook, or both.
- Use millimeters consistently for MRI geometry controls such as FOV, slice
  thickness, slice gap, and spatial probe positions; simulations and exports
  continue to use SI meters internally.
- Preserve Pulseq acquisition labels for ordered repetitions, echoes, slices,
  segments, and partitions in the result metadata.

When installing the Python package, enable Pulseq and GUI support with:

```bash
pip install "blochsimulator[gui,pulseq]"
```

The standalone desktop application already bundles the dependencies required
for the Sequence mode.

### Spectral and dynamic phantom designer

The **Phantom Designer** creates multi-shape phantoms from boxes and
ellipsoids and assigns spatially resolved spectral peaks and relaxation
properties to them. It supports per-shape B0 offsets as well as analytic
linear or radial B0 inhomogeneity maps.

Dynamic phantoms extend the same design with a hyperpolarized
pyruvate-to-lactate model. Pool-specific initial magnetization and relaxation,
spatial `kPL` regions, tabulated pyruvate inflow, and a time-dependent B0 offset
can be configured in the **Kinetics / kPL** tab. Total and pool-resolved signals
and magnetization remain available after simulation.

### Results and export

Sequence simulations retain the chronological ADC signal, k-space coordinates,
acquisition labels, final magnetization, and optional checkpoints. Cartesian
and spectroscopic acquisitions additionally provide ready-to-use k-space,
reconstruction, FID, and spectrum arrays where applicable.

Use `SequenceSimulationResult.to_xarray()` for an in-memory `xarray.Dataset`,
or export from **Export results…**. The default export writes both a NetCDF
dataset and an analysis notebook; NetCDF-only, HDF5, and NumPy archives are
also available.

The **Bruker raw dataset** export is experimental. It writes simulated complex
ADC data as `fid`, `rawdata.job0`, or both, together with Bruker-style `acqp`,
`method`, `visu_pars`, and `pulseprogram` files. Export metadata should be
reviewed before using these datasets in scanner-specific reconstruction
pipelines.

## Usage

### GUI application

Once installed, launch the GUI from a terminal or from the applications folder:

```bash
blochsimulator-gui
```

Features:

- Design RF pulses (rectangular, sinc, Gaussian)
- Configure tissue parameters (T1, T2)
- Select pulse sequences (spin echo, gradient echo, etc.)
- Real-time 3D magnetization visualization
- Signal analysis and frequency spectra

### Jupyter Notebook

You can launch the interactive GUI directly from a cell in your local Jupyter
Notebook. You can also export the selected GUI simulation as a notebook. See
the [spin-echo reproduction](examples/spin_echo_reproduction.ipynb) and
[spin-echo analysis](examples/spin_echo_analysis.ipynb) examples.

```python
# Install from PyPI once, if needed
!pip install blochsimulator

# Launch the GUI
!blochsimulator-gui
```

This requires Jupyter to run on your local machine; it does not work on a
headless remote server or Google Colab.

### Python API

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

## Theory

The simulator solves the Bloch equations:

$$
\frac{d\mathbf{M}}{dt}
=
\gamma\left(\mathbf{M}\times\mathbf{B}\right)
-\frac{M_x}{T_2}\,\hat{\mathbf{x}}
-\frac{M_y}{T_2}\,\hat{\mathbf{y}}
-\frac{M_z-M_0}{T_1}\,\hat{\mathbf{z}}
$$

Using:

- Rotation matrices for RF and gradient effects
- Exponential decay for relaxation
- Cayley-Klein parameters for efficient rotation calculation

## Development

For detailed packaging, release workflows, and CI/CD information, see the
[Developer Guide](docs/DEVELOPER_GUIDE.md).

<details>
<summary><strong>Developer setup and manual desktop build</strong></summary>

### Install from source

To run from source, you need Python 3.9 or later and a C compiler.

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
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python setup.py build_ext --inplace
PYINSTALLER_CONFIG_DIR=.pyinstaller pyinstaller bloch_gui.spec --noconfirm
```

The artifact is written to `dist/BlochSimulator` as a single binary, with an
`.exe` suffix on Windows.

Alternatively, use the build helper:

```bash
./scripts/build_pyinstaller.sh
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

Luca Nagel
