# Bloch Equation Simulator for Python

A fast Bloch equation simulator with a C core, Python API and interactive GUI,
based on the solver by Brian Hargreaves (Stanford). Use it to explore spin
physics in **Free Mode**, or to simulate complete [Pulseq](https://github.com/pulseq/pypulseq)
acquisitions on 3D phantoms in **Sequence Mode**.

## Installation

**Python package** (GUI, Pulseq support and Python API):

```bash
pip install blochsimulator
blochsimulator-gui
```

In a Jupyter notebook, use `%pip install blochsimulator` so the package lands in
the notebook's own environment.

**Desktop application** for macOS and Windows (no Python needed): download from
[GitHub Releases](https://github.com/LucaNagel/bloch_sim_gui/releases). Builds
target 64-bit systems; mainly tested on macOS 26.

<details>
<summary><strong>macOS: the app is blocked on first launch</strong></summary>

The app is not signed with an Apple Developer ID, so macOS quarantines it.
Move `BlochSimulator.app` to **Applications** and launch it. If it is blocked:

1. Dismiss the warning.
2. Open **System Settings > Privacy & Security** and scroll to **Security**.
3. Click **Open Anyway** next to the `BlochSimulator.app` message.
4. Launch **BlochSimulator** again.

Alternatively, if you trust the download, remove the quarantine flag:

```bash
xattr -cr /Applications/BlochSimulator.app
```

</details>

## Demo

![Sequence workspace](docs/_static/media/sequence_mode_demo.gif)

***Sequence Mode:** EPI variants on a spherical object, including
multi-repetition and multi-slice acquisitions with B0 inhomogeneity.*

![Spin Echo Animation](docs/_static/media/spin_echo_v240.gif)

***Free Mode:** a spin-echo simulation.*

## Features

**Free Mode** — understand spin behaviour over frequency and position:

- Endpoint or time-resolved simulation of configurable tissues (T1, T2, PD)
- RF pulses: rectangular, sinc, Gaussian, adiabatic half/full passage, BIR-4,
  or loaded from file
- FID, spin echo, gradient echo, inversion recovery, slice selection, EPI, SSFP
- Magnetization, signal, spectrum, spatial-profile, heatmap and 3D-vector views,
  GIF/MP4 export
- Parameter sweeps with final-state or full time-resolved result collection

**Sequence Mode** — simulate what the scanner would measure:

- **Pulseq sequences**, loaded or generated: EPI (bipolar/flyback), spiral,
  EPSI/MGE, CSI, FLASH, inversion recovery with T1 fit, PRESS, slice-selective
  FID, Cartesian 3D bSSFP, [spectrally selective 3D bSSFP](https://doi.org/10.1002/mrm.29676)
  and Cartesian or [radial multi-echo 3D bSSFP](https://doi.org/10.1002/mrm.30614)
- **RF design** including [spectral-spatial pulses](docs/SPECTRAL_SPATIAL_RF.md)
  with per-metabolite pulse schedules
- **Phantoms**: 3D spectral phantoms with B0 maps, hyperpolarized
  pyruvate→lactate kinetics with inflow, and a mouse perfusion phantom for
  hyperpolarized ¹³C and for ¹H (tissue water, gadolinium DCE)
- **B1 Tx/Rx fields**: uniform, birdcage, surface coil
- **Receiver noise** calibrated from a measured phantom FID, to predict SNR
  of e.g. hyperpolarized in-vivo experiments
- **Reconstruction Explorer** for 2D/3D images, k-space and CSI spectra, with
  the phantom shown on the image grid
- **PSF and spoiling analysis** of the current sequence
- **Export** to `.seq`, NetCDF/HDF5, Bruker raw data (experimental) and
  auto-generated Jupyter notebooks; complete projects as `.blochproj`

See the **[User Guide](https://github.com/LucaNagel/bloch_sim_gui/blob/main/docs/USER_GUIDE.md)** for how to use these and what to
watch out for.

## Python API

Exported notebooks reproduce any GUI simulation; see the
[spin-echo reproduction](examples/spin_echo_reproduction.ipynb) and
[analysis](examples/spin_echo_analysis.ipynb) examples. The GUI can also be
started from a local Jupyter notebook with `!blochsimulator-gui` (not on remote
servers or Colab).

### Basic simulation

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

#### Load a saved project

```python
from blochsimulator.project_io import load_project

project = load_project("simulation.blochproj")
phantom = project["phantom"]
result = project["sequence_result"]
```

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


## Development

For detailed packaging, release workflows, and CI/CD information, see the
[Developer Guide](docs/DEVELOPER_GUIDE.md).

<details>
<summary><strong>Developer setup and manual desktop build</strong></summary>

### Install from source

The Python package supports Python 3.10 or later. Desktop GUI development and
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

### Conda environment for notebooks and the Python API

From the repository root, create an environment with an explicit Python version:

```bash
conda create -n bloch-simulator python=3.12 pip
conda activate bloch-simulator
python -m pip install -e .
```

Select this environment as the kernel in your notebook editor.

Pip installs `h5py` automatically because it is declared in `pyproject.toml`.
If no compatible pre-built wheel is available, pip attempts a source build,
which needs a separately installed HDF5 library. To use Conda's pre-built
`h5py` package and its HDF5 dependency, optionally add `h5py` to the
`conda create` command before running pip. See the
[Conda setup notes](docs/DEVELOPER_GUIDE.md#conda-for-notebooks-and-the-python-api)
for details.

### Build the desktop application

Standalone applications for macOS, Windows, and Linux are automatically built
and attached to GitHub Releases whenever a new version tag is pushed. The
instructions below are for manual local builds. One build per operating system
is required.

Prerequisites:

- macOS: Xcode CLT; `brew install libomp`
- Windows: Python 3.10+ and MSVC Build Tools for the C extension
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
- [xarray team](https://docs.xarray.dev/en/stable/)
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
