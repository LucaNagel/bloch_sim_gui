# Bloch Equation Simulator for Python

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=github)](https://lucanagel.github.io/bloch_sim_gui/)

A high-performance Python implementation of the Bloch equation solver originally developed by Brian Hargreaves at Stanford University. This package provides a fast C-based core with Python bindings, parallel processing support, and an interactive GUI for MRI pulse sequence simulation.

## Features

The Bloch Simulator can be accessed from 3 different directions:

### Standalone **[GUI](https://github.com/LucaNagel/bloch_sim_gui/releases)**:

Download and install **built package (*.exe*, *.app*)** **[here](https://github.com/LucaNagel/bloch_sim_gui/releases)**. Features:
* High performance GUI for nice visualization
* Easily modify and live update simulation:
    * **Sample Parameters**: *T1*, *T2*, *Polarization level (enables simulation of hyperpolarized magnetization*), etc.
    * **RF Pulses**: *Duration*, *Tip Angle*, *B1*, *Frequency*, *Phase*, *Pulse type* such as *sinc*, *Gaussian*, *Adiabatic Half Passage*, etc.
    * **Pulse Sequence**: *Blockpulse*, *Spin Echo*, *Slice Selection*, *SSFP*, *Inversion Recovery*, etc.
* Export:
    * Python compatible simulation results (*.npy*, *.npz*, *.hdf5*)
    * Automically generated IPython (*.ipynb*) notebooks for repeatility and modifications
    * Animations (*.mp4*, *.gif*) and figures (*.svg*, *.png*)


### Python package **[blochsimulator](https://pypi.org/project/blochsimulator/)**
**Install bloch simulator** package from [pypi.org](https://pypi.org/project/blochsimulator/) via `pip instal blochsimulator`. Features:

* Access simulation capabilites from IPython notebooks and python scripts
* Highly customizable simulations

### Online [GUI](https://lucanagel.github.io/bloch_sim_gui/)
Access [here online](https://lucanagel.github.io/bloch_sim_gui/). Features:

* No installation required
* Simple UI, live simulation
* Simulate RF Pulse Parameters and Slice Selection Parameters

## Documentation

For detailed instructions on installation, GUI features, and Python API usage, please refer to the **[User Guide](https://github.com/LucaNagel/bloch_sim_gui/blob/main/docs/USER_GUIDE.md)**.

## Installation

### Method A: Direct Install (PyPI)
*Recommended for most users.*  Package is availabe on [pypi.org](https://pypi.org/project/blochsimulator/).
```bash
pip install blochsimulator
```

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
1.  **Navigate** to the directory where you want to install the GUI in your terminal.
    ```bash
    cd /path/to/the/directory/
    ```
2.  **Clone or download** the repository.
    ```bash
    git clone git@github.com:LucaNagel/bloch_sim_gui.git
    ```
2. **Or** if that fails:
    ```bash
    git clone https://github.com/LucaNagel/bloch_sim_gui.git
    ```
3. **Navigate** to the cloned repository.
    ```bash
    cd bloch_sim_gui/
    ```
4.  **Install** in editable mode:
    ```bash
    pip install -e .
    ```

### Verification (not necessary for the installation)

Test the installation:

```python
from blochsimulator import BlochSimulator, TissueParameters

sim = BlochSimulator()
tissue = TissueParameters.gray_matter(3.0)
print(f"T1: {tissue.t1:.3f}s, T2: {tissue.t2:.3f}s")
```

### Method C: Standalone App
*No installation required. Recommended for non-technical users.*

Download the most reason version for your OS in **[Releases](https://github.com/LucaNagel/bloch_sim_gui/releases)**

#### Activation (MacOS):
In case of the MacOS app, this requires you to manually remove the *quarantine* flag that macOS puts on the downloaded app. Only perform this step if you trust the distributor.

1.  Unzip the file and move `BlochSimulator.app` to your **Applications** folder.
2. Launch **BlochSimulator** from your Applications folder and dimiss the warning.
3. Go to System Settings > Privacy & Security, scroll down to the Security section, here you should see a message "BlochSimulator.app was blocked...". Click "Open Anyway".
4.  Launch **BlochSimulator** from your Applications folder.

**Alternative Activation**:
1.  Unzip the file and move `BlochSimulator.app` to your **Applications** folder.
2.  **Crucial Step:** Open Terminal and run this command to fix the "App is damaged" error:
    ```bash
    xattr -cr /Applications/BlochSimulator.app
    ```
3.  Launch **BlochSimulator** from your Applications folder.



## Usage

### 🚀 Jupyter Notebook (Recommended)

You can launch the interactive GUI directly from a cell in your Jupyter Notebook.

```python
# 0. Install from PyPI (run once) if not done before
!pip install blochsimulator

# 1. Launch GUI
!blochsimulator-gui
```

*Note: This works if Jupyter is running on your local machine. It will not work on headless remote servers or Google Colab.*

### 1. GUI Application

Once installed, you can launch the GUI from any terminal or shell:

```bash
blochsimulator-gui
```

Features:
- Design RF pulses (rectangular, sinc, Gaussian)
- Configure tissue parameters (T1, T2)
- Select pulse sequences (spin echo, gradient echo, etc.)
- Real-time 3D magnetization visualization
- Signal analysis and frequency spectra

### 2. Python API

#### Basic Simulation

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

#### Spin Echo Sequence

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

#### Custom Pulse Design

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

#### Parallel Simulation

```python
# Simulate multiple positions and frequencies in parallel
positions = np.random.randn(100, 3) * 0.01  # Random positions in 1cm cube
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

### 3. Sequence Library

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

### 4. Tissue Parameter Library

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

## Desktop app build (PyInstaller)

*For detailed packaging, release workflows, and CI/CD info, see the [Developer Guide](docs/DEVELOPER_GUIDE.md).*

**Note:** Standalone applications for macOS, Windows, and Linux are automatically built and attached to GitHub Releases whenever a new version tag is pushed. The instructions below are for manual/local builds.

One build per OS is required (macOS build won’t run on Windows/Linux).

### Prereqs
- macOS: Xcode CLT; `brew install libomp`.
- Windows: Python 3.8+ and MSVC Build Tools (for C extension).
- Linux: gcc/g++; ensure `libgomp` available.

### Quick build (any OS)
```bash
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python setup.py build_ext --inplace
PYINSTALLER_CONFIG_DIR=.pyinstaller pyinstaller bloch_gui.spec --noconfirm
```
Artifact: `dist/BlochSimulator` (single binary; `.exe` on Windows).

### One-liner helper
```bash
./scripts/build_pyinstaller.sh   # creates a venv, installs deps, builds, packages
```

### Run the packaged app
- macOS/Linux: `./dist/BlochSimulator`
- Windows: `dist\\BlochSimulator.exe`

### Runtime data/exports
- `rfpulses/` is bundled automatically.
- Exports default to per-user data dirs:
  - macOS: `~/Library/Application Support/BlochSimulator/exports`
  - Windows: `%APPDATA%\\BlochSimulator\\exports`
  - Linux: `~/.local/share/BlochSimulator/exports`
- Override with `BLOCH_APP_DIR` or `BLOCH_EXPORT_DIR` if you need a custom location.

## Theory

The simulator solves the Bloch equations:

$$
\frac{d\vec{M}}{dt} = \gamma (\vec{M} \times \vec{B}) - \begin{pmatrix} M_x / T_2 \\ M_y / T_2 \\ (M_z - M_0) / T_1 \end{pmatrix}
$$

Using:
- Rotation matrices for RF and gradient effects
- Exponential decay for relaxation
- Cayley-Klein parameters for efficient rotation calculation

## Troubleshooting

### Build Issues

1. **Missing compiler**: Install gcc (Linux), Xcode (macOS), or Visual Studio (Windows)
2. **OpenMP not found**: The code will still work but without parallelization
3. **Import error**: Ensure the .so/.pyd file is in the same directory

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

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

This project is based on [code](http://mrsrl.stanford.edu/~brian/blochsim/) originally developed by Brian Hargreaves at Stanford University. Currently (01/2026) it is unfortunately not available. A python adaption of this code can be found [here](https://github.com/ZhengguoTan/BlochSim).

- Original Bloch simulator by Brian Hargreaves, Stanford University
- NumPy and SciPy communities
- PyQt/PySide developers
- OpenMP project
- Built partially with [codex](https://openai.com/codex/), [claude code](https://claude.ai/) and [gemini cli](https://github.com/google-gemini/gemini-cli)

## Contact

Luca Nagel

## Appendix: File Structure

```
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
