# Bloch Equation Simulator for Python

A high-performance Python implementation of the Bloch equation solver originally developed by Brian Hargreaves at Stanford University. This package provides a fast C-based core with Python bindings, parallel processing support, and an interactive GUI for MRI pulse sequence simulation.

## Features

- **High Performance**: C implementation with Cython bindings maintains original speed
- **Parallel Processing**: OpenMP support for multi-core acceleration
- **Interactive GUI**: Real-time visualization and parameter adjustment
- **Flexible API**: Easy-to-use Python interface for scripting
- **Comprehensive**: Supports arbitrary RF pulses, gradient waveforms, and tissue parameters
- **Visualization**: 3D magnetization vectors, time evolution plots, and frequency spectra

## Installation

### Prerequisites

1. **Python 3.7+**
2. **C Compiler**:
   - Linux: `gcc` (usually pre-installed)
   - macOS: Xcode Command Line Tools (`xcode-select --install`)
   - Windows: Visual Studio Build Tools
3. **OpenMP** (optional but recommended):
   - Linux: Included with gcc
   - macOS: `brew install libomp`
   - Windows: Included with Visual Studio

### Quick Setup

```bash
# Clone or download the repository
cd bloch_simulator

# Install dependencies
pip install -r requirements.txt

# Build the extension
python build.py

# Or manually:
python setup.py build_ext --inplace
```

### Verification

Test the installation:

```python
from bloch_simulator import BlochSimulator, TissueParameters

sim = BlochSimulator()
tissue = TissueParameters.gray_matter(3.0)
print(f"T1: {tissue.t1:.3f}s, T2: {tissue.t2:.3f}s")
```

## Usage

### 1. GUI Application

Launch the interactive GUI:

```bash
python bloch_gui.py
```

Features:
- Design RF pulses (rectangular, sinc, Gaussian)
- Configure tissue parameters (T1, T2, proton density)
- Select pulse sequences (spin echo, gradient echo, etc.)
- Real-time 3D magnetization visualization
- Signal analysis and frequency spectra

### 2. Python API

#### Basic Simulation

```python
import numpy as np
from bloch_simulator import BlochSimulator, TissueParameters

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
from bloch_simulator import BlochSimulator, SpinEcho, TissueParameters

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
from bloch_simulator import design_rf_pulse

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
from bloch_simulator import SpinEcho, GradientEcho

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
from bloch_simulator import TissueParameters

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

## Performance

Benchmarks on Intel i7-10700K (8 cores):

| Simulation Size | Single Thread | 8 Threads | Speedup |
|-----------------|---------------|-----------|---------|
| 1 spin, 1000 points | 0.8 ms | - | - |
| 256x256 image | 450 ms | 65 ms | 6.9x |
| 64x64x64 volume | 8.2 s | 1.3 s | 6.3x |

## Advanced Features

### GPU Acceleration (Optional)

If CUDA is available:

```python
# Install CuPy
pip install cupy-cuda11x

# Enable GPU acceleration (future feature)
sim = BlochSimulator(use_gpu=True)
```

### Steady-State Simulations

```python
# Mode 1: Steady-state endpoint
result = sim.simulate(sequence, tissue, mode=1)

# Mode 3: Steady-state with time evolution  
result = sim.simulate(sequence, tissue, mode=3)
```

### Save/Load Results

```python
# Save to HDF5
sim.save_results("simulation_results.h5")

# Load later
sim.load_results("simulation_results.h5")
```

## Validation

The simulator has been validated against:
- Analytical solutions for simple cases
- Original MATLAB implementation
- Published literature values

## Theory

The simulator solves the Bloch equations:

```
dM/dt = γ(M × B) - [Mx/T2, My/T2, (Mz-M0)/T1]
```

Using:
- Rotation matrices for RF and gradient effects
- Exponential decay for relaxation
- Cayley-Klein parameters for efficient rotation calculation

## Troubleshooting

### Build Issues

1. **Missing compiler**: Install gcc (Linux), Xcode (macOS), or Visual Studio (Windows)
2. **OpenMP not found**: The code will still work but without parallelization
3. **Import error**: Ensure the .so/.pyd file is in the same directory

### Performance Issues

1. Enable parallel processing: `use_parallel=True`
2. Increase threads: `num_threads=8`
3. Reduce simulation points where possible
4. Use endpoint mode (0) instead of time-resolved (2) when appropriate

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{bloch_simulator_python,
  title={Python Bloch Equation Simulator GUI and API},
  author={Luca Nagel},
  year={2025},
  url={https://github.com/LucaNagel/bloch_sim_gui}
}
```

## Acknowledgments

This project is based on [code](http://mrsrl.stanford.edu/~brian/blochsim/) originally developed by Brian Hargreaves at Stanford University. Currently (11/2025) it is unfortunately not available. A python adaption of this code can be found [here](https://github.com/ZhengguoTan/BlochSim).

- Original Bloch simulator by Brian Hargreaves, Stanford University
- NumPy and SciPy communities
- PyQt/PySide developers
- OpenMP project

## Contact

Luca Nagel

## Appendix: File Structure

```
bloch_simulator/
├── bloch_core_modified.c   # C implementation (from original)
├── bloch_core.h            # C header file
├── bloch_wrapper.pyx       # Cython wrapper
├── setup.py                # Build configuration
├── build.py                # Build script
├── bloch_simulator.py      # Python API
├── bloch_gui.py           # GUI application
├── requirements.txt        # Dependencies
└── README.md              # This file
```
