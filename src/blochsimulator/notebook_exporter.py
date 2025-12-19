"""
notebook_exporter.py - Jupyter Notebook Export for Bloch Simulator

This module generates executable Jupyter notebooks from simulation parameters.

Two export modes:
- Mode A: Load data from HDF5 file (for analysis/visualization)
- Mode B: Re-run simulation from parameters (reproducibility)

Author: Bloch Simulator Team
Date: 2024
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from . import __version__


class NotebookExporter:
    """Generate Jupyter notebooks from Bloch Simulator parameters."""

    def __init__(self):
        self.nb_version = 4

    def create_notebook_mode_a(
        self,
        h5_filename: str,
        sequence_params: Dict,
        simulation_params: Dict,
        tissue_params: Dict,
        title: str = "Bloch Simulation Analysis"
    ) -> nbformat.NotebookNode:
        """
        Create notebook that loads data from HDF5 file (Mode A).

        Parameters
        ----------
        h5_filename : str
            Path to HDF5 data file
        sequence_params : dict
            Sequence parameters
        simulation_params : dict
            Simulation parameters
        tissue_params : dict
            Tissue parameters
        title : str
            Notebook title

        Returns
        -------
        nbformat.NotebookNode
            Jupyter notebook object
        """
        nb = new_notebook()
        cells = []

        # Title
        cells.append(new_markdown_cell(f"# {title}\n\n"
                                       f"**BlochSimulator Version**: {__version__}\n\n"
                                       f"**Mode**: Load data from HDF5 file\n\n"
                                       f"**Data file**: `{h5_filename}`\n\n"
                                       f"This notebook loads pre-computed simulation data and provides "
                                       f"visualization and analysis tools."))

        # Cell 1: Imports
        cells.append(new_markdown_cell("## Setup and Imports"))
        cells.append(new_code_cell(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import h5py\n"
            "from pathlib import Path\n\n"
            "# Set matplotlib style\n"
            "plt.style.use('seaborn-v0_8-darkgrid')\n"
            "%matplotlib inline"
        ))

        # Cell 2: Load data
        cells.append(new_markdown_cell("## Load Simulation Data"))
        cells.append(new_code_cell(self._generate_load_data_code(h5_filename)))

        # Cell 3: Display parameters
        cells.append(new_markdown_cell("## Simulation Parameters"))
        cells.append(new_code_cell(self._generate_display_params_code(
            tissue_params, sequence_params, simulation_params
        )))

        # Cell 4: Quick analysis
        cells.append(new_markdown_cell("## Quick Analysis"))
        cells.append(new_code_cell(self._generate_quick_analysis_code()))

        # Cell 5: Magnetization evolution plot
        cells.append(new_markdown_cell("## Magnetization Evolution"))
        cells.append(new_code_cell(self._generate_magnetization_plot_code()))

        # Cell 6: Signal plot
        cells.append(new_markdown_cell("## MRI Signal"))
        cells.append(new_code_cell(self._generate_signal_plot_code()))

        # Cell 7: Spatial profile (if applicable)
        if simulation_params.get('num_positions', 1) > 1:
            cells.append(new_markdown_cell("## Spatial Profile"))
            cells.append(new_code_cell(self._generate_spatial_profile_code()))

        # Cell 8: Custom analysis section
        cells.append(new_markdown_cell(
            "## Custom Analysis\n\n"
            "Add your custom analysis code here. Available data:\n"
            "- `data['mx']`, `data['my']`, `data['mz']` - Magnetization components\n"
            "- `data['signal']` - Complex signal\n"
            "- `data['time']` - Time points\n"
            "- `data['positions']` - Spatial positions\n"
            "- `data['frequencies']` - Off-resonance frequencies"
        ))
        cells.append(new_code_cell("# Your custom analysis code here\n"))

        nb['cells'] = cells
        return nb

    def create_notebook_mode_b(
        self,
        sequence_params: Dict,
        simulation_params: Dict,
        tissue_params: Dict,
        rf_waveform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        title: str = "Bloch Simulation - Reproducible",
        waveform_filename: Optional[str] = None
    ) -> nbformat.NotebookNode:
        """
        Create notebook that re-runs simulation (Mode B).

        Parameters
        ----------
        sequence_params : dict
            Sequence parameters
        simulation_params : dict
            Simulation parameters
        tissue_params : dict
            Tissue parameters
        rf_waveform : tuple, optional
            (b1, time) RF pulse waveform
        title : str
            Notebook title
        waveform_filename : str, optional
            Path to save/load large waveforms (e.g. .npz)

        Returns
        -------
        nbformat.NotebookNode
            Jupyter notebook object
        """
        nb = new_notebook()
        cells = []

        # Title
        cells.append(new_markdown_cell(
            f"# {title}\n\n"
            f"**BlochSimulator Version**: {__version__}\n\n"
            f"**Mode**: Re-run simulation from parameters\n\n"
            f"This notebook reproduces the simulation from scratch using the "
            f"exported parameters."
        ))

        # Cell 1: Imports
        cells.append(new_markdown_cell("## Setup and Imports"))
        cells.append(new_code_cell(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from blochsimulator import (\n"
            "    BlochSimulator, TissueParameters,\n"
            "    SpinEcho, SpinEchoTipAxis, GradientEcho,\n"
            "    SliceSelectRephase, design_rf_pulse\n"
            ")\n\n"
            "# Set matplotlib style\n"
            "plt.style.use('seaborn-v0_8-darkgrid')\n"
            "%matplotlib inline"
        ))

        # Cell 2: Define parameters
        cells.append(new_markdown_cell("## Simulation Parameters"))
        cells.append(new_code_cell(self._generate_parameter_definition_code(
            tissue_params, sequence_params, simulation_params, waveform_filename
        )))

        # Cell 3: Create simulator and tissue
        cells.append(new_markdown_cell("## Initialize Simulator"))
        cells.append(new_code_cell(self._generate_simulator_init_code(
            tissue_params, simulation_params
        )))

        # Cell 4: Define pulse sequence
        cells.append(new_markdown_cell("## Define Pulse Sequence"))
        cells.append(new_code_cell(self._generate_sequence_definition_code(
            sequence_params, rf_waveform
        )))

        # Cell 5: Define positions and frequencies
        cells.append(new_markdown_cell("## Spatial and Frequency Sampling"))
        cells.append(new_code_cell(self._generate_sampling_code(simulation_params)))

        # Cell 6: Run simulation
        cells.append(new_markdown_cell("## Run Simulation"))
        cells.append(new_code_cell(self._generate_simulation_run_code(simulation_params)))

        # Cell 7: Visualize results
        cells.append(new_markdown_cell("## Visualization"))
        cells.append(new_code_cell(self._generate_magnetization_plot_code()))

        # Cell 8: Signal analysis
        cells.append(new_markdown_cell("## Signal Analysis"))
        cells.append(new_code_cell(self._generate_signal_plot_code()))

        # Cell 9: Save results (optional)
        cells.append(new_markdown_cell("## Save Results (Optional)"))
        cells.append(new_code_cell(
            "# Uncomment to save results\n"
            "# sim.save_results('simulation_results.h5', sequence_params, simulation_params)\n"
            "# print('Results saved!')"
        ))

        nb['cells'] = cells
        return nb

    # ========================================================================
    # Code Generation Methods
    # ========================================================================

    def _generate_load_data_code(self, h5_filename: str) -> str:
        """Generate code to load HDF5 data."""
        return f"""# Load data from HDF5 file
data_file = '{h5_filename}'

if not Path(data_file).exists():
    raise FileNotFoundError(f"Data file not found: {{data_file}}")

print(f"Loading data from: {{data_file}}")

data = {{}}
with h5py.File(data_file, 'r') as f:
    # Load magnetization data
    data['mx'] = f['mx'][...]
    data['my'] = f['my'][...]
    data['mz'] = f['mz'][...]
    data['signal'] = f['signal'][...]

    # Load coordinate arrays
    data['time'] = f['time'][...]
    data['positions'] = f['positions'][...]
    data['frequencies'] = f['frequencies'][...]

    # Load tissue parameters
    data['tissue'] = {{}}
    for key in ['name', 't1', 't2', 'density', 't2_star']:
        try:
            data['tissue'][key] = f['tissue'].attrs[key]
        except KeyError:
            pass

    print(f"Data loaded successfully!")
    print(f"  Shape: {{data['mx'].shape}}")
    print(f"  Duration: {{data['time'][-1]*1000:.3f}} ms")
"""

    def _generate_display_params_code(
        self,
        tissue_params: Dict,
        sequence_params: Dict,
        simulation_params: Dict
    ) -> str:
        """Generate code to display parameters."""
        # Filter out numpy arrays from parameters for display
        seq_params_filtered = {k: v for k, v in sequence_params.items()
                              if not isinstance(v, np.ndarray)}
        sim_params_filtered = {k: v for k, v in simulation_params.items()
                              if not isinstance(v, np.ndarray)}

        return f"""# Display simulation parameters
print("="*60)
print("SIMULATION PARAMETERS")
print("="*60)

print("\\nTissue:")
print(f"  Name: {{data['tissue'].get('name', 'Unknown')}}")
print(f"  T1: {{data['tissue'].get('t1', 0)*1000:.1f}} ms")
print(f"  T2: {{data['tissue'].get('t2', 0)*1000:.1f}} ms")

print("\\nSequence:")
for key, value in {seq_params_filtered}.items():
    print(f"  {{key}}: {{value}}")

print("\\nSimulation:")
for key, value in {sim_params_filtered}.items():
    print(f"  {{key}}: {{value}}")

print("="*60)
"""

    def _generate_quick_analysis_code(self) -> str:
        """Generate quick analysis code."""
        return """# Quick analysis
print("\\nData Statistics:")
print(f"  Time points: {len(data['time'])}")
print(f"  Positions: {data['positions'].shape[0]}")
print(f"  Frequencies: {len(data['frequencies'])}")

if data['mx'].ndim == 3:  # Time-resolved
    mx_final = data['mx'][-1]
    my_final = data['my'][-1]
    mz_final = data['mz'][-1]

    print("\\nFinal Magnetization:")
    print(f"  Mx range: [{mx_final.min():.4f}, {mx_final.max():.4f}]")
    print(f"  My range: [{my_final.min():.4f}, {my_final.max():.4f}]")
    print(f"  Mz range: [{mz_final.min():.4f}, {mz_final.max():.4f}]")

    # Find peak transverse magnetization
    mxy = np.sqrt(data['mx']**2 + data['my']**2)
    max_mxy = mxy.max()
    max_idx = np.unravel_index(mxy.argmax(), mxy.shape)

    print(f"\\n  Peak |Mxy|: {max_mxy:.4f}")
    print(f"  At time: {data['time'][max_idx[0]]*1000:.3f} ms")
"""

    def _generate_magnetization_plot_code(self) -> str:
        """Generate magnetization plotting code."""
        return """# Plot magnetization evolution
position_idx = 0  # Change to plot different position
freq_idx = 0      # Change to plot different frequency

if data['mx'].ndim == 3:  # Time-resolved
    time_ms = data['time'] * 1000
    mx = data['mx'][:, position_idx, freq_idx]
    my = data['my'][:, position_idx, freq_idx]
    mz = data['mz'][:, position_idx, freq_idx]
    mxy = np.sqrt(mx**2 + my**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(time_ms, mx, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Mx')
    axes[0, 0].set_title('Transverse Magnetization (x)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_ms, my, 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('My')
    axes[0, 1].set_title('Transverse Magnetization (y)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_ms, mz, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Mz')
    axes[1, 0].set_title('Longitudinal Magnetization')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(time_ms, mxy, color='purple', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('|Mxy|')
    axes[1, 1].set_title('Transverse Magnitude')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Magnetization Evolution - Position {position_idx}, Frequency {freq_idx}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Endpoint data - no time evolution to plot")
"""

    def _generate_signal_plot_code(self) -> str:
        """Generate signal plotting code."""
        return """# Plot signal
if data['signal'].ndim == 3:  # Time-resolved
    signal = data['signal'][:, position_idx, freq_idx]
    time_ms = data['time'] * 1000

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(time_ms, np.real(signal), 'b-', label='Real', linewidth=1.5)
    axes[0].plot(time_ms, np.imag(signal), 'r-', label='Imaginary', linewidth=1.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Complex Signal Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_ms, np.abs(signal), color='purple', linewidth=1.5)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('|Signal|')
    axes[1].set_title('Signal Magnitude')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'MRI Signal - Position {position_idx}, Frequency {freq_idx}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Endpoint data - no time evolution to plot")
"""

    def _generate_spatial_profile_code(self) -> str:
        """Generate spatial profile plotting code."""
        return """# Plot spatial profile
time_idx = -1  # Final time point
freq_idx = 0

if data['mz'].ndim == 3:
    mz = data['mz'][time_idx, :, freq_idx]
    mx = data['mx'][time_idx, :, freq_idx]
    my = data['my'][time_idx, :, freq_idx]
elif data['mz'].ndim == 2:
    mz = data['mz'][:, freq_idx]
    mx = data['mx'][:, freq_idx]
    my = data['my'][:, freq_idx]

mxy = np.sqrt(mx**2 + my**2)
z_pos = data['positions'][:, 2] * 100  # Convert to cm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(z_pos, mz, 'go-', linewidth=2, markersize=6)
ax1.set_xlabel('Position (cm)')
ax1.set_ylabel('Mz')
ax1.set_title('Longitudinal Magnetization Profile')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

ax2.plot(z_pos, mxy, 'mo-', linewidth=2, markersize=6)
ax2.set_xlabel('Position (cm)')
ax2.set_ylabel('|Mxy|')
ax2.set_title('Transverse Magnetization Profile')
ax2.grid(True, alpha=0.3)

freq = data['frequencies'][freq_idx]
plt.suptitle(f'Spatial Profile - Frequency: {freq:.1f} Hz',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""

    def _generate_parameter_definition_code(
        self,
        tissue_params: Dict,
        sequence_params: Dict,
        simulation_params: Dict,
        waveform_filename: Optional[str] = None
    ) -> str:
        """Generate parameter definition code."""
        code = "# Define simulation parameters\n\n"

        # Tissue parameters
        code += "# Tissue parameters\n"
        code += f"tissue_name = '{tissue_params.get('name', 'Custom')}'\n"
        code += f"t1 = {tissue_params.get('t1', 1.0):.6f}  # seconds\n"
        code += f"t2 = {tissue_params.get('t2', 0.1):.6f}  # seconds\n"
        code += f"density = {tissue_params.get('density', 1.0):.3f}\n\n"

        # Sequence parameters
        code += "# Sequence parameters\n"
        code += f"sequence_type = '{sequence_params.get('sequence_type', 'Custom')}'\n"
        if 'te' in sequence_params:
            code += f"te = {sequence_params['te']:.6f}  # seconds\n"
        if 'tr' in sequence_params:
            code += f"tr = {sequence_params['tr']:.6f}  # seconds\n"
        if 'flip_angle' in sequence_params:
            code += f"flip_angle = {sequence_params.get('flip_angle', 90):.1f}  # degrees\n"
        code += "\n"

        # Simulation parameters
        code += "# Simulation parameters\n"
        code += f"num_positions = {simulation_params.get('num_positions', 1)}\n"
        code += f"num_frequencies = {simulation_params.get('num_frequencies', 1)}\n"
        code += f"time_step_us = {simulation_params.get('time_step_us', 1.0):.3f}\n"
        mode_str = simulation_params.get('mode', 'endpoint')
        code += f"mode = 2 if '{mode_str}' == 'time-resolved' else 0\n"

        # Create dictionary for compatibility
        code += "\n# Parameter dictionary (used for some sequence types)\n"
        
        # Check if we have waveforms to save
        waveforms_to_save = {}
        for k, v in sequence_params.items():
            if isinstance(v, np.ndarray):
                waveforms_to_save[k] = v
        
        if waveforms_to_save and waveform_filename:
            # Save to file
            np.savez(waveform_filename, **waveforms_to_save)
            rel_path = Path(waveform_filename).name
            code += f"# Load large waveforms from external file\n"
            code += f"loaded_waveforms = {{}}\n"
            code += f"wf_file = Path('{rel_path}')\n"
            code += f"if wf_file.exists():\n"
            code += f"    with np.load(wf_file) as wf_data:\n"
            code += f"        loaded_waveforms = {{k: wf_data[k] for k in wf_data.files}}\n"
            code += f"else:\n"
            code += f"    print(f'Warning: Waveform file {{wf_file}} not found!')\n\n"
            
            code += "sequence_params = {\n"
            code += f"    'sequence_type': '{sequence_params.get('sequence_type', 'Custom')}',\n"
            for k, v in sequence_params.items():
                if k == 'sequence_type': continue
                if k in waveforms_to_save:
                    code += f"    '{k}': loaded_waveforms.get('{k}'),\n"
                elif isinstance(v, str):
                    code += f"    '{k}': '{v}',\n"
                elif v is None:
                    code += f"    '{k}': None,\n"
                else:
                    code += f"    '{k}': {v},\n"
            code += "}\n"
        else:
            code += "sequence_params = {\n"
            code += f"    'sequence_type': '{sequence_params.get('sequence_type', 'Custom')}',\n"
            for k, v in sequence_params.items():
                if k == 'sequence_type': continue
                if isinstance(v, str):
                    code += f"    '{k}': '{v}',\n"
                elif v is None:
                    code += f"    '{k}': None,\n"
                else:
                    # Note: numpy arrays will be truncated here if not saved to file
                    code += f"    '{k}': {v},\n"
            code += "}\n"

        return code

    def _generate_simulator_init_code(
        self,
        tissue_params: Dict,
        simulation_params: Dict
    ) -> str:
        """Generate simulator initialization code."""
        return f"""# Create simulator
use_parallel = {simulation_params.get('use_parallel', False)}
num_threads = {simulation_params.get('num_threads', 4)}

sim = BlochSimulator(use_parallel=use_parallel, num_threads=num_threads)

# Create tissue
tissue = TissueParameters(
    name=tissue_name,
    t1=t1,
    t2=t2,
    density=density
)

print(f"Simulator initialized")
print(f"  Tissue: {{tissue.name}}")
print(f"  T1: {{tissue.t1*1000:.1f}} ms, T2: {{tissue.t2*1000:.1f}} ms")
"""

    def _generate_sequence_definition_code(
        self,
        sequence_params: Dict,
        rf_waveform: Optional[Tuple] = None
    ) -> str:
        """Generate pulse sequence definition code."""
        seq_type = sequence_params.get('sequence_type', 'Spin Echo')

        # Use full waveforms if available (preferred for accuracy and complex sequences)
        if 'b1_waveform' in sequence_params and 'time_waveform' in sequence_params:
            return """# Use the full simulated waveforms exported from the GUI
b1 = sequence_params.get('b1_waveform')
time = sequence_params.get('time_waveform')
gradients = sequence_params.get('gradients_waveform')

if b1 is None or time is None:
    print("Warning: Waveforms missing from sequence_params dictionary!")
    # Fallback or error
    raise ValueError("B1 or time waveform missing. Ensure the .npz file was exported and loaded correctly.")

if gradients is None:
    gradients = np.zeros((len(b1), 3))

sequence = (b1, gradients, time)
print(f"Sequence created from full exported waveforms ({len(b1)} points)")
"""

        if 'Spin Echo' in seq_type and 'Tip' not in seq_type:
            return f"""# Create Spin Echo sequence
sequence = SpinEcho(
    te=te,
    tr=tr
)
print(f"Spin Echo sequence: TE={{te*1000:.1f}} ms, TR={{tr*1000:.1f}} ms")
"""
        elif 'Gradient Echo' in seq_type:
            return f"""# Create Gradient Echo sequence
sequence = GradientEcho(
    te=te,
    tr=tr,
    flip_angle=flip_angle
)
print(f"Gradient Echo: TE={{te*1000:.1f}} ms, TR={{tr*1000:.1f}} ms, FA={{flip_angle:.1f}}°")
"""
        elif 'Slice Select' in seq_type:
            return f"""# Create Slice Select + Rephase sequence
sequence = SliceSelectRephase(
    flip_angle=flip_angle,
    pulse_duration={sequence_params.get('rf_duration', 3e-3):.6f}
)
print(f"Slice Select + Rephase: FA={{flip_angle:.1f}}°")
"""
        elif 'Free Induction Decay' in seq_type:
            return f"""# Create Free Induction Decay (FID) sequence
# Using a simple pulse followed by readout
dt = time_step_us * 1e-6
duration = {sequence_params.get('duration', 0.01)}
npoints = int(duration / dt)
time = np.arange(npoints) * dt
b1 = np.zeros(npoints, dtype=complex)
gradients = np.zeros((npoints, 3))

# RF Pulse
flip = {sequence_params.get('flip_angle', 90.0)}
pulse, _ = design_rf_pulse('gaussian', duration=1e-3, flip_angle=flip, npoints=int(1e-3/dt))
n_pulse = min(len(pulse), npoints)
b1[:n_pulse] = pulse[:n_pulse]

sequence = (b1, gradients, time)
print(f"FID sequence created: duration={{duration:.3f}}s, flip={{flip}}°")
"""
        elif 'SSFP' in seq_type:
            return f"""# Create SSFP sequence
# Simplified implementation for notebook
# Note: For full SSFP features, consider exporting HDF5 data instead
dt = time_step_us * 1e-6
tr = {sequence_params.get('tr', 0.01)}
n_reps = {int(sequence_params.get('ssfp_repeats', 10))}
flip = {sequence_params.get('flip_angle', 30.0)}
alpha_rad = np.deg2rad(flip)

# Create a single TR block
n_tr = int(tr / dt)
b1_block = np.zeros(n_tr, dtype=complex)
pulse, _ = design_rf_pulse('sinc', duration=0.001, flip_angle=flip, npoints=int(0.001/dt))
n_pulse = min(len(pulse), n_tr)
b1_block[:n_pulse] = pulse[:n_pulse]

# Repeat blocks
b1 = np.tile(b1_block, n_reps)
# Alternate phase (0-180)
for i in range(1, n_reps, 2):
    start = i * n_tr
    end = start + n_pulse
    b1[start:end] *= -1

gradients = np.zeros((len(b1), 3))
time = np.arange(len(b1)) * dt
sequence = (b1, gradients, time)
print(f"SSFP sequence: TR={{tr*1000:.1f}}ms, FA={{flip}}°, {{n_reps}} reps")
"""
        else:
            # Custom sequence with RF pulse
            return """# Create custom sequence from parameters
# NOTE: This sequence type requires custom waveform definitions not fully exported to this notebook.
# You can define your own 'b1', 'gradients', and 'time' arrays here.

print("Custom/Complex sequence selected. Arrays must be defined manually.")
# Example placeholder:
# time = np.arange(1000) * 1e-5
# b1 = np.zeros_like(time, dtype=complex)
# gradients = np.zeros((1000, 3))
# sequence = (b1, gradients, time)

raise NotImplementedError("This sequence type requires manual definition of waveforms in this notebook.")
"""

    def _generate_sampling_code(self, simulation_params: Dict) -> str:
        """Generate position/frequency sampling code."""
        pos_range = simulation_params.get('position_range_cm', 0.0) / 100.0  # to meters
        freq_range = simulation_params.get('frequency_range_hz', 0.0)

        return f"""# Define spatial positions
positions = np.zeros((num_positions, 3))
if num_positions > 1:
    positions[:, 2] = np.linspace(-{pos_range/2:.6f}, {pos_range/2:.6f}, num_positions)

# Define off-resonance frequencies
if num_frequencies > 1:
    frequencies = np.linspace(-{freq_range/2:.1f}, {freq_range/2:.1f}, num_frequencies)
else:
    frequencies = np.array([0.0])

print(f"Sampling:")
print(f"  Positions: {{num_positions}}")
print(f"  Frequencies: {{num_frequencies}}")
"""

    def _generate_simulation_run_code(self, simulation_params: Dict) -> str:
        """Generate simulation execution code."""
        return """# Run simulation
print("\\nRunning simulation...")

result = sim.simulate(
    sequence,
    tissue,
    positions=positions,
    frequencies=frequencies,
    mode=mode
)

# Extract results for easier access
data = {
    'mx': result['mx'],
    'my': result['my'],
    'mz': result['mz'],
    'signal': result['signal'],
    'time': result['time'],
    'positions': result['positions'],
    'frequencies': result['frequencies'],
    'tissue': {'name': tissue.name, 't1': tissue.t1, 't2': tissue.t2}
}

print(f"Simulation complete!")
print(f"  Result shape: {result['mx'].shape}")
print(f"  Duration: {result['time'][-1]*1000:.3f} ms")
"""

    def save_notebook(self, nb: nbformat.NotebookNode, filename: str):
        """
        Save notebook to file.

        Parameters
        ----------
        nb : nbformat.NotebookNode
            Notebook object
        filename : str
            Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)


# ============================================================================
# Convenience Functions
# ============================================================================

def export_notebook(
    mode: str,
    filename: str,
    sequence_params: Dict,
    simulation_params: Dict,
    tissue_params: Dict,
    h5_filename: Optional[str] = None,
    rf_waveform: Optional[Tuple] = None,
    title: Optional[str] = None,
    waveform_filename: Optional[str] = None
):
    """
    Export Jupyter notebook (convenience function).

    Parameters
    ----------
    mode : str
        'load_data' (Mode A) or 'resimulate' (Mode B)
    filename : str
        Output .ipynb filename
    sequence_params : dict
        Sequence parameters
    simulation_params : dict
        Simulation parameters
    tissue_params : dict
        Tissue parameters
    h5_filename : str, optional
        HDF5 data file (Mode A only)
    rf_waveform : tuple, optional
        (b1, time) RF waveform (Mode B only)
    title : str, optional
        Notebook title
    waveform_filename : str, optional
        Path to save/load large waveforms (e.g. .npz)
    """
    exporter = NotebookExporter()

    if mode.lower() in ['load_data', 'a', 'mode_a']:
        if h5_filename is None:
            raise ValueError("Mode A requires h5_filename parameter")

        nb = exporter.create_notebook_mode_a(
            h5_filename,
            sequence_params,
            simulation_params,
            tissue_params,
            title or "Bloch Simulation Analysis"
        )
    elif mode.lower() in ['resimulate', 'b', 'mode_b']:
        nb = exporter.create_notebook_mode_b(
            sequence_params,
            simulation_params,
            tissue_params,
            rf_waveform,
            title or "Bloch Simulation - Reproducible",
            waveform_filename=waveform_filename
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'load_data' or 'resimulate'")

    exporter.save_notebook(nb, filename)
    print(f"Notebook exported: {filename}")


if __name__ == "__main__":
    print("Notebook Exporter for Bloch Simulator")
    print("=" * 60)
    print("\nUsage:")
    print("  from notebook_exporter import export_notebook")
    print("  export_notebook('load_data', 'analysis.ipynb', ...)")
    print("  export_notebook('resimulate', 'reproduce.ipynb', ...)")
