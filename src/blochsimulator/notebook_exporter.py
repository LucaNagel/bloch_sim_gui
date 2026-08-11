"""
notebook_exporter.py - Jupyter Notebook Export for Bloch Simulator

This module generates executable Jupyter notebooks from simulation parameters.

Two export modes:
- Mode A: Load data from HDF5 file (for analysis/visualization)
- Mode B: Re-run simulation from parameters (reproducibility)

Author: Bloch Simulator Team
Date: 2024
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pprint import pformat
from textwrap import dedent

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False
    nbformat = None
import numpy as np
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
        title: str = "Bloch Simulation Analysis",
    ) -> Any:
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
        cells.append(
            new_markdown_cell(
                f"# {title}\n\n"
                f"**BlochSimulator Version**: {__version__}\n\n"
                f"**Mode**: Load data from HDF5 file\n\n"
                f"**Data file**: `{h5_filename}`\n\n"
                f"This notebook loads pre-computed simulation data and provides "
                f"visualization and analysis tools."
            )
        )

        # Installation Instructions
        cells.append(
            new_markdown_cell(
                "## Installation\n\n"
                "If you haven't installed the `blochsimulator` package yet, you can do so using pip:\n\n"
                "```bash\n"
                "# From GitHub (latest version)\n"
                "!pip install git+https://github.com/LucaNagel/bloch_sim_gui.git\n\n"
                "# From local directory (if you have the source code)\n"
                "# !pip install .\n"
                "```"
            )
        )

        # Cell 1: Imports
        cells.append(new_markdown_cell("## Setup and Imports"))
        cells.append(
            new_code_cell(
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import h5py\n"
                "import xarray as xr\n"
                "from pathlib import Path\n"
                "from blochsimulator import BlochSimulator\n\n"
                "# Set matplotlib style\n"
                "plt.style.use('seaborn-v0_8-darkgrid')\n"
                "%matplotlib inline"
            )
        )

        # Cell 2: Load data
        cells.append(new_markdown_cell("## Load Simulation Data"))
        cells.append(new_code_cell(self._generate_load_data_code_mode_a(h5_filename)))

        # Cell 3: Xarray Integration
        cells.append(new_markdown_cell("## Xarray Dataset"))
        cells.append(new_code_cell(self._generate_xarray_code()))

        # Cell 4: Display parameters
        cells.append(new_markdown_cell("## Simulation Parameters"))
        cells.append(
            new_code_cell(
                self._generate_display_params_code(
                    tissue_params, sequence_params, simulation_params
                )
            )
        )

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
        if simulation_params.get("num_positions", 1) > 1:
            cells.append(new_markdown_cell("## Spatial Profile"))
            cells.append(new_code_cell(self._generate_spatial_profile_code()))

        # Cell 8: Custom analysis section
        cells.append(
            new_markdown_cell(
                "## Custom Analysis\n\n"
                "Add your custom analysis code here. Available data:\n"
                "- `data['mx']`, `data['my']`, `data['mz']` - Magnetization components\n"
                "- `data['signal']` - Complex signal\n"
                "- `data['time']` - Time points\n"
                "- `data['positions']` - Spatial positions\n"
                "- `data['frequencies']` - Off-resonance frequencies"
            )
        )
        cells.append(new_code_cell("# Your custom analysis code here\n"))

        nb["cells"] = cells
        return nb

    def create_notebook_sweep_analysis(
        self,
        data_filename: str,
        param_name: str,
        metrics: List[str],
        title: str = "Parameter Sweep Analysis",
        is_dynamic: bool = False,
    ) -> Any:
        """
        Create notebook for parameter sweep analysis.

        Parameters
        ----------
        data_filename : str
            Path to the data file (NPZ or CSV)
        param_name : str
            Name of the swept parameter
        metrics : list
            List of collected metrics
        title : str
            Notebook title
        is_dynamic : bool
            Whether the sweep contains time-resolved data
        """
        nb = new_notebook()
        cells = []

        # Title
        cells.append(
            new_markdown_cell(
                f"# {title}\n\n"
                f"**BlochSimulator Version**: {__version__}\n\n"
                f"**Sweep Parameter**: {param_name}\n\n"
                f"**Data file**: `{data_filename}`\n\n"
                f"**Mode**: {'Dynamic (Time-Resolved)' if is_dynamic else 'Static (Final State)'}"
            )
        )

        # Installation Instructions
        cells.append(
            new_markdown_cell(
                "## Installation\n\n"
                "If you haven't installed the `blochsimulator` package yet, you can do so using pip:\n\n"
                "```bash\n"
                "# From GitHub (latest version)\n"
                "!pip install git+https://github.com/LucaNagel/bloch_sim_gui.git\n\n"
                "# From local directory (if you have the source code)\n"
                "# !pip install .\n"
                "```"
            )
        )

        # Imports
        cells.append(new_markdown_cell("## Setup and Imports"))
        cells.append(
            new_code_cell(
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import json\n"
                "import xarray as xr\n"
                "from pathlib import Path\n\n"
                "# Set matplotlib style\n"
                "plt.style.use('seaborn-v0_8-darkgrid')\n"
                "%matplotlib inline"
            )
        )

        # Load Data
        cells.append(new_markdown_cell("## Load Sweep Data"))
        load_code = f"filename = '{data_filename}'\n"
        load_code += f"is_dynamic = {is_dynamic}\n"
        load_code += "file_path = Path(filename)\n\n"
        load_code += "constant_params = {}\n"
        load_code += "time_vector = None\n\n"

        load_code += "if file_path.suffix == '.npz':\n"
        load_code += "    data = np.load(file_path, allow_pickle=True)\n"
        load_code += "    param_values = data['parameter_values']\n"
        load_code += f"    param_name = str(data['parameter_name'])\n"
        load_code += "    # Load constant params\n"
        load_code += "    if 'constant_params' in data:\n"
        load_code += "        try:\n"
        load_code += "            val = data['constant_params']\n"
        load_code += "            if hasattr(val, 'item'): val = val.item()\n"
        load_code += "            constant_params = json.loads(str(val))\n"
        load_code += "        except:\n"
        load_code += "            pass\n"
        load_code += "    if 'time' in data:\n"
        load_code += "        time_vector = data['time']\n"
        load_code += "    # Load metrics into a dictionary\n"
        load_code += "    results = {k: data[k] for k in data.files if k not in ['parameter_values', 'parameter_name', 'constant_params', 'time']}\n"
        load_code += "elif file_path.suffix == '.csv':\n"
        load_code += "    # Load CSV using numpy (ignoring header row)\n"
        load_code += "    with open(file_path, 'r') as f:\n"
        load_code += "        header_lines = []\n"
        load_code += "        pos = f.tell()\n"
        load_code += "        line = f.readline()\n"
        load_code += "        while line.startswith('#'):\n"
        load_code += "            header_lines.append(line)\n"
        load_code += "            pos = f.tell()\n"
        load_code += "            line = f.readline()\n"
        load_code += "        f.seek(pos) # Go back to first data line\n"
        load_code += "        col_header = line.strip().split(',')\n"
        load_code += "    \n"
        load_code += "    # Parse constant params from header\n"
        load_code += "    for line in header_lines:\n"
        load_code += "        if 'Constant Parameters:' in line:\n"
        load_code += "            try:\n"
        load_code += "                json_str = line.split('Constant Parameters:', 1)[1].strip()\n"
        load_code += "                constant_params = json.loads(json_str)\n"
        load_code += "            except:\n"
        load_code += "                pass\n"
        load_code += "    \n"
        load_code += "    raw_data = np.genfromtxt(file_path, delimiter=',', comments='#', skip_header=1)\n"
        load_code += "    # If only one line, genfromtxt returns 1D array\n"
        load_code += "    if raw_data.ndim == 1:\n"
        load_code += "        raw_data = raw_data.reshape(1, -1)\n"
        load_code += "    \n"
        load_code += "    param_name = col_header[0]\n"
        load_code += "    param_values = raw_data[:, 0]\n"
        load_code += "    \n"
        load_code += "    results = {}\n"
        load_code += "    for i, col_name in enumerate(col_header[1:]):\n"
        load_code += "        results[col_name] = raw_data[:, i+1]\n"
        load_code += "        \n"
        load_code += "    # Check for array sidecar\n"
        load_code += (
            "    array_path = file_path.with_name(file_path.stem + '_arrays.npz')\n"
        )
        load_code += "    if array_path.exists():\n"
        load_code += "        print(f'Loading array data from {array_path.name}')\n"
        load_code += "        arrays = np.load(array_path, allow_pickle=True)\n"
        load_code += "        if 'time' in arrays:\n"
        load_code += "             time_vector = arrays['time']\n"
        load_code += (
            "        # Load constant params from sidecar if not in CSV header\n"
        )
        load_code += "        if not constant_params and 'constant_params' in arrays:\n"
        load_code += "            try:\n"
        load_code += "                val = arrays['constant_params']\n"
        load_code += "                if hasattr(val, 'item'): val = val.item()\n"
        load_code += "                constant_params = json.loads(str(val))\n"
        load_code += "            except: pass\n"
        load_code += "        for k in arrays.files:\n"
        load_code += "            if k not in ['parameter_name', 'parameter_values', 'constant_params', 'time']:\n"
        load_code += "                results[k] = arrays[k]\n"
        load_code += "else:\n"
        load_code += "    raise ValueError('Unsupported file format')\n\n"
        load_code += "print(f'Loaded sweep data for parameter: {param_name}')\n"
        load_code += "print(f'Steps: {len(param_values)}')\n"
        load_code += "print(f'Metrics: {list(results.keys())}')"
        cells.append(new_code_cell(load_code))

        # Xarray Integration
        cells.append(new_markdown_cell("## Xarray Dataset Construction"))
        xr_code = f"""# Create xarray Dataset from sweep results
data_vars = {{}}
coords = {{param_name: param_values}}

if time_vector is not None:
    coords['time'] = time_vector

# Extract spatial/frequency info from constant params
n_pos = constant_params.get('num_positions', 1)
n_freq = constant_params.get('num_frequencies', 1)
n_time = len(time_vector) if time_vector is not None else 0

for k, v in results.items():
    if np.ndim(v) == 1 and len(v) == len(param_values):
        # Scalar metric vs parameter
        data_vars[k] = ([param_name], v)
    elif np.ndim(v) > 1 and len(v) == len(param_values):
        # Dynamic/Multi-dim metric: (param_steps, ...)
        dims = [param_name]
        remaining_shape = v.shape[1:]

        # Try to intelligently name dimensions
        for i, dim_len in enumerate(remaining_shape):
            if n_time > 0 and dim_len == n_time:
                dims.append('time')
            elif n_pos > 1 and dim_len == n_pos:
                dims.append('position')
            elif n_freq > 1 and dim_len == n_freq:
                dims.append('frequency')
            else:
                dims.append(f'dim_{{i+1}}')

        # Handle duplicate dimension names (if any)
        seen = {{}}
        for i, d in enumerate(dims):
            if d in seen:
                seen[d] += 1
                dims[i] = f"{{d}}_{{seen[d]}}"
            else:
                seen[d] = 0

        data_vars[k] = (dims, v)

ds = xr.Dataset(
    data_vars,
    coords=coords
)
# Add constant params as attrs
if constant_params:
    ds.attrs.update(constant_params)

print('Xarray Dataset created:')
print(ds)"""
        cells.append(new_code_cell(xr_code))

        # Display Constant Parameters
        cells.append(new_markdown_cell("## Simulation Configuration"))
        config_code = """print(f'Sweep Mode: {"Dynamic (Time-Resolved)" if is_dynamic else "Static (Final State)"}')
print('\\nConstant Parameters (Fixed during sweep):')

# Organize parameters for display if possible
categories = {'Tissue': [], 'Sequence': [], 'Simulation': [], 'Other': []}

if constant_params:
    for k, v in sorted(constant_params.items()):
        if k in ['t1', 't2', 't2_star', 'density', 'name', 'tissue_name']:
            categories['Tissue'].append((k, v))
        elif k in ['te', 'tr', 'flip_angle', 'sequence_type']:
            categories['Sequence'].append((k, v))
        elif k in ['num_positions', 'num_frequencies', 'time_step_us']:
            categories['Simulation'].append((k, v))
        else:
            categories['Other'].append((k, v))

    for cat, items in categories.items():
        if items:
            print(f'\\n{cat}:')
            for k, v in items:
                print(f'  {k}: {v}')
else:
    print('  No constant parameters found in metadata.')

if time_vector is not None:
    print(f'\\nTime vector loaded: {len(time_vector)} points, duration={time_vector[-1]*1000:.1f} ms')

# Example: Extracting specific parameters for further calculation
t1_ms = constant_params.get('t1', 0) * 1000
te_ms = constant_params.get('te', 0) * 1000
print(f'\\nSelected T1: {t1_ms:.1f} ms, TE: {te_ms:.1f} ms')"""
        cells.append(new_code_cell(config_code))

        # Plot Scalar Metrics
        cells.append(new_markdown_cell("## Scalar Metrics vs Parameter"))
        plot_code = """fig, ax = plt.subplots(figsize=(10, 6))

# Plot all scalar metrics using xarray
has_scalar = False
for var_name in ds.data_vars:
    if ds[var_name].ndim == 1:
        has_scalar = True
        ds[var_name].plot(ax=ax, marker='o', label=var_name)

if has_scalar:
    ax.set_title(f'Sweep Results: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
else:
    print('No scalar metrics found to plot.')
    plt.close()"""
        cells.append(new_code_cell(plot_code))

        # Advanced Analysis (Dynamic Data) - Only if dynamic mode
        if is_dynamic:
            cells.append(new_markdown_cell("## Dynamic Data Analysis"))
            cells.append(
                new_markdown_cell(
                    "Analysis of time-resolved signals across the parameter sweep."
                )
            )

            # 1. Heatmap
            heatmap_code = """# 1. Heatmap of the signal magnitude
dynamic_vars = [v for v in ds.data_vars if ds[v].ndim > 1]
if dynamic_vars:
    target = 'Signal' if 'Signal' in dynamic_vars else dynamic_vars[0]
    print(f'Plotting heatmap for: {target}')

    plt.figure(figsize=(12, 6))
    plot_data = np.abs(ds[target])

    # Reduce dimensions until 2D (sweep_dim, time_dim)
    while plot_data.ndim > 2:
        # Average over intermediate dims (e.g. spatial)
        plot_data = plot_data.mean(dim=plot_data.dims[1])

    plot_data.plot(cmap='viridis')
    plt.title(f'{target} Heatmap')
    plt.show()"""
            cells.append(new_code_cell(heatmap_code))

            # 2. Coordinate vs Data Plot (requested feature)
            coord_plot_code = """# 2. Coordinate Selection Plot (Data vs Time)
# Demonstrates xarray's powerful selection capabilities
if dynamic_vars and 'time' in ds.coords:
    target = 'Signal' if 'Signal' in dynamic_vars else dynamic_vars[0]

    # Select 3 evenly spaced points from the sweep parameter
    param_vals = ds[param_name].values
    indices = np.linspace(0, len(param_vals)-1, 3, dtype=int)
    selected_vals = param_vals[indices]

    plt.figure(figsize=(10, 6))

    for val in selected_vals:
        # Use .sel() to select data by coordinate value
        trace = np.abs(ds[target].sel({param_name: val}, method='nearest'))
        # Handle extra dims if any
        if trace.ndim > 1:
            trace = trace.mean(axis=tuple(range(trace.ndim-1)))

        trace.plot(label=f'{param_name}={val:.2f}')

    plt.title(f'{target} Evolution for selected {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print('Skipping coordinate plot (requires time dimension)')"""
            cells.append(new_code_cell(coord_plot_code))

        nb["cells"] = cells
        return nb

    def create_notebook_mode_b(
        self,
        sequence_params: Dict,
        simulation_params: Dict,
        tissue_params: Dict,
        rf_waveform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        title: str = "Bloch Simulation - Reproducible",
        waveform_filename: Optional[str] = None,
    ) -> Any:
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
        cells.append(
            new_markdown_cell(
                f"# {title}\n\n"
                f"**BlochSimulator Version**: {__version__}\n\n"
                f"**Mode**: Re-run simulation from parameters\n\n"
                f"This notebook reproduces the simulation from scratch using the "
                f"exported parameters."
            )
        )

        # Installation Instructions
        cells.append(
            new_markdown_cell(
                "## Installation\n\n"
                "If you haven't installed the `blochsimulator` package yet, you can do so using pip:\n\n"
                "```bash\n"
                "# From GitHub (latest version)\n"
                "!pip install git+https://github.com/LucaNagel/bloch_sim_gui.git\n\n"
                "# From local directory (if you have the source code)\n"
                "# !pip install .\n"
                "```"
            )
        )

        # Cell 1: Imports
        cells.append(new_markdown_cell("## Setup and Imports"))
        cells.append(
            new_code_cell(
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import xarray as xr\n"
                "from pathlib import Path\n"
                "from blochsimulator import (\n"
                "    BlochSimulator, TissueParameters,\n"
                "    SpinEcho, SpinEchoTipAxis, GradientEcho,\n"
                "    SliceSelectRephase, design_rf_pulse\n"
                ")\n\n"
                "# Set matplotlib style\n"
                "plt.style.use('seaborn-v0_8-darkgrid')\n"
                "%matplotlib inline"
            )
        )

        # Cell 2: Define parameters
        cells.append(new_markdown_cell("## Simulation Parameters"))
        cells.append(
            new_code_cell(
                self._generate_parameter_definition_code(
                    tissue_params, sequence_params, simulation_params, waveform_filename
                )
            )
        )

        # Cell 3: Create simulator and tissue
        cells.append(new_markdown_cell("## Initialize Simulator"))
        cells.append(
            new_code_cell(
                self._generate_simulator_init_code(tissue_params, simulation_params)
            )
        )

        # Cell 4: Define pulse sequence
        cells.append(new_markdown_cell("## Define Pulse Sequence"))
        cells.append(
            new_code_cell(
                self._generate_sequence_definition_code(sequence_params, rf_waveform)
            )
        )

        # Cell 5: Define positions and frequencies
        cells.append(new_markdown_cell("## Spatial and Frequency Sampling"))
        cells.append(new_code_cell(self._generate_sampling_code(simulation_params)))

        # Cell 6: Run simulation
        cells.append(new_markdown_cell("## Run Simulation"))
        cells.append(
            new_code_cell(self._generate_simulation_run_code(simulation_params))
        )

        # Cell 6b: Xarray Dataset
        cells.append(new_markdown_cell("## Xarray Dataset"))
        cells.append(new_code_cell(self._generate_xarray_code()))

        # Cell 7: Visualize results
        cells.append(new_markdown_cell("## Visualization"))
        cells.append(new_code_cell(self._generate_magnetization_plot_code()))

        # Cell 8: Signal analysis
        cells.append(new_markdown_cell("## Signal Analysis"))
        cells.append(new_code_cell(self._generate_signal_plot_code()))

        # Cell 9: Save results (optional)
        cells.append(new_markdown_cell("## Save Results (Optional)"))
        cells.append(
            new_code_cell(
                "# Uncomment to save results\n"
                "# sim.save_results('simulation_results.h5', sequence_params, simulation_params)\n"
                "# print('Results saved!')"
            )
        )

        nb["cells"] = cells
        return nb

    # ========================================================================
    # Code Generation Methods
    # ========================================================================

    def _generate_load_data_code_mode_a(self, h5_filename: str) -> str:
        """Generate code to load HDF5 data using BlochSimulator."""
        return f"""# Load data from HDF5 file
data_file = '{h5_filename}'

if not Path(data_file).exists():
    raise FileNotFoundError(f"Data file not found: {{data_file}}")

print(f"Loading data from: {{data_file}}")

# Initialize simulator to handle data loading
sim = BlochSimulator()
sim.load_results(data_file)
data = sim.last_result

# Convert tissue to dictionary for consistent access
from dataclasses import asdict
if hasattr(data['tissue'], '__dataclass_fields__'):
    data['tissue'] = asdict(data['tissue'])

# Load additional parameters (metadata) not loaded by the simulator core
with h5py.File(data_file, 'r') as f:
    # Load sequence parameters
    data['sequence_params'] = {{}}
    if 'sequence_parameters' in f:
        grp = f['sequence_parameters']
        for key in grp.attrs.keys():
            data['sequence_params'][key] = grp.attrs[key]
        for key in grp.keys():
            if isinstance(grp[key], h5py.Dataset):
                data['sequence_params'][key] = grp[key][...]

    # Load simulation parameters
    data['simulation_params'] = {{}}
    if 'simulation_parameters' in f:
        grp = f['simulation_parameters']
        for key in grp.attrs.keys():
            data['simulation_params'][key] = grp.attrs[key]
        for key in grp.keys():
            if isinstance(grp[key], h5py.Dataset):
                data['simulation_params'][key] = grp[key][...]

print(f"Data loaded successfully!")
if 'mx' in data:
    print(f"  Shape: {{data['mx'].shape}}")
if 'time' in data:
    print(f"  Duration: {{data['time'][-1]*1000:.3f}} ms")
"""

    def _generate_xarray_code(self) -> str:
        """Generate code to convert simulation data to an xarray Dataset."""
        return """# Convert to xarray Dataset for advanced analysis
# Extract info from metadata
n_pos = data.get('simulation_params', {}).get('num_positions', 1)
n_freq = data.get('simulation_params', {}).get('num_frequencies', 1)
time = data.get('time')
n_time = len(time) if time is not None else 0

# Create DataArray for each component
vars = {}
coords = {}
if time is not None: coords['time'] = time

for k in ['mx', 'my', 'mz', 'signal']:
    v = data[k]
    dims = []

    # Try to intelligently name dimensions
    for i, dim_len in enumerate(v.shape):
        if n_time > 0 and dim_len == n_time:
            dims.append('time')
        elif n_pos > 1 and dim_len == n_pos:
            dims.append('position')
        elif n_freq > 1 and dim_len == n_freq:
            dims.append('frequency')
        else:
            dims.append(f'dim_{i}')

    vars[k] = (dims, v)

ds = xr.Dataset(vars, coords=coords)
# Add metadata
ds.attrs.update(data.get('simulation_params', {}))
ds.attrs.update(data.get('sequence_params', {}))

print('Xarray Dataset created as "ds":')
print(ds)"""

    def _generate_load_data_code(self, h5_filename: str) -> str:
        """Generate code to load HDF5 data (Legacy Manual Method)."""
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
    if 'tissue' in f:
        for key in f['tissue'].attrs.keys():
            data['tissue'][key] = f['tissue'].attrs[key]

    # Load sequence parameters
    data['sequence_params'] = {{}}
    if 'sequence_parameters' in f:
        grp = f['sequence_parameters']
        # Load attributes
        for key in grp.attrs.keys():
            data['sequence_params'][key] = grp.attrs[key]
        # Load datasets (e.g., waveforms)
        for key in grp.keys():
            if isinstance(grp[key], h5py.Dataset):
                data['sequence_params'][key] = grp[key][...]

    # Load simulation parameters
    data['simulation_params'] = {{}}
    if 'simulation_parameters' in f:
        grp = f['simulation_parameters']
        for key in grp.attrs.keys():
            data['simulation_params'][key] = grp.attrs[key]
        for key in grp.keys():
            if isinstance(grp[key], h5py.Dataset):
                data['simulation_params'][key] = grp[key][...]

    print(f"Data loaded successfully!")
    print(f"  Shape: {{data['mx'].shape}}")
    print(f"  Duration: {{data['time'][-1]*1000:.3f}} ms")
"""

    def _generate_display_params_code(
        self, tissue_params: Dict, sequence_params: Dict, simulation_params: Dict
    ) -> str:
        """Generate code to display parameters."""
        return """# Display simulation parameters
print("="*60)
print("SIMULATION PARAMETERS")
print("="*60)

print("\\nTissue:")
for key, value in data['tissue'].items():
    if key in ['t1', 't2', 't2_star'] and value is not None:
        print(f"  {key}: {value*1000:.1f} ms")
    elif value is not None:
        print(f"  {key}: {value}")

print("\\nSequence:")
for key, value in data['sequence_params'].items():
    if not isinstance(value, np.ndarray):
        print(f"  {key}: {value}")

print("\\nSimulation:")
for key, value in data['simulation_params'].items():
    if not isinstance(value, np.ndarray):
        print(f"  {key}: {value}")

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
# Always choose central index for position and frequency
position_idx = data['positions'].shape[0] // 2
freq_idx = len(data['frequencies']) // 2

# Get actual values for title
pos_z_mm = data['positions'][position_idx, 2] * 1000
freq_hz = data['frequencies'][freq_idx]

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

    plt.suptitle(f'Magnetization Evolution - Pos: {pos_z_mm:.2f} mm, Freq: {freq_hz:.1f} Hz',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Endpoint data - no time evolution to plot")
"""

    def _generate_signal_plot_code(self) -> str:
        """Generate signal plotting code."""
        return """# Plot signal
# Re-use central indices
position_idx = data['positions'].shape[0] // 2
freq_idx = len(data['frequencies']) // 2

pos_z_mm = data['positions'][position_idx, 2] * 1000
freq_hz = data['frequencies'][freq_idx]

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

    plt.suptitle(f'MRI Signal - Pos: {pos_z_mm:.2f} mm, Freq: {freq_hz:.1f} Hz',
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
z_pos = data['positions'][:, 2] * 1000  # Convert to mm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(z_pos, mz, 'go-', linewidth=2, markersize=6)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Mz')
ax1.set_title('Longitudinal Magnetization Profile')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

ax2.plot(z_pos, mxy, 'mo-', linewidth=2, markersize=6)
ax2.set_xlabel('Position (mm)')
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
        waveform_filename: Optional[str] = None,
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
        if "te" in sequence_params:
            code += f"te = {sequence_params['te']:.6f}  # seconds\n"
        if "tr" in sequence_params:
            code += f"tr = {sequence_params['tr']:.6f}  # seconds\n"
        if "flip_angle" in sequence_params:
            code += (
                f"flip_angle = {sequence_params.get('flip_angle', 90):.1f}  # degrees\n"
            )
        code += "\n"

        # Simulation parameters
        code += "# Simulation parameters\n"
        code += f"num_positions = {simulation_params.get('num_positions', 1)}\n"
        code += f"num_frequencies = {simulation_params.get('num_frequencies', 1)}\n"
        code += f"time_step_us = {simulation_params.get('time_step_us', 1.0):.3f}\n"
        mode_str = simulation_params.get("mode", "endpoint")
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
            code += (
                f"        loaded_waveforms = {{k: wf_data[k] for k in wf_data.files}}\n"
            )
            code += f"else:\n"
            code += f"    print(f'Warning: Waveform file {{wf_file}} not found!')\n\n"

            code += "sequence_params = {\n"
            code += f"    'sequence_type': '{sequence_params.get('sequence_type', 'Custom')}',\n"
            for k, v in sequence_params.items():
                if k == "sequence_type":
                    continue
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
                if k == "sequence_type":
                    continue
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
        self, tissue_params: Dict, simulation_params: Dict
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
        self, sequence_params: Dict, rf_waveform: Optional[Tuple] = None
    ) -> str:
        """Generate pulse sequence definition code."""
        seq_type = sequence_params.get("sequence_type", "Spin Echo")

        # Use full waveforms if available (preferred for accuracy and complex sequences)
        if "b1_waveform" in sequence_params and "time_waveform" in sequence_params:
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

        if "Spin Echo" in seq_type and "Tip" not in seq_type:
            return f"""# Create Spin Echo sequence
sequence = SpinEcho(
    te=te,
    tr=tr
)
print(f"Spin Echo sequence: TE={{te*1000:.1f}} ms, TR={{tr*1000:.1f}} ms")
"""
        elif "Gradient Echo" in seq_type:
            return f"""# Create Gradient Echo sequence
sequence = GradientEcho(
    te=te,
    tr=tr,
    flip_angle=flip_angle
)
print(f"Gradient Echo: TE={{te*1000:.1f}} ms, TR={{tr*1000:.1f}} ms, FA={{flip_angle:.1f}}°")
"""
        elif "Slice Select" in seq_type:
            dur = sequence_params.get("rf_duration", 3e-3)
            return f"""# Create Slice Select + Rephase sequence
sequence = SliceSelectRephase(
    flip_angle=flip_angle,
    pulse_duration={dur:.6f}
)
print(f"Slice Select + Rephase: FA={{flip_angle:.1f}}°")
"""
        elif "Free Induction Decay" in seq_type:
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
        elif "SSFP" in seq_type:
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
        if "position_range_mm" in simulation_params:
            pos_range = simulation_params["position_range_mm"] / 1000.0
        else:
            pos_range = simulation_params.get("position_range_cm", 0.0) / 100.0
        freq_range = simulation_params.get("frequency_range_hz", 0.0)
        freq_center = simulation_params.get("frequency_center_hz", 0.0)

        return f"""# Define spatial positions
positions = np.zeros((num_positions, 3))
if num_positions > 1:
    positions[:, 2] = np.linspace(-{pos_range/2:.6f}, {pos_range/2:.6f}, num_positions)

# Define off-resonance frequencies
if num_frequencies > 1:
    frequencies = np.linspace({freq_center-freq_range/2:.1f}, {freq_center+freq_range/2:.1f}, num_frequencies)
else:
    frequencies = np.array([{freq_center:.1f}])

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
    mode=mode,
    dt=time_step_us * 1e-6
)

# Extract results for easier access
from dataclasses import asdict
data = {
    'mx': result['mx'],
    'my': result['my'],
    'mz': result['mz'],
    'signal': result['signal'],
    'time': result['time'],
    'positions': result['positions'],
    'frequencies': result['frequencies'],
    'tissue': asdict(tissue),
    'sequence_params': sequence_params,
    'simulation_params': {
        'num_positions': len(positions),
        'num_frequencies': len(frequencies),
        'mode': 'time-resolved' if mode == 2 else 'endpoint',
        'use_parallel': use_parallel,
        'num_threads': num_threads
    }
}

print(f"Simulation complete!")
print(f"  Result shape: {result['mx'].shape}")
print(f"  Duration: {result['time'][-1]*1000:.3f} ms")
"""

    def save_notebook(self, nb: Any, filename: str):
        """
        Save notebook to file.

        Parameters
        ----------
        nb : nbformat.NotebookNode
            Notebook object
        filename : str
            Output filename
        """
        with open(filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)


# ============================================================================
# Convenience Functions
# ============================================================================


def export_notebook(
    mode: str,
    filename: str,
    sequence_params: Optional[Dict] = None,
    simulation_params: Optional[Dict] = None,
    tissue_params: Optional[Dict] = None,
    h5_filename: Optional[str] = None,
    rf_waveform: Optional[Tuple] = None,
    title: Optional[str] = None,
    waveform_filename: Optional[str] = None,
    # Sweep specific
    data_filename: Optional[str] = None,
    param_name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    is_dynamic: bool = False,
):
    """
    Export Jupyter notebook (convenience function).

    Parameters
    ----------
    mode : str
        'load_data' (Mode A), 'resimulate' (Mode B), or 'sweep'
    filename : str
        Output .ipynb filename
    ... (other params)
    is_dynamic : bool
        Whether sweep data is time-resolved (sweep mode only)
    """
    exporter = NotebookExporter()

    if mode.lower() in ["load_data", "a", "mode_a"]:
        if h5_filename is None:
            raise ValueError("Mode A requires h5_filename parameter")
        # Ensure params are provided
        if not all([sequence_params, simulation_params, tissue_params]):
            raise ValueError("Mode A requires sequence, simulation, and tissue params")

        nb = exporter.create_notebook_mode_a(
            h5_filename,
            sequence_params,
            simulation_params,
            tissue_params,
            title or "Bloch Simulation Analysis",
        )
    elif mode.lower() in ["resimulate", "b", "mode_b"]:
        if not all([sequence_params, simulation_params, tissue_params]):
            raise ValueError("Mode B requires sequence, simulation, and tissue params")
        nb = exporter.create_notebook_mode_b(
            sequence_params,
            simulation_params,
            tissue_params,
            rf_waveform,
            title or "Bloch Simulation - Reproducible",
            waveform_filename=waveform_filename,
        )
    elif mode.lower() == "sweep":
        if not all([data_filename, param_name]):
            raise ValueError("Sweep mode requires data_filename and param_name")
        nb = exporter.create_notebook_sweep_analysis(
            data_filename,
            param_name,
            metrics or [],
            title or f"Sweep Analysis: {param_name}",
            is_dynamic=is_dynamic,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'load_data', 'resimulate', or 'sweep'"
        )

    exporter.save_notebook(nb, filename)
    print(f"Notebook exported: {filename}")


def _sequence_result_reconstruction_code() -> str:
    """Return portable Cartesian reconstruction helpers for result notebooks."""
    return dedent(
        """
        from itertools import product


        def _canonical_cartesian_axis(raw_axis):
            raw_axis = np.asarray(raw_axis, dtype=float)
            size = raw_axis.size
            cells = np.arange(size, dtype=float) - size // 2
            if size < 2:
                return np.zeros(size, dtype=float), cells
            step = float(np.median(np.diff(raw_axis)))
            tolerance = max(1e-12, 1e-9 * np.max(np.abs(raw_axis)))
            if not np.isfinite(step) or step <= tolerance:
                raise ValueError('Cartesian coordinate axis is not strictly increasing')
            offset_cells = float(np.median(raw_axis / step - cells))
            # Remove whole-grid moment origins (for example volume spoilers),
            # while retaining a genuine half-cell readout offset.
            offset_cells -= float(np.rint(offset_cells))
            cells = cells + offset_cells
            return cells * step, cells


        def _cartesian_coordinate_levels(values):
            values = np.sort(np.asarray(values, dtype=float).reshape(-1))
            if not values.size or not np.all(np.isfinite(values)):
                raise ValueError('Cartesian coordinates must be finite and non-empty')
            tolerance = max(
                1e-12,
                64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(values))),
            )
            clusters = [[values[0]]]
            for value in values[1:]:
                if abs(value - np.mean(clusters[-1])) <= tolerance:
                    clusters[-1].append(value)
                else:
                    clusters.append([value])
            return np.asarray([np.mean(cluster) for cluster in clusters])


        def _cartesian_orientation(dataset):
            basis_value = dataset.attrs.get('cartesian_encoding_basis_xyz')
            if basis_value is None:
                basis = np.eye(3)
            elif isinstance(basis_value, str):
                basis = np.fromstring(basis_value, sep=',', dtype=float)
            else:
                basis = np.asarray(basis_value, dtype=float).reshape(-1)
            if basis.size != 9:
                raise ValueError(
                    'cartesian_encoding_basis_xyz must contain nine values'
                )
            basis = np.asarray(basis, dtype=float).reshape(3, 3)

            axes = str(
                dataset.attrs.get('cartesian_encoding_axes', '+x +y +z')
            ).split()
            if len(axes) != 3:
                raise ValueError(
                    'cartesian_encoding_axes must contain read, phase, and partition'
                )
            roles = ('read', 'phase', 'partition')
            dimensions = tuple(
                f'{role}_{axis[-1].lower()}'
                for role, axis in zip(roles, axes)
            )
            return basis, tuple(axes), dimensions


        def _cartesian_spatial_dims(kspace):
            result = []
            for role in ('partition', 'phase', 'read'):
                matches = [
                    dimension
                    for dimension in kspace.dims
                    if dimension.startswith(f'{role}_')
                ]
                if matches:
                    result.append(matches[0])
            if len(result) not in (2, 3):
                raise ValueError(
                    'Cartesian k-space needs read_*/phase_* dimensions and an '
                    'optional partition_* dimension'
                )
            return tuple(result)


        def _cartesian_from_adc(dataset, signal_name='signal'):
            required = {
                'adc_event_index', 'readout_sample_index', 'kx', 'ky', 'kz'
            }
            missing = sorted(required.difference(dataset.coords))
            if missing:
                raise ValueError(
                    'raw Cartesian reconstruction requires coordinates: '
                    + ', '.join(missing)
                )
            if signal_name not in dataset:
                raise ValueError(f'{signal_name!r} is not present in the dataset')

            event_index = np.asarray(dataset.adc_event_index.values)
            sample_order = np.argsort(event_index, kind='stable')
            boundaries = np.flatnonzero(np.diff(event_index[sample_order])) + 1
            event_samples = [
                values for values in np.split(sample_order, boundaries) if values.size
            ]
            if not event_samples:
                raise ValueError('the ADC stream contains no readout events')
            read_matrix = event_samples[0].size
            if read_matrix < 1 or any(
                samples.size != read_matrix for samples in event_samples
            ):
                raise ValueError(
                    'all Cartesian readouts must contain the same number of samples'
                )
            expected_read_indices = np.arange(read_matrix)
            for samples in event_samples:
                indices = np.sort(
                    np.asarray(dataset.readout_sample_index.values)[samples]
                )
                if not np.array_equal(indices, expected_read_indices):
                    raise ValueError('readout_sample_index is incomplete within an event')

            label_axes = (
                ('slice_index', 'slice'),
                ('echo_index', 'echo'),
                ('repetition_index', 'repetition'),
                ('segment_index', 'segment'),
            )
            event_labels = {}
            for coordinate, _ in label_axes:
                values = (
                    np.asarray(dataset.coords[coordinate].values)
                    if coordinate in dataset.coords
                    else np.zeros(dataset.sizes['adc'], dtype=int)
                )
                selected = []
                for samples in event_samples:
                    if np.any(values[samples] != values[samples[0]]):
                        raise ValueError(f'{coordinate} changes within an ADC event')
                    selected.append(values[samples[0]])
                event_labels[coordinate] = np.asarray(selected)

            if 'partition_index' in dataset.coords:
                partition_per_sample = np.asarray(dataset.partition_index.values)
                partition_values_per_event = []
                for samples in event_samples:
                    if np.any(
                        partition_per_sample[samples]
                        != partition_per_sample[samples[0]]
                    ):
                        raise ValueError('partition_index changes within an ADC event')
                    partition_values_per_event.append(
                        partition_per_sample[samples[0]]
                    )
                partition_values_per_event = np.asarray(partition_values_per_event)
            else:
                partition_values_per_event = np.zeros(len(event_samples), dtype=int)

            outer_axes = [
                (coordinate, dimension)
                for coordinate, dimension in label_axes
                if np.unique(event_labels[coordinate]).size > 1
            ]
            outer_values = {
                dimension: np.sort(np.unique(event_labels[coordinate]))
                for coordinate, dimension in outer_axes
            }
            outer_keys = list(
                product(*(outer_values[dimension] for _, dimension in outer_axes))
            )
            if not outer_keys:
                outer_keys = [()]
            outer_positions = {
                key: tuple(
                    int(np.flatnonzero(outer_values[dimension] == value)[0])
                    for (_, dimension), value in zip(outer_axes, key)
                )
                for key in outer_keys
            }

            records = []
            kx = np.asarray(dataset.kx.values, dtype=float)
            ky = np.asarray(dataset.ky.values, dtype=float)
            kz = np.asarray(dataset.kz.values, dtype=float)
            basis, encoding_axes, encoding_dims = _cartesian_orientation(dataset)
            logical_moments = np.column_stack((kx, ky, kz)) @ basis
            k_read, k_phase, k_partition = logical_moments.T
            read_dim, phase_dim, partition_dim = encoding_dims
            for event, samples in enumerate(event_samples):
                outer_key = tuple(
                    event_labels[coordinate][event]
                    for coordinate, _ in outer_axes
                )
                records.append(
                    {
                        'samples': samples,
                        'outer': outer_key,
                        'partition': partition_values_per_event[event],
                        'k_phase': float(np.median(k_phase[samples])),
                        'k_partition': float(np.median(k_partition[samples])),
                    }
                )

            first_outer = outer_keys[0]
            labelled_partitions = np.unique(partition_values_per_event)
            if labelled_partitions.size > 1:
                partition_values = np.asarray(
                    sorted(
                        labelled_partitions,
                        key=lambda value: np.median(
                            [
                                record['k_partition']
                                for record in records
                                if record['outer'] == first_outer
                                and record['partition'] == value
                            ]
                        ),
                    )
                )
                for record in records:
                    record['partition_group'] = record['partition']
            else:
                # Older and third-party result files may not carry Pulseq PAR
                # labels. Recover the partition grouping from the trajectory
                # within each outer frame instead of folding every kz plane into
                # the phase axis. This is especially visible when ny == nz.
                levels_by_outer = {
                    outer: _cartesian_coordinate_levels(
                        [
                            record['k_partition']
                            for record in records
                            if record['outer'] == outer
                        ]
                    )
                    for outer in outer_keys
                }
                partition_counts = {
                    levels.size for levels in levels_by_outer.values()
                }
                if len(partition_counts) != 1:
                    raise ValueError(
                        'Cartesian outer frames contain unequal kz level counts'
                    )
                partition_values = np.arange(partition_counts.pop())
                for record in records:
                    levels = levels_by_outer[record['outer']]
                    record['partition_group'] = int(
                        np.argmin(np.abs(levels - record['k_partition']))
                    )
            is_3d = partition_values.size > 1
            groups = {}
            for record in records:
                key = (
                    record['outer'],
                    record['partition_group'] if is_3d else None,
                )
                groups.setdefault(key, []).append(record)

            expected_group_keys = [
                (outer, partition if is_3d else None)
                for outer in outer_keys
                for partition in (partition_values if is_3d else [None])
            ]
            if any(key not in groups for key in expected_group_keys):
                raise ValueError('the Cartesian outer/partition grid is incomplete')
            phase_counts = {len(groups[key]) for key in expected_group_keys}
            if len(phase_counts) != 1:
                raise ValueError('Cartesian partitions contain unequal phase-line counts')
            phase_matrix = phase_counts.pop()

            signal = dataset[signal_name]
            leading_dims = [dimension for dimension in signal.dims if dimension != 'adc']
            signal_values = np.asarray(signal.transpose(*leading_dims, 'adc').values)
            outer_dims = [dimension for _, dimension in outer_axes]
            spatial_dims = (
                [partition_dim, phase_dim, read_dim]
                if is_3d
                else [phase_dim, read_dim]
            )
            output_shape = (
                tuple(signal_values.shape[:-1])
                + tuple(len(outer_values[dimension]) for dimension in outer_dims)
                + ((partition_values.size,) if is_3d else ())
                + (phase_matrix, read_matrix)
            )
            kspace_values = np.empty(output_shape, dtype=signal_values.dtype)
            leading_index = (slice(None),) * len(leading_dims)
            for outer in outer_keys:
                outer_position = outer_positions[outer]
                partitions = partition_values if is_3d else [None]
                for partition_position, partition in enumerate(partitions):
                    phase_records = sorted(
                        groups[(outer, partition if is_3d else None)],
                        key=lambda record: record['k_phase'],
                    )
                    for phase_position, record in enumerate(phase_records):
                        samples = record['samples']
                        samples = samples[
                            np.argsort(k_read[samples], kind='stable')
                        ]
                        index = leading_index + outer_position
                        if is_3d:
                            index += (partition_position, phase_position, slice(None))
                        else:
                            index += (phase_position, slice(None))
                        kspace_values[index] = signal_values[..., samples]

            first_partition = partition_values[0] if is_3d else None
            first_phase_records = sorted(
                groups[(first_outer, first_partition)],
                key=lambda record: record['k_phase'],
            )
            read_axes = [
                np.sort(k_read[record['samples']])
                for record in first_phase_records
            ]
            read_axis, _ = _canonical_cartesian_axis(
                np.median(read_axes, axis=0)
            )
            phase_axis, _ = _canonical_cartesian_axis(
                [record['k_phase'] for record in first_phase_records]
            )
            coordinate_values = {
                read_dim: np.arange(read_matrix),
                phase_dim: np.arange(phase_matrix),
                'cartesian_k_read_cyc_per_m': (read_dim, read_axis),
                'cartesian_k_phase_cyc_per_m': (phase_dim, phase_axis),
            }
            logical_axes = {
                'read': (read_dim, read_axis),
                'phase': (phase_dim, phase_axis),
            }
            if is_3d:
                partition_axis, _ = _canonical_cartesian_axis(
                    [
                        float(
                            np.median(
                                [
                                    record['k_partition']
                                    for record in groups[(first_outer, partition)]
                                ]
                            )
                        )
                        for partition in partition_values
                    ]
                )
                coordinate_values.update(
                    {
                        partition_dim: np.arange(partition_values.size),
                        'cartesian_k_partition_cyc_per_m': (
                            partition_dim,
                            partition_axis,
                        ),
                    }
                )
                logical_axes['partition'] = (partition_dim, partition_axis)
            for (role, values), axis_code in zip(
                logical_axes.items(), encoding_axes
            ):
                sign = -1.0 if axis_code.startswith('-') else 1.0
                coordinate_values[f'cartesian_k{axis_code[-1]}_cyc_per_m'] = (
                    values[0],
                    sign * values[1],
                )
            for dimension in leading_dims:
                if dimension in signal.coords:
                    coordinate_values[dimension] = signal.coords[dimension]
            for dimension in outer_dims:
                coordinate_values[dimension] = outer_values[dimension]
            dims = leading_dims + outer_dims + spatial_dims
            return xr.DataArray(
                kspace_values,
                dims=dims,
                coords=coordinate_values,
                name=('cartesian_3d_kspace' if is_3d else 'cartesian_kspace'),
                attrs={
                    'source': f'reconstructed from chronological {signal_name}',
                    'adc_sorting': (
                        'adc_event_index, outer labels, logical phase, and '
                        'logical read coordinates'
                    ),
                    'cartesian_encoding_axes': ' '.join(encoding_axes),
                },
            )


        def _cartesian_ifft(kspace, spatial_dims):
            coordinate_names = {
                'read': 'cartesian_k_read_cyc_per_m',
                'phase': 'cartesian_k_phase_cyc_per_m',
                'partition': 'cartesian_k_partition_cyc_per_m',
            }
            centre_phase = np.ones(kspace.shape, dtype=np.complex128)
            for dimension in spatial_dims:
                size = kspace.sizes[dimension]
                role = dimension.split('_', 1)[0]
                coordinate_name = coordinate_names[role]
                if coordinate_name not in kspace.coords:
                    physical_name = f'cartesian_k{dimension[-1]}_cyc_per_m'
                    coordinate_name = physical_name
                if coordinate_name in kspace.coords:
                    _, cells = _canonical_cartesian_axis(
                        np.asarray(kspace.coords[coordinate_name].values)
                    )
                else:
                    cells = np.arange(size, dtype=float) - size // 2
                shape = [1] * kspace.ndim
                shape[kspace.get_axis_num(dimension)] = size
                centre_phase *= np.exp(1j * np.pi * cells / size).reshape(shape)
            axes = tuple(kspace.get_axis_num(dimension) for dimension in spatial_dims)
            corrected = np.asarray(kspace.values) * centre_phase
            image = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(corrected, axes=axes), axes=axes
                ),
                axes=axes,
            )
            return xr.DataArray(
                image,
                dims=kspace.dims,
                coords=kspace.coords,
                attrs={
                    'source': f'centred IFFT of {kspace.name}',
                    'voxel_centered_phase_correction': True,
                },
            )


        if 'cartesian_3d_kspace' not in ds and 'cartesian_kspace' not in ds:
            try:
                reconstructed_kspace = _cartesian_from_adc(ds)
            except ValueError as exc:
                print(f'Automatic Cartesian reconstruction unavailable: {exc}')
            else:
                ds[reconstructed_kspace.name] = reconstructed_kspace
                print(
                    f'Built {reconstructed_kspace.name} from chronological ADC data: '
                    f'{reconstructed_kspace.dims} {reconstructed_kspace.shape}'
                )

        if 'cartesian_3d_kspace' in ds:
            spatial_dims = _cartesian_spatial_dims(ds.cartesian_3d_kspace)
            notebook_image = _cartesian_ifft(
                ds.cartesian_3d_kspace,
                spatial_dims,
            )
            ds['notebook_cartesian_3d_image'] = notebook_image
            ds['notebook_cartesian_3d_image_magnitude'] = np.abs(notebook_image)
            print('Reconstructed notebook_cartesian_3d_image with a centred 3D IFFT.')
        elif 'cartesian_kspace' in ds:
            spatial_dims = _cartesian_spatial_dims(ds.cartesian_kspace)
            notebook_image = _cartesian_ifft(
                ds.cartesian_kspace, spatial_dims
            )
            ds['notebook_cartesian_image'] = notebook_image
            ds['notebook_cartesian_image_magnitude'] = np.abs(notebook_image)
            print('Reconstructed notebook_cartesian_image with a centred 2D IFFT.')

        if 'species_signal' in ds:
            species_name = (
                'species_cartesian_3d_kspace'
                if 'cartesian_3d_kspace' in ds
                else 'species_cartesian_kspace'
            )
            try:
                species_kspace = _cartesian_from_adc(ds, 'species_signal')
            except ValueError:
                species_kspace = None
            if species_kspace is not None:
                ds[species_name] = species_kspace
                species_spatial_dims = _cartesian_spatial_dims(species_kspace)
                species_image = _cartesian_ifft(
                    species_kspace, species_spatial_dims
                )
                image_name = species_name.replace('kspace', 'image')
                ds[image_name] = species_image
                ds[f'{image_name}_magnitude'] = np.abs(species_image)
        """
    ).strip()


def _sequence_result_explorer_code() -> str:
    """Return the adaptive ipywidgets explorer used by result notebooks."""
    return dedent(
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "The interactive result explorer requires ipywidgets. "
                "Install it with `%pip install ipywidgets`."
            ) from exc

        def _role_dimension(data, role, default):
            matches = [
                dimension
                for dimension in data.dims
                if dimension.startswith(f'{role}_')
            ]
            return matches[0] if matches else default


        if (
            'radial_3d_image_magnitude' in ds
            or 'radial_3d_image' in ds
        ):
            explorer_kind = 'radial_3d'
            orientation_source = (
                ds.radial_3d_image_magnitude
                if 'radial_3d_image_magnitude' in ds
                else ds.radial_3d_image
            )
            x_dim, y_dim, z_dim = 'radial_x', 'radial_y', 'radial_z'
            repetition_dim = 'repetition' if 'repetition' in ds.dims else None
            spectral_dim = None
        elif (
            'notebook_cartesian_3d_image_magnitude' in ds
            or 'cartesian_3d_image_magnitude' in ds
        ):
            explorer_kind = 'cartesian_3d'
            orientation_source = (
                ds.notebook_cartesian_3d_image_magnitude
                if 'notebook_cartesian_3d_image_magnitude' in ds
                else ds.cartesian_3d_image_magnitude
            )
            x_dim = _role_dimension(orientation_source, 'read', 'read_x')
            y_dim = _role_dimension(orientation_source, 'phase', 'phase_y')
            z_dim = _role_dimension(
                orientation_source, 'partition', 'partition_z'
            )
            repetition_dim = 'repetition' if 'repetition' in ds.dims else None
            spectral_dim = None
        elif 'csi_kspace' in ds:
            explorer_kind = 'csi'
            orientation_source = ds.csi_spatial_fid
            x_dim, y_dim, z_dim = 'phase_x', 'phase_y', None
            repetition_dim = 'repetition' if 'repetition' in ds.dims else None
            spectral_dim = 'spectral_point'
        elif (
            'notebook_cartesian_image_magnitude' in ds
            or 'cartesian_image_magnitude' in ds
            or 'cartesian_image' in ds
        ):
            explorer_kind = 'cartesian_2d'
            orientation_source = (
                ds.notebook_cartesian_image_magnitude
                if 'notebook_cartesian_image_magnitude' in ds
                else (
                    ds.cartesian_image_magnitude
                    if 'cartesian_image_magnitude' in ds
                    else ds.cartesian_image
                )
            )
            x_dim = _role_dimension(orientation_source, 'read', 'read_x')
            y_dim = _role_dimension(orientation_source, 'phase', 'phase_y')
            z_dim = None
            repetition_dim = (
                'cartesian_frame' if 'cartesian_frame' in ds.dims else None
            )
            spectral_dim = None
        elif 'spiral_image_magnitude' in ds:
            explorer_kind = 'spiral_2d'
            orientation_source = ds.spiral_image_magnitude
            x_dim, y_dim, z_dim = 'read_x', 'phase_y', None
            repetition_dim = 'spiral_frame' if 'spiral_frame' in ds.dims else None
            spectral_dim = None
        else:
            explorer_kind = 'raw_signal'
            orientation_source = None
            x_dim = y_dim = z_dim = repetition_dim = spectral_dim = None

        outer_dims = []
        if orientation_source is not None:
            excluded_dims = {
                x_dim, y_dim, z_dim, spectral_dim, repetition_dim, 'coil', 'pool', None
            }
            outer_dims = [
                dim for dim in orientation_source.dims
                if dim not in excluded_dims and ds.sizes[dim] > 1
            ]


        def _index_slider(label, dim, initial=None):
            available = dim is not None and dim in ds.sizes
            size = int(ds.sizes[dim]) if available else 1
            if initial is None:
                initial = (
                    size // 2
                    if dim is not None
                    and dim.startswith(('read_', 'phase_', 'partition_'))
                    else 0
                )
            return widgets.IntSlider(
                value=min(int(initial), size - 1),
                min=0,
                max=size - 1,
                step=1,
                description=label if available else f'{label} (n/a)',
                disabled=not available or size == 1,
                continuous_update=True,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='260px'),
            )


        def _rss_magnitude(data):
            if 'coil' in data.dims:
                return np.sqrt((np.abs(data) ** 2).sum('coil'))
            return np.abs(data)


        def _select_outer(data, keep_dims, repetition, outer_indices=None):
            outer_indices = {} if outer_indices is None else dict(outer_indices)
            selectors = {}
            for dim in data.dims:
                if dim in keep_dims or dim == 'coil':
                    continue
                if dim == repetition_dim:
                    selectors[dim] = repetition
                else:
                    selectors[dim] = int(outer_indices.get(dim, 0))
            return data.isel(selectors)


        def _crosshair(axis, horizontal, vertical):
            axis.axhline(horizontal, color='cyan', linewidth=0.8, alpha=0.8)
            axis.axvline(vertical, color='cyan', linewidth=0.8, alpha=0.8)


        def _display_figure_once(fig):
            # interactive_output flushes inline Matplotlib figures after every
            # callback. Close explicitly after display so that the same figure
            # is not emitted again by that flush or at the end of the cell.
            display(fig)
            plt.close(fig)


        def _display_range_source():
            if explorer_kind == 'radial_3d':
                name = (
                    'radial_3d_image_magnitude'
                    if 'radial_3d_image_magnitude' in ds
                    else 'radial_3d_image'
                )
            elif explorer_kind == 'cartesian_3d':
                name = (
                    'notebook_cartesian_3d_image_magnitude'
                    if 'notebook_cartesian_3d_image_magnitude' in ds
                    else 'cartesian_3d_image_magnitude'
                )
            elif explorer_kind == 'cartesian_2d':
                name = (
                    'notebook_cartesian_image_magnitude'
                    if 'notebook_cartesian_image_magnitude' in ds
                    else (
                        'cartesian_image_magnitude'
                        if 'cartesian_image_magnitude' in ds
                        else 'cartesian_image'
                    )
                )
            elif explorer_kind == 'spiral_2d':
                name = 'spiral_image_magnitude'
            elif explorer_kind == 'csi':
                name = 'csi_spectrum'
            else:
                return None
            return _rss_magnitude(ds[name])


        def _display_range_slider():
            source = _display_range_source()
            if source is None:
                return widgets.FloatRangeSlider(
                    value=(0.0, 1.0),
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    description='Display range (n/a)',
                    disabled=True,
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='520px'),
                )
            data_max = float(source.max(skipna=True).item())
            if not np.isfinite(data_max):
                data_max = 1.0
            slider_max = data_max if data_max > 0.0 else 1.0
            label = (
                'Spectrum y-range'
                if explorer_kind == 'csi'
                else 'Image range'
            )
            return widgets.FloatRangeSlider(
                value=(0.0, slider_max),
                min=0.0,
                max=slider_max,
                step=slider_max / 1000.0,
                description=label,
                disabled=data_max <= 0.0,
                continuous_update=False,
                readout_format='.4g',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='520px'),
            )


        def _display_limits(display_range):
            display_min, display_max = map(float, display_range)
            if display_max <= display_min:
                display_max = np.nextafter(display_min, np.inf)
            return display_min, display_max


        def _repetition_label(index):
            if repetition_dim is None:
                return 'n/a'
            coordinate_name = repetition_dim
            if (
                explorer_kind in {'cartesian_2d', 'spiral_2d'}
                and 'cartesian_frame_repetition_index' in ds.coords
            ):
                coordinate_name = 'cartesian_frame_repetition_index'
            if (
                explorer_kind == 'spiral_2d'
                and 'spiral_frame_repetition_index' in ds.coords
            ):
                coordinate_name = 'spiral_frame_repetition_index'
            if coordinate_name in ds.coords:
                value = np.asarray(ds.coords[coordinate_name].values)[index]
                return str(value.item() if hasattr(value, 'item') else value)
            return str(index)


        def _outer_label(repetition, outer_indices):
            labels = []
            if repetition_dim is not None:
                labels.append(f'repetition={_repetition_label(repetition)}')
            for dim in outer_dims:
                index = int(outer_indices.get(dim, 0))
                if dim in ds.coords:
                    value = np.asarray(ds.coords[dim].values)[index]
                    value = value.item() if hasattr(value, 'item') else value
                else:
                    value = index
                labels.append(f'{dim}={value}')
            return ', '.join(labels) if labels else 'single acquisition'


        def _show_cartesian_3d(
            x, y, z, repetition, display_range, display_auto, outer_indices
        ):
            spatial_dims = {z_dim, y_dim, x_dim}
            if explorer_kind == 'radial_3d':
                image_name = (
                    'radial_3d_image_magnitude'
                    if 'radial_3d_image_magnitude' in ds
                    else 'radial_3d_image'
                )
                kspace_name = 'radial_3d_gridded_kspace'
            else:
                image_name = (
                    'notebook_cartesian_3d_image_magnitude'
                    if 'notebook_cartesian_3d_image_magnitude' in ds
                    else 'cartesian_3d_image_magnitude'
                )
                kspace_name = 'cartesian_3d_kspace'
            image = _select_outer(
                ds[image_name], spatial_dims, repetition, outer_indices
            )
            kspace = _select_outer(
                ds[kspace_name], spatial_dims, repetition, outer_indices
            )
            image = _rss_magnitude(image).transpose(
                z_dim, y_dim, x_dim
            )
            kspace = _rss_magnitude(kspace).transpose(
                z_dim, y_dim, x_dim
            )
            volume = np.asarray(image)
            kspace_volume = np.asarray(kspace)
            if display_auto:
                display_range = (float(volume[z].min()), float(volume[z].max()))
            display_min, display_max = _display_limits(display_range)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            axes[0, 0].imshow(
                volume[z],
                origin='lower',
                cmap='gray',
                aspect='auto',
                vmin=display_min,
                vmax=display_max,
            )
            _crosshair(axes[0, 0], y, x)
            axes[0, 0].set(
                title=f'{x_dim}/{y_dim} reconstruction at {z_dim}={z}',
                xlabel=x_dim,
                ylabel=y_dim,
            )

            axes[0, 1].imshow(
                volume[:, y, :],
                origin='lower',
                cmap='gray',
                aspect='auto',
                vmin=display_min,
                vmax=display_max,
            )
            _crosshair(axes[0, 1], z, x)
            axes[0, 1].set(
                title=f'{x_dim}/{z_dim} reconstruction at {y_dim}={y}',
                xlabel=x_dim,
                ylabel=z_dim,
            )

            axes[1, 0].imshow(
                volume[:, :, x],
                origin='lower',
                cmap='gray',
                aspect='auto',
                vmin=display_min,
                vmax=display_max,
            )
            _crosshair(axes[1, 0], z, y)
            axes[1, 0].set(
                title=f'{y_dim}/{z_dim} reconstruction at {x_dim}={x}',
                xlabel=y_dim,
                ylabel=z_dim,
            )

            axes[1, 1].imshow(
                np.log1p(kspace_volume[z]),
                origin='lower',
                cmap='magma',
                aspect='auto',
            )
            _crosshair(axes[1, 1], y, x)
            axes[1, 1].set(
                title=f'log(1 + |k-space|) at {z_dim} index {z}',
                xlabel='k_read',
                ylabel='k_phase',
            )
            fig.suptitle(
                f'{_outer_label(repetition, outer_indices)} · '
                f'voxel ({x_dim}={x}, {y_dim}={y}, {z_dim}={z}) · '
                f'magnitude={volume[z, y, x]:.5g}'
            )
            _display_figure_once(fig)


        def _show_cartesian_2d(
            x, y, repetition, display_range, display_auto, outer_indices
        ):
            if explorer_kind == 'spiral_2d':
                display_x_dim, display_y_dim = 'read_x', 'phase_y'
                image_name = 'spiral_image_magnitude'
                kspace_name = 'spiral_gridded_kspace'
            else:
                display_x_dim, display_y_dim = x_dim, y_dim
                image_name = (
                    'notebook_cartesian_image_magnitude'
                    if 'notebook_cartesian_image_magnitude' in ds
                    else (
                        'cartesian_image_magnitude'
                        if 'cartesian_image_magnitude' in ds
                        else 'cartesian_image'
                    )
                )
                kspace_name = 'cartesian_kspace'
            spatial_dims = {display_y_dim, display_x_dim}
            image = _rss_magnitude(
                _select_outer(ds[image_name], spatial_dims, repetition, outer_indices)
            ).transpose(display_y_dim, display_x_dim)
            kspace = _rss_magnitude(
                _select_outer(ds[kspace_name], spatial_dims, repetition, outer_indices)
            ).transpose(display_y_dim, display_x_dim)
            image_values = np.asarray(image)
            kspace_values = np.asarray(kspace)
            if display_auto:
                display_range = (float(image_values.min()), float(image_values.max()))
            display_min, display_max = _display_limits(display_range)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            axes[0].imshow(
                np.log1p(kspace_values), origin='lower', cmap='magma', aspect='auto'
            )
            _crosshair(axes[0], y, x)
            axes[0].set(
                title='log(1 + |k-space|)',
                xlabel='k_read',
                ylabel='k_phase',
            )
            axes[1].imshow(
                image_values,
                origin='lower',
                cmap='gray',
                aspect='auto',
                vmin=display_min,
                vmax=display_max,
            )
            _crosshair(axes[1], y, x)
            axes[1].set(
                title='Reconstruction',
                xlabel=display_x_dim,
                ylabel=display_y_dim,
            )
            fig.suptitle(
                f'{_outer_label(repetition, outer_indices)} · '
                f'pixel ({display_x_dim}={x}, {display_y_dim}={y}) · '
                f'magnitude={image_values[y, x]:.5g}'
            )
            _display_figure_once(fig)


        def _show_csi(
            x, y, spectral_point, repetition, display_range, outer_indices
        ):
            cube_dims = {'phase_y', 'phase_x', 'spectral_point'}
            kspace = _rss_magnitude(
                _select_outer(ds['csi_kspace'], cube_dims, repetition, outer_indices)
            ).transpose('phase_y', 'phase_x', 'spectral_point')
            spatial_fid = _rss_magnitude(
                _select_outer(
                    ds['csi_spatial_fid'], cube_dims, repetition, outer_indices
                )
            ).transpose('phase_y', 'phase_x', 'spectral_point')
            spectrum = _rss_magnitude(
                _select_outer(ds['csi_spectrum'], cube_dims, repetition, outer_indices)
            ).transpose('phase_y', 'phase_x', 'spectral_point')

            kspace_map = np.asarray(kspace.isel(spectral_point=spectral_point))
            fid_map = np.asarray(spatial_fid.isel(spectral_point=spectral_point))
            spectrum_line = np.asarray(spectrum.isel(phase_y=y, phase_x=x))
            if 'spectral_frequency_hz' in ds.coords:
                spectral_axis = np.asarray(ds.spectral_frequency_hz)
                spectral_label = 'Frequency (Hz)'
            else:
                spectral_axis = np.arange(spectrum_line.size)
                spectral_label = 'Spectral point'

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            axes[0].imshow(kspace_map, origin='lower', cmap='magma', aspect='auto')
            _crosshair(axes[0], y, x)
            axes[0].set(
                title=f'CSI k-space · point {spectral_point}',
                xlabel='kx index',
                ylabel='ky index',
            )
            axes[1].imshow(fid_map, origin='lower', cmap='gray', aspect='auto')
            _crosshair(axes[1], y, x)
            axes[1].set(
                title=f'Spatial FID magnitude · point {spectral_point}',
                xlabel='x',
                ylabel='y',
            )
            axes[2].plot(spectral_axis, spectrum_line)
            axes[2].axvline(
                spectral_axis[spectral_point], color='tab:red', linewidth=1.0
            )
            axes[2].set(
                title=f'Spectrum at (x={x}, y={y})',
                xlabel=spectral_label,
                ylabel='Magnitude',
            )
            axes[2].set_ylim(*_display_limits(display_range))
            fig.suptitle(
                f'Spectral point {spectral_point} · '
                f'spectrum magnitude={spectrum_line[spectral_point]:.5g}'
            )
            _display_figure_once(fig)


        def _render_explorer(
            x, y, z, repetition, spectral_point, display_range, display_auto,
            **outer_values,
        ):
            outer_indices = {
                key[len('outer__'):]: value
                for key, value in outer_values.items()
                if key.startswith('outer__')
            }
            if explorer_kind in {'cartesian_3d', 'radial_3d'}:
                _show_cartesian_3d(
                    x, y, z, repetition, display_range, display_auto, outer_indices
                )
            elif explorer_kind in {'cartesian_2d', 'spiral_2d'}:
                _show_cartesian_2d(
                    x, y, repetition, display_range, display_auto, outer_indices
                )
            elif explorer_kind == 'csi':
                _show_csi(
                    x, y, spectral_point, repetition, display_range, outer_indices
                )
            else:
                print(
                    'No gridded Cartesian, spiral, or CSI data were found. '
                    'Inspect signal with the ADC-coordinate table above.'
                )


        x_slider = _index_slider(x_dim or 'x', x_dim)
        y_slider = _index_slider(y_dim or 'y', y_dim)
        z_slider = _index_slider(z_dim or 'z', z_dim)
        repetition_slider = _index_slider('Repetition', repetition_dim, initial=0)
        spectral_point_slider = _index_slider(
            'Spectral point', spectral_dim, initial=0
        )
        display_range_slider = _display_range_slider()
        display_auto_checkbox = widgets.Checkbox(
            value=True,
            description='Auto display range',
            disabled=False,)
        controls = {
            'x': x_slider,
            'y': y_slider,
            'z': z_slider,
            'repetition': repetition_slider,
            'spectral_point': spectral_point_slider,
            'display_range': display_range_slider,
            'display_auto': display_auto_checkbox,
        }
        controls.update(
            {
                f'outer__{dim}': _index_slider(dim.replace('_', ' ').title(), dim, 0)
                for dim in outer_dims
            }
        )
        output = widgets.interactive_output(_render_explorer, controls)
        control_row = widgets.Box(
            list(controls.values()),
            layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='center'
            ),
        )
        display(
            widgets.VBox(
                [
                    widgets.HTML(
                        f'<b>Detected data:</b> {explorer_kind}. '
                        'Move a slider to update all linked views. '
                        'Use the range control to set reconstruction contrast '
                        'or the spectrum y-axis.'
                    ),
                    control_row,
                    output,
                ]
            )
        )
        """
    ).strip()


def export_pulseq_generation_notebook(
    filename: str,
    sequence_kind: str,
    parameters: Dict[str, Any],
    *,
    seq_filename: Optional[str] = None,
) -> Path:
    """Create a notebook that regenerates one GUI-built Pulseq sequence."""
    if not HAS_NBFORMAT:
        raise ImportError("Jupyter notebook export requires nbformat")
    builders = {
        "epi": "make_pulseq_epi",
        "spiral": "make_pulseq_spiral",
        "csi": "make_pulseq_csi",
        "flash": "make_pulseq_flash",
        "bssfp_3d": "make_pulseq_bssfp",
        "spectral_bssfp_3d": "make_pulseq_spectral_selective_bssfp",
        "me_bssfp_3d": "make_pulseq_me_bssfp",
        "radial_me_bssfp_3d": "make_pulseq_radial_me_bssfp",
    }
    try:
        builder_name = builders[str(sequence_kind)]
    except KeyError as exc:
        raise ValueError(
            "sequence_kind must be 'epi', 'spiral', 'csi', 'flash', 'bssfp_3d', "
            "'spectral_bssfp_3d', 'me_bssfp_3d', or 'radial_me_bssfp_3d'"
        ) from exc
    notebook_path = Path(filename)
    if notebook_path.suffix.lower() != ".ipynb":
        notebook_path = notebook_path.with_suffix(".ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    output_name = (
        Path(seq_filename).name
        if seq_filename is not None
        else f"{notebook_path.stem}.seq"
    )
    parameter_literal = pformat(dict(parameters), sort_dicts=False, width=88)
    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# Reproduce Pulseq sequence\n\n"
                f"Generated by BlochSimulator {__version__}. This notebook uses "
                f"`{builder_name}` with the exact parameters selected in the "
                "Sequence Simulation workspace at export time."
            ),
            new_code_cell(
                "from pathlib import Path\n"
                f"from blochsimulator.sequence import {builder_name}\n\n"
                f"parameters = {parameter_literal}\n"
                "parameters"
            ),
            new_code_cell(
                f"sequence = {builder_name}(**parameters)\n"
                f"output_path = Path({output_name!r})\n"
                "sequence.write(str(output_path), v141_compat=True)\n"
                "print(f'Wrote {output_path.resolve()}')\n"
                "sequence"
            ),
        ]
    )
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return notebook_path


def export_sequence_result_notebook(filename: str, data_filename: str) -> Path:
    """Create an xarray-based analysis notebook for sparse sequence output."""
    if not HAS_NBFORMAT:
        raise ImportError("Jupyter notebook export requires nbformat")
    notebook_path = Path(filename)
    data_path = Path(data_filename)
    relative_data = os.path.relpath(
        data_path.resolve(), start=notebook_path.parent.resolve()
    )
    absolute_data = str(data_path.resolve())
    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# Sequence simulation result\n\n"
                f"BlochSimulator {__version__} sparse event-based result. "
                f"The xarray dataset is stored in `{relative_data}`."
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import xarray as xr\n"
                "import matplotlib.pyplot as plt\n\n"
                f"data_path = Path({relative_data!r})\n"
                "if not data_path.exists():\n"
                f"    original_data_path = Path({absolute_data!r})\n"
                "    if original_data_path.exists():\n"
                "        data_path = original_data_path\n"
                "    else:\n"
                "        raise FileNotFoundError(\n"
                "            f'Could not find result data at {data_path} or '\n"
                "            f'{original_data_path}. Move the .nc file next to '\n"
                "            'the notebook or update data_path.'\n"
                "        )\n"
                "ds = xr.open_dataset(data_path)\n"
                "for name in list(ds.data_vars):\n"
                "    if not name.endswith('_real'):\n"
                "        continue\n"
                "    base = name[:-5]\n"
                "    imag = f'{base}_imag'\n"
                "    if imag in ds:\n"
                "        ds[base] = ds[name] + 1j * ds[imag]\n"
                "ds"
            ),
            new_markdown_cell("## ADC signal"),
            new_code_cell(
                "signal = ds.signal.values\n"
                "time_ms = ds.adc_time_s.values * 1e3\n"
                "fig, ax = plt.subplots(figsize=(9, 4))\n"
                "if signal.ndim == 1:\n"
                "    ax.plot(time_ms, np.abs(signal), label='Magnitude')\n"
                "else:\n"
                "    for coil, values in enumerate(signal):\n"
                "        ax.plot(time_ms, np.abs(values), label=f'Coil {coil + 1}')\n"
                "ax.set(xlabel='Time (ms)', ylabel='Signal (a.u.)')\n"
                "ax.legend(); ax.grid(True); plt.show()"
            ),
            new_markdown_cell(
                "## ADC order and k-space coordinates\n\n"
                "`signal` is stored in chronological ADC order. Every sample has "
                "the same `adc` coordinate as `kx`, `ky`, `kz`, "
                "`adc_event_index`, and `readout_sample_index`. Pulseq outer "
                "labels are available as `slice_index`, `echo_index`, "
                "`repetition_index`, `segment_index`, and `partition_index`. "
                "Use these coordinates for auditing and grouping; do not infer "
                "spatial ordering from array length. Cartesian/EPI exports contain "
                "the already sorted `cartesian_kspace(phase_*, read_*)` array and "
                "`cartesian_image`. Cartesian 3D acquisitions additionally contain "
                "`cartesian_3d_kspace(..., partition_*, phase_*, read_*)` and the "
                "corresponding 3D reconstruction. Spiral exports contain linearly "
                "gridded `spiral_gridded_kspace` and `spiral_image` arrays. "
                "Supported radial 3D exports contain density-compensated "
                "`radial_3d_gridded_kspace` and `radial_3d_image` arrays. CSI "
                "exports contain the already sorted "
                "`csi_kspace(phase_y, phase_x, spectral_point)` array."
            ),
            new_code_cell(
                "coordinate_names = [name for name in (\n"
                "    'kx', 'ky', 'kz', 'adc_event_index', "
                "'readout_sample_index',\n"
                "    'slice_index', 'echo_index', 'repetition_index',\n"
                "    'segment_index', 'partition_index'\n"
                ") if name in ds.coords]\n"
                "adc_table = ds[coordinate_names].to_dataframe()\n"
                "adc_table['signal'] = ds.signal.values if ds.signal.ndim == 1 else list(ds.signal.values.T)\n"
                "adc_table.head()"
            ),
            new_markdown_cell(
                "## Reconstruction preparation\n\n"
                "This section performs the centered inverse FFT inside the notebook. "
                "If an older result file has only chronological ADC data, the helper "
                "first validates and sorts it with the exported event, outer-label, "
                "partition, and logical encoding coordinates. It does not reshape based on "
                "the sample count alone. Pool-resolved `species_signal` data are "
                "reconstructed as separate variables when available."
            ),
            new_code_cell(_sequence_result_reconstruction_code()),
            new_code_cell(
                "if 'radial_3d_gridded_kspace' in ds:\n"
                "    kspace_3d = ds.radial_3d_gridded_kspace\n"
                "    image_3d = ds.radial_3d_image_magnitude\n"
                "    spatial_dims = {'radial_z', 'radial_y', 'radial_x'}\n"
                "    selectors = {dim: 0 for dim in kspace_3d.dims if dim not in spatial_dims | {'coil'}}\n"
                "    kspace_volume = kspace_3d.isel(selectors)\n"
                "    image_volume = image_3d.isel(selectors)\n"
                "    if 'coil' in kspace_volume.dims:\n"
                "        kspace_volume = np.sqrt((abs(kspace_volume) ** 2).sum('coil'))\n"
                "        image_volume = np.sqrt((abs(image_volume) ** 2).sum('coil'))\n"
                "    z_mid = kspace_volume.sizes['radial_z'] // 2\n"
                "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
                "    axes[0].imshow(np.log1p(abs(kspace_volume.isel(radial_z=z_mid))), origin='lower', cmap='magma')\n"
                "    axes[0].set_title('Central radial gridded k-space plane')\n"
                "    axes[1].imshow(abs(image_volume.isel(radial_z=z_mid)), origin='lower', cmap='gray')\n"
                "    axes[1].set_title('Central radial reconstruction slice')\n"
                "    plt.tight_layout(); plt.show()\n"
                "elif 'cartesian_3d_kspace' in ds:\n"
                "    kspace_3d = ds.cartesian_3d_kspace\n"
                "    image_3d = ds.notebook_cartesian_3d_image_magnitude\n"
                "    partition_dim, phase_dim, read_dim = _cartesian_spatial_dims(kspace_3d)\n"
                "    spatial_dims = {partition_dim, phase_dim, read_dim}\n"
                "    selectors = {dim: 0 for dim in kspace_3d.dims "
                "if dim not in spatial_dims | {'coil'}}\n"
                "    kspace_volume = kspace_3d.isel(selectors)\n"
                "    image_volume = image_3d.isel(selectors)\n"
                "    if 'coil' in kspace_volume.dims:\n"
                "        kspace_volume = np.sqrt((abs(kspace_volume) ** 2).sum('coil'))\n"
                "        image_volume = np.sqrt((abs(image_volume) ** 2).sum('coil'))\n"
                "    partition_mid = kspace_volume.sizes[partition_dim] // 2\n"
                "    image_partition_mid = image_volume.sizes[partition_dim] // 2\n"
                "    print('Sorted Cartesian 3D k-space:', kspace_3d.dims, kspace_3d.shape)\n"
                "    print('Encoding axes:', ds.attrs.get('cartesian_encoding_axes', '+x +y +z'))\n"
                "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
                "    axes[0].imshow(np.log1p(abs(kspace_volume.isel({partition_dim: partition_mid}))), origin='lower', cmap='magma')\n"
                "    axes[0].set_title(f'Central k-partition plane ({partition_dim})')\n"
                "    axes[1].imshow(abs(image_volume.isel({partition_dim: image_partition_mid})), origin='lower', cmap='gray')\n"
                "    axes[1].set_title(f'Central reconstructed {partition_dim} slice')\n"
                "    plt.tight_layout(); plt.show()\n"
                "elif 'cartesian_kspace' in ds:\n"
                "    kspace = ds.cartesian_kspace\n"
                "    image = ds.notebook_cartesian_image_magnitude\n"
                "    phase_dim, read_dim = _cartesian_spatial_dims(kspace)\n"
                "    spatial_dims = {phase_dim, read_dim}\n"
                "    selectors = {dim: 0 for dim in kspace.dims "
                "if dim not in spatial_dims | {'coil'}}\n"
                "    kspace = kspace.isel(selectors)\n"
                "    image = image.isel(selectors)\n"
                "    if 'coil' in kspace.dims:\n"
                "        kspace_display = np.sqrt((abs(kspace) ** 2).sum('coil'))\n"
                "        image_display = np.sqrt((abs(image) ** 2).sum('coil'))\n"
                "    else:\n"
                "        kspace_display = abs(kspace)\n"
                "        image_display = abs(image)\n"
                "    print('Sorted Cartesian/EPI k-space:', kspace.dims, kspace.shape)\n"
                "    read_coord = 'cartesian_k_read_cyc_per_m' if 'cartesian_k_read_cyc_per_m' in ds.coords else f'cartesian_k{read_dim[-1]}_cyc_per_m'\n"
                "    phase_coord = 'cartesian_k_phase_cyc_per_m' if 'cartesian_k_phase_cyc_per_m' in ds.coords else f'cartesian_k{phase_dim[-1]}_cyc_per_m'\n"
                "    print('k-read axis:', ds.coords[read_coord].values[:4], '...')\n"
                "    print('k-phase axis:', ds.coords[phase_coord].values[:4], '...')\n"
                "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
                "    axes[0].imshow(np.log1p(kspace_display.values), origin='lower', cmap='magma')\n"
                "    axes[0].set_title('log(1 + |k-space|)')\n"
                "    axes[0].set_xlabel(f'{read_dim} / k-read'); axes[0].set_ylabel(f'{phase_dim} / k-phase')\n"
                "    axes[1].imshow(image_display.values, origin='lower', cmap='gray')\n"
                "    axes[1].set_title('|IFFT2 image|')\n"
                "    axes[1].set_xlabel(read_dim); axes[1].set_ylabel(phase_dim)\n"
                "    plt.tight_layout(); plt.show()\n"
                "elif 'spiral_gridded_kspace' in ds:\n"
                "    kspace = ds.spiral_gridded_kspace\n"
                "    image = ds.spiral_image_magnitude\n"
                "    selectors = {dim: 0 for dim in kspace.dims "
                "if dim not in {'coil', 'phase_y', 'read_x'}}\n"
                "    kspace = kspace.isel(selectors); image = image.isel(selectors)\n"
                "    if 'coil' in kspace.dims:\n"
                "        kspace = np.sqrt((abs(kspace) ** 2).sum('coil'))\n"
                "        image = np.sqrt((abs(image) ** 2).sum('coil'))\n"
                "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
                "    axes[0].imshow(np.log1p(abs(kspace)), origin='lower', cmap='magma')\n"
                "    axes[0].set_title('Linearly gridded spiral k-space')\n"
                "    axes[1].imshow(abs(image), origin='lower', cmap='gray')\n"
                "    axes[1].set_title('Spiral reconstruction')\n"
                "    plt.tight_layout(); plt.show()\n"
                "elif 'csi_kspace' in ds:\n"
                "    kspace = ds.csi_kspace\n"
                "    print('Sorted CSI k-space:', kspace.dims, kspace.shape)\n"
                "elif {'kx', 'ky'}.issubset(ds.coords):\n"
                "    print('Chronological kx/ky samples are present, but the validated "
                "Cartesian reconstruction above was unavailable.')"
            ),
            new_markdown_cell(
                "## Interactive multidimensional explorer\n\n"
                "The controls below are connected to the gridded xarray data. "
                "Use the `x`, `y`, and `z` sliders to move the crosshair and "
                "orthogonal slices, `Repetition` to select a dynamic volume or "
                "2D frame, and `Spectral point` to inspect CSI data. Controls "
                "without a matching dataset dimension are disabled automatically. "
                "The two-handle range control adjusts reconstruction `vmin`/`vmax` "
                "or the spectrum y-axis limits."
            ),
            new_code_cell(_sequence_result_explorer_code()),
            new_markdown_cell("## Final longitudinal magnetization"),
            new_code_cell(
                "mz = ds.final_magnetization.sel(component='mz').values\n"
                "while mz.ndim > 2:\n"
                "    mz = np.take(mz, mz.shape[-1] // 2, axis=-1)\n"
                "fig, ax = plt.subplots(figsize=(6, 5))\n"
                "if mz.ndim == 1:\n"
                "    ax.plot(mz)\n"
                "else:\n"
                "    image = ax.imshow(mz.T, origin='lower', cmap='viridis')\n"
                "    fig.colorbar(image, ax=ax, label='Mz')\n"
                "ax.set_title('Final Mz (central slice)'); plt.show()"
            ),
        ]
    )
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return notebook_path


if __name__ == "__main__":
    print("Notebook Exporter for Bloch Simulator")
    print("=" * 60)
    print("\nUsage:")
    print("  from notebook_exporter import export_notebook")
    print("  export_notebook('load_data', 'analysis.ipynb', ...)")
    print("  export_notebook('resimulate', 'reproduce.ipynb', ...)")
