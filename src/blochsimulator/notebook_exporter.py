"""
notebook_exporter.py - Jupyter Notebook Export for Bloch Simulator

This module generates executable Jupyter notebooks from simulation parameters.

Two export modes:
- Mode A: Load data from HDF5 file (for analysis/visualization)
- Mode B: Re-run simulation from parameters (reproducibility)

Author: Bloch Simulator Team
Date: 2024
"""

from typing import List, Dict, Any, Optional
import json

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False
    nbformat = None
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
        xr_code = """# Convert to xarray Dataset for advanced analysis
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

print('Xarray Dataset created:')
print(ds)"""
        cells.append(new_code_cell(xr_code))

        # Cell 3b: Compatibility conversion
        cells.append(
            new_code_cell(
                "# Prepare data for display (convert objects to dictionaries)\n"
                "from dataclasses import asdict\n"
                "if hasattr(data['tissue'], 't1'): # Check if object\n"
                "    data['tissue'] = asdict(data['tissue'])"
            )
        )

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
        xr_code = """# Convert to xarray Dataset for advanced analysis
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

print('Xarray Dataset created:')
print(ds)"""
        cells.append(new_code_cell(xr_code))

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
pos_z_cm = data['positions'][position_idx, 2] * 100
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

    plt.suptitle(f'Magnetization Evolution - Pos: {pos_z_cm:.2f} cm, Freq: {freq_hz:.1f} Hz',
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

pos_z_cm = data['positions'][position_idx, 2] * 100
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

    plt.suptitle(f'MRI Signal - Pos: {pos_z_cm:.2f} cm, Freq: {freq_hz:.1f} Hz',
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
        pos_range = simulation_params.get("position_range_cm", 0.0) / 100.0  # to meters
        freq_range = simulation_params.get("frequency_range_hz", 0.0)

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
    'tissue': {'name': tissue.name, 't1': tissue.t1, 't2': tissue.t2},
    'simulation_params': {
        'num_positions': len(positions),
        'num_frequencies': len(frequencies)
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


if __name__ == "__main__":
    print("Notebook Exporter for Bloch Simulator")
    print("=" * 60)
    print("\nUsage:")
    print("  from notebook_exporter import export_notebook")
    print("  export_notebook('load_data', 'analysis.ipynb', ...)")
    print("  export_notebook('resimulate', 'reproduce.ipynb', ...)")
