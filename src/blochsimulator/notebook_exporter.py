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
from dataclasses import asdict, is_dataclass
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

    @staticmethod
    def _normalise_tissue_params(tissue_params: Dict) -> Dict:
        """Return one unit-explicit tissue parameter representation.

        The GUI state uses ``t1_ms``/``t2_ms`` while older callers of the
        notebook API pass ``t1``/``t2`` in seconds.  Notebook code should not
        expose both schemas or silently fall back to the default tissue.
        """

        raw = dict(tissue_params or {})

        def seconds(name: str, default: float) -> float:
            if f"{name}_s" in raw:
                return float(raw[f"{name}_s"])
            if name in raw:
                return float(raw[name])
            if f"{name}_ms" in raw:
                return float(raw[f"{name}_ms"]) / 1000.0
            return default

        result = {
            "name": raw.get("name", raw.get("preset", "Custom")),
            "t1_s": seconds("t1", 1.0),
            "t2_s": seconds("t2", 0.1),
            "density": float(raw.get("density", 1.0)),
        }
        t2_star = raw.get("t2_star_s", raw.get("t2_star"))
        if t2_star is None and raw.get("t2_star_ms") is not None:
            t2_star = float(raw["t2_star_ms"]) / 1000.0
        if t2_star is not None:
            result["t2_star_s"] = float(t2_star)
        return result

    @staticmethod
    def _normalise_pulse_params(pulse_params: Dict) -> Dict:
        """Make RF pulse units explicit without changing their values."""

        raw = dict(pulse_params or {})
        renamed = {
            "flip_angle": "flip_angle_deg",
            "b1_amplitude": "b1_amplitude_g",
            "phase": "phase_deg",
            "freq_offset": "frequency_offset_hz",
            "loaded_pulse_b1": "loaded_b1_waveform_g",
            "loaded_pulse_time": "loaded_time_waveform_s",
        }
        result = {}
        for key, value in raw.items():
            if key == "duration":
                result["duration_s"] = float(value) / 1000.0
            else:
                result[renamed.get(key, key)] = value
        return result

    @classmethod
    def _normalise_sequence_params(cls, sequence_params: Dict) -> Dict:
        """Canonicalise legacy/GUI sequence metadata for notebook use.

        GUI timing controls are stored in milliseconds as ``te``, ``tr`` and
        ``ti`` and accompanied by computed ``*_s`` aliases.  Older public API
        callers use the unsuffixed names for seconds.  Explicit ``*_s`` values
        therefore take precedence and the ambiguous aliases are never emitted.
        """

        raw = dict(sequence_params or {})
        sequence_type = raw.pop("sequence_type", raw.get("type", "Custom"))
        raw.pop("type", None)

        timings = {}
        for name in ("te", "tr", "ti"):
            value_s = raw.pop(f"{name}_s", None)
            value_ms = raw.pop(f"{name}_ms", None)
            legacy_value = raw.pop(name, None)
            if value_s is not None:
                timings[f"{name}_s"] = float(value_s)
            elif value_ms is not None:
                timings[f"{name}_s"] = float(value_ms) / 1000.0
            elif legacy_value is not None:
                # Backward-compatible NotebookExporter API: unsuffixed timing
                # values were documented and tested as seconds.
                timings[f"{name}_s"] = float(legacy_value)

        type_lower = str(sequence_type).lower()
        if "ssfp" in type_lower:
            active_timings = {"tr_s"}
        elif "inversion recovery" in type_lower:
            active_timings = {"te_s", "tr_s", "ti_s"}
        elif "spin echo" in type_lower or "gradient echo" in type_lower:
            active_timings = {"te_s", "tr_s"}
        elif "slice select" in type_lower:
            active_timings = {"te_s"}
        elif "free induction" in type_lower:
            active_timings = {"tr_s"}
        else:
            active_timings = set(timings)

        result = {"sequence_type": sequence_type}
        result.update(
            (key, value) for key, value in timings.items() if key in active_timings
        )

        # An analytic Free Induction Decay export is intentionally regenerated
        # from its editable RF settings. Keeping the GUI's frozen waveform in
        # that case would silently ignore edits made in the notebook. A custom
        # pulse cannot be regenerated, so it keeps the exact exported arrays.
        configured_pulse_type = raw.get("rf_pulse_type")
        excitation_state = (raw.get("pulse_states") or {}).get("Excitation", {})
        if isinstance(excitation_state, dict):
            configured_pulse_type = excitation_state.get(
                "pulse_type", configured_pulse_type
            )
        regenerate_fid = (
            "free induction" in type_lower
            and str(configured_pulse_type or "gaussian").lower() != "custom"
        )
        if regenerate_fid:
            for key in (
                "b1_waveform",
                "time_waveform",
                "gradients_waveform",
                "b1_waveform_g",
                "time_waveform_s",
                "gradients_waveform_g_per_cm",
            ):
                raw.pop(key, None)

        has_exact_waveforms = (
            raw.get("b1_waveform") is not None and raw.get("time_waveform") is not None
        ) or (
            raw.get("b1_waveform_g") is not None
            and raw.get("time_waveform_s") is not None
        )
        result["sequence_definition_source"] = (
            "exact_exported_waveforms"
            if has_exact_waveforms
            else "regenerated_from_parameters"
        )

        # Unit-explicit top-level sequence fields.
        rename = {
            "flip_angle": "flip_angle_deg",
            "duration": "duration_s",
            "rephase_pct": "rephase_percent",
            "slice_thickness": "slice_thickness_mm",
            "slice_gradient": "slice_gradient_g_per_cm",
            "ssfp_start_phase": "ssfp_start_phase_deg",
            "b1_waveform": "b1_waveform_g",
            "time_waveform": "time_waveform_s",
            "gradients_waveform": "gradients_waveform_g_per_cm",
        }

        # Separate the currently visible RF designer state from pulse-role
        # states.  They are different concepts and used to appear as competing
        # pulse definitions in exported notebooks.
        rf_field_map = {
            "rf_pulse_type": "pulse_type",
            "rf_flip_angle": "flip_angle_deg",
            "rf_duration_s": "duration_s",
            "rf_time_bw_product": "time_bandwidth_product",
            "rf_phase": "phase_deg",
            "rf_freq_offset": "frequency_offset_hz",
            "rf_b1_amplitude": "b1_amplitude_g",
            "rf_sinc_lobes": "sinc_lobes",
            "rf_slr_sharpness": "slr_sharpness",
            "rf_apodization": "apodization",
        }
        rf_snapshot = {}
        for source, target in rf_field_map.items():
            if source in raw:
                rf_snapshot[target] = raw.pop(source)

        pulse_states = raw.pop("pulse_states", {}) or {}
        roles_by_type = {
            "spin echo": {"Excitation", "Refocusing"},
            "spin echo (tip-axis 180)": {"Excitation", "Refocusing"},
            "inversion recovery": {"Inversion", "Excitation"},
            "gradient echo": {"Excitation"},
            "free induction decay": {"Excitation"},
            "flash": {"Excitation"},
            "epi": {"Excitation"},
            "custom": {"Custom Pulse"},
        }
        active_roles = roles_by_type.get(type_lower, set())
        active_pulses = {
            role: cls._normalise_pulse_params(state)
            for role, state in pulse_states.items()
            if role in active_roles and isinstance(state, dict)
        }
        if active_pulses:
            result["sequence_role_pulses"] = active_pulses
        elif rf_snapshot:
            # Sequences without named pulse roles (for example the SSFP loop)
            # use the currently active RF designer configuration.  For named
            # roles the role-specific settings above are the clearer source.
            result["rf_designer_snapshot"] = rf_snapshot

        use_ratios = bool(raw.get("ssfp_use_ratios", False))
        if "ssfp" in type_lower:
            start_delay_ms = raw.pop("ssfp_start_tr", None)
            start_flip_deg = raw.pop("ssfp_start_flip", None)
            if not use_ratios:
                if start_delay_ms is not None:
                    result["ssfp_start_delay_s"] = float(start_delay_ms) / 1000.0
                if start_flip_deg is not None:
                    result["ssfp_start_flip_angle_deg"] = float(start_flip_deg)
            else:
                # In ratio mode the absolute controls are inactive.
                raw.pop("ssfp_start_tr", None)
                raw.pop("ssfp_start_flip", None)
        else:
            # SSFP-only controls are inactive for every other sequence type.
            for key in tuple(raw):
                if key.startswith("ssfp_"):
                    raw.pop(key)

        # Other sequence-specific controls that are visibly inactive should not
        # masquerade as parameters of the selected sequence.
        if "spin echo" not in type_lower:
            raw.pop("echo_count", None)
        if not any(
            name in type_lower
            for name in (
                "spin echo",
                "gradient echo",
                "slice select",
                "epi",
                "inversion recovery",
            )
        ):
            raw.pop("slice_thickness", None)
            raw.pop("slice_gradient", None)
        if "slice select" not in type_lower:
            raw.pop("rephase_pct", None)

        for key, value in raw.items():
            result[rename.get(key, key)] = value
        return result

    @staticmethod
    def _normalise_simulation_params(simulation_params: Dict) -> Dict:
        """Remove duplicate ranges and label coordinate-array units."""

        raw = dict(simulation_params or {})
        result = {}
        rename = {
            "position_axis": "position_axis_m",
            "frequency_axis": "frequency_axis_hz",
            "effective_frequency_axis": "effective_frequency_axis_hz",
        }
        for key, value in raw.items():
            # ``position_range_cm`` is a duplicate compatibility alias whenever
            # the GUI's explicitly labelled millimetre value is present.
            if key == "position_range_cm" and "position_range_mm" in raw:
                continue
            result[rename.get(key, key)] = value
        return result

    @staticmethod
    def _plain_parameter_value(value):
        """Convert parameter objects to executable, portable Python values."""

        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return NotebookExporter._plain_parameter_value(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): NotebookExporter._plain_parameter_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [NotebookExporter._plain_parameter_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(
                NotebookExporter._plain_parameter_value(item) for item in value
            )
        if value is None or isinstance(value, (str, int, float, complex, bool)):
            return value
        return str(value)

    @classmethod
    def _parameter_source(cls, value, arrays: Dict[str, np.ndarray], path=()) -> str:
        """Format parameters as Python and replace arrays with NPZ lookups."""

        references = {}

        def replace_arrays(item, item_path):
            item = cls._plain_parameter_value(item)
            if isinstance(item, np.ndarray):
                key = "__".join(str(part) for part in item_path) or "array"
                candidate = key
                suffix = 2
                while candidate in arrays:
                    candidate = f"{key}_{suffix}"
                    suffix += 1
                arrays[candidate] = item
                token = f"__BLOCH_ARRAY_REFERENCE_{len(references)}__"
                references[token] = candidate
                return token
            if isinstance(item, dict):
                return {
                    key: replace_arrays(child, (*item_path, key))
                    for key, child in item.items()
                }
            if isinstance(item, list):
                return [
                    replace_arrays(child, (*item_path, index))
                    for index, child in enumerate(item)
                ]
            if isinstance(item, tuple):
                return tuple(
                    replace_arrays(child, (*item_path, index))
                    for index, child in enumerate(item)
                )
            return item

        prepared = replace_arrays(value, path)
        source = pformat(prepared, sort_dicts=False, width=88)
        for token, key in references.items():
            source = source.replace(repr(token), f"loaded_arrays[{key!r}]")
        return source

    @classmethod
    def _summarise_arrays(cls, value):
        """Replace arrays by readable summaries for analysis notebook metadata."""

        value = cls._plain_parameter_value(value)
        if isinstance(value, np.ndarray):
            return f"<array shape={value.shape}, dtype={value.dtype}>"
        if isinstance(value, dict):
            return {key: cls._summarise_arrays(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._summarise_arrays(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._summarise_arrays(item) for item in value)
        return value

    @staticmethod
    def _assignment_source(target: str, value_source: str) -> str:
        """Align multiline literals underneath their assignment target."""

        continuation = " " * (len(target) + 3)
        return f"{target} = {value_source.replace(chr(10), chr(10) + continuation)}"

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
        tissue_params = self._normalise_tissue_params(tissue_params)
        sequence_params = self._normalise_sequence_params(sequence_params)
        simulation_params = self._normalise_simulation_params(simulation_params)

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
        cells.append(
            new_code_cell(
                self._generate_canonical_metadata_code(
                    sequence_params, simulation_params
                )
            )
        )

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
        tissue_params = self._normalise_tissue_params(tissue_params)
        sequence_params = self._normalise_sequence_params(sequence_params)
        simulation_params = self._normalise_simulation_params(simulation_params)

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

        # Cell 4b: Verify the exact RF and gradient arrays used downstream.
        cells.append(new_markdown_cell("## Sequence Waveforms"))
        cells.append(new_code_cell(self._generate_sequence_visualization_code()))

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

    def _generate_canonical_metadata_code(
        self, sequence_params: Dict, simulation_params: Dict
    ) -> str:
        """Use clear metadata in analysis cells while retaining the file values."""

        sequence_summary = self._summarise_arrays(sequence_params)
        simulation_summary = self._summarise_arrays(simulation_params)
        sequence_assignment = self._assignment_source(
            "data['sequence_params']",
            pformat(sequence_summary, sort_dicts=False, width=88),
        )
        simulation_assignment = self._assignment_source(
            "data['simulation_params']",
            pformat(simulation_summary, sort_dicts=False, width=88),
        )
        return (
            "# Use one canonical, unit-explicit parameter schema in this notebook.\n"
            "# The values read verbatim from the HDF5 file remain available under\n"
            "# *_params_file for auditing or compatibility with older exports.\n"
            "data['sequence_params_file'] = data.get('sequence_params', {})\n"
            "data['simulation_params_file'] = data.get('simulation_params', {})\n"
            f"{sequence_assignment}\n"
            f"{simulation_assignment}\n"
        )

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
        return """# Display the canonical, unit-explicit simulation parameters
from pprint import pprint

print("="*60)
print("SIMULATION PARAMETERS")
print("="*60)

print("\\nTissue:")
for key, value in data['tissue'].items():
    if key in ['t1', 't2', 't2_star'] and value is not None:
        print(f"  {key}_s: {value:.9g}")
    elif value is not None:
        print(f"  {key}: {value}")

print("\\nSequence (unit suffixes are authoritative):")
pprint(data['sequence_params'], sort_dicts=False)

print("\\nSimulation (unit suffixes are authoritative):")
pprint(data['simulation_params'], sort_dicts=False)

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
        """Generate one canonical parameter cell with explicit units."""

        arrays = {}
        tissue_source = self._parameter_source(tissue_params, arrays, ("tissue",))
        sequence_source = self._parameter_source(sequence_params, arrays, ("sequence",))
        simulation_source = self._parameter_source(
            simulation_params, arrays, ("simulation",)
        )

        code = (
            "# Canonical simulation parameters\n"
            "# Unit suffixes (_s, _ms, _us, _hz, _deg, _g, _m) are authoritative.\n"
            "# When sequence_definition_source is exact_exported_waveforms, the stored\n"
            "# B1/gradient/time arrays are authoritative; otherwise the editable pulse\n"
            "# and timing parameters below regenerate the sequence.\n\n"
        )

        if arrays:
            if not waveform_filename:
                raise ValueError(
                    "Array-valued parameters require a waveform_filename for a "
                    "reproducible notebook export."
                )
            np.savez(waveform_filename, **arrays)
            rel_path = Path(waveform_filename).name
            code += (
                "# Load array-valued inputs from the companion archive.\n"
                f"array_file = Path({rel_path!r})\n"
                "if not array_file.exists():\n"
                "    raise FileNotFoundError(\n"
                "        f'Missing companion array file: {array_file}'\n"
                "    )\n"
                "with np.load(array_file, allow_pickle=False) as array_data:\n"
                "    loaded_arrays = {key: array_data[key] for key in array_data.files}\n\n"
            )
        else:
            code += "loaded_arrays = {}\n\n"

        code += f"{self._assignment_source('tissue_params', tissue_source)}\n\n"
        code += f"{self._assignment_source('sequence_params', sequence_source)}\n\n"
        code += f"{self._assignment_source('simulation_params', simulation_source)}\n"
        return code

    def _generate_simulator_init_code(
        self, tissue_params: Dict, simulation_params: Dict
    ) -> str:
        """Generate simulator initialization code."""
        return """# Create simulator
use_parallel = bool(simulation_params.get('use_parallel', False))
num_threads = int(simulation_params.get('num_threads', 4))

sim = BlochSimulator(use_parallel=use_parallel, num_threads=num_threads)

# Create tissue
tissue = TissueParameters(
    name=tissue_params['name'],
    t1=tissue_params['t1_s'],
    t2=tissue_params['t2_s'],
    t2_star=tissue_params.get('t2_star_s'),
    density=tissue_params['density'],
)

print(f"Simulator initialized")
print(f"  Tissue: {tissue.name}")
print(f"  T1: {tissue.t1*1000:.1f} ms, T2: {tissue.t2*1000:.1f} ms")
"""

    def _generate_sequence_definition_code(
        self, sequence_params: Dict, rf_waveform: Optional[Tuple] = None
    ) -> str:
        """Generate pulse sequence definition code."""
        seq_type = sequence_params.get("sequence_type", "Spin Echo")

        if "Free Induction Decay" in seq_type:
            return """# Create Free Induction Decay (FID) sequence
# Analytic pulses are regenerated from the editable canonical RF parameters.
rf_params = sequence_params.get('sequence_role_pulses', {}).get('Excitation')
if rf_params is None:
    rf_params = sequence_params.get('rf_designer_snapshot', {})

rf_type_aliases = {
    'rectangle': 'rect',
    'adiabatic half passage': 'adiabatic_half',
    'adiabatic full passage': 'adiabatic_full',
    'bir-4': 'bir4',
}
rf_type = str(rf_params.get('pulse_type', 'gaussian')).lower()
rf_design_type = rf_type_aliases.get(rf_type, rf_type)
rf_duration_s = float(rf_params.get('duration_s', 1e-3))
rf_flip_angle_deg = float(rf_params.get('flip_angle_deg', 90.0))
rf_b1_amplitude_g = float(rf_params.get('b1_amplitude_g', 0.0))
rf_phase_deg = float(rf_params.get('phase_deg', 0.0))
rf_frequency_offset_hz = float(rf_params.get('frequency_offset_hz', 0.0))
rf_sinc_lobes = int(rf_params.get('sinc_lobes', 3))
rf_slr_sharpness = int(rf_params.get('slr_sharpness', 1))
rf_apodization = rf_params.get('apodization', 'None')

dt = float(simulation_params.get('time_step_us', 1.0)) * 1e-6
if rf_design_type == 'custom':
    b1 = np.asarray(sequence_params.get('b1_waveform_g'), dtype=complex).copy()
    time = np.asarray(sequence_params.get('time_waveform_s'), dtype=float).copy()
    gradients_value = sequence_params.get('gradients_waveform_g_per_cm')
    gradients = (
        np.zeros((len(b1), 3), dtype=float)
        if gradients_value is None
        else np.asarray(gradients_value, dtype=float).copy()
    )
    if b1.ndim != 1 or time.ndim != 1 or b1.shape != time.shape:
        raise ValueError('The exported custom RF waveform is missing or invalid.')
else:
    rf_points = max(32, int(np.ceil(rf_duration_s / dt)))
    if rf_design_type == 'sinc':
        shape_parameter = float(max(1, rf_sinc_lobes) + 1)
    elif rf_design_type in {'adiabatic_half', 'adiabatic_full', 'bir4'}:
        shape_parameter = 4.0
    else:
        shape_parameter = float(rf_params.get('time_bandwidth_product', 4.0))
    if rf_design_type in {'adiabatic_half', 'adiabatic_full'} and rf_b1_amplitude_g <= 0:
        raise ValueError('AHP/AFP require b1_amplitude_g > 0; flip angle is not used.')
    design_flip_angle = (
        180.0 if rf_design_type == 'adiabatic_full' else
        90.0 if rf_design_type == 'adiabatic_half' else
        rf_flip_angle_deg
    )
    pulse, _ = design_rf_pulse(
        rf_design_type,
        duration=rf_duration_s,
        flip_angle=design_flip_angle,
        time_bw_product=shape_parameter,
        npoints=rf_points,
        freq_offset=0.0,
        slr_sharpness=rf_slr_sharpness,
    )
    pulse_dt = rf_duration_s / len(pulse)
    windows = {
        'Hamming': np.hamming,
        'Hanning': np.hanning,
        'Blackman': np.blackman,
    }
    if rf_design_type == 'sinc' and rf_apodization in windows and len(pulse) > 1:
        pulse = pulse * windows[rf_apodization](len(pulse))

    if rf_b1_amplitude_g > 0:
        peak = np.max(np.abs(pulse))
        if peak == 0:
            raise ValueError('RF pulse has zero amplitude and cannot be rescaled.')
        pulse = pulse * (rf_b1_amplitude_g / peak)
    elif rf_design_type not in {'adiabatic_half', 'adiabatic_full'}:
        target_area = np.deg2rad(rf_flip_angle_deg) / (2 * np.pi * 4258.0)
        area = np.sum(pulse) * pulse_dt
        if abs(area) < 1e-15:
            raise ValueError('RF pulse integral is too small for flip-angle scaling.')
        pulse = pulse * target_area / area

    total_duration_s = max(rf_duration_s, float(sequence_params.get('tr_s', 0.01)))
    total_duration_s += max(
        0.0, float(simulation_params.get('extra_tail_ms', 0.0))
    ) * 1e-3
    total_points = max(len(pulse), int(np.ceil(total_duration_s / pulse_dt)))
    b1 = np.pad(pulse, (0, total_points - len(pulse)))
    time = (np.arange(total_points, dtype=float) + 0.5) * pulse_dt
    gradients = np.zeros((total_points, 3), dtype=float)

    b1 *= np.exp(1j * np.deg2rad(rf_phase_deg))
    if rf_frequency_offset_hz != 0:
        b1 *= np.exp(2j * np.pi * rf_frequency_offset_hz * time)

sequence = (b1, gradients, time)
print(
    f'FID sequence created: {len(time)} points, {time[-1] * 1e3:.3f} ms, '
    f'requested flip={rf_flip_angle_deg:g}°'
)
"""

        # Use full waveforms if available (preferred for accuracy and complex sequences)
        if "b1_waveform_g" in sequence_params and "time_waveform_s" in sequence_params:
            return """# Use the full simulated waveforms exported from the GUI
b1 = sequence_params.get('b1_waveform_g')
time = sequence_params.get('time_waveform_s')
gradients = sequence_params.get('gradients_waveform_g_per_cm')

if b1 is None or time is None:
    raise ValueError(
        "B1 or time waveform missing. Ensure the companion NPZ file is present."
    )

if gradients is None:
    gradients = np.zeros((len(b1), 3))

sequence = (b1, gradients, time)
print(f"Sequence created from exact exported waveforms ({len(b1)} points)")
"""

        if "Spin Echo" in seq_type and "Tip" not in seq_type:
            return """# Create Spin Echo sequence from canonical SI parameters
sequence = SpinEcho(
    te=sequence_params['te_s'],
    tr=sequence_params['tr_s'],
)
print(
    f"Spin Echo sequence: TE={sequence_params['te_s']*1000:.1f} ms, "
    f"TR={sequence_params['tr_s']*1000:.1f} ms"
)
"""
        elif "Spin Echo" in seq_type and "Tip" in seq_type:
            return """# Create tip-axis Spin Echo sequence from canonical SI parameters
sequence = SpinEchoTipAxis(
    te=sequence_params['te_s'],
    tr=sequence_params['tr_s'],
)
print(
    f"Tip-axis Spin Echo: TE={sequence_params['te_s']*1000:.1f} ms, "
    f"TR={sequence_params['tr_s']*1000:.1f} ms"
)
"""
        elif "Gradient Echo" in seq_type:
            return """# Create Gradient Echo sequence from canonical SI parameters
sequence = GradientEcho(
    te=sequence_params['te_s'],
    tr=sequence_params['tr_s'],
    flip_angle=sequence_params.get('flip_angle_deg', 90.0),
)
print(
    f"Gradient Echo: TE={sequence_params['te_s']*1000:.1f} ms, "
    f"TR={sequence_params['tr_s']*1000:.1f} ms, "
    f"FA={sequence_params.get('flip_angle_deg', 90.0):.1f}°"
)
"""
        elif "Slice Select" in seq_type:
            snapshot = sequence_params.get("rf_designer_snapshot", {})
            dur = snapshot.get("duration_s", sequence_params.get("rf_duration_s", 3e-3))
            return f"""# Create Slice Select + Rephase sequence
sequence = SliceSelectRephase(
    flip_angle=sequence_params.get('flip_angle_deg', 90.0),
    pulse_duration={dur:.6f}
)
print(
    f"Slice Select + Rephase: "
    f"FA={{sequence_params.get('flip_angle_deg', 90.0):.1f}}°"
)
"""
        elif "SSFP" in seq_type:
            return """# Create SSFP sequence
# Simplified implementation for notebook
# Note: For full SSFP features, consider exporting HDF5 data instead
dt = simulation_params['time_step_us'] * 1e-6
tr = sequence_params.get('tr_s', 0.01)
n_reps = int(sequence_params.get('ssfp_repeats', 10))
rf_snapshot = sequence_params.get('rf_designer_snapshot', {})
flip = rf_snapshot.get('flip_angle_deg', sequence_params.get('flip_angle_deg', 30.0))

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
print(f"SSFP sequence: TR={tr*1000:.1f}ms, FA={flip}°, {n_reps} reps")
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

    def _generate_sequence_visualization_code(self) -> str:
        """Generate a compact RF/gradient plot of the simulated sequence."""
        return """# Plot the exact arrays that will be passed to the simulator
if isinstance(sequence, tuple):
    sequence_b1, sequence_gradients, sequence_time = sequence
else:
    sequence_b1, sequence_gradients, sequence_time = sequence.compile(
        dt=simulation_params.get('time_step_us', 1.0) * 1e-6
    )

sequence_b1 = np.asarray(sequence_b1, dtype=complex)
sequence_gradients = np.asarray(sequence_gradients, dtype=float)
sequence_time_ms = np.asarray(sequence_time, dtype=float) * 1e3
if sequence_gradients.ndim == 1:
    sequence_gradients = sequence_gradients[:, None]
if sequence_gradients.shape[1] < 3:
    sequence_gradients = np.pad(
        sequence_gradients,
        ((0, 0), (0, 3 - sequence_gradients.shape[1])),
    )

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axes[0].plot(sequence_time_ms, np.abs(sequence_b1), label='|B1|', linewidth=1.5)
axes[0].plot(sequence_time_ms, np.real(sequence_b1), label='Re(B1)', alpha=0.8)
axes[0].plot(sequence_time_ms, np.imag(sequence_b1), label='Im(B1)', alpha=0.8)
axes[0].set_ylabel('RF (G)')
axes[0].legend(loc='upper right', ncol=3)

gradient_labels = ('Gx', 'Gy', 'Gz')
gradient_colors = ('tab:red', 'tab:green', 'tab:blue')
for axis, values, label, color in zip(
    axes[1:], sequence_gradients.T[:3], gradient_labels, gradient_colors
):
    axis.plot(sequence_time_ms, values, color=color, linewidth=1.2)
    axis.set_ylabel(f'{label}\\n(G/cm)')

axes[-1].set_xlabel('Time (ms)')
for axis in axes:
    axis.grid(True, alpha=0.3)
fig.suptitle(f"Simulated sequence: {sequence_params['sequence_type']}")
fig.tight_layout()
plt.show()
"""

    def _generate_sampling_code(self, simulation_params: Dict) -> str:
        """Generate position/frequency sampling code."""
        return """# Use the exact sampled axes when they were captured with the run.
if 'position_axis_m' in simulation_params:
    positions = np.asarray(simulation_params['position_axis_m'], dtype=float)
else:
    num_positions = int(simulation_params.get('num_positions', 1))
    if 'position_range_mm' in simulation_params:
        position_range_m = simulation_params['position_range_mm'] / 1000.0
    else:
        position_range_m = simulation_params.get('position_range_cm', 0.0) / 100.0
    positions = np.zeros((num_positions, 3))
    if num_positions > 1:
        positions[:, 2] = np.linspace(
            -position_range_m / 2.0,
            position_range_m / 2.0,
            num_positions,
        )

if 'frequency_axis_hz' in simulation_params:
    frequencies = np.asarray(simulation_params['frequency_axis_hz'], dtype=float)
else:
    num_frequencies = int(simulation_params.get('num_frequencies', 1))
    frequency_center_hz = simulation_params.get('frequency_center_hz', 0.0)
    frequency_range_hz = simulation_params.get('frequency_range_hz', 0.0)
    if num_frequencies > 1:
        frequencies = np.linspace(
            frequency_center_hz - frequency_range_hz / 2.0,
            frequency_center_hz + frequency_range_hz / 2.0,
            num_frequencies,
        )
    else:
        frequencies = np.array([frequency_center_hz])

print(f"Sampling:")
print(f"  Positions: {len(positions)}")
print(f"  Frequencies: {len(frequencies)}")
"""

    def _generate_simulation_run_code(self, simulation_params: Dict) -> str:
        """Generate simulation execution code."""
        return """# Run simulation
print("\\nRunning simulation...")

mode = 2 if simulation_params.get('mode') == 'time-resolved' else 0
time_step_s = simulation_params.get('time_step_us', 1.0) * 1e-6

result = sim.simulate(
    sequence,
    tissue,
    positions=positions,
    frequencies=frequencies,
    initial_magnetization=simulation_params.get('initial_mz'),
    mode=mode,
    dt=time_step_s,
    rf_carrier_offset=simulation_params.get('rf_carrier_offset_hz', 0.0),
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
    'simulation_params': simulation_params,
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
        if waveform_filename is None:
            notebook_path = Path(filename)
            waveform_filename = str(
                notebook_path.with_name(f"{notebook_path.stem}_arrays.npz")
            )
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


def _sequence_result_access_code() -> str:
    """Return clear, stable aliases for data used in result notebooks."""
    return dedent(
        """
        # Stable, descriptive entry points for subsequent analysis.
        raw_adc_signal = result_dataset['signal']
        try:
            build_cartesian_kspace_from_raw = _cartesian_from_adc
            reconstruct_cartesian_image = _cartesian_ifft
        except NameError:
            build_cartesian_kspace_from_raw = None
            reconstruct_cartesian_image = None
        reconstructed_kspace = None
        reconstructed_image = None
        reconstructed_image_magnitude = None
        reconstructed_spectrum = None
        reconstruction_kind = 'raw_adc_only'
        reconstruction_source_names = {}

        if 'radial_3d_image' in result_dataset:
            reconstruction_kind = 'radial_3d'
            reconstruction_source_names = {
                'kspace': 'radial_3d_gridded_kspace',
                'image': 'radial_3d_image',
                'magnitude': 'radial_3d_image_magnitude',
            }
        elif (
            'notebook_cartesian_3d_image' in result_dataset
            or 'cartesian_3d_image' in result_dataset
        ):
            reconstruction_kind = 'cartesian_3d'
            reconstruction_source_names = {
                'kspace': 'cartesian_3d_kspace',
                'image': (
                    'notebook_cartesian_3d_image'
                    if 'notebook_cartesian_3d_image' in result_dataset
                    else 'cartesian_3d_image'
                ),
                'magnitude': (
                    'notebook_cartesian_3d_image_magnitude'
                    if 'notebook_cartesian_3d_image_magnitude' in result_dataset
                    else 'cartesian_3d_image_magnitude'
                ),
            }
        elif (
            'notebook_cartesian_image' in result_dataset
            or 'cartesian_image' in result_dataset
        ):
            reconstruction_kind = 'cartesian_2d'
            reconstruction_source_names = {
                'kspace': 'cartesian_kspace',
                'image': (
                    'notebook_cartesian_image'
                    if 'notebook_cartesian_image' in result_dataset
                    else 'cartesian_image'
                ),
                'magnitude': (
                    'notebook_cartesian_image_magnitude'
                    if 'notebook_cartesian_image_magnitude' in result_dataset
                    else 'cartesian_image_magnitude'
                ),
            }
        elif 'spiral_image' in result_dataset:
            reconstruction_kind = 'spiral_2d'
            reconstruction_source_names = {
                'kspace': 'spiral_gridded_kspace',
                'image': 'spiral_image',
                'magnitude': 'spiral_image_magnitude',
            }
        elif 'csi_spectrum' in result_dataset:
            reconstruction_kind = 'csi'
            reconstruction_source_names = {
                'kspace': 'csi_kspace',
                'image': 'csi_spatial_fid',
                'spectrum': 'csi_spectrum',
            }

        kspace_name = reconstruction_source_names.get('kspace')
        image_name = reconstruction_source_names.get('image')
        magnitude_name = reconstruction_source_names.get('magnitude')
        spectrum_name = reconstruction_source_names.get('spectrum')
        if kspace_name in result_dataset:
            reconstructed_kspace = result_dataset[kspace_name]
        if image_name in result_dataset:
            reconstructed_image = result_dataset[image_name]
        if magnitude_name in result_dataset:
            reconstructed_image_magnitude = result_dataset[magnitude_name]
        elif reconstructed_image is not None:
            reconstructed_image_magnitude = np.abs(reconstructed_image)
        if spectrum_name in result_dataset:
            reconstructed_spectrum = result_dataset[spectrum_name]

        # One obvious default for users who simply want to work with the result.
        # Images use magnitude data; CSI uses the complex reconstructed spectrum.
        reconstructed_data = (
            reconstructed_spectrum
            if reconstructed_spectrum is not None
            else reconstructed_image_magnitude
        )
        reconstruction = {
            'kind': reconstruction_kind,
            'raw_adc_signal': raw_adc_signal,
            'kspace': reconstructed_kspace,
            'image': reconstructed_image,
            'image_magnitude': reconstructed_image_magnitude,
            'spectrum': reconstructed_spectrum,
            'data': reconstructed_data,
            'dataset_variable_names': reconstruction_source_names,
        }

        print(f'Reconstruction type: {reconstruction_kind}')
        print("Use `reconstructed_data` for the primary reconstructed result.")
        print("Use `raw_adc_signal` for chronological raw ADC samples.")
        print("All data and metadata remain available in `result_dataset` (`ds` is an alias).")
        if reconstructed_data is not None:
            print(
                'reconstructed_data:',
                reconstructed_data.dims,
                reconstructed_data.shape,
                reconstructed_data.dtype,
            )
        else:
            print('No image or spectrum reconstruction is available for this result.')
        """
    ).strip()


def _sequence_result_raw_reconstruction_example_code() -> str:
    """Return a compact raw-ADC Cartesian reconstruction example."""
    return dedent(
        """
        # Complete Cartesian example: chronological raw ADC -> sorted k-space -> image.
        # The helper groups samples by ADC event and labels, sorts the logical
        # partition/phase/read coordinates, and validates that the grid is complete.
        if reconstruction_kind.startswith('cartesian'):
            try:
                example_kspace_from_raw = build_cartesian_kspace_from_raw(
                    result_dataset,
                    signal_name='signal',
                )
            except ValueError as exc:
                print(f'Raw Cartesian example unavailable: {exc}')
            else:
                example_spatial_dims = _cartesian_spatial_dims(
                    example_kspace_from_raw
                )
                example_image_from_raw = reconstruct_cartesian_image(
                    example_kspace_from_raw,
                    example_spatial_dims,
                )
                example_image_magnitude_from_raw = np.abs(
                    example_image_from_raw
                )
                print('Raw ADC samples:', raw_adc_signal.shape)
                print(
                    'Sorted k-space:',
                    example_kspace_from_raw.dims,
                    example_kspace_from_raw.shape,
                )
                print(
                    'Reconstructed image:',
                    example_image_from_raw.dims,
                    example_image_from_raw.shape,
                )
        else:
            print(
                'This explicit raw-data example applies to Cartesian acquisitions. '
                'For this result, use `reconstruction` to inspect the exported '
                'gridded k-space and reconstructed data.'
            )
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


def _sequence_result_cartesian_point_trace_code() -> str:
    """Return reusable Cartesian frame and pixel-trace helpers."""
    return dedent(
        r'''
        def dimension_cartesian_frames(dataset, variable='cartesian_image'):
            """Expose flat Cartesian frames as named xarray dimensions."""
            data = dataset[variable]
            if 'cartesian_frame' not in data.dims:
                return data

            frame_axes = []
            frame_values = {}
            for axis in ('slice', 'echo', 'repetition', 'segment', 'partition'):
                coordinate = f'cartesian_frame_{axis}_index'
                if coordinate not in dataset.coords:
                    continue
                values = np.asarray(dataset.coords[coordinate])
                if np.unique(values).size > 1:
                    frame_axes.append(axis)
                    frame_values[axis] = values
            if not frame_axes:
                frame_axes = ['frame']
                frame_values['frame'] = np.arange(data.sizes['cartesian_frame'])

            storage_coordinates = [
                name
                for name in data.coords
                if name.startswith('cartesian_frame_')
            ]
            compact = data.reset_coords(storage_coordinates, drop=True)
            compact = compact.assign_coords(
                {
                    axis: ('cartesian_frame', frame_values[axis])
                    for axis in frame_axes
                }
            )
            if len(frame_axes) == 1:
                result = compact.swap_dims({'cartesian_frame': frame_axes[0]})
                result = result.reset_coords('cartesian_frame', drop=True)
            else:
                result = compact.set_index(
                    cartesian_frame=frame_axes
                ).unstack('cartesian_frame')
            remaining = [dim for dim in result.dims if dim not in frame_axes]
            result = result.transpose(*frame_axes, *remaining)

            echo_times_by_index = _echo_times_by_index(dataset)
            if 'echo' in frame_axes and echo_times_by_index:
                echo_times = [echo_times_by_index[value] for value in result.echo.values]
                result = result.assign_coords(
                    echo_time_s=('echo', np.asarray(echo_times))
                )
                result.echo_time_s.attrs['units'] = 's'
            return result


        def _echo_times_by_index(dataset):
            """Read echo times from new coordinates or older result attributes."""
            coordinate = 'cartesian_frame_echo_index'
            time_coordinate = 'cartesian_frame_echo_time_s'
            if coordinate in dataset.coords and time_coordinate in dataset.coords:
                labels = np.asarray(dataset.coords[coordinate])
                times = np.asarray(dataset.coords[time_coordinate], dtype=float)
                return {
                    value.item() if hasattr(value, 'item') else value: float(
                        times[np.flatnonzero(labels == value)[0]]
                    )
                    for value in np.unique(labels)
                }

            stored = dataset.attrs.get('echo_times_s')
            if stored is None:
                return {}
            if isinstance(stored, str):
                times = np.fromstring(stored, sep=',')
            else:
                times = np.asarray(stored, dtype=float).reshape(-1)
            return {index: float(value) for index, value in enumerate(times)}


        def cartesian_echo_spectrum(signal, echo_times_s):
            """FFT an echo signal and return its physical frequency axis in Hz."""
            values = np.asarray(signal)
            times = np.asarray(echo_times_s, dtype=float)
            if values.ndim != 1 or times.ndim != 1 or values.size != times.size:
                raise ValueError('signal and echo_times_s must be matching 1D arrays')
            if values.size < 2:
                raise ValueError('at least two echoes are required for an FFT')
            intervals = np.diff(times)
            spacing_s = float(np.median(intervals))
            if spacing_s <= 0 or not np.allclose(
                intervals, spacing_s, rtol=1e-4, atol=1e-12
            ):
                raise ValueError(
                    'echo times must be increasing and uniformly spaced for an FFT'
                )
            frequency_hz = np.fft.fftshift(
                np.fft.fftfreq(values.size, d=spacing_s)
            )
            spectrum = np.fft.fftshift(np.fft.fft(values))
            return frequency_hz, spectrum


        def cartesian_point_trace(dataset, x, y, *, over='echo', **fixed):
            """Return one reconstructed image pixel over echo/repetition/etc.

            ``x`` and ``y`` are reconstruction pixel indices, not indices on
            the generally finer phantom simulation grid. ``fixed`` selects
            the other frame labels, for example ``repetition=0``.
            """
            image = dimension_cartesian_frames(dataset, 'cartesian_image')
            read_dim = next(
                dim for dim in image.dims if dim.startswith('read_')
            )
            phase_dim = next(
                dim for dim in image.dims if dim.startswith('phase_')
            )
            if over not in image.dims:
                raise ValueError(
                    f'{over!r} is not an available image dimension; '
                    f'choose from {image.dims!r}'
                )
            point = image.isel({read_dim: int(x), phase_dim: int(y)})
            selectors = {
                dim: fixed.get(dim, point.coords[dim].values[0])
                for dim in point.dims
                if dim != over and dim in point.coords
            }
            if selectors:
                point = point.sel(selectors)
            if over == 'echo' and 'echo_time_s' in point.coords:
                axis = 1e3 * np.asarray(point.echo_time_s)
                axis_label = 'Echo time (ms)'
            else:
                axis = np.asarray(point.coords[over])
                axis_label = over.replace('_', ' ').title()
            return point, axis, axis_label
        '''
    ).strip()


def _sequence_result_cartesian_point_example_code() -> str:
    """Return the short, editable Cartesian pixel-trace example."""
    return dedent(
        r"""
        # User settings: reconstruction pixel and dimension to plot.
        mxy_image = dimension_cartesian_frames(ds, 'cartesian_image')
        read_dim = next(dim for dim in mxy_image.dims if dim.startswith('read_'))
        phase_dim = next(dim for dim in mxy_image.dims if dim.startswith('phase_'))
        x = mxy_image.sizes[read_dim] // 2
        y = mxy_image.sizes[phase_dim] // 2

        # EPSI normally varies over echo; repeated scans also expose repetition.
        available_trace_axes = [
            dim
            for dim in mxy_image.dims
            if dim not in {read_dim, phase_dim, 'coil'}
            and mxy_image.sizes[dim] > 1
        ]

        if available_trace_axes:
            trace_axis = (
                'echo' if 'echo' in available_trace_axes
                else 'repetition' if 'repetition' in available_trace_axes
                else available_trace_axes[0]
            )
            point_signal, trace_values, trace_label = cartesian_point_trace(
                ds, x, y, over=trace_axis
            )
            # Repetition example at a fixed echo:
            # point_signal, trace_values, trace_label = cartesian_point_trace(
            #     ds, x, y, over='repetition', echo=0
            # )
            # EPSI spectrum of this pixel (trace_values is in ms for echo):
            # frequency_hz, point_spectrum = cartesian_echo_spectrum(
            #     point_signal, 1e-3 * trace_values
            # )

            first_image = mxy_image
            for dim in first_image.dims:
                if dim not in {read_dim, phase_dim, 'coil'}:
                    first_image = first_image.isel({dim: 0})
            if 'coil' in first_image.dims:
                first_image = np.sqrt((np.abs(first_image) ** 2).sum('coil'))

            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            axes[0].imshow(np.abs(first_image), origin='lower', cmap='gray')
            axes[0].axvline(x, color='cyan', linewidth=0.8)
            axes[0].axhline(y, color='cyan', linewidth=0.8)
            axes[0].set(
                title='Reconstruction and selected image pixel',
                xlabel=read_dim,
                ylabel=phase_dim,
            )
            axes[1].plot(trace_values, np.abs(point_signal), 'o-', label='Magnitude')
            if np.iscomplexobj(point_signal):
                axes[1].plot(trace_values, point_signal.real, alpha=0.65, label='Real')
                axes[1].plot(trace_values, point_signal.imag, alpha=0.65, label='Imaginary')
            axes[1].set(
                title=f'Pixel ({read_dim}={x}, {phase_dim}={y})',
                xlabel=trace_label,
                ylabel='Reconstructed signal (a.u.)',
            )
            axes[1].grid(True)
            axes[1].legend()
            plt.show()
        else:
            point_signal = mxy_image.isel({read_dim: x, phase_dim: y})
            print('This result contains one reconstructed Cartesian image.')
        """
    ).strip()


def _sequence_result_checkpoint_code() -> str:
    """Return the optional true phantom-grid Mxy checkpoint example."""
    return dedent(
        r"""
        if 'checkpoint_magnetization' not in ds:
            print(
                'No voxel-resolved magnetization checkpoints were stored for '
                'this run. The reconstructed image above is the spatially '
                'resolved measured signal. Configure checkpoint times when you '
                'need the true Mx+iMy evolution on the phantom simulation grid.'
            )
        else:
            checkpoint_mxy = (
                ds.checkpoint_magnetization.sel(component='mx')
                + 1j * ds.checkpoint_magnetization.sel(component='my')
            )
            spatial_dims = [
                dim for dim in checkpoint_mxy.dims if dim != 'checkpoint'
            ]
            centre = {
                dim: checkpoint_mxy.sizes[dim] // 2 for dim in spatial_dims
            }
            phantom_point_mxy = checkpoint_mxy.isel(centre)
            plt.figure(figsize=(8, 4))
            plt.plot(
                1e3 * ds.checkpoint,
                np.abs(phantom_point_mxy),
                'o-',
                label=r'$|M_{xy}|$',
            )
            plt.xlabel('Checkpoint time (ms)')
            plt.ylabel('Magnetization (simulation units)')
            plt.grid(True)
            plt.legend()
            plt.show()
        """
    ).strip()


def _sequence_result_cartesian_2d_explorer_code() -> str:
    """Return a compact frame-aware explorer for Cartesian 2D/EPSI data."""
    return dedent(
        r"""
        try:
            import ipywidgets as widgets
            from IPython.display import clear_output, display
        except ImportError as exc:
            raise ImportError(
                'The interactive result explorer requires ipywidgets. '
                'Install it with `%pip install ipywidgets`.'
            ) from exc

        image_data = ds['cartesian_image']
        kspace_data = ds['cartesian_kspace']
        echo_times_by_index = _echo_times_by_index(ds)
        x_dim = next(dim for dim in image_data.dims if dim.startswith('read_'))
        y_dim = next(dim for dim in image_data.dims if dim.startswith('phase_'))
        frame_prefix = 'cartesian_frame_'

        outer_axes = {}
        if 'cartesian_frame' in image_data.dims:
            for axis in ('slice', 'echo', 'repetition', 'segment', 'partition'):
                coordinate = f'{frame_prefix}{axis}_index'
                if coordinate not in ds.coords:
                    continue
                values = np.unique(np.asarray(ds.coords[coordinate]))
                if values.size > 1:
                    outer_axes[axis] = {
                        'coordinate': coordinate,
                        'values': values,
                    }
            if not outer_axes and image_data.sizes['cartesian_frame'] > 1:
                outer_axes['frame'] = {
                    'coordinate': None,
                    'values': np.arange(image_data.sizes['cartesian_frame']),
                }
        else:
            for dim in image_data.dims:
                if dim in {x_dim, y_dim, 'coil'} or image_data.sizes[dim] <= 1:
                    continue
                values = (
                    np.asarray(image_data.coords[dim])
                    if dim in image_data.coords
                    else np.arange(image_data.sizes[dim])
                )
                outer_axes[dim] = {'coordinate': dim, 'values': values}


        def _plain(value):
            return value.item() if hasattr(value, 'item') else value


        def _axis_title(axis):
            return {
                'echo': 'Echo',
                'repetition': 'Acquisition repetition',
                'slice': 'Slice',
                'segment': 'Segment',
                'partition': 'Partition',
                'frame': 'Frame',
            }.get(axis, axis.replace('_', ' ').title())


        def _option_label(axis, value):
            if axis == 'echo' and _plain(value) in echo_times_by_index:
                time_ms = 1e3 * echo_times_by_index[_plain(value)]
                return f'{_plain(value)} ({time_ms:.4g} ms)'
            return str(_plain(value))


        def _select_outer(data, selections):
            if 'cartesian_frame' in data.dims:
                candidates = np.arange(data.sizes['cartesian_frame'])
                for axis, details in outer_axes.items():
                    value = selections.get(axis, details['values'][0])
                    if axis == 'frame':
                        candidates = candidates[candidates == int(value)]
                    else:
                        coordinate = np.asarray(ds.coords[details['coordinate']])
                        candidates = candidates[coordinate[candidates] == value]
                if candidates.size != 1:
                    raise ValueError(
                        'Outer selections do not identify exactly one image frame'
                    )
                return data.isel(cartesian_frame=int(candidates[0]))
            selectors = {
                axis: selections.get(axis, details['values'][0])
                for axis, details in outer_axes.items()
                if axis in data.dims
            }
            return data.sel(selectors) if selectors else data


        def _rss_magnitude(data):
            if 'coil' in data.dims:
                return np.sqrt((np.abs(data) ** 2).sum('coil'))
            return np.abs(data)


        def _trace(data, x, y, over, selections):
            if not over:
                selected = _select_outer(data, selections)
                return np.asarray([_plain(selected.isel({x_dim: x, y_dim: y}))]), [0]
            values = outer_axes[over]['values']
            signal = []
            horizontal = []
            for value in values:
                current = dict(selections)
                current[over] = value
                selected = _select_outer(data, current)
                if 'coil' in selected.dims:
                    selected = np.sqrt((np.abs(selected) ** 2).sum('coil'))
                signal.append(_plain(selected.isel({x_dim: x, y_dim: y})))
                if over == 'echo' and _plain(value) in echo_times_by_index:
                    horizontal.append(1e3 * echo_times_by_index[_plain(value)])
                else:
                    horizontal.append(_plain(value))
            return np.asarray(signal), np.asarray(horizontal)


        def _render(
            x,
            y,
            display_range,
            display_auto,
            kspace_scale,
            trace_axis,
            trace_view,
            **values,
        ):
            # Some notebook frontends do not honour interactive_output's deferred
            # clear reliably during rapid slider updates. Clear immediately so one
            # widget instance always owns exactly one figure.
            clear_output(wait=False)
            selections = {
                name[len('outer__'):]: value
                for name, value in values.items()
                if name.startswith('outer__')
            }
            image = _rss_magnitude(
                _select_outer(image_data, selections)
            ).transpose(y_dim, x_dim)
            kspace = _rss_magnitude(
                _select_outer(kspace_data, selections)
            ).transpose(y_dim, x_dim)
            image_values = np.asarray(image)
            kspace_values = np.asarray(kspace)
            if kspace_scale == 'log':
                kspace_values = np.log1p(kspace_values)
                kspace_title = 'log(1 + |k-space|)'
            else:
                kspace_title = '|k-space|'
            if display_auto:
                low, high = float(image_values.min()), float(image_values.max())
            else:
                low, high = map(float, display_range)
            if high <= low:
                high = np.nextafter(low, np.inf)

            signal, horizontal = _trace(
                image_data, x, y, trace_axis, selections
            )
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            axes[0].imshow(kspace_values, origin='lower', cmap='magma')
            axes[0].set(title=kspace_title, xlabel='k-read', ylabel='k-phase')
            axes[1].imshow(
                image_values,
                origin='lower',
                cmap='gray',
                vmin=low,
                vmax=high,
            )
            axes[1].axvline(x, color='cyan', linewidth=0.8)
            axes[1].axhline(y, color='cyan', linewidth=0.8)
            axes[1].set(title='Reconstruction', xlabel=x_dim, ylabel=y_dim)
            if trace_view == 'spectrum':
                if trace_axis != 'echo' or not echo_times_by_index:
                    axes[2].text(
                        0.5,
                        0.5,
                        'A frequency spectrum requires the echo axis\n'
                        'and physical echo times.',
                        ha='center',
                        va='center',
                        transform=axes[2].transAxes,
                    )
                    axes[2].set_axis_off()
                else:
                    try:
                        frequency_hz, spectrum = cartesian_echo_spectrum(
                            signal, 1e-3 * horizontal
                        )
                    except ValueError as exc:
                        axes[2].text(
                            0.5,
                            0.5,
                            str(exc),
                            ha='center',
                            va='center',
                            wrap=True,
                            transform=axes[2].transAxes,
                        )
                        axes[2].set_axis_off()
                    else:
                        spacing_ms = float(np.median(np.diff(horizontal)))
                        axes[2].plot(frequency_hz, np.abs(spectrum))
                        axes[2].axvline(0.0, color='0.6', linewidth=0.8)
                        axes[2].set(
                            title=f'Pixel spectrum (echo spacing {spacing_ms:.4g} ms)',
                            xlabel='Frequency offset (Hz)',
                            ylabel='FFT magnitude (a.u.)',
                        )
                        axes[2].grid(True)
            else:
                trace_label = (
                    'Echo time (ms)'
                    if trace_axis == 'echo' and echo_times_by_index
                    else _axis_title(trace_axis) if trace_axis else 'Image'
                )
                axes[2].plot(horizontal, np.abs(signal), 'o-', label='Magnitude')
                if np.iscomplexobj(signal):
                    axes[2].plot(horizontal, signal.real, alpha=0.65, label='Real')
                    axes[2].plot(
                        horizontal, signal.imag, alpha=0.65, label='Imaginary'
                    )
                axes[2].set(
                    title=f'Image pixel ({x}, {y})',
                    xlabel=trace_label,
                    ylabel='Reconstructed signal (a.u.)',
                )
                axes[2].grid(True)
                axes[2].legend()
            selected_text = ', '.join(
                f'{_axis_title(axis)}={_plain(value)}'
                for axis, value in selections.items()
            )
            fig.suptitle(selected_text or 'Single acquisition')
            display(fig)
            plt.close(fig)


        x_slider = widgets.IntSlider(
            value=image_data.sizes[x_dim] // 2,
            min=0,
            max=image_data.sizes[x_dim] - 1,
            description=x_dim,
            continuous_update=False,
        )
        y_slider = widgets.IntSlider(
            value=image_data.sizes[y_dim] // 2,
            min=0,
            max=image_data.sizes[y_dim] - 1,
            description=y_dim,
            continuous_update=False,
        )
        outer_controls = {
            f'outer__{axis}': widgets.SelectionSlider(
                options=[
                    (_option_label(axis, value), _plain(value))
                    for value in details['values']
                ],
                description=_axis_title(axis),
                continuous_update=False,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='320px'),
            )
            for axis, details in outer_axes.items()
        }
        trace_options = [
            (_axis_title(axis), axis) for axis in outer_axes
        ] or [('Single image', '')]
        trace_axis_dropdown = widgets.Dropdown(
            options=trace_options,
            value=(
                'echo' if 'echo' in outer_axes
                else 'repetition' if 'repetition' in outer_axes
                else trace_options[0][1]
            ),
            description='Plot pixel over',
            style={'description_width': 'initial'},
        )
        trace_view_dropdown = widgets.Dropdown(
            options=[('Signal', 'signal'), ('Spectrum (FFT)', 'spectrum')],
            value='signal',
            description='Pixel view',
            style={'description_width': 'initial'},
        )
        kspace_scale_dropdown = widgets.Dropdown(
            options=[('Magnitude', 'magnitude'), ('Log magnitude', 'log')],
            value='log',
            description='K-space view',
            style={'description_width': 'initial'},
        )
        data_max = float(_rss_magnitude(image_data).max(skipna=True))
        slider_max = data_max if np.isfinite(data_max) and data_max > 0 else 1.0
        display_range_slider = widgets.FloatRangeSlider(
            value=(0.0, slider_max),
            min=0.0,
            max=slider_max,
            step=slider_max / 1000.0,
            description='Image range',
            continuous_update=False,
            readout_format='.4g',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='520px'),
        )
        display_auto_checkbox = widgets.Checkbox(
            value=True,
            description='Auto image range',
        )
        controls = {
            'x': x_slider,
            'y': y_slider,
            'display_range': display_range_slider,
            'display_auto': display_auto_checkbox,
            'kspace_scale': kspace_scale_dropdown,
            'trace_axis': trace_axis_dropdown,
            'trace_view': trace_view_dropdown,
            **outer_controls,
        }
        output = widgets.interactive_output(_render, controls)
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
                        '<b>Cartesian 2D / EPSI:</b> frame controls use their '
                        'actual sequence meaning. The right plot follows the '
                        'selected image pixel over echo or acquisition repetition. '
                        'Spectrum (FFT) uses the physical echo spacing and shows '
                        'frequency offset in Hz. K-space view switches between '
                        'linear magnitude and log magnitude.'
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
    pulseq_definitions: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a notebook that regenerates one GUI-built Pulseq sequence."""
    if not HAS_NBFORMAT:
        raise ImportError("Jupyter notebook export requires nbformat")
    builders = {
        "epi": "make_pulseq_epi",
        "epsi_mge": "make_pulseq_epsi_mge",
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
            "sequence_kind must be 'epi', 'epsi_mge', 'spiral', 'csi', 'flash', "
            "'bssfp_3d', 'spectral_bssfp_3d', 'me_bssfp_3d', or "
            "'radial_me_bssfp_3d'"
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
    definitions_literal = pformat(
        dict(pulseq_definitions or {}), sort_dicts=False, width=88
    )
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
                f"pulseq_definitions = {definitions_literal}\n"
                "for name, value in pulseq_definitions.items():\n"
                "    sequence.set_definition(name, value)\n"
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


def _sequence_result_dataset_kind(data_path: Path) -> tuple[str, bool]:
    """Inspect a result file without loading its arrays into memory."""
    try:
        import xarray as xr

        with xr.open_dataset(data_path) as dataset:
            names = set(dataset.data_vars)
    except Exception:
        return "generic", False

    def present(name: str) -> bool:
        return name in names or f"{name}_real" in names

    if present("radial_3d_image"):
        return "radial_3d", True
    if present("cartesian_3d_image"):
        return "cartesian_3d", True
    if present("csi_spatial_fid"):
        return "csi", True
    if present("spiral_image"):
        return "spiral_2d", True
    if present("cartesian_image"):
        return "cartesian_2d", True
    return "raw_signal", False


def export_sequence_result_notebook(filename: str, data_filename: str) -> Path:
    """Create a concise, result-aware xarray analysis notebook."""
    if not HAS_NBFORMAT:
        raise ImportError("Jupyter notebook export requires nbformat")
    notebook_path = Path(filename)
    data_path = Path(data_filename)
    relative_data = os.path.relpath(
        data_path.resolve(), start=notebook_path.parent.resolve()
    )
    absolute_data = str(data_path.resolve())
    result_kind, has_reconstruction = _sequence_result_dataset_kind(data_path)

    cells = [
        new_markdown_cell(
            "# Sequence simulation result\n\n"
            f"Generated by BlochSimulator {__version__}. This notebook detected "
            f"`{result_kind}` data in `{relative_data}`. Start with the short "
            "examples below; storage details and raw ADC inspection are in the "
            "advanced section at the end."
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
            "result_dataset = xr.open_dataset(data_path)\n"
            "stored_ds = result_dataset\n"
            "ds = stored_ds.copy()\n"
            "storage_components = []\n"
            "for name in list(ds.data_vars):\n"
            "    if not name.endswith('_real'):\n"
            "        continue\n"
            "    base = name[:-5]\n"
            "    imag = f'{base}_imag'\n"
            "    if imag in ds:\n"
            "        complex_values = ds[name] + 1j * ds[imag]\n"
            "        complex_values.attrs = dict(ds[name].attrs)\n"
            "        ds[base] = complex_values\n"
            "        storage_components.extend((name, imag))\n"
            "# Keep the analysis view compact. The on-disk components remain in stored_ds.\n"
            "ds = ds.drop_vars(storage_components)\n"
            "print(f'Loaded {data_path}')\n"
            'print(f\'Sequence: {ds.attrs.get("sequence_source", "unknown")}\')\n'
            "for name in (\n"
            "    'signal', 'cartesian_image', 'cartesian_3d_image',\n"
            "    'spiral_image', 'radial_3d_image', 'csi_spatial_fid',\n"
            "    'checkpoint_magnetization', 'final_magnetization',\n"
            "):\n"
            "    if name in ds:\n"
            "        print(f'{name:28s} dims={ds[name].dims} shape={ds[name].shape}')"
        ),
        new_markdown_cell(
            "## Start here: what the important variables mean\n\n"
            "- `signal` is the complex receiver signal in chronological ADC order. "
            "It is summed over the phantom and is not voxel-resolved.\n"
            "- `cartesian_image` is the complex spatial reconstruction and is the "
            "usual starting point for a measured pixel signal.\n"
            "- `species_cartesian_image` contains the same reconstruction separated "
            "by the `pool` coordinate when pool-resolved simulation was enabled.\n"
            "- `checkpoint_magnetization`, when present, contains true voxel-grid "
            "$M_x$, $M_y$, and $M_z$ at explicitly configured times.\n"
            "- `final_magnetization` contains the voxel-grid state after the sequence.\n\n"
            "A reconstruction pixel and a phantom simulation voxel need not have the "
            "same size. Cartesian image-position coordinates ending in `_position_m` "
            "refer to reconstruction pixel centres."
        ),
    ]

    if not has_reconstruction:
        cells.extend(
            [
                new_markdown_cell(
                    "## Reconstruction preparation\n\n"
                    "This cell validates and sorts chronological Cartesian ADC data "
                    "when gridded reconstruction arrays are not already available."
                ),
                new_code_cell(
                    _sequence_result_reconstruction_code(),
                    metadata={"jupyter": {"source_hidden": True}, "collapsed": True},
                ),
            ]
        )

    if not has_reconstruction:
        cells.extend(
            [
                new_markdown_cell(
                    "## Named analysis entry points\n\n"
                    "Use `reconstructed_data` for the primary reconstructed result, "
                    "`raw_adc_signal` for chronological samples, and `reconstruction` "
                    "for all named representations."
                ),
                new_code_cell("result_dataset = ds\n" + _sequence_result_access_code()),
                new_markdown_cell(
                    "## Reconstruct Cartesian data from raw ADC samples\n\n"
                    "This explicit example repeats the validated raw-data path and "
                    "keeps the resulting dimensioned k-space and image available."
                ),
                new_code_cell(_sequence_result_raw_reconstruction_example_code()),
            ]
        )

    if result_kind == "cartesian_2d":
        cells.extend(
            [
                new_markdown_cell(
                    "## Reconstructed signal of one image pixel\n\n"
                    "`mxy_image` is the complex measured and reconstructed signal, "
                    "with flat storage frames replaced by meaningful dimensions. "
                    "Change `x`, `y`, and `trace_axis` in the short example below. "
                    "For EPSI, `echo` plots the pixel across echo times. With "
                    "repeated complete acquisitions, the commented example shows "
                    "how to use `over='repetition'` at a fixed echo."
                ),
                new_code_cell(
                    _sequence_result_cartesian_point_trace_code(),
                    metadata={"jupyter": {"source_hidden": True}, "collapsed": True},
                ),
                new_code_cell(_sequence_result_cartesian_point_example_code()),
                new_markdown_cell(
                    "## Interactive Cartesian 2D / EPSI explorer\n\n"
                    "Frame controls are named from their actual Pulseq labels. "
                    "Echo and acquisition repetition therefore have separate controls. "
                    "Use `Pixel view` to switch between the complex echo signal "
                    "and its centred FFT spectrum. The frequency-offset axis in Hz "
                    "is calculated from the physical echo spacing. "
                    "The implementation cell is collapsed; execute it to show the "
                    "controls."
                ),
                new_code_cell(
                    _sequence_result_cartesian_2d_explorer_code(),
                    metadata={"jupyter": {"source_hidden": True}, "collapsed": True},
                ),
            ]
        )
    elif result_kind != "raw_signal":
        cells.extend(
            [
                new_markdown_cell("## Interactive multidimensional explorer"),
                new_code_cell(
                    _sequence_result_explorer_code(),
                    metadata={"jupyter": {"source_hidden": True}, "collapsed": True},
                ),
            ]
        )

    cells.extend(
        [
            new_markdown_cell(
                "## True phantom-grid transverse magnetization at checkpoints\n\n"
                "This is distinct from the reconstructed receive image. It is only "
                "available when checkpoint times were requested before simulation."
            ),
            new_code_cell(_sequence_result_checkpoint_code()),
            new_markdown_cell("## Final longitudinal magnetization"),
            new_code_cell(
                "mz = ds.final_magnetization.sel(component='mz')\n"
                "while mz.ndim > 2:\n"
                "    mz = mz.isel({mz.dims[-1]: mz.sizes[mz.dims[-1]] // 2})\n"
                "fig, ax = plt.subplots(figsize=(6, 5))\n"
                "if mz.ndim == 1:\n"
                "    ax.plot(mz)\n"
                "else:\n"
                "    image = ax.imshow(mz.T, origin='lower', cmap='viridis')\n"
                "    fig.colorbar(image, ax=ax, label='Mz')\n"
                "ax.set_title('Final Mz (central slice)'); plt.show()"
            ),
            new_markdown_cell(
                "## Advanced: chronological ADC samples\n\n"
                "Each `signal` sample shares the `adc` dimension with its time, "
                "k-space coordinate, event index, readout-sample index, and Pulseq "
                "labels. Keep these xarray labels when grouping the data."
            ),
            new_code_cell(
                "signal = ds['signal']\n"
                "time_ms = ds['adc_time_s'] * 1e3\n"
                "fig, ax = plt.subplots(figsize=(9, 4))\n"
                "if 'coil' not in signal.dims:\n"
                "    ax.plot(time_ms, np.abs(signal), label='Magnitude')\n"
                "else:\n"
                "    for coil in signal.coil.values:\n"
                "        ax.plot(\n"
                "            time_ms, np.abs(signal.sel(coil=coil)),\n"
                "            label=f'Coil {coil}',\n"
                "        )\n"
                "ax.set(xlabel='ADC time (ms)', ylabel='Received signal (a.u.)')\n"
                "ax.legend(); ax.grid(True); plt.show()"
            ),
            new_code_cell(
                "coordinate_names = [name for name in (\n"
                "    'adc_time_s', 'kx', 'ky', 'kz', 'adc_event_index',\n"
                "    'readout_sample_index', 'slice_index', 'echo_index',\n"
                "    'repetition_index', 'segment_index', 'partition_index',\n"
                ") if name in ds.coords]\n"
                "adc_table = ds[coordinate_names].to_dataframe()\n"
                "adc_table['signal'] = (\n"
                "    ds.signal.values if ds.signal.ndim == 1\n"
                "    else list(ds.signal.transpose('adc', 'coil').values)\n"
                ")\n"
                "adc_table.head()",
                metadata={"jupyter": {"source_hidden": True}, "collapsed": True},
            ),
        ]
    )

    notebook = new_notebook(cells=cells)
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
