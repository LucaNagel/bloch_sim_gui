"""Example notebook that loads exported simulation projects.

The notebook is written next to the exported ``.blochproj`` files and shows
how to read them with :func:`blochsimulator.project_io.load_project`, how to
reach the sequence, phantom, result and parameter-loop metadata, and how to
collect the runs of a parameter loop into arrays. It is plain notebook JSON,
so writing it does not need Jupyter or nbformat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import __version__

DEFAULT_NOTEBOOK_NAME = "load_simulations.ipynb"


def _cell(kind: str, source: str) -> dict:
    lines = source.strip("\n").splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def project_loader_notebook(directory, *, loop_id: Optional[str] = None) -> dict:
    """Return the example notebook for the projects in ``directory``."""
    directory = Path(directory).expanduser().resolve()
    loop_note = (
        "It belongs to one parameter loop; `LOOP_ID` selects that loop's files "
        "if the folder holds several loops. Set it to `None` to load every project."
        if loop_id
        else "Set `LOOP_ID` to a loop id to keep only the runs of one parameter loop."
    )
    cells = [
        _cell(
            "markdown",
            f"""
# Load exported Bloch Simulator projects

Every `.blochproj` file in this folder is one complete simulation: the sequence
program, the phantom, the simulation result and the GUI state. {loop_note}

Written by Bloch Simulator {__version__}.
""",
        ),
        _cell(
            "code",
            f"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blochsimulator.project_io import load_project

# The folder of this notebook. Change it if you move the notebook.
sim_dir = Path(r"{directory}")
if not sim_dir.is_dir():
    sim_dir = Path.cwd()
LOOP_ID = {loop_id!r}

files = sorted(sim_dir.glob("*.blochproj"))
print(f"{{len(files)}} projects in {{sim_dir}}")
""",
        ),
        _cell(
            "markdown",
            """
## One project

`load_project` returns a dictionary:

| key | content |
| --- | --- |
| `program` | the simulated sequence (`events`, `rf_events`, `gradient_events`, `adc_events`, `metadata`) |
| `phantom` | the simulated object (`None` for a spectral spin probe) |
| `sequence_result` | the result, see below |
| `state` | the GUI state, including the sequence controls |
| `tx_field`, `rx_field` | optional B1 transmit/receive fields |

A sequence simulation result (`SequenceSimulationResult`) holds the ADC `signal`
(`(adc,)` or `(coil, adc)`), `adc_times_s`, the `final_magnetization`
(`(..., 3)` for Mx, My, Mz on the phantom grid) and, for multi-pool phantoms,
`pool_names`, `species_signal` and `final_pool_magnetization`.
A spin-probe result (`SequenceProbeResult`) holds `time_s`, `positions_m`,
`frequency_offsets_hz` and `magnetization` with shape
`(time, position, frequency, 3)`.

A run of a parameter loop stores the varied parameter in
`result.metadata["parameter_loop"]`.
""",
        ),
        _cell(
            "code",
            """
project = load_project(files[0])
program = project["program"]
phantom = project["phantom"]
result = project["sequence_result"]

print("Result type:", type(result).__name__)
print("Sequence duration:", f"{program.duration_s * 1e3:.3f} ms")
print("Sequence definitions:", program.metadata.get("definitions", {}))
if phantom is not None:
    print("Phantom:", type(phantom).__name__, getattr(phantom, "name", ""))
    print("Phantom shape:", getattr(phantom, "shape", None))
    pools = getattr(phantom, "pools", None) or getattr(phantom, "species", None)
    for pool in pools or ():
        print(f"  {pool.name}: T1 = {pool.t1:g} s, T2 = {pool.t2:g} s")

for index, rf_event in enumerate(program.rf_events):
    duration_ms = rf_event.samples_hz.size * rf_event.raster_s * 1e3
    flip_angle_deg = 360.0 * abs(np.sum(rf_event.samples_hz) * rf_event.raster_s)
    print(f"RF {index}: {duration_ms:.3f} ms, {flip_angle_deg:.3f}°")
    if index == 4:
        print(f"... {len(program.rf_events)} RF pulses in total")
        break

print("Parameter loop:", result.metadata.get("parameter_loop"))
""",
        ),
        _cell(
            "code",
            """
def transverse(result):
    \"\"\"Return the final complex transverse magnetization Mx + iMy.\"\"\"
    if hasattr(result, "final_magnetization"):
        m = np.asarray(result.final_magnetization)
    else:
        # Spin probe: last stored time point, (position, frequency)
        m = np.asarray(result.magnetization)[-1]
    return m[..., 0] + 1j * m[..., 1]


def loop_info(result):
    return dict(result.metadata.get("parameter_loop") or {})


mxy = transverse(result)
print("Mxy shape:", mxy.shape)
if hasattr(result, "signal"):
    print("Signal shape:", np.shape(result.signal), "ADC samples:", result.adc_times_s.size)
""",
        ),
        _cell(
            "markdown",
            """
## All runs of the loop

Projects are loaded one after another and only the derived arrays are kept,
so large loops fit into memory. The runs are ordered by their loop index.
""",
        ),
        _cell(
            "code",
            """
runs = []
for file in files:
    project = load_project(file)
    result = project["sequence_result"]
    if result is None:
        continue
    info = loop_info(result)
    if LOOP_ID is not None and info.get("loop_id") != LOOP_ID:
        continue
    runs.append(
        {
            "file": file.name,
            "index": info.get("index", len(runs)),
            "parameter": info.get("parameter_label", ""),
            "unit": info.get("unit", ""),
            "value": info.get("value", np.nan),
            "mxy": transverse(result),
            "signal": np.asarray(getattr(result, "signal", np.zeros(0))),
        }
    )
    del project, result  # release the full project before loading the next

runs.sort(key=lambda run: run["index"])
values = np.array([run["value"] for run in runs], dtype=float)
for run in runs:
    print(f"{run['index']:3d}  {run['parameter']} = {run['value']:g} {run['unit']}  {run['file']}")
""",
        ),
        _cell(
            "code",
            """
def image(mxy):
    \"\"\"Reduce Mxy to a 2D image or a 1D profile for plotting.\"\"\"
    mxy = np.abs(np.squeeze(mxy))
    if mxy.ndim == 3:
        mxy = mxy[:, :, mxy.shape[2] // 2]  # central z slice
    return mxy


columns = min(8, max(1, len(runs)))
rows = int(np.ceil(len(runs) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
cmap = plt.colormaps["turbo"].copy()
cmap.set_bad(color="none")
for ax, run in zip(axes.flat, runs):
    data = image(run["mxy"])
    if data.ndim == 2:
        ax.imshow(np.rot90(np.where(data == 0, np.nan, data)), cmap=cmap)
        ax.axis("off")
    else:
        ax.plot(np.atleast_1d(data))
    ax.set_title(f"{run['value']:g} {run['unit']}", fontsize=8)
for ax in list(axes.flat)[len(runs):]:
    ax.axis("off")
fig.suptitle(f"|Mxy| at the end of the sequence vs {runs[0]['parameter'] if runs else ''}")
fig.tight_layout()

has_signal = any(run["signal"].size for run in runs)
fig, axes = plt.subplots(1, 2 if has_signal else 1, figsize=(9 if has_signal else 4.5, 3.5), squeeze=False)
axes = axes[0]
axes[0].plot(values, [np.mean(np.abs(run["mxy"])) for run in runs], "o-")
axes[0].set_ylabel("mean |Mxy|")
if has_signal:
    axes[1].plot(values, [np.max(np.abs(run["signal"])) if run["signal"].size else np.nan for run in runs], "s-")
    axes[1].set_ylabel("max |signal|")
for ax in axes:
    ax.set_xlabel(f"{runs[0]['parameter']} ({runs[0]['unit']})" if runs else "value")
fig.tight_layout()
""",
        ),
    ]
    for index, cell in enumerate(cells):
        # nbformat 4.5 requires a unique id per cell.
        cell["id"] = f"cell-{index}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_project_loader_notebook(
    directory, *, loop_id: Optional[str] = None, filename: str = DEFAULT_NOTEBOOK_NAME
) -> Path:
    """Write the example notebook into ``directory`` and return its path.

    An existing notebook with the same name is kept; the new one receives a
    numbered name instead, so user edits are never overwritten.
    """
    directory = Path(directory).expanduser().resolve()
    path = directory / filename
    stem, suffix = path.stem, path.suffix
    number = 2
    while path.exists():
        path = directory / f"{stem}_{number}{suffix}"
        number += 1
    notebook = project_loader_notebook(directory, loop_id=loop_id)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return path
