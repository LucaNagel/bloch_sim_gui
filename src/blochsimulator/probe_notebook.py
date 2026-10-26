"""Example notebook that loads exported spin-probe results.

The notebook is written next to spin-probe exports (``.nc`` or ``.npz`` from
:meth:`SequenceProbeResult.save`) or next to the ``.blochproj`` files of a
spin-probe parameter loop. It shows one probe as a time course and as a
spectrum or spatial profile, and then collects every probe in the folder,
ordered by its parameter-loop index. Like :mod:`project_notebook` it is plain
notebook JSON, so writing it does not need Jupyter or nbformat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import __version__
from .project_notebook import _cell, unique_notebook_path

DEFAULT_PROBE_NOTEBOOK_NAME = "load_spin_probes.ipynb"


def probe_loader_notebook(
    directory, *, data_file: Optional[str] = None, loop_id: Optional[str] = None
) -> dict:
    """Return the example notebook for the spin probes in ``directory``."""
    directory = Path(directory).expanduser().resolve()
    focus = (
        f"It was written together with `{data_file}`, which the first part loads."
        if data_file
        else "The first part loads the first spin probe it finds."
    )
    loop_note = (
        "The probes belong to one parameter loop; `LOOP_ID` keeps only that "
        "loop's files. Set it to `None` to load every probe in the folder."
        if loop_id
        else "Set `LOOP_ID` to a loop id to keep only the runs of one parameter loop."
    )
    cells = [
        _cell(
            "markdown",
            f"""
# Load exported spin probes

A spin probe follows single isochromats through the sequence and stores their
magnetization at every checkpoint. A *spectral* probe has one position and many
frequency offsets, a *geometry* probe has many positions at one offset.

This notebook reads spin-probe exports (`.nc`, `.npz`) and the `.blochproj`
projects of spin-probe parameter loops in this folder. {focus}
{loop_note}

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
from blochsimulator.sequence import SequenceProbeResult

# The folder of this notebook. Change it if you move the notebook.
sim_dir = Path(r"{directory}")
if not sim_dir.is_dir():
    sim_dir = Path.cwd()
DATA_FILE = {data_file!r}
LOOP_ID = {loop_id!r}


def load_probe(path):
    \"\"\"Return the spin probe stored in ``path``, or None for other files.\"\"\"
    path = Path(path)
    if path.suffix == ".blochproj":
        result = load_project(path)["sequence_result"]
        return result if isinstance(result, SequenceProbeResult) else None
    try:
        return SequenceProbeResult.load(path)
    except (ValueError, KeyError, OSError):
        return None


files = sorted(
    path for path in sim_dir.iterdir() if path.suffix in {{".nc", ".npz", ".blochproj"}}
)
print(f"{{len(files)}} candidate files in {{sim_dir}}")
""",
        ),
        _cell(
            "markdown",
            """
## One spin probe

A `SequenceProbeResult` holds

| field | content |
| --- | --- |
| `time_s` | checkpoint times `(time,)` |
| `positions_m` | probe positions `(position, 3)` |
| `frequency_offsets_hz` | off-resonance of every isochromat `(frequency,)` |
| `magnetization` | Mx, My, Mz with shape `(time, position, frequency, 3)` |
| `metadata` | T1/T2, probe type, ideal spoiler times, parameter loop, ... |

`probe.mxy`, `probe.mz` and `probe.coherent_mxy` (complex mean over positions,
what a receiver would see) are shortcuts. The frequency axis shown in the GUI
(Hz or ppm) is stored in `metadata["frequency_axis_unit"]`.
""",
        ),
        _cell(
            "code",
            """
if DATA_FILE is not None and (sim_dir / DATA_FILE).exists():
    path = sim_dir / DATA_FILE
else:
    path = next(file for file in files if load_probe(file) is not None)
probe = load_probe(path)
meta = probe.metadata

unit = str(meta.get("frequency_axis_unit", "Hz"))
display_axis = np.asarray(
    meta.get(f"frequency_offsets_{unit.lower()}", probe.frequency_offsets_hz), dtype=float
)
if display_axis.shape != probe.frequency_offsets_hz.shape:
    display_axis, unit = probe.frequency_offsets_hz, "Hz"
spoilers_ms = 1e3 * np.asarray(meta.get("ideal_spoiler_end_times_s") or [], dtype=float)
t_ms = 1e3 * probe.time_s

print("File:", path.name)
print("Probe type:", meta.get("ui_probe_mode") or meta.get("probe_type"))
print("Sequence:", meta.get("sequence_source"))
print(f"Time points: {t_ms.size} ({t_ms[0]:.3f} ... {t_ms[-1]:.3f} ms)")
print("Positions:", probe.positions_m.shape[0], " Frequencies:", display_axis.size)
print(f"T1 = {meta.get('t1_s')} s, T2 = {meta.get('t2_s')} s")
print("Parameter loop:", meta.get("parameter_loop"))
""",
        ),
        _cell(
            "markdown",
            """
### Time course

Coherent |Mxy| and mean Mz of one frequency offset. Dashed grey lines mark the
ends of ideal spoilers: signal after them is new signal from the following RF.
""",
        ),
        _cell(
            "code",
            """
mxy = probe.coherent_mxy  # (time, frequency)
mz = probe.mz.mean(axis=1)  # (time, frequency)
FREQUENCY_INDEX = int(np.argmin(np.abs(display_axis)))  # closest to on resonance

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t_ms, np.abs(mxy[:, FREQUENCY_INDEX]), label="|Mxy|")
ax.plot(t_ms, mz[:, FREQUENCY_INDEX], label="Mz")
for t in spoilers_ms:
    ax.axvline(t, color="0.6", ls="--", lw=0.8)
ax.set(
    xlabel="time (ms)",
    ylabel="magnetization",
    title=f"offset {display_axis[FREQUENCY_INDEX]:g} {unit}",
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
""",
        ),
        _cell(
            "markdown",
            """
### Spectrum or spatial profile at one time point

A spectral probe is plotted against its frequency offset, a geometry probe
against the position along its longest axis.
""",
        ),
        _cell(
            "code",
            """
TIME_INDEX = -1  # last checkpoint; e.g. int(np.argmin(np.abs(t_ms - 12.5))) for 12.5 ms

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
if display_axis.size > 1:
    x, xlabel = display_axis, f"frequency offset ({unit})"
    m_xy, m_z = mxy[TIME_INDEX], mz[TIME_INDEX]
else:
    positions_mm = 1e3 * probe.positions_m
    axis = int(np.argmax(np.ptp(positions_mm, axis=0)))
    order = np.argsort(positions_mm[:, axis])
    x, xlabel = positions_mm[order, axis], f"{'xyz'[axis]} (mm)"
    m_xy = probe.mxy[TIME_INDEX, order, 0]
    m_z = probe.mz[TIME_INDEX, order, 0]
axes[0].plot(x, np.abs(m_xy))
axes[0].set_ylabel("|Mxy|")
axes[1].plot(x, np.angle(m_xy))
axes[1].set_ylabel("phase (rad)")
axes[2].plot(x, m_z)
axes[2].set_ylabel("Mz")
axes[2].set_xlabel(xlabel)
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle(f"t = {t_ms[TIME_INDEX]:.3f} ms")
fig.tight_layout()
""",
        ),
        _cell(
            "markdown",
            """
### As an xarray dataset

`to_xarray()` labels every dimension, which helps with selections such as
`ds.mz.sel(frequency=0, method="nearest")`.
""",
        ),
        _cell("code", "ds = probe.to_xarray()\nds"),
        _cell(
            "markdown",
            """
## All spin probes in this folder

Every probe is loaded once and only the coherent Mxy and the mean Mz are kept,
so large loops fit into memory. Loop runs are ordered by their loop index and
labelled with the varied parameter; other probes keep their file name.
""",
        ),
        _cell(
            "code",
            """
runs = []
for file in files:
    result = load_probe(file)
    if result is None:
        continue
    info = dict(result.metadata.get("parameter_loop") or {})
    if LOOP_ID is not None and info.get("loop_id") != LOOP_ID:
        continue
    runs.append(
        {
            "file": file.name,
            "index": info.get("index", len(runs)),
            "parameter": info.get("parameter_label", ""),
            "unit": info.get("unit", ""),
            "value": info.get("value", np.nan),
            "time_ms": 1e3 * result.time_s,
            "frequency_hz": result.frequency_offsets_hz,
            "mxy": result.coherent_mxy,
            "mz": result.mz.mean(axis=1),
        }
    )
    del result  # release the full probe before loading the next

runs.sort(key=lambda run: run["index"])
for run in runs:
    label = f"{run['parameter']} = {run['value']:g} {run['unit']}" if run["parameter"] else ""
    print(f"{run['index']:3d}  {label}  {run['file']}")


def run_label(run):
    if run["parameter"]:
        return f"{run['value']:g} {run['unit']}"
    return run["file"]
""",
        ),
        _cell(
            "code",
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
colors = plt.colormaps["viridis"](np.linspace(0, 1, max(1, len(runs))))
for color, run in zip(colors, runs):
    centre = int(np.argmin(np.abs(run["frequency_hz"])))
    axes[0].plot(run["time_ms"], np.abs(run["mxy"][:, centre]), color=color, label=run_label(run))
    if run["frequency_hz"].size > 1:
        axes[1].plot(run["frequency_hz"], np.abs(run["mxy"][-1]), color=color)
axes[0].set(xlabel="time (ms)", ylabel="|Mxy| on resonance")
axes[1].set(xlabel="frequency offset (Hz)", ylabel="|Mxy| at the last checkpoint")
if len(runs) <= 12:
    axes[0].legend(fontsize=7)
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.tight_layout()

values = np.array([run["value"] for run in runs], dtype=float)
if runs and np.isfinite(values).all():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(values, [np.abs(run["mxy"][-1]).mean() for run in runs], "o-", label="mean |Mxy|")
    ax.plot(values, [run["mz"][-1].mean() for run in runs], "s-", label="mean Mz")
    ax.set(xlabel=f"{runs[0]['parameter']} ({runs[0]['unit']})", ylabel="last checkpoint")
    ax.legend()
    ax.grid(True, alpha=0.3)
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


def write_probe_loader_notebook(
    directory,
    *,
    data_file: Optional[str] = None,
    loop_id: Optional[str] = None,
    filename: str = DEFAULT_PROBE_NOTEBOOK_NAME,
    overwrite: bool = False,
) -> Path:
    """Write the spin-probe notebook into ``directory`` and return its path.

    Without ``overwrite`` an existing notebook is kept and the new one receives
    a numbered name, so user edits are never lost.
    """
    directory = Path(directory).expanduser().resolve()
    path = directory / filename
    if not overwrite:
        path = unique_notebook_path(path)
    notebook = probe_loader_notebook(directory, data_file=data_file, loop_id=loop_id)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return path
