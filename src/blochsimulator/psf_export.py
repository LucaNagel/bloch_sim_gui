"""Save PSF results as a zip archive with data and an example notebook.

:func:`write_psf_results` packs the PSF of the PSF Analysis tab into one
``.zip`` file: ``<name>.npz`` holds the arrays and the metrics, and
``<name>.ipynb`` loads them and redraws the PSF image, the profiles through
the peak, the k-space modulation and, where present, the spectral line and
the slice profile. The notebook is plain notebook JSON, so writing it does not
need Jupyter or nbformat, and the data load with ``numpy.load`` alone, without
pickles or the simulator.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence
import zipfile

import numpy as np

from . import __version__
from .project_notebook import _cell


def _finite(value):
    """JSON-safe float: NaN and infinities become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _source_metadata(source) -> dict:
    return {
        "position_m": [float(value) for value in source.position_m],
        "t1_s": _finite(source.t1_s),
        "t2_s": _finite(source.t2_s),
        "t2_star_s": _finite(source.t2_star_s),
        "off_resonance_hz": _finite(source.off_resonance_hz),
        "equilibrium_mz": _finite(source.equilibrium_mz),
    }


def _axis_metrics(psf) -> dict:
    metrics = {}
    for role, axis in psf.axis_metrics.items():
        metrics[str(role)] = {
            "voxel_size_m": _finite(axis.voxel_size_m),
            "fwhm_m": _finite(axis.fwhm_m),
            "fwhm_voxels": _finite(axis.fwhm_voxels),
            "ideal_fwhm_m": _finite(axis.ideal_fwhm_m),
            "broadening": _finite(axis.broadening),
            "peak_offset_m": _finite(axis.peak_offset_m),
            "peak_offset_voxels": _finite(axis.peak_offset_voxels),
            "peak_sidelobe_ratio": _finite(axis.peak_sidelobe_ratio),
            "ideal_peak_sidelobe_ratio": _finite(axis.ideal_peak_sidelobe_ratio),
        }
    return metrics


def _spectral_metrics(metrics) -> Optional[dict]:
    if metrics is None:
        return None
    return {
        "resolution_hz": _finite(metrics.resolution_hz),
        "fwhm_hz": _finite(metrics.fwhm_hz),
        "fwhm_points": _finite(metrics.fwhm_points),
        "ideal_fwhm_hz": _finite(metrics.ideal_fwhm_hz),
        "broadening": _finite(metrics.broadening),
        "peak_offset_hz": _finite(metrics.peak_offset_hz),
        "peak_sidelobe_ratio": _finite(metrics.peak_sidelobe_ratio),
        "ideal_peak_sidelobe_ratio": _finite(metrics.ideal_peak_sidelobe_ratio),
    }


def _slice_metrics(slice_profile) -> list:
    rows = []
    for label, metrics in zip(slice_profile.frame_labels, slice_profile.metrics):
        rows.append(
            {
                "frame": str(label),
                "nominal_thickness_m": _finite(metrics.nominal_thickness_m),
                "nominal_centre_m": _finite(metrics.nominal_centre_m),
                "fwhm_m": _finite(metrics.fwhm_m),
                "thickness_ratio": _finite(metrics.thickness_ratio),
                "centre_m": _finite(metrics.centre_m),
                "out_of_slice_fraction": _finite(metrics.out_of_slice_fraction),
            }
        )
    return rows


def psf_result_arrays(
    psf,
    *,
    slice_profile=None,
    title: str = "",
    role_labels: Optional[Sequence[str]] = None,
    settings: Optional[Mapping] = None,
) -> dict:
    """Arrays and metadata of ``psf`` (a :class:`PSFResult`) for ``np.savez``.

    ``metadata_json`` holds the source, the metrics and the labels as a JSON
    string; every other entry is a numeric or string array, so the file loads
    without ``allow_pickle``. ``settings`` adds display settings to the
    metadata, such as the colour scale of the GUI.
    """
    roles = [str(role) for role in psf.roles]
    labels = (
        [str(label) for label in role_labels]
        if role_labels is not None
        else [role.capitalize() for role in roles]
    )
    data = {
        "psf": np.asarray(psf.psf),
        "ideal_psf": np.asarray(psf.ideal_psf),
        "modulation": np.asarray(psf.modulation),
        "voxel_size_m": np.asarray(psf.voxel_size_m, dtype=float),
        "roles": np.asarray(roles),
    }
    for role, offsets in zip(roles, psf.offsets_m):
        data[f"offsets_m_{role}"] = np.asarray(offsets, dtype=float)
    for role, k in zip(roles, psf.k_cyc_per_m):
        data[f"k_cyc_per_m_{role}"] = np.asarray(k, dtype=float)
    for role in roles:
        line, values, ideal = psf.profiles[role]
        data[f"profile_offsets_m_{role}"] = np.asarray(line, dtype=float)
        data[f"profile_{role}"] = np.asarray(values)
        data[f"profile_ideal_{role}"] = np.asarray(ideal)
    if psf.spectral_profile is not None:
        frequency, line, ideal = psf.spectral_profile
        data["spectral_frequency_hz"] = np.asarray(frequency, dtype=float)
        data["spectral_line"] = np.asarray(line)
        data["spectral_ideal_line"] = np.asarray(ideal)
    if slice_profile is not None:
        data["slice_positions_m"] = np.asarray(slice_profile.positions_m, dtype=float)
        data["slice_profiles"] = np.asarray(slice_profile.profiles)
        data["slice_normal_xyz"] = np.asarray(slice_profile.normal_xyz, dtype=float)
        data["slice_frame_labels"] = np.asarray(
            [str(label) for label in slice_profile.frame_labels]
        )
    metadata = {
        "written_by": f"Bloch Simulator {__version__}",
        "title": str(title),
        "label": str(psf.label),
        "kind": str(psf.kind),
        "frame": int(psf.frame),
        "frame_label": str(psf.frame_labels[psf.frame]),
        "frame_labels": [str(label) for label in psf.frame_labels],
        "roles": roles,
        "role_labels": labels,
        "voxel_size_m": [float(value) for value in psf.voxel_size_m],
        "gradient_error": str(psf.gradient_error),
        "source": _source_metadata(psf.source),
        "peak_efficiency": _finite(psf.peak_efficiency),
        "voxel_energy_fraction": _finite(psf.voxel_energy_fraction),
        "ideal_voxel_energy_fraction": _finite(psf.ideal_voxel_energy_fraction),
        "axis_metrics": _axis_metrics(psf),
        "spectral_metrics": _spectral_metrics(psf.spectral_metrics),
        "slice_metrics": (
            None if slice_profile is None else _slice_metrics(slice_profile)
        ),
        "settings": dict(settings or {}),
    }
    data["metadata_json"] = np.asarray(json.dumps(metadata, indent=1))
    return data


def psf_results_notebook(data_file: str) -> dict:
    """Return the example notebook that loads the PSF results in ``data_file``."""
    cells = [
        _cell(
            "markdown",
            f"""
# PSF results

The point spread function (PSF) is the image of a single point source. This
notebook loads `{data_file}`, written by the PSF Analysis tab of Bloch
Simulator {__version__}, prints the metrics and redraws the plots of the tab:

1. the PSF image, normalised to its peak, with offsets in voxels from the source,
2. the profiles through the peak along each encoding direction, with the ideal
   PSF (uniform k-space weighting) dashed,
3. the k-space modulation (the transfer function),
4. the spectral line (CSI) or the slice profile (2D), when they were computed.

**Ideal PSF (uniform k-space weighting):** the same k-space samples and
reconstruction, but every sample has the measured k-space-centre amplitude, so
there is no relaxation, hyperpolarized depletion, T2* or gradient error along
the k-space order. It is not a point: finite k-space coverage gives a sinc-like
PSF (about 1.21 voxels FWHM for Cartesian). Broadening and peak efficiency are
relative to it. For CSI, the ideal spectral line is that of an undamped FID.

Keep the notebook next to `{data_file}`. Only NumPy and Matplotlib are needed.
""",
        ),
        _cell(
            "code",
            f"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = Path({data_file!r})
data = np.load(DATA_FILE)
meta = json.loads(str(data["metadata_json"]))
roles = meta["roles"]
labels = dict(zip(roles, meta["role_labels"]))
voxel = dict(zip(roles, meta["voxel_size_m"]))
psf = data["psf"]
ideal = data["ideal_psf"]
scale = float(np.abs(ideal).max()) or 1.0

print(meta["title"] or meta["label"])
print(f"Frame {{meta['frame_label']}}, {{meta['kind']}} layout, gradient errors: {{meta['gradient_error']}}")
print("Source:", meta["source"])
print("Arrays:", ", ".join(name for name in data.files if name != "metadata_json"))
""",
        ),
        _cell("markdown", "## Metrics"),
        _cell(
            "code",
            """
def show(value, pattern="{:.3g}"):
    return "—" if value is None else pattern.format(value)


print(f"Peak efficiency:       {show(meta['peak_efficiency'])} (peak / ideal peak, uniform k-space weighting)")
print(f"Energy in own voxel:   {show(meta['voxel_energy_fraction'])}"
      f" (ideal, uniform k-space weighting: {show(meta['ideal_voxel_energy_fraction'])})")
print()
print(f"{'Direction':<24}{'FWHM [vox]':>12}{'Broadening':>12}{'Shift [vox]':>13}{'PSLR':>10}")
for role in roles:
    m = meta["axis_metrics"].get(role)
    if m is None:
        continue
    print(f"{labels[role]:<24}{show(m['fwhm_voxels']):>12}{show(m['broadening']):>12}"
          f"{show(m['peak_offset_voxels']):>13}{show(m['peak_sidelobe_ratio']):>10}")
print("Broadening: FWHM / FWHM of the ideal PSF (uniform k-space weighting).")
if meta["spectral_metrics"]:
    s = meta["spectral_metrics"]
    print()
    print(f"Spectral line: FWHM {show(s['fwhm_hz'])} Hz ({show(s['fwhm_points'])} points),"
          f" broadening {show(s['broadening'])}, shift {show(s['peak_offset_hz'])} Hz")
for row in meta["slice_metrics"] or []:
    print(f"Slice {row['frame']}: FWHM {show(row['fwhm_m'] and row['fwhm_m'] * 1e3)} mm,"
          f" thickness ratio {show(row['thickness_ratio'])},"
          f" out-of-slice fraction {show(row['out_of_slice_fraction'])}")
""",
        ),
        _cell(
            "markdown",
            """
## 1 · PSF image

A 3D PSF is shown in the plane through its peak that contains the last two
directions; change `PLANE_AXES` for another plane. `DB = True` shows side lobes
on a logarithmic scale.
""",
        ),
        _cell(
            "code",
            """
DB = False
PLANE_AXES = (-2, -1)  # array axes of psf shown as rows and columns

magnitude = np.abs(psf) / (np.abs(psf).max() or 1.0)
peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)
if magnitude.ndim == 1:
    image, shown = magnitude[None, :], (roles[0], roles[0])
else:
    rows, cols = (axis % magnitude.ndim for axis in PLANE_AXES)
    index = tuple(slice(None) if axis in (rows, cols) else peak[axis]
                  for axis in range(magnitude.ndim))
    image = magnitude[index]
    if rows > cols:
        image = image.T
    shown = (roles[rows], roles[cols])
if DB:
    image = 20 * np.log10(np.maximum(image, 1e-4))


def extent_vox(role):
    offsets = data[f"offsets_m_{role}"] / voxel[role]
    step = offsets[1] - offsets[0] if offsets.size > 1 else 1.0
    return offsets[0] - step / 2, offsets[-1] + step / 2


fig, ax = plt.subplots(figsize=(5, 4.5))
x0, x1 = extent_vox(shown[1])
y0, y1 = extent_vox(shown[0])
shown_image = ax.imshow(image, origin="lower", extent=(x0, x1, y0, y1),
                        aspect="auto", cmap="magma",
                        vmin=-80 if DB else 0, vmax=0 if DB else 1)
fig.colorbar(shown_image, label="|PSF| [dB]" if DB else "|PSF| / peak")
ax.set_xlabel(f"{labels[shown[1]]} offset [voxel]")
ax.set_ylabel(f"{labels[shown[0]]} offset [voxel]")
ax.set_title("PSF image")
plt.show()
""",
        ),
        _cell("markdown", "## 2 · Profiles through the peak"),
        _cell(
            "code",
            """
fig, ax = plt.subplots(figsize=(6, 3.5))
for role in roles:
    x = data[f"profile_offsets_m_{role}"] / voxel[role]
    line, = ax.plot(x, np.abs(data[f"profile_{role}"]) / scale, label=labels[role])
    ax.plot(x, np.abs(data[f"profile_ideal_{role}"]) / scale, "--",
            color=line.get_color(), linewidth=1)
ax.plot([], [], "k--", linewidth=1, label="ideal (uniform k-space weighting)")
ax.set_xlabel("Offset from the source [voxel]")
ax.set_ylabel("|PSF| / ideal peak (uniform k-space weighting)")
ax.set_ylim(bottom=0)
ax.legend()
plt.show()
""",
        ),
        _cell(
            "markdown",
            """
## 3 · k-space modulation

The sampled k-space with the source's encoding phase removed. Cartesian
layouts are a grid over k; spiral layouts hold one value per sample.
""",
        ),
        _cell(
            "code",
            """
modulation = data["modulation"]
k = [data[f"k_cyc_per_m_{role}"] for role in roles]
fig, ax = plt.subplots(figsize=(5, 4.5))
if meta["kind"] == "spiral":
    points = ax.scatter(k[-1], k[-2] if len(k) > 1 else np.zeros_like(k[-1]),
                        c=np.abs(modulation), s=4, cmap="viridis")
    fig.colorbar(points, label="|modulation|")
    ax.set_xlabel(f"k {labels[roles[-1]]} [1/m]")
    ax.set_ylabel(f"k {labels[roles[-2]]} [1/m]" if len(k) > 1 else "")
    ax.set_aspect("equal")
elif modulation.ndim == 1:
    ax.plot(k[0], np.abs(modulation))
    ax.set_xlabel(f"k {labels[roles[0]]} [1/m]")
    ax.set_ylabel("|modulation|")
else:
    centre = tuple(int(np.argmin(np.abs(values))) for values in k)
    index = tuple(slice(None) if axis >= modulation.ndim - 2 else centre[axis]
                  for axis in range(modulation.ndim))
    plane = np.abs(modulation[index])
    shown_plane = ax.imshow(plane, origin="lower", aspect="auto", cmap="viridis",
                            extent=(k[-1][0], k[-1][-1], k[-2][0], k[-2][-1]))
    fig.colorbar(shown_plane, label="|modulation|")
    ax.set_xlabel(f"k {labels[roles[-1]]} [1/m]")
    ax.set_ylabel(f"k {labels[roles[-2]]} [1/m]")
ax.set_title("k-space modulation")
plt.show()
""",
        ),
        _cell("markdown", "## 4 · Spectral line and slice profile"),
        _cell(
            "code",
            """
if "spectral_line" in data.files:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    frequency = data["spectral_frequency_hz"]
    peak_line = np.abs(data["spectral_ideal_line"]).max() or 1.0
    ax.plot(frequency, np.abs(data["spectral_line"]) / peak_line, label="line")
    ax.plot(frequency, np.abs(data["spectral_ideal_line"]) / peak_line, "k--",
            linewidth=1, label="ideal (undamped FID)")
    ax.set_xlabel("Frequency offset from the source [Hz]")
    ax.set_ylabel("|line| / ideal peak (undamped FID)")
    ax.legend()
    plt.show()

if "slice_profiles" in data.files:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    profiles = np.abs(data["slice_profiles"])
    profiles = profiles / (profiles.max() or 1.0)
    for label, row in zip(data["slice_frame_labels"], profiles):
        ax.plot(data["slice_positions_m"] * 1e3, row, label=str(label))
    ax.set_xlabel("Source position along the slice normal [mm]")
    ax.set_ylabel("|signal| / peak")
    ax.legend()
    plt.show()

if "spectral_line" not in data.files and "slice_profiles" not in data.files:
    print("No spectral line or slice profile in this file.")
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


def write_psf_results(path, psf, **arrays_options) -> Path:
    """Write ``psf`` and its notebook into the zip archive ``path``.

    The archive holds ``<stem>.npz`` and ``<stem>.ipynb``; ``arrays_options``
    go to :func:`psf_result_arrays`. Returns the archive path, with ``.zip``
    appended when ``path`` has another suffix.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() != ".zip":
        path = path.with_name(path.name + ".zip")
    stem = path.stem
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **psf_result_arrays(psf, **arrays_options))
    notebook = psf_results_notebook(f"{stem}.npz")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}.npz", buffer.getvalue())
        archive.writestr(
            f"{stem}.ipynb",
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        )
    return path
