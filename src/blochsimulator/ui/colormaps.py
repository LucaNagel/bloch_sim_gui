"""Shared colormap choices for every user-selectable colormap in the GUI.

Names are matplotlib names, so ``pg.colormap.get`` resolves them from
pyqtgraph's own maps first and from matplotlib otherwise. Importing this
module also adds the maps to the right-click gradient menu of every
pyqtgraph histogram LUT that is created afterwards.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

# Grouped as sequential, diverging, cyclic (for phase) and rainbow maps.
SEQUENTIAL_COLORMAPS = (
    "gray",
    "gray_r",
    "bone",
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "cividis",
    "turbo",
    "hot",
    "afmhot",
    "copper",
    "cubehelix",
    "gnuplot2",
)
DIVERGING_COLORMAPS = ("coolwarm", "RdBu_r", "seismic", "bwr", "PiYG")
CYCLIC_COLORMAPS = ("twilight", "twilight_shifted", "hsv")
RAINBOW_COLORMAPS = ("jet", "rainbow", "Spectral_r", "nipy_spectral")
COLORMAP_NAMES = (
    SEQUENTIAL_COLORMAPS + DIVERGING_COLORMAPS + CYCLIC_COLORMAPS + RAINBOW_COLORMAPS
)


def colormap(name: str) -> pg.ColorMap:
    """Return a pyqtgraph colormap for a name from :data:`COLORMAP_NAMES`."""
    if name == "gray":
        return pg.ColorMap(
            np.asarray([0.0, 1.0]),
            np.asarray([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte),
        )
    if name == "gray_r":
        return pg.ColorMap(
            np.asarray([0.0, 1.0]),
            np.asarray([[255, 255, 255, 255], [0, 0, 0, 255]], dtype=np.ubyte),
        )
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.get(name, source="matplotlib")


def _register_gradient_presets(stops: int = 16) -> None:
    """Add the shared maps to pyqtgraph's histogram LUT preset menu."""
    for name in COLORMAP_NAMES:
        if name in Gradients or name == "gray":
            continue
        try:
            positions = np.linspace(0.0, 1.0, stops)
            colors = colormap(name).map(positions, mode="byte")
        except Exception:
            continue
        Gradients[name] = {
            "ticks": [
                (float(position), tuple(int(value) for value in color))
                for position, color in zip(positions, colors)
            ],
            "mode": "rgb",
        }


_register_gradient_presets()
