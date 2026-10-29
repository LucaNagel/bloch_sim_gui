"""Shared interaction behavior for pyqtgraph plots."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


AXIS_ZOOM_TOOLTIP = (
    "Pinch or scroll to zoom both axes. Hold Shift to zoom only horizontally "
    "(X), or Alt/Option to zoom only vertically (Y). Pinching directly over "
    "an axis also limits zoom to that axis."
)


def zoom_axis_for_modifiers(modifiers):
    """Return the pyqtgraph axis selected by the active keyboard modifier."""
    horizontal = bool(modifiers & Qt.ShiftModifier)
    vertical = bool(modifiers & Qt.AltModifier)
    if horizontal == vertical:
        return None
    return pg.ViewBox.XAxis if horizontal else pg.ViewBox.YAxis


def _event_modifiers(event):
    modifier_getter = getattr(event, "modifiers", None)
    if callable(modifier_getter):
        return modifier_getter()
    return QApplication.keyboardModifiers()


def install_axis_constrained_zoom():
    """Add Shift-X and Alt/Option-Y zoom constraints to every ViewBox."""
    current_handler = pg.ViewBox.wheelEvent
    if getattr(current_handler, "_blochsimulator_axis_zoom", False):
        return

    def wheel_event(view_box, event, axis=None):
        if axis is None:
            axis = zoom_axis_for_modifiers(_event_modifiers(event))
        return current_handler(view_box, event, axis=axis)

    wheel_event._blochsimulator_axis_zoom = True
    pg.ViewBox.wheelEvent = wheel_event


# Fraction of the data extent that zooming out may add on each side. The X
# margin is small so that a time axis stays within 0 to the measurement time.
_DATA_LIMIT_X_MARGIN = 0.05
_DATA_LIMIT_Y_MARGIN = 0.5


def _data_limits(view_box):
    """Return (xLimits, yLimits) from the plotted data, ``None`` where unknown."""
    if not getattr(view_box, "auto_data_limits", True):
        return (None, None), (None, None)
    items = [item for item in view_box.addedItems if item.isVisible()]
    if not items:
        return (None, None), (None, None)
    try:
        bounds = view_box.childrenBounds()
    except Exception:
        return (None, None), (None, None)
    images_only = all(isinstance(item, pg.ImageItem) for item in items)
    limits = []
    for axis_bounds, margin, enabled in (
        (bounds[0], _DATA_LIMIT_X_MARGIN, True),
        (bounds[1], _DATA_LIMIT_Y_MARGIN, images_only),
    ):
        low, high = (None, None) if axis_bounds is None else axis_bounds
        if (
            not enabled
            or low is None
            or high is None
            or not np.isfinite(low)
            or not np.isfinite(high)
            or high <= low
        ):
            limits.append((None, None))
            continue
        pad = (high - low) * margin
        limits.append((low - pad, high + pad))
    return limits[0], limits[1]


def install_data_range_zoom_limit():
    """Keep zoom and pan of every ViewBox within the range of its data.

    Zooming out of a plot shows nothing new beyond the data (for a time axis
    the measurement), so the view range stops a small margin outside it.
    Explicit ``setLimits`` still apply; the tighter limit wins. A ViewBox with
    ``auto_data_limits = False`` keeps pyqtgraph's unrestricted zoom.
    """
    current = pg.ViewBox._effectiveLimits
    if getattr(current, "_blochsimulator_data_limits", False):
        return

    def effective_limits(view_box):
        explicit = current(view_box)
        automatic = _data_limits(view_box)
        merged = []
        for (low, high), (auto_low, auto_high) in zip(explicit, automatic):
            if auto_low is not None:
                low = auto_low if low is None else max(low, auto_low)
            if auto_high is not None:
                high = auto_high if high is None else min(high, auto_high)
            merged.append((low, high))
        return tuple(merged)

    effective_limits._blochsimulator_data_limits = True
    pg.ViewBox._effectiveLimits = effective_limits
