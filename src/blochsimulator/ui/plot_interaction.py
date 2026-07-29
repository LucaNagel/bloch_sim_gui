"""Shared interaction behavior for pyqtgraph plots."""

from __future__ import annotations

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
