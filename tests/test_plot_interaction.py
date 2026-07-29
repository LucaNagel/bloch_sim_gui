import pytest
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from blochsimulator.ui.plot_interaction import (
    install_axis_constrained_zoom,
    zoom_axis_for_modifiers,
)


class _WheelEvent:
    def __init__(self, view_box, modifiers):
        self._view_box = view_box
        self._modifiers = modifiers
        self.accepted = False

    def delta(self):
        return 120

    def pos(self):
        return self._view_box.boundingRect().center()

    def modifiers(self):
        return self._modifiers

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


@pytest.mark.parametrize(
    ("modifiers", "expected_axis"),
    (
        (Qt.NoModifier, None),
        (Qt.ShiftModifier, pg.ViewBox.XAxis),
        (Qt.AltModifier, pg.ViewBox.YAxis),
        (Qt.ShiftModifier | Qt.AltModifier, None),
    ),
)
def test_zoom_axis_for_modifiers(modifiers, expected_axis):
    assert zoom_axis_for_modifiers(modifiers) == expected_axis


@pytest.mark.parametrize(
    ("modifiers", "x_changes", "y_changes"),
    (
        (Qt.NoModifier, True, True),
        (Qt.ShiftModifier, True, False),
        (Qt.AltModifier, False, True),
    ),
)
def test_trackpad_zoom_can_be_constrained_to_one_axis(modifiers, x_changes, y_changes):
    app = QApplication.instance() or QApplication([])
    install_axis_constrained_zoom()
    plot = pg.PlotWidget()
    plot.resize(600, 400)
    plot.show()
    app.processEvents()
    view_box = plot.getViewBox()
    view_box.setRange(xRange=(0.0, 10.0), yRange=(0.0, 20.0), padding=0.0)
    app.processEvents()
    before_x, before_y = view_box.viewRange()
    event = _WheelEvent(view_box, modifiers)

    view_box.wheelEvent(event)

    after_x, after_y = view_box.viewRange()
    assert (after_x != before_x) is x_changes
    assert (after_y != before_y) is y_changes
    assert event.accepted
    plot.close()
    plot.deleteLater()
    app.processEvents()
