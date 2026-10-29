import pytest
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from blochsimulator.ui.plot_interaction import (
    install_axis_constrained_zoom,
    install_data_range_zoom_limit,
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


def test_zoom_out_stops_near_the_plotted_data():
    app = QApplication.instance() or QApplication([])
    install_data_range_zoom_limit()
    plot = pg.PlotWidget()
    plot.resize(600, 400)
    plot.show()
    plot.plot([0.0, 10.0], [0.0, 1.0])
    view_box = plot.getViewBox()
    app.processEvents()

    view_box.setRange(xRange=(-1e6, 1e6), padding=0.0)
    app.processEvents()

    # Curves limit the abscissa (time, frequency); the ordinate stays free so
    # plots with a deliberately fixed amplitude range keep it.
    x_min, x_max = view_box.viewRange()[0]
    assert -1.0 <= x_min <= 0.0 and 10.0 <= x_max <= 11.0
    plot.close()
    plot.deleteLater()
    app.processEvents()


def test_explicit_limits_and_opt_out_are_respected():
    app = QApplication.instance() or QApplication([])
    install_data_range_zoom_limit()
    plot = pg.PlotWidget()
    plot.resize(600, 400)
    plot.show()
    plot.plot([0.0, 10.0], [0.0, 1.0])
    view_box = plot.getViewBox()
    view_box.setLimits(xMin=0.0, xMax=10.0)
    view_box.setRange(xRange=(-100.0, 100.0), padding=0.0)
    app.processEvents()
    assert view_box.viewRange()[0] == pytest.approx([0.0, 10.0])

    view_box.setLimits(xMin=None, xMax=None)
    view_box.auto_data_limits = False
    view_box.setRange(xRange=(-100.0, 100.0), padding=0.0)
    app.processEvents()
    assert view_box.viewRange()[0] == pytest.approx([-100.0, 100.0])
    plot.close()
    plot.deleteLater()
    app.processEvents()


def test_tab_bar_wheel_slides_tabs_without_switching():
    from PyQt5.QtCore import QPoint, QPointF
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QLabel, QTabWidget

    from blochsimulator.ui.widgets import install_tab_bar_wheel_scroll

    app = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    for index in range(12):
        tabs.addTab(QLabel(str(index)), f"Tab number {index}")
    tabs.resize(300, 200)
    tabs.show()
    app.processEvents()
    install_tab_bar_wheel_scroll(tabs)
    bar = tabs.tabBar()
    first_x = bar.tabRect(0).x()

    def wheel(delta):
        event = QWheelEvent(
            QPointF(20, 10),
            QPointF(20, 10),
            QPoint(0, 0),
            QPoint(0, delta),
            0,
            Qt.Vertical,
            Qt.NoButton,
            Qt.NoModifier,
        )
        app.sendEvent(bar, event)
        app.processEvents()

    wheel(-240)
    assert bar.tabRect(0).x() < first_x
    assert tabs.currentIndex() == 0
    wheel(240)
    assert bar.tabRect(0).x() == first_x
