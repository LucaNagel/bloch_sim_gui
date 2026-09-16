"""Tests for predictable mouse-wheel behavior on numeric inputs."""

import pytest
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from blochsimulator.ui.widgets import SpinBoxWheelGuard


def _wheel_event(target, delta, modifiers=Qt.NoModifier):
    position = target.rect().center()
    return QWheelEvent(
        QPointF(position),
        QPointF(target.mapToGlobal(position)),
        QPoint(),
        QPoint(0, delta),
        Qt.NoButton,
        modifiers,
        Qt.NoScrollPhase,
        False,
    )


@pytest.mark.parametrize("spin_box_type", [QSpinBox, QDoubleSpinBox])
def test_spin_box_wheel_requires_control_and_scrolls_form(
    qt_application, spin_box_type
):
    guard = SpinBoxWheelGuard()
    qt_application.installEventFilter(guard)

    scroll_area = QScrollArea()
    scroll_area.resize(240, 120)
    content = QWidget()
    layout = QVBoxLayout(content)
    spin_box = spin_box_type()
    spin_box.setRange(-100, 100)
    spin_box.setSingleStep(1)
    spin_box.setValue(5)
    layout.addWidget(spin_box)
    for index in range(20):
        layout.addWidget(QLabel(f"Row {index}"))
    scroll_area.setWidget(content)
    scroll_area.setWidgetResizable(True)
    scroll_area.show()
    qt_application.processEvents()

    try:
        target = spin_box.lineEdit()
        QApplication.sendEvent(target, _wheel_event(target, -120))

        assert spin_box.value() == pytest.approx(5)
        assert scroll_area.verticalScrollBar().value() > 0

        scroll_area.verticalScrollBar().setValue(0)
        QApplication.sendEvent(
            target,
            _wheel_event(target, 120, Qt.ControlModifier),
        )

        assert spin_box.value() == pytest.approx(6)
        assert scroll_area.verticalScrollBar().value() == 0
    finally:
        qt_application.removeEventFilter(guard)
        scroll_area.close()


def test_unmodified_wheel_does_not_change_standalone_spin_box(qt_application):
    guard = SpinBoxWheelGuard()
    qt_application.installEventFilter(guard)
    spin_box = QSpinBox()
    spin_box.setValue(5)
    spin_box.show()
    qt_application.processEvents()

    try:
        QApplication.sendEvent(spin_box, _wheel_event(spin_box, 120))
        assert spin_box.value() == 5
    finally:
        qt_application.removeEventFilter(guard)
        spin_box.close()
