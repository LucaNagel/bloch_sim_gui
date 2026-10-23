import sys

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

from blochsimulator.ui.spin_minimum import ZeroToMinimum, minimum_ms_from_message


def _spin(minimum, maximum, value, decimals=2):
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


def test_smallest_feasible_bisects_and_uses_hints():
    app = QApplication.instance() or QApplication(sys.argv)
    spin = _spin(0.1, 100.0, 15.0)
    calls = []

    def feasible(control):
        calls.append(control.value())
        return control.value() >= 7.834

    plain = ZeroToMinimum(feasible)
    assert plain.smallest_feasible(spin) == pytest.approx(7.84)
    assert spin.value() == 15.0
    bisection_calls = len(calls)

    calls.clear()
    hinted = ZeroToMinimum(feasible, hint=lambda control: 7.834)
    assert hinted.smallest_feasible(spin) == pytest.approx(7.84)
    assert len(calls) < bisection_calls

    # A wrong hint still ends at the true minimum.
    assert ZeroToMinimum(feasible, hint=lambda c: 50.0).smallest_feasible(
        spin
    ) == pytest.approx(7.84)
    assert ZeroToMinimum(lambda c: False).smallest_feasible(spin) is None
    assert ZeroToMinimum(lambda c: True).smallest_feasible(spin) == pytest.approx(0.1)


def test_typed_zero_sets_the_minimum_only_where_zero_is_invalid():
    app = QApplication.instance() or QApplication(sys.argv)
    tr = _spin(0.1, 100.0, 15.0)
    offset = _spin(-10.0, 10.0, 3.0)
    count = QSpinBox()
    count.setRange(1, 64)
    count.setValue(32)
    changes = []
    tr.valueChanged.connect(changes.append)
    zero = ZeroToMinimum(lambda c: c.value() >= (8.0 if c is tr else 4))
    zero.install([tr, offset, count])
    for spin in (tr, offset, count):
        spin.show()
        spin.setFocus()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, "0")
        QTest.keyClick(spin, Qt.Key_Return)
        app.processEvents()
    assert tr.value() == pytest.approx(8.0)
    # Qt may repeat the pending edit's signal on Return; 8 is the only value.
    assert set(changes) == {8.0}
    assert offset.value() == 0.0
    assert count.value() == 4


def test_minimum_from_builder_messages():
    assert minimum_ms_from_message(
        "TR 6.290 ms is too short to center the ADC at TR/2; the minimum is 6.730 ms."
    ) == pytest.approx(6.73)
    assert minimum_ms_from_message(
        "repetition_time_s is too short; minimum is 0.00783 s"
    ) == pytest.approx(7.83)
    assert minimum_ms_from_message("flip angle must be positive") is None
