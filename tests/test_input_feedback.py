"""Tests for the application-wide input feedback on editable fields."""

import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDoubleSpinBox

from blochsimulator.ui.widgets import install_input_feedback


@pytest.fixture
def millimetre_spin(qt_application):
    install_input_feedback(qt_application)
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1000.0)
    spin.setDecimals(2)
    spin.setSuffix(" mm")
    spin.setValue(32.0)
    yield spin
    spin.deleteLater()


def test_decimal_point_replaces_the_old_fraction(millimetre_spin):
    spin = millimetre_spin
    # A double click selects only the integer part of "32.00 mm".
    spin.lineEdit().setSelection(0, 2)
    QTest.keyClicks(spin, "64.30")
    assert spin.lineEdit().text() == "64.30 mm"
    assert spin.value() == pytest.approx(64.3)


def test_comma_types_the_decimal_point(millimetre_spin):
    spin = millimetre_spin
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "7,5")
    assert spin.value() == pytest.approx(7.5)


def test_decimal_point_after_the_fraction_changes_nothing(millimetre_spin):
    spin = millimetre_spin
    spin.lineEdit().end(False)
    QTest.keyClicks(spin, ".")
    assert spin.lineEdit().text() == "32.00 mm"
