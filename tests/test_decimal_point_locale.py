"""The GUI shows "." as decimal separator regardless of the OS region."""

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDoubleSpinBox, QLineEdit

from blochsimulator.ui.widgets import install_decimal_point_locale


@pytest.fixture
def system_locale(request):
    QLocale.setDefault(QLocale(request.param))
    install_decimal_point_locale()
    yield request.param
    install_decimal_point_locale()


@pytest.mark.parametrize("system_locale", ["de_DE", "fr_FR", "en_US"], indirect=True)
def test_spin_boxes_use_decimal_point_for_any_system_locale(system_locale):
    spin = QDoubleSpinBox()
    spin.setDecimals(2)
    spin.setRange(0.0, 10000.0)
    spin.setValue(1234.5)
    assert spin.text() == "1234.50"

    spin.lineEdit().setText("12.5")
    spin.interpretText()
    assert spin.value() == 12.5
    spin.close()


@pytest.mark.parametrize("system_locale", ["de_DE"], indirect=True)
def test_comma_key_types_decimal_point_in_decimal_fields(system_locale):
    spin = QDoubleSpinBox()
    spin.setDecimals(3)
    spin.setRange(0.0, 100.0)
    spin.show()
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "1,5")
    assert spin.lineEdit().text() == "1.5"
    spin.interpretText()
    assert spin.value() == 1.5

    edit = QLineEdit()
    edit.setValidator(QDoubleValidator(edit))
    QTest.keyClicks(edit, "2,25")
    assert edit.text() == "2.25"

    # Plain text fields keep the comma.
    plain = QLineEdit()
    QTest.keyClicks(plain, "a,b")
    assert plain.text() == "a,b"
    for widget in (spin, edit, plain):
        widget.close()
