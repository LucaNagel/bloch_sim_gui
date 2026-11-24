"""B0, nucleus and sequence reference ask before a user changes them."""

import pytest
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QApplication

from blochsimulator.ui import fundamental_inputs
from blochsimulator.ui.fundamental_inputs import (
    B0_REASON,
    NUCLEUS_REASON,
    ConfirmedComboBox,
    ConfirmedDoubleSpinBox,
)


@pytest.fixture
def answers(monkeypatch):
    """Record each question and answer it with the next queued reply."""
    state = {"replies": [], "asked": []}

    def confirm(_parent, label, old, new, _reason):
        state["asked"].append((label, old, new))
        return state["replies"].pop(0)

    monkeypatch.setattr(fundamental_inputs, "confirm_fundamental_change", confirm)
    return state


def _spin():
    spin = ConfirmedDoubleSpinBox("B0", B0_REASON)
    spin.setRange(0.01, 30.0)
    spin.setDecimals(4)
    spin.setSuffix(" T")
    spin.setValue(3.0)
    return spin


def _enter(widget):
    for kind in (QEvent.KeyPress, QEvent.KeyRelease):
        QApplication.sendEvent(
            widget, QKeyEvent(kind, Qt.Key_Return, Qt.NoModifier, "\r")
        )


def test_programmatic_changes_are_not_asked(answers):
    spin = _spin()
    spin.setValue(7.0)
    combo = ConfirmedComboBox("the nucleus", NUCLEUS_REASON)
    combo.addItems(["H1", "C13"])
    combo.setCurrentText("C13")
    assert spin.value() == 7.0 and combo.currentText() == "C13"
    assert answers["asked"] == []


def test_arrow_steps_need_confirmation(answers):
    spin = _spin()
    answers["replies"] = [False, True]
    spin.stepBy(1)
    assert spin.value() == pytest.approx(3.0)
    spin.stepBy(1)
    assert spin.value() == pytest.approx(4.0)
    assert answers["asked"] == [("B0", "3.0000 T", "4.0000 T")] * 2


def test_typed_value_needs_confirmation_on_enter(answers):
    spin = _spin()
    spin.show()
    answers["replies"] = [False]
    spin.lineEdit().setText("9.4 T")
    _enter(spin)
    assert spin.value() == pytest.approx(3.0)
    assert spin.text() == "3.0000 T"

    answers["replies"] = [True]
    spin.lineEdit().setText("9.4 T")
    _enter(spin)
    assert spin.value() == pytest.approx(9.4)
    assert [new for _, _, new in answers["asked"]] == ["9.4000 T", "9.4000 T"]
    spin.close()


def test_wheel_never_changes_the_value(answers):
    spin = _spin()
    combo = ConfirmedComboBox("the nucleus", NUCLEUS_REASON)
    combo.addItems(["H1", "C13"])
    wheel = QWheelEvent(
        QPointF(1, 1),
        QPointF(1, 1),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    spin.wheelEvent(wheel)
    combo.wheelEvent(wheel)
    assert spin.value() == pytest.approx(3.0) and combo.currentIndex() == 0
    assert answers["asked"] == []


def test_popup_choice_needs_confirmation(answers):
    combo = ConfirmedComboBox("the nucleus", NUCLEUS_REASON)
    combo.addItems(["H1", "C13", "Xe129"])
    changes = []
    combo.currentTextChanged.connect(changes.append)
    combo.show()
    combo.showPopup()
    view = combo.view()
    rect = view.visualRect(view.model().index(1, 0))

    def release():
        QApplication.sendEvent(
            view.viewport(),
            QMouseEvent(
                QEvent.MouseButtonRelease,
                rect.center(),
                Qt.LeftButton,
                Qt.NoButton,
                Qt.NoModifier,
            ),
        )

    answers["replies"] = [False]
    release()
    assert combo.currentText() == "H1" and changes == []

    combo.showPopup()
    answers["replies"] = [True]
    release()
    assert combo.currentText() == "C13" and changes == ["C13"]
    assert answers["asked"] == [("the nucleus", "H1", "C13")] * 2
    combo.close()


def test_arrow_keys_open_the_popup_instead_of_changing(answers):
    combo = ConfirmedComboBox("the nucleus", NUCLEUS_REASON)
    combo.addItems(["H1", "C13"])
    combo.show()
    QApplication.sendEvent(
        combo, QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
    )
    QApplication.sendEvent(
        combo, QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.NoModifier, "c")
    )
    assert combo.currentText() == "H1"
    assert answers["asked"] == []
    combo.hidePopup()
    combo.close()


def test_application_fields_use_confirmed_inputs():
    from blochsimulator.ui.mouse_phantom_designer import MousePhantomDesignerDialog
    from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    sequence = SequenceSimulationWidget()
    designer = SpectralPhantomDesignerDialog()
    mouse = MousePhantomDesignerDialog()
    widgets = [
        sequence.nucleus,
        sequence.sequence_reference_ppm,
        sequence.field_strength_t,
        designer.nucleus,
        designer.field_strength_t,
        mouse.nucleus,
        mouse.field_strength,
    ]
    assert all(
        isinstance(w, (ConfirmedComboBox, ConfirmedDoubleSpinBox)) for w in widgets
    )
