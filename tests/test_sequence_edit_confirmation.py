"""Keyboard and tab navigation must preserve unapplied editor changes."""

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialogButtonBox, QMessageBox

from blochsimulator.b1_fields import B1Field
from blochsimulator.mouse_phantom import MousePerfusionConfig
from blochsimulator.phantom import PhantomFactory
from blochsimulator.phantom_design import PhantomDesign, ShapeDefinition
from blochsimulator.ui.b1_widgets import B1WorkspaceWidget
from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.mouse_phantom_designer import MousePhantomDesignerDialog
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog


@pytest.mark.parametrize("kind", ["transmit", "receive"])
@pytest.mark.parametrize("destination", ["editor", "workspace"])
@pytest.mark.parametrize(
    "answer", [QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel]
)
def test_b1_tab_change_confirms_unapplied_fields(
    qt_application, monkeypatch, kind, destination, answer
):
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    window.tab_widget.setCurrentIndex(window.b1_tab_index)
    workspace = window.b1_widget
    workspace.auto_apply.setChecked(False)
    phantom = PhantomFactory.uniform((3, 3, 2), (0.03, 0.03, 0.02), 1.0, 0.1)
    workspace.set_phantom(phantom)
    workspace.apply_fields()
    editor_index = 0 if kind == "transmit" else 1
    workspace.editor_tabs.setCurrentIndex(editor_index)
    editor = workspace.tx_editor if kind == "transmit" else workspace.rx_editor
    editor.set_field(B1Field.uniform(phantom.shape, phantom.fov, kind=kind, value=0.5j))
    questions = []

    def answer_question(*args):
        questions.append(args)
        return answer

    monkeypatch.setattr(QMessageBox, "question", answer_question)
    if destination == "editor":
        workspace.editor_tabs.setCurrentIndex(1 - editor_index)
        expected_index = (
            editor_index if answer == QMessageBox.Cancel else 1 - editor_index
        )
        assert workspace.editor_tabs.currentIndex() == expected_index
        assert workspace.preview_tabs.currentIndex() == expected_index
    else:
        window.tab_widget.setCurrentIndex(window.simulation_explorer_tab_index)
        expected_index = (
            window.b1_tab_index
            if answer == QMessageBox.Cancel
            else window.simulation_explorer_tab_index
        )
        assert window.tab_widget.currentIndex() == expected_index

    assert len(questions) == 1
    applied_map = (
        phantom.tx_sensitivity_map
        if kind == "transmit"
        else phantom.rx_sensitivity_maps
    )
    np.testing.assert_allclose(applied_map, 0.5j if answer == QMessageBox.Yes else 1.0)
    assert workspace._fields_pending == (answer != QMessageBox.Yes)
    assert editor.field is not None
    window.close()


def test_b1_apply_button_clears_pending_geometry_and_unity_changes(
    qt_application, monkeypatch
):
    workspace = B1WorkspaceWidget()
    workspace.auto_apply.setChecked(False)
    phantom = PhantomFactory.uniform((3, 3, 2), (0.03, 0.03, 0.02), 1.0, 0.1)
    workspace.set_phantom(phantom)
    workspace.tx_editor.set_field(
        B1Field.uniform(phantom.shape, phantom.fov, value=0.7)
    )
    workspace.apply_button.click()
    workspace.tx_editor.rotation_spins[2].setValue(30.0)
    assert workspace._fields_pending
    workspace.apply_button.click()
    workspace.tx_editor.clear_button.click()
    assert workspace._fields_pending
    workspace.apply_button.click()

    def unexpected_question(*args):
        pytest.fail("Applied fields must not trigger a tab-change question")

    monkeypatch.setattr(QMessageBox, "question", unexpected_question)
    workspace.editor_tabs.setCurrentIndex(1)
    assert workspace.editor_tabs.currentIndex() == 1
    np.testing.assert_allclose(phantom.tx_sensitivity_map, 1.0)
    assert not workspace._fields_pending
    workspace.close()


def test_b1_disabling_auto_apply_stops_scheduled_application(qt_application):
    workspace = B1WorkspaceWidget()
    phantom = PhantomFactory.uniform((3, 3, 2), (0.03, 0.03, 0.02), 1.0, 0.1)
    workspace.set_phantom(phantom)
    workspace.tx_editor.set_field(
        B1Field.uniform(phantom.shape, phantom.fov, value=0.7)
    )
    assert workspace._apply_timer.isActive()
    workspace.auto_apply.setChecked(False)
    QTest.qWait(150)
    np.testing.assert_allclose(phantom.tx_sensitivity_map, 1.0)
    assert workspace._fields_pending
    workspace.close()


def test_b1_auto_apply_finishes_before_tab_change(qt_application, monkeypatch):
    workspace = B1WorkspaceWidget()
    phantom = PhantomFactory.uniform((3, 3, 2), (0.03, 0.03, 0.02), 1.0, 0.1)
    workspace.set_phantom(phantom)
    workspace.tx_editor.set_field(
        B1Field.uniform(phantom.shape, phantom.fov, value=0.7)
    )
    applied = []
    workspace.phantom_updated.connect(applied.append)

    def unexpected_question(*args):
        pytest.fail("Automatic application must not need confirmation")

    monkeypatch.setattr(QMessageBox, "question", unexpected_question)
    workspace.editor_tabs.setCurrentIndex(1)
    np.testing.assert_allclose(phantom.tx_sensitivity_map, 0.7)
    assert workspace.editor_tabs.currentIndex() == 1
    assert not workspace._apply_timer.isActive()
    QTest.qWait(150)
    assert applied == [phantom]
    workspace.close()


def test_b1_application_failure_keeps_edits_and_current_tab(
    qt_application, monkeypatch
):
    workspace = B1WorkspaceWidget()
    workspace.auto_apply.setChecked(False)
    phantom = PhantomFactory.uniform((3, 3, 2), (0.03, 0.03, 0.02), 1.0, 0.1)
    workspace.set_phantom(phantom)
    field = B1Field.uniform(phantom.shape, phantom.fov, value=0.7)
    workspace.tx_editor.set_field(field)

    def fail_resampling(_phantom):
        raise ValueError("Invalid test geometry")

    monkeypatch.setattr(field, "resample_to_phantom", fail_resampling)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    workspace.editor_tabs.setCurrentIndex(1)

    assert workspace.editor_tabs.currentIndex() == 0
    assert workspace.preview_tabs.currentIndex() == 0
    assert workspace._fields_pending
    np.testing.assert_allclose(phantom.tx_sensitivity_map, 1.0)
    assert "Invalid test geometry" in warnings[0][2]
    workspace.close()


@pytest.fixture(params=["spectral", "mouse"])
def phantom_editor(request, qt_application):
    if request.param == "spectral":
        dialog = SpectralPhantomDesignerDialog(
            design=PhantomDesign(shape=(4, 4, 2), shapes=[ShapeDefinition(name="Test")])
        )
        spin = dialog.field_strength_t
    else:
        dialog = MousePhantomDesignerDialog(
            config=MousePerfusionConfig(shape=(8, 8, 16))
        )
        spin = dialog.field_strength
    dialog.open()
    qt_application.processEvents()
    yield dialog, spin
    dialog.close()


def test_phantom_enter_commits_inputs_and_escape_closes(phantom_editor):
    dialog, spin = phantom_editor
    accepted = []
    rejected = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.rejected.connect(lambda: rejected.append(True))
    for key in (Qt.Key_Return, Qt.Key_Enter):
        spin.setFocus()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "7.5")
        QTest.keyClick(spin, key)
        assert spin.value() == pytest.approx(7.5)
        assert dialog.isVisible()
        dialog.name_edit.setFocus()
        QTest.keyClick(dialog.name_edit, key)
        assert dialog.isVisible()
        buttons = dialog.findChild(QDialogButtonBox)
        ok = buttons.button(QDialogButtonBox.Ok)
        ok.setFocus()
        QTest.keyClick(ok, key)
        assert dialog.isVisible()
        QTest.keyClick(dialog, key)
        assert dialog.isVisible()

    assert accepted == []
    assert rejected == []
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key_Escape)
    assert not dialog.isVisible()
    assert rejected == [True]


def test_phantom_ok_button_still_accepts(phantom_editor, monkeypatch):
    dialog, _spin = phantom_editor
    phantom = SimpleNamespace(metadata={})
    if isinstance(dialog, SpectralPhantomDesignerDialog):
        monkeypatch.setattr(dialog.design, "build", lambda: phantom)
    else:
        monkeypatch.setattr(dialog, "_build_phantom", lambda: phantom)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    buttons = dialog.findChild(QDialogButtonBox)
    QTest.mouseClick(buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)
    assert accepted == [True]
    assert dialog.get_phantom() is phantom
    assert not dialog.isVisible()
