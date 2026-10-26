import time

import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMessageBox

from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.ui.widget_state import capture_widget_state, restore_widget_state


def _ss_bssfp_widget():
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.SS_BSSFP_SOURCE)
    return widget


def _loop_parameters(widget):
    return [
        widget.parameter_loop_parameter.itemData(index)
        for index in range(widget.parameter_loop_parameter.count())
    ]


def test_target_table_mirrors_the_stored_target_lists(qt_application):
    widget = _ss_bssfp_widget()
    table = widget.ss_bssfp_target_table

    names, values = table.targets()
    assert names == ["Lac", "Py"]
    assert values["rf_offset_hz"] == pytest.approx([1655.0, -245.0])
    assert values["receiver_offset_hz"] == pytest.approx([925.44725, 0.0])
    assert values["flip_angle_deg"] == pytest.approx([90.0, 4.0])
    assert values["startup_flip_angle_deg"] == pytest.approx([45.0, 2.0])

    table.rows[1].spins["flip_angle_deg"].setValue(7.5)
    table.rows[0].spins["rf_offset_hz"].setValue(1600.0)
    parameters = widget._ss_bssfp_pulseq_parameters()
    assert parameters["flip_angle_deg"] == pytest.approx((90.0, 7.5))
    assert parameters["target_frequency_offsets_hz"] == pytest.approx((1600.0, -245.0))

    widget.ss_bssfp_target_names.setText("Lac, Py, Ala")
    widget.ss_bssfp_target_offsets_hz.setText("1655, -245, 300")
    assert len(table.rows) == 3
    assert table.rows[2].name.text() == "Ala"
    assert table.rows[2].spins["rf_offset_hz"].value() == pytest.approx(300.0)

    # Removing a metabolite rewrites every stored list consistently.
    table.rows[1].remove.click()
    assert widget.ss_bssfp_target_names.text() == "Lac, Ala"
    assert widget.ss_bssfp_target_offsets_hz.text() == "1655, 300"
    table.add_button.click()
    assert widget.ss_bssfp_target_names.text() == "Lac, Ala, Target 3"
    widget.deleteLater()


def test_target_table_follows_restored_project_state(qt_application):
    widget = _ss_bssfp_widget()
    widget.ss_bssfp_flip_angles_deg.setText("30, 12")
    state = capture_widget_state(widget)
    other = _ss_bssfp_widget()
    restore_widget_state(other, state, block_signals=True)
    other.refresh_after_session_control_restore()

    assert other.ss_bssfp_target_table.targets()[1]["flip_angle_deg"] == (
        pytest.approx([30.0, 12.0])
    )
    widget.deleteLater()
    other.deleteLater()


def test_per_metabolite_values_are_loop_parameters(qt_application):
    widget = _ss_bssfp_widget()
    widget.show()
    QTest.qWait(10)
    labels = {parameter.key: parameter.label for parameter in _loop_parameters(widget)}

    assert labels["ss_bssfp_target[0].flip_angle_deg"] == "Lac: Target flip angle"
    assert labels["ss_bssfp_target[1].rf_offset_hz"] == "Py: RF target offset"
    assert labels["ss_bssfp_target[1].receiver_offset_hz"] == "Py: Receiver offset"
    assert "ss_bssfp_target[1].startup_flip_angle_deg" in labels

    # Startup flips only exist when explicit startup angles are used.
    widget.ss_bssfp_alpha_half_use_ratios.setChecked(True)
    widget._refresh_parameter_loop_parameters()
    keys = {parameter.key for parameter in _loop_parameters(widget)}
    assert "ss_bssfp_target[1].startup_flip_angle_deg" not in keys
    assert "ss_bssfp_target[1].flip_angle_deg" in keys
    widget.close()
    widget.deleteLater()


def test_loop_over_one_metabolite_flip_angle_regenerates_the_sequence(
    qt_application, monkeypatch
):
    pytest.importorskip("pypulseq")
    widget = _ss_bssfp_widget()
    widget.show()
    # Accept the FOV/resolution warnings of the coarse built-in phantom.
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: QMessageBox.Yes)
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget.ss_bssfp_read_matrix.setValue(2)
    widget.ss_bssfp_phase_matrix.setValue(2)
    widget.ss_bssfp_partition_matrix.setValue(2)
    widget.ss_bssfp_repetitions.setValue(1)
    QTest.qWait(10)
    widget._refresh_parameter_loop_parameters()
    parameters = _loop_parameters(widget)
    index = next(
        index
        for index, parameter in enumerate(parameters)
        if parameter.key == "ss_bssfp_target[1].flip_angle_deg"
    )
    widget.parameter_loop_parameter.setCurrentIndex(index)
    widget.parameter_loop_start.setValue(5.0)
    widget.parameter_loop_stop.setValue(10.0)
    widget.parameter_loop_steps.setValue(2)
    flips = []
    register = widget._register_session_simulation_run

    def capture(result, **kwargs):
        flips.append(widget._ss_bssfp_pulseq_parameters()["flip_angle_deg"])
        return register(result, **kwargs)

    monkeypatch.setattr(widget, "_register_session_simulation_run", capture)
    widget._start_parameter_loop()
    deadline = time.monotonic() + 60.0
    while widget._parameter_loop_state is not None and time.monotonic() < deadline:
        QTest.qWait(10)

    assert widget._parameter_loop_state is None
    assert flips == [pytest.approx((90.0, 5.0)), pytest.approx((90.0, 10.0))]
    assert widget.ss_bssfp_flip_angles_deg.text() == "90, 10"
    widget.close()
    widget.deleteLater()
