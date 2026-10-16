import time

import numpy as np
import pytest
from PyQt5.QtTest import QTest

from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
from blochsimulator.sequence.parameter_loop import (
    MAX_PARAMETER_LOOP_POINTS,
    manual_parameter_loop_values,
    parameter_loop_values,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("x", lambda x: x),
        ("sin(x)", np.sin),
        ("np.sin(x)", np.sin),
        ("x^2", lambda x: x**2),
        ("y = x**2", lambda x: x**2),
        ("2 + cos(pi*x)/2", lambda x: 2 + np.cos(np.pi * x) / 2),
        ("exp(-x)", lambda x: np.exp(-x)),
        ("clip(x, 1, 4)", lambda x: np.clip(x, 1, 4)),
        ("3", lambda x: np.full(x.shape, 3.0)),
    ],
)
def test_formula_maps_evenly_spaced_x_values(formula, expected):
    x = np.linspace(0, 5, 11)
    np.testing.assert_allclose(parameter_loop_values(formula, 0, 5, 11), expected(x))


def test_formula_preserves_descending_and_repeated_points():
    np.testing.assert_allclose(
        parameter_loop_values("sin(x)", 5, 0, 11), np.sin(np.linspace(5, 0, 11))
    )
    np.testing.assert_allclose(parameter_loop_values("x^2", 2, 2, 3), [4, 4, 4])
    np.testing.assert_allclose(parameter_loop_values("x^2", 2, 5, 1), [4])


@pytest.mark.parametrize(
    "formula",
    [
        "",
        "sin(",
        "unknown(x)",
        "sqrt(-x)",
        "1 / (x - 1)",
        "exp(1000*x)",
        "__import__('os').getcwd()",
        "x.__class__",
        "np.load('data.npy')",
        "[x for x in (1, 2)]",
        "True",
        "2j",
        "sin(x, x)",
    ],
)
def test_formula_rejects_invalid_values_and_python_code(formula):
    with pytest.raises(ValueError):
        parameter_loop_values(formula, 0, 2, 3)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.5, 1, 1.75", [0.5, 1, 1.75]),
        ("3, -2, 3, 1e-3", [3, -2, 3, 0.001]),
        ("  -2.5  ", [-2.5]),
        (" 1e3,\n +2.5E-2, 0 ", [1000, 0.025, 0]),
        (",".join(["1"] * MAX_PARAMETER_LOOP_POINTS), [1] * MAX_PARAMETER_LOOP_POINTS),
    ],
)
def test_manual_values_preserve_entered_numbers_and_order(text, expected):
    np.testing.assert_array_equal(manual_parameter_loop_values(text), expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "1,",
        ",1",
        "1, ,2",
        "1; 2",
        "1, nan",
        "inf, 1",
        "-inf",
        "1e309",
        "1, 2j",
        "1, sin(2)",
        "__import__('os').getcwd()",
        ",".join(["1"] * (MAX_PARAMETER_LOOP_POINTS + 1)),
    ],
)
def test_manual_values_reject_invalid_or_oversized_lists(text):
    with pytest.raises(ValueError):
        manual_parameter_loop_values(text)


def _loop_parameters(widget):
    return [
        widget.parameter_loop_parameter.itemData(index)
        for index in range(widget.parameter_loop_parameter.count())
    ]


def test_manual_values_in_parameter_loop_tab(qt_application):
    from blochsimulator.ui.main_window import BlochSimulatorGUI

    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.open_parameter_loop_button.click()
    qt_application.processEvents()

    assert window.tab_widget.currentIndex() == window.sequence_parameter_loop_tab_index
    assert widget.parameter_loop_input_mode.currentData() == "formula"
    assert not widget.parameter_loop_manual_values.isEnabled()
    widget.parameter_loop_formula.setText("x^2")
    widget.parameter_loop_start.setValue(0)
    widget.parameter_loop_stop.setValue(2)
    widget.parameter_loop_steps.setValue(3)
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData("manual")
    )
    widget.parameter_loop_manual_values.setText("1.5, 0.5, 1.5, 2")

    assert widget.parameter_loop_manual_values.isEnabled()
    assert not any(
        control.isEnabled()
        for control in (
            widget.parameter_loop_formula,
            widget.parameter_loop_start,
            widget.parameter_loop_stop,
            widget.parameter_loop_steps,
        )
    )
    assert widget._parameter_loop_values(
        widget.parameter_loop_parameter.currentData()
    ).tolist() == [1.5, 0.5, 1.5, 2]
    assert widget.parameter_loop_preview.text() == "1.5, 0.5, 1.5, 2 ×"
    x, y = widget.parameter_loop_plot_curve.getData()
    np.testing.assert_array_equal(x, [0, 1, 2, 3])
    np.testing.assert_array_equal(y, [1.5, 0.5, 1.5, 2])
    assert widget.parameter_loop_plot.getAxis("left").labelUnits == "×"

    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData("formula")
    )
    assert widget.parameter_loop_formula.isEnabled()
    assert not widget.parameter_loop_manual_values.isEnabled()
    assert widget.parameter_loop_preview.text() == "0, 1, 4 ×"
    x, y = widget.parameter_loop_plot_curve.getData()
    np.testing.assert_array_equal(x, [0, 1, 2])
    np.testing.assert_array_equal(y, [0, 1, 4])
    assert widget.parameter_loop_manual_values.text() == "1.5, 0.5, 1.5, 2"
    window.close()
    window.deleteLater()
    qt_application.processEvents()


def test_parameter_loop_discovers_controls_for_every_sequence_source(qt_application):
    widget = SequenceSimulationWidget()
    expected_flip_controls = {
        widget.EPI_SOURCE: "epi_flip_angle_deg",
        widget.CSI_SOURCE: "csi_flip_angle_deg",
        widget.BSSFP_SOURCE: "bssfp_flip_angle_deg",
        widget.RADIAL_ME_BSSFP_SOURCE: "radial_me_flip_angle_deg",
        widget.ME_BSSFP_SOURCE: "me_bssfp_flip_angle_deg",
        widget.FLASH_SOURCE: "flash_flip_angle_deg",
        widget.EPSI_MGE_SOURCE: "epsi_flip_angle_deg",
    }

    for source_index in range(widget.sequence_source.count()):
        widget.sequence_source.setCurrentIndex(source_index)
        qt_application.processEvents()
        parameters = _loop_parameters(widget)
        keys = {parameter.key for parameter in parameters}
        assert "__rf_scale__" in keys
        assert "sequence_reference_ppm" in keys
        assert "simulation_timestep_us" in keys
        if source_index in expected_flip_controls:
            assert expected_flip_controls[source_index] in keys

    # SS-bSSFP stores a vector of target flip angles in a text field. The
    # universal RF scale keeps this sequence loopable without special casing.
    widget.sequence_source.setCurrentIndex(widget.SS_BSSFP_SOURCE)
    assert _loop_parameters(widget)[0].key == "__rf_scale__"

    # Conditional controls enter and leave the generic list with their form row.
    widget.sequence_source.setCurrentIndex(widget.BSSFP_SOURCE)
    assert "bssfp_alpha_half_tr_ratio" in {
        parameter.key for parameter in _loop_parameters(widget)
    }
    widget.bssfp_alpha_half.setChecked(False)
    qt_application.processEvents()
    assert "bssfp_alpha_half_tr_ratio" not in {
        parameter.key for parameter in _loop_parameters(widget)
    }


def test_parameter_loop_action_stays_available_while_sequence_is_pending(
    qt_application,
):
    widget = SequenceSimulationWidget()
    widget.program = None

    widget.prepare_parameter_loop_workspace()

    assert widget.parameter_loop_parameter.count() > 0
    assert widget.parameter_loop_button.isEnabled()


def test_integer_sequence_parameters_produce_integer_loop_values(qt_application):
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    parameters = _loop_parameters(widget)
    matrix_index = next(
        index
        for index, parameter in enumerate(parameters)
        if parameter.key == "read_matrix"
    )
    widget.parameter_loop_parameter.setCurrentIndex(matrix_index)
    widget.parameter_loop_start.setValue(4)
    widget.parameter_loop_stop.setValue(8)
    widget.parameter_loop_steps.setValue(3)

    parameter = widget.parameter_loop_parameter.currentData()
    values = widget._parameter_loop_values(parameter)

    assert parameter.integer
    assert values.dtype.kind in "iu"
    assert values.tolist() == [4, 6, 8]


def test_manual_integer_values_are_rounded_without_reordering(qt_application):
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    matrix_index = next(
        index
        for index, parameter in enumerate(_loop_parameters(widget))
        if parameter.key == "read_matrix"
    )
    widget.parameter_loop_parameter.setCurrentIndex(matrix_index)
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData("manual")
    )
    widget.parameter_loop_manual_values.setText("514.2, 4.1, 7.9, 4.1")
    parameter = widget.parameter_loop_parameter.currentData()

    assert widget._parameter_loop_values(parameter).tolist() == [514, 4, 8, 4]
    assert widget.parameter_loop_preview.text() == "514, 4, 8, 4"
    np.testing.assert_array_equal(
        widget.parameter_loop_plot_curve.getData()[1], [514, 4, 8, 4]
    )
    assert widget.parameter_loop_plot.getAxis("left").labelUnits == ""
    widget.parameter_loop_manual_values.setText("2147483648")
    with pytest.raises(ValueError, match="32-bit integer"):
        widget._parameter_loop_values(parameter)
    assert widget.parameter_loop_plot_curve.getData()[1] is None


def test_formula_x_range_is_independent_of_integer_parameter_range(qt_application):
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    matrix_index = next(
        index
        for index, parameter in enumerate(_loop_parameters(widget))
        if parameter.key == "read_matrix"
    )
    widget.parameter_loop_parameter.setCurrentIndex(matrix_index)
    widget.parameter_loop_formula.setText("4 + x^2")
    widget.parameter_loop_start.setValue(0)
    widget.parameter_loop_stop.setValue(2)
    widget.parameter_loop_steps.setValue(5)

    values = widget._parameter_loop_values(
        widget.parameter_loop_parameter.currentData()
    )

    assert widget.parameter_loop_start.value() == 0
    assert values.tolist() == [4, 4, 5, 6, 8]
    assert widget.parameter_loop_preview.text() == "4, 4, 5, 6, 8"
    np.testing.assert_array_equal(
        widget.parameter_loop_plot_curve.getData()[1], [4, 4, 5, 6, 8]
    )


@pytest.mark.parametrize(
    "text, expected",
    [("-2", [-2]), ("3, -2, 3, 0.001", [3, -2, 3, 0.001])],
)
def test_loop_plot_updates_and_clears_invalid_input(qt_application, text, expected):
    widget = SequenceSimulationWidget()
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData("manual")
    )
    widget.parameter_loop_manual_values.setText(text)
    x, y = widget.parameter_loop_plot_curve.getData()
    np.testing.assert_array_equal(x, np.arange(len(expected)))
    assert widget.parameter_loop_plot.getAxis("bottom").labelText == "Index"
    assert widget.parameter_loop_plot.getAxis("left").labelText == "Parameter value"
    np.testing.assert_array_equal(y, expected)
    widget.parameter_loop_manual_values.setText("1,,2")
    assert widget.parameter_loop_plot_curve.getData()[1] is None
    widget.parameter_loop_manual_values.setText("2, 1")
    np.testing.assert_array_equal(widget.parameter_loop_plot_curve.getData()[1], [2, 1])


@pytest.mark.parametrize(
    "input_mode, text",
    [
        ("formula", "sqrt(-x)"),
        ("formula", "1/(x-x)"),
        ("manual", ""),
        ("manual", "1,,2"),
    ],
)
def test_invalid_input_does_not_start_a_loop(
    qt_application, monkeypatch, input_mode, text
):
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData(input_mode)
    )
    input_field = (
        widget.parameter_loop_manual_values
        if input_mode == "manual"
        else widget.parameter_loop_formula
    )
    input_field.setText(text)
    widget.parameter_loop_start.setValue(0)
    widget.parameter_loop_stop.setValue(5)
    messages = []
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QMessageBox.warning",
        lambda _parent, _title, message: messages.append(message),
    )

    widget._start_parameter_loop()

    assert messages
    assert widget._parameter_loop_state is None
    assert widget.worker is None
    assert input_field.isEnabled()


def test_universal_rf_scale_preserves_timing_and_adc():
    program = SequenceProgram(
        events=(
            RFEvent(0.0, np.array([1.0 + 2.0j, -3.0j]), 1e-3),
            RFEvent(3e-3, np.array([4.0]), 1e-3),
            ADCEvent(5e-3, 2, 1e-3),
        ),
        duration_s=6e-3,
        source="test",
    )

    scaled = SequenceSimulationWidget._program_with_rf_scale(program, 1.5)

    assert scaled.duration_s == program.duration_s
    assert scaled.adc_events == program.adc_events
    assert np.allclose(
        scaled.rf_events[0].samples_hz, 1.5 * program.rf_events[0].samples_hz
    )
    assert np.allclose(
        scaled.rf_events[1].samples_hz, 1.5 * program.rf_events[1].samples_hz
    )
    assert scaled.metadata["ParameterLoopRFScale"] == pytest.approx(1.5)


def test_rf_scale_formula_accepts_negative_and_large_values(qt_application):
    widget = SequenceSimulationWidget()
    widget.parameter_loop_parameter.setCurrentIndex(0)
    widget.parameter_loop_formula.setText("12*sin(x)")
    widget.parameter_loop_start.setValue(0)
    widget.parameter_loop_stop.setValue(5)
    widget.parameter_loop_steps.setValue(11)

    values = widget._parameter_loop_values(
        widget.parameter_loop_parameter.currentData()
    )

    np.testing.assert_allclose(values, 12 * np.sin(np.linspace(0, 5, 11)))
    assert values.min() < 0
    assert values.max() > 10
    assert "must be between" not in widget.parameter_loop_preview.text()


@pytest.mark.parametrize(
    "parameter_key, start, stop, expected",
    [
        ("sequence_reference_ppm", -20000.0, 20000.0, [-20000.0, 0.0, 20000.0]),
        ("read_matrix", 512, 514, [512, 513, 514]),
    ],
)
def test_loop_values_are_not_clipped_by_form_controls(
    qt_application, monkeypatch, parameter_key, start, stop, expected
):
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    parameter_index = next(
        index
        for index, parameter in enumerate(_loop_parameters(widget))
        if parameter.key == parameter_key
    )
    widget.parameter_loop_parameter.setCurrentIndex(parameter_index)
    widget.parameter_loop_start.setValue(start)
    widget.parameter_loop_stop.setValue(stop)
    widget.parameter_loop_steps.setValue(3)
    parameter = widget.parameter_loop_parameter.currentData()
    observed = []

    def fake_run():
        observed.append(parameter.widget.value())
        widget._parameter_loop_iteration_finished()
        return True

    monkeypatch.setattr(widget, "_run", fake_run)
    monkeypatch.setattr(widget, "_ensure_current_generated_sequence", lambda: True)
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QTimer.singleShot",
        lambda _delay, callback: callback(),
    )

    widget._start_parameter_loop()

    assert observed == expected
    assert parameter.widget.value() == expected[-1]
    assert widget._parameter_loop_state is None


@pytest.mark.parametrize("input_mode", ["formula", "manual"])
def test_parameter_loop_runs_each_value_against_the_same_base_program(
    qt_application, monkeypatch, input_mode
):
    widget = SequenceSimulationWidget()
    base_samples = np.array(widget.program.rf_events[0].samples_hz, copy=True)
    widget.parameter_loop_parameter.setCurrentIndex(0)
    widget.parameter_loop_start.setValue(0.5)
    widget.parameter_loop_stop.setValue(1.5)
    widget.parameter_loop_steps.setValue(3)
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData(input_mode)
    )
    widget.parameter_loop_manual_values.setText("1.5, 0.5, 1.5, -12")
    observed = []

    def fake_run():
        assert not widget.parameter_loop_input_mode.isEnabled()
        assert not widget.parameter_loop_formula.isEnabled()
        assert not widget.parameter_loop_manual_values.isEnabled()
        observed.append(np.array(widget.program.rf_events[0].samples_hz, copy=True))
        widget._parameter_loop_iteration_finished()
        return True

    monkeypatch.setattr(widget, "_run", fake_run)
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QTimer.singleShot",
        lambda _delay, callback: callback(),
    )

    widget._start_parameter_loop()

    scales = [1.5, 0.5, 1.5, -12] if input_mode == "manual" else [0.5, 1.0, 1.5]
    assert len(observed) == len(scales)
    for samples, scale in zip(observed, scales):
        np.testing.assert_allclose(samples, scale * base_samples)
    assert widget._parameter_loop_state is None
    assert "Parameter loop complete" in widget.status.text()
    assert widget.parameter_loop_input_mode.isEnabled()
    assert widget.parameter_loop_manual_values.isEnabled() == (input_mode == "manual")
    assert widget.parameter_loop_formula.isEnabled() == (input_mode == "formula")


@pytest.mark.parametrize("input_mode", ["formula", "manual"])
def test_parameter_loop_chains_real_sequence_workers(
    qt_application, monkeypatch, input_mode
):
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget.parameter_loop_parameter.setCurrentIndex(0)
    widget.parameter_loop_start.setValue(-1.0)
    widget.parameter_loop_stop.setValue(1.0)
    widget.parameter_loop_steps.setValue(2)
    widget.parameter_loop_formula.setText("12*x")
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData(input_mode)
    )
    widget.parameter_loop_manual_values.setText("-12, 5, -12")
    loop_metadata = []
    register = widget._register_session_simulation_run

    def capture(result, **kwargs):
        loop_metadata.append(dict(result.metadata["parameter_loop"]))
        return register(result, **kwargs)

    monkeypatch.setattr(widget, "_register_session_simulation_run", capture)
    widget._start_parameter_loop()
    deadline = time.monotonic() + 10.0
    while widget._parameter_loop_state is not None and time.monotonic() < deadline:
        QTest.qWait(10)

    assert widget._parameter_loop_state is None
    expected = [-12.0, 5.0, -12.0] if input_mode == "manual" else [-12.0, 12.0]
    assert [metadata["value"] for metadata in loop_metadata] == expected
    assert [metadata["index"] for metadata in loop_metadata] == list(
        range(len(expected))
    )
    assert all(metadata["count"] == len(expected) for metadata in loop_metadata)
    assert all(metadata["input_mode"] == input_mode for metadata in loop_metadata)
    if input_mode == "manual":
        assert all(
            metadata["manual_values"] == "-12, 5, -12" for metadata in loop_metadata
        )
        assert all(
            "formula" not in metadata and "x" not in metadata
            for metadata in loop_metadata
        )
    else:
        assert all(metadata["formula"] == "12*x" for metadata in loop_metadata)
        assert [metadata["x"] for metadata in loop_metadata] == [-1.0, 1.0]
    explorer = widget.reconstruction_explorer
    assert explorer.simulation_combo.count() == len(expected)
    assert "-12" in explorer.simulation_combo.itemText(0)
    final_result = widget.result
    explorer.simulation_combo.setCurrentIndex(0)
    assert (
        explorer.simulation_combo.currentData() == explorer.simulation_combo.itemData(0)
    )
    assert widget.result is final_result
    explorer.simulation_combo.setCurrentIndex(1)
    assert (
        explorer.simulation_combo.currentData() == explorer.simulation_combo.itemData(1)
    )
    assert widget.result is final_result
    QTest.qWait(10)
    assert widget.run_button.isEnabled()
    assert widget.parameter_loop_button.isEnabled()
    assert widget.parameter_loop_input_mode.isEnabled()
    assert widget.parameter_loop_manual_values.isEnabled() == (input_mode == "manual")
    assert widget.parameter_loop_formula.isEnabled() == (input_mode == "formula")


@pytest.mark.parametrize("slice_gradient_enabled", [True, False])
def test_slice_selection_loop_runs_are_selectable_in_spatial_magnetization(
    qt_application,
    slice_gradient_enabled,
):
    pytest.importorskip("pypulseq")
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget.sequence_source.setCurrentIndex(widget.SLICE_SELECTION_SOURCE)
    widget.slice_selection_gradient_enabled.setChecked(slice_gradient_enabled)
    widget.slice_selection_rephase.setChecked(True)
    flip_index = next(
        index
        for index, parameter in enumerate(_loop_parameters(widget))
        if parameter.key == "slice_selection_flip_angle_deg"
    )
    widget.parameter_loop_parameter.setCurrentIndex(flip_index)
    widget.parameter_loop_start.setValue(30)
    widget.parameter_loop_stop.setValue(60)
    widget.parameter_loop_steps.setValue(2)
    widget._start_parameter_loop()
    deadline = time.monotonic() + 20.0
    while widget._parameter_loop_state is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert widget._parameter_loop_state is None

    viewer = widget.result_volume_viewer
    assert viewer.simulation_combo.count() == 2
    assert "60°" in viewer.simulation_combo.currentText()
    final_result, final_program, final_phantom = (
        widget.result,
        widget.program,
        widget.phantom,
    )
    viewer.map_combo.setCurrentText("Mz")
    viewer.state_combo.setCurrentIndex(1 if slice_gradient_enabled else 0)
    final_display = viewer.volume.data.copy()
    viewer.simulation_combo.setCurrentIndex(0)
    assert viewer.result.metadata["parameter_loop"]["value"] == 30
    assert viewer.map_combo.currentText() == "Mz"
    assert viewer.state_combo.currentData() == (
        "SliceSelectionEndTime" if slice_gradient_enabled else "RFExcitationEndTime"
    )
    np.testing.assert_allclose(
        viewer.volume.data,
        (
            viewer.result.checkpoint_magnetization[0, ..., 2]
            if slice_gradient_enabled
            else viewer.result.mz
        ),
    )
    assert not np.allclose(viewer.volume.data, final_display)
    assert widget.result is final_result
    assert widget.program is final_program
    assert widget.phantom is final_phantom
    assert widget.slice_selection_flip_angle_deg.value() == 60

    first = viewer._session_runs[viewer.simulation_combo.currentData()]
    assert widget.restore_session_simulation_run(first)
    assert viewer.simulation_combo.currentData() == first.run_id
    viewer.simulation_combo.setCurrentIndex(1)
    widget.forget_session_simulation_run(first)
    assert widget.result is None
    assert viewer.result is final_result
    assert viewer.simulation_combo.count() == 1
    assert viewer.simulation_combo.currentIndex() == 0


@pytest.mark.parametrize("flip_angles", [(30.0, 60.0), (390.0, 420.0)])
def test_parameter_loop_regenerates_active_sequence_for_each_value(
    qt_application, monkeypatch, flip_angles
):
    pytest.importorskip("pypulseq")
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    widget.read_matrix.setValue(2)
    widget.phase_matrix.setValue(2)
    widget.epi_repetition_time_ms.setValue(100.0)
    parameters = _loop_parameters(widget)
    flip_index = next(
        index
        for index, parameter in enumerate(parameters)
        if parameter.key == "epi_flip_angle_deg"
    )
    widget.parameter_loop_parameter.setCurrentIndex(flip_index)
    widget.parameter_loop_start.setValue(flip_angles[0])
    widget.parameter_loop_stop.setValue(flip_angles[1])
    widget.parameter_loop_steps.setValue(2)
    loop_metadata = []
    register = widget._register_session_simulation_run

    def capture(result, **kwargs):
        loop_metadata.append(dict(result.metadata["parameter_loop"]))
        return register(result, **kwargs)

    monkeypatch.setattr(widget, "_register_session_simulation_run", capture)
    widget._start_parameter_loop()
    deadline = time.monotonic() + 20.0
    while widget._parameter_loop_state is not None and time.monotonic() < deadline:
        QTest.qWait(10)

    assert widget._parameter_loop_state is None
    assert [metadata["value"] for metadata in loop_metadata] == list(flip_angles)
    assert widget.epi_flip_angle_deg.value() == pytest.approx(flip_angles[1])
    assert widget.program.metadata["definitions"]["FlipAngleDeg"] == pytest.approx(
        flip_angles[1]
    )
    explorer = widget.reconstruction_explorer
    assert explorer.simulation_combo.count() == 2
    final_display = np.array(explorer._current_display, copy=True)
    final_result = widget.result
    explorer.simulation_combo.setCurrentIndex(0)
    assert explorer._result.metadata["parameter_loop"]["value"] == flip_angles[0]
    assert explorer._current_display.shape == final_display.shape
    assert not np.allclose(explorer._current_display, final_display)
    assert widget.result is final_result
