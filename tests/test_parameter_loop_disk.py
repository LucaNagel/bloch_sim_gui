import threading
import time
import weakref

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest

from blochsimulator.project_io import load_project, save_project
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def _configure_disk_loop(widget, directory, *, input_mode="manual"):
    widget.sequence_source.setCurrentIndex(widget.INTERNAL_SOURCE)
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget.parameter_loop_parameter.setCurrentIndex(0)
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData(input_mode)
    )
    widget.parameter_loop_manual_values.setText("1.5, 0.5, 1.5")
    widget.parameter_loop_start.setValue(0.5)
    widget.parameter_loop_stop.setValue(1.5)
    widget.parameter_loop_steps.setValue(3)
    widget.parameter_loop_output_mode.setCurrentIndex(
        widget.parameter_loop_output_mode.findData("disk")
    )
    widget.parameter_loop_output_directory.setText(str(directory))


def _wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


@pytest.mark.parametrize("input_mode", ["formula", "manual"])
def test_disk_loop_saves_before_next_run_and_releases_previous_results(
    qt_application, tmp_path, monkeypatch, input_mode
):
    widget = SequenceSimulationWidget()
    directory = tmp_path / "loop results"
    _configure_disk_loop(widget, directory, input_mode=input_mode)
    references = []
    run_references = []
    registered = widget._register_session_simulation_run
    started = []
    run_simulation = widget._run

    def capture(result, **kwargs):
        run = registered(result, **kwargs)
        references.append(weakref.ref(result))
        run_references.append(weakref.ref(run))
        return run

    def start_simulation():
        assert len(list(directory.glob("*.blochproj"))) == len(started)
        assert widget._parameter_loop_save_worker is None
        assert not widget.parameter_loop_output_mode.isEnabled()
        assert not widget.parameter_loop_output_directory.isEnabled()
        started.append(widget._parameter_loop_state["current_value"])
        return run_simulation()

    monkeypatch.setattr(widget, "_register_session_simulation_run", capture)
    monkeypatch.setattr(widget, "_run", start_simulation)
    widget._start_parameter_loop()
    _wait_until(lambda: widget._parameter_loop_state is None)

    expected = [1.5, 0.5, 1.5] if input_mode == "manual" else [0.5, 1, 1.5]
    paths = sorted(directory.glob("*.blochproj"))
    assert len(paths) == len(expected)
    assert not list(directory.glob("*.tmp"))
    for index, (path, value) in enumerate(zip(paths, expected)):
        saved = load_project(path)
        metadata = saved["sequence_result"].metadata["parameter_loop"]
        assert metadata["index"] == index
        assert metadata["count"] == len(expected)
        assert metadata["value"] == value
        assert metadata["input_mode"] == input_mode
        assert saved["program"].metadata["ParameterLoopRFScale"] == value
        assert saved["phantom"].shape == (2, 2, 1)
    assert widget.reconstruction_explorer._session_runs == {}
    assert widget.result_volume_viewer._session_runs == {}
    assert all(reference() is None for reference in references[:-1])
    assert all(reference() is None for reference in run_references)
    assert widget.result is references[-1]()
    assert "3 projects saved" in widget.status.text()
    assert widget.parameter_loop_output_mode.isEnabled()
    assert widget.parameter_loop_output_directory.isEnabled()


def test_disk_loop_uses_custom_numbering_and_preserves_existing_files(
    qt_application, tmp_path
):
    widget = SequenceSimulationWidget()
    _configure_disk_loop(widget, tmp_path)
    controls = widget.parameter_loop_export_naming
    controls.mode.setCurrentIndex(controls.mode.findData("numbered"))
    controls.prefix.setText("simulation")
    controls.start_number.setValue(7)
    controls.digits.setValue(2)
    existing = tmp_path / "simulation_07.blochproj"
    existing.write_bytes(b"preserved")
    widget._start_parameter_loop()
    assert not controls.isEnabled()
    _wait_until(lambda: widget._parameter_loop_state is None)
    assert existing.read_bytes() == b"preserved"
    for index, number in enumerate((8, 9, 10)):
        saved = load_project(tmp_path / f"simulation_{number:02d}.blochproj")
        assert saved["sequence_result"].metadata["parameter_loop"]["index"] == index
    assert controls.isEnabled()


def test_disk_loop_preserves_gui_setup_and_existing_session_runs(
    qt_application, tmp_path, monkeypatch
):
    from blochsimulator.ui.main_window import BlochSimulatorGUI

    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    _configure_disk_loop(widget, tmp_path)
    widget.open_parameter_loop_button.click()
    widget.parameter_loop_steps.setValue(1)
    widget.parameter_loop_manual_values.setText("0.5")
    widget._start_parameter_loop()
    _wait_until(lambda: widget._parameter_loop_state is None)

    path = next(tmp_path.glob("*.blochproj"))
    saved = load_project(path)
    assert saved["state"]["workspace_mode"] == "sequence"
    assert saved["state"]["sequence_controls"]["matrix_size"]["value"] == 2
    assert saved["state"]["sequence_controls"]["object_source"]["index"] == 1
    assert window.simulation_explorer.runs == ()
    assert saved["sequence_result"].metadata["parameter_loop"]["value"] == 0.5

    # Earlier in-memory runs survive subsequent disk-only loops.
    previous = widget._register_session_simulation_run(widget.result)
    original_bytes = path.read_bytes()
    messages = []
    monkeypatch.setattr(
        "blochsimulator.ui.main_window.QMessageBox.warning",
        lambda _parent, _title, message: messages.append(message),
    )
    widget.parameter_loop_button.click()
    state = widget._parameter_loop_state
    window.open_project(str(path))
    window._open_session_simulation_run(previous)
    assert len(messages) == 2
    assert widget._parameter_loop_state is state
    _wait_until(lambda: widget._parameter_loop_state is None)
    assert len(list(tmp_path.glob("*.blochproj"))) == 2
    assert path.read_bytes() == original_bytes
    assert window.simulation_explorer.runs == (previous,)
    assert set(widget.reconstruction_explorer._session_runs) == {previous.run_id}
    assert set(widget.result_volume_viewer._session_runs) == {previous.run_id}
    widget.parameter_loop_output_mode.setCurrentIndex(
        widget.parameter_loop_output_mode.findData("memory")
    )
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData("formula")
    )
    window.open_project(str(path))
    assert widget.result.metadata["parameter_loop"]["value"] == 0.5
    assert widget.parameter_loop_output_mode.currentData() == "disk"
    assert widget.parameter_loop_output_directory.isEnabled()
    assert widget.parameter_loop_input_mode.currentData() == "manual"
    assert widget.parameter_loop_manual_values.isEnabled()
    window.close()
    window.deleteLater()
    qt_application.processEvents()


@pytest.mark.parametrize("folder", ["", "file"])
def test_disk_loop_rejects_invalid_destination_before_simulating(
    qt_application, tmp_path, monkeypatch, folder
):
    widget = SequenceSimulationWidget()
    _configure_disk_loop(widget, tmp_path)
    target = tmp_path / "file"
    target.write_text("existing file")
    widget.parameter_loop_output_directory.setText(str(target) if folder else "")
    messages = []
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QMessageBox.warning",
        lambda _parent, _title, message: messages.append(message),
    )
    monkeypatch.setattr(widget, "_run", lambda: pytest.fail("Simulation started"))

    widget.parameter_loop_button.click()

    assert messages
    assert widget._parameter_loop_state is None
    assert widget._parameter_loop_save_worker is None
    assert target.read_text() == "existing file"


def test_cancel_during_save_keeps_current_file_and_gui_responsive(
    qt_application, tmp_path, monkeypatch
):
    widget = SequenceSimulationWidget()
    _configure_disk_loop(widget, tmp_path)
    saving = threading.Event()
    release = threading.Event()

    def pause_save(filename, **arguments):
        save_project(filename, **arguments)
        saving.set()
        assert release.wait(10)

    monkeypatch.setattr("blochsimulator.project_io.save_project", pause_save)
    widget._start_parameter_loop()
    try:
        _wait_until(saving.is_set)
        ticked = []
        QTimer.singleShot(0, lambda: ticked.append(True))
        _wait_until(lambda: bool(ticked))
        assert widget.cancel_button.isEnabled()
        widget.cancel_button.click()
        assert widget._parameter_loop_state["cancel_requested"]
    finally:
        release.set()
        _wait_until(lambda: widget._parameter_loop_state is None)

    paths = list(tmp_path.glob("*.blochproj"))
    assert len(paths) == 1
    assert (
        load_project(paths[0])["sequence_result"].metadata["parameter_loop"]["index"]
        == 0
    )
    assert "cancelled after 1/3" in widget.status.text()
    assert widget.parameter_loop_output_directory.isEnabled()


def test_save_failure_stops_loop_and_retains_unsaved_result(
    qt_application, tmp_path, monkeypatch
):
    widget = SequenceSimulationWidget()
    _configure_disk_loop(widget, tmp_path)
    paths = []
    messages = []

    def fail_second_save(filename, **arguments):
        paths.append(filename)
        if len(paths) == 2:
            raise OSError("Disk full")
        save_project(filename, **arguments)

    monkeypatch.setattr("blochsimulator.project_io.save_project", fail_second_save)
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QMessageBox.critical",
        lambda _parent, _title, message: messages.append(message),
    )
    widget._start_parameter_loop()
    _wait_until(lambda: widget._parameter_loop_state is None)

    assert len(paths) == 2
    assert len(list(tmp_path.glob("*.blochproj"))) == 1
    assert "Disk full" in messages[0]
    assert "Saving failed" in widget.status.text()
    assert widget.result.metadata["parameter_loop"]["index"] == 1
    assert widget.export_button.isEnabled()
    assert len(widget.reconstruction_explorer._session_runs) == 1
    assert len(widget.result_volume_viewer._session_runs) == 1
