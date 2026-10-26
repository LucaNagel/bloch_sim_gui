import json
import time

from PyQt5.QtTest import QTest

from blochsimulator.project_io import load_project
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def _wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def _parameters(widget):
    return {
        parameter.key: parameter for parameter in widget._parameter_loop_parameters()
    }


def test_probe_category_lists_probe_settings_only_in_probe_mode(qt_application):
    widget = SequenceSimulationWidget()
    category = widget.parameter_loop_category
    category.setCurrentIndex(category.findData("probe"))
    widget.object_source.setCurrentIndex(1)
    assert _parameters(widget) == {}
    assert not widget.parameter_loop_button.isEnabled()
    assert "Select Spin probe" in widget.parameter_loop_parameter_info.text()

    widget.object_source.setCurrentIndex(2)
    parameters = _parameters(widget)
    assert {"probe_t1_ms", "probe_t2_ms", "probe_initial_mz"} <= set(parameters)
    assert all(parameter.category == "probe" for parameter in parameters.values())
    assert parameters["probe_t1_ms"].widget is widget.probe_t1_ms
    assert not widget.parameter_loop_probe_mode.isHidden()


def test_probe_loop_saves_probe_projects_and_notebook(qt_application, tmp_path):
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.INTERNAL_SOURCE)
    widget.object_source.setCurrentIndex(2)
    widget.probe_points.setValue(8)
    category = widget.parameter_loop_category
    category.setCurrentIndex(category.findData("probe"))
    combo = widget.parameter_loop_parameter
    combo.setCurrentIndex(
        next(
            index
            for index in range(combo.count())
            if combo.itemData(index).key == "probe_t2_ms"
        )
    )
    widget.parameter_loop_input_mode.setCurrentIndex(
        widget.parameter_loop_input_mode.findData("manual")
    )
    widget.parameter_loop_manual_values.setText("20, 80")
    widget.parameter_loop_output_mode.setCurrentIndex(
        widget.parameter_loop_output_mode.findData("disk")
    )
    widget.parameter_loop_output_directory.setText(str(tmp_path))
    widget.parameter_loop_write_notebook.setChecked(True)

    widget._start_parameter_loop()
    assert not widget.probe_controls.isEnabled()
    _wait_until(lambda: widget._parameter_loop_state is None)

    assert widget.probe_controls.isEnabled()
    paths = sorted(tmp_path.glob("*.blochproj"))
    assert len(paths) == 2
    loop_ids = set()
    for index, (path, value) in enumerate(zip(paths, [20.0, 80.0])):
        result = load_project(path)["sequence_result"]
        assert type(result).__name__ == "SequenceProbeResult"
        metadata = result.metadata["parameter_loop"]
        assert metadata["index"] == index
        assert metadata["category"] == "probe"
        assert metadata["value"] == value
        loop_ids.add(metadata["loop_id"])
    assert len(loop_ids) == 1
    notebook = tmp_path / "load_simulations.ipynb"
    assert loop_ids.pop() in notebook.read_text()
    json.loads(notebook.read_text())
    assert "notebook load_simulations.ipynb written" in widget.status.text()
    probe_notebook = tmp_path / "load_spin_probes.ipynb"
    json.loads(probe_notebook.read_text())
    assert metadata["loop_id"] in probe_notebook.read_text()
    assert "spin-probe notebook load_spin_probes.ipynb written" in (
        widget.status.text()
    )
