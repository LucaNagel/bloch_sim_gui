import json

import matplotlib
import numpy as np
import pytest
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from blochsimulator.probe_notebook import (
    DEFAULT_PROBE_NOTEBOOK_NAME,
    write_probe_loader_notebook,
)
from blochsimulator.sequence import SequenceProbeResult


def _probe(t2_s=0.05, loop=None, finished="2026-10-26T10:20:30+00:00"):
    time_s = np.linspace(0.0, 0.02, 5)
    frequencies = np.array([-100.0, 0.0, 100.0])
    magnetization = np.zeros((time_s.size, 1, frequencies.size, 3))
    magnetization[..., 0] = np.exp(-time_s / t2_s)[:, None, None]
    magnetization[..., 2] = 0.5
    metadata = {
        "probe_type": "spectral",
        "ui_probe_mode": "spectral",
        "simulation_finished_at_utc": finished,
        "simulation_wall_time_s": 0.1,
        "t1_s": 1.0,
        "t2_s": t2_s,
        "frequency_axis_unit": "ppm",
        "frequency_offsets_ppm": np.array([-1.0, 0.0, 1.0]),
        "ideal_spoiler_end_times_s": [0.01],
    }
    if loop is not None:
        metadata["parameter_loop"] = loop
    return SequenceProbeResult(
        time_s=time_s,
        positions_m=np.array([[0.0, 0.001, 0.0]]),
        frequency_offsets_hz=frequencies,
        magnetization=magnetization,
        metadata=metadata,
    )


@pytest.mark.parametrize("suffix", [".nc", ".npz"])
def test_probe_result_save_load_round_trip(tmp_path, suffix):
    result = _probe(loop={"loop_id": "abc", "index": 1, "value": 20.0})
    path = tmp_path / f"probe{suffix}"
    result.save(path)
    loaded = SequenceProbeResult.load(path)

    np.testing.assert_allclose(loaded.time_s, result.time_s)
    np.testing.assert_allclose(loaded.positions_m, result.positions_m)
    np.testing.assert_allclose(loaded.frequency_offsets_hz, result.frequency_offsets_hz)
    np.testing.assert_allclose(loaded.magnetization, result.magnetization)
    assert loaded.metadata["parameter_loop"]["loop_id"] == "abc"
    assert loaded.metadata["frequency_offsets_ppm"] == [-1.0, 0.0, 1.0]
    assert loaded.metadata["t2_s"] == 0.05


def test_probe_result_save_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        _probe().save(tmp_path / "probe.csv")
    np.savez(tmp_path / "other.npz", values=np.zeros(3))
    with pytest.raises(ValueError):
        SequenceProbeResult.load(tmp_path / "other.npz")


def test_probe_notebook_runs_on_exported_probes(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    for index, t2_s in enumerate([0.02, 0.08]):
        _probe(
            t2_s, loop={"loop_id": "loop1", "index": index, "value": 1e3 * t2_s}
        ).save(tmp_path / f"probe_{index}.nc")
    _probe().save(tmp_path / "unrelated.npz")
    (tmp_path / "notes.nc").write_text("not a probe")

    first = write_probe_loader_notebook(
        tmp_path, data_file="probe_1.nc", loop_id="loop1"
    )
    second = write_probe_loader_notebook(tmp_path)
    assert first.name == DEFAULT_PROBE_NOTEBOOK_NAME
    assert second.name == "load_spin_probes_2.ipynb"

    notebook = json.loads(first.read_text())
    assert notebook["nbformat"] == 4
    namespace = {}
    monkeypatch.chdir(tmp_path)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "cell", "exec"), namespace)
    assert namespace["path"].name == "probe_1.nc"
    assert namespace["unit"] == "ppm"
    assert [run["file"] for run in namespace["runs"]] == ["probe_0.nc", "probe_1.nc"]
    matplotlib.pyplot.close("all")


def _finish_probe(widget, result):
    widget._probe_finished(result, result.frequency_offsets_hz.copy(), "Hz", "spectral")


def test_spin_probe_selector_hint_and_export(qt_application, tmp_path, monkeypatch):
    from blochsimulator.ui.main_window import BlochSimulatorGUI

    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(2)

    assert not widget.probe_empty_hint.isHidden()
    assert "Spin probe" in widget.probe_empty_hint.text()
    assert not widget.probe_run_combo.isEnabled()
    assert not widget.probe_export_button.isEnabled()

    first, second = _probe(0.02), _probe(0.08, finished="2026-10-26T10:21:00+00:00")
    _finish_probe(widget, first)
    _finish_probe(widget, second)
    assert widget.probe_empty_hint.isHidden()
    assert widget.probe_run_combo.count() == 2
    assert widget.probe_result is second
    assert widget.probe_run_combo.currentIndex() == 1

    widget.probe_run_combo.setCurrentIndex(0)
    assert widget.probe_result is first

    run = window.simulation_explorer.runs[0]
    assert widget.probe_run_combo.itemText(0) == run.display_name
    run.display_name = "Short T2"
    window.simulation_explorer.run_renamed.emit(run)
    assert widget.probe_run_combo.itemText(0) == "Short T2"

    path = tmp_path / "shown.nc"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    widget._export_probe_results()
    np.testing.assert_allclose(
        SequenceProbeResult.load(path).magnetization, first.magnetization
    )
    notebook = json.loads((tmp_path / "shown.ipynb").read_text())
    assert "DATA_FILE = 'shown.nc'" in "".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )

    folder = tmp_path / "all"
    folder.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(folder))
    widget._export_all_probe_results()
    exported = sorted(folder.glob("*.nc"))
    assert len(exported) == 2
    assert "Short_T2" in exported[0].name
    assert (folder / DEFAULT_PROBE_NOTEBOOK_NAME).exists()

    widget.remove_probe_session_run(run)
    assert widget.probe_run_combo.count() == 1

    window.close()
    window.deleteLater()
    qt_application.processEvents()
