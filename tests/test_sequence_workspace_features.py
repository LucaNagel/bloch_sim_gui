import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from blochsimulator import BlochSimulator
from blochsimulator.paths import workspace_directory
from blochsimulator.phantom_design import PhantomDesign
from blochsimulator.sequence import BrukerExportOptions
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.ui.widgets import IMAGE_CANVAS_BACKGROUND, IMAGE_FOV_BORDER


def _wait_until(predicate, timeout_ms=10_000):
    deadline = time.monotonic() + timeout_ms / 1_000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    return predicate()


def test_workspace_directories_follow_configured_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOCHSIMULATOR_DATA_DIR", str(tmp_path))

    assert workspace_directory("sequences") == tmp_path / "sequences"
    assert workspace_directory("phantoms") == tmp_path / "phantoms"
    assert workspace_directory("exports") == tmp_path / "exports"


def test_phantom_designer_defaults_to_isotropic_128_cube():
    design = PhantomDesign()
    assert design.shape == (128, 128, 128)
    assert design.fov_m == (0.22, 0.22, 0.22)


def test_sequence_2d_image_views_show_the_fov_against_a_dark_gray_canvas():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()

    for view in (
        widget.kspace_view,
        widget.reconstruction_view,
    ):
        canvas_rgb = view.ui.graphicsView.backgroundBrush().color().getRgb()[:3]
        border_rgb = view.getImageItem().border.color().getRgb()[:3]
        assert canvas_rgb == IMAGE_CANVAS_BACKGROUND
        assert border_rgb == IMAGE_FOV_BORDER

    split_canvas_rgb = widget.split_image_plot.backgroundBrush().color().getRgb()[:3]
    split_border_rgb = widget.split_image_item.border.color().getRgb()[:3]
    assert split_canvas_rgb == IMAGE_CANVAS_BACKGROUND
    assert split_border_rgb == IMAGE_FOV_BORDER
    widget.close()
    app.processEvents()


def test_file_dialogs_start_in_sequence_and_phantom_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOCHSIMULATOR_DATA_DIR", str(tmp_path))
    app = QApplication.instance() or QApplication([])
    sequence_widget = SequenceSimulationWidget()
    designer = SpectralPhantomDesignerDialog()

    with patch(
        "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ) as sequence_dialog:
        sequence_widget._load_pulseq_file()
    with patch(
        "blochsimulator.ui.phantom_designer.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ) as phantom_dialog:
        designer._load()

    assert Path(sequence_dialog.call_args.args[2]) == tmp_path / "sequences"
    assert Path(phantom_dialog.call_args.args[2]) == tmp_path / "phantoms"
    sequence_widget.close()
    designer.close()
    app.processEvents()


def test_pulseq_file_dialog_loads_in_background(tmp_path):
    pypulseq = pytest.importorskip("pypulseq")
    sequence = pypulseq.Sequence()
    sequence.add_block(pypulseq.make_adc(num_samples=4, dwell=10e-6))
    path = tmp_path / "background.seq"
    sequence.write(str(path))
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()

    with patch(
        "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getOpenFileName",
        return_value=(str(path), "Pulseq sequence (*.seq)"),
    ):
        widget._load_pulseq_file()

    worker = widget.pulseq_load_worker
    assert worker is not None
    assert not widget.load_pulseq_button.isEnabled()
    assert widget.progress.minimum() == 0
    assert widget.progress.maximum() == 0
    assert worker.wait(10_000)
    app.processEvents()

    assert widget.pulseq_load_worker is None
    assert widget.load_pulseq_button.isEnabled()
    assert widget.run_button.isEnabled()
    assert widget.program.source == str(path)
    assert widget.progress.format() == "Pulseq loaded"
    widget.close()
    app.processEvents()


def test_python_sequence_script_runs_in_gui_and_loads_generated_pulseq(
    tmp_path, monkeypatch
):
    pytest.importorskip("pypulseq")
    monkeypatch.setenv("BLOCHSIMULATOR_DATA_DIR", str(tmp_path))
    scripts = workspace_directory("sequences") / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "generate_test_sequence.py"
    script.write_text(
        "from pathlib import Path\n"
        "import pypulseq as pp\n"
        "sequence = pp.Sequence()\n"
        "sequence.add_block(pp.make_adc(num_samples=4, dwell=10e-6))\n"
        "output = Path(__file__).with_suffix('.seq')\n"
        "sequence.write(str(output), v141_compat=True)\n"
        "print(f'generated {output.name}')\n",
        encoding="utf-8",
    )
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()

    with patch(
        "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getOpenFileName",
        return_value=(str(script), "Python script (*.py)"),
    ):
        widget._run_python_script()

    process = widget.script_process
    assert process is not None
    generated = script.with_suffix(".seq")
    assert _wait_until(
        lambda: widget.program is not None
        and widget.program.source == str(generated)
        and widget.script_process is None
        and widget.pulseq_load_worker is None
    )

    assert generated.is_file()
    assert widget.program.source == str(generated)
    assert "generated generate_test_sequence.seq" in widget.script_output.toPlainText()
    assert widget.sequence_source.currentIndex() == widget.PULSEQ_SOURCE
    assert widget.run_script_button.isEnabled()

    widget.close()
    app.processEvents()


def test_sequence_progress_is_determinate_from_simulation_start():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(4)
    widget.z_matrix_size.setValue(4)
    widget._build_phantom()

    work_units = widget._estimated_work_units()
    widget.progress.setRange(0, work_units)
    widget.progress.setValue(0)
    assert work_units == 32
    assert widget.progress.minimum() == 0
    assert widget.progress.maximum() == work_units
    widget.close()
    app.processEvents()


def test_sequence_progress_estimate_excludes_compile_time_and_uses_run_rate(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget._simulation_started_at = 100.0
    timestamps = iter((110.0, 120.0))
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.time.monotonic",
        lambda: next(timestamps),
    )

    widget._progress(25, 100)
    assert widget.progress.format() == "25% · Estimating remaining time…"
    widget._progress(50, 100)

    assert widget.progress.value() == 50
    assert widget.progress.format() == "50% · ETA 20s"
    assert widget.status.text() == "Chunk 50/100 · 50% · approximately 20s remaining"
    widget.close()
    app.processEvents()


def test_completed_sequence_progress_shows_percentage_and_elapsed_time(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.progress.setRange(0, 100)
    widget.progress.setValue(75)
    widget._simulation_started_at = 100.0
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.time.monotonic",
        lambda: 112.2,
    )

    widget._reset_run_controls(completed=True)

    assert widget.progress.value() == 100
    assert widget.progress.format() == "100% · Complete in 13s"
    assert widget.simulation_time_label.text() == ("Total runtime: 13s · Remaining: 0s")
    assert widget._simulation_started_at is None
    widget.close()
    app.processEvents()


def test_sequence_time_label_updates_between_progress_events():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget._simulation_started_at = 100.0
    widget._simulation_last_progress_at = 110.0
    widget._simulation_last_progress_done = 20
    widget._simulation_last_progress_total = 100
    widget._simulation_progress_rate = 2.0

    widget._update_simulation_time_label(now=115.0)

    assert widget.simulation_time_label.text() == (
        "Elapsed: 15s · Remaining: approximately 35s"
    )
    widget.close()
    app.processEvents()


def test_completed_sequence_records_wall_time_in_result_metadata():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget._simulation_started_at = 100.0
    widget._simulation_started_at_utc = "2026-08-14T10:00:00+00:00"
    result = SimpleNamespace(metadata={})

    elapsed_s = widget._record_simulation_timing(
        result,
        now=112.5,
        finished_at_utc="2026-08-14T10:00:12.500000+00:00",
    )

    assert elapsed_s == pytest.approx(12.5)
    assert result.metadata == {
        "simulation_wall_time_s": 12.5,
        "simulation_started_at_utc": "2026-08-14T10:00:00+00:00",
        "simulation_finished_at_utc": "2026-08-14T10:00:12.500000+00:00",
        "simulation_time_measurement": "wall_clock",
    }
    widget.close()
    app.processEvents()


def test_live_sequence_preview_moves_cursor_and_updates_cartesian_views():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(4)
    widget._load_cartesian_epi()
    widget._configure_frame_selector()

    signal = np.ones(widget.acquisition.num_samples, dtype=np.complex128)
    widget._preview(0.5, signal)

    expected_position = widget.program.duration_s * 500.0
    assert widget.rf_progress_cursor.value() == pytest.approx(expected_position)
    assert len(widget.signal_plot.listDataItems()) == 3
    assert (
        "Live received ADC signal" in widget.signal_plot.getPlotItem().titleLabel.text
    )
    assert "Current simulation" in widget.spectrum_info.text()
    assert "Live k-space" in widget.kspace_info.text()
    assert "Live |IFFT2|" in widget.reconstruction_info.text()
    widget.close()
    app.processEvents()


def test_new_sequence_run_clears_previous_signal_and_images():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.result = object()
    widget._split_csi_data = object()
    widget.signal_plot.plot([0.0, 1.0], [1.0, 2.0])
    widget.kspace_view.setImage(np.ones((2, 2)))
    widget.reconstruction_view.setImage(np.ones((2, 2)))

    widget._clear_previous_simulation_views()

    assert widget.result is None
    assert widget._split_csi_data is None
    assert widget.signal_plot.listDataItems() == []
    assert widget.kspace_view.image is None
    assert widget.reconstruction_view.image is None
    assert "current simulation" in widget.spectrum_info.text().lower()
    assert "current simulation" in widget.kspace_info.text().lower()
    assert "current simulation" in widget.reconstruction_info.text().lower()
    widget.close()
    app.processEvents()


def test_sequence_workspace_exports_xarray_result(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.field_strength_t.setValue(9.4)
    widget.nucleus.setCurrentText("C13")
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget._build_phantom()
    widget.result = BlochSimulator(use_parallel=False).simulate_sequence(
        widget.program, widget.phantom
    )
    widget.result.metadata.update(
        simulation_wall_time_s=12.5,
        simulation_started_at_utc="2026-08-14T10:00:00+00:00",
        simulation_finished_at_utc="2026-08-14T10:00:12.500000+00:00",
        simulation_time_measurement="wall_clock",
    )
    output = tmp_path / "sequence_result.nc"

    with (
        patch(
            "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getSaveFileName",
            return_value=(str(output), "xarray NetCDF (*.nc)"),
        ),
        patch("blochsimulator.ui.sequence_simulation_widget.QMessageBox.information"),
    ):
        widget._export_results()

    assert output.is_file()
    import xarray as xr

    with xr.open_dataset(output) as exported:
        assert exported.attrs["field_strength_t"] == pytest.approx(9.4)
        assert exported.attrs["nucleus"] == "C13"
        assert exported.attrs["simulation_wall_time_s"] == pytest.approx(12.5)
        assert exported.attrs["simulation_time_measurement"] == "wall_clock"
        metadata = json.loads(exported.attrs["metadata_json"])
        assert metadata["field_strength_t"] == pytest.approx(9.4)
        assert metadata["nucleus"] == "C13"
        assert metadata["phantom_metadata"]["field_strength_t"] == pytest.approx(9.4)
        assert metadata["phantom_metadata"]["nucleus"] == "C13"
        assert metadata["simulation_wall_time_s"] == pytest.approx(12.5)
        assert metadata["simulation_started_at_utc"].startswith("2026-08-14T10:00")
    widget.close()
    app.processEvents()


def test_sequence_workspace_exports_result_data_and_notebook_by_default(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget._build_phantom()
    widget.result = BlochSimulator(use_parallel=False).simulate_sequence(
        widget.program, widget.phantom
    )
    output = tmp_path / "sequence_result.nc"
    combined_filter = "xarray NetCDF + Jupyter notebook (*.nc)"

    with (
        patch(
            "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getSaveFileName",
            return_value=(str(output), combined_filter),
        ) as save_dialog,
        patch("blochsimulator.ui.sequence_simulation_widget.QMessageBox.information"),
    ):
        widget._export_results()

    notebook = output.with_suffix(".ipynb")
    assert output.is_file()
    assert notebook.is_file()
    assert "sequence_result.nc" in notebook.read_text(encoding="utf-8")
    assert save_dialog.call_args.args[3].startswith(combined_filter)

    widget.close()
    app.processEvents()


def test_sequence_workspace_exports_bruker_raw_directory(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget._build_phantom()
    widget.result = BlochSimulator(use_parallel=False).simulate_sequence(
        widget.program, widget.phantom
    )
    output = tmp_path / "bruker_export" / "1"

    with (
        patch(
            "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getSaveFileName",
            return_value=(str(output), "Bruker raw dataset (directory)"),
        ),
        patch.object(
            widget,
            "_prompt_bruker_export_options",
            return_value=BrukerExportOptions(
                method_name="Bruker:RARE",
                raw_data_files="both",
            ),
        ),
        patch("blochsimulator.ui.sequence_simulation_widget.QMessageBox.information"),
    ):
        widget._export_results()

    assert (output / "fid").is_file()
    assert (output / "rawdata.job0").is_file()
    assert (output / "acqp").is_file()
    assert (output / "method").is_file()
    widget.close()
    app.processEvents()
