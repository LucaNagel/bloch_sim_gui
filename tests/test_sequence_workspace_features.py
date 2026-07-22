from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator import BlochSimulator
from blochsimulator.paths import workspace_directory
from blochsimulator.phantom_design import PhantomDesign
from blochsimulator.sequence import BrukerExportOptions
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def test_workspace_directories_follow_configured_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOCHSIMULATOR_DATA_DIR", str(tmp_path))

    assert workspace_directory("sequences") == tmp_path / "sequences"
    assert workspace_directory("phantoms") == tmp_path / "phantoms"
    assert workspace_directory("exports") == tmp_path / "exports"


def test_phantom_designer_defaults_to_isotropic_128_cube():
    design = PhantomDesign()
    assert design.shape == (128, 128, 128)
    assert design.fov_m == (0.22, 0.22, 0.22)


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
    assert work_units == 1
    assert widget.progress.minimum() == 0
    assert widget.progress.maximum() == 1
    widget.close()
    app.processEvents()


def test_sequence_progress_shows_percentage_and_estimated_time(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget._simulation_started_at = 100.0
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.time.monotonic",
        lambda: 110.0,
    )

    widget._progress(25, 100)

    assert widget.progress.value() == 25
    assert widget.progress.format() == "25% · ETA 30s"
    assert widget.status.text() == "Chunk 25/100 · 25% · approximately 30s remaining"
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
    assert widget._simulation_started_at is None
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
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget._build_phantom()
    widget.result = BlochSimulator(use_parallel=False).simulate_sequence(
        widget.program, widget.phantom
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
