from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator import BlochSimulator
from blochsimulator.paths import workspace_directory
from blochsimulator.phantom_design import PhantomDesign
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
    assert "Live k-space" in widget.kspace_info.text()
    assert "Live |IFFT2|" in widget.reconstruction_info.text()
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
