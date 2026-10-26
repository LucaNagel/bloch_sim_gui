import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication, QDialogButtonBox

from blochsimulator.project_io import load_project
from blochsimulator.project_naming import ProjectExportNamer, ProjectExportNaming
from blochsimulator.sequence.result import SequenceSimulationResult
from blochsimulator.ui.project_export_dialog import (
    SimulationProjectExportDialog,
)


def test_current_names_are_reserved_before_the_batch_is_written(tmp_path):
    namer = ProjectExportNamer(tmp_path)
    paths = [
        namer.next_path("bloch_project_20261015_120000", "unused") for _ in range(3)
    ]
    assert [path.name for path in paths] == [
        "bloch_project_20261015_120000.blochproj",
        "bloch_project_20261015_120000_2.blochproj",
        "bloch_project_20261015_120000_3.blochproj",
    ]
    assert namer.next_path("Renamed run", "unused").name == "Renamed run.blochproj"


def test_timestamp_prefix_is_customizable_and_avoids_existing_files(tmp_path):
    existing = tmp_path / "study_20261015_120000.blochproj"
    existing.write_bytes(b"preserved")
    namer = ProjectExportNamer(
        tmp_path, ProjectExportNaming(mode="timestamp", prefix="study")
    )
    assert (
        namer.next_path("old name", "20261015_120000").name
        == "study_20261015_120000_2.blochproj"
    )
    assert (
        namer.next_path("old name", "20261015_120000").name
        == "study_20261015_120000_3.blochproj"
    )
    assert existing.read_bytes() == b"preserved"


def test_numbering_supports_start_padding_and_skips_occupied_numbers(tmp_path):
    for number in (7, 9):
        (tmp_path / f"simulation_{number:03d}.blochproj").write_bytes(b"preserved")
    namer = ProjectExportNamer(
        tmp_path,
        ProjectExportNaming(
            mode="numbered", prefix="simulation_", start_number=7, digits=3
        ),
    )
    assert [namer.next_path("unused", "unused").name for _ in range(3)] == [
        "simulation_008.blochproj",
        "simulation_010.blochproj",
        "simulation_011.blochproj",
    ]
    assert (tmp_path / "simulation_007.blochproj").read_bytes() == b"preserved"


def test_number_width_expands_and_zero_is_a_valid_start(tmp_path):
    namer = ProjectExportNamer(
        tmp_path, ProjectExportNaming(mode="numbered", start_number=99, digits=2)
    )
    assert namer.next_path("unused", "unused").name == "simulation_99.blochproj"
    assert namer.next_path("unused", "unused").name == "simulation_100.blochproj"
    zero = ProjectExportNamer(
        tmp_path, ProjectExportNaming(mode="numbered", start_number=0)
    )
    assert zero.next_path("unused", "unused").name == "simulation_00.blochproj"


@pytest.mark.parametrize(
    "prefix", ["", "   ", "../study", "a/b", "a\\b", "a:b", "a?b", ".", ".."]
)
def test_invalid_prefixes_are_rejected(prefix, tmp_path):
    with pytest.raises(ValueError):
        ProjectExportNamer(
            tmp_path, ProjectExportNaming(mode="numbered", prefix=prefix)
        )


def test_naming_controls_preview_and_validation(qt_application, tmp_path, monkeypatch):
    dialog = SimulationProjectExportDialog(
        tmp_path,
        [("Renamed run", "20261015_120000"), ("Another run", "20261015_120000")],
    )
    controls = dialog.naming
    assert not controls.prefix.isEnabled()
    assert "Renamed run.blochproj" in controls.preview.text()
    controls.mode.setCurrentIndex(controls.mode.findData("numbered"))
    controls.prefix.setText("simulation")
    controls.start_number.setValue(5)
    controls.digits.setValue(3)
    assert controls.prefix.isEnabled()
    assert controls.start_number.isEnabled()
    assert (
        controls.preview.text() == "simulation_005.blochproj\nsimulation_006.blochproj"
    )
    messages = []
    monkeypatch.setattr(
        "blochsimulator.ui.project_export_dialog.QMessageBox.warning",
        lambda _parent, _title, message: messages.append(message),
    )
    accepted = QSignalSpy(dialog.accepted)
    controls.prefix.setText("../simulation")
    dialog.buttons.button(QDialogButtonBox.Save).click()
    assert messages
    assert not accepted
    controls.prefix.setText("simulation")
    dialog.buttons.button(QDialogButtonBox.Save).click()
    assert len(accepted) == 1
    directory, naming = dialog.export_options()
    assert directory == tmp_path
    assert naming.start_number == 5
    assert naming.digits == 3


def test_batch_export_uses_actual_naming_dialog_and_preserves_each_result(
    qt_application, tmp_path, monkeypatch
):
    from blochsimulator.ui.main_window import BlochSimulatorGUI

    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget._build_phantom()
    runs = []
    for value in (1, 2):
        result = SequenceSimulationResult(
            signal=np.full(4, value, dtype=complex),
            adc_times_s=np.arange(4) * 1e-4,
            final_magnetization=np.zeros(widget.phantom.shape + (3,)),
            checkpoint_magnetization=None,
            checkpoint_times_s=np.empty(0),
        )
        run = widget._register_session_simulation_run(result)
        run.created_at_utc = "2026-10-15T12:00:00"
        runs.append(run)
    occupied = tmp_path / "simulation_11.blochproj"
    occupied.write_bytes(b"preserved")

    def choose_names():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, SimulationProjectExportDialog)
        dialog.directory.setText(str(tmp_path))
        dialog.naming.mode.setCurrentIndex(dialog.naming.mode.findData("numbered"))
        dialog.naming.prefix.setText("simulation")
        dialog.naming.start_number.setValue(11)
        dialog.naming.digits.setValue(2)
        assert "simulation_12.blochproj" in dialog.naming.preview.text()
        assert not dialog.write_notebook.isChecked()
        dialog.write_notebook.setChecked(True)
        dialog.buttons.button(QDialogButtonBox.Save).click()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(
        lambda: (
            QApplication.activeModalWidget().reject()
            if QApplication.activeModalWidget() is not None
            else None
        )
    )
    watchdog.start(5000)
    QTimer.singleShot(0, choose_names)
    window._export_session_simulation_runs(runs)
    watchdog.stop()
    assert occupied.read_bytes() == b"preserved"
    for number, value in ((12, 1), (13, 2)):
        project = load_project(tmp_path / f"simulation_{number}.blochproj")
        np.testing.assert_array_equal(project["sequence_result"].signal, value)
    assert window._simulation_export_naming.prefix == "simulation"
    assert (tmp_path / "load_simulations.ipynb").is_file()
    monkeypatch.setattr(
        window,
        "_prompt_session_project_export",
        lambda _runs: (tmp_path, ProjectExportNaming()),
    )
    window._export_session_simulation_runs(runs)
    timestamped = sorted(tmp_path.glob("bloch_project_*.blochproj"))
    assert len(timestamped) == 2
    assert timestamped[1].stem == f"{timestamped[0].stem}_2"
    # The notebook choice is remembered; an existing notebook is never replaced.
    assert (tmp_path / "load_simulations_2.ipynb").is_file()
    for path, value in zip(timestamped, (1, 2)):
        np.testing.assert_array_equal(
            load_project(path)["sequence_result"].signal, value
        )
    window.close()
    window.deleteLater()
    qt_application.processEvents()
