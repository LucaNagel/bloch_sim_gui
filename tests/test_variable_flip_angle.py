from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence import (
    CartesianAcquisition,
    load_pulseq,
    make_cartesian_epi,
    make_pulseq_csi,
    make_pulseq_epi,
    variable_flip_angle_schedule,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


pypulseq = pytest.importorskip("pypulseq")


def _write_and_load(sequence, path: Path):
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _rf_flip_angles_deg(program):
    return np.asarray(
        [
            360.0 * abs(np.sum(event.samples_hz) * event.raster_s)
            for event in program.rf_events
        ]
    )


def test_variable_flip_angle_schedule_matches_four_scan_nagashima_schedule():
    schedule = variable_flip_angle_schedule(4)

    assert schedule == pytest.approx([30.0, 35.26438968, 45.0, 90.0])
    angles = np.deg2rad(schedule)
    remaining_before_pulse = np.concatenate(([1.0], np.cumprod(np.cos(angles[:-1]))))
    assert remaining_before_pulse * np.sin(angles) == pytest.approx(np.full(4, 0.5))


def test_csi_vfa_changes_each_phase_encode_and_restarts_each_repetition(tmp_path):
    sequence = make_pulseq_csi(
        matrix=(2, 2),
        spectral_points=8,
        spectral_bandwidth_hz=2000.0,
        repetition_time_s=30e-3,
        repetitions=2,
        variable_flip_angle=True,
    )
    program = _write_and_load(sequence, tmp_path / "csi_vfa.seq")
    schedule = variable_flip_angle_schedule(4)
    definitions = program.metadata["definitions"]

    assert _rf_flip_angles_deg(program) == pytest.approx(np.tile(schedule, 2), abs=2e-3)
    assert bool(definitions["VariableFlipAngle"])
    assert definitions["VariableFlipAngleDimension"] == "phase_encode"
    assert definitions["FlipAngleScheduleDeg"] == pytest.approx(schedule)
    assert definitions["VariableFlipAngleReferenceDOI"] == ("10.1016/j.jmr.2007.10.011")


def test_epi_vfa_changes_per_repetition_for_internal_and_pulseq_sequences(tmp_path):
    acquisition = CartesianAcquisition.epi(
        read_matrix=4,
        phase_matrix=2,
        fov_m=(0.08, 0.06),
        dwell_s=40e-6,
    )
    internal = make_cartesian_epi(
        acquisition,
        n_slices=2,
        slice_thickness_m=3e-3,
        repetitions=4,
        repetition_time_s=50e-3,
        variable_flip_angle=True,
    )
    pulseq = make_pulseq_epi(
        fov_m=acquisition.fov_m,
        matrix=(acquisition.read_matrix, acquisition.phase_matrix),
        sampling_bandwidth_hz=25_000.0,
        n_slices=2,
        slice_thickness_m=3e-3,
        repetitions=4,
        repetition_time_s=50e-3,
        variable_flip_angle=True,
    )
    imported = _write_and_load(pulseq, tmp_path / "epi_vfa.seq")
    schedule = variable_flip_angle_schedule(4)
    expected = np.repeat(schedule, 2)

    assert _rf_flip_angles_deg(internal) == pytest.approx(expected)
    assert _rf_flip_angles_deg(imported) == pytest.approx(expected, abs=2e-3)
    assert internal.metadata["definitions"]["FlipAngleScheduleDeg"] == pytest.approx(
        schedule
    )
    assert imported.metadata["definitions"]["VariableFlipAngleDimension"] == (
        "repetition"
    )


def test_gui_enables_and_exports_epi_and_csi_vfa(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()

    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(2)
    widget.epi_repetitions.setValue(4)
    widget.epi_repetition_time_ms.setValue(50.0)
    widget.epi_variable_flip_angle.setChecked(True)
    widget.epi_vfa_final_flip_angle_deg.setValue(60.0)
    widget.sequence_source.setCurrentIndex(1)
    epi_path = widget._write_pulseq_path(tmp_path / "gui_epi_vfa.seq")
    epi_definitions = load_pulseq(epi_path).metadata["definitions"]

    assert not widget.epi_flip_angle_deg.isEnabled()
    assert widget.epi_vfa_final_flip_angle_deg.isEnabled()
    assert "60°" in widget.epi_vfa_info.text()
    assert epi_definitions["FlipAngleScheduleDeg"] == pytest.approx(
        variable_flip_angle_schedule(4, final_flip_angle_deg=60.0)
    )

    widget.csi_read_matrix.setValue(2)
    widget.csi_phase_matrix.setValue(2)
    widget.csi_spectral_points.setValue(8)
    widget.csi_repetition_time_ms.setValue(30.0)
    widget.csi_repetitions.setValue(2)
    widget.csi_variable_flip_angle.setChecked(True)
    widget.sequence_source.setCurrentIndex(2)
    csi_path = widget._write_pulseq_path(tmp_path / "gui_csi_vfa.seq")
    csi_definitions = load_pulseq(csi_path).metadata["definitions"]

    assert not widget.csi_flip_angle_deg.isEnabled()
    assert widget.csi_vfa_final_flip_angle_deg.isEnabled()
    assert "4 phase encodes" in widget.csi_vfa_info.text()
    assert csi_definitions["FlipAngleScheduleDeg"] == pytest.approx(
        variable_flip_angle_schedule(4)
    )

    widget.close()
    widget.deleteLater()
    app.processEvents()
