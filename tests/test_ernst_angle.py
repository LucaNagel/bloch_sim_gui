import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence import ernst_angle_deg, load_pulseq, make_pulseq_epi
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


pypulseq = pytest.importorskip("pypulseq")


def test_ernst_angle_uses_tr_and_t1_only():
    expected = np.rad2deg(np.arccos(np.exp(-0.1 / 1.0)))

    assert float(ernst_angle_deg(0.1, 1.0)) == pytest.approx(expected)
    assert ernst_angle_deg(0.1, np.asarray([0.5, 1.0])).shape == (2,)
    with pytest.raises(ValueError, match="repetition_time_s"):
        ernst_angle_deg(0.0, 1.0)
    with pytest.raises(ValueError, match="t1_s"):
        ernst_angle_deg(0.1, 0.0)


def test_epi_builder_applies_and_records_rf_spoiling(tmp_path):
    sequence = make_pulseq_epi(
        matrix=(2, 2),
        sampling_bandwidth_hz=25_000.0,
        repetitions=3,
        repetition_time_s=30e-3,
        echo_time_s=5e-3,
        rf_spoiling=True,
        rf_spoiling_increment_deg=117.0,
    )
    path = tmp_path / "rf_spoiled_epi.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    phases_deg = np.mod(
        np.rad2deg([event.phase_offset_rad for event in program.rf_events]), 360.0
    )

    assert phases_deg == pytest.approx([0.0, 117.0, 351.0], abs=1e-3)
    assert bool(program.metadata["definitions"]["RFSpoiling"])
    assert program.metadata["definitions"]["RFSpoilingIncrementDeg"] == pytest.approx(
        117.0
    )


def test_gui_ernst_angle_state_and_summary_table():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.t1_ms.setValue(1000.0)

    assert widget.epi_use_ernst_angle.isEnabled()
    assert not widget.epi_rf_spoiling_increment_deg.isEnabled()
    widget.epi_rf_spoiling.setChecked(True)
    assert widget.epi_use_ernst_angle.isEnabled()
    assert widget.epi_rf_spoiling_increment_deg.isEnabled()
    widget.epi_repetition_time_ms.setValue(100.0)
    widget.epi_use_ernst_angle.setChecked(True)
    expected = float(ernst_angle_deg(0.1, 1.0))
    assert widget.epi_flip_angle_deg.value() == pytest.approx(expected, abs=0.01)
    assert not widget.epi_flip_angle_deg.isEnabled()
    assert "T2 is not used" in widget.epi_ernst_info.text()
    t2_independent_angle = widget.epi_flip_angle_deg.value()
    widget.t2_ms.setValue(750.0)
    assert widget.epi_flip_angle_deg.value() == t2_independent_angle

    epi_parameters = widget._epi_pulseq_parameters()
    assert epi_parameters["rf_spoiling"]
    assert epi_parameters["flip_angle_deg"] == pytest.approx(expected)

    widget.epi_rf_spoiling.setChecked(False)
    assert widget.epi_use_ernst_angle.isChecked()
    assert widget.epi_use_ernst_angle.isEnabled()
    assert not widget.epi_flip_angle_deg.isEnabled()
    assert not widget.epi_rf_spoiling_increment_deg.isEnabled()
    unspoiled_parameters = widget._epi_pulseq_parameters()
    assert not unspoiled_parameters["rf_spoiling"]
    assert unspoiled_parameters["flip_angle_deg"] == pytest.approx(expected)

    assert widget.flash_use_ernst_angle.isEnabled()
    widget.flash_rf_spoiling.setChecked(False)
    assert widget.flash_use_ernst_angle.isEnabled()
    assert not widget.flash_rf_spoiling_increment_deg.isEnabled()
    assert widget.csi_use_ernst_angle.isEnabled()
    assert not widget.csi_rf_spoiling_increment_deg.isEnabled()
    widget.csi_rf_spoiling.setChecked(True)
    assert widget.csi_rf_spoiling_increment_deg.isEnabled()

    widget.flash_rf_spoiling.setChecked(True)
    widget.flash_use_ernst_angle.setChecked(True)
    widget.flash_read_matrix.setValue(4)
    widget.flash_phase_matrix.setValue(2)
    widget.flash_repetition_time_ms.setValue(15.0)
    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
    widget.generate_sequence_button.click()
    app.processEvents()

    definitions = widget.program.metadata["definitions"]
    assert bool(definitions["UseErnstAngle"])
    assert not bool(definitions["ErnstUsesT2"])
    assert not widget.sequence_summary_table.isHidden()
    summary = {
        widget.sequence_summary_table.item(row, 0)
        .text(): widget.sequence_summary_table.item(row, 1)
        .text()
        for row in range(widget.sequence_summary_table.rowCount())
    }
    assert summary == {"Sequence name": "internal-flash-2d"}

    widget.close()
    widget.deleteLater()
    app.processEvents()
