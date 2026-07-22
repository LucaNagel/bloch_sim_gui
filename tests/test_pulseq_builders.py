from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence import (
    SequenceCompiler,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    infer_spectroscopic_acquisition,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_epi,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


pypulseq = pytest.importorskip("pypulseq")


def _write_and_load(sequence, path: Path):
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def test_configurable_csi_builder_round_trips_as_spectroscopic_pulseq(tmp_path):
    sequence = make_pulseq_csi(
        fov_m=(0.08, 0.06),
        matrix=(2, 3),
        spectral_bandwidth_hz=2000.0,
        spectral_points=8,
        phase_encoding_order="centric",
        repetition_time_s=30e-3,
    )
    program = _write_and_load(sequence, tmp_path / "csi.seq")
    acquisition = infer_spectroscopic_acquisition(program)

    assert sequence.check_timing()[0]
    assert acquisition.matrix == (2, 3)
    assert acquisition.spectral_points == 8
    assert acquisition.spectral_bandwidth_hz == pytest.approx(2000.0)
    assert program.metadata["definitions"]["Name"] == "csi_2d"


def test_configurable_bssfp_builder_round_trips_as_dynamic_3d_pulseq(tmp_path):
    sequence = make_pulseq_bssfp(
        fov_m=(0.08, 0.06, 0.04),
        matrix=(4, 2, 2),
        repetitions=2,
        dummy_repetitions=1,
        repetition_time_s=10e-3,
    )
    program = _write_and_load(sequence, tmp_path / "bssfp.seq")
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert sequence.check_timing()[0]
    assert compiled.adc_times_s.size == 4 * 2 * 2 * 2
    assert volumes.matrix == (4, 2, 2)
    assert volumes.num_volumes == 2
    assert volumes.varying_axes == ("repetition",)
    assert program.metadata["definitions"]["RFPhaseIncrementDeg"] == 180.0


def test_epi_builder_uses_configured_receiver_bandwidth(tmp_path):
    sequence = make_pulseq_epi(
        fov_m=(0.08, 0.06),
        matrix=(4, 3),
        sampling_bandwidth_hz=25_000.0,
        repetition_time_s=50e-3,
    )
    program = _write_and_load(sequence, tmp_path / "epi.seq")
    compiled = SequenceCompiler().compile(program)

    assert sequence.check_timing()[0]
    assert compiled.adc_times_s.size == 12
    assert program.metadata["definitions"]["SamplingBandwidth"] == pytest.approx(
        25_000.0
    )


def test_sequence_workspace_builds_and_exports_csi_and_bssfp(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    assert [widget.sequence_source.itemText(index) for index in range(5)] == [
        "Internal FID",
        "EPI",
        "CSI",
        "bSSFP (3D)",
        "Pulseq .seq file",
    ]

    widget.csi_read_matrix.setValue(2)
    widget.csi_phase_matrix.setValue(2)
    widget.csi_spectral_points.setValue(8)
    widget.csi_repetition_time_ms.setValue(30.0)
    widget.sequence_source.setCurrentIndex(2)
    csi_path = widget._write_pulseq_path(tmp_path / "interactive_csi")

    assert not widget.csi_group.isHidden()
    assert widget.program.source == "internal-csi"
    assert widget.spectroscopic_acquisition.matrix == (2, 2)
    assert load_pulseq(csi_path).metadata["definitions"]["Name"] == "csi_2d"

    widget.bssfp_read_matrix.setValue(4)
    widget.bssfp_phase_matrix.setValue(2)
    widget.bssfp_partition_matrix.setValue(2)
    widget.bssfp_repetitions.setValue(2)
    widget.sequence_source.setCurrentIndex(3)
    bssfp_path = widget._write_pulseq_path(tmp_path / "interactive_bssfp.seq")

    assert not widget.bssfp_group.isHidden()
    assert widget.program.source == "internal-bssfp-3d"
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget.acquisition_volumes.num_volumes == 2
    assert load_pulseq(bssfp_path).metadata["definitions"]["Name"] == "bssfp_3d"

    widget.close()
    widget.deleteLater()
    app.processEvents()
