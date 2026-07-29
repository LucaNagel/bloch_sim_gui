from pathlib import Path
from unittest.mock import patch

import numpy as np
import nbformat
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence import (
    SequenceCompiler,
    SequenceSimulationResult,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    infer_spectroscopic_acquisition,
    infer_spiral_acquisition,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_epi,
    make_pulseq_spiral,
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
        repetitions=3,
    )
    program = _write_and_load(sequence, tmp_path / "csi.seq")
    acquisition = infer_spectroscopic_acquisition(program)

    assert sequence.check_timing()[0]
    assert acquisition.matrix == (2, 3)
    assert acquisition.num_repetitions == 3
    assert acquisition.num_encodings == 2 * 3 * 3
    assert acquisition.spectral_points == 8
    assert acquisition.spectral_bandwidth_hz == pytest.approx(2000.0)
    assert program.metadata["definitions"]["Name"] == "csi_2d"
    assert program.metadata["definitions"]["Repetitions"] == 3
    chronological = np.arange(acquisition.num_samples)
    grid = acquisition.reshape_signal(chronological)
    assert grid.shape == (3, 3, 2, 8)
    for repetition in range(3):
        center_event = acquisition.encoding_event_index(repetition, 1, 1)
        expected = chronological[center_event * 8 : (center_event + 1) * 8]
        assert np.array_equal(grid[repetition, 1, 1], expected)
    result = SequenceSimulationResult(
        signal=chronological.astype(np.complex128),
        adc_times_s=SequenceCompiler().compile(program).adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={"spectroscopic_acquisition": acquisition.to_metadata()},
    )
    assert result.to_xarray()["csi_kspace"].dims == (
        "repetition",
        "phase_y",
        "phase_x",
        "spectral_point",
    )


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


def test_epi_builder_applies_edge_to_edge_slice_gap(tmp_path):
    sequence = make_pulseq_epi(
        matrix=(4, 3),
        n_slices=3,
        slice_thickness_m=4e-3,
        slice_gap_m=2e-3,
        repetition_time_s=100e-3,
    )
    program = _write_and_load(sequence, tmp_path / "epi_gap.seq")
    definitions = program.metadata["definitions"]

    assert definitions["SliceGap"] == pytest.approx(2e-3)
    assert definitions["SliceSpacing"] == pytest.approx(6e-3)
    assert definitions["SlicePositions"] == pytest.approx([-6e-3, 0.0, 6e-3])
    assert definitions["FOV"] == pytest.approx([0.22, 0.22, 16e-3])
    assert len({event.frequency_offset_hz for event in program.rf_events}) == 3


@pytest.mark.parametrize(
    ("pulse_type", "expected_tbw"),
    [("sinc", 3.5), ("slr", 3.5), ("block", 1.0)],
)
def test_epi_builder_exports_configurable_rf_pulse_properties(
    tmp_path, pulse_type, expected_tbw
):
    sequence = make_pulseq_epi(
        matrix=(4, 3),
        flip_angle_deg=35.0,
        rf_pulse_type=pulse_type,
        rf_duration_s=2.5e-3,
        rf_time_bandwidth_product=3.5,
        rf_apodization=0.25,
        rf_slr_sharpness=5.0,
        repetition_time_s=50e-3,
    )
    program = _write_and_load(sequence, tmp_path / f"epi_{pulse_type}.seq")
    definitions = program.metadata["definitions"]
    rf = program.rf_events[0]

    assert sequence.check_timing()[0]
    assert definitions["RFPulseType"] == pulse_type
    assert definitions["RFDuration"] == pytest.approx(2.5e-3)
    assert definitions["RFTimeBandwidthProduct"] == pytest.approx(expected_tbw)
    assert rf.samples_hz.size * rf.raster_s == pytest.approx(2.5e-3)
    assert 360.0 * abs(np.sum(rf.samples_hz) * rf.raster_s) == pytest.approx(
        35.0, abs=2e-3
    )
    if pulse_type == "sinc":
        assert definitions["RFApodization"] == pytest.approx(0.25)
    if pulse_type == "slr":
        assert definitions["RFSLRSharpness"] == pytest.approx(5.0)


def test_epi_builder_preserves_rf_designer_complex_shape_and_rescales_flip(tmp_path):
    raster_s = 10e-6
    sample_count = 100
    reference_flip_angle_deg = 30.0
    phase_rad = np.pi / 4.0
    reference_amplitude_hz = reference_flip_angle_deg / (
        360.0 * sample_count * raster_s
    )
    waveform_hz = np.full(
        sample_count,
        reference_amplitude_hz * np.exp(1j * phase_rad),
        dtype=np.complex128,
    )
    sequence = make_pulseq_epi(
        matrix=(4, 3),
        flip_angle_deg=60.0,
        rf_pulse_type="designer",
        rf_duration_s=sample_count * raster_s,
        rf_time_bandwidth_product=3.0,
        rf_custom_waveform_hz=waveform_hz,
        rf_custom_raster_s=raster_s,
        rf_custom_flip_angle_deg=reference_flip_angle_deg,
        rf_custom_name="Gaussian",
        rf_frequency_offset_hz=125.0,
        repetition_time_s=50e-3,
    )
    program = _write_and_load(sequence, tmp_path / "epi_designer.seq")
    definitions = program.metadata["definitions"]
    rf = program.rf_events[0]
    integral = np.sum(rf.samples_hz) * rf.raster_s

    assert sequence.check_timing()[0]
    assert definitions["RFPulseType"] == "designer"
    assert definitions["RFDesignerPulseName"] == "Gaussian"
    assert definitions["RFDesignerFlipAngleDeg"] == pytest.approx(30.0)
    assert definitions["RFFrequencyOffset"] == pytest.approx(125.0)
    assert rf.frequency_offset_hz == pytest.approx(125.0)
    assert 360.0 * abs(integral) == pytest.approx(60.0, abs=2e-3)
    assert np.angle(integral) == pytest.approx(phase_rad, abs=2e-6)


def test_spiral_builder_round_trips_and_reconstructs_frames(tmp_path):
    sequence = make_pulseq_spiral(
        fov_m=(0.08, 0.06),
        matrix=(8, 8),
        sampling_bandwidth_hz=50_000.0,
        spiral_turns=4.0,
        n_slices=2,
        slice_thickness_m=4e-3,
        slice_gap_m=2e-3,
        repetitions=2,
        repetition_time_s=100e-3,
        spoil_after_slice=False,
    )
    program = _write_and_load(sequence, tmp_path / "spiral.seq")
    compiled = SequenceCompiler().compile_acquisition(program)
    acquisition = infer_spiral_acquisition(program, compiled=compiled)

    assert sequence.check_timing()[0]
    assert acquisition.matrix == (8, 8)
    assert acquisition.num_frames == 4
    assert acquisition.varying_axes == ("slice", "repetition")
    assert compiled.adc_times_s.size == 8 * 8 * 2 * 2
    assert program.metadata["definitions"]["Trajectory"] == "spiral"
    assert program.metadata["definitions"]["SliceGap"] == pytest.approx(2e-3)
    trajectory = acquisition.trajectory(compiled.adc_gradient_moment_cyc_per_m, 0)
    assert np.ptp(trajectory[:, 0]) > 1.0 / acquisition.fov_m[0]
    assert np.ptp(trajectory[:, 1]) > 1.0 / acquisition.fov_m[1]

    result = SequenceSimulationResult(
        signal=np.ones(acquisition.num_samples, dtype=np.complex128),
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "acquisition_dimensions": acquisition.dimensions.to_metadata(),
            "spiral_acquisition": acquisition.to_metadata(),
        },
    )
    dataset = result.to_xarray()
    assert dataset["spiral_gridded_kspace"].shape == (4, 8, 8)
    assert dataset["spiral_image"].shape == (4, 8, 8)


def test_sequence_workspace_builds_and_exports_configurable_fov(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    assert [widget.sequence_source.itemText(index) for index in range(5)] == [
        "Internal FID",
        "EPI",
        "CSI",
        "bSSFP (3D)",
        "Pulseq .seq file",
    ]

    widget.epi_read_fov_mm.setValue(80.0)
    widget.epi_phase_fov_mm.setValue(60.0)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(2)
    widget.epi_repetition_time_ms.setValue(50.0)
    widget.sequence_source.setCurrentIndex(1)
    epi_path = widget._write_pulseq_path(tmp_path / "interactive_epi")

    assert widget.acquisition.fov_m == pytest.approx((0.08, 0.06))
    assert load_pulseq(epi_path).metadata["definitions"]["FOV"] == pytest.approx(
        [0.08, 0.06, 0.003]
    )

    widget.csi_read_fov_mm.setValue(90.0)
    widget.csi_phase_fov_mm.setValue(70.0)
    widget.csi_read_matrix.setValue(2)
    widget.csi_phase_matrix.setValue(2)
    widget.csi_spectral_points.setValue(8)
    widget.csi_repetition_time_ms.setValue(30.0)
    widget.csi_repetitions.setValue(2)
    widget.sequence_source.setCurrentIndex(2)
    csi_path = widget._write_pulseq_path(tmp_path / "interactive_csi")

    assert not widget.csi_group.isHidden()
    assert widget.program.source == "internal-csi"
    assert widget.spectroscopic_acquisition.matrix == (2, 2)
    assert widget.spectroscopic_acquisition.fov_m == pytest.approx((0.09, 0.07))
    assert widget.spectroscopic_acquisition.num_repetitions == 2
    assert widget.csi_repetition_selector.maximum() == 1
    csi_definitions = load_pulseq(csi_path).metadata["definitions"]
    assert csi_definitions["Name"] == "csi_2d"
    assert csi_definitions["FOV"] == pytest.approx([0.09, 0.07, 0.01])
    compiled_csi = SequenceCompiler().compile(widget.program)
    widget.result = SequenceSimulationResult(
        signal=np.ones(widget.spectroscopic_acquisition.num_samples, dtype=complex),
        adc_times_s=compiled_csi.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=(compiled_csi.adc_gradient_moment_cyc_per_m),
        metadata={
            "spectroscopic_acquisition": (
                widget.spectroscopic_acquisition.to_metadata()
            )
        },
    )
    widget.csi_repetition_selector.setValue(1)
    widget._show_spectroscopic_result(widget.result)
    assert "repetition=1" in widget.spectrum_info.text()

    widget.bssfp_read_fov_mm.setValue(100.0)
    widget.bssfp_phase_fov_mm.setValue(80.0)
    widget.bssfp_partition_fov_mm.setValue(50.0)
    widget.bssfp_read_matrix.setValue(4)
    widget.bssfp_phase_matrix.setValue(2)
    widget.bssfp_partition_matrix.setValue(2)
    widget.bssfp_repetitions.setValue(2)
    widget.sequence_source.setCurrentIndex(3)
    bssfp_path = widget._write_pulseq_path(tmp_path / "interactive_bssfp.seq")

    assert not widget.bssfp_group.isHidden()
    assert widget.program.source == "internal-bssfp-3d"
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget.acquisition_volumes.fov_m == pytest.approx((0.1, 0.08, 0.05))
    assert widget.acquisition_volumes.fov_z_m == pytest.approx(0.05)
    assert widget.acquisition_volumes.num_volumes == 2
    bssfp_definitions = load_pulseq(bssfp_path).metadata["definitions"]
    assert bssfp_definitions["Name"] == "bssfp_3d"
    assert bssfp_definitions["FOV"] == pytest.approx([0.1, 0.08, 0.05])

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_exports_pulseq_and_reproduction_notebook_by_default(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(1)
    widget.epi_read_fov_mm.setValue(80.0)
    widget.epi_phase_fov_mm.setValue(60.0)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(3)
    widget.sampling_bandwidth_khz.setValue(25.0)
    widget.epi_flip_angle_deg.setValue(33.0)
    output = tmp_path / "reproducible_epi.seq"
    both_filter = "Pulseq + Jupyter notebook (*.seq)"

    with (
        patch(
            "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getSaveFileName",
            return_value=(str(output), both_filter),
        ) as save_dialog,
        patch("blochsimulator.ui.sequence_simulation_widget.QMessageBox.information"),
    ):
        widget._export_pulseq()

    notebook_path = output.with_suffix(".ipynb")
    assert output.is_file()
    assert notebook_path.is_file()
    assert save_dialog.call_args.args[3].startswith(both_filter)

    original_sequence = output.read_text(encoding="utf-8")
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert "make_pulseq_epi" in code_cells[0]
    assert "'matrix': (4, 3)" in code_cells[0]
    assert "'sampling_bandwidth_hz': 25000.0" in code_cells[0]
    assert "'flip_angle_deg': 33.0" in code_cells[0]

    output.unlink()
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in code_cells:
        exec(compile(source, str(notebook_path), "exec"), namespace)
    assert output.read_text(encoding="utf-8") == original_sequence

    widget.close()
    widget.deleteLater()
    app.processEvents()
