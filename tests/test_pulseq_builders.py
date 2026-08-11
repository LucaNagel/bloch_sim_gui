from pathlib import Path
from unittest.mock import patch

import numpy as np
import nbformat
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    SequenceCompiler,
    SequenceSimulationResult,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    infer_spectroscopic_acquisition,
    infer_spiral_acquisition,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_epi,
    make_pulseq_flash,
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


def test_3d_bssfp_inference_accepts_bracketed_numeric_definitions(tmp_path):
    sequence = make_pulseq_bssfp(
        fov_m=(0.08, 0.06, 0.04),
        matrix=(4, 2, 2),
        repetitions=2,
        dummy_repetitions=0,
        use_alpha_half=False,
        repetition_time_s=10e-3,
    )
    sequence.definitions.update(
        {
            "MatrixSize": "[4. 2. 2.]",
            "EncodingMatrixSize": "[4. 2. 2.]",
            "FOV": "[0.08 0.06 0.04]",
            "EncodingFOV": "[0.08 0.06 0.04]",
            "EncodingBasisXYZ": "[1. 0. 0. 0. 1. 0. 0. 0. 1.]",
        }
    )

    program = _write_and_load(sequence, tmp_path / "bssfp_string_defs.seq")
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert volumes.matrix == (4, 2, 2)
    assert volumes.num_volumes == 2
    assert volumes.varying_axes == ("repetition",)


def test_cartesian_3d_builder_maps_read_phase_partition_to_scanner_axes(tmp_path):
    sequence = make_pulseq_bssfp(
        fov_m=(0.08, 0.06, 0.04),
        matrix=(4, 2, 2),
        repetitions=1,
        dummy_repetitions=0,
        use_alpha_half=False,
        encoding_axes=("+z", "+y", "-x"),
    )
    program = _write_and_load(sequence, tmp_path / "bssfp_read_z.seq")
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert volumes.encoding_frame.axis_codes == ("+z", "+y", "-x")
    assert volumes.matrix == (4, 2, 2)
    assert volumes.read_dimension == "read_z"
    assert volumes.partition_dimension == "partition_x"
    definitions = program.metadata["definitions"]
    assert definitions["ReadoutAxis"] == "+z"
    assert definitions["PhaseEncodingAxis"] == "+y"
    assert definitions["PartitionEncodingAxis"] == "-x"

    result = SequenceSimulationResult(
        signal=np.zeros(compiled.adc_times_s.size, dtype=np.complex128),
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={"cartesian_acquisition_volumes": volumes.to_metadata()},
    )
    dataset = result.to_xarray()
    assert dataset["cartesian_3d_kspace"].dims == (
        "partition_x",
        "phase_y",
        "read_z",
    )
    assert dataset.attrs["cartesian_encoding_axes"] == "+z +y -x"


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
    spoiler_times = np.atleast_1d(
        program.metadata["definitions"]["IdealSpoilerEndTimes"]
    )
    assert spoiler_times.size == 1
    assert compiled.transverse_crush_times_s == pytest.approx(spoiler_times)


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


def test_epi_and_csi_export_slice_orientation_offset_and_echo_time(tmp_path):
    epi = make_pulseq_epi(
        matrix=(4, 4),
        echo_time_s=15e-3,
        repetition_time_s=50e-3,
        slice_offset_m=7e-3,
        encoding_axes=("+x", "+z", "-y"),
    )
    epi_program = _write_and_load(epi, tmp_path / "epi_coronal.seq")
    epi_definitions = epi_program.metadata["definitions"]

    assert epi_definitions["ReadoutAxis"] == "+x"
    assert epi_definitions["PhaseEncodingAxis"] == "+z"
    assert epi_definitions["PartitionEncodingAxis"] == "-y"
    assert epi_definitions["SliceOffset"] == pytest.approx(7e-3)
    assert epi_definitions["SlicePositions"] == pytest.approx(7e-3)
    assert epi_definitions["TE"] == pytest.approx(15e-3)
    assert epi_program.rf_events[0].frequency_offset_hz != 0.0

    csi = make_pulseq_csi(
        matrix=(2, 2),
        spectral_points=8,
        echo_time_s=8e-3,
        repetition_time_s=30e-3,
        slice_offset_m=-4e-3,
        encoding_axes=("+y", "+z", "+x"),
    )
    csi_program = _write_and_load(csi, tmp_path / "csi_sagittal.seq")
    csi_definitions = csi_program.metadata["definitions"]

    assert csi_definitions["ReadoutAxis"] == "+y"
    assert csi_definitions["PhaseEncodingAxis"] == "+z"
    assert csi_definitions["PartitionEncodingAxis"] == "+x"
    assert csi_definitions["SliceOffset"] == pytest.approx(-4e-3)
    assert csi_definitions["TE"] == pytest.approx(8e-3, abs=10e-6)
    assert csi_program.rf_events[0].frequency_offset_hz != 0.0


def test_flash_builder_round_trips_with_spoiling_and_cartesian_frames(tmp_path):
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        repetitions=2,
        slice_offset_m=4e-3,
        encoding_axes=("+x", "+z", "-y"),
        spoiler_cycles_per_voxel=1.0,
    )
    program = _write_and_load(sequence, tmp_path / "flash.seq")
    compiled = SequenceCompiler().compile_acquisition(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    definitions = program.metadata["definitions"]

    assert sequence.check_timing()[0]
    assert compiled.adc_times_s.size == 8 * 4 * 2
    assert frames.num_frames == 2
    assert frames.varying_axes == ("repetition",)
    assert frames.acquisitions[0].encoding_frame.axis_codes == (
        "+x",
        "+z",
        "-y",
    )
    assert definitions["Name"] == "flash_2d"
    assert definitions["TE"] == pytest.approx(5e-3)
    assert definitions["TR"] == pytest.approx(15e-3)
    assert definitions["RFSpoilingIncrementDeg"] == pytest.approx(117.0)
    assert definitions["SpoilerCyclesPerVoxel"] == pytest.approx(1.0)
    assert definitions["SliceOffset"] == pytest.approx(4e-3)
    assert np.asarray(definitions["SpoilerEndTimes"]).size == 8


def test_flash_in_plane_spoiler_remains_one_cartesian_2d_image(tmp_path):
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        repetitions=1,
        spoiler_cycles_per_voxel=1.0,
    )
    program = _write_and_load(sequence, tmp_path / "flash_in_plane_spoiler.seq")
    compiled = SequenceCompiler().compile_acquisition(program)

    acquisition = infer_cartesian_acquisition(program, compiled=compiled)

    assert acquisition.read_matrix == 8
    assert acquisition.phase_matrix == 4
    assert np.ptp(np.asarray(acquisition.moment_origins_cyc_per_m)[:, 0]) > 0.0
    acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)

    shape = (8, 4, 1)
    phantom = Phantom(
        shape=shape,
        fov=(0.08, 0.06, 3e-3),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    dataset = result.to_xarray()

    reference_sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        repetitions=1,
        spoiler_cycles_per_voxel=0.0,
    )
    reference_program = _write_and_load(
        reference_sequence, tmp_path / "flash_without_in_plane_spoiler.seq"
    )
    reference = BlochSimulator(use_parallel=False).simulate_sequence(
        reference_program, phantom
    )
    reference_dataset = reference.to_xarray()

    assert result.cartesian_acquisition is not None
    assert result.cartesian_acquisition_volumes is None
    assert dataset["cartesian_kspace"].shape == (4, 8)
    assert dataset["cartesian_image_magnitude"].shape == (4, 8)
    assert "cartesian_3d_kspace" not in dataset
    assert result.signal == pytest.approx(reference.signal, abs=1e-10)
    np.testing.assert_allclose(
        dataset["cartesian_image_magnitude"].values,
        reference_dataset["cartesian_image_magnitude"].values,
        rtol=0.0,
        atol=1e-10,
    )


def test_full_image_repetitions_use_requested_start_to_start_intervals(tmp_path):
    cases = (
        (
            "csi",
            make_pulseq_csi(
                matrix=(1, 2),
                spectral_points=4,
                repetition_time_s=20e-3,
                repetitions=3,
                acquisition_interval_s=80e-3,
            ),
            80e-3,
            (0.0, 80e-3, 160e-3),
            2,
        ),
        (
            "flash",
            make_pulseq_flash(
                matrix=(4, 2),
                repetition_time_s=15e-3,
                repetitions=3,
                acquisition_interval_s=50e-3,
            ),
            50e-3,
            (0.0, 50e-3, 100e-3),
            2,
        ),
        (
            "bssfp",
            make_pulseq_bssfp(
                matrix=(4, 1, 1),
                repetition_time_s=10e-3,
                repetitions=3,
                dummy_repetitions=0,
                use_alpha_half=False,
                acquisition_interval_s=30e-3,
            ),
            30e-3,
            (0.0, 30e-3, 60e-3),
            1,
        ),
    )

    for (
        name,
        sequence,
        expected_interval,
        expected_starts,
        rf_events_per_image,
    ) in cases:
        program = _write_and_load(sequence, tmp_path / f"{name}_interval.seq")
        definitions = program.metadata["definitions"]
        starts = np.asarray(definitions["AcquisitionStartTimes"], dtype=float).reshape(
            -1
        )
        assert definitions["AcquisitionIntervalReference"] == "start-to-start"
        assert definitions["RequestedAcquisitionInterval"] == pytest.approx(
            expected_interval
        )
        assert definitions["AcquisitionInterval"] == pytest.approx(expected_interval)
        assert starts == pytest.approx(expected_starts)
        rf_starts = np.asarray([event.start_s for event in program.rf_events])
        assert np.diff(rf_starts[::rf_events_per_image]) == pytest.approx(
            np.full(len(expected_starts) - 1, expected_interval)
        )


def test_flash_rejects_an_interval_shorter_than_one_complete_image():
    with pytest.raises(ValueError, match="too short for one FLASH image"):
        make_pulseq_flash(
            matrix=(4, 2),
            repetition_time_s=15e-3,
            repetitions=2,
            acquisition_interval_s=20e-3,
        )


@pytest.mark.parametrize(
    ("pulse_type", "expected_tbw"),
    [("sinc", 3.5), ("slr", 3.5), ("gaussian", 3.5), ("block", 1.0)],
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
    assert program.metadata["definitions"]["AcquisitionInterval"] == pytest.approx(
        100e-3
    )
    assert np.asarray(
        program.metadata["definitions"]["AcquisitionStartTimes"]
    ) == pytest.approx((0.0, 100e-3))
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
    assert [
        widget.sequence_source.itemText(index)
        for index in range(widget.sequence_source.count())
    ] == [
        "Internal FID",
        "EPI",
        "CSI",
        "bSSFP (3D)",
        "SS-bSSFP (3D)",
        "Radial ME-bSSFP (3D)",
        "ME-bSSFP (3D, Cartesian)",
        "FLASH (2D)",
        "Pulseq .seq file",
    ]

    widget.epi_read_fov_mm.setValue(80.0)
    widget.epi_phase_fov_mm.setValue(60.0)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(2)
    widget.epi_repetition_time_ms.setValue(50.0)
    widget.sequence_source.setCurrentIndex(1)
    widget.generate_sequence_button.click()
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
    widget.generate_sequence_button.click()
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
    widget.bssfp_read_gradient_axis.setCurrentText("+Z")
    widget.bssfp_phase_gradient_axis.setCurrentText("+Y")
    widget.generate_sequence_button.click()
    bssfp_path = widget._write_pulseq_path(tmp_path / "interactive_bssfp.seq")

    assert not widget.bssfp_group.isHidden()
    assert widget.program.source == "internal-bssfp-3d"
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget.acquisition_volumes.fov_m == pytest.approx((0.1, 0.08, 0.05))
    assert widget.acquisition_volumes.fov_z_m == pytest.approx(0.05)
    assert widget.acquisition_volumes.num_volumes == 2
    assert widget.acquisition_volumes.encoding_frame.axis_codes == (
        "+z",
        "+y",
        "-x",
    )
    assert widget.bssfp_partition_gradient_axis.text().startswith("-X")
    bssfp_definitions = load_pulseq(bssfp_path).metadata["definitions"]
    assert bssfp_definitions["Name"] == "bssfp_3d"
    assert bssfp_definitions["FOV"] == pytest.approx([0.1, 0.08, 0.05])
    assert bssfp_definitions["ReadoutAxis"] == "+z"
    assert bssfp_definitions["PhaseEncodingAxis"] == "+y"
    assert bssfp_definitions["PartitionEncodingAxis"] == "-x"

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
