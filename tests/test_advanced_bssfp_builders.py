import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator.sequence import (
    AcquisitionDimensions,
    SequenceCompiler,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    load_pulseq,
    make_pulseq_me_bssfp,
    make_pulseq_radial_me_bssfp,
    make_pulseq_spectral_selective_bssfp,
    spiral_phyllotaxis_directions,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


RADIAL_SCRIPT_MAIN = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "sequences"
        / "scripts"
        / "generate_3d_radial_me_bssfp.py"
    )
)["main"]
ME_BSSFP_SCRIPT_MAIN = runpy.run_path(
    str(Path(__file__).parents[1] / "sequences" / "scripts" / "generate_3d_me_bssfp.py")
)["main"]


def _round_trip(sequence, path):
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def test_spectral_selective_builder_cycles_targets_by_dynamic_volume(tmp_path):
    sequence = make_pulseq_spectral_selective_bssfp(
        fov_m=(56e-3, 28e-3, 21e-3),
        matrix=(4, 2, 2),
        target_frequency_offsets_hz=(-245.0, 735.0),
        receiver_frequency_offsets_hz=(0.0, 980.0),
        target_metabolite_names=("Py", "Lac"),
        flip_angle_deg=(4.0, 30.0),
        spectral_rf_duration_s=2.333e-3,
        spectral_rf_bandwidth_hz=900.0,
        encoding_duration_s=0.5e-3,
        repetition_time_s=8e-3,
        repetitions=2,
        use_alpha_half=False,
    )
    program = _round_trip(sequence, tmp_path / "ss_bssfp.seq")
    compiled = SequenceCompiler().compile_acquisition(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert sequence.check_timing()[0]
    assert volumes.matrix == (4, 2, 2)
    assert volumes.num_volumes == 2
    assert sorted(
        {round(event.frequency_offset_hz) for event in program.rf_events}
    ) == [
        -245,
        735,
    ]
    assert sorted(
        {round(event.frequency_offset_hz) for event in program.adc_events}
    ) == [0, 980]
    definitions = program.metadata["definitions"]
    assert definitions["Name"] == "spectral_selective_bssfp_3d"
    assert definitions["ReferenceDOI"] == "10.1002/mrm.29676"
    assert definitions["SpectralTargetNames"].split() == ["Py", "Lac"]
    assert definitions["DynamicFrames"] == 2


def test_spectral_selective_builder_matches_paper_readout_and_auto_encoding():
    sequence = make_pulseq_spectral_selective_bssfp(
        fov_m=(56e-3, 28e-3, 21e-3),
        matrix=(32, 16, 12),
        spectral_rf_duration_s=2.333e-3,
        sampling_bandwidth_hz=10_000.0,
        repetition_time_s=6.29e-3,
        repetitions=1,
        use_alpha_half=False,
        scanner_parameters={"max_grad_mtm": 100.0, "max_slew_tms": 1000.0},
    )

    definitions = sequence.definitions
    assert sequence.check_timing()[0]
    assert definitions["TR"] == pytest.approx(6.29e-3)
    assert definitions["TE"] == pytest.approx(6.29e-3 / 2)
    assert definitions["ADCDwell"] == pytest.approx(100e-6)
    assert definitions["ReadoutDuration"] == pytest.approx(3.2e-3)
    assert definitions["EncodingLobeDurationMode"] == "automatic"
    assert definitions["EncodingLobeDuration"] == pytest.approx(0.18e-3)

    rf_centres = sequence.rf_times()[0]
    first_adc_times = sequence.adc_times()[0][:32]
    adc_centre = (first_adc_times[0] + first_adc_times[-1]) / 2
    assert adc_centre - rf_centres[0] == pytest.approx(
        6.29e-3 / 2,
        abs=5e-6,
    )


@pytest.mark.parametrize(
    "builder,extra_parameters",
    [
        (
            make_pulseq_spectral_selective_bssfp,
            {
                "repetitions": 1,
                "use_alpha_half": False,
                "repetition_time_s": 8e-3,
                "encoding_duration_s": 0.5e-3,
            },
        ),
        (
            make_pulseq_me_bssfp,
            {"repetitions": 1, "use_alpha_half": False},
        ),
    ],
)
def test_cartesian_advanced_builders_share_read_z_encoding_frame(
    tmp_path, builder, extra_parameters
):
    sequence = builder(
        matrix=(4, 2, 2),
        encoding_axes=("+z", "+x", "+y"),
        **extra_parameters,
    )
    program = _round_trip(sequence, tmp_path / f"{builder.__name__}_read_z.seq")
    compiled = SequenceCompiler().compile_acquisition(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert volumes.encoding_frame.axis_codes == ("+z", "+x", "+y")
    assert program.metadata["definitions"]["ReadoutAxis"] == "+z"


@pytest.mark.parametrize("readout_strategy", ["flyback", "symmetric"])
def test_cartesian_me_bssfp_builds_echo_volumes_and_balances_gradients(
    tmp_path, readout_strategy
):
    sequence = make_pulseq_me_bssfp(
        matrix=(8, 4, 3),
        readout_strategy=readout_strategy,
        repetitions=2,
    )
    program = _round_trip(sequence, tmp_path / f"me_bssfp_{readout_strategy}.seq")
    compiled = SequenceCompiler().compile_acquisition(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )
    definitions = program.metadata["definitions"]

    assert sequence.check_timing()[0]
    assert definitions["Name"] == "me_bssfp_3d"
    assert definitions["TrajectoryType"] == "cartesian_3d_multi_echo"
    assert definitions["ReadoutStrategy"] == readout_strategy
    assert definitions["Echoes"] == 5
    assert definitions["EchoSpacing"] == pytest.approx(1.32e-3)
    assert definitions["EchoTimes"] == pytest.approx(
        [1.71e-3, 3.03e-3, 4.35e-3, 5.67e-3, 6.99e-3]
    )
    assert volumes.matrix == (8, 4, 3)
    assert volumes.num_volumes == 10
    assert volumes.varying_axes == ("echo", "repetition")
    assert len(program.adc_events) == 2 * 3 * 4 * 5
    for acquisition, frame_index in zip(frames.acquisitions, frames.frame_indices):
        echo_index = frame_index[1]
        expected_direction = (
            1 if readout_strategy == "flyback" or echo_index % 2 == 0 else -1
        )
        assert set(acquisition.readout_directions) == {expected_direction}

    total_moment = np.zeros(3)
    axis_index = {"x": 0, "y": 1, "z": 2}
    for event in program.gradient_events:
        total_moment[axis_index[event.axis]] += (
            np.sum(event.samples_hz_per_m) * event.raster_s
        )
    # Pulseq text serialization rounds the repeated read/flyback amplitudes.
    assert total_moment == pytest.approx(np.zeros(3), abs=2e-2)


def test_cartesian_me_bssfp_script_uses_short_tr_thesis_preset():
    sequence = ME_BSSFP_SCRIPT_MAIN()
    definitions = sequence.definitions

    assert sequence.check_timing()[0]
    assert definitions["MatrixSize"] == [32, 16, 14]
    assert definitions["ReadoutStrategy"] == "flyback"
    assert definitions["Echoes"] == 5
    assert definitions["TR"] == pytest.approx(8.7e-3)
    assert definitions["RequestedTR"] == pytest.approx(8.696e-3)
    assert definitions["RequestedSamplingBandwidth"] == pytest.approx(39_682.5)
    assert definitions["AcquisitionTimePerVolume"] == pytest.approx(16 * 14 * 8.7e-3)


def test_radial_me_bssfp_matches_published_timing_and_balances_gradients(tmp_path):
    sequence = make_pulseq_radial_me_bssfp(
        spokes_per_measurement=1,
        measurements=1,
        use_alpha_half=False,
        use_tip_back=False,
    )
    program = _round_trip(sequence, tmp_path / "radial_me_bssfp.seq")
    compiled = SequenceCompiler().compile_acquisition(program)
    definitions = program.metadata["definitions"]

    assert sequence.check_timing()[0]
    assert definitions["Name"] == "radial_me_bssfp_3d"
    assert definitions["ReferenceDOI"] == "10.1002/mrm.30614"
    assert definitions["TR"] == pytest.approx(16e-3)
    assert definitions["Echoes"] == 5
    assert definitions["EchoSpacing"] == pytest.approx(2e-3)
    assert definitions["EchoTimes"] == pytest.approx([4e-3, 6e-3, 8e-3, 10e-3, 12e-3])
    assert definitions["BaseResolution"] == 32
    assert definitions["ReadoutOversampling"] == 2
    assert definitions["ReadoutSamples"] == 64
    assert definitions["RequestedPixelBandwidthHz"] == pytest.approx(1000.0)
    assert definitions["PixelBandwidthHz"] == pytest.approx(1041.6666667)
    assert definitions["AcquisitionTimePerMeasurement"] == pytest.approx(16e-3)
    assert definitions["RadialAcquisitionTime"] == pytest.approx(16e-3)
    assert compiled.adc_times_s.size == 5 * 64

    total_moment = np.zeros(3)
    axis_index = {"x": 0, "y": 1, "z": 2}
    for event in program.gradient_events:
        total_moment[axis_index[event.axis]] += (
            np.sum(event.samples_hz_per_m) * event.raster_s
        )
    # Pulseq text serialization rounds gradient amplitudes; the residual is
    # negligible relative to the roughly 180 cycles/m readout moment.
    assert total_moment == pytest.approx(np.zeros(3), abs=5e-3)


def test_radial_me_bssfp_labels_echoes_spokes_and_dynamic_measurements(tmp_path):
    sequence = RADIAL_SCRIPT_MAIN(
        base_resolution=4,
        readout_oversampling=2,
        spokes_per_measurement=3,
        measurements=2,
    )
    program = _round_trip(sequence, tmp_path / "radial_script.seq")
    dimensions = AcquisitionDimensions.from_program(program)

    assert len(program.adc_events) == 2 * 3 * 5
    assert dimensions.repetition_indices[:15] == (0,) * 15
    assert dimensions.repetition_indices[15:] == (1,) * 15
    assert dimensions.echo_indices[:5] == (0, 1, 2, 3, 4)
    line_indices = tuple(program.metadata["adc_label_values"]["LIN"])
    assert line_indices[:5] == (0,) * 5
    assert line_indices[5:10] == (1,) * 5


def test_spiral_phyllotaxis_rotates_each_measurement_about_z():
    directions = spiral_phyllotaxis_directions(
        7, measurements=2, inter_measurement_rotation_deg=90.0
    )
    expected = np.column_stack(
        (-directions[0, :, 1], directions[0, :, 0], directions[0, :, 2])
    )

    assert directions.shape == (2, 7, 3)
    assert np.linalg.norm(directions, axis=-1) == pytest.approx(np.ones((2, 7)))
    assert directions[1] == pytest.approx(expected)


def test_sequence_workspace_builds_all_advanced_bssfp_modes(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()

    widget.ss_bssfp_read_matrix.setValue(4)
    widget.ss_bssfp_phase_matrix.setValue(2)
    widget.ss_bssfp_partition_matrix.setValue(2)
    widget.ss_bssfp_repetition_time_ms.setValue(8.0)
    widget.sequence_source.setCurrentIndex(4)
    widget.generate_sequence_button.click()
    ss_path = widget._write_pulseq_path(tmp_path / "interactive_ss_bssfp.seq")

    assert not widget.ss_bssfp_group.isHidden()
    assert widget.program.source == "internal-spectral-selective-bssfp-3d"
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget.acquisition_volumes.num_volumes == 2
    ss_definitions = load_pulseq(ss_path).metadata["definitions"]
    assert ss_definitions["ReferenceDOI"] == "10.1002/mrm.29676"
    assert ss_definitions["FieldStrengthT"] == pytest.approx(7.0)
    assert ss_definitions["Nucleus"] == "C13"

    widget.radial_me_base_resolution.setValue(4)
    widget.radial_me_spokes.setValue(3)
    widget.radial_me_measurements.setValue(2)
    widget.sequence_source.setCurrentIndex(5)
    widget.generate_sequence_button.click()
    radial_path = widget._write_pulseq_path(tmp_path / "interactive_radial.seq")

    assert not widget.radial_me_bssfp_group.isHidden()
    assert widget.program.source == "internal-radial-me-bssfp-3d"
    assert "radial ME-bSSFP" in widget.acquisition_note
    radial_definitions = load_pulseq(radial_path).metadata["definitions"]
    assert radial_definitions["Measurements"] == 2
    assert radial_definitions["SpokesPerMeasurement"] == 3
    assert radial_definitions["FieldStrengthT"] == pytest.approx(3.0)
    assert radial_definitions["Nucleus"] == "C13"

    widget.me_bssfp_read_matrix.setValue(4)
    widget.me_bssfp_phase_matrix.setValue(2)
    widget.me_bssfp_partition_matrix.setValue(2)
    widget.me_bssfp_repetitions.setValue(2)
    widget.sequence_source.setCurrentIndex(6)
    widget.generate_sequence_button.click()
    me_path = widget._write_pulseq_path(tmp_path / "interactive_me_bssfp.seq")

    assert not widget.me_bssfp_group.isHidden()
    assert widget.program.source == "internal-me-bssfp-3d"
    assert "Cartesian 3D ME-bSSFP" in widget.acquisition_note
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget.acquisition_volumes.num_volumes == 10
    me_definitions = load_pulseq(me_path).metadata["definitions"]
    assert me_definitions["ReadoutStrategy"] == "flyback"
    assert me_definitions["Echoes"] == 5
    assert me_definitions["Repetitions"] == 2
    assert me_definitions["FieldStrengthT"] == pytest.approx(7.0)
    assert me_definitions["Nucleus"] == "C13"
    assert widget._pulseq_export_spec()[0] == "me_bssfp_3d"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_generates_paper_ss_bssfp_timing():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.set_scanner_parameters({"max_grad_mtm": 100.0, "max_slew_tms": 1000.0})
    widget.ss_bssfp_read_matrix.setValue(32)
    widget.ss_bssfp_phase_matrix.setValue(16)
    widget.ss_bssfp_partition_matrix.setValue(12)
    widget.ss_bssfp_rf_duration_ms.setValue(2.333)
    widget.ss_bssfp_bandwidth_khz.setValue(10.0)
    widget.ss_bssfp_repetition_time_ms.setValue(6.29)
    widget.ss_bssfp_repetitions.setValue(1)
    widget.ss_bssfp_alpha_half.setChecked(False)
    widget.sequence_source.setCurrentIndex(4)
    widget.generate_sequence_button.click()

    assert widget.program.source == "internal-spectral-selective-bssfp-3d"
    assert widget._generated_pulseq_sequence.definitions["TR"] == pytest.approx(6.29e-3)
    assert widget._generated_pulseq_sequence.definitions[
        "ReadoutDuration"
    ] == pytest.approx(3.2e-3)
    assert widget.ss_bssfp_encoding_duration_ms.value() == pytest.approx(0.18)

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_generation_is_explicit_by_default_and_live_preview_is_opt_in():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()

    assert not widget.sequence_live_preview.isChecked()
    widget.ss_bssfp_read_matrix.setValue(4)
    widget.ss_bssfp_phase_matrix.setValue(2)
    widget.ss_bssfp_partition_matrix.setValue(2)
    widget.ss_bssfp_repetition_time_ms.setValue(8.0)
    widget.sequence_source.setCurrentIndex(4)

    assert widget.program.source == "internal-fid"
    assert widget._sequence_generation_pending
    assert "Generate sequence" in widget.sequence_info.text()

    widget.generate_sequence_button.click()
    assert widget.program.source == "internal-spectral-selective-bssfp-3d"
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert not widget._sequence_generation_pending

    widget.ss_bssfp_read_matrix.setValue(5)
    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget._sequence_generation_pending

    widget.sequence_live_preview.setChecked(True)
    assert widget.acquisition_volumes.matrix == (5, 2, 2)
    assert not widget._sequence_generation_pending

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_spin_probe_generates_pending_sequence_and_reports_invalid_parameters():
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.ss_bssfp_read_matrix.setValue(4)
    widget.ss_bssfp_phase_matrix.setValue(2)
    widget.ss_bssfp_partition_matrix.setValue(2)
    widget.ss_bssfp_repetition_time_ms.setValue(8.0)
    widget.sequence_source.setCurrentIndex(4)
    widget.object_source.setCurrentIndex(2)

    assert widget._sequence_generation_pending
    assert widget._can_start_probe()
    assert widget.program.source == "internal-spectral-selective-bssfp-3d"
    assert not widget._sequence_generation_pending

    last_program = widget.program
    widget.ss_bssfp_repetition_time_ms.setValue(1.0)
    with patch(
        "blochsimulator.ui.sequence_simulation_widget.QMessageBox.warning"
    ) as warning:
        assert not widget._can_start_probe()

    assert widget.program is last_program
    assert widget._sequence_generation_pending
    assert warning.call_args.args[1] == "Sequence generation failed"
    assert "too short" in warning.call_args.args[2]
    assert "last valid sequence preview" in widget.sequence_info.text()

    widget.close()
    widget.deleteLater()
    app.processEvents()
