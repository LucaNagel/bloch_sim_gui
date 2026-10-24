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
    make_pulseq_epsi_mge,
    make_pulseq_epi,
    make_pulseq_flash,
    make_pulseq_me_bssfp,
    make_pulseq_spectral_selective_bssfp,
    make_pulseq_spiral,
)
from blochsimulator.sequence.model import ADCEvent
from blochsimulator.sequence.rf_pulses import (
    rf_time_bandwidth_product_from_envelope,
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
        alpha_half_phase_deg=73.0,
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
    assert program.metadata["definitions"]["AlphaHalfPhaseDeg"] == 73.0
    assert np.mod(np.rad2deg(program.rf_events[0].phase_offset_rad), 360.0) == (
        pytest.approx(73.0)
    )


def test_bssfp_alpha_half_phase_defaults_to_preceding_phase_cycle_member():
    sequence = make_pulseq_bssfp(
        matrix=(2, 1, 1),
        dummy_repetitions=0,
        rf_phase_start_deg=70.0,
        rf_phase_increment_deg=30.0,
    )

    assert sequence.definitions["AlphaHalfPhaseDeg"] == pytest.approx(40.0)
    assert sequence.definitions["AlphaHalfFlipAngleDeg"] == pytest.approx(7.5)
    assert sequence.definitions["AlphaHalfCenterSpacing"] == pytest.approx(5e-3)


def test_bssfp_configured_start_phase_tr_and_flip_reach_rf_events(tmp_path):
    start_phase = 70.0
    phase_increment = 30.0
    flip_angle = 42.0
    requested_tr = 12e-3
    sequence = make_pulseq_bssfp(
        matrix=(2, 2, 1),
        dummy_repetitions=0,
        flip_angle_deg=flip_angle,
        repetition_time_s=requested_tr,
        rf_phase_start_deg=start_phase,
        rf_phase_increment_deg=phase_increment,
        alpha_half_use_ratios=True,
        alpha_half_tr_ratio=0.4,
        alpha_half_flip_ratio=0.25,
    )
    program = _write_and_load(sequence, tmp_path / "bssfp_rf_audit.seq")
    starter, first_rf, second_rf = program.rf_events[:3]

    def center_phase(event):
        return np.mod(
            np.rad2deg(
                event.phase_offset_rad
                + 2
                * np.pi
                * event.frequency_offset_hz
                * event.raster_s
                * event.samples_hz.size
                / 2
            ),
            360.0,
        )

    def rf_flip(event):
        return 360.0 * abs(np.sum(event.samples_hz) * event.raster_s)

    definitions = program.metadata["definitions"]

    assert definitions["TR"] == pytest.approx(requested_tr)
    assert definitions["FlipAngleDeg"] == pytest.approx(flip_angle)
    assert center_phase(starter) == pytest.approx(start_phase - phase_increment)
    assert center_phase(first_rf) == pytest.approx(start_phase)
    assert center_phase(second_rf) == pytest.approx(start_phase + phase_increment)
    assert first_rf.start_s - starter.start_s == pytest.approx(0.4 * requested_tr)
    assert second_rf.start_s - first_rf.start_s == pytest.approx(requested_tr)
    assert rf_flip(starter) == pytest.approx(0.25 * flip_angle, rel=1e-5)
    assert rf_flip(first_rf) == pytest.approx(flip_angle, rel=1e-5)


def test_bssfp_startup_phase_is_independent_of_regular_phase_cycle(tmp_path):
    """Sequence Mode keeps its startup phase separate from regular RF cycling."""
    sequence = make_pulseq_bssfp(
        matrix=(2, 2, 2),
        dummy_repetitions=0,
        repetitions=1,
        repetition_time_s=10e-3,
        alpha_half_phase_deg=90.0,
        rf_phase_start_deg=180.0,
        rf_phase_increment_deg=180.0,
    )
    program = _write_and_load(sequence, tmp_path / "bssfp_independent_startup.seq")
    measured_phases_deg = np.mod(
        np.rad2deg([event.phase_offset_rad for event in program.rf_events[:5]]),
        360.0,
    )

    assert measured_phases_deg == pytest.approx(
        [90.0, 180.0, 0.0, 180.0, 0.0], abs=1e-3
    )


@pytest.mark.parametrize(
    ("startup_parameters", "expected_flip_deg", "expected_spacing_s"),
    (
        (
            {
                "alpha_half_use_ratios": True,
                "alpha_half_flip_ratio": 0.25,
                "alpha_half_tr_ratio": 0.4,
            },
            20.0,
            4e-3,
        ),
        (
            {
                "alpha_half_use_ratios": False,
                "alpha_half_flip_angle_deg": 37.0,
                "alpha_half_center_spacing_s": 3e-3,
            },
            37.0,
            3e-3,
        ),
    ),
)
def test_bssfp_startup_flip_and_first_tr_support_ratio_and_absolute_modes(
    tmp_path,
    startup_parameters,
    expected_flip_deg,
    expected_spacing_s,
):
    main_flip_deg = 80.0
    sequence = make_pulseq_bssfp(
        matrix=(2, 1, 1),
        flip_angle_deg=main_flip_deg,
        repetition_time_s=10e-3,
        dummy_repetitions=0,
        **startup_parameters,
    )
    program = _write_and_load(sequence, tmp_path / "bssfp_startup.seq")
    startup_rf, first_regular_rf = program.rf_events[:2]
    startup_integral = abs(np.sum(startup_rf.samples_hz) * startup_rf.raster_s)
    regular_integral = abs(
        np.sum(first_regular_rf.samples_hz) * first_regular_rf.raster_s
    )
    definitions = program.metadata["definitions"]

    assert definitions["AlphaHalfFlipAngleDeg"] == pytest.approx(expected_flip_deg)
    assert definitions["AlphaHalfCenterSpacing"] == pytest.approx(expected_spacing_s)
    assert first_regular_rf.start_s - startup_rf.start_s == pytest.approx(
        expected_spacing_s
    )
    assert startup_integral / regular_integral == pytest.approx(
        expected_flip_deg / main_flip_deg,
        rel=1e-5,
    )


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


@pytest.mark.parametrize("builder", (make_pulseq_epi, make_pulseq_flash))
def test_centered_cartesian_adc_accepts_non_rf_raster_bandwidth(builder):
    sequence = builder(
        matrix=(8, 4),
        sampling_bandwidth_hz=150_000.0,
    )

    assert sequence.check_timing()[0]
    assert sequence.definitions["RequestedSamplingBandwidth"] == pytest.approx(
        150_000.0
    )
    assert sequence.definitions["SamplingBandwidth"] == pytest.approx(
        142_857.14285714287
    )


@pytest.mark.parametrize(
    ("strategy", "expected_directions"),
    [
        ("bipolar", (1, -1, 1, -1)),
        ("flyback", (1, 1, 1, 1)),
    ],
)
def test_epi_builder_supports_bipolar_and_flyback_readouts(
    tmp_path, strategy, expected_directions
):
    sequence = make_pulseq_epi(
        fov_m=(0.08, 0.06),
        matrix=(4, 4),
        readout_strategy=strategy,
        repetition_time_s=50e-3,
    )
    program = _write_and_load(sequence, tmp_path / f"epi_{strategy}.seq")
    acquisition = infer_cartesian_acquisition(program)

    assert sequence.check_timing()[0]
    assert acquisition.readout_directions == expected_directions
    assert program.metadata["definitions"]["ReadoutStrategy"] == strategy


@pytest.mark.parametrize("strategy", ["bipolar", "flyback"])
def test_epsi_mge_builder_round_trips_as_echo_frames(tmp_path, strategy):
    sequence = make_pulseq_epsi_mge(
        fov_m=(0.08, 0.06),
        matrix=(4, 3),
        echoes=4,
        echo_time_s=6e-3,
        echo_spacing_s=2e-3,
        readout_strategy=strategy,
        repetition_time_s=50e-3,
    )
    program = _write_and_load(sequence, tmp_path / f"epsi_mge_{strategy}.seq")
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    definitions = program.metadata["definitions"]

    assert sequence.check_timing()[0]
    assert compiled.adc_times_s.size == 4 * 3 * 4
    assert frames.num_frames == 4
    assert frames.varying_axes == ("echo",)
    assert all(frame.read_matrix == 4 for frame in frames.acquisitions)
    assert all(frame.phase_matrix == 3 for frame in frames.acquisitions)
    expected_signs = (1, -1, 1, -1) if strategy == "bipolar" else (1, 1, 1, 1)
    assert tuple(frame.readout_directions[0] for frame in frames.acquisitions) == (
        expected_signs
    )
    assert definitions["Name"] == "epsi_mge_2d"
    assert definitions["TrajectoryType"] == "cartesian_2d_multi_echo"
    assert definitions["EchoTimes"] == pytest.approx([6e-3, 8e-3, 10e-3, 12e-3])
    assert definitions["EchoSpectralBandwidth"] == pytest.approx(500.0)
    assert definitions["EchoSpectralResolution"] == pytest.approx(125.0)
    assert definitions["ReadoutStrategy"] == strategy

    result = SequenceSimulationResult(
        signal=np.zeros(compiled.adc_times_s.size, dtype=np.complex128),
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "cartesian_acquisition_frames": frames.to_metadata(),
            "sequence_definitions": definitions,
        },
    )
    dataset = result.to_xarray()

    assert dataset.cartesian_image.dims == (
        "cartesian_frame",
        "phase_y",
        "read_x",
    )
    assert dataset.cartesian_frame_echo_time_s.values == pytest.approx(
        [6e-3, 8e-3, 10e-3, 12e-3]
    )
    assert np.all(np.diff(dataset.cartesian_frame_first_adc_time_s) > 0)
    assert dataset.cartesian_read_position_m.values == pytest.approx(
        [-0.03, -0.01, 0.01, 0.03]
    )
    assert dataset.cartesian_phase_position_m.values == pytest.approx(
        [-0.02, 0.0, 0.02]
    )
    assert "measured ADC signal" in dataset.cartesian_image.attrs["description"]
    assert "not a voxel-resolved" in dataset.signal.attrs["description"]


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


@pytest.mark.parametrize("bandwidth_hz", [3000.0, 7000.0])
def test_csi_pads_adc_block_to_block_raster(tmp_path, bandwidth_hz):
    # 256 samples at the 3000 Hz raster-rounded dwell (333.3 us) do not sum to
    # the 10 us block raster on their own.
    sequence = make_pulseq_csi(
        matrix=(1, 1),
        spectral_bandwidth_hz=bandwidth_hz,
        spectral_points=256,
        repetition_time_s=0.2,
        nucleus="C13",
    )
    program = _write_and_load(sequence, tmp_path / "csi.seq")
    acquisition = infer_spectroscopic_acquisition(program)

    assert sequence.check_timing()[0]
    assert acquisition.spectral_points == 256
    assert sequence.duration()[0] == pytest.approx(0.2)


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


@pytest.mark.parametrize(
    ("encoding_axes", "n_slices"),
    ((("+x", "+y", "+z"), 1), (("-y", "+x", "+z"), 2), (("+y", "-x", "+z"), 1)),
)
def test_flash_read_phase_offsets_centre_an_off_centre_object(
    tmp_path, encoding_axes, n_slices
):
    from blochsimulator.sequence.encoding import resolve_encoding_frame
    from blochsimulator.sequence.image_export import extract_image_series

    shape = (32, 32, 1)
    positions = (np.arange(32) - 15.5) * 2e-3
    x, y = np.meshgrid(positions, positions, indexing="ij")
    disc = np.array([0.012, -0.008, 0.0])
    pd_map = ((x - disc[0]) ** 2 + (y - disc[1]) ** 2 < 0.004**2).astype(float)
    phantom = Phantom(
        shape=shape,
        fov=(0.064, 0.064, 3e-3),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
        pd_map=pd_map[..., None],
    )
    frame = resolve_encoding_frame(encoding_axes).matrix
    read_offset, phase_offset = float(disc @ frame[:, 0]), float(disc @ frame[:, 1])
    sequence = make_pulseq_flash(
        fov_m=(0.032, 0.032),
        matrix=(16, 16),
        sampling_bandwidth_hz=20e3,
        echo_time_s=4e-3,
        repetition_time_s=10e-3,
        n_slices=n_slices,
        read_offset_m=read_offset,
        phase_offset_m=phase_offset,
        encoding_axes=encoding_axes,
    )
    program = _write_and_load(sequence, tmp_path / "flash_offsets.seq")
    definitions = program.metadata["definitions"]
    assert float(definitions["ReadOffset"]) == pytest.approx(read_offset)
    assert float(definitions["PhaseOffset"]) == pytest.approx(phase_offset)

    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    acquisition = (
        result.cartesian_acquisition
        if n_slices == 1
        else result.cartesian_acquisition_frames.acquisitions[0]
    )
    assert acquisition.center_m == pytest.approx((read_offset, phase_offset))
    restored = type(acquisition).from_metadata(acquisition.to_metadata())
    assert restored.center_m == acquisition.center_m
    dataset = result.to_xarray()
    image = dataset["cartesian_image_magnitude"]
    if "cartesian_frame" in image.dims:
        image = image.isel(cartesian_frame=0)
    phase_dim, read_dim = image.dims
    weights = np.asarray(image) / float(image.sum())
    # The sequence demodulates the offsets: the disc sits in the image centre.
    centre_pixel = (
        float(weights.sum(axis=1) @ np.arange(16)),
        float(weights.sum(axis=0) @ np.arange(16)),
    )
    assert centre_pixel == pytest.approx((7.5, 7.5), abs=1e-3)
    # The image coordinates carry the shifted FOV centre.
    read_positions = dataset.coords["cartesian_read_position_m"].values
    phase_positions = dataset.coords["cartesian_phase_position_m"].values
    assert float(weights.sum(axis=0) @ read_positions) == pytest.approx(
        read_offset, abs=1e-6
    )
    assert float(weights.sum(axis=1) @ phase_positions) == pytest.approx(
        phase_offset, abs=1e-6
    )

    # DICOM, NIfTI and Bruker export place the image centre on the disc.
    series = extract_image_series(result, include_pools=False)[0]
    row_spacing, column_spacing = series.pixel_spacing_m
    first = series.frames[0]
    rows, columns = first.magnitude.shape
    centre = (
        np.asarray(first.position_m)
        + 0.5 * (columns - 1) * column_spacing * np.asarray(series.row_direction)
        + 0.5 * (rows - 1) * row_spacing * np.asarray(series.column_direction)
    )
    slice_position = -1.5e-3 if n_slices == 2 else 0.0
    assert centre == pytest.approx(disc + slice_position * frame[:, 2], abs=1e-9)


def _off_centre_disc_phantom():
    """Return a 4 mm disc at (12, -8) mm in a single-slice 64 mm phantom."""
    shape = (32, 32, 1)
    positions = (np.arange(32) - 15.5) * 2e-3
    x, y = np.meshgrid(positions, positions, indexing="ij")
    disc = np.array([0.012, -0.008, 0.0])
    pd_map = ((x - disc[0]) ** 2 + (y - disc[1]) ** 2 < 0.004**2).astype(float)
    phantom = Phantom(
        shape=shape,
        fov=(0.064, 0.064, 3e-3),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
        pd_map=pd_map[..., None],
    )
    return disc, phantom


def _exported_centre(result):
    """Return the scanner position of the first exported image's centre."""
    from blochsimulator.sequence.image_export import extract_image_series

    series = extract_image_series(result, include_pools=False)[0]
    row_spacing, column_spacing = series.pixel_spacing_m
    first = series.frames[0]
    rows, columns = first.magnitude.shape
    return (
        np.asarray(first.position_m)
        + 0.5 * (columns - 1) * column_spacing * np.asarray(series.row_direction)
        + 0.5 * (rows - 1) * row_spacing * np.asarray(series.column_direction)
    )


_ME_BSSFP_OPTIONS = dict(
    fov_m=(0.032, 0.032, 0.012),
    matrix=(16, 16, 2),
    echoes=3,
    nucleus="H1",
    sampling_bandwidth_hz=40e3,
    repetition_time_s=12e-3,
    dummy_repetitions=0,
    # A receiver offset would add a chemical-shift displacement.
    receiver_frequency_offset_hz=0.0,
)


@pytest.mark.parametrize(
    ("builder", "encoding_axes", "options"),
    (
        (
            make_pulseq_epi,
            ("+x", "+y", "+z"),
            dict(readout_strategy="bipolar", echo_time_s=40e-3),
        ),
        (
            make_pulseq_epi,
            ("-y", "+x", "+z"),
            dict(readout_strategy="flyback", echo_time_s=40e-3),
        ),
        (
            make_pulseq_epsi_mge,
            ("+y", "-x", "+z"),
            dict(
                readout_strategy="bipolar",
                echoes=3,
                echo_spacing_s=2e-3,
                echo_time_s=6e-3,
                repetition_time_s=20e-3,
            ),
        ),
        (
            make_pulseq_epsi_mge,
            ("+x", "+y", "+z"),
            dict(
                readout_strategy="flyback",
                echoes=3,
                echo_spacing_s=2e-3,
                echo_time_s=6e-3,
                repetition_time_s=20e-3,
            ),
        ),
        (
            make_pulseq_bssfp,
            ("-y", "+x", "+z"),
            dict(fov_m=(0.032, 0.032, 0.012), matrix=(16, 16, 2), dummy_repetitions=0),
        ),
        (
            make_pulseq_spectral_selective_bssfp,
            ("+x", "+y", "+z"),
            dict(
                fov_m=(0.032, 0.032, 0.012),
                matrix=(16, 16, 2),
                nucleus="H1",
                target_metabolite_names=("water",),
                target_frequency_offsets_hz=(0.0,),
                receiver_frequency_offsets_hz=(0.0,),
                flip_angle_deg=(30.0,),
                repetition_time_s=None,
            ),
        ),
        (
            make_pulseq_me_bssfp,
            ("+y", "-x", "+z"),
            dict(_ME_BSSFP_OPTIONS, readout_strategy="flyback", echo_spacing_s=1.5e-3),
        ),
        (
            make_pulseq_me_bssfp,
            ("+x", "+y", "+z"),
            dict(_ME_BSSFP_OPTIONS, readout_strategy="symmetric", echo_spacing_s=1e-3),
        ),
    ),
    ids=(
        "epi-bipolar",
        "epi-flyback",
        "epsi-bipolar",
        "epsi-flyback",
        "bssfp",
        "ss-bssfp",
        "me-bssfp-flyback",
        "me-bssfp-symmetric",
    ),
)
def test_cartesian_read_phase_offsets_centre_an_off_centre_object(
    tmp_path, builder, encoding_axes, options
):
    from blochsimulator.sequence.encoding import resolve_encoding_frame

    disc, phantom = _off_centre_disc_phantom()
    frame = resolve_encoding_frame(encoding_axes).matrix
    read_offset, phase_offset = float(disc @ frame[:, 0]), float(disc @ frame[:, 1])
    parameters = dict(fov_m=(0.032, 0.032), matrix=(16, 16))
    parameters["sampling_bandwidth_hz"] = 20e3
    parameters.update(options)
    sequence = builder(
        read_offset_m=read_offset,
        phase_offset_m=phase_offset,
        encoding_axes=encoding_axes,
        **parameters,
    )
    program = _write_and_load(sequence, tmp_path / "offsets.seq")
    definitions = program.metadata["definitions"]
    assert float(definitions["ReadOffset"]) == pytest.approx(read_offset)
    assert float(definitions["PhaseOffset"]) == pytest.approx(phase_offset)
    assert "OffsetEncoding" not in definitions

    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    dataset = result.to_xarray()
    image = dataset["cartesian_image_magnitude"]
    image = image.sum([dim for dim in image.dims[:-2]])
    weights = np.asarray(image) / float(image.sum())
    # The sequence demodulates the offsets: the disc sits in the image centre.
    centre_pixel = (
        float(weights.sum(axis=1) @ np.arange(16)),
        float(weights.sum(axis=0) @ np.arange(16)),
    )
    assert centre_pixel == pytest.approx((7.5, 7.5), abs=1e-3)
    read_positions = dataset.coords["cartesian_read_position_m"].values
    phase_positions = dataset.coords["cartesian_phase_position_m"].values
    assert float(weights.sum(axis=0) @ read_positions) == pytest.approx(
        read_offset, abs=1e-6
    )
    assert float(weights.sum(axis=1) @ phase_positions) == pytest.approx(
        phase_offset, abs=1e-6
    )
    centre = _exported_centre(result)
    assert (centre @ frame[:, 0], centre @ frame[:, 1]) == pytest.approx(
        (read_offset, phase_offset), abs=1e-9
    )


def test_csi_read_phase_offsets_centre_an_off_centre_object(tmp_path):
    disc, phantom = _off_centre_disc_phantom()
    sequence = make_pulseq_csi(
        fov_m=(0.032, 0.032),
        matrix=(16, 16),
        spectral_points=16,
        spectral_bandwidth_hz=2000.0,
        repetition_time_s=20e-3,
        slice_thickness_m=3e-3,
        read_offset_m=float(disc[0]),
        phase_offset_m=float(disc[1]),
    )
    program = _write_and_load(sequence, tmp_path / "csi_offsets.seq")
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    spatial = result.to_xarray()["csi_spatial_fid"].isel(spectral_point=0)
    weights = np.abs(np.asarray(spatial))
    weights /= weights.sum()
    # The phase-encoding steps carry the offsets: the disc sits in the centre.
    assert (
        float(weights.sum(axis=1) @ np.arange(16)),
        float(weights.sum(axis=0) @ np.arange(16)),
    ) == pytest.approx((7.5, 7.5), abs=1e-3)
    assert _exported_centre(result)[:2] == pytest.approx(disc[:2], abs=1e-9)


@pytest.mark.parametrize("encoding_axes", (("+x", "+y", "+z"), ("-y", "+x", "+z")))
def test_spiral_reconstruction_applies_read_phase_offsets(tmp_path, encoding_axes):
    from blochsimulator.sequence.encoding import resolve_encoding_frame

    disc, phantom = _off_centre_disc_phantom()
    frame = resolve_encoding_frame(encoding_axes).matrix
    read_offset, phase_offset = float(disc @ frame[:, 0]), float(disc @ frame[:, 1])
    sequence = make_pulseq_spiral(
        fov_m=(0.032, 0.032),
        matrix=(16, 16),
        sampling_bandwidth_hz=50e3,
        read_offset_m=read_offset,
        phase_offset_m=phase_offset,
        encoding_axes=encoding_axes,
    )
    program = _write_and_load(sequence, tmp_path / "spiral_offsets.seq")
    assert program.metadata["definitions"]["OffsetEncoding"] == "reconstruction"
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert result.spiral_acquisition.offset_in_reconstruction
    restored = type(result.spiral_acquisition).from_metadata(
        result.spiral_acquisition.to_metadata()
    )
    assert restored.fov_center_xyz_m == pytest.approx(tuple(disc))
    assert restored.offset_in_reconstruction

    dataset = result.to_xarray()
    image = dataset["spiral_image_magnitude"].isel(spiral_frame=0)
    # Squared magnitude keeps the gridding blur from biasing the centroid.
    weights = np.asarray(image) ** 2
    weights /= weights.sum()
    # The reconstruction phase ramp moves the disc to the image centre, N // 2.
    assert (
        float(weights.sum(axis=1) @ np.arange(16)),
        float(weights.sum(axis=0) @ np.arange(16)),
    ) == pytest.approx((8.0, 8.0), abs=0.05)
    x_positions = dataset["spiral_position_x_m"].values
    y_positions = dataset["spiral_position_y_m"].values
    assert (
        float(weights.sum(axis=0) @ x_positions),
        float(weights.sum(axis=1) @ y_positions),
    ) == pytest.approx(tuple(disc[:2]), abs=1e-4)
    # Voxel N // 2 holds the FOV centre, half a voxel above the grid middle.
    assert _exported_centre(result)[:2] == pytest.approx(disc[:2] - 1e-3, abs=1e-9)


def test_flash_rejects_an_interval_shorter_than_one_complete_image():
    with pytest.raises(ValueError, match="too short for one FLASH image"):
        make_pulseq_flash(
            matrix=(4, 2),
            repetition_time_s=15e-3,
            repetitions=2,
            acquisition_interval_s=20e-3,
        )


@pytest.mark.parametrize("pulse_type", ["sinc", "slr", "gaussian", "block"])
def test_epi_builder_exports_configurable_rf_pulse_properties(tmp_path, pulse_type):
    sequence = make_pulseq_epi(
        matrix=(4, 3),
        flip_angle_deg=35.0,
        rf_pulse_type=pulse_type,
        rf_duration_s=2.5e-3,
        rf_time_bandwidth_product=3.5,
        rf_apodization=0.25,
        rf_slr_sharpness=5.0,
        slice_thickness_m=20e-3,
        repetition_time_s=50e-3,
    )
    program = _write_and_load(sequence, tmp_path / f"epi_{pulse_type}.seq")
    definitions = program.metadata["definitions"]
    rf = program.rf_events[0]

    assert sequence.check_timing()[0]
    assert definitions["RFPulseType"] == pulse_type
    assert definitions["RFDuration"] == pytest.approx(2.5e-3)
    expected_tbw = rf_time_bandwidth_product_from_envelope(rf.samples_hz)
    assert definitions["RFTimeBandwidthProduct"] == pytest.approx(
        expected_tbw, rel=1e-6
    )
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


@pytest.mark.parametrize("builder_name", ["csi", "flash"])
def test_other_2d_builders_accept_the_same_loaded_rf_pulse(builder_name):
    raster_s = 10e-6
    sample_count = 100
    reference_flip_angle_deg = 30.0
    waveform_hz = np.full(
        sample_count,
        reference_flip_angle_deg / (360.0 * sample_count * raster_s),
        dtype=np.complex128,
    )
    common = {
        "matrix": (2, 2),
        "flip_angle_deg": 30.0,
        "rf_pulse_type": "designer",
        "rf_duration_s": sample_count * raster_s,
        "rf_custom_waveform_hz": waveform_hz,
        "rf_custom_raster_s": raster_s,
        "rf_custom_flip_angle_deg": reference_flip_angle_deg,
        "rf_custom_name": "loaded.exc",
        "rf_frequency_offset_hz": 125.0,
    }
    if builder_name == "csi":
        sequence = make_pulseq_csi(
            spectral_points=8,
            echo_time_s=10e-3,
            repetition_time_s=100e-3,
            **common,
        )
    else:
        sequence = make_pulseq_flash(
            sampling_bandwidth_hz=10_000.0,
            echo_time_s=6e-3,
            repetition_time_s=20e-3,
            **common,
        )

    assert sequence.check_timing()[0]
    assert sequence.definitions["RFPulseType"] == "designer"
    assert sequence.definitions["RFDesignerPulseName"] == "loaded.exc"
    assert sequence.definitions["RFDesignerFlipAngleDeg"] == pytest.approx(30.0)
    assert sequence.definitions["RFFrequencyOffset"] == pytest.approx(125.0)


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
        "EPSI / MGE (2D)",
        "Pulseq .seq file",
        "RF + Slice Selection",
        "PRESS (single voxel)",
        "Inversion Recovery (2D)",
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
    assert csi_definitions["FOV"] == pytest.approx([0.09, 0.07, 0.003])
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
    widget.bssfp_alpha_half_phase_deg.setValue(73.0)
    widget.bssfp_alpha_half_use_ratios.setChecked(False)
    widget.bssfp_alpha_half_center_spacing_ms.setValue(3.0)
    widget.bssfp_alpha_half_flip_angle_deg.setValue(11.0)
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
    assert bssfp_definitions["AlphaHalfPhaseDeg"] == pytest.approx(73.0)
    assert bssfp_definitions["AlphaHalfUsesRatios"] == 0
    assert bssfp_definitions["AlphaHalfCenterSpacing"] == pytest.approx(3e-3)
    assert bssfp_definitions["AlphaHalfFlipAngleDeg"] == pytest.approx(11.0)

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("bandwidth_hz", [50e3, 100e3, 150e3, 200e3, 77e3])
@pytest.mark.parametrize("strategy", ["bipolar", "flyback"])
def test_epsi_mge_timing_plan_matches_built_sequence(bandwidth_hz, strategy):
    common = dict(
        matrix=(16, 3),
        echoes=4,
        readout_strategy=strategy,
        sampling_bandwidth_hz=bandwidth_hz,
    )
    plan = make_pulseq_epsi_mge(
        **common,
        echo_spacing_s=1e-6,
        echo_time_s=1e-6,
        repetition_time_s=1e-6,
        timing_only=True,
    )
    # Every bandwidth is moved onto a raster-compatible one, so the ADC delay
    # check cannot fail, and the planned minima are exactly buildable.
    sequence = make_pulseq_epsi_mge(
        **common,
        echo_spacing_s=plan.minimum_echo_spacing_s,
        echo_time_s=plan.minimum_echo_time_s + 1e-5,
        repetition_time_s=plan.minimum_repetition_time_s + 1e-5,
    )
    assert sequence.check_timing()[0]
    assert sequence.definitions["SamplingBandwidth"] == pytest.approx(
        plan.sampling_bandwidth_hz
    )
    assert sequence.definitions["RequestedSamplingBandwidth"] == bandwidth_hz
    assert sequence.definitions["EchoSpacing"] == pytest.approx(
        plan.minimum_echo_spacing_s
    )


@pytest.mark.parametrize("bandwidth_hz", [50e3, 100e3, 150e3, 77e3])
def test_flash_timing_plan_matches_built_sequence(bandwidth_hz):
    common = dict(matrix=(16, 3), sampling_bandwidth_hz=bandwidth_hz)
    plan = make_pulseq_flash(
        **common, echo_time_s=1e-6, repetition_time_s=1e-6, timing_only=True
    )
    sequence = make_pulseq_flash(
        **common,
        echo_time_s=plan.minimum_echo_time_s,
        repetition_time_s=plan.minimum_repetition_time_s,
    )
    assert sequence.check_timing()[0]
    assert sequence.definitions["SamplingBandwidth"] == pytest.approx(
        plan.sampling_bandwidth_hz
    )
    assert sequence.definitions["TE"] == pytest.approx(plan.minimum_echo_time_s)
    assert sequence.definitions["TR"] == pytest.approx(plan.minimum_repetition_time_s)
    # A too-short request is a plan, not an error, when only planning.
    with pytest.raises(ValueError, match="too short"):
        make_pulseq_flash(**common, echo_time_s=1e-6)


@pytest.mark.parametrize("live_preview", [False, True])
def test_sequence_workspace_resolves_flash_timing_conflicts(live_preview):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(live_preview)
    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)

    widget.flash_sampling_bandwidth_khz.setValue(150.0)
    widget._settle_flash_timing()
    assert widget.flash_sampling_bandwidth_khz.value() != pytest.approx(150.0)
    warning = widget.status.text()
    assert "Bandwidth 150" in warning
    assert widget._reload_selected_generated_sequence()
    assert widget.status.text() == warning
    assert widget.log_widget.toPlainText().count(warning) == 1

    # Raising TE pushes TR up; lowering TE afterwards never lowers TR again.
    widget.flash_repetition_time_ms.setValue(15.0)
    widget.flash_echo_time_ms.setValue(40.0)
    assert widget.flash_repetition_time_ms.value() > 40.0
    assert "TR" in widget.status.text()
    assert "FLASH timing [Warning]" in widget.log_widget.toPlainText()
    raised_tr = widget.flash_repetition_time_ms.value()
    widget.flash_echo_time_ms.setValue(5.0)
    assert widget.flash_repetition_time_ms.value() == pytest.approx(raised_tr)

    # A longer RF pulse needs a longer minimum TE and TR.
    widget.flash_rf_duration_ms.setValue(8.0)
    assert widget.flash_echo_time_ms.value() > 5.0
    assert widget._reload_selected_generated_sequence()

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("strategy", ["bipolar", "flyback"])
@pytest.mark.parametrize("bandwidth_hz", [50e3, 150e3])
def test_epi_timing_plan_matches_built_sequence(bandwidth_hz, strategy):
    common = dict(
        matrix=(16, 5),
        n_slices=3,
        readout_strategy=strategy,
        sampling_bandwidth_hz=bandwidth_hz,
    )
    plan = make_pulseq_epi(
        **common, echo_time_s=1e-6, repetition_time_s=1e-6, timing_only=True
    )
    sequence = make_pulseq_epi(
        **common,
        echo_time_s=plan.minimum_echo_time_s,
        repetition_time_s=plan.minimum_repetition_time_s,
    )
    assert sequence.check_timing()[0]
    assert sequence.definitions["TE"] == pytest.approx(plan.minimum_echo_time_s)
    assert sequence.definitions["MinimumRepetitionTime"] == pytest.approx(
        plan.minimum_repetition_time_s
    )
    with pytest.raises(ValueError, match="too short"):
        make_pulseq_epi(**common, echo_time_s=1e-6)


@pytest.mark.parametrize("live_preview", [False, True])
def test_sequence_workspace_resolves_epi_timing_conflicts(live_preview):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(live_preview)
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    widget.epi_readout_trajectory.setCurrentText("Cartesian EPI")

    widget.sampling_bandwidth_khz.setValue(150.0)
    widget._settle_epi_timing()
    assert widget.sampling_bandwidth_khz.value() != pytest.approx(150.0)
    warning = widget.status.text()
    assert "Bandwidth 150" in warning
    assert widget._reload_selected_generated_sequence()
    assert widget.status.text() == warning
    assert widget.log_widget.toPlainText().count(warning) == 1

    # A TE or slice count that no longer fits raises the interval, not vice versa.
    widget.epi_repetition_time_ms.setValue(100.0)
    widget.epi_echo_time_ms.setValue(150.0)
    assert widget.epi_repetition_time_ms.value() > 150.0
    assert "Acquisition interval" in widget.status.text()
    assert "EPI timing [Warning]" in widget.log_widget.toPlainText()
    raised = widget.epi_repetition_time_ms.value()
    widget.epi_echo_time_ms.setValue(20.0)
    assert widget.epi_repetition_time_ms.value() == pytest.approx(raised)

    widget.epi_slice_count.setValue(40)
    assert widget.epi_repetition_time_ms.value() > raised
    assert widget._reload_selected_generated_sequence()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_flyback_epi_and_epsi_mge(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(False)

    assert [
        widget.epi_readout_trajectory.itemText(index)
        for index in range(widget.epi_readout_trajectory.count())
    ] == ["Cartesian EPI", "Flyback", "Spiral"]
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    widget.epi_readout_trajectory.setCurrentText("Flyback")
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(4)
    widget.epi_repetition_time_ms.setValue(50.0)
    assert widget._reload_selected_generated_sequence()
    assert widget.acquisition.readout_directions == (1, 1, 1, 1)

    widget.sequence_source.setCurrentIndex(widget.EPSI_MGE_SOURCE)
    widget.epsi_read_matrix.setValue(4)
    widget.epsi_phase_matrix.setValue(3)
    widget.epsi_echoes.setValue(4)
    widget.epsi_readout_strategy.setCurrentText("Flyback")
    assert not widget.epsi_mge_group.isHidden()
    assert widget._reload_selected_generated_sequence()
    assert widget.program.source == "internal-epsi-mge-2d"
    assert widget.acquisition_frames.num_frames == 4
    assert widget.acquisition_frames.varying_axes == ("echo",)
    assert all(
        acquisition.readout_directions == (1, 1, 1)
        for acquisition in widget.acquisition_frames.acquisitions
    )
    output = widget._write_pulseq_path(tmp_path / "interactive_epsi_mge")
    assert load_pulseq(output).metadata["definitions"]["Name"] == "epsi_mge_2d"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_exports_pulseq_and_reproduction_notebook_by_default(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.field_strength_t.setValue(9.4)
    widget.nucleus.setCurrentText("C13")
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
    assert "'FieldStrengthT': 9.4" in code_cells[1]
    assert "'Nucleus': 'C13'" in code_cells[1]

    output.unlink()
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in code_cells:
        exec(compile(source, str(notebook_path), "exec"), namespace)
    assert output.read_text(encoding="utf-8") == original_sequence

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("live_preview", [False, True])
def test_sequence_workspace_resolves_epsi_mge_timing_conflicts(live_preview):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(live_preview)
    widget.sequence_source.setCurrentIndex(widget.EPSI_MGE_SOURCE)

    # 150 kHz is not raster-compatible: the field moves to the nearest usable
    # bandwidth and the sequence builds without a timing error.
    widget.epsi_sampling_bandwidth_khz.setValue(150.0)
    widget._settle_epsi_mge_timing()
    assert widget.epsi_sampling_bandwidth_khz.value() != pytest.approx(150.0)
    warning = widget.status.text()
    assert "Bandwidth 150" in warning
    assert widget._reload_selected_generated_sequence()
    assert widget.status.text() == warning
    assert widget.log_widget.toPlainText().count(warning) == 1

    # Raising TE1 pushes TR up, never the other way round.
    widget.epsi_repetition_time_ms.setValue(50.0)
    widget.epsi_echo_time_ms.setValue(60.0)
    assert widget.epsi_repetition_time_ms.value() > 60.0
    assert "TR" in widget.status.text()
    assert "EPSI / MGE timing [Warning]" in widget.log_widget.toPlainText()
    raised_tr = widget.epsi_repetition_time_ms.value()
    widget.epsi_echo_time_ms.setValue(6.0)
    assert widget.epsi_repetition_time_ms.value() == pytest.approx(raised_tr)

    # A longer echo train raises the echo-spacing-dependent minimum TR.
    widget.epsi_echoes.setValue(64)
    assert widget.epsi_repetition_time_ms.value() > raised_tr
    assert widget._reload_selected_generated_sequence()

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("source_name", ["FLASH_SOURCE", "EPI_SOURCE"])
def test_sequence_workspace_bandwidth_does_not_drift_on_regeneration(source_name):
    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(False)
    widget.sequence_source.setCurrentIndex(getattr(widget, source_name))
    control = (
        widget.flash_sampling_bandwidth_khz
        if source_name == "FLASH_SOURCE"
        else widget.sampling_bandwidth_khz
    )
    control.setValue(150.0)

    assert widget._reload_selected_generated_sequence()
    settled = control.value()
    assert settled == pytest.approx(142.857, abs=0.01)
    for _ in range(3):
        assert widget._reload_selected_generated_sequence()
        assert control.value() == pytest.approx(settled)

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize(
    "source_name, control_name",
    [
        ("EPI_SOURCE", "epi_repetition_time_ms"),
        ("FLASH_SOURCE", "flash_repetition_time_ms"),
        ("EPSI_MGE_SOURCE", "epsi_repetition_time_ms"),
    ],
)
def test_timing_fields_are_not_resolved_while_typing(source_name, control_name):
    from PyQt5.QtTest import QTest
    from PyQt5.QtCore import Qt

    app = QApplication.instance() or QApplication([])
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(False)
    widget.sequence_source.setCurrentIndex(getattr(widget, source_name))
    widget.show()
    control = getattr(widget, control_name)
    control.setValue(50.0)
    control.setFocus()
    control.selectAll()
    for key in "3000":
        QTest.keyClicks(control, key)
    # Nothing is applied (or corrected) until the entry is confirmed.
    assert control.value() == pytest.approx(50.0)
    QTest.keyClick(control, Qt.Key_Return)
    assert control.value() == pytest.approx(3000.0)

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_bssfp_startup_is_coherent_with_the_rf_carrier_offset(tmp_path):
    """The alpha/2 phase advance must use the realised spacing at the carrier.

    Spins that sit at the same offset from the RF carrier must see the same
    transient, independent of the absolute carrier frequency.
    """
    detunings_hz = np.array([0.0, 20.0, 40.0])
    mz = {}
    for carrier_hz in (0.0, 137.0):
        program = _write_and_load(
            make_pulseq_bssfp(
                matrix=(4, 8, 4),
                flip_angle_deg=40.0,
                repetition_time_s=5e-3,
                rf_frequency_offset_hz=carrier_hz,
                dummy_repetitions=0,
            ),
            tmp_path / f"bssfp_{carrier_hz:g}.seq",
        )
        adc_centres = np.array(
            [
                event.start_s + 0.5 * (event.num_samples - 1) * event.dwell_s
                for event in program.events
                if isinstance(event, ADCEvent)
            ]
        )
        result = BlochSimulator().simulate_sequence_probes(
            program,
            np.zeros((1, 3)),
            carrier_hz + detunings_hz,
            checkpoints_s=adc_centres,
            t1_s=50.0,
            t2_s=10.0,
        )
        mz[carrier_hz] = result.magnetization[:, 0, :, 2]

    np.testing.assert_allclose(mz[137.0], mz[0.0], atol=2e-3)
    # On resonance the alpha/2 - TR/2 catalyst leaves Mz nearly constant.
    assert np.ptp(mz[0.0][:, 0]) < 0.02
