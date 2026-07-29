import runpy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    AcquisitionDimensions,
    SequenceCompiler,
    SequenceSimulationResult,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    load_pulseq,
)
from blochsimulator.spectral_phantom import ChemicalSpecies


SPECTRAL_BSSFP_MAIN = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "sequences"
        / "scripts"
        / "generate_3d_bssfp_spectral_selective.py"
    )
)["main"]


def test_spectral_selective_3d_bssfp_cycles_rf_and_receiver_offsets(tmp_path):
    sequence = SPECTRAL_BSSFP_MAIN(
        n_read=4,
        n_phase=2,
        n_partition=2,
        n_repetition=2,
        target_frequency_offsets_hz=(-245.0, 735.0),
        receiver_frequency_offsets_hz=(0.0, 980.0),
        target_metabolite_names=("Py", "Lac"),
        flip_angle_deg=30.0,
        spectral_rf_duration=4e-3,
        spectral_rf_bandwidth_hz=150.0,
        dummy_repetitions=0,
        use_alpha_half=False,
        target_tr=15.3e-3,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    path = tmp_path / "spectral_selective_3d_bssfp.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    dimensions = AcquisitionDimensions.from_program(program)

    assert dimensions.repetition_indices == (0, 0, 0, 0, 1, 1, 1, 1)
    assert dimensions.partition_indices == (0, 0, 1, 1, 0, 0, 1, 1)
    assert sorted(
        {round(event.frequency_offset_hz) for event in program.rf_events}
    ) == [
        -245,
        735,
    ]
    assert sorted(
        {round(event.frequency_offset_hz) for event in program.adc_events}
    ) == [
        0,
        980,
    ]
    definitions = program.metadata["definitions"]
    assert definitions["DynamicFrames"] == 2
    assert definitions["SpectralTargetOffsetsHz"] == pytest.approx([-245.0, 735.0])
    assert definitions["SpectralReceiverOffsetsHz"] == pytest.approx([0.0, 980.0])
    assert definitions["FlipAngleDeg"] == pytest.approx([30.0, 30.0])
    assert definitions["SpectralRFBandwidthHz"] == pytest.approx(150.0)
    assert definitions["SpectralRFDuration"] == pytest.approx(4e-3)
    assert definitions["SpectralRFPulseType"] == "slr"
    assert definitions["SpectralSLRSharpness"] == pytest.approx(1.0)
    assert definitions["SpectralRFPulseFile"].endswith("rfpulses/SLR_sharpness_1.txt")

    rf_magnitudes = np.abs(program.rf_events[0].samples_hz)
    assert rf_magnitudes.size > 100
    assert not np.allclose(rf_magnitudes, rf_magnitudes[0])


def test_spectral_selective_3d_bssfp_defaults_match_skinner_paper():
    sequence = SPECTRAL_BSSFP_MAIN(
        n_phase=2,
        n_partition=2,
        n_repetition=1,
        dummy_repetitions=0,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    definitions = sequence.definitions
    assert definitions["TR"] == pytest.approx(6.29e-3)
    assert definitions["TE"] == pytest.approx(3.145e-3)
    assert definitions["FOV"] == pytest.approx([56e-3, 28e-3, 21e-3])
    assert definitions["MatrixSize"] == [32, 2, 2]
    assert definitions["FieldStrengthT"] == pytest.approx(7.0)
    assert definitions["Nucleus"] == "C13"
    assert definitions["SpectralTargetOffsetsHz"] == pytest.approx([1655.0, -245.0])
    assert definitions["SpectralReceiverOffsetsHz"] == pytest.approx([925.44725, 0.0])
    assert definitions["SpectralTargetNames"] == ["Lac", "Py"]
    assert definitions["FlipAngleDeg"] == pytest.approx([90.0, 4.0])
    assert definitions["SpectralRFDuration"] == pytest.approx(2.33e-3)
    assert definitions["SpectralRFBandwidthFactorHzMs"] == pytest.approx(2100.0)
    assert definitions["SpectralRFBandwidthHz"] == pytest.approx(2100.0 / 2.33)
    assert definitions["SpectralRFFWHM"] == pytest.approx(900.0)
    assert definitions["ReadoutBandwidthHz"] == pytest.approx(10_000.0)
    assert definitions["AlphaHalfCenterSpacing"] == pytest.approx(4.31e-3)
    assert definitions["EndImageSpoilerCyclesPerFOV"] == pytest.approx(4.0)
    assert definitions["EndImageSpoilerDuration"] == pytest.approx(1e-3)
    assert definitions["EndImageSpoilerAxes"] == "xyz"


def test_spectral_selective_3d_bssfp_adds_published_starter_and_volume_spoiler(
    tmp_path,
):
    fov = (56e-3, 28e-3, 21e-3)
    sequence = SPECTRAL_BSSFP_MAIN(
        fov=fov,
        n_read=4,
        n_phase=2,
        n_partition=1,
        n_repetition=2,
        dummy_repetitions=0,
    )
    path = tmp_path / "spectral_selective_spoiler.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    assert program.rf_events[1].start_s - program.rf_events[0].start_s == pytest.approx(
        4.31e-3,
        abs=1e-9,
    )
    spoiler_end_times = np.asarray(
        program.metadata["definitions"]["EndImageSpoilerEndTimes"], dtype=float
    ).reshape(-1)
    assert spoiler_end_times.size == 2
    for spoiler_end in spoiler_end_times:
        ending_events = [
            event
            for event in program.gradient_events
            if np.isclose(event.end_s, spoiler_end, rtol=0.0, atol=1e-9)
        ]
        assert {event.axis for event in ending_events} == {"x", "y", "z"}
        moments = {
            event.axis: np.sum(event.samples_hz_per_m) * event.raster_s
            for event in ending_events
        }
        for axis, axis_fov in zip("xyz", fov):
            assert moments[axis] * axis_fov == pytest.approx(4.0, abs=2e-5)


def test_spectral_selective_3d_bssfp_allows_adjusting_initial_pulse_spacing():
    requested_spacing = 5.0e-3
    sequence = SPECTRAL_BSSFP_MAIN(
        n_read=4,
        n_phase=1,
        n_partition=1,
        n_repetition=1,
        dummy_repetitions=0,
        alpha_half_center_spacing=requested_spacing,
        target_tr=8e-3,
        end_image_spoiler_cycles_per_fov=0,
    )

    rf_center_times = sequence.rf_times()[0]
    assert rf_center_times[1] - rf_center_times[0] == pytest.approx(
        requested_spacing,
        abs=1e-9,
    )
    assert sequence.definitions["AlphaHalfCenterSpacing"] == pytest.approx(
        requested_spacing
    )


def test_spectral_selective_bssfp_cartesian_inference_keeps_serialization_residual(
    tmp_path,
):
    sequence = SPECTRAL_BSSFP_MAIN(
        n_phase=16,
        n_partition=1,
        n_repetition=1,
        dummy_repetitions=0,
    )
    path = tmp_path / "spectral_selective_inference.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    acquisition = infer_cartesian_acquisition(
        program,
        compiled=SequenceCompiler().compile(program),
    )

    assert acquisition.phase_matrix == 16
    assert acquisition.read_matrix == 32
    # Text serialization leaves a tiny line-to-line residual.  It is retained
    # instead of being rounded to a half-cell that no longer validates.
    assert acquisition.kx_offset_cells == pytest.approx(0.49902336, abs=2e-6)


def test_spectral_selective_bssfp_builds_position_sorted_3d_volumes(tmp_path):
    fov = (56e-3, 28e-3, 21e-3)
    sequence = SPECTRAL_BSSFP_MAIN(
        fov=fov,
        n_read=4,
        n_phase=3,
        n_partition=2,
        n_repetition=2,
        dummy_repetitions=0,
        use_alpha_half=False,
        target_tr=8e-3,
    )
    path = tmp_path / "position_sorted_3d_bssfp.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert volumes.matrix == (4, 3, 2)
    assert volumes.num_volumes == 2
    assert volumes.varying_axes == ("repetition",)
    assert volumes.volume_indices == ((0, 0, 0, 0), (0, 0, 1, 0))
    assert volumes.volume_frame_indices == ((0, 1), (2, 3))
    assert volumes.kz_cyc_per_m * fov[2] == pytest.approx([-1.0, 0.0])

    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program,
        Phantom(
            shape=(1, 1, 1),
            fov=fov,
            t1_map=np.full((1, 1, 1), 10.0),
            t2_map=np.full((1, 1, 1), 1.0),
        ),
        simulation_timestep_s=20e-6,
    )
    inferred = result.cartesian_acquisition_volumes
    assert inferred is not None
    assert result.to_cartesian_3d_kspace().shape == (2, 2, 3, 4)

    known_image = np.zeros((2, 2, 3, 4), dtype=np.complex128)
    known_image[0, 0, 1, 2] = 1.0
    known_image[0, 1, 2, 0] = 0.25 - 0.5j
    known_image[1, 0, 0, 3] = 0.75 + 0.25j
    known_image[1, 1, 1, 1] = 0.5
    dx = inferred.fov_m[0] / inferred.read_matrix
    dy = inferred.fov_m[1] / inferred.phase_matrix
    dz = inferred.fov_m[2] / inferred.partition_matrix
    centre_phase = np.exp(
        2j
        * np.pi
        * (
            inferred.kz_cyc_per_m[:, None, None] * dz / 2
            + inferred.ky_cyc_per_m[None, :, None] * dy / 2
            + inferred.kx_cyc_per_m[None, None, :] * dx / 2
        )
    )
    axes = (-3, -2, -1)
    corrected_kspace = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(known_image, axes=axes), axes=axes),
        axes=axes,
    )
    sorted_kspace = corrected_kspace / centre_phase
    chronological = np.zeros(result.adc_times_s.size, dtype=np.complex128)
    for volume, frame_group in enumerate(inferred.volume_frame_indices):
        for z_index, frame in enumerate(frame_group):
            acquisition = inferred.frames.acquisitions[frame]
            raw = np.empty(
                (acquisition.phase_matrix, acquisition.read_matrix),
                dtype=np.complex128,
            )
            for acquired_line, phase_index in enumerate(acquisition.phase_indices):
                line = sorted_kspace[volume, z_index, phase_index]
                if acquisition.readout_directions[acquired_line] < 0:
                    line = line[::-1]
                raw[acquired_line] = line
            chronological[np.asarray(inferred.frames.sample_indices[frame])] = (
                raw.reshape(-1)
            )

    synthetic = SequenceSimulationResult(
        signal=chronological,
        adc_times_s=result.adc_times_s,
        final_magnetization=result.final_magnetization,
        checkpoint_magnetization=None,
        checkpoint_times_s=np.zeros(0),
        metadata=result.metadata,
        adc_gradient_moment_cyc_per_m=result.adc_gradient_moment_cyc_per_m,
    )
    assert np.allclose(synthetic.reconstruct_cartesian_3d(), known_image, atol=1e-12)

    dataset = synthetic.to_xarray()
    assert dataset["cartesian_3d_kspace"].dims == (
        "repetition",
        "partition_z",
        "phase_y",
        "read_x",
    )
    assert dataset["cartesian_3d_kspace"].shape == (2, 2, 3, 4)
    assert np.array_equal(dataset["repetition"], [0, 1])

    output = synthetic.save(tmp_path / "position_sorted_3d_bssfp.npz")
    with np.load(output) as exported:
        assert exported["cartesian_3d_kspace"].shape == (2, 2, 3, 4)
        assert exported["cartesian_3d_image"].shape == (2, 2, 3, 4)
        assert np.array_equal(exported["cartesian_3d_repetition_index"], [0, 1])


def test_dynamic_spectral_bssfp_exports_inferred_3d_volumes(tmp_path):
    fov = (56e-3, 28e-3, 21e-3)
    sequence = SPECTRAL_BSSFP_MAIN(
        fov=fov,
        n_read=4,
        n_phase=2,
        n_partition=2,
        n_repetition=2,
        dummy_repetitions=0,
        use_alpha_half=False,
        target_tr=8e-3,
    )
    path = tmp_path / "dynamic_spectral_3d_bssfp.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    phantom = DynamicSpectralPhantom(
        shape=(1, 1, 1),
        fov=fov,
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones((1, 1, 1)),
            "Lactate": np.zeros((1, 1, 1)),
        },
        kpl_map_s_inv=np.zeros((1, 1, 1)),
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=20e-6,
    )
    dataset = result.to_xarray()

    assert result.cartesian_acquisition_volumes is not None
    assert dataset["cartesian_3d_kspace"].dims == (
        "repetition",
        "partition_z",
        "phase_y",
        "read_x",
    )
    assert dataset["cartesian_3d_kspace"].shape == (2, 2, 2, 4)
    assert dataset["cartesian_3d_image_magnitude"].shape == (2, 2, 2, 4)
    output = result.save(tmp_path / "dynamic_spectral_3d_bssfp.nc")
    with xr.open_dataset(output) as exported:
        assert exported["cartesian_3d_kspace_real"].shape == (2, 2, 2, 4)
        assert exported["cartesian_3d_kspace_imag"].shape == (2, 2, 2, 4)
        assert exported["cartesian_3d_image_magnitude"].shape == (2, 2, 2, 4)


def test_spectral_selective_3d_bssfp_uses_metabolite_specific_flip_angles(tmp_path):
    sequence = SPECTRAL_BSSFP_MAIN(
        n_phase=1,
        n_partition=1,
        n_repetition=2,
        dummy_repetitions=0,
        use_alpha_half=False,
    )
    path = tmp_path / "spectral_selective_3d_bssfp_flip_angles.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    by_offset = {}
    for event in program.rf_events:
        by_offset.setdefault(round(event.frequency_offset_hz), []).append(
            np.max(np.abs(event.samples_hz))
        )

    assert max(by_offset[1655]) / max(by_offset[-245]) == pytest.approx(
        90.0 / 4.0,
        rel=0.02,
    )


def test_spectral_selective_3d_bssfp_accepts_readout_bandwidth():
    sequence = SPECTRAL_BSSFP_MAIN(
        n_read=32,
        n_phase=2,
        n_partition=2,
        n_repetition=1,
        readout_bandwidth_hz=20_000.0,
        dummy_repetitions=0,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    definitions = sequence.definitions
    assert definitions["ReadoutBandwidthHz"] == pytest.approx(20_000.0)
    assert definitions["ADCDwell"] == pytest.approx(50e-6)


def test_spectral_selective_3d_bssfp_can_use_legacy_rf_center_receiver_offsets(
    tmp_path,
):
    sequence = SPECTRAL_BSSFP_MAIN(
        n_read=4,
        n_phase=1,
        n_partition=1,
        n_repetition=2,
        target_frequency_offsets_hz=(-245.0, 735.0),
        receiver_frequency_offsets_hz=None,
        target_metabolite_names=("Py", "Lac"),
        flip_angle_deg=30.0,
        spectral_rf_duration=4e-3,
        spectral_rf_bandwidth_hz=150.0,
        dummy_repetitions=0,
        use_alpha_half=False,
        target_tr=15.3e-3,
    )
    path = tmp_path / "spectral_selective_3d_bssfp_legacy_adc.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    assert sorted(
        {round(event.frequency_offset_hz) for event in program.adc_events}
    ) == [-245, 735]
    assert program.metadata["definitions"][
        "SpectralReceiverOffsetsHz"
    ] == pytest.approx([-245.0, 735.0])


@pytest.mark.parametrize("value", [(), (float("nan"),), (float("inf"),)])
def test_spectral_selective_3d_bssfp_rejects_invalid_offsets(value):
    with pytest.raises(ValueError, match="target_frequency_offsets_hz"):
        SPECTRAL_BSSFP_MAIN(target_frequency_offsets_hz=value)


@pytest.mark.parametrize("value", [(), (0.0,), (0.0, float("nan"))])
def test_spectral_selective_3d_bssfp_rejects_invalid_receiver_offsets(value):
    with pytest.raises(ValueError, match="receiver_frequency_offsets_hz"):
        SPECTRAL_BSSFP_MAIN(receiver_frequency_offsets_hz=value)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_spectral_selective_3d_bssfp_rejects_invalid_frame_count(value):
    with pytest.raises(ValueError, match="n_repetition"):
        SPECTRAL_BSSFP_MAIN(n_repetition=value)
