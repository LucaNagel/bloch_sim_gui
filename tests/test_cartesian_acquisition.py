from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    AcquisitionDimensions,
    CartesianAcquisition,
    CartesianAcquisitionFrames,
    SequenceCompiler,
    SequenceProgram,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_volumes,
    make_cartesian_epi,
)


def test_acquisition_dimensions_expand_event_indices_and_round_trip_metadata():
    dimensions = AcquisitionDimensions(
        adc_event_sample_counts=(3, 2),
        slice_indices=(4, 4),
        echo_indices=(0, 1),
        repetition_indices=(2, 2),
        segment_indices=(0, 0),
        partition_indices=(7, 7),
        source="test",
    )

    assert dimensions.num_adc_events == 2
    assert dimensions.num_samples == 5
    assert dimensions.varying_axes == ("echo",)
    assert np.array_equal(dimensions.sample_indices("echo"), [0, 0, 0, 1, 1])
    assert AcquisitionDimensions.from_metadata(dimensions.to_metadata()) == dimensions


def test_3d_volume_inference_sorts_kz_positions_not_chronological_frames():
    acquisition = CartesianAcquisition(
        read_matrix=2,
        phase_matrix=2,
        fov_m=(0.02, 0.02),
        dwell_s=10e-6,
        phase_indices=(1, 0),
        readout_directions=(-1, 1),
        kx_offset_cells=0.5,
    )
    dimensions = AcquisitionDimensions(
        adc_event_sample_counts=(2,) * 8,
        repetition_indices=(0, 0, 0, 0, 1, 1, 1, 1),
        partition_indices=(1, 1, 0, 0, 1, 1, 0, 0),
        source="test",
    )
    frames = CartesianAcquisitionFrames(
        acquisitions=(acquisition,) * 4,
        sample_indices=(
            tuple(range(0, 4)),
            tuple(range(4, 8)),
            tuple(range(8, 12)),
            tuple(range(12, 16)),
        ),
        frame_indices=(
            (0, 0, 0, 0, 1),
            (0, 0, 0, 0, 0),
            (0, 0, 1, 0, 1),
            (0, 0, 1, 0, 0),
        ),
        dimensions=dimensions,
        moment_origins_cyc_per_m=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (200.0, 200.0, 200.0),
            (200.0, 200.0, 200.0),
        ),
    )
    moments = np.empty((16, 3), dtype=float)
    for frame, kz in enumerate((0.0, -50.0, 0.0, -50.0)):
        origin = np.asarray(frames.moment_origins_cyc_per_m[frame])
        raw = np.empty((2, 2, 3), dtype=float)
        for acquired_line, phase_index in enumerate(acquisition.phase_indices):
            kx = acquisition.kx_cyc_per_m
            if acquisition.readout_directions[acquired_line] < 0:
                kx = kx[::-1]
            raw[acquired_line, :, 0] = kx
            raw[acquired_line, :, 1] = acquisition.ky_cyc_per_m[phase_index]
            raw[acquired_line, :, 2] = kz
        moments[np.asarray(frames.sample_indices[frame])] = raw.reshape(-1, 3) + origin

    program = SequenceProgram(
        events=(),
        duration_s=0.0,
        metadata={"definitions": {"FOV": (0.02, 0.02, 0.02), "MatrixSize": (2, 2, 2)}},
    )
    compiled = SimpleNamespace(
        adc_times_s=np.arange(16, dtype=float),
        adc_gradient_moment_cyc_per_m=moments,
    )
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert volumes.volume_frame_indices == ((1, 0), (3, 2))
    assert volumes.volume_indices == ((0, 0, 0, 0), (0, 0, 1, 0))
    assert volumes.kz_cyc_per_m == pytest.approx([-50.0, 0.0])


@pytest.mark.parametrize("suffix", [".npz", ".h5"])
def test_sparse_exports_include_per_sample_kspace_and_outer_indices(tmp_path, suffix):
    acquisition = CartesianAcquisition.epi(2, 2, (0.02, 0.02), 10e-6)
    program = make_cartesian_epi(acquisition)
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program,
        Phantom(
            shape=(1, 1),
            fov=(0.02, 0.02),
            t1_map=np.ones((1, 1)),
            t2_map=np.ones((1, 1)),
        ),
    )
    path = tmp_path / f"result{suffix}"
    result.save(path)

    if suffix == ".npz":
        with np.load(path) as exported:
            values = {name: exported[name] for name in exported.files}
    else:
        with h5py.File(path) as exported:
            values = {name: exported[name][...] for name in exported.keys()}

    for name in (
        "kx_cyc_per_m",
        "ky_cyc_per_m",
        "kz_cyc_per_m",
        "adc_event_index",
        "readout_sample_index",
        "repetition_index",
        "partition_index",
    ):
        assert values[name].shape == result.signal.shape
    assert np.array_equal(
        values["kx_cyc_per_m"], result.adc_gradient_moment_cyc_per_m[:, 0]
    )
    assert values["cartesian_kspace"].shape == (2, 2)
    assert values["cartesian_image"].shape == (2, 2)
    assert np.array_equal(
        values["cartesian_kspace"], acquisition.reshape_signal(result.signal)
    )
    assert np.array_equal(values["cartesian_kx_cyc_per_m"], acquisition.kx_cyc_per_m)
    assert np.array_equal(values["cartesian_ky_cyc_per_m"], acquisition.ky_cyc_per_m)


def test_cartesian_epi_xarray_export_contains_sorted_grid_and_image(tmp_path):
    shape = (4, 4)
    phantom = Phantom(
        shape=shape,
        fov=(0.04, 0.04),
        t1_map=np.full(shape, 1e9),
        t2_map=np.full(shape, 1e9),
        pd_map=np.eye(shape[0]),
    )
    acquisition = CartesianAcquisition.epi(
        read_matrix=shape[0],
        phase_matrix=shape[1],
        fov_m=phantom.fov,
        dwell_s=50e-6,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        make_cartesian_epi(
            acquisition,
            rf_duration_s=100e-6,
            prephaser_duration_s=100e-6,
            blip_duration_s=50e-6,
        ),
        phantom,
    )

    dataset = result.to_xarray()
    assert dataset["cartesian_kspace"].dims == ("phase_y", "read_x")
    assert dataset["cartesian_kspace"].shape == shape
    assert np.array_equal(
        dataset["cartesian_kspace"], acquisition.reshape_signal(result.signal)
    )
    assert np.array_equal(dataset["cartesian_kx_cyc_per_m"], acquisition.kx_cyc_per_m)
    assert np.array_equal(dataset["cartesian_ky_cyc_per_m"], acquisition.ky_cyc_per_m)

    output = result.save(tmp_path / "epi_result.nc")
    import xarray as xr

    with xr.open_dataset(output) as saved:
        assert saved["cartesian_kspace_real"].shape == shape
        assert saved["cartesian_kspace_imag"].shape == shape
        assert saved["cartesian_image_magnitude"].shape == shape


def test_single_2d_inference_rejects_varying_outer_dimensions_explicitly():
    acquisition = CartesianAcquisition.epi(4, 2, (0.04, 0.02), 10e-6)
    base = make_cartesian_epi(acquisition)
    dimensions = AcquisitionDimensions(
        adc_event_sample_counts=(4, 4),
        repetition_indices=(0, 1),
    )
    program = SequenceProgram(
        base.events,
        duration_s=base.duration_s,
        metadata={
            "acquisition_dimensions": dimensions.to_metadata(),
            "definitions": {"FOV": (0.04, 0.02)},
        },
    )

    with pytest.raises(ValueError, match="varying outer.*repetition"):
        infer_cartesian_acquisition(program)


def test_cartesian_layout_bandwidth_and_epi_reordering():
    acquisition = CartesianAcquisition.epi(
        read_matrix=4,
        phase_matrix=3,
        fov_m=(0.04, 0.03),
        dwell_s=10e-6,
    )
    assert acquisition.sampling_bandwidth_hz == pytest.approx(100_000.0)
    assert acquisition.pixel_bandwidth_hz == pytest.approx(25_000.0)

    raw = np.arange(12)
    grid = acquisition.reshape_signal(raw)
    assert np.array_equal(grid[0], [0, 1, 2, 3])
    assert np.array_equal(grid[1], [7, 6, 5, 4])
    assert np.array_equal(grid[2], [8, 9, 10, 11])


def test_cartesian_epi_compiles_to_declared_adc_grid():
    acquisition = CartesianAcquisition.epi(
        read_matrix=6,
        phase_matrix=4,
        fov_m=(0.06, 0.04),
        dwell_s=20e-6,
    )
    program = make_cartesian_epi(acquisition)
    compiled = SequenceCompiler().compile(program)
    assert compiled.adc_times_s.size == acquisition.num_samples
    acquisition.validate_adc_times(compiled.adc_times_s)
    acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)


def test_cartesian_epi_adds_configurable_spoilers_after_each_slice():
    acquisition = CartesianAcquisition.epi(
        read_matrix=4,
        phase_matrix=2,
        fov_m=(0.04, 0.02),
        dwell_s=20e-6,
    )
    slice_thickness = 4e-3
    program = make_cartesian_epi(
        acquisition,
        n_slices=2,
        slice_thickness_m=slice_thickness,
        spoil_after_slice=True,
        spoiler_cycles_per_slice=6.0,
        spoiler_cycles_per_voxel=0.25,
        spoiler_duration_s=2e-3,
    )
    definitions = program.metadata["definitions"]

    assert definitions["SpoilAfterSlice"]
    assert definitions["SpoilerAxes"] == "xyz"
    assert definitions["SpoilerDuration"] == pytest.approx(2e-3)
    assert len(definitions["SpoilerEndTimes"]) == 2
    for spoiler_end in definitions["SpoilerEndTimes"]:
        events = [
            event
            for event in program.gradient_events
            if event.end_s == pytest.approx(spoiler_end)
        ]
        assert {event.axis for event in events} == {"x", "y", "z"}
        extents = {
            "x": acquisition.fov_m[0] / acquisition.read_matrix,
            "y": acquisition.fov_m[1] / acquisition.phase_matrix,
            "z": slice_thickness,
        }
        expected = {"x": 0.25, "y": 0.25, "z": 6.0}
        for event in events:
            moment = np.sum(event.samples_hz_per_m) * event.raster_s
            assert moment * extents[event.axis] == pytest.approx(expected[event.axis])


def test_cartesian_epi_requires_slice_thickness_for_through_slice_spoiler():
    acquisition = CartesianAcquisition.epi(4, 2, (0.04, 0.02), 20e-6)
    with pytest.raises(ValueError, match="through-slice spoiler"):
        make_cartesian_epi(acquisition, spoil_after_slice=True)


def test_cartesian_epi_end_to_end_fft_recovers_object_magnitude():
    shape = (4, 4)
    pd = np.zeros(shape)
    pd[1, 2] = 1.0
    pd[3, 0] = 0.5
    phantom = Phantom(
        shape=shape,
        fov=(0.04, 0.04),
        t1_map=np.full(shape, 1e9),
        t2_map=np.full(shape, 1e9),
        pd_map=pd,
    )
    acquisition = CartesianAcquisition.epi(
        read_matrix=shape[0],
        phase_matrix=shape[1],
        fov_m=phantom.fov,
        dwell_s=50e-6,
    )
    program = make_cartesian_epi(
        acquisition,
        rf_duration_s=100e-6,
        prephaser_duration_s=100e-6,
        blip_duration_s=50e-6,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    kspace = result.to_cartesian_kspace(acquisition)
    image = result.reconstruct_cartesian(acquisition)
    assert kspace.shape == shape
    # Cartesian arrays use conventional (phase/y, read/x) display order.
    assert np.allclose(np.abs(image), pd.T, atol=1e-8)


def test_cartesian_half_cell_readout_offset_reconstructs_magnitude():
    shape = (4, 4)
    pd = np.zeros(shape)
    pd[0, 1] = 1.0
    pd[2, 3] = 0.25
    phantom = Phantom(
        shape=shape,
        fov=(0.04, 0.04),
        t1_map=np.full(shape, 1e9),
        t2_map=np.full(shape, 1e9),
        pd_map=pd,
    )
    acquisition = CartesianAcquisition(
        read_matrix=4,
        phase_matrix=4,
        fov_m=phantom.fov,
        dwell_s=50e-6,
        readout_directions=(1, -1, 1, -1),
        kx_offset_cells=0.5,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        make_cartesian_epi(
            acquisition,
            rf_duration_s=100e-6,
            prephaser_duration_s=100e-6,
            blip_duration_s=50e-6,
        ),
        phantom,
    )

    image = result.reconstruct_cartesian(acquisition)
    assert np.allclose(np.abs(image), pd.T, atol=1e-8)


def test_voxel_volume_signal_weighting_is_explicit_and_resolution_aware():
    shape = (1, 1, 1)
    phantom = Phantom(
        shape=shape,
        fov=(0.02, 0.03, 0.04),
        t1_map=np.full(shape, 1e9),
        t2_map=np.full(shape, 1e9),
        m0_map=np.asarray([[[[1.0, 0.0, 0.0]]]]),
    )
    acquisition = CartesianAcquisition(
        read_matrix=1,
        phase_matrix=1,
        fov_m=(0.02, 0.03),
        dwell_s=1e-3,
    )
    program = make_cartesian_epi(acquisition, flip_angle_deg=0.0)
    simulator = BlochSimulator(use_parallel=False)
    voxel = simulator.simulate_sequence(program, phantom)
    volume = simulator.simulate_sequence(
        program, phantom, signal_weighting="voxel_volume"
    )
    assert voxel.signal[0] == pytest.approx(1.0 + 0j)
    assert volume.signal[0] == pytest.approx(phantom.voxel_volume_m3 + 0j)

    fine_shape = (2, 2, 2)
    fine = Phantom(
        shape=fine_shape,
        fov=phantom.fov,
        t1_map=np.full(fine_shape, 1e9),
        t2_map=np.full(fine_shape, 1e9),
        m0_map=np.broadcast_to([1.0, 0.0, 0.0], fine_shape + (3,)).copy(),
    )
    fine_volume = simulator.simulate_sequence(
        program, fine, signal_weighting="voxel_volume"
    )
    assert fine_volume.signal[0] == pytest.approx(volume.signal[0])


def test_voxel_volume_weighting_requires_explicit_3d_thickness():
    shape = (1, 1)
    phantom = Phantom(
        shape=shape,
        fov=(0.02, 0.03),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
    )
    acquisition = CartesianAcquisition(1, 1, phantom.fov, 1e-3)
    with pytest.raises(ValueError, match="requires a 3D phantom"):
        BlochSimulator(use_parallel=False).simulate_sequence(
            make_cartesian_epi(acquisition),
            phantom,
            signal_weighting="voxel_volume",
        )
