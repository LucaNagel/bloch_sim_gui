import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    AcquisitionDimensions,
    CartesianAcquisition,
    SequenceCompiler,
    SequenceProgram,
    infer_cartesian_acquisition,
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
