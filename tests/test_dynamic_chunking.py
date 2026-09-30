import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicB0,
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    SpinSampling,
)
from blochsimulator.spectral_phantom import ChemicalSpecies


def _chunking_fixture():
    shape = (5, 1, 1)
    grid = np.linspace(0.0, 1.0, np.prod(shape)).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.05, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": 1.0 + 0.2 * grid,
            "Lactate": 0.05 + 0.03 * grid,
        },
        initial_spin_density_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.full(shape, 0.1),
        },
        kpl_map_s_inv=0.02 + 0.08 * grid,
        b0_map=-4.0 + 8.0 * grid,
        equilibrium_polarization=0.02,
        nucleus="C13",
    )
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 8e-4), (0.1, 0.4), "linear", "hold"),
        0.8 + 0.4 * grid,
        TimeCurve((0.0, 8e-4), (0.5, 0.3), "linear", "hold"),
    )
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve((0.0, 8e-4), (-2.0, 3.0), "linear", "hold"),
        0.5 + grid,
    )
    phantom.tx_sensitivity_map = (0.8 + 0.3 * grid) * np.exp(1j * (-0.2 + 0.4 * grid))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(shape, dtype=np.complex128),
            (0.7 + 0.2 * grid) * np.exp(1j * (0.1 + 0.3 * grid)),
        )
    )
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(0.0, np.linspace(35.0 + 5.0j, 90.0 - 15.0j, 12), raster),
            GradientEvent("x", 0.0, np.linspace(-100.0, 140.0, 24), raster),
            ADCEvent(0.0, 13, 40e-6, phase_offset_rad=0.31),
        ),
        duration_s=8e-4,
    )
    return phantom, program


def test_parent_voxel_chunk_sizes_have_one_deterministic_reduction_tree():
    phantom, program = _chunking_fixture()
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    kwargs = dict(
        sequence_kernel="native_parallel",
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 2, 1)),
        checkpoints_s=(0.0, 4e-4, 8e-4),
        simulation_timestep_s=10e-6,
    )

    chunked = [
        simulator.simulate_dynamic_sequence(
            program, phantom, parent_voxel_chunk_size=size, **kwargs
        )
        for size in (1, 2, 4)
    ]

    for candidate in chunked[1:]:
        for name in (
            "signal",
            "species_signal",
            "final_magnetization",
            "final_pool_magnetization",
            "checkpoint_magnetization",
            "checkpoint_pool_magnetization",
        ):
            assert np.array_equal(getattr(candidate, name), getattr(chunked[0], name))
    for result, expected_size in zip(chunked, (1, 2, 4)):
        assert result.metadata["dynamic_parent_chunking"] is True
        assert result.metadata["dynamic_parent_chunk_size"] == expected_size
        assert result.metadata["dynamic_reduction_group_parent_voxels"] == 1
        assert result.metadata["dynamic_signal_reduction"] == (
            "fixed-parent-streaming-pairwise"
        )
        assert result.metadata["numerical_contract"] == "float64-close"


def test_parent_voxel_chunking_matches_single_block_float64_physics():
    phantom, program = _chunking_fixture()
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    kwargs = dict(
        sequence_kernel="native_parallel",
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 2, 1)),
        checkpoints_s=(0.0, 4e-4, 8e-4),
        simulation_timestep_s=10e-6,
    )
    single = simulator.simulate_dynamic_sequence(program, phantom, **kwargs)
    chunked = simulator.simulate_dynamic_sequence(
        program, phantom, parent_voxel_chunk_size=2, **kwargs
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        **{**kwargs, "sequence_kernel": "optimized"},
    )

    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(chunked, name), getattr(single, name), rtol=2e-13, atol=2e-13
        )
        np.testing.assert_allclose(
            getattr(single, name),
            getattr(reference, name),
            rtol=2e-13,
            atol=2e-13,
        )


def test_chunked_native_reduction_is_thread_count_reproducible():
    phantom, program = _chunking_fixture()
    kwargs = dict(
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 2, 1)),
        checkpoints_s=(0.0, 4e-4, 8e-4),
        simulation_timestep_s=10e-6,
        parent_voxel_chunk_size=2,
    )
    serial = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", **kwargs
    )
    parallel = BlochSimulator(
        use_parallel=True, num_threads=4
    ).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel", **kwargs
    )

    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(serial, name), getattr(parallel, name))


def test_chunk_memory_budget_fails_before_expanded_state_allocation():
    phantom, program = _chunking_fixture()
    with pytest.raises(MemoryError, match="before dynamic state allocation"):
        BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
            program,
            phantom,
            sequence_kernel="optimized",
            spoiler_mode="gradient",
            spin_sampling=SpinSampling((2, 2, 1)),
            parent_voxel_chunk_size=1,
            memory_budget_bytes=1024,
        )


def test_native_execution_survives_the_previous_total_spin_limit():
    shape = (2, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.asarray([0.02, 0.08]).reshape(shape),
        nucleus="C13",
    )
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([70.0 + 10.0j]), raster),
            ADCEvent(raster, 1, raster),
        ),
        duration_s=2 * raster,
    )

    result = BlochSimulator(use_parallel=True, num_threads=2).simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="native_parallel",
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((65537, 1, 1)),
        simulation_timestep_s=raster,
    )

    assert result.metadata["n_simulated_spins"] == 131074
    assert result.metadata["dynamic_parent_chunking"] is True
    assert result.metadata["dynamic_parent_chunk_count"] == 2
    assert result.metadata["dynamic_parent_chunk_size"] == 1
    assert result.metadata["native_full_plan_enabled"] is True
    assert (
        result.metadata["native_full_plan_intervals"]
        == 2 * result.metadata["compiled_interval_count"]
    )
    assert np.all(np.isfinite(result.signal))
