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


def _native_plan_fixture():
    shape = (4, 1, 1)
    spatial = np.linspace(0.0, 1.0, 4).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.04, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": 1.0 + 0.2 * spatial,
            "Lactate": 0.04 + 0.03 * spatial,
        },
        kpl_map_s_inv=0.01 + 0.09 * spatial,
        b0_map=-3.0 + 6.0 * spatial,
        conversion_start_s=60e-6,
        nucleus="C13",
    )
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([75.0 + 15.0j, 30.0 - 8.0j]), raster),
            GradientEvent("x", 0.0, np.linspace(-120.0, 160.0, 12), raster),
            ADCEvent(4 * raster, 6, raster, phase_offset_rad=0.27),
        ),
        duration_s=12 * raster,
        metadata={"definitions": {"IdealSpoilerEndTimes": [6 * raster]}},
    )
    return phantom, program, raster


def test_native_plan_driver_executes_rf_adc_crusher_and_checkpoints():
    phantom, program, raster = _native_plan_fixture()
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="ideal",
    )
    simulator = BlochSimulator(use_parallel=False)
    reference = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", **kwargs
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_blocks"] >= 1
    assert native.metadata["native_full_plan_intervals"] == (
        native.metadata["compiled_interval_count"]
    )
    assert native.metadata["numerical_contract"] == "float64-close"
    assert native.metadata["effective_threads"] == 1
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


def test_native_plan_adc_windows_match_single_block_signal():
    phantom, program, raster = _native_plan_fixture()
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="ideal",
        sequence_kernel="native_serial",
        # Exercise the bounded voxel-major ADC windows in both runs.
        _native_spin_major=False,
    )
    simulator = BlochSimulator(use_parallel=False)
    single = simulator.simulate_dynamic_sequence(
        program, phantom, memory_budget_bytes=None, **kwargs
    )
    previews = []
    windowed = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        memory_budget_bytes=4096,
        preview_callback=lambda fraction, signal: previews.append(
            (fraction, signal.copy())
        ),
        **kwargs,
    )

    assert single.metadata["native_full_plan_blocks"] == 1
    assert windowed.metadata["native_full_plan_enabled"] is True
    assert windowed.metadata["native_full_plan_blocks"] > 1
    np.testing.assert_array_equal(windowed.species_signal, single.species_signal)
    assert previews[-1][0] == 1.0
    np.testing.assert_array_equal(previews[-1][1], single.species_signal.sum(axis=0))


def test_native_plan_driver_supports_dynamic_b0_half_interval_integrals():
    phantom, program, raster = _native_plan_fixture()
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve(
            (0.0, 6 * raster, 12 * raster),
            (-7.0, 9.0, 2.5),
            "linear",
            "hold",
        ),
        np.linspace(0.35, 1.15, phantom.nvoxels).reshape(phantom.shape),
        (1.0, 0.65),
    )
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="ideal",
    )
    simulator = BlochSimulator(use_parallel=False)
    reference = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", **kwargs
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_intervals"] == (
        native.metadata["compiled_interval_count"]
    )
    assert native.metadata["numerical_contract"] == "float64-close"
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


def test_native_plan_driver_supports_coupled_polarized_inflow():
    phantom, program, raster = _native_plan_fixture()
    grid = np.linspace(0.0, 1.0, phantom.nvoxels).reshape(phantom.shape)
    phantom.initial_spin_density_maps = {
        "Pyruvate": 1.0 + 0.1 * grid,
        "Lactate": 0.06 + 0.02 * grid,
    }
    phantom.equilibrium_polarization = 0.15
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve(
            (0.0, 5 * raster, 12 * raster),
            (0.2, 0.7, 0.35),
            "linear",
            "hold",
        ),
        0.5 + 0.6 * grid,
        TimeCurve(
            (0.0, 7 * raster, 12 * raster),
            (0.8, 0.45, 0.3),
            "linear",
            "hold",
        ),
        arrival_delay_map_s=2 * raster * grid,
        max_step_s=raster,
    )
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve(
            (0.0, 5 * raster, 12 * raster),
            (-4.0, 7.0, 1.0),
            "linear",
            "hold",
        ),
        0.3 + 0.9 * grid,
        (1.0, 0.75),
    )
    phantom.tx_sensitivity_map = (0.7 + 0.35 * grid) * np.exp(1j * (-0.2 + 0.35 * grid))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(phantom.shape, dtype=np.complex128),
            (0.75 + 0.2 * grid) * np.exp(1j * (0.1 + 0.25 * grid)),
        )
    )
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 1, 1)),
        parent_voxel_chunk_size=2,
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = BlochSimulator(use_parallel=True, num_threads=2).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel", **kwargs
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_model"] == ("coupled-concentration-inflow")
    assert native.metadata["spatial_inflow_arrival_delays"] is True
    assert native.metadata["native_full_plan_intervals"] == 2 * (
        native.metadata["compiled_interval_count"]
    )
    assert native.metadata["native_rf_fused_blocks"] == 0
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


@pytest.mark.parametrize(
    ("source_case", "expected_model"),
    (
        ("tracked_concentration", "tracked-concentration"),
        ("direct_inflow", "direct-magnetization-inflow"),
        ("direct_inflow_with_concentration", "direct-magnetization-inflow"),
        ("polarized_inflow", "polarized-magnetization-inflow"),
    ),
)
def test_native_plan_driver_supports_remaining_two_pool_source_models(
    source_case, expected_model
):
    phantom, program, raster = _native_plan_fixture()
    grid = np.linspace(0.0, 1.0, phantom.nvoxels).reshape(phantom.shape)
    if source_case in {"tracked_concentration", "direct_inflow_with_concentration"}:
        phantom.initial_spin_density_maps = {
            "Pyruvate": 0.9 + 0.2 * grid,
            "Lactate": 0.03 + 0.04 * grid,
        }
        phantom.equilibrium_polarization = 0.11
    if source_case != "tracked_concentration":
        phantom.pyruvate_inflow = PyruvateInflow(
            TimeCurve(
                (0.0, 5 * raster, 12 * raster),
                (0.15, 0.8, 0.25),
                "linear",
                "hold",
            ),
            0.4 + 0.7 * grid,
            (
                TimeCurve(
                    (0.0, 7 * raster, 12 * raster),
                    (0.85, 0.5, 0.2),
                    "linear",
                    "hold",
                )
                if source_case == "polarized_inflow"
                else None
            ),
            arrival_delay_map_s=(
                raster * grid if source_case == "direct_inflow" else None
            ),
            max_step_s=raster,
        )
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 1, 1)),
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = BlochSimulator(use_parallel=True, num_threads=2).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel", **kwargs
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_model"] == expected_model
    assert (
        native.metadata["native_full_plan_intervals"]
        == native.metadata["compiled_interval_count"]
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
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


def test_native_plan_driver_supports_spatial_tx_subspins_and_receive_coils():
    phantom, program, raster = _native_plan_fixture()
    grid = np.linspace(0.0, 1.0, phantom.nvoxels).reshape(phantom.shape)
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve(
            (0.0, 5 * raster, 12 * raster),
            (4.0, -6.0, 3.0),
            "linear",
            "hold",
        ),
        0.4 + 0.8 * grid,
        (1.0, 0.7),
    )
    phantom.tx_sensitivity_map = (0.75 + 0.35 * grid) * np.exp(
        1j * (-0.15 + 0.3 * grid)
    )
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(phantom.shape, dtype=np.complex128),
            (0.8 + 0.1 * grid) * np.exp(1j * (0.2 + 0.2 * grid)),
        )
    )
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 1, 1)),
        parent_voxel_chunk_size=2,
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    reference = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel", **kwargs
    )

    assert native.signal.shape == (2, 6)
    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_intervals"] == 2 * (
        native.metadata["compiled_interval_count"]
    )
    assert native.metadata["tx_sensitivity"] == "spatial"
    assert native.metadata["n_rx_coils"] == 2
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


def test_native_plan_driver_supports_additional_product_pool_conversion():
    shape = (4, 1, 1)
    grid = np.linspace(0.0, 1.0, np.prod(shape)).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.04, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
            ChemicalSpecies("Alanine", -7.0, 18.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": 1.0 + 0.2 * grid,
            "Lactate": 0.03 + 0.02 * grid,
            "Alanine": 0.01 + 0.01 * grid,
        },
        initial_spin_density_maps={
            "Pyruvate": 0.9 + 0.2 * grid,
            "Lactate": 0.04 + 0.02 * grid,
            "Alanine": 0.02 + 0.01 * grid,
        },
        equilibrium_polarization=0.08,
        kpl_map_s_inv=0.03 + 0.08 * grid,
        b0_map=-3.0 + 6.0 * grid,
        conversion_start_s=40e-6,
        nucleus="C13",
    )
    phantom.kpa_map_s_inv = 0.02 + 0.05 * grid
    raster = 20e-6
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve(
            (0.0, 5 * raster, 12 * raster),
            (0.15, 0.75, 0.3),
            "linear",
            "hold",
        ),
        0.5 + 0.5 * grid,
        TimeCurve(
            (0.0, 7 * raster, 12 * raster),
            (0.8, 0.45, 0.25),
            "linear",
            "hold",
        ),
        arrival_delay_map_s=raster * grid,
        max_step_s=raster,
    )
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve(
            (0.0, 6 * raster, 12 * raster),
            (-4.0, 6.0, 1.5),
            "linear",
            "hold",
        ),
        0.4 + 0.7 * grid,
        (1.0, 0.7, 1.2),
    )
    phantom.tx_sensitivity_map = (0.75 + 0.3 * grid) * np.exp(1j * (-0.2 + 0.3 * grid))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(shape, dtype=np.complex128),
            (0.8 + 0.1 * grid) * np.exp(1j * (0.1 + 0.2 * grid)),
        )
    )
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([70.0 + 12.0j, 25.0 - 5.0j]), raster),
            GradientEvent("x", 0.0, np.linspace(-110.0, 140.0, 12), raster),
            ADCEvent(4 * raster, 6, raster, phase_offset_rad=0.19),
        ),
        duration_s=12 * raster,
        metadata={"definitions": {"IdealSpoilerEndTimes": [6 * raster]}},
    )
    kwargs = dict(
        checkpoints_s=(6 * raster, 12 * raster),
        simulation_timestep_s=raster,
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 1, 1)),
        parent_voxel_chunk_size=2,
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = BlochSimulator(use_parallel=True, num_threads=2).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel", **kwargs
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_model"] == (
        "multi-pool-coupled-concentration-inflow"
    )
    assert native.metadata["additional_product_conversion"] is True
    assert (
        native.metadata["native_full_plan_intervals"]
        == 2 * native.metadata["compiled_interval_count"]
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
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


def test_native_multi_pool_plan_without_inflow():
    shape = (3, 1, 1)
    grid = np.linspace(0.0, 1.0, np.prod(shape)).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.03, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
            ChemicalSpecies("Alanine", -7.0, 18.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": 1.0 + 0.2 * grid,
            "Lactate": 0.03 + 0.01 * grid,
            "Alanine": 0.02 + 0.01 * grid,
        },
        kpl_map_s_inv=0.02 + 0.08 * grid,
        b0_map=-2.0 + 4.0 * grid,
        nucleus="C13",
    )
    phantom.kpa_map_s_inv = 0.01 + 0.04 * grid
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([65.0 + 8.0j]), raster),
            ADCEvent(raster, 3, raster, phase_offset_rad=0.12),
        ),
        duration_s=4 * raster,
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", simulation_timestep_s=raster
    )
    native = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", simulation_timestep_s=raster
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_model"] == "multi-pool-no-inflow"
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


def test_native_multi_pool_initial_adc_uses_safe_optimized_fallback():
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
            ChemicalSpecies("Alanine", -7.0, 18.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
            "Alanine": np.zeros(shape),
        },
        kpl_map_s_inv=np.full(shape, 0.05),
        nucleus="C13",
    )
    phantom.kpa_map_s_inv = np.full(shape, 0.02)
    raster = 20e-6
    program = SequenceProgram((ADCEvent(0.0, 2, raster),), duration_s=2 * raster)
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", simulation_timestep_s=raster
    )
    requested_native = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", simulation_timestep_s=raster
    )

    assert requested_native.metadata["requested_sequence_kernel"] == "native_serial"
    assert requested_native.metadata["sequence_kernel"] == "optimized"
    assert requested_native.metadata["native_full_plan_enabled"] is False
    assert "initial-time ADC" in requested_native.metadata["native_fallback_reason"]
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
    ):
        assert np.array_equal(getattr(requested_native, name), getattr(reference, name))


@pytest.mark.parametrize(
    ("polarized", "expected_model"),
    (
        (False, "multi-pool-direct-magnetization-inflow"),
        (True, "multi-pool-polarized-magnetization-inflow"),
    ),
)
def test_native_multi_pool_plan_supports_untracked_inflow_modes(
    polarized, expected_model
):
    shape = (2, 1, 1)
    grid = np.linspace(0.0, 1.0, np.prod(shape)).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
            ChemicalSpecies("Alanine", -7.0, 18.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": 0.8 + 0.2 * grid,
            "Lactate": 0.02 + 0.01 * grid,
            "Alanine": 0.01 + 0.01 * grid,
        },
        kpl_map_s_inv=0.03 + 0.05 * grid,
        nucleus="C13",
    )
    phantom.kpa_map_s_inv = 0.01 + 0.03 * grid
    raster = 20e-6
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 4 * raster), (0.2, 0.7), "linear", "hold"),
        0.5 + 0.4 * grid,
        (
            TimeCurve((0.0, 4 * raster), (0.8, 0.3), "linear", "hold")
            if polarized
            else None
        ),
    )
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([55.0 + 7.0j]), raster),
            ADCEvent(raster, 3, raster),
        ),
        duration_s=4 * raster,
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", simulation_timestep_s=raster
    )
    native = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", simulation_timestep_s=raster
    )

    assert native.metadata["native_full_plan_enabled"] is True
    assert native.metadata["native_full_plan_model"] == expected_model
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=2e-13, atol=2e-13
        )


@pytest.mark.parametrize("sequence_kernel", ("optimized", "native_serial"))
def test_dynamic_flash_without_timestep_uses_native_rf_raster(
    tmp_path, sequence_kernel
):
    pytest.importorskip("pypulseq")
    from blochsimulator.sequence import load_pulseq, make_pulseq_flash

    phantom, _program, _raster = _native_plan_fixture()
    sequence = make_pulseq_flash(
        fov_m=phantom.fov[:2],
        matrix=(4, 2),
        sampling_bandwidth_hz=20_000.0,
        flip_angle_deg=10.0,
        rf_duration_s=0.5e-3,
        slice_thickness_m=phantom.fov[2],
        nucleus="C13",
    )
    path = tmp_path / "flash.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    assert program.rf_events[0].raster_s == pytest.approx(1e-6)

    simulator = BlochSimulator(use_parallel=False)
    native_raster = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel=sequence_kernel,
        simulation_timestep_s=None,
    )
    explicit = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel=sequence_kernel,
        simulation_timestep_s=1e-6,
    )

    assert native_raster.metadata["simulation_timestep_s"] is None
    assert explicit.metadata["simulation_timestep_s"] == pytest.approx(1e-6)
    assert native_raster.signal.size == 4 * 2
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native_raster, name),
            getattr(explicit, name),
            rtol=1e-12,
            atol=1e-12,
        )
