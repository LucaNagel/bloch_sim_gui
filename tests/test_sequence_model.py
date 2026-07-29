import numpy as np
import pytest

from blochsimulator.phantom import Phantom, PhantomFactory
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
)
from blochsimulator.units import (
    gradient_g_per_cm_to_hz_per_m,
    gradient_hz_per_m_to_g_per_cm,
    gradient_t_per_m_to_g_per_cm,
    rf_gauss_to_hz,
    rf_hz_to_gauss,
)


def test_unit_conversions_round_trip():
    rf = np.array([1 + 2j, -0.5j])
    gradient = np.array([1.0, -2.0, 0.25])
    assert np.allclose(rf_hz_to_gauss(rf_gauss_to_hz(rf)), rf)
    assert np.allclose(
        gradient_hz_per_m_to_g_per_cm(gradient_g_per_cm_to_hz_per_m(gradient)),
        gradient,
    )
    assert gradient_t_per_m_to_g_per_cm(1.0) == pytest.approx(100.0)


def test_phantom_coordinates_are_voxel_centres():
    phantom = PhantomFactory.uniform((2, 2, 2), (0.02, 0.04, 0.06), 1.0, 0.1)
    assert np.allclose(np.unique(phantom.positions[:, 0]), [-0.005, 0.005])
    assert np.allclose(np.unique(phantom.positions[:, 1]), [-0.01, 0.01])
    assert np.allclose(np.unique(phantom.positions[:, 2]), [-0.015, 0.015])


def test_split_off_resonance_maps_and_legacy_conflict():
    shape = (2, 2)
    phantom = Phantom(
        shape,
        (0.02, 0.02),
        np.ones(shape),
        np.ones(shape),
        b0_map=np.ones(shape) * 10,
        chemical_shift_map=np.ones(shape) * -3,
    )
    assert np.array_equal(phantom.effective_df_map, np.ones(shape) * 7)
    with pytest.raises(ValueError, match="cannot be combined"):
        Phantom(
            shape,
            (0.02, 0.02),
            np.ones(shape),
            np.ones(shape),
            df_map=np.zeros(shape),
            b0_map=np.zeros(shape),
        )


def test_phantom_tx_rx_defaults_and_shape_validation():
    shape = (2, 3)
    phantom = Phantom(
        shape,
        (0.02, 0.03),
        np.ones(shape),
        np.ones(shape),
    )
    assert phantom.tx_sensitivity_map.shape == shape
    assert phantom.rx_sensitivity_maps.shape == (1, *shape)
    assert phantom.n_rx_coils == 1
    with pytest.raises(ValueError, match="Tx sensitivity"):
        Phantom(
            shape,
            (0.02, 0.03),
            np.ones(shape),
            np.ones(shape),
            tx_sensitivity_map=np.ones((2, 2)),
        )
    with pytest.raises(ValueError, match="Rx sensitivity"):
        Phantom(
            shape,
            (0.02, 0.03),
            np.ones(shape),
            np.ones(shape),
            rx_sensitivity_maps=np.ones(shape),
        )


def test_sparse_compiler_collapses_long_rf_free_gradient():
    gradient = GradientEvent("x", 0.0, np.ones(100_000), 1e-6)
    adc = ADCEvent(0.05, 2, 0.025)
    program = SequenceProgram((gradient, adc), duration_s=0.1)
    compiled = SequenceCompiler().compile(program)
    # Event/ADC boundaries only; the 100,000 gradient raster points are collapsed.
    assert compiled.n_intervals == 3
    assert np.allclose(compiled.adc_times_s, [0.05, 0.075])
    assert np.sum(compiled.gradient_hz_per_m[:, 0] * compiled.dt_s) == pytest.approx(
        0.1
    )


def test_compiler_retains_extra_dynamic_boundaries():
    program = SequenceProgram((), duration_s=2.0)

    compiled = SequenceCompiler().compile(program, extra_boundaries_s=(0.25, 1.25))

    assert compiled.interval_end_s == pytest.approx([0.25, 1.25, 2.0])
    assert compiled.metadata["extra_boundary_count"] == 2


def test_compiler_rf_gradient_overlap_uses_fine_boundaries():
    rf = RFEvent(0.0, np.array([10.0, 20.0]), 1e-3)
    gradient = GradientEvent("z", 0.0, np.array([1.0, 2.0, 3.0, 4.0]), 0.5e-3)
    program = SequenceProgram((rf, gradient), duration_s=2e-3)
    compiled = SequenceCompiler().compile(program)
    assert np.allclose(compiled.dt_s, 0.5e-3)
    assert np.allclose(compiled.rf_hz, [10, 10, 20, 20])
    assert np.allclose(compiled.gradient_hz_per_m[:, 2], [1, 2, 3, 4])


def test_compiler_reports_meaningful_status_stages():
    messages = []
    program = SequenceProgram(
        (
            RFEvent(0.0, np.ones(2), 1e-3),
            ADCEvent(2e-3, 2, 1e-3),
        ),
        duration_s=4e-3,
    )

    SequenceCompiler().compile(program, status_callback=messages.append)

    assert any("Validating" in message for message in messages)
    assert any("sparse sequence timeline" in message for message in messages)
    assert any("Finalizing" in message for message in messages)


def test_acquisition_compiler_matches_full_adc_gradient_moments_without_rf_raster():
    rf = RFEvent(0.0, np.linspace(1.0, 2.0, 1000), 1e-6)
    gradient = GradientEvent("x", 0.0, np.linspace(-2.0, 3.0, 200), 10e-6)
    adc = ADCEvent(0.25e-3, 6, 0.25e-3)
    program = SequenceProgram((rf, gradient, adc), duration_s=2e-3)

    full = SequenceCompiler().compile(program)
    acquisition = SequenceCompiler().compile_acquisition(program)

    assert acquisition.metadata["acquisition_only"] is True
    assert acquisition.n_intervals < full.n_intervals
    assert np.all(acquisition.rf_hz == 0.0)
    np.testing.assert_allclose(acquisition.adc_times_s, full.adc_times_s, atol=0.0)
    np.testing.assert_allclose(
        acquisition.adc_gradient_moment_cyc_per_m,
        full.adc_gradient_moment_cyc_per_m,
        rtol=1e-13,
        atol=1e-15,
    )


def test_compiler_uses_configured_rf_active_simulation_timestep():
    rf = RFEvent(0.0, np.arange(1.0, 11.0), 1e-6)
    program = SequenceProgram((rf,), duration_s=10e-6)

    native = SequenceCompiler().compile(program)
    coarse = SequenceCompiler().compile(program, simulation_timestep_s=5e-6)

    assert native.n_intervals == 10
    assert coarse.n_intervals == 2
    assert np.allclose(coarse.dt_s, [5e-6, 5e-6])
    assert np.allclose(coarse.rf_hz, [3.0, 8.0])


def test_compiler_rejects_invalid_simulation_timestep():
    program = SequenceProgram((), duration_s=0.0)
    with pytest.raises(ValueError, match="simulation_timestep_s"):
        SequenceCompiler().compile(program, simulation_timestep_s=0.0)


def test_compiler_coalesces_numerically_duplicate_raster_boundaries():
    rf = RFEvent(100e-6, np.ones(3000), 1e-6)
    gradient = GradientEvent("z", 0.0, np.ones(400), 10e-6)
    program = SequenceProgram((rf, gradient), duration_s=4e-3)
    compiled = SequenceCompiler().compile(program)

    assert np.min(compiled.dt_s) == pytest.approx(1e-6)
    assert np.all(np.abs(compiled.rf_hz[compiled.rf_hz != 0]) > 0)


def test_adc_zero_and_checkpoint_state_indices():
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 0.5),),
        duration_s=1.5,
    )
    compiled = SequenceCompiler().compile(program, checkpoints_s=(0.25, 1.5))
    assert np.array_equal(compiled.adc_state_indices, [0, 2, 3])
    assert np.array_equal(compiled.checkpoint_state_indices, [1, 4])


def test_compiler_maps_observation_to_nearest_coalesced_boundary():
    event_end = 1.2129899999999882
    checkpoint = 1.21299
    program = SequenceProgram(
        events=(GradientEvent("x", 0.0, np.array([1.0]), event_end),),
        duration_s=1.3,
    )

    compiled = SequenceCompiler().compile(
        program,
        checkpoints_s=(checkpoint,),
        simulation_timestep_s=5e-6,
    )

    state_index = compiled.checkpoint_state_indices[0]
    boundaries = np.concatenate(([0.0], compiled.interval_end_s))
    assert compiled.checkpoint_times_s == pytest.approx([checkpoint])
    assert boundaries[state_index] == pytest.approx(event_end, abs=1e-13)


def test_same_axis_gradient_overlap_rejected():
    program = SequenceProgram(
        (
            GradientEvent("x", 0.0, np.ones(2), 1.0),
            GradientEvent("x", 1.0, np.ones(2), 1.0),
        ),
        duration_s=3.0,
    )
    with pytest.raises(ValueError, match="overlapping gradient"):
        SequenceCompiler().compile(program)


def test_legacy_adapter_requires_uniform_time_and_converts_units():
    time = np.arange(4) * 1e-3
    program = SequenceProgram.from_legacy(
        np.ones(4, dtype=complex),
        np.ones((4, 3)),
        time,
        adc_times_s=(0.0, 0.003),
    )
    compiled = SequenceCompiler().compile(program)
    assert compiled.rf_hz[0] == pytest.approx(rf_gauss_to_hz(1.0))
    assert compiled.gradient_hz_per_m[0, 0] == pytest.approx(
        gradient_g_per_cm_to_hz_per_m(1.0)
    )
    with pytest.raises(ValueError, match="uniform"):
        SequenceProgram.from_legacy(
            np.ones(3), np.zeros((3, 3)), np.array([0.0, 1e-3, 3e-3])
        )
