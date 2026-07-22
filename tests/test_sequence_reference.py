import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    simulate_reference_sequence,
)


def test_reference_free_precession_and_relaxation_match_analytic_solution():
    duration = 40e-3
    checkpoints = (0.0, 10e-3, duration)
    frequencies = np.array([13.0, -27.0])
    t1 = np.array([0.7, 1.2])
    t2 = np.array([0.08, 0.05])
    initial = np.array([0.6, -0.2, 0.3])
    result = simulate_reference_sequence(
        SequenceProgram((), duration_s=duration),
        frequency_offsets_hz=frequencies,
        t1_s=t1,
        t2_s=t2,
        initial_magnetization=initial,
        checkpoints_s=checkpoints,
    )

    for checkpoint_index, time_s in enumerate(checkpoints):
        expected_transverse = (initial[0] + 1j * initial[1]) * np.exp(
            -time_s / t2 - 2j * np.pi * frequencies * time_s
        )
        expected_mz = 1.0 + (initial[2] - 1.0) * np.exp(-time_s / t1)
        assert result.checkpoint_magnetization[checkpoint_index, :, 0] == pytest.approx(
            expected_transverse.real, abs=2e-14
        )
        assert result.checkpoint_magnetization[checkpoint_index, :, 1] == pytest.approx(
            expected_transverse.imag, abs=2e-14
        )
        assert result.checkpoint_magnetization[checkpoint_index, :, 2] == pytest.approx(
            expected_mz, abs=2e-14
        )

    assert result.final_magnetization == pytest.approx(
        result.checkpoint_magnetization[-1], abs=0.0
    )


def test_reference_hard_90_degree_pulse_has_expected_rotation_sign():
    duration = 1e-3
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0]), duration),),
        duration_s=duration,
    )

    result = simulate_reference_sequence(program, t1_s=1e12, t2_s=1e12)

    assert result.final_magnetization[0] == pytest.approx([0.0, 1.0, 0.0], abs=2e-12)


def test_reference_records_adc_state_at_sample_centres():
    dwell = 1e-3
    frequency = 37.0
    program = SequenceProgram(
        (ADCEvent(0.0, 3, dwell),),
        duration_s=3 * dwell,
    )

    result = simulate_reference_sequence(
        program,
        frequency_offsets_hz=frequency,
        t1_s=1e12,
        t2_s=1e12,
        initial_magnetization=(1.0, 0.0, 0.0),
    )

    expected = np.exp(-2j * np.pi * frequency * result.adc_times_s)
    actual = result.adc_magnetization[:, 0, 0] + 1j * result.adc_magnetization[:, 0, 1]
    assert result.adc_times_s == pytest.approx([0.0, dwell, 2 * dwell])
    assert actual == pytest.approx(expected, abs=2e-14)


def test_native_sequence_kernel_matches_reference_without_relevant_relaxation():
    raster = 20e-6
    checkpoints = tuple(np.arange(6, dtype=float) * raster)
    program = SequenceProgram(
        (
            RFEvent(
                0.0,
                np.array([80 + 10j, 120 - 20j, -40 + 30j, 90 - 5j]),
                raster,
                frequency_offset_hz=750.0,
                phase_offset_rad=0.23,
            ),
            GradientEvent("z", 0.0, np.array([1000.0, -500.0, 250.0, 800.0]), raster),
        ),
        duration_s=5 * raster,
    )
    position = np.array([[0.0, 0.0, 0.012]])
    frequency = np.array([35.0])

    reference = simulate_reference_sequence(
        program,
        positions_m=position,
        frequency_offsets_hz=frequency,
        t1_s=1e9,
        t2_s=1e9,
        checkpoints_s=checkpoints,
    )
    native = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        position,
        frequency,
        checkpoints_s=checkpoints,
        t1_s=1e9,
        t2_s=1e9,
    )

    assert reference.interval_count == 5
    assert native.magnetization[:, 0, 0] == pytest.approx(
        reference.checkpoint_magnetization[:, 0], abs=2e-12
    )


def test_reference_detects_noncommuting_rf_coarsening_error():
    raster = 0.5e-3
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0, 250.0j]), raster),),
        duration_s=2 * raster,
    )
    position = np.zeros((1, 3))
    frequency = np.zeros(1)
    checkpoints = (program.duration_s,)

    reference = simulate_reference_sequence(
        program,
        t1_s=1e12,
        t2_s=1e12,
        checkpoints_s=checkpoints,
    )
    simulator = BlochSimulator(use_parallel=False)
    native = simulator.simulate_sequence_probes(
        program,
        position,
        frequency,
        checkpoints_s=checkpoints,
        t1_s=1e12,
        t2_s=1e12,
    )
    coarse = simulator.simulate_sequence_probes(
        program,
        position,
        frequency,
        checkpoints_s=checkpoints,
        t1_s=1e12,
        t2_s=1e12,
        simulation_timestep_s=program.duration_s,
    )

    expected = reference.final_magnetization[0]
    assert native.magnetization[-1, 0, 0] == pytest.approx(expected, abs=2e-12)
    assert np.linalg.norm(coarse.magnetization[-1, 0, 0] - expected) > 0.1


def test_rf_free_gradient_and_relaxation_match_native_sparse_kernel():
    raster = 0.5e-3
    duration = 4 * raster
    checkpoints = (0.0, 0.75e-3, duration)
    program = SequenceProgram(
        (GradientEvent("x", 0.0, np.array([200.0, -50.0, 400.0, 10.0]), raster),),
        duration_s=duration,
    )
    positions = np.array([[-0.01, 0.0, 0.0], [0.013, 0.0, 0.0]])
    initial = np.array([0.7, -0.1, 0.2])

    reference = simulate_reference_sequence(
        program,
        positions_m=positions,
        frequency_offsets_hz=17.0,
        t1_s=0.8,
        t2_s=0.06,
        initial_magnetization=initial,
        checkpoints_s=checkpoints,
    )
    native = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        positions,
        np.array([17.0]),
        checkpoints_s=checkpoints,
        t1_s=0.8,
        t2_s=0.06,
        initial_magnetization=initial,
    )

    assert native.magnetization[:, :, 0] == pytest.approx(
        reference.checkpoint_magnetization, abs=3e-14
    )


def test_reference_rejects_incompatible_spin_dimensions():
    program = SequenceProgram((), duration_s=1e-3)

    with pytest.raises(ValueError, match="frequency_offsets_hz has 2 entries"):
        simulate_reference_sequence(
            program,
            positions_m=np.zeros((3, 3)),
            frequency_offsets_hz=np.zeros(2),
        )
