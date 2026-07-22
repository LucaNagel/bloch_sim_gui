import numpy as np
import pytest

from blochsimulator.sequence import (
    ConvergenceCriteria,
    RFEvent,
    SequenceProgram,
    SpinProbeEnsemble,
    default_probe_checkpoints,
    run_timestep_convergence,
)


def test_probe_ensemble_expands_cartesian_parameter_axes():
    probes = SpinProbeEnsemble.from_axes(
        [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
        frequency_offsets_hz=[-100.0, 100.0],
        b1_scales=[0.8, 1.2 + 0.1j],
        relaxation_times_s=[(0.8, 0.05), (1.5, 0.2)],
    )

    assert probes.n_spins == 16
    assert probes.describe(0) == {
        "index": 0,
        "position_m": [-0.01, 0.0, 0.0],
        "frequency_offset_hz": -100.0,
        "b1_scale": 0.8 + 0j,
        "t1_s": 0.8,
        "t2_s": 0.05,
        "initial_magnetization": [0.0, 0.0, 1.0],
    }
    assert probes.describe(1)["t1_s"] == pytest.approx(1.5)
    assert probes.describe(2)["b1_scale"] == pytest.approx(1.2 + 0.1j)
    assert probes.describe(8)["position_m"] == [0.01, 0.0, 0.0]
    assert not probes.positions_m.flags.writeable
    assert not probes.b1_scales.flags.writeable


def test_probe_ensemble_keeps_explicit_relaxation_pairs():
    probes = SpinProbeEnsemble.from_axes(
        [0.0, 0.0, 0.0],
        relaxation_times_s=[(0.5, 0.03), (2.0, 0.3)],
    )

    assert np.allclose(
        np.column_stack((probes.t1_s, probes.t2_s)),
        [[0.5, 0.03], [2.0, 0.3]],
    )
    with pytest.raises(ValueError, match="positive"):
        SpinProbeEnsemble.from_axes([0.0, 0.0, 0.0], relaxation_times_s=[(1.0, 0.0)])


def test_default_checkpoints_sample_rf_ends_without_entering_rf_events():
    raster = 10e-6
    events = tuple(
        RFEvent(2 * index * raster, np.array([100.0]), raster) for index in range(20)
    )
    program = SequenceProgram(events, duration_s=41 * raster)

    checkpoints = default_probe_checkpoints(program, max_rf_checkpoints=5)
    rf_ends = {event.end_s for event in events}

    assert checkpoints.size <= 6
    assert checkpoints[-1] == pytest.approx(program.duration_s)
    assert all(time_s in rf_ends for time_s in checkpoints[:-1])
    assert events[0].end_s in checkpoints
    assert events[-1].end_s in checkpoints


def test_convergence_sweep_identifies_first_noncommuting_coarsening_failure():
    raster = 0.5e-3
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0, 250.0j]), raster),),
        duration_s=2 * raster,
    )
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=[0.0, 100.0],
        b1_scales=[0.8, 1.0],
        relaxation_times_s=[(1e9, 1e9)],
    )

    result = run_timestep_convergence(
        program,
        probes,
        timesteps_s=(None, raster, 2 * raster),
        criteria=ConvergenceCriteria(
            max_vector_error=1e-3,
            rms_vector_error=1e-3,
        ),
    )

    native, raster_point, coarse = result.points
    assert native.passed
    assert native.max_vector_error < 1e-11
    assert raster_point.passed
    assert raster_point.interval_count == 2
    assert coarse.interval_count == 1
    assert not coarse.passed
    assert coarse.max_vector_error > 0.1
    assert result.coarsest_passing_timestep_s == pytest.approx(raster)
    assert result.probes.describe(coarse.worst_probe_index)[
        "frequency_offset_hz"
    ] == pytest.approx(100.0)

    records = result.to_records()
    assert records[0]["timestep"] == "native"
    assert records[-1]["simulation_timestep_us"] == pytest.approx(1000.0)
    assert records[-1]["passed"] is False


def test_explicit_checkpoints_always_include_sequence_end():
    duration = 1e-3
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0]), duration),), duration_s=duration
    )
    probes = SpinProbeEnsemble.from_axes([0.0, 0.0, 0.0])

    result = run_timestep_convergence(
        program,
        probes,
        timesteps_s=(None,),
        checkpoints_s=(),
    )

    assert result.checkpoint_times_s == pytest.approx([duration])
    assert result.native_point is result.points[0]
    assert result.points[0].worst_time_s == pytest.approx(duration)


def test_convergence_inputs_reject_invalid_tolerances_and_timesteps():
    program = SequenceProgram((), duration_s=1e-3)
    probes = SpinProbeEnsemble.from_axes([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="max_vector_error"):
        ConvergenceCriteria(max_vector_error=0.0)
    with pytest.raises(ValueError, match="duplicates"):
        run_timestep_convergence(program, probes, timesteps_s=(1e-6, 1e-6))
    with pytest.raises(ValueError, match="finite and positive"):
        run_timestep_convergence(program, probes, timesteps_s=(0.0,))
