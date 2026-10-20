"""Probe T1 recovery for thermal and normalized hyperpolarized Mz, and spoiling."""

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.sequence import GradientEvent, RFEvent, SequenceProgram

KERNELS = ("optimized", "strict_reference")
T1_S = 25.0
T2_S = 1.0
ENHANCEMENT = 1.0e4


def _rf_train(n_pulses=20, tr_s=0.01, pulse_s=1e-3, flip_deg=10.0):
    """Hard pulses whose sinc spectrum has a null at every 1/pulse_s Hz."""
    amplitude_hz = flip_deg / 360.0 / pulse_s
    events = tuple(
        RFEvent(index * tr_s, np.array([amplitude_hz + 0j]), pulse_s)
        for index in range(n_pulses)
    )
    return SequenceProgram(events, duration_s=n_pulses * tr_s)


def _probe(program, checkpoints, frequencies, **kwargs):
    kwargs.setdefault("t1_s", T1_S)
    kwargs.setdefault("t2_s", T2_S)
    return BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        positions_m=np.zeros((1, 3)),
        frequency_offsets_hz=np.asarray(frequencies, dtype=float),
        checkpoints_s=checkpoints,
        **kwargs,
    )


def _hp_relaxation(t_s):
    """Normalized HP Mz relaxing from above toward the thermal level 1/ε."""
    thermal = 1.0 / ENHANCEMENT
    return thermal + (1.0 - thermal) * np.exp(-np.asarray(t_s) / T1_S)


@pytest.mark.parametrize("kernel", KERNELS)
def test_hyperpolarized_free_decay_relaxes_toward_thermal_level(kernel):
    checkpoints = np.array([0.0, 4.0, 16.0, 200.0])

    result = _probe(
        SequenceProgram((), duration_s=200.0),
        checkpoints,
        [0.0],
        initial_magnetization=(0.0, 0.0, 1.0),
        equilibrium_mz=1.0 / ENHANCEMENT,
        sequence_kernel=kernel,
    )

    assert result.mz[:, 0, 0] == pytest.approx(_hp_relaxation(checkpoints), rel=1e-9)
    assert np.all(result.mz[:, 0, 0] > 1.0 / ENHANCEMENT)
    assert result.metadata["longitudinal_equilibrium"] == "custom"
    assert result.metadata["equilibrium_mz"] == 1.0 / ENHANCEMENT


@pytest.mark.parametrize("kernel", KERNELS)
def test_normalized_hyperpolarization_matches_thermal_units(kernel):
    # The same spins in thermal units start at Mz = ε and relax toward 1.
    program = _rf_train()
    checkpoints = np.linspace(0.0, program.duration_s, 5)
    frequencies = [-3000.0, -40.0, 0.0]

    normalized = _probe(
        program,
        checkpoints,
        frequencies,
        initial_magnetization=(0.0, 0.0, 1.0),
        equilibrium_mz=1.0 / ENHANCEMENT,
        sequence_kernel=kernel,
    )
    thermal_units = _probe(
        program,
        checkpoints,
        frequencies,
        initial_magnetization=(0.0, 0.0, ENHANCEMENT),
        sequence_kernel=kernel,
    )

    np.testing.assert_allclose(
        normalized.magnetization * ENHANCEMENT,
        thermal_units.magnetization,
        rtol=1e-9,
        atol=1e-9 * ENHANCEMENT,
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_hyperpolarized_far_off_resonant_spin_decays_through_rf(kernel):
    # The 1 ms hard pulses have a spectral null at -3000 Hz, so the spin only
    # relaxes; T1 recovery is applied after every RF sub-step as well.
    program = _rf_train()
    checkpoints = np.array([0.0, program.duration_s])

    hp = _probe(
        program,
        checkpoints,
        [-3000.0],
        initial_magnetization=(0.0, 0.0, 1.0),
        equilibrium_mz=1.0 / ENHANCEMENT,
        sequence_kernel=kernel,
    )
    thermal = _probe(
        program,
        checkpoints,
        [-3000.0],
        initial_magnetization=(0.0, 0.0, 1.0),
        sequence_kernel=kernel,
    )

    assert hp.mz[-1, 0, 0] == pytest.approx(
        _hp_relaxation(program.duration_s), rel=1e-6
    )
    assert thermal.mz[-1, 0, 0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("kernel", KERNELS)
def test_saturated_hyperpolarized_mz_recovers_only_to_thermal_level(kernel):
    checkpoints = np.array([0.0, 5.0, 20.0])
    program = SequenceProgram((), duration_s=20.0)
    kwargs = dict(initial_magnetization=(0.0, 0.0, 0.0), sequence_kernel=kernel)

    hp = _probe(
        program,
        checkpoints,
        [0.0, -3000.0],
        equilibrium_mz=1.0 / ENHANCEMENT,
        **kwargs,
    )
    thermal = _probe(program, checkpoints, [0.0, -3000.0], **kwargs)

    recovery = 1.0 - np.exp(-checkpoints / T1_S)
    for spin in range(2):
        assert thermal.mz[:, 0, spin] == pytest.approx(recovery, rel=1e-9)
        assert hp.mz[:, 0, spin] == pytest.approx(recovery / ENHANCEMENT, rel=1e-9)


@pytest.mark.parametrize("kernel", KERNELS)
def test_explicit_thermal_equilibrium_matches_default_exactly(kernel):
    program = SequenceProgram(
        (
            *_rf_train(n_pulses=5).events,
            GradientEvent("x", 0.0, np.full(50, 200.0), 1e-3),
        ),
        duration_s=0.05,
    )
    checkpoints = np.linspace(0.0, program.duration_s, 6)
    frequencies = [-3000.0, -50.0, 0.0, 120.0]

    default = _probe(program, checkpoints, frequencies, sequence_kernel=kernel)
    explicit = _probe(
        program, checkpoints, frequencies, equilibrium_mz=1.0, sequence_kernel=kernel
    )

    np.testing.assert_array_equal(default.magnetization, explicit.magnetization)
    assert default.metadata["longitudinal_equilibrium"] == "thermal"
    assert default.metadata["equilibrium_mz"] == 1.0


def test_custom_equilibrium_relaxes_toward_value():
    result = _probe(
        SequenceProgram((), duration_s=10.0),
        np.array([10.0]),
        [0.0],
        initial_magnetization=(0.0, 0.0, 1.0),
        equilibrium_mz=0.25,
    )

    decay = np.exp(-10.0 / T1_S)
    assert result.mz[0, 0, 0] == pytest.approx(decay + 0.25 * (1.0 - decay))
    assert result.metadata["longitudinal_equilibrium"] == "custom"


def test_non_finite_equilibrium_is_rejected():
    with pytest.raises(ValueError, match="equilibrium_mz"):
        _probe(
            SequenceProgram((), duration_s=0.0),
            (0.0,),
            [0.0],
            equilibrium_mz=np.nan,
        )


@pytest.mark.parametrize("module", ["blochsimulator_cy", "blochsimulator_strict_cy"])
def test_native_chunk_rejects_mismatched_equilibrium_length(module):
    simulate_sequence_chunk = pytest.importorskip(
        f"blochsimulator.{module}"
    ).simulate_sequence_chunk

    with pytest.raises(ValueError, match="mz_equilibrium length"):
        simulate_sequence_chunk(
            np.zeros(1, dtype=np.complex128),
            np.zeros((1, 3)),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.zeros(1),
            np.zeros((1, 3)),
            np.ones(1),
            np.ones(1, dtype=np.complex128),
            np.ones((1, 1), dtype=np.complex128),
            np.array([[0.0, 0.0, 1.0]]),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.complex128),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            mz_equilibrium=np.zeros(2),
        )


def _ideal_spoiler_program():
    # A 90 degree pulse followed by a declared ideal crusher at 2 ms.
    pulse_s = 1e-4
    rf = RFEvent(0.0, np.array([0.25 / pulse_s + 0j]), pulse_s)
    return SequenceProgram(
        (rf,),
        duration_s=4e-3,
        metadata={"definitions": {"IdealSpoilerEndTimes": [2e-3]}},
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_probe_ideal_spoiling_toggle(kernel):
    program = _ideal_spoiler_program()
    checkpoints = np.array([1e-3, 3e-3])
    kwargs = dict(
        initial_magnetization=(0.0, 0.0, 1.0),
        t1_s=1e9,
        t2_s=1e9,
        sequence_kernel=kernel,
    )

    spoiled = _probe(program, checkpoints, [0.0], **kwargs)
    unspoiled = _probe(program, checkpoints, [0.0], ideal_spoiling=False, **kwargs)

    spoiled_mxy = np.abs(spoiled.mx + 1j * spoiled.my)[:, 0, 0]
    unspoiled_mxy = np.abs(unspoiled.mx + 1j * unspoiled.my)[:, 0, 0]
    assert spoiled_mxy[0] == pytest.approx(1.0, abs=1e-6)
    assert spoiled_mxy[1] == pytest.approx(0.0, abs=1e-12)
    assert unspoiled_mxy == pytest.approx([1.0, 1.0], abs=1e-6)

    assert spoiled.metadata["spoiler_mode"] == "ideal"
    assert spoiled.metadata["ideal_spoiling_applied"] is True
    assert spoiled.metadata["ideal_spoiler_end_times_s"] == pytest.approx([2e-3])
    assert unspoiled.metadata["spoiler_mode"] == "gradient"
    assert unspoiled.metadata["ideal_spoiling_applied"] is False
    assert unspoiled.metadata["ideal_spoiler_end_times_s"] == []
    assert unspoiled.metadata["declared_ideal_spoiler_end_times_s"] == pytest.approx(
        [2e-3]
    )
