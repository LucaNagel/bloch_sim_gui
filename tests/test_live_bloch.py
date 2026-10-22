import numpy as np
import pytest

from blochsimulator.live_bloch import LiveSpinSystem
from blochsimulator.simulator import BlochSimulator, TissueParameters

# The kernel integrates with the proton gyromagnetic ratio.
KERNEL_GAMMA_HZ_PER_T = 42.576e6


def _single_spin(position_m=(0.0, 0.0, 0.0), offset_hz=0.0, **kwargs):
    return LiveSpinSystem([position_m], [offset_hz], KERNEL_GAMMA_HZ_PER_T, **kwargs)


@pytest.mark.parametrize(
    "phase_deg, expected",
    [(0.0, [0, 1, 0]), (90.0, [1, 0, 0]), (180.0, [0, -1, 0]), (270.0, [-1, 0, 0])],
)
def test_ninety_degree_pulse_tips_like_the_kernel(phase_deg, expected):
    system = _single_spin()
    system.apply_pulse(90.0, phase_deg)
    np.testing.assert_allclose(system.magnetization[0, 0], expected, atol=1e-12)
    assert system.time_s == 0.0


def test_positive_frequency_turns_clockwise_seen_from_plus_z():
    # 1 cm along x in 1 mT/m precesses at gamma * 1e-5 T.
    system = _single_spin(position_m=(0.01, 0.0, 0.0))
    system.apply_pulse(90.0, 0.0)
    frequency = KERNEL_GAMMA_HZ_PER_T * 1e-5
    system.evolve(0.25 / frequency, gradient_t_per_m=(1e-3, 0.0, 0.0))
    np.testing.assert_allclose(system.magnetization[0, 0], [1, 0, 0], atol=1e-12)
    assert system.time_s == pytest.approx(0.25 / frequency)


def test_spin_echo_refocuses_off_resonance_and_gradient():
    positions = np.column_stack([np.linspace(-5e-3, 5e-3, 7), np.zeros(7), np.zeros(7)])
    system = LiveSpinSystem(positions, [-80.0, 0.0, 35.0], KERNEL_GAMMA_HZ_PER_T)
    gradient = (2e-3, 0.0, 0.0)
    system.apply_pulse(90.0, 0.0)
    system.evolve(3e-3, gradient)
    assert np.linalg.norm(system.magnetization.mean(axis=(0, 1))) < 0.5
    system.apply_pulse(180.0, 0.0)
    system.evolve(3e-3, gradient)
    np.testing.assert_allclose(
        system.magnetization.reshape(-1, 3),
        np.tile([0.0, -1.0, 0.0], (21, 1)),
        atol=1e-12,
    )


def test_relaxation_follows_t1_and_t2():
    system = _single_spin(t1_s=0.1, t2_s=0.02)
    system.apply_pulse(90.0, 0.0)
    system.evolve(0.02)
    mx, my, mz = system.magnetization[0, 0]
    assert np.hypot(mx, my) == pytest.approx(np.exp(-1.0))
    assert mz == pytest.approx(1.0 - np.exp(-0.2))

    system.reset()
    system.apply_pulse(90.0, 0.0)
    system.evolve(0.02, relax=False)
    np.testing.assert_allclose(system.magnetization[0, 0], [0, 1, 0], atol=1e-12)


def test_reset_returns_to_equilibrium():
    system = _single_spin(position_m=(0.0, 0.003, 0.0))
    system.apply_pulse(90.0, 90.0)
    system.evolve(1e-3, (0.0, 5e-3, 0.0))
    system.reset()
    np.testing.assert_allclose(system.magnetization[0, 0], [0, 0, 1])
    assert system.time_s == 0.0


def _hard_pulse(flip_deg, phase_deg, dt, duration=10e-6):
    n = int(round(duration / dt))
    amplitude_g = np.radians(flip_deg) / (2 * np.pi * 4257.6 * duration)
    b1 = np.full(n, amplitude_g * np.exp(1j * np.radians(phase_deg)))
    return b1, np.zeros((n, 3))


def _gradient(g_mt_per_m, duration, dt):
    n = int(round(duration / dt))
    return np.zeros(n, complex), np.tile(np.asarray(g_mt_per_m) * 0.1, (n, 1))


def test_sequence_matches_the_simulation_kernel():
    dt = 1e-6
    steps = [
        ("pulse", 90.0, 0.0),
        ("gradient", [2.0, 0.0, 1.0], 1e-3),
        ("pulse", 180.0, 90.0),
        ("gradient", [0.0, -3.0, 2.0], 0.7e-3),
        ("pulse", 90.0, 180.0),
        ("gradient", [1.0, 1.0, 1.0], 0.3e-3),
    ]
    positions = np.array([[4e-3, -2e-3, 3e-3], [-5e-3, 1e-3, 0.0], [0.0, 0.0, 0.0]])
    offsets = np.array([-40.0, 0.0, 25.0])
    live = LiveSpinSystem(positions, offsets, KERNEL_GAMMA_HZ_PER_T, 0.5, 0.05)
    segments = []
    for kind, a, b in steps:
        if kind == "pulse":
            live.apply_pulse(a, b)
            segments.append(_hard_pulse(a, b, dt))
        else:
            live.evolve(b, np.asarray(a) * 1e-3)
            segments.append(_gradient(a, b, dt))
    b1 = np.concatenate([s[0] for s in segments])
    gradients = np.concatenate([s[1] for s in segments])
    time = np.arange(len(b1)) * dt
    try:
        result = BlochSimulator().simulate(
            (b1, gradients, time),
            TissueParameters(name="test", t1=0.5, t2=0.05),
            positions=positions,
            frequencies=offsets,
            dt=dt,
            mode=0,
        )
    except NotImplementedError:
        pytest.skip("simulation kernel not built")
    kernel = np.stack([np.asarray(result[k]) for k in ("mx", "my", "mz")], axis=-1)
    # The kernel's pulses last 10 µs, the live mode's are instantaneous.
    np.testing.assert_allclose(
        kernel.reshape(live.magnetization.shape), live.magnetization, atol=5e-3
    )
