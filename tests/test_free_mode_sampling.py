"""Independent numerical, sampling and responsiveness contracts for Free Mode."""

import threading
import time

import numpy as np
import pytest
from scipy.linalg import expm

from blochsimulator.adaptive import SimulationCancelled
from blochsimulator.simulator import BlochSimulator, TissueParameters, design_rf_pulse
from blochsimulator.units import LEGACY_GAMMA_RAD_PER_S_PER_GAUSS as GAMMA


@pytest.mark.parametrize("pulse_type", ["adiabatic_half", "adiabatic_full"])
@pytest.mark.parametrize("tbw", [0.0, 4.0])
def test_adiabatic_phase_is_independent_of_design_raster(pulse_type, tbw):
    coarse, coarse_time = design_rf_pulse(
        pulse_type, duration=0.005, npoints=100, time_bw_product=tbw
    )
    fine, fine_time = design_rf_pulse(
        pulse_type, duration=0.005, npoints=500, time_bw_product=tbw
    )
    np.testing.assert_allclose(coarse_time, fine_time[2::5], atol=1e-15)
    np.testing.assert_allclose(coarse, fine[2::5], atol=1e-15)


@pytest.mark.parametrize("duration,nfreq", [(0.005, 11), (0.010, 31)])
def test_strong_ahp_matches_ten_ns_midpoint_reference(duration, nfreq):
    dt = 1e-8
    pulse, _ = design_rf_pulse(
        "adiabatic_half", duration=duration, npoints=round(duration / dt)
    )
    pulse *= 5 / np.abs(pulse).max()
    b1 = np.r_[pulse, np.zeros(round((0.015 - duration) / dt))]
    time_axis = (np.arange(len(b1)) + 0.5) * dt
    gradients = np.zeros((len(b1), 3))
    frequencies = np.linspace(-50, 50, nfreq)
    tissue = TissueParameters("test", 1, 0.1)
    simulator = BlochSimulator(use_parallel=False)
    adaptive = simulator.simulate(
        (b1, gradients, time_axis),
        tissue,
        frequencies=frequencies,
        mode=2,
        adaptive_rf=True,
        output_step_s=1e-6,
    )
    fixed = simulator.simulate(
        (b1, gradients, time_axis), tissue, frequencies=frequencies, mode=0
    )
    for key in ("mx", "my", "mz"):
        np.testing.assert_allclose(adaptive[key][-1], fixed[key], atol=5e-7, rtol=0)
    assert len(adaptive["time"]) == 15001


def test_adaptive_constant_rf_matches_affine_matrix_exponential():
    dt, duration = 1e-6, 200e-6
    t = (np.arange(500) + 0.5) * dt
    b1 = np.zeros(len(t), complex)
    b1[:200] = 2 + 0.5j
    tissue = TissueParameters("test", 0.3, 0.04)
    frequencies = np.array([-800.0, 0.0, 550.0])
    initial = np.array([0.1, -0.3, 0.8])
    result = BlochSimulator().simulate(
        (b1, np.zeros((len(t), 3)), t),
        tissue,
        frequencies=frequencies,
        initial_magnetization=initial,
        adaptive_rf=True,
        output_step_s=1e-6,
        mode=2,
    )
    for fi, offset in enumerate(frequencies):
        wx, wy, wz = -2 * GAMMA, 0.5 * GAMMA, -2 * np.pi * offset
        generator = np.array(
            [
                [-1 / tissue.t2, -wz, wy, 0],
                [wz, -1 / tissue.t2, -wx, 0],
                [-wy, wx, -1 / tissue.t1, 1 / tissue.t1],
                [0, 0, 0, 0],
            ]
        )
        expected = (expm(generator * duration) @ np.r_[initial, 1])[:3]
        index = np.argmin(abs(result["time"] - duration))
        observed = np.array([result[k][index, 0, fi] for k in ("mx", "my", "mz")])
        np.testing.assert_allclose(observed, expected, atol=2e-8)
        decay = result["time"][-1] - duration
        expected_xy = complex(*expected[:2]) * np.exp(
            -decay / tissue.t2 - 2j * np.pi * offset * decay
        )
        np.testing.assert_allclose(result["signal"][-1, 0, fi], expected_xy, atol=2e-8)
        np.testing.assert_allclose(
            result["mz"][-1, 0, fi],
            1 + (expected[2] - 1) * np.exp(-decay / tissue.t1),
            atol=2e-8,
        )
    np.testing.assert_allclose(np.diff(result["time"]), 1e-6, atol=1e-15)


@pytest.mark.parametrize("npoints", [8193, 18001])
def test_saved_sampling_preserves_fixed_kernel_values_and_final_state(npoints):
    dt = 1e-6
    t = np.arange(npoints) * dt
    b1 = 0.1 * np.exp(2j * np.pi * 150 * t)
    positions = np.array([[0, 0, -0.001], [0, 0, 0.001]])
    frequencies = np.array([-50, 0, 70])
    sequence = (b1, np.zeros((len(t), 3)), t)
    simulator = BlochSimulator(use_parallel=False)
    tissue = TissueParameters("test", 1, 0.1)
    initial = np.arange(18).reshape(3, 6) / 20
    full = simulator.simulate(
        sequence, tissue, positions, frequencies, initial_magnetization=initial, mode=2
    )
    sampled = simulator.simulate(
        sequence,
        tissue,
        positions,
        frequencies,
        initial_magnetization=initial,
        mode=2,
        output_step_s=11 * dt,
    )
    for key in ("mx", "my", "mz", "signal", "time"):
        np.testing.assert_allclose(sampled[key], full[key][::11], atol=1e-13)
    np.testing.assert_allclose(
        sampled["integration"]["final_magnetization"],
        np.stack([full[k][-1] for k in ("mx", "my", "mz")]),
        atol=1e-13,
    )


def test_fixed_and_adaptive_simulations_observe_cancellation():
    t = np.arange(1000) * 1e-6
    sequence = (np.ones(1000, complex), np.zeros((1000, 3)), t)
    for adaptive in (False, True):
        with pytest.raises(SimulationCancelled):
            BlochSimulator().simulate(
                sequence,
                TissueParameters("test", 1, 0.1),
                mode=2,
                adaptive_rf=adaptive,
                output_step_s=1e-6,
                cancel_check=lambda: True,
            )


def test_serial_native_kernel_allows_other_python_threads_to_run():
    from blochsimulator.blochsimulator_cy import simulate_bloch

    ticks = []
    stop = threading.Event()
    ready = threading.Event()

    def heartbeat():
        ready.set()
        while not stop.wait(0.002):
            ticks.append(time.monotonic())

    n = 500000
    rf = np.full(n, 0.1 + 0j)
    gradients = np.zeros((n, 3))
    steps = np.full(n, 1e-7)
    frequencies = np.linspace(-50, 50, 31)
    positions = np.zeros((1, 3))
    thread = threading.Thread(target=heartbeat)
    thread.start()
    ready.wait()
    try:
        begin = time.monotonic()
        simulate_bloch(
            rf,
            gradients,
            steps,
            1.0,
            0.1,
            frequencies,
            positions,
            mode=0,
        )
        end = time.monotonic()
        # Exclude wrapper preparation/cleanup; the middle of this long call
        # must permit Python progress while the native kernel is computing.
        margin = (end - begin) / 4
        assert (
            len([tick for tick in ticks if begin + margin < tick < end - margin]) >= 3
        )
    finally:
        stop.set()
        thread.join()
