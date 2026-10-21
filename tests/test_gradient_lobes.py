"""Contracts of extra gradient lobes and slice-axis remapping."""

import numpy as np
import pytest

from blochsimulator.gradient_lobes import (
    GradientLobe,
    add_gradient_lobes,
    gradient_moments_mt_ms_per_m,
    phase_twist_cycles_per_mm,
    remap_slice_axis,
)
from blochsimulator.position_sampling import build_line_positions
from blochsimulator.simulator import BlochSimulator, TissueParameters


def _empty_sequence(n, dt):
    return np.zeros(n, complex), np.zeros((n, 3)), np.arange(n) * dt


@pytest.mark.parametrize("dt", [1e-5, 3.7e-5])
def test_lobe_moment_is_exact_on_any_raster(dt):
    lobe = GradientLobe("y", start_ms=0.123, duration_ms=1.5, ramp_ms=0.2)
    b1, gradients, time = add_gradient_lobes(*_empty_sequence(400, dt), [lobe], dt)

    moments = gradient_moments_mt_ms_per_m(gradients, time)
    np.testing.assert_allclose(moments, [0.0, 10.0 * 1.3, 0.0], atol=1e-12)
    assert lobe.moment_mt_ms_per_m == pytest.approx(13.0)
    # Plateau samples carry the full amplitude (10 mT/m = 1 G/cm).
    plateau = (time >= 0.5e-3) & (time < 1.3e-3)
    np.testing.assert_allclose(gradients[plateau, 1], 1.0)


def test_long_ramps_give_a_triangle():
    lobe = GradientLobe("x", duration_ms=1.0, amplitude_mt_per_m=-20.0, ramp_ms=0.8)

    assert lobe.effective_ramp_ms == 0.5
    assert lobe.moment_mt_ms_per_m == pytest.approx(-10.0)


def test_lobes_past_the_end_extend_the_sequence_without_rf():
    dt = 1e-5
    b1 = np.full(100, 0.1 + 0j)
    lobe = GradientLobe("x", start_ms=0.8, duration_ms=0.6, ramp_ms=0.0)
    b1, gradients, time = add_gradient_lobes(
        b1, np.zeros((100, 3)), np.arange(100) * dt, [lobe], dt
    )

    assert len(time) == len(b1) == len(gradients) == 140
    np.testing.assert_allclose(np.diff(time), dt)
    assert np.all(b1[100:] == 0.0)
    np.testing.assert_allclose(gradients[80:140, 0], 1.0)
    assert np.all(gradients[:80, 0] == 0.0)


def test_lobes_add_onto_existing_gradients():
    dt = 1e-5
    b1, gradients, time = _empty_sequence(100, dt)
    gradients[:, 2] = 0.5
    lobes = [
        GradientLobe("z", start_ms=0.0, duration_ms=0.5, ramp_ms=0.0),
        GradientLobe("z", start_ms=0.0, duration_ms=0.5, ramp_ms=0.0),
    ]
    _, gradients, _ = add_gradient_lobes(b1, gradients, time, lobes, dt)

    np.testing.assert_allclose(gradients[:50, 2], 2.5)
    np.testing.assert_allclose(gradients[50:, 2], 0.5)


@pytest.mark.parametrize("axis, column", [("x", 0), ("y", 1), ("z", 2)])
def test_slice_axis_remap_moves_the_slice_channel(axis, column):
    gradients = np.zeros((4, 3))
    gradients[:, 0] = 1.0
    gradients[:, 2] = 3.0
    remapped = remap_slice_axis(gradients, axis)

    np.testing.assert_allclose(remapped[:, column], 3.0)
    assert remapped.sum() == pytest.approx(gradients.sum())
    assert gradients[0, 2] == 3.0


def test_lobe_state_round_trip_and_validation():
    lobe = GradientLobe("Y", 1.0, 2.0, -5.0, 0.3)

    assert lobe.axis == "y"
    assert GradientLobe.from_state(lobe.to_state()) == lobe
    with pytest.raises(ValueError):
        GradientLobe("x", duration_ms=0.0)
    with pytest.raises(ValueError):
        GradientLobe("w")


def test_x_lobe_winds_the_predicted_phase_along_x():
    dt = 1e-5
    n_rf = 20
    b1 = np.zeros(400, complex)
    # Hard 90° pulse: gamma * B1 * T = pi / 2 with gamma = 26753 rad/s/G.
    b1[:n_rf] = (np.pi / 2) / (26753.0 * n_rf * dt)
    lobe = GradientLobe("x", start_ms=0.5, duration_ms=2.0, ramp_ms=0.25)
    b1, gradients, time = add_gradient_lobes(
        b1, np.zeros((400, 3)), np.arange(400) * dt, [lobe], dt
    )
    positions = build_line_positions(9, 0.004, "x")
    tissue = TissueParameters("slow", 10.0, 10.0)
    result = BlochSimulator(use_parallel=False).simulate(
        (b1, gradients, time), tissue, positions, np.array([0.0]), mode=0
    )

    mxy = (result["mx"] + 1j * result["my"]).reshape(-1)
    np.testing.assert_allclose(np.abs(mxy), 1.0, atol=1e-3)
    twist = phase_twist_cycles_per_mm([lobe.moment_mt_ms_per_m])[0]
    expected = -2 * np.pi * twist * positions[:, 0] * 1000.0
    relative = np.angle(mxy * np.conj(mxy[4]))
    np.testing.assert_allclose(relative, np.angle(np.exp(1j * expected)), atol=1e-6)
