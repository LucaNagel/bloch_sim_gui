from pathlib import Path

import numpy as np
import pytest

from blochsimulator.simulator import BlochSimulator, CustomPulse, TissueParameters
from blochsimulator.units import LEGACY_GAMMA_RAD_PER_S_PER_GAUSS

SPSP_PULSE = (
    Path(__file__).parents[1]
    / "rfpulses"
    / "bruker"
    / "13C_Ultimate_SPSP_Pulse_QuEMRT.exc"
)


def _write_hard_pulse(tmp_path, flip_deg=90.0):
    """1 ms hard pulse with 100 samples whose edges are not zero."""
    amplitude = np.deg2rad(flip_deg) / (LEGACY_GAMMA_RAD_PER_S_PER_GAUSS * 1e-3)
    path = tmp_path / "hard_1ms.dat"
    np.savetxt(path, np.column_stack([np.full(100, amplitude), np.zeros(100)]))
    return path


def _simulated_flip_deg(pulse, dt):
    result = BlochSimulator(use_parallel=False).simulate(
        pulse, TissueParameters("test", t1=1e3, t2=1e3), dt=dt
    )
    mxy = np.hypot(result["mx"][0, 0], result["my"][0, 0])
    return np.rad2deg(np.arctan2(mxy, result["mz"][0, 0]))


# Native raster of the library pulse is 24.72 ms / 2473 = 9.996 us.
@pytest.mark.parametrize("dt", [1e-6, 4e-6, 1e-5, 2e-5])
def test_library_pulse_keeps_duration_and_rf_area_on_any_raster(dt):
    pulse = CustomPulse(SPSP_PULSE)
    source_dt = float(np.median(np.diff(pulse.time)))
    source_area = np.sum(pulse.b1) * source_dt
    duration = len(pulse.b1) * source_dt

    b1, gradients, time = pulse.compile(dt)

    assert len(b1) == round(duration / dt)
    np.testing.assert_allclose(np.diff(time), dt, rtol=1e-9)
    assert gradients.shape == (len(b1), 3)
    np.testing.assert_allclose(np.sum(b1) * dt, source_area, rtol=1e-12)


@pytest.mark.parametrize("dt", [1e-6, 3e-6, 1e-5, 3e-5])
def test_hard_pulse_flip_angle_survives_resampling(tmp_path, dt):
    pulse = CustomPulse(_write_hard_pulse(tmp_path))

    # Resampling over the sample span used to drop one raster step (~1/N).
    assert _simulated_flip_deg(pulse, dt) == pytest.approx(90.0, abs=1e-3)


def test_custom_gradients_are_resampled_with_b1(tmp_path):
    n = 100
    ramp = np.linspace(0.0, 1.0, 10)
    slice_select = 0.4 * np.r_[ramp, np.ones(n - 20), ramp[::-1]]
    bipolar = 0.2 * np.r_[np.ones(n // 2), -np.ones(n // 2)]
    native_gradients = np.column_stack([bipolar, np.zeros(n), slice_select])
    pulse = CustomPulse(_write_hard_pulse(tmp_path), gradients=native_gradients)
    source_dt = 1e-5
    dt = 3e-6

    b1, gradients, time = pulse.compile(dt)

    assert gradients.shape == (len(b1), 3) == (333, 3)
    np.testing.assert_allclose(
        np.sum(gradients[:, 2]) * dt, np.sum(slice_select) * source_dt, rtol=1e-12
    )
    np.testing.assert_array_equal(gradients[:, 1], 0.0)
    # The bipolar lobe still switches sign half way through the pulse.
    first_negative = time[np.argmax(gradients[:, 0] < 0.0)] - time[0]
    assert first_negative == pytest.approx(0.5e-3, abs=dt)
    lobe = np.sum(bipolar[: n // 2]) * source_dt
    assert np.sum(np.clip(gradients[:, 0], 0.0, None)) * dt == pytest.approx(
        lobe, rel=1e-2
    )

    # Gradients that already sit on the target raster are used as given.
    on_target = np.ones((len(b1), 3))
    pulse.custom_gradients = on_target
    np.testing.assert_array_equal(pulse.compile(dt)[1], on_target)
