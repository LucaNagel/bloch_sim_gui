"""Error-controlled integration of a single RF pulse followed by free decay.

The waveform raster, internal solver steps and uniform saved time axis are
independent. This path uses the legacy kernel's units and sign convention.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .memory import enforce_memory_budget
from .units import LEGACY_GAMMA_RAD_PER_S_PER_GAUSS as GAMMA


class SimulationCancelled(Exception):
    """A worker observed a cancellation request between integration steps."""


def integrate_rf_pulse(
    b1,
    time,
    tissue,
    positions,
    frequencies,
    initial,
    *,
    output_step_s,
    budget,
    mode=2,
    rtol=1e-8,
    atol=1e-10,
    cancel_check=None,
    progress=None,
):
    """Integrate one contiguous RF pulse, then propagate RF-free decay exactly.

    DOP853 controls the local magnetization error, including amplitude, phase,
    off-resonance and relaxation. Linear interpolation preserves the supplied
    RF waveform; refining integration cannot recover a poorly sampled input.
    Input timestamps are RF sample centres, as returned by the RF designer.
    Integration begins and ends at the surrounding half-raster boundaries.
    """
    if mode not in (0, 2):
        raise ValueError(
            "Adaptive RF integration supports endpoint and transient modes"
        )
    if not np.isfinite(output_step_s) or output_step_s <= 0:
        raise ValueError("output_step_s must be positive and finite")
    if time.size < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError(
            "Adaptive RF integration needs a strictly increasing time axis"
        )
    if not np.all(np.isfinite(b1)):
        raise ValueError("RF samples must be finite")
    if tissue.t1 <= 0 or tissue.t2 <= 0:
        raise ValueError("T1 and T2 must be positive")
    active = np.flatnonzero(b1 != 0)
    if active.size and (active[0] != 0 or np.any(np.diff(active) != 1)):
        raise ValueError(
            "Adaptive RF integration requires one initial contiguous pulse"
        )
    end = float(time[-1] + 0.5 * (time[-1] - time[-2]))
    start = float(time[0] - 0.5 * (time[1] - time[0]))
    pulse_end = (
        float(0.5 * (time[active[-1]] + time[active[-1] + 1]))
        if active.size and active[-1] + 1 < time.size
        else end
    )
    if not active.size:
        pulse_end = start
    npos, nfreq = len(positions), len(frequencies)
    count = int(np.ceil((end - start) / output_step_s - 1e-9)) + 1 if mode == 2 else 1
    # Includes solver output, assembled components, signal and working copies.
    enforce_memory_budget(
        count * npos * nfreq * 128 + time.size * 64 + npos * nfreq * 2048,
        budget,
        description=f"Adaptive RF: {count:,} saved times × {npos:,} positions × {nfreq:,} frequencies",
        suggestions="Increase the output time step or reduce positions or frequencies",
    )
    output_time = np.linspace(start, end, count) if mode == 2 else np.array([end])
    if initial is None:
        state = np.zeros((3, npos, nfreq))
        state[2] = 1.0
    else:
        # Legacy initial buffers are frequency-major, then position.
        state = np.asarray(initial).reshape(3, nfreq, npos).transpose(0, 2, 1).copy()
    result = np.empty((3, count, npos, nfreq))
    calls = 0
    omega_z = -2 * np.pi * frequencies[None, :]
    wave_real = np.ascontiguousarray(b1.real)
    wave_imag = np.ascontiguousarray(b1.imag)
    max_step = min(20e-6, (pulse_end - start) / 32) if pulse_end > start else 20e-6
    if active.size > 1:
        phase_rate = np.max(
            np.abs(np.diff(np.unwrap(np.angle(b1[: active.size]))))
            / np.diff(time[: active.size])
        )
        if phase_rate > 0:
            max_step = min(max_step, 0.2 / phase_rate)

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise SimulationCancelled()

    def rhs(t, y):
        nonlocal calls
        check_cancel()
        calls += 1
        if progress is not None and calls % 256 == 0:
            progress(min(99, int(100 * (t - start) / (end - start))))
        field = complex(np.interp(t, time, wave_real), np.interp(t, time, wave_imag))
        # Hold the final RF sample until the exact pulse boundary.
        if active.size and t >= time[active[-1]]:
            field = b1[active[-1]]
        mx, my, mz = y.reshape(3, npos, nfreq)
        wx, wy = -GAMMA * field.real, GAMMA * field.imag
        return np.stack(
            (
                wy * mz - omega_z * my - mx / tissue.t2,
                omega_z * mx - wx * mz - my / tissue.t2,
                wx * my - wy * mx + (1 - mz) / tissue.t1,
            )
        ).ravel()

    check_cancel()
    during = output_time <= pulse_end
    if pulse_end > start:
        evaluation = np.unique(np.r_[output_time[during], pulse_end])
        solution = solve_ivp(
            rhs,
            (start, pulse_end),
            state.ravel(),
            method="DOP853",
            t_eval=evaluation,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        values = solution.y.reshape(3, npos, nfreq, -1).transpose(0, 3, 1, 2)
        result[:, during] = values[:, : np.count_nonzero(during)]
        state = values[:, -1]
    elif np.any(during):
        result[:, during] = state[:, None]
    check_cancel()
    after = ~during
    delay = (output_time[after] - pulse_end)[:, None, None]
    transverse = (state[0] + 1j * state[1])[None] * np.exp(
        -delay / tissue.t2 + 1j * omega_z[None] * delay
    )
    result[0, after] = transverse.real
    result[1, after] = transverse.imag
    result[2, after] = 1 + (state[2][None] - 1) * np.exp(-delay / tissue.t1)
    metadata = {
        "solver": "DOP853",
        "rtol": rtol,
        "atol": atol,
        "rhs_evaluations": calls,
        "output_step_s": float(output_time[1] - output_time[0]) if count > 1 else None,
        "input_points": int(time.size),
    }
    return result, output_time, metadata
