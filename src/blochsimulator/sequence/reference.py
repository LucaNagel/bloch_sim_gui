"""Independent high-accuracy reference integration for sequence tests.

This module intentionally does not use :class:`SequenceCompiler` or either
native sequence kernel.  It expands the piecewise-constant event model at its
native RF and gradient rasters and solves the affine Bloch equation exactly on
every resulting interval with a matrix exponential.

The implementation is designed as a correctness oracle for small spin-probe
ensembles.  It is deliberately much slower than the production streaming
solver and should not be used for full imaging phantoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm

from .model import GradientEvent, RFEvent, SequenceProgram


@dataclass(frozen=True)
class ReferenceSimulationResult:
    """Magnetization states produced by the independent reference solver.

    Magnetization arrays use ``(observation, spin, xyz)`` order.  The final
    state has shape ``(spin, xyz)``.  ADC states are not receiver-weighted or
    demodulated; retaining the individual spin states avoids cancellation
    hiding a local integration error.
    """

    final_magnetization: np.ndarray
    adc_times_s: np.ndarray
    adc_magnetization: np.ndarray
    checkpoint_times_s: np.ndarray
    checkpoint_magnetization: np.ndarray
    interval_count: int

    def __post_init__(self) -> None:
        for name in (
            "final_magnetization",
            "adc_times_s",
            "adc_magnetization",
            "checkpoint_times_s",
            "checkpoint_magnetization",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def simulate_reference_sequence(
    program: SequenceProgram,
    *,
    positions_m=(0.0, 0.0, 0.0),
    frequency_offsets_hz=0.0,
    t1_s=1.0,
    t2_s=0.1,
    initial_magnetization=(0.0, 0.0, 1.0),
    tx_sensitivity=1.0 + 0.0j,
    checkpoints_s: Sequence[float] = (),
) -> ReferenceSimulationResult:
    """Propagate a small spin ensemble with an exact affine interval solve.

    Parameters are either scalar/single-spin values or arrays with one entry
    per spin. ``positions_m`` and ``initial_magnetization`` use trailing xyz
    dimensions.  Single values are broadcast to the largest supplied spin
    dimension.

    RF is interpreted in Hz, gradients in Hz/m, positions in metres, frequency
    offsets in Hz, and all time values in seconds, matching ``SequenceProgram``.
    The rotation signs match the event-based production kernel.
    """

    if not isinstance(program, SequenceProgram):
        raise TypeError(f"program must be SequenceProgram, got {type(program)}")

    positions = _vectors(positions_m, "positions_m")
    initial = _vectors(initial_magnetization, "initial_magnetization")
    frequencies = _values(frequency_offsets_hz, np.float64, "frequency_offsets_hz")
    t1 = _values(t1_s, np.float64, "t1_s")
    t2 = _values(t2_s, np.float64, "t2_s")
    tx = _values(tx_sensitivity, np.complex128, "tx_sensitivity")
    n_spins = max(
        positions.shape[0],
        initial.shape[0],
        frequencies.size,
        t1.size,
        t2.size,
        tx.size,
    )
    positions = _broadcast_rows(positions, n_spins, "positions_m")
    state = _broadcast_rows(initial, n_spins, "initial_magnetization").copy()
    frequencies = _broadcast_values(frequencies, n_spins, "frequency_offsets_hz")
    t1 = _broadcast_values(t1, n_spins, "t1_s")
    t2 = _broadcast_values(t2, n_spins, "t2_s")
    tx = _broadcast_values(tx, n_spins, "tx_sensitivity")

    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("t1_s and t2_s must be positive")

    checkpoints = _checkpoints(checkpoints_s, program.duration_s)
    adc_times = _adc_times(program)
    boundaries = _native_boundaries(program, adc_times, checkpoints)
    adc_state_indices = _observation_indices(boundaries, adc_times)
    checkpoint_state_indices = _observation_indices(boundaries, checkpoints)
    adc_states = np.empty((adc_times.size, n_spins, 3), dtype=np.float64)
    checkpoint_states = np.empty((checkpoints.size, n_spins, 3), dtype=np.float64)

    adc_by_state = _indices_by_state(adc_state_indices)
    checkpoints_by_state = _indices_by_state(checkpoint_state_indices)
    _record_observations(adc_states, adc_by_state.get(0, ()), state)
    _record_observations(checkpoint_states, checkpoints_by_state.get(0, ()), state)

    for interval_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        dt = float(end - start)
        midpoint = float(start + 0.5 * dt)
        rf_hz = _rf_at(program, midpoint)
        gradient_hz_per_m = _gradient_at(program, midpoint)

        for spin in range(n_spins):
            effective_rf = rf_hz * tx[spin]
            effective_frequency = frequencies[spin] + np.dot(
                gradient_hz_per_m, positions[spin]
            )
            transition = expm(
                _affine_bloch_generator(
                    effective_rf,
                    float(effective_frequency),
                    float(t1[spin]),
                    float(t2[spin]),
                )
                * dt
            )
            homogeneous_state = np.empty(4, dtype=np.float64)
            homogeneous_state[:3] = state[spin]
            homogeneous_state[3] = 1.0
            state[spin] = (transition @ homogeneous_state)[:3]

        state_index = interval_index + 1
        _record_observations(adc_states, adc_by_state.get(state_index, ()), state)
        _record_observations(
            checkpoint_states,
            checkpoints_by_state.get(state_index, ()),
            state,
        )

    return ReferenceSimulationResult(
        final_magnetization=state,
        adc_times_s=adc_times,
        adc_magnetization=adc_states,
        checkpoint_times_s=checkpoints,
        checkpoint_magnetization=checkpoint_states,
        interval_count=max(0, boundaries.size - 1),
    )


def _affine_bloch_generator(
    rf_hz: complex, frequency_hz: float, t1_s: float, t2_s: float
) -> np.ndarray:
    """Return the homogeneous 4x4 generator for one constant interval."""

    omega = (
        2.0
        * np.pi
        * np.array([-rf_hz.real, rf_hz.imag, -frequency_hz], dtype=np.float64)
    )
    wx, wy, wz = omega
    generator = np.zeros((4, 4), dtype=np.float64)
    generator[:3, :3] = (
        (0.0, -wz, wy),
        (wz, 0.0, -wx),
        (-wy, wx, 0.0),
    )
    generator[0, 0] -= 1.0 / t2_s
    generator[1, 1] -= 1.0 / t2_s
    generator[2, 2] -= 1.0 / t1_s
    generator[2, 3] = 1.0 / t1_s
    return generator


def _native_boundaries(
    program: SequenceProgram, adc_times: np.ndarray, checkpoints: np.ndarray
) -> np.ndarray:
    values = [0.0, float(program.duration_s)]
    values.extend(adc_times.tolist())
    values.extend(checkpoints.tolist())
    for event in (*program.rf_events, *program.gradient_events):
        sample_count = (
            event.samples_hz.size
            if isinstance(event, RFEvent)
            else event.samples_hz_per_m.size
        )
        values.extend(
            (
                event.start_s
                + np.arange(sample_count + 1, dtype=np.float64) * event.raster_s
            ).tolist()
        )
    return _coalesce_boundaries(values, program.duration_s)


def _coalesce_boundaries(values, duration_s: float) -> np.ndarray:
    tolerance = max(1e-15, 32 * np.spacing(max(1.0, abs(duration_s))))
    boundaries = np.sort(np.asarray(values, dtype=np.float64))
    boundaries[np.abs(boundaries) <= tolerance] = 0.0
    boundaries[np.abs(boundaries - duration_s) <= tolerance] = duration_s
    boundaries = boundaries[(boundaries >= 0.0) & (boundaries <= duration_s)]
    if boundaries.size == 0:
        return np.zeros(0, dtype=np.float64)

    result = [float(boundaries[0])]
    anchor = float(boundaries[0])
    for value in boundaries[1:]:
        value = float(value)
        if value - anchor <= tolerance:
            if value == duration_s:
                result[-1] = value
            continue
        result.append(value)
        anchor = value
    return np.asarray(result, dtype=np.float64)


def _rf_at(program: SequenceProgram, time_s: float) -> complex:
    value = 0.0j
    for event in program.rf_events:
        index = _sample_index(
            time_s, event.start_s, event.raster_s, event.samples_hz.size
        )
        if index is None:
            continue
        cell_start = event.start_s + index * event.raster_s
        phase = event.phase_offset_rad + 2.0 * np.pi * event.frequency_offset_hz * (
            cell_start - event.start_s
        )
        value += event.samples_hz[index] * np.exp(1j * phase)
    return complex(value)


def _gradient_at(program: SequenceProgram, time_s: float) -> np.ndarray:
    value = np.zeros(3, dtype=np.float64)
    for event in program.gradient_events:
        index = _sample_index(
            time_s,
            event.start_s,
            event.raster_s,
            event.samples_hz_per_m.size,
        )
        if index is not None:
            value["xyz".index(event.axis)] += event.samples_hz_per_m[index]
    return value


def _sample_index(
    time_s: float, event_start_s: float, raster_s: float, sample_count: int
):
    relative = (time_s - event_start_s) / raster_s
    if relative < 0.0 or relative >= sample_count:
        return None
    return min(int(np.floor(relative)), sample_count - 1)


def _adc_times(program: SequenceProgram) -> np.ndarray:
    values = []
    order = 0
    for event in program.adc_events:
        for time_s in event.sample_times_s:
            values.append((float(time_s), order))
            order += 1
    values.sort(key=lambda item: (item[0], item[1]))
    return np.asarray([value for value, _ in values], dtype=np.float64)


def _checkpoints(values: Sequence[float], duration_s: float) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.size == 0:
        return np.zeros(0, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("checkpoints_s must be a finite one-dimensional sequence")
    if np.any(result < 0.0) or np.any(result > duration_s):
        raise ValueError("checkpoint times must lie within the sequence")
    return np.unique(result)


def _observation_indices(boundaries: np.ndarray, times: np.ndarray) -> np.ndarray:
    if times.size == 0:
        return np.zeros(0, dtype=np.int64)
    insertions = np.searchsorted(boundaries, times, side="left")
    right = np.clip(insertions, 0, boundaries.size - 1)
    left = np.clip(insertions - 1, 0, boundaries.size - 1)
    use_left = np.abs(boundaries[left] - times) <= np.abs(boundaries[right] - times)
    indices = np.where(use_left, left, right)
    tolerance = max(1e-12, 32 * np.spacing(max(1.0, abs(boundaries[-1]))))
    if np.any(np.abs(boundaries[indices] - times) > tolerance):
        raise RuntimeError("reference timeline lost an observation boundary")
    return indices.astype(np.int64)


def _indices_by_state(state_indices: np.ndarray) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for observation, state_index in enumerate(state_indices):
        result.setdefault(int(state_index), []).append(observation)
    return result


def _record_observations(target, observations, state) -> None:
    for observation in observations:
        target[observation] = state


def _values(value, dtype, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a scalar or non-empty 1D array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vectors(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (3,):
        result = result.reshape(1, 3)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (n_spins, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _broadcast_values(values: np.ndarray, size: int, name: str) -> np.ndarray:
    if values.size == size:
        return np.asarray(values)
    if values.size == 1:
        return np.full(size, values.item(), dtype=values.dtype)
    raise ValueError(f"{name} has {values.size} entries, expected 1 or {size}")


def _broadcast_rows(values: np.ndarray, size: int, name: str) -> np.ndarray:
    if values.shape[0] == size:
        return np.asarray(values)
    if values.shape[0] == 1:
        return np.broadcast_to(values, (size, 3)).copy()
    raise ValueError(f"{name} has {values.shape[0]} rows, expected 1 or {size}")
