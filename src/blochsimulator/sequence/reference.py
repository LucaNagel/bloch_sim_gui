"""Independent reference integration (Gold standard) for sequence tests.

This module intentionally does not use :class:`SequenceCompiler` or any
native sequence kernel.  It expands the piecewise-constant event model at its
native RF and gradient rasters (plus ADC samples, checkpoints and optional
source knots) and advances the magnetization interval by interval with one of
two methods:

``method="exact"``
    Solves the affine Bloch(-McConnell) equation exactly on every interval
    with a matrix exponential.  Several spectral pools, longitudinal exchange
    (for example pyruvate-to-lactate conversion), relaxation towards a per-pool
    equilibrium and a longitudinal source that is linear within each interval
    (inflow) are part of one generator, so the only approximation is the
    piecewise-constant sampling of RF and gradients.

``method="hargreaves"``
    Reproduces the time-stepping of Brian Hargreaves' ``bloch.c``: for every
    interval, first the rotation about ``n = (-2*pi*Re(B1), 2*pi*Im(B1),
    -2*pi*(df + G.r)) * dt`` built from Cayley-Klein parameters, then the decay
    ``M <- diag(E2, E2, E1) M + (0, 0, M0 (1 - E1))``.  RF is in Hz and
    gradients in Hz/m instead of Gauss and G/cm, so ``GAMMA`` cancels; ``2*pi``
    is not truncated as in the original ``TWOPI`` constant.  Exchange and
    sources are not part of that model and are rejected.

``steady_state=True`` corresponds to Hargreaves' mode 3: the start state is the
fixed point of the affine map of the whole program, and the transient is then
recorded from it.

The implementation is a correctness oracle for small spin-probe ensembles.  It
is deliberately much slower than the production solvers and should not be used
for full imaging phantoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import expm

from .model import GradientEvent, RFEvent, SequenceProgram

REFERENCE_METHODS = ("exact", "hargreaves")


@dataclass(frozen=True)
class ReferencePool:
    """One spectral pool of the reference model.

    ``frequency_offset_hz`` is added to every spin's own offset.
    ``equilibrium`` is the longitudinal magnetization the pool relaxes to (use
    the thermal value, not zero, for hyperpolarized pools).
    """

    frequency_offset_hz: float = 0.0
    t1_s: float = 1.0
    t2_s: float = 0.1
    equilibrium: float = 1.0
    initial_magnetization: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ReferenceSimulationResult:
    """Magnetization states produced by the independent reference solver.

    Magnetization arrays use ``(observation, spin, xyz)`` order.  The final
    state has shape ``(spin, xyz)``.  When the solver was called with
    ``pools``, a pool axis is inserted before xyz, e.g. ``(observation, spin,
    pool, xyz)``.  ADC states are not receiver-weighted or demodulated;
    retaining the individual spin states avoids cancellation hiding a local
    integration error.  ``initial_magnetization`` is the start state, which is
    the steady state when ``steady_state=True`` was requested.
    """

    final_magnetization: np.ndarray
    adc_times_s: np.ndarray
    adc_magnetization: np.ndarray
    checkpoint_times_s: np.ndarray
    checkpoint_magnetization: np.ndarray
    interval_count: int
    initial_magnetization: np.ndarray | None = None
    method: str = "exact"

    def __post_init__(self) -> None:
        for name in (
            "final_magnetization",
            "adc_times_s",
            "adc_magnetization",
            "checkpoint_times_s",
            "checkpoint_magnetization",
            "initial_magnetization",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def simulate_reference_sequence(
    program: SequenceProgram,
    *,
    positions_m=(0.0, 0.0, 0.0),
    frequency_offsets_hz=0.0,
    t1_s=None,
    t2_s=None,
    initial_magnetization=None,
    tx_sensitivity=1.0 + 0.0j,
    checkpoints_s: Sequence[float] = (),
    method: str = "exact",
    pools: Sequence[ReferencePool] | None = None,
    exchange_rates_per_s=None,
    longitudinal_source: Callable[[float], object] | None = None,
    source_knots_s: Sequence[float] = (),
    steady_state: bool = False,
) -> ReferenceSimulationResult:
    """Propagate a small spin ensemble with the reference solver.

    Parameters are either scalar/single-spin values or arrays with one entry
    per spin. ``positions_m`` and ``initial_magnetization`` use trailing xyz
    dimensions.  Single values are broadcast to the largest supplied spin
    dimension.  Without ``pools`` the model has one pool with equilibrium 1;
    ``t1_s`` and ``t2_s`` default to 1 s and 0.1 s.

    RF is interpreted in Hz, gradients in Hz/m, positions in metres, frequency
    offsets in Hz, and all time values in seconds, matching ``SequenceProgram``.
    The rotation signs match the event-based production kernel and
    Hargreaves' ``bloch.c``.

    ``pools`` replaces ``t1_s``, ``t2_s`` and ``initial_magnetization`` by
    per-pool values.  ``exchange_rates_per_s[j, i]`` is the longitudinal rate
    from pool ``i`` into pool ``j`` in 1/s; diagonal entries are added as they
    are, so a conversion ``i -> j`` with rate ``k`` needs ``[j, i] = k`` and
    ``[i, i] = -k``.  ``longitudinal_source(t)`` returns the Mz source per
    second, broadcastable to ``(spin, pool)``; it is evaluated at interval
    boundaries and integrated exactly as a linear function within each
    interval.  Add the times where it has kinks to ``source_knots_s``.
    """

    if not isinstance(program, SequenceProgram):
        raise TypeError(f"program must be SequenceProgram, got {type(program)}")
    if method not in REFERENCE_METHODS:
        raise ValueError(f"method must be one of {REFERENCE_METHODS}, got {method!r}")

    positions = _vectors(positions_m, "positions_m")
    frequencies = _values(frequency_offsets_hz, np.float64, "frequency_offsets_hz")
    tx = _values(tx_sensitivity, np.complex128, "tx_sensitivity")
    pooled = pools is not None
    if pooled:
        if t1_s is not None or t2_s is not None or initial_magnetization is not None:
            raise ValueError("pass t1_s, t2_s and initial_magnetization through pools")
        pools = tuple(pools)
        if not pools or not all(isinstance(pool, ReferencePool) for pool in pools):
            raise TypeError("pools must be a non-empty sequence of ReferencePool")
        pool_t1 = np.array([pool.t1_s for pool in pools], dtype=np.float64)
        pool_t2 = np.array([pool.t2_s for pool in pools], dtype=np.float64)
        pool_offsets = np.array(
            [pool.frequency_offset_hz for pool in pools], dtype=np.float64
        )
        pool_equilibrium = np.array(
            [pool.equilibrium for pool in pools], dtype=np.float64
        )
        pool_initial = _vectors(
            [pool.initial_magnetization for pool in pools], "pool magnetization"
        )
        for name, value in (
            ("pool t1_s", pool_t1),
            ("pool t2_s", pool_t2),
            ("pool frequency_offset_hz", pool_offsets),
            ("pool equilibrium", pool_equilibrium),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        n_spins = max(positions.shape[0], frequencies.size, tx.size)
        n_pools = len(pools)
        t1 = np.broadcast_to(pool_t1, (n_spins, n_pools)).copy()
        t2 = np.broadcast_to(pool_t2, (n_spins, n_pools)).copy()
        equilibrium = np.broadcast_to(pool_equilibrium, (n_spins, n_pools)).copy()
        state = np.broadcast_to(pool_initial, (n_spins, n_pools, 3)).copy()
    else:
        initial = _vectors(
            (0.0, 0.0, 1.0) if initial_magnetization is None else initial_magnetization,
            "initial_magnetization",
        )
        t1_values = _values(1.0 if t1_s is None else t1_s, np.float64, "t1_s")
        t2_values = _values(0.1 if t2_s is None else t2_s, np.float64, "t2_s")
        n_spins = max(
            positions.shape[0],
            initial.shape[0],
            frequencies.size,
            t1_values.size,
            t2_values.size,
            tx.size,
        )
        n_pools = 1
        pool_offsets = np.zeros(1, dtype=np.float64)
        t1 = _broadcast_values(t1_values, n_spins, "t1_s")[:, None]
        t2 = _broadcast_values(t2_values, n_spins, "t2_s")[:, None]
        equilibrium = np.ones((n_spins, 1), dtype=np.float64)
        state = _broadcast_rows(initial, n_spins, "initial_magnetization")[
            :, None, :
        ].copy()
    positions = _broadcast_rows(positions, n_spins, "positions_m")
    frequencies = _broadcast_values(frequencies, n_spins, "frequency_offsets_hz")
    tx = _broadcast_values(tx, n_spins, "tx_sensitivity")
    pool_frequencies = frequencies[:, None] + pool_offsets[None, :]

    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("t1_s and t2_s must be positive")

    exchange = None
    if exchange_rates_per_s is not None:
        exchange = np.asarray(exchange_rates_per_s, dtype=np.float64)
        if exchange.shape != (n_pools, n_pools):
            raise ValueError(
                f"exchange_rates_per_s must have shape ({n_pools}, {n_pools})"
            )
        if not np.all(np.isfinite(exchange)):
            raise ValueError("exchange_rates_per_s must be finite")
        if not np.any(exchange):
            exchange = None
    if method == "hargreaves" and (
        exchange is not None or longitudinal_source is not None
    ):
        raise ValueError(
            "method='hargreaves' replicates bloch.c and supports neither "
            "exchange nor longitudinal sources"
        )
    if steady_state and longitudinal_source is not None:
        raise ValueError("steady_state is undefined with a longitudinal source")

    def source_at(time_s: float) -> np.ndarray:
        value = np.asarray(longitudinal_source(float(time_s)), dtype=np.float64)
        try:
            value = np.broadcast_to(value, (n_spins, n_pools))
        except ValueError as exc:
            raise ValueError(
                "longitudinal_source must return values broadcastable to "
                f"({n_spins}, {n_pools})"
            ) from exc
        if not np.all(np.isfinite(value)):
            raise ValueError("longitudinal_source returned non-finite values")
        return value

    checkpoints = _checkpoints(checkpoints_s, program.duration_s)
    knots = (
        _checkpoints(source_knots_s, program.duration_s)
        if longitudinal_source is not None
        else np.zeros(0, dtype=np.float64)
    )
    adc_times = _adc_times(program)
    boundaries = _native_boundaries(
        program, adc_times, np.concatenate((checkpoints, knots))
    )
    adc_state_indices = _observation_indices(boundaries, adc_times)
    checkpoint_state_indices = _observation_indices(boundaries, checkpoints)

    def interval_map(start: float, end: float):
        dt = float(end - start)
        midpoint = float(start + 0.5 * dt)
        rf_hz = _rf_at(program, midpoint)
        gradient_hz_per_m = _gradient_at(program, midpoint)
        effective_rf = rf_hz * tx
        frequency = pool_frequencies + (positions @ gradient_hz_per_m)[:, None]
        if method == "hargreaves":
            return _hargreaves_map(effective_rf, frequency, t1, t2, equilibrium, dt)
        sources = None
        if longitudinal_source is not None:
            sources = (source_at(start), source_at(end))
        return _exact_map(
            effective_rf, frequency, t1, t2, equilibrium, exchange, sources, dt
        )

    intervals = list(zip(boundaries[:-1], boundaries[1:]))
    if steady_state:
        dimension = 3 * n_pools
        total_matrix = np.broadcast_to(
            np.eye(dimension), (n_spins, dimension, dimension)
        ).copy()
        total_offset = np.zeros((n_spins, dimension), dtype=np.float64)
        for start, end in intervals:
            matrix, offset = interval_map(start, end)
            total_matrix = matrix @ total_matrix
            total_offset = np.einsum("sij,sj->si", matrix, total_offset) + offset
        try:
            fixed_point = np.linalg.solve(
                np.eye(dimension)[None] - total_matrix, total_offset[..., None]
            )[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the program has no unique steady state (I - A is singular)"
            ) from exc
        state = fixed_point.reshape(n_spins, n_pools, 3)

    start_state = state.copy()
    adc_states = np.empty((adc_times.size, n_spins, n_pools, 3), dtype=np.float64)
    checkpoint_states = np.empty(
        (checkpoints.size, n_spins, n_pools, 3), dtype=np.float64
    )
    adc_by_state = _indices_by_state(adc_state_indices)
    checkpoints_by_state = _indices_by_state(checkpoint_state_indices)
    _record_observations(adc_states, adc_by_state.get(0, ()), state)
    _record_observations(checkpoint_states, checkpoints_by_state.get(0, ()), state)

    for interval_index, (start, end) in enumerate(intervals):
        matrix, offset = interval_map(start, end)
        flat = state.reshape(n_spins, 3 * n_pools)
        state = (np.einsum("sij,sj->si", matrix, flat) + offset).reshape(
            n_spins, n_pools, 3
        )
        state_index = interval_index + 1
        _record_observations(adc_states, adc_by_state.get(state_index, ()), state)
        _record_observations(
            checkpoint_states,
            checkpoints_by_state.get(state_index, ()),
            state,
        )

    if not pooled:
        state = state[:, 0]
        start_state = start_state[:, 0]
        adc_states = adc_states[:, :, 0]
        checkpoint_states = checkpoint_states[:, :, 0]
    return ReferenceSimulationResult(
        final_magnetization=state,
        adc_times_s=adc_times,
        adc_magnetization=adc_states,
        checkpoint_times_s=checkpoints,
        checkpoint_magnetization=checkpoint_states,
        interval_count=max(0, boundaries.size - 1),
        initial_magnetization=start_state,
        method=method,
    )


def _rotation_vectors(rf_hz: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    """Return angular frequencies ``(spin, pool, xyz)`` in rad/s."""

    omega = np.empty((*frequency_hz.shape, 3), dtype=np.float64)
    omega[..., 0] = -2.0 * np.pi * rf_hz.real[:, None]
    omega[..., 1] = 2.0 * np.pi * rf_hz.imag[:, None]
    omega[..., 2] = -2.0 * np.pi * frequency_hz
    return omega


def _exact_map(rf_hz, frequency_hz, t1, t2, equilibrium, exchange, sources, dt):
    """Return the exact affine interval map ``M -> A M + b`` per spin."""

    n_spins, n_pools = frequency_hz.shape
    dimension = 3 * n_pools
    constant = dimension
    ramp = dimension + 1
    size = dimension + (2 if sources is not None else 1)
    omega = _rotation_vectors(rf_hz, frequency_hz)
    generator = np.zeros((n_spins, size, size), dtype=np.float64)
    for pool in range(n_pools):
        x, y, z = 3 * pool, 3 * pool + 1, 3 * pool + 2
        wx, wy, wz = omega[:, pool, 0], omega[:, pool, 1], omega[:, pool, 2]
        generator[:, x, y] = -wz
        generator[:, x, z] = wy
        generator[:, y, x] = wz
        generator[:, y, z] = -wx
        generator[:, z, x] = -wy
        generator[:, z, y] = wx
        generator[:, x, x] = -1.0 / t2[:, pool]
        generator[:, y, y] = -1.0 / t2[:, pool]
        generator[:, z, z] = -1.0 / t1[:, pool]
        generator[:, z, constant] = equilibrium[:, pool] / t1[:, pool]
        if exchange is not None:
            for source_pool in range(n_pools):
                generator[:, z, 3 * source_pool + 2] += exchange[pool, source_pool]
        if sources is not None:
            start_value, end_value = sources
            generator[:, z, constant] += start_value[:, pool]
            generator[:, z, ramp] = (end_value[:, pool] - start_value[:, pool]) / dt
    if sources is not None:
        generator[:, ramp, constant] = 1.0
    transition = expm(generator * dt)
    return transition[:, :dimension, :dimension], transition[:, :dimension, constant]


def _cayley_klein_rotation(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrices for rotation vectors ``(..., 3)`` as in ``bloch.c``."""

    nx, ny, nz = rotation[..., 0], rotation[..., 1], rotation[..., 2]
    phi = np.sqrt(nx * nx + ny * ny + nz * nz)
    zero = phi == 0.0
    safe_phi = np.where(zero, 1.0, phi)
    half_sine = np.where(zero, 0.0, np.sin(0.5 * phi) / safe_phi)
    ar = np.cos(0.5 * phi)
    ai = -nz * half_sine
    br = ny * half_sine
    bi = -nx * half_sine
    matrix = np.empty((*rotation.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = ar * ar - ai * ai - br * br + bi * bi
    matrix[..., 1, 0] = -2.0 * (ar * ai + br * bi)
    matrix[..., 2, 0] = 2.0 * (ai * bi - ar * br)
    matrix[..., 0, 1] = 2.0 * (ar * ai - br * bi)
    matrix[..., 1, 1] = ar * ar - ai * ai + br * br - bi * bi
    matrix[..., 2, 1] = -2.0 * (ai * br + ar * bi)
    matrix[..., 0, 2] = 2.0 * (ar * br + ai * bi)
    matrix[..., 1, 2] = 2.0 * (ar * bi - ai * br)
    matrix[..., 2, 2] = ar * ar + ai * ai - br * br - bi * bi
    matrix[zero] = np.eye(3)
    return matrix


def _hargreaves_map(rf_hz, frequency_hz, t1, t2, equilibrium, dt):
    """Rotate, then relax, as one time step of Hargreaves' ``bloch.c``."""

    n_spins, n_pools = frequency_hz.shape
    rotation = _cayley_klein_rotation(_rotation_vectors(rf_hz, frequency_hz) * dt)
    e1 = np.exp(-dt / t1)
    e2 = np.exp(-dt / t2)
    rotation[..., 0:2, :] *= e2[..., None, None]
    rotation[..., 2, :] *= e1[..., None]
    dimension = 3 * n_pools
    matrix = np.zeros((n_spins, dimension, dimension), dtype=np.float64)
    offset = np.zeros((n_spins, dimension), dtype=np.float64)
    for pool in range(n_pools):
        block = slice(3 * pool, 3 * pool + 3)
        matrix[:, block, block] = rotation[:, pool]
        offset[:, 3 * pool + 2] = equilibrium[:, pool] * (1.0 - e1[:, pool])
    return matrix, offset


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
