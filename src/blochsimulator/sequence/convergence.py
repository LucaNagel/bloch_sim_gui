"""Small-ensemble convergence testing for RF-active simulation time steps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .model import SequenceProgram
from .reference import ReferenceSimulationResult, simulate_reference_sequence


@dataclass(frozen=True)
class SpinProbeEnsemble:
    """Paired physical parameters for a small collection of test spins."""

    positions_m: np.ndarray
    frequency_offsets_hz: np.ndarray
    b1_scales: np.ndarray
    t1_s: np.ndarray
    t2_s: np.ndarray
    initial_magnetization: np.ndarray

    def __post_init__(self) -> None:
        positions = _vectors(self.positions_m, "positions_m")
        n_spins = positions.shape[0]
        values = {
            "frequency_offsets_hz": _per_spin_values(
                self.frequency_offsets_hz,
                np.float64,
                n_spins,
                "frequency_offsets_hz",
            ),
            "b1_scales": _per_spin_values(
                self.b1_scales, np.complex128, n_spins, "b1_scales"
            ),
            "t1_s": _per_spin_values(self.t1_s, np.float64, n_spins, "t1_s"),
            "t2_s": _per_spin_values(self.t2_s, np.float64, n_spins, "t2_s"),
        }
        initial = _vectors(self.initial_magnetization, "initial_magnetization")
        if initial.shape[0] == 1 and n_spins > 1:
            initial = np.broadcast_to(initial, (n_spins, 3)).copy()
        elif initial.shape[0] != n_spins:
            raise ValueError(
                "initial_magnetization must have one row or one row per spin"
            )
        if np.any(values["t1_s"] <= 0) or np.any(values["t2_s"] <= 0):
            raise ValueError("t1_s and t2_s must be positive")

        arrays = {"positions_m": positions, "initial_magnetization": initial, **values}
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def n_spins(self) -> int:
        return int(self.positions_m.shape[0])

    @classmethod
    def from_axes(
        cls,
        positions_m,
        *,
        frequency_offsets_hz=(0.0,),
        b1_scales=(1.0,),
        relaxation_times_s=((1.0, 0.1),),
        initial_magnetization=(0.0, 0.0, 1.0),
    ) -> "SpinProbeEnsemble":
        """Create the Cartesian product of position, frequency, B1 and T1/T2.

        ``relaxation_times_s`` contains explicit ``(T1, T2)`` pairs rather than
        two independent axes.  This prevents the helper from silently creating
        physically unintended T1/T2 combinations.
        """

        positions = _vectors(positions_m, "positions_m")
        frequencies = _axis_values(
            frequency_offsets_hz, np.float64, "frequency_offsets_hz"
        )
        b1 = _axis_values(b1_scales, np.complex128, "b1_scales")
        relaxation = np.asarray(relaxation_times_s, dtype=np.float64)
        if relaxation.shape == (2,):
            relaxation = relaxation.reshape(1, 2)
        if (
            relaxation.ndim != 2
            or relaxation.shape[0] == 0
            or relaxation.shape[1] != 2
            or not np.all(np.isfinite(relaxation))
        ):
            raise ValueError("relaxation_times_s must contain finite (T1, T2) pairs")
        if np.any(relaxation <= 0):
            raise ValueError("relaxation_times_s values must be positive")

        combinations = np.asarray(
            tuple(
                product(
                    range(positions.shape[0]),
                    range(frequencies.size),
                    range(b1.size),
                    range(relaxation.shape[0]),
                )
            ),
            dtype=np.int64,
        )
        return cls(
            positions_m=positions[combinations[:, 0]],
            frequency_offsets_hz=frequencies[combinations[:, 1]],
            b1_scales=b1[combinations[:, 2]],
            t1_s=relaxation[combinations[:, 3], 0],
            t2_s=relaxation[combinations[:, 3], 1],
            initial_magnetization=initial_magnetization,
        )

    def describe(self, index: int) -> dict:
        """Return the physical parameters of one expanded probe."""

        if not 0 <= int(index) < self.n_spins:
            raise IndexError("probe index is out of range")
        index = int(index)
        return {
            "index": index,
            "position_m": self.positions_m[index].tolist(),
            "frequency_offset_hz": float(self.frequency_offsets_hz[index]),
            "b1_scale": complex(self.b1_scales[index]),
            "t1_s": float(self.t1_s[index]),
            "t2_s": float(self.t2_s[index]),
            "initial_magnetization": self.initial_magnetization[index].tolist(),
        }


@dataclass(frozen=True)
class ConvergenceCriteria:
    """Pass/fail limits for local normalized magnetization vectors."""

    max_vector_error: float = 1e-3
    rms_vector_error: float = 2e-4

    def __post_init__(self) -> None:
        for name in ("max_vector_error", "rms_vector_error"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TimestepConvergencePoint:
    """Accuracy and runtime measurements for one production time step."""

    simulation_timestep_s: float | None
    interval_count: int
    runtime_s: float
    max_vector_error: float
    rms_vector_error: float
    final_max_vector_error: float
    worst_probe_index: int
    worst_time_s: float
    passed: bool

    @property
    def simulation_timestep_us(self) -> float | None:
        if self.simulation_timestep_s is None:
            return None
        return self.simulation_timestep_s * 1e6


@dataclass(frozen=True)
class TimestepConvergenceResult:
    """Gold-reference comparison over multiple candidate time steps."""

    probes: SpinProbeEnsemble
    criteria: ConvergenceCriteria
    checkpoint_times_s: np.ndarray
    reference: ReferenceSimulationResult
    points: tuple[TimestepConvergencePoint, ...]

    def __post_init__(self) -> None:
        times = np.ascontiguousarray(self.checkpoint_times_s, dtype=np.float64)
        times.setflags(write=False)
        object.__setattr__(self, "checkpoint_times_s", times)
        object.__setattr__(self, "points", tuple(self.points))

    @property
    def native_point(self) -> TimestepConvergencePoint | None:
        return next(
            (point for point in self.points if point.simulation_timestep_s is None),
            None,
        )

    @property
    def coarsest_passing_timestep_s(self) -> float | None:
        values = [
            point.simulation_timestep_s
            for point in self.points
            if point.passed and point.simulation_timestep_s is not None
        ]
        return max(values, default=None)

    def to_records(self) -> list[dict]:
        """Return flat records suitable for CSV or dataframe export."""

        return [
            {
                "timestep": (
                    "native"
                    if point.simulation_timestep_s is None
                    else f"{point.simulation_timestep_us:g} us"
                ),
                "simulation_timestep_s": point.simulation_timestep_s,
                "simulation_timestep_us": point.simulation_timestep_us,
                "interval_count": point.interval_count,
                "runtime_s": point.runtime_s,
                "max_vector_error": point.max_vector_error,
                "rms_vector_error": point.rms_vector_error,
                "final_max_vector_error": point.final_max_vector_error,
                "worst_probe_index": point.worst_probe_index,
                "worst_time_s": point.worst_time_s,
                "passed": point.passed,
            }
            for point in self.points
        ]


DEFAULT_TIMESTEPS_S = (
    None,
    1e-6,
    2e-6,
    5e-6,
    10e-6,
    20e-6,
    50e-6,
    100e-6,
)


def default_probe_checkpoints(
    program: SequenceProgram, *, max_rf_checkpoints: int = 64
) -> np.ndarray:
    """Select RF-end observations without adding boundaries inside RF events."""

    if not isinstance(program, SequenceProgram):
        raise TypeError(f"program must be SequenceProgram, got {type(program)}")
    if int(max_rf_checkpoints) != max_rf_checkpoints or max_rf_checkpoints <= 0:
        raise ValueError("max_rf_checkpoints must be a positive integer")
    rf_ends = np.unique(
        np.asarray([event.end_s for event in program.rf_events], dtype=np.float64)
    )
    limit = int(max_rf_checkpoints)
    if rf_ends.size > limit:
        early_count = min(8, max(1, limit // 4))
        late_indices = np.linspace(
            early_count,
            rf_ends.size - 1,
            num=limit - early_count,
            dtype=np.int64,
        )
        selected = np.unique(
            np.concatenate((np.arange(early_count, dtype=np.int64), late_indices))
        )
        rf_ends = rf_ends[selected]
    return np.unique(np.append(rf_ends, program.duration_s))


def run_timestep_convergence(
    program: SequenceProgram,
    probes: SpinProbeEnsemble,
    *,
    timesteps_s: Iterable[float | None] = DEFAULT_TIMESTEPS_S,
    checkpoints_s: Sequence[float] | None = None,
    criteria: ConvergenceCriteria | None = None,
    max_rf_checkpoints: int = 64,
    simulator=None,
) -> TimestepConvergenceResult:
    """Compare production time steps with the independent gold reference.

    This routine is intended for tens to a few thousand probes, not full image
    phantoms.  The default observations are RF event ends plus sequence end, so
    the checkpoints do not insert additional boundaries inside an RF event and
    artificially improve a coarse candidate.
    """

    if not isinstance(program, SequenceProgram):
        raise TypeError(f"program must be SequenceProgram, got {type(program)}")
    if not isinstance(probes, SpinProbeEnsemble):
        raise TypeError(f"probes must be SpinProbeEnsemble, got {type(probes)}")
    criteria = ConvergenceCriteria() if criteria is None else criteria
    if not isinstance(criteria, ConvergenceCriteria):
        raise TypeError("criteria must be ConvergenceCriteria")
    timesteps = _timesteps(timesteps_s)
    checkpoints = (
        default_probe_checkpoints(program, max_rf_checkpoints=max_rf_checkpoints)
        if checkpoints_s is None
        else _checkpoint_times(checkpoints_s, program.duration_s)
    )

    reference = simulate_reference_sequence(
        program,
        positions_m=probes.positions_m,
        frequency_offsets_hz=probes.frequency_offsets_hz,
        t1_s=probes.t1_s,
        t2_s=probes.t2_s,
        initial_magnetization=probes.initial_magnetization,
        tx_sensitivity=probes.b1_scales,
        checkpoints_s=checkpoints,
    )
    phantom = _probe_phantom(probes)
    if simulator is None:
        from ..simulator import BlochSimulator

        simulator = BlochSimulator(use_parallel=False)

    points = []
    for timestep_s in timesteps:
        start = perf_counter()
        candidate = simulator.simulate_sequence(
            program,
            phantom,
            checkpoints_s=checkpoints,
            chunk_voxels=probes.n_spins,
            simulation_timestep_s=timestep_s,
        )
        runtime_s = perf_counter() - start
        candidate_checkpoints = np.asarray(candidate.checkpoint_magnetization)
        checkpoint_difference = (
            candidate_checkpoints - reference.checkpoint_magnetization
        )
        checkpoint_errors = np.linalg.norm(checkpoint_difference, axis=-1)
        worst_flat = int(np.argmax(checkpoint_errors))
        worst_checkpoint, worst_probe = np.unravel_index(
            worst_flat, checkpoint_errors.shape
        )
        max_error = float(checkpoint_errors[worst_checkpoint, worst_probe])
        rms_error = float(np.sqrt(np.mean(checkpoint_errors**2)))
        final_errors = np.linalg.norm(
            np.asarray(candidate.final_magnetization) - reference.final_magnetization,
            axis=-1,
        )
        final_max_error = float(np.max(final_errors))
        points.append(
            TimestepConvergencePoint(
                simulation_timestep_s=timestep_s,
                interval_count=int(candidate.metadata["n_intervals"]),
                runtime_s=float(runtime_s),
                max_vector_error=max(max_error, final_max_error),
                rms_vector_error=rms_error,
                final_max_vector_error=final_max_error,
                worst_probe_index=int(worst_probe),
                worst_time_s=float(checkpoints[worst_checkpoint]),
                passed=(
                    max(max_error, final_max_error) <= criteria.max_vector_error
                    and rms_error <= criteria.rms_vector_error
                ),
            )
        )

    return TimestepConvergenceResult(
        probes=probes,
        criteria=criteria,
        checkpoint_times_s=checkpoints,
        reference=reference,
        points=tuple(points),
    )


def _probe_phantom(probes: SpinProbeEnsemble):
    from ..phantom import Phantom

    shape = (probes.n_spins,)
    phantom = Phantom(
        shape=shape,
        fov=(1.0,),
        t1_map=np.array(probes.t1_s, copy=True),
        t2_map=np.array(probes.t2_s, copy=True),
        pd_map=np.ones(shape, dtype=np.float64),
        chemical_shift_map=np.array(probes.frequency_offsets_hz, copy=True),
        m0_map=np.array(probes.initial_magnetization, copy=True),
        mask=np.ones(shape, dtype=bool),
        tx_sensitivity_map=np.array(probes.b1_scales, copy=True),
        name="Time-step convergence probes",
        metadata={"probe_ensemble": True},
    )
    phantom.positions = np.array(probes.positions_m, copy=True)
    phantom.x = phantom.positions[:, 0].copy()
    phantom.y = phantom.positions[:, 1].copy()
    phantom.z = phantom.positions[:, 2].copy()
    return phantom


def _timesteps(values: Iterable[float | None]) -> tuple[float | None, ...]:
    result = []
    seen = set()
    for value in values:
        normalized = None if value is None else float(value)
        if normalized is not None and (not np.isfinite(normalized) or normalized <= 0):
            raise ValueError("timesteps_s values must be None or finite and positive")
        key = ("native",) if normalized is None else ("value", normalized)
        if key in seen:
            raise ValueError("timesteps_s must not contain duplicates")
        seen.add(key)
        result.append(normalized)
    if not result:
        raise ValueError("timesteps_s must contain at least one candidate")
    return tuple(result)


def _checkpoint_times(values: Sequence[float], duration_s: float) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("checkpoints_s must be a finite one-dimensional sequence")
    if np.any(result < 0) or np.any(result > duration_s):
        raise ValueError("checkpoint times must lie within the sequence")
    return np.unique(np.append(result, duration_s))


def _vectors(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (3,):
        result = result.reshape(1, 3)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (n, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _axis_values(value, dtype, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite non-empty one-dimensional axis")
    return result


def _per_spin_values(value, dtype, n_spins: int, name: str) -> np.ndarray:
    result = _axis_values(value, dtype, name)
    if result.size == 1 and n_spins > 1:
        return np.full(n_spins, result.item(), dtype=dtype)
    if result.size != n_spins:
        raise ValueError(f"{name} must have one value or one value per spin")
    return result
