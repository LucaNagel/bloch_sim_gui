"""Small-ensemble convergence testing for RF-active simulation time steps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .model import ADCEvent, SequenceProgram
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
    _static_spin_major: bool | None = None,
) -> TimestepConvergenceResult:
    """Compare production time steps with the independent gold reference.

    This routine is intended for tens to a few thousand probes, not full image
    phantoms.  The default observations are RF event ends plus sequence end, so
    the checkpoints do not insert additional boundaries inside an RF event and
    artificially improve a coarse candidate. The simulator's static kernel
    runs the candidates; the private switch ``_static_spin_major`` is passed to
    ``simulate_sequence``.
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
            _static_spin_major=_static_spin_major,
        )
        runtime_s = perf_counter() - start
        # Declared MagnetizationCheckpointTimes join the requested checkpoints;
        # compare only the rows the reference observed.
        candidate_checkpoints = np.asarray(candidate.checkpoint_magnetization)[
            _matching_rows(candidate.checkpoint_times_s, checkpoints)
        ]
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


TIMESTEP_ERROR_REFERENCE_RATIO = 4
DEFAULT_TIMESTEP_ERROR_INTERVAL_BUDGET = 400_000
DEFAULT_TIMESTEP_ERROR_LINE_SAMPLES = 9
# Directions whose |RF|-weighted gradient moment is below this fraction of the
# strongest one do not select along their axis and get no probe line.
_GRADIENT_DIRECTION_FRACTION = 1e-2
_NO_RELAXATION_S = 1e6


@dataclass(frozen=True)
class TimestepErrorEstimate:
    """Self-convergence estimate of the RF-active time-step error.

    ``max_vector_error`` is the largest ``|M(step) - M(reference)|`` over
    RF-end checkpoints and the end of the checked part. ``source`` says what
    was simulated and sets the unit: ``"static_probes"`` (thermal probe spins
    in the static kernel, in units of the equilibrium magnetization) or
    ``"dynamic_kernel"`` (phantom voxels in the dynamic kernel, with exchange,
    inflow and washout; per voxel and pool in units of that pool's peak
    magnetization in that voxel). ``status`` is ``"checked"``,
    ``"truncated"`` (only the first ``checked_duration_s`` fit the interval
    budget), ``"identical_discretization"`` (the reference grid equals the
    step grid inside every RF event, so the error is exactly zero),
    ``"native_raster"`` (no step was set) or ``"finer_than_reference"``.
    """

    status: str
    simulation_timestep_s: float | None
    reference_timestep_s: float | None
    max_vector_error: float
    worst_time_s: float | None
    worst_probe: dict | None
    probe_count: int
    checked_duration_s: float
    sequence_duration_s: float
    reference_interval_count: int
    runtime_s: float
    source: str = "static_probes"

    @property
    def simulated(self) -> bool:
        return self.status in ("checked", "truncated")

    def exceeds(self, threshold: float) -> bool:
        """Whether a simulated estimate is above ``threshold``."""

        return self.simulated and self.max_vector_error > float(threshold)

    def to_metadata(self) -> dict:
        """Return JSON-safe values for result metadata and project files.

        An error that was not simulated is ``None``; the probe B1 scale is
        always one and is left out.
        """

        error = float(self.max_vector_error)
        probe = None
        if self.worst_probe is not None:
            probe = {
                key: value
                for key, value in self.worst_probe.items()
                if key != "b1_scale"
            }
        return {
            "status": self.status,
            "source": self.source,
            "simulation_timestep_s": self.simulation_timestep_s,
            "reference_timestep_s": self.reference_timestep_s,
            "max_vector_error": error if np.isfinite(error) else None,
            "worst_time_s": self.worst_time_s,
            "worst_probe": probe,
            "probe_count": self.probe_count,
            "checked_duration_s": self.checked_duration_s,
            "sequence_duration_s": self.sequence_duration_s,
            "reference_interval_count": self.reference_interval_count,
            "runtime_s": self.runtime_s,
        }


def estimate_timestep_error(
    program: SequenceProgram,
    simulation_timestep_s: float | None,
    *,
    positions_m=(0.0, 0.0, 0.0),
    frequency_offsets_hz=(0.0,),
    t1_s=_NO_RELAXATION_S,
    t2_s=_NO_RELAXATION_S,
    reference_timestep_s: float | None = None,
    line_samples: int = DEFAULT_TIMESTEP_ERROR_LINE_SAMPLES,
    max_reference_intervals: int = DEFAULT_TIMESTEP_ERROR_INTERVAL_BUDGET,
    max_rf_checkpoints: int = 256,
    spoiler_mode: str = "ideal",
    simulator=None,
    cancel_callback=None,
    _static_spin_major: bool | None = None,
) -> TimestepErrorEstimate:
    """Estimate the RF time-step error of a whole sequence from a few probes.

    The same production kernel runs the sequence twice, at
    ``simulation_timestep_s`` and at ``reference_timestep_s`` (default: a
    quarter of the step). The integrator is second order in the RF step, so
    the difference underestimates the error against the converged solution by
    the factor ``1 - (reference / step)**2`` (6 % for the default ratio).
    Whole-sequence runs matter: RF trains such as balanced SSFP amplify
    per-pulse errors by orders of magnitude.

    Probe spins sit on lines through the centre of ``positions_m`` along every
    direction the gradients select while RF is active, spanning the projected
    extent of ``positions_m`` with ``line_samples`` points, combined with every
    ``frequency_offsets_hz`` entry. ``t1_s`` and ``t2_s`` are scalars or one
    value per frequency; the default is effectively no relaxation. ADC events
    are dropped because they do not change the magnetization. When the
    reference needs more than ``max_reference_intervals`` RF intervals, only
    the sequence up to the first RF event beyond that budget is checked.
    Both runs use the simulator's static kernel (spin-major for
    ``"production"``), so the estimate measures the kernel that produces the
    result; the private switch ``_static_spin_major`` overrides that.
    """

    start = perf_counter()
    step, reference, skip = _timestep_error_steps(
        program, simulation_timestep_s, reference_timestep_s
    )
    if int(line_samples) != line_samples or line_samples < 2:
        raise ValueError("line_samples must be an integer of at least 2")
    _check_interval_budget(max_reference_intervals)
    frequencies = _axis_values(frequency_offsets_hz, np.float64, "frequency_offsets_hz")
    t1 = _per_spin_values(t1_s, np.float64, frequencies.size, "t1_s")
    t2 = _per_spin_values(t2_s, np.float64, frequencies.size, "t2_s")
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("t1_s and t2_s must be positive")
    sample_positions = _vectors(positions_m, "positions_m")
    if skip is not None:
        return _skipped_timestep_error(skip, program, step, reference, start)

    checked, truncated = _timestep_error_program(
        program, reference, int(max_reference_intervals)
    )
    probe_positions = _timestep_error_positions(
        checked, sample_positions, int(line_samples)
    )
    index = np.asarray(
        tuple(product(range(probe_positions.shape[0]), range(frequencies.size))),
        dtype=np.int64,
    )
    probes = SpinProbeEnsemble(
        positions_m=probe_positions[index[:, 0]],
        frequency_offsets_hz=frequencies[index[:, 1]],
        b1_scales=1.0,
        t1_s=t1[index[:, 1]],
        t2_s=t2[index[:, 1]],
        initial_magnetization=(0.0, 0.0, 1.0),
    )
    checkpoints = default_probe_checkpoints(
        checked, max_rf_checkpoints=max_rf_checkpoints
    )
    phantom = _probe_phantom(probes)
    if simulator is None:
        from ..simulator import BlochSimulator

        simulator = BlochSimulator(use_parallel=False)

    runs = []
    for timestep_s in (step, reference):
        runs.append(
            simulator.simulate_sequence(
                checked,
                phantom,
                checkpoints_s=checkpoints,
                chunk_voxels=probes.n_spins,
                simulation_timestep_s=timestep_s,
                spoiler_mode=spoiler_mode,
                cancel_callback=cancel_callback,
                _static_spin_major=_static_spin_major,
            )
        )
    candidate, gold = runs
    errors = np.linalg.norm(
        np.asarray(candidate.checkpoint_magnetization)
        - np.asarray(gold.checkpoint_magnetization),
        axis=-1,
    )
    worst_checkpoint, worst_probe = np.unravel_index(
        int(np.argmax(errors)), errors.shape
    )
    return TimestepErrorEstimate(
        status="truncated" if truncated else "checked",
        simulation_timestep_s=step,
        reference_timestep_s=reference,
        max_vector_error=float(errors[worst_checkpoint, worst_probe]),
        # Declared MagnetizationCheckpointTimes join the requested checkpoints.
        worst_time_s=float(np.asarray(gold.checkpoint_times_s)[worst_checkpoint]),
        worst_probe=probes.describe(int(worst_probe)),
        probe_count=probes.n_spins,
        checked_duration_s=float(checked.duration_s),
        sequence_duration_s=float(program.duration_s),
        reference_interval_count=int(gold.metadata["n_intervals"]),
        runtime_s=perf_counter() - start,
    )


def _timestep_error_steps(
    program: SequenceProgram,
    simulation_timestep_s: float | None,
    reference_timestep_s: float | None,
) -> tuple[float | None, float | None, str | None]:
    """Return the step, the reference step and the skip status, if any."""

    if not isinstance(program, SequenceProgram):
        raise TypeError(f"program must be SequenceProgram, got {type(program)}")
    from .compiler import SequenceCompiler

    step = None if simulation_timestep_s is None else float(simulation_timestep_s)
    if step is not None and (not np.isfinite(step) or step <= 0):
        raise ValueError("simulation_timestep_s must be None or finite and positive")
    if reference_timestep_s is None:
        reference = None if step is None else step / TIMESTEP_ERROR_REFERENCE_RATIO
    else:
        reference = float(reference_timestep_s)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("reference_timestep_s must be finite and positive")
    if step is None:
        return step, reference, "native_raster"
    if reference >= step:
        return step, reference, "finer_than_reference"
    rf_step = SequenceCompiler._rf_step
    if all(
        rf_step(event, step) == rf_step(event, reference) for event in program.rf_events
    ):
        return step, reference, "identical_discretization"
    return step, reference, None


def _check_interval_budget(max_reference_intervals) -> None:
    if int(max_reference_intervals) != max_reference_intervals or (
        max_reference_intervals <= 0
    ):
        raise ValueError("max_reference_intervals must be a positive integer")


def _skipped_timestep_error(
    status: str,
    program: SequenceProgram,
    step: float | None,
    reference: float | None,
    start: float,
    source: str = "static_probes",
) -> TimestepErrorEstimate:
    return TimestepErrorEstimate(
        status=status,
        simulation_timestep_s=step,
        reference_timestep_s=reference,
        max_vector_error=0.0 if status == "identical_discretization" else np.nan,
        worst_time_s=None,
        worst_probe=None,
        probe_count=0,
        checked_duration_s=0.0,
        sequence_duration_s=float(program.duration_s),
        reference_interval_count=0,
        runtime_s=perf_counter() - start,
        source=source,
    )


DEFAULT_DYNAMIC_TIMESTEP_ERROR_MAX_VOXELS = 256
# A pool's error in a voxel is relative to its peak magnetization there, but
# at least to this fraction of the pool's largest peak in the checked voxels,
# so nearly empty voxels do not dominate.
DYNAMIC_TIMESTEP_ERROR_PEAK_FLOOR = 1e-2


def dynamic_timestep_error_voxels(
    phantom,
    *,
    field_strength_t: float | None = None,
    nucleus: str | None = None,
    sequence_reference_ppm: float | None = None,
    max_voxels: int = DEFAULT_DYNAMIC_TIMESTEP_ERROR_MAX_VOXELS,
) -> np.ndarray:
    """Return flat indices of the active voxels the dynamic check simulates.

    A phantom with at most ``max_voxels`` active voxels is checked whole.
    Otherwise the extremes of every voxel map the dynamic kernel uses come
    first: per pool the median, lowest and highest off-resonance (B0 plus
    chemical shift relative to the sequence reference), the lowest and
    highest Tx sensitivity, kPL and kPA, inflow delivery and arrival delay,
    washout loss and dynamic B0 scale, one voxel per washout and contrast
    compartment and the brightest voxel per pool. Evenly spaced active
    voxels fill the rest. The indices are sorted and unique.
    """

    if int(max_voxels) != max_voxels or max_voxels < 1:
        raise ValueError("max_voxels must be a positive integer")
    active = np.flatnonzero(np.asarray(phantom.mask).ravel()).astype(np.int64)
    if active.size == 0:
        raise ValueError("the phantom has no active voxels")
    if active.size <= max_voxels:
        return active

    def values(array, dtype=np.float64):
        return np.asarray(array, dtype=dtype).ravel()[active]

    picks: list[int] = []

    def extremes(array, median=False):
        order = np.argsort(array, kind="stable")
        if median:
            picks.append(int(order[(order.size - 1) // 2]))
        picks.extend((int(order[0]), int(order[-1])))

    def one_per_group(array):
        _, first = np.unique(array, return_index=True)
        picks.extend(int(index) for index in first)

    frequencies = timestep_error_frequencies(
        phantom,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        sequence_reference_ppm=sequence_reference_ppm,
    )
    for pool_frequencies in frequencies:
        extremes(pool_frequencies[active], median=True)
    tx = getattr(phantom, "tx_sensitivity_map", None)
    if tx is not None:
        extremes(np.abs(values(tx, np.complex128)))
    extremes(values(phantom.kpl_map_s_inv))
    kpa = getattr(phantom, "kpa_map_s_inv", None)
    if kpa is not None and phantom.n_species > 2:
        extremes(values(kpa))
    inflow = phantom.pyruvate_inflow
    if inflow is not None:
        extremes(values(inflow.delivery_map))
        if inflow.arrival_delay_map_s is not None:
            extremes(values(inflow.arrival_delay_map_s))
    washout = phantom.voxel_washout
    if washout is not None:
        extremes(values(washout.loss_map_s_inv))
        if washout.compartment_map is not None:
            one_per_group(values(washout.compartment_map, np.int64))
    contrast = getattr(phantom, "contrast_relaxation", None)
    if contrast is not None:
        one_per_group(values(contrast.compartment_map, np.int64))
    if phantom.dynamic_b0 is not None:
        extremes(values(phantom.dynamic_b0.spatial_scale_map))
    for pool in phantom.pools:
        density = phantom.initial_concentration_maps.get(pool.name)
        if density is not None:
            picks.append(int(np.argmax(np.abs(values(density)))))

    chosen = list(dict.fromkeys(picks))[:max_voxels]
    remaining = np.setdiff1d(np.arange(active.size), chosen, assume_unique=True)
    fill = max_voxels - len(chosen)
    if fill > 0 and remaining.size:
        spaced = np.linspace(0, remaining.size - 1, min(fill, remaining.size))
        chosen.extend(remaining[np.unique(np.round(spaced).astype(np.int64))])
    return np.unique(active[np.asarray(chosen, dtype=np.int64)])


def timestep_error_frequencies(
    phantom,
    *,
    field_strength_t: float | None = None,
    nucleus: str | None = None,
    sequence_reference_ppm: float | None = None,
) -> np.ndarray:
    """Return the static off-resonance of every dynamic pool and voxel in Hz.

    The shape is ``(n_pools, n_voxels)`` over the flattened phantom: B0 plus
    the pool's chemical shift (and shift map) relative to the sequence
    reference, as the dynamic kernel sees them before dynamic B0.
    """

    from ..units import ppm_to_hz

    field = phantom.field_strength if field_strength_t is None else field_strength_t
    field = float(field)
    effective_nucleus = phantom.nucleus if nucleus is None else str(nucleus)
    reference_ppm = (
        phantom.spectral_reference_ppm
        if sequence_reference_ppm is None
        else float(sequence_reference_ppm)
    )
    b0 = np.asarray(phantom.b0_offset_hz(field, effective_nucleus), dtype=float)
    b0 = np.broadcast_to(b0, phantom.shape).ravel()
    reference_shift_hz = float(
        ppm_to_hz(
            phantom.spectral_reference_ppm - reference_ppm, field, effective_nucleus
        )
    )
    pool_shifts = phantom.pool_shift_offsets_hz(field, effective_nucleus)
    frequencies = np.empty((len(phantom.pools), b0.size), dtype=np.float64)
    for index, pool in enumerate(phantom.pools):
        frequencies[index] = (
            b0
            + pool.get_frequency_offset(field, effective_nucleus)
            + reference_shift_hz
        )
        if pool_shifts is not None:
            frequencies[index] += pool_shifts[index].ravel()
    return frequencies


def estimate_dynamic_timestep_error(
    program: SequenceProgram,
    phantom,
    simulation_timestep_s: float | None,
    *,
    field_strength_t: float | None = None,
    nucleus: str | None = None,
    sequence_reference_ppm: float | None = None,
    reference_timestep_s: float | None = None,
    voxel_indices=None,
    max_voxels: int = DEFAULT_DYNAMIC_TIMESTEP_ERROR_MAX_VOXELS,
    max_reference_intervals: int = DEFAULT_TIMESTEP_ERROR_INTERVAL_BUDGET,
    max_rf_checkpoints: int = 256,
    spoiler_mode: str = "ideal",
    spin_sampling=None,
    simulator=None,
    cancel_callback=None,
) -> TimestepErrorEstimate:
    """Estimate the RF time-step error of a dynamic phantom run.

    The dynamic kernel of ``simulator`` runs the sequence on a voxel subset
    (``voxel_indices`` or :func:`dynamic_timestep_error_voxels`) at
    ``simulation_timestep_s`` and at ``reference_timestep_s`` (default: a
    quarter of the step), with the phantom's conversion, inflow, washout,
    Tx and B0 maps. Each voxel keeps its full-phantom result because voxels
    do not couple. The error of a pool in a voxel is ``|M(step) -
    M(reference)|`` relative to that pool's peak ``|M|`` there (at least
    ``DYNAMIC_TIMESTEP_ERROR_PEAK_FLOOR`` of the pool's largest peak), at
    RF-end checkpoints and the end of the checked part. A small pool such as
    lactate thus reports its own relative error. The estimate is the largest
    over voxels and pools; the integrator is second order, so it is about
    6 % below the error against the converged solution.

    ADC events are dropped. When the reference needs more than
    ``max_reference_intervals`` RF intervals, only the sequence up to the
    first RF event beyond that budget is checked (``status="truncated"``);
    errors that build up over a long RF train are then underestimated.
    """

    start = perf_counter()
    step, reference, skip = _timestep_error_steps(
        program, simulation_timestep_s, reference_timestep_s
    )
    _check_interval_budget(max_reference_intervals)
    if voxel_indices is None:
        voxels = dynamic_timestep_error_voxels(
            phantom,
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            sequence_reference_ppm=sequence_reference_ppm,
            max_voxels=max_voxels,
        )
    else:
        voxels = np.unique(np.asarray(voxel_indices, dtype=np.int64).ravel())
    if skip is not None:
        return _skipped_timestep_error(
            skip, program, step, reference, start, "dynamic_kernel"
        )

    checked, truncated = _timestep_error_program(
        program, reference, int(max_reference_intervals)
    )
    checkpoints = np.union1d(
        default_probe_checkpoints(checked, max_rf_checkpoints=max_rf_checkpoints),
        (0.0,),
    )
    if simulator is None:
        from ..simulator import BlochSimulator

        simulator = BlochSimulator(use_parallel=False)
    options = dict(
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        sequence_reference_ppm=sequence_reference_ppm,
        checkpoints_s=checkpoints,
        spoiler_mode=spoiler_mode,
        cancel_callback=cancel_callback,
        _voxel_indices=voxels,
    )
    if spin_sampling is not None:
        options["spin_sampling"] = spin_sampling
    candidate, gold = (
        simulator.simulate_dynamic_sequence(
            checked, phantom, simulation_timestep_s=timestep_s, **options
        )
        for timestep_s in (step, reference)
    )
    # (checkpoint, pool, voxel, xyz)
    gold_states = np.asarray(gold.checkpoint_pool_magnetization, dtype=np.float64)
    differences = np.linalg.norm(
        np.asarray(candidate.checkpoint_pool_magnetization, dtype=np.float64)
        - gold_states,
        axis=-1,
    )
    peaks = np.linalg.norm(gold_states, axis=-1).max(axis=0)
    scales = np.maximum(
        peaks, DYNAMIC_TIMESTEP_ERROR_PEAK_FLOOR * peaks.max(axis=1, keepdims=True)
    )
    errors = np.divide(
        differences,
        scales[None],
        out=np.zeros_like(differences),
        where=scales[None] > 0,
    )
    worst_checkpoint, worst_pool, worst_voxel = np.unravel_index(
        int(np.argmax(errors)), errors.shape
    )
    voxel = int(voxels[worst_voxel])
    frequency = timestep_error_frequencies(
        phantom,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        sequence_reference_ppm=sequence_reference_ppm,
    )[worst_pool, voxel]
    metadata = gold.metadata
    intervals = metadata.get("compiled_interval_count", metadata.get("n_intervals"))
    return TimestepErrorEstimate(
        status="truncated" if truncated else "checked",
        simulation_timestep_s=step,
        reference_timestep_s=reference,
        max_vector_error=float(errors[worst_checkpoint, worst_pool, worst_voxel]),
        worst_time_s=float(np.asarray(gold.checkpoint_times_s)[worst_checkpoint]),
        worst_probe={
            "voxel_index": voxel,
            "position_m": [
                float(value) for value in np.asarray(phantom.positions)[voxel]
            ],
            "pool": phantom.pools[worst_pool].name,
            "frequency_offset_hz": float(frequency),
            "peak_magnetization": float(peaks[worst_pool, worst_voxel]),
        },
        probe_count=int(voxels.size),
        checked_duration_s=float(checked.duration_s),
        sequence_duration_s=float(program.duration_s),
        reference_interval_count=int(intervals or 0),
        runtime_s=perf_counter() - start,
        source="dynamic_kernel",
    )


DEFAULT_TIMESTEP_ERROR_MAX_FREQUENCIES = 12
_FREQUENCY_MATCH_HZ = 0.5


def timestep_error_inputs(
    phantom,
    *,
    field_strength_t: float | None = None,
    nucleus: str | None = None,
    sequence_reference_ppm: float | None = None,
    max_frequencies: int = DEFAULT_TIMESTEP_ERROR_MAX_FREQUENCIES,
) -> dict:
    """Return :func:`estimate_timestep_error` probe inputs for a phantom.

    Every frequency group (the voxels of a static phantom, each spectral
    component or each dynamic pool) contributes the voxels with its median,
    lowest and highest off-resonance (B0 plus chemical shift relative to the
    sequence reference), each with that voxel's T1/T2. Group medians come
    first, then the extremes; frequencies within 0.5 Hz of a chosen one are
    skipped and at most ``max_frequencies`` are kept. ``positions_m`` are the
    active voxel centres, whose bounding box the probe lines span.
    """

    from ..dynamic_phantom import DynamicSpectralPhantom
    from ..spectral_phantom import SpectralPhantom

    if int(max_frequencies) != max_frequencies or max_frequencies < 1:
        raise ValueError("max_frequencies must be a positive integer")
    groups = []
    positions = []
    if isinstance(phantom, DynamicSpectralPhantom):
        active = np.flatnonzero(np.asarray(phantom.mask).ravel())
        positions.append(np.asarray(phantom.positions, dtype=np.float64)[active])
        frequencies = timestep_error_frequencies(
            phantom,
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            sequence_reference_ppm=sequence_reference_ppm,
        )[:, active]
        for pool, pool_frequencies in zip(phantom.pools, frequencies):
            groups.append(
                (
                    pool_frequencies,
                    np.full(active.size, float(pool.t1)),
                    np.full(active.size, float(pool.t2)),
                )
            )
    else:
        if isinstance(phantom, SpectralPhantom):
            components = [
                component
                for _, component in phantom.to_component_phantoms(
                    field_strength_t, nucleus, sequence_reference_ppm
                )
            ]
        else:
            components = [phantom]
        for component in components:
            properties = component.get_active_properties()
            positions.append(np.asarray(properties["positions"], dtype=np.float64))
            groups.append(
                tuple(
                    np.asarray(properties[key], dtype=np.float64).ravel()
                    for key in ("df", "t1", "t2")
                )
            )

    positions = np.concatenate(positions) if positions else np.zeros((0, 3))
    if positions.shape[0] == 0:
        raise ValueError("the phantom has no active voxels")
    picks = [[], [], []]
    for frequencies, t1, t2 in groups:
        if frequencies.size == 0:
            continue
        order = np.argsort(frequencies, kind="stable")
        for rank, index in enumerate(
            (order[(order.size - 1) // 2], order[0], order[-1])
        ):
            picks[rank].append((frequencies[index], t1[index], t2[index]))
    chosen = []
    for frequency, t1, t2 in (pick for rank in picks for pick in rank):
        if len(chosen) == max_frequencies:
            break
        if all(abs(frequency - other[0]) > _FREQUENCY_MATCH_HZ for other in chosen):
            chosen.append((float(frequency), float(t1), float(t2)))
    if not chosen:
        raise ValueError("the phantom has no active voxels")
    chosen = np.asarray(chosen, dtype=np.float64)
    return {
        "positions_m": positions,
        "frequency_offsets_hz": chosen[:, 0],
        "t1_s": chosen[:, 1],
        "t2_s": chosen[:, 2],
    }


def _timestep_error_program(
    program: SequenceProgram, reference_s: float, max_intervals: int
) -> tuple[SequenceProgram, bool]:
    """Keep the events and definitions that change the magnetization.

    ADC events and their label/dimension metadata are dropped. With a budget
    cut, the program ends at the start of the first RF event beyond
    ``max_intervals`` (or later, where an earlier event lasts longer), keeps
    only events starting before it and trims the time definitions.
    """

    from .compiler import (
        IDEAL_SPOILER_DEFINITION,
        LEGACY_IDEAL_SPOILER_DEFINITIONS,
        SequenceCompiler,
    )

    cut = None
    total = 0
    for position, event in enumerate(program.rf_events):
        duration = event.samples_hz.size * event.raster_s
        step = SequenceCompiler._rf_step(event, reference_s)
        count = max(1, int(np.ceil(duration / step * (1.0 - 1e-9))))
        if position and total + count > max_intervals:
            cut = event.start_s
            break
        total += count
    events = tuple(
        event
        for event in program.events
        if not isinstance(event, ADCEvent) and (cut is None or event.start_s < cut)
    )
    duration = (
        program.duration_s
        if cut is None
        else max(cut, max(event.end_s for event in events))
    )
    definitions = program.metadata.get("definitions", {})
    definitions = dict(definitions) if isinstance(definitions, dict) else {}
    if cut is not None:
        tolerance = max(1e-15, duration * 1e-12)
        for key in (
            IDEAL_SPOILER_DEFINITION,
            *LEGACY_IDEAL_SPOILER_DEFINITIONS,
            "MagnetizationCheckpointTimes",
        ):
            if definitions.get(key) is None:
                continue
            values = np.asarray(definitions[key], dtype=float).reshape(-1)
            definitions[key] = values[values <= duration + tolerance].tolist()
    checked = SequenceProgram(
        events,
        duration,
        source=program.source,
        version=program.version,
        metadata={"definitions": definitions},
    )
    return checked, cut is not None


def _timestep_error_positions(
    program: SequenceProgram, positions: np.ndarray, line_samples: int
) -> np.ndarray:
    """Return the box centre plus lines along the RF-active gradient axes."""

    centre = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    moment = _rf_gradient_moment(program)
    points = [centre.reshape(1, 3)]
    if np.trace(moment) > 0:
        values, vectors = np.linalg.eigh(moment)
        for value, direction in zip(values, vectors.T):
            if value < _GRADIENT_DIRECTION_FRACTION * values[-1]:
                continue
            projected = (positions - centre) @ direction
            low, high = float(projected.min()), float(projected.max())
            if high > low:
                distances = np.linspace(low, high, line_samples)
                points.append(centre + np.outer(distances, direction))
    return np.unique(np.concatenate(points), axis=0)


def _rf_gradient_moment(program: SequenceProgram) -> np.ndarray:
    """Return ``sum |B1| g g^T dt`` over all RF samples (3 x 3)."""

    rf_events = program.rf_events
    if not rf_events:
        return np.zeros((3, 3), dtype=np.float64)
    times = np.concatenate(
        [
            event.start_s + (np.arange(event.samples_hz.size) + 0.5) * event.raster_s
            for event in rf_events
        ]
    )
    weights = np.concatenate(
        [np.abs(event.samples_hz) * event.raster_s for event in rf_events]
    )
    gradients = np.zeros((times.size, 3), dtype=np.float64)
    for column, axis in enumerate("xyz"):
        events = sorted(
            (event for event in program.gradient_events if event.axis == axis),
            key=lambda event: event.start_s,
        )
        if not events:
            continue
        starts = np.asarray([event.start_s for event in events])
        rasters = np.asarray([event.raster_s for event in events])
        sizes = np.asarray([event.samples_hz_per_m.size for event in events])
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        samples = np.concatenate([event.samples_hz_per_m for event in events])
        owner = np.searchsorted(starts, times, side="right") - 1
        valid = owner >= 0
        owner = np.where(valid, owner, 0)
        sample = np.floor((times - starts[owner]) / rasters[owner]).astype(np.int64)
        valid &= (sample >= 0) & (sample < sizes[owner])
        gradients[valid, column] = samples[offsets[owner[valid]] + sample[valid]]
    return (gradients * weights[:, None]).T @ gradients


def _probe_phantom(probes: SpinProbeEnsemble):
    from ..phantom import Phantom

    shape = (probes.n_spins,)
    return Phantom.from_positions(
        probes.positions_m,
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


def _matching_rows(observed_s, requested_s) -> np.ndarray:
    """Return the rows of sorted observed times that hold the requested times."""
    observed = np.asarray(observed_s, dtype=np.float64)
    requested = np.asarray(requested_s, dtype=np.float64)
    rows = np.minimum(np.searchsorted(observed, requested), max(observed.size - 1, 0))
    if observed.size == 0 or not np.array_equal(observed[rows], requested):
        raise RuntimeError("simulation did not observe every requested checkpoint")
    return rows


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
