"""Coherence analysis and subvoxel-grid recommendations for gradient spoiling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from .spin_sampling import SpinSampling


@dataclass(frozen=True)
class SpoilerTrainAnalysis:
    """Retained transverse coherence for a sequence of accumulated phases."""

    phase_cycles_per_voxel: tuple[tuple[float, float, float], ...]
    continuous_coherence: tuple[float, ...]
    sampled_coherence: tuple[float, ...]
    alias_threshold: float = 0.05
    continuous_null_threshold: float = 0.01

    @property
    def n_observations(self) -> int:
        return len(self.sampled_coherence)

    @property
    def single_continuous_coherence(self) -> float:
        return self.continuous_coherence[0] if self.continuous_coherence else 1.0

    @property
    def single_sampled_coherence(self) -> float:
        return self.sampled_coherence[0] if self.sampled_coherence else 1.0

    @property
    def absolute_sampling_error(self) -> tuple[float, ...]:
        return tuple(
            abs(sampled - continuous)
            for sampled, continuous in zip(
                self.sampled_coherence, self.continuous_coherence
            )
        )

    @property
    def maximum_sampling_error(self) -> float:
        errors = self.absolute_sampling_error
        return max(errors, default=0.0)

    @property
    def worst_error_observation(self) -> int | None:
        errors = self.absolute_sampling_error
        return int(np.argmax(errors)) + 1 if errors else None

    @property
    def first_alias_observation(self) -> int | None:
        for index, (continuous, sampled) in enumerate(
            zip(self.continuous_coherence, self.sampled_coherence), start=1
        ):
            if (
                continuous < self.continuous_null_threshold
                and sampled > self.alias_threshold
            ):
                return index
        return None


@dataclass(frozen=True)
class SpinGridRecommendation:
    """Smallest tested spin grid meeting a train-wide quadrature tolerance."""

    counts_xyz: tuple[int, int, int]
    method: str
    spins_per_voxel: int
    maximum_sampling_error: float
    target_error: float
    meets_target: bool


def analyze_phase_cycle_train(
    phase_cycles_per_voxel,
    spin_sampling: SpinSampling,
    *,
    alias_threshold: float = 0.05,
    continuous_null_threshold: float = 0.01,
) -> SpoilerTrainAnalysis:
    """Compare continuous-voxel and sampled coherence for accumulated phases.

    Each row contains the X/Y/Z phase cycles accumulated across one complete
    phantom voxel at an observation time. Rows may come from repeated identical
    crushers or directly from differences between actual ADC moment origins.
    """
    phases = np.asarray(phase_cycles_per_voxel, dtype=np.float64)
    if phases.size == 0:
        phases = np.empty((0, 3), dtype=np.float64)
    if phases.ndim != 2 or phases.shape[1] != 3 or not np.all(np.isfinite(phases)):
        raise ValueError("phase_cycles_per_voxel must be a finite N x 3 array")
    if alias_threshold < 0.0 or continuous_null_threshold < 0.0:
        raise ValueError("coherence thresholds must be non-negative")

    continuous = np.prod(np.abs(np.sinc(phases)), axis=1)
    normalized_offsets, weights = spin_sampling.normalized_offsets_and_weights()
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("spin sampling weights must have a positive sum")
    normalized_weights = weights / weight_sum
    sampled = np.abs(
        np.exp(2j * np.pi * (phases @ normalized_offsets.T)) @ normalized_weights
    )
    return SpoilerTrainAnalysis(
        phase_cycles_per_voxel=tuple(
            tuple(float(value) for value in row) for row in phases
        ),
        continuous_coherence=tuple(float(value) for value in continuous),
        sampled_coherence=tuple(float(value) for value in sampled),
        alias_threshold=float(alias_threshold),
        continuous_null_threshold=float(continuous_null_threshold),
    )


def analyze_repeated_spoiler_train(
    cycles_per_voxel_xyz,
    spin_sampling: SpinSampling,
    excitation_count: int,
    **kwargs,
) -> SpoilerTrainAnalysis:
    """Analyze all accumulated spoiler orders in an RF excitation train."""
    cycles = _finite_xyz(cycles_per_voxel_xyz, "cycles_per_voxel_xyz")
    excitation_count = int(excitation_count)
    if excitation_count <= 0:
        raise ValueError("excitation_count must be positive")
    # Include one crusher even for a single-excitation sequence so the familiar
    # single-spoiler check remains available.
    orders = np.arange(1, max(2, excitation_count), dtype=np.float64)
    return analyze_phase_cycle_train(
        orders[:, None] * cycles[None, :], spin_sampling, **kwargs
    )


def analyze_adc_moment_train(
    moment_origins_cyc_per_m,
    voxel_basis_m,
    spin_sampling: SpinSampling,
    **kwargs,
) -> SpoilerTrainAnalysis:
    """Analyze coherence from actual ADC gradient-moment origins.

    The first ADC is the phase reference. Every following row is transformed to
    cycles across the phantom's three physical voxel basis vectors.
    """
    moments = np.asarray(moment_origins_cyc_per_m, dtype=np.float64)
    basis = np.asarray(voxel_basis_m, dtype=np.float64)
    if moments.ndim != 2 or moments.shape[1] != 3 or not np.all(np.isfinite(moments)):
        raise ValueError("moment_origins_cyc_per_m must be a finite N x 3 array")
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("voxel_basis_m must be a finite 3 x 3 matrix")
    phase_cycles = (moments[1:] - moments[:1]) @ basis
    return analyze_phase_cycle_train(phase_cycles, spin_sampling, **kwargs)


@lru_cache(maxsize=256)
def recommend_spin_grid(
    cycles_per_voxel_xyz: tuple[float, float, float],
    excitation_count: int,
    minimum_counts_xyz: tuple[int, int, int] = (1, 1, 1),
    method: str = "midpoint",
    target_error: float = 0.01,
    maximum_spins: int = 512,
    maximum_axis_count: int = 32,
) -> SpinGridRecommendation:
    """Find the least expensive grid whose whole train matches a uniform voxel.

    Candidate grids are ordered by total spin count and then by their longest
    axis. The error criterion is sampled versus continuous coherence, so a weak
    physical spoiler is not mistaken for a sampling failure.
    """
    cycles = _finite_xyz(cycles_per_voxel_xyz, "cycles_per_voxel_xyz")
    excitation_count = int(excitation_count)
    minimum = tuple(int(value) for value in minimum_counts_xyz)
    method = str(method).strip().lower()
    if excitation_count <= 0:
        raise ValueError("excitation_count must be positive")
    if len(minimum) != 3 or any(value <= 0 for value in minimum):
        raise ValueError("minimum_counts_xyz must contain three positive integers")
    if method not in {"midpoint", "stratified"}:
        raise ValueError("method must be 'midpoint' or 'stratified'")
    if not np.isfinite(target_error) or target_error <= 0.0:
        raise ValueError("target_error must be positive and finite")
    maximum_spins = int(maximum_spins)
    maximum_axis_count = int(maximum_axis_count)
    if maximum_spins < int(np.prod(minimum)):
        raise ValueError("maximum_spins is smaller than the minimum grid")
    if maximum_axis_count < max(minimum):
        raise ValueError("maximum_axis_count is smaller than a minimum axis count")

    axis_ranges = []
    for cycle, lower in zip(cycles, minimum):
        upper = maximum_axis_count if abs(cycle) > 1e-12 else lower
        axis_ranges.append(range(lower, upper + 1))
    candidates = [
        tuple(int(value) for value in counts)
        for counts in product(*axis_ranges)
        if int(np.prod(counts)) <= maximum_spins
    ]
    candidates.sort(key=lambda counts: (int(np.prod(counts)), max(counts), counts))

    best_counts = candidates[0]
    best_error = float("inf")
    for counts in candidates:
        error = _repeated_train_sampling_error(cycles, counts, method, excitation_count)
        if error < best_error:
            best_counts, best_error = counts, error
        if error <= target_error:
            return SpinGridRecommendation(
                counts_xyz=counts,
                method=method,
                spins_per_voxel=int(np.prod(counts)),
                maximum_sampling_error=float(error),
                target_error=float(target_error),
                meets_target=True,
            )
    return SpinGridRecommendation(
        counts_xyz=best_counts,
        method=method,
        spins_per_voxel=int(np.prod(best_counts)),
        maximum_sampling_error=float(best_error),
        target_error=float(target_error),
        meets_target=False,
    )


def recommend_spin_grid_for_phase_train(
    phase_cycles_per_voxel,
    minimum_counts_xyz: tuple[int, int, int] = (1, 1, 1),
    method: str = "midpoint",
    target_error: float = 0.01,
    maximum_spins: int = 512,
    maximum_axis_count: int = 32,
) -> SpinGridRecommendation:
    """Recommend a grid for arbitrary phases, such as actual ADC moments."""
    phases = np.asarray(phase_cycles_per_voxel, dtype=np.float64)
    if phases.ndim != 2 or phases.shape[1] != 3 or not np.all(np.isfinite(phases)):
        raise ValueError("phase_cycles_per_voxel must be a finite N x 3 array")
    minimum = tuple(int(value) for value in minimum_counts_xyz)
    method = str(method).strip().lower()
    if len(minimum) != 3 or any(value <= 0 for value in minimum):
        raise ValueError("minimum_counts_xyz must contain three positive integers")
    if method not in {"midpoint", "stratified"}:
        raise ValueError("method must be 'midpoint' or 'stratified'")
    if not np.isfinite(target_error) or target_error <= 0.0:
        raise ValueError("target_error must be positive and finite")
    maximum_spins = int(maximum_spins)
    maximum_axis_count = int(maximum_axis_count)
    if maximum_spins < int(np.prod(minimum)):
        raise ValueError("maximum_spins is smaller than the minimum grid")
    if maximum_axis_count < max(minimum):
        raise ValueError("maximum_axis_count is smaller than a minimum axis count")

    active_axes = np.any(np.abs(phases) > 1e-12, axis=0)
    axis_ranges = [
        range(lower, maximum_axis_count + 1) if active else range(lower, lower + 1)
        for lower, active in zip(minimum, active_axes)
    ]
    candidates = [
        tuple(int(value) for value in counts)
        for counts in product(*axis_ranges)
        if int(np.prod(counts)) <= maximum_spins
    ]
    candidates.sort(key=lambda counts: (int(np.prod(counts)), max(counts), counts))
    best_counts = candidates[0]
    best_error = float("inf")
    for counts in candidates:
        error = _phase_train_sampling_error(phases, counts, method)
        if error < best_error:
            best_counts, best_error = counts, error
        if error <= target_error:
            return SpinGridRecommendation(
                counts_xyz=counts,
                method=method,
                spins_per_voxel=int(np.prod(counts)),
                maximum_sampling_error=float(error),
                target_error=float(target_error),
                meets_target=True,
            )
    return SpinGridRecommendation(
        counts_xyz=best_counts,
        method=method,
        spins_per_voxel=int(np.prod(best_counts)),
        maximum_sampling_error=float(best_error),
        target_error=float(target_error),
        meets_target=False,
    )


def _finite_xyz(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return array


def _repeated_train_sampling_error(
    cycles: np.ndarray,
    counts: tuple[int, int, int],
    method: str,
    excitation_count: int,
) -> float:
    """Evaluate a candidate cheaply, retaining the public calculation exactly."""
    if method != "midpoint":
        sampling = SpinSampling(counts, method=method)
        return analyze_repeated_spoiler_train(
            cycles, sampling, excitation_count
        ).maximum_sampling_error

    orders = np.arange(1, max(2, excitation_count), dtype=np.float64)
    phases = orders[:, None] * cycles[None, :]
    continuous = np.prod(np.abs(np.sinc(phases)), axis=1)
    sampled = np.ones(orders.size, dtype=np.float64)
    for axis_cycles, count in zip(cycles, counts):
        offsets = (np.arange(count, dtype=np.float64) + 0.5) / count - 0.5
        sampled *= np.abs(
            np.mean(
                np.exp(2j * np.pi * np.outer(orders * axis_cycles, offsets)),
                axis=1,
            )
        )
    return float(np.max(np.abs(sampled - continuous), initial=0.0))


def _phase_train_sampling_error(
    phases: np.ndarray,
    counts: tuple[int, int, int],
    method: str,
) -> float:
    if phases.shape[0] == 0:
        return 0.0
    if method != "midpoint":
        return analyze_phase_cycle_train(
            phases, SpinSampling(counts, method=method)
        ).maximum_sampling_error
    continuous = np.prod(np.abs(np.sinc(phases)), axis=1)
    sampled = np.ones(phases.shape[0], dtype=np.float64)
    for axis_index, count in enumerate(counts):
        offsets = (np.arange(count, dtype=np.float64) + 0.5) / count - 0.5
        sampled *= np.abs(
            np.mean(
                np.exp(2j * np.pi * np.outer(phases[:, axis_index], offsets)),
                axis=1,
            )
        )
    return float(np.max(np.abs(sampled - continuous), initial=0.0))
