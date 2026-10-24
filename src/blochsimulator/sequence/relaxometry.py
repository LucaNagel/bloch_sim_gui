"""Inversion-recovery T1 fitting for reconstructed IR image series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IR_T1_FIT_REFERENCE_DOI = "10.1002/mrm.22497"


@dataclass(frozen=True)
class InversionRecoveryFit:
    """Voxelwise fit of ``S(TI) = a - b exp(-TI / T1)``.

    ``inversion_efficiency`` is ``b / a - 1``: one for a perfect inversion
    with complete recovery, zero for saturation. ``polarity_flips`` counts
    the shortest-TI magnitude samples whose sign was restored before fitting.
    Voxels below ``minimum_signal_fraction`` are NaN.
    """

    t1_s: np.ndarray
    a: np.ndarray
    b: np.ndarray
    inversion_efficiency: np.ndarray
    rms_residual: np.ndarray
    polarity_flips: np.ndarray


def inversion_recovery_signal(
    inversion_times_s,
    t1_s,
    *,
    a: float = 1.0,
    inversion_efficiency: float = 1.0,
) -> np.ndarray:
    """Return ``a (1 - (1 + efficiency) exp(-TI / T1))``."""
    inversion_times = np.asarray(inversion_times_s, dtype=float)
    t1 = np.asarray(t1_s, dtype=float)
    return a * (1.0 - (1.0 + inversion_efficiency) * np.exp(-inversion_times / t1))


def _grid_fit(inversion_times, signal, t1_grid):
    """Least-squares ``a, b`` for every T1 candidate.

    ``signal`` is ``(..., n)``; ``t1_grid`` is ``(g,)`` or matches the
    leading axes of ``signal`` with a trailing ``g``. Returns cost, ``a`` and
    ``b`` per candidate from the normal equations without forming the model.
    """
    decay = np.exp(-inversion_times / t1_grid[..., None])
    count = inversion_times.size
    sum_decay = decay.sum(axis=-1)
    sum_decay_sq = np.square(decay).sum(axis=-1)
    sum_signal = signal.sum(axis=-1)[..., None]
    sum_signal_sq = np.square(signal).sum(axis=-1)[..., None]
    sum_decay_signal = np.einsum("...n,...gn->...g", signal, decay)
    determinant = count * sum_decay_sq - np.square(sum_decay)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (sum_decay_sq * sum_signal - sum_decay * sum_decay_signal) / determinant
        b = (sum_decay * sum_signal - count * sum_decay_signal) / determinant
    cost = (
        sum_signal_sq
        + count * np.square(a)
        + np.square(b) * sum_decay_sq
        - 2.0 * a * sum_signal
        + 2.0 * b * sum_decay_signal
        - 2.0 * a * b * sum_decay
    )
    return np.where(np.isfinite(cost), np.maximum(cost, 0.0), np.inf), a, b


def fit_inversion_recovery_t1(
    inversion_times_s,
    signal,
    *,
    magnitude: bool | None = None,
    t1_bounds_s: tuple[float, float] = (1e-3, 20.0),
    coarse_points: int = 160,
    fine_points: int = 81,
    minimum_signal_fraction: float = 0.02,
) -> InversionRecoveryFit:
    """Fit T1 voxelwise from an inversion-recovery series.

    ``signal`` has the inversion times on its last axis. Complex data is
    rotated by the phase of the longest-TI sample and fitted on its real
    part, which keeps the sign of inverted samples. Magnitude data uses
    polarity restoration: the fit tries flipping the first ``k`` sorted
    samples and keeps the ``k`` with the smallest residual (NLS-PR, Barral
    et al., MRM 2010). The fit is a log-spaced T1 grid search with two
    local refinements and closed-form ``a``/``b`` at every candidate.
    """
    inversion_times = np.asarray(inversion_times_s, dtype=float).reshape(-1)
    data = np.asarray(signal)
    if data.shape[-1:] != inversion_times.shape:
        raise ValueError("signal must have the inversion times on its last axis")
    if inversion_times.size < 3:
        raise ValueError("an inversion-recovery fit needs at least three TIs")
    if not np.all(np.isfinite(inversion_times)) or np.any(inversion_times < 0):
        raise ValueError("inversion_times_s must be non-negative and finite")
    low, high = (float(value) for value in t1_bounds_s)
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("t1_bounds_s must be increasing positive values")
    if int(coarse_points) < 3 or int(fine_points) < 3:
        raise ValueError("coarse_points and fine_points must be at least three")
    order = np.argsort(inversion_times, kind="stable")
    inversion_times = inversion_times[order]
    data = data[..., order]
    voxel_shape = data.shape[:-1]
    data = data.reshape(-1, inversion_times.size)
    if magnitude is None:
        magnitude = not np.iscomplexobj(data) and bool(np.all(data >= 0))
    if np.iscomplexobj(data):
        reference_phase = np.angle(data[:, -1])
        values = np.real(data * np.exp(-1j * reference_phase)[:, None])
        if magnitude:
            values = np.abs(values)
    else:
        values = np.asarray(data, dtype=float)
        if magnitude:
            values = np.abs(values)

    if magnitude:
        # The longest TI is taken as recovered, so at most n - 1 flips.
        flips = np.arange(inversion_times.size)
        signs = np.where(
            np.arange(inversion_times.size)[None, :] < flips[:, None], -1.0, 1.0
        )
        candidates = values[:, None, :] * signs[None, :, :]
    else:
        flips = np.zeros(1, dtype=int)
        candidates = values[:, None, :]

    coarse_grid = np.geomspace(low, high, int(coarse_points))
    best_flip = np.zeros(candidates.shape[0], dtype=int)
    best_grid = np.zeros(candidates.shape[0], dtype=int)
    chunk = max(1, 4_000_000 // (candidates.shape[1] * coarse_grid.size))
    for start in range(0, candidates.shape[0], chunk):
        cost, _, _ = _grid_fit(
            inversion_times, candidates[start : start + chunk], coarse_grid
        )
        flat_best = np.argmin(cost.reshape(cost.shape[0], -1), axis=1)
        flip_index, grid_index = np.unravel_index(flat_best, cost.shape[1:])
        best_flip[start : start + chunk] = flip_index
        best_grid[start : start + chunk] = grid_index

    rows = np.arange(candidates.shape[0])
    best = None
    # Near the null the coarse grid cannot separate neighbouring polarities,
    # so the refinement also tries one flip more and one less.
    for flip_shift in (0, -1, 1) if magnitude else (0,):
        flip = np.clip(best_flip + flip_shift, 0, flips.size - 1)
        selected = candidates[rows, flip]
        t1 = coarse_grid[best_grid]
        half_window = np.log(coarse_grid[1] / coarse_grid[0])
        # Two local passes, each spanning one spacing of the previous grid.
        for _ in range(2):
            offsets = np.linspace(-half_window, half_window, int(fine_points))
            fine_grid = np.clip(t1[:, None] * np.exp(offsets)[None, :], low, high)
            fine_cost, fine_a, fine_b = _grid_fit(inversion_times, selected, fine_grid)
            best_fine = np.argmin(fine_cost, axis=1)
            t1 = fine_grid[rows, best_fine]
            half_window = offsets[1] - offsets[0]
        trial = (
            fine_cost[rows, best_fine],
            t1,
            fine_a[rows, best_fine],
            fine_b[rows, best_fine],
            flip,
        )
        if best is None:
            best = trial
        else:
            better = trial[0] < best[0]
            best = tuple(np.where(better, new, old) for new, old in zip(trial, best))
    cost, t1, a, b, best_flip = best
    rms = np.sqrt(cost / inversion_times.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = b / a - 1.0

    peak = np.max(np.abs(values), axis=1)
    global_peak = float(np.max(peak)) if peak.size else 0.0
    masked = peak <= float(minimum_signal_fraction) * global_peak
    if global_peak == 0.0:
        masked[:] = True
    result = {}
    for name, array in (
        ("t1_s", t1),
        ("a", a),
        ("b", b),
        ("inversion_efficiency", efficiency),
        ("rms_residual", rms),
    ):
        array = np.asarray(array, dtype=float).copy()
        array[masked] = np.nan
        result[name] = array.reshape(voxel_shape)
    return InversionRecoveryFit(
        **result,
        polarity_flips=np.asarray(flips[best_flip], dtype=int).reshape(voxel_shape),
    )
