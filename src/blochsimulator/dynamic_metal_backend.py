"""Experimental Metal support for dynamic two-pool simulation.

The low-level functions remain numerical probes. The GUI-facing hybrid wrapper
adds independent Float64 CPU checks and returns a fresh exact CPU result unless
the mixed-precision accuracy gate passes.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import resources
import platform
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .dynamic_phantom import (
    DynamicSpectralPhantom,
    _advance_longitudinal_kinetics,
    _prepare_rf_rotation,
    kinetic_preroll_start_s,
)
from .sequence import SequenceCompiler
from .sequence.spin_sampling import coerce_spin_sampling, phantom_voxel_basis_m


FLOAT32_PRECISION_STRATEGY = (
    "float32 state retained in one GPU thread; Float64-prepared RF coefficients "
    "rounded once; fast math disabled; per-spin ADC values reduced on CPU in "
    "deterministic Float64 pairwise order"
)
DOUBLE_SINGLE_PRECISION_STRATEGY = (
    "double-single two-Float32 pool state retained in one GPU thread; "
    "Float64-prepared RF coefficients rounded once; fast math disabled; "
    "per-spin ADC values reduced on CPU in deterministic Float64 pairwise order"
)
HYBRID_PRECISION_STRATEGY = (
    "GPU prediction for the complete subvoxel grid; disjoint Float64 CPU "
    "calibration and validation offsets in every active voxel; species-wise "
    "complex gain correction and independent held-out accuracy gate"
)


def metal_capability() -> dict:
    """Report probe availability without importing Metal on CPU-only systems."""
    result = {
        "available": False,
        "supported_platform": (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        ),
        "device_name": None,
        "apple_gpu_family": 0,
        "recommended_max_working_set_bytes": 0,
        "reason": None,
        "probe_extension_available": False,
    }
    if platform.system() != "Darwin":
        result["reason"] = "Metal is available only on macOS"
        return result
    if platform.machine() != "arm64":
        result["reason"] = "the experimental backend targets Apple Silicon arm64"
        return result
    try:
        from . import _dynamic_metal_probe
    except ImportError as error:
        result["reason"] = f"the private Metal probe extension is unavailable: {error}"
        return result
    result["probe_extension_available"] = True
    native = dict(_dynamic_metal_probe.capability())
    result.update(native)
    return result


def _pairwise_sum_complex128(values: np.ndarray) -> np.ndarray:
    """Reduce the leading axis using a fixed binary tree in complex128."""
    work = np.asarray(values, dtype=np.complex128)
    if work.ndim < 1 or work.shape[0] == 0:
        return np.zeros(work.shape[1:], dtype=np.complex128)
    work = np.ascontiguousarray(work)
    while work.shape[0] > 1:
        pair_count = work.shape[0] // 2
        reduced = work[0 : 2 * pair_count : 2] + work[1 : 2 * pair_count : 2]
        if work.shape[0] % 2:
            reduced = np.concatenate((reduced, work[-1:]), axis=0)
        work = np.ascontiguousarray(reduced)
    return work[0]


class _PairwiseComplexAccumulator:
    """Stream arrays through the same fixed binary-tree addition topology."""

    def __init__(self):
        self.levels = []

    def add(self, value) -> None:
        carry = np.ascontiguousarray(value, dtype=np.complex128)
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            carry = self.levels[level] + carry
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(carry)
        else:
            self.levels[level] = carry

    def finish(self) -> np.ndarray:
        result = None
        for value in reversed(self.levels):
            if value is None:
                continue
            result = value if result is None else result + value
        if result is None:
            raise ValueError("cannot finish an empty pairwise accumulator")
        return np.ascontiguousarray(result, dtype=np.complex128)


def _relative_nrmse(reference, candidate) -> float:
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    difference_norm = float(np.linalg.norm((candidate - reference).reshape(-1)))
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    return difference_norm / reference_norm if reference_norm else difference_norm


def _farthest_subvoxel_order(sampling) -> np.ndarray:
    """Return a deterministic spatially dispersed ordering of grid points."""
    offsets, _ = sampling.normalized_offsets_and_weights()
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape[0] != sampling.grid_spins_per_voxel:
        raise ValueError("hybrid sampling requires the complete subvoxel grid")
    selected = np.zeros(offsets.shape[0], dtype=bool)
    order = np.empty(offsets.shape[0], dtype=np.int64)
    next_index = int(np.argmin(np.sum(np.square(offsets), axis=1)))
    minimum_distance = np.full(offsets.shape[0], np.inf, dtype=np.float64)
    for cursor in range(offsets.shape[0]):
        order[cursor] = next_index
        selected[next_index] = True
        distance = np.sum(np.square(offsets - offsets[next_index]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
        if cursor + 1 < offsets.shape[0]:
            next_index = int(np.argmax(minimum_distance))
    return order


def _hybrid_subvoxel_partition(
    sampling,
    calibration_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose disjoint, spatially spread CPU offsets for every active voxel."""
    calibration_fraction = float(calibration_fraction)
    validation_fraction = float(validation_fraction)
    if (
        not np.isfinite(calibration_fraction)
        or not np.isfinite(validation_fraction)
        or calibration_fraction <= 0.0
        or validation_fraction <= 0.0
        or calibration_fraction + validation_fraction >= 1.0
    ):
        raise ValueError(
            "hybrid calibration and validation fractions must be positive and "
            "sum to less than one"
        )
    grid_count = sampling.grid_spins_per_voxel
    if sampling.selected_indices is not None or grid_count < 3:
        raise ValueError(
            "hybrid correction requires a complete grid with at least three "
            "subvoxel spins per voxel"
        )
    calibration_count = max(1, int(np.ceil(grid_count * calibration_fraction)))
    validation_count = max(1, int(np.ceil(grid_count * validation_fraction)))
    while calibration_count + validation_count >= grid_count:
        if calibration_count >= validation_count and calibration_count > 1:
            calibration_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            break

    offsets, _ = sampling.normalized_offsets_and_weights()
    offsets = np.asarray(offsets, dtype=np.float64)
    order = _farthest_subvoxel_order(sampling)
    used = np.zeros(grid_count, dtype=bool)
    symmetric_groups = []
    for index in order:
        index = int(index)
        if used[index]:
            continue
        partner = int(np.argmin(np.sum(np.square(offsets + offsets[index]), axis=1)))
        group = tuple(sorted({index, partner}))
        used[list(group)] = True
        symmetric_groups.append(group)
    if len(symmetric_groups) < 3:
        raise ValueError(
            "hybrid correction requires at least three centre-symmetric "
            "subvoxel groups"
        )

    calibration = []
    validation = []
    # Keep at least one complete symmetry group as GPU-only work.
    for group in symmetric_groups[:-1]:
        calibration_progress = len(calibration) / calibration_count
        validation_progress = len(validation) / validation_count
        if len(calibration) < calibration_count and (
            len(validation) >= validation_count
            or calibration_progress <= validation_progress
        ):
            calibration.extend(group)
        elif len(validation) < validation_count:
            validation.extend(group)
        if (
            len(calibration) >= calibration_count
            and len(validation) >= validation_count
        ):
            break
    if not calibration or not validation:
        raise ValueError("hybrid correction could not form two CPU reference groups")
    return (
        np.sort(np.asarray(calibration, dtype=np.int64)),
        np.sort(np.asarray(validation, dtype=np.int64)),
    )


def _hybrid_signal_correction(
    gpu_full,
    gpu_calibration,
    cpu_calibration,
    gpu_validation,
    cpu_validation,
    *,
    calibration_weight_fraction: float,
    validation_weight_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Apply and independently assess a complex-gain subvoxel correction."""
    gpu_full = np.asarray(gpu_full, dtype=np.complex128)
    gpu_calibration = np.asarray(gpu_calibration, dtype=np.complex128)
    cpu_calibration = np.asarray(cpu_calibration, dtype=np.complex128)
    gpu_validation = np.asarray(gpu_validation, dtype=np.complex128)
    cpu_validation = np.asarray(cpu_validation, dtype=np.complex128)
    if not (
        gpu_full.shape
        == gpu_calibration.shape
        == cpu_calibration.shape
        == gpu_validation.shape
        == cpu_validation.shape
    ):
        raise ValueError("hybrid CPU and GPU species signals must have equal shapes")
    calibration_weight_fraction = float(calibration_weight_fraction)
    validation_weight_fraction = float(validation_weight_fraction)
    if calibration_weight_fraction <= 0.0 or validation_weight_fraction <= 0.0:
        raise ValueError("hybrid sample weight fractions must be positive")

    # Phase drift acts approximately as a small complex gain. Applying the
    # sampled gain preserves physical cancellation between subvoxel signals;
    # extrapolating an absolute error does not. Near-zero calibration samples
    # are deliberately left uncorrected and are still exercised by validation.
    gain = np.ones_like(gpu_calibration, dtype=np.complex128)
    reliable = np.zeros_like(gpu_calibration, dtype=bool)
    for pool in range(gpu_calibration.shape[0]):
        peak = float(np.max(np.abs(gpu_calibration[pool]), initial=0.0))
        threshold = max(np.finfo(np.float64).eps, peak * 1.0e-8)
        reliable[pool] = np.abs(gpu_calibration[pool]) > threshold
        gain[pool, reliable[pool]] = (
            cpu_calibration[pool, reliable[pool]]
            / gpu_calibration[pool, reliable[pool]]
        )
    corrected = gpu_full * gain
    predicted_validation = gpu_validation * gain
    validation_residual = predicted_validation - cpu_validation

    estimated_species_nrmse = []
    local_species_nrmse = []
    for pool in range(gpu_full.shape[0]):
        estimated_species_nrmse.append(
            _relative_nrmse(
                corrected[pool],
                corrected[pool]
                + validation_residual[pool] / validation_weight_fraction,
            )
        )
        local_species_nrmse.append(
            _relative_nrmse(cpu_validation[pool], predicted_validation[pool])
        )
    corrected_total = corrected.sum(axis=0)
    validation_total = cpu_validation.sum(axis=0)
    predicted_total = predicted_validation.sum(axis=0)
    estimated_total_nrmse = _relative_nrmse(
        corrected_total,
        corrected_total + validation_residual.sum(axis=0) / validation_weight_fraction,
    )
    metrics = {
        "estimated_full_total_signal_nrmse": estimated_total_nrmse,
        "estimated_full_species_signal_nrmse": estimated_species_nrmse,
        "held_out_total_signal_nrmse": _relative_nrmse(
            validation_total, predicted_total
        ),
        "held_out_species_signal_nrmse": local_species_nrmse,
        "correction_norm_relative_to_gpu_signal": _relative_nrmse(gpu_full, corrected),
        "calibration_gain_unreliable_sample_count": int(np.count_nonzero(~reliable)),
        "calibration_gain_sample_count": int(reliable.size),
    }
    return corrected, metrics


def _build_interval_plan(compiled, phantom) -> np.ndarray:
    """Prepare compact Float64 interval values, then round once to Float32."""
    interval_count = compiled.n_intervals
    plan64 = np.zeros((interval_count, 16), dtype=np.float64)
    starts = np.concatenate(([0.0], compiled.interval_end_s[:-1]))
    mids = starts + compiled.dt_s / 2.0
    plan64[:, 0:3] = compiled.gradient_hz_per_m
    plan64[:, 3] = compiled.dt_s
    plan64[:, 6] = 1.0
    for index, (rf_hz, duration) in enumerate(zip(compiled.rf_hz, compiled.dt_s)):
        prepared = _prepare_rf_rotation(rf_hz, duration)
        if prepared is None:
            continue
        plan64[index, 4] = prepared[0]
        plan64[index, 5] = prepared[1]
        plan64[index, 6] = prepared[2]
        plan64[index, 7] = prepared[3]
        plan64[index, 8] = prepared[4]

    inflow_curve = phantom.inflow_curve_on_sequence_timeline
    polarization_curve = phantom.inflow_polarization_curve_on_sequence_timeline
    if inflow_curve is not None:
        for index, (start, mid, end) in enumerate(
            zip(starts, mids, compiled.interval_end_s)
        ):
            rate_start, rate_mid = inflow_curve.interval_values(start, mid)
            _, rate_end = inflow_curve.interval_values(mid, end)
            plan64[index, 9:12] = (rate_start, rate_mid, rate_end)
            if polarization_curve is None:
                plan64[index, 12:15] = 1.0
            else:
                pol_start, pol_mid = polarization_curve.interval_values(start, mid)
                _, pol_end = polarization_curve.interval_values(mid, end)
                plan64[index, 12:15] = (pol_start, pol_mid, pol_end)
    plan64[:, 15] = (mids >= phantom.conversion_start_on_sequence_timeline_s).astype(
        np.float64
    )
    return np.ascontiguousarray(plan64, dtype=np.float32)


def _prepare_active_state(
    phantom,
    active,
    sampling,
    real_positions,
    *,
    field_strength_t=None,
    nucleus=None,
):
    spins_per_voxel = sampling.spins_per_voxel
    state = (
        np.asarray(phantom.initial_magnetization, dtype=np.float64)
        .reshape(2, phantom.nvoxels, 3)[:, active]
        .copy()
    )
    if sampling.enabled:
        state = np.repeat(state, spins_per_voxel, axis=1)
    concentration_state = None
    if phantom.initial_spin_density_maps is not None:
        density = np.asarray(phantom.initial_spin_density, dtype=np.float64).reshape(
            2, phantom.nvoxels
        )[:, active]
        if sampling.enabled:
            density = np.repeat(density, spins_per_voxel, axis=1)
        concentration_state = np.zeros_like(state)
        concentration_state[:, :, 2] = density

    inflow_curve = phantom.inflow_curve_on_sequence_timeline
    polarization_curve = phantom.inflow_polarization_curve_on_sequence_timeline
    delivery = None
    if phantom.pyruvate_inflow is not None:
        delivery = np.asarray(
            phantom.pyruvate_inflow.delivery_map.ravel()[active], dtype=np.float64
        )
        if sampling.enabled:
            delivery = np.repeat(delivery, spins_per_voxel)
    preroll_start = kinetic_preroll_start_s(
        (
            None
            if phantom.pyruvate_inflow is None
            else phantom.pyruvate_inflow.rate_curve_s_inv
        ),
        phantom.conversion_start_s,
        phantom.kinetics_time_offset_s,
    )
    kpl = np.asarray(phantom.kpl_map_s_inv.ravel()[active], dtype=np.float64)
    if sampling.enabled:
        kpl = np.repeat(kpl, spins_per_voxel)
    if preroll_start < 0.0:
        _advance_longitudinal_kinetics(
            state,
            kpl,
            np.asarray([1.0 / pool.t1 for pool in phantom.pools]),
            preroll_start,
            0.0,
            inflow_curve=inflow_curve,
            inflow_delivery=delivery,
            conversion_start_s=phantom.conversion_start_on_sequence_timeline_s,
            inflow_polarization_curve=polarization_curve,
            concentration_state=concentration_state,
            equilibrium_polarization=phantom.equilibrium_polarization,
        )

    spin_count = state.shape[1]
    initial = np.zeros((spin_count, 2, 4), dtype=np.float32)
    initial[:, :, :3] = np.transpose(state, (1, 0, 2)).astype(np.float32)
    if concentration_state is not None:
        initial[:, :, 3] = np.transpose(concentration_state[:, :, 2], (1, 0))

    spatial = np.zeros((spin_count, 4), dtype=np.float32)
    spatial[:, :3] = np.asarray(real_positions, dtype=np.float32)
    field = (
        phantom.field_strength if field_strength_t is None else float(field_strength_t)
    )
    effective_nucleus = phantom.nucleus if nucleus is None else str(nucleus)
    spatial[:, 3] = np.asarray(
        phantom.b0_offset_hz(field, effective_nucleus).ravel()[active],
        dtype=np.float32,
    ).repeat(spins_per_voxel if sampling.enabled else 1)
    kinetic = np.zeros((spin_count, 4), dtype=np.float32)
    kinetic[:, 0] = kpl.astype(np.float32)
    if delivery is not None:
        kinetic[:, 1] = delivery.astype(np.float32)
    return initial, spatial, kinetic, concentration_state is not None, preroll_start


def run_metal_precision_probe(
    program,
    phantom: DynamicSpectralPhantom,
    *,
    simulation_timestep_s: float,
    signal_weighting: str = "voxel",
    spin_sampling=None,
    spoiler_mode: str = "ideal",
    precision_strategy: str = "float32",
    field_strength_t=None,
    nucleus=None,
    memory_budget_bytes: int | None = None,
    spin_chunk_size: int | str | None = None,
    capture_spin_indices=(),
    capture_spin_groups=(),
    cancel_callback=None,
) -> dict:
    """Run the isolated full-prefix Metal probe on a supported object.

    ``spin_chunk_size="auto"`` bounds the per-spin ADC output used by this
    feasibility implementation. ``capture_spin_indices`` retains only the
    requested individual trajectories for diagnostics. ``capture_spin_groups``
    reduces each requested group immediately and is the scalable hybrid path;
    all other trajectories are also reduced in deterministic Float64 order.
    """
    capability = metal_capability()
    if not capability["available"]:
        raise RuntimeError(capability["reason"] or "Metal is unavailable")
    if not isinstance(phantom, DynamicSpectralPhantom):
        raise TypeError("the Metal precision probe requires a dynamic phantom")
    if phantom.dynamic_b0 is not None:
        raise ValueError("the Metal precision probe does not support dynamic B0")
    tx_map = getattr(phantom, "tx_sensitivity_map", None)
    if tx_map is not None and not np.all(np.asarray(tx_map) == (1.0 + 0.0j)):
        raise ValueError("the Metal precision probe requires uniform transmit")
    rx_maps = getattr(phantom, "rx_sensitivity_maps", None)
    if rx_maps is not None:
        rx_maps = np.asarray(rx_maps)
        if rx_maps.shape[0] != 1 or not np.all(rx_maps[0] == (1.0 + 0.0j)):
            raise ValueError(
                "the Metal precision probe requires one uniform receive coil"
            )
    if signal_weighting not in {"voxel", "voxel_volume"}:
        raise ValueError("signal_weighting must be 'voxel' or 'voxel_volume'")
    if precision_strategy not in {"float32", "double_single"}:
        raise ValueError("precision_strategy must be 'float32' or 'double_single'")
    spoiler_mode = str(spoiler_mode).strip().lower()
    if spoiler_mode not in {"ideal", "gradient"}:
        raise ValueError("spoiler_mode must be 'ideal' or 'gradient'")

    sampling = coerce_spin_sampling(spin_sampling)
    sampling.validate_phantom_dimensions(phantom.ndim)
    compiled = SequenceCompiler().compile(
        program,
        extra_boundaries_s=phantom.dynamic_breakpoints_s(program.duration_s),
        simulation_timestep_s=simulation_timestep_s,
    )
    active = np.flatnonzero(phantom.mask.ravel())
    if active.size == 0:
        raise ValueError("dynamic phantom has no active spins")
    offsets, weights = sampling.offsets_m(phantom_voxel_basis_m(phantom))
    positions = np.asarray(phantom.positions[active], dtype=np.float64)
    if sampling.enabled:
        positions = np.repeat(positions, sampling.spins_per_voxel, axis=0) + np.tile(
            offsets, (active.size, 1)
        )
    field = (
        phantom.field_strength if field_strength_t is None else float(field_strength_t)
    )
    effective_nucleus = phantom.nucleus if nucleus is None else str(nucleus)
    initial, spatial, kinetic, track_concentration, preroll_start = (
        _prepare_active_state(
            phantom,
            active,
            sampling,
            positions,
            field_strength_t=field,
            nucleus=effective_nucleus,
        )
    )
    kinetic[:, 2] = np.tile(weights, active.size).astype(np.float32)
    interval_plan = _build_interval_plan(compiled, phantom)
    rf_active = np.asarray(compiled.rf_hz != 0.0, dtype=bool)
    rf_block_count = int(
        np.count_nonzero(rf_active & ~np.concatenate(([False], rf_active[:-1])))
    )
    adc_states = np.ascontiguousarray(compiled.adc_state_indices, dtype=np.uint32)
    adc_demod = np.ascontiguousarray(
        np.column_stack(
            (compiled.adc_demodulation.real, compiled.adc_demodulation.imag)
        ),
        dtype=np.float32,
    )
    crusher_states = np.ascontiguousarray(
        (
            compiled.transverse_crush_state_indices
            if spoiler_mode == "ideal"
            else np.zeros(0, dtype=np.uint32)
        ),
        dtype=np.uint32,
    )
    pool_offsets = [
        pool.get_frequency_offset(field, effective_nucleus) for pool in phantom.pools
    ]
    signal_scale = (
        phantom.voxel_volume_m3 if signal_weighting == "voxel_volume" else 1.0
    )
    physical_constants = np.ascontiguousarray(
        (
            pool_offsets[0],
            pool_offsets[1],
            1.0 / phantom.pools[0].t1,
            1.0 / phantom.pools[1].t1,
            phantom.pools[0].t2,
            phantom.pools[1].t2,
            phantom.equilibrium_polarization,
            signal_scale,
            float(track_concentration),
        ),
        dtype=np.float32,
    )
    spin_count = int(initial.shape[0])
    captured_indices = np.asarray(tuple(capture_spin_indices), dtype=np.int64)
    if captured_indices.ndim != 1:
        raise ValueError("capture_spin_indices must be one-dimensional")
    if captured_indices.size:
        if np.any(captured_indices < 0) or np.any(captured_indices >= spin_count):
            raise ValueError("captured Metal spin index is outside the active grid")
        if np.unique(captured_indices).size != captured_indices.size:
            raise ValueError("captured Metal spin indices must be unique")
        captured_indices = np.sort(captured_indices)
    captured_groups = []
    captured_group_union = []
    for values in tuple(capture_spin_groups):
        group = np.asarray(tuple(values), dtype=np.int64)
        if group.ndim != 1 or not group.size:
            raise ValueError("each captured Metal spin group must be non-empty")
        if np.any(group < 0) or np.any(group >= spin_count):
            raise ValueError("captured Metal spin group is outside the active grid")
        if np.unique(group).size != group.size:
            raise ValueError("captured Metal spin group indices must be unique")
        group = np.sort(group)
        captured_groups.append(group)
        captured_group_union.append(group)
    if captured_group_union:
        combined_groups = np.concatenate(captured_group_union)
        if np.unique(combined_groups).size != combined_groups.size:
            raise ValueError("captured Metal spin groups must be disjoint")
    state_bytes = int(initial.nbytes + spatial.nbytes + kinetic.nbytes + initial.nbytes)
    plan_bytes = int(
        interval_plan.nbytes
        + adc_states.nbytes
        + adc_demod.nbytes
        + crusher_states.nbytes
    )
    state_bytes_per_spin = int(
        initial.strides[0]
        + spatial.strides[0]
        + kinetic.strides[0]
        + initial.strides[0]
    )
    reduction_bytes_per_spin = int(2 * adc_states.size * 2 * 4)
    working_bytes_per_spin = state_bytes_per_spin + reduction_bytes_per_spin
    recommended = int(capability["recommended_max_working_set_bytes"])
    automatic_budget = min(1024**3, max(64 * 1024**2, recommended // 10))
    effective_budget = (
        automatic_budget if memory_budget_bytes is None else int(memory_budget_bytes)
    )
    if effective_budget <= plan_bytes or working_bytes_per_spin <= 0:
        raise MemoryError(
            "Metal precision probe memory limit exceeded: estimated working set "
            f"cannot fit one spin plus the immutable plan inside the "
            f"{effective_budget / 1024**2:.2f} MiB probe budget. Use a smaller "
            "synthetic phantom or a shorter acquisition prefix."
        )
    maximum_budget_chunk = max(
        0, (effective_budget - plan_bytes) // working_bytes_per_spin
    )
    if spin_chunk_size == "auto":
        effective_chunk_size = min(spin_count, maximum_budget_chunk)
    elif spin_chunk_size is None:
        effective_chunk_size = spin_count
    else:
        if isinstance(spin_chunk_size, (bool, np.bool_)):
            raise ValueError("spin_chunk_size must be a positive integer or 'auto'")
        effective_chunk_size = int(spin_chunk_size)
        if effective_chunk_size != spin_chunk_size or effective_chunk_size <= 0:
            raise ValueError("spin_chunk_size must be a positive integer or 'auto'")
        effective_chunk_size = min(spin_count, effective_chunk_size)
    if effective_chunk_size <= 0:
        raise MemoryError(
            "Metal precision probe memory limit exceeded: the plan leaves no "
            "room for one spin's ADC trajectory. Use fewer ADC samples."
        )
    reduction_bytes = int(effective_chunk_size * reduction_bytes_per_spin)
    chunk_state_bytes = int(effective_chunk_size * state_bytes_per_spin)
    peak_estimate = chunk_state_bytes + plan_bytes + reduction_bytes
    if peak_estimate > effective_budget:
        raise MemoryError(
            "Metal precision probe memory limit exceeded: estimated working set "
            f"{peak_estimate / 1024**2:.2f} MiB is above the "
            f"{effective_budget / 1024**2:.2f} MiB probe budget. Use "
            "spin_chunk_size='auto', a smaller object, or a shorter acquisition."
        )
    source = (
        resources.files("blochsimulator")
        .joinpath("metal/dynamic_bloch_precision.metal")
        .read_bytes()
    )
    from . import _dynamic_metal_probe

    species_signal_accumulator = _PairwiseComplexAccumulator()
    final_active = np.zeros((2, active.size, 3), dtype=np.float64)
    captured_signal = np.empty(
        (captured_indices.size, 2, adc_states.size), dtype=np.complex128
    )
    captured_final = np.empty((captured_indices.size, 2, 3), dtype=np.float32)
    captured_group_signal_accumulators = [
        _PairwiseComplexAccumulator() for _ in captured_groups
    ]
    captured_group_final_active = np.zeros(
        (len(captured_groups), 2, active.size, 3), dtype=np.float64
    )
    compile_seconds = 0.0
    simulation_seconds = 0.0
    chunk_count = 0
    start = perf_counter()
    for chunk_start in range(0, spin_count, effective_chunk_size):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Simulation cancelled")
        chunk_stop = min(chunk_start + effective_chunk_size, spin_count)
        native = _dynamic_metal_probe.run_probe(
            source,
            interval_plan,
            adc_states,
            adc_demod,
            crusher_states,
            np.ascontiguousarray(initial[chunk_start:chunk_stop]),
            np.ascontiguousarray(spatial[chunk_start:chunk_stop]),
            np.ascontiguousarray(kinetic[chunk_start:chunk_stop]),
            physical_constants,
            1 if precision_strategy == "double_single" else 0,
        )
        chunk_count += 1
        compile_seconds += float(native["pipeline_compile_seconds"])
        simulation_seconds += float(native["simulation_seconds"])
        raw_signal = np.asarray(native["per_spin_species_signal"])
        complex_signal = raw_signal[..., 0].astype(np.float64) + 1j * raw_signal[
            ..., 1
        ].astype(np.float64)
        species_signal_accumulator.add(_pairwise_sum_complex128(complex_signal))

        final_spin = np.asarray(native["final_pool_state"])[..., :3]
        global_indices = np.arange(chunk_start, chunk_stop, dtype=np.int64)
        parent_indices = global_indices // sampling.spins_per_voxel
        subvoxel_indices = global_indices % sampling.spins_per_voxel
        spin_weights = np.asarray(weights, dtype=np.float64)[subvoxel_indices]
        for pool in range(2):
            np.add.at(
                final_active[pool],
                parent_indices,
                final_spin[:, pool] * spin_weights[:, None],
            )

        capture_start = int(np.searchsorted(captured_indices, chunk_start))
        capture_stop = int(np.searchsorted(captured_indices, chunk_stop))
        if capture_stop > capture_start:
            destination = slice(capture_start, capture_stop)
            local = captured_indices[destination] - chunk_start
            captured_signal[destination] = complex_signal[local]
            captured_final[destination] = final_spin[local]
        for group_index, group in enumerate(captured_groups):
            group_start = int(np.searchsorted(group, chunk_start))
            group_stop = int(np.searchsorted(group, chunk_stop))
            if group_stop <= group_start:
                continue
            group_global = group[group_start:group_stop]
            group_local = group_global - chunk_start
            captured_group_signal_accumulators[group_index].add(
                _pairwise_sum_complex128(complex_signal[group_local])
            )
            group_parent = group_global // sampling.spins_per_voxel
            group_subvoxel = group_global % sampling.spins_per_voxel
            group_weights = np.asarray(weights, dtype=np.float64)[group_subvoxel]
            for pool in range(2):
                np.add.at(
                    captured_group_final_active[group_index, pool],
                    group_parent,
                    final_spin[group_local, pool] * group_weights[:, None],
                )
    wall_seconds = perf_counter() - start
    species_signal = species_signal_accumulator.finish()

    final_pool = np.zeros((2, phantom.nvoxels, 3), dtype=np.float32)
    final_pool[:, active] = final_active
    final_pool = final_pool.reshape((2,) + phantom.shape + (3,))
    result = {
        "signal": species_signal.sum(axis=0),
        "species_signal": species_signal,
        "final_pool_magnetization": final_pool,
        "final_magnetization": final_pool.sum(axis=0),
        "adc_times_s": compiled.adc_times_s,
        "metadata": {
            **capability,
            "requested_backend": "metal_precision_probe",
            "actual_backend": "metal_precision_probe",
            "precision_strategy": (
                DOUBLE_SINGLE_PRECISION_STRATEGY
                if precision_strategy == "double_single"
                else FLOAT32_PRECISION_STRATEGY
            ),
            "state_precision": precision_strategy,
            "fast_math": False,
            "compiled_interval_count": compiled.n_intervals,
            "adc_sample_count": int(compiled.adc_times_s.size),
            "rf_active_interval_count": int(np.count_nonzero(compiled.rf_hz)),
            "crusher_boundary_count": int(crusher_states.size),
            "spin_count": spin_count,
            "effective_spin_chunk_size": effective_chunk_size,
            "spin_chunk_count": chunk_count,
            "state_buffer_bytes": state_bytes,
            "gpu_chunk_state_buffer_bytes": chunk_state_bytes,
            "coefficient_plan_buffer_bytes": plan_bytes,
            "partial_reduction_bytes": reduction_bytes,
            "partial_reduction": (
                "bounded per-spin chunk output; fixed CPU Float64 pairwise trees"
            ),
            "pipeline_compile_seconds": compile_seconds,
            "warm_simulation_seconds": simulation_seconds,
            "end_to_end_probe_seconds": wall_seconds,
            "memory_budget_bytes": effective_budget,
            "peak_working_set_estimate_bytes": peak_estimate,
            "compiled_gpu_segments": {"full_prefix_probe": 1},
            "rf_block_count": rf_block_count,
            "adc_readout_block_count": len(program.adc_events),
            "kinetic_preroll_start_s": preroll_start,
            "field_strength_t": field,
            "nucleus": effective_nucleus,
            "probe_only": True,
        },
    }
    if captured_indices.size:
        result["captured_spin_indices"] = captured_indices
        result["captured_spin_species_signal"] = captured_signal
        result["captured_final_pool_state"] = captured_final
    if captured_groups:
        result["captured_group_species_signal"] = np.stack(
            [
                accumulator.finish()
                for accumulator in captured_group_signal_accumulators
            ],
            axis=0,
        )
        captured_group_final_pool = np.zeros(
            (len(captured_groups), 2, phantom.nvoxels, 3), dtype=np.float64
        )
        captured_group_final_pool[:, :, active] = captured_group_final_active
        result["captured_group_final_pool_magnetization"] = (
            captured_group_final_pool.reshape(
                (len(captured_groups), 2) + phantom.shape + (3,)
            )
        )
        result["captured_group_spin_counts"] = np.asarray(
            [group.size for group in captured_groups], dtype=np.int64
        )
    return result


def _expanded_spin_indices(active_count, grid_count, subvoxel_indices):
    subvoxel_indices = np.asarray(subvoxel_indices, dtype=np.int64)
    return np.sort(
        (
            np.arange(int(active_count), dtype=np.int64)[:, None] * int(grid_count)
            + subvoxel_indices[None, :]
        ).reshape(-1)
    )


def _run_cpu_float64_sample(
    program,
    phantom,
    *,
    sampling,
    simulation_timestep_s,
    signal_weighting,
    spoiler_mode,
    checkpoints_s=(),
    field_strength_t=None,
    nucleus=None,
    memory_budget_bytes=None,
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
    checkpoint_dtype=None,
):
    from .simulator import BlochSimulator

    simulator = BlochSimulator(
        use_parallel=False,
        dynamic_sequence_kernel="optimized",
        dynamic_sequence_precision="float64",
    )
    start = perf_counter()
    result = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        checkpoints_s=checkpoints_s,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        progress_callback=progress_callback,
        preview_callback=preview_callback,
        cancel_callback=cancel_callback,
        status_callback=status_callback,
        simulation_timestep_s=simulation_timestep_s,
        signal_weighting=signal_weighting,
        spin_sampling=sampling,
        spoiler_mode=spoiler_mode,
        memory_budget_bytes=memory_budget_bytes,
        checkpoint_dtype=checkpoint_dtype,
    )
    return result, perf_counter() - start


def _cpu_result_as_probe(result, metadata):
    return {
        "signal": np.asarray(result.signal),
        "species_signal": np.asarray(result.species_signal),
        "final_pool_magnetization": np.asarray(result.final_pool_magnetization),
        "final_magnetization": np.asarray(result.final_magnetization),
        "adc_times_s": np.asarray(result.adc_times_s),
        "metadata": metadata,
        "_sequence_result": result,
    }


def run_metal_hybrid_probe(
    program,
    phantom: DynamicSpectralPhantom,
    *,
    simulation_timestep_s: float,
    signal_weighting: str = "voxel",
    spin_sampling=None,
    spoiler_mode: str = "ideal",
    precision_strategy: str = "float32",
    field_strength_t=None,
    nucleus=None,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.05,
    signal_nrmse_gate: float = 1.0e-3,
    require_species_gate: bool = True,
    fallback_to_cpu: bool = True,
    run_concurrently: bool = True,
    memory_budget_bytes: int | None = None,
    spin_chunk_size: int | str | None = "auto",
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
) -> dict:
    """Run the opt-in CPU/Metal subvoxel correction experiment.

    The GPU predicts the complete grid. Float64 CPU calculations use disjoint
    calibration and held-out validation offsets in every active voxel. The
    validation sample never influences the correction. A failed gate either
    returns the untouched Float64 CPU result or raises, so an untrusted hybrid
    signal is never returned as a successful result.
    """
    from .sequence.spin_sampling import coerce_spin_sampling

    capability = metal_capability()
    if not capability["available"]:
        raise RuntimeError(capability["reason"] or "Metal is unavailable")
    sampling = coerce_spin_sampling(spin_sampling)
    sampling.validate_phantom_dimensions(phantom.ndim)
    calibration_subvoxels, validation_subvoxels = _hybrid_subvoxel_partition(
        sampling, calibration_fraction, validation_fraction
    )
    calibration_sampling = sampling.select(calibration_subvoxels)
    validation_sampling = sampling.select(validation_subvoxels)
    active = np.flatnonzero(phantom.mask.ravel())
    grid_count = sampling.grid_spins_per_voxel
    calibration_indices = _expanded_spin_indices(
        active.size, grid_count, calibration_subvoxels
    )
    validation_indices = _expanded_spin_indices(
        active.size, grid_count, validation_subvoxels
    )
    _, full_weights = sampling.normalized_offsets_and_weights()
    calibration_weight_fraction = float(
        np.sum(np.asarray(full_weights)[calibration_subvoxels])
    )
    validation_weight_fraction = float(
        np.sum(np.asarray(full_weights)[validation_subvoxels])
    )
    if not np.isfinite(signal_nrmse_gate) or signal_nrmse_gate <= 0.0:
        raise ValueError("signal_nrmse_gate must be finite and positive")

    def run_gpu():
        return run_metal_precision_probe(
            program,
            phantom,
            simulation_timestep_s=simulation_timestep_s,
            signal_weighting=signal_weighting,
            spin_sampling=sampling,
            spoiler_mode=spoiler_mode,
            precision_strategy=precision_strategy,
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            memory_budget_bytes=memory_budget_bytes,
            spin_chunk_size=spin_chunk_size,
            capture_spin_groups=(calibration_indices, validation_indices),
            cancel_callback=cancel_callback,
        )

    def run_cpu(sample, *, report_progress=False, report_preview=False):
        return _run_cpu_float64_sample(
            program,
            phantom,
            sampling=sample,
            simulation_timestep_s=simulation_timestep_s,
            signal_weighting=signal_weighting,
            spoiler_mode=spoiler_mode,
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=progress_callback if report_progress else None,
            preview_callback=preview_callback if report_preview else None,
            cancel_callback=cancel_callback,
        )

    hybrid_start = perf_counter()
    if run_concurrently:
        with ThreadPoolExecutor(max_workers=3) as executor:
            gpu_future = executor.submit(run_gpu)
            calibration_future = executor.submit(
                run_cpu, calibration_sampling, report_progress=True
            )
            validation_future = executor.submit(run_cpu, validation_sampling)
            gpu = gpu_future.result()
            calibration_result, calibration_seconds = calibration_future.result()
            validation_result, validation_seconds = validation_future.result()
    else:
        gpu = run_gpu()
        calibration_result, calibration_seconds = run_cpu(
            calibration_sampling, report_progress=True
        )
        validation_result, validation_seconds = run_cpu(validation_sampling)

    gpu_calibration, gpu_validation = np.asarray(gpu["captured_group_species_signal"])
    corrected_species, validation_metrics = _hybrid_signal_correction(
        gpu["species_signal"],
        gpu_calibration,
        calibration_result.species_signal,
        gpu_validation,
        validation_result.species_signal,
        calibration_weight_fraction=calibration_weight_fraction,
        validation_weight_fraction=validation_weight_fraction,
    )

    gpu_calibration_final, gpu_validation_final = np.asarray(
        gpu["captured_group_final_pool_magnetization"], dtype=np.float64
    )
    final_mean_error = (
        np.asarray(calibration_result.final_pool_magnetization, dtype=np.float64)
        - gpu_calibration_final
    ) / calibration_weight_fraction
    corrected_final_pool = (
        np.asarray(gpu["final_pool_magnetization"], dtype=np.float64) + final_mean_error
    )
    predicted_validation_final = (
        gpu_validation_final + final_mean_error * validation_weight_fraction
    )
    validation_final_residual = predicted_validation_final - np.asarray(
        validation_result.final_pool_magnetization, dtype=np.float64
    )
    validation_metrics["estimated_full_final_pool_nrmse"] = _relative_nrmse(
        corrected_final_pool,
        corrected_final_pool + validation_final_residual / validation_weight_fraction,
    )
    validation_metrics["held_out_final_pool_nrmse"] = _relative_nrmse(
        validation_result.final_pool_magnetization,
        predicted_validation_final,
    )

    gate_values = [validation_metrics["estimated_full_total_signal_nrmse"]]
    if require_species_gate:
        gate_values.extend(validation_metrics["estimated_full_species_signal_nrmse"])
    validation_passed = bool(
        all(np.isfinite(value) and value <= signal_nrmse_gate for value in gate_values)
    )
    shared_metadata = {
        **gpu["metadata"],
        "requested_backend": "metal_cpu_subvoxel_hybrid_probe",
        "precision_strategy": HYBRID_PRECISION_STRATEGY,
        "hybrid_calibration_fraction_requested": float(calibration_fraction),
        "hybrid_validation_fraction_requested": float(validation_fraction),
        "hybrid_calibration_weight_fraction": calibration_weight_fraction,
        "hybrid_validation_weight_fraction": validation_weight_fraction,
        "hybrid_calibration_subvoxel_indices": calibration_subvoxels.tolist(),
        "hybrid_validation_subvoxel_indices": validation_subvoxels.tolist(),
        "hybrid_cpu_calibration_spin_count": int(calibration_indices.size),
        "hybrid_cpu_validation_spin_count": int(validation_indices.size),
        "hybrid_gpu_spin_count": int(active.size * grid_count),
        "hybrid_cpu_calibration_seconds": calibration_seconds,
        "hybrid_cpu_validation_seconds": validation_seconds,
        "hybrid_concurrent_execution": bool(run_concurrently),
        "hybrid_validation_metrics": validation_metrics,
        "hybrid_signal_nrmse_gate": float(signal_nrmse_gate),
        "hybrid_species_gate_required": bool(require_species_gate),
        "hybrid_validation_passed": validation_passed,
        "hybrid_correction": (
            "species-wise complex CPU/GPU gain from calibration offsets; "
            "near-zero calibration samples remain uncorrected; validation "
            "offsets are disjoint and held out"
        ),
        "probe_only": True,
    }
    if not validation_passed:
        if not fallback_to_cpu:
            raise RuntimeError(
                "hybrid Metal validation failed its accuracy gate; no corrected "
                "result was returned"
            )
        if status_callback is not None:
            status_callback(
                "GPU accuracy check did not pass; calculating the exact CPU result."
            )
        fallback_result, fallback_seconds = run_cpu(
            sampling, report_progress=True, report_preview=True
        )
        shared_metadata.update(
            {
                "actual_backend": "cpu_float64_fallback",
                "hybrid_fallback_used": True,
                "hybrid_fallback_reason": (
                    "held-out subvoxel validation exceeded the accuracy gate"
                ),
                "hybrid_fallback_seconds": fallback_seconds,
                "end_to_end_hybrid_seconds": perf_counter() - hybrid_start,
            }
        )
        return _cpu_result_as_probe(fallback_result, shared_metadata)

    shared_metadata.update(
        {
            "actual_backend": "metal_cpu_subvoxel_hybrid_probe",
            "hybrid_fallback_used": False,
            "hybrid_fallback_reason": None,
            "hybrid_fallback_seconds": 0.0,
            "end_to_end_hybrid_seconds": perf_counter() - hybrid_start,
        }
    )
    return {
        "signal": corrected_species.sum(axis=0),
        "species_signal": corrected_species,
        "final_pool_magnetization": corrected_final_pool,
        "final_magnetization": corrected_final_pool.sum(axis=0),
        "adc_times_s": np.asarray(gpu["adc_times_s"]),
        "metadata": shared_metadata,
        "_sequence_result": calibration_result,
    }


def _run_exact_cpu_fallback(
    program,
    phantom,
    *,
    sampling,
    simulation_timestep_s,
    signal_weighting,
    spoiler_mode,
    checkpoints_s,
    field_strength_t,
    nucleus,
    memory_budget_bytes,
    progress_callback,
    preview_callback,
    cancel_callback,
    status_callback,
    reason,
    checkpoint_dtype=None,
):
    """Return an ordinary result while recording why the GPU path was skipped."""
    result, fallback_seconds = _run_cpu_float64_sample(
        program,
        phantom,
        sampling=sampling,
        simulation_timestep_s=simulation_timestep_s,
        signal_weighting=signal_weighting,
        spoiler_mode=spoiler_mode,
        checkpoints_s=checkpoints_s,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        memory_budget_bytes=memory_budget_bytes,
        progress_callback=progress_callback,
        preview_callback=preview_callback,
        cancel_callback=cancel_callback,
        status_callback=status_callback,
        checkpoint_dtype=checkpoint_dtype,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "requested_backend": "metal_cpu_subvoxel_hybrid",
            "actual_backend": "cpu_float64_fallback",
            "requested_sequence_kernel": "metal_hybrid",
            "hybrid_fallback_used": True,
            "hybrid_fallback_reason": str(reason),
            "hybrid_fallback_seconds": fallback_seconds,
            "hybrid_validation_passed": False,
            "probe_only": False,
        }
    )
    return replace(result, metadata=metadata)


def run_metal_hybrid_sequence(
    program,
    phantom: DynamicSpectralPhantom,
    *,
    checkpoints_s=(),
    field_strength_t=None,
    nucleus=None,
    progress_callback=None,
    preview_callback=None,
    cancel_callback=None,
    status_callback=None,
    simulation_timestep_s=1e-6,
    signal_weighting="voxel",
    spin_sampling=None,
    spoiler_mode="ideal",
    memory_budget_bytes=None,
    checkpoint_dtype=None,
    **_ignored,
):
    """Run the opt-in hybrid as a regular GUI-compatible sequence result.

    Unsupported cases and failed accuracy checks return a fresh Float64 CPU
    result. The experimental GPU estimate is therefore never exposed as a
    successful simulation unless its independent validation gate passes.
    """
    from .sequence.spin_sampling import coerce_spin_sampling

    sampling = coerce_spin_sampling(spin_sampling)
    checkpoints_s = tuple(checkpoints_s)
    status = status_callback if status_callback is not None else lambda _message: None

    def cancelled():
        return bool(cancel_callback is not None and cancel_callback())

    if cancelled():
        raise RuntimeError("Simulation cancelled")
    if checkpoints_s:
        reason = "stored checkpoints currently require the exact CPU path"
        status("Hybrid GPU mode cannot store checkpoints; using the exact CPU path.")
        return _run_exact_cpu_fallback(
            program,
            phantom,
            sampling=sampling,
            simulation_timestep_s=simulation_timestep_s,
            signal_weighting=signal_weighting,
            spoiler_mode=spoiler_mode,
            checkpoints_s=checkpoints_s,
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            cancel_callback=cancel_callback,
            status_callback=status_callback,
            reason=reason,
            checkpoint_dtype=checkpoint_dtype,
        )

    status(
        "Experimental CPU + Apple GPU mode: calculating all subvoxel spins "
        "and checking independent CPU samples."
    )
    try:
        raw = run_metal_hybrid_probe(
            program,
            phantom,
            simulation_timestep_s=simulation_timestep_s,
            signal_weighting=signal_weighting,
            spin_sampling=sampling,
            spoiler_mode=spoiler_mode,
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            cancel_callback=cancel_callback,
            status_callback=status_callback,
        )
    except (ImportError, MemoryError, RuntimeError, ValueError) as error:
        if cancelled():
            raise RuntimeError("Simulation cancelled") from error
        status("The GPU path is not usable for this run; using the exact CPU path.")
        return _run_exact_cpu_fallback(
            program,
            phantom,
            sampling=sampling,
            simulation_timestep_s=simulation_timestep_s,
            signal_weighting=signal_weighting,
            spoiler_mode=spoiler_mode,
            checkpoints_s=(),
            field_strength_t=field_strength_t,
            nucleus=nucleus,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            cancel_callback=cancel_callback,
            status_callback=status_callback,
            reason=error,
            checkpoint_dtype=checkpoint_dtype,
        )
    if cancelled():
        raise RuntimeError("Simulation cancelled")

    template = raw["_sequence_result"]
    metadata = dict(template.metadata)
    metadata.update(raw["metadata"])
    metadata.update(
        {
            "requested_backend": "metal_cpu_subvoxel_hybrid",
            "requested_sequence_kernel": "metal_hybrid",
            "sequence_kernel": (
                "optimized"
                if raw["metadata"].get("hybrid_fallback_used")
                else "metal_hybrid"
            ),
            "simulation_precision": "mixed_gpu_float32_cpu_float64",
            "state_dtype": (
                "float64"
                if raw["metadata"].get("hybrid_fallback_used")
                else "mixed_float32_float64"
            ),
            "signal_dtype": "complex128",
            "spin_sampling": sampling.to_metadata(),
            "subvoxel_spin_counts_xyz": sampling.counts_xyz,
            "subvoxel_spins_per_voxel": sampling.spins_per_voxel,
            "n_simulated_spins": int(phantom.n_active * sampling.spins_per_voxel),
            "probe_only": False,
        }
    )
    result = replace(
        template,
        signal=np.asarray(raw["signal"]),
        adc_times_s=np.asarray(raw["adc_times_s"]),
        final_magnetization=np.asarray(raw["final_magnetization"]),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.zeros(0, dtype=np.float64),
        metadata=metadata,
        species_signal=np.asarray(raw["species_signal"]),
        final_pool_magnetization=np.asarray(raw["final_pool_magnetization"]),
        checkpoint_pool_magnetization=None,
    )
    if progress_callback is not None:
        progress_callback(1, 1)
    if preview_callback is not None:
        preview_callback(1.0, result.signal)
    if metadata.get("hybrid_fallback_used"):
        status("GPU accuracy check did not pass; the exact CPU result was used.")
    else:
        status("GPU accuracy check passed; using the checked CPU + GPU result.")
    return result
