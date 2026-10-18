"""Shared exact execution-plan representation for dynamic CPU/GPU backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

import numpy as np


SEGMENT_FREE = 0
SEGMENT_RF = 1

ACTION_CRUSH = 1
ACTION_ADC = 2
ACTION_CHECKPOINT = 3


def _readonly_contiguous(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DynamicExecutionPlan:
    """Backend-neutral plan with exact interval and boundary action ordering."""

    dt_s: np.ndarray
    interval_start_s: np.ndarray
    interval_mid_s: np.ndarray
    interval_end_s: np.ndarray
    # (mid - start, end - mid) per interval. Inside RF pulses these come from
    # the compiler's pulse-relative grid, so identical pulses get identical
    # half steps wherever they sit on the absolute timeline.
    interval_half_s: np.ndarray
    rf_hz: np.ndarray
    gradient_hz_per_m: np.ndarray
    conversion_active: np.ndarray
    source_rate_samples: Optional[np.ndarray]
    source_polarization_samples: Optional[np.ndarray]
    spatial_source_delays: bool
    dynamic_b0_first_half_integral_hz_s: np.ndarray
    dynamic_b0_second_half_integral_hz_s: np.ndarray
    action_state_indices: np.ndarray
    action_kinds: np.ndarray
    action_payload_indices: np.ndarray
    segment_start_intervals: np.ndarray
    segment_end_intervals: np.ndarray
    segment_kinds: np.ndarray
    fingerprint: str

    def __post_init__(self):
        dtypes = {
            "dt_s": np.float64,
            "interval_start_s": np.float64,
            "interval_mid_s": np.float64,
            "interval_end_s": np.float64,
            "interval_half_s": np.float64,
            "rf_hz": np.complex128,
            "gradient_hz_per_m": np.float64,
            "conversion_active": np.bool_,
            "dynamic_b0_first_half_integral_hz_s": np.float64,
            "dynamic_b0_second_half_integral_hz_s": np.float64,
            "action_state_indices": np.int64,
            "action_kinds": np.int8,
            "action_payload_indices": np.int64,
            "segment_start_intervals": np.int64,
            "segment_end_intervals": np.int64,
            "segment_kinds": np.int8,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(
                self, name, _readonly_contiguous(getattr(self, name), dtype)
            )
        for name in ("source_rate_samples", "source_polarization_samples"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly_contiguous(value, np.float64))

    @property
    def n_intervals(self) -> int:
        return int(self.dt_s.size)

    @property
    def n_segments(self) -> int:
        return int(self.segment_start_intervals.size)

    @property
    def n_actions(self) -> int:
        return int(self.action_kinds.size)

    @property
    def nbytes(self) -> int:
        total = 0
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
        return total


def _action_table(compiled):
    actions = []
    for payload, state_index in enumerate(compiled.transverse_crush_state_indices):
        actions.append((int(state_index), ACTION_CRUSH, payload))
    for payload, state_index in enumerate(compiled.adc_state_indices):
        actions.append((int(state_index), ACTION_ADC, payload))
    for payload, state_index in enumerate(compiled.checkpoint_state_indices):
        actions.append((int(state_index), ACTION_CHECKPOINT, payload))
    # The kind values encode the solver's established boundary order: ideal
    # crusher first, then every ADC sample, then every checkpoint.
    actions.sort(key=lambda value: (value[0], value[1], value[2]))
    if not actions:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    return tuple(np.asarray(values) for values in zip(*actions))


def _segments(rf_hz, action_state_indices):
    interval_count = int(rf_hz.size)
    if interval_count == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0, dtype=np.int8)
    protected = set(int(value) for value in action_state_indices)
    starts = []
    ends = []
    kinds = []
    start = 0
    kind = SEGMENT_RF if rf_hz[0] != 0.0 else SEGMENT_FREE
    for interval in range(interval_count):
        next_kind = (
            SEGMENT_RF
            if interval + 1 < interval_count and rf_hz[interval + 1] != 0.0
            else SEGMENT_FREE
        )
        boundary_has_action = interval + 1 in protected
        mode_changes = interval + 1 < interval_count and next_kind != kind
        if boundary_has_action or mode_changes or interval + 1 == interval_count:
            starts.append(start)
            ends.append(interval + 1)
            kinds.append(kind)
            start = interval + 1
            kind = next_kind
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(kinds, dtype=np.int8),
    )


def _fingerprint(arrays, flags):
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(array)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    digest.update(repr(tuple(flags)).encode("ascii"))
    return digest.hexdigest()


def build_dynamic_execution_plan(compiled, phantom) -> DynamicExecutionPlan:
    """Build one exact immutable plan without time or field quantization."""
    dt_s = np.asarray(compiled.dt_s, dtype=np.float64)
    interval_end_s = np.asarray(compiled.interval_end_s, dtype=np.float64)
    interval_start_s = (
        np.concatenate(([0.0], interval_end_s[:-1]))
        if interval_end_s.size
        else np.zeros(0, dtype=np.float64)
    )
    interval_mid_s = getattr(compiled, "interval_mid_s", None)
    interval_half_s = getattr(compiled, "interval_half_s", None)
    if interval_mid_s is None or interval_half_s is None:
        interval_mid_s = interval_start_s + dt_s / 2.0
        interval_half_s = np.column_stack(
            (interval_mid_s - interval_start_s, interval_end_s - interval_mid_s)
        ).reshape(-1, 2)
    interval_mid_s = np.asarray(interval_mid_s, dtype=np.float64)
    interval_half_s = np.asarray(interval_half_s, dtype=np.float64)
    conversion_active = (
        interval_mid_s >= phantom.conversion_start_on_sequence_timeline_s
    )

    spatial_source_delays = bool(
        phantom.pyruvate_inflow is not None
        and phantom.pyruvate_inflow.has_spatial_delays
    )
    sample_times = np.column_stack((interval_start_s, interval_mid_s, interval_end_s))
    source_rate_samples = None
    source_polarization_samples = None
    inflow_curve = phantom.inflow_curve_on_sequence_timeline
    polarization_curve = phantom.inflow_polarization_curve_on_sequence_timeline
    if inflow_curve is not None and not spatial_source_delays:
        source_rate_samples = np.empty((dt_s.size, 4), dtype=np.float64)
        source_polarization_samples = np.empty_like(source_rate_samples)
        for index, (start, middle, end) in enumerate(sample_times):
            source_rate_samples[index, 0:2] = inflow_curve.interval_values(
                float(start), float(middle)
            )
            source_rate_samples[index, 2:4] = inflow_curve.interval_values(
                float(middle), float(end)
            )
            if polarization_curve is None:
                source_polarization_samples[index] = 1.0
            else:
                source_polarization_samples[index, 0:2] = (
                    polarization_curve.interval_values(float(start), float(middle))
                )
                source_polarization_samples[index, 2:4] = (
                    polarization_curve.interval_values(float(middle), float(end))
                )

    first_dynamic_b0 = np.zeros(dt_s.size, dtype=np.float64)
    second_dynamic_b0 = np.zeros(dt_s.size, dtype=np.float64)
    if phantom.dynamic_b0 is not None:
        curve = phantom.dynamic_b0.offset_curve_hz
        for index, (start, middle, end) in enumerate(sample_times):
            first_dynamic_b0[index] = curve.integral(float(start), float(middle))
            second_dynamic_b0[index] = curve.integral(float(middle), float(end))

    action_states, action_kinds, action_payloads = _action_table(compiled)
    segment_starts, segment_ends, segment_kinds = _segments(
        np.asarray(compiled.rf_hz), action_states
    )
    fingerprint_arrays = (
        dt_s,
        interval_end_s,
        interval_mid_s,
        interval_half_s,
        compiled.rf_hz,
        compiled.gradient_hz_per_m,
        conversion_active,
        source_rate_samples,
        source_polarization_samples,
        first_dynamic_b0,
        second_dynamic_b0,
        action_states,
        action_kinds,
        action_payloads,
    )
    fingerprint = _fingerprint(fingerprint_arrays, (spatial_source_delays,))
    return DynamicExecutionPlan(
        dt_s=dt_s,
        interval_start_s=interval_start_s,
        interval_mid_s=interval_mid_s,
        interval_end_s=interval_end_s,
        interval_half_s=interval_half_s,
        rf_hz=compiled.rf_hz,
        gradient_hz_per_m=compiled.gradient_hz_per_m,
        conversion_active=conversion_active,
        source_rate_samples=source_rate_samples,
        source_polarization_samples=source_polarization_samples,
        spatial_source_delays=spatial_source_delays,
        dynamic_b0_first_half_integral_hz_s=first_dynamic_b0,
        dynamic_b0_second_half_integral_hz_s=second_dynamic_b0,
        action_state_indices=action_states,
        action_kinds=action_kinds,
        action_payload_indices=action_payloads,
        segment_start_intervals=segment_starts,
        segment_end_intervals=segment_ends,
        segment_kinds=segment_kinds,
        fingerprint=fingerprint,
    )
