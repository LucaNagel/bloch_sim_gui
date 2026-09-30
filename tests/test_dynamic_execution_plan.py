from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from blochsimulator.dynamic_execution import (
    ACTION_ADC,
    ACTION_CHECKPOINT,
    ACTION_CRUSH,
    SEGMENT_FREE,
    SEGMENT_RF,
    build_dynamic_execution_plan,
)
from blochsimulator.dynamic_phantom import TimeCurve
from blochsimulator.sequence.compiler import CompiledSequence


def _compiled_sequence():
    return CompiledSequence(
        dt_s=np.asarray([0.1, 0.2, 0.3]),
        interval_end_s=np.asarray([0.1, 0.3, 0.6]),
        rf_hz=np.asarray([0.0, 10.0 + 2.0j, 10.0 + 2.0j]),
        gradient_hz_per_m=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        ),
        adc_times_s=np.asarray([0.1, 0.3, 0.3]),
        adc_state_indices=np.asarray([1, 2, 2]),
        adc_demodulation=np.ones(3, dtype=np.complex128),
        checkpoint_times_s=np.asarray([0.3]),
        checkpoint_state_indices=np.asarray([2]),
        transverse_crush_times_s=np.asarray([0.3]),
        transverse_crush_state_indices=np.asarray([2]),
        duration_s=0.6,
        metadata={},
    )


def _phantom(*, spatial_source_delays=False):
    inflow_curve = TimeCurve((0.0, 0.3, 0.6), (1.0, 2.0, 4.0), outside="hold")
    polarization_curve = TimeCurve((0.0, 0.6), (0.25, 0.75), outside="hold")
    b0_curve = TimeCurve((0.0, 0.6), (-2.0, 3.0), outside="hold")
    return SimpleNamespace(
        conversion_start_on_sequence_timeline_s=0.2,
        pyruvate_inflow=SimpleNamespace(has_spatial_delays=spatial_source_delays),
        inflow_curve_on_sequence_timeline=inflow_curve,
        inflow_polarization_curve_on_sequence_timeline=polarization_curve,
        dynamic_b0=SimpleNamespace(offset_curve_hz=b0_curve),
    )


def test_execution_plan_is_exact_immutable_and_fingerprinted():
    compiled = _compiled_sequence()
    first = build_dynamic_execution_plan(compiled, _phantom())
    second = build_dynamic_execution_plan(compiled, _phantom())

    assert np.array_equal(first.dt_s, compiled.dt_s)
    assert np.array_equal(first.rf_hz, compiled.rf_hz)
    assert np.array_equal(first.gradient_hz_per_m, compiled.gradient_hz_per_m)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.nbytes > compiled.dt_s.nbytes
    with pytest.raises(ValueError):
        first.dt_s[0] = 1.0

    changed_rf = compiled.rf_hz.copy()
    changed_rf[2] = np.nextafter(changed_rf[2].real, np.inf) + 2.0j
    changed = replace(compiled, rf_hz=changed_rf)
    assert (
        build_dynamic_execution_plan(changed, _phantom()).fingerprint
        != first.fingerprint
    )


def test_execution_plan_preserves_boundary_action_order_and_segments():
    plan = build_dynamic_execution_plan(_compiled_sequence(), _phantom())

    assert plan.action_state_indices.tolist() == [1, 2, 2, 2, 2]
    assert plan.action_kinds.tolist() == [
        ACTION_ADC,
        ACTION_CRUSH,
        ACTION_ADC,
        ACTION_ADC,
        ACTION_CHECKPOINT,
    ]
    assert plan.action_payload_indices.tolist() == [0, 0, 1, 2, 0]
    assert plan.segment_start_intervals.tolist() == [0, 1, 2]
    assert plan.segment_end_intervals.tolist() == [1, 2, 3]
    assert plan.segment_kinds.tolist() == [SEGMENT_FREE, SEGMENT_RF, SEGMENT_RF]


def test_execution_plan_precomputes_exact_dynamic_samples_and_integrals():
    phantom = _phantom()
    plan = build_dynamic_execution_plan(_compiled_sequence(), phantom)

    assert plan.conversion_active.tolist() == [False, True, True]
    for index, (start, middle, end) in enumerate(
        zip(plan.interval_start_s, plan.interval_mid_s, plan.interval_end_s)
    ):
        assert np.array_equal(
            plan.source_rate_samples[index, 0:2],
            phantom.inflow_curve_on_sequence_timeline.interval_values(start, middle),
        )
        assert np.array_equal(
            plan.source_rate_samples[index, 2:4],
            phantom.inflow_curve_on_sequence_timeline.interval_values(middle, end),
        )
        assert plan.dynamic_b0_first_half_integral_hz_s[index] == pytest.approx(
            phantom.dynamic_b0.offset_curve_hz.integral(start, middle), rel=0, abs=0
        )
        assert plan.dynamic_b0_second_half_integral_hz_s[index] == pytest.approx(
            phantom.dynamic_b0.offset_curve_hz.integral(middle, end), rel=0, abs=0
        )


def test_execution_plan_defers_source_sampling_for_spatial_arrival_delays():
    plan = build_dynamic_execution_plan(
        _compiled_sequence(), _phantom(spatial_source_delays=True)
    )

    assert plan.spatial_source_delays is True
    assert plan.source_rate_samples is None
    assert plan.source_polarization_samples is None
