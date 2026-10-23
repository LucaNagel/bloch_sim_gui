"""Weighted coherence pathways and the subvoxel grid recommendation."""

import numpy as np
import pytest

from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    SpinSampling,
    coherence_pathways_from_program,
    declares_ideal_spoilers,
    deduplicate_observations,
    recommend_spin_grid_for_pathways,
    weighted_pathway_sampling_error,
)
from blochsimulator.sequence.spoiling import _AxisTables

RASTER_S = 10e-6
VOXEL_M = 2e-3
BASIS_M = np.diag([VOXEL_M] * 3)


def _block_pulse(start_s, flip_rad, phase_rad=0.0, samples=100):
    amplitude_hz = flip_rad / (2.0 * np.pi * samples * RASTER_S)
    return RFEvent(
        start_s,
        np.full(samples, amplitude_hz, dtype=complex),
        RASTER_S,
        phase_offset_rad=phase_rad,
    )


def _crusher(start_s, cycles_per_voxel, axis="z", samples=100):
    amplitude = cycles_per_voxel / VOXEL_M / (samples * RASTER_S)
    return GradientEvent(axis, start_s, np.full(samples, amplitude), RASTER_S)


def _flash(n_tr, flip_deg, tr_s=10e-3, cycles=1.0):
    events = []
    for index in range(n_tr):
        start = index * tr_s
        events.extend(
            (
                _block_pulse(start, np.deg2rad(flip_deg)),
                ADCEvent(start + 2e-3, 1, 20e-6),
                _crusher(start + 5e-3, cycles),
            )
        )
    return SequenceProgram(tuple(events), n_tr * tr_s)


def _integer_epg(n_tr, flip_rad, tr_s, t1_s, t2_s, max_order=200):
    """Reference EPG (Weigel 2015) with one dephasing order per TR."""
    f_plus = np.zeros(max_order + 1, complex)
    f_minus = np.zeros(max_order + 1, complex)
    z = np.zeros(max_order + 1, complex)
    z[0] = 1.0
    c2, s2 = np.cos(flip_rad / 2) ** 2, np.sin(flip_rad / 2) ** 2
    s, c = np.sin(flip_rad), np.cos(flip_rad)
    e1, e2 = np.exp(-tr_s / t1_s), np.exp(-tr_s / t2_s)
    for _ in range(n_tr):
        f_plus, f_minus, z = (
            c2 * f_plus + s2 * f_minus - 1j * s * z,
            s2 * f_plus + c2 * f_minus + 1j * s * z,
            -0.5j * s * f_plus + 0.5j * s * f_minus + c * z,
        )
        observed = (np.abs(f_plus).copy(), np.abs(f_minus).copy())
        f_plus, f_minus, z = f_plus * e2, f_minus * e2, z * e1
        z[0] += 1.0 - e1
        f_plus = np.concatenate(([np.conj(f_minus[1])], f_plus[:-1]))
        f_minus = np.concatenate((f_minus[1:], [0.0]))
    return observed


def test_pathway_weights_match_integer_epg_for_flash_train():
    n_tr, flip_deg, t1_s, t2_s = 48, 15.0, 1.0, 0.2
    train = coherence_pathways_from_program(
        _flash(n_tr, flip_deg),
        BASIS_M,
        t1_s=t1_s,
        t2_s=t2_s,
        weight_threshold=0.0,
    )
    f_plus, f_minus = _integer_epg(n_tr, np.deg2rad(flip_deg), 10e-3, t1_s, t2_s)
    last = train.adc_indices == train.adc_indices.max()
    phases = train.phase_cycles_per_voxel[last, 2]
    weights = train.weights[last]

    assert train.rf_pulse_count == n_tr
    assert train.refocusing_pulse_count == 0
    assert train.evaluated_adc_count == n_tr
    for order in range(1, 6):
        model = np.sort(
            [weights[np.isclose(phases, sign * order)].max() for sign in (1, -1)]
        )
        reference = np.sort([f_plus[order], f_minus[order]]) / f_plus[0]
        np.testing.assert_allclose(model, reference, atol=2e-3)


def test_spin_echo_counts_refocusing_and_finds_crusher_pathways():
    events = []
    for index in range(4):
        start = index * 0.5
        events.extend(
            (
                _block_pulse(start, np.pi / 2),
                _crusher(start + 5e-3, 3.0),
                _block_pulse(start + 10e-3, 0.9 * np.pi, phase_rad=np.pi / 2),
                _crusher(start + 12e-3, 3.0),
                ADCEvent(start + 19e-3, 16, 20e-6),
            )
        )
    train = coherence_pathways_from_program(
        SequenceProgram(tuple(events), 2.0), BASIS_M, t1_s=1.0, t2_s=0.1
    )
    phases, weights = deduplicate_observations(
        train.phase_cycles_per_voxel, train.weights
    )

    assert train.refocusing_pulse_count == 4
    # The spin echo itself is the reference; the FID of each pulse and the
    # imperfect refocusing leave pathways at +-3 crusher cycles.
    assert set(np.round(phases[weights > 0.1, 2]).astype(int)) == {-3, 3}
    recommendation = recommend_spin_grid_for_pathways(phases, weights, (1, 1, 2))
    assert recommendation.meets_target
    assert recommendation.method == "midpoint"
    assert recommendation.counts_xyz == (1, 1, 4)


def test_pathway_recommendation_prefers_midpoint_and_falls_back_to_rms():
    phases = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    weights = np.array([1.0, 0.5])
    recommendation = recommend_spin_grid_for_pathways(phases, weights, (1, 1, 2))
    assert recommendation.method == "midpoint"
    assert recommendation.meets_target
    # Three points alias only at multiples of three cycles.
    assert recommendation.counts_xyz == (1, 1, 3)
    assert not recommendation.error_is_rms

    # Strong pathways at every order up to 40 cycles cannot be resolved by a
    # 32-point midpoint grid; the best grid is reported with an RMS error.
    dense = np.column_stack((np.zeros(40), np.zeros(40), np.arange(1.0, 41.0)))
    fallback = recommend_spin_grid_for_pathways(dense, np.ones(40), (1, 1, 2))
    assert not fallback.meets_target
    assert fallback.method == "stratified"
    assert fallback.error_is_rms

    current = weighted_pathway_sampling_error(
        phases, weights, SpinSampling((1, 1, 2), method="midpoint")
    )
    # Two midpoint spins rephase completely at 2 and 4 cycles.
    assert current.maximum_error == pytest.approx(1.0)
    assert current.worst_phase_cycles_per_voxel == (0.0, 0.0, 2.0)


def test_closed_form_midpoint_factor_matches_explicit_grid():
    rng = np.random.default_rng(3)
    phases = rng.uniform(-40.0, 40.0, size=(2000, 3))
    phases[:20, 0] = np.arange(20) * 3.0
    tables = _AxisTables(phases)
    for count in (2, 3, 4, 7, 32):
        offsets = (np.arange(count) + 0.5) / count - 0.5
        explicit = np.mean(np.cos(2 * np.pi * np.outer(phases[:, 0], offsets)), axis=1)
        np.testing.assert_allclose(
            tables._midpoint_factor(0, count), explicit, atol=1e-9
        )


def test_stratified_offsets_vary_per_voxel_and_are_chunk_independent():
    sampling = SpinSampling((2, 3, 4), method="stratified")
    assert sampling.varies_per_voxel
    all_offsets = sampling.normalized_voxel_offsets(np.arange(6))
    chunk = sampling.normalized_voxel_offsets(np.array([4, 5]))

    np.testing.assert_array_equal(all_offsets[4:], chunk)
    assert not np.allclose(all_offsets[0], all_offsets[1])
    # Every point stays inside its own grid cell.
    counts = np.array(sampling.counts_xyz)
    cells = np.floor((all_offsets + 0.5) * counts)
    np.testing.assert_array_equal(cells[0], cells[1])

    midpoint = SpinSampling((2, 3, 4), method="midpoint")
    assert not midpoint.varies_per_voxel
    midpoint_offsets = midpoint.normalized_voxel_offsets(np.arange(3))
    np.testing.assert_array_equal(midpoint_offsets[0], midpoint_offsets[2])


def test_declares_ideal_spoilers_recognizes_current_and_legacy_markers():
    plain = SequenceProgram((), 0.0)
    assert not declares_ideal_spoilers(plain)
    for name in ("IdealSpoilerEndTimes", "SpoilerEndTimes", "EndImageSpoilerEndTimes"):
        marked = SequenceProgram((), 0.0, metadata={"definitions": {name: []}})
        assert declares_ideal_spoilers(marked)
