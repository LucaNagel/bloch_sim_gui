import os

import numpy as np
import pytest

from blochsimulator.sequence import (
    SpinSampling,
    analyze_adc_moment_train,
    analyze_repeated_spoiler_train,
    recommend_spin_grid,
    recommend_spin_grid_for_phase_train,
)


@pytest.mark.parametrize("matrix_size", [8, 32, 64, 128])
@pytest.mark.parametrize("spoiler_strength", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("counts", [(1, 1, 1), (3, 3, 3), (3, 4, 5), (5, 13, 1)])
def test_spoiler_train_sweep_is_finite_and_covers_every_phase_encode(
    matrix_size, spoiler_strength, counts
):
    report = analyze_repeated_spoiler_train(
        (spoiler_strength,) * 3,
        SpinSampling(counts),
        matrix_size,
    )

    assert report.n_observations == matrix_size - 1
    assert np.all(np.isfinite(report.continuous_coherence))
    assert np.all(np.isfinite(report.sampled_coherence))
    assert np.all(np.asarray(report.sampled_coherence) <= 1.0 + 1e-12)


def test_regular_3_cube_rephases_on_third_integer_cycle_crusher():
    report = analyze_repeated_spoiler_train(
        (1.0, 1.0, 1.0), SpinSampling((3, 3, 3)), 64
    )

    assert report.single_sampled_coherence == pytest.approx(0.0, abs=1e-14)
    assert report.first_alias_observation == 3
    assert report.sampled_coherence[2] == pytest.approx(1.0)
    assert report.maximum_sampling_error == pytest.approx(1.0)


def test_project_geometry_recommends_resolved_nonrecurring_64_line_grid():
    recommendation = recommend_spin_grid(
        (1.0, 1.0, 1.0),
        64,
        (2, 2, 2),
        "midpoint",
        0.01,
        512,
        32,
    )

    assert recommendation.counts_xyz == (2, 3, 11)
    assert recommendation.spins_per_voxel == 66
    assert recommendation.meets_target


@pytest.mark.parametrize("matrix_size", [8, 32, 64, 128])
@pytest.mark.parametrize("spoiler_strength", [0.5, 1.0, 2.0])
def test_recommended_grid_matches_continuous_voxel_over_complete_train(
    matrix_size, spoiler_strength
):
    recommendation = recommend_spin_grid(
        (spoiler_strength,) * 3,
        matrix_size,
        (1, 1, 1),
        "midpoint",
        0.01,
        1024,
        64,
    )
    report = analyze_repeated_spoiler_train(
        (spoiler_strength,) * 3,
        SpinSampling(recommendation.counts_xyz, method=recommendation.method),
        matrix_size,
    )

    assert report.maximum_sampling_error == pytest.approx(
        recommendation.maximum_sampling_error
    )
    baseline = analyze_repeated_spoiler_train(
        (spoiler_strength,) * 3, SpinSampling(), matrix_size
    )
    assert report.maximum_sampling_error <= baseline.maximum_sampling_error + 1e-12
    if recommendation.meets_target:
        assert report.maximum_sampling_error <= 0.01 + 1e-12
    else:
        assert recommendation.spins_per_voxel <= 1024


def test_stratified_sampling_is_reproducible_and_avoids_exact_short_recurrence():
    first = SpinSampling((3, 3, 3), method="stratified")
    second = SpinSampling((3, 3, 3), method="stratified")
    first_offsets, first_weights = first.normalized_offsets_and_weights()
    second_offsets, second_weights = second.normalized_offsets_and_weights()
    report = analyze_repeated_spoiler_train((1.0, 1.0, 1.0), first, 64)

    np.testing.assert_array_equal(first_offsets, second_offsets)
    np.testing.assert_array_equal(first_weights, second_weights)
    assert np.max(report.sampled_coherence) < 0.5
    assert report.sampled_coherence[2] < 0.5


def test_actual_adc_moment_analysis_matches_repeated_spoiler_model():
    orders = np.arange(5, dtype=float)
    moments = np.column_stack((orders * 1000.0, orders * 500.0, orders * 2000.0))
    voxel_basis = np.diag((1e-3, 2e-3, 0.5e-3))
    sampling = SpinSampling((3, 4, 5))

    actual = analyze_adc_moment_train(moments, voxel_basis, sampling)
    repeated = analyze_repeated_spoiler_train((1.0, 1.0, 1.0), sampling, 5)

    np.testing.assert_allclose(
        actual.phase_cycles_per_voxel, repeated.phase_cycles_per_voxel
    )
    assert actual.sampled_coherence == pytest.approx(repeated.sampled_coherence)

    recommendation = recommend_spin_grid_for_phase_train(
        actual.phase_cycles_per_voxel,
        (1, 1, 1),
        "midpoint",
        0.01,
        512,
        32,
    )
    recommended_report = analyze_adc_moment_train(
        moments,
        voxel_basis,
        SpinSampling(recommendation.counts_xyz, method=recommendation.method),
    )
    assert recommendation.meets_target
    assert recommended_report.maximum_sampling_error <= 0.01 + 1e-12


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BLOCHSIMULATOR_RUN_SLOW_SPOILER_TESTS") != "1",
    reason="set BLOCHSIMULATOR_RUN_SLOW_SPOILER_TESTS=1 for the full FLASH sweep",
)
def test_project_like_flash_phantom_matrix_strength_and_spin_sweep(tmp_path):
    from benchmarks.benchmark_flash_spoiler_train import parse_grid, run_benchmark

    records = run_benchmark(
        output_dir=tmp_path,
        matrix_sizes=(8, 16, 32),
        spoiler_strengths=(0.5, 1.0, 2.0),
        samplings=(
            parse_grid("midpoint:1x1x1"),
            parse_grid("midpoint:3x3x3"),
            parse_grid("midpoint:3x4x5"),
            parse_grid("stratified:3x3x3"),
        ),
        phantom_shape=(12, 12, 24),
        timestep_us=10.0,
        threads=1,
    )

    assert len(records) == 3 * 3 * 5
    assert {record["matrix_size"] for record in records} == {8, 16, 32}
    assert {record["spoiler_cycles_per_phantom_voxel"] for record in records} == {
        0.5,
        1.0,
        2.0,
    }
    assert all(np.isfinite(record["simulation_time_s"]) for record in records)
