"""Remaining-time estimates of sequence runs, spin probes and loops."""

import pytest

from blochsimulator.ui.run_progress import (
    LoopProgressEstimator,
    RunProgressEstimator,
    format_duration,
)


def _estimator(reports, total=100, started_at=0.0):
    estimator = RunProgressEstimator(started_at, total)
    for time_s, done in reports:
        estimator.report(done, total, time_s)
    return estimator


def test_rate_ignores_the_fast_start_of_a_throttled_run():
    # 10 units/s for the first 10 s, then 5 units/s.
    estimator = _estimator([(0.0, 0), (5.0, 50), (10.0, 100), (20.0, 150)], 400)

    assert estimator.rate == pytest.approx(5.0)
    assert estimator.remaining_s(20.0) == pytest.approx(50.0)


def test_preparation_before_the_first_report_does_not_count():
    estimator = _estimator([(30.0, 0), (40.0, 20)], started_at=0.0)

    assert estimator.rate == pytest.approx(2.0)
    fraction, text = estimator.status(40.0)
    assert fraction == pytest.approx(0.2)
    assert text == "20% · 40s elapsed · ~40s left"


def test_overdue_report_raises_the_remaining_time_instead_of_sticking():
    # Chunks of 25 units every 5 s, then the third chunk takes longer.
    estimator = _estimator([(0.0, 0), (5.0, 25), (10.0, 50)])

    assert estimator.remaining_s(12.0) == pytest.approx(8.0)
    # The next report was due at 15 s; at 20 s at most 75 units are done.
    assert estimator.remaining_s(20.0) == pytest.approx(25.0 / (25.0 / 10.0))
    assert estimator.remaining_s(30.0) > estimator.remaining_s(20.0)
    assert estimator.status(30.0)[0] == pytest.approx(0.75)


def test_small_leading_chunk_does_not_cap_the_next_report():
    # Simulator runs start with about 1/64 of the voxels, then larger chunks.
    estimator = _estimator([(10.0, 0), (10.5, 489)], total=31288, started_at=5.0)

    fraction, text = estimator.status(13.5)

    assert fraction == pytest.approx((489 + 978 * 3) / 31288)
    assert text == "10% · 9s elapsed · ~29s left"


def test_overdue_final_report_reads_almost_done():
    estimator = _estimator([(0.0, 0), (10.0, 50)])

    fraction, text = estimator.status(30.0)

    assert fraction == pytest.approx(0.99)
    assert text == "99% · 30s elapsed · almost done…"


def test_displayed_fraction_never_moves_back_and_completes_on_report():
    estimator = _estimator([(0.0, 0), (10.0, 50)])
    assert estimator.status(19.0)[0] == pytest.approx(0.95)

    estimator.report(60, 100, 20.0)
    assert estimator.status(20.0)[0] == pytest.approx(0.95)

    estimator.report(100, 100, 30.0)
    assert estimator.status(30.0) == (1.0, "100% · 30s elapsed · finishing…")


def test_restarted_count_starts_a_new_rate():
    estimator = _estimator([(0.0, 0), (10.0, 50)])
    estimator.report(10, 100, 12.0)

    assert estimator.rate is None
    estimator.report(30, 100, 14.0)
    assert estimator.rate == pytest.approx(10.0)


def test_first_loop_run_estimates_the_loop_from_its_own_estimate():
    loop = LoopProgressEstimator(0.0, 4)

    assert loop.status(5.0) == (0.0, "Loop 1/4 · 5s elapsed · estimating time left…")
    fraction, text = loop.status(10.0, run_fraction=0.5, run_remaining_s=10.0)

    assert fraction == pytest.approx(0.125)
    # 10 s left of this run plus three runs of 20 s.
    assert text == "Loop 1/4 · 10s elapsed · ~1m 10s left"


def test_later_loop_runs_use_the_mean_finished_run_duration():
    loop = LoopProgressEstimator(0.0, 4)
    loop.run_finished(20.0)
    loop.run_finished(60.0)

    # Between runs, e.g. while compiling, the mean duration of 30 s applies.
    assert loop.remaining_s(65.0) == pytest.approx(25.0 + 30.0)
    assert loop.remaining_s(65.0, run_remaining_s=5.0) == pytest.approx(35.0)
    fraction, text = loop.status(65.0, run_fraction=0.5, run_remaining_s=5.0)
    assert fraction == pytest.approx(0.625)
    assert text.startswith("Loop 3/4 · 1m 05s elapsed · ~35s left")

    # Saving after the run reports no run fraction; the bar keeps its place.
    assert loop.status(70.0)[0] == pytest.approx(0.625)


def test_format_duration():
    assert format_duration(0.4) == "<1s"
    assert format_duration(59.2) == "1m 00s"
    assert format_duration(3725) == "1h 02m"
