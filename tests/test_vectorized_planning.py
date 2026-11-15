"""Vectorized compile and plan helpers must match their scalar versions bit for bit."""

import dataclasses

import numpy as np
import pytest

from blochsimulator.dynamic_phantom import (
    TimeCurve,
    _prepare_rf_rotation,
    _prepare_rf_rotations,
    _unique_rows,
)
from blochsimulator.sequence import SequenceCompiler
from blochsimulator.sequence.model import RFEvent


def _scalar_event_integral(event_start, raster, samples, start, end):
    """The compiler's former per-interval gradient integral."""
    overlap_start = max(start, event_start)
    event_end = event_start + samples.size * raster
    overlap_end = min(end, event_end)
    if overlap_start >= overlap_end:
        return samples.dtype.type(0)
    first = max(0, int(np.floor((overlap_start - event_start) / raster)))
    last = min(samples.size - 1, int(np.ceil((overlap_end - event_start) / raster)) - 1)
    total = samples.dtype.type(0)
    for index in range(first, last + 1):
        cell_start = event_start + index * raster
        cell_end = cell_start + raster
        width = max(0.0, min(overlap_end, cell_end) - max(overlap_start, cell_start))
        total += samples[index] * width
    return total


@pytest.mark.parametrize("interpolation", ["linear", "step"])
@pytest.mark.parametrize("outside", ["zero", "hold"])
def test_interval_values_array_matches_scalar_interval_values(interpolation, outside):
    rng = np.random.default_rng(1)
    for _ in range(50):
        count = int(rng.integers(1, 12))
        times = np.cumsum(rng.uniform(0.01, 1.0, count)) - rng.uniform(0.0, 2.0)
        curve = TimeCurve(
            tuple(times), tuple(rng.normal(size=count)), interpolation, outside
        )
        grid = np.concatenate(
            [
                times,
                rng.uniform(times[0] - 1.0, times[-1] + 1.0, 50),
                [times[0] - 1e-12, times[-1] + 1e-12],
            ]
        )
        starts = np.sort(rng.choice(grid, 60))
        ends = starts + np.where(rng.random(60) < 0.2, 0.0, rng.uniform(0, 0.5, 60))

        vectorized = curve.interval_values_array(starts, ends)
        scalar = np.array(
            [curve.interval_values(float(a), float(b)) for a, b in zip(starts, ends)]
        )
        assert vectorized.tobytes() == scalar.tobytes()


@pytest.mark.parametrize("interpolation", ["linear", "step"])
@pytest.mark.parametrize("outside", ["zero", "hold"])
def test_integrals_match_scalar_integral(interpolation, outside):
    rng = np.random.default_rng(6)
    for _ in range(50):
        count = int(rng.integers(1, 8))
        times = np.cumsum(rng.uniform(0.01, 1.0, count)) - rng.uniform(0.0, 2.0)
        curve = TimeCurve(
            tuple(times), tuple(rng.normal(size=count)), interpolation, outside
        )
        grid = np.concatenate([times, rng.uniform(times[0] - 1.0, times[-1] + 1.0, 40)])
        starts = rng.choice(grid, 80)
        # Zero, short knot-free, knot-spanning and reversed intervals.
        ends = starts + rng.choice([0.0, 1e-4, 0.7, -0.3], 80)

        vectorized = curve.integrals(starts, ends)
        scalar = np.array([curve.integral(a, b) for a, b in zip(starts, ends)])
        assert vectorized.tobytes() == scalar.tobytes()


def test_interval_values_array_rejects_reversed_intervals():
    curve = TimeCurve((0.0, 1.0), (0.0, 1.0), "linear", "hold")
    with pytest.raises(ValueError, match="must not precede"):
        curve.interval_values_array([0.5], [0.4])


def test_prepare_rf_rotations_matches_scalar_rotation():
    rng = np.random.default_rng(2)
    rf = rng.normal(scale=300.0, size=200) + 1j * rng.normal(scale=300.0, size=200)
    rf[:5] = [0.0, 1.0, -1j, 1e-300, -2.5 + 0.0j]
    duration = rng.choice([20e-6, 9.5e-6, 1e-6], size=rf.size)
    rows = np.column_stack((rf.real, rf.imag, duration))

    vectorized = _prepare_rf_rotations(rows)
    scalar = [_prepare_rf_rotation(value, dt) for value, dt in zip(rf, duration)]

    assert vectorized[0] is None
    assert [
        None if row is None else np.asarray(row).tobytes() for row in vectorized
    ] == [None if row is None else np.asarray(row).tobytes() for row in scalar]


def test_unique_rows_matches_numpy_unique():
    rng = np.random.default_rng(3)
    cases = [
        rng.integers(-3, 4, size=(500, 3)).astype(float),
        np.column_stack(
            (rng.choice([1e-5, 9.7e-6, 2.3e-6], 400), rng.choice([0.0, 1.0], 400))
        ),
        rng.normal(size=(300, 3)),
        np.zeros((0, 2)),
        np.ones((1, 2)),
    ]
    for rows in cases:
        expected, expected_inverse = np.unique(rows, axis=0, return_inverse=True)
        unique, inverse = _unique_rows(rows)
        assert unique.tobytes() == expected.tobytes()
        assert np.array_equal(inverse, expected_inverse.reshape(-1))


def test_piecewise_integrals_match_scalar_loop():
    rng = np.random.default_rng(4)
    events = []
    for _ in range(12):
        size = int(rng.integers(1, 40))
        events.append(
            (
                float(rng.uniform(-1e-4, 1e-3)),
                float(rng.choice([10e-6, 4e-6, 7.3e-6])),
                rng.normal(size=size) * 1e3,
            )
        )
    samples = np.concatenate([values for _, _, values in events])
    counts = np.array([values.size for _, _, values in events])
    offsets = np.cumsum(counts) - counts
    rows = 400
    event_index = rng.integers(0, len(events), rows)
    event_start = np.array([events[i][0] for i in event_index])
    raster = np.array([events[i][1] for i in event_index])
    start = event_start + rng.uniform(-50e-6, 400e-6, rows)
    end = start + np.where(rng.random(rows) < 0.1, 0.0, rng.uniform(0, 300e-6, rows))

    vectorized = SequenceCompiler._piecewise_integrals(
        event_start,
        raster,
        samples,
        offsets[event_index],
        counts[event_index],
        start,
        end,
    )
    scalar = np.array(
        [
            _scalar_event_integral(
                event_start[row], raster[row], events[i][2], start[row], end[row]
            )
            for row, i in enumerate(event_index)
        ]
    )
    assert vectorized.tobytes() == scalar.tobytes()


def test_rf_frame_cache_shares_shapes_across_phase_offsets():
    rng = np.random.default_rng(5)
    samples = rng.normal(size=64) * 200.0 + 1j * rng.normal(size=64) * 50.0
    first = RFEvent(
        start_s=1e-3, samples_hz=samples, raster_s=1e-6, frequency_offset_hz=120.0
    )
    second = dataclasses.replace(first, start_s=7e-3, phase_offset_rad=2.3)
    local = np.concatenate(([0.0], np.sort(rng.uniform(0.0, 64e-6, 9)), [64e-6]))
    cache = {}

    integrals = [
        SequenceCompiler._rf_frame_integrals(event, local, cache)
        for event in (first, second)
    ]

    assert len(cache) == 1
    for event, frame in zip((first, second), integrals):
        scalar = np.array(
            [
                SequenceCompiler._rf_event_integral(
                    event, local[i], local[i + 1], event_start_s=0.0
                )
                for i in range(local.size - 1)
            ],
            dtype=np.complex128,
        )
        assert frame.tobytes() == scalar.tobytes()
