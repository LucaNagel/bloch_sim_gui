"""Geometry contracts of Free Mode position sampling."""

import numpy as np
import pytest

from blochsimulator.position_sampling import (
    build_line_positions,
    build_plane_positions,
    build_volume_positions,
    infer_layout,
)


@pytest.mark.parametrize("axis, column", [("x", 0), ("y", 1), ("z", 2)])
def test_line_varies_only_its_axis(axis, column):
    positions = build_line_positions(5, 0.02, axis)

    assert positions.shape == (5, 3)
    np.testing.assert_allclose(positions[:, column], np.linspace(-0.01, 0.01, 5))
    others = [c for c in range(3) if c != column]
    assert np.all(positions[:, others] == 0.0)


def test_single_line_position_sits_at_the_center():
    positions = build_line_positions(1, 0.02, "x", center_m=(0.001, 0.002, 0.003))

    np.testing.assert_allclose(positions, [[0.001, 0.002, 0.003]])


def test_plane_is_row_major_with_first_axis_fastest():
    positions = build_plane_positions((3, 2), (0.02, 0.01), "xy")

    np.testing.assert_allclose(
        positions[:, :2],
        [
            [-0.01, -0.005],
            [0.0, -0.005],
            [0.01, -0.005],
            [-0.01, 0.005],
            [0.0, 0.005],
            [0.01, 0.005],
        ],
    )
    assert np.all(positions[:, 2] == 0.0)
    image = positions.reshape(2, 3, 3)
    np.testing.assert_allclose(image[:, 0, 1], [-0.005, 0.005])
    np.testing.assert_allclose(image[0, :, 0], [-0.01, 0.0, 0.01])


def test_plane_center_offsets_the_normal_axis():
    positions = build_plane_positions(
        (2, 2), (0.01, 0.01), "xz", center_m=(0.0, 0.004, 0.0)
    )

    np.testing.assert_allclose(positions[:, 1], 0.004)
    np.testing.assert_allclose(positions[:, 0], [-0.005, 0.005, -0.005, 0.005])
    np.testing.assert_allclose(positions[:, 2], [-0.005, -0.005, 0.005, 0.005])


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_layout_round_trips_planes(plane):
    center = np.zeros(3)
    center[3 - sum("xyz".index(name) for name in plane)] = 0.001
    positions = build_plane_positions((4, 3), (0.03, 0.02), plane, center)
    layout = infer_layout(positions)

    assert layout.is_grid
    assert layout.plane_name == plane
    assert layout.counts == (4, 3)
    assert layout.shape == (3, 4)
    np.testing.assert_allclose(layout.coordinates[0], np.linspace(-0.015, 0.015, 4))
    np.testing.assert_allclose(layout.coordinates[1], np.linspace(-0.01, 0.01, 3))
    values = np.arange(12)
    np.testing.assert_array_equal(layout.reshape(values)[1], [4, 5, 6, 7])


def test_layout_of_a_line_and_a_point():
    line = infer_layout(build_line_positions(7, 0.02, "y"))
    point = infer_layout(np.zeros((1, 3)))

    assert line.axes == (1,) and line.counts == (7,)
    assert line.primary_axis == 1
    assert point.ndim == 0 and point.primary_axis == 2
    assert point.describe() == "single position"


def test_plane_with_one_sample_per_axis_is_a_line():
    layout = infer_layout(build_plane_positions((1, 5), (0.02, 0.02), "xy"))

    assert layout.axes == (1,)
    assert layout.counts == (5,)


def test_diagonal_line_falls_back_to_largest_span():
    positions = np.zeros((5, 3))
    positions[:, 0] = np.linspace(-0.01, 0.01, 5)
    positions[:, 1] = np.linspace(-0.02, 0.02, 5)
    layout = infer_layout(positions)

    assert not layout.is_grid
    assert layout.axes == (1,)
    np.testing.assert_allclose(layout.coordinates[0], positions[:, 1])


def test_grid_indices_subsample_each_axis_and_keep_the_grid():
    positions = build_plane_positions((9, 5), (0.08, 0.04), "xy")
    layout = infer_layout(positions)
    indices = layout.grid_indices(3)
    sub = layout.sub_layout(indices)

    np.testing.assert_allclose(sub.coordinates[0], [-0.04, 0.0, 0.04])
    np.testing.assert_allclose(sub.coordinates[1], [-0.02, 0.0, 0.02])
    sub_inferred = infer_layout(positions[indices])
    assert sub_inferred.counts == sub.counts
    np.testing.assert_allclose(sub_inferred.coordinates[0], sub.coordinates[0])
    np.testing.assert_allclose(sub_inferred.coordinates[1], sub.coordinates[1])


def test_position_labels_name_the_varying_axes():
    plane = infer_layout(build_plane_positions((2, 2), (0.002, 0.004), "xy"))
    line = infer_layout(build_line_positions(3, 0.02, "z"))

    assert plane.label([0.001, -0.002, 0.0]) == "(x, y) = (1.00, -2.00) mm"
    assert line.label([0.0, 0.0, 0.01]) == "z = 10.00 mm"


def test_volume_is_row_major_with_x_fastest_and_z_slowest():
    positions = build_volume_positions((3, 2, 2), (0.02, 0.01, 0.004))

    assert positions.shape == (12, 3)
    volume = positions.reshape(2, 2, 3, 3)
    np.testing.assert_allclose(volume[0, 0, :, 0], [-0.01, 0.0, 0.01])
    np.testing.assert_allclose(volume[0, :, 0, 1], [-0.005, 0.005])
    np.testing.assert_allclose(volume[:, 0, 0, 2], [-0.002, 0.002])


def test_layout_round_trips_a_volume():
    positions = build_volume_positions(
        (4, 3, 2), (0.03, 0.02, 0.01), center_m=(0.0, 0.0, 0.005)
    )
    layout = infer_layout(positions)

    assert layout.is_grid
    assert layout.ndim == 3
    assert layout.plane_name == "xyz"
    assert layout.counts == (4, 3, 2)
    assert layout.shape == (2, 3, 4)
    assert layout.strides == (1, 4, 12)
    np.testing.assert_allclose(layout.coordinates[2], [0.0, 0.01])
    values = np.arange(24)
    np.testing.assert_array_equal(layout.reshape(values)[1, 2], [20, 21, 22, 23])
    assert layout.describe() == "4 × 3 × 2 volume in xyz"
    assert layout.label([0.001, 0.002, 0.003]) == ("(x, y, z) = (1.00, 2.00, 3.00) mm")


def test_volume_with_one_sample_along_an_axis_is_a_plane():
    layout = infer_layout(build_volume_positions((3, 1, 4), (0.02, 0.02, 0.03)))

    assert layout.axes == (0, 2)
    assert layout.counts == (3, 4)


def test_volume_grid_indices_subsample_every_axis():
    positions = build_volume_positions((9, 5, 7), (0.08, 0.04, 0.06))
    layout = infer_layout(positions)
    indices = layout.grid_indices(3)
    sub = layout.sub_layout(indices)

    assert indices.size == 27
    np.testing.assert_allclose(sub.coordinates[0], [-0.04, 0.0, 0.04])
    np.testing.assert_allclose(sub.coordinates[1], [-0.02, 0.0, 0.02])
    np.testing.assert_allclose(sub.coordinates[2], [-0.03, 0.0, 0.03])
    sub_inferred = infer_layout(positions[indices])
    assert sub_inferred.counts == sub.counts
    for inferred, expected in zip(sub_inferred.coordinates, sub.coordinates):
        np.testing.assert_allclose(inferred, expected)


@pytest.mark.parametrize("along", [0, 1, 2])
def test_volume_slice_matches_the_inferred_plane(along):
    positions = build_volume_positions((4, 3, 5), (0.03, 0.02, 0.04))
    layout = infer_layout(positions)

    indices, plane = layout.slice_plane(along, 1)

    sliced = positions[indices]
    np.testing.assert_allclose(sliced[:, along], layout.coordinates[along][1])
    inferred = infer_layout(sliced)
    assert plane.axes == inferred.axes
    assert plane.counts == inferred.counts
    for expected, got in zip(inferred.coordinates, plane.coordinates):
        np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(plane.center, sliced[0])


def test_slice_plane_needs_a_volume():
    layout = infer_layout(build_plane_positions((3, 2), (0.02, 0.01), "xy"))

    with pytest.raises(ValueError):
        layout.slice_plane(0, 0)


def test_shuffled_volume_is_not_a_grid():
    positions = build_volume_positions((3, 3, 3), (0.02, 0.02, 0.02))
    rng = np.random.default_rng(0)
    layout = infer_layout(positions[rng.permutation(27)])

    assert not layout.is_grid
