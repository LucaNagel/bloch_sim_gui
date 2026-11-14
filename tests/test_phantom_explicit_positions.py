"""Free-spin phantoms keep their explicit positions through every code path."""

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import GradientEvent, RFEvent, SequenceProgram

# Off-grid positions: a 1D grid phantom would put every spin on y = z = 0.
POSITIONS_M = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.013, 0.05],
        [-0.07, 0.02, -0.12],
    ]
)


def _free_phantom(positions=POSITIONS_M):
    count = len(positions)
    return Phantom.from_positions(
        positions,
        t1_map=np.full(count, 1e6),
        t2_map=np.full(count, 1e6),
    )


def _excite_and_encode_program():
    """90 degree hard pulse, then constant Gx and Gz for 1 ms."""
    rf_duration_s = 100e-6
    gradient_raster_s = 10e-6
    samples = int(round(1e-3 / gradient_raster_s))
    events = (
        RFEvent(0.0, np.array([0.25 / rf_duration_s]), rf_duration_s),
        GradientEvent("x", rf_duration_s, np.full(samples, 1000.0), gradient_raster_s),
        GradientEvent("z", rf_duration_s, np.full(samples, -600.0), gradient_raster_s),
    )
    return SequenceProgram(events, duration_s=rf_duration_s + 1e-3)


def test_from_positions_keeps_positions_through_copy():
    phantom = _free_phantom()

    assert phantom.shape == (3,)
    assert phantom.has_explicit_positions
    for candidate in (phantom, phantom.copy()):
        assert np.array_equal(candidate.positions, POSITIONS_M)
        assert np.array_equal(
            np.column_stack((candidate.x, candidate.y, candidate.z)), POSITIONS_M
        )
        assert np.array_equal(
            candidate.get_active_properties()["positions"], POSITIONS_M
        )
    grid = Phantom(shape=(4,), fov=(0.1,), t1_map=np.ones(4), t2_map=np.ones(4))
    assert not grid.has_explicit_positions


def test_simulation_uses_explicit_positions_not_grid_coordinates():
    phantom = _free_phantom()
    program = _excite_and_encode_program()

    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program, phantom, checkpoints_s=(program.duration_s,)
    )
    state = np.asarray(result.checkpoint_magnetization)[0]
    transverse = state[:, 0] + 1j * state[:, 1]
    phase = np.angle(transverse * np.conj(transverse[0]))
    expected = (
        2 * np.pi * 1e-3 * (1000.0 * POSITIONS_M[:, 0] - 600.0 * POSITIONS_M[:, 2])
    )

    assert np.allclose(np.abs(transverse), 1.0, atol=1e-6)
    assert np.allclose(phase, expected, atol=1e-6) or np.allclose(
        phase, -expected, atol=1e-6
    )


def test_subvoxel_sampling_is_rejected_for_explicit_positions():
    phantom = _free_phantom()
    program = _excite_and_encode_program()

    with pytest.raises(ValueError, match="explicit spin positions"):
        BlochSimulator(use_parallel=False).simulate_sequence(
            program, phantom, spoiler_mode="gradient", spin_sampling=(2, 1, 1)
        )


def test_grid_only_operations_reject_explicit_positions(tmp_path):
    phantom = _free_phantom()

    with pytest.raises(ValueError, match="voxel volume"):
        phantom.voxel_volume_m3
    with pytest.raises(ValueError, match="explicit positions"):
        phantom.save(tmp_path / "probe.npz")
    with pytest.raises(ValueError, match="explicit positions"):
        phantom.to_xarray()


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.zeros((3, 2)), "shape"),
        (np.array([[0.0, 0.0, np.nan]]), "finite"),
    ],
)
def test_from_positions_validates_positions(positions, message):
    with pytest.raises(ValueError, match=message):
        Phantom.from_positions(
            positions,
            t1_map=np.ones(len(positions)),
            t2_map=np.ones(len(positions)),
        )


def test_explicit_positions_require_a_matching_1d_shape():
    with pytest.raises(ValueError, match="1D"):
        Phantom(
            shape=(2, 1),
            fov=(1.0, 1.0),
            t1_map=np.ones((2, 1)),
            t2_map=np.ones((2, 1)),
            positions_m=np.zeros((2, 3)),
        )
    with pytest.raises(ValueError, match="should be"):
        Phantom(
            shape=(2,),
            fov=(1.0,),
            t1_map=np.ones(2),
            t2_map=np.ones(2),
            positions_m=np.zeros((3, 3)),
        )
    with pytest.raises(TypeError, match="sets fov"):
        Phantom.from_positions(
            np.zeros((1, 3)), t1_map=np.ones(1), t2_map=np.ones(1), fov=(0.1,)
        )
