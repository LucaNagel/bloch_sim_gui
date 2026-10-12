import numpy as np
from PyQt5.QtCore import Qt

from blochsimulator.phantom import Phantom
from blochsimulator.sequence import SequenceSimulationResult
from blochsimulator.ui.simulation_explorer import SessionSimulationRun
from blochsimulator.ui.volume_viewer import SequenceResultVolumeViewer


def _run(run_id, shape, value, *, extra_checkpoint=False):
    phantom = Phantom(
        shape, tuple(0.01 * n for n in shape), np.ones(shape), np.full(shape, 0.1)
    )
    magnetization = np.broadcast_to(
        np.array([0.1, 0.2, 0.3]) * value, shape + (3,)
    ).copy()
    times = np.array([0.001, 0.002] if extra_checkpoint else [0.002])
    checkpoints = np.stack([magnetization * t / 0.004 for t in times])
    names = ("B", "A") if extra_checkpoint else ("A", "B")
    weights = {"A": 0.25, "B": 0.75}
    result = SequenceSimulationResult(
        signal=np.empty(0, dtype=complex),
        adc_times_s=np.empty(0),
        final_magnetization=magnetization,
        checkpoint_magnetization=checkpoints,
        checkpoint_times_s=times,
        pool_names=names,
        final_pool_magnetization=np.stack([magnetization * weights[n] for n in names]),
        checkpoint_pool_magnetization=np.stack(
            [checkpoints * weights[n] for n in names], axis=1
        ),
        metadata={
            "duration_s": 0.004,
            "sequence_definitions": {
                "MagnetizationOnly": 1,
                "SliceSelectionEndTime": 0.002,
                "RephasingEndTime": 0.004,
            },
        },
    )
    return SessionSimulationRun(
        run_id=run_id,
        created_at_utc="2026-10-12T10:00:00+00:00",
        sequence_name="rf_slice_selection",
        phantom_name=f"Phantom {run_id}",
        phantom_shape=shape,
        sequence_duration_s=0.004,
        adc_samples=0,
        runtime_s=0.1,
        kernel="reference",
        result=result,
        state={"phantom": phantom},
        display_name=f"Loop {run_id}/2 · flip angle = {30 * value}°",
    )


def test_run_selection_uses_own_geometry_and_preserves_map_state_and_pool(
    qt_application,
):
    viewer = SequenceResultVolumeViewer()
    first = _run("1", (2, 3, 4), 1)
    second = _run("2", (3, 2, 1), 2, extra_checkpoint=True)
    for run in (first, second):
        viewer.set_result(run.result, run.state["phantom"])
        viewer.add_session_run(run, current=True)
    assert viewer.simulation_combo.count() == 2
    assert viewer.simulation_combo.currentData() == second.run_id
    assert "60°" in viewer.simulation_combo.currentText()
    assert "Phantom 2" in viewer.simulation_combo.itemData(1, Qt.ToolTipRole)
    assert viewer._session_runs[first.run_id].result is first.result

    viewer.map_combo.setCurrentText("Mz")
    viewer.state_combo.setCurrentIndex(2)
    viewer.pool_combo.setCurrentText("B")
    viewer.simulation_combo.setCurrentIndex(0)
    assert viewer.result is first.result
    assert viewer.phantom is first.state["phantom"]
    assert viewer.map_combo.currentText() == "Mz"
    assert viewer.state_combo.currentData() == "SliceSelectionEndTime"
    assert viewer.state_combo.currentIndex() == 1
    assert viewer.pool_combo.currentText() == "B"
    np.testing.assert_allclose(viewer.volume.data, 0.3 * 0.5 * 0.75)
    assert viewer.volume.data.shape == first.phantom_shape
    assert viewer.volume.fov_m == first.state["phantom"].fov
    np.testing.assert_array_equal(viewer.volume.mask, first.state["phantom"].mask)

    viewer.simulation_combo.setCurrentIndex(1)
    assert viewer.result is second.result
    assert viewer.state_combo.currentIndex() == 2
    assert viewer.pool_combo.currentIndex() == 1
    np.testing.assert_allclose(viewer.volume.data, 0.6 * 0.5 * 0.75)
    assert viewer.volume.data.shape == second.phantom_shape

    viewer.set_result(first.result, first.state["phantom"])
    assert viewer.simulation_combo.currentData() == first.run_id


def test_run_list_tracks_rename_delete_and_survives_clear(qt_application):
    viewer = SequenceResultVolumeViewer()
    first = _run("1", (2, 3, 4), 1)
    second = _run("2", (3, 2, 1), 2)
    viewer.add_session_run(first)
    viewer.add_session_run(second)
    viewer.simulation_combo.setCurrentIndex(0)
    first.display_name = "Baseline"
    viewer.add_session_run(first)
    assert viewer.simulation_combo.count() == 2
    assert viewer.simulation_combo.currentText() == "Baseline"
    viewer.remove_session_run(second)
    assert viewer.result is first.result
    assert viewer.simulation_combo.currentData() == first.run_id

    viewer.clear("Next simulation running")
    assert viewer.result is None
    assert viewer.simulation_combo.count() == 1
    assert viewer.simulation_combo.currentIndex() == -1
    assert viewer.status.text() == "Next simulation running"
    assert viewer.volume.isHidden()
    viewer.simulation_combo.setCurrentIndex(0)
    assert viewer.result is first.result
    assert not viewer.volume.isHidden()
    viewer.remove_session_run(first)
    assert viewer.result is None
    assert viewer.phantom is None
    assert viewer.volume.data.shape == (1, 1, 1)
    assert not viewer.simulation_combo.isEnabled()
    assert not viewer._session_runs
