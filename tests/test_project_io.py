import numpy as np

from blochsimulator.b1_fields import B1Field
from blochsimulator.phantom import Phantom
from blochsimulator.project_io import load_project, save_project
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
from blochsimulator.sequence.result import SequenceSimulationResult


def test_complete_project_round_trip(tmp_path):
    phantom = Phantom(
        shape=(2, 3),
        fov=(0.1, 0.2),
        t1_map=np.ones((2, 3)),
        t2_map=np.full((2, 3), 0.1),
        name="saved",
    )
    tx = B1Field.uniform((2, 3), phantom.fov, kind="transmit", value=0.8 + 0.2j)
    program = SequenceProgram(
        (
            RFEvent(0.0, np.array([100 + 20j]), 1e-4, 12.0, 0.3),
            ADCEvent(2e-4, 2, 1e-4, 5.0, 0.1),
        ),
        duration_s=4e-4,
        metadata={"label": "test", "definitions": {"FOV": np.array([0.1, 0.2])}},
    )
    result = SequenceSimulationResult(
        signal=np.array([1 + 2j, 3 + 4j]),
        adc_times_s=np.array([2e-4, 3e-4]),
        final_magnetization=np.zeros((2, 3, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={"ok": True},
    )
    path = tmp_path / "session.blochproj"
    save_project(
        path,
        {"workspace_mode": "sequence"},
        phantom,
        tx,
        None,
        program,
        {"time": np.arange(3)},
        result,
    )

    loaded = load_project(path)
    assert loaded["state"]["workspace_mode"] == "sequence"
    assert loaded["phantom"].name == "saved"
    np.testing.assert_allclose(loaded["tx_field"].data, tx.data)
    np.testing.assert_allclose(loaded["program"].rf_events[0].samples_hz, [100 + 20j])
    assert loaded["program"].rf_events[0].frequency_offset_hz == 12.0
    np.testing.assert_allclose(
        loaded["program"].metadata["definitions"]["FOV"], [0.1, 0.2]
    )
    np.testing.assert_allclose(loaded["legacy_result"]["time"], np.arange(3))
    np.testing.assert_allclose(loaded["sequence_result"].signal, result.signal)


def test_project_io_reads_legacy_numpy_array_metadata_strings():
    from blochsimulator.project_io import _decode_legacy_array_strings

    decoded = _decode_legacy_array_strings(
        {"definitions": {"Times": "[ 1.2  2.4\n 3.6 ]"}}
    )
    np.testing.assert_allclose(decoded["definitions"]["Times"], [1.2, 2.4, 3.6])
