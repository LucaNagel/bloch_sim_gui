import numpy as np

from blochsimulator.b1_fields import B1Field
from blochsimulator.phantom import Phantom
from blochsimulator.project_io import (
    load_project,
    read_project_metadata,
    save_project,
    scan_project_folders,
)
from blochsimulator.sequence import (
    ADCEvent,
    RFEvent,
    SequenceProbeResult,
    SequenceProgram,
)
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


def test_spin_probe_project_round_trip(tmp_path):
    result = SequenceProbeResult(
        time_s=np.array([0.0, 0.01]),
        positions_m=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        frequency_offsets_hz=np.array([-100.0, 100.0]),
        magnetization=np.arange(24, dtype=float).reshape(2, 2, 2, 3),
        metadata={"probe_type": "spectral", "axis": np.array([-1.0, 1.0])},
    )
    path = tmp_path / "spin_probe.blochproj"

    save_project(
        path,
        {"workspace_mode": "sequence", "sequence_view_name": "Spin Probe"},
        sequence_result=result,
    )
    loaded = load_project(path)

    assert isinstance(loaded["sequence_result"], SequenceProbeResult)
    np.testing.assert_allclose(loaded["sequence_result"].time_s, result.time_s)
    np.testing.assert_allclose(
        loaded["sequence_result"].magnetization, result.magnetization
    )
    np.testing.assert_allclose(loaded["sequence_result"].metadata["axis"], [-1.0, 1.0])
    metadata = read_project_metadata(path)
    assert metadata["contents"]["sequence_result"] == {
        "kind": "spin-probe",
        "time_samples": 2,
        "positions": 2,
        "frequencies": 2,
        "magnetization_shape": [2, 2, 2, 3],
        "metadata_keys": ["axis", "probe_type"],
    }


def test_project_io_reads_legacy_numpy_array_metadata_strings():
    from blochsimulator.project_io import _decode_legacy_array_strings

    decoded = _decode_legacy_array_strings(
        {"definitions": {"Times": "[ 1.2  2.4\n 3.6 ]"}}
    )
    np.testing.assert_allclose(decoded["definitions"]["Times"], [1.2, 2.4, 3.6])


def test_project_explorer_reads_manifest_metadata_without_loading_arrays(
    tmp_path, monkeypatch
):
    phantom = Phantom(
        shape=(2, 3),
        fov=(0.1, 0.2),
        t1_map=np.ones((2, 3)),
        t2_map=np.full((2, 3), 0.1),
        name="Explorer phantom",
    )
    program = SequenceProgram(
        (RFEvent(0.0, np.array([100 + 0j]), 1e-4), ADCEvent(2e-4, 4, 1e-4)),
        duration_s=6e-4,
        source="explorer test",
    )
    result = SequenceSimulationResult(
        signal=np.ones(4, dtype=complex),
        adc_times_s=np.arange(4) * 1e-4,
        final_magnetization=np.zeros((2, 3, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "indexed.blochproj"
    save_project(
        path,
        {"application_version": "9.9", "workspace_mode": "sequence"},
        phantom=phantom,
        program=program,
        sequence_result=result,
    )

    # Metadata indexing must not use the full project loader.
    monkeypatch.setattr(
        "blochsimulator.project_io.load_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data loaded")),
    )
    metadata = read_project_metadata(path)

    assert metadata["contents"]["phantom"]["shape"] == [2, 3]
    assert metadata["contents"]["sequence"]["event_types"] == {
        "rf": 1,
        "gradient": 0,
        "adc": 1,
    }
    assert metadata["contents"]["sequence_result"]["signal_shape"] == [4]
    assert scan_project_folders([tmp_path], recursive=False) == []
    assert [item["path"] for item in scan_project_folders([tmp_path])] == [
        str(path.resolve())
    ]
