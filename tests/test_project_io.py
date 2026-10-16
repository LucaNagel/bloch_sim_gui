import io
import json
import zipfile

import numpy as np
import pytest

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
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo("arrays.npz").compress_type == zipfile.ZIP_STORED
        assert archive.getinfo("phantom.npz").compress_type == zipfile.ZIP_STORED


@pytest.mark.parametrize(
    "word_dtype, float_dtype, words",
    [
        (
            np.uint32,
            np.float32,
            [
                0x00000000,
                0x80000000,
                0x3F800000,
                0x3F800001,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0xFFC00002,
                0x00000001,
                0x80000001,
                0x3F000000,
                0xBF000000,
            ],
        ),
        (
            np.uint64,
            np.float64,
            [
                0x0000000000000000,
                0x8000000000000000,
                0x3FF0000000000000,
                0x3FF0000000000001,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000000001,
                0xFFF8000000000002,
                0x0000000000000001,
                0x8000000000000001,
                0x3FE0000000000000,
                0xBFE0000000000000,
            ],
        ),
    ],
)
def test_sequence_project_preserves_magnetization_bits_without_component_copies(
    tmp_path, word_dtype, float_dtype, words
):
    # Include signed zeros, adjacent floats, infinities, NaN payloads, and
    # subnormal values in a non-contiguous array. No tolerance is appropriate.
    magnetization = (
        np.asarray(words, dtype=word_dtype)
        .view(float_dtype)
        .reshape(2, 2, 3)
        .transpose(1, 0, 2)
    )
    assert not magnetization.flags.c_contiguous
    result = SequenceSimulationResult(
        signal=np.empty(0, dtype=complex),
        adc_times_s=np.empty(0),
        final_magnetization=magnetization,
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
    )
    path = tmp_path / "magnetization.blochproj"
    save_project(path, {"workspace_mode": "sequence"}, sequence_result=result)

    with zipfile.ZipFile(path) as archive:
        values = json.loads(archive.read("manifest.json"))["sequence_result"]["value"]
        assert {"mx", "my", "mz"}.isdisjoint(values)
        with np.load(io.BytesIO(archive.read("arrays.npz"))) as arrays:
            stored = arrays[values["final_magnetization"]["$array"]]
            assert stored.dtype == magnetization.dtype
            assert stored.shape == magnetization.shape
            assert stored.tobytes() == magnetization.tobytes()

    loaded = load_project(path)["sequence_result"]
    assert loaded.final_magnetization.dtype == magnetization.dtype
    assert loaded.final_magnetization.shape == magnetization.shape
    assert loaded.final_magnetization.tobytes() == magnetization.tobytes()
    for index, name in enumerate(("mx", "my", "mz")):
        expected = magnetization[..., index]
        for candidate in (result, loaded):
            component = getattr(candidate, name)
            assert component.dtype == expected.dtype
            assert component.shape == expected.shape
            assert component.tobytes() == expected.tobytes()
            assert np.shares_memory(component, candidate.final_magnetization)
            assert candidate.to_dict()[name].tobytes() == expected.tobytes()


def test_sequence_project_loads_legacy_magnetization_component_arrays(tmp_path):
    magnetization = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    arrays = {
        "signal": np.empty(0, dtype=complex),
        "adc_times_s": np.empty(0),
        "final_magnetization": magnetization,
        "checkpoint_times_s": np.empty(0),
        "mx": magnetization[..., 0],
        "my": magnetization[..., 1],
        "mz": magnetization[..., 2],
    }
    # Reproduce the previous on-disk schema independently of the current writer.
    manifest = {
        "format": "bloch-simulator-project",
        "version": 1,
        "sequence_result": {
            "kind": "sequence",
            "value": {
                **{name: {"$array": name} for name in arrays},
                "checkpoint_magnetization": None,
                "metadata": {"legacy_components": True},
            },
        },
    }
    path = tmp_path / "legacy.blochproj"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", buffer.getvalue())

    loaded = load_project(path)["sequence_result"]
    assert loaded.final_magnetization.tobytes() == magnetization.tobytes()
    assert loaded.metadata == {"legacy_components": True}
    for name in ("mx", "my", "mz"):
        assert getattr(loaded, name).tobytes() == arrays[name].tobytes()


def test_failed_project_write_preserves_previous_file_and_removes_partial_output(
    tmp_path, monkeypatch
):
    path = tmp_path / "saved.blochproj"
    save_project(path, {"workspace_mode": "sequence"})
    previous = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr("blochsimulator.project_io.np.savez_compressed", fail_write)
    with pytest.raises(OSError, match="Disk full"):
        save_project(path, {"workspace_mode": "free"})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_project_state_round_trips_complex_rf_waveform_arrays(tmp_path):
    waveform = np.asarray([1.0 + 2.0j, 3.0 - 4.0j])
    gradient = np.asarray([0.0, 120.0, -120.0, 0.0])
    path = tmp_path / "rf_state.blochproj"

    save_project(
        path,
        {
            "sequence_rf_pulse": {"waveform_hz": waveform},
            "sequence_spectral_spatial_pulse": {
                "rf_hz": waveform,
                "gradient_hz_per_m": gradient,
                "bands": [{"center_hz": 0.0, "width_hz": 80.0}],
            },
        },
    )

    loaded = load_project(path)
    np.testing.assert_array_equal(
        loaded["state"]["sequence_rf_pulse"]["waveform_hz"], waveform
    )
    np.testing.assert_array_equal(
        loaded["state"]["sequence_spectral_spatial_pulse"]["rf_hz"], waveform
    )
    np.testing.assert_array_equal(
        loaded["state"]["sequence_spectral_spatial_pulse"]["gradient_hz_per_m"],
        gradient,
    )


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
