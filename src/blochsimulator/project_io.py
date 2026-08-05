"""Versioned, pickle-free persistence for complete GUI projects."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .b1_fields import B1Field
from .ui.phantom_designer import load_any_phantom
from .sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from .sequence.result import SequenceSimulationResult


FORMAT = "bloch-simulator-project"
VERSION = 1


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _Arrays:
    def __init__(self):
        self.values = {}

    def add(self, value, prefix="array"):
        key = f"{prefix}_{len(self.values)}"
        self.values[key] = np.asarray(value)
        return {"$array": key}


def _encode(value, arrays, prefix="array"):
    if isinstance(value, np.ndarray):
        return arrays.add(value, prefix)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _encode(v, arrays, prefix) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(v, arrays, prefix) for v in value]}
    if isinstance(value, list):
        return [_encode(v, arrays, prefix) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _decode(value, arrays):
    if isinstance(value, dict) and set(value) == {"$array"}:
        return np.array(arrays[value["$array"]], copy=True)
    if isinstance(value, dict) and set(value) == {"$tuple"}:
        return tuple(_decode(v, arrays) for v in value["$tuple"])
    if isinstance(value, dict):
        return {k: _decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    return value


def _program_to_data(program, arrays):
    if program is None:
        return None
    events = []
    for event in program.events:
        common = {"start_s": event.start_s}
        if isinstance(event, RFEvent):
            events.append(
                {
                    "type": "rf",
                    **common,
                    "raster_s": event.raster_s,
                    "samples": arrays.add(event.samples_hz, "rf"),
                    "frequency_offset_hz": event.frequency_offset_hz,
                    "phase_offset_rad": event.phase_offset_rad,
                }
            )
        elif isinstance(event, GradientEvent):
            events.append(
                {
                    "type": "gradient",
                    **common,
                    "axis": event.axis,
                    "raster_s": event.raster_s,
                    "samples": arrays.add(event.samples_hz_per_m, "gradient"),
                }
            )
        elif isinstance(event, ADCEvent):
            events.append(
                {
                    "type": "adc",
                    **common,
                    "num_samples": event.num_samples,
                    "dwell_s": event.dwell_s,
                    "frequency_offset_hz": event.frequency_offset_hz,
                    "phase_offset_rad": event.phase_offset_rad,
                }
            )
    return {
        "duration_s": program.duration_s,
        "source": program.source,
        "version": program.version,
        "metadata": _json_value(program.metadata),
        "events": events,
    }


def _program_from_data(data, arrays):
    if data is None:
        return None
    events = []
    for item in data["events"]:
        if item["type"] == "rf":
            events.append(
                RFEvent(
                    item["start_s"],
                    _decode(item["samples"], arrays),
                    item["raster_s"],
                    item.get("frequency_offset_hz", 0.0),
                    item.get("phase_offset_rad", 0.0),
                )
            )
        elif item["type"] == "gradient":
            events.append(
                GradientEvent(
                    item["axis"],
                    item["start_s"],
                    _decode(item["samples"], arrays),
                    item["raster_s"],
                )
            )
        else:
            events.append(
                ADCEvent(
                    item["start_s"],
                    item["num_samples"],
                    item["dwell_s"],
                    item.get("frequency_offset_hz", 0.0),
                    item.get("phase_offset_rad", 0.0),
                )
            )
    return SequenceProgram(
        tuple(events),
        data["duration_s"],
        data.get("source", "project"),
        data.get("version", "1.0"),
        data.get("metadata", {}),
    )


def _field_to_data(field, arrays):
    if field is None:
        return None
    return {
        "data": arrays.add(field.data, "b1"),
        "fov_m": field.fov_m,
        "kind": field.kind,
        "spatial_ndim": field.spatial_ndim,
        "name": field.name,
        "scale_xyz": field.scale_xyz,
        "rotation_deg_xyz": field.rotation_deg_xyz,
        "source_path": field.source_path,
    }


def _field_from_data(data, arrays):
    if data is None:
        return None
    values = dict(data)
    values["data"] = _decode(values["data"], arrays)
    return B1Field(**values)


def _result_to_data(result, arrays):
    if result is None:
        return None
    if isinstance(result, SequenceSimulationResult):
        return {
            "kind": "sequence",
            "value": _encode(result.to_dict(), arrays, "result"),
        }
    if isinstance(result, dict):
        return {"kind": "legacy", "value": _encode(result, arrays, "result")}
    return None


def _result_from_data(data, arrays):
    if data is None:
        return None
    value = _decode(data["value"], arrays)
    if data["kind"] == "legacy":
        return value
    keys = (
        "signal",
        "adc_times_s",
        "final_magnetization",
        "checkpoint_magnetization",
        "checkpoint_times_s",
        "metadata",
        "adc_gradient_moment_cyc_per_m",
        "pool_names",
        "species_signal",
        "final_pool_magnetization",
        "checkpoint_pool_magnetization",
    )
    return SequenceSimulationResult(**{key: value.get(key) for key in keys})


def save_project(
    filename,
    state,
    phantom=None,
    tx_field=None,
    rx_field=None,
    program=None,
    legacy_result=None,
    sequence_result=None,
):
    """Write one self-contained project archive."""
    filename = Path(filename)
    arrays = _Arrays()
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "state": _json_value(state),
        "b1": {
            "tx": _field_to_data(tx_field, arrays),
            "rx": _field_to_data(rx_field, arrays),
        },
        "program": _program_to_data(program, arrays),
        "legacy_result": _result_to_data(legacy_result, arrays),
        "sequence_result": _result_to_data(sequence_result, arrays),
        "phantom_entry": None,
    }
    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if phantom is not None:
            with tempfile.TemporaryDirectory() as directory:
                phantom_path = Path(directory) / "phantom.npz"
                phantom.save(phantom_path)
                archive.write(phantom_path, "phantom.npz")
                manifest["phantom_entry"] = "phantom.npz"
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays.values)
        archive.writestr("arrays.npz", buffer.getvalue())
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))


def load_project(filename):
    """Read a project archive and return its independent state objects."""
    with zipfile.ZipFile(filename, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("This is not a Bloch Simulator project file")
        if int(manifest.get("version", 0)) > VERSION:
            raise ValueError("This project was created by a newer application version")
        with np.load(
            io.BytesIO(archive.read("arrays.npz")), allow_pickle=False
        ) as stored:
            arrays = {key: stored[key] for key in stored.files}
        phantom = None
        if manifest.get("phantom_entry"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "phantom.npz"
                path.write_bytes(archive.read(manifest["phantom_entry"]))
                phantom = load_any_phantom(path)
    return {
        "state": manifest.get("state", {}),
        "phantom": phantom,
        "tx_field": _field_from_data(manifest.get("b1", {}).get("tx"), arrays),
        "rx_field": _field_from_data(manifest.get("b1", {}).get("rx"), arrays),
        "program": _program_from_data(manifest.get("program"), arrays),
        "legacy_result": _result_from_data(manifest.get("legacy_result"), arrays),
        "sequence_result": _result_from_data(manifest.get("sequence_result"), arrays),
    }
