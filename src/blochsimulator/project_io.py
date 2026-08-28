"""Versioned, pickle-free persistence for complete GUI projects."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .b1_fields import B1Field
from .ui.phantom_designer import load_any_phantom
from .sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProbeResult,
    SequenceProgram,
)
from .sequence.result import SequenceSimulationResult


FORMAT = "bloch-simulator-project"
VERSION = 1


def _shape(value):
    """Return a JSON-safe array shape without retaining any array data."""
    if value is None:
        return None
    try:
        return [int(size) for size in np.shape(value)]
    except Exception:
        return None


def _field_summary(field):
    if field is None:
        return None
    return {
        "name": str(field.name),
        "kind": str(field.kind),
        "shape": _shape(field.data),
        "fov_m": [float(value) for value in field.fov_m],
    }


def _program_summary(program):
    if program is None:
        return None
    event_types = {
        "rf": sum(isinstance(event, RFEvent) for event in program.events),
        "gradient": sum(isinstance(event, GradientEvent) for event in program.events),
        "adc": sum(isinstance(event, ADCEvent) for event in program.events),
    }
    metadata = program.metadata if isinstance(program.metadata, dict) else {}
    definitions = metadata.get("definitions", {})
    return {
        "source": str(program.source),
        "duration_s": float(program.duration_s),
        "event_count": int(len(program.events)),
        "event_types": event_types,
        "definition_keys": sorted(str(key) for key in definitions),
    }


def _result_summary(result):
    if result is None:
        return None
    if isinstance(result, SequenceSimulationResult):
        return {
            "kind": "sequence",
            "signal_shape": _shape(result.signal),
            "adc_samples": int(np.size(result.adc_times_s)),
            "final_magnetization_shape": _shape(result.final_magnetization),
            "checkpoint_count": int(np.size(result.checkpoint_times_s)),
            "pool_names": [str(name) for name in result.pool_names],
            "metadata_keys": sorted(str(key) for key in result.metadata),
        }
    if isinstance(result, SequenceProbeResult):
        return {
            "kind": "spin-probe",
            "time_samples": int(np.size(result.time_s)),
            "positions": int(result.positions_m.shape[0]),
            "frequencies": int(result.frequency_offsets_hz.size),
            "magnetization_shape": _shape(result.magnetization),
            "metadata_keys": sorted(str(key) for key in result.metadata),
        }
    if isinstance(result, dict):
        arrays = {
            str(key): _shape(value)
            for key, value in result.items()
            if isinstance(value, np.ndarray)
        }
        return {
            "kind": "free-mode",
            "keys": sorted(str(key) for key in result if not str(key).startswith("_")),
            "array_shapes": arrays,
        }
    return {"kind": type(result).__name__}


def _project_metadata(
    filename,
    state,
    phantom,
    tx_field,
    rx_field,
    program,
    legacy_result,
    sequence_result,
):
    phantom_summary = None
    if phantom is not None:
        phantom_summary = {
            "name": str(phantom.name),
            "type": type(phantom).__name__,
            "shape": [int(size) for size in phantom.shape],
            "fov_m": [float(value) for value in phantom.fov],
        }
        components = getattr(phantom, "species", None) or getattr(
            phantom, "pools", None
        )
        if components:
            phantom_summary["components"] = [
                str(getattr(component, "name", component)) for component in components
            ]
        if getattr(phantom, "nucleus", None):
            phantom_summary["nucleus"] = str(phantom.nucleus)
        if getattr(phantom, "field_strength", None) is not None:
            phantom_summary["field_strength_t"] = float(phantom.field_strength)
    return {
        "name": Path(filename).stem,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "application_version": str(state.get("application_version", "")),
        "workspace_mode": str(state.get("workspace_mode", "")),
        "contents": {
            "phantom": phantom_summary,
            "tx_field": _field_summary(tx_field),
            "rx_field": _field_summary(rx_field),
            "sequence": _program_summary(program),
            "free_mode_result": _result_summary(legacy_result),
            "sequence_result": _result_summary(sequence_result),
        },
    }


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


def _decode_legacy_array_strings(value):
    """Convert NumPy array repr strings from older project files."""
    if isinstance(value, dict):
        return {key: _decode_legacy_array_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_legacy_array_strings(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            body = text[1:-1].replace(",", " ")
            if not body.strip():
                return np.empty(0, dtype=float)
            try:
                parsed = np.fromstring(body, sep=" ")
            except ValueError:
                parsed = np.empty(0, dtype=float)
            if parsed.size:
                return parsed
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
        "metadata": _encode(program.metadata, arrays, "metadata"),
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
    metadata = _decode(data.get("metadata", {}), arrays)
    metadata = _decode_legacy_array_strings(metadata)
    return SequenceProgram(
        tuple(events),
        data["duration_s"],
        data.get("source", "project"),
        data.get("version", "1.0"),
        metadata,
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
    if isinstance(result, SequenceProbeResult):
        return {
            "kind": "spin-probe",
            "value": _encode(
                {
                    "time_s": result.time_s,
                    "positions_m": result.positions_m,
                    "frequency_offsets_hz": result.frequency_offsets_hz,
                    "magnetization": result.magnetization,
                    "metadata": result.metadata,
                },
                arrays,
                "probe_result",
            ),
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
    if data["kind"] == "spin-probe":
        return SequenceProbeResult(
            time_s=np.asarray(value["time_s"]),
            positions_m=np.asarray(value["positions_m"]),
            frequency_offsets_hz=np.asarray(value["frequency_offsets_hz"]),
            magnetization=np.asarray(value["magnetization"]),
            metadata=dict(value.get("metadata", {})),
        )
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
        "sequence_waveforms",
        "physical_field_maps",
    )
    arguments = {key: value.get(key) for key in keys}
    arguments["pool_names"] = tuple(arguments.get("pool_names") or ())
    arguments["sequence_waveforms"] = arguments.get("sequence_waveforms") or {}
    arguments["physical_field_maps"] = arguments.get("physical_field_maps") or {}
    return SequenceSimulationResult(**arguments)


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
        "project_metadata": _project_metadata(
            filename,
            state,
            phantom,
            tx_field,
            rx_field,
            program,
            legacy_result,
            sequence_result,
        ),
        "state": _json_value(state),
        "b1": {
            "tx": _field_to_data(tx_field, arrays),
            "rx": _field_to_data(rx_field, arrays),
        },
        "program": _program_to_data(program, arrays),
        "legacy_result": _result_to_data(legacy_result, arrays),
        "sequence_result": _result_to_data(sequence_result, arrays),
        "phantom_entry": None,
        "phantom_name": None if phantom is None else str(phantom.name),
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


def _legacy_project_metadata(manifest):
    """Build an explorer summary for projects saved before metadata was added."""
    state = manifest.get("state", {})
    b1 = manifest.get("b1", {})
    program = manifest.get("program")
    event_types = {"rf": 0, "gradient": 0, "adc": 0}
    if program:
        for event in program.get("events", []):
            kind = str(event.get("type", ""))
            if kind in event_types:
                event_types[kind] += 1
    sequence = None
    if program:
        sequence = {
            "source": str(program.get("source", "project")),
            "duration_s": float(program.get("duration_s", 0.0)),
            "event_count": len(program.get("events", [])),
            "event_types": event_types,
            "definition_keys": [],
        }
    return {
        "name": "",
        "saved_at": "",
        "application_version": str(state.get("application_version", "")),
        "workspace_mode": str(state.get("workspace_mode", "")),
        "contents": {
            "phantom": (
                {"name": str(manifest.get("phantom_name") or "Phantom")}
                if manifest.get("phantom_entry")
                else None
            ),
            "tx_field": {"kind": "transmit"} if b1.get("tx") else None,
            "rx_field": {"kind": "receive"} if b1.get("rx") else None,
            "sequence": sequence,
            "free_mode_result": (
                {"kind": "free-mode"} if manifest.get("legacy_result") else None
            ),
            "sequence_result": (
                {"kind": "sequence"} if manifest.get("sequence_result") else None
            ),
        },
    }


def read_project_metadata(filename):
    """Read only a project's small JSON manifest for explorer display.

    Unlike :func:`load_project`, this never opens ``arrays.npz`` or the embedded
    phantom, so scanning folders does not preload simulation data into memory.
    """
    path = Path(filename)
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("format") != FORMAT:
        raise ValueError("This is not a Bloch Simulator project file")

    metadata = manifest.get("project_metadata")
    if not isinstance(metadata, dict):
        metadata = _legacy_project_metadata(manifest)
    else:
        # Work on an independent JSON-compatible object and tolerate metadata
        # written by intermediate development versions.
        metadata = json.loads(json.dumps(metadata))
        metadata.setdefault("contents", {})
        metadata.setdefault("application_version", "")
        metadata.setdefault("workspace_mode", "")
        metadata.setdefault("saved_at", "")

    stat = path.stat()
    metadata.update(
        {
            "name": str(metadata.get("name") or path.stem),
            "path": str(path.resolve()),
            "folder": str(path.resolve().parent),
            "file_size": int(stat.st_size),
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "format_version": int(manifest.get("version", 0)),
        }
    )
    return metadata


def scan_project_folders(folders, *, recursive=True):
    """Return metadata records for every ``.blochproj`` in the given folders."""
    projects = []
    seen = set()
    for folder in folders:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        candidates = (
            root.rglob("*.blochproj") if recursive else root.glob("*.blochproj")
        )
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path.absolute()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            try:
                projects.append(read_project_metadata(path))
            except Exception as exc:
                try:
                    stat = path.stat()
                    file_size = int(stat.st_size)
                    modified_at = datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat()
                except OSError:
                    file_size = 0
                    modified_at = ""
                projects.append(
                    {
                        "name": path.stem,
                        "path": str(resolved),
                        "folder": str(resolved.parent),
                        "file_size": file_size,
                        "modified_at": modified_at,
                        "contents": {},
                        "error": str(exc),
                    }
                )
    projects.sort(
        key=lambda item: (str(item.get("modified_at", "")), item["name"].lower()),
        reverse=True,
    )
    return projects


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
                if manifest.get("phantom_name") is not None:
                    phantom.name = str(manifest["phantom_name"])
    return {
        "state": manifest.get("state", {}),
        "phantom": phantom,
        "tx_field": _field_from_data(manifest.get("b1", {}).get("tx"), arrays),
        "rx_field": _field_from_data(manifest.get("b1", {}).get("rx"), arrays),
        "program": _program_from_data(manifest.get("program"), arrays),
        "legacy_result": _result_from_data(manifest.get("legacy_result"), arrays),
        "sequence_result": _result_from_data(manifest.get("sequence_result"), arrays),
    }
