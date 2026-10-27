"""Named sequence presets: sequence settings saved without phantom or results.

A preset holds the sequence-tab controls of every source (so switching the
source after loading keeps the stored values), the RF pulse and SPSP choices,
the receiver noise model and the spoiler configuration. It deliberately leaves
out the phantom, object, probe, parameter loop, viewer selectors, nucleus and
field, which belong to the experiment rather than to the sequence protocol.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .. import __version__
from ..paths import get_app_data_dir
from .widget_state import capture_widget_state, restore_widget_state

PRESET_FORMAT = "blochsimulator-sequence-preset"
PRESET_VERSION = 1
PRESET_SUFFIX = ".json"
STARTUP_PRESET_KEY = "sequence/startup_preset"

# Control prefixes owned by one sequence source each.
SEQUENCE_CONTROL_PREFIXES = (
    "bssfp_",
    "csi_",
    "epi_",
    "epsi_",
    "fid_",
    "flash_",
    "ir_",
    "me_bssfp_",
    "press_",
    "radial_me_",
    "slice_fid_",
    "slice_selection_",
    "ss_bssfp_",
)
# Shared acquisition controls that are part of a sequence protocol.
SEQUENCE_SHARED_CONTROLS = (
    "sequence_source",
    "sampling_bandwidth_khz",
    "read_matrix",
    "phase_matrix",
    "simulation_timestep_us",
    "signal_weighting",
    "sequence_reference_ppm",
    "spoiling_auto_adjust",
)


def is_sequence_control(name: str) -> bool:
    """Return whether a sequence-widget attribute belongs in a preset."""
    return name in SEQUENCE_SHARED_CONTROLS or name.startswith(
        SEQUENCE_CONTROL_PREFIXES
    )


def capture_sequence_preset(widget) -> dict:
    """Return the preset dictionary for a SequenceSimulationWidget."""
    controls = {
        name: saved
        for name, saved in capture_widget_state(widget).items()
        if is_sequence_control(name)
    }
    return {
        "format": PRESET_FORMAT,
        "version": PRESET_VERSION,
        "application_version": __version__,
        "controls": controls,
        "receiver_noise": widget.receiver_noise.get_state(),
        "rf_pulse": widget.rf_designer_pulse_state(),
        "spectral_spatial_pulse_library": (
            widget.spectral_spatial_pulse_library_state()
        ),
        "spectral_spatial_schedule": widget.spectral_spatial_schedule_state(),
        "spoiler_configuration": {
            "mode": widget.spoiler_mode,
            "counts_xyz": list(widget.subvoxel_spin_counts),
            "sampling_method": widget.subvoxel_sampling_method,
            "auto_adjust": widget.spoiling_auto_adjust.isChecked(),
        },
    }


def apply_sequence_preset(widget, preset: dict) -> None:
    """Restore a preset into a SequenceSimulationWidget and regenerate."""
    if preset.get("format") != PRESET_FORMAT:
        raise ValueError("This file is not a Bloch simulator sequence preset.")
    widget.restore_rf_designer_pulse_state(preset.get("rf_pulse"))
    widget.restore_spectral_spatial_schedule_state(
        preset.get("spectral_spatial_pulse_library"),
        preset.get("spectral_spatial_schedule"),
    )
    controls = {
        name: saved
        for name, saved in dict(preset.get("controls", {})).items()
        if is_sequence_control(name)
    }
    restore_widget_state(widget, controls, block_signals=True)
    if preset.get("receiver_noise"):
        widget.receiver_noise.restore_state(preset["receiver_noise"])
    spoiler = preset.get("spoiler_configuration") or {}
    if spoiler:
        widget.set_spoiler_configuration(
            spoiler.get("mode", widget.spoiler_mode),
            spoiler.get("counts_xyz", widget.subvoxel_spin_counts),
            spoiler.get("sampling_method", widget.subvoxel_sampling_method),
            auto_adjust=spoiler.get("auto_adjust", False),
        )
    widget.refresh_after_session_control_restore()
    widget._reload_selected_generated_sequence()


def preset_file_name(name: str) -> str:
    """Return a safe file name for a preset display name."""
    stem = re.sub(r"[^\w\- .]+", "_", name.strip()).strip(" .")
    if not stem:
        raise ValueError("Enter a preset name.")
    return stem + PRESET_SUFFIX


class SequencePresetStore:
    """JSON preset files in one directory, listed by their display names."""

    def __init__(self, directory: Path | None = None):
        self.directory = Path(
            directory
            if directory is not None
            else get_app_data_dir() / "sequence_presets"
        )

    def path(self, name: str) -> Path:
        return self.directory / preset_file_name(name)

    def names(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(
            (path.stem for path in self.directory.glob("*" + PRESET_SUFFIX)),
            key=str.casefold,
        )

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def save(self, name: str, preset: dict) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(name)
        write_preset_file(path, preset)
        return path

    def load(self, name: str) -> dict:
        return read_preset_file(self.path(name))

    def delete(self, name: str) -> None:
        self.path(name).unlink(missing_ok=True)


_NDARRAY_KEY = "__ndarray__"


def _json_default(value):
    """Encode numpy arrays (also complex) and scalars for the preset file."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                _NDARRAY_KEY: "complex",
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return {_NDARRAY_KEY: "real", "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_object_hook(obj):
    kind = obj.get(_NDARRAY_KEY)
    if kind == "complex":
        return np.asarray(obj["real"]) + 1j * np.asarray(obj["imag"])
    if kind == "real":
        return np.asarray(obj["data"])
    return obj


def write_preset_file(path: Path, preset: dict) -> None:
    Path(path).write_text(
        json.dumps(preset, indent=2, default=_json_default), encoding="utf-8"
    )


def read_preset_file(path: Path) -> dict:
    preset = json.loads(
        Path(path).read_text(encoding="utf-8"), object_hook=_json_object_hook
    )
    if not isinstance(preset, dict) or preset.get("format") != PRESET_FORMAT:
        raise ValueError(f"{Path(path).name} is not a Bloch simulator sequence preset.")
    return preset
