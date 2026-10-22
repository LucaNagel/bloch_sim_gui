"""Scanner hardware limits shared by Pulseq sequence builders and the GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ScannerParameters:
    """Physical gradient limits and scanner timing constraints.

    Gradient amplitude and slew rate use the conventional scanner units shown
    in the UI. All timing values are stored in seconds so they can be passed to
    :class:`pypulseq.Opts` without an implicit unit conversion.
    """

    max_grad_mtm: float = 32.0
    max_slew_tms: float = 130.0
    grad_raster_time_s: float = 10e-6
    rf_raster_time_s: float = 1e-6
    adc_raster_time_s: float = 0.1e-6
    block_duration_raster_s: float = 10e-6
    rf_ringdown_time_s: float = 30e-6
    rf_dead_time_s: float = 100e-6
    adc_dead_time_s: float = 20e-6

    def __post_init__(self) -> None:
        positive = (
            "max_grad_mtm",
            "max_slew_tms",
            "grad_raster_time_s",
            "rf_raster_time_s",
            "adc_raster_time_s",
            "block_duration_raster_s",
        )
        non_negative = (
            "rf_ringdown_time_s",
            "rf_dead_time_s",
            "adc_dead_time_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in non_negative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | "ScannerParameters" | None
    ) -> "ScannerParameters":
        """Return validated parameters, accepting partial mapping overrides."""
        if values is None:
            return cls()
        if isinstance(values, cls):
            return values
        known_fields = {field.name for field in fields(cls)}
        unknown = set(values) - known_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown scanner parameter(s): {names}")
        merged = asdict(cls())
        merged.update(values)
        return cls(**{name: float(value) for name, value in merged.items()})

    def to_dict(self) -> dict[str, float]:
        """Return a notebook- and JSON-friendly representation."""
        return {name: float(value) for name, value in asdict(self).items()}

    def to_pypulseq_kwargs(self) -> dict[str, float | str]:
        """Translate the profile to explicit :class:`pypulseq.Opts` kwargs."""
        return {
            "max_grad": float(self.max_grad_mtm),
            "grad_unit": "mT/m",
            "max_slew": float(self.max_slew_tms),
            "slew_unit": "T/m/s",
            "grad_raster_time": float(self.grad_raster_time_s),
            "rf_raster_time": float(self.rf_raster_time_s),
            "adc_raster_time": float(self.adc_raster_time_s),
            "block_duration_raster": float(self.block_duration_raster_s),
            "rf_ringdown_time": float(self.rf_ringdown_time_s),
            "rf_dead_time": float(self.rf_dead_time_s),
            "adc_dead_time": float(self.adc_dead_time_s),
        }


_SETTING_FIELDS = {
    "max_grad_mtm": ("scanner/max_grad_mtm", 1.0),
    "max_slew_tms": ("scanner/max_slew_tms", 1.0),
    "grad_raster_time_s": ("scanner/grad_raster_time_us", 1e-6),
    "rf_raster_time_s": ("scanner/rf_raster_time_us", 1e-6),
    "adc_raster_time_s": ("scanner/adc_raster_time_us", 1e-6),
    "block_duration_raster_s": ("scanner/block_duration_raster_us", 1e-6),
    "rf_ringdown_time_s": ("scanner/rf_ringdown_time_us", 1e-6),
    "rf_dead_time_s": ("scanner/rf_dead_time_us", 1e-6),
    "adc_dead_time_s": ("scanner/adc_dead_time_us", 1e-6),
}


def load_scanner_parameters(settings) -> ScannerParameters:
    """Load a scanner profile from a QSettings-compatible object.

    Malformed individual values fall back to their defaults. If the resulting
    profile is inconsistent, the complete default profile is returned.
    """
    defaults = ScannerParameters()
    if settings is None:
        return defaults
    values = {}
    for name, (key, scale) in _SETTING_FIELDS.items():
        default_value = float(getattr(defaults, name)) / scale
        try:
            stored_value = float(settings.value(key, default_value))
        except (TypeError, ValueError):
            stored_value = default_value
        values[name] = stored_value * scale
    try:
        return ScannerParameters(**values)
    except ValueError:
        return defaults


def save_scanner_parameters(settings, parameters: ScannerParameters) -> None:
    """Persist a scanner profile in the same units that are shown in the UI."""
    parameters = ScannerParameters.from_mapping(parameters)
    for name, (key, scale) in _SETTING_FIELDS.items():
        settings.setValue(key, float(getattr(parameters, name)) / scale)


_PROFILES_KEY = "scanner/profiles"


def load_scanner_profiles(settings) -> dict[str, ScannerParameters]:
    """Load the named scanner profiles stored in a QSettings-compatible object.

    Profiles are stored as one JSON document. Entries that are malformed or
    physically inconsistent are skipped so one bad profile cannot hide the rest.
    """
    if settings is None:
        return {}
    try:
        raw = json.loads(str(settings.value(_PROFILES_KEY, "") or "{}"))
    except (TypeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    profiles: dict[str, ScannerParameters] = {}
    for name, values in raw.items():
        name = str(name).strip()
        if not name or not isinstance(values, dict):
            continue
        try:
            profiles[name] = ScannerParameters.from_mapping(values)
        except (TypeError, ValueError):
            continue
    return profiles


def save_scanner_profiles(settings, profiles: Mapping[str, ScannerParameters]) -> None:
    """Persist named scanner profiles as a JSON document."""
    payload = {
        str(name).strip(): ScannerParameters.from_mapping(values).to_dict()
        for name, values in profiles.items()
        if str(name).strip()
    }
    settings.setValue(_PROFILES_KEY, json.dumps(payload))
