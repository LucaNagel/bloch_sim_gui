"""Native spectral-spatial RF pulse design and Bloch validation.

This independent Python implementation follows the flyback construction
described by Kerr and Larson.  It does not call MATLAB or copy the reference
toolbox's bundled ``rf_tools`` code.  RF and gradient samples use the
simulator's frequency units: Hz and Hz/m, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
from scipy.signal import freqz, minimum_phase, remez

from ..units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm, ppm_to_hz


LARSON_TOOLBOX_URL = "https://github.com/LarsonLab/Spectral-Spatial-RF-Pulse-Design"
DESIGN_IMPLEMENTATION = "native-python-larson-kerr-flyback-v2"
SUPPORTED_SPECTRAL_PHASE_TYPES = ("linear", "minimum")
SUPPORTED_METHODS = ("flyback",)
SPECTRAL_SPATIAL_STATE_VERSION = 1


def _stable_id(prefix: str) -> str:
    """Return a project-stable identifier for one newly created object."""
    return f"{prefix}-{uuid4().hex}"


def _readonly_1d(values, dtype, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _ripple(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return value


@dataclass(frozen=True)
class SpectralBand:
    """One periodic pass band relative to the RF carrier.

    The pass band is constrained over ``width_hz``.  The interval immediately
    outside it, up to ``transition_width_hz`` on each side, is unconstrained.
    Ripple values are fractions of the strongest requested pass band.
    """

    center_hz: float
    width_hz: float
    amplitude: float = 1.0
    phase_deg: float = 0.0
    name: str = "Target"
    transition_width_hz: float | None = None
    passband_ripple: float = 0.01
    stopband_ripple: float = 0.01

    def __post_init__(self) -> None:
        if not np.all(
            np.isfinite((self.center_hz, self.width_hz, self.amplitude, self.phase_deg))
        ):
            raise ValueError("spectral-band values must be finite")
        if self.width_hz <= 0:
            raise ValueError("spectral-band width_hz must be positive")
        if self.amplitude < 0:
            raise ValueError("spectral-band amplitude must be non-negative")
        transition = self.transition_width_hz
        if transition is None:
            transition = 0.15 * float(self.width_hz)
        if not np.isfinite(transition) or transition < 0:
            raise ValueError(
                "spectral-band transition_width_hz must be non-negative and finite"
            )
        object.__setattr__(self, "transition_width_hz", float(transition))
        object.__setattr__(
            self,
            "passband_ripple",
            _ripple("spectral-band passband_ripple", self.passband_ripple),
        )
        object.__setattr__(
            self,
            "stopband_ripple",
            _ripple("spectral-band stopband_ripple", self.stopband_ripple),
        )

    @property
    def lower_hz(self) -> float:
        return float(self.center_hz - self.width_hz / 2.0)

    @property
    def upper_hz(self) -> float:
        return float(self.center_hz + self.width_hz / 2.0)

    @property
    def complex_amplitude(self) -> complex:
        return complex(self.amplitude * np.exp(1j * np.deg2rad(self.phase_deg)))

    @classmethod
    def from_value(cls, value) -> "SpectralBand":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        values = tuple(value)
        if len(values) == 2:
            return cls(values[0], values[1])
        if len(values) == 3:
            return cls(values[0], values[1], values[2])
        return cls(*values)

    def to_dict(self) -> dict:
        return {
            "center_hz": float(self.center_hz),
            "width_hz": float(self.width_hz),
            "amplitude": float(self.amplitude),
            "phase_deg": float(self.phase_deg),
            "name": str(self.name),
            "transition_width_hz": float(self.transition_width_hz),
            "passband_ripple": float(self.passband_ripple),
            "stopband_ripple": float(self.stopband_ripple),
        }


@dataclass(frozen=True)
class SpectralStopBand:
    """An optional explicit unwrapped stop-band constraint."""

    lower_hz: float
    upper_hz: float
    maximum_amplitude: float = 0.01
    name: str = "Stop band"

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_hz) or not np.isfinite(self.upper_hz):
            raise ValueError("stop-band edges must be finite")
        if self.upper_hz <= self.lower_hz:
            raise ValueError("stop-band upper_hz must exceed lower_hz")
        object.__setattr__(
            self,
            "maximum_amplitude",
            _ripple("stop-band maximum_amplitude", self.maximum_amplitude),
        )

    @classmethod
    def from_value(cls, value) -> "SpectralStopBand":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        return cls(*tuple(value))

    def to_dict(self) -> dict:
        return {
            "lower_hz": float(self.lower_hz),
            "upper_hz": float(self.upper_hz),
            "maximum_amplitude": float(self.maximum_amplitude),
            "name": str(self.name),
        }


@dataclass(frozen=True)
class AliasCompatibility:
    """Result of folding an unwrapped prescription periodically."""

    compatible: bool
    sampling_rate_hz: float
    reason: str


@dataclass(frozen=True)
class SpectralSpatialPulse:
    """A reproducible coupled RF/gradient excitation waveform."""

    rf_hz: np.ndarray
    rf_raster_s: float
    gradient_hz_per_m: np.ndarray
    gradient_raster_s: float
    reference_flip_angle_deg: float
    reference_thickness_m: float
    nucleus: str
    bands: tuple[SpectralBand, ...]
    name: str = "Spectral-spatial pulse"
    method: str = "flyback"
    spatial_time_bandwidth_product: float = 3.0
    subpulse_duration_s: float = 0.0
    subpulse_period_s: float = 0.0
    excitation_center_s: float = 0.0
    max_gradient_t_per_m: float = 0.0
    max_slew_t_per_m_s: float = 0.0
    reference_spatial_center_m: float = 0.0
    reference_spatial_transition_m: float = 0.0
    stopbands: tuple[SpectralStopBand, ...] = ()
    spectral_phase_type: str = "linear"
    spatial_passband_ripple: float = 0.01
    spatial_stopband_ripple: float = 0.01
    peak_b1_t: float = 0.0
    max_b1_t: float | None = None
    max_duration_s: float | None = None
    design_metadata: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        rf = _readonly_1d(self.rf_hz, np.complex128, "rf_hz")
        gradient = _readonly_1d(self.gradient_hz_per_m, np.float64, "gradient_hz_per_m")
        object.__setattr__(self, "rf_hz", rf)
        object.__setattr__(self, "gradient_hz_per_m", gradient)
        object.__setattr__(
            self, "bands", tuple(SpectralBand.from_value(value) for value in self.bands)
        )
        object.__setattr__(
            self,
            "stopbands",
            tuple(SpectralStopBand.from_value(value) for value in self.stopbands),
        )
        object.__setattr__(self, "design_metadata", dict(self.design_metadata))
        if self.nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            raise ValueError(f"unsupported nucleus {self.nucleus!r}")
        for parameter in (
            "rf_raster_s",
            "gradient_raster_s",
            "reference_flip_angle_deg",
            "reference_thickness_m",
            "spatial_time_bandwidth_product",
            "subpulse_duration_s",
            "subpulse_period_s",
        ):
            _positive_finite(parameter, getattr(self, parameter))
        for parameter in ("max_gradient_t_per_m", "max_slew_t_per_m_s", "peak_b1_t"):
            value = float(getattr(self, parameter))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{parameter} must be non-negative and finite")
        if self.max_b1_t is not None:
            _positive_finite("max_b1_t", self.max_b1_t)
        if self.max_duration_s is not None:
            _positive_finite("max_duration_s", self.max_duration_s)
        _ripple("spatial_passband_ripple", self.spatial_passband_ripple)
        _ripple("spatial_stopband_ripple", self.spatial_stopband_ripple)
        if self.spectral_phase_type not in SUPPORTED_SPECTRAL_PHASE_TYPES:
            raise ValueError("spectral_phase_type must be 'linear' or 'minimum'")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported spectral-spatial method {self.method!r}")
        if not self.bands:
            raise ValueError("bands must not be empty")
        if not np.isfinite(self.reference_spatial_center_m):
            raise ValueError("reference_spatial_center_m must be finite")
        if (
            not np.isfinite(self.reference_spatial_transition_m)
            or self.reference_spatial_transition_m <= 0
        ):
            raise ValueError(
                "reference_spatial_transition_m must be positive and finite"
            )
        rf_duration = rf.size * float(self.rf_raster_s)
        gradient_duration = gradient.size * float(self.gradient_raster_s)
        if not np.isclose(
            rf_duration,
            gradient_duration,
            rtol=0.0,
            atol=max(self.rf_raster_s, self.gradient_raster_s) / 2,
        ):
            raise ValueError("RF and gradient durations must match")
        if (
            not np.isfinite(self.excitation_center_s)
            or self.excitation_center_s < 0
            or self.excitation_center_s > rf_duration
        ):
            raise ValueError("excitation_center_s must lie within the pulse")

    @property
    def duration_s(self) -> float:
        return float(self.rf_hz.size * self.rf_raster_s)

    @property
    def spectral_alias_bandwidth_hz(self) -> float:
        return 1.0 / float(self.subpulse_period_s)

    def to_dict(self) -> dict:
        return {
            "rf_hz": np.array(self.rf_hz, copy=True),
            "rf_raster_s": float(self.rf_raster_s),
            "gradient_hz_per_m": np.array(self.gradient_hz_per_m, copy=True),
            "gradient_raster_s": float(self.gradient_raster_s),
            "reference_flip_angle_deg": float(self.reference_flip_angle_deg),
            "reference_thickness_m": float(self.reference_thickness_m),
            "nucleus": str(self.nucleus),
            "bands": [band.to_dict() for band in self.bands],
            "stopbands": [band.to_dict() for band in self.stopbands],
            "reference_spatial_center_m": float(self.reference_spatial_center_m),
            "reference_spatial_transition_m": float(
                self.reference_spatial_transition_m
            ),
            "name": str(self.name),
            "method": str(self.method),
            "spectral_phase_type": str(self.spectral_phase_type),
            "spatial_time_bandwidth_product": float(
                self.spatial_time_bandwidth_product
            ),
            "spatial_passband_ripple": float(self.spatial_passband_ripple),
            "spatial_stopband_ripple": float(self.spatial_stopband_ripple),
            "subpulse_duration_s": float(self.subpulse_duration_s),
            "subpulse_period_s": float(self.subpulse_period_s),
            "excitation_center_s": float(self.excitation_center_s),
            "max_gradient_t_per_m": float(self.max_gradient_t_per_m),
            "max_slew_t_per_m_s": float(self.max_slew_t_per_m_s),
            "peak_b1_t": float(self.peak_b1_t),
            "max_b1_t": None if self.max_b1_t is None else float(self.max_b1_t),
            "max_duration_s": (
                None if self.max_duration_s is None else float(self.max_duration_s)
            ),
            "design_metadata": dict(self.design_metadata),
        }

    @classmethod
    def from_mapping(cls, value) -> "SpectralSpatialPulse":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("spectral-spatial pulse must be a mapping")
        data = dict(value)
        if float(data.get("reference_spatial_transition_m", 0.0)) <= 0:
            thickness = float(data["reference_thickness_m"])
            tbw = float(data.get("spatial_time_bandwidth_product", 3.0))
            data["reference_spatial_transition_m"] = thickness / tbw
        return cls(**data)

    def scaled_waveforms(
        self,
        *,
        flip_angle_deg: float,
        thickness_m: float | None = None,
        spatial_offset_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return RF and gradient samples for one sequence excitation."""
        flip_angle_deg = float(flip_angle_deg)
        thickness_m = (
            self.reference_thickness_m if thickness_m is None else float(thickness_m)
        )
        if not np.isfinite(flip_angle_deg) or flip_angle_deg < 0:
            raise ValueError("flip_angle_deg must be non-negative and finite")
        if not np.isfinite(thickness_m) or thickness_m <= 0:
            raise ValueError("thickness_m must be positive and finite")
        if spatial_offset_m is None:
            spatial_offset_m = self.reference_spatial_center_m
        if not np.isfinite(spatial_offset_m):
            raise ValueError("spatial_offset_m must be finite")
        rf = np.asarray(self.rf_hz) * (flip_angle_deg / self.reference_flip_angle_deg)
        if self.max_b1_t is not None:
            gamma_hz_per_t = float(NUCLEUS_GAMMA_HZ_PER_T[self.nucleus])
            peak_b1_t = float(np.max(np.abs(rf)) / gamma_hz_per_t)
            if peak_b1_t > self.max_b1_t * (1 + 1e-10):
                raise ValueError(
                    "target flip angle makes the spectral-spatial RF exceed "
                    "the stored B1 limit"
                )
        gradient = np.asarray(self.gradient_hz_per_m) * (
            self.reference_thickness_m / thickness_m
        )
        if spatial_offset_m:
            gradient_on_rf = _resample_piecewise_constant(
                gradient,
                self.gradient_raster_s,
                rf.size,
                self.rf_raster_s,
            )
            k_future = np.cumsum(gradient_on_rf[::-1])[::-1] * self.rf_raster_s
            rf = rf * np.exp(-2j * np.pi * k_future * float(spatial_offset_m))
        return np.asarray(rf, np.complex128), np.asarray(gradient, float)


@dataclass
class SpectralSpatialPulseLibrary:
    """Ordered collection of reusable SPSP pulses with project-stable IDs.

    The library owns pulse identities. Schedule entries refer to these IDs and
    never contain a private waveform copy, so editing one library pulse updates
    every schedule occurrence in a controlled and serializable way.
    """

    pulses: dict[str, SpectralSpatialPulse] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, SpectralSpatialPulse] = {}
        for pulse_id, pulse in dict(self.pulses).items():
            pulse_id = str(pulse_id).strip()
            if not pulse_id:
                raise ValueError("spectral-spatial pulse IDs must not be empty")
            if pulse_id in normalized:
                raise ValueError(f"duplicate spectral-spatial pulse ID {pulse_id!r}")
            normalized[pulse_id] = SpectralSpatialPulse.from_mapping(pulse)
        self.pulses = normalized

    def __len__(self) -> int:
        return len(self.pulses)

    def __iter__(self):
        return iter(self.pulses)

    def get(self, pulse_id: str) -> SpectralSpatialPulse:
        try:
            return self.pulses[str(pulse_id)]
        except KeyError as exc:
            raise KeyError(f"unknown spectral-spatial pulse ID {pulse_id!r}") from exc

    def add(
        self, pulse: SpectralSpatialPulse | Mapping, *, pulse_id: str | None = None
    ) -> str:
        pulse = SpectralSpatialPulse.from_mapping(pulse)
        pulse_id = _stable_id("spsp") if pulse_id is None else str(pulse_id).strip()
        if not pulse_id:
            raise ValueError("spectral-spatial pulse ID must not be empty")
        if pulse_id in self.pulses:
            raise ValueError(f"spectral-spatial pulse ID {pulse_id!r} already exists")
        self.pulses[pulse_id] = pulse
        return pulse_id

    def replace(self, pulse_id: str, pulse: SpectralSpatialPulse | Mapping) -> None:
        self.get(pulse_id)
        self.pulses[str(pulse_id)] = SpectralSpatialPulse.from_mapping(pulse)

    def duplicate(self, pulse_id: str, *, name: str | None = None) -> str:
        pulse = self.get(pulse_id)
        duplicate = replace(pulse, name=name or f"{pulse.name} copy")
        return self.add(duplicate)

    def remove(self, pulse_id: str) -> SpectralSpatialPulse:
        self.get(pulse_id)
        return self.pulses.pop(str(pulse_id))

    def to_dict(self) -> dict:
        return {
            "version": SPECTRAL_SPATIAL_STATE_VERSION,
            "pulses": [
                {"id": pulse_id, "pulse": pulse.to_dict()}
                for pulse_id, pulse in self.pulses.items()
            ],
        }

    @classmethod
    def from_mapping(cls, value) -> "SpectralSpatialPulseLibrary":
        if isinstance(value, cls):
            return cls(dict(value.pulses))
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("spectral-spatial pulse library must be a mapping")
        data = dict(value)
        raw = data.get("pulses", data)
        if isinstance(raw, Mapping):
            return cls(
                {
                    str(pulse_id): SpectralSpatialPulse.from_mapping(pulse)
                    for pulse_id, pulse in raw.items()
                }
            )
        pulses = {}
        for item in raw:
            if not isinstance(item, Mapping) or "id" not in item:
                raise ValueError("each pulse-library item needs id and pulse fields")
            pulse_id = str(item["id"])
            pulse_value = item.get("pulse", item.get("value"))
            if pulse_value is None:
                raise ValueError(f"pulse-library item {pulse_id!r} has no pulse")
            if pulse_id in pulses:
                raise ValueError(f"duplicate spectral-spatial pulse ID {pulse_id!r}")
            pulses[pulse_id] = SpectralSpatialPulse.from_mapping(pulse_value)
        return cls(pulses)


@dataclass(frozen=True)
class SpectralSpatialScheduleEntry:
    """One ordered SPSP use with either amplitude or flip-angle scaling.

    If ``flip_angle_deg`` is set, it is the requested flip angle of the
    strongest pass band and ``amplitude_scale`` is informational only. If it is
    ``None``, the pulse reference waveform is multiplied by
    ``amplitude_scale``. ``repeat_count`` expands adjacent uses before cyclic
    schedule selection.
    """

    pulse_id: str
    display_name: str = ""
    target_metabolite: str = ""
    amplitude_scale: float = 1.0
    flip_angle_deg: float | None = None
    enabled: bool = True
    repeat_count: int = 1
    entry_id: str = field(default_factory=lambda: _stable_id("step"))

    def __post_init__(self) -> None:
        if not str(self.pulse_id).strip():
            raise ValueError("schedule pulse_id must not be empty")
        if not str(self.entry_id).strip():
            raise ValueError("schedule entry_id must not be empty")
        amplitude_scale = float(self.amplitude_scale)
        if not np.isfinite(amplitude_scale) or amplitude_scale < 0:
            raise ValueError("schedule amplitude_scale must be non-negative and finite")
        object.__setattr__(self, "amplitude_scale", amplitude_scale)
        if self.flip_angle_deg is not None:
            flip_angle_deg = float(self.flip_angle_deg)
            if not np.isfinite(flip_angle_deg) or flip_angle_deg < 0:
                raise ValueError(
                    "schedule flip_angle_deg must be non-negative and finite or None"
                )
            object.__setattr__(self, "flip_angle_deg", flip_angle_deg)
        repeat_count = int(self.repeat_count)
        if repeat_count < 1 or repeat_count != self.repeat_count:
            raise ValueError("schedule repeat_count must be a positive integer")
        object.__setattr__(self, "repeat_count", repeat_count)
        object.__setattr__(self, "pulse_id", str(self.pulse_id).strip())
        object.__setattr__(self, "entry_id", str(self.entry_id).strip())
        object.__setattr__(self, "display_name", str(self.display_name))
        object.__setattr__(self, "target_metabolite", str(self.target_metabolite))

    @property
    def scale_mode(self) -> str:
        return "flip_angle" if self.flip_angle_deg is not None else "amplitude"

    def effective_flip_angle_deg(self, pulse: SpectralSpatialPulse | Mapping) -> float:
        pulse = SpectralSpatialPulse.from_mapping(pulse)
        if self.flip_angle_deg is not None:
            return float(self.flip_angle_deg)
        return float(pulse.reference_flip_angle_deg * self.amplitude_scale)

    def effective_amplitude_scale(self, pulse: SpectralSpatialPulse | Mapping) -> float:
        pulse = SpectralSpatialPulse.from_mapping(pulse)
        return self.effective_flip_angle_deg(pulse) / pulse.reference_flip_angle_deg

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "pulse_id": self.pulse_id,
            "display_name": self.display_name,
            "target_metabolite": self.target_metabolite,
            "amplitude_scale": float(self.amplitude_scale),
            "flip_angle_deg": (
                None if self.flip_angle_deg is None else float(self.flip_angle_deg)
            ),
            "enabled": bool(self.enabled),
            "repeat_count": int(self.repeat_count),
        }

    @classmethod
    def from_mapping(cls, value) -> "SpectralSpatialScheduleEntry":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("spectral-spatial schedule entry must be a mapping")
        data = dict(value)
        if "repeat_count" not in data and "repetitions" in data:
            data["repeat_count"] = data.pop("repetitions")
        if "target_metabolite" not in data and "target_name" in data:
            data["target_metabolite"] = data.pop("target_name")
        return cls(**data)


@dataclass
class SpectralSpatialSchedule:
    """Ordered, optionally cyclic sequence of SPSP library references."""

    entries: list[SpectralSpatialScheduleEntry] = field(default_factory=list)
    cyclic: bool = True
    selection_level: str = "frame"

    def __post_init__(self) -> None:
        self.entries = [
            SpectralSpatialScheduleEntry.from_mapping(entry) for entry in self.entries
        ]
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("spectral-spatial schedule entry IDs must be unique")
        self.selection_level = str(self.selection_level).strip().lower()
        if self.selection_level not in {"frame", "tr"}:
            raise ValueError("schedule selection_level must be 'frame' or 'tr'")
        self.cyclic = bool(self.cyclic)

    def add(self, entry: SpectralSpatialScheduleEntry | Mapping, index=None) -> str:
        entry = SpectralSpatialScheduleEntry.from_mapping(entry)
        if any(existing.entry_id == entry.entry_id for existing in self.entries):
            raise ValueError(f"schedule entry ID {entry.entry_id!r} already exists")
        if index is None:
            self.entries.append(entry)
        else:
            self.entries.insert(int(index), entry)
        return entry.entry_id

    def remove(self, index: int) -> SpectralSpatialScheduleEntry:
        return self.entries.pop(int(index))

    def duplicate(self, index: int) -> str:
        source = self.entries[int(index)]
        duplicate = replace(
            source,
            entry_id=_stable_id("step"),
            display_name=(
                f"{source.display_name} copy" if source.display_name else "Copy"
            ),
        )
        self.entries.insert(int(index) + 1, duplicate)
        return duplicate.entry_id

    def move(self, index: int, new_index: int) -> None:
        index = int(index)
        new_index = int(new_index)
        if not 0 <= index < len(self.entries) or not 0 <= new_index < len(self.entries):
            raise IndexError("schedule move index is out of range")
        self.entries.insert(new_index, self.entries.pop(index))

    def expanded_entries(self) -> tuple[SpectralSpatialScheduleEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.enabled
            for _ in range(entry.repeat_count)
        )

    def entry_for_index(self, index: int) -> tuple[int, SpectralSpatialScheduleEntry]:
        expanded = self.expanded_entries()
        if not expanded:
            raise ValueError("spectral-spatial schedule has no enabled entries")
        index = int(index)
        if index < 0:
            raise IndexError("schedule index must be non-negative")
        if self.cyclic:
            index %= len(expanded)
        elif index >= len(expanded):
            raise IndexError("non-cyclic spectral-spatial schedule is exhausted")
        return index, expanded[index]

    def referenced_pulse_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(entry.pulse_id for entry in self.entries))

    def to_dict(self) -> dict:
        return {
            "version": SPECTRAL_SPATIAL_STATE_VERSION,
            "cyclic": bool(self.cyclic),
            "selection_level": self.selection_level,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_mapping(cls, value) -> "SpectralSpatialSchedule":
        if isinstance(value, cls):
            return cls(list(value.entries), value.cyclic, value.selection_level)
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("spectral-spatial schedule must be a mapping")
        data = dict(value)
        return cls(
            entries=list(data.get("entries", ())),
            cyclic=bool(data.get("cyclic", True)),
            selection_level=str(data.get("selection_level", "frame")),
        )


def migrate_spectral_spatial_state(
    *, library=None, schedule=None, legacy_pulse=None
) -> tuple[SpectralSpatialPulseLibrary, SpectralSpatialSchedule, bool]:
    """Normalize new state or migrate the historical singleton representation.

    Returns ``(library, schedule, migrated_legacy)``. The deterministic legacy
    IDs make repeated loads of an old project stable before it is first saved
    in the new format.
    """
    pulse_library = SpectralSpatialPulseLibrary.from_mapping(library)
    pulse_schedule = SpectralSpatialSchedule.from_mapping(schedule)
    migrated = False
    if not pulse_library.pulses and legacy_pulse:
        pulse = SpectralSpatialPulse.from_mapping(legacy_pulse)
        pulse_id = "legacy-single-pulse"
        pulse_library.add(pulse, pulse_id=pulse_id)
        pulse_schedule = SpectralSpatialSchedule(
            entries=[
                SpectralSpatialScheduleEntry(
                    pulse_id=pulse_id,
                    display_name=pulse.name,
                    target_metabolite=pulse.bands[0].name if pulse.bands else "",
                    amplitude_scale=1.0,
                    entry_id="legacy-single-entry",
                )
            ]
        )
        migrated = True
    missing = set(pulse_schedule.referenced_pulse_ids()) - set(pulse_library.pulses)
    if missing:
        raise ValueError(
            "spectral-spatial schedule references missing pulse IDs: "
            + ", ".join(sorted(missing))
        )
    return pulse_library, pulse_schedule, migrated


def validate_spectral_spatial_schedule(
    library,
    schedule,
    *,
    nucleus: str,
    thickness_m: float | None = None,
    scanner_parameters=None,
    maximum_duration_s: float | None = None,
) -> tuple[str, ...]:
    """Validate every enabled step after its exact RF/spatial scaling.

    Hardware violations raise ``ValueError``. An unmet sampled FIR design is
    returned as an explicit warning so callers can display it instead of
    presenting the requested stop-band attenuation as guaranteed.
    """
    pulse_library = SpectralSpatialPulseLibrary.from_mapping(library)
    pulse_schedule = SpectralSpatialSchedule.from_mapping(schedule)
    expanded = pulse_schedule.expanded_entries()
    if not expanded:
        raise ValueError("spectral-spatial schedule has no enabled entries")
    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        raise ValueError(f"unsupported nucleus {nucleus!r}")
    scanner = None
    if scanner_parameters is not None:
        from .scanner import ScannerParameters

        scanner = ScannerParameters.from_mapping(scanner_parameters)
    warnings = []
    for step_index, entry in enumerate(expanded):
        pulse = pulse_library.get(entry.pulse_id)
        if pulse.nucleus != nucleus:
            raise ValueError(
                f"schedule step {step_index} uses a {pulse.nucleus} pulse, not {nucleus}"
            )
        target_thickness = (
            pulse.reference_thickness_m if thickness_m is None else float(thickness_m)
        )
        flip = entry.effective_flip_angle_deg(pulse)
        rf, gradient = pulse.scaled_waveforms(
            flip_angle_deg=flip, thickness_m=target_thickness
        )
        gamma = float(NUCLEUS_GAMMA_HZ_PER_T[nucleus])
        peak_b1_t = float(np.max(np.abs(rf)) / gamma)
        if pulse.max_b1_t is not None and peak_b1_t > pulse.max_b1_t * (1 + 1e-10):
            raise ValueError(f"schedule step {step_index} exceeds the pulse B1 limit")
        if maximum_duration_s is not None and pulse.duration_s > float(
            maximum_duration_s
        ) * (1 + 1e-12):
            raise ValueError(f"schedule step {step_index} exceeds the duration limit")
        if scanner is not None:
            physical_gradient = np.asarray(gradient, dtype=float) / gamma
            if np.max(np.abs(physical_gradient)) > scanner.max_grad_mtm * 1e-3 * (
                1 + 1e-10
            ):
                raise ValueError(
                    f"schedule step {step_index} gradient exceeds the scanner amplitude limit"
                )
            slew = (
                np.concatenate(
                    (
                        [2.0 * physical_gradient[0]],
                        np.diff(physical_gradient),
                        [-2.0 * physical_gradient[-1]],
                    )
                )
                / pulse.gradient_raster_s
            )
            if np.max(np.abs(slew)) > scanner.max_slew_tms * (1 + 1e-10):
                raise ValueError(
                    f"schedule step {step_index} gradient exceeds the scanner slew limit"
                )
        if (
            pulse.design_metadata.get("metrics", {}).get(
                "all_filter_specifications_met"
            )
            is False
        ):
            warnings.append(
                f"Schedule step {step_index + 1} ({entry.display_name or pulse.name}) "
                "does not meet all sampled pass-/stop-band specifications."
            )
    return tuple(warnings)


def _resample_piecewise_constant(
    samples: np.ndarray,
    source_raster_s: float,
    target_count: int,
    target_raster_s: float,
) -> np.ndarray:
    times = (np.arange(target_count, dtype=float) + 0.5) * target_raster_s
    indices = np.floor(times / source_raster_s).astype(int)
    return np.asarray(samples)[np.clip(indices, 0, len(samples) - 1)]


def _trapezoid_samples(amplitude: float, rise_samples: int, flat_samples: int):
    ramp = (np.arange(rise_samples, dtype=float) + 0.5) / rise_samples
    return np.concatenate(
        (amplitude * ramp, np.full(flat_samples, amplitude), amplitude * ramp[::-1])
    )


def _minimum_time_rewinder(
    area_t_s_per_m: float,
    *,
    raster_s: float,
    max_gradient_t_per_m: float,
    max_slew_t_per_m_s: float,
) -> np.ndarray:
    """Return the shortest rasterized trapezoid with the requested area."""
    area = abs(float(area_t_s_per_m))
    if area == 0:
        return np.zeros(1, dtype=float)
    lower = max(2, int(np.ceil(2 * np.sqrt(area / max_slew_t_per_m_s) / raster_s)))
    for total_samples in range(max(2, lower - 4), 1_000_001):
        for rise_samples in range(1, total_samples // 2 + 1):
            flat_samples = total_samples - 2 * rise_samples
            amplitude = area / ((rise_samples + flat_samples) * raster_s)
            if amplitude > max_gradient_t_per_m * (1 + 1e-12):
                continue
            if amplitude / (rise_samples * raster_s) > max_slew_t_per_m_s * (1 + 1e-12):
                continue
            return -_trapezoid_samples(amplitude, rise_samples, flat_samples)
    raise ValueError("no rasterized flyback rewinder satisfies the hardware limits")


def _wrap_frequency(frequency_hz, alias_bandwidth_hz):
    half = alias_bandwidth_hz / 2.0
    return (np.asarray(frequency_hz) + half) % alias_bandwidth_hz - half


def _periodic_interval_mask(grid, lower_hz, upper_hz, sampling_rate_hz):
    width = float(upper_hz - lower_hz)
    if width >= sampling_rate_hz:
        return np.ones(np.asarray(grid).shape, dtype=bool)
    center = (float(lower_hz) + float(upper_hz)) / 2.0
    return np.abs(_wrap_frequency(np.asarray(grid) - center, sampling_rate_hz)) <= (
        width / 2.0
    )


def check_spectral_alias_compatibility(
    bands: Sequence[SpectralBand | Mapping | Sequence[float]],
    sampling_rate_hz: float,
    *,
    stopbands: Sequence[SpectralStopBand | Mapping | Sequence[float]] = (),
    grid_points: int = 32768,
) -> AliasCompatibility:
    """Check whether pass/stop specifications remain consistent after folding."""
    bands = tuple(SpectralBand.from_value(value) for value in bands)
    stopbands = tuple(SpectralStopBand.from_value(value) for value in stopbands)
    sampling_rate_hz = _positive_finite("sampling_rate_hz", sampling_rate_hz)
    grid = np.linspace(
        -sampling_rate_hz / 2,
        sampling_rate_hz / 2,
        max(2048, int(grid_points)),
        endpoint=False,
    )
    assigned = np.zeros(grid.size, dtype=bool)
    desired = np.zeros(grid.size, dtype=np.complex128)
    for band in bands:
        if band.width_hz + 2 * band.transition_width_hz >= sampling_rate_hz:
            return AliasCompatibility(
                False,
                sampling_rate_hz,
                f"pass band {band.name!r} and its transitions span the alias interval",
            )
        mask = _periodic_interval_mask(
            grid, band.lower_hz, band.upper_hz, sampling_rate_hz
        )
        overlap = mask & assigned
        if np.any(overlap):
            tolerance = max(band.passband_ripple, 1e-9)
            if np.any(np.abs(desired[overlap] - band.complex_amplitude) > tolerance):
                return AliasCompatibility(
                    False,
                    sampling_rate_hz,
                    f"pass band {band.name!r} aliases onto a different pass target",
                )
        desired[mask] = band.complex_amplitude
        assigned |= mask
    for stopband in stopbands:
        mask = _periodic_interval_mask(
            grid, stopband.lower_hz, stopband.upper_hz, sampling_rate_hz
        )
        if np.any(mask & assigned):
            return AliasCompatibility(
                False,
                sampling_rate_hz,
                f"stop band {stopband.name!r} aliases onto a pass band",
            )
    return AliasCompatibility(True, sampling_rate_hz, "compatible")


def find_compatible_spectral_sampling_rate(
    bands: Sequence[SpectralBand | Mapping | Sequence[float]],
    *,
    minimum_period_samples: int,
    raster_s: float,
    stopbands: Sequence[SpectralStopBand | Mapping | Sequence[float]] = (),
    maximum_period_samples: int | None = None,
) -> tuple[float, AliasCompatibility, int]:
    """Find the highest raster-compatible sampling rate without alias conflicts."""
    raster_s = _positive_finite("raster_s", raster_s)
    minimum_period_samples = int(minimum_period_samples)
    if minimum_period_samples <= 0:
        raise ValueError("minimum_period_samples must be positive")
    if maximum_period_samples is None:
        maximum_period_samples = 8 * minimum_period_samples
    maximum_period_samples = int(maximum_period_samples)
    if maximum_period_samples < minimum_period_samples:
        raise ValueError("maximum period is shorter than the gradient waveform")
    last = None
    for samples in range(minimum_period_samples, maximum_period_samples + 1):
        rate = 1.0 / (samples * raster_s)
        last = check_spectral_alias_compatibility(bands, rate, stopbands=stopbands)
        if last.compatible:
            return rate, last, samples
    assert last is not None
    raise ValueError(
        "no alias-compatible spectral sampling rate exists within the duration "
        f"limit; last conflict: {last.reason}"
    )


def _spectral_design_grid(bands, stopbands, sampling_rate_hz, grid_points):
    grid = np.linspace(
        -sampling_rate_hz / 2,
        sampling_rate_hz / 2,
        grid_points,
        endpoint=False,
    )
    desired = np.zeros(grid_points, dtype=np.complex128)
    weights = np.full(
        grid_points,
        1.0 / min(band.stopband_ripple for band in bands),
        dtype=float,
    )
    transition = np.zeros(grid_points, dtype=bool)
    for band in bands:
        distance = np.abs(_wrap_frequency(grid - band.center_hz, sampling_rate_hz))
        inside = distance <= band.width_hz / 2.0
        edge = (distance > band.width_hz / 2.0) & (
            distance <= band.width_hz / 2.0 + band.transition_width_hz
        )
        # The sequence simulator uses the MRI rotating-frame convention in
        # which the resulting transverse phase is conjugate to the complex RF
        # spectral response. Design the RF response against the conjugated
        # target so ``SpectralBand.phase_deg`` remains the requested Mxy phase.
        desired[inside] = np.conj(band.complex_amplitude)
        weights[inside] = 1.0 / band.passband_ripple
        transition |= edge
    for stopband in stopbands:
        mask = _periodic_interval_mask(
            grid, stopband.lower_hz, stopband.upper_hz, sampling_rate_hz
        )
        weights[mask] = np.maximum(weights[mask], 1.0 / stopband.maximum_amplitude)
    weights[transition] = 0.0
    return grid, desired, weights


def _linear_phase_spectral_weights(bands, stopbands, count, sampling_rate_hz):
    points = max(4096, 256 * count)
    grid, desired, weights = _spectral_design_grid(
        bands, stopbands, sampling_rate_hz, points
    )
    centered_index = np.arange(count, dtype=float) - (count - 1) / 2.0
    matrix = np.exp(
        -2j * np.pi * grid[:, None] * centered_index[None, :] / sampling_rate_hz
    )
    sqrt_weight = np.sqrt(weights / np.max(weights))
    weighted = matrix * sqrt_weight[:, None]
    target = desired * sqrt_weight
    regularization = 1e-8 * np.eye(count)
    return np.linalg.solve(
        weighted.conj().T @ weighted + regularization,
        weighted.conj().T @ target,
    )


def _minimum_phase_lowpass(count, pass_edge, stop_edge, pass_ripple, stop_ripple):
    pass_edge = max(float(pass_edge), 1e-8)
    stop_edge = min(float(stop_edge), 0.5 - 1e-8)
    if stop_edge <= pass_edge:
        raise ValueError("spectral transition is too narrow for the sampling rate")
    try:
        linear = remez(
            count,
            (0.0, pass_edge, stop_edge, 0.5),
            (1.0, 0.0),
            weight=(1.0 / pass_ripple, 1.0 / stop_ripple),
            fs=1.0,
            maxiter=200,
        )
    except ValueError:
        index = np.arange(count, dtype=float) - (count - 1) / 2.0
        cutoff = (pass_edge + stop_edge) / 2.0
        linear = 2.0 * cutoff * np.sinc(2.0 * cutoff * index)
        linear *= np.kaiser(count, 7.5)
    if count <= 2:
        result = np.asarray(linear, dtype=float)
    else:
        try:
            result = minimum_phase(linear, method="homomorphic", half=False)
        except (TypeError, ValueError):
            half = minimum_phase(linear, method="homomorphic")
            result = np.pad(half, (0, max(0, count - len(half))))[:count]
    total = np.sum(result)
    if not np.isfinite(total) or abs(total) <= np.finfo(float).eps:
        raise ValueError("minimum-phase spectral factorization failed")
    return np.asarray(result / total, dtype=np.complex128)


def _minimum_phase_spectral_weights(bands, count, sampling_rate_hz):
    times = np.arange(count, dtype=float) / sampling_rate_hz
    basis = []
    for band in bands:
        prototype = _minimum_phase_lowpass(
            count,
            band.width_hz / (2.0 * sampling_rate_hz),
            (band.width_hz / 2.0 + band.transition_width_hz) / sampling_rate_hz,
            band.passband_ripple,
            band.stopband_ripple,
        )
        basis.append(prototype * np.exp(2j * np.pi * band.center_hz * times))
    basis = np.asarray(basis, dtype=np.complex128).T
    centers = np.asarray([band.center_hz for band in bands], dtype=float)
    response = np.exp(-2j * np.pi * centers[:, None] * times[None, :]) @ basis
    target = np.conj(np.asarray([band.complex_amplitude for band in bands]))
    weights = np.linalg.solve(
        response.conj().T @ response + 1e-8 * np.eye(len(bands)),
        response.conj().T @ target,
    )
    return basis @ weights


def _spectral_weights(bands, stopbands, count, sampling_rate_hz, phase_type):
    if phase_type == "linear":
        result = _linear_phase_spectral_weights(
            bands, stopbands, count, sampling_rate_hz
        )
    else:
        result = _minimum_phase_spectral_weights(bands, count, sampling_rate_hz)
    return np.asarray(result, dtype=np.complex128)


def _design_spatial_subpulse(
    sample_count,
    *,
    time_bandwidth_product,
    slice_thickness_m,
    transition_width_m,
    passband_ripple,
    stopband_ripple,
):
    pass_edge = time_bandwidth_product / (2.0 * sample_count)
    stop_edge = pass_edge + (
        time_bandwidth_product * transition_width_m / (slice_thickness_m * sample_count)
    )
    if not 0 < pass_edge < stop_edge < 0.5:
        raise ValueError(
            "spatial transition/TBW cannot be represented on the RF raster; "
            "increase subpulse duration or reduce the spatial TBW"
        )
    try:
        envelope = remez(
            sample_count,
            (0.0, pass_edge, stop_edge, 0.5),
            (1.0, 0.0),
            weight=(1.0 / passband_ripple, 1.0 / stopband_ripple),
            fs=1.0,
            maxiter=200,
        )
    except ValueError as exc:
        raise ValueError(
            "spatial equiripple subpulse design did not converge; increase the "
            "subpulse duration or transition width"
        ) from exc
    coherent = float(np.sum(envelope))
    if not np.isfinite(coherent) or abs(coherent) <= np.finfo(float).eps:
        raise ValueError("spatial subpulse has no coherent integral")
    return np.asarray(envelope / coherent, dtype=np.complex128)


def _spatial_response_metrics(
    envelope,
    *,
    time_bandwidth_product,
    slice_thickness_m,
    transition_width_m,
    passband_ripple,
    stopband_ripple,
):
    sample_count = len(envelope)
    pass_edge = time_bandwidth_product / (2.0 * sample_count)
    stop_edge = pass_edge + (
        time_bandwidth_product * transition_width_m / (slice_thickness_m * sample_count)
    )
    angular_frequency, response = freqz(envelope, worN=16385, whole=False)
    frequencies = angular_frequency / (2.0 * np.pi)
    pass_error = np.abs(np.abs(response[frequencies <= pass_edge]) - 1.0)
    stop_amplitude = np.abs(response[frequencies >= stop_edge])
    pass_max = float(np.max(pass_error))
    stop_max = float(np.max(stop_amplitude))
    return {
        "spatial_passband_rms_error": float(np.sqrt(np.mean(pass_error**2))),
        "spatial_passband_max_error": pass_max,
        "spatial_stopband_max_amplitude": stop_max,
        "spatial_filter_specifications_met": bool(
            pass_max <= passband_ripple * (1 + 1e-6)
            and stop_max <= stopband_ripple * (1 + 1e-6)
        ),
    }


def _response_metrics(coefficients, bands, stopbands, sampling_rate_hz, phase_type):
    grid, desired, weights = _spectral_design_grid(
        bands, stopbands, sampling_rate_hz, 16384
    )
    if phase_type == "linear":
        sample_index = np.arange(len(coefficients)) - (len(coefficients) - 1) / 2
    else:
        sample_index = np.arange(len(coefficients))
    response = (
        np.exp(-2j * np.pi * grid[:, None] * sample_index[None, :] / sampling_rate_hz)
        @ coefficients
    )
    constrained_pass = np.abs(desired) > 0
    constrained_stop = (weights > 0) & ~constrained_pass
    if phase_type == "minimum":
        errors = np.abs(
            np.abs(response[constrained_pass]) - np.abs(desired[constrained_pass])
        )
    else:
        errors = np.abs(response[constrained_pass] - desired[constrained_pass])
    center_matrix = np.exp(
        -2j
        * np.pi
        * np.asarray([band.center_hz for band in bands])[:, None]
        * sample_index[None, :]
        / sampling_rate_hz
    )
    center_response = center_matrix @ coefficients
    center_target = np.conj(np.asarray([band.complex_amplitude for band in bands]))
    phase_error_deg = np.rad2deg(np.angle(center_response * np.conj(center_target)))
    return {
        "spectral_passband_rms_error": (
            float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0
        ),
        "spectral_passband_max_error": float(np.max(errors)) if errors.size else 0.0,
        "spectral_stopband_max_amplitude": (
            float(np.max(np.abs(response[constrained_stop])))
            if np.any(constrained_stop)
            else 0.0
        ),
        "spectral_band_center_max_amplitude_error": float(
            np.max(np.abs(np.abs(center_response) - np.abs(center_target)))
        ),
        "spectral_band_center_max_phase_error_deg": float(
            np.max(np.abs(phase_error_deg))
        ),
    }


def design_spectral_spatial_pulse(
    *,
    bands: Sequence[SpectralBand | Mapping | Sequence[float]],
    stopbands: Sequence[SpectralStopBand | Mapping | Sequence[float]] = (),
    flip_angle_deg: float = 30.0,
    slice_thickness_m: float = 10e-3,
    spatial_time_bandwidth_product: float = 3.0,
    spatial_transition_m: float | None = None,
    spatial_passband_ripple: float = 0.01,
    spatial_stopband_ripple: float = 0.01,
    subpulse_duration_s: float = 0.7e-3,
    subpulse_count: int = 11,
    spectral_phase_type: str = "linear",
    spectral_sampling_rate_hz: float | None = None,
    rf_raster_s: float = 1e-6,
    gradient_raster_s: float = 10e-6,
    max_gradient_t_per_m: float = 0.05,
    max_slew_t_per_m_s: float = 200.0,
    max_b1_t: float | None = None,
    max_duration_s: float | None = None,
    nucleus: str = "C13",
    name: str = "Spectral-spatial pulse",
    spatial_center_m: float = 0.0,
    method: str = "flyback",
) -> SpectralSpatialPulse:
    """Design a constrained native-Python flyback SPSP excitation.

    ``echo_planar`` is reserved but deliberately not implemented until
    numerical reference vectors are available for that trajectory.
    """
    method = str(method).strip().lower().replace("-", "_")
    if method in {"ep", "epi", "echo_planar", "bipolar"}:
        raise NotImplementedError(
            "echo_planar/bipolar SPSP design is not implemented; use method='flyback'"
        )
    if method != "flyback":
        raise ValueError("method must be 'flyback'")
    spectral_phase_type = str(spectral_phase_type).strip().lower()
    if spectral_phase_type not in SUPPORTED_SPECTRAL_PHASE_TYPES:
        raise ValueError("spectral_phase_type must be 'linear' or 'minimum'")
    bands = tuple(SpectralBand.from_value(value) for value in bands)
    stopbands = tuple(SpectralStopBand.from_value(value) for value in stopbands)
    if not bands:
        raise ValueError("at least one spectral band is required")
    if any(band.transition_width_hz <= 0 for band in bands):
        raise ValueError(
            "new designs require a positive transition_width_hz for every band"
        )
    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        raise ValueError(f"unsupported nucleus {nucleus!r}")
    if not np.isfinite(spatial_center_m):
        raise ValueError("spatial_center_m must be finite")
    flip_angle_deg = _positive_finite("flip_angle_deg", flip_angle_deg)
    slice_thickness_m = _positive_finite("slice_thickness_m", slice_thickness_m)
    spatial_time_bandwidth_product = _positive_finite(
        "spatial_time_bandwidth_product", spatial_time_bandwidth_product
    )
    subpulse_duration_s = _positive_finite("subpulse_duration_s", subpulse_duration_s)
    rf_raster_s = _positive_finite("rf_raster_s", rf_raster_s)
    gradient_raster_s = _positive_finite("gradient_raster_s", gradient_raster_s)
    max_gradient_t_per_m = _positive_finite(
        "max_gradient_t_per_m", max_gradient_t_per_m
    )
    max_slew_t_per_m_s = _positive_finite("max_slew_t_per_m_s", max_slew_t_per_m_s)
    spatial_passband_ripple = _ripple(
        "spatial_passband_ripple", spatial_passband_ripple
    )
    spatial_stopband_ripple = _ripple(
        "spatial_stopband_ripple", spatial_stopband_ripple
    )
    if max_b1_t is not None:
        max_b1_t = _positive_finite("max_b1_t", max_b1_t)
    if max_duration_s is not None:
        max_duration_s = _positive_finite("max_duration_s", max_duration_s)
    if spatial_transition_m is None:
        spatial_transition_m = slice_thickness_m / spatial_time_bandwidth_product
    spatial_transition_m = _positive_finite(
        "spatial_transition_m", spatial_transition_m
    )
    if (
        isinstance(subpulse_count, (bool, np.bool_))
        or int(subpulse_count) != subpulse_count
    ):
        raise ValueError("subpulse_count must be an integer")
    subpulse_count = int(subpulse_count)
    if subpulse_count < 3:
        raise ValueError("subpulse_count must be at least 3")

    ratio = gradient_raster_s / rf_raster_s
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
        raise ValueError("gradient_raster_s must be an integer multiple of rf_raster_s")
    gamma_hz_per_t = float(NUCLEUS_GAMMA_HZ_PER_T[nucleus])
    flat_gradient_samples = max(
        1, int(np.ceil(subpulse_duration_s / gradient_raster_s))
    )
    actual_subpulse_duration_s = flat_gradient_samples * gradient_raster_s
    excitation_gradient = spatial_time_bandwidth_product / (
        actual_subpulse_duration_s * gamma_hz_per_t * slice_thickness_m
    )
    if excitation_gradient > max_gradient_t_per_m * (1 + 1e-12):
        minimum_duration = spatial_time_bandwidth_product / (
            max_gradient_t_per_m * gamma_hz_per_t * slice_thickness_m
        )
        raise ValueError(
            "spatial subpulse exceeds the gradient limit; use a duration of at "
            f"least {minimum_duration * 1e3:.3f} ms or a thicker slice"
        )
    rise_samples = max(
        1,
        int(np.ceil(excitation_gradient / (max_slew_t_per_m_s * gradient_raster_s))),
    )
    positive = _trapezoid_samples(
        excitation_gradient, rise_samples, flat_gradient_samples
    )
    positive_area = float(np.sum(positive) * gradient_raster_s)
    rewinder = _minimum_time_rewinder(
        positive_area,
        raster_s=gradient_raster_s,
        max_gradient_t_per_m=max_gradient_t_per_m,
        max_slew_t_per_m_s=max_slew_t_per_m_s,
    )
    minimum_period_samples = len(positive) + len(rewinder)
    if max_duration_s is None:
        maximum_period_samples = 8 * minimum_period_samples
    else:
        maximum_period_samples = int(
            np.floor(max_duration_s / (subpulse_count * gradient_raster_s))
        )
        if maximum_period_samples < minimum_period_samples:
            minimum_duration = (
                subpulse_count * minimum_period_samples * gradient_raster_s
            )
            raise ValueError(
                "gradient lobes exceed max_duration_s; the minimum duration is "
                f"{minimum_duration * 1e3:.3f} ms"
            )
    if spectral_sampling_rate_hz is None:
        sampling_rate_hz, alias_result, period_samples = (
            find_compatible_spectral_sampling_rate(
                bands,
                minimum_period_samples=minimum_period_samples,
                raster_s=gradient_raster_s,
                stopbands=stopbands,
                maximum_period_samples=maximum_period_samples,
            )
        )
    else:
        requested_rate = _positive_finite(
            "spectral_sampling_rate_hz", spectral_sampling_rate_hz
        )
        period_samples = int(np.ceil(1.0 / (requested_rate * gradient_raster_s)))
        if period_samples < minimum_period_samples:
            raise ValueError(
                "requested spectral sampling rate leaves insufficient time for "
                "the excitation and flyback gradients"
            )
        if period_samples > maximum_period_samples:
            raise ValueError("requested spectral sampling rate exceeds max_duration_s")
        sampling_rate_hz = 1.0 / (period_samples * gradient_raster_s)
        alias_result = check_spectral_alias_compatibility(
            bands, sampling_rate_hz, stopbands=stopbands
        )
        if not alias_result.compatible:
            raise ValueError(
                "requested spectral sampling rate has incompatible aliases: "
                + alias_result.reason
            )

    dead_samples = period_samples - minimum_period_samples
    period_gradient_t_per_m = np.concatenate(
        (positive, rewinder, np.zeros(dead_samples, dtype=float))
    )
    gradient_t_per_m_wave = np.tile(period_gradient_t_per_m, subpulse_count)
    gradient_hz_per_m = gradient_t_per_m_wave * gamma_hz_per_t
    period_s = period_samples * gradient_raster_s
    total_duration_s = subpulse_count * period_s
    if max_duration_s is not None and total_duration_s > max_duration_s * (1 + 1e-12):
        raise RuntimeError("internal duration search exceeded max_duration_s")

    subpulse_rf_count = int(round(actual_subpulse_duration_s / rf_raster_s))
    envelope = _design_spatial_subpulse(
        subpulse_rf_count,
        time_bandwidth_product=spatial_time_bandwidth_product,
        slice_thickness_m=slice_thickness_m,
        transition_width_m=spatial_transition_m,
        passband_ripple=spatial_passband_ripple,
        stopband_ripple=spatial_stopband_ripple,
    )
    spatial_metrics = _spatial_response_metrics(
        envelope,
        time_bandwidth_product=spatial_time_bandwidth_product,
        slice_thickness_m=slice_thickness_m,
        transition_width_m=spatial_transition_m,
        passband_ripple=spatial_passband_ripple,
        stopband_ripple=spatial_stopband_ripple,
    )
    coefficients = _spectral_weights(
        bands,
        stopbands,
        subpulse_count,
        sampling_rate_hz,
        spectral_phase_type,
    )
    rf_count = int(round(total_duration_s / rf_raster_s))
    rf_hz = np.zeros(rf_count, dtype=np.complex128)
    for index, coefficient in enumerate(coefficients):
        start_s = index * period_s + rise_samples * gradient_raster_s
        start = int(round(start_s / rf_raster_s))
        rf_hz[start : start + subpulse_rf_count] += coefficient * envelope

    reference_band = max(bands, key=lambda band: band.amplitude)
    times = (np.arange(rf_count, dtype=float) + 0.5) * rf_raster_s
    coherent = (
        np.sum(rf_hz * np.exp(-2j * np.pi * reference_band.center_hz * times))
        * rf_raster_s
    )
    if not np.isfinite(coherent) or abs(coherent) <= 1e-15:
        raise ValueError("spectral design produced no coherent RF integral")
    rf_hz *= (flip_angle_deg / 360.0) / abs(coherent)
    peak_b1_t = float(np.max(np.abs(rf_hz)) / gamma_hz_per_t)
    if max_b1_t is not None and peak_b1_t > max_b1_t * (1 + 1e-10):
        raise ValueError(
            f"designed RF requires {peak_b1_t * 1e6:.3f} uT, exceeding the "
            f"{max_b1_t * 1e6:.3f} uT B1 limit; increase duration or subpulse count"
        )

    slew = (
        np.concatenate(
            (
                [2.0 * gradient_t_per_m_wave[0]],
                np.diff(gradient_t_per_m_wave),
                [-2.0 * gradient_t_per_m_wave[-1]],
            )
        )
        / gradient_raster_s
    )
    actual_max_slew = float(np.max(np.abs(slew)))
    if actual_max_slew > max_slew_t_per_m_s * (1 + 1e-9):
        raise RuntimeError("internal gradient construction exceeded the slew limit")
    residual_moment = float(np.sum(gradient_t_per_m_wave) * gradient_raster_s)
    if abs(residual_moment) > max(1e-14, positive_area * subpulse_count * 1e-10):
        raise RuntimeError("spectral-spatial gradient is not balanced")

    metrics = _response_metrics(
        coefficients,
        bands,
        stopbands,
        sampling_rate_hz,
        spectral_phase_type,
    )
    metrics.update(spatial_metrics)
    spectral_stop_limit = min(
        [band.stopband_ripple for band in bands]
        + [band.maximum_amplitude for band in stopbands]
    )
    metrics["spectral_filter_specifications_met"] = bool(
        metrics["spectral_passband_max_error"]
        <= min(band.passband_ripple for band in bands) * (1 + 1e-6)
        and metrics["spectral_stopband_max_amplitude"]
        <= spectral_stop_limit * (1 + 1e-6)
    )
    metrics["all_filter_specifications_met"] = bool(
        metrics["spectral_filter_specifications_met"]
        and metrics["spatial_filter_specifications_met"]
    )
    metrics.update(
        {
            "duration_s": float(total_duration_s),
            "peak_b1_t": peak_b1_t,
            "peak_gradient_t_per_m": float(np.max(np.abs(gradient_t_per_m_wave))),
            "peak_slew_t_per_m_s": actual_max_slew,
            "gradient_moment_t_s_per_m": residual_moment,
            "positive_lobe_samples": int(len(positive)),
            "rewinder_samples": int(len(rewinder)),
            "period_samples": int(period_samples),
        }
    )
    metadata = {
        "schema_version": 2,
        "implementation": DESIGN_IMPLEMENTATION,
        "reference_method": "Larson/Kerr spectral-spatial RF pulse design",
        "reference_toolbox_url": LARSON_TOOLBOX_URL,
        "matlab_reference_validated": False,
        "method": "flyback",
        "spectral_phase_type": spectral_phase_type,
        "inputs": {
            "nucleus": str(nucleus),
            "reference_flip_angle_deg": float(flip_angle_deg),
            "slice_thickness_m": float(slice_thickness_m),
            "spatial_center_m": float(spatial_center_m),
            "spatial_transition_m": float(spatial_transition_m),
            "spatial_time_bandwidth_product": float(spatial_time_bandwidth_product),
            "requested_subpulse_duration_s": float(subpulse_duration_s),
            "actual_subpulse_duration_s": float(actual_subpulse_duration_s),
            "subpulse_count": int(subpulse_count),
            "rf_raster_s": float(rf_raster_s),
            "gradient_raster_s": float(gradient_raster_s),
            "bands": [band.to_dict() for band in bands],
            "stopbands": [band.to_dict() for band in stopbands],
        },
        "spectral_coefficients": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in coefficients
        ],
        "alias_search": {
            "sampling_rate_hz": float(sampling_rate_hz),
            "period_samples": int(period_samples),
            "minimum_period_samples": int(minimum_period_samples),
            "maximum_period_samples": int(maximum_period_samples),
            "result": alias_result.reason,
        },
        "constraints": {
            "max_gradient_t_per_m": float(max_gradient_t_per_m),
            "max_slew_t_per_m_s": float(max_slew_t_per_m_s),
            "max_b1_t": None if max_b1_t is None else float(max_b1_t),
            "max_duration_s": (
                None if max_duration_s is None else float(max_duration_s)
            ),
            "spatial_passband_ripple": float(spatial_passband_ripple),
            "spatial_stopband_ripple": float(spatial_stopband_ripple),
        },
        "metrics": metrics,
        "unsupported": [
            "echo_planar/bipolar trajectory",
            "large-tip two-dimensional SLR correction",
            "B1 VERSE optimization",
            "MATLAB numerical-reference equivalence",
        ],
    }
    # Pulseq sequence builders use this isodelay to center balanced readouts.
    # A centered reference is exact for the linear-phase design and remains a
    # stable, conservative timing reference for minimum-phase excitation.
    excitation_center_s = total_duration_s / 2.0
    return SpectralSpatialPulse(
        rf_hz=rf_hz,
        rf_raster_s=rf_raster_s,
        gradient_hz_per_m=gradient_hz_per_m,
        gradient_raster_s=gradient_raster_s,
        reference_flip_angle_deg=flip_angle_deg,
        reference_thickness_m=slice_thickness_m,
        nucleus=nucleus,
        bands=bands,
        stopbands=stopbands,
        reference_spatial_center_m=float(spatial_center_m),
        reference_spatial_transition_m=float(spatial_transition_m),
        name=name,
        method="flyback",
        spectral_phase_type=spectral_phase_type,
        spatial_time_bandwidth_product=spatial_time_bandwidth_product,
        spatial_passband_ripple=spatial_passband_ripple,
        spatial_stopband_ripple=spatial_stopband_ripple,
        subpulse_duration_s=actual_subpulse_duration_s,
        subpulse_period_s=period_s,
        excitation_center_s=excitation_center_s,
        max_gradient_t_per_m=float(np.max(np.abs(gradient_t_per_m_wave))),
        max_slew_t_per_m_s=actual_max_slew,
        peak_b1_t=peak_b1_t,
        max_b1_t=max_b1_t,
        max_duration_s=max_duration_s,
        design_metadata=metadata,
    )


def simulate_spectral_spatial_profile(
    pulse: SpectralSpatialPulse | Mapping,
    positions_m,
    frequency_offsets_hz,
    *,
    flip_angle_deg: float | None = None,
    thickness_m: float | None = None,
    spatial_offset_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Run a relaxation-free Bloch simulation on a position/frequency grid."""
    pulse = SpectralSpatialPulse.from_mapping(pulse)
    positions = np.asarray(positions_m, dtype=float).reshape(-1)
    frequencies = np.asarray(frequency_offsets_hz, dtype=float).reshape(-1)
    if positions.size == 0 or frequencies.size == 0:
        raise ValueError("positions and frequencies must not be empty")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("positions and frequencies must be finite")
    rf, gradient = pulse.scaled_waveforms(
        flip_angle_deg=(
            pulse.reference_flip_angle_deg
            if flip_angle_deg is None
            else float(flip_angle_deg)
        ),
        thickness_m=thickness_m,
        spatial_offset_m=spatial_offset_m,
    )
    gradient_rf = _resample_piecewise_constant(
        gradient,
        pulse.gradient_raster_s,
        rf.size,
        pulse.rf_raster_s,
    )
    position_grid, frequency_grid = np.meshgrid(positions, frequencies, indexing="ij")
    mx = np.zeros(position_grid.size, dtype=float)
    my = np.zeros_like(mx)
    mz = np.ones_like(mx)
    positions_flat = position_grid.reshape(-1)
    frequencies_flat = frequency_grid.reshape(-1)
    dt = pulse.rf_raster_s
    for b1, gradient_value in zip(rf, gradient_rf):
        # Match the simulator's canonical rotating-frame convention exactly:
        # rotx=-2*pi*Re(B1)*dt, roty=+2*pi*Im(B1)*dt, and
        # rotz=-2*pi*(off-resonance + gradient position)*dt.
        bx = np.full(mx.size, -float(np.real(b1)))
        by = np.full(mx.size, float(np.imag(b1)))
        bz = -(frequencies_flat + gradient_value * positions_flat)
        magnitude = np.sqrt(bx * bx + by * by + bz * bz)
        active = magnitude > 0
        if not np.any(active):
            continue
        nx = np.divide(bx, magnitude, out=np.zeros_like(bx), where=active)
        ny = np.divide(by, magnitude, out=np.zeros_like(by), where=active)
        nz = np.divide(bz, magnitude, out=np.zeros_like(bz), where=active)
        angle = 2.0 * np.pi * magnitude * dt
        cosine = np.cos(angle)
        sine = np.sin(angle)
        dot = nx * mx + ny * my + nz * mz
        cross_x = ny * mz - nz * my
        cross_y = nz * mx - nx * mz
        cross_z = nx * my - ny * mx
        one_minus = 1.0 - cosine
        mx, my, mz = (
            mx * cosine + cross_x * sine + nx * dot * one_minus,
            my * cosine + cross_y * sine + ny * dot * one_minus,
            mz * cosine + cross_z * sine + nz * dot * one_minus,
        )
    shape = position_grid.shape
    mxy = (mx + 1j * my).reshape(shape)
    mz = mz.reshape(shape)
    flip = np.rad2deg(np.arctan2(np.abs(mxy), mz))
    return {
        "positions_m": positions,
        "frequency_offsets_hz": frequencies,
        "mxy": mxy,
        "mz": mz,
        "flip_angle_deg": flip,
    }


@dataclass(frozen=True)
class SpectralSpatialPeak:
    """One preview peak expressed in absolute and sequence-relative units."""

    name: str
    absolute_ppm: float
    sequence_offset_hz: float
    carrier_offset_hz: float
    t1_s: float = np.inf
    t2_star_s: float = np.inf
    relative_weight: float = 1.0
    source: str = "phantom"

    def __post_init__(self) -> None:
        finite = (
            self.absolute_ppm,
            self.sequence_offset_hz,
            self.carrier_offset_hz,
            self.relative_weight,
        )
        if not np.all(np.isfinite(finite)):
            raise ValueError("spectral peak positions and weight must be finite")
        if self.relative_weight < 0:
            raise ValueError("spectral peak relative_weight must be non-negative")
        for name in ("t1_s", "t2_star_s"):
            value = float(getattr(self, name))
            if not (np.isinf(value) or np.isfinite(value) and value > 0):
                raise ValueError(f"spectral peak {name} must be positive or infinite")

    def to_dict(self) -> dict:
        return {
            "name": str(self.name),
            "absolute_ppm": float(self.absolute_ppm),
            "sequence_offset_hz": float(self.sequence_offset_hz),
            "carrier_offset_hz": float(self.carrier_offset_hz),
            "t1_s": float(self.t1_s),
            "t2_star_s": float(self.t2_star_s),
            "relative_weight": float(self.relative_weight),
            "source": str(self.source),
        }

    @classmethod
    def from_mapping(cls, value) -> "SpectralSpatialPeak":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("spectral peak must be a mapping")
        return cls(**dict(value))


def fold_frequency_offset_hz(offset_hz: float, alias_period_hz: float) -> float:
    """Fold one carrier-relative offset into ``[-period/2, period/2)``."""
    alias_period_hz = _positive_finite("alias_period_hz", alias_period_hz)
    offset_hz = float(offset_hz)
    if not np.isfinite(offset_hz):
        raise ValueError("offset_hz must be finite")
    return float(
        (offset_hz + alias_period_hz / 2.0) % alias_period_hz - alias_period_hz / 2.0
    )


def spectral_spatial_peaks(
    *,
    phantom=None,
    presets: Sequence | None = None,
    field_strength_t: float,
    nucleus: str,
    sequence_reference_ppm: float,
    rf_carrier_offset_hz: float = 0.0,
) -> tuple[SpectralSpatialPeak, ...]:
    """Resolve preview peaks against the actual sequence and RF references.

    Phantom components take precedence. Their stored chemical shifts are
    relative to ``phantom.spectral_reference_ppm``; the returned absolute ppm
    position is converted to Hz relative to ``sequence_reference_ppm`` and then
    to the SPSP carrier by subtracting ``rf_carrier_offset_hz``. Presets are
    treated as absolute ppm positions and are used only when no compatible
    phantom was supplied.
    """
    field_strength_t = _positive_finite("field_strength_t", field_strength_t)
    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        raise ValueError(f"unsupported nucleus {nucleus!r}")
    sequence_reference_ppm = float(sequence_reference_ppm)
    rf_carrier_offset_hz = float(rf_carrier_offset_hz)
    if not np.isfinite(sequence_reference_ppm) or not np.isfinite(rf_carrier_offset_hz):
        raise ValueError("sequence reference and RF carrier offset must be finite")

    components = None
    source = "preset"
    phantom_reference_ppm = 0.0
    if phantom is not None:
        components = getattr(phantom, "species", None)
        if components is None:
            components = getattr(phantom, "pools", None)
        if components is not None:
            source = "phantom"
            phantom_reference_ppm = float(
                getattr(phantom, "spectral_reference_ppm", 0.0)
            )
    if components is None:
        components = tuple(presets or ())

    peaks = []
    for component in components:
        fixed_hz = getattr(component, "frequency_offset_hz", None)
        if source == "phantom":
            relative_ppm = (
                float(hz_to_ppm(fixed_hz, field_strength_t, nucleus))
                if fixed_hz is not None
                else float(component.chemical_shift_ppm)
            )
            absolute_ppm = phantom_reference_ppm + relative_ppm
        else:
            absolute_ppm = float(component.chemical_shift_ppm)
        sequence_offset_hz = float(
            ppm_to_hz(
                absolute_ppm - sequence_reference_ppm,
                field_strength_t,
                nucleus,
            )
        )
        t1_s = float(getattr(component, "t1_s", getattr(component, "t1", np.inf)))
        t2_star_s = float(
            getattr(
                component,
                "t2_star_s",
                getattr(component, "t2_star", getattr(component, "t2", np.inf)),
            )
        )
        peaks.append(
            SpectralSpatialPeak(
                name=str(component.name),
                absolute_ppm=absolute_ppm,
                sequence_offset_hz=sequence_offset_hz,
                carrier_offset_hz=sequence_offset_hz - rf_carrier_offset_hz,
                t1_s=t1_s,
                t2_star_s=t2_star_s,
                source=source,
            )
        )
    return tuple(peaks)


def spectral_spatial_schedule_warnings(
    library,
    schedule,
    peaks: Sequence[SpectralSpatialPeak | Mapping],
    *,
    phantom_sequence_reference_ppm: float | None = None,
    sequence_reference_ppm: float | None = None,
) -> tuple[str, ...]:
    """Return user-facing target, reference, and design-quality warnings."""
    pulse_library = SpectralSpatialPulseLibrary.from_mapping(library)
    pulse_schedule = SpectralSpatialSchedule.from_mapping(schedule)
    peaks = tuple(SpectralSpatialPeak.from_mapping(peak) for peak in peaks)

    def key(value):
        return "".join(
            character for character in str(value).lower() if character.isalnum()
        )

    by_name = {key(peak.name): peak for peak in peaks}
    warnings = []
    if (
        phantom_sequence_reference_ppm is not None
        and sequence_reference_ppm is not None
        and not np.isclose(
            float(phantom_sequence_reference_ppm),
            float(sequence_reference_ppm),
            atol=1e-6,
            rtol=0.0,
        )
    ):
        warnings.append(
            "The phantom's stored sequence reference differs from the current "
            "sequence reference; peak offsets were recalculated from absolute ppm."
        )
    for step_index, entry in enumerate(pulse_schedule.expanded_entries()):
        pulse = pulse_library.get(entry.pulse_id)
        target_key = key(entry.target_metabolite)
        if target_key:
            if target_key in by_name:
                matches = [by_name[target_key]]
            else:
                matches = [
                    peak
                    for peak_key, peak in by_name.items()
                    if target_key in peak_key or peak_key in target_key
                ]
            if len(matches) != 1:
                warnings.append(
                    f"Schedule step {step_index + 1} target "
                    f"{entry.target_metabolite!r} is not present uniquely in the displayed peaks."
                )
            else:
                peak = matches[0]
                folded = fold_frequency_offset_hz(
                    peak.carrier_offset_hz, pulse.spectral_alias_bandwidth_hz
                )
                in_passband = any(
                    abs(
                        fold_frequency_offset_hz(
                            folded - band.center_hz,
                            pulse.spectral_alias_bandwidth_hz,
                        )
                    )
                    <= band.width_hz / 2.0
                    for band in pulse.bands
                )
                if not in_passband:
                    warnings.append(
                        f"Schedule step {step_index + 1} target {peak.name!r} folds to "
                        f"{folded:+.3f} Hz, outside every requested pass band."
                    )
        if (
            pulse.design_metadata.get("metrics", {}).get(
                "all_filter_specifications_met"
            )
            is False
        ):
            warnings.append(
                f"Schedule step {step_index + 1} pulse {pulse.name!r} does not meet "
                "all sampled filter specifications; stop-band suppression is not guaranteed."
            )
    return tuple(dict.fromkeys(warnings))


def _apply_spectral_spatial_waveform(
    pulse: SpectralSpatialPulse,
    rf: np.ndarray,
    gradient: np.ndarray,
    positions: np.ndarray,
    frequencies: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one complex SPSP waveform to arbitrary initial magnetization."""
    gradient_rf = _resample_piecewise_constant(
        gradient,
        pulse.gradient_raster_s,
        rf.size,
        pulse.rf_raster_s,
    )
    position_grid, frequency_grid = np.meshgrid(positions, frequencies, indexing="ij")
    positions_flat = position_grid.reshape(-1)
    frequencies_flat = frequency_grid.reshape(-1)
    mx = np.asarray(mx, dtype=float).reshape(-1).copy()
    my = np.asarray(my, dtype=float).reshape(-1).copy()
    mz = np.asarray(mz, dtype=float).reshape(-1).copy()
    if mx.size != positions_flat.size or my.size != mx.size or mz.size != mx.size:
        raise ValueError("magnetization shape does not match position/frequency grid")
    dt = pulse.rf_raster_s
    for b1, gradient_value in zip(rf, gradient_rf):
        bx = np.full(mx.size, -float(np.real(b1)))
        by = np.full(mx.size, float(np.imag(b1)))
        bz = -(frequencies_flat + gradient_value * positions_flat)
        magnitude = np.sqrt(bx * bx + by * by + bz * bz)
        active = magnitude > 0
        if not np.any(active):
            continue
        nx = np.divide(bx, magnitude, out=np.zeros_like(bx), where=active)
        ny = np.divide(by, magnitude, out=np.zeros_like(by), where=active)
        nz = np.divide(bz, magnitude, out=np.zeros_like(bz), where=active)
        angle = 2.0 * np.pi * magnitude * dt
        cosine = np.cos(angle)
        sine = np.sin(angle)
        dot = nx * mx + ny * my + nz * mz
        cross_x = ny * mz - nz * my
        cross_y = nz * mx - nx * mz
        cross_z = nx * my - ny * mx
        one_minus = 1.0 - cosine
        mx, my, mz = (
            mx * cosine + cross_x * sine + nx * dot * one_minus,
            my * cosine + cross_y * sine + ny * dot * one_minus,
            mz * cosine + cross_z * sine + nz * dot * one_minus,
        )
    shape = (positions.size, frequencies.size)
    return mx.reshape(shape), my.reshape(shape), mz.reshape(shape)


def simulate_spectral_spatial_schedule_preview(
    library,
    schedule,
    peaks: Sequence[SpectralSpatialPeak | Mapping],
    positions_m,
    *,
    thickness_m: float | None = None,
    spatial_offset_m: float = 0.0,
    pulse_spacing_s: float | Sequence[float] = 0.0,
    include_relaxation: bool = True,
    spoiling: str = "ideal",
) -> dict[str, object]:
    """Fast cumulative Bloch preview on discrete peaks and a reduced z grid.

    The calculation uses the complete complex RF and gradient samples for every
    expanded schedule step. Between pulse end and the next pulse start it
    applies off-resonance phase evolution and, when requested, T1/T2* recovery.
    ``spoiling='ideal'`` clears transverse magnetization only after recording
    each step endpoint; ``'none'`` preserves it coherently.
    """
    pulse_library = SpectralSpatialPulseLibrary.from_mapping(library)
    pulse_schedule = SpectralSpatialSchedule.from_mapping(schedule)
    steps = pulse_schedule.expanded_entries()
    if not steps:
        raise ValueError("spectral-spatial schedule has no enabled entries")
    peaks = tuple(SpectralSpatialPeak.from_mapping(peak) for peak in peaks)
    if not peaks:
        raise ValueError("schedule preview needs at least one phantom or preset peak")
    positions = np.asarray(positions_m, dtype=float).reshape(-1)
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("preview positions must be non-empty and finite")
    spoiling = str(spoiling).strip().lower()
    if spoiling not in {"ideal", "none"}:
        raise ValueError("preview spoiling must be 'ideal' or 'none'")
    if np.isscalar(pulse_spacing_s):
        spacing = np.full(len(steps), float(pulse_spacing_s), dtype=float)
    else:
        spacing = np.asarray(pulse_spacing_s, dtype=float).reshape(-1)
        if spacing.size != len(steps):
            raise ValueError("pulse_spacing_s must be scalar or one value per step")
    if not np.all(np.isfinite(spacing)) or np.any(spacing < 0):
        raise ValueError("pulse spacing must be non-negative and finite")

    frequencies = np.asarray([peak.carrier_offset_hz for peak in peaks], dtype=float)
    t1 = np.asarray([peak.t1_s for peak in peaks], dtype=float)[None, :]
    t2_star = np.asarray([peak.t2_star_s for peak in peaks], dtype=float)[None, :]
    shape = (positions.size, len(peaks))
    mx = np.zeros(shape, dtype=float)
    my = np.zeros(shape, dtype=float)
    mz = np.ones(shape, dtype=float)
    cumulative_mxy = []
    cumulative_mz = []
    isolated_mxy = []
    isolated_mz = []
    effective_flip_angles = []
    folded_offsets = []
    step_metadata = []
    elapsed_s = 0.0
    for step_index, (entry, start_spacing) in enumerate(zip(steps, spacing)):
        pulse = pulse_library.get(entry.pulse_id)
        flip = entry.effective_flip_angle_deg(pulse)
        target_thickness = (
            pulse.reference_thickness_m if thickness_m is None else float(thickness_m)
        )
        rf, gradient = pulse.scaled_waveforms(
            flip_angle_deg=flip,
            thickness_m=target_thickness,
            spatial_offset_m=spatial_offset_m,
        )
        mx, my, mz = _apply_spectral_spatial_waveform(
            pulse, rf, gradient, positions, frequencies, mx, my, mz
        )
        cumulative_mxy.append(mx + 1j * my)
        cumulative_mz.append(mz.copy())
        isolated = simulate_spectral_spatial_profile(
            pulse,
            positions,
            frequencies,
            flip_angle_deg=flip,
            thickness_m=target_thickness,
            spatial_offset_m=spatial_offset_m,
        )
        isolated_mxy.append(isolated["mxy"])
        isolated_mz.append(isolated["mz"])
        effective_flip_angles.append(flip)
        folded_offsets.append(
            [
                fold_frequency_offset_hz(
                    peak.carrier_offset_hz, pulse.spectral_alias_bandwidth_hz
                )
                for peak in peaks
            ]
        )
        step_metadata.append(
            {
                "step_index": step_index,
                "entry_id": entry.entry_id,
                "pulse_id": entry.pulse_id,
                "display_name": entry.display_name or pulse.name,
                "target_metabolite": entry.target_metabolite,
                "scale_mode": entry.scale_mode,
                "amplitude_scale": entry.effective_amplitude_scale(pulse),
                "flip_angle_deg": flip,
                "pulse_duration_s": pulse.duration_s,
                "start_time_s": elapsed_s,
            }
        )
        gap_s = float(start_spacing) - pulse.duration_s if start_spacing else 0.0
        if gap_s < -1e-12:
            raise ValueError(
                f"preview spacing for step {step_index + 1} is shorter than its pulse"
            )
        gap_s = max(0.0, gap_s)
        if spoiling == "ideal":
            mx.fill(0.0)
            my.fill(0.0)
        if gap_s:
            mxy = (mx + 1j * my) * np.exp(2j * np.pi * frequencies[None, :] * gap_s)
            if include_relaxation:
                mxy *= np.exp(-gap_s / t2_star)
                mz = 1.0 + (mz - 1.0) * np.exp(-gap_s / t1)
            mx = np.real(mxy)
            my = np.imag(mxy)
        elapsed_s += pulse.duration_s + gap_s

    return {
        "positions_m": positions,
        "peaks": [peak.to_dict() for peak in peaks],
        "steps": step_metadata,
        "isolated_mxy": np.asarray(isolated_mxy),
        "isolated_mz": np.asarray(isolated_mz),
        "isolated_flip_angle_deg": np.rad2deg(
            np.arctan2(np.abs(isolated_mxy), np.asarray(isolated_mz))
        ),
        "cumulative_mxy": np.asarray(cumulative_mxy),
        "cumulative_mz": np.asarray(cumulative_mz),
        "effective_flip_angle_deg": np.asarray(effective_flip_angles),
        "folded_offsets_hz": np.asarray(folded_offsets),
        "include_relaxation": bool(include_relaxation),
        "spoiling": spoiling,
        "approximation": (
            "Reduced spatial grid and discrete peak frequencies; complete complex "
            "SPSP RF/gradient waveforms, coherent state propagation, inter-pulse "
            "off-resonance, optional T1/T2* relaxation, and the selected ideal/no "
            "spoiling assumption are included. Readout and encoding gradients, "
            "motion, exchange, B0 distributions, and receive reconstruction are omitted."
        ),
    }
