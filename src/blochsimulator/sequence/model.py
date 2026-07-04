"""Immutable event model for spatially resolved sequence simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np

from ..units import gradient_g_per_cm_to_hz_per_m, rf_gauss_to_hz


def _readonly_1d(values, dtype, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RFEvent:
    """Piecewise-constant complex RF event in nutation-frequency Hz."""

    start_s: float
    samples_hz: np.ndarray
    raster_s: float
    frequency_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0

    def __post_init__(self) -> None:
        samples = _readonly_1d(self.samples_hz, np.complex128, "samples_hz")
        object.__setattr__(self, "samples_hz", samples)
        _validate_time(self.start_s, "start_s", allow_zero=True)
        _validate_time(self.raster_s, "raster_s")
        _validate_finite(self.frequency_offset_hz, "frequency_offset_hz")
        _validate_finite(self.phase_offset_rad, "phase_offset_rad")

    @property
    def end_s(self) -> float:
        return float(self.start_s + self.samples_hz.size * self.raster_s)


@dataclass(frozen=True)
class GradientEvent:
    """Piecewise-constant gradient event in Hz/m on one physical axis."""

    axis: str
    start_s: float
    samples_hz_per_m: np.ndarray
    raster_s: float

    def __post_init__(self) -> None:
        axis = str(self.axis).lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("gradient axis must be 'x', 'y', or 'z'")
        object.__setattr__(self, "axis", axis)
        samples = _readonly_1d(self.samples_hz_per_m, np.float64, "samples_hz_per_m")
        object.__setattr__(self, "samples_hz_per_m", samples)
        _validate_time(self.start_s, "start_s", allow_zero=True)
        _validate_time(self.raster_s, "raster_s")

    @property
    def end_s(self) -> float:
        return float(self.start_s + self.samples_hz_per_m.size * self.raster_s)


@dataclass(frozen=True)
class ADCEvent:
    """Uniform ADC sampling event with receiver frequency and phase."""

    start_s: float
    num_samples: int
    dwell_s: float
    frequency_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0

    def __post_init__(self) -> None:
        _validate_time(self.start_s, "start_s", allow_zero=True)
        if int(self.num_samples) != self.num_samples or self.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        _validate_time(self.dwell_s, "dwell_s")
        _validate_finite(self.frequency_offset_hz, "frequency_offset_hz")
        _validate_finite(self.phase_offset_rad, "phase_offset_rad")

    @property
    def sample_times_s(self) -> np.ndarray:
        return self.start_s + np.arange(self.num_samples, dtype=float) * self.dwell_s

    @property
    def end_s(self) -> float:
        return float(self.start_s + (self.num_samples - 1) * self.dwell_s)


SequenceEvent = Union[RFEvent, GradientEvent, ADCEvent]


@dataclass(frozen=True)
class SequenceProgram:
    """Validated event collection in canonical sequence units."""

    events: Tuple[SequenceEvent, ...]
    duration_s: float
    source: str = "internal"
    version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(
            not isinstance(event, (RFEvent, GradientEvent, ADCEvent))
            for event in events
        ):
            raise TypeError("events contains an unsupported event type")
        _validate_time(self.duration_s, "duration_s", allow_zero=True)
        latest = max((event.end_s for event in events), default=0.0)
        tolerance = max(1e-15, abs(self.duration_s) * 1e-12)
        if latest > self.duration_s + tolerance:
            raise ValueError(
                f"duration_s={self.duration_s} ends before the latest event at {latest}"
            )
        ordered = tuple(sorted(events, key=lambda event: (event.start_s, event.end_s)))
        object.__setattr__(self, "events", ordered)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def rf_events(self) -> Tuple[RFEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, RFEvent))

    @property
    def gradient_events(self) -> Tuple[GradientEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, GradientEvent))

    @property
    def adc_events(self) -> Tuple[ADCEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, ADCEvent))

    @classmethod
    def from_legacy(
        cls,
        b1_gauss,
        gradients_g_per_cm,
        time_s,
        *,
        adc_times_s: Sequence[float] = (),
        source: str = "legacy",
    ) -> "SequenceProgram":
        """Convert the existing dense `(B1, gradients, time)` representation.

        The legacy representation must have a uniform, strictly increasing time
        grid. ADC times are optional and must lie on or within the sequence.
        """
        b1 = _readonly_1d(b1_gauss, np.complex128, "b1_gauss")
        time = _readonly_1d(time_s, np.float64, "time_s")
        if b1.size != time.size:
            raise ValueError("B1 and time arrays must have equal lengths")
        gradients = np.asarray(gradients_g_per_cm, dtype=np.float64)
        if gradients.ndim == 1:
            gradients = gradients.reshape(-1, 1)
        if gradients.ndim != 2 or gradients.shape[0] != b1.size:
            raise ValueError("gradients must have shape (ntime, 1..3)")
        if gradients.shape[1] > 3 or gradients.shape[1] < 1:
            raise ValueError("gradients must have between one and three columns")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("gradients contains NaN or infinite values")
        if time.size == 1:
            raise ValueError("legacy conversion requires at least two time points")
        differences = np.diff(time)
        if np.any(differences <= 0):
            raise ValueError("time_s must be strictly increasing")
        raster = float(np.median(differences))
        if not np.allclose(differences, raster, rtol=1e-10, atol=1e-15):
            raise ValueError("legacy conversion currently requires a uniform time grid")

        start = float(time[0])
        if start < 0:
            raise ValueError("time_s must not start before zero")
        events = [RFEvent(start, rf_gauss_to_hz(b1), raster)]
        converted = gradient_g_per_cm_to_hz_per_m(gradients)
        for axis_index, axis in enumerate("xyz"):
            samples = (
                converted[:, axis_index]
                if axis_index < converted.shape[1]
                else np.zeros(b1.size)
            )
            events.append(GradientEvent(axis, start, samples, raster))

        duration = float(time[-1] + raster)
        adc_times = np.asarray(tuple(adc_times_s), dtype=float)
        if adc_times.size:
            if adc_times.ndim != 1 or not np.all(np.isfinite(adc_times)):
                raise ValueError(
                    "adc_times_s must be a finite one-dimensional sequence"
                )
            if np.any(adc_times < 0) or np.any(adc_times > duration):
                raise ValueError("ADC times must lie within the sequence duration")
            for sample_time in adc_times:
                # Single-sample events preserve irregular or sparse sampling exactly.
                events.append(ADCEvent(float(sample_time), 1, raster))

        return cls(
            events=tuple(events),
            duration_s=duration,
            source=source,
            metadata={"legacy_time_origin_s": start},
        )


def _validate_time(value: float, name: str, allow_zero: bool = False) -> None:
    _validate_finite(value, name)
    if value < 0 or (value == 0 and not allow_zero):
        relation = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {relation}")


def _validate_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
